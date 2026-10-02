"""Label-flipping poisoning of the cloud model-update pipeline and a trusted-reference defense.

Threat model
  The cloud periodically retrains the classifier head on images uploaded by many devices
  (crowd pool U). An adversary controlling a fraction rho of uploads of HIGH-severity diseases
  relabels them as 'normal' (concealment attack), or flips labels at random (random attack).
  A small expert-verified seed set T (trusted) and a trusted calibration set are assumed.

Defenses
  none        train on U (poisoned) + T
  CL-filter   reference model g_T trained on T flags upload u when its label is unlikely
              under g_T: p_T(y_u | x_u) < t_{y_u} and argmax p_T(x_u) != y_u, where t_k is the
              mean self-confidence of class k on held-out trusted images (confident learning,
              Northcutt et al., 2021); flagged uploads are dropped.
  kNN-filter  flag u when fewer than half of its k nearest trusted neighbours share its label.

Evaluation on the clean held-out set: macro-F1, high-severity recall, concealment rate
(high-severity images predicted 'normal'), detection precision/recall, and the
conformal URGENT recall with a trusted vs. a poisoned calibration set.
"""
import argparse
import json
import time
import warnings
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from trust.conformal import calibrate, lac_scores, predict_sets  # noqa: E402


def fit_head(X, y, K, C=0.5, seed=0):
    """Multinomial logistic regression (lbfgs, at most 500 iterations, identical for every condition)."""
    clf = LogisticRegression(C=C, max_iter=500, tol=1e-4, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X, y)
    return clf


def proba(clf, X, K):
    p = np.zeros((len(X), K))
    p[:, clf.classes_] = clf.predict_proba(X)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    proc = pathlib.Path(cfg["paths"]["processed"])
    res = pathlib.Path(cfg["paths"]["results"])
    classes = cfg["classes"]
    K = len(classes)
    H = [classes.index(c) for c in cfg["risk"]["high_severity"]]
    normal = classes.index("normal")

    df = pd.read_csv(proc / "paddy_index.csv")
    y_all = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    F = np.load(res / "raw" / "imagenet_feats_paddy.npy").astype(np.float32)
    split = df.split.to_numpy()
    groups = df.group.to_numpy()
    tr_idx, ho_idx = np.nonzero(split == "train")[0], np.nonzero(split == "holdout")[0]

    # dimensionality reduction fitted on the unlabeled training features only
    scaler = StandardScaler().fit(F[tr_idx])
    pca = PCA(n_components=256, random_state=cfg["seed"]).fit(scaler.transform(F[tr_idx]))
    Z = pca.transform(scaler.transform(F)).astype(np.float32)

    rows, attacks = [], ["concealment", "random"]
    rhos = cfg["poisoning"]["rhos"]
    for rep in range(args.reps):
        rng = np.random.default_rng(1000 + rep)
        # trusted seed set: a random 10 % of training GROUPS
        tg = np.unique(groups[tr_idx])
        trusted_groups = set(rng.choice(tg, int(cfg["poisoning"]["trusted_fraction"] * len(tg)), replace=False))
        is_T = np.array([g in trusted_groups for g in groups[tr_idx]])
        T, U = tr_idx[is_T], tr_idx[~is_T]
        # held-out split (group-aware) into calibration / test halves
        hg = np.unique(groups[ho_idx])
        cal_groups = set(rng.choice(hg, len(hg) // 2, replace=False))
        is_cal = np.array([g in cal_groups for g in groups[ho_idx]])
        CAL, TEST = ho_idx[is_cal], ho_idx[~is_cal]

        # reference model on trusted data, thresholds from 5-fold out-of-fold self-confidence on T
        g_T = fit_head(Z[T], y_all[T], K, seed=rep)
        oof = np.zeros((len(T), K))
        fold = rng.integers(0, 5, len(T))
        for f in range(5):
            m = fold == f
            oof[m] = proba(fit_head(Z[T][~m], y_all[T][~m], K, seed=rep), Z[T][m], K)
        t_k = np.array([oof[y_all[T] == k, k].mean() for k in range(K)])
        nn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(Z[T])

        for attack in attacks:
            for rho in rhos:
                y_u = y_all[U].copy()
                if attack == "concealment":
                    cand = np.nonzero(np.isin(y_u, H))[0]
                else:
                    cand = np.arange(len(y_u))
                n_poison = int(round(rho * len(cand)))
                pois = rng.choice(cand, n_poison, replace=False) if n_poison else np.array([], int)
                if attack == "concealment":
                    y_u[pois] = normal
                else:
                    y_u[pois] = (y_u[pois] + rng.integers(1, K, len(pois))) % K
                is_pois = np.zeros(len(U), bool)
                is_pois[pois] = True

                pT = proba(g_T, Z[U], K)
                flag_cl = (pT[np.arange(len(U)), y_u] < t_k[y_u]) & (pT.argmax(1) != y_u)
                _, nbr = nn.kneighbors(Z[U])
                agree = (y_all[T][nbr] == y_u[:, None]).mean(1)
                flag_knn = agree < 0.5

                for defense, flag in [("none", np.zeros(len(U), bool)), ("CL-filter", flag_cl),
                                      ("kNN-filter", flag_knn)]:
                    keep = ~flag
                    Xtr = np.concatenate([Z[T], Z[U][keep]])
                    ytr = np.concatenate([y_all[T], y_u[keep]])
                    clf = fit_head(Xtr, ytr, K, seed=rep)
                    p_te, p_cal = proba(clf, Z[TEST], K), proba(clf, Z[CAL], K)
                    y_te = y_all[TEST]
                    pred = p_te.argmax(1)
                    inH = np.isin(y_te, H)
                    row = dict(rep=rep, attack=attack, rho=rho, defense=defense,
                               macro_f1=f1_score(y_te, pred, average="macro"),
                               accuracy=float((pred == y_te).mean()),
                               high_sev_recall=float(np.isin(pred[inH], H).mean()),
                               concealment_rate=float((pred[inH] == normal).mean()),
                               n_poisoned=int(is_pois.sum()), n_flagged=int(flag.sum()),
                               det_precision=float((flag & is_pois).sum() / max(flag.sum(), 1)),
                               det_recall=float((flag & is_pois).sum() / max(is_pois.sum(), 1)),
                               clean_removed=float((flag & ~is_pois).sum() / max((~is_pois).sum(), 1)))
                    # conformal URGENT recall with trusted calibration vs. poisoned calibration
                    for cal_kind in ("trusted", "poisoned"):
                        y_cal = y_all[CAL].copy()
                        if cal_kind == "poisoned" and rho > 0:
                            if attack == "concealment":
                                c = np.nonzero(np.isin(y_cal, H))[0]
                                pc = rng.choice(c, int(round(rho * len(c))), replace=False)
                                y_cal[pc] = normal
                            else:
                                pc = rng.choice(len(y_cal), int(round(rho * len(y_cal))), replace=False)
                                y_cal[pc] = (y_cal[pc] + rng.integers(1, K, len(pc))) % K
                        q = calibrate(lac_scores(p_cal), y_cal, args.alpha)
                        sets = predict_sets(lac_scores(p_te), q)
                        urgent = sets[:, H].any(1)
                        row[f"urgent_recall_{cal_kind}_cal"] = float(urgent[inH].mean())
                        row[f"coverage_{cal_kind}_cal"] = float(sets[np.arange(len(y_te)), y_te].mean())
                        row[f"set_size_{cal_kind}_cal"] = float(sets.sum(1).mean())
                    rows.append(row)
        print(f"rep {rep} done {time.strftime('%H:%M:%S')}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(res / "raw" / "poisoning_results.csv", index=False)
    print(out.groupby(["attack", "rho", "defense"]).mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
