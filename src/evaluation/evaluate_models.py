"""E1-E4, E7: accuracy and calibration, conformal prediction, uncertainty-gated cascade,
cross-region shift (Sethy) with target recalibration, and per-variety subgroup analysis.

Reads the logits written by src/models/train.py for every seed and writes
results/raw/eval_*.json|csv. All numbers in the manuscript tables come from these files.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from evaluation.stats import cluster_bootstrap, mean_sd  # noqa: E402
from trust.calibration import brier, classwise_ece, ece, fit_temperature, nll, reliability_curve  # noqa: E402
from trust.conformal import (aps_scores, calibrate, lac_scores, per_class_coverage,  # noqa: E402
                             predict_sets, set_metrics)


def load(cfg):
    res = pathlib.Path(cfg["paths"]["results"]) / "raw" / "models"
    seeds = [s for s in cfg["train"]["seeds"] if (res / f"seed{s}" / "run.json").exists()]
    L = {k: np.stack([np.load(res / f"seed{s}" / f"logits_{k}.npy") for s in seeds]) for k in ("val", "holdout", "sethy")}
    va = np.load(res / f"seed{seeds[0]}" / "index_val.npy")
    ho = np.load(res / f"seed{seeds[0]}" / "index_holdout.npy")
    return seeds, L, va, ho


def calibrated_probs(L, y_val):
    """Per-seed temperature scaling; ensemble = mean of calibrated members, re-tempered on val."""
    S = L["val"].shape[0]
    T = [fit_temperature(L["val"][s], y_val) for s in range(S)]
    P = {k: np.stack([softmax(L[k][s] / T[s], axis=1) for s in range(S)]) for k in L}
    P_raw = {k: np.stack([softmax(L[k][s], axis=1) for s in range(S)]) for k in L}
    ens_log = {k: np.log(np.clip(P[k].mean(0), 1e-12, 1)) for k in L}
    T_ens = fit_temperature(ens_log["val"], y_val)
    ens = {k: softmax(ens_log[k] / T_ens, axis=1) for k in L}
    ens_raw = {k: P_raw[k].mean(0) for k in L}
    return T, T_ens, P, P_raw, ens, ens_raw


def cls_metrics(p, y, K):
    pred = p.argmax(1)
    return {"accuracy": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro", labels=range(K))),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "nll": nll(p, y), "ece": ece(p, y),
            "classwise_ece": classwise_ece(p, y), "brier": brier(p, y)}


def tiers_from_sets(sets, H):
    urgent = sets[:, H].any(1)
    size = sets.sum(1)
    strategic = ~urgent & (size != 1)
    routine = ~urgent & ~strategic
    return np.where(urgent, "URGENT", np.where(strategic, "STRATEGIC", "ROUTINE"))


def group_split(groups, idx, rng, frac=0.5):
    g = np.unique(groups[idx])
    cal_g = set(rng.choice(g, int(len(g) * frac), replace=False))
    m = np.array([x in cal_g for x in groups[idx]])
    return idx[m], idx[~m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    out_dir = pathlib.Path(cfg["paths"]["results"]) / "raw"
    proc = pathlib.Path(cfg["paths"]["processed"])
    classes = cfg["classes"]
    K = len(classes)
    cidx = {c: i for i, c in enumerate(classes)}
    H = [cidx[c] for c in cfg["risk"]["high_severity"]]
    Hb = [cidx[c] for c in cfg["risk"]["high_severity_broad"]]
    alpha = cfg["risk"]["alpha"]

    df = pd.read_csv(proc / "paddy_index.csv")
    y_all = df.label.map(cidx).to_numpy()
    groups = df.group.to_numpy()
    sdf = pd.read_csv(proc / "sethy_index.csv")
    y_s = sdf.label.map(cidx).to_numpy()
    g_s = sdf.group.to_numpy()

    seeds, L, va, ho = load(cfg)
    S = len(seeds)
    y_va, y_ho = y_all[va], y_all[ho]
    g_ho = groups[ho]
    T, T_ens, P, P_raw, ENS, ENS_raw = calibrated_probs(L, y_va)
    summary = {"seeds": seeds, "temperatures": T, "temperature_ensemble": T_ens,
               "n_val": int(len(va)), "n_holdout": int(len(ho)), "n_holdout_groups": int(len(np.unique(g_ho)))}

    # ---------------- E1: accuracy + calibration on the full holdout ----------------
    e1 = {"edge_raw": [], "edge_ts": []}
    for s in range(S):
        e1["edge_raw"].append(cls_metrics(P_raw["holdout"][s], y_ho, K))
        e1["edge_ts"].append(cls_metrics(P["holdout"][s], y_ho, K))
    e1["ensemble_raw"] = cls_metrics(ENS_raw["holdout"], y_ho, K)
    e1["ensemble_ts"] = cls_metrics(ENS["holdout"], y_ho, K)
    agg = {}
    for k in ("edge_raw", "edge_ts"):
        agg[k] = {m: mean_sd([r[m] for r in e1[k]]) for m in e1[k][0]}
    for k in ("ensemble_raw", "ensemble_ts"):
        agg[k] = e1[k]
    # cluster-bootstrap CIs for the ensemble (calibrated) and seed-0 edge
    ci = {}
    for name, p in [("ensemble_ts", ENS["holdout"]), ("edge_ts_seed0", P["holdout"][0])]:
        pred = p.argmax(1)
        ci[name] = {
            "accuracy": cluster_bootstrap(lambda i: (pred[i] == y_ho[i]).mean(), g_ho),
            "macro_f1": cluster_bootstrap(lambda i: f1_score(y_ho[i], pred[i], average="macro", labels=range(K)), g_ho, B=1000),
            "ece": cluster_bootstrap(lambda i: ece(p[i], y_ho[i]), g_ho, B=1000),
        }
    per_class = pd.DataFrame({
        "class": classes,
        "n_holdout": [int((y_ho == k).sum()) for k in range(K)],
        "edge_f1_mean": np.mean([f1_score(y_ho, P["holdout"][s].argmax(1), average=None, labels=range(K)) for s in range(S)], 0),
        "edge_f1_sd": np.std([f1_score(y_ho, P["holdout"][s].argmax(1), average=None, labels=range(K)) for s in range(S)], 0, ddof=1),
        "ensemble_f1": f1_score(y_ho, ENS["holdout"].argmax(1), average=None, labels=range(K)),
    })
    per_class.to_csv(out_dir / "eval_per_class.csv", index=False)
    conf = pd.crosstab(pd.Series(y_ho, name="true"), pd.Series(ENS["holdout"].argmax(1), name="pred")).reindex(index=range(K), columns=range(K), fill_value=0)
    conf.to_csv(out_dir / "eval_confusion_ensemble.csv")
    rel = {"edge_raw_seed0": reliability_curve(P_raw["holdout"][0], y_ho), "edge_ts_seed0": reliability_curve(P["holdout"][0], y_ho),
           "ensemble_ts": reliability_curve(ENS["holdout"], y_ho)}
    json.dump({"per_seed": e1, "aggregate": agg, "cluster_ci": ci, "reliability": rel, **summary},
              open(out_dir / "eval_e1_accuracy_calibration.json", "w"), indent=1)
    print("E1", json.dumps({k: {m: v for m, v in agg[k].items() if m in ("accuracy", "macro_f1", "ece", "nll")} for k in agg}, indent=0))

    # ---------------- E2 + E3: conformal and cascade over random group splits ----------------
    rng = np.random.default_rng(cfg["seed"])
    rows_cp, rows_cas = [], []
    for r in range(cfg["conformal"]["n_splits"]):
        cal, te = group_split(groups, ho, rng)
        pos = {i: k for k, i in enumerate(ho)}
        ci_, ti_ = np.array([pos[i] for i in cal]), np.array([pos[i] for i in te])
        s_edge = r % S
        y_c, y_t = y_ho[ci_], y_ho[ti_]
        for model, p in [("edge", P["holdout"][s_edge]), ("ensemble", ENS["holdout"])]:
            pc, pt = p[ci_], p[ti_]
            for a in cfg["risk"]["alphas"]:
                for method in ("LAC-marginal", "LAC-classwise", "APS-marginal"):
                    if method.startswith("LAC"):
                        sc, st = lac_scores(pc), lac_scores(pt)
                    else:
                        sc, st = aps_scores(pc, rng), aps_scores(pt, rng)
                    q = calibrate(sc, y_c, a, "class" if method == "LAC-classwise" else "marginal", K)
                    sets = predict_sets(st, q)
                    m = set_metrics(sets, y_t, K)
                    pcc = per_class_coverage(sets, y_t, K)
                    tiers = tiers_from_sets(sets, H)
                    inH = np.isin(y_t, H)
                    m.update(split=r, model=model, alpha=a, method=method,
                             high_sev_min_class_coverage=float(np.nanmin(pcc[H])),
                             urgent_recall=float((tiers[inH] == "URGENT").mean()),
                             urgent_false_alarm=float((tiers[~inH] == "URGENT").mean()),
                             rate_urgent=float((tiers == "URGENT").mean()),
                             rate_strategic=float((tiers == "STRATEGIC").mean()),
                             rate_routine=float((tiers == "ROUTINE").mean()),
                             urgent_recall_broadH=float(sets[np.isin(y_t, Hb)][:, Hb].any(1).mean()))
                    for k in range(K):
                        m[f"cov_{classes[k]}"] = float(pcc[k])
                    rows_cp.append(m)

        # ---- cascade (alpha = default, LAC-classwise on the edge model) ----
        pe, pE = P["holdout"][s_edge], ENS["holdout"]
        q = calibrate(lac_scores(pe[ci_]), y_c, alpha, "class", K)
        sets = predict_sets(lac_scores(pe[ti_]), q)
        tiers = tiers_from_sets(sets, H)
        esc = tiers != "ROUTINE"
        rate = esc.mean()
        n_esc = int(esc.sum())
        conf_e = pe[ti_].max(1)
        ent = -(pe[ti_] * np.log(np.clip(pe[ti_], 1e-12, 1))).sum(1)
        riskH = pe[ti_][:, H].sum(1)
        order = {
            "conformal-tier (proposed)": esc,
            "max-softmax": np.isin(np.arange(len(ti_)), np.argsort(conf_e)[:n_esc]),
            "entropy": np.isin(np.arange(len(ti_)), np.argsort(-ent)[:n_esc]),
            "risk-mass": np.isin(np.arange(len(ti_)), np.argsort(-riskH)[:n_esc]),
            "random": np.isin(np.arange(len(ti_)), rng.choice(len(ti_), n_esc, replace=False)),
            "edge-only": np.zeros(len(ti_), bool),
            "cloud-only": np.ones(len(ti_), bool),
        }
        inH = np.isin(y_t, H)
        for gate, e in order.items():
            final = np.where(e, pE[ti_].argmax(1), pe[ti_].argmax(1))
            rows_cas.append(dict(split=r, gate=gate, escalation_rate=float(e.mean()),
                                 accuracy=float((final == y_t).mean()),
                                 macro_f1=float(f1_score(y_t, final, average="macro", labels=range(K))),
                                 high_sev_recall=float(np.isin(final[inH], H).mean()),
                                 high_sev_missed=float((~np.isin(final[inH], H)).mean()),
                                 high_sev_recall_or_flag=float((np.isin(final[inH], H) | (tiers[inH] == "URGENT")).mean()
                                                               if gate.startswith("conformal") else np.isin(final[inH], H).mean()),
                                 compute_units_per_image=float(1 + 5 * e.mean()) if gate != "cloud-only" else 5.0))
    cp = pd.DataFrame(rows_cp)
    cas = pd.DataFrame(rows_cas)
    cp.to_csv(out_dir / "eval_e2_conformal_splits.csv", index=False)
    cas.to_csv(out_dir / "eval_e3_cascade_splits.csv", index=False)
    print("E2", cp[cp.alpha == alpha].groupby(["model", "method"])[["coverage", "avg_set_size", "min_class_coverage", "urgent_recall", "urgent_false_alarm", "rate_routine"]].mean().round(4).to_string())
    print("E3", cas.groupby("gate")[["escalation_rate", "accuracy", "macro_f1", "high_sev_recall", "compute_units_per_image"]].mean().round(4).to_string())

    # ---------------- E4: cross-region shift (Sethy) ----------------
    e4 = {}
    shared = sorted(set(y_s))
    inH_s = np.isin(y_s, H)

    for model, ps, pcal in [("ensemble", ENS["sethy"], ENS["holdout"])] + \
                           [(f"edge_seed{seeds[s]}", P["sethy"][s], P["holdout"][s]) for s in range(S)]:
        pred = ps.argmax(1)
        q = calibrate(lac_scores(pcal), y_ho, alpha, "class", K)
        sets = predict_sets(lac_scores(ps), q)
        tiers = tiers_from_sets(sets, H)
        acc = cluster_bootstrap(lambda i: (pred[i] == y_s[i]).mean(), g_s, B=1000)
        cov = cluster_bootstrap(lambda i: sets[i, y_s[i]].mean(), g_s, B=1000)
        urg = cluster_bootstrap(lambda i: (tiers[i][inH_s[i]] == "URGENT").mean(), g_s, B=1000)
        e4[model] = {"accuracy": acc, "coverage_paddy_calibrated": cov, "urgent_recall_H": urg,
                     "avg_set_size": float(sets.sum(1).mean()),
                     "group_mean_accuracy": float(pd.Series(pred == y_s).groupby(g_s).mean().mean()),
                     "high_sev_recall": float(np.isin(pred[inH_s], H).mean()),
                     "per_class_recall": {classes[k]: float((pred[y_s == k] == k).mean()) for k in shared},
                     "per_class_coverage": {classes[k]: float(sets[y_s == k, k].mean()) for k in shared},
                     "pred_distribution": {classes[k]: float((pred == k).mean()) for k in range(K)},
                     "tier_rates": {t: float((tiers == t).mean()) for t in ("URGENT", "STRATEGIC", "ROUTINE")}}
    # ---- target recalibration with m labelled Sethy groups ----
    rows_rc = []
    ug = np.unique(g_s)
    for m_groups in (5, 10, 20, 40, 80):
        for rep in range(200):
            r2 = np.random.default_rng(10_000 + 100 * m_groups + rep)
            calg = set(r2.choice(ug, m_groups, replace=False))
            mcal = np.array([g in calg for g in g_s])
            s_e = rep % S
            for model, ps, p_src in [("ensemble", ENS["sethy"], ENS["holdout"]),
                                     ("edge", P["sethy"][s_e], P["holdout"][s_e])]:
                for mode in ("marginal", "class"):
                    sc = lac_scores(ps[mcal])
                    present = np.unique(y_s[mcal])
                    if mode == "class":
                        q = np.full(K, np.inf)
                        qq = calibrate(sc, y_s[mcal], alpha, "class", K)
                        q[present] = qq[present]
                        # classes absent from the small target calibration set keep the source (Paddy) threshold
                        qp = calibrate(lac_scores(p_src), y_ho, alpha, "class", K)
                        absent = np.setdiff1d(np.arange(K), present)
                        q[absent] = qp[absent]
                    else:
                        q = calibrate(sc, y_s[mcal], alpha, "marginal", K)
                    sets = predict_sets(lac_scores(ps[~mcal]), q)
                    yt = y_s[~mcal]
                    tiers = tiers_from_sets(sets, H)
                    rows_rc.append(dict(m_groups=m_groups, rep=rep, model=model, mode=mode,
                                        n_cal_images=int(mcal.sum()),
                                        coverage=float(sets[np.arange(len(yt)), yt].mean()),
                                        avg_set_size=float(sets.sum(1).mean()),
                                        urgent_recall_H=float((tiers[np.isin(yt, H)] == "URGENT").mean())))
    rc = pd.DataFrame(rows_rc)
    rc.to_csv(out_dir / "eval_e4_recalibration.csv", index=False)
    json.dump(e4, open(out_dir / "eval_e4_shift.json", "w"), indent=1)
    print("E4", json.dumps({k: {"acc": v["accuracy"], "cov": v["coverage_paddy_calibrated"], "urg": v["urgent_recall_H"]} for k, v in e4.items()}, indent=0))
    print("E4 recal", rc.groupby(["model", "mode", "m_groups"])[["coverage", "avg_set_size", "urgent_recall_H"]].mean().round(4).to_string())

    # ---------------- E7: per-variety subgroup analysis ----------------
    var = df.variety.to_numpy()[ho]
    rows_v = []
    rng = np.random.default_rng(cfg["seed"] + 7)
    covs = {v: [] for v in np.unique(var)}
    for r in range(cfg["conformal"]["n_splits"]):
        cal, te = group_split(groups, ho, rng)
        pos = {i: k for k, i in enumerate(ho)}
        ci_, ti_ = np.array([pos[i] for i in cal]), np.array([pos[i] for i in te])
        q = calibrate(lac_scores(ENS["holdout"][ci_]), y_ho[ci_], alpha, "marginal", K)
        sets = predict_sets(lac_scores(ENS["holdout"][ti_]), q)
        cov = sets[np.arange(len(ti_)), y_ho[ti_]]
        for v in covs:
            m = var[ti_] == v
            if m.any():
                covs[v].append(cov[m].mean())
    predE = ENS["holdout"].argmax(1)
    for v in np.unique(var):
        m = var == v
        rows_v.append(dict(variety=v, n_holdout=int(m.sum()), n_groups=int(len(np.unique(g_ho[m]))),
                           ensemble_accuracy=float((predE[m] == y_ho[m]).mean()),
                           edge_accuracy_mean=float(np.mean([(P["holdout"][s][m].argmax(1) == y_ho[m]).mean() for s in range(S)])),
                           marginal_coverage_mean=float(np.mean(covs[v])) if covs[v] else np.nan,
                           marginal_coverage_p05=float(np.percentile(covs[v], 5)) if covs[v] else np.nan))
    pd.DataFrame(rows_v).to_csv(out_dir / "eval_e7_variety.csv", index=False)
    print("E7", pd.DataFrame(rows_v).round(3).to_string())


if __name__ == "__main__":
    main()
