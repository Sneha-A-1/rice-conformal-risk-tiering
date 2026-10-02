"""E0: how much do near-duplicates inflate image-level evaluation?

Frozen ImageNet MobileNetV3-Large features (never fine-tuned on Paddy Doctor) are classified with
(i) a 1-nearest-neighbour rule and (ii) a logistic-regression head, under two 80/20 splits of the
full Paddy Doctor set: a random image-level split and a group-aware split that keeps near-duplicate
groups together. Ten repetitions each. Any gap between the two protocols is attributable to
near-duplicate leakage, because the model and features are identical.
"""
import argparse
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--reps", type=int, default=10)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    proc = pathlib.Path(cfg["paths"]["processed"])
    res = pathlib.Path(cfg["paths"]["results"]) / "raw"
    df = pd.read_csv(proc / "paddy_index.csv")
    y = df.label.map({c: i for i, c in enumerate(cfg["classes"])}).to_numpy()
    g = df.group.to_numpy()
    F = np.load(res / "imagenet_feats_paddy.npy").astype(np.float32)
    rows = []
    for rep in range(10):
        for proto, splitter in (("random image-level", ShuffleSplit(1, test_size=0.2, random_state=rep)),
                                ("group-aware", GroupShuffleSplit(1, test_size=0.2, random_state=rep))):
            tr, te = next(splitter.split(F, y, groups=g))
            sc = StandardScaler().fit(F[tr])
            pca = PCA(256, random_state=rep).fit(sc.transform(F[tr]))
            Ztr, Zte = pca.transform(sc.transform(F[tr])), pca.transform(sc.transform(F[te]))
            for name, clf in (("1-NN", KNeighborsClassifier(1, metric="cosine")),
                              ("logistic regression", LogisticRegression(C=0.5, max_iter=500))):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    clf.fit(Ztr, y[tr])
                pred = clf.predict(Zte)
                # share of test images with a near-duplicate (same group) in the training set
                share = float(np.isin(g[te], g[tr]).mean())
                rows.append(dict(rep=rep, protocol=proto, classifier=name, n_test=len(te),
                                 accuracy=float((pred == y[te]).mean()),
                                 macro_f1=float(f1_score(y[te], pred, average="macro")),
                                 test_with_neardup_in_train=share))
        print("rep", rep, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(res / "eval_e0_leakage.csv", index=False)
    print(out.groupby(["classifier", "protocol"])[["accuracy", "macro_f1", "test_with_neardup_in_train"]]
          .agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
