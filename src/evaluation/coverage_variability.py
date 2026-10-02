"""E2b: variability of realised conformal coverage across calibration draws, group-aware vs image-level splits.

For split conformal prediction with n calibration points, the coverage conditional on the calibration set follows
Beta(n + 1 - l, l), l = floor((n + 1) * alpha) (Angelopoulos & Bates, 2023), so its spread across
calibration draws is known for exchangeable data. Near-duplicate groups violate image-level exchangeability; this
script measures how much more the realised coverage fluctuates when calibration and test images are split by group
(as in deployment, where new fields are never near-duplicates of calibration images) than when they are split
image by image, for every edge seed (temperature-scaled, LAC marginal, alpha = 0.1).

Output: results/raw/eval_e2b_split_variability.csv  (seed, mode, split, n_cal, n_test, coverage)
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from trust.calibration import fit_temperature, probs  # noqa: E402
from trust.conformal import calibrate, lac_scores, predict_sets  # noqa: E402


def main(n_splits=200, alpha=0.1):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    classes = cfg["classes"]
    res = pathlib.Path(cfg["paths"]["results"]) / "raw"
    df = pd.read_csv(pathlib.Path(cfg["paths"]["processed"]) / "paddy_index.csv")
    y = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    rows = []
    for seed in cfg["train"]["seeds"]:
        d = res / "models" / f"seed{seed}"
        lv, lh = np.load(d / "logits_val.npy"), np.load(d / "logits_holdout.npy")
        iv, ih = np.load(d / "index_val.npy"), np.load(d / "index_holdout.npy")
        p = probs(lh, fit_temperature(lv, y[iv]))
        yh, g = y[ih], df.group.to_numpy()[ih]
        ug = np.unique(g)
        rng = np.random.default_rng(5000 + seed)
        for mode in ("group-aware", "image-level"):
            for s in range(n_splits):
                if mode == "group-aware":
                    cg = set(rng.choice(ug, len(ug) // 2, replace=False))
                    cal = np.array([x in cg for x in g])
                else:
                    cal = np.zeros(len(yh), bool)
                    cal[rng.choice(len(yh), len(yh) // 2, replace=False)] = True
                q = calibrate(lac_scores(p[cal]), yh[cal], alpha)
                S = predict_sets(lac_scores(p[~cal]), q)
                rows.append({"seed": seed, "mode": mode, "split": s, "n_cal": int(cal.sum()), "n_test": int((~cal).sum()),
                             "coverage": float(S[np.arange((~cal).sum()), yh[~cal]].mean())})
    out = pd.DataFrame(rows)
    out.to_csv(res / "eval_e2b_split_variability.csv", index=False)
    print(out.groupby(["seed", "mode"]).coverage.agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
