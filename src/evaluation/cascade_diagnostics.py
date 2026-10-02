"""E3b: what the conformal tier rule escalates.

Over 100 random group-aware calibration/test splits of the hold-out partition (edge seed = split mod 5,
temperature-scaled, class-conditional LAC, alpha = 0.1, as in the cascade experiment), records the share of escalated
images that are URGENT, the edge model's accuracy on URGENT, STRATEGIC and ROUTINE images, and the share of URGENT
images whose prediction set is a single (high-severity) class.

Output: results/raw/eval_e3b_tier_diagnostics.csv
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.special import softmax

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from trust.calibration import fit_temperature  # noqa: E402
from trust.conformal import calibrate, lac_scores, predict_sets  # noqa: E402


def main(n_splits=100):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    classes = cfg["classes"]
    K = len(classes)
    H = [classes.index(c) for c in cfg["risk"]["high_severity"]]
    alpha = cfg["risk"]["alpha"]
    res = pathlib.Path(cfg["paths"]["results"]) / "raw"
    df = pd.read_csv(pathlib.Path(cfg["paths"]["processed"]) / "paddy_index.csv")
    y = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    P = []
    for s in cfg["train"]["seeds"]:
        d = res / "models" / f"seed{s}"
        lv, lh = np.load(d / "logits_val.npy"), np.load(d / "logits_holdout.npy")
        iv, ih = np.load(d / "index_val.npy"), np.load(d / "index_holdout.npy")
        P.append(softmax(lh / fit_temperature(lv, y[iv]), axis=1))
    yh, g = y[ih], df.group.to_numpy()[ih]
    ug = np.unique(g)
    rng = np.random.default_rng(7000)
    rows = []
    for r in range(n_splits):
        cg = set(rng.choice(ug, len(ug) // 2, replace=False))
        cal = np.array([x in cg for x in g])
        p = P[r % len(P)]
        q = calibrate(lac_scores(p[cal]), yh[cal], alpha, "class", K)
        S = predict_sets(lac_scores(p[~cal]), q)
        yt, pred = yh[~cal], p[~cal].argmax(1)
        urg = S[:, H].any(1)
        size = S.sum(1)
        strat = ~urg & (size != 1)
        rout = ~urg & ~strat
        esc = urg | strat
        rows.append({"split": r, "seed": r % len(P), "share_escalated_urgent": float(urg.sum() / max(esc.sum(), 1)),
                     "edge_acc_urgent": float((pred[urg] == yt[urg]).mean()),
                     "edge_acc_strategic": float((pred[strat] == yt[strat]).mean()) if strat.any() else np.nan,
                     "edge_acc_routine": float((pred[rout] == yt[rout]).mean()),
                     "urgent_singleton_share": float((size[urg] == 1).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(res / "eval_e3b_tier_diagnostics.csv", index=False)
    print(out.mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
