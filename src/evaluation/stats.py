"""Statistical helpers: cluster bootstrap, paired tests with Holm correction, effect sizes."""
import numpy as np
from scipy.stats import wilcoxon


def cluster_bootstrap(metric_fn, groups, B=2000, seed=0, ci=0.95):
    """Percentile CI for metric_fn(idx) when observations are clustered (Field & Welsh, 2007).
    Resamples whole clusters (near-duplicate groups) with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inv == g)[0] for g in range(len(uniq))]
    stats = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        stats.append(metric_fn(idx))
    lo, hi = np.percentile(stats, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(metric_fn(np.arange(len(groups)))), float(lo), float(hi)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for r, i in enumerate(order):
        running = max(running, min(1.0, (m - r) * p[i]))
        adj[i] = running
    return adj


def paired_tests(proposed, baselines: dict, lower_is_better=True):
    """Wilcoxon signed-rank (two-sided) proposed vs each baseline on paired replications,
    Holm-adjusted; effect size = matched-pairs rank-biserial correlation and Cliff's delta."""
    rows, pv = [], []
    x = np.asarray(proposed, float)
    for name, b in baselines.items():
        y = np.asarray(b, float)
        d = x - y
        if np.allclose(d, 0):
            p, rbc = 1.0, 0.0
        else:
            res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            p = float(res.pvalue)
            nz = d[d != 0]
            ranks = np.argsort(np.argsort(np.abs(nz))) + 1
            rbc = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
        rows.append({"baseline": name, "median_diff": float(np.median(d)), "p_value": p,
                     "rank_biserial": rbc, "cliffs_delta": cliffs_delta(x, y)})
        pv.append(p)
    for r, a in zip(rows, holm(pv)):
        r["p_holm"] = float(a)
    return rows


def mean_sd(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def t_ci(x, ci=0.95):
    from scipy.stats import t
    x = np.asarray(x, float)
    m, s = x.mean(), x.std(ddof=1)
    h = t.ppf((1 + ci) / 2, len(x) - 1) * s / np.sqrt(len(x))
    return float(m), float(m - h), float(m + h)


def corrected_resampled_t(diff, n_test, n_train):
    """Corrected resampled t-test (Nadeau & Bengio, 2003) for J paired differences obtained from J random
    train/test (here calibration/test) splits of the same data: the variance is inflated by (1/J + n_test/n_train)
    to account for the overlap between splits. Returns (mean, t, two-sided p)."""
    from scipy.stats import t as tdist
    d = np.asarray(diff, float)
    J = len(d)
    m, v = d.mean(), d.var(ddof=1)
    if v == 0:
        return float(m), 0.0, 1.0
    tt = m / np.sqrt((1.0 / J + n_test / n_train) * v)
    return float(m), float(tt), float(2 * tdist.sf(abs(tt), J - 1))


def corrected_resampled_ci(diff, n_test, n_train, ci=0.95):
    """Confidence interval for the mean of J resampled paired differences with the Nadeau-Bengio variance correction."""
    from scipy.stats import t as tdist
    d = np.asarray(diff, float)
    J = len(d)
    m = d.mean()
    h = tdist.ppf((1 + ci) / 2, J - 1) * np.sqrt((1.0 / J + n_test / n_train) * d.var(ddof=1))
    return float(m), float(m - h), float(m + h)
