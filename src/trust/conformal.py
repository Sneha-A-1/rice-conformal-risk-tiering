"""Split conformal prediction for classification.

Scores
  LAC  s(x, y) = 1 - p_y(x)                       (Sadinle et al., 2019)
  APS  s(x, y) = sum of probabilities ranked >= y, randomised (Romano et al., 2020)
Calibration
  marginal        one threshold for all classes          -> P(Y in C) >= 1 - alpha
  class-conditional (Mondrian) one threshold per class   -> P(Y in C | Y = k) >= 1 - alpha_k
"""
import numpy as np


def conformal_quantile(scores, alpha):
    """Finite-sample corrected (1 - alpha) quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


# ---------------- scores ----------------
def lac_scores(p):
    return 1.0 - p


def aps_scores(p, rng, randomized=True):
    """APS conformity score for every class: total mass of classes ranked at or above y,
    minus U * p_y when randomised."""
    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    s_sorted = cum - (rng.uniform(size=(p.shape[0], 1)) * sorted_p if randomized else 0)
    s = np.empty_like(p)
    np.put_along_axis(s, order, s_sorted, axis=1)
    return s


# ---------------- calibration ----------------
def calibrate(score_matrix_cal, y_cal, alpha, mode="marginal", n_classes=None, alpha_per_class=None):
    """Return a per-class threshold vector qhat (length K)."""
    K = n_classes or score_matrix_cal.shape[1]
    true_scores = score_matrix_cal[np.arange(len(y_cal)), y_cal]
    if mode == "marginal":
        return np.full(K, conformal_quantile(true_scores, alpha))
    if mode == "class":
        q = np.empty(K)
        for k in range(K):
            a = alpha if alpha_per_class is None else alpha_per_class[k]
            q[k] = conformal_quantile(true_scores[y_cal == k], a)
        return q
    raise ValueError(mode)


def predict_sets(score_matrix, qhat):
    """Boolean matrix (n, K): class k in C(x) iff s(x, k) <= qhat[k]."""
    return score_matrix <= qhat[None, :]


# ---------------- evaluation ----------------
def set_metrics(sets, y, n_classes):
    cov = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    out = {
        "coverage": float(cov.mean()),
        "avg_set_size": float(size.mean()),
        "singleton_rate": float((size == 1).mean()),
        "empty_rate": float((size == 0).mean()),
    }
    cc = [cov[y == k].mean() for k in range(n_classes) if (y == k).any()]
    out["min_class_coverage"] = float(np.min(cc))
    out["class_coverage_gap"] = float(np.mean(np.abs(np.array(cc) - out["coverage"])))
    return out


def per_class_coverage(sets, y, n_classes):
    cov = sets[np.arange(len(y)), y]
    return np.array([cov[y == k].mean() if (y == k).any() else np.nan for k in range(n_classes)])
