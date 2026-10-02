"""Probability calibration and calibration metrics."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def fit_temperature(logits, y):
    """Temperature T > 0 minimising validation NLL of softmax(logits / T) (Guo et al., 2017)."""
    def nll(logt):
        lp = log_softmax(logits / np.exp(logt), axis=1)
        return -lp[np.arange(len(y)), y].mean()
    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def probs(logits, T=1.0):
    return softmax(logits / T, axis=1)


def ece(p, y, n_bins=15):
    """Expected calibration error (top-label, equal-width bins)."""
    conf = p.max(1)
    acc = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def classwise_ece(p, y, n_bins=15):
    """Mean over classes of the one-vs-rest calibration error."""
    K = p.shape[1]
    out = []
    edges = np.linspace(0, 1, n_bins + 1)
    for k in range(K):
        pk, yk = p[:, k], (y == k).astype(float)
        e = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pk > lo) & (pk <= hi)
            if m.any():
                e += m.mean() * abs(yk[m].mean() - pk[m].mean())
        out.append(e)
    return float(np.mean(out))


def nll(p, y, eps=1e-12):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], eps, 1)).mean())


def brier(p, y):
    oh = np.eye(p.shape[1])[y]
    return float(((p - oh) ** 2).sum(1).mean())


def reliability_curve(p, y, n_bins=15):
    conf = p.max(1)
    acc = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append((lo, hi, int(m.sum()), float(conf[m].mean()), float(acc[m].mean())))
    return rows
