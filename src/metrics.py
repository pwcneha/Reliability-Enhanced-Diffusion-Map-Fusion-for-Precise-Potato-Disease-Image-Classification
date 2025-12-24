from __future__ import annotations
import numpy as np


def nll(P, y, eps: float = 1e-12) -> float:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    P = np.clip(P, eps, 1.0)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def accuracy(P, y) -> float:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    return float(np.mean(P.argmax(axis=1) == y))


def brier(P, y) -> float:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    N, C = P.shape
    Y = np.zeros((N, C), dtype=float)
    Y[np.arange(N), y] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def ece(P, y, n_bins: int = 15) -> float:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    acc = (pred == y).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i < n_bins - 1 else (conf > lo) & (conf <= hi + 1e-9)
        if np.any(m):
            ece_val += float(np.abs(acc[m].mean() - conf[m].mean()) * (m.mean()))
    return float(ece_val)


def metrics_dict(P, y) -> dict:
    return {
        "acc": accuracy(P, y),
        "nll": nll(P, y),
        "brier": brier(P, y),
        "ece": ece(P, y),
    }


def mcnemar_exact(yhat_a, yhat_b, ytrue):
    yhat_a = np.asarray(yhat_a)
    yhat_b = np.asarray(yhat_b)
    ytrue = np.asarray(ytrue)

    a_correct = (yhat_a == ytrue)
    b_correct = (yhat_b == ytrue)

    n01 = int(np.sum(a_correct & (~b_correct)))
    n10 = int(np.sum((~a_correct) & b_correct))

    # Exact binomial test (two-sided) on min(n01,n10) with p=0.5, n=n01+n10
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10

    k = min(n01, n10)

    # compute 2*sum_{i=0..k} C(n,i) / 2^n
    from math import comb
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    p = min(1.0, float(p))
    return p, n01, n10


def bootstrap_delta_nll_ci(P_base, P_fused, y, B: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    P_base = np.asarray(P_base, dtype=float)
    P_fused = np.asarray(P_fused, dtype=float)
    y = np.asarray(y, dtype=int)
    N = len(y)

    deltas = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        deltas[b] = nll(P_fused[idx], y[idx]) - nll(P_base[idx], y[idx])

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_nll_mean": float(deltas.mean()), "ci95": [float(lo), float(hi)]}
