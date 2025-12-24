import numpy as np


def nll(P, y, eps=1e-12):
    P = np.clip(P, eps, 1.0)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def accuracy(P, y):
    return float(np.mean(P.argmax(1) == y))


def brier(P, y):
    N, C = P.shape
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def ece(P, y, bins=15):
    conf = P.max(1)
    pred = P.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)

    ece_val = 0.0
    for i in range(bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.any():
            ece_val += abs(acc[mask].mean() - conf[mask].mean()) * mask.mean()
    return float(ece_val)


def metrics_dict(P, y):
    return {
        "accuracy": accuracy(P, y),
        "nll": nll(P, y),
        "brier": brier(P, y),
        "ece": ece(P, y),
    }


def mcnemar_exact(yhat_base, yhat_fuse, ytrue):
    base_correct = (yhat_base == ytrue)
    fuse_correct = (yhat_fuse == ytrue)

    n01 = int((base_correct & ~fuse_correct).sum())
    n10 = int((~base_correct & fuse_correct).sum())

    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10

    from math import comb
    k = min(n01, n10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p), n01, n10


def bootstrap_delta_nll_ci(P_base, P_fused, y, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        deltas.append(nll(P_fused[idx], y[idx]) - nll(P_base[idx], y[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"mean": float(np.mean(deltas)), "ci95": [float(lo), float(hi)]}


