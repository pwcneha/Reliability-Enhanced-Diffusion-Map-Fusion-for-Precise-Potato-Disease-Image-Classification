import numpy as np
from math import comb


def nll(P, y):
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    r = np.arange(P.shape[0])
    return float(-np.log(np.clip(P[r, y], 1e-12, None)).mean())


def accuracy(P, y):
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    return float((P.argmax(1) == y).mean())


def brier(P, y):
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    C = P.shape[1]
    Y = np.eye(C)[y]
    return float(np.mean(((P - Y) ** 2).sum(1) / C))


def ece(P, y, n_bins=15):
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    conf = P.max(1)
    pred = P.argmax(1)
    corr = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1] + (1e-9 if i == n_bins - 1 else 0.0)
        m = (conf >= lo) & (conf < hi)
        if m.any():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def metrics_dict(P, y):
    return {
        "acc": accuracy(P, y),
        "nll": nll(P, y),
        "brier": brier(P, y),
        "ece": ece(P, y),
    }


def mcnemar_exact(yhat_base, yhat_fuse, y_true):
    """
    Exact McNemar test on discordant pairs.
    Returns dict with p_value, n01, n10, discordant.
    """
    yhat_base = np.asarray(yhat_base, dtype=int)
    yhat_fuse = np.asarray(yhat_fuse, dtype=int)
    y_true = np.asarray(y_true, dtype=int)

    b_correct = (yhat_base == y_true)
    f_correct = (yhat_fuse == y_true)

    n01 = int(np.sum(b_correct & (~f_correct)))  # base right, fuse wrong
    n10 = int(np.sum((~b_correct) & f_correct))  # base wrong, fuse right
    n = n01 + n10

    if n == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        tail = sum(comb(n, i) for i in range(0, k + 1))
        p = min(1.0, 2.0 * tail * (0.5 ** n))

    return {
        "p_value": float(p),
        "n01_base_correct_fuse_wrong": int(n01),
        "n10_base_wrong_fuse_correct": int(n10),
        "discordant": int(n),
    }


def bootstrap_delta_nll_ci(P_a, P_b, y, seed=42, n_boot=800):
    """
    ΔNLL = NLL(P_a) - NLL(P_b)
    Returns mean + 95% CI.
    """
    P_a = np.asarray(P_a, dtype=float)
    P_b = np.asarray(P_b, dtype=float)
    y = np.asarray(y, dtype=int)

    rng = np.random.default_rng(int(seed))
    N = P_a.shape[0]
    idx_all = np.arange(N)

    deltas = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx = rng.choice(idx_all, size=N, replace=True)
        deltas[i] = nll(P_a[idx], y[idx]) - nll(P_b[idx], y[idx])

    deltas.sort()
    lo = float(deltas[int(0.025 * n_boot)])
    hi = float(deltas[int(0.975 * n_boot)])
    return {
        "delta_nll_mean": float(deltas.mean()),
        "ci95": [lo, hi],
        "definition": "ΔNLL = NLL(A) - NLL(B)",
    }


