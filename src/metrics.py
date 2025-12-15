import numpy as np
from math import comb

# ------------------------- basic helpers -------------------------
def _clip_probs(P, eps=1e-12):
    P = np.asarray(P, dtype=float)
    P = np.clip(P, eps, None)
    P = P / np.clip(P.sum(axis=1, keepdims=True), eps, None)
    return P

def accuracy(P, y):
    P = _clip_probs(P)
    y = np.asarray(y, dtype=int)
    return float((P.argmax(1) == y).mean())

def nll(P, y, eps=1e-12):
    P = _clip_probs(P, eps=eps)
    y = np.asarray(y, dtype=int)
    rows = np.arange(P.shape[0])
    return float(-np.log(np.clip(P[rows, y], eps, None)).mean())

def brier(P, y):
    P = _clip_probs(P)
    y = np.asarray(y, dtype=int)
    C = P.shape[1]
    Y = np.eye(C)[y]
    return float(np.mean(((P - Y) ** 2).sum(axis=1) / C))

def ece(P, y, n_bins=15):
    P = _clip_probs(P)
    y = np.asarray(y, dtype=int)

    conf = P.max(1)
    pred = P.argmax(1)
    correct = (pred == y).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1] + (1e-9 if i == n_bins - 1 else 0.0)
        m = (conf >= lo) & (conf < hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)

# ------------------------- metrics bundle -------------------------
def metrics_dict(P, y, n_bins=15):
    """
    Returns a dict of common metrics for probability predictions.
    P: (N,C) probabilities
    y: (N,) integer labels
    """
    P = _clip_probs(P)
    y = np.asarray(y, dtype=int)
    return {
        "acc": accuracy(P, y),
        "nll": nll(P, y),
        "brier": brier(P, y),
        "ece": ece(P, y, n_bins=n_bins),
        "n": int(len(y)),
        "c": int(P.shape[1]),
    }

# ------------------------- statistical tests -------------------------
def mcnemar_exact(y_pred_a, y_pred_b, y_true):
    """
    Exact two-sided McNemar test p-value based on discordant pairs.
    y_pred_a / y_pred_b: (N,) predicted labels from method A/B
    y_true: (N,) true labels
    Returns: (p_value, n01, n10)
      n01 = A correct, B wrong
      n10 = A wrong, B correct
    """
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)
    y_true   = np.asarray(y_true, dtype=int)

    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)

    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10

    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(0, k + 1))
    p = 2.0 * tail * (0.5 ** n)
    p = float(min(1.0, p))
    return p, n01, n10

def bootstrap_delta_nll_ci(P_a, P_b, y, n_boot=800, seed=42):
    """
    Bootstrap CI for ΔNLL = NLL(B) - NLL(A)
    Negative mean => B better than A.
    Returns: dict(mean, ci95=[lo,hi])
    """
    P_a = _clip_probs(P_a)
    P_b = _clip_probs(P_b)
    y = np.asarray(y, dtype=int)

    rng = np.random.default_rng(seed)
    N = len(y)
    idx_all = np.arange(N)

    deltas = []
    for _ in range(int(n_boot)):
        idx = rng.choice(idx_all, size=N, replace=True)
        deltas.append(nll(P_b[idx], y[idx]) - nll(P_a[idx], y[idx]))

    deltas = np.sort(np.asarray(deltas, dtype=float))
    lo = float(deltas[int(0.025 * len(deltas))])
    hi = float(deltas[int(0.975 * len(deltas))])
    return {"mean": float(deltas.mean()), "ci95": [lo, hi]}
