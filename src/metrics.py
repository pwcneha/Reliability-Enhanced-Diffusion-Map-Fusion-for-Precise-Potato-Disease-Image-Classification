# src/metrics.py
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)


def nll(P: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    P = np.clip(P, eps, 1.0)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def metrics_dict(P: np.ndarray, y: np.ndarray) -> dict:
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    yhat = P.argmax(axis=1)

    out = {
        "accuracy": float(accuracy_score(y, yhat)),
        "macro_f1": float(f1_score(y, yhat, average="macro")),
        "kappa": float(cohen_kappa_score(y, yhat)),
        "nll": float(nll(P, y)),
        "confusion_matrix": confusion_matrix(y, yhat).tolist(),
        "classification_report": classification_report(y, yhat, digits=4),
    }
    return out


def mcnemar_exact(yhat_a: np.ndarray, yhat_b: np.ndarray, y: np.ndarray):
    """
    Exact McNemar test using binomial test on discordant pairs.
    Returns: (p_value, n01, n10)
    where
      n01 = a correct, b wrong
      n10 = a wrong, b correct
    """
    yhat_a = np.asarray(yhat_a, dtype=int)
    yhat_b = np.asarray(yhat_b, dtype=int)
    y = np.asarray(y, dtype=int)

    a_correct = (yhat_a == y)
    b_correct = (yhat_b == y)

    n01 = int(np.sum(a_correct & (~b_correct)))
    n10 = int(np.sum((~a_correct) & b_correct))
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10

    # two-sided exact binomial p-value under p=0.5
    from math import comb
    k = min(n01, n10)
    p = 0.0
    for i in range(0, k + 1):
        p += comb(n, i) * (0.5 ** n)
    p_value = min(1.0, 2.0 * p)
    return float(p_value), n01, n10


def bootstrap_delta_nll_ci(P_base, P_fused, y, n_boot=2000, seed=42):
    """
    CI for delta NLL = (fused - base). Negative is better.
    """
    rng = np.random.default_rng(seed)
    P_base = np.asarray(P_base, float)
    P_fused = np.asarray(P_fused, float)
    y = np.asarray(y, int)

    N = len(y)
    deltas = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, N, size=N)
        deltas.append(nll(P_fused[idx], y[idx]) - nll(P_base[idx], y[idx]))

    deltas = np.asarray(deltas, float)
    return {
        "delta_nll_mean": float(deltas.mean()),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }
