"""Metrics used for locked post-policy evaluation."""

from __future__ import annotations

from math import comb

import numpy as np
from sklearn.metrics import f1_score, log_loss


def normalise(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a two-dimensional array.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probability encountered.")
    probabilities = np.clip(probabilities, 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def accuracy(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    probabilities = normalise(probabilities)
    y_true = np.asarray(y_true, dtype=int)
    return float(np.mean(probabilities.argmax(axis=1) == y_true))


def macro_f1(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    probabilities = normalise(probabilities)
    y_true = np.asarray(y_true, dtype=int)
    return float(
        f1_score(
            y_true,
            probabilities.argmax(axis=1),
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )


def nll(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    probabilities = normalise(probabilities)
    y_true = np.asarray(y_true, dtype=int)
    return float(log_loss(y_true, probabilities, labels=[0, 1, 2]))


def ece(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    bins: int = 15,
) -> float:
    probabilities = normalise(probabilities)
    y_true = np.asarray(y_true, dtype=int)
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0

    for bin_index in range(bins):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]
        if bin_index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            bin_accuracy = (prediction[mask] == y_true[mask]).mean()
            result += mask.mean() * abs(
                bin_accuracy - confidence[mask].mean()
            )
    return float(result)


def normalised_multiclass_brier(
    probabilities: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Return squared probability error averaged over images and classes."""

    probabilities = normalise(probabilities)
    y_true = np.asarray(y_true, dtype=int)
    n_images, n_classes = probabilities.shape
    one_hot = np.eye(n_classes, dtype=np.float64)[y_true]
    return float(
        np.sum((probabilities - one_hot) ** 2)
        / (n_images * n_classes)
    )


def metrics_dict(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    ece_bins: int = 15,
) -> dict[str, float]:
    return {
        "accuracy": accuracy(probabilities, y_true),
        "macro_f1": macro_f1(probabilities, y_true),
        "nll": nll(probabilities, y_true),
        f"ece_{ece_bins}": ece(probabilities, y_true, bins=ece_bins),
        "brier_norm": normalised_multiclass_brier(
            probabilities, y_true
        ),
    }


def mcnemar_exact(
    baseline_prediction: np.ndarray,
    comparator_prediction: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float | int]:
    """Exact two-sided McNemar test from paired hard predictions."""

    baseline_prediction = np.asarray(baseline_prediction, dtype=int)
    comparator_prediction = np.asarray(comparator_prediction, dtype=int)
    y_true = np.asarray(y_true, dtype=int)
    baseline_correct = baseline_prediction == y_true
    comparator_correct = comparator_prediction == y_true

    harmful = int((baseline_correct & ~comparator_correct).sum())
    corrective = int((~baseline_correct & comparator_correct).sum())
    discordant = harmful + corrective
    if discordant == 0:
        p_value = 1.0
    else:
        lower_tail = min(harmful, corrective)
        p_value = min(
            1.0,
            2.0
            * sum(
                comb(discordant, index)
                for index in range(lower_tail + 1)
            )
            / (2**discordant),
        )
    return {
        "corrective": corrective,
        "harmful": harmful,
        "discordant": discordant,
        "p_value": float(p_value),
    }
