"""Path-independent implementation of the locked Budget-Gated Fusion policy.

This module extracts the policy logic used in the final group-aware analysis.
Private storage paths, experiment-stage identifiers, immutable marker handling,
and project-specific file orchestration have been removed. The regional
selection, eligibility conditions, candidate definitions, deterministic
ranking rules, and per-fold intervention ceiling are preserved.

Held-out labels are not accepted by the policy-construction or gate-application
functions. Development labels are used only to build regional reliability
profiles and to fit optional post-policy temperature scaling.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans
from sklearn.metrics import log_loss


DEFAULT_EXPERT_NAMES = (
    "Snapshot MLP ensemble",
    "Transformer ensemble",
    "DeiT-III Small",
)


@dataclass(frozen=True)
class BGFConfig:
    """Locked BGF-v1 configuration used in the reported analysis."""

    n_regions: int = 3
    minimum_region_n: int = 10
    minimum_nll_gain: float = 0.003
    maximum_ece_drift: float = 0.015
    minimum_expert_margin: float = 0.08
    maximum_edit_rate: float = 0.05
    random_seed: int = 42
    kmeans_n_init: int = 50
    kmeans_max_iter: int = 1000


def normalise(probabilities: np.ndarray) -> np.ndarray:
    """Return finite row-normalised class probabilities."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional probability array, got "
            f"{probabilities.shape}."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probability encountered.")
    probabilities = np.clip(probabilities, 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _normalise_tensor(probability_tensor: np.ndarray) -> np.ndarray:
    probability_tensor = np.asarray(probability_tensor, dtype=np.float64)
    if probability_tensor.ndim != 3:
        raise ValueError(
            "Expected probability_tensor with shape "
            "(n_images, n_models, n_classes)."
        )
    return np.stack(
        [
            normalise(probability_tensor[:, model_index, :])
            for model_index in range(probability_tensor.shape[1])
        ],
        axis=1,
    )


def ece_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> float:
    """Compute equal-width expected calibration error."""

    probabilities = normalise(probabilities)
    y_true = np.asarray(y_true, dtype=int)
    if len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities have different lengths.")

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


def fit_temperature(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    lower: float = 0.5,
    upper: float = 10.0,
) -> tuple[float, str]:
    """Fit a scalar temperature using development predictions only."""

    probabilities = normalise(probabilities)
    y_true = np.asarray(y_true, dtype=int)
    if np.all(probabilities.argmax(axis=1) == y_true):
        return 1.0, "Perfect development prediction; neutral temperature"

    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))

    def objective(log_temperature: float) -> float:
        logits = log_probabilities / np.exp(log_temperature)
        logits -= logits.max(axis=1, keepdims=True)
        scaled = np.exp(logits)
        scaled /= scaled.sum(axis=1, keepdims=True)
        return float(log_loss(y_true, scaled, labels=[0, 1, 2]))

    result = minimize_scalar(
        objective,
        bounds=(np.log(lower), np.log(upper)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    temperature = float(np.exp(result.x))
    tolerance = 1e-4
    if temperature <= lower + tolerance:
        status = "Lower-bound development solution"
    elif temperature >= upper - tolerance:
        status = "Upper-bound development solution"
    else:
        status = "Interior development solution"
    return temperature, status


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply scalar temperature scaling without changing the gate ledger."""

    probabilities = normalise(probabilities)
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    logits /= float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    return normalise(np.exp(logits))


def build_regions(
    development_dmap: np.ndarray,
    heldout_dmap: np.ndarray,
    config: BGFConfig = BGFConfig(),
) -> tuple[np.ndarray, np.ndarray, KMeans, float]:
    """Fit fold-local regions on development DMAP coordinates.

    The first two fold-specific DMAP coordinates are divided by one global
    development RMS scalar before k-means fitting. The fitted model is then
    applied unchanged to held-out coordinates.
    """

    development_dmap = np.asarray(development_dmap, dtype=np.float64)
    heldout_dmap = np.asarray(heldout_dmap, dtype=np.float64)
    if development_dmap.ndim != 2 or development_dmap.shape[1] < 2:
        raise ValueError("development_dmap must contain at least two columns.")
    if heldout_dmap.ndim != 2 or heldout_dmap.shape[1] < 2:
        raise ValueError("heldout_dmap must contain at least two columns.")

    development_leading = development_dmap[:, :2]
    heldout_leading = heldout_dmap[:, :2]
    global_rms = float(np.sqrt(np.mean(development_leading**2)))
    if not np.isfinite(global_rms) or global_rms <= 0:
        raise ValueError("Invalid development DMAP global RMS.")

    development_coordinates = development_leading / global_rms
    heldout_coordinates = heldout_leading / global_rms
    model = KMeans(
        n_clusters=config.n_regions,
        random_state=config.random_seed,
        n_init=config.kmeans_n_init,
        max_iter=config.kmeans_max_iter,
    )
    model.fit(development_coordinates)
    return (
        model.predict(development_coordinates).astype(np.int16),
        model.predict(heldout_coordinates).astype(np.int16),
        model,
        global_rms,
    )


def build_regional_policy(
    development_labels: np.ndarray,
    development_probability_tensor: np.ndarray,
    development_regions: np.ndarray,
    config: BGFConfig = BGFConfig(),
    expert_names: Sequence[str] = DEFAULT_EXPERT_NAMES,
) -> tuple[dict[int, dict | None], pd.DataFrame]:
    """Build a regional expert policy from development predictions.

    Model index zero is the designated baseline. Remaining model indices are
    experts in the same order as ``expert_names``.
    """

    y_true = np.asarray(development_labels, dtype=int)
    tensor = _normalise_tensor(development_probability_tensor)
    regions = np.asarray(development_regions, dtype=int)

    if len(y_true) != len(tensor) or len(regions) != len(tensor):
        raise ValueError(
            "Development labels, probabilities, and regions must align."
        )
    if tensor.shape[1] != len(expert_names) + 1:
        raise ValueError(
            "The probability tensor must contain one baseline followed by "
            "one stream for each expert name."
        )

    baseline = tensor[:, 0, :]
    policy: dict[int, dict | None] = {}
    audit_rows: list[dict] = []

    for region_id in range(config.n_regions):
        mask = regions == region_id
        region_n = int(mask.sum())
        if region_n == 0:
            policy[region_id] = None
            continue

        baseline_nll = float(
            log_loss(y_true[mask], baseline[mask], labels=[0, 1, 2])
        )
        baseline_ece = ece_score(y_true[mask], baseline[mask])
        expert_rows = []

        for expert_index, expert_name in enumerate(expert_names, start=1):
            expert = tensor[mask, expert_index, :]
            expert_nll = float(
                log_loss(y_true[mask], expert, labels=[0, 1, 2])
            )
            expert_ece = ece_score(y_true[mask], expert)
            nll_gain = float(baseline_nll - expert_nll)
            ece_drift = float(expert_ece - baseline_ece)
            eligible = bool(
                region_n >= config.minimum_region_n
                and nll_gain >= config.minimum_nll_gain
                and ece_drift <= config.maximum_ece_drift
            )
            row = {
                "Region_ID": region_id,
                "Region_N": region_n,
                "Expert_Index": expert_index,
                "Expert_Name": expert_name,
                "Base_NLL": baseline_nll,
                "Base_ECE": baseline_ece,
                "Expert_NLL": expert_nll,
                "Expert_ECE": expert_ece,
                "Regional_NLL_Gain": nll_gain,
                "Regional_ECE_Drift": ece_drift,
                "Eligible": eligible,
            }
            expert_rows.append(row)
            audit_rows.append(row.copy())

        admissible = [row for row in expert_rows if row["Eligible"]]
        if admissible:
            selected = sorted(
                admissible,
                key=lambda row: (
                    -row["Regional_NLL_Gain"],
                    row["Regional_ECE_Drift"],
                    row["Expert_Name"],
                ),
            )[0]
            region_eligible = True
        else:
            selected = sorted(
                expert_rows,
                key=lambda row: (
                    row["Expert_NLL"],
                    row["Regional_ECE_Drift"],
                    row["Expert_Name"],
                ),
            )[0]
            region_eligible = False

        policy[region_id] = {
            "region_id": region_id,
            "region_n": region_n,
            "expert_index": int(selected["Expert_Index"]),
            "expert_name": selected["Expert_Name"],
            "nll_gain": float(selected["Regional_NLL_Gain"]),
            "ece_drift": float(selected["Regional_ECE_Drift"]),
            "eligible": bool(region_eligible),
        }

    return policy, pd.DataFrame(audit_rows)


def apply_regional_gate(
    probability_tensor: np.ndarray,
    regions: np.ndarray,
    source_ids: Sequence[str],
    policy: Mapping[int, dict | None],
    gate_version: str,
    config: BGFConfig = BGFConfig(),
) -> dict:
    """Apply one locked BGF policy without using held-out labels.

    Parameters
    ----------
    probability_tensor:
        Array shaped ``(n_images, 1 + n_experts, n_classes)``. Model index zero
        is the designated baseline.
    regions:
        Region assignments produced by the development-fitted k-means model.
    source_ids:
        Stable identifiers used only for deterministic tie-breaking.
    policy:
        Output of :func:`build_regional_policy`.
    gate_version:
        ``"as_executed"`` or ``"equation_consistent"``.
    """

    tensor = _normalise_tensor(probability_tensor)
    regions = np.asarray(regions, dtype=int)
    source_ids = np.asarray(source_ids).astype(str)
    if len(tensor) != len(regions) or len(tensor) != len(source_ids):
        raise ValueError(
            "Probabilities, regions, and source identifiers must align."
        )

    version = gate_version.strip().lower().replace("-", "_")
    if version not in {"as_executed", "equation_consistent"}:
        raise ValueError(
            "gate_version must be 'as_executed' or "
            "'equation_consistent'."
        )

    baseline = tensor[:, 0, :]
    baseline_prediction = baseline.argmax(axis=1)
    candidates: list[dict] = []

    for sample_index in range(len(baseline)):
        rule = policy.get(int(regions[sample_index]))
        if rule is None or not rule["eligible"]:
            continue

        expert_index = int(rule["expert_index"])
        expert = tensor[sample_index, expert_index, :]
        expert_prediction = int(expert.argmax())
        if expert_prediction == int(baseline_prediction[sample_index]):
            continue

        sorted_expert = np.sort(expert)[::-1]
        equation_margin = float(sorted_expert[0] - sorted_expert[1])
        implemented_margin = float(
            expert[expert_prediction]
            - baseline[sample_index, expert_prediction]
        )
        baseline_confidence = float(baseline[sample_index].max())
        equation_priority = float(
            rule["nll_gain"]
            * equation_margin
            * (1.0 - baseline_confidence)
        )

        if version == "as_executed":
            if implemented_margin < config.minimum_expert_margin:
                continue
            active_priority = implemented_margin
        else:
            if equation_margin < config.minimum_expert_margin:
                continue
            active_priority = equation_priority

        candidates.append(
            {
                "Sample_Index": sample_index,
                "Source_ID": source_ids[sample_index],
                "Region_ID": int(regions[sample_index]),
                "Selected_Expert": rule["expert_name"],
                "Expert_Index": expert_index,
                "Base_Prediction": int(
                    baseline_prediction[sample_index]
                ),
                "Expert_Prediction": expert_prediction,
                "Base_Confidence": baseline_confidence,
                "Expert_Confidence": float(expert.max()),
                "Implemented_Margin": implemented_margin,
                "Equation_Margin": equation_margin,
                "Regional_NLL_Gain": float(rule["nll_gain"]),
                "Regional_ECE_Drift": float(rule["ece_drift"]),
                "Equation_Priority": equation_priority,
                "Active_Priority": active_priority,
                "Gate_Version": version,
            }
        )

    if version == "as_executed":
        candidates.sort(
            key=lambda row: (
                -row["Implemented_Margin"],
                row["Sample_Index"],
            )
        )
    else:
        candidates.sort(
            key=lambda row: (
                -row["Equation_Priority"],
                row["Source_ID"],
            )
        )

    maximum_edits = int(
        np.floor(config.maximum_edit_rate * len(baseline))
    )
    accepted = candidates[:maximum_edits]
    output = baseline.copy()
    accepted_mask = np.zeros(len(baseline), dtype=bool)
    selected_expert_indices = np.full(
        len(baseline), -1, dtype=np.int16
    )

    for accepted_rank, row in enumerate(accepted, start=1):
        sample_index = int(row["Sample_Index"])
        expert_index = int(row["Expert_Index"])
        output[sample_index] = tensor[sample_index, expert_index, :]
        accepted_mask[sample_index] = True
        selected_expert_indices[sample_index] = expert_index
        row["Accepted"] = True
        row["Accepted_Rank"] = accepted_rank

    accepted_indices = {
        int(row["Sample_Index"]) for row in accepted
    }
    for candidate_rank, row in enumerate(candidates, start=1):
        row["Candidate_Rank"] = candidate_rank
        if int(row["Sample_Index"]) not in accepted_indices:
            row["Accepted"] = False
            row["Accepted_Rank"] = np.nan

    return {
        "probabilities": normalise(output),
        "candidate_table": pd.DataFrame(candidates),
        "accepted_mask": accepted_mask,
        "selected_expert_indices": selected_expert_indices,
        "candidate_n": len(candidates),
        "accepted_n": int(accepted_mask.sum()),
        "maximum_edits": maximum_edits,
    }
