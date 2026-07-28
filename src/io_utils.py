"""Input validation and output helpers for the public BGF runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_POLICY_KEYS = (
    "development_dmap",
    "heldout_dmap",
    "development_labels",
    "development_source_ids",
    "heldout_source_ids",
    "development_probability_tensor",
    "heldout_probability_tensor",
)


def _clean_ids(values: np.ndarray) -> np.ndarray:
    cleaned = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        cleaned.append(str(value).strip())
    result = np.asarray(cleaned, dtype=str)
    if len(np.unique(result)) != len(result):
        raise ValueError("Source identifiers must be unique within each split.")
    return result


def load_policy_input_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load one fold-local BGF input package.

    Held-out labels are intentionally not part of the accepted schema.
    """

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_POLICY_KEYS if key not in archive]
        if missing:
            raise ValueError(
                f"{path} is missing required keys: {', '.join(missing)}"
            )
        data = {key: np.asarray(archive[key]) for key in archive.files}

    forbidden = {
        "heldout_labels",
        "heldout_y",
        "outer_test_labels",
        "y_test",
        "test_labels",
    }
    present_forbidden = sorted(forbidden.intersection(data))
    if present_forbidden:
        raise ValueError(
            "The policy input contains held-out labels, which are forbidden "
            f"during construction: {present_forbidden}"
        )

    data["development_labels"] = np.asarray(
        data["development_labels"], dtype=int
    )
    data["development_source_ids"] = _clean_ids(
        data["development_source_ids"]
    )
    data["heldout_source_ids"] = _clean_ids(data["heldout_source_ids"])
    data["development_dmap"] = np.asarray(
        data["development_dmap"], dtype=np.float64
    )
    data["heldout_dmap"] = np.asarray(
        data["heldout_dmap"], dtype=np.float64
    )
    data["development_probability_tensor"] = np.asarray(
        data["development_probability_tensor"], dtype=np.float64
    )
    data["heldout_probability_tensor"] = np.asarray(
        data["heldout_probability_tensor"], dtype=np.float64
    )

    n_development = len(data["development_labels"])
    n_heldout = len(data["heldout_source_ids"])
    if len(data["development_source_ids"]) != n_development:
        raise ValueError("Development labels and identifiers do not align.")
    if len(data["development_dmap"]) != n_development:
        raise ValueError("Development labels and DMAP rows do not align.")
    if len(data["development_probability_tensor"]) != n_development:
        raise ValueError(
            "Development labels and probability rows do not align."
        )
    if len(data["heldout_dmap"]) != n_heldout:
        raise ValueError("Held-out identifiers and DMAP rows do not align.")
    if len(data["heldout_probability_tensor"]) != n_heldout:
        raise ValueError(
            "Held-out identifiers and probability rows do not align."
        )

    development_shape = data["development_probability_tensor"].shape
    heldout_shape = data["heldout_probability_tensor"].shape
    if len(development_shape) != 3 or development_shape[1:] != (4, 3):
        raise ValueError(
            "development_probability_tensor must have shape "
            "(n_development, 4, 3)."
        )
    if len(heldout_shape) != 3 or heldout_shape[1:] != (4, 3):
        raise ValueError(
            "heldout_probability_tensor must have shape "
            "(n_heldout, 4, 3)."
        )
    if set(np.unique(data["development_labels"])) - {0, 1, 2}:
        raise ValueError("Development labels must use class IDs 0, 1 and 2.")

    return data


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_table(path: str | Path, table: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
