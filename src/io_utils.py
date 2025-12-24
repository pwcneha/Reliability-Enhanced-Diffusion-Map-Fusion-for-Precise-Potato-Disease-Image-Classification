from __future__ import annotations
from pathlib import Path
import json
import numpy as np


def load_npz_xy(path: str):
    z = np.load(path, allow_pickle=False)
    if "X" not in z or "y" not in z:
        raise KeyError(f"{path} must contain keys: 'X' and 'y'")
    X = np.asarray(z["X"])
    y = np.asarray(z["y"]).astype(int)
    return X, y


def load_probs_csv(path: str):
    arr = np.loadtxt(path, delimiter=",", dtype=float)
    if arr.ndim == 1:
        # single sample edge case -> (1, C)
        arr = arr[None, :]
    return arr


def save_csv(path: Path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    A = np.asarray(arr, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"save_csv expects 2D array, got shape {A.shape}")
    np.savetxt(str(path), A, delimiter=",")


def save_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
