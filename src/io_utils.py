# src/io_utils.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np


def load_npz_xy(path: str):
    path = str(path)
    data = np.load(path, allow_pickle=False)
    if "X" not in data or "y" not in data:
        raise KeyError(f"{path} must contain keys 'X' and 'y'. Found: {list(data.keys())}")
    X = data["X"]
    y = data["y"]
    return X, y


def load_probs_csv(path: str):
    """
    Loads a CSV containing probabilities of shape (N, C).
    Supports optional header. Uses numpy only (no pandas dependency).
    """
    path = str(path)
    try:
        P = np.loadtxt(path, delimiter=",")
        # If it accidentally loaded a single row as 1D, force 2D
        if P.ndim == 1:
            P = P.reshape(1, -1)
        return P
    except ValueError:
        # likely header exists
        P = np.loadtxt(path, delimiter=",", skiprows=1)
        if P.ndim == 1:
            P = P.reshape(1, -1)
        return P


def save_csv(path: str | Path, arr):
    path = Path(path)
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"save_csv expected 2D array, got shape {arr.shape}")
    np.savetxt(str(path), arr.astype(float), delimiter=",", fmt="%.10f")


def save_json(path: str | Path, obj):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
