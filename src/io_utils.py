from pathlib import Path
import json
import numpy as np


def load_npz_xy(path):
    data = np.load(path)
    if "X" not in data or "y" not in data:
        raise ValueError(f"{path} must contain keys 'X' and 'y'")
    return data["X"], data["y"].astype(int)


def load_probs_csv(path):
    arr = np.loadtxt(path, delimiter=",", dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def save_csv(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    np.savetxt(path, arr, delimiter=",")


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

