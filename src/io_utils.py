import json
import numpy as np
from pathlib import Path


def load_npz_xy(npz_path: str):
    """
    Load DMAP feature file (.npz).

    Expected keys:
      - 'X': array of shape (N, d)
      - 'y': array of shape (N,)

    Fallback:
      - arr_0 -> X
      - arr_1 -> y
    """
    data = np.load(npz_path, allow_pickle=True)

    if "X" in data.files and "y" in data.files:
        X = data["X"]
        y = data["y"]
    else:
        # Fallback for generic np.savez
        if len(data.files) < 2:
            raise ValueError(
                f"{npz_path} must contain X,y or arr_0,arr_1. Found keys: {data.files}"
            )
        X = data[data.files[0]]
        y = data[data.files[1]]

    return X, y.astype(int)


def load_probs_csv(csv_path: str):
    """
    Load class probability CSV of shape (N, C).

    - One row per sample
    - One column per class
    - Rows are normalized defensively
    """
    P = np.loadtxt(csv_path, delimiter=",")

    if P.ndim != 2:
        raise ValueError(f"{csv_path}: expected 2D array, got shape {P.shape}")

    # Defensive normalization
    P = np.clip(P, 1e-12, None)
    row_sum = P.sum(axis=1, keepdims=True)
    P = P / np.clip(row_sum, 1e-12, None)

    return P


def save_csv(path, array):
    """
    Save array to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), array, delimiter=",")


def save_json(path, obj):
    """
    Save dictionary/object to JSON with indentation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
