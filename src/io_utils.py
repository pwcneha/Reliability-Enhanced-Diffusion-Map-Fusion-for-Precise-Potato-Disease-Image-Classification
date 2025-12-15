import json
import numpy as np
from pathlib import Path


def load_npz_xy(path):
    z = np.load(path, allow_pickle=True)
    if "X" in z.files and "y" in z.files:
        X, y = z["X"], z["y"]
    else:
        # fallback for unnamed arrays
        X = z["arr_0"]
        y = z["arr_1"] if "arr_1" in z.files else None
        if y is None:
            raise ValueError(f"{path}: could not find labels 'y' or arr_1 in npz")
    return X, y.astype(int)


def load_probs_csv(path):
    P = np.loadtxt(path, delimiter=",")
    P = np.asarray(P, dtype=float)
    # defensively normalize
    P = np.clip(P, 1e-12, None)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    return P


def save_csv(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), np.asarray(arr, dtype=float), delimiter=",")


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


