import argparse
from pathlib import Path
import numpy as np

from src.io_utils import load_npz_xy, load_probs_csv, save_csv, save_json
from src.metrics import metrics_dict, mcnemar_exact, bootstrap_delta_nll_ci
from src.bgf_gate import fit_cluster_reliability, budgeted_gate_strict


def parse_args():
    p = argparse.ArgumentParser(
        description="Budget-Gated Fusion (BGF) using DMAP clusters and calibrated expert probabilities"
    )
    p.add_argument("--dmap_train", required=True)
    p.add_argument("--dmap_val", required=True)
    p.add_argument("--dmap_test", required=True)

    p.add_argument("--base_val", required=True)
    p.add_argument("--base_test", required=True)

    p.add_argument(
        "--expert",
        action="append",
        nargs=3,
        metavar=("NAME", "VAL_CSV", "TEST_CSV"),
        required=True,
        help="Repeat for each expert"
    )

    p.add_argument("--out_dir", required=True)
    p.add_argument("--k_clust", type=int, default=12)
    p.add_argument("--budget", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    Xtr, ytr = load_npz_xy(args.dmap_train)
    Xv, yv = load_npz_xy(args.dmap_val)
    Xt, yt = load_npz_xy(args.dmap_test)

    base_val = load_probs_csv(args.base_val)
    base_test = load_probs_csv(args.base_test)

    experts_val = []
    experts_test = []

    for name, v_csv, t_csv in args.expert:
        experts_val.append((name, load_probs_csv(v_csv)))
        experts_test.append((name, load_probs_csv(t_csv)))

    cluster_nll, ct_test = fit_cluster_reliability(
        Xtr, Xv, Xt, yv,
        base_val,
        experts_val,
        k_clust=args.k_clust,
        seed=args.seed
    )

    fused_test = budgeted_gate_strict(
        base_test,
        experts_test,
        cluster_nll,
        ct_test,
        budget=args.budget
    )

    save_csv(out / "fused_probs_test.csv", fused_test)

    metrics = metrics_dict(fused_test, yt)
    save_json(out / "metrics.json", metrics)


if __name__ == "__main__":
    main()
