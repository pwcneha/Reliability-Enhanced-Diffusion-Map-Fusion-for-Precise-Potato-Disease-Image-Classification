# run_fusion.py
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
        help="Repeat for each expert: --expert name val.csv test.csv",
    )

    p.add_argument("--out_dir", required=True)
    p.add_argument("--k_clust", type=int, default=12)
    p.add_argument("--budget", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--margin_min_nll", type=float, default=0.0)
    p.add_argument("--delta_conf", type=float, default=0.0)
    return p.parse_args()


def _check_probs(name, P, N, C):
    P = np.asarray(P)
    if P.ndim != 2:
        raise ValueError(f"{name}: expected 2D probs array, got shape {P.shape}")
    if P.shape != (N, C):
        raise ValueError(f"{name}: expected shape {(N, C)}, got {P.shape}")
    row_sum_err = np.abs(P.sum(axis=1) - 1.0).mean()
    if row_sum_err > 1e-2:
        print(f"[WARN] {name}: mean |sum(p)-1| = {row_sum_err:.3e} (are these probabilities?)")


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    Xtr, ytr = load_npz_xy(args.dmap_train)
    Xv, yv = load_npz_xy(args.dmap_val)
    Xt, yt = load_npz_xy(args.dmap_test)

    base_val = load_probs_csv(args.base_val)
    base_test = load_probs_csv(args.base_test)

    N_val, N_test = len(yv), len(yt)
    C = base_val.shape[1]

    _check_probs("base_val", base_val, N_val, C)
    _check_probs("base_test", base_test, N_test, C)

    experts_val, experts_test = [], []
    for name, v_csv, t_csv in args.expert:
        Pv = load_probs_csv(v_csv)
        Pt = load_probs_csv(t_csv)
        _check_probs(f"{name}_val", Pv, N_val, C)
        _check_probs(f"{name}_test", Pt, N_test, C)
        experts_val.append((name, Pv))
        experts_test.append((name, Pt))

    cluster_nll, ct_test = fit_cluster_reliability(
        Xtr, Xv, Xt, yv,
        base_val,
        experts_val,
        k_clust=args.k_clust,
        seed=args.seed,
    )

    fused_test = budgeted_gate_strict(
        base_test,
        experts_test,
        cluster_nll,
        ct_test,
        budget=args.budget,
        margin_min_nll=args.margin_min_nll,
        delta_conf=args.delta_conf,
    )

    # save outputs
    save_csv(out / "base_probs_test.csv", base_test)
    save_csv(out / "fused_probs_test.csv", fused_test)

    base_metrics = metrics_dict(base_test, yt)
    fused_metrics = metrics_dict(fused_test, yt)

    yhat_base = base_test.argmax(1)
    yhat_fuse = fused_test.argmax(1)
    p_mcnemar, n01, n10 = mcnemar_exact(yhat_base, yhat_fuse, yt)
    delta_ci = bootstrap_delta_nll_ci(base_test, fused_test, yt, seed=args.seed)

    report = {
        "config": {
            "k_clust": args.k_clust,
            "budget": args.budget,
            "seed": args.seed,
            "margin_min_nll": args.margin_min_nll,
            "delta_conf": args.delta_conf,
            "experts": [n for (n, _) in experts_test],
        },
        "base_metrics": base_metrics,
        "fused_metrics": fused_metrics,
        "mcnemar": {
            "p_value": p_mcnemar,
            "n01_base_correct_fuse_wrong": int(n01),
            "n10_base_wrong_fuse_correct": int(n10),
        },
        "delta_nll_ci": delta_ci,
    }

    save_json(out / "report.json", report)
    print(f"[OK] Saved outputs to: {out}")


if __name__ == "__main__":
    main()


