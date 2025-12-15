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

    # strict gate knobs (make manuscript-reproducible)
    p.add_argument("--min_cluster_val", type=int, default=3)
    p.add_argument("--margin_min_nll", type=float, default=0.005)
    p.add_argument("--delta_conf", type=float, default=0.08)

    return p.parse_args()


def _check_probs(name, P, N, C):
    if not isinstance(P, np.ndarray):
        raise ValueError(f"{name}: expected numpy array, got {type(P)}")
    if P.ndim != 2:
        raise ValueError(f"{name}: expected 2D probs array, got shape {P.shape}")
    if P.shape != (N, C):
        raise ValueError(f"{name}: expected shape {(N, C)}, got {P.shape}")

    # Basic sanity
    if np.any(P < -1e-6):
        raise ValueError(f"{name}: contains negative probabilities.")
    row_err = float(np.mean(np.abs(P.sum(axis=1) - 1.0)))
    if row_err > 1e-2:
        print(f"[WARN] {name}: mean |sum(p)-1| = {row_err:.3e} (check normalization)")


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load DMAP features + labels
    Xtr, ytr = load_npz_xy(args.dmap_train)
    Xv, yv = load_npz_xy(args.dmap_val)
    Xt, yt = load_npz_xy(args.dmap_test)

    # Load baseline probabilities
    base_val = load_probs_csv(args.base_val)
    base_test = load_probs_csv(args.base_test)

    N_val = int(len(yv))
    N_test = int(len(yt))
    C = int(base_val.shape[1])

    _check_probs("base_val", base_val, N_val, C)
    _check_probs("base_test", base_test, N_test, C)

    # Load experts (VAL + TEST)
    experts_val = []
    experts_test = []
    for name, v_csv, t_csv in args.expert:
        Pv = load_probs_csv(v_csv)
        Pt = load_probs_csv(t_csv)
        _check_probs(f"{name}_val", Pv, N_val, C)
        _check_probs(f"{name}_test", Pt, N_test, C)
        experts_val.append((name, Pv))
        experts_test.append((name, Pt))

    # Fit cluster-wise expected NLL map (on VAL labels)
    cluster_nll, ct_test = fit_cluster_reliability(
        Xtr=Xtr, Xv=Xv, Xt=Xt, yv=yv,
        base_val=base_val,
        experts_val=experts_val,
        k_clust=args.k_clust,
        seed=args.seed,
        min_cluster_val=args.min_cluster_val,
    )

    # Fuse on TEST with strict gate (budgeted)
    fused_test, gate_info = budgeted_gate_strict(
        base_test=base_test,
        experts_test=experts_test,
        cluster_nll=cluster_nll,
        ct_test=ct_test,
        budget=args.budget,
        margin_min_nll=args.margin_min_nll,
        delta_conf=args.delta_conf,
    )

    # Save probability outputs
    save_csv(out / "base_probs_test.csv", base_test)
    save_csv(out / "fused_probs_test.csv", fused_test)

    # Metrics
    base_metrics = metrics_dict(base_test, yt)
    fused_metrics = metrics_dict(fused_test, yt)

    # McNemar
    yhat_base = base_test.argmax(1)
    yhat_fuse = fused_test.argmax(1)
    mc = mcnemar_exact(yhat_base, yhat_fuse, yt)  # dict

    # Bootstrap CI for ΔNLL = NLL(fused) - NLL(base) (negative is better)
    delta_ci = bootstrap_delta_nll_ci(fused_test, base_test, yt, seed=args.seed)  # IMPORTANT order

    flip_mask = (yhat_fuse != yhat_base)

    report = {
        "config": {
            "k_clust": int(args.k_clust),
            "budget": float(args.budget),
            "seed": int(args.seed),
            "min_cluster_val": int(args.min_cluster_val),
            "margin_min_nll": float(args.margin_min_nll),
            "delta_conf": float(args.delta_conf),
            "experts": [n for (n, _) in experts_test],
        },
        "base_metrics": base_metrics,
        "fused_metrics": fused_metrics,
        "flips": {
            "count": int(flip_mask.sum()),
            "frac": float(flip_mask.mean()),
        },
        "gate_info": gate_info,
        "mcnemar": mc,
        "delta_nll_ci": delta_ci,
    }

    save_json(out / "report.json", report)
    print(f"[OK] Saved outputs to: {out}")


if __name__ == "__main__":
    main()



