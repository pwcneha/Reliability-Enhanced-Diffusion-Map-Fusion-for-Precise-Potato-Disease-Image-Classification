import argparse
import numpy as np
from pathlib import Path

from src.io_utils import load_npz_xy, load_probs_csv, save_csv, save_json
from src.metrics import metrics_dict, mcnemar_exact, bootstrap_delta_nll_ci
from src.bgf_gate import fit_cluster_reliability, budgeted_gate_strict


def parse_args():
    p = argparse.ArgumentParser("Budget-Gated Fusion (BGF)")
    p.add_argument("--dmap_train", required=True)
    p.add_argument("--dmap_val", required=True)
    p.add_argument("--dmap_test", required=True)
    p.add_argument("--base_val", required=True)
    p.add_argument("--base_test", required=True)
    p.add_argument("--expert", action="append", nargs=3, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k_clust", type=int, default=12)
    p.add_argument("--budget", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--margin_min_nll", type=float, default=0.0)
    p.add_argument("--delta_conf", type=float, default=0.0)
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

    experts_val, experts_test = [], []
    for name, v, t in args.expert:
        experts_val.append((name, load_probs_csv(v)))
        experts_test.append((name, load_probs_csv(t)))

    cluster_nll, ct_test = fit_cluster_reliability(
        Xtr, Xv, Xt, yv, base_val, experts_val,
        k_clust=args.k_clust, seed=args.seed
    )

    fused_test = budgeted_gate_strict(
        base_test, experts_test, cluster_nll, ct_test,
        budget=args.budget,
        margin_min_nll=args.margin_min_nll,
        delta_conf=args.delta_conf
    )

    save_csv(out / "base_probs_test.csv", base_test)
    save_csv(out / "fused_probs_test.csv", fused_test)

    base_metrics = metrics_dict(base_test, yt)
    fused_metrics = metrics_dict(fused_test, yt)

    yhat_base = base_test.argmax(1)
    yhat_fuse = fused_test.argmax(1)

    p_val, n01, n10 = mcnemar_exact(yhat_base, yhat_fuse, yt)
    delta_ci = bootstrap_delta_nll_ci(base_test, fused_test, yt)

    changed_probs = int((np.abs(base_test - fused_test).max(1) > 1e-12).sum())
    changed_labels = int((yhat_base != yhat_fuse).sum())

    report = {
        "config": vars(args),
        "base_metrics": base_metrics,
        "fused_metrics": fused_metrics,
        "override_stats": {
            "overridden_prob_rows": changed_probs,
            "overridden_prob_fraction": changed_probs / len(yt),
            "label_flips": changed_labels,
            "label_flip_fraction": changed_labels / len(yt),
        },
        "mcnemar": {"p_value": p_val, "n01": n01, "n10": n10},
        "delta_nll_ci": delta_ci,
    }

    save_json(out / "report.json", report)
    print(f"[OK] Saved outputs to {out}")


if __name__ == "__main__":
    main()

