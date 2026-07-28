"""Construct both locked BGF-v1 policies for one outer fold.

The input package contains development labels and fold-local development and
held-out probability streams. Held-out labels are forbidden. Performance
evaluation is intentionally separate from policy construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.bgf_gate import (
    BGFConfig,
    apply_regional_gate,
    apply_temperature,
    build_regional_policy,
    build_regions,
    fit_temperature,
)
from src.io_utils import (
    load_policy_input_npz,
    save_json,
    save_npz,
    save_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct the as-executed and equation-consistent BGF "
            "policies for one group-aware outer fold."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Fold-local NPZ package without held-out labels.",
    )
    parser.add_argument(
        "--config",
        default="configs/bgf_thresholds.json",
        help="Path to the locked public BGF configuration.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for predictions and audit tables.",
    )
    parser.add_argument(
        "--fold-id",
        default="unspecified",
        help="Descriptive fold identifier written to the summary.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> tuple[BGFConfig, dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    config = BGFConfig(
        n_regions=int(raw["n_regions"]),
        minimum_region_n=int(raw["minimum_region_n"]),
        minimum_nll_gain=float(raw["minimum_nll_gain"]),
        maximum_ece_drift=float(raw["maximum_ece_drift"]),
        minimum_expert_margin=float(raw["minimum_expert_margin"]),
        maximum_edit_rate=float(raw["maximum_edit_rate"]),
        random_seed=int(raw["random_seed"]),
        kmeans_n_init=int(raw["kmeans_n_init"]),
        kmeans_max_iter=int(raw["kmeans_max_iter"]),
    )
    return config, raw


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config, raw_config = load_config(args.config)
    data = load_policy_input_npz(args.input)

    development_regions, heldout_regions, _, global_rms = build_regions(
        data["development_dmap"],
        data["heldout_dmap"],
        config=config,
    )
    policy, regional_audit = build_regional_policy(
        data["development_labels"],
        data["development_probability_tensor"],
        development_regions,
        config=config,
    )
    save_table(
        output_dir / "regional_expert_policy_audit.csv",
        regional_audit,
    )

    summary = {
        "fold_id": str(args.fold_id),
        "development_n": int(len(data["development_labels"])),
        "heldout_n": int(len(data["heldout_source_ids"])),
        "global_dmap_rms": float(global_rms),
        "heldout_labels_accessed": False,
        "configuration": raw_config,
        "policies": {},
    }

    for version in ("as_executed", "equation_consistent"):
        development_result = apply_regional_gate(
            data["development_probability_tensor"],
            development_regions,
            data["development_source_ids"],
            policy,
            version,
            config=config,
        )
        heldout_result = apply_regional_gate(
            data["heldout_probability_tensor"],
            heldout_regions,
            data["heldout_source_ids"],
            policy,
            version,
            config=config,
        )
        temperature, temperature_status = fit_temperature(
            data["development_labels"],
            development_result["probabilities"],
        )
        heldout_post = apply_temperature(
            heldout_result["probabilities"],
            temperature,
        )

        save_table(
            output_dir / f"{version}_candidates.csv",
            heldout_result["candidate_table"],
        )
        save_npz(
            output_dir / f"{version}_heldout_predictions.npz",
            source_ids=data["heldout_source_ids"],
            regions=heldout_regions,
            probabilities_pre=heldout_result["probabilities"],
            probabilities_post=heldout_post,
            accepted_mask=heldout_result["accepted_mask"],
            selected_expert_indices=heldout_result[
                "selected_expert_indices"
            ],
            temperature=np.asarray([temperature], dtype=np.float64),
            heldout_labels_accessed=np.asarray([False], dtype=bool),
        )
        summary["policies"][version] = {
            "candidate_n": int(heldout_result["candidate_n"]),
            "accepted_n": int(heldout_result["accepted_n"]),
            "maximum_edits": int(heldout_result["maximum_edits"]),
            "temperature": float(temperature),
            "temperature_status": temperature_status,
        }

    save_json(output_dir / "bgf_construction_summary.json", summary)
    print(
        "BGF construction complete. Held-out labels were not accepted "
        f"or accessed. Outputs: {output_dir}"
    )


if __name__ == "__main__":
    main()
