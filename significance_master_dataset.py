from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scanner.significance import (
    SignificanceConfig,
    evaluate_segment_significance,
)


DEFAULT_INPUT = Path(
    "data/research/edge_reports/event_outcomes_adapted.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/research/significance_reports"
)


GROUPINGS: dict[str, list[str]] = {
    "initial_setup": ["initial_setup"],
    "price": ["price_bucket"],
    "rvol": ["rvol_bucket"],
    "sma20_extension": ["sma20_extension_bucket"],
    "move_1d": ["move_1d_bucket"],
    "move_3d": ["move_3d_bucket"],
    "move_5d": ["move_5d_bucket"],
    "setup_x_rvol": ["initial_setup", "rvol_bucket"],
    "setup_x_price": ["initial_setup", "price_bucket"],
    "setup_x_sma20": [
        "initial_setup",
        "sma20_extension_bucket",
    ],
    "price_x_rvol": ["price_bucket", "rvol_bucket"],
    "rvol_x_move5": ["rvol_bucket", "move_5d_bucket"],
    "setup_x_rvol_x_move5": [
        "initial_setup",
        "rvol_bucket",
        "move_5d_bucket",
    ],
}


def run_significance(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    min_n: int = 30,
    bootstrap_samples: int = 4000,
) -> dict[str, Path]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Adapted event file does not exist: {input_path}\n"
            "Run research_master_dataset.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading adapted events: {input_path}")
    frame = pd.read_csv(input_path, low_memory=False)
    print(f"Events loaded:           {len(frame):,}")
    print(f"Minimum N:               {min_n}")
    print(f"Bootstrap samples:       {bootstrap_samples}")
    print()

    config = SignificanceConfig(
        bootstrap_samples=bootstrap_samples,
    )

    outputs: dict[str, Path] = {}
    combined: list[pd.DataFrame] = []

    for name, group_by in GROUPINGS.items():
        missing = [
            column
            for column in group_by
            if column not in frame.columns
        ]
        if missing:
            print(
                f"{name:<28} SKIP missing "
                + ", ".join(missing)
            )
            continue

        report = evaluate_segment_significance(
            frame,
            group_by=group_by,
            min_n=min_n,
            config=config,
        )

        if report.empty:
            print(f"{name:<28} no segments reached min_n")
            continue

        report.insert(0, "segmentation", name)

        path = output_dir / f"significance_{name}.csv"
        report.to_csv(path, index=False)
        outputs[name] = path
        combined.append(report)

        supported = int(
            report["statistically_supported"].sum()
        )

        print(
            f"{name:<28} "
            f"{supported:>4}/{len(report):<4} statistically supported"
        )

    if combined:
        all_results = pd.concat(
            combined,
            ignore_index=True,
        )

        # IMPORTANT: each grouping report already applies BH internally.
        # This combined file is for browsing; we also apply a second, global
        # correction across every tested row to create the strictest shortlist.
        from scanner.significance import benjamini_hochberg

        global_bh = benjamini_hochberg(
            all_results["win_rate_p_value"],
            alpha=config.fdr_alpha,
        )

        all_results["global_q_value"] = (
            global_bh["q_value"].to_numpy()
        )
        all_results["global_fdr_significant"] = (
            global_bh["fdr_significant"].to_numpy()
        )

        all_results["globally_supported"] = (
            all_results["global_fdr_significant"]
            & (all_results["win_rate_ci_low"] > 0.50)
            & (all_results["median_return_ci_low"] > 0.0)
        )

        all_path = output_dir / "all_significance_results.csv"
        all_results.to_csv(all_path, index=False)
        outputs["all_results"] = all_path

        strongest = all_results[
            all_results["globally_supported"]
        ].copy()

        strongest = strongest.sort_values(
            [
                "horizon_days",
                "confidence_score",
                "n",
            ],
            ascending=[True, False, False],
            kind="stable",
        ).reset_index(drop=True)

        strongest_path = (
            output_dir / "statistically_strongest_edges.csv"
        )
        strongest.to_csv(strongest_path, index=False)
        outputs["strongest"] = strongest_path

    print()
    print("=" * 92)
    print("SIGNIFICANCE ANALYSIS COMPLETE")
    print("=" * 92)
    print(f"Reports: {output_dir}")
    print()
    print(
        "The strict shortlist requires a globally FDR-corrected significant "
        "win rate, a >50% lower Wilson bound, and a >0 lower bootstrap "
        "median-return bound."
    )

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate statistical confidence and false-discovery control "
            "for historical anomaly segments."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=4000,
    )
    args = parser.parse_args()

    run_significance(
        input_path=args.input,
        output_dir=args.output_dir,
        min_n=args.min_n,
        bootstrap_samples=args.bootstrap_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
