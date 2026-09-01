from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scanner.walkforward import (
    WalkForwardConfig,
    summarize_walk_forward,
    walk_forward_validate,
)


DEFAULT_INPUT = Path(
    "data/research/edge_reports/event_outcomes_adapted.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/research/walkforward_reports"
)


GROUPINGS: dict[str, list[str]] = {
    "initial_setup": ["initial_setup"],
    "rvol": ["rvol_bucket"],
    "sma20_extension": ["sma20_extension_bucket"],
    "move_1d": ["move_1d_bucket"],
    "move_3d": ["move_3d_bucket"],
    "move_5d": ["move_5d_bucket"],
    "setup_x_rvol": ["initial_setup", "rvol_bucket"],
    "setup_x_sma20": [
        "initial_setup",
        "sma20_extension_bucket",
    ],
    "rvol_x_move5": ["rvol_bucket", "move_5d_bucket"],
    "setup_x_rvol_x_move5": [
        "initial_setup",
        "rvol_bucket",
        "move_5d_bucket",
    ],
}


def run_walkforward(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    first_test_year: int = 2021,
    min_train_n: int = 30,
    min_test_n: int = 10,
    top_n: int = 25,
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
    print(f"First test year:         {first_test_year}")
    print(f"Minimum train N:         {min_train_n}")
    print(f"Minimum test N:          {min_test_n}")
    print()

    config = WalkForwardConfig(
        min_train_n=min_train_n,
        min_test_n=min_test_n,
        top_n_per_fold=top_n,
    )

    outputs: dict[str, Path] = {}
    all_fold_results: list[pd.DataFrame] = []
    all_persistence: list[pd.DataFrame] = []

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

        fold_results = walk_forward_validate(
            frame,
            group_by=group_by,
            config=config,
            first_test_year=first_test_year,
        )

        if fold_results.empty:
            print(f"{name:<28} no qualifying folds")
            continue

        fold_results.insert(0, "segmentation", name)

        folds_path = output_dir / f"walkforward_{name}.csv"
        fold_results.to_csv(folds_path, index=False)
        outputs[f"{name}_folds"] = folds_path
        all_fold_results.append(fold_results)

        persistence = summarize_walk_forward(
            fold_results,
            group_by=group_by,
        )

        if not persistence.empty:
            persistence.insert(0, "segmentation", name)
            persistence_path = (
                output_dir / f"persistence_{name}.csv"
            )
            persistence.to_csv(
                persistence_path,
                index=False,
            )
            outputs[f"{name}_persistence"] = persistence_path
            all_persistence.append(persistence)

        validated = int(fold_results["validated"].sum())
        total = int(len(fold_results))

        print(
            f"{name:<28} "
            f"{validated:>4}/{total:<4} fold-segments validated"
        )

    if all_fold_results:
        combined_folds = pd.concat(
            all_fold_results,
            ignore_index=True,
        )
        path = output_dir / "all_walkforward_folds.csv"
        combined_folds.to_csv(path, index=False)
        outputs["all_folds"] = path

    if all_persistence:
        combined_persistence = pd.concat(
            all_persistence,
            ignore_index=True,
        )

        combined_persistence = combined_persistence.sort_values(
            [
                "horizon_days",
                "validation_rate",
                "total_test_n",
                "median_test_median_return",
            ],
            ascending=[True, False, False, False],
            kind="stable",
        ).reset_index(drop=True)

        path = output_dir / "all_persistent_edges.csv"
        combined_persistence.to_csv(path, index=False)
        outputs["all_persistent_edges"] = path

        strongest = combined_persistence[
            (combined_persistence["validation_rate"] >= 0.60)
            & (
                combined_persistence[
                    "median_test_median_return"
                ]
                > 0
            )
        ].copy()

        strongest_path = (
            output_dir / "strongest_persistent_edges.csv"
        )
        strongest.to_csv(strongest_path, index=False)
        outputs["strongest_persistent_edges"] = strongest_path

    print()
    print("=" * 92)
    print("WALK-FORWARD VALIDATION COMPLETE")
    print("=" * 92)
    print(f"Reports: {output_dir}")
    print()
    print(
        "A setup validates in a test fold only when the untouched test "
        "year has enough observations, >50% contrarian wins, and a "
        "positive median contrarian return."
    )

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run expanding-window out-of-sample validation on "
            "historical anomaly research."
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
        "--first-test-year",
        type=int,
        default=2021,
    )
    parser.add_argument(
        "--min-train-n",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--min-test-n",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
    )
    args = parser.parse_args()

    run_walkforward(
        input_path=args.input,
        output_dir=args.output_dir,
        first_test_year=args.first_test_year,
        min_train_n=args.min_train_n,
        min_test_n=args.min_test_n,
        top_n=args.top_n,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
