from __future__ import annotations

import argparse
from pathlib import Path

from scanner.promotion import (
    PromotionConfig,
    build_promotion_report,
)


DEFAULT_SIGNIFICANCE = Path(
    "data/research/significance_reports/"
    "all_significance_results.csv"
)
DEFAULT_PERSISTENCE = Path(
    "data/research/walkforward_reports/"
    "all_persistent_edges.csv"
)
DEFAULT_OUTPUT = Path(
    "data/research/final_evidence/"
    "promoted_research_setups.csv"
)


GROUPING_MAP: dict[str, list[str]] = {
    "initial_setup": ["initial_setup"],
    "price": ["price_bucket"],
    "rvol": ["rvol_bucket"],
    "sma20_extension": ["sma20_extension_bucket"],
    "move_1d": ["move_1d_bucket"],
    "move_3d": ["move_3d_bucket"],
    "move_5d": ["move_5d_bucket"],
    "setup_x_rvol": [
        "initial_setup",
        "rvol_bucket",
    ],
    "setup_x_price": [
        "initial_setup",
        "price_bucket",
    ],
    "setup_x_sma20": [
        "initial_setup",
        "sma20_extension_bucket",
    ],
    "price_x_rvol": [
        "price_bucket",
        "rvol_bucket",
    ],
    "rvol_x_move5": [
        "rvol_bucket",
        "move_5d_bucket",
    ],
    "setup_x_rvol_x_move5": [
        "initial_setup",
        "rvol_bucket",
        "move_5d_bucket",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Combine statistical significance and walk-forward "
            "persistence into PROMOTE/WATCH/REJECT research decisions."
        )
    )
    parser.add_argument(
        "--significance",
        type=Path,
        default=DEFAULT_SIGNIFICANCE,
    )
    parser.add_argument(
        "--persistence",
        type=Path,
        default=DEFAULT_PERSISTENCE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    report = build_promotion_report(
        significance_path=args.significance,
        persistence_path=args.persistence,
        output_path=args.output,
        grouping_map=GROUPING_MAP,
        config=PromotionConfig(),
    )

    print()
    print("=" * 92)
    print("FINAL EVIDENCE REPORT")
    print("=" * 92)

    if report.empty:
        print("No matched research setups were available.")
        print(f"Output: {args.output}")
        return 0

    counts = (
        report["disposition"]
        .value_counts()
        .reindex(
            ["PROMOTE", "WATCH", "REJECT"],
            fill_value=0,
        )
    )

    print(f"PROMOTE: {int(counts['PROMOTE']):>6,}")
    print(f"WATCH:   {int(counts['WATCH']):>6,}")
    print(f"REJECT:  {int(counts['REJECT']):>6,}")
    print()
    print(f"Output: {args.output}")
    print()
    print(
        "PROMOTE is a research designation only. It does not yet account "
        "for borrow availability, slippage, fees, position sizing, or "
        "live execution constraints."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
