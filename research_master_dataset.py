from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scanner.segmentation import (
    SegmentationConfig,
    add_research_buckets,
    rank_best_segments,
    summarize_segments,
)


DEFAULT_EVENTS = Path("data/research/master_anomalies_2019_2025.csv")
DEFAULT_OUTPUT_DIR = Path("data/research/edge_reports")
DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "Master event CSV is missing required columns: "
            + ", ".join(missing)
        )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def adapt_master_events(
    events: pd.DataFrame,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Convert the scanner's finalized master event rows into the standardized
    shape consumed by scanner.segmentation.

    IMPORTANT RESEARCH CONVENTION
    -----------------------------
    The current scanner is researching upside/overextension anomalies and
    their subsequent mean reversion. Therefore the research position is the
    CONTRARIAN SHORT side:

        stock executable return:  -8%
        contrarian short return:  +8%

    Event-level forward returns already exist in the master dataset and are
    anchored from the next open after the first signal. We reuse those values
    rather than recomputing them from raw bars.

    MFE/MAE are intentionally left NaN at each individual horizon because the
    current master event schema stores event-level path extrema, not separate
    path extrema for every 1/2/3/5/10/20-day window. We refuse to fabricate
    horizon-specific values.
    """
    horizons = tuple(sorted({int(h) for h in horizons}))
    if not horizons or horizons[0] <= 0:
        raise ValueError("horizons must contain positive integers.")

    required = [
        "event_number",
        "ticker",
        "event_start_date",
        "entry_signal_close",
        "entry_relative_volume",
        "entry_distance_sma_20_pct",
        "entry_return_1d",
        "entry_return_3d",
        "entry_return_5d",
    ]
    required.extend(f"executable_return_{h}d" for h in horizons)
    _require_columns(events, required)

    result = events.copy()

    result["event_id"] = (
        result["ticker"].astype(str).str.upper().str.strip()
        + "|"
        + result["event_start_date"].astype(str)
        + "|"
        + result["event_number"].astype(str)
    )
    result["date"] = pd.to_datetime(
        result["event_start_date"],
        errors="coerce",
    ).dt.normalize()

    # Current research universe: upside anomaly -> contrarian SHORT.
    result["event_direction"] = 1

    # Canonical feature names expected by SegmentationConfig.
    result["event_close"] = _numeric(result, "entry_signal_close")
    result["rvol"] = _numeric(result, "entry_relative_volume")
    result["sma20_extension_pct"] = _numeric(
        result,
        "entry_distance_sma_20_pct",
    )
    result["return_1d_event"] = _numeric(result, "entry_return_1d")
    result["return_3d_event"] = _numeric(result, "entry_return_3d")
    result["return_5d_event"] = _numeric(result, "entry_return_5d")

    # The existing master dataset does not currently contain market cap or a
    # canonical dollar-volume feature on event rows. Keep these absent rather
    # than inventing them; add_research_buckets() simply skips missing inputs.

    for horizon in horizons:
        suffix = f"{horizon}d"
        source = f"executable_return_{suffix}"
        raw_return = _numeric(result, source)

        result[f"outcome_available_{suffix}"] = raw_return.notna()
        result[f"return_{suffix}"] = raw_return

        # Short-side contrarian return. This is a research return transform,
        # not a fully executable P&L model (borrow/slippage/fees not included).
        result[f"contrarian_return_{suffix}"] = -raw_return
        result[f"reversed_{suffix}"] = raw_return.lt(0).astype("boolean")

        # Do not misuse the event-wide extrema as horizon-specific extrema.
        result[f"contrarian_mfe_{suffix}"] = np.nan
        result[f"contrarian_mae_{suffix}"] = np.nan

    return result


def build_reports(
    *,
    events_path: Path = DEFAULT_EVENTS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    min_n: int = 30,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> dict[str, Path]:
    """
    Build reproducible edge-discovery CSVs from the finalized master event CSV.
    """
    if not events_path.exists():
        raise FileNotFoundError(
            f"Master event file does not exist: {events_path}\n"
            "Finish/finalize build_master_dataset.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading master events: {events_path}")
    events = pd.read_csv(events_path, low_memory=False)
    print(f"Master events loaded:  {len(events):,}")

    adapted = adapt_master_events(events, horizons=horizons)

    bucketed = add_research_buckets(
        adapted,
        SegmentationConfig(
            price_col="event_close",
            rvol_col="rvol",
            sma20_extension_col="sma20_extension_pct",
            move_1d_col="return_1d_event",
            move_3d_col="return_3d_event",
            move_5d_col="return_5d_event",
        ),
    )

    outputs: dict[str, Path] = {}

    adapted_path = output_dir / "event_outcomes_adapted.csv"
    bucketed.to_csv(adapted_path, index=False)
    outputs["adapted_events"] = adapted_path

    groupings: dict[str, list[str]] = {
        "initial_setup": ["initial_setup"],
        "price": ["price_bucket"],
        "rvol": ["rvol_bucket"],
        "sma20_extension": ["sma20_extension_bucket"],
        "move_1d": ["move_1d_bucket"],
        "move_3d": ["move_3d_bucket"],
        "move_5d": ["move_5d_bucket"],
        "setup_x_rvol": ["initial_setup", "rvol_bucket"],
        "setup_x_price": ["initial_setup", "price_bucket"],
        "setup_x_sma20": ["initial_setup", "sma20_extension_bucket"],
        "price_x_rvol": ["price_bucket", "rvol_bucket"],
        "rvol_x_move5": ["rvol_bucket", "move_5d_bucket"],
        "setup_x_rvol_x_move5": [
            "initial_setup",
            "rvol_bucket",
            "move_5d_bucket",
        ],
    }

    all_summaries: list[pd.DataFrame] = []

    for name, columns in groupings.items():
        missing = [column for column in columns if column not in bucketed.columns]
        if missing:
            print(
                f"Skipping grouping {name}: missing "
                + ", ".join(missing)
            )
            continue

        summary = summarize_segments(
            bucketed,
            group_by=columns,
            horizons=horizons,
            min_n=min_n,
        )

        if summary.empty:
            print(f"{name}: no segments reached min_n={min_n}")
            continue

        summary.insert(0, "segmentation", name)

        path = output_dir / f"segments_{name}.csv"
        summary.to_csv(path, index=False)
        outputs[name] = path
        all_summaries.append(summary)

        print(
            f"{name:<28} "
            f"{len(summary):>6,} ranked segment/horizon rows"
        )

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)

        combined_path = output_dir / "all_ranked_segments.csv"
        combined.to_csv(combined_path, index=False)
        outputs["all_ranked_segments"] = combined_path

        for horizon in horizons:
            best = rank_best_segments(
                combined,
                horizon_days=horizon,
                top_n=100,
            )

            best_path = output_dir / f"best_{horizon}d_segments.csv"
            best.to_csv(best_path, index=False)
            outputs[f"best_{horizon}d"] = best_path

    print()
    print("=" * 88)
    print("EDGE REPORT COMPLETE")
    print("=" * 88)
    print(f"Output directory: {output_dir}")
    print(f"Minimum sample:   {min_n}")
    print()
    print(
        "NOTE: rankings are hypothesis-generation results, not proof of a "
        "tradable strategy."
    )
    print(
        "Borrow availability, slippage, fees, out-of-sample validation, and "
        "multiple-testing controls still need to be added."
    )

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rank historical anomaly setups from the finalized master event "
            "dataset."
        )
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_EVENTS,
        help=f"Master event CSV (default: {DEFAULT_EVENTS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Report directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=30,
        help="Minimum event count for a segment to be reported (default: 30)",
    )
    args = parser.parse_args()

    if args.min_n < 1:
        raise SystemExit("--min-n must be at least 1")

    build_reports(
        events_path=args.events,
        output_dir=args.output_dir,
        min_n=args.min_n,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
