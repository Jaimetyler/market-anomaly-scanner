from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SAMPLES_PATH = Path(
    "data/research/replay_reports/raw_median_top_n_samples.csv"
)

DEFAULT_EVENTS_PATH = Path(
    "data/research/edge_reports/event_outcomes_adapted.csv"
)

DEFAULT_REPORT_DIR = Path(
    "data/research/replay_reports"
)

FOCUS_TOP_N = 5

TAIL_THRESHOLDS = (
    -25.0,
    -50.0,
    -100.0,
    -200.0,
)


def load_event_metadata(
    path: str | Path = DEFAULT_EVENTS_PATH,
) -> pd.DataFrame:
    """
    Load event-level metadata that was intentionally not carried through
    the large replay anchor report.

    We join this back by event_id for tail attribution.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Event outcomes file not found: {path}"
        )

    print(
        f"Loading event metadata: {path}"
    )

    header = pd.read_csv(
        path,
        nrows=0,
    )

    required = {
        "event_id",
        "event_direction",
        "initial_setup",
    }

    missing = sorted(
        required - set(header.columns)
    )

    if missing:
        raise ValueError(
            "Event outcomes file is missing required columns: "
            + ", ".join(missing)
        )

    metadata = pd.read_csv(
        path,
        usecols=[
            "event_id",
            "event_direction",
            "initial_setup",
        ],
    )

    metadata = (
        metadata.drop_duplicates(
            subset=["event_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        f"Loaded metadata for "
        f"{metadata['event_id'].nunique():,} events."
    )

    return metadata


def attach_event_metadata(
    samples: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach direction/setup metadata to the RAW_MEDIAN sample rows.

    Existing columns are preserved if they already exist.
    """
    df = samples.copy()

    columns_needed = [
        column
        for column in (
            "event_direction",
            "initial_setup",
        )
        if column not in df.columns
    ]

    if not columns_needed:
        return df

    join_columns = [
        "event_id",
        *columns_needed,
    ]

    metadata_subset = (
        metadata[
            join_columns
        ]
        .drop_duplicates(
            subset=["event_id"],
            keep="first",
        )
    )

    df = df.merge(
        metadata_subset,
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    return df


def load_samples(
    path: str | Path = DEFAULT_SAMPLES_PATH,
    *,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Top-N sample file not found: {path}"
        )

    print(
        f"Loading RAW_MEDIAN samples: {path}"
    )

    df = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    base_required = {
        "event_id",
        "ticker",
        "date",
        "test_year",
        "horizon_days",
        "top_n",
        "segmentation",
        "train_n",
        "train_win_rate",
        "train_median_return",
        "ranking_score",
        "realized_contrarian_return",
        "condition",
    }

    missing_base = sorted(
        base_required - set(df.columns)
    )

    if missing_base:
        raise ValueError(
            "Top-N sample file is missing required columns: "
            + ", ".join(missing_base)
        )

    metadata_required = {
        "event_direction",
        "initial_setup",
    }

    if not metadata_required.issubset(
        df.columns
    ):
        metadata = load_event_metadata(
            events_path
        )

        df = attach_event_metadata(
            df,
            metadata,
        )

    missing_after_join = sorted(
        (
            base_required
            | metadata_required
        )
        - set(df.columns)
    )

    if missing_after_join:
        raise ValueError(
            "Tail analysis still lacks required columns after "
            "metadata join: "
            + ", ".join(missing_after_join)
        )

    df["test_year"] = pd.to_numeric(
        df["test_year"],
        errors="coerce",
    ).astype("Int64")

    df["horizon_days"] = pd.to_numeric(
        df["horizon_days"],
        errors="coerce",
    ).astype("Int64")

    df["top_n"] = pd.to_numeric(
        df["top_n"],
        errors="coerce",
    ).astype("Int64")

    df[
        "realized_contrarian_return"
    ] = pd.to_numeric(
        df["realized_contrarian_return"],
        errors="coerce",
    )

    missing_direction = (
        df["event_direction"]
        .isna()
        .sum()
    )

    missing_setup = (
        df["initial_setup"]
        .isna()
        .sum()
    )

    print(
        f"Loaded {len(df):,} sample rows."
    )

    print(
        f"Metadata gaps: "
        f"direction={missing_direction:,}, "
        f"setup={missing_setup:,}"
    )

    return df


def focus_samples(
    samples: pd.DataFrame,
    *,
    top_n: int = FOCUS_TOP_N,
) -> pd.DataFrame:
    df = samples[
        samples["top_n"] == top_n
    ].copy()

    df = df.dropna(
        subset=[
            "realized_contrarian_return",
            "event_direction",
        ]
    )

    return df


def direction_label(
    value: object,
) -> str:
    """
    Preserve the raw direction while providing a stable grouping.

    We deliberately do not assume yet whether positive direction maps
    to the short or long contrarian side. The empirical tail analysis
    will tell us which side is dangerous before we assign trade labels.
    """
    try:
        numeric = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return str(value)

    if numeric > 0:
        return "DIRECTION_POSITIVE"

    if numeric < 0:
        return "DIRECTION_NEGATIVE"

    return "DIRECTION_ZERO"


def add_tail_flags(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    df = samples.copy()

    df["direction_label"] = (
        df["event_direction"]
        .map(direction_label)
    )

    returns = df[
        "realized_contrarian_return"
    ]

    for threshold in TAIL_THRESHOLDS:
        name = (
            f"loss_le_{abs(int(threshold))}"
        )

        df[name] = (
            returns <= threshold
        )

    return df


def summarize_direction(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        samples.groupby(
            [
                "direction_label",
                "horizon_days",
            ],
            observed=True,
        )
        .agg(
            observations=(
                "event_id",
                "size",
            ),
            unique_events=(
                "event_id",
                "nunique",
            ),
            win_rate=(
                "realized_contrarian_return",
                lambda values: (
                    pd.to_numeric(
                        values,
                        errors="coerce",
                    )
                    > 0
                ).mean(),
            ),
            mean_return=(
                "realized_contrarian_return",
                "mean",
            ),
            median_return=(
                "realized_contrarian_return",
                "median",
            ),
            p05_return=(
                "realized_contrarian_return",
                lambda values: (
                    pd.to_numeric(
                        values,
                        errors="coerce",
                    )
                    .dropna()
                    .quantile(0.05)
                ),
            ),
            worst_return=(
                "realized_contrarian_return",
                "min",
            ),
            loss_le_25_rate=(
                "loss_le_25",
                "mean",
            ),
            loss_le_50_rate=(
                "loss_le_50",
                "mean",
            ),
            loss_le_100_rate=(
                "loss_le_100",
                "mean",
            ),
            loss_le_200_rate=(
                "loss_le_200",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary


def summarize_setup_tail(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        samples.groupby(
            [
                "horizon_days",
                "direction_label",
                "initial_setup",
            ],
            observed=True,
        )
        .agg(
            observations=(
                "event_id",
                "size",
            ),
            mean_return=(
                "realized_contrarian_return",
                "mean",
            ),
            median_return=(
                "realized_contrarian_return",
                "median",
            ),
            p05_return=(
                "realized_contrarian_return",
                lambda values: (
                    pd.to_numeric(
                        values,
                        errors="coerce",
                    )
                    .dropna()
                    .quantile(0.05)
                ),
            ),
            worst_return=(
                "realized_contrarian_return",
                "min",
            ),
            loss_le_50_count=(
                "loss_le_50",
                "sum",
            ),
            loss_le_100_count=(
                "loss_le_100",
                "sum",
            ),
            loss_le_200_count=(
                "loss_le_200",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "loss_le_50_rate"
    ] = (
        summary["loss_le_50_count"]
        / summary["observations"]
    )

    summary[
        "loss_le_100_rate"
    ] = (
        summary["loss_le_100_count"]
        / summary["observations"]
    )

    summary[
        "loss_le_200_rate"
    ] = (
        summary["loss_le_200_count"]
        / summary["observations"]
    )

    return summary


def summarize_segmentation_tail(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        samples.groupby(
            [
                "horizon_days",
                "direction_label",
                "segmentation",
            ],
            observed=True,
        )
        .agg(
            observations=(
                "event_id",
                "size",
            ),
            mean_return=(
                "realized_contrarian_return",
                "mean",
            ),
            median_return=(
                "realized_contrarian_return",
                "median",
            ),
            worst_return=(
                "realized_contrarian_return",
                "min",
            ),
            loss_le_50_count=(
                "loss_le_50",
                "sum",
            ),
            loss_le_100_count=(
                "loss_le_100",
                "sum",
            ),
            loss_le_200_count=(
                "loss_le_200",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "loss_le_50_rate"
    ] = (
        summary["loss_le_50_count"]
        / summary["observations"]
    )

    summary[
        "loss_le_100_rate"
    ] = (
        summary["loss_le_100_count"]
        / summary["observations"]
    )

    summary[
        "loss_le_200_rate"
    ] = (
        summary["loss_le_200_count"]
        / summary["observations"]
    )

    return summary


def summarize_year_tail(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        samples.groupby(
            [
                "test_year",
                "horizon_days",
                "direction_label",
            ],
            observed=True,
        )
        .agg(
            observations=(
                "event_id",
                "size",
            ),
            mean_return=(
                "realized_contrarian_return",
                "mean",
            ),
            median_return=(
                "realized_contrarian_return",
                "median",
            ),
            worst_return=(
                "realized_contrarian_return",
                "min",
            ),
            loss_le_100_count=(
                "loss_le_100",
                "sum",
            ),
            loss_le_200_count=(
                "loss_le_200",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "loss_le_100_rate"
    ] = (
        summary["loss_le_100_count"]
        / summary["observations"]
    )

    summary[
        "loss_le_200_rate"
    ] = (
        summary["loss_le_200_count"]
        / summary["observations"]
    )

    return summary


def worst_events(
    samples: pd.DataFrame,
    *,
    limit_per_horizon: int = 25,
) -> pd.DataFrame:
    blocks = []

    for horizon in sorted(
        samples["horizon_days"]
        .dropna()
        .unique()
    ):
        subset = (
            samples[
                samples["horizon_days"]
                == horizon
            ]
            .sort_values(
                "realized_contrarian_return",
                ascending=True,
            )
            .head(
                limit_per_horizon
            )
            .copy()
        )

        subset[
            "tail_rank"
        ] = np.arange(
            1,
            len(subset) + 1,
        )

        blocks.append(
            subset
        )

    if not blocks:
        return pd.DataFrame()

    return pd.concat(
        blocks,
        ignore_index=True,
    )


def repeated_tail_events(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    tail = samples[
        samples[
            "realized_contrarian_return"
        ]
        <= -50.0
    ].copy()

    if tail.empty:
        return pd.DataFrame()

    summary = (
        tail.groupby(
            [
                "event_id",
                "ticker",
                "date",
                "direction_label",
                "initial_setup",
            ],
            observed=True,
        )
        .agg(
            horizons_hit=(
                "horizon_days",
                "nunique",
            ),
            worst_return=(
                "realized_contrarian_return",
                "min",
            ),
            median_return=(
                "realized_contrarian_return",
                "median",
            ),
        )
        .reset_index()
    )

    return summary.sort_values(
        [
            "horizons_hit",
            "worst_return",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )


def write_tail_reports(
    *,
    direction_summary: pd.DataFrame,
    setup_summary: pd.DataFrame,
    segmentation_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
    worst: pd.DataFrame,
    repeated: pd.DataFrame,
    output_dir: str | Path = DEFAULT_REPORT_DIR,
) -> None:
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    direction_summary.to_csv(
        output_dir
        / "raw_median_tail_by_direction.csv",
        index=False,
    )

    setup_summary.to_csv(
        output_dir
        / "raw_median_tail_by_setup.csv",
        index=False,
    )

    segmentation_summary.to_csv(
        output_dir
        / "raw_median_tail_by_segmentation.csv",
        index=False,
    )

    year_summary.to_csv(
        output_dir
        / "raw_median_tail_by_year.csv",
        index=False,
    )

    worst.to_csv(
        output_dir
        / "raw_median_worst_events.csv",
        index=False,
    )

    repeated.to_csv(
        output_dir
        / "raw_median_repeated_tail_events.csv",
        index=False,
    )