from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EVENTS_PATH = Path(
    "data/research/edge_reports/event_outcomes_adapted.csv"
)

REPORT_DIR = Path(
    "data/research/replay_reports"
)

HORIZONS = (
    1,
    2,
    3,
    5,
    10,
    20,
)

ABS_RETURN_THRESHOLDS = (
    100,
    200,
    500,
    1000,
    5000,
    10000,
)


def find_columns(
    columns: list[str],
    fragments: tuple[str, ...],
) -> list[str]:
    matches = []

    for column in columns:
        lower = column.lower()

        if any(
            fragment in lower
            for fragment in fragments
        ):
            matches.append(
                column
            )

    return matches


def load_events() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing event dataset: {EVENTS_PATH}"
        )

    print(
        f"Loading: {EVENTS_PATH}"
    )

    df = pd.read_csv(
        EVENTS_PATH,
        low_memory=False,
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    return df


def print_relevant_columns(
    df: pd.DataFrame,
) -> None:
    columns = list(
        df.columns
    )

    print()
    print("=" * 120)
    print("POTENTIALLY RELEVANT COLUMNS")
    print("=" * 120)

    groups = {
        "PRICE": (
            "price",
            "close",
            "open",
        ),
        "RETURN": (
            "return",
            "ret_",
        ),
        "CONTRARIAN": (
            "contrarian",
        ),
        "FORWARD": (
            "forward",
            "fwd",
            "future",
        ),
        "EVENT": (
            "event",
        ),
    }

    for label, fragments in groups.items():
        matches = find_columns(
            columns,
            fragments,
        )

        print()
        print(label)
        print("-" * 120)

        for column in matches:
            print(
                column
            )


def contrarian_return_columns(
    df: pd.DataFrame,
) -> dict[int, str]:
    result: dict[int, str] = {}

    for horizon in HORIZONS:
        candidates = [
            f"contrarian_return_{horizon}d",
            f"contrarian_ret_{horizon}d",
            f"contrarian_{horizon}d_return",
        ]

        for candidate in candidates:
            if candidate in df.columns:
                result[
                    horizon
                ] = candidate

                break

    return result


def raw_return_columns(
    df: pd.DataFrame,
) -> dict[int, str]:
    result: dict[int, str] = {}

    for horizon in HORIZONS:
        candidates = [
            f"forward_return_{horizon}d",
            f"return_{horizon}d",
            f"fwd_return_{horizon}d",
            f"future_return_{horizon}d",
        ]

        for candidate in candidates:
            if candidate in df.columns:
                result[
                    horizon
                ] = candidate

                break

    return result


def build_long_returns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    contra = contrarian_return_columns(
        df
    )

    raw = raw_return_columns(
        df
    )

    print()
    print("=" * 120)
    print("DETECTED HORIZON RETURN COLUMNS")
    print("=" * 120)

    print(
        f"Contrarian: {contra}"
    )

    print(
        f"Raw/forward: {raw}"
    )

    id_columns = [
        column
        for column in (
            "event_id",
            "ticker",
            "date",
            "event_date",
            "event_close",
            "event_direction",
            "initial_setup",
        )
        if column in df.columns
    ]

    blocks = []

    for horizon, column in contra.items():
        usecols = [
            *id_columns,
            column,
        ]

        if horizon in raw:
            usecols.append(
                raw[horizon]
            )

        block = df[
            usecols
        ].copy()

        block[
            "horizon_days"
        ] = horizon

        block[
            "contrarian_return"
        ] = pd.to_numeric(
            block[column],
            errors="coerce",
        )

        if horizon in raw:
            block[
                "raw_forward_return"
            ] = pd.to_numeric(
                block[
                    raw[horizon]
                ],
                errors="coerce",
            )
        else:
            block[
                "raw_forward_return"
            ] = np.nan

        block[
            "source_contrarian_column"
        ] = column

        block = block.drop(
            columns=[
                column,
                *(
                    [raw[horizon]]
                    if horizon in raw
                    else []
                ),
            ],
            errors="ignore",
        )

        blocks.append(
            block
        )

    if not blocks:
        raise ValueError(
            "Could not find any contrarian return columns."
        )

    return pd.concat(
        blocks,
        ignore_index=True,
    )


def summarize_integrity(
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for horizon, group in long_df.groupby(
        "horizon_days",
        observed=True,
    ):
        returns = (
            pd.to_numeric(
                group[
                    "contrarian_return"
                ],
                errors="coerce",
            )
            .dropna()
        )

        row = {
            "horizon_days": int(
                horizon
            ),
            "observations": len(
                returns
            ),
            "min_return": (
                returns.min()
            ),
            "max_return": (
                returns.max()
            ),
            "p001": (
                returns.quantile(
                    0.001
                )
            ),
            "p01": (
                returns.quantile(
                    0.01
                )
            ),
            "p99": (
                returns.quantile(
                    0.99
                )
            ),
            "p999": (
                returns.quantile(
                    0.999
                )
            ),
        }

        for threshold in (
            ABS_RETURN_THRESHOLDS
        ):
            row[
                f"abs_ge_{threshold}"
            ] = (
                returns.abs()
                >= threshold
            ).sum()

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def build_extreme_rows(
    long_df: pd.DataFrame,
    threshold: float = 500.0,
) -> pd.DataFrame:
    extreme = long_df[
        long_df[
            "contrarian_return"
        ].abs()
        >= threshold
    ].copy()

    extreme[
        "abs_contrarian_return"
    ] = extreme[
        "contrarian_return"
    ].abs()

    return extreme.sort_values(
        [
            "abs_contrarian_return",
            "ticker",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )


def print_extremes(
    extreme: pd.DataFrame,
) -> None:
    print()
    print("=" * 150)
    print(
        "LARGEST ABSOLUTE CONTRARIAN RETURNS"
    )
    print("=" * 150)

    columns = [
        column
        for column in (
            "ticker",
            "date",
            "event_date",
            "event_id",
            "event_close",
            "event_direction",
            "initial_setup",
            "horizon_days",
            "contrarian_return",
            "raw_forward_return",
        )
        if column in extreme.columns
    ]

    if extreme.empty:
        print(
            "No extreme observations."
        )

        return

    print(
        extreme[
            columns
        ]
        .head(75)
        .to_string(
            index=False
        )
    )


def print_problem_tickers(
    extreme: pd.DataFrame,
) -> None:
    if (
        extreme.empty
        or "ticker"
        not in extreme.columns
    ):
        return

    summary = (
        extreme.groupby(
            "ticker",
            observed=True,
        )
        .agg(
            extreme_rows=(
                "contrarian_return",
                "size",
            ),
            worst_abs_return=(
                "abs_contrarian_return",
                "max",
            ),
            first_horizon=(
                "horizon_days",
                "min",
            ),
            last_horizon=(
                "horizon_days",
                "max",
            ),
        )
        .sort_values(
            [
                "extreme_rows",
                "worst_abs_return",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index()
    )

    print()
    print("=" * 120)
    print("REPEATED EXTREME TICKERS")
    print("=" * 120)

    print(
        summary.head(
            50
        ).to_string(
            index=False
        )
    )


def main() -> None:
    print("=" * 120)
    print(
        "HISTORICAL RETURN INTEGRITY AUDIT"
    )
    print("=" * 120)

    df = load_events()

    print_relevant_columns(
        df
    )

    long_df = build_long_returns(
        df
    )

    summary = summarize_integrity(
        long_df
    )

    extreme = build_extreme_rows(
        long_df,
        threshold=500.0,
    )

    print()
    print("=" * 120)
    print("RETURN DISTRIBUTION AUDIT")
    print("=" * 120)

    print(
        summary.to_string(
            index=False
        )
    )

    print_extremes(
        extreme
    )

    print_problem_tickers(
        extreme
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        REPORT_DIR
        / "return_integrity_summary.csv",
        index=False,
    )

    extreme.to_csv(
        REPORT_DIR
        / "return_integrity_extremes.csv",
        index=False,
    )

    print()
    print("=" * 120)
    print("REPORTS WRITTEN")
    print("=" * 120)

    print(
        REPORT_DIR
        / "return_integrity_summary.csv"
    )

    print(
        REPORT_DIR
        / "return_integrity_extremes.csv"
    )


if __name__ == "__main__":
    main()