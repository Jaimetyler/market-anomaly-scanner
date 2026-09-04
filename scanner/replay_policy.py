from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ANCHORS_PATH = Path(
    "data/research/replay_reports/replay_method_anchors.csv"
)

DEFAULT_EVENTS_PATH = Path(
    "data/research/edge_reports/event_outcomes_adapted.csv"
)

DEFAULT_REPORT_DIR = Path(
    "data/research/replay_reports"
)

RANKING_METHOD = "RAW_MEDIAN"

DEVELOPMENT_YEARS = (
    2021,
    2022,
    2023,
    2024,
)

VALIDATION_YEAR = 2025

TOP_LEVELS = (
    1,
    3,
    5,
    10,
)


@dataclass(frozen=True)
class ScannerPolicy:
    name: str
    min_price: float | None


POLICIES = (
    ScannerPolicy(
        name="BASELINE",
        min_price=None,
    ),
    ScannerPolicy(
        name="PRICE_GE_2",
        min_price=2.0,
    ),
    ScannerPolicy(
        name="PRICE_GE_5",
        min_price=5.0,
    ),
    ScannerPolicy(
        name="PRICE_GE_10",
        min_price=10.0,
    ),
)


def load_inputs(
    anchors_path: str | Path = DEFAULT_ANCHORS_PATH,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
) -> pd.DataFrame:
    anchors_path = Path(
        anchors_path
    )

    events_path = Path(
        events_path
    )

    if not anchors_path.exists():
        raise FileNotFoundError(
            f"Anchor report not found: {anchors_path}"
        )

    if not events_path.exists():
        raise FileNotFoundError(
            f"Event outcomes not found: {events_path}"
        )

    print(
        f"Loading ranking anchors: {anchors_path}"
    )

    anchors = pd.read_csv(
        anchors_path,
        parse_dates=["date"],
    )

    anchors = anchors[
        anchors["ranking_method"]
        == RANKING_METHOD
    ].copy()

    print(
        f"RAW_MEDIAN anchors: {len(anchors):,}"
    )

    print(
        f"Loading event prices: {events_path}"
    )

    metadata = pd.read_csv(
        events_path,
        usecols=[
            "event_id",
            "event_close",
        ],
    )

    metadata = (
        metadata.drop_duplicates(
            subset=["event_id"],
            keep="first",
        )
    )

    anchors = anchors.drop(
        columns=[
            "event_close",
        ],
        errors="ignore",
    )

    df = anchors.merge(
        metadata,
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    numeric_columns = (
        "test_year",
        "horizon_days",
        "ranking_score",
        "realized_contrarian_return",
        "event_close",
    )

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    missing_price = (
        df["event_close"]
        .isna()
        .sum()
    )

    print(
        f"Missing event prices: {missing_price:,}"
    )

    return df


def apply_policy(
    anchors: pd.DataFrame,
    policy: ScannerPolicy,
) -> pd.DataFrame:
    df = anchors.copy()

    if policy.min_price is None:
        return df

    return df[
        df["event_close"]
        >= policy.min_price
    ].copy()


def rank_policy_universe(
    anchors: pd.DataFrame,
    policy: ScannerPolicy,
) -> pd.DataFrame:
    """
    Apply risk eligibility FIRST, then rerank the surviving universe.

    This mirrors intended live-scanner behavior.
    """
    eligible = apply_policy(
        anchors,
        policy,
    )

    eligible = eligible.sort_values(
        [
            "date",
            "horizon_days",
            "ranking_score",
            "train_n",
            "condition_count",
        ],
        ascending=[
            True,
            True,
            False,
            False,
            False,
        ],
        kind="mergesort",
    ).copy()

    eligible["policy"] = (
        policy.name
    )

    eligible["daily_rank"] = (
        eligible.groupby(
            [
                "date",
                "horizon_days",
            ],
            observed=True,
        )
        .cumcount()
        + 1
    )

    return eligible


def build_policy_samples(
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    blocks = []

    for policy in POLICIES:
        print(
            f"Reranking policy: {policy.name}"
        )

        ranked = rank_policy_universe(
            anchors,
            policy,
        )

        for top_n in TOP_LEVELS:
            block = ranked[
                ranked["daily_rank"]
                <= top_n
            ].copy()

            block["top_n"] = top_n

            blocks.append(
                block
            )

    return pd.concat(
        blocks,
        ignore_index=True,
    )


def tail_mean(
    values: pd.Series,
    q: float = 0.05,
) -> float:
    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return np.nan

    cutoff = clean.quantile(
        q
    )

    tail = clean[
        clean <= cutoff
    ]

    if tail.empty:
        return np.nan

    return float(
        tail.mean()
    )


def profit_factor(
    values: pd.Series,
) -> float:
    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return np.nan

    gains = clean[
        clean > 0
    ].sum()

    losses = clean[
        clean < 0
    ].sum()

    if losses == 0:
        if gains > 0:
            return np.inf

        return np.nan

    return float(
        gains
        / abs(losses)
    )


def summarize_group(
    group: pd.DataFrame,
) -> pd.Series:
    returns = pd.to_numeric(
        group[
            "realized_contrarian_return"
        ],
        errors="coerce",
    ).dropna()

    if returns.empty:
        return pd.Series(
            dtype=float
        )

    return pd.Series(
        {
            "observations": len(
                returns
            ),
            "trading_days": group[
                "date"
            ].nunique(),
            "win_rate": (
                returns > 0
            ).mean(),
            "mean_return": (
                returns.mean()
            ),
            "median_return": (
                returns.median()
            ),
            "p05_return": (
                returns.quantile(
                    0.05
                )
            ),
            "p10_return": (
                returns.quantile(
                    0.10
                )
            ),
            "cvar_05": tail_mean(
                returns,
                0.05,
            ),
            "worst_return": (
                returns.min()
            ),
            "loss_le_50_rate": (
                returns <= -50
            ).mean(),
            "loss_le_100_rate": (
                returns <= -100
            ).mean(),
            "loss_le_200_rate": (
                returns <= -200
            ).mean(),
            "profit_factor": (
                profit_factor(
                    returns
                )
            ),
            "median_entry_price": (
                pd.to_numeric(
                    group[
                        "event_close"
                    ],
                    errors="coerce",
                )
                .median()
            ),
        }
    )


def summarize_yearly(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    return (
        samples.groupby(
            [
                "policy",
                "test_year",
                "horizon_days",
                "top_n",
            ],
            observed=True,
        )
        .apply(
            summarize_group,
            include_groups=False,
        )
        .reset_index()
    )


def summarize_periods(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    df = samples.copy()

    df["period"] = np.where(
        df["test_year"].isin(
            DEVELOPMENT_YEARS
        ),
        "DEVELOPMENT_2021_2024",
        np.where(
            df["test_year"]
            == VALIDATION_YEAR,
            "VALIDATION_2025",
            "OTHER",
        ),
    )

    df = df[
        df["period"]
        != "OTHER"
    ]

    return (
        df.groupby(
            [
                "policy",
                "period",
                "horizon_days",
                "top_n",
            ],
            observed=True,
        )
        .apply(
            summarize_group,
            include_groups=False,
        )
        .reset_index()
    )


def write_reports(
    samples: pd.DataFrame,
    yearly: pd.DataFrame,
    periods: pd.DataFrame,
    *,
    output_dir: str | Path = DEFAULT_REPORT_DIR,
) -> None:
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples.to_csv(
        output_dir
        / "scanner_policy_samples.csv",
        index=False,
    )

    yearly.to_csv(
        output_dir
        / "scanner_policy_by_year.csv",
        index=False,
    )

    periods.to_csv(
        output_dir
        / "scanner_policy_summary.csv",
        index=False,
    )