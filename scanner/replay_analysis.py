from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_REPLAY_PATH = Path(
    "data/research/replay_reports/historical_evidence_replay.csv"
)

DEFAULT_REPORT_DIR = Path(
    "data/research/replay_reports"
)

ANALYSIS_YEARS = (2021, 2022, 2023, 2024)
HOLDOUT_YEAR = 2025


@dataclass(frozen=True)
class RankingMethod:
    name: str
    score_column: str
    ascending: bool = False


RANKING_METHODS = (
    RankingMethod(
        name="RAW_EDGE",
        score_column="score_raw_edge",
    ),
    RankingMethod(
        name="RAW_WR",
        score_column="score_raw_wr",
    ),
    RankingMethod(
        name="WILSON_WR",
        score_column="score_wilson_wr",
    ),
    RankingMethod(
        name="RAW_MEDIAN",
        score_column="score_raw_median",
    ),
    RankingMethod(
        name="TRAIN_RANK",
        score_column="score_train_rank",
    ),
)


REPLAY_USECOLS = (
    "event_id",
    "ticker",
    "date",
    "test_year",
    "horizon_days",
    "segmentation",
    "train_rank",
    "train_n",
    "train_win_rate",
    "train_median_return",
    "train_edge_score",
    "realized_contrarian_return",
    "realized_win",
    "condition",
    "condition_count",
)


def wilson_lower_bound(
    wins: float,
    n: float,
    *,
    z: float = 1.96,
) -> float:
    """
    Wilson score lower bound for a binomial proportion.

    This gives us a conservative win-rate estimate that naturally
    penalizes small samples without imposing an arbitrary minimum N.
    """
    if n <= 0:
        return 0.0

    p = wins / n
    z2 = z * z

    denominator = 1.0 + z2 / n

    centre = (
        p
        + z2 / (2.0 * n)
    )

    margin = z * math.sqrt(
        (
            p * (1.0 - p) / n
            + z2 / (4.0 * n * n)
        )
    )

    return (
        centre - margin
    ) / denominator


def add_ranking_scores(
    replay: pd.DataFrame,
) -> pd.DataFrame:
    df = replay.copy()

    df["score_raw_edge"] = df[
        "train_edge_score"
    ].astype(float)

    df["score_raw_wr"] = df[
        "train_win_rate"
    ].astype(float)

    df["score_raw_median"] = df[
        "train_median_return"
    ].astype(float)

    # Lower train_rank is better, so negate it so that all ranking
    # methods can consistently sort higher score = stronger evidence.
    df["score_train_rank"] = -df[
        "train_rank"
    ].astype(float)

    wins = (
        df["train_win_rate"].astype(float)
        * df["train_n"].astype(float)
    )

    df["score_wilson_wr"] = [
        wilson_lower_bound(
            win_count,
            n,
        )
        for win_count, n in zip(
            wins,
            df["train_n"].astype(float),
        )
    ]

    return df


def load_replay(
    path: str | Path = DEFAULT_REPLAY_PATH,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Replay dataset not found: {path}"
        )

    print(f"Loading replay dataset: {path}")

    df = pd.read_csv(
        path,
        usecols=list(REPLAY_USECOLS),
        parse_dates=["date"],
    )

    print(
        f"Loaded {len(df):,} evidence rows "
        f"for {df['event_id'].nunique():,} events."
    )

    return df


def select_anchor_matches(
    scored_replay: pd.DataFrame,
    method: RankingMethod,
) -> pd.DataFrame:
    """
    Pick one evidence anchor for each event/horizon using only the
    selected training-derived score.
    """
    sort_columns = [
        "event_id",
        "horizon_days",
        method.score_column,
        "train_n",
        "condition_count",
    ]

    ascending = [
        True,
        True,
        False,
        False,
        False,
    ]

    ordered = scored_replay.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
    )

    anchors = ordered.drop_duplicates(
        subset=[
            "event_id",
            "horizon_days",
        ],
        keep="first",
    ).copy()

    anchors["ranking_method"] = method.name
    anchors["ranking_score"] = anchors[
        method.score_column
    ].astype(float)

    return anchors


def build_method_anchors(
    scored_replay: pd.DataFrame,
) -> pd.DataFrame:
    blocks = []

    for method in RANKING_METHODS:
        print(
            f"Selecting anchors: {method.name}"
        )

        anchors = select_anchor_matches(
            scored_replay,
            method,
        )

        blocks.append(anchors)

    return pd.concat(
        blocks,
        ignore_index=True,
    )


def summarize_method_performance(
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Overall OOS performance for each method/year/horizon.
    """
    summary = (
        anchors.groupby(
            [
                "ranking_method",
                "test_year",
                "horizon_days",
            ],
            observed=True,
        )
        .agg(
            events=(
                "event_id",
                "nunique",
            ),
            realized_win_rate=(
                "realized_win",
                "mean",
            ),
            realized_mean_return=(
                "realized_contrarian_return",
                "mean",
            ),
            realized_median_return=(
                "realized_contrarian_return",
                "median",
            ),
            median_train_n=(
                "train_n",
                "median",
            ),
            median_anchor_score=(
                "ranking_score",
                "median",
            ),
        )
        .reset_index()
    )

    return summary


def add_daily_ranks(
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    ranked = anchors.copy()

    ranked["daily_rank"] = (
        ranked.groupby(
            [
                "ranking_method",
                "date",
                "horizon_days",
            ],
            observed=True,
        )["ranking_score"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    return ranked


def summarize_daily_top_n(
    ranked_anchors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Test actual scanner-style usefulness:
    if we only looked at the top 1/3/5/10 names each day, what happened?
    """
    top_levels = (
        1,
        3,
        5,
        10,
    )

    blocks = []

    for top_n in top_levels:
        subset = ranked_anchors[
            ranked_anchors["daily_rank"]
            <= top_n
        ]

        summary = (
            subset.groupby(
                [
                    "ranking_method",
                    "test_year",
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
                trading_days=(
                    "date",
                    "nunique",
                ),
                realized_win_rate=(
                    "realized_win",
                    "mean",
                ),
                realized_mean_return=(
                    "realized_contrarian_return",
                    "mean",
                ),
                realized_median_return=(
                    "realized_contrarian_return",
                    "median",
                ),
            )
            .reset_index()
        )

        summary["top_n"] = top_n

        blocks.append(summary)

    all_summary = (
        ranked_anchors.groupby(
            [
                "ranking_method",
                "test_year",
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
            trading_days=(
                "date",
                "nunique",
            ),
            realized_win_rate=(
                "realized_win",
                "mean",
            ),
            realized_mean_return=(
                "realized_contrarian_return",
                "mean",
            ),
            realized_median_return=(
                "realized_contrarian_return",
                "median",
            ),
        )
        .reset_index()
    )

    all_summary["top_n"] = 0

    blocks.append(all_summary)

    return pd.concat(
        blocks,
        ignore_index=True,
    )


def summarize_score_deciles(
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check calibration/monotonicity.

    For each method/year/horizon, split anchor scores into ten
    approximately equal-sized buckets.

    Decile 10 contains the strongest ranking scores and decile 1 the
    weakest. If the ranking method is useful, realized performance
    should generally improve as decile increases.

    This implementation avoids DataFrameGroupBy.apply() so grouping
    columns remain intact across pandas versions.
    """
    df = anchors.copy()

    group_columns = [
        "ranking_method",
        "test_year",
        "horizon_days",
    ]

    group_size = (
        df.groupby(
            group_columns,
            observed=True,
        )["ranking_score"]
        .transform("size")
    )

    percentile_rank = (
        df.groupby(
            group_columns,
            observed=True,
        )["ranking_score"]
        .rank(
            method="first",
            pct=True,
            ascending=True,
        )
    )

    valid = group_size >= 10

    df = df.loc[valid].copy()
    percentile_rank = percentile_rank.loc[valid]

    df["score_decile"] = (
        percentile_rank
        .mul(10.0)
        .apply(math.ceil)
        .clip(
            lower=1,
            upper=10,
        )
        .astype(int)
    )

    summary = (
        df.groupby(
            [
                "ranking_method",
                "test_year",
                "horizon_days",
                "score_decile",
            ],
            observed=True,
        )
        .agg(
            events=(
                "event_id",
                "nunique",
            ),
            realized_win_rate=(
                "realized_win",
                "mean",
            ),
            realized_mean_return=(
                "realized_contrarian_return",
                "mean",
            ),
            realized_median_return=(
                "realized_contrarian_return",
                "median",
            ),
            median_train_n=(
                "train_n",
                "median",
            ),
            median_ranking_score=(
                "ranking_score",
                "median",
            ),
        )
        .reset_index()
    )

    return summary


def method_scorecard(
    top_n_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact development-period scorecard.

    Uses 2021-2024 only. 2025 remains untouched as the final holdout.

    This table is descriptive rather than a new optimized composite
    score. It lets us compare the ranking methods on the metrics that
    matter before choosing one.
    """
    dev = top_n_summary[
        top_n_summary["test_year"].isin(
            ANALYSIS_YEARS
        )
    ].copy()

    top5 = dev[
        dev["top_n"] == 5
    ]

    scorecard = (
        top5.groupby(
            [
                "ranking_method",
                "horizon_days",
            ],
            observed=True,
        )
        .agg(
            years=(
                "test_year",
                "nunique",
            ),
            observations=(
                "observations",
                "sum",
            ),
            avg_yearly_win_rate=(
                "realized_win_rate",
                "mean",
            ),
            worst_year_win_rate=(
                "realized_win_rate",
                "min",
            ),
            avg_yearly_median_return=(
                "realized_median_return",
                "mean",
            ),
            worst_year_median_return=(
                "realized_median_return",
                "min",
            ),
            avg_yearly_mean_return=(
                "realized_mean_return",
                "mean",
            ),
        )
        .reset_index()
    )

    return scorecard.sort_values(
        [
            "horizon_days",
            "avg_yearly_median_return",
            "avg_yearly_win_rate",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).reset_index(drop=True)


def write_reports(
    anchors: pd.DataFrame,
    overall: pd.DataFrame,
    top_n: pd.DataFrame,
    deciles: pd.DataFrame,
    scorecard: pd.DataFrame,
    *,
    output_dir: str | Path = DEFAULT_REPORT_DIR,
) -> None:
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    anchors.to_csv(
        output_dir
        / "replay_method_anchors.csv",
        index=False,
    )

    overall.to_csv(
        output_dir
        / "replay_method_performance.csv",
        index=False,
    )

    top_n.to_csv(
        output_dir
        / "replay_daily_top_n.csv",
        index=False,
    )

    deciles.to_csv(
        output_dir
        / "replay_score_deciles.csv",
        index=False,
    )

    scorecard.to_csv(
        output_dir
        / "replay_development_scorecard.csv",
        index=False,
    )