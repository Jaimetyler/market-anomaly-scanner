from __future__ import annotations

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

HOLDOUT_YEAR = 2025

TOP_LEVELS = (
    1,
    3,
    5,
    10,
)


ANCHOR_USECOLS = (
    "event_id",
    "ticker",
    "date",
    "test_year",
    "horizon_days",
    "segmentation",
    "train_n",
    "train_win_rate",
    "train_median_return",
    "ranking_method",
    "ranking_score",
    "realized_contrarian_return",
    "condition",
    "condition_count",
)


def load_raw_median_anchors(
    path: str | Path = DEFAULT_ANCHORS_PATH,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Anchor dataset not found: {path}"
        )

    print(f"Loading anchor dataset: {path}")

    df = pd.read_csv(
        path,
        usecols=list(ANCHOR_USECOLS),
        parse_dates=["date"],
    )

    df = df[
        df["ranking_method"] == RANKING_METHOD
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"No {RANKING_METHOD} anchors found."
        )

    df["test_year"] = (
        df["test_year"]
        .astype(int)
    )

    df["horizon_days"] = (
        df["horizon_days"]
        .astype(int)
    )

    print(
        f"Loaded {len(df):,} "
        f"{RANKING_METHOD} anchor rows "
        f"for {df['event_id'].nunique():,} events."
    )

    return df


def add_daily_ranks(
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank RAW_MEDIAN anchors within each date and horizon.

    Rank 1 = strongest training-period median contrarian return.
    """
    df = anchors.copy()

    df["daily_rank"] = (
        df.groupby(
            [
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

    return df


def load_excursion_outcomes(
    events_path: str | Path = DEFAULT_EVENTS_PATH,
) -> pd.DataFrame:
    """
    Convert horizon-specific historical MFE/MAE columns into a
    long-form event/horizon table.

    Missing excursion columns are tolerated. This lets the analysis
    still run if some horizons were not populated.
    """
    events_path = Path(events_path)

    if not events_path.exists():
        raise FileNotFoundError(
            f"Historical event outcomes not found: "
            f"{events_path}"
        )

    print(
        f"Loading excursion outcomes: "
        f"{events_path}"
    )

    header = pd.read_csv(
        events_path,
        nrows=0,
    )

    available_columns = set(
        header.columns
    )

    horizons = (
        1,
        2,
        3,
        5,
        10,
        20,
    )

    requested_columns = {
        "event_id",
    }

    for horizon in horizons:
        requested_columns.add(
            f"contrarian_mfe_{horizon}d"
        )
        requested_columns.add(
            f"contrarian_mae_{horizon}d"
        )

    usecols = [
        column
        for column in requested_columns
        if column in available_columns
    ]

    if "event_id" not in usecols:
        raise ValueError(
            "Event outcomes file does not contain event_id."
        )

    events = pd.read_csv(
        events_path,
        usecols=usecols,
    )

    blocks = []

    for horizon in horizons:
        mfe_column = (
            f"contrarian_mfe_{horizon}d"
        )

        mae_column = (
            f"contrarian_mae_{horizon}d"
        )

        if (
            mfe_column not in events.columns
            and mae_column not in events.columns
        ):
            continue

        block = pd.DataFrame(
            {
                "event_id": events[
                    "event_id"
                ],
                "horizon_days": horizon,
            }
        )

        if mfe_column in events.columns:
            block["contrarian_mfe"] = (
                pd.to_numeric(
                    events[mfe_column],
                    errors="coerce",
                )
            )
        else:
            block["contrarian_mfe"] = np.nan

        if mae_column in events.columns:
            block["contrarian_mae"] = (
                pd.to_numeric(
                    events[mae_column],
                    errors="coerce",
                )
            )
        else:
            block["contrarian_mae"] = np.nan

        blocks.append(block)

    if not blocks:
        return pd.DataFrame(
            columns=[
                "event_id",
                "horizon_days",
                "contrarian_mfe",
                "contrarian_mae",
            ]
        )

    excursions = pd.concat(
        blocks,
        ignore_index=True,
    )

    excursions = (
        excursions.drop_duplicates(
            subset=[
                "event_id",
                "horizon_days",
            ],
            keep="first",
        )
    )

    return excursions


def attach_excursions(
    anchors: pd.DataFrame,
    excursions: pd.DataFrame,
) -> pd.DataFrame:
    df = anchors.copy()

    if excursions.empty:
        df["contrarian_mfe"] = np.nan
        df["contrarian_mae"] = np.nan
        return df

    merged = df.merge(
        excursions,
        on=[
            "event_id",
            "horizon_days",
        ],
        how="left",
        validate="many_to_one",
    )

    return merged


def _tail_mean(
    values: pd.Series,
    quantile: float,
) -> float:
    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return np.nan

    cutoff = clean.quantile(
        quantile
    )

    tail = clean[
        clean <= cutoff
    ]

    if tail.empty:
        return np.nan

    return float(
        tail.mean()
    )


def _average_win(
    values: pd.Series,
) -> float:
    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    winners = clean[
        clean > 0
    ]

    if winners.empty:
        return np.nan

    return float(
        winners.mean()
    )


def _average_loss(
    values: pd.Series,
) -> float:
    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    losers = clean[
        clean < 0
    ]

    if losers.empty:
        return np.nan

    return float(
        losers.mean()
    )


def _profit_factor(
    values: pd.Series,
) -> float:
    """
    Gross positive return divided by absolute gross negative return.

    This is a distribution diagnostic, NOT a literal portfolio
    backtest. Events can overlap and no position sizing is applied.
    """
    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return np.nan

    gross_profit = clean[
        clean > 0
    ].sum()

    gross_loss = clean[
        clean < 0
    ].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf

        return np.nan

    return float(
        gross_profit
        / abs(gross_loss)
    )


def _payoff_ratio(
    values: pd.Series,
) -> float:
    avg_win = _average_win(
        values
    )

    avg_loss = _average_loss(
        values
    )

    if (
        pd.isna(avg_win)
        or pd.isna(avg_loss)
        or avg_loss == 0
    ):
        return np.nan

    return float(
        avg_win
        / abs(avg_loss)
    )


def _period_label(
    year: int,
) -> str:
    if year in DEVELOPMENT_YEARS:
        return "DEVELOPMENT_2021_2024"

    if year == HOLDOUT_YEAR:
        return "HOLDOUT_2025"

    return f"YEAR_{year}"


def build_top_n_samples(
    ranked_anchors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Materialize top 1/3/5/10 daily samples.

    A single anchor can therefore appear in multiple top-N groups.
    """
    blocks = []

    for top_n in TOP_LEVELS:
        block = ranked_anchors[
            ranked_anchors["daily_rank"]
            <= top_n
        ].copy()

        block["top_n"] = top_n

        blocks.append(block)

    return pd.concat(
        blocks,
        ignore_index=True,
    )


def summarize_risk_group(
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

    avg_win = _average_win(
        returns
    )

    avg_loss = _average_loss(
        returns
    )

    mfe = pd.to_numeric(
        group.get(
            "contrarian_mfe",
            pd.Series(
                dtype=float,
            ),
        ),
        errors="coerce",
    ).dropna()

    mae = pd.to_numeric(
        group.get(
            "contrarian_mae",
            pd.Series(
                dtype=float,
            ),
        ),
        errors="coerce",
    ).dropna()

    return pd.Series(
        {
            "observations": len(
                returns
            ),
            "unique_events": group[
                "event_id"
            ].nunique(),
            "trading_days": group[
                "date"
            ].nunique(),
            "win_rate": (
                returns > 0
            ).mean(),
            "loss_rate": (
                returns < 0
            ).mean(),
            "mean_return": (
                returns.mean()
            ),
            "median_return": (
                returns.median()
            ),
            "stdev_return": (
                returns.std(
                    ddof=1
                )
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
            "p25_return": (
                returns.quantile(
                    0.25
                )
            ),
            "p75_return": (
                returns.quantile(
                    0.75
                )
            ),
            "p90_return": (
                returns.quantile(
                    0.90
                )
            ),
            "p95_return": (
                returns.quantile(
                    0.95
                )
            ),
            "worst_return": (
                returns.min()
            ),
            "best_return": (
                returns.max()
            ),
            "average_win": avg_win,
            "average_loss": avg_loss,
            "payoff_ratio": (
                _payoff_ratio(
                    returns
                )
            ),
            "profit_factor": (
                _profit_factor(
                    returns
                )
            ),
            "cvar_05": (
                _tail_mean(
                    returns,
                    0.05,
                )
            ),
            "cvar_10": (
                _tail_mean(
                    returns,
                    0.10,
                )
            ),
            "mfe_coverage": (
                len(mfe)
                / len(group)
                if len(group)
                else np.nan
            ),
            "mae_coverage": (
                len(mae)
                / len(group)
                if len(group)
                else np.nan
            ),
            "median_mfe": (
                mfe.median()
                if not mfe.empty
                else np.nan
            ),
            "p90_mfe": (
                mfe.quantile(
                    0.90
                )
                if not mfe.empty
                else np.nan
            ),
            "median_mae": (
                mae.median()
                if not mae.empty
                else np.nan
            ),
            "p10_mae": (
                mae.quantile(
                    0.10
                )
                if not mae.empty
                else np.nan
            ),
        }
    )


def summarize_by_year(
    top_samples: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        top_samples.groupby(
            [
                "test_year",
                "horizon_days",
                "top_n",
            ],
            observed=True,
        )
        .apply(
            summarize_risk_group,
            include_groups=False,
        )
        .reset_index()
    )

    summary["period"] = (
        summary["test_year"]
        .astype(int)
        .map(
            _period_label
        )
    )

    return summary


def summarize_period(
    top_samples: pd.DataFrame,
) -> pd.DataFrame:
    df = top_samples.copy()

    df["period"] = np.where(
        df["test_year"].isin(
            DEVELOPMENT_YEARS
        ),
        "DEVELOPMENT_2021_2024",
        np.where(
            df["test_year"]
            == HOLDOUT_YEAR,
            "HOLDOUT_2025",
            "OTHER",
        ),
    )

    df = df[
        df["period"] != "OTHER"
    ]

    summary = (
        df.groupby(
            [
                "period",
                "horizon_days",
                "top_n",
            ],
            observed=True,
        )
        .apply(
            summarize_risk_group,
            include_groups=False,
        )
        .reset_index()
    )

    return summary


def write_risk_reports(
    top_samples: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    period_summary: pd.DataFrame,
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

    top_samples.to_csv(
        output_dir
        / "raw_median_top_n_samples.csv",
        index=False,
    )

    yearly_summary.to_csv(
        output_dir
        / "raw_median_risk_by_year.csv",
        index=False,
    )

    period_summary.to_csv(
        output_dir
        / "raw_median_risk_summary.csv",
        index=False,
    )