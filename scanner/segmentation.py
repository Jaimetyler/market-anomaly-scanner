from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SegmentationConfig:
    """
    Column names used by the research segmentation layer.

    The defaults are intentionally generic. When we wire this into the
    master dataset, we can either rename columns in a thin adapter or pass
    a custom config.
    """

    price_col: str = "event_close"
    rvol_col: str = "rvol"
    sma20_extension_col: str = "sma20_extension_pct"
    move_1d_col: str = "return_1d_event"
    move_3d_col: str = "return_3d_event"
    move_5d_col: str = "return_5d_event"
    dollar_volume_col: str = "dollar_volume"
    market_cap_col: str = "market_cap"


def _as_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def add_research_buckets(
    outcomes: pd.DataFrame,
    config: SegmentationConfig | None = None,
) -> pd.DataFrame:
    """
    Add stable, human-readable research buckets.

    This is deliberately non-destructive: a copy is returned.

    Buckets added when the source column exists:
        price_bucket
        rvol_bucket
        sma20_extension_bucket
        move_1d_bucket
        move_3d_bucket
        move_5d_bucket
        dollar_volume_bucket
        market_cap_bucket

    Percentage-like move/extension columns are expected as decimal returns:
        0.25 == +25%
       -0.25 == -25%
    """
    config = config or SegmentationConfig()
    result = outcomes.copy()

    price = _as_numeric(result, config.price_col)
    if config.price_col in result.columns:
        result["price_bucket"] = pd.cut(
            price,
            bins=[-np.inf, 2, 5, 10, 20, 50, 100, np.inf],
            labels=[
                "<$2",
                "$2-$5",
                "$5-$10",
                "$10-$20",
                "$20-$50",
                "$50-$100",
                "$100+",
            ],
            right=False,
        )

    rvol = _as_numeric(result, config.rvol_col)
    if config.rvol_col in result.columns:
        result["rvol_bucket"] = pd.cut(
            rvol,
            bins=[-np.inf, 1, 1.5, 2, 3, 5, 10, np.inf],
            labels=[
                "<1x",
                "1-1.5x",
                "1.5-2x",
                "2-3x",
                "3-5x",
                "5-10x",
                "10x+",
            ],
            right=False,
        )

    sma20 = _as_numeric(result, config.sma20_extension_col)
    if config.sma20_extension_col in result.columns:
        result["sma20_extension_bucket"] = pd.cut(
            sma20,
            bins=[
                -np.inf,
                -0.50,
                -0.30,
                -0.20,
                -0.10,
                0.10,
                0.20,
                0.30,
                0.50,
                np.inf,
            ],
            labels=[
                "<-50%",
                "-50% to -30%",
                "-30% to -20%",
                "-20% to -10%",
                "-10% to +10%",
                "+10% to +20%",
                "+20% to +30%",
                "+30% to +50%",
                "+50%+",
            ],
            right=False,
        )

    for source_col, output_col in (
        (config.move_1d_col, "move_1d_bucket"),
        (config.move_3d_col, "move_3d_bucket"),
        (config.move_5d_col, "move_5d_bucket"),
    ):
        if source_col not in result.columns:
            continue

        values = _as_numeric(result, source_col)
        result[output_col] = pd.cut(
            values,
            bins=[
                -np.inf,
                -0.50,
                -0.30,
                -0.20,
                -0.10,
                0.10,
                0.20,
                0.30,
                0.50,
                1.00,
                np.inf,
            ],
            labels=[
                "<-50%",
                "-50% to -30%",
                "-30% to -20%",
                "-20% to -10%",
                "-10% to +10%",
                "+10% to +20%",
                "+20% to +30%",
                "+30% to +50%",
                "+50% to +100%",
                "+100%+",
            ],
            right=False,
        )

    dollar_volume = _as_numeric(result, config.dollar_volume_col)
    if config.dollar_volume_col in result.columns:
        result["dollar_volume_bucket"] = pd.cut(
            dollar_volume,
            bins=[
                -np.inf,
                1_000_000,
                5_000_000,
                10_000_000,
                25_000_000,
                50_000_000,
                100_000_000,
                np.inf,
            ],
            labels=[
                "<$1M",
                "$1M-$5M",
                "$5M-$10M",
                "$10M-$25M",
                "$25M-$50M",
                "$50M-$100M",
                "$100M+",
            ],
            right=False,
        )

    market_cap = _as_numeric(result, config.market_cap_col)
    if config.market_cap_col in result.columns:
        result["market_cap_bucket"] = pd.cut(
            market_cap,
            bins=[
                -np.inf,
                300_000_000,
                2_000_000_000,
                10_000_000_000,
                50_000_000_000,
                200_000_000_000,
                np.inf,
            ],
            labels=[
                "<$300M",
                "$300M-$2B",
                "$2B-$10B",
                "$10B-$50B",
                "$50B-$200B",
                "$200B+",
            ],
            right=False,
        )

    return result


def summarize_segments(
    outcomes: pd.DataFrame,
    *,
    group_by: Sequence[str],
    horizons: Iterable[int] = (1, 2, 3, 5, 10, 20),
    min_n: int = 20,
) -> pd.DataFrame:
    """
    Summarize contrarian performance by one or more segment columns.

    Expected outcome columns come from scanner.backtest.compute_event_outcomes().

    Metrics:
        n
        reversal_rate
        contrarian_win_rate
        mean_contrarian_return
        median_contrarian_return
        mean_mfe
        mean_mae
        payoff_ratio
        edge_score

    edge_score is a simple research-ranking statistic, not a trading signal:
        median contrarian return * sqrt(n)

    This intentionally rewards both effect size and sample size without
    pretending to be a full statistical significance test.
    """
    group_by = list(group_by)
    if not group_by:
        raise ValueError("group_by must contain at least one column.")

    missing_groups = [column for column in group_by if column not in outcomes.columns]
    if missing_groups:
        raise ValueError(f"Missing grouping columns: {missing_groups}")

    horizons = tuple(sorted({int(h) for h in horizons}))
    if not horizons or horizons[0] <= 0:
        raise ValueError("horizons must contain positive integers.")

    if min_n < 1:
        raise ValueError("min_n must be at least 1.")

    rows: list[dict] = []

    grouped = outcomes.groupby(
        group_by[0] if len(group_by) == 1 else group_by,
        dropna=False,
        observed=True,
        sort=True,
    )

    for group_key, frame in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_values = dict(zip(group_by, group_key))

        for horizon in horizons:
            suffix = f"{horizon}d"

            required = [
                f"outcome_available_{suffix}",
                f"contrarian_return_{suffix}",
                f"reversed_{suffix}",
                f"contrarian_mfe_{suffix}",
                f"contrarian_mae_{suffix}",
            ]
            missing = [column for column in required if column not in frame.columns]
            if missing:
                raise ValueError(
                    f"Outcome data is missing columns for {horizon}d: {missing}"
                )

            usable = frame[
                frame[f"outcome_available_{suffix}"].fillna(False)
            ].copy()

            returns = pd.to_numeric(
                usable[f"contrarian_return_{suffix}"],
                errors="coerce",
            )
            valid = returns.notna()
            returns = returns[valid]

            if returns.empty:
                continue

            n = int(len(returns))
            if n < min_n:
                continue

            usable_valid = usable.loc[valid].copy()

            reversals = usable_valid[f"reversed_{suffix}"].astype("boolean")
            mfe = pd.to_numeric(
                usable_valid[f"contrarian_mfe_{suffix}"],
                errors="coerce",
            )
            mae = pd.to_numeric(
                usable_valid[f"contrarian_mae_{suffix}"],
                errors="coerce",
            )

            mean_return = float(returns.mean())
            median_return = float(returns.median())
            win_rate = float((returns > 0).mean())
            reversal_rate = (
                float(reversals.mean())
                if reversals.notna().any()
                else np.nan
            )
            mean_mfe = float(mfe.mean()) if mfe.notna().any() else np.nan
            mean_mae = float(mae.mean()) if mae.notna().any() else np.nan

            avg_win = returns[returns > 0].mean()
            avg_loss = returns[returns < 0].mean()

            if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0:
                payoff_ratio = float(avg_win / abs(avg_loss))
            else:
                payoff_ratio = np.nan

            edge_score = float(median_return * np.sqrt(n))

            rows.append(
                {
                    **group_values,
                    "horizon_days": horizon,
                    "n": n,
                    "reversal_rate": reversal_rate,
                    "contrarian_win_rate": win_rate,
                    "mean_contrarian_return": mean_return,
                    "median_contrarian_return": median_return,
                    "mean_mfe": mean_mfe,
                    "mean_mae": mean_mae,
                    "payoff_ratio": payoff_ratio,
                    "edge_score": edge_score,
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        ["horizon_days", "edge_score", "n"],
        ascending=[True, False, False],
        kind="stable",
    ).reset_index(drop=True)


def rank_best_segments(
    summary: pd.DataFrame,
    *,
    horizon_days: int,
    top_n: int = 25,
    min_win_rate: float = 0.0,
    min_median_return: float = -np.inf,
) -> pd.DataFrame:
    """
    Filter and rank already summarized research segments.

    This is for hypothesis generation only. Ranking a setup highly does not
    prove it is tradable; later stages must address out-of-sample testing,
    borrow constraints, liquidity, slippage, and multiple-testing bias.
    """
    required = [
        "horizon_days",
        "n",
        "contrarian_win_rate",
        "median_contrarian_return",
        "edge_score",
    ]
    missing = [column for column in required if column not in summary.columns]
    if missing:
        raise ValueError(f"summary is missing required columns: {missing}")

    filtered = summary[
        (summary["horizon_days"] == int(horizon_days))
        & (summary["contrarian_win_rate"] >= float(min_win_rate))
        & (summary["median_contrarian_return"] >= float(min_median_return))
    ].copy()

    return (
        filtered.sort_values(
            ["edge_score", "n"],
            ascending=[False, False],
            kind="stable",
        )
        .head(int(top_n))
        .reset_index(drop=True)
    )
