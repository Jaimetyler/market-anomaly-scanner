from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class OutcomeResult:
    signal_price: float

    return_1d: float | None
    return_2d: float | None
    return_3d: float | None
    return_5d: float | None
    return_10d: float | None
    return_20d: float | None

    max_future_gain_pct: float | None
    max_future_decline_pct: float | None

    mfe_pct: float | None
    mae_pct: float | None

    sessions_to_max_gain: int | None
    sessions_to_max_decline: int | None


def _percent_change(
    future_price: float,
    signal_price: float,
) -> float | None:
    if signal_price == 0:
        return None

    return (
        (future_price / signal_price)
        - 1
    ) * 100


def calculate_outcomes(
    future_bars: pd.DataFrame,
    signal_price: float,
) -> OutcomeResult:
    """
    Measure what happened AFTER an anomaly signal.

    future_bars must begin with the session immediately
    following the signal session.

    MFE:
        Maximum Favorable Excursion for a SHORT thesis.
        Therefore falling prices are favorable.

    MAE:
        Maximum Adverse Excursion for a SHORT thesis.
        Therefore rising prices are adverse.

    We also store direction-neutral maximum future
    gain/decline so the research database is not tied
    exclusively to short strategies.
    """

    if signal_price <= 0:
        raise ValueError(
            "signal_price must be greater than zero."
        )

    if future_bars.empty:
        return OutcomeResult(
            signal_price=signal_price,
            return_1d=None,
            return_2d=None,
            return_3d=None,
            return_5d=None,
            return_10d=None,
            return_20d=None,
            max_future_gain_pct=None,
            max_future_decline_pct=None,
            mfe_pct=None,
            mae_pct=None,
            sessions_to_max_gain=None,
            sessions_to_max_decline=None,
        )

    required = {
        "high",
        "low",
        "close",
    }

    missing = (
        required
        - set(
            future_bars.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing required outcome columns: "
            f"{sorted(missing)}"
        )

    df = (
        future_bars
        .reset_index(drop=True)
        .copy()
    )

    def ending_return(
        sessions: int,
    ) -> float | None:
        if len(df) < sessions:
            return None

        close = float(
            df.iloc[
                sessions - 1
            ]["close"]
        )

        return _percent_change(
            close,
            signal_price,
        )

    # Use up to 20 future sessions for initial research.
    window = (
        df.head(20)
        .copy()
    )

    highs = (
        window["high"]
        .astype(float)
    )

    lows = (
        window["low"]
        .astype(float)
    )

    highest_price = float(
        highs.max()
    )

    lowest_price = float(
        lows.min()
    )

    max_future_gain_pct = (
        _percent_change(
            highest_price,
            signal_price,
        )
    )

    max_future_decline_pct = (
        _percent_change(
            lowest_price,
            signal_price,
        )
    )

    max_gain_index = int(
        highs.idxmax()
    )

    max_decline_index = int(
        lows.idxmin()
    )

    # +1 because index zero is the first
    # session AFTER the signal.
    sessions_to_max_gain = (
        max_gain_index + 1
    )

    sessions_to_max_decline = (
        max_decline_index + 1
    )

    # Short-oriented excursion terminology.
    #
    # A 30% decline is +30% favorable excursion.
    # A 20% rally is +20% adverse excursion.

    mfe_pct = (
        abs(
            min(
                max_future_decline_pct,
                0,
            )
        )
        if max_future_decline_pct
        is not None
        else None
    )

    mae_pct = (
        max(
            max_future_gain_pct,
            0,
        )
        if max_future_gain_pct
        is not None
        else None
    )

    return OutcomeResult(
        signal_price=signal_price,

        return_1d=ending_return(1),
        return_2d=ending_return(2),
        return_3d=ending_return(3),
        return_5d=ending_return(5),
        return_10d=ending_return(10),
        return_20d=ending_return(20),

        max_future_gain_pct=(
            max_future_gain_pct
        ),

        max_future_decline_pct=(
            max_future_decline_pct
        ),

        mfe_pct=mfe_pct,
        mae_pct=mae_pct,

        sessions_to_max_gain=(
            sessions_to_max_gain
        ),

        sessions_to_max_decline=(
            sessions_to_max_decline
        ),
    )