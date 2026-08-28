from __future__ import annotations

from datetime import time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


MARKET_TIMEZONE = ZoneInfo("America/New_York")
REGULAR_MARKET_CLOSE = time(16, 0)


# ============================================================
# DATA PREPARATION
# ============================================================

def bars_to_dataframe(
    bars: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert Massive aggregate bars into a clean pandas DataFrame.

    Expected Massive fields:
        o  = open
        h  = high
        l  = low
        c  = close
        v  = volume
        t  = timestamp in milliseconds
        vw = volume-weighted average price (optional)
    """

    if not bars:
        raise ValueError("No market bars were provided.")

    df = pd.DataFrame(bars).copy()

    required = {"o", "h", "l", "c", "v", "t"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required bar fields: {sorted(missing)}"
        )

    df = df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "t": "timestamp",
            "vw": "vwap",
        }
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if "vwap" in df.columns:
        numeric_columns.append("vwap")

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df
        .dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# BASIC CALCULATIONS
# ============================================================

def percent_change(
    current: float,
    previous: float,
) -> float | None:
    """
    Calculate percentage change from previous to current.
    """

    if previous is None or previous == 0:
        return None

    return ((current / previous) - 1) * 100


def return_over_sessions(
    closes: pd.Series,
    sessions: int,
) -> float | None:
    """
    Calculate close-to-close return over N trading sessions.

    Example:
        sessions=5 compares today's close with the close
        five trading sessions earlier.
    """

    if sessions <= 0:
        raise ValueError(
            "sessions must be greater than zero."
        )

    if len(closes) <= sessions:
        return None

    current = float(closes.iloc[-1])
    previous = float(
        closes.iloc[-1 - sessions]
    )

    return percent_change(
        current=current,
        previous=previous,
    )


# ============================================================
# MOVING AVERAGES
# ============================================================

def sma(
    series: pd.Series,
    period: int,
) -> float | None:
    """
    Calculate a simple moving average.
    """

    if period <= 0:
        raise ValueError(
            "period must be greater than zero."
        )

    if len(series) < period:
        return None

    value = (
        series
        .rolling(window=period)
        .mean()
        .iloc[-1]
    )

    if pd.isna(value):
        return None

    return float(value)


def distance_from_level_pct(
    price: float,
    level: float | None,
) -> float | None:
    """
    Calculate how far price is above or below a reference level.

    Positive:
        price is above the level.

    Negative:
        price is below the level.
    """

    if level is None or level == 0:
        return None

    return ((price / level) - 1) * 100


# ============================================================
# VOLUME
# ============================================================

def average_volume(
    volumes: pd.Series,
    period: int = 20,
    exclude_current: bool = True,
) -> float | None:
    """
    Calculate average volume.

    By default, today's volume is excluded so today's abnormal
    volume does not contaminate the baseline used to calculate
    relative volume.
    """

    if period <= 0:
        raise ValueError(
            "period must be greater than zero."
        )

    if exclude_current:
        series = volumes.iloc[:-1]
    else:
        series = volumes

    if len(series) < period:
        return None

    value = series.tail(period).mean()

    if pd.isna(value):
        return None

    return float(value)


def relative_volume(
    current_volume: float,
    average_volume_value: float | None,
) -> float | None:
    """
    Calculate relative volume.

    Example:
        5.0 means current volume is five times normal volume.
    """

    if (
        average_volume_value is None
        or average_volume_value == 0
    ):
        return None

    return current_volume / average_volume_value


# ============================================================
# WILDER SMOOTHING
# ============================================================

def wilder_smoothed_series(
    values: pd.Series,
    period: int,
) -> pd.Series:
    """
    Calculate a Wilder-smoothed series.

    Wilder's method differs from simply calling pandas ewm().
    The initial value is the arithmetic mean of the first
    `period` valid observations.

    Every subsequent value is:

        ((previous * (period - 1)) + current) / period

    The returned Series uses the same index as the input.
    Values before the initial seed are NaN.
    """

    if period <= 0:
        raise ValueError(
            "period must be greater than zero."
        )

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).astype(float)

    result = pd.Series(
        float("nan"),
        index=numeric.index,
        dtype="float64",
    )

    valid_positions = [
        position
        for position, value in enumerate(
            numeric.tolist()
        )
        if not pd.isna(value)
    ]

    if len(valid_positions) < period:
        return result

    seed_positions = valid_positions[:period]

    seed = float(
        numeric.iloc[seed_positions].mean()
    )

    seed_position = seed_positions[-1]
    result.iloc[seed_position] = seed

    previous = seed

    for position in valid_positions[period:]:
        current = float(
            numeric.iloc[position]
        )

        previous = (
            (
                previous * (period - 1)
            )
            + current
        ) / period

        result.iloc[position] = previous

    return result


# ============================================================
# RSI
# ============================================================

def rsi(
    closes: pd.Series,
    period: int = 14,
) -> float | None:
    """
    Calculate RSI using Wilder's original smoothing method.

    The initial average gain and average loss are simple
    arithmetic averages of the first `period` price changes.

    Subsequent averages use Wilder's recursive smoothing.
    """

    if period <= 0:
        raise ValueError(
            "period must be greater than zero."
        )

    closes = pd.to_numeric(
        closes,
        errors="coerce",
    ).astype(float)

    if len(closes) < period + 1:
        return None

    if closes.isna().any():
        return None

    delta = closes.diff().iloc[1:]

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain_series = wilder_smoothed_series(
        gains,
        period,
    )

    avg_loss_series = wilder_smoothed_series(
        losses,
        period,
    )

    gain = avg_gain_series.iloc[-1]
    loss = avg_loss_series.iloc[-1]

    if pd.isna(gain) or pd.isna(loss):
        return None

    if gain == 0 and loss == 0:
        return 50.0

    if loss == 0:
        return 100.0

    if gain == 0:
        return 0.0

    rs = gain / loss

    value = 100 - (
        100 / (1 + rs)
    )

    return float(value)


# ============================================================
# TRUE RANGE / ATR
# ============================================================

def true_range_series(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate True Range for every bar.

    True Range is the maximum of:

        high - low
        abs(high - previous close)
        abs(low - previous close)

    For the first bar, where no previous close exists,
    high - low is used.
    """

    previous_close = df["close"].shift(1)

    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"]
                - previous_close
            ).abs(),
            (
                df["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    )

    return ranges.max(axis=1)


def atr_series(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Calculate the full Wilder ATR series.

    The first ATR is the arithmetic mean of the first
    `period` True Range observations.

    Each later ATR uses Wilder's recursive smoothing.
    """

    if period <= 0:
        raise ValueError(
            "period must be greater than zero."
        )

    true_range = true_range_series(df)

    return wilder_smoothed_series(
        true_range,
        period,
    )


def atr(
    df: pd.DataFrame,
    period: int = 14,
) -> float | None:
    """
    Calculate Average True Range using Wilder's
    original smoothing method.
    """

    if period <= 0:
        raise ValueError(
            "period must be greater than zero."
        )

    if len(df) < period:
        return None

    series = atr_series(
        df,
        period=period,
    )

    value = series.iloc[-1]

    if pd.isna(value):
        return None

    return float(value)


def atr_expansion(
    df: pd.DataFrame,
    atr_period: int = 14,
    baseline_period: int = 20,
) -> float | None:
    """
    Compare current Wilder ATR with its recent baseline.

    Example:
        2.0 means current ATR is approximately twice
        its recent average level.

    The baseline excludes the current ATR observation.
    """

    if (
        atr_period <= 0
        or baseline_period <= 0
    ):
        raise ValueError(
            "ATR and baseline periods must be "
            "greater than zero."
        )

    minimum_rows = (
        atr_period
        + baseline_period
    )

    if len(df) < minimum_rows:
        return None

    series = atr_series(
        df,
        period=atr_period,
    )

    current_atr = series.iloc[-1]

    baseline = (
        series
        .iloc[:-1]
        .dropna()
        .tail(baseline_period)
        .mean()
    )

    if (
        pd.isna(current_atr)
        or pd.isna(baseline)
        or baseline == 0
    ):
        return None

    return float(
        current_atr / baseline
    )


# ============================================================
# COMPLETED SESSION HANDLING
# ============================================================

def remove_incomplete_session(
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Remove today's still-forming U.S. daily candle.

    Daily aggregate timestamps are converted to
    America/New_York before determining the session date.

    If the newest bar belongs to today's New York trading
    date and regular trading has not yet reached 4:00 PM ET,
    that bar is removed.

    At or after 4:00 PM ET, today's regular session is
    considered complete and the bar is retained.

    `now` is injectable so this behavior can be tested
    deterministically.

    This function intentionally handles regular-session
    completion only. Intraday scanning is separate.
    """

    if df.empty:
        return df

    if now is None:
        current = pd.Timestamp.now(
            tz=MARKET_TIMEZONE,
        )
    else:
        current = pd.Timestamp(now)

        if current.tzinfo is None:
            raise ValueError(
                "now must be timezone-aware."
            )

        current = current.tz_convert(
            MARKET_TIMEZONE
        )

    latest_timestamp = pd.Timestamp(
        df.iloc[-1]["timestamp"]
    )

    if latest_timestamp.tzinfo is None:
        latest_timestamp = (
            latest_timestamp.tz_localize("UTC")
        )

    latest_market_time = (
        latest_timestamp.tz_convert(
            MARKET_TIMEZONE
        )
    )

    latest_session_date = (
        latest_market_time.date()
    )

    current_market_date = (
        current.date()
    )

    market_close = pd.Timestamp.combine(
        current_market_date,
        REGULAR_MARKET_CLOSE,
    ).tz_localize(
        MARKET_TIMEZONE
    )

    if (
        len(df) >= 2
        and latest_session_date
        == current_market_date
        and current < market_close
    ):
        return (
            df
            .iloc[:-1]
            .reset_index(drop=True)
        )

    return df


# ============================================================
# SNAPSHOT BUILDER
# ============================================================

def build_snapshot(
    bars: list[dict[str, Any]],
) -> dict[str, float | int | str | None]:
    """
    Build the complete V1 quantitative snapshot for one ticker.

    This function intentionally performs deterministic calculations
    only. Candidate rules, scoring, AI analysis, fundamentals, and
    trade decisions do not belong here.
    """

    df = bars_to_dataframe(bars)

    df = remove_incomplete_session(df)

    if df.empty:
        raise ValueError(
            "No completed market sessions are available."
        )

    latest = df.iloc[-1]

    close = float(latest["close"])
    volume = float(latest["volume"])

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    sma_10 = sma(
        df["close"],
        10,
    )

    sma_20 = sma(
        df["close"],
        20,
    )

    sma_50 = sma(
        df["close"],
        50,
    )

    sma_200 = sma(
        df["close"],
        200,
    )

    # --------------------------------------------------------
    # VOLUME BASELINE
    # --------------------------------------------------------

    avg_volume_20d = average_volume(
        df["volume"],
        period=20,
        exclude_current=True,
    )

    # --------------------------------------------------------
    # BUILD SNAPSHOT
    # --------------------------------------------------------

    snapshot = {
        # DATA TIMESTAMP
        "as_of": (
            latest["timestamp"].isoformat()
        ),

        # PRICE
        "price": close,
        "open": float(latest["open"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),

        # VOLUME
        "volume": int(volume),
        "dollar_volume": close * volume,

        # RETURNS
        "return_1d": return_over_sessions(
            df["close"],
            1,
        ),
        "return_2d": return_over_sessions(
            df["close"],
            2,
        ),
        "return_3d": return_over_sessions(
            df["close"],
            3,
        ),
        "return_5d": return_over_sessions(
            df["close"],
            5,
        ),
        "return_10d": return_over_sessions(
            df["close"],
            10,
        ),
        "return_20d": return_over_sessions(
            df["close"],
            20,
        ),

        # VOLUME ANALYSIS
        "avg_volume_20d": avg_volume_20d,
        "relative_volume": relative_volume(
            current_volume=volume,
            average_volume_value=(
                avg_volume_20d
            ),
        ),

        # MOMENTUM
        "rsi_14": rsi(
            df["close"],
            period=14,
        ),

        # MOVING AVERAGES
        "sma_10": sma_10,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_200": sma_200,

        # PRICE EXTENSION
        "distance_sma_10_pct": (
            distance_from_level_pct(
                price=close,
                level=sma_10,
            )
        ),
        "distance_sma_20_pct": (
            distance_from_level_pct(
                price=close,
                level=sma_20,
            )
        ),
        "distance_sma_50_pct": (
            distance_from_level_pct(
                price=close,
                level=sma_50,
            )
        ),
        "distance_sma_200_pct": (
            distance_from_level_pct(
                price=close,
                level=sma_200,
            )
        ),

        # VOLATILITY
        "atr_14": atr(
            df,
            period=14,
        ),

        "atr_expansion": atr_expansion(
            df,
            atr_period=14,
            baseline_period=20,
        ),
    }

    return snapshot