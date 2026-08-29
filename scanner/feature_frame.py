from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scanner.indicators import (
    atr_series,
    bars_to_dataframe,
    remove_incomplete_session,
    wilder_smoothed_series,
)


RETURN_SESSIONS = (1, 2, 3, 5, 10, 20)
SMA_PERIODS = (10, 20, 50, 200)
RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_BASELINE_PERIOD = 20
VOLUME_BASELINE_PERIOD = 20


def _safe_scalar(value: Any) -> float | int | str | None:
    """Convert pandas/numpy scalars into snapshot-friendly Python values."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, np.generic):
        return value.item()

    return value


def _percent_change_series(
    current: pd.Series,
    previous: pd.Series,
) -> pd.Series:
    """Vectorized equivalent of indicators.percent_change()."""

    result = ((current / previous) - 1.0) * 100.0
    return result.where(previous != 0)


def _distance_series(
    price: pd.Series,
    level: pd.Series,
) -> pd.Series:
    """Vectorized equivalent of distance_from_level_pct()."""

    result = ((price / level) - 1.0) * 100.0
    return result.where(level != 0)


def _rsi_series_exact_wilder(
    closes: pd.Series,
    period: int = RSI_PERIOD,
) -> pd.Series:
    """
    Produce the same RSI values as scanner.indicators.rsi(), but once
    across the entire ticker history instead of recomputing every prefix.
    """

    output = pd.Series(
        np.nan,
        index=closes.index,
        dtype="float64",
    )

    if len(closes) < period + 1:
        return output

    numeric = pd.to_numeric(
        closes,
        errors="coerce",
    )

    if numeric.isna().any():
        return output

    delta = numeric.diff().iloc[1:]
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = wilder_smoothed_series(
        gains,
        period=period,
    )
    avg_loss = wilder_smoothed_series(
        losses,
        period=period,
    )

    valid = avg_gain.notna() & avg_loss.notna()

    flat = valid & (avg_gain == 0) & (avg_loss == 0)
    no_losses = valid & (avg_loss == 0) & (avg_gain > 0)
    no_gains = valid & (avg_gain == 0) & (avg_loss > 0)
    normal = valid & (avg_gain > 0) & (avg_loss > 0)

    rsi_values = pd.Series(
        np.nan,
        index=avg_gain.index,
        dtype="float64",
    )

    rsi_values.loc[flat] = 50.0
    rsi_values.loc[no_losses] = 100.0
    rsi_values.loc[no_gains] = 0.0

    rs = (
        avg_gain.loc[normal]
        / avg_loss.loc[normal]
    )

    rsi_values.loc[normal] = (
        100.0
        - (100.0 / (1.0 + rs))
    )

    output.loc[rsi_values.index] = rsi_values
    return output


def build_feature_frame(
    bars: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Build every deterministic V1 scanner feature once for the full
    ticker history.

    This is intentionally equivalent to repeatedly calling
    build_snapshot() on each historical prefix, but avoids rebuilding
    rolling calculations thousands of times.
    """

    df = bars_to_dataframe(
        bars
    )

    df = remove_incomplete_session(
        df
    )

    if df.empty:
        raise ValueError(
            "No completed market sessions are available."
        )

    frame = df.copy()

    frame["session_date"] = (
        frame["timestamp"]
        .dt.date
        .astype(str)
    )

    close = frame["close"]
    volume = frame["volume"]

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------
    for sessions in RETURN_SESSIONS:
        frame[f"return_{sessions}d"] = (
            _percent_change_series(
                current=close,
                previous=close.shift(sessions),
            )
        )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------
    frame["avg_volume_20d"] = (
        volume
        .shift(1)
        .rolling(
            window=VOLUME_BASELINE_PERIOD,
            min_periods=VOLUME_BASELINE_PERIOD,
        )
        .mean()
    )

    frame["relative_volume"] = (
        volume
        / frame["avg_volume_20d"]
    ).where(
        frame["avg_volume_20d"] != 0
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------
    frame["rsi_14"] = (
        _rsi_series_exact_wilder(
            close,
            period=RSI_PERIOD,
        )
    )

    # --------------------------------------------------------
    # MOVING AVERAGES + EXTENSION
    # --------------------------------------------------------
    for period in SMA_PERIODS:
        sma_column = f"sma_{period}"
        distance_column = (
            f"distance_sma_{period}_pct"
        )

        frame[sma_column] = (
            close
            .rolling(
                window=period,
                min_periods=period,
            )
            .mean()
        )

        frame[distance_column] = (
            _distance_series(
                price=close,
                level=frame[sma_column],
            )
        )

    # --------------------------------------------------------
    # ATR + ATR EXPANSION
    # --------------------------------------------------------
    frame["atr_14"] = atr_series(
        frame,
        period=ATR_PERIOD,
    )

    atr_baseline = (
        frame["atr_14"]
        .shift(1)
        .rolling(
            window=ATR_BASELINE_PERIOD,
            min_periods=ATR_BASELINE_PERIOD,
        )
        .mean()
    )

    frame["atr_expansion"] = (
        frame["atr_14"]
        / atr_baseline
    ).where(
        atr_baseline != 0
    )

    return frame


def snapshot_from_feature_row(
    row: pd.Series,
) -> dict[str, float | int | str | None]:
    """
    Convert one precomputed feature-frame row into the exact snapshot
    shape expected by detect_anomaly() and classify_setup().
    """

    close = float(row["close"])
    volume = float(row["volume"])

    return {
        "as_of": row["timestamp"].isoformat(),
        "price": close,
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "volume": int(volume),
        "dollar_volume": close * volume,
        "return_1d": _safe_scalar(row["return_1d"]),
        "return_2d": _safe_scalar(row["return_2d"]),
        "return_3d": _safe_scalar(row["return_3d"]),
        "return_5d": _safe_scalar(row["return_5d"]),
        "return_10d": _safe_scalar(row["return_10d"]),
        "return_20d": _safe_scalar(row["return_20d"]),
        "avg_volume_20d": _safe_scalar(row["avg_volume_20d"]),
        "relative_volume": _safe_scalar(row["relative_volume"]),
        "rsi_14": _safe_scalar(row["rsi_14"]),
        "sma_10": _safe_scalar(row["sma_10"]),
        "sma_20": _safe_scalar(row["sma_20"]),
        "sma_50": _safe_scalar(row["sma_50"]),
        "sma_200": _safe_scalar(row["sma_200"]),
        "distance_sma_10_pct": _safe_scalar(
            row["distance_sma_10_pct"]
        ),
        "distance_sma_20_pct": _safe_scalar(
            row["distance_sma_20_pct"]
        ),
        "distance_sma_50_pct": _safe_scalar(
            row["distance_sma_50_pct"]
        ),
        "distance_sma_200_pct": _safe_scalar(
            row["distance_sma_200_pct"]
        ),
        "atr_14": _safe_scalar(row["atr_14"]),
        "atr_expansion": _safe_scalar(row["atr_expansion"]),
    }


def snapshots_by_session(
    bars: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | str | None]]:
    """Precompute snapshot dictionaries keyed by ISO session date."""

    frame = build_feature_frame(
        bars
    )

    return {
        str(row["session_date"]): (
            snapshot_from_feature_row(
                row
            )
        )
        for _, row in frame.iterrows()
    }