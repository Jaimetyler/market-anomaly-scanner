from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)


@dataclass(frozen=True)
class HorizonExcursion:
    horizon: int
    mfe_pct: float | None
    mae_pct: float | None


def _normalize_horizons(
    horizons: Iterable[int],
) -> tuple[int, ...]:
    normalized = tuple(
        sorted(
            {
                int(horizon)
                for horizon in horizons
            }
        )
    )

    if not normalized:
        raise ValueError(
            "At least one horizon is required."
        )

    if normalized[0] <= 0:
        raise ValueError(
            "Horizons must be positive integers."
        )

    return normalized


def _require_price_columns(
    bars: pd.DataFrame,
) -> None:
    required = {
        "high",
        "low",
    }

    missing = required - set(bars.columns)

    if missing:
        raise ValueError(
            "Bars are missing required columns: "
            + ", ".join(sorted(missing))
        )


def contrarian_short_path_excursion(
    bars: pd.DataFrame,
    *,
    entry_price: float,
) -> tuple[float, float]:
    """
    Measure best/worst intraperiod excursion for a SHORT trade.

    Returns percentage points:

        MFE +25.0
            Price traded 25% in the short's favor.

        MAE -30.0
            Price traded 30% against the short.

    MAE is deliberately NEGATIVE because that is the convention used by
    scanner.replay_risk and the replay reporting layer.

    The supplied bars must represent only sessions occurring after the
    anomaly signal. For executable research, the first bar should be the
    next trading session, where entry occurs at that session's open.
    """
    if not np.isfinite(entry_price) or entry_price <= 0:
        raise ValueError(
            "entry_price must be a finite value greater than zero."
        )

    if bars.empty:
        return np.nan, np.nan

    _require_price_columns(bars)

    highs = pd.to_numeric(
        bars["high"],
        errors="coerce",
    ).dropna()

    lows = pd.to_numeric(
        bars["low"],
        errors="coerce",
    ).dropna()

    if highs.empty or lows.empty:
        return np.nan, np.nan

    lowest_price = float(lows.min())
    highest_price = float(highs.max())

    # Short trade:
    # falling price = favorable
    # rising price  = adverse
    mfe_pct = max(
        0.0,
        (
            (entry_price - lowest_price)
            / entry_price
        )
        * 100.0,
    )

    mae_pct = min(
        0.0,
        (
            (entry_price - highest_price)
            / entry_price
        )
        * 100.0,
    )

    return float(mfe_pct), float(mae_pct)


def calculate_horizon_excursions(
    future_bars: pd.DataFrame,
    *,
    entry_price: float,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> dict[str, float]:
    """
    Calculate executable short-side MFE/MAE for each requested horizon.

    future_bars must begin with the execution session immediately after the
    anomaly signal. The entry itself is assumed to occur at that session's
    open, using entry_price supplied by the master event's next_open field.

    A 5D excursion therefore uses the high/low path from the first executable
    session through the fifth executable session.

    Missing horizons remain NaN rather than being fabricated from a shorter
    path.
    """
    normalized_horizons = _normalize_horizons(
        horizons
    )

    if not np.isfinite(entry_price) or entry_price <= 0:
        raise ValueError(
            "entry_price must be a finite value greater than zero."
        )

    _require_price_columns(future_bars)

    bars = future_bars.reset_index(
        drop=True
    ).copy()

    output: dict[str, float] = {}

    for horizon in normalized_horizons:
        suffix = f"{horizon}d"

        if len(bars) < horizon:
            output[f"contrarian_mfe_{suffix}"] = np.nan
            output[f"contrarian_mae_{suffix}"] = np.nan
            continue

        path = bars.iloc[:horizon]

        mfe_pct, mae_pct = (
            contrarian_short_path_excursion(
                path,
                entry_price=entry_price,
            )
        )

        output[f"contrarian_mfe_{suffix}"] = (
            mfe_pct
        )
        output[f"contrarian_mae_{suffix}"] = (
            mae_pct
        )

    return output
