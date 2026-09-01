from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)


@dataclass(frozen=True)
class BacktestConfig:
    """
    Configuration for event-outcome measurement.

    event_direction convention:
        +1 = upside anomaly / spike; contrarian trade is SHORT
        -1 = downside anomaly / selloff; contrarian trade is LONG
    """

    ticker_col: str = "ticker"
    date_col: str = "date"
    close_col: str = "close"
    high_col: str = "high"
    low_col: str = "low"
    direction_col: str = "event_direction"
    event_id_col: str = "event_id"
    horizons: tuple[int, ...] = DEFAULT_HORIZONS


def _normalize_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    cleaned = tuple(sorted({int(h) for h in horizons}))
    if not cleaned:
        raise ValueError("At least one horizon is required.")
    if cleaned[0] <= 0:
        raise ValueError("All horizons must be positive trading-day counts.")
    return cleaned


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _normalize_prices(prices: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    required = [
        config.ticker_col,
        config.date_col,
        config.close_col,
        config.high_col,
        config.low_col,
    ]
    _require_columns(prices, required, "prices")

    result = prices.copy()

    result[config.ticker_col] = (
        result[config.ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )
    result[config.date_col] = pd.to_datetime(
        result[config.date_col],
        errors="coerce",
    ).dt.normalize()

    for column in (config.close_col, config.high_col, config.low_col):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(
        subset=[
            config.ticker_col,
            config.date_col,
            config.close_col,
            config.high_col,
            config.low_col,
        ]
    )

    result = result[result[config.close_col] > 0].copy()

    duplicated = result.duplicated(
        subset=[config.ticker_col, config.date_col],
        keep=False,
    )
    if duplicated.any():
        sample = (
            result.loc[duplicated, [config.ticker_col, config.date_col]]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "prices contains duplicate ticker/date rows. "
            f"Examples: {sample}"
        )

    result = result.sort_values(
        [config.ticker_col, config.date_col],
        kind="stable",
    ).reset_index(drop=True)

    result["_trading_index"] = result.groupby(
        config.ticker_col,
        sort=False,
    ).cumcount()

    return result


def _normalize_events(events: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    required = [
        config.ticker_col,
        config.date_col,
        config.direction_col,
    ]
    _require_columns(events, required, "events")

    result = events.copy()

    result[config.ticker_col] = (
        result[config.ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )
    result[config.date_col] = pd.to_datetime(
        result[config.date_col],
        errors="coerce",
    ).dt.normalize()

    result[config.direction_col] = pd.to_numeric(
        result[config.direction_col],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            config.ticker_col,
            config.date_col,
            config.direction_col,
        ]
    )

    invalid_direction = ~result[config.direction_col].isin([-1, 1])
    if invalid_direction.any():
        bad = (
            result.loc[invalid_direction, config.direction_col]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"{config.direction_col} must contain only +1 or -1. "
            f"Found: {bad}"
        )

    result[config.direction_col] = result[config.direction_col].astype(int)

    if config.event_id_col not in result.columns:
        result[config.event_id_col] = (
            result[config.ticker_col]
            + "|"
            + result[config.date_col].dt.strftime("%Y-%m-%d")
            + "|"
            + result.groupby(
                [config.ticker_col, config.date_col],
                sort=False,
            ).cumcount().astype(str)
        )

    return result.reset_index(drop=True)


def _contrarian_path_metrics(
    *,
    direction: int,
    entry_close: float,
    highs: np.ndarray,
    lows: np.ndarray,
) -> tuple[float, float]:
    """
    Return (MFE, MAE) for the CONTRARIAN trade.

    Values are decimal returns:
        +0.10 = +10%
        -0.05 = -5%

    For an upside anomaly (+1), the contrarian trade is short.
    For a downside anomaly (-1), the contrarian trade is long.
    """
    if len(highs) == 0 or len(lows) == 0:
        return np.nan, np.nan

    if direction == 1:
        # Short after an upside anomaly.
        best = (entry_close - float(np.nanmin(lows))) / entry_close
        worst = (entry_close - float(np.nanmax(highs))) / entry_close
    else:
        # Long after a downside anomaly.
        best = (float(np.nanmax(highs)) - entry_close) / entry_close
        worst = (float(np.nanmin(lows)) - entry_close) / entry_close

    return best, worst


def compute_event_outcomes(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """
    Attach forward trading-day outcomes to anomaly events.

    This function is deliberately read-only with respect to the supplied
    DataFrames. It does not write files, mutate the mining cache, or make
    network requests.

    Output metrics for each horizon H:
        close_Hd
        return_Hd
            Underlying stock return from event close to H trading days later.

        contrarian_return_Hd
            Return of the simple contrarian position:
              upside anomaly (+1) -> short
              downside anomaly (-1) -> long

        reversed_Hd
            True when the stock's H-day return is opposite the anomaly direction.

        contrarian_mfe_Hd
            Best intraperiod return available to the contrarian position.

        contrarian_mae_Hd
            Worst intraperiod return suffered by the contrarian position.

        outcome_available_Hd
            False near the end of the price history when H future sessions
            do not exist.

    Event-day close is used as the theoretical entry reference. This does NOT
    model slippage, borrow availability, commissions, gaps after the close,
    or execution timing. Those belong in a later execution layer.
    """
    config = config or BacktestConfig()
    horizons = _normalize_horizons(config.horizons)

    px = _normalize_prices(prices, config)
    ev = _normalize_events(events, config)

    price_lookup = px.set_index(
        [config.ticker_col, config.date_col],
        drop=False,
    )

    ticker_groups = {
        ticker: group.reset_index(drop=True)
        for ticker, group in px.groupby(config.ticker_col, sort=False)
    }

    rows: list[dict] = []

    for _, event in ev.iterrows():
        ticker = event[config.ticker_col]
        event_date = event[config.date_col]
        direction = int(event[config.direction_col])

        output = event.to_dict()
        output["event_matched"] = False
        output["event_close"] = np.nan

        key = (ticker, event_date)

        if key not in price_lookup.index:
            output["event_match_error"] = "no_exact_ticker_date_price_row"
            for horizon in horizons:
                suffix = f"{horizon}d"
                output[f"outcome_available_{suffix}"] = False
                output[f"close_{suffix}"] = np.nan
                output[f"return_{suffix}"] = np.nan
                output[f"contrarian_return_{suffix}"] = np.nan
                output[f"reversed_{suffix}"] = pd.NA
                output[f"contrarian_mfe_{suffix}"] = np.nan
                output[f"contrarian_mae_{suffix}"] = np.nan
            rows.append(output)
            continue

        event_price_row = price_lookup.loc[key]
        if isinstance(event_price_row, pd.DataFrame):
            raise RuntimeError(
                "Unexpected duplicate ticker/date price rows after normalization."
            )

        entry_close = float(event_price_row[config.close_col])
        event_index = int(event_price_row["_trading_index"])

        output["event_matched"] = True
        output["event_match_error"] = None
        output["event_close"] = entry_close

        ticker_prices = ticker_groups[ticker]

        for horizon in horizons:
            suffix = f"{horizon}d"
            target_index = event_index + horizon

            if target_index >= len(ticker_prices):
                output[f"outcome_available_{suffix}"] = False
                output[f"close_{suffix}"] = np.nan
                output[f"return_{suffix}"] = np.nan
                output[f"contrarian_return_{suffix}"] = np.nan
                output[f"reversed_{suffix}"] = pd.NA
                output[f"contrarian_mfe_{suffix}"] = np.nan
                output[f"contrarian_mae_{suffix}"] = np.nan
                continue

            target = ticker_prices.iloc[target_index]
            target_close = float(target[config.close_col])
            raw_return = (target_close / entry_close) - 1.0

            # Contrarian side is opposite the event direction.
            contrarian_return = -direction * raw_return
            reversed_flag = bool(raw_return * direction < 0)

            path = ticker_prices.iloc[
                event_index + 1 : target_index + 1
            ]

            mfe, mae = _contrarian_path_metrics(
                direction=direction,
                entry_close=entry_close,
                highs=path[config.high_col].to_numpy(dtype=float),
                lows=path[config.low_col].to_numpy(dtype=float),
            )

            output[f"outcome_available_{suffix}"] = True
            output[f"close_{suffix}"] = target_close
            output[f"return_{suffix}"] = raw_return
            output[f"contrarian_return_{suffix}"] = contrarian_return
            output[f"reversed_{suffix}"] = reversed_flag
            output[f"contrarian_mfe_{suffix}"] = mfe
            output[f"contrarian_mae_{suffix}"] = mae

        rows.append(output)

    result = pd.DataFrame(rows)

    # Keep a stable order for reproducible research output.
    preferred = [
        config.event_id_col,
        config.ticker_col,
        config.date_col,
        config.direction_col,
        "event_matched",
        "event_match_error",
        "event_close",
    ]
    remaining = [column for column in result.columns if column not in preferred]
    ordered = [column for column in preferred if column in result.columns] + remaining

    return result.loc[:, ordered]


def summarize_outcomes(
    outcomes: pd.DataFrame,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    group_by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Produce a compact research summary from compute_event_outcomes() output.

    For each requested horizon, reports:
        n
        reversal_rate
        contrarian_win_rate
        mean_contrarian_return
        median_contrarian_return
        mean_mfe
        mean_mae

    group_by can be used later for setup type, market-cap bucket, RVOL bucket,
    sector, regime, spike-size bucket, etc.
    """
    horizons = _normalize_horizons(horizons)
    group_by = list(group_by or [])

    _require_columns(outcomes, group_by, "outcomes")

    summary_rows: list[dict] = []

    if group_by:
        grouper = group_by[0] if len(group_by) == 1 else group_by
        grouped = outcomes.groupby(grouper, dropna=False, sort=True)
    else:
        grouped = [((), outcomes)]

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
            _require_columns(frame, required, "outcomes")

            usable = frame[
                frame[f"outcome_available_{suffix}"].fillna(False)
            ].copy()

            returns = pd.to_numeric(
                usable[f"contrarian_return_{suffix}"],
                errors="coerce",
            )
            reversals = usable[f"reversed_{suffix}"].astype("boolean")
            mfe = pd.to_numeric(
                usable[f"contrarian_mfe_{suffix}"],
                errors="coerce",
            )
            mae = pd.to_numeric(
                usable[f"contrarian_mae_{suffix}"],
                errors="coerce",
            )

            row = {
                **group_values,
                "horizon_days": horizon,
                "n": int(returns.notna().sum()),
                "reversal_rate": float(reversals.mean())
                if reversals.notna().any()
                else np.nan,
                "contrarian_win_rate": float((returns > 0).mean())
                if returns.notna().any()
                else np.nan,
                "mean_contrarian_return": float(returns.mean())
                if returns.notna().any()
                else np.nan,
                "median_contrarian_return": float(returns.median())
                if returns.notna().any()
                else np.nan,
                "mean_mfe": float(mfe.mean())
                if mfe.notna().any()
                else np.nan,
                "mean_mae": float(mae.mean())
                if mae.notna().any()
                else np.nan,
            }
            summary_rows.append(row)

    return pd.DataFrame(summary_rows)