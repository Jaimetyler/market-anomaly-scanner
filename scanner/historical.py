from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

from scanner.indicators import build_snapshot
from scanner.outcomes import (
    OutcomeResult,
    calculate_outcomes,
)


@dataclass
class HistoricalSlice:
    """
    Point-in-time view of one historical signal date.

    history_bars:
        Contains ONLY information available through
        the signal date.

    future_bars:
        Contains ONLY sessions after the signal date.

    signal_close:
        Closing price on the signal date.

    next_open:
        Opening price of the first future session,
        when available.
    """

    signal_date: date
    history_bars: list[dict[str, Any]]
    future_bars: list[dict[str, Any]]
    signal_close: float
    next_open: float | None


@dataclass
class HistoricalOutcomeBundle:
    """
    Two ways to evaluate the same historical signal.

    research:
        Measures future movement relative to the
        signal-day closing price.

    executable:
        Measures future movement relative to the
        next-session opening price.

    This helps separate:

        "What happened after we detected it?"

    from:

        "What could a trader plausibly have captured?"
    """

    signal_date: date
    signal_close: float
    next_open: float | None

    research: OutcomeResult
    executable: OutcomeResult | None


def _normalize_date(
    value: str | date | datetime,
) -> date:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    return date.fromisoformat(
        value
    )


def bar_session_date(
    bar: dict[str, Any],
) -> date:
    """
    Convert a Massive daily aggregate timestamp
    into its calendar session date.

    Massive aggregate timestamps are Unix
    milliseconds.
    """

    timestamp = bar.get("t")

    if timestamp is None:
        raise ValueError(
            "Historical bar is missing timestamp 't'."
        )

    parsed = pd.to_datetime(
        timestamp,
        unit="ms",
        utc=True,
    )

    return parsed.date()


def sort_bars(
    bars: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Return bars sorted chronologically.
    """

    return sorted(
        bars,
        key=lambda bar: bar["t"],
    )


def create_historical_slice(
    bars: list[dict[str, Any]],
    signal_date: str | date | datetime,
) -> HistoricalSlice:
    """
    Create a strict point-in-time split.

    The history side may contain bars dated
    <= signal_date.

    The future side may contain bars dated
    > signal_date.

    A bar for the exact signal date MUST exist.

    This function is the core information wall
    used by the historical research engine.
    """

    signal_date = _normalize_date(
        signal_date
    )

    if not bars:
        raise ValueError(
            "No historical bars were supplied."
        )

    ordered = sort_bars(
        bars
    )

    history_bars = []
    future_bars = []

    signal_bar = None

    for bar in ordered:
        session_date = bar_session_date(
            bar
        )

        if session_date <= signal_date:
            history_bars.append(
                bar
            )
        else:
            future_bars.append(
                bar
            )

        if session_date == signal_date:
            signal_bar = bar

    if signal_bar is None:
        raise ValueError(
            "No bar exists for signal date "
            f"{signal_date.isoformat()}."
        )

    signal_close = signal_bar.get(
        "c"
    )

    if signal_close is None:
        raise ValueError(
            "Signal bar is missing close price 'c'."
        )

    signal_close = float(
        signal_close
    )

    if signal_close <= 0:
        raise ValueError(
            "Signal close must be greater than zero."
        )

    next_open = None

    if future_bars:
        open_price = future_bars[
            0
        ].get("o")

        if open_price is not None:
            next_open = float(
                open_price
            )

            if next_open <= 0:
                next_open = None

    return HistoricalSlice(
        signal_date=signal_date,
        history_bars=history_bars,
        future_bars=future_bars,
        signal_close=signal_close,
        next_open=next_open,
    )


def build_snapshot_as_of(
    bars: list[dict[str, Any]],
    signal_date: str | date | datetime,
) -> dict[str, Any]:
    """
    Build scanner indicators exactly as they
    would have appeared on signal_date.

    Future bars are physically removed BEFORE
    build_snapshot() is called.

    This is our protection against look-ahead bias.
    """

    historical_slice = (
        create_historical_slice(
            bars=bars,
            signal_date=signal_date,
        )
    )

    return build_snapshot(
        historical_slice.history_bars
    )


def bars_to_outcome_dataframe(
    bars: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert Massive daily bars into the format
    expected by calculate_outcomes().
    """

    rows = []

    for bar in sort_bars(
        bars
    ):
        rows.append(
            {
                "session_date": (
                    bar_session_date(
                        bar
                    )
                ),
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "session_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    df = pd.DataFrame(
        rows
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return (
        df
        .sort_values(
            "session_date"
        )
        .reset_index(
            drop=True
        )
    )


def calculate_historical_outcomes(
    bars: list[dict[str, Any]],
    signal_date: str | date | datetime,
) -> HistoricalOutcomeBundle:
    """
    Grade one historical signal in two ways.

    RESEARCH OUTCOME
    ----------------
    Entry reference:
        signal-day close

    Purpose:
        Measure what happened after detection.

    EXECUTABLE OUTCOME
    ------------------
    Entry reference:
        next-session open

    Purpose:
        Estimate the path available to someone
        who only learned about the signal after
        the signal session had completed.

    Both use ONLY bars after signal_date for
    future outcome calculations.
    """

    historical_slice = (
        create_historical_slice(
            bars=bars,
            signal_date=signal_date,
        )
    )

    future_df = (
        bars_to_outcome_dataframe(
            historical_slice.future_bars
        )
    )

    research = calculate_outcomes(
        future_bars=future_df,
        signal_price=(
            historical_slice.signal_close
        ),
    )

    executable = None

    if (
        historical_slice.next_open
        is not None
    ):
        executable = calculate_outcomes(
            future_bars=future_df,
            signal_price=(
                historical_slice.next_open
            ),
        )

    return HistoricalOutcomeBundle(
        signal_date=(
            historical_slice.signal_date
        ),
        signal_close=(
            historical_slice.signal_close
        ),
        next_open=(
            historical_slice.next_open
        ),
        research=research,
        executable=executable,
    )