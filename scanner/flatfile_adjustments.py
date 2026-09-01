from __future__ import annotations

# FLATFILE ADJUSTMENTS v2 — per-event split-ratio normalization
# DESTINATION: scanner/flatfile_adjustments.py

from dataclasses import replace
from datetime import date
from typing import Iterable

from scanner.corporate_actions import get_stock_splits
from scanner.flatfile_history import FlatFileBar


def _as_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _event_date(event) -> date | None:
    value = getattr(event, "execution_date", None)
    if value is None:
        value = getattr(event, "date", None)
    if value is None:
        return None
    return _as_date(value)


def _event_price_factor(event) -> float | None:
    """
    Per-event adjustment factor for historical prices.

    Massive split semantics:
      forward 10-for-1: split_from=1,   split_to=10  -> price factor 0.1
      reverse 250:1:   split_from=250, split_to=1   -> price factor 250

    Do NOT use historical_adjustment_factor here. That field can be cumulative
    across multiple later splits.
    """
    split_from = getattr(event, "split_from", None)
    split_to = getattr(event, "split_to", None)

    raw = getattr(event, "raw", None)
    if isinstance(raw, dict):
        if split_from is None:
            split_from = raw.get("split_from")
        if split_to is None:
            split_to = raw.get("split_to")

    try:
        split_from = float(split_from)
        split_to = float(split_to)
    except (TypeError, ValueError):
        return None

    if split_from <= 0 or split_to <= 0:
        return None

    return split_from / split_to


def applicable_split_events(events: Iterable, bar_date: str | date) -> list:
    """
    Only splits AFTER the bar apply. The execution-date bar is already on the
    post-split basis and must remain unchanged.
    """
    session = _as_date(bar_date)
    result = []

    for event in events:
        execution = _event_date(event)
        if execution is not None and execution > session:
            result.append(event)

    result.sort(key=lambda e: _event_date(e) or date.max)
    return result


def cumulative_price_adjustment_factor(
    events: Iterable,
    bar_date: str | date,
) -> float:
    factor = 1.0

    for event in applicable_split_events(events, bar_date):
        event_factor = _event_price_factor(event)
        if event_factor is not None:
            factor *= event_factor

    return factor


def adjust_flatfile_bar_for_splits(
    bar: FlatFileBar,
    events: Iterable,
) -> FlatFileBar:
    factor = cumulative_price_adjustment_factor(events, bar.session_date)

    if factor == 1.0:
        return bar

    return replace(
        bar,
        open=bar.open * factor,
        high=bar.high * factor,
        low=bar.low * factor,
        close=bar.close * factor,
        volume=bar.volume / factor,
        vwap=None if bar.vwap is None else bar.vwap * factor,
    )


def adjust_flatfile_history_for_splits(
    bars: Iterable[FlatFileBar],
    events: Iterable,
) -> list[FlatFileBar]:
    event_list = list(events)
    return [
        adjust_flatfile_bar_for_splits(bar, event_list)
        for bar in bars
    ]


def get_adjusted_flatfile_history(
    ticker: str,
    bars: Iterable[FlatFileBar],
) -> list[FlatFileBar]:
    return adjust_flatfile_history_for_splits(
        bars,
        get_stock_splits(ticker=ticker),
    )