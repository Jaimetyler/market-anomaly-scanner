from __future__ import annotations

# FLATFILE ADJUSTMENTS TESTS v2
# DESTINATION: tests/test_flatfile_adjustments.py

from dataclasses import dataclass

import pytest

from scanner.flatfile_adjustments import (
    adjust_flatfile_bar_for_splits,
    cumulative_price_adjustment_factor,
)
from scanner.flatfile_history import FlatFileBar


@dataclass(frozen=True)
class FakeSplit:
    execution_date: str
    split_from: float
    split_to: float
    historical_adjustment_factor: float | None = None


def _bar(
    session_date: str,
    close: float = 100.0,
    volume: float = 1_000.0,
) -> FlatFileBar:
    return FlatFileBar(
        ticker="TEST",
        session_date=session_date,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
        transactions=10,
        window_start_ns=1,
        vwap=close,
    )


def test_forward_split_uses_ratio_not_cumulative_factor():
    event = FakeSplit(
        execution_date="2024-06-10",
        split_from=1.0,
        split_to=10.0,
        historical_adjustment_factor=0.000001,
    )

    adjusted = adjust_flatfile_bar_for_splits(
        _bar("2024-06-07", close=100.0, volume=1_000.0),
        [event],
    )

    assert adjusted.close == pytest.approx(10.0)
    assert adjusted.volume == pytest.approx(10_000.0)


def test_reverse_split_uses_ratio_not_cumulative_factor():
    event = FakeSplit(
        execution_date="2025-03-17",
        split_from=250.0,
        split_to=1.0,
        historical_adjustment_factor=6_102_000.0,
    )

    adjusted = adjust_flatfile_bar_for_splits(
        _bar("2025-03-14", close=0.04, volume=250_000.0),
        [event],
    )

    assert adjusted.close == pytest.approx(10.0)
    assert adjusted.volume == pytest.approx(1_000.0)


def test_execution_date_is_unchanged():
    event = FakeSplit("2025-03-17", 250.0, 1.0, 6_102_000.0)
    original = _bar("2025-03-17", close=10.0, volume=1_000.0)
    assert adjust_flatfile_bar_for_splits(original, [event]) == original


def test_post_split_bar_is_unchanged():
    event = FakeSplit("2025-03-17", 250.0, 1.0, 6_102_000.0)
    original = _bar("2025-03-18", close=12.0, volume=2_000.0)
    assert adjust_flatfile_bar_for_splits(original, [event]) == original


def test_multiple_event_ratios_compound():
    events = [
        FakeSplit("2024-10-02", 40.0, 1.0, 244_080_000.0),
        FakeSplit("2025-03-17", 250.0, 1.0, 6_102_000.0),
    ]
    factor = cumulative_price_adjustment_factor(events, "2024-10-01")
    assert factor == pytest.approx(40.0 * 250.0)


def test_bar_between_two_splits_only_gets_later_split():
    events = [
        FakeSplit("2024-10-02", 40.0, 1.0, 244_080_000.0),
        FakeSplit("2025-03-17", 250.0, 1.0, 6_102_000.0),
    ]
    factor = cumulative_price_adjustment_factor(events, "2024-10-03")
    assert factor == pytest.approx(250.0)