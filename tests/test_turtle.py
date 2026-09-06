from __future__ import annotations

import pytest

from scanner.turtle import (
    SYSTEM_1,
    SYSTEM_2,
    initial_stop_price,
    previous_channel,
    pyramid_levels,
    theoretical_breakout_fill,
    turtle_breakout_signals_at_index,
    wilder_n,
)


def make_bar(
    day: int,
    *,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
) -> dict:
    return {
        "session_date": f"2025-01-{day:02d}",
        "o": open_,
        "h": high,
        "l": low,
        "c": close,
    }


def make_flat_bars(
    count: int,
) -> list[dict]:
    bars = []

    for i in range(count):
        bars.append(
            {
                "session_date": f"session-{i}",
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "c": 100.0,
            }
        )

    return bars


def test_previous_channel_excludes_current_bar():
    bars = make_flat_bars(21)

    bars[20] = {
        "session_date": "breakout-day",
        "o": 100.0,
        "h": 500.0,
        "l": 1.0,
        "c": 250.0,
    }

    high, low = previous_channel(
        bars,
        index=20,
        lookback=20,
    )

    assert high == pytest.approx(101.0)
    assert low == pytest.approx(99.0)


def test_wilder_n_flat_range():
    bars = make_flat_bars(30)

    values = wilder_n(
        bars,
        period=20,
    )

    assert values[18] is None
    assert values[19] == pytest.approx(2.0)
    assert values[20] == pytest.approx(2.0)
    assert values[29] == pytest.approx(2.0)


def test_system_1_long_breakout():
    bars = make_flat_bars(25)

    bars[20] = {
        "session_date": "breakout-day",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
    }

    n_values = wilder_n(
        bars,
        period=20,
    )

    signals = turtle_breakout_signals_at_index(
        bars,
        index=20,
        system=SYSTEM_1,
        n_values=n_values,
    )

    assert len(signals) == 1

    signal = signals[0]

    assert signal.system == "SYSTEM_1"
    assert signal.side == "LONG"
    assert signal.breakout_level == pytest.approx(101.0)

    # No opening gap. Theoretical fill occurs at breakout.
    assert signal.entry_price == pytest.approx(101.0)

    # Prior-day N is 2.
    assert signal.n == pytest.approx(2.0)

    # 2N initial stop.
    assert signal.stop_price == pytest.approx(97.0)

    assert signal.requires_skip_rule_evaluation is True

    assert signal.pyramid_levels == pytest.approx(
        (
            102.0,
            103.0,
            104.0,
        )
    )


def test_long_gap_through_breakout_enters_at_open():
    bars = make_flat_bars(25)

    bars[20] = {
        "session_date": "gap-day",
        "o": 105.0,
        "h": 107.0,
        "l": 104.0,
        "c": 106.0,
    }

    signals = turtle_breakout_signals_at_index(
        bars,
        index=20,
        system=SYSTEM_1,
    )

    assert len(signals) == 1

    signal = signals[0]

    assert signal.side == "LONG"
    assert signal.breakout_level == pytest.approx(101.0)
    assert signal.entry_price == pytest.approx(105.0)


def test_system_2_short_breakout():
    bars = make_flat_bars(60)

    bars[55] = {
        "session_date": "short-breakout-day",
        "o": 100.0,
        "h": 100.0,
        "l": 98.0,
        "c": 98.5,
    }

    signals = turtle_breakout_signals_at_index(
        bars,
        index=55,
        system=SYSTEM_2,
    )

    assert len(signals) == 1

    signal = signals[0]

    assert signal.system == "SYSTEM_2"
    assert signal.side == "SHORT"
    assert signal.breakout_level == pytest.approx(99.0)
    assert signal.entry_price == pytest.approx(99.0)

    # N = 2, so short stop is 2N above entry.
    assert signal.stop_price == pytest.approx(103.0)

    assert signal.requires_skip_rule_evaluation is False

    assert signal.pyramid_levels == pytest.approx(
        (
            98.0,
            97.0,
            96.0,
        )
    )


def test_short_gap_through_breakout_enters_at_open():
    bars = make_flat_bars(60)

    bars[55] = {
        "session_date": "gap-down-day",
        "o": 95.0,
        "h": 97.0,
        "l": 94.0,
        "c": 95.0,
    }

    signals = turtle_breakout_signals_at_index(
        bars,
        index=55,
        system=SYSTEM_2,
    )

    assert len(signals) == 1

    signal = signals[0]

    assert signal.side == "SHORT"
    assert signal.breakout_level == pytest.approx(99.0)
    assert signal.entry_price == pytest.approx(95.0)


def test_same_bar_two_sided_breakout_is_marked_ambiguous():
    bars = make_flat_bars(25)

    bars[20] = {
        "session_date": "wild-day",
        "o": 100.0,
        "h": 103.0,
        "l": 97.0,
        "c": 100.0,
    }

    signals = turtle_breakout_signals_at_index(
        bars,
        index=20,
        system=SYSTEM_1,
    )

    assert len(signals) == 2

    assert {
        signal.side
        for signal in signals
    } == {
        "LONG",
        "SHORT",
    }

    assert all(
        signal.ambiguous_same_bar
        for signal in signals
    )


def test_stop_math():
    assert initial_stop_price(
        side="LONG",
        entry_price=100.0,
        n=4.0,
    ) == pytest.approx(92.0)

    assert initial_stop_price(
        side="SHORT",
        entry_price=100.0,
        n=4.0,
    ) == pytest.approx(108.0)


def test_pyramid_spacing():
    assert pyramid_levels(
        side="LONG",
        entry_price=100.0,
        n=4.0,
    ) == pytest.approx(
        (
            102.0,
            104.0,
            106.0,
        )
    )

    assert pyramid_levels(
        side="SHORT",
        entry_price=100.0,
        n=4.0,
    ) == pytest.approx(
        (
            98.0,
            96.0,
            94.0,
        )
    )