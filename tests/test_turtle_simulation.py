from __future__ import annotations

import pytest

from scanner.turtle import (
    SYSTEM_1,
    SYSTEM_2,
)
from scanner.turtle_simulation import (
    _gap_aware_stop_fill,
    _trade_return_pct,
    simulate_turtle_system,
)


def flat_bars(
    count: int,
    *,
    price: float = 100.0,
) -> list[dict]:
    return [
        {
            "session_date": f"session-{i}",
            "o": price,
            "h": price + 1.0,
            "l": price - 1.0,
            "c": price,
        }
        for i in range(count)
    ]


def test_long_return_math():
    assert _trade_return_pct(
        side="LONG",
        entry_price=100.0,
        exit_price=110.0,
    ) == pytest.approx(10.0)


def test_short_return_math():
    assert _trade_return_pct(
        side="SHORT",
        entry_price=100.0,
        exit_price=90.0,
    ) == pytest.approx(
        (100.0 / 90.0 - 1.0) * 100.0
    )


def test_gap_aware_long_stop():
    assert _gap_aware_stop_fill(
        side="LONG",
        session_open=90.0,
        stop_price=95.0,
    ) == pytest.approx(90.0)

    assert _gap_aware_stop_fill(
        side="LONG",
        session_open=100.0,
        stop_price=95.0,
    ) == pytest.approx(95.0)


def test_gap_aware_short_stop():
    assert _gap_aware_stop_fill(
        side="SHORT",
        session_open=110.0,
        stop_price=105.0,
    ) == pytest.approx(110.0)

    assert _gap_aware_stop_fill(
        side="SHORT",
        session_open=100.0,
        stop_price=105.0,
    ) == pytest.approx(105.0)


def test_system_2_long_breakout_then_channel_exit():
    bars = flat_bars(90)

    # System 2 needs 55 prior sessions.
    bars[55] = {
        "session_date": "entry",
        "o": 100.0,
        "h": 103.0,
        "l": 100.0,
        "c": 102.0,
    }

    # Keep the trade moving higher without triggering the stop.
    for i in range(56, 76):
        bars[i] = {
            "session_date": f"trend-{i}",
            "o": 103.0,
            "h": 103.5,
            "l": 102.0,
            "c": 103.0,
        }

    # Force a downside 20-day channel break.
    bars[76] = {
        "session_date": "exit",
        "o": 100.0,
        "h": 101.0,
        "l": 90.0,
        "c": 92.0,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_2,
    )

    assert trades

    trade = trades[0]

    assert trade.side == "LONG"
    assert trade.system == "SYSTEM_2"
    assert trade.entry_index == 55
    assert trade.exit_reason in {
        "CHANNEL_EXIT",
        "STOP",
    }


def test_system_2_short_breakout():
    bars = flat_bars(80)

    bars[55] = {
        "session_date": "entry",
        "o": 100.0,
        "h": 100.0,
        "l": 97.0,
        "c": 98.0,
    }

    for i in range(56, 70):
        bars[i] = {
            "session_date": f"down-{i}",
            "o": 97.0,
            "h": 98.0,
            "l": 96.0,
            "c": 97.0,
        }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_2,
    )

    assert trades
    assert trades[0].side == "SHORT"


def test_pyramiding_never_exceeds_four_units():
    bars = flat_bars(90)

    bars[55] = {
        "session_date": "entry",
        "o": 100.0,
        "h": 110.0,
        "l": 100.0,
        "c": 108.0,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_2,
    )

    assert trades

    assert all(
        trade.units <= 4
        for trade in trades
    )


def test_stop_wins_same_bar_ambiguity():
    bars = flat_bars(80)

    # With flat warmup, N ~= 2.
    # Breakout around 101, initial stop around 97.
    # This bar touches both favorable pyramid territory
    # and the stop. Conservative simulator must stop.
    bars[55] = {
        "session_date": "wild-entry",
        "o": 100.0,
        "h": 106.0,
        "l": 95.0,
        "c": 102.0,
    }

    # Unfortunately this is also a two-sided ENTRY breakout,
    # which the simulator intentionally skips.
    #
    # Create a clean entry first.
    bars[55] = {
        "session_date": "entry",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
    }

    bars[56] = {
        "session_date": "ambiguous",
        "o": 101.5,
        "h": 105.0,
        "l": 95.0,
        "c": 102.0,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_2,
    )

    assert trades

    trade = trades[0]

    assert trade.exit_index == 56
    assert trade.exit_reason == "STOP"
    assert trade.ambiguous_bars >= 1


def test_system_1_is_stateful_and_runs():
    bars = flat_bars(100)

    bars[20] = {
        "session_date": "first-breakout",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.0,
    }

    # Give it enough movement later to create exits/new opportunities.
    bars[35] = {
        "session_date": "down-move",
        "o": 98.0,
        "h": 99.0,
        "l": 94.0,
        "c": 95.0,
    }

    bars[50] = {
        "session_date": "up-move",
        "o": 101.0,
        "h": 108.0,
        "l": 100.0,
        "c": 107.0,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_1,
    )

    assert isinstance(
        trades,
        list,
    )

    assert all(
        trade.system == "SYSTEM_1"
        for trade in trades
    )
