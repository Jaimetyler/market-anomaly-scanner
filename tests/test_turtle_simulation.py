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

    bars[55] = {
        "session_date": "entry",
        "o": 100.0,
        "h": 103.0,
        "l": 100.0,
        "c": 102.0,
    }

    for i in range(56, 76):
        bars[i] = {
            "session_date": f"trend-{i}",
            "o": 103.0,
            "h": 103.5,
            "l": 102.0,
            "c": 103.0,
        }

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


# ---------------------------------------------------------------------------
# Canonical System 1 breakout-state tests
# ---------------------------------------------------------------------------


def test_system_1_first_valid_20_day_breakout_is_taken():
    bars = flat_bars(50)

    bars[20] = {
        "session_date": "first-breakout",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_1,
    )

    assert trades
    assert trades[0].system == "SYSTEM_1"
    assert trades[0].side == "LONG"
    assert trades[0].entry_index == 20


def test_system_1_losing_breakout_allows_next_20_day_breakout():
    bars = flat_bars(70)

    bars[20] = {
        "session_date": "first-breakout",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
    }

    bars[21] = {
        "session_date": "first-breakout-loses",
        "o": 100.0,
        "h": 100.5,
        "l": 96.0,
        "c": 97.0,
    }

    bars[42] = {
        "session_date": "second-breakout",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_1,
    )

    entry_indices = [
        trade.entry_index
        for trade in trades
    ]

    assert 20 in entry_indices
    assert 42 in entry_indices


def test_system_1_winning_breakout_skips_next_20_day_breakout_and_uses_55_day_failsafe():
    bars = flat_bars(90)

    bars[0] = {
        "session_date": "old-low",
        "o": 100.0,
        "h": 101.0,
        "l": 95.0,
        "c": 100.0,
    }

    bars[20] = {
        "session_date": "winning-long-entry",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
    }

    for i in range(21, 32):
        bars[i] = {
            "session_date": f"winning-trend-{i}",
            "o": 103.0,
            "h": 104.5,
            "l": 102.5,
            "c": 103.5,
        }

    bars[32] = {
        "session_date": "winning-channel-exit",
        "o": 103.0,
        "h": 103.2,
        "l": 102.4,
        "c": 102.6,
    }

    bars[53] = {
        "session_date": "skipped-short-20d",
        "o": 100.0,
        "h": 100.0,
        "l": 98.0,
        "c": 98.5,
    }

    bars[54] = {
        "session_date": "hold-before-failsafe",
        "o": 99.5,
        "h": 100.0,
        "l": 98.5,
        "c": 99.5,
    }

    bars[55] = {
        "session_date": "short-55d-failsafe",
        "o": 99.0,
        "h": 100.0,
        "l": 94.0,
        "c": 94.5,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_1,
    )

    entry_indices = [
        trade.entry_index
        for trade in trades
    ]

    assert 20 in entry_indices
    assert 53 not in entry_indices
    assert 55 in entry_indices

    failsafe_trade = next(
        trade
        for trade in trades
        if trade.entry_index == 55
    )

    assert failsafe_trade.side == "SHORT"
    assert failsafe_trade.system == "SYSTEM_1"


# ---------------------------------------------------------------------------
# Canonical pyramiding / stop mechanics
# ---------------------------------------------------------------------------


def test_pyramiding_adds_every_half_n_and_stops_at_four_units():
    bars = flat_bars(80)

    bars[55] = {
        "session_date": "entry",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
    }

    bars[56] = {
        "session_date": "three-adds",
        "o": 101.5,
        "h": 104.2,
        "l": 101.2,
        "c": 104.0,
    }

    bars[57] = {
        "session_date": "stop-after-pyramid",
        "o": 100.0,
        "h": 101.0,
        "l": 99.0,
        "c": 100.0,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_2,
    )

    assert trades

    trade = trades[0]

    assert trade.entry_index == 55
    assert trade.units == 4
    assert trade.final_stop_price == pytest.approx(
        100.0,
        abs=1e-6,
    )


def test_pyramiding_stop_moves_half_n_with_each_new_unit():
    bars = flat_bars(80)

    bars[55] = {
        "session_date": "entry",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
    }

    bars[56] = {
        "session_date": "pyramid",
        "o": 101.5,
        "h": 104.2,
        "l": 101.2,
        "c": 104.0,
    }

    bars[57] = {
        "session_date": "resolve",
        "o": 100.0,
        "h": 101.0,
        "l": 99.0,
        "c": 100.0,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_2,
    )

    assert trades

    trade = trades[0]

    assert trade.units == 4
    assert trade.initial_entry_price == pytest.approx(
        101.0,
        abs=1e-6,
    )
    assert trade.initial_n == pytest.approx(
        2.0,
        abs=1e-6,
    )
    assert trade.initial_stop_price == pytest.approx(
        97.0,
        abs=1e-6,
    )
    assert trade.final_stop_price == pytest.approx(
        100.0,
        abs=1e-6,
    )


def test_gap_add_uses_actual_fill_as_origin_for_next_half_n_add():
    bars = flat_bars(80)

    bars[55] = {
        "session_date": "entry",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
    }

    bars[56] = {
        "session_date": "gap-add",
        "o": 104.5,
        "h": 105.0,
        "l": 104.0,
        "c": 104.8,
    }

    bars[57] = {
        "session_date": "resolve",
        "o": 100.0,
        "h": 101.0,
        "l": 99.0,
        "c": 100.0,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_2,
    )

    assert trades

    trade = trades[0]

    assert trade.units == 2
    assert trade.final_stop_price == pytest.approx(
        100.5,
        abs=1e-6,
    )


def test_stop_wins_same_bar_ambiguity():
    bars = flat_bars(80)

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


def test_all_completed_trades_respect_four_unit_limit():
    bars = flat_bars(90)

    bars[55] = {
        "session_date": "entry",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
    }

    bars[56] = {
        "session_date": "huge-trend-bar",
        "o": 101.5,
        "h": 120.0,
        "l": 101.0,
        "c": 118.0,
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


# ---------------------------------------------------------------------------
# Gap / daily-OHLC execution semantics
# ---------------------------------------------------------------------------


def test_long_entry_gap_through_breakout_fills_at_open():
    bars = flat_bars(80)

    bars[55] = {
        "session_date": "gap-entry",
        "o": 105.0,
        "h": 106.0,
        "l": 104.0,
        "c": 105.5,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_2,
    )

    assert trades

    trade = trades[0]

    assert trade.entry_index == 55
    assert trade.side == "LONG"
    assert trade.initial_entry_price == pytest.approx(
        105.0,
    )


def test_short_entry_gap_through_breakout_fills_at_open():
    bars = flat_bars(80)

    bars[55] = {
        "session_date": "gap-entry",
        "o": 95.0,
        "h": 96.0,
        "l": 94.0,
        "c": 94.5,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_2,
    )

    assert trades

    trade = trades[0]

    assert trade.entry_index == 55
    assert trade.side == "SHORT"
    assert trade.initial_entry_price == pytest.approx(
        95.0,
    )


def test_long_stop_gap_through_stop_fills_at_open():
    bars = flat_bars(80)

    bars[55] = {
        "session_date": "entry",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
    }

    bars[56] = {
        "session_date": "gap-stop",
        "o": 94.0,
        "h": 95.0,
        "l": 93.0,
        "c": 94.0,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_2,
    )

    assert trades

    trade = trades[0]

    assert trade.exit_reason == "STOP"
    assert trade.exit_index == 56
    assert trade.exit_price == pytest.approx(
        94.0,
    )


def test_short_stop_gap_through_stop_fills_at_open():
    bars = flat_bars(80)

    bars[55] = {
        "session_date": "entry",
        "o": 100.0,
        "h": 100.0,
        "l": 98.0,
        "c": 98.5,
    }

    bars[56] = {
        "session_date": "gap-stop",
        "o": 106.0,
        "h": 107.0,
        "l": 105.0,
        "c": 106.0,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_2,
    )

    assert trades

    trade = trades[0]

    assert trade.exit_reason == "STOP"
    assert trade.exit_index == 56
    assert trade.exit_price == pytest.approx(
        106.0,
    )


def test_long_channel_exit_gap_fills_at_open():
    bars = flat_bars(100)

    # Flat warmup:
    # previous 55-day high ~= 101
    # N ~= 2
    #
    # Enter around 101.
    bars[55] = {
        "session_date": "entry",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
    }

    # Keep price above the old flat-bar lows so the 20-day
    # exit channel eventually rises to ~100.5.
    #
    # IMPORTANT:
    # Keep highs below the first add level (~102), so the
    # position stays at ONE unit and the stop remains ~97.
    for i in range(56, 76):
        bars[i] = {
            "session_date": f"hold-{i}",
            "o": 101.5,
            "h": 101.8,
            "l": 100.5,
            "c": 101.5,
        }

    # Previous 20-day low is now ~100.5.
    #
    # Gap below that channel exit level, but remain safely
    # ABOVE the hard stop (~97).
    bars[76] = {
        "session_date": "gap-channel-exit",
        "o": 99.5,
        "h": 100.0,
        "l": 99.0,
        "c": 99.5,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_2,
    )

    assert trades

    trade = trades[0]

    assert trade.units == 1
    assert trade.exit_reason == "CHANNEL_EXIT"
    assert trade.exit_index == 76

    # Gap through channel -> fill at actual open,
    # not at the better historical channel level.
    assert trade.exit_price == pytest.approx(
        99.5,
    )


def test_two_sided_entry_bar_is_skipped_as_daily_bar_ambiguity():
    bars = flat_bars(80)

    bars[55] = {
        "session_date": "two-sided-breakout",
        "o": 100.0,
        "h": 105.0,
        "l": 95.0,
        "c": 100.0,
    }

    trades = simulate_turtle_system(
        bars,
        system=SYSTEM_2,
    )

    assert not any(
        trade.entry_index == 55
        for trade in trades
    )