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

    # System 2 uses a 55-day breakout.
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

    # Force the opposite 20-day channel exit.
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
    """
    Original Turtle rule:

    With no previous 20-day breakout result, the first valid System 1
    breakout is permitted.
    """
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
    """
    Original Turtle System 1 rule:

    If the previous hypothetical 20-day breakout was a loser,
    the next 20-day breakout is taken.
    """
    bars = flat_bars(70)

    # First 20-day breakout.
    bars[20] = {
        "session_date": "first-breakout",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
    }

    # Flat warmup gives N ~= 2.
    # Entry ~= 101, so 2N stop ~= 97.
    # Force both the real trade and the hypothetical breakout to lose.
    bars[21] = {
        "session_date": "first-breakout-loses",
        "o": 100.0,
        "h": 100.5,
        "l": 96.0,
        "c": 97.0,
    }

    # Wait until the original breakout high has rolled out of
    # the 20-day entry channel.
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
    """
    Canonical System 1 behavior:

    1. First 20-day breakout is taken.
    2. Its hypothetical "Always Trader" result is a winner.
    3. The next 20-day breakout is skipped.
    4. Direction of that next breakout does not matter.
    5. A later 55-day breakout is taken as the failsafe.

    The first breakout here is LONG.
    The skipped breakout and failsafe are SHORT.

    That specifically verifies that previous-breakout direction
    does not control the winner/loser skip state.
    """
    bars = flat_bars(90)

    # Give the 55-day downside channel an older low that is lower
    # than the ordinary flat-bar 20-day channel.
    #
    # This lets us produce:
    #     short 20D breakout at 98 -> skipped
    #     short 55D breakout at 94 -> failsafe
    bars[0] = {
        "session_date": "old-low",
        "o": 100.0,
        "h": 101.0,
        "l": 95.0,
        "c": 100.0,
    }

    # First System 1 breakout: LONG.
    bars[20] = {
        "session_date": "winning-long-entry",
        "o": 100.0,
        "h": 102.0,
        "l": 100.0,
        "c": 101.5,
    }

    # Move price favorably and keep it above the stop.
    #
    # This can also pyramid the real trade, which is fine.
    for i in range(21, 32):
        bars[i] = {
            "session_date": f"winning-trend-{i}",
            "o": 103.0,
            "h": 104.5,
            "l": 102.5,
            "c": 103.5,
        }

    # The previous ten bars now have lows around 102.5.
    # Touch the 10-day channel at a price above the original
    # hypothetical entry (~101), making that hypothetical
    # 20-day breakout a winner.
    bars[32] = {
        "session_date": "winning-channel-exit",
        "o": 103.0,
        "h": 103.2,
        "l": 102.4,
        "c": 102.6,
    }

    # By index 53, the recent 20-day downside channel is near 99,
    # but the 55-day channel still contains the older low of 95.
    #
    # This is therefore a SHORT 20-day breakout but NOT a
    # 55-day breakout. Because the previous hypothetical breakout
    # was a winner, System 1 must skip it.
    bars[53] = {
        "session_date": "skipped-short-20d",
        "o": 100.0,
        "h": 100.0,
        "l": 98.0,
        "c": 98.5,
    }

    # Keep the skipped hypothetical short unresolved for one bar.
    bars[54] = {
        "session_date": "hold-before-failsafe",
        "o": 99.5,
        "h": 100.0,
        "l": 98.5,
        "c": 99.5,
    }

    # Break below the older 55-day low.
    #
    # This is the System 1 failsafe entry.
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

    # Initial long was traded.
    assert 20 in entry_indices

    # Winning previous breakout forces this 20D short breakout
    # to be skipped.
    assert 53 not in entry_indices

    # But the 55D failsafe must be taken.
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
    """
    Flat warmup produces N ~= 2.

    System 2 long breakout entry ~= 101.

    Therefore theoretical add levels are:

        Unit 1: 101
        Unit 2: 102
        Unit 3: 103
        Unit 4: 104

    A bar reaching 104.2 should fill all three adds but never
    create a fifth unit.
    """
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

    # Final stop after the fourth unit should be near 100.
    # Force it to resolve so the completed trade is returned.
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

    # N ~= 2.
    #
    # Newest fill ~= 104.
    # Canonical cascading stop:
    #
    #     104 - 2N
    #     104 - 4
    #     100
    assert trade.final_stop_price == pytest.approx(
        100.0,
        abs=1e-6,
    )


def test_pyramiding_stop_moves_half_n_with_each_new_unit():
    """
    Canonical Turtle cascading-stop progression for a long trade
    with N = 2:

        Initial entry 101 -> stop 97
        Add at 102       -> stop 98
        Add at 103       -> stop 99
        Add at 104       -> stop 100

    This is mathematically equivalent to keeping the active stop
    2N behind the newest actual unit fill.
    """
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
    """
    This is an important original-Turtle execution rule.

    Add levels are based on the ACTUAL previous fill, not merely
    the theoretical level that triggered the order.

    With N = 2:

        Initial entry ~= 101
        First add trigger = 102

    Then suppose the next session gaps to 104.5.

    The first add should fill at the actual open of 104.5.

    Therefore the NEXT add level must be:

        104.5 + 0.5N
        104.5 + 1.0
        105.5

    If the bar high is only 105.0, there must be exactly TWO
    total units.

    An implementation incorrectly anchored to the theoretical
    102 add level could cascade extra fills on this bar.
    """
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

    # After the gap add:
    #
    # actual second-unit fill = 104.5
    # stop = 104.5 - 2N = 100.5
    #
    # Resolve the completed trade next bar.
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
    """
    Daily OHLC cannot establish intraday ordering.

    Our explicit daily-equity adaptation is conservative:

        if active stop and add/exit are both touched,
        STOP wins.

    The trade is also marked ambiguous for later audit.
    """
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