from __future__ import annotations

from dataclasses import asdict

import pytest

from scanner.turtle import SYSTEM_1, SYSTEM_2
from scanner.turtle_simulation import simulate_turtle_system
from scanner.turtle_stateful import TurtleMarketState


def _bar(date, open_, high, low, close):
    return {
        "session_date": date,
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
    }


def _build_regression_bars():
    """
    Deterministic synthetic history with:
    - enough warmup for 55D
    - long and short breakouts
    - pyramids
    - stops/channel exits
    - repeated later opportunities

    The purpose is not realism. It is to force both engines through the same
    canonical mechanics and require their completed trades to match exactly.
    """
    bars = []

    # 60-day flat-ish warmup.
    for i in range(60):
        base = 100.0 + ((i % 5) - 2) * 0.2
        bars.append(
            _bar(
                f"2024-01-{i + 1:02d}",
                base,
                base + 1.0,
                base - 1.0,
                base,
            )
        )

    # Long breakout and pyramid.
    bars.extend(
        [
            _bar("2024-03-01", 101.0, 103.5, 100.5, 103.0),
            _bar("2024-03-02", 103.0, 106.5, 102.5, 106.0),
            _bar("2024-03-03", 106.0, 109.0, 105.5, 108.0),
            _bar("2024-03-04", 108.0, 109.0, 106.0, 107.0),
            _bar("2024-03-05", 107.0, 108.0, 104.0, 105.0),
            _bar("2024-03-06", 105.0, 106.0, 101.0, 102.0),
            _bar("2024-03-07", 102.0, 103.0, 98.0, 99.0),
        ]
    )

    # Trend lower enough to create short breakouts.
    price = 99.0
    for i in range(35):
        price -= 0.9
        bars.append(
            _bar(
                f"2024-04-{i + 1:02d}",
                price + 0.3,
                price + 1.0,
                price - 1.2,
                price,
            )
        )

    # Short breakout continuation and possible adds.
    bars.extend(
        [
            _bar("2024-05-10", 66.0, 67.0, 63.0, 64.0),
            _bar("2024-05-11", 64.0, 65.0, 60.5, 61.0),
            _bar("2024-05-12", 61.0, 62.0, 57.5, 58.0),
            _bar("2024-05-13", 58.0, 60.0, 57.0, 59.0),
            _bar("2024-05-14", 59.0, 63.0, 58.0, 62.0),
            _bar("2024-05-15", 62.0, 67.0, 61.0, 66.0),
        ]
    )

    # Recovery/uptrend for another cycle.
    price = 66.0
    for i in range(50):
        price += 1.0
        bars.append(
            _bar(
                f"2024-06-{i + 1:02d}",
                price - 0.3,
                price + 1.2,
                price - 1.0,
                price,
            )
        )

    return bars


def _run_stateful_accept_everything(bars, system):
    state = TurtleMarketState(
        bars,
        system=system,
    )

    for index in range(len(bars)):
        actions = state.propose(index)

        if not actions:
            continue

        action = actions[0]

        if action.action_type == "ENTRY":
            state.accept(action)
            continue

        if action.action_type == "EXIT":
            state.accept(action)
            continue

        if action.action_type == "ADD":
            # Match standalone behavior: after accepting one 0.5N add, the
            # same daily bar may have crossed another add level.
            current = action

            while current is not None:
                state.accept(current)
                current = state.propose_current_add(index)

    state.close_end_of_data()

    return state.completed_trades


def _assert_trade_equal(expected, actual):
    assert actual.system == expected.system
    assert actual.side == expected.side

    assert actual.entry_index == expected.entry_index
    assert actual.exit_index == expected.exit_index
    assert actual.entry_date == expected.entry_date
    assert actual.exit_date == expected.exit_date

    assert actual.initial_entry_price == pytest.approx(
        expected.initial_entry_price
    )
    assert actual.average_entry_price == pytest.approx(
        expected.average_entry_price
    )
    assert actual.exit_price == pytest.approx(
        expected.exit_price
    )
    assert actual.initial_n == pytest.approx(
        expected.initial_n
    )
   

    assert actual.exit_reason == expected.exit_reason
    assert actual.ambiguous_bars == expected.ambiguous_bars
    assert actual.units == expected.units

    assert len(actual.unit_entries) == len(expected.unit_entries)

    for expected_unit, actual_unit in zip(
        expected.unit_entries,
        actual.unit_entries,
        strict=True,
    ):
        assert actual_unit.entry_index == expected_unit.entry_index
        assert actual_unit.entry_date == expected_unit.entry_date
        assert actual_unit.entry_price == pytest.approx(
            expected_unit.entry_price
        )
        assert actual_unit.n == pytest.approx(
            expected_unit.n
        )


@pytest.mark.parametrize(
    "system",
    [
        SYSTEM_1,
        SYSTEM_2,
    ],
)
def test_stateful_accept_everything_matches_standalone(system):
    bars = _build_regression_bars()

    expected = simulate_turtle_system(
        bars,
        system=system,
    )
    actual = _run_stateful_accept_everything(
        bars,
        system,
    )

    assert len(actual) == len(expected)

    for expected_trade, actual_trade in zip(
        expected,
        actual,
        strict=True,
    ):
        _assert_trade_equal(
            expected_trade,
            actual_trade,
        )
