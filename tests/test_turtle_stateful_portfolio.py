from __future__ import annotations

import pytest

from scanner.turtle import SYSTEM_2
from scanner.turtle_portfolio import TurtleChronologicalConfig
from scanner.turtle_stateful import TurtleMarketState
from scanner.turtle_stateful_portfolio import (
    TurtleStatefulSegment,
    run_stateful_turtle_portfolio,
)


def _bar(date, open_, high, low, close):
    return {
        "session_date": date,
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
    }


def _base_bars():
    bars = []
    for i in range(60):
        bars.append(
            _bar(
                f"2024-01-{i + 1:02d}",
                100, 101, 99, 100,
            )
        )
    return bars


def test_rejected_add_does_not_move_stop_and_can_retry():
    bars = _base_bars()
    bars += [
        _bar("2024-03-01", 101, 103, 100, 102),
        _bar("2024-03-02", 102, 104, 101, 103),
        _bar("2024-03-03", 102, 104, 101, 103),
    ]

    state = TurtleMarketState(bars, system=SYSTEM_2)
    state.accept(state.propose(60)[0])

    original_stop = state.stop_price
    add = state.propose(61)[0]
    state.reject(add)

    assert state.units == 1
    assert state.stop_price == pytest.approx(original_stop)

    retry = state.propose(62)[0]
    assert retry.action_type == "ADD"
    assert retry.unit_number == 2
    assert retry.price == pytest.approx(add.price)


def test_stateful_portfolio_rejected_add_does_not_change_trade_path():
    # Ticker A occupies the only portfolio unit first.
    a = _base_bars()
    a += [
        _bar("2024-03-01", 101, 103, 100, 102),
        _bar("2024-03-02", 102, 104, 101, 103),
        _bar("2024-03-03", 103, 103, 100, 101),
        _bar("2024-03-04", 101, 101, 96, 97),
    ]

    result = run_stateful_turtle_portfolio(
        [
            TurtleStatefulSegment(
                ticker="A",
                segment_number=1,
                bars=a,
            )
        ],
        system=SYSTEM_2,
        config=TurtleChronologicalConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
            max_units_per_trade=4,
            max_total_units=1,
            max_direction_units=1,
        ),
        mark_prices_by_ticker={
            "A": {
                bar["session_date"]: bar["close"]
                for bar in a
            }
        },
    )

    assert any(
        skip.reason == "ADD_TOTAL_UNIT_LIMIT"
        for skip in result.skipped_trades
    )
    assert result.accepted_trades
    trade = result.accepted_trades[0]
    assert trade.units == 1


def test_stateful_portfolio_allows_multiple_same_bar_adds_when_capacity_exists():
    bars = _base_bars()
    # With N ~2, breakout around 101 and a high of 106 crosses multiple 0.5N adds.
    bars += [
        _bar("2024-03-01", 101, 103, 100, 102),
        _bar("2024-03-02", 102, 108, 101, 107),
        _bar("2024-03-03", 107, 108, 106, 107),
    ]

    result = run_stateful_turtle_portfolio(
        [
            TurtleStatefulSegment(
                ticker="A",
                segment_number=1,
                bars=bars,
            )
        ],
        system=SYSTEM_2,
        config=TurtleChronologicalConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
            max_units_per_trade=4,
            max_total_units=12,
            max_direction_units=12,
        ),
        mark_prices_by_ticker={
            "A": {
                bar["session_date"]: bar["close"]
                for bar in bars
            }
        },
    )

    assert result.accepted_trades
    assert result.accepted_trades[0].units == 4
