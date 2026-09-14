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
        bars.append(_bar(f"2024-01-{i + 1:02d}", 100, 101, 99, 100))
    return bars


def _marks(ticker, bars):
    return {
        ticker: {
            bar["session_date"]: bar["close"]
            for bar in bars
        }
    }


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
    a = _base_bars()
    a += [
        _bar("2024-03-01", 101, 103, 100, 102),
        _bar("2024-03-02", 102, 104, 101, 103),
        _bar("2024-03-03", 103, 103, 100, 101),
        _bar("2024-03-04", 101, 101, 96, 97),
    ]

    result = run_stateful_turtle_portfolio(
        [TurtleStatefulSegment(ticker="A", segment_number=1, bars=a)],
        system=SYSTEM_2,
        config=TurtleChronologicalConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
            max_units_per_trade=4,
            max_total_units=1,
            max_direction_units=1,
        ),
        mark_prices_by_ticker=_marks("A", a),
    )

    assert any(
        skip.reason == "ADD_TOTAL_UNIT_LIMIT"
        for skip in result.skipped_trades
    )
    assert result.accepted_trades
    assert result.accepted_trades[0].units == 1


def test_stateful_portfolio_allows_multiple_same_bar_adds_when_capacity_exists():
    bars = _base_bars()
    bars += [
        _bar("2024-03-01", 101, 103, 100, 102),
        _bar("2024-03-02", 102, 108, 101, 107),
        _bar("2024-03-03", 107, 108, 106, 107),
    ]

    result = run_stateful_turtle_portfolio(
        [TurtleStatefulSegment(ticker="A", segment_number=1, bars=bars)],
        system=SYSTEM_2,
        config=TurtleChronologicalConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
            max_units_per_trade=4,
            max_total_units=12,
            max_direction_units=12,
        ),
        mark_prices_by_ticker=_marks("A", bars),
    )

    assert result.accepted_trades
    assert result.accepted_trades[0].units == 4


def _same_day_competition(seed, ticker_order):
    bars_by_ticker = {
        ticker: _base_bars() + [_bar("2024-03-01", 101, 103, 100, 102)]
        for ticker in ticker_order
    }
    marks = {
        ticker: {
            bar["session_date"]: bar["close"]
            for bar in bars
        }
        for ticker, bars in bars_by_ticker.items()
    }
    return run_stateful_turtle_portfolio(
        [
            TurtleStatefulSegment(ticker=ticker, segment_number=1, bars=bars_by_ticker[ticker])
            for ticker in ticker_order
        ],
        system=SYSTEM_2,
        config=TurtleChronologicalConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
            max_units_per_trade=4,
            max_total_units=1,
            max_direction_units=1,
            entry_priority_seed=seed,
        ),
        mark_prices_by_ticker=marks,
    )


def test_same_day_priority_is_independent_of_input_and_ticker_order():
    forward = _same_day_competition(0, ["A", "Z"])
    reverse = _same_day_competition(0, ["Z", "A"])

    assert [trade.ticker for trade in forward.accepted_trades] == [
        trade.ticker for trade in reverse.accepted_trades
    ]
    assert forward.accepted_trades[0].entry_priority_rank == 1
    assert forward.accepted_trades[0].entry_priority_token


def test_priority_seed_can_change_capacity_winner_reproducibly():
    seed_zero = _same_day_competition(0, ["A", "Z"])
    seed_one = _same_day_competition(1, ["A", "Z"])

    assert seed_zero.accepted_trades[0].ticker != seed_one.accepted_trades[0].ticker
    assert seed_zero.accepted_trades[0].entry_priority_seed == 0
    assert seed_one.accepted_trades[0].entry_priority_seed == 1
    rejected = [
        skip for skip in seed_zero.skipped_trades
        if skip.reason == "TOTAL_UNIT_LIMIT"
    ]
    assert rejected
    assert rejected[0].priority_rank == 2
    assert rejected[0].priority_token
