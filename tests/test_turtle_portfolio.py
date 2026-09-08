from __future__ import annotations

from types import SimpleNamespace

import pytest

from scanner.turtle_portfolio import (
    TurtleChronologicalConfig,
    TurtleDatedTrade,
    run_chronological_turtle_portfolio,
    unit_shares,
)


def _trade(
    *,
    system: str = "SYSTEM_2",
    side: str = "LONG",
    entry_index: int = 10,
    exit_index: int = 20,
    entry_date: str = "2024-01-02",
    exit_date: str = "2024-01-10",
    average_entry_price: float = 100.0,
    exit_price: float = 110.0,
    units: int = 1,
    initial_n: float = 2.0,
    exit_reason: str = "CHANNEL_EXIT",
):
    return SimpleNamespace(
        system=system,
        side=side,
        entry_index=entry_index,
        exit_index=exit_index,
        entry_date=entry_date,
        exit_date=exit_date,
        average_entry_price=average_entry_price,
        exit_price=exit_price,
        units=units,
        initial_n=initial_n,
        exit_reason=exit_reason,
    )


def test_unit_shares_uses_equity_risk_divided_by_n():
    assert unit_shares(
        equity=5000.0,
        n=2.0,
        risk_fraction=0.01,
    ) == 25


def test_overlapping_positions_share_one_chronological_account():
    a = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            entry_date="2024-01-02",
            exit_date="2024-01-10",
            average_entry_price=100.0,
            exit_price=110.0,
            units=2,
        ),
    )

    b = TurtleDatedTrade(
        ticker="BBB",
        trade=_trade(
            entry_index=11,
            exit_index=21,
            entry_date="2024-01-03",
            exit_date="2024-01-11",
            average_entry_price=50.0,
            exit_price=55.0,
            units=2,
        ),
    )

    result = run_chronological_turtle_portfolio(
        [b, a],
        config=TurtleChronologicalConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
            max_total_units=12,
            max_direction_units=12,
        ),
    )

    assert len(result.accepted_trades) == 2
    assert not result.skipped_trades
    assert result.max_concurrent_positions == 2
    assert result.max_concurrent_units == 4

    # Both entries size from the same realized $5,000 because neither trade
    # has exited yet: 1% of 5000 / N=2 = 25 shares per unit.
    assert all(
        trade.shares_per_unit == 25
        for trade in result.accepted_trades
    )

    # AAA: 50 shares * $10 = $500
    # BBB: 50 shares * $5 = $250
    assert result.ending_equity == pytest.approx(
        5750.0
    )


def test_total_unit_limit_rejects_overlapping_candidate():
    a = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            units=4,
            entry_date="2024-01-02",
            exit_date="2024-01-10",
        ),
    )

    b = TurtleDatedTrade(
        ticker="BBB",
        trade=_trade(
            entry_index=11,
            exit_index=21,
            units=4,
            entry_date="2024-01-03",
            exit_date="2024-01-11",
        ),
    )

    result = run_chronological_turtle_portfolio(
        [a, b],
        config=TurtleChronologicalConfig(
            max_total_units=4,
            max_direction_units=12,
        ),
    )

    assert len(result.accepted_trades) == 1
    assert len(result.skipped_trades) == 1
    assert result.skipped_trades[0].ticker == "BBB"
    assert (
        result.skipped_trades[0].reason
        == "TOTAL_UNIT_LIMIT"
    )


def test_direction_unit_limit_rejects_same_side_exposure():
    long_a = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            side="LONG",
            units=4,
            entry_date="2024-01-02",
            exit_date="2024-01-10",
        ),
    )

    long_b = TurtleDatedTrade(
        ticker="BBB",
        trade=_trade(
            side="LONG",
            entry_index=11,
            exit_index=21,
            units=4,
            entry_date="2024-01-03",
            exit_date="2024-01-11",
        ),
    )

    result = run_chronological_turtle_portfolio(
        [long_a, long_b],
        config=TurtleChronologicalConfig(
            max_total_units=12,
            max_direction_units=4,
        ),
    )

    assert len(result.accepted_trades) == 1
    assert len(result.skipped_trades) == 1
    assert (
        result.skipped_trades[0].reason
        == "DIRECTION_UNIT_LIMIT"
    )


def test_opposite_directions_can_coexist_under_direction_cap():
    long_trade = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            side="LONG",
            units=4,
            entry_date="2024-01-02",
            exit_date="2024-01-10",
        ),
    )

    short_trade = TurtleDatedTrade(
        ticker="BBB",
        trade=_trade(
            side="SHORT",
            entry_index=11,
            exit_index=21,
            units=4,
            entry_date="2024-01-03",
            exit_date="2024-01-11",
            average_entry_price=100.0,
            exit_price=90.0,
        ),
    )

    result = run_chronological_turtle_portfolio(
        [long_trade, short_trade],
        config=TurtleChronologicalConfig(
            max_total_units=8,
            max_direction_units=4,
        ),
    )

    assert len(result.accepted_trades) == 2
    assert not result.skipped_trades
    assert result.max_concurrent_units == 8


def test_same_day_exit_frees_units_before_new_entry():
    first = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            units=4,
            entry_date="2024-01-02",
            exit_date="2024-01-10",
        ),
    )

    replacement = TurtleDatedTrade(
        ticker="BBB",
        trade=_trade(
            entry_index=11,
            exit_index=21,
            units=4,
            entry_date="2024-01-10",
            exit_date="2024-01-20",
        ),
    )

    result = run_chronological_turtle_portfolio(
        [first, replacement],
        config=TurtleChronologicalConfig(
            max_total_units=4,
            max_direction_units=4,
        ),
    )

    assert len(result.accepted_trades) == 2
    assert not result.skipped_trades

    events_on_transition_day = [
        (
            point.event_type,
            point.ticker,
        )
        for point in result.equity_curve
        if point.event_date == "2024-01-10"
    ]

    assert events_on_transition_day == [
        ("EXIT", "AAA"),
        ("ENTRY", "BBB"),
    ]


def test_later_entry_sizes_from_realized_equity_after_prior_exit():
    first = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            entry_date="2024-01-02",
            exit_date="2024-01-03",
            average_entry_price=100.0,
            exit_price=120.0,
            units=1,
            initial_n=2.0,
        ),
    )

    second = TurtleDatedTrade(
        ticker="BBB",
        trade=_trade(
            entry_index=11,
            exit_index=21,
            entry_date="2024-01-04",
            exit_date="2024-01-05",
            average_entry_price=100.0,
            exit_price=100.0,
            units=1,
            initial_n=2.0,
        ),
    )

    result = run_chronological_turtle_portfolio(
        [first, second],
        config=TurtleChronologicalConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
        ),
    )

    accepted_by_ticker = {
        trade.ticker: trade
        for trade in result.accepted_trades
    }

    # First trade: 25 shares * $20 = +$500 -> realized equity $5,500.
    assert (
        accepted_by_ticker["AAA"].realized_equity_after_exit
        == pytest.approx(5500.0)
    )

    # Second unit size: 1% of $5,500 / $2 N = floor(27.5) = 27 shares.
    assert (
        accepted_by_ticker["BBB"].shares_per_unit
        == 27
    )


def test_final_trade_units_are_reserved_from_initial_entry():
    a = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            units=4,
            entry_date="2024-01-02",
            exit_date="2024-01-10",
        ),
    )

    result = run_chronological_turtle_portfolio(
        [a],
        config=TurtleChronologicalConfig(
            max_total_units=12,
            max_direction_units=12,
        ),
    )

    entry_point = next(
        point
        for point in result.equity_curve
        if point.event_type == "ENTRY"
    )

    # Until TurtleTrade exposes the date of each 0.5N add, the portfolio layer
    # intentionally reserves the final completed trade unit count at entry.
    assert entry_point.active_units == 4
