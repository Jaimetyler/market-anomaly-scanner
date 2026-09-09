from __future__ import annotations

from types import SimpleNamespace

import pytest

from scanner.turtle_portfolio import (
    TurtleChronologicalConfig,
    TurtleDatedTrade,
    run_chronological_turtle_portfolio,
    unit_shares,
)
from scanner.turtle_simulation import TurtleUnit


def _unit(
    *,
    price: float,
    index: int,
    date: str,
    n: float = 2.0,
) -> TurtleUnit:
    return TurtleUnit(
        entry_price=price,
        n=n,
        entry_index=index,
        entry_date=date,
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
    unit_entries: tuple[TurtleUnit, ...] | None = None,
):
    if unit_entries is None:
        unit_entries = tuple(
            _unit(
                price=average_entry_price,
                index=entry_index + offset,
                date=entry_date,
                n=initial_n,
            )
            for offset in range(units)
        )

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
        unit_entries=unit_entries,
    )


def test_unit_shares_uses_equity_risk_divided_by_n():
    assert unit_shares(
        equity=5000.0,
        n=2.0,
        risk_fraction=0.01,
    ) == 25


def test_actual_pyramid_adds_consume_units_on_their_real_dates():
    trade = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            units=4,
            unit_entries=(
                _unit(
                    price=100.0,
                    index=10,
                    date="2024-01-02",
                ),
                _unit(
                    price=101.0,
                    index=11,
                    date="2024-01-03",
                ),
                _unit(
                    price=102.0,
                    index=12,
                    date="2024-01-04",
                ),
                _unit(
                    price=103.0,
                    index=13,
                    date="2024-01-05",
                ),
            ),
        ),
    )

    result = run_chronological_turtle_portfolio(
        [trade],
        config=TurtleChronologicalConfig(
            max_total_units=12,
            max_direction_units=12,
        ),
    )

    events = [
        (
            point.event_date,
            point.event_type,
            point.active_units,
        )
        for point in result.equity_curve
        if point.ticker == "AAA"
    ]

    assert events[:4] == [
        ("2024-01-02", "ENTRY", 1),
        ("2024-01-03", "ADD_2", 2),
        ("2024-01-04", "ADD_3", 3),
        ("2024-01-05", "ADD_4", 4),
    ]


def test_overlapping_positions_share_one_chronological_account():
    a = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            entry_date="2024-01-02",
            exit_date="2024-01-10",
            exit_price=110.0,
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
    assert result.max_concurrent_units == 2

    assert all(
        trade.shares_per_unit == 25
        for trade in result.accepted_trades
    )

    # AAA: 25 shares * $10 = $250
    # BBB: 25 shares * $5 = $125
    assert result.ending_equity == pytest.approx(
        5375.0
    )


def test_total_unit_limit_can_reject_an_add_without_rejecting_initial_trade():
    a = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            units=2,
            unit_entries=(
                _unit(
                    price=100.0,
                    index=10,
                    date="2024-01-02",
                ),
                _unit(
                    price=101.0,
                    index=12,
                    date="2024-01-04",
                ),
            ),
        ),
    )

    b = TurtleDatedTrade(
        ticker="BBB",
        trade=_trade(
            entry_index=11,
            exit_index=21,
            entry_date="2024-01-03",
            exit_date="2024-01-11",
            units=1,
        ),
    )

    result = run_chronological_turtle_portfolio(
        [a, b],
        config=TurtleChronologicalConfig(
            max_total_units=2,
            max_direction_units=12,
        ),
    )

    assert len(result.accepted_trades) == 2

    aaa = next(
        trade
        for trade in result.accepted_trades
        if trade.ticker == "AAA"
    )

    assert aaa.units == 1
    assert any(
        skip.ticker == "AAA"
        and skip.reason == "ADD_TOTAL_UNIT_LIMIT"
        for skip in result.skipped_trades
    )


def test_rejected_add_breaks_later_precomputed_add_chain():
    a = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            units=3,
            unit_entries=(
                _unit(
                    price=100.0,
                    index=10,
                    date="2024-01-02",
                ),
                _unit(
                    price=101.0,
                    index=12,
                    date="2024-01-04",
                ),
                _unit(
                    price=102.0,
                    index=13,
                    date="2024-01-05",
                ),
            ),
        ),
    )

    blocker = TurtleDatedTrade(
        ticker="BBB",
        trade=_trade(
            entry_index=11,
            exit_index=14,
            entry_date="2024-01-03",
            exit_date="2024-01-06",
            units=1,
        ),
    )

    result = run_chronological_turtle_portfolio(
        [a, blocker],
        config=TurtleChronologicalConfig(
            max_total_units=2,
            max_direction_units=12,
        ),
    )

    aaa = next(
        trade
        for trade in result.accepted_trades
        if trade.ticker == "AAA"
    )

    assert aaa.units == 1

    aaa_reasons = [
        skip.reason
        for skip in result.skipped_trades
        if skip.ticker == "AAA"
    ]

    assert "ADD_TOTAL_UNIT_LIMIT" in aaa_reasons
    assert "ADD_CHAIN_BROKEN" in aaa_reasons


def test_direction_unit_limit_rejects_same_side_initial_exposure():
    long_a = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            side="LONG",
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
            entry_date="2024-01-03",
            exit_date="2024-01-11",
        ),
    )

    result = run_chronological_turtle_portfolio(
        [long_a, long_b],
        config=TurtleChronologicalConfig(
            max_total_units=12,
            max_direction_units=1,
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
            entry_date="2024-01-03",
            exit_date="2024-01-11",
            average_entry_price=100.0,
            exit_price=90.0,
        ),
    )

    result = run_chronological_turtle_portfolio(
        [long_trade, short_trade],
        config=TurtleChronologicalConfig(
            max_total_units=2,
            max_direction_units=1,
        ),
    )

    assert len(result.accepted_trades) == 2
    assert not result.skipped_trades
    assert result.max_concurrent_units == 2


def test_same_day_exit_frees_units_before_new_entry():
    first = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            entry_date="2024-01-02",
            exit_date="2024-01-10",
        ),
    )

    replacement = TurtleDatedTrade(
        ticker="BBB",
        trade=_trade(
            entry_index=11,
            exit_index=21,
            entry_date="2024-01-10",
            exit_date="2024-01-20",
        ),
    )

    result = run_chronological_turtle_portfolio(
        [first, replacement],
        config=TurtleChronologicalConfig(
            max_total_units=1,
            max_direction_units=1,
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


def test_same_day_exit_frees_capacity_before_existing_position_add():
    exiting = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            entry_date="2024-01-02",
            exit_date="2024-01-04",
        ),
    )

    pyramiding = TurtleDatedTrade(
        ticker="BBB",
        trade=_trade(
            entry_index=11,
            exit_index=30,
            entry_date="2024-01-03",
            exit_date="2024-01-20",
            units=2,
            unit_entries=(
                _unit(
                    price=100.0,
                    index=11,
                    date="2024-01-03",
                ),
                _unit(
                    price=101.0,
                    index=12,
                    date="2024-01-04",
                ),
            ),
        ),
    )

    result = run_chronological_turtle_portfolio(
        [exiting, pyramiding],
        config=TurtleChronologicalConfig(
            max_total_units=2,
            max_direction_units=2,
        ),
    )

    events = [
        (
            point.event_type,
            point.ticker,
            point.active_units,
        )
        for point in result.equity_curve
        if point.event_date == "2024-01-04"
    ]

    assert events == [
        ("EXIT", "AAA", 1),
        ("ADD_2", "BBB", 2),
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

    # First trade: 25 shares * $20 = +$500 -> $5,500.
    assert (
        accepted_by_ticker["AAA"].realized_equity_after_exit
        == pytest.approx(5500.0)
    )

    assert (
        accepted_by_ticker["BBB"].shares_per_unit
        == 27
    )


def test_pnl_uses_only_actual_accepted_unit_fill_prices():
    a = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            exit_price=110.0,
            units=2,
            unit_entries=(
                _unit(
                    price=100.0,
                    index=10,
                    date="2024-01-02",
                ),
                _unit(
                    price=105.0,
                    index=12,
                    date="2024-01-04",
                ),
            ),
        ),
    )

    result = run_chronological_turtle_portfolio(
        [a],
        config=TurtleChronologicalConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
            max_total_units=2,
            max_direction_units=2,
        ),
    )

    accepted = result.accepted_trades[0]

    # 25 shares per unit:
    # first unit:  (110 - 100) * 25 = 250
    # second unit: (110 - 105) * 25 = 125
    assert accepted.pnl_dollars == pytest.approx(
        375.0
    )
    assert accepted.entry_price == pytest.approx(
        102.5
    )
    assert result.ending_equity == pytest.approx(
        5375.0
    )



def test_daily_mtm_curve_marks_open_long_position_to_close():
    trade = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            side="LONG",
            entry_date="2024-01-02",
            exit_date="2024-01-04",
            average_entry_price=100.0,
            exit_price=105.0,
            units=1,
        ),
    )

    result = run_chronological_turtle_portfolio(
        [trade],
        config=TurtleChronologicalConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
        ),
        mark_prices_by_ticker={
            "AAA": {
                "2024-01-02": 98.0,
                "2024-01-03": 90.0,
                "2024-01-04": 105.0,
            }
        },
    )

    points = {
        point.session_date: point
        for point in result.daily_mtm_curve
    }

    # 1N=$2, so $50 risk / $2 = 25 shares.
    assert points["2024-01-02"].unrealized_pnl == pytest.approx(-50.0)
    assert points["2024-01-02"].mtm_equity == pytest.approx(4950.0)

    assert points["2024-01-03"].unrealized_pnl == pytest.approx(-250.0)
    assert points["2024-01-03"].mtm_equity == pytest.approx(4750.0)
    assert points["2024-01-03"].mtm_drawdown_pct == pytest.approx(-5.0)

    # Exit is processed before the end-of-day mark, so open P&L is zero and
    # the realized account contains the completed trade P&L.
    assert points["2024-01-04"].unrealized_pnl == pytest.approx(0.0)
    assert points["2024-01-04"].mtm_equity == pytest.approx(5125.0)

    assert result.max_mtm_drawdown_pct == pytest.approx(-5.0)
    assert result.worst_mtm_drawdown_date == "2024-01-03"


def test_daily_mtm_curve_marks_short_position_correctly():
    trade = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            side="SHORT",
            entry_date="2024-01-02",
            exit_date="2024-01-04",
            average_entry_price=100.0,
            exit_price=95.0,
            units=1,
        ),
    )

    result = run_chronological_turtle_portfolio(
        [trade],
        mark_prices_by_ticker={
            "AAA": {
                "2024-01-02": 102.0,
                "2024-01-03": 110.0,
                "2024-01-04": 95.0,
            }
        },
    )

    points = {
        point.session_date: point
        for point in result.daily_mtm_curve
    }

    assert points["2024-01-02"].unrealized_pnl == pytest.approx(-50.0)
    assert points["2024-01-03"].unrealized_pnl == pytest.approx(-250.0)
    assert points["2024-01-04"].unrealized_pnl == pytest.approx(0.0)


def test_daily_mtm_uses_only_portfolio_accepted_units():
    trade = TurtleDatedTrade(
        ticker="AAA",
        trade=_trade(
            side="LONG",
            entry_date="2024-01-02",
            exit_date="2024-01-05",
            exit_price=104.0,
            units=2,
            unit_entries=(
                _unit(
                    price=100.0,
                    index=10,
                    date="2024-01-02",
                ),
                _unit(
                    price=101.0,
                    index=12,
                    date="2024-01-04",
                ),
            ),
        ),
    )

    blocker = TurtleDatedTrade(
        ticker="BBB",
        trade=_trade(
            side="LONG",
            entry_index=11,
            exit_index=30,
            entry_date="2024-01-03",
            exit_date="2024-01-10",
            average_entry_price=50.0,
            exit_price=50.0,
            units=1,
        ),
    )

    result = run_chronological_turtle_portfolio(
        [trade, blocker],
        config=TurtleChronologicalConfig(
            max_total_units=2,
            max_direction_units=12,
        ),
        mark_prices_by_ticker={
            "AAA": {
                "2024-01-02": 100.0,
                "2024-01-03": 100.0,
                "2024-01-04": 104.0,
                "2024-01-05": 104.0,
            },
            "BBB": {
                "2024-01-03": 50.0,
                "2024-01-04": 50.0,
                "2024-01-05": 50.0,
                "2024-01-10": 50.0,
            },
        },
    )

    day = next(
        point
        for point in result.daily_mtm_curve
        if point.session_date == "2024-01-04"
    )

    # AAA's second unit is rejected by the total-unit cap. MTM therefore
    # includes only the accepted first unit: 25 shares * ($104-$100) = $100.
    assert day.unrealized_pnl == pytest.approx(100.0)
    assert day.active_units == 2
