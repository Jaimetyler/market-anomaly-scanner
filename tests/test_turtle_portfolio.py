from __future__ import annotations

import pytest

from scanner.turtle_portfolio import (
    TurtlePortfolioConfig,
    run_turtle_portfolio,
    unit_shares,
)
from scanner.turtle_simulation import (
    TurtleTrade,
)


def make_trade(
    *,
    side: str = "LONG",
    average_entry_price: float = 100.0,
    exit_price: float = 110.0,
    n: float = 2.0,
    units: int = 1,
) -> TurtleTrade:
    return TurtleTrade(
        system="SYSTEM_2",
        side=side,
        entry_index=55,
        exit_index=65,
        entry_date="entry",
        exit_date="exit",
        initial_entry_price=(
            average_entry_price
        ),
        average_entry_price=(
            average_entry_price
        ),
        exit_price=exit_price,
        units=units,
        initial_n=n,
        initial_stop_price=(
            average_entry_price
            - 2 * n
            if side == "LONG"
            else average_entry_price
            + 2 * n
        ),
        final_stop_price=(
            average_entry_price
            - 2 * n
            if side == "LONG"
            else average_entry_price
            + 2 * n
        ),
        exit_reason="CHANNEL_EXIT",
        return_pct=0.0,
        unit_weighted_return_pct=0.0,
        bars_held=11,
        ambiguous_bars=0,
    )


def test_unit_shares_uses_equity_risk_over_n():
    # 1% of $5,000 = $50.
    # N = $2.
    # $50 / $2 = 25 shares.
    assert unit_shares(
        equity=5000.0,
        n=2.0,
        risk_fraction=0.01,
    ) == 25


def test_long_profit_updates_equity():
    trade = make_trade(
        side="LONG",
        average_entry_price=100.0,
        exit_price=110.0,
        n=2.0,
        units=1,
    )

    result = run_turtle_portfolio(
        [trade],
        config=TurtlePortfolioConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
        ),
    )

    # 25 shares * $10 profit.
    assert result.ending_equity == pytest.approx(
        5250.0
    )

    assert result.total_pnl == pytest.approx(
        250.0
    )


def test_short_profit_updates_equity():
    trade = make_trade(
        side="SHORT",
        average_entry_price=100.0,
        exit_price=90.0,
        n=2.0,
        units=1,
    )

    result = run_turtle_portfolio(
        [trade],
        config=TurtlePortfolioConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
        ),
    )

    assert result.ending_equity == pytest.approx(
        5250.0
    )


def test_multiple_units_scale_position():
    trade = make_trade(
        side="LONG",
        average_entry_price=100.0,
        exit_price=110.0,
        n=2.0,
        units=4,
    )

    result = run_turtle_portfolio(
        [trade],
        config=TurtlePortfolioConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
        ),
    )

    # 25 shares per unit * 4 units * $10.
    assert result.ending_equity == pytest.approx(
        6000.0
    )


def test_unit_cap_is_respected():
    trade = make_trade(
        side="LONG",
        average_entry_price=100.0,
        exit_price=110.0,
        n=2.0,
        units=4,
    )

    result = run_turtle_portfolio(
        [trade],
        config=TurtlePortfolioConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
            max_units_per_trade=2,
        ),
    )

    assert result.trades[0].units == 2
    assert result.trades[0].total_shares == 50


def test_losing_trade_creates_drawdown():
    trade = make_trade(
        side="LONG",
        average_entry_price=100.0,
        exit_price=96.0,
        n=2.0,
        units=1,
    )

    result = run_turtle_portfolio(
        [trade],
        config=TurtlePortfolioConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
        ),
    )

    assert result.ending_equity == pytest.approx(
        4900.0
    )

    assert result.max_drawdown_pct == pytest.approx(
        -2.0
    )


def test_compounding_changes_next_unit_size():
    first = make_trade(
        average_entry_price=100.0,
        exit_price=120.0,
        n=2.0,
        units=1,
    )

    second = make_trade(
        average_entry_price=100.0,
        exit_price=110.0,
        n=2.0,
        units=1,
    )

    result = run_turtle_portfolio(
        [first, second],
        config=TurtlePortfolioConfig(
            starting_equity=5000.0,
            risk_fraction_per_unit=0.01,
        ),
    )

    assert result.trades[0].shares_per_unit == 25

    # First trade adds $500:
    # equity becomes $5,500.
    # 1% = $55.
    # $55 / N2 = 27 whole shares.
    assert result.trades[1].shares_per_unit == 27


def test_tiny_account_can_skip_trade():
    trade = make_trade(
        n=1000.0,
    )

    result = run_turtle_portfolio(
        [trade],
        config=TurtlePortfolioConfig(
            starting_equity=100.0,
            risk_fraction_per_unit=0.01,
        ),
    )

    assert result.trades == ()
    assert result.ending_equity == pytest.approx(
        100.0
    )
