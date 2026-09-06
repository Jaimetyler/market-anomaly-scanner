from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scanner.turtle_simulation import TurtleTrade


@dataclass(frozen=True)
class TurtlePortfolioConfig:
    starting_equity: float = 5000.0
    risk_fraction_per_unit: float = 0.01
    max_units_per_trade: int = 4


@dataclass(frozen=True)
class TurtlePortfolioTrade:
    system: str
    side: str

    entry_index: int
    exit_index: int

    entry_price: float
    exit_price: float

    shares_per_unit: int
    units: int
    total_shares: int

    risk_dollars_per_unit: float
    pnl_dollars: float
    return_on_equity_pct: float

    equity_before: float
    equity_after: float

    exit_reason: str


@dataclass(frozen=True)
class TurtleEquityPoint:
    trade_number: int
    equity: float
    peak_equity: float
    drawdown_pct: float


@dataclass(frozen=True)
class TurtlePortfolioResult:
    starting_equity: float
    ending_equity: float
    total_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: tuple[TurtlePortfolioTrade, ...]
    equity_curve: tuple[TurtleEquityPoint, ...]


def unit_shares(
    *,
    equity: float,
    n: float,
    risk_fraction: float,
) -> int:
    """
    Turtle-style normalized position sizing.

    One unit risks approximately:

        equity * risk_fraction

    against a 1N adverse move.

    shares = risk_dollars / N
    """
    if equity <= 0:
        return 0

    if n <= 0:
        return 0

    if risk_fraction <= 0:
        raise ValueError(
            "risk_fraction must be positive"
        )

    risk_dollars = (
        equity
        * risk_fraction
    )

    return max(
        0,
        int(
            risk_dollars
            // n
        ),
    )


def _trade_pnl(
    *,
    side: str,
    entry_price: float,
    exit_price: float,
    shares: int,
) -> float:
    side = side.upper()

    if side == "LONG":
        return (
            exit_price
            - entry_price
        ) * shares

    if side == "SHORT":
        return (
            entry_price
            - exit_price
        ) * shares

    raise ValueError(
        f"Unknown side: {side!r}"
    )


def _drawdown_pct(
    *,
    equity: float,
    peak_equity: float,
) -> float:
    if peak_equity <= 0:
        return 0.0

    return (
        (equity / peak_equity)
        - 1.0
    ) * 100.0


def run_turtle_portfolio(
    trades: Sequence[TurtleTrade],
    *,
    config: TurtlePortfolioConfig | None = None,
) -> TurtlePortfolioResult:
    """
    Sequential single-position portfolio simulation.

    This is intentionally the first/simple portfolio layer.

    Assumptions:
    - trades are applied sequentially in the order received
    - one unit is sized using current equity and initial N
    - every pyramid unit uses the same shares-per-unit
    - no margin interest, commissions, slippage, borrow fees, or taxes yet
    - no overlapping-position portfolio constraint yet
    - fractional shares are not used
    - if unit size rounds below 1 share, the trade is skipped
    """
    if config is None:
        config = TurtlePortfolioConfig()

    if config.starting_equity <= 0:
        raise ValueError(
            "starting_equity must be positive"
        )

    if not (
        0
        < config.risk_fraction_per_unit
        < 1
    ):
        raise ValueError(
            "risk_fraction_per_unit must be between 0 and 1"
        )

    equity = config.starting_equity
    peak_equity = equity
    max_drawdown = 0.0

    portfolio_trades: list[
        TurtlePortfolioTrade
    ] = []

    curve: list[
        TurtleEquityPoint
    ] = [
        TurtleEquityPoint(
            trade_number=0,
            equity=equity,
            peak_equity=peak_equity,
            drawdown_pct=0.0,
        )
    ]

    for trade in trades:
        shares_per_unit = unit_shares(
            equity=equity,
            n=trade.initial_n,
            risk_fraction=(
                config.risk_fraction_per_unit
            ),
        )

        if shares_per_unit < 1:
            continue

        actual_units = min(
            trade.units,
            config.max_units_per_trade,
        )

        total_shares = (
            shares_per_unit
            * actual_units
        )

        equity_before = equity

        pnl = _trade_pnl(
            side=trade.side,
            entry_price=(
                trade.average_entry_price
            ),
            exit_price=trade.exit_price,
            shares=total_shares,
        )

        equity = max(
            0.0,
            equity
            + pnl,
        )

        return_on_equity_pct = (
            (
                equity
                / equity_before
            )
            - 1.0
        ) * 100.0

        portfolio_trades.append(
            TurtlePortfolioTrade(
                system=trade.system,
                side=trade.side,
                entry_index=(
                    trade.entry_index
                ),
                exit_index=(
                    trade.exit_index
                ),
                entry_price=(
                    trade.average_entry_price
                ),
                exit_price=(
                    trade.exit_price
                ),
                shares_per_unit=(
                    shares_per_unit
                ),
                units=actual_units,
                total_shares=(
                    total_shares
                ),
                risk_dollars_per_unit=(
                    equity_before
                    * config.risk_fraction_per_unit
                ),
                pnl_dollars=pnl,
                return_on_equity_pct=(
                    return_on_equity_pct
                ),
                equity_before=(
                    equity_before
                ),
                equity_after=(
                    equity
                ),
                exit_reason=(
                    trade.exit_reason
                ),
            )
        )

        peak_equity = max(
            peak_equity,
            equity,
        )

        dd = _drawdown_pct(
            equity=equity,
            peak_equity=peak_equity,
        )

        max_drawdown = min(
            max_drawdown,
            dd,
        )

        curve.append(
            TurtleEquityPoint(
                trade_number=len(
                    portfolio_trades
                ),
                equity=equity,
                peak_equity=(
                    peak_equity
                ),
                drawdown_pct=dd,
            )
        )

        if equity <= 0:
            break

    total_pnl = (
        equity
        - config.starting_equity
    )

    total_return_pct = (
        (
            equity
            / config.starting_equity
        )
        - 1.0
    ) * 100.0

    return TurtlePortfolioResult(
        starting_equity=(
            config.starting_equity
        ),
        ending_equity=equity,
        total_pnl=total_pnl,
        total_return_pct=(
            total_return_pct
        ),
        max_drawdown_pct=(
            max_drawdown
        ),
        trades=tuple(
            portfolio_trades
        ),
        equity_curve=tuple(
            curve
        ),
    )
