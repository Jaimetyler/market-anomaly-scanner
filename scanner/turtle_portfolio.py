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


# ---------------------------------------------------------------------------
# Chronological overlapping-portfolio layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurtleDatedTrade:
    """
    One completed single-market Turtle trade plus its ticker identity.

    The underlying TurtleTrade already contains entry_date / exit_date.
    Keeping ticker here lets the portfolio engine order and audit overlapping
    positions across the whole stock universe.
    """

    ticker: str
    trade: TurtleTrade


@dataclass(frozen=True)
class TurtleChronologicalConfig:
    starting_equity: float = 5000.0
    risk_fraction_per_unit: float = 0.01
    max_units_per_trade: int = 4

    # Original Turtle portfolio risk was expressed in "units".
    #
    # A stock-specific sector/correlation map is not wired yet, so this first
    # chronological layer enforces:
    #   - a total active-unit cap
    #   - a same-direction active-unit cap
    #
    # Correlated-market group caps come next.
    max_total_units: int = 12
    max_direction_units: int = 12


@dataclass(frozen=True)
class TurtleChronologicalTrade:
    ticker: str
    system: str
    side: str
    entry_date: str
    exit_date: str

    shares_per_unit: int
    units: int
    total_shares: int

    entry_price: float
    exit_price: float
    initial_n: float

    risk_dollars_per_unit: float
    pnl_dollars: float

    realized_equity_at_entry: float
    realized_equity_before_exit: float
    realized_equity_after_exit: float

    exit_reason: str


@dataclass(frozen=True)
class TurtlePortfolioSkip:
    ticker: str
    system: str
    side: str
    entry_date: str
    exit_date: str
    requested_units: int
    reason: str


@dataclass(frozen=True)
class TurtleChronologicalEquityPoint:
    event_date: str
    event_type: str
    ticker: str
    realized_equity: float
    peak_realized_equity: float
    realized_drawdown_pct: float
    active_positions: int
    active_units: int
    active_long_units: int
    active_short_units: int


@dataclass(frozen=True)
class TurtleChronologicalResult:
    starting_equity: float
    ending_equity: float
    total_pnl: float
    total_return_pct: float
    max_realized_drawdown_pct: float

    accepted_trades: tuple[TurtleChronologicalTrade, ...]
    skipped_trades: tuple[TurtlePortfolioSkip, ...]
    equity_curve: tuple[TurtleChronologicalEquityPoint, ...]

    max_concurrent_positions: int
    max_concurrent_units: int


@dataclass
class _ActivePosition:
    ticker: str
    trade: TurtleTrade
    shares_per_unit: int
    units: int
    total_shares: int
    risk_dollars_per_unit: float
    realized_equity_at_entry: float


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


def _validate_chronological_config(
    config: TurtleChronologicalConfig,
) -> None:
    if config.starting_equity <= 0:
        raise ValueError(
            "starting_equity must be positive"
        )

    if not (
        0.0
        < config.risk_fraction_per_unit
        < 1.0
    ):
        raise ValueError(
            "risk_fraction_per_unit must be between 0 and 1"
        )

    if config.max_units_per_trade < 1:
        raise ValueError(
            "max_units_per_trade must be at least 1"
        )

    if config.max_total_units < 1:
        raise ValueError(
            "max_total_units must be at least 1"
        )

    if config.max_direction_units < 1:
        raise ValueError(
            "max_direction_units must be at least 1"
        )


def _trade_date(
    trade: TurtleTrade,
    *,
    which: str,
) -> str:
    value = (
        trade.entry_date
        if which == "entry"
        else trade.exit_date
    )

    if value is None:
        return ""

    return str(value)


def _active_unit_counts(
    active: Sequence[_ActivePosition],
) -> tuple[int, int, int]:
    total = sum(
        position.units
        for position in active
    )

    long_units = sum(
        position.units
        for position in active
        if position.trade.side.upper() == "LONG"
    )

    short_units = sum(
        position.units
        for position in active
        if position.trade.side.upper() == "SHORT"
    )

    return (
        total,
        long_units,
        short_units,
    )


def run_chronological_turtle_portfolio(
    dated_trades: Sequence[TurtleDatedTrade],
    *,
    config: TurtleChronologicalConfig | None = None,
) -> TurtleChronologicalResult:
    """
    Run one shared Turtle account through trades in calendar order.

    This fixes the largest flaw in the old sequential proxy:
    overlapping positions are now active at the same time and compete for the
    same portfolio unit budget.

    Event ordering:
    - exits are processed before new entries on the same date
    - new entries are then processed in deterministic ticker/system/index order
    - equity used for sizing is realized equity available at that entry event

    IMPORTANT CURRENT LIMITATION:
    TurtleTrade currently records only the completed trade's final unit count
    and average entry price; it does not expose the timestamp of each 0.5N
    pyramid add. Therefore this layer conservatively reserves the trade's final
    unit count from its initial entry date through exit.

    That means:
    - overlapping-trade accounting is real chronological accounting
    - portfolio unit competition is conservative
    - exact intratrade add timing is not yet modeled
    - realized equity drawdown is not mark-to-market drawdown
    """
    if config is None:
        config = TurtleChronologicalConfig()

    _validate_chronological_config(
        config
    )

    candidates = [
        item
        for item in dated_trades
        if _trade_date(
            item.trade,
            which="entry",
        )
        and _trade_date(
            item.trade,
            which="exit",
        )
    ]

    entries_by_date: dict[
        str,
        list[TurtleDatedTrade],
    ] = {}

    exits_by_date: dict[
        str,
        list[TurtleDatedTrade],
    ] = {}

    for item in candidates:
        entry_date = _trade_date(
            item.trade,
            which="entry",
        )

        exit_date = _trade_date(
            item.trade,
            which="exit",
        )

        entries_by_date.setdefault(
            entry_date,
            [],
        ).append(item)

        exits_by_date.setdefault(
            exit_date,
            [],
        ).append(item)

    all_dates = sorted(
        set(entries_by_date)
        | set(exits_by_date)
    )

    realized_equity = (
        config.starting_equity
    )

    peak_realized_equity = (
        realized_equity
    )

    max_realized_drawdown = 0.0

    active_by_key: dict[
        tuple[
            str,
            str,
            int,
            int,
            str,
            str,
        ],
        _ActivePosition,
    ] = {}

    accepted: list[
        TurtleChronologicalTrade
    ] = []

    skipped: list[
        TurtlePortfolioSkip
    ] = []

    curve: list[
        TurtleChronologicalEquityPoint
    ] = []

    max_concurrent_positions = 0
    max_concurrent_units = 0

    def key_for(
        item: TurtleDatedTrade,
    ) -> tuple[
        str,
        str,
        int,
        int,
        str,
        str,
    ]:
        trade = item.trade

        return (
            item.ticker,
            trade.system,
            trade.entry_index,
            trade.exit_index,
            _trade_date(
                trade,
                which="entry",
            ),
            _trade_date(
                trade,
                which="exit",
            ),
        )

    def append_curve_point(
        *,
        event_date: str,
        event_type: str,
        ticker: str,
    ) -> None:
        nonlocal max_concurrent_positions
        nonlocal max_concurrent_units

        active = list(
            active_by_key.values()
        )

        (
            total_units,
            long_units,
            short_units,
        ) = _active_unit_counts(
            active
        )

        max_concurrent_positions = max(
            max_concurrent_positions,
            len(active),
        )

        max_concurrent_units = max(
            max_concurrent_units,
            total_units,
        )

        drawdown = _drawdown_pct(
            equity=realized_equity,
            peak_equity=(
                peak_realized_equity
            ),
        )

        curve.append(
            TurtleChronologicalEquityPoint(
                event_date=event_date,
                event_type=event_type,
                ticker=ticker,
                realized_equity=(
                    realized_equity
                ),
                peak_realized_equity=(
                    peak_realized_equity
                ),
                realized_drawdown_pct=(
                    drawdown
                ),
                active_positions=(
                    len(active)
                ),
                active_units=total_units,
                active_long_units=(
                    long_units
                ),
                active_short_units=(
                    short_units
                ),
            )
        )

    for event_date in all_dates:
        # --------------------------------------------------------------
        # Exits first.
        #
        # If a position exits today, its units are available for another
        # position that enters today.
        # --------------------------------------------------------------
        exiting = sorted(
            exits_by_date.get(
                event_date,
                [],
            ),
            key=lambda item: (
                item.ticker,
                item.trade.system,
                item.trade.entry_index,
                item.trade.exit_index,
            ),
        )

        for item in exiting:
            key = key_for(item)
            position = active_by_key.pop(
                key,
                None,
            )

            # A trade can be absent because it was rejected at entry.
            if position is None:
                continue

            trade = position.trade

            equity_before_exit = (
                realized_equity
            )

            pnl = _trade_pnl(
                side=trade.side,
                entry_price=(
                    trade.average_entry_price
                ),
                exit_price=(
                    trade.exit_price
                ),
                shares=(
                    position.total_shares
                ),
            )

            realized_equity = max(
                0.0,
                realized_equity
                + pnl,
            )

            peak_realized_equity = max(
                peak_realized_equity,
                realized_equity,
            )

            drawdown = _drawdown_pct(
                equity=realized_equity,
                peak_equity=(
                    peak_realized_equity
                ),
            )

            max_realized_drawdown = min(
                max_realized_drawdown,
                drawdown,
            )

            accepted.append(
                TurtleChronologicalTrade(
                    ticker=item.ticker,
                    system=trade.system,
                    side=trade.side,
                    entry_date=_trade_date(
                        trade,
                        which="entry",
                    ),
                    exit_date=_trade_date(
                        trade,
                        which="exit",
                    ),
                    shares_per_unit=(
                        position.shares_per_unit
                    ),
                    units=(
                        position.units
                    ),
                    total_shares=(
                        position.total_shares
                    ),
                    entry_price=(
                        trade.average_entry_price
                    ),
                    exit_price=(
                        trade.exit_price
                    ),
                    initial_n=(
                        trade.initial_n
                    ),
                    risk_dollars_per_unit=(
                        position.risk_dollars_per_unit
                    ),
                    pnl_dollars=pnl,
                    realized_equity_at_entry=(
                        position.realized_equity_at_entry
                    ),
                    realized_equity_before_exit=(
                        equity_before_exit
                    ),
                    realized_equity_after_exit=(
                        realized_equity
                    ),
                    exit_reason=(
                        trade.exit_reason
                    ),
                )
            )

            append_curve_point(
                event_date=event_date,
                event_type="EXIT",
                ticker=item.ticker,
            )

        if realized_equity <= 0:
            break

        # --------------------------------------------------------------
        # Entries second.
        # --------------------------------------------------------------
        entering = sorted(
            entries_by_date.get(
                event_date,
                [],
            ),
            key=lambda item: (
                item.ticker,
                item.trade.system,
                item.trade.entry_index,
                item.trade.exit_index,
            ),
        )

        for item in entering:
            trade = item.trade

            requested_units = min(
                trade.units,
                config.max_units_per_trade,
            )

            if requested_units < 1:
                skipped.append(
                    TurtlePortfolioSkip(
                        ticker=item.ticker,
                        system=trade.system,
                        side=trade.side,
                        entry_date=_trade_date(
                            trade,
                            which="entry",
                        ),
                        exit_date=_trade_date(
                            trade,
                            which="exit",
                        ),
                        requested_units=(
                            requested_units
                        ),
                        reason="NO_UNITS",
                    )
                )
                continue

            shares_per_unit = unit_shares(
                equity=realized_equity,
                n=trade.initial_n,
                risk_fraction=(
                    config.risk_fraction_per_unit
                ),
            )

            if shares_per_unit < 1:
                skipped.append(
                    TurtlePortfolioSkip(
                        ticker=item.ticker,
                        system=trade.system,
                        side=trade.side,
                        entry_date=_trade_date(
                            trade,
                            which="entry",
                        ),
                        exit_date=_trade_date(
                            trade,
                            which="exit",
                        ),
                        requested_units=(
                            requested_units
                        ),
                        reason="UNIT_SIZE_BELOW_ONE_SHARE",
                    )
                )
                continue

            active = list(
                active_by_key.values()
            )

            (
                total_units,
                long_units,
                short_units,
            ) = _active_unit_counts(
                active
            )

            if (
                total_units
                + requested_units
                > config.max_total_units
            ):
                skipped.append(
                    TurtlePortfolioSkip(
                        ticker=item.ticker,
                        system=trade.system,
                        side=trade.side,
                        entry_date=_trade_date(
                            trade,
                            which="entry",
                        ),
                        exit_date=_trade_date(
                            trade,
                            which="exit",
                        ),
                        requested_units=(
                            requested_units
                        ),
                        reason="TOTAL_UNIT_LIMIT",
                    )
                )
                continue

            side = trade.side.upper()

            directional_units = (
                long_units
                if side == "LONG"
                else short_units
            )

            if (
                directional_units
                + requested_units
                > config.max_direction_units
            ):
                skipped.append(
                    TurtlePortfolioSkip(
                        ticker=item.ticker,
                        system=trade.system,
                        side=trade.side,
                        entry_date=_trade_date(
                            trade,
                            which="entry",
                        ),
                        exit_date=_trade_date(
                            trade,
                            which="exit",
                        ),
                        requested_units=(
                            requested_units
                        ),
                        reason="DIRECTION_UNIT_LIMIT",
                    )
                )
                continue

            total_shares = (
                shares_per_unit
                * requested_units
            )

            risk_dollars_per_unit = (
                realized_equity
                * config.risk_fraction_per_unit
            )

            active_by_key[
                key_for(item)
            ] = _ActivePosition(
                ticker=item.ticker,
                trade=trade,
                shares_per_unit=(
                    shares_per_unit
                ),
                units=requested_units,
                total_shares=total_shares,
                risk_dollars_per_unit=(
                    risk_dollars_per_unit
                ),
                realized_equity_at_entry=(
                    realized_equity
                ),
            )

            append_curve_point(
                event_date=event_date,
                event_type="ENTRY",
                ticker=item.ticker,
            )

    # Anything still open here is only possible when the input set itself is
    # malformed (every completed TurtleTrade should have an exit date that was
    # included in all_dates). We intentionally do not manufacture an exit.
    total_pnl = (
        realized_equity
        - config.starting_equity
    )

    total_return_pct = (
        (
            realized_equity
            / config.starting_equity
        )
        - 1.0
    ) * 100.0

    return TurtleChronologicalResult(
        starting_equity=(
            config.starting_equity
        ),
        ending_equity=(
            realized_equity
        ),
        total_pnl=total_pnl,
        total_return_pct=(
            total_return_pct
        ),
        max_realized_drawdown_pct=(
            max_realized_drawdown
        ),
        accepted_trades=tuple(
            accepted
        ),
        skipped_trades=tuple(
            skipped
        ),
        equity_curve=tuple(
            curve
        ),
        max_concurrent_positions=(
            max_concurrent_positions
        ),
        max_concurrent_units=(
            max_concurrent_units
        ),
    )


# ---------------------------------------------------------------------------
# Legacy sequential proxy
#
# Kept for backward compatibility with existing reports/tests. New research
# should use run_chronological_turtle_portfolio().
# ---------------------------------------------------------------------------


def run_turtle_portfolio(
    trades: Sequence[TurtleTrade],
    *,
    config: TurtlePortfolioConfig | None = None,
) -> TurtlePortfolioResult:
    """
    Legacy sequential single-position proxy.

    This function is intentionally retained so older reports/tests continue
    to work. It is NOT a historical multi-position portfolio simulation.
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
