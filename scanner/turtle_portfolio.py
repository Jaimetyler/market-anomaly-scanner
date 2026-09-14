from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from scanner.turtle_simulation import TurtleTrade, TurtleUnit


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
    # Neutral, reproducible ordering for same-session stateful proposals.
    # Ticker names are used only as a final collision tie-breaker.
    entry_priority_seed: int = 0


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
    entry_priority_seed: int = 0
    entry_priority_rank: int = 0
    entry_priority_token: str = ""


@dataclass(frozen=True)
class TurtlePortfolioSkip:
    ticker: str
    system: str
    side: str
    entry_date: str
    exit_date: str
    requested_units: int
    reason: str
    priority_seed: int = 0
    priority_rank: int = 0
    priority_token: str = ""


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
class TurtleDailyMtmPoint:
    session_date: str
    realized_equity: float
    unrealized_pnl: float
    mtm_equity: float
    peak_mtm_equity: float
    mtm_drawdown_pct: float
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
    max_mtm_drawdown_pct: float
    worst_mtm_drawdown_date: str

    accepted_trades: tuple[TurtleChronologicalTrade, ...]
    skipped_trades: tuple[TurtlePortfolioSkip, ...]
    equity_curve: tuple[TurtleChronologicalEquityPoint, ...]
    daily_mtm_curve: tuple[TurtleDailyMtmPoint, ...]

    max_concurrent_positions: int
    max_concurrent_units: int


@dataclass
class _ActivePosition:
    ticker: str
    trade: TurtleTrade
    shares_per_unit: int
    accepted_units: list[TurtleUnit]
    risk_dollars_per_unit: float
    realized_equity_at_entry: float
    add_chain_open: bool = True

    @property
    def units(self) -> int:
        return len(self.accepted_units)

    @property
    def total_shares(self) -> int:
        return self.shares_per_unit * self.units


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


def _unit_entries(
    trade: TurtleTrade,
) -> tuple[TurtleUnit, ...]:
    """
    Return the simulator's actual per-unit execution timeline.

    Backward compatibility:
    older TurtleTrade-like objects may not expose unit_entries. In that case
    we synthesize the old conservative reservation behavior so legacy callers
    do not break. New production simulations should always provide the real
    timeline.
    """
    timeline = tuple(
        getattr(
            trade,
            "unit_entries",
            (),
        )
        or ()
    )

    if timeline:
        return timeline

    units = max(
        0,
        int(
            getattr(
                trade,
                "units",
                0,
            )
        ),
    )

    if units < 1:
        return ()

    entry_price = float(
        getattr(
            trade,
            "average_entry_price",
            0.0,
        )
    )

    n = float(
        getattr(
            trade,
            "initial_n",
            0.0,
        )
    )

    entry_index = int(
        getattr(
            trade,
            "entry_index",
            -1,
        )
    )

    entry_date = _trade_date(
        trade,
        which="entry",
    )

    return tuple(
        TurtleUnit(
            entry_price=entry_price,
            n=n,
            entry_index=entry_index,
            entry_date=entry_date,
        )
        for _ in range(units)
    )


def _accepted_entry_price(
    position: _ActivePosition,
) -> float:
    if not position.accepted_units:
        return 0.0

    return sum(
        unit.entry_price
        for unit in position.accepted_units
    ) / len(
        position.accepted_units
    )


def _position_pnl(
    *,
    position: _ActivePosition,
    exit_price: float,
) -> float:
    return sum(
        _trade_pnl(
            side=position.trade.side,
            entry_price=unit.entry_price,
            exit_price=exit_price,
            shares=position.shares_per_unit,
        )
        for unit in position.accepted_units
    )


def _position_unrealized_pnl(
    *,
    position: _ActivePosition,
    mark_price: float,
) -> float:
    return sum(
        _trade_pnl(
            side=position.trade.side,
            entry_price=unit.entry_price,
            exit_price=mark_price,
            shares=position.shares_per_unit,
        )
        for unit in position.accepted_units
    )


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
    mark_prices_by_ticker: Mapping[
        str,
        Mapping[str, float],
    ] | None = None,
) -> TurtleChronologicalResult:
    """
    Run one shared Turtle account through trades in calendar order.

    The single-market simulator now exposes the actual initial-entry and
    0.5N pyramid-add dates for every completed trade. Portfolio capacity is
    therefore consumed one unit at a time on the dates those executions
    actually occurred.

    Same-date portfolio event ordering is deterministic:
    1. exits
    2. pyramid adds for already-open positions
    3. new initial entries

    This preserves the existing convention that exits free capacity before
    other same-day actions and gives an already-open Turtle position priority
    for a valid pyramid add over a brand-new position.

    If a portfolio limit rejects a pyramid add, later precomputed adds for that
    trade are not executed. Those later levels were generated by the
    single-market path assuming the rejected add had occurred, so continuing
    the chain would invent a path the portfolio never actually held.

    Daily mark-to-market reporting:
    - when mark_prices_by_ticker is supplied, end-of-day MTM equity is
      calculated from each open accepted unit using that ticker's adjusted
      daily close
    - missing ticker closes carry the most recent known close forward
    - position sizing still uses realized equity; MTM is reporting-only here

    Remaining limitations:
    - position sizing is still realized-equity based
    - no sector/correlation group limits yet
    - no margin, borrow, commissions, or slippage
    - daily OHLC cannot establish cross-market intraday event ordering
    """
    if config is None:
        config = TurtleChronologicalConfig()

    _validate_chronological_config(config)

    candidates = [
        item
        for item in dated_trades
        if _trade_date(item.trade, which="entry")
        and _trade_date(item.trade, which="exit")
    ]

    entries_by_date: dict[str, list[TurtleDatedTrade]] = {}
    exits_by_date: dict[str, list[TurtleDatedTrade]] = {}
    adds_by_date: dict[
        str,
        list[tuple[TurtleDatedTrade, int, TurtleUnit]],
    ] = {}

    for item in candidates:
        trade = item.trade
        timeline = _unit_entries(trade)

        entry_date = _trade_date(
            trade,
            which="entry",
        )
        exit_date = _trade_date(
            trade,
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

        # The first unit is the initial entry. Later units are actual adds.
        for unit_number, unit in enumerate(
            timeline[1:],
            start=2,
        ):
            add_date = str(
                unit.entry_date
                or ""
            )

            if not add_date:
                continue

            adds_by_date.setdefault(
                add_date,
                [],
            ).append(
                (
                    item,
                    unit_number,
                    unit,
                )
            )

    marks = mark_prices_by_ticker or {}

    mark_dates: set[str] = set()
    for ticker_marks in marks.values():
        mark_dates.update(
            str(session_date)
            for session_date in ticker_marks
        )

    all_dates = sorted(
        set(entries_by_date)
        | set(exits_by_date)
        | set(adds_by_date)
        | mark_dates
    )

    realized_equity = config.starting_equity
    peak_realized_equity = realized_equity
    max_realized_drawdown = 0.0

    active_by_key: dict[
        tuple[str, str, int, int, str, str],
        _ActivePosition,
    ] = {}

    accepted: list[TurtleChronologicalTrade] = []
    skipped: list[TurtlePortfolioSkip] = []
    curve: list[TurtleChronologicalEquityPoint] = []
    daily_mtm_curve: list[TurtleDailyMtmPoint] = []

    last_mark_by_ticker: dict[str, float] = {}
    peak_mtm_equity = config.starting_equity
    max_mtm_drawdown = 0.0
    worst_mtm_drawdown_date = ""

    max_concurrent_positions = 0
    max_concurrent_units = 0

    def key_for(
        item: TurtleDatedTrade,
    ) -> tuple[str, str, int, int, str, str]:
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
        ) = _active_unit_counts(active)

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
            peak_equity=peak_realized_equity,
        )

        curve.append(
            TurtleChronologicalEquityPoint(
                event_date=event_date,
                event_type=event_type,
                ticker=ticker,
                realized_equity=realized_equity,
                peak_realized_equity=(
                    peak_realized_equity
                ),
                realized_drawdown_pct=drawdown,
                active_positions=len(active),
                active_units=total_units,
                active_long_units=long_units,
                active_short_units=short_units,
            )
        )

    def append_daily_mtm_point(
        *,
        session_date: str,
    ) -> None:
        nonlocal peak_mtm_equity
        nonlocal max_mtm_drawdown
        nonlocal worst_mtm_drawdown_date

        active = list(
            active_by_key.values()
        )

        (
            total_units,
            long_units,
            short_units,
        ) = _active_unit_counts(active)

        unrealized_pnl = 0.0

        for position in active:
            ticker = position.ticker
            mark_price = last_mark_by_ticker.get(
                ticker
            )

            if mark_price is None:
                # A newly-entered position should normally have a same-day
                # close. If not, use its latest accepted fill until the first
                # market close arrives rather than fabricating a zero mark.
                if position.accepted_units:
                    mark_price = (
                        position.accepted_units[-1].entry_price
                    )
                else:
                    continue

            unrealized_pnl += (
                _position_unrealized_pnl(
                    position=position,
                    mark_price=mark_price,
                )
            )

        mtm_equity = max(
            0.0,
            realized_equity + unrealized_pnl,
        )

        peak_mtm_equity = max(
            peak_mtm_equity,
            mtm_equity,
        )

        mtm_drawdown = _drawdown_pct(
            equity=mtm_equity,
            peak_equity=peak_mtm_equity,
        )

        if mtm_drawdown < max_mtm_drawdown:
            max_mtm_drawdown = mtm_drawdown
            worst_mtm_drawdown_date = session_date

        daily_mtm_curve.append(
            TurtleDailyMtmPoint(
                session_date=session_date,
                realized_equity=realized_equity,
                unrealized_pnl=unrealized_pnl,
                mtm_equity=mtm_equity,
                peak_mtm_equity=peak_mtm_equity,
                mtm_drawdown_pct=mtm_drawdown,
                active_positions=len(active),
                active_units=total_units,
                active_long_units=long_units,
                active_short_units=short_units,
            )
        )

    def limit_reason(
        *,
        side: str,
        requested_units: int,
    ) -> str | None:
        active = list(
            active_by_key.values()
        )

        (
            total_units,
            long_units,
            short_units,
        ) = _active_unit_counts(active)

        if (
            total_units
            + requested_units
            > config.max_total_units
        ):
            return "TOTAL_UNIT_LIMIT"

        directional_units = (
            long_units
            if side.upper() == "LONG"
            else short_units
        )

        if (
            directional_units
            + requested_units
            > config.max_direction_units
        ):
            return "DIRECTION_UNIT_LIMIT"

        return None

    for event_date in all_dates:
        for ticker, ticker_marks in marks.items():
            if event_date in ticker_marks:
                last_mark_by_ticker[ticker] = float(
                    ticker_marks[event_date]
                )

        # --------------------------------------------------------------
        # 1. Exits
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

            # Rejected initial entries never became active.
            if position is None:
                continue

            trade = position.trade
            equity_before_exit = realized_equity

            pnl = _position_pnl(
                position=position,
                exit_price=trade.exit_price,
            )

            realized_equity = max(
                0.0,
                realized_equity + pnl,
            )

            peak_realized_equity = max(
                peak_realized_equity,
                realized_equity,
            )

            drawdown = _drawdown_pct(
                equity=realized_equity,
                peak_equity=peak_realized_equity,
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
                    units=position.units,
                    total_shares=(
                        position.total_shares
                    ),
                    entry_price=(
                        _accepted_entry_price(
                            position
                        )
                    ),
                    exit_price=trade.exit_price,
                    initial_n=trade.initial_n,
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
                    exit_reason=trade.exit_reason,
                )
            )

            append_curve_point(
                event_date=event_date,
                event_type="EXIT",
                ticker=item.ticker,
            )

        if realized_equity <= 0:
            append_daily_mtm_point(
                session_date=event_date,
            )
            break

        # --------------------------------------------------------------
        # 2. Actual pyramid adds
        # --------------------------------------------------------------
        adding = sorted(
            adds_by_date.get(
                event_date,
                [],
            ),
            key=lambda event: (
                event[0].ticker,
                event[0].trade.system,
                event[0].trade.entry_index,
                event[1],
            ),
        )

        for item, unit_number, unit in adding:
            key = key_for(item)

            position = active_by_key.get(key)

            # Trade may have been rejected at initial entry or may already
            # have exited earlier on this same date.
            if position is None:
                continue

            if not position.add_chain_open:
                skipped.append(
                    TurtlePortfolioSkip(
                        ticker=item.ticker,
                        system=item.trade.system,
                        side=item.trade.side,
                        entry_date=_trade_date(
                            item.trade,
                            which="entry",
                        ),
                        exit_date=_trade_date(
                            item.trade,
                            which="exit",
                        ),
                        requested_units=1,
                        reason="ADD_CHAIN_BROKEN",
                    )
                )
                continue

            if (
                position.units
                >= config.max_units_per_trade
            ):
                position.add_chain_open = False

                skipped.append(
                    TurtlePortfolioSkip(
                        ticker=item.ticker,
                        system=item.trade.system,
                        side=item.trade.side,
                        entry_date=_trade_date(
                            item.trade,
                            which="entry",
                        ),
                        exit_date=_trade_date(
                            item.trade,
                            which="exit",
                        ),
                        requested_units=1,
                        reason="TRADE_UNIT_LIMIT",
                    )
                )
                continue

            reason = limit_reason(
                side=item.trade.side,
                requested_units=1,
            )

            if reason is not None:
                # Later simulator add levels assume this add happened.
                # Once this link is rejected, do not invent downstream fills.
                position.add_chain_open = False

                skipped.append(
                    TurtlePortfolioSkip(
                        ticker=item.ticker,
                        system=item.trade.system,
                        side=item.trade.side,
                        entry_date=_trade_date(
                            item.trade,
                            which="entry",
                        ),
                        exit_date=_trade_date(
                            item.trade,
                            which="exit",
                        ),
                        requested_units=1,
                        reason=f"ADD_{reason}",
                    )
                )
                continue

            position.accepted_units.append(unit)

            append_curve_point(
                event_date=event_date,
                event_type=f"ADD_{unit_number}",
                ticker=item.ticker,
            )

        # --------------------------------------------------------------
        # 3. Initial entries
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
            timeline = _unit_entries(trade)

            if not timeline:
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
                        requested_units=0,
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
                        requested_units=1,
                        reason=(
                            "UNIT_SIZE_BELOW_ONE_SHARE"
                        ),
                    )
                )
                continue

            reason = limit_reason(
                side=trade.side,
                requested_units=1,
            )

            if reason is not None:
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
                        requested_units=1,
                        reason=reason,
                    )
                )
                continue

            risk_dollars_per_unit = (
                realized_equity
                * config.risk_fraction_per_unit
            )

            active_by_key[key_for(item)] = (
                _ActivePosition(
                    ticker=item.ticker,
                    trade=trade,
                    shares_per_unit=(
                        shares_per_unit
                    ),
                    accepted_units=[
                        timeline[0]
                    ],
                    risk_dollars_per_unit=(
                        risk_dollars_per_unit
                    ),
                    realized_equity_at_entry=(
                        realized_equity
                    ),
                )
            )

            append_curve_point(
                event_date=event_date,
                event_type="ENTRY",
                ticker=item.ticker,
            )

        append_daily_mtm_point(
            session_date=event_date,
        )

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
        starting_equity=config.starting_equity,
        ending_equity=realized_equity,
        total_pnl=total_pnl,
        total_return_pct=total_return_pct,
        max_realized_drawdown_pct=(
            max_realized_drawdown
        ),
        max_mtm_drawdown_pct=(
            max_mtm_drawdown
        ),
        worst_mtm_drawdown_date=(
            worst_mtm_drawdown_date
        ),
        accepted_trades=tuple(accepted),
        skipped_trades=tuple(skipped),
        equity_curve=tuple(curve),
        daily_mtm_curve=tuple(
            daily_mtm_curve
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
