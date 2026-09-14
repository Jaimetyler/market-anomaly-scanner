from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from scanner.turtle import TurtleSystem, bar_close
from scanner.turtle_portfolio import (
    TurtleChronologicalConfig,
    TurtleChronologicalEquityPoint,
    TurtleChronologicalResult,
    TurtleChronologicalTrade,
    TurtleDailyMtmPoint,
    TurtlePortfolioSkip,
    _drawdown_pct,
    _trade_pnl,
    _validate_chronological_config,
    unit_shares,
)
from scanner.turtle_stateful import TurtleMarketState, TurtleStateAction


@dataclass(frozen=True)
class TurtleStatefulSegment:
    ticker: str
    segment_number: int
    bars: Sequence[Any]


@dataclass
class _LivePosition:
    ticker: str
    segment_number: int
    state: TurtleMarketState
    shares_per_unit: int
    risk_dollars_per_unit: float
    realized_equity_at_entry: float
    entry_priority_seed: int
    entry_priority_rank: int
    entry_priority_token: str

    @property
    def units(self) -> int:
        return self.state.units

    @property
    def side(self) -> str:
        return self.state.side

    @property
    def accepted_units(self):
        return self.state.accepted_units


def _priority_token(
    *,
    seed: int,
    key: tuple[str, int],
    action: TurtleStateAction,
) -> str:
    """Stable neutral priority without using post-fill market information."""
    payload = "|".join(
        (
            str(seed),
            action.session_date,
            action.action_type,
            action.system,
            action.side,
            key[0],
            str(key[1]),
        )
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _rank_proposals(
    proposals: Sequence[tuple[tuple[str, int], TurtleStateAction]],
    *,
    seed: int,
) -> list[tuple[tuple[str, int], TurtleStateAction, int, str]]:
    decorated = [
        (key, action, _priority_token(seed=seed, key=key, action=action))
        for key, action in proposals
    ]
    decorated.sort(key=lambda item: (item[2], item[0][0], item[0][1]))
    return [
        (key, action, rank, token)
        for rank, (key, action, token) in enumerate(decorated, start=1)
    ]

def _bar_date(bar: Any) -> str:
    if isinstance(bar, dict):
        for key in ("session_date", "date", "signal_date"):
            value = bar.get(key)
            if value is not None:
                return value.isoformat() if hasattr(value, "isoformat") else str(value)
        return ""

    for key in ("session_date", "date", "signal_date"):
        if hasattr(bar, key):
            value = getattr(bar, key)
            if value is not None:
                return value.isoformat() if hasattr(value, "isoformat") else str(value)
    return ""


def run_stateful_turtle_portfolio(
    segments: Sequence[TurtleStatefulSegment],
    *,
    system: TurtleSystem,
    config: TurtleChronologicalConfig | None = None,
    mark_prices_by_ticker: Mapping[str, Mapping[str, float]] | None = None,
) -> TurtleChronologicalResult:
    """
    Shared chronological portfolio whose limits participate in trade state.

    Same-date convention:
      1. market states advance and propose their highest-priority action
      2. exits execute first
      3. pyramid adds execute/reject next
      4. new entries execute/reject last
      5. segment-end positions close at that segment's final adjusted close
      6. end-of-day MTM is recorded

    A rejected add does not move a stop, does not create a unit, and does not
    create downstream add levels. If the still-unfilled level is touched on a
    later bar, it can be proposed again.

    Position sizing uses realized equity. MTM is reporting-only.
    """
    if config is None:
        config = TurtleChronologicalConfig()
    _validate_chronological_config(config)

    marks = mark_prices_by_ticker or {}

    states: dict[tuple[str, int], TurtleMarketState] = {}
    bar_index_by_date: dict[str, list[tuple[tuple[str, int], int]]] = {}
    segment_end_by_date: dict[str, list[tuple[str, int]]] = {}

    for segment in segments:
        bars = list(segment.bars)
        if not bars:
            continue

        key = (segment.ticker, segment.segment_number)
        states[key] = TurtleMarketState(bars, system=system)

        for index, bar in enumerate(bars):
            session_date = _bar_date(bar)
            if not session_date:
                continue
            bar_index_by_date.setdefault(session_date, []).append((key, index))

        final_date = _bar_date(bars[-1])
        if final_date:
            segment_end_by_date.setdefault(final_date, []).append(key)

    mark_dates = {
        str(session_date)
        for ticker_marks in marks.values()
        for session_date in ticker_marks
    }
    all_dates = sorted(
        set(bar_index_by_date)
        | set(segment_end_by_date)
        | mark_dates
    )

    realized_equity = config.starting_equity
    peak_realized_equity = realized_equity
    max_realized_drawdown = 0.0

    peak_mtm_equity = config.starting_equity
    max_mtm_drawdown = 0.0
    worst_mtm_drawdown_date = ""

    active: dict[tuple[str, int], _LivePosition] = {}
    accepted: list[TurtleChronologicalTrade] = []
    skipped: list[TurtlePortfolioSkip] = []
    curve: list[TurtleChronologicalEquityPoint] = []
    daily_mtm_curve: list[TurtleDailyMtmPoint] = []

    last_mark_by_ticker: dict[str, float] = {}

    max_concurrent_positions = 0
    max_concurrent_units = 0

    def counts() -> tuple[int, int, int]:
        total = sum(position.units for position in active.values())
        longs = sum(
            position.units
            for position in active.values()
            if position.side.upper() == "LONG"
        )
        shorts = sum(
            position.units
            for position in active.values()
            if position.side.upper() == "SHORT"
        )
        return total, longs, shorts

    def limit_reason(side: str) -> str | None:
        total, longs, shorts = counts()
        if total + 1 > config.max_total_units:
            return "TOTAL_UNIT_LIMIT"
        directional = longs if side.upper() == "LONG" else shorts
        if directional + 1 > config.max_direction_units:
            return "DIRECTION_UNIT_LIMIT"
        return None

    def append_curve(
        *,
        session_date: str,
        event_type: str,
        ticker: str,
    ) -> None:
        nonlocal max_concurrent_positions, max_concurrent_units
        total, longs, shorts = counts()
        max_concurrent_positions = max(max_concurrent_positions, len(active))
        max_concurrent_units = max(max_concurrent_units, total)
        curve.append(
            TurtleChronologicalEquityPoint(
                event_date=session_date,
                event_type=event_type,
                ticker=ticker,
                realized_equity=realized_equity,
                peak_realized_equity=peak_realized_equity,
                realized_drawdown_pct=_drawdown_pct(
                    equity=realized_equity,
                    peak_equity=peak_realized_equity,
                ),
                active_positions=len(active),
                active_units=total,
                active_long_units=longs,
                active_short_units=shorts,
            )
        )

    def close_position(
        *,
        key: tuple[str, int],
        closed_trade,
        session_date: str,
    ) -> None:
        nonlocal realized_equity
        nonlocal peak_realized_equity
        nonlocal max_realized_drawdown

        position = active.pop(key)
        equity_before_exit = realized_equity

        pnl = sum(
            _trade_pnl(
                side=closed_trade.side,
                entry_price=unit.entry_price,
                exit_price=closed_trade.exit_price,
                shares=position.shares_per_unit,
            )
            for unit in closed_trade.unit_entries
        )

        realized_equity = max(0.0, realized_equity + pnl)
        peak_realized_equity = max(peak_realized_equity, realized_equity)
        dd = _drawdown_pct(
            equity=realized_equity,
            peak_equity=peak_realized_equity,
        )
        max_realized_drawdown = min(max_realized_drawdown, dd)

        average_entry = (
            sum(unit.entry_price for unit in closed_trade.unit_entries)
            / len(closed_trade.unit_entries)
        )

        accepted.append(
            TurtleChronologicalTrade(
                ticker=position.ticker,
                system=closed_trade.system,
                side=closed_trade.side,
                entry_date=closed_trade.entry_date,
                exit_date=closed_trade.exit_date,
                shares_per_unit=position.shares_per_unit,
                units=len(closed_trade.unit_entries),
                total_shares=(
                    position.shares_per_unit
                    * len(closed_trade.unit_entries)
                ),
                entry_price=average_entry,
                exit_price=closed_trade.exit_price,
                initial_n=closed_trade.initial_n,
                risk_dollars_per_unit=position.risk_dollars_per_unit,
                pnl_dollars=pnl,
                realized_equity_at_entry=position.realized_equity_at_entry,
                realized_equity_before_exit=equity_before_exit,
                realized_equity_after_exit=realized_equity,
                exit_reason=closed_trade.exit_reason,
                entry_priority_seed=position.entry_priority_seed,
                entry_priority_rank=position.entry_priority_rank,
                entry_priority_token=position.entry_priority_token,
            )
        )
        append_curve(
            session_date=session_date,
            event_type="EXIT",
            ticker=position.ticker,
        )

    def append_mtm(session_date: str) -> None:
        nonlocal peak_mtm_equity
        nonlocal max_mtm_drawdown
        nonlocal worst_mtm_drawdown_date

        total, longs, shorts = counts()
        unrealized = 0.0

        for position in active.values():
            mark = last_mark_by_ticker.get(position.ticker)
            if mark is None:
                units = position.accepted_units
                if not units:
                    continue
                mark = units[-1].entry_price

            unrealized += sum(
                _trade_pnl(
                    side=position.side,
                    entry_price=unit.entry_price,
                    exit_price=mark,
                    shares=position.shares_per_unit,
                )
                for unit in position.accepted_units
            )

        mtm_equity = max(0.0, realized_equity + unrealized)
        peak_mtm_equity = max(peak_mtm_equity, mtm_equity)
        mtm_dd = _drawdown_pct(
            equity=mtm_equity,
            peak_equity=peak_mtm_equity,
        )
        if mtm_dd < max_mtm_drawdown:
            max_mtm_drawdown = mtm_dd
            worst_mtm_drawdown_date = session_date

        daily_mtm_curve.append(
            TurtleDailyMtmPoint(
                session_date=session_date,
                realized_equity=realized_equity,
                unrealized_pnl=unrealized,
                mtm_equity=mtm_equity,
                peak_mtm_equity=peak_mtm_equity,
                mtm_drawdown_pct=mtm_dd,
                active_positions=len(active),
                active_units=total,
                active_long_units=longs,
                active_short_units=shorts,
            )
        )

    for session_date in all_dates:
        for ticker, ticker_marks in marks.items():
            if session_date in ticker_marks:
                last_mark_by_ticker[ticker] = float(ticker_marks[session_date])

        proposals: list[
            tuple[tuple[str, int], TurtleStateAction]
        ] = []

        for key, index in sorted(
            bar_index_by_date.get(session_date, []),
            key=lambda item: (item[0][0], item[0][1]),
        ):
            state = states[key]
            for action in state.propose(index):
                proposals.append((key, action))

        # 1. Exits.
        for key, action in sorted(
            (
                item for item in proposals
                if item[1].action_type == "EXIT"
            ),
            key=lambda item: (item[0][0], item[0][1]),
        ):
            if key not in active:
                raise RuntimeError("state proposed exit without active portfolio position")
            closed = states[key].accept(action)
            if closed is None:
                raise RuntimeError("accepted exit did not close trade")
            close_position(
                key=key,
                closed_trade=closed,
                session_date=session_date,
            )

        if realized_equity <= 0:
            append_mtm(session_date)
            break

        # 2. Adds. Multiple levels may execute on the same bar.
        for key, first_action, priority_rank, priority_token in _rank_proposals(
            [
                item for item in proposals
                if item[1].action_type == "ADD"
            ],
            seed=config.entry_priority_seed,
        ):
            if key not in active:
                continue

            state = states[key]
            action: TurtleStateAction | None = first_action

            while action is not None:
                if state.units >= config.max_units_per_trade:
                    skipped.append(
                        TurtlePortfolioSkip(
                            ticker=key[0],
                            system=system.name,
                            side=action.side,
                            entry_date=(
                                state.open_trade.entry_date
                                if state.open_trade is not None
                                else ""
                            ),
                            exit_date="",
                            requested_units=1,
                            reason="TRADE_UNIT_LIMIT",
                            priority_seed=config.entry_priority_seed,
                            priority_rank=priority_rank,
                            priority_token=priority_token,
                        )
                    )
                    state.reject(action)
                    break

                reason = limit_reason(action.side)
                if reason is not None:
                    skipped.append(
                        TurtlePortfolioSkip(
                            ticker=key[0],
                            system=system.name,
                            side=action.side,
                            entry_date=(
                                state.open_trade.entry_date
                                if state.open_trade is not None
                                else ""
                            ),
                            exit_date="",
                            requested_units=1,
                            reason=f"ADD_{reason}",
                            priority_seed=config.entry_priority_seed,
                            priority_rank=priority_rank,
                            priority_token=priority_token,
                        )
                    )
                    state.reject(action)
                    break

                state.accept(action)
                append_curve(
                    session_date=session_date,
                    event_type=f"ADD_{state.units}",
                    ticker=key[0],
                )
                action = state.propose_current_add(action.bar_index)

        # 3. New entries.
        for key, action, priority_rank, priority_token in _rank_proposals(
            [
                item for item in proposals
                if item[1].action_type == "ENTRY"
            ],
            seed=config.entry_priority_seed,
        ):
            state = states[key]

            shares_per_unit = unit_shares(
                equity=realized_equity,
                n=action.n,
                risk_fraction=config.risk_fraction_per_unit,
            )
            if shares_per_unit < 1:
                skipped.append(
                    TurtlePortfolioSkip(
                        ticker=key[0],
                        system=system.name,
                        side=action.side,
                        entry_date=action.session_date,
                        exit_date="",
                        requested_units=1,
                        reason="UNIT_SIZE_BELOW_ONE_SHARE",
                        priority_seed=config.entry_priority_seed,
                        priority_rank=priority_rank,
                        priority_token=priority_token,
                    )
                )
                state.reject(action)
                continue

            reason = limit_reason(action.side)
            if reason is not None:
                skipped.append(
                    TurtlePortfolioSkip(
                        ticker=key[0],
                        system=system.name,
                        side=action.side,
                        entry_date=action.session_date,
                        exit_date="",
                        requested_units=1,
                        reason=reason,
                        priority_seed=config.entry_priority_seed,
                        priority_rank=priority_rank,
                        priority_token=priority_token,
                    )
                )
                state.reject(action)
                continue

            risk_dollars = realized_equity * config.risk_fraction_per_unit
            state.accept(action)
            active[key] = _LivePosition(
                ticker=key[0],
                segment_number=key[1],
                state=state,
                shares_per_unit=shares_per_unit,
                risk_dollars_per_unit=risk_dollars,
                realized_equity_at_entry=realized_equity,
                entry_priority_seed=config.entry_priority_seed,
                entry_priority_rank=priority_rank,
                entry_priority_token=priority_token,
            )
            append_curve(
                session_date=session_date,
                event_type="ENTRY",
                ticker=key[0],
            )

        # 4. Corporate-action-safe segment boundary / end of available segment.
        # Normal stop/channel exits above retain priority. Anything still open
        # is closed at the final adjusted close, exactly like standalone EOD.
        for key in sorted(
            segment_end_by_date.get(session_date, []),
            key=lambda item: (item[0], item[1]),
        ):
            if key not in active:
                continue
            closed = states[key].close_end_of_data()
            if closed is None:
                continue
            close_position(
                key=key,
                closed_trade=closed,
                session_date=session_date,
            )

        append_mtm(session_date)

    total_pnl = realized_equity - config.starting_equity
    total_return_pct = (
        (realized_equity / config.starting_equity) - 1.0
    ) * 100.0

    return TurtleChronologicalResult(
        starting_equity=config.starting_equity,
        ending_equity=realized_equity,
        total_pnl=total_pnl,
        total_return_pct=total_return_pct,
        max_realized_drawdown_pct=max_realized_drawdown,
        max_mtm_drawdown_pct=max_mtm_drawdown,
        worst_mtm_drawdown_date=worst_mtm_drawdown_date,
        accepted_trades=tuple(accepted),
        skipped_trades=tuple(skipped),
        equity_curve=tuple(curve),
        daily_mtm_curve=tuple(daily_mtm_curve),
        max_concurrent_positions=max_concurrent_positions,
        max_concurrent_units=max_concurrent_units,
    )
