from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from scanner.turtle import (
    DEFAULT_MAX_UNITS,
    SYSTEM_1,
    SYSTEM_2,
    TurtleSystem,
    bar_close,
)
from scanner.turtle_simulation import (
    TurtleTrade,
    TurtleUnit,
    _HypotheticalBreakout,
    _OpenTrade,
    _add_fill,
    _add_touched,
    _channel,
    _channel_exit_fill,
    _channel_exit_touched,
    _close_trade,
    _date,
    _entry_candidate,
    _hypothetical_breakout,
    _initial_stop,
    _make_open_trade,
    _next_add_price,
    _stop_fill,
    _stop_touched,
    _update_hypothetical,
    _wilder_n_values,
)


@dataclass(frozen=True)
class TurtleStateAction:
    action_type: str
    system: str
    side: str
    bar_index: int
    session_date: str
    price: float
    n: float
    unit_number: int = 0
    exit_reason: str = ""
    ambiguous_same_bar: bool = False


class TurtleMarketState:
    """
    One continuous ticker segment / one Turtle system.

    Market-rule state and portfolio-execution state are deliberately separated:
    - System 1 hypothetical winner/loser state advances from market bars.
    - Real units/stops change only after the portfolio accepts an action.
    """

    def __init__(
        self,
        bars: Sequence[Any],
        *,
        system: TurtleSystem,
    ) -> None:
        if system not in (SYSTEM_1, SYSTEM_2):
            raise ValueError("unsupported Turtle system")

        self.bars = bars
        self.system = system
        self.n_values = _wilder_n_values(bars)

        self.open_trade: _OpenTrade | None = None
        self.hypothetical: _HypotheticalBreakout | None = None
        self.last_breakout_was_winner = False

        self.completed_trades: list[TurtleTrade] = []
        self._last_index = -1

    @property
    def is_open(self) -> bool:
        return self.open_trade is not None

    @property
    def units(self) -> int:
        return 0 if self.open_trade is None else len(self.open_trade.units)

    @property
    def side(self) -> str:
        return "" if self.open_trade is None else self.open_trade.side

    @property
    def stop_price(self) -> float | None:
        return None if self.open_trade is None else self.open_trade.stop_price

    @property
    def accepted_units(self) -> tuple[TurtleUnit, ...]:
        if self.open_trade is None:
            return ()
        return tuple(self.open_trade.units)

    def _previous_n(self, index: int) -> float | None:
        if index <= 0:
            return None
        return self.n_values[index - 1]

    def _update_hypothetical(self, index: int) -> None:
        if self.system != SYSTEM_1 or self.hypothetical is None:
            return

        outcome = _update_hypothetical(
            hypothetical=self.hypothetical,
            bars=self.bars,
            index=index,
        )
        if outcome is not None:
            self.last_breakout_was_winner = outcome
            self.hypothetical = None

    def _start_hypothetical_if_flat(
        self,
        *,
        index: int,
        previous_n: float,
        twenty_candidate: tuple[str, float] | None,
    ) -> None:
        # Match the proven standalone simulator: a new Always-Trader
        # hypothetical is started from the flat-state 20D breakout logic.
        if (
            self.system == SYSTEM_1
            and self.hypothetical is None
            and twenty_candidate is not None
        ):
            self.hypothetical = _hypothetical_breakout(
                bars=self.bars,
                index=index,
                n=previous_n,
            )

    def propose(self, index: int) -> list[TurtleStateAction]:
        if index <= self._last_index:
            raise ValueError("bars must advance in strictly increasing order")
        if index < 0 or index >= len(self.bars):
            raise IndexError(index)

        self._last_index = index
        bar = self.bars[index]
        session_date = _date(bar)

        self._update_hypothetical(index)

        if self.open_trade is not None:
            trade = self.open_trade

            if index == trade.entry_index:
                return []

            exit_channel = _channel(
                self.bars,
                index=index,
                lookback=trade.system.exit_lookback,
            )
            exit_level: float | None = None
            if exit_channel is not None:
                upper, lower = exit_channel
                exit_level = lower if trade.side == "LONG" else upper

            stop_hit = _stop_touched(
                bar,
                side=trade.side,
                stop_price=trade.stop_price,
            )
            exit_hit = (
                exit_level is not None
                and _channel_exit_touched(
                    bar,
                    side=trade.side,
                    exit_level=exit_level,
                )
            )

            next_add = _next_add_price(trade)
            add_hit = (
                next_add is not None
                and _add_touched(
                    bar,
                    side=trade.side,
                    price=next_add,
                )
            )

            ambiguous = stop_hit and (exit_hit or add_hit)
            if ambiguous:
                trade.ambiguous_bars += 1

            if stop_hit:
                return [
                    TurtleStateAction(
                        action_type="EXIT",
                        system=trade.system.name,
                        side=trade.side,
                        bar_index=index,
                        session_date=session_date,
                        price=_stop_fill(
                            bar,
                            side=trade.side,
                            stop_price=trade.stop_price,
                        ),
                        n=trade.units[-1].n,
                        exit_reason="STOP",
                        ambiguous_same_bar=ambiguous,
                    )
                ]

            if exit_hit:
                return [
                    TurtleStateAction(
                        action_type="EXIT",
                        system=trade.system.name,
                        side=trade.side,
                        bar_index=index,
                        session_date=session_date,
                        price=_channel_exit_fill(
                            bar,
                            side=trade.side,
                            exit_level=float(exit_level),
                        ),
                        n=trade.units[-1].n,
                        exit_reason="CHANNEL_EXIT",
                    )
                ]

            add = self.propose_current_add(index)
            return [] if add is None else [add]

        previous_n = self._previous_n(index)
        if previous_n is None or previous_n <= 0:
            return []

        if self.system == SYSTEM_2:
            candidate = _entry_candidate(
                bars=self.bars,
                index=index,
                lookback=55,
            )
        else:
            twenty_candidate = _entry_candidate(
                bars=self.bars,
                index=index,
                lookback=20,
            )
            self._start_hypothetical_if_flat(
                index=index,
                previous_n=previous_n,
                twenty_candidate=twenty_candidate,
            )

            if twenty_candidate is None:
                return []

            if not self.last_breakout_was_winner:
                candidate = twenty_candidate
            else:
                candidate = _entry_candidate(
                    bars=self.bars,
                    index=index,
                    lookback=55,
                )

        if candidate is None:
            return []

        side, breakout_level = candidate
        preview = _make_open_trade(
            system=self.system,
            side=side,
            index=index,
            bar=bar,
            breakout_level=breakout_level,
            n=previous_n,
        )

        return [
            TurtleStateAction(
                action_type="ENTRY",
                system=self.system.name,
                side=side,
                bar_index=index,
                session_date=session_date,
                price=preview.initial_entry_price,
                n=previous_n,
                unit_number=1,
            )
        ]

    def propose_current_add(self, index: int) -> TurtleStateAction | None:
        """
        Re-evaluate the next 0.5N level on the already-advanced current bar.

        This allows multiple accepted pyramid levels on one daily bar, matching
        the standalone simulator, without advancing the market state twice.
        """
        if index != self._last_index:
            raise ValueError("current-bar add requires the already-advanced bar")
        if self.open_trade is None:
            return None
        if index == self.open_trade.entry_index:
            return None
        if len(self.open_trade.units) >= DEFAULT_MAX_UNITS:
            return None

        next_price = _next_add_price(self.open_trade)
        if next_price is None:
            return None

        bar = self.bars[index]
        if not _add_touched(
            bar,
            side=self.open_trade.side,
            price=next_price,
        ):
            return None

        return TurtleStateAction(
            action_type="ADD",
            system=self.system.name,
            side=self.open_trade.side,
            bar_index=index,
            session_date=_date(bar),
            price=_add_fill(
                bar,
                side=self.open_trade.side,
                price=next_price,
            ),
            n=self.open_trade.units[-1].n,
            unit_number=len(self.open_trade.units) + 1,
        )

    def accept(self, action: TurtleStateAction) -> TurtleTrade | None:
        if action.system != self.system.name:
            raise ValueError("action belongs to another Turtle system")

        bar = self.bars[action.bar_index]

        if action.action_type == "ENTRY":
            if self.open_trade is not None:
                raise RuntimeError("cannot enter while trade is already open")

            previous_n = self._previous_n(action.bar_index)
            if previous_n is None or previous_n <= 0:
                raise RuntimeError("entry has no valid previous N")

            if self.system == SYSTEM_2:
                lookback = 55
            elif self.last_breakout_was_winner:
                lookback = 55
            else:
                lookback = 20

            candidate = _entry_candidate(
                bars=self.bars,
                index=action.bar_index,
                lookback=lookback,
            )
            if candidate is None:
                raise RuntimeError("accepted entry is no longer executable")

            side, breakout_level = candidate
            self.open_trade = _make_open_trade(
                system=self.system,
                side=side,
                index=action.bar_index,
                bar=bar,
                breakout_level=breakout_level,
                n=previous_n,
            )
            return None

        if self.open_trade is None:
            raise RuntimeError("ADD/EXIT requires an open trade")

        trade = self.open_trade

        if action.action_type == "ADD":
            next_price = _next_add_price(trade)
            if next_price is None:
                raise RuntimeError("trade has no remaining pyramid level")
            if not _add_touched(
                bar,
                side=trade.side,
                price=next_price,
            ):
                raise RuntimeError("accepted add is not currently executable")

            fill = _add_fill(
                bar,
                side=trade.side,
                price=next_price,
            )
            unit_n = trade.units[-1].n
            trade.units.append(
                TurtleUnit(
                    entry_price=fill,
                    n=unit_n,
                    entry_index=action.bar_index,
                    entry_date=_date(bar),
                )
            )
            trade.stop_price = _initial_stop(
                fill,
                unit_n,
                side=trade.side,
            )
            return None

        if action.action_type == "EXIT":
            closed = _close_trade(
                trade=trade,
                exit_index=action.bar_index,
                exit_bar=bar,
                exit_price=action.price,
                exit_reason=action.exit_reason,
            )
            self.completed_trades.append(closed)
            self.open_trade = None
            return closed

        raise ValueError(f"unsupported action type: {action.action_type}")

    def reject(self, action: TurtleStateAction) -> None:
        if action.system != self.system.name:
            raise ValueError("action belongs to another Turtle system")
        # Deliberate no-op. Rejected real actions never mutate trade state.

    def close_end_of_data(self) -> TurtleTrade | None:
        if self.open_trade is None or not self.bars:
            return None

        final_index = len(self.bars) - 1
        final_bar = self.bars[final_index]
        closed = _close_trade(
            trade=self.open_trade,
            exit_index=final_index,
            exit_bar=final_bar,
            exit_price=bar_close(final_bar),
            exit_reason="END_OF_DATA",
        )
        self.completed_trades.append(closed)
        self.open_trade = None
        return closed
