from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from scanner.turtle import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_MAX_UNITS,
    DEFAULT_PYRAMID_STEP_N,
    DEFAULT_STOP_N,
    SYSTEM_1,
    SYSTEM_2,
    TurtleSystem,
    bar_close,
    bar_high,
    bar_low,
    bar_open,
)


@dataclass(frozen=True)
class TurtleUnit:
    entry_price: float
    n: float
    entry_index: int = -1
    entry_date: str = ""


@dataclass(frozen=True)
class TurtleTrade:
    system: str
    side: str

    entry_index: int
    exit_index: int

    entry_date: str
    exit_date: str

    initial_entry_price: float
    average_entry_price: float
    exit_price: float

    units: int

    initial_n: float
    initial_stop_price: float
    final_stop_price: float

    exit_reason: str

    return_pct: float
    unit_weighted_return_pct: float

    bars_held: int
    ambiguous_bars: int

    # Actual unit execution timeline. Item 0 is the initial entry;
    # later items are 0.5N pyramid adds.
    unit_entries: tuple[TurtleUnit, ...] = ()


@dataclass
class _OpenTrade:
    system: TurtleSystem
    side: str

    entry_index: int
    entry_date: str

    initial_entry_price: float
    initial_n: float
    initial_stop_price: float

    units: list[TurtleUnit]

    stop_price: float
    ambiguous_bars: int = 0


@dataclass
class _HypotheticalBreakout:
    """
    The System 1 "Always Trader".

    This hypothetical trader exists only to determine whether the most
    recent 20-day breakout was a winner or loser.

    The original rule considers a breakout a loser if price moves 2N
    against the position before a profitable 10-day exit occurs.
    """

    side: str
    entry_index: int
    entry_price: float
    n: float
    stop_price: float


def _date(bar: Any) -> str:
    if isinstance(bar, dict):
        for key in (
            "session_date",
            "date",
            "signal_date",
        ):
            value = bar.get(key)

            if value is not None:
                if hasattr(value, "isoformat"):
                    return value.isoformat()

                return str(value)

        return ""

    for key in (
        "session_date",
        "date",
        "signal_date",
    ):
        if hasattr(bar, key):
            value = getattr(
                bar,
                key,
            )

            if value is not None:
                if hasattr(value, "isoformat"):
                    return value.isoformat()

                return str(value)

    return ""


def _true_range(
    current: Any,
    previous: Any | None,
) -> float:
    high = bar_high(current)
    low = bar_low(current)

    if previous is None:
        return high - low

    previous_close = bar_close(
        previous
    )

    return max(
        high - low,
        abs(
            high
            - previous_close
        ),
        abs(
            low
            - previous_close
        ),
    )


def _wilder_n_values(
    bars: Sequence[Any],
    *,
    period: int = DEFAULT_ATR_PERIOD,
) -> list[float | None]:
    """
    Classic Wilder-smoothed True Range.

    N for today's entry decision uses yesterday's completed N.
    """

    if period < 1:
        raise ValueError(
            "period must be >= 1"
        )

    n_values: list[
        float | None
    ] = [
        None
        for _ in bars
    ]

    if len(bars) < period:
        return n_values

    true_ranges: list[
        float
    ] = []

    for index, bar in enumerate(
        bars
    ):
        previous = (
            bars[index - 1]
            if index > 0
            else None
        )

        true_ranges.append(
            _true_range(
                bar,
                previous,
            )
        )

    initial = sum(
        true_ranges[:period]
    ) / period

    n_values[
        period - 1
    ] = initial

    previous_n = initial

    for index in range(
        period,
        len(bars),
    ):
        current_n = (
            (
                (period - 1)
                * previous_n
            )
            + true_ranges[index]
        ) / period

        n_values[
            index
        ] = current_n

        previous_n = current_n

    return n_values


def _channel(
    bars: Sequence[Any],
    *,
    index: int,
    lookback: int,
) -> tuple[
    float,
    float,
] | None:
    """
    Highest high / lowest low of PREVIOUS lookback bars.

    Current bar is never included.
    """

    start = (
        index
        - lookback
    )

    if start < 0:
        return None

    previous_bars = bars[
        start:index
    ]

    if len(
        previous_bars
    ) != lookback:
        return None

    high = max(
        bar_high(bar)
        for bar in previous_bars
    )

    low = min(
        bar_low(bar)
        for bar in previous_bars
    )

    return (
        high,
        low,
    )


def _breakout_side(
    bar: Any,
    *,
    upper: float,
    lower: float,
) -> str | None:
    """
    Original rules require price to exceed the channel.

    If daily OHLC says BOTH sides were exceeded, sequence is unknowable.
    We return None instead of inventing an intraday ordering.
    """

    broke_long = (
        bar_high(bar)
        > upper
    )

    broke_short = (
        bar_low(bar)
        < lower
    )

    if (
        broke_long
        and broke_short
    ):
        return None

    if broke_long:
        return "LONG"

    if broke_short:
        return "SHORT"

    return None


def _entry_fill(
    bar: Any,
    *,
    side: str,
    breakout_level: float,
) -> float:
    open_price = bar_open(
        bar
    )

    if side == "LONG":
        return max(
            open_price,
            breakout_level,
        )

    if side == "SHORT":
        return min(
            open_price,
            breakout_level,
        )

    raise ValueError(
        f"unsupported side: {side}"
    )


def _initial_stop(
    entry_price: float,
    n: float,
    *,
    side: str,
) -> float:
    if side == "LONG":
        return (
            entry_price
            - DEFAULT_STOP_N
            * n
        )

    if side == "SHORT":
        return (
            entry_price
            + DEFAULT_STOP_N
            * n
        )

    raise ValueError(
        f"unsupported side: {side}"
    )


def _average_entry(
    units: Sequence[
        TurtleUnit
    ],
) -> float:
    if not units:
        raise ValueError(
            "trade must contain units"
        )

    return sum(
        unit.entry_price
        for unit in units
    ) / len(units)


def _trade_return_pct(
    *,
    side: str,
    entry_price: float,
    exit_price: float,
) -> float:
    if side == "LONG":
        return (
            exit_price
            / entry_price
            - 1.0
        ) * 100.0

    if side == "SHORT":
        return (
            entry_price
            / exit_price
            - 1.0
        ) * 100.0

    raise ValueError(
        f"unsupported side: {side}"
    )


def _weighted_return_pct(
    *,
    side: str,
    units: Sequence[
        TurtleUnit
    ],
    exit_price: float,
) -> float:
    returns = [
        _trade_return_pct(
            side=side,
            entry_price=(
                unit.entry_price
            ),
            exit_price=(
                exit_price
            ),
        )
        for unit in units
    ]

    return sum(
        returns
    ) / len(returns)


def _stop_touched(
    bar: Any,
    *,
    side: str,
    stop_price: float,
) -> bool:
    if side == "LONG":
        return (
            bar_low(bar)
            <= stop_price
        )

    return (
        bar_high(bar)
        >= stop_price
    )


def _stop_fill(
    bar: Any,
    *,
    side: str,
    stop_price: float,
) -> float:
    """
    Gap-aware stop fill.
    """

    open_price = bar_open(
        bar
    )

    if side == "LONG":
        if (
            open_price
            < stop_price
        ):
            return open_price

        return stop_price

    if (
        open_price
        > stop_price
    ):
        return open_price

    return stop_price


def _channel_exit_touched(
    bar: Any,
    *,
    side: str,
    exit_level: float,
) -> bool:
    if side == "LONG":
        return (
            bar_low(bar)
            < exit_level
        )

    return (
        bar_high(bar)
        > exit_level
    )


def _channel_exit_fill(
    bar: Any,
    *,
    side: str,
    exit_level: float,
) -> float:
    open_price = bar_open(
        bar
    )

    if side == "LONG":
        return min(
            open_price,
            exit_level,
        )

    return max(
        open_price,
        exit_level,
    )


def _next_add_price(
    trade: _OpenTrade,
) -> float | None:
    if (
        len(trade.units)
        >= DEFAULT_MAX_UNITS
    ):
        return None

    last_unit = (
        trade.units[-1]
    )

    step = (
        DEFAULT_PYRAMID_STEP_N
        * last_unit.n
    )

    if trade.side == "LONG":
        return (
            last_unit.entry_price
            + step
        )

    return (
        last_unit.entry_price
        - step
    )


def _add_touched(
    bar: Any,
    *,
    side: str,
    price: float,
) -> bool:
    if side == "LONG":
        return (
            bar_high(bar)
            >= price
        )

    return (
        bar_low(bar)
        <= price
    )


def _add_fill(
    bar: Any,
    *,
    side: str,
    price: float,
) -> float:
    open_price = bar_open(
        bar
    )

    if side == "LONG":
        return max(
            open_price,
            price,
        )

    return min(
        open_price,
        price,
    )


def _make_open_trade(
    *,
    system: TurtleSystem,
    side: str,
    index: int,
    bar: Any,
    breakout_level: float,
    n: float,
) -> _OpenTrade:
    entry_price = _entry_fill(
        bar,
        side=side,
        breakout_level=(
            breakout_level
        ),
    )

    stop_price = _initial_stop(
        entry_price,
        n,
        side=side,
    )

    return _OpenTrade(
        system=system,
        side=side,
        entry_index=index,
        entry_date=_date(bar),
        initial_entry_price=(
            entry_price
        ),
        initial_n=n,
        initial_stop_price=(
            stop_price
        ),
        units=[
            TurtleUnit(
                entry_price=(
                    entry_price
                ),
                n=n,
                entry_index=index,
                entry_date=_date(bar),
            )
        ],
        stop_price=stop_price,
    )


def _close_trade(
    *,
    trade: _OpenTrade,
    exit_index: int,
    exit_bar: Any,
    exit_price: float,
    exit_reason: str,
) -> TurtleTrade:
    average_entry = (
        _average_entry(
            trade.units
        )
    )

    return TurtleTrade(
        system=(
            trade.system.name
        ),
        side=trade.side,
        entry_index=(
            trade.entry_index
        ),
        exit_index=(
            exit_index
        ),
        entry_date=(
            trade.entry_date
        ),
        exit_date=_date(
            exit_bar
        ),
        initial_entry_price=(
            trade.initial_entry_price
        ),
        average_entry_price=(
            average_entry
        ),
        exit_price=(
            exit_price
        ),
        units=len(
            trade.units
        ),
        initial_n=(
            trade.initial_n
        ),
        initial_stop_price=(
            trade.initial_stop_price
        ),
        final_stop_price=(
            trade.stop_price
        ),
        exit_reason=(
            exit_reason
        ),
        return_pct=(
            _trade_return_pct(
                side=trade.side,
                entry_price=(
                    average_entry
                ),
                exit_price=(
                    exit_price
                ),
            )
        ),
        unit_weighted_return_pct=(
            _weighted_return_pct(
                side=trade.side,
                units=(
                    trade.units
                ),
                exit_price=(
                    exit_price
                ),
            )
        ),
        bars_held=(
            exit_index
            - trade.entry_index
            + 1
        ),
        ambiguous_bars=(
            trade.ambiguous_bars
        ),
        unit_entries=tuple(
            trade.units
        ),
    )


def _hypothetical_breakout(
    *,
    bars: Sequence[Any],
    index: int,
    n: float,
) -> _HypotheticalBreakout | None:
    channel = _channel(
        bars,
        index=index,
        lookback=20,
    )

    if channel is None:
        return None

    upper, lower = channel

    side = _breakout_side(
        bars[index],
        upper=upper,
        lower=lower,
    )

    if side is None:
        return None

    level = (
        upper
        if side == "LONG"
        else lower
    )

    entry_price = (
        _entry_fill(
            bars[index],
            side=side,
            breakout_level=(
                level
            ),
        )
    )

    return _HypotheticalBreakout(
        side=side,
        entry_index=index,
        entry_price=(
            entry_price
        ),
        n=n,
        stop_price=(
            _initial_stop(
                entry_price,
                n,
                side=side,
            )
        ),
    )


def _update_hypothetical(
    *,
    hypothetical: _HypotheticalBreakout,
    bars: Sequence[Any],
    index: int,
) -> bool | None:
    """
    Returns:
        False -> hypothetical breakout is a loser
        True  -> hypothetical breakout is a winner
        None  -> outcome not resolved yet

    IMPORTANT:
    We do NOT evaluate stop/exit on the breakout bar itself because daily
    OHLC cannot tell whether the opposing extreme happened before or after
    the breakout entry.

    This avoids falsely declaring a breakout loser from a low/high that may
    have occurred before the trade existed.
    """

    if (
        index
        <= hypothetical.entry_index
    ):
        return None

    bar = bars[index]

    # Original loser test:
    # 2N adverse movement before a profitable 10-day exit.
    if _stop_touched(
        bar,
        side=hypothetical.side,
        stop_price=(
            hypothetical.stop_price
        ),
    ):
        return False

    exit_channel = _channel(
        bars,
        index=index,
        lookback=10,
    )

    if exit_channel is None:
        return None

    upper, lower = (
        exit_channel
    )

    exit_level = (
        lower
        if hypothetical.side
        == "LONG"
        else upper
    )

    if not _channel_exit_touched(
        bar,
        side=(
            hypothetical.side
        ),
        exit_level=(
            exit_level
        ),
    ):
        return None

    exit_price = (
        _channel_exit_fill(
            bar,
            side=(
                hypothetical.side
            ),
            exit_level=(
                exit_level
            ),
        )
    )

    return (
        _trade_return_pct(
            side=(
                hypothetical.side
            ),
            entry_price=(
                hypothetical.entry_price
            ),
            exit_price=(
                exit_price
            ),
        )
        > 0.0
    )


def _entry_candidate(
    *,
    bars: Sequence[Any],
    index: int,
    lookback: int,
) -> tuple[
    str,
    float,
] | None:
    channel = _channel(
        bars,
        index=index,
        lookback=lookback,
    )

    if channel is None:
        return None

    upper, lower = channel

    side = _breakout_side(
        bars[index],
        upper=upper,
        lower=lower,
    )

    if side is None:
        return None

    return (
        side,
        (
            upper
            if side == "LONG"
            else lower
        ),
    )



# ---------------------------------------------------------------------------
# Backward-compatible private helper names used by the existing regression
# tests. These delegate to the rewritten canonical implementations above.
# ---------------------------------------------------------------------------

def _gap_aware_stop_fill(
    *,
    side: str,
    session_open: float,
    stop_price: float,
) -> float:
    """
    Backward-compatible gap-aware stop helper.

    LONG:
        gap below stop -> fill at open
        otherwise      -> fill at stop

    SHORT:
        gap above stop -> fill at open
        otherwise      -> fill at stop
    """
    if side == "LONG":
        return min(
            session_open,
            stop_price,
        )

    if side == "SHORT":
        return max(
            session_open,
            stop_price,
        )

    raise ValueError(
        f"unsupported side: {side}"
    )


def _gap_aware_exit_fill(
    *,
    side: str,
    session_open: float,
    exit_level: float,
) -> float:
    """
    Backward-compatible gap-aware channel-exit helper.
    """
    if side == "LONG":
        return min(
            session_open,
            exit_level,
        )

    if side == "SHORT":
        return max(
            session_open,
            exit_level,
        )

    raise ValueError(
        f"unsupported side: {side}"
    )


def simulate_turtle_system(
    bars: Sequence[Any],
    *,
    system: TurtleSystem,
) -> list[TurtleTrade]:
    """
    Stateful Turtle simulation.

    SYSTEM 1
    --------
    - 20-day breakout entry.
    - Ignore 20-day breakout if prior hypothetical 20-day breakout was winner.
    - Direction of previous breakout does not matter.
    - If skipped due to previous winner, enter at 55-day failsafe.
    - 10-day exit.

    SYSTEM 2
    --------
    - Take every 55-day breakout.
    - 20-day exit.

    Shared mechanics
    ----------------
    - previous completed day's N used for entry
    - 2N stop
    - add every 0.5N
    - max 4 units
    - gap-aware fills
    - daily-bar ambiguity handled conservatively
    """

    if system not in (
        SYSTEM_1,
        SYSTEM_2,
    ):
        raise ValueError(
            "unsupported Turtle system"
        )

    if not bars:
        return []

    n_values = (
        _wilder_n_values(
            bars
        )
    )

    trades: list[
        TurtleTrade
    ] = []

    open_trade: (
        _OpenTrade
        | None
    ) = None

    # System 1 "Always Trader" state.
    hypothetical: (
        _HypotheticalBreakout
        | None
    ) = None

    # No previous breakout means first valid 20-day breakout is permitted.
    last_breakout_was_winner = False

    for index, bar in enumerate(
        bars
    ):
        previous_n = (
            n_values[index - 1]
            if index > 0
            else None
        )

        # --------------------------------------------------------------
        # Update System 1 hypothetical breakout state.
        # --------------------------------------------------------------

        if (
            system == SYSTEM_1
            and hypothetical
            is not None
        ):
            outcome = (
                _update_hypothetical(
                    hypothetical=(
                        hypothetical
                    ),
                    bars=bars,
                    index=index,
                )
            )

            if outcome is not None:
                last_breakout_was_winner = (
                    outcome
                )

                hypothetical = None

        # --------------------------------------------------------------
        # Manage real open trade first.
        # --------------------------------------------------------------

        if open_trade is not None:
            # Do not evaluate stop/add/exit on the entry bar.
            #
            # Daily OHLC cannot establish whether the day's adverse
            # extreme happened before or after the breakout.
            if (
                index
                == open_trade.entry_index
            ):
                continue

            exit_channel = (
                _channel(
                    bars,
                    index=index,
                    lookback=(
                        open_trade
                        .system
                        .exit_lookback
                    ),
                )
            )

            exit_level = None

            if exit_channel:
                upper, lower = (
                    exit_channel
                )

                exit_level = (
                    lower
                    if open_trade.side
                    == "LONG"
                    else upper
                )

            stop_hit = (
                _stop_touched(
                    bar,
                    side=(
                        open_trade.side
                    ),
                    stop_price=(
                        open_trade
                        .stop_price
                    ),
                )
            )

            exit_hit = (
                exit_level
                is not None
                and _channel_exit_touched(
                    bar,
                    side=(
                        open_trade.side
                    ),
                    exit_level=(
                        exit_level
                    ),
                )
            )

            # Daily OHLC cannot tell us intraday ordering when the
            # active stop and another actionable level are both touched.
            #
            # Deterministic research rule:
            #     STOP WINS
            #
            # But preserve the ambiguity count so these trades remain
            # inspectable rather than silently pretending sequencing is known.
            next_add_price = (
                _next_add_price(
                    open_trade
                )
            )

            add_hit = (
                next_add_price is not None
                and _add_touched(
                    bar,
                    side=(
                        open_trade.side
                    ),
                    price=(
                        next_add_price
                    ),
                )
            )

            if (
                stop_hit
                and (
                    exit_hit
                    or add_hit
                )
            ):
                open_trade.ambiguous_bars += 1

            # Stop wins if multiple actionable levels are touched.
            if stop_hit:
                exit_price = (
                    _stop_fill(
                        bar,
                        side=(
                            open_trade.side
                        ),
                        stop_price=(
                            open_trade
                            .stop_price
                        ),
                    )
                )

                trades.append(
                    _close_trade(
                        trade=(
                            open_trade
                        ),
                        exit_index=index,
                        exit_bar=bar,
                        exit_price=(
                            exit_price
                        ),
                        exit_reason=(
                            "STOP"
                        ),
                    )
                )

                open_trade = None
                continue

            if exit_hit:
                exit_price = (
                    _channel_exit_fill(
                        bar,
                        side=(
                            open_trade.side
                        ),
                        exit_level=(
                            float(
                                exit_level
                            )
                        ),
                    )
                )

                trades.append(
                    _close_trade(
                        trade=(
                            open_trade
                        ),
                        exit_index=index,
                        exit_bar=bar,
                        exit_price=(
                            exit_price
                        ),
                        exit_reason=(
                            "CHANNEL_EXIT"
                        ),
                    )
                )

                open_trade = None
                continue

            # Pyramiding.
            #
            # Multiple levels can be crossed in one daily bar.
            while (
                len(
                    open_trade.units
                )
                < DEFAULT_MAX_UNITS
            ):
                next_price = (
                    _next_add_price(
                        open_trade
                    )
                )

                if next_price is None:
                    break

                if not _add_touched(
                    bar,
                    side=(
                        open_trade.side
                    ),
                    price=next_price,
                ):
                    break

                fill = (
                    _add_fill(
                        bar,
                        side=(
                            open_trade.side
                        ),
                        price=next_price,
                    )
                )

                unit_n = (
                    open_trade
                    .units[-1]
                    .n
                )

                open_trade.units.append(
                    TurtleUnit(
                        entry_price=fill,
                        n=unit_n,
                        entry_index=index,
                        entry_date=_date(bar),
                    )
                )

                # Original Turtle stop concept:
                # adding at +0.5N also raises/lowers protective stops.
                open_trade.stop_price = (
                    _initial_stop(
                        fill,
                        unit_n,
                        side=(
                            open_trade.side
                        ),
                    )
                )

            continue

        # --------------------------------------------------------------
        # Flat: no valid N means no trade.
        # --------------------------------------------------------------

        if (
            previous_n is None
            or previous_n <= 0
        ):
            continue

        # --------------------------------------------------------------
        # System 1 hypothetical 20-day breakout.
        #
        # Start the hypothetical "Always Trader" whenever its previous
        # hypothetical trade has resolved and a new 20D breakout occurs.
        # --------------------------------------------------------------

        twenty_candidate = None

        if system == SYSTEM_1:
            twenty_candidate = (
                _entry_candidate(
                    bars=bars,
                    index=index,
                    lookback=20,
                )
            )

            if (
                hypothetical
                is None
                and twenty_candidate
                is not None
            ):
                hypothetical = (
                    _hypothetical_breakout(
                        bars=bars,
                        index=index,
                        n=previous_n,
                    )
                )

        # --------------------------------------------------------------
        # Real entries.
        # --------------------------------------------------------------

        if system == SYSTEM_2:
            candidate = (
                _entry_candidate(
                    bars=bars,
                    index=index,
                    lookback=55,
                )
            )

            if candidate is None:
                continue

            side, level = (
                candidate
            )

            open_trade = (
                _make_open_trade(
                    system=system,
                    side=side,
                    index=index,
                    bar=bar,
                    breakout_level=(
                        level
                    ),
                    n=previous_n,
                )
            )

            continue

        # --------------------------------------------------------------
        # SYSTEM 1
        # --------------------------------------------------------------

        if twenty_candidate is None:
            continue

        twenty_side, twenty_level = (
            twenty_candidate
        )

        if not last_breakout_was_winner:
            open_trade = (
                _make_open_trade(
                    system=system,
                    side=twenty_side,
                    index=index,
                    bar=bar,
                    breakout_level=(
                        twenty_level
                    ),
                    n=previous_n,
                )
            )

            continue

        # Previous hypothetical breakout was a winner:
        # skip the 20-day entry and use the 55-day failsafe instead.
        failsafe = (
            _entry_candidate(
                bars=bars,
                index=index,
                lookback=55,
            )
        )

        if failsafe is None:
            continue

        side, level = (
            failsafe
        )

        open_trade = (
            _make_open_trade(
                system=system,
                side=side,
                index=index,
                bar=bar,
                breakout_level=(
                    level
                ),
                n=previous_n,
            )
        )

    # --------------------------------------------------------------
    # Close anything still open at final close.
    # --------------------------------------------------------------

    if open_trade is not None:
        final_index = (
            len(bars)
            - 1
        )

        final_bar = bars[
            final_index
        ]

        exit_price = (
            bar_close(
                final_bar
            )
        )

        trades.append(
            _close_trade(
                trade=open_trade,
                exit_index=(
                    final_index
                ),
                exit_bar=(
                    final_bar
                ),
                exit_price=(
                    exit_price
                ),
                exit_reason=(
                    "END_OF_DATA"
                ),
            )
        )

    return trades


def simulate_turtle_system_1(
    bars: Sequence[Any],
) -> list[TurtleTrade]:
    return (
        simulate_turtle_system(
            bars,
            system=SYSTEM_1,
        )
    )


def simulate_turtle_system_2(
    bars: Sequence[Any],
) -> list[TurtleTrade]:
    return (
        simulate_turtle_system(
            bars,
            system=SYSTEM_2,
        )
    )
