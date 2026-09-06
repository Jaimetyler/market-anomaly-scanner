from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


# =============================================================================
# TURTLE TRADING
# =============================================================================
#
# Canonical daily Turtle building blocks.
#
# System 1:
#   Entry: previous 20-session breakout
#   Exit:  previous 10-session opposite breakout
#   Skip rule: yes -- requires trade-history state and is deliberately handled
#              by the future execution/simulation layer, not this signal layer.
#
# System 2:
#   Entry: previous 55-session breakout
#   Exit:  previous 20-session opposite breakout
#   Skip rule: no
#
# Shared:
#   N:          20-session Wilder-smoothed True Range
#   Stop:       2N
#   Pyramid:    add every 0.5N
#   Max units:  4
#
# Important:
#   Original Turtle entries occurred when the breakout was exceeded intraday.
#   If the market gapped through the breakout, entry occurred at the open.
#
# This module therefore uses daily OHLC data to detect whether the threshold
# was touched and produces a gap-aware theoretical fill.
#
# Daily OHLC cannot determine intraday event ordering when multiple thresholds
# are crossed during the same session. The later trade simulator must treat
# those cases conservatively.
# =============================================================================


DEFAULT_ATR_PERIOD = 20
DEFAULT_STOP_N = 2.0
DEFAULT_PYRAMID_STEP_N = 0.5
DEFAULT_MAX_UNITS = 4


@dataclass(frozen=True)
class TurtleSystem:
    name: str
    entry_lookback: int
    exit_lookback: int
    uses_skip_rule: bool


SYSTEM_1 = TurtleSystem(
    name="SYSTEM_1",
    entry_lookback=20,
    exit_lookback=10,
    uses_skip_rule=True,
)

SYSTEM_2 = TurtleSystem(
    name="SYSTEM_2",
    entry_lookback=55,
    exit_lookback=20,
    uses_skip_rule=False,
)


@dataclass(frozen=True)
class TurtleSignal:
    system: str
    side: str
    bar_index: int
    session_date: str | None

    breakout_level: float
    entry_price: float

    n: float
    stop_price: float

    entry_lookback: int
    exit_lookback: int

    pyramid_levels: tuple[float, ...]
    max_units: int

    requires_skip_rule_evaluation: bool
    ambiguous_same_bar: bool


def _value(
    bar: Any,
    *names: str,
) -> Any:
    if isinstance(bar, Mapping):
        for name in names:
            if name in bar:
                return bar[name]

    for name in names:
        if hasattr(bar, name):
            return getattr(bar, name)

    raise KeyError(
        f"Could not find any of fields {names!r} on bar {bar!r}"
    )


def _float_value(
    bar: Any,
    *names: str,
) -> float:
    value = _value(bar, *names)
    return float(value)


def _session_date(bar: Any) -> str | None:
    names = (
        "session_date",
        "date",
        "signal_date",
    )

    try:
        value = _value(bar, *names)
    except KeyError:
        return None

    if value is None:
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


def bar_open(bar: Any) -> float:
    return _float_value(bar, "o", "open")


def bar_high(bar: Any) -> float:
    return _float_value(bar, "h", "high")


def bar_low(bar: Any) -> float:
    return _float_value(bar, "l", "low")


def bar_close(bar: Any) -> float:
    return _float_value(bar, "c", "close")


def true_range(
    bar: Any,
    previous_close: float | None,
) -> float:
    """
    Standard True Range.

    On the first available bar, where no previous close exists, high-low is
    used.
    """
    high = bar_high(bar)
    low = bar_low(bar)

    if previous_close is None:
        return high - low

    return max(
        high - low,
        abs(high - previous_close),
        abs(low - previous_close),
    )


def true_ranges(
    bars: Sequence[Any],
) -> list[float]:
    result: list[float] = []
    previous_close: float | None = None

    for bar in bars:
        result.append(
            true_range(
                bar,
                previous_close,
            )
        )
        previous_close = bar_close(bar)

    return result


def wilder_n(
    bars: Sequence[Any],
    *,
    period: int = DEFAULT_ATR_PERIOD,
) -> list[float | None]:
    """
    Calculate Turtle N using Wilder smoothing.

    The first N is the arithmetic mean of the first `period` True Range
    observations.

    Thereafter:

        N_today = ((period - 1) * N_yesterday + TR_today) / period

    Returns one value per input bar. Values before the initial warmup are None.
    """
    if period <= 0:
        raise ValueError("period must be positive")

    ranges = true_ranges(bars)

    result: list[float | None] = [
        None
        for _ in bars
    ]

    if len(ranges) < period:
        return result

    initial = sum(
        ranges[:period]
    ) / period

    result[period - 1] = initial
    previous_n = initial

    for index in range(
        period,
        len(ranges),
    ):
        current = (
            ((period - 1) * previous_n)
            + ranges[index]
        ) / period

        result[index] = current
        previous_n = current

    return result


def previous_channel(
    bars: Sequence[Any],
    *,
    index: int,
    lookback: int,
) -> tuple[float, float]:
    """
    Return the highest high / lowest low of the sessions immediately preceding
    `index`.

    The current bar is NEVER included.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")

    if index < lookback:
        raise ValueError(
            f"Need at least {lookback} prior bars for index {index}"
        )

    previous = bars[
        index - lookback:index
    ]

    high = max(
        bar_high(bar)
        for bar in previous
    )

    low = min(
        bar_low(bar)
        for bar in previous
    )

    return high, low


def turtle_exit_level(
    bars: Sequence[Any],
    *,
    index: int,
    side: str,
    lookback: int,
) -> float:
    """
    Current opposite-channel exit threshold using only information available
    before the current bar.

    Long:
        previous lookback-session LOW.

    Short:
        previous lookback-session HIGH.
    """
    high, low = previous_channel(
        bars,
        index=index,
        lookback=lookback,
    )

    side_normalized = side.upper()

    if side_normalized == "LONG":
        return low

    if side_normalized == "SHORT":
        return high

    raise ValueError(
        f"Unknown side: {side!r}"
    )


def theoretical_breakout_fill(
    *,
    side: str,
    session_open: float,
    breakout_level: float,
) -> float:
    """
    Gap-aware daily approximation of the original Turtle entry mechanics.

    Long:
        If the market opens above the breakout, fill at the open.
        Otherwise fill at the breakout level.

    Short:
        If the market opens below the breakout, fill at the open.
        Otherwise fill at the breakout level.

    Tick-size/slippage modeling belongs in the execution layer.
    """
    side_normalized = side.upper()

    if side_normalized == "LONG":
        return max(
            session_open,
            breakout_level,
        )

    if side_normalized == "SHORT":
        return min(
            session_open,
            breakout_level,
        )

    raise ValueError(
        f"Unknown side: {side!r}"
    )


def initial_stop_price(
    *,
    side: str,
    entry_price: float,
    n: float,
    stop_n: float = DEFAULT_STOP_N,
) -> float:
    if entry_price <= 0:
        raise ValueError(
            "entry_price must be positive"
        )

    if n <= 0:
        raise ValueError(
            "n must be positive"
        )

    if stop_n <= 0:
        raise ValueError(
            "stop_n must be positive"
        )

    distance = stop_n * n

    side_normalized = side.upper()

    if side_normalized == "LONG":
        return entry_price - distance

    if side_normalized == "SHORT":
        return entry_price + distance

    raise ValueError(
        f"Unknown side: {side!r}"
    )


def pyramid_levels(
    *,
    side: str,
    entry_price: float,
    n: float,
    step_n: float = DEFAULT_PYRAMID_STEP_N,
    max_units: int = DEFAULT_MAX_UNITS,
) -> tuple[float, ...]:
    """
    Return prices for add-on units.

    The initial entry is unit #1, so a four-unit maximum produces three
    additional levels.
    """
    if max_units < 1:
        raise ValueError(
            "max_units must be at least 1"
        )

    if step_n <= 0:
        raise ValueError(
            "step_n must be positive"
        )

    if n <= 0:
        raise ValueError(
            "n must be positive"
        )

    side_normalized = side.upper()

    if side_normalized == "LONG":
        direction = 1.0
    elif side_normalized == "SHORT":
        direction = -1.0
    else:
        raise ValueError(
            f"Unknown side: {side!r}"
        )

    levels = []

    for add_number in range(
        1,
        max_units,
    ):
        levels.append(
            entry_price
            + (
                direction
                * add_number
                * step_n
                * n
            )
        )

    return tuple(levels)


def turtle_breakout_signals_at_index(
    bars: Sequence[Any],
    *,
    index: int,
    system: TurtleSystem,
    n_values: Sequence[float | None] | None = None,
    atr_period: int = DEFAULT_ATR_PERIOD,
    stop_n: float = DEFAULT_STOP_N,
    pyramid_step_n: float = DEFAULT_PYRAMID_STEP_N,
    max_units: int = DEFAULT_MAX_UNITS,
) -> list[TurtleSignal]:
    """
    Detect raw Turtle breakout signals for one session.

    This intentionally does NOT resolve the System 1 previous-winner skip rule.
    That rule requires state from prior hypothetical breakout outcomes and
    belongs in the trade-state simulator.

    A sufficiently wide daily bar can cross both long and short entry channels.
    Since OHLC data cannot reveal which occurred first, both signals are
    returned and marked ambiguous_same_bar=True.
    """
    if index <= 0:
        return []

    if index < system.entry_lookback:
        return []

    if n_values is None:
        n_values = wilder_n(
            bars,
            period=atr_period,
        )

    # Entry happens during the current session. Use volatility information
    # known before that session, avoiding lookahead from the current day's
    # range.
    n = n_values[index - 1]

    if n is None or n <= 0:
        return []

    channel_high, channel_low = previous_channel(
        bars,
        index=index,
        lookback=system.entry_lookback,
    )

    current = bars[index]

    current_open = bar_open(current)
    current_high = bar_high(current)
    current_low = bar_low(current)

    long_breakout = (
        current_high > channel_high
    )

    short_breakout = (
        current_low < channel_low
    )

    ambiguous = (
        long_breakout
        and short_breakout
    )

    signals: list[TurtleSignal] = []

    if long_breakout:
        fill = theoretical_breakout_fill(
            side="LONG",
            session_open=current_open,
            breakout_level=channel_high,
        )

        signals.append(
            TurtleSignal(
                system=system.name,
                side="LONG",
                bar_index=index,
                session_date=_session_date(
                    current
                ),
                breakout_level=channel_high,
                entry_price=fill,
                n=n,
                stop_price=initial_stop_price(
                    side="LONG",
                    entry_price=fill,
                    n=n,
                    stop_n=stop_n,
                ),
                entry_lookback=system.entry_lookback,
                exit_lookback=system.exit_lookback,
                pyramid_levels=pyramid_levels(
                    side="LONG",
                    entry_price=fill,
                    n=n,
                    step_n=pyramid_step_n,
                    max_units=max_units,
                ),
                max_units=max_units,
                requires_skip_rule_evaluation=(
                    system.uses_skip_rule
                ),
                ambiguous_same_bar=ambiguous,
            )
        )

    if short_breakout:
        fill = theoretical_breakout_fill(
            side="SHORT",
            session_open=current_open,
            breakout_level=channel_low,
        )

        signals.append(
            TurtleSignal(
                system=system.name,
                side="SHORT",
                bar_index=index,
                session_date=_session_date(
                    current
                ),
                breakout_level=channel_low,
                entry_price=fill,
                n=n,
                stop_price=initial_stop_price(
                    side="SHORT",
                    entry_price=fill,
                    n=n,
                    stop_n=stop_n,
                ),
                entry_lookback=system.entry_lookback,
                exit_lookback=system.exit_lookback,
                pyramid_levels=pyramid_levels(
                    side="SHORT",
                    entry_price=fill,
                    n=n,
                    step_n=pyramid_step_n,
                    max_units=max_units,
                ),
                max_units=max_units,
                requires_skip_rule_evaluation=(
                    system.uses_skip_rule
                ),
                ambiguous_same_bar=ambiguous,
            )
        )

    return signals


def scan_turtle_breakouts(
    bars: Sequence[Any],
    *,
    systems: Iterable[TurtleSystem] = (
        SYSTEM_1,
        SYSTEM_2,
    ),
    atr_period: int = DEFAULT_ATR_PERIOD,
    stop_n: float = DEFAULT_STOP_N,
    pyramid_step_n: float = DEFAULT_PYRAMID_STEP_N,
    max_units: int = DEFAULT_MAX_UNITS,
) -> list[TurtleSignal]:
    """
    Scan an entire bar history for raw Turtle breakout opportunities.

    System 1 results still require the historical previous-winner skip rule to
    be evaluated by the future stateful simulator.
    """
    systems = tuple(systems)

    if not systems:
        return []

    n_values = wilder_n(
        bars,
        period=atr_period,
    )

    signals: list[TurtleSignal] = []

    minimum_index = min(
        system.entry_lookback
        for system in systems
    )

    for index in range(
        minimum_index,
        len(bars),
    ):
        for system in systems:
            signals.extend(
                turtle_breakout_signals_at_index(
                    bars,
                    index=index,
                    system=system,
                    n_values=n_values,
                    atr_period=atr_period,
                    stop_n=stop_n,
                    pyramid_step_n=pyramid_step_n,
                    max_units=max_units,
                )
            )

    return signals