from __future__ import annotations

import pytest

from scanner.turtle import SYSTEM_2
from scanner.turtle_stateful import TurtleMarketState


def _bar(
    date: str,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> dict[str, float | str]:
    return {
        "session_date": date,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }


def _breakout_bars() -> list[dict[str, float | str]]:
    # 60 quiet bars make N deterministic enough and establish a 55D channel.
    bars = []
    for i in range(60):
        bars.append(
            _bar(
                f"2024-01-{i + 1:02d}",
                100.0,
                101.0,
                99.0,
                100.0,
            )
        )

    # Long breakout.
    bars.append(
        _bar("2024-03-01", 101.0, 103.0, 100.0, 102.0)
    )
    # Touch first 0.5N add.
    bars.append(
        _bar("2024-03-02", 102.0, 104.0, 101.0, 103.0)
    )
    # Touch it again later.
    bars.append(
        _bar("2024-03-03", 102.0, 104.0, 101.0, 103.0)
    )
    return bars


def test_rejected_add_does_not_change_trade_state():
    bars = _breakout_bars()
    state = TurtleMarketState(bars, system=SYSTEM_2)

    entry = state.propose(60)[0]
    assert entry.action_type == "ENTRY"
    state.accept(entry)

    before_units = state.units
    before_stop = state.stop_price

    add = state.propose(61)[0]
    assert add.action_type == "ADD"

    state.reject(add)

    assert state.units == before_units
    assert state.stop_price == pytest.approx(before_stop)


def test_rejected_add_can_be_offered_again_on_later_bar():
    bars = _breakout_bars()
    state = TurtleMarketState(bars, system=SYSTEM_2)

    state.accept(state.propose(60)[0])

    first_add = state.propose(61)[0]
    state.reject(first_add)

    second_add = state.propose(62)[0]

    assert second_add.action_type == "ADD"
    assert second_add.unit_number == 2
    assert second_add.price == pytest.approx(first_add.price)


def test_accepted_add_changes_stop_and_next_unit_number():
    bars = _breakout_bars()
    state = TurtleMarketState(bars, system=SYSTEM_2)

    state.accept(state.propose(60)[0])
    old_stop = state.stop_price

    add = state.propose(61)[0]
    state.accept(add)

    assert state.units == 2
    assert state.stop_price != pytest.approx(old_stop)

    # The next bar may offer another add; if it does, it must be unit 3,
    # proving the accepted action—not a precomputed timeline—controls state.
    actions = state.propose(62)
    if actions:
        assert actions[0].action_type == "ADD"
        assert actions[0].unit_number == 3
