from __future__ import annotations

from scanner.turtle import SYSTEM_2
from scanner.turtle_simulation import simulate_turtle_system


def _bar(date: str, o: float, h: float, l: float, c: float) -> dict[str, object]:
    return {
        "session_date": date,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
    }


def _pyramid_fixture() -> list[dict[str, object]]:
    bars: list[dict[str, object]] = []
    for i in range(60):
        bars.append(_bar(f"2026-01-{i + 1:02d}", 100.0, 101.0, 99.0, 100.0))

    bars.append(_bar("2026-03-02", 101.0, 102.0, 100.5, 101.5))
    bars.append(_bar("2026-03-03", 102.0, 103.2, 101.5, 103.0))
    bars.append(_bar("2026-03-04", 103.0, 104.4, 102.8, 104.0))
    bars.append(_bar("2026-03-05", 104.0, 105.6, 103.8, 105.0))
    bars.append(_bar("2026-03-06", 105.0, 105.2, 104.5, 105.0))
    return bars


def test_trade_exposes_actual_unit_entry_timeline():
    trades = simulate_turtle_system(_pyramid_fixture(), system=SYSTEM_2)
    assert trades

    trade = trades[0]
    assert trade.units == len(trade.unit_entries)
    assert trade.unit_entries[0].entry_index == trade.entry_index
    assert trade.unit_entries[0].entry_date == trade.entry_date

    indices = [unit.entry_index for unit in trade.unit_entries]
    assert indices == sorted(indices)
    assert all(unit.entry_index >= trade.entry_index for unit in trade.unit_entries)
    assert all(unit.entry_index <= trade.exit_index for unit in trade.unit_entries)


def test_pyramid_adds_are_timestamped_after_initial_entry():
    trades = simulate_turtle_system(_pyramid_fixture(), system=SYSTEM_2)
    assert trades

    trade = trades[0]
    assert trade.units >= 2
    assert len(trade.unit_entries) == trade.units

    initial = trade.unit_entries[0]
    adds = trade.unit_entries[1:]

    assert initial.entry_index == trade.entry_index
    assert initial.entry_date == trade.entry_date
    assert all(add.entry_index > trade.entry_index for add in adds)
    assert all(add.entry_date != trade.entry_date for add in adds)
