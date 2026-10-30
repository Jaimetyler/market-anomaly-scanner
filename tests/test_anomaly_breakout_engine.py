from experiments.anomaly_breakout.engine import Bar, Signal, detect_signals, simulate
import pytest


def test_next_open_and_no_lookahead():
    bars = []
    for i in range(22):
        high = 18 if i < 15 else 12
        low = 6 if i < 15 else 9
        close = 10
        volume = 100
        if i == 20:
            high, low, close, volume = 25, 10, 24, 200
        bars.append(Bar(f"2020-01-{i+1:02d}", close, high, low, close, volume))
    # Last bar is a future entry. Its high/close must not change the signal.
    bars[-1] = Bar("2020-01-22", 24, 100, 1, 100, 100)
    # Fix artificial high/low validity on the future bar.
    signals = detect_signals("XYZ", bars, {"2020-01-19"}, {"2020-01-21": True})
    assert len(signals) == 1
    assert signals[0].entry_date == "2020-01-22"
    assert signals[0].anomaly is True


def test_gap_stop_is_worse_than_stop():
    bars = [
        Bar("2020-01-01", 100, 105, 96, 100, 100),
        Bar("2020-01-02", 90, 92, 85, 88, 100),
    ]
    signal = Signal("XYZ", "2019-12-31", "2020-01-01", 95, True)
    trades, curve = simulate([signal], {"XYZ": bars}, slippage_bps=0)
    assert len(trades) == 1
    assert trades[0].reason == "gap_stop"
    assert trades[0].exit_price == 90
    assert trades[0].pnl < 0


def test_filter_and_cash_constraint():
    bars = [
        Bar("2020-01-01", 100, 101, 100, 100, 100),
        Bar("2020-01-02", 101, 102, 100, 101, 100),
    ]
    s = Signal("XYZ", "2019-12-31", "2020-01-01", 99, False)
    trades, curve = simulate([s], {"XYZ": bars}, anomaly_only=True)
    assert trades == []
    assert curve[-1][1] == 100000
    # Missing liquidation policy is rejected rather than silently ignoring positions.
    with pytest.raises(ValueError, match="Open positions"):
        simulate([s], {"XYZ": bars})


def test_no_future_outcome_input():
    # API accepts only OHLCV bars and dated anomaly membership, not outcome fields.
    assert "research_return" not in detect_signals.__code__.co_varnames
