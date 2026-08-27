import pandas as pd
import pytest

from scanner.indicators import (
    atr,
    average_volume,
    distance_from_level_pct,
    percent_change,
    relative_volume,
    return_over_sessions,
    rsi,
    sma,
)


# ============================================================
# PERCENT CHANGE
# ============================================================

def test_percent_change_gain():
    result = percent_change(120, 100)

    assert result == pytest.approx(20.0)


def test_percent_change_loss():
    result = percent_change(80, 100)

    assert result == pytest.approx(-20.0)


def test_percent_change_zero_previous():
    assert percent_change(100, 0) is None


# ============================================================
# SESSION RETURNS
# ============================================================

def test_return_over_one_session():
    closes = pd.Series([
        100.0,
        105.0,
        110.0,
    ])

    result = return_over_sessions(
        closes,
        1,
    )

    expected = (
        (110.0 / 105.0) - 1
    ) * 100

    assert result == pytest.approx(expected)


def test_return_over_two_sessions():
    closes = pd.Series([
        100.0,
        105.0,
        110.0,
    ])

    result = return_over_sessions(
        closes,
        2,
    )

    assert result == pytest.approx(10.0)


def test_return_requires_enough_history():
    closes = pd.Series([
        100.0,
        105.0,
    ])

    assert return_over_sessions(
        closes,
        2,
    ) is None


# ============================================================
# SIMPLE MOVING AVERAGE
# ============================================================

def test_sma():
    values = pd.Series([
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ])

    result = sma(
        values,
        3,
    )

    # Average of 30, 40, 50
    assert result == pytest.approx(40.0)


def test_sma_requires_enough_history():
    values = pd.Series([
        10.0,
        20.0,
    ])

    assert sma(
        values,
        3,
    ) is None


# ============================================================
# DISTANCE FROM MOVING AVERAGE / LEVEL
# ============================================================

def test_distance_above_level():
    result = distance_from_level_pct(
        price=120.0,
        level=100.0,
    )

    assert result == pytest.approx(20.0)


def test_distance_below_level():
    result = distance_from_level_pct(
        price=80.0,
        level=100.0,
    )

    assert result == pytest.approx(-20.0)


# ============================================================
# AVERAGE VOLUME
# ============================================================

def test_average_volume_excludes_current():
    volumes = pd.Series([
        100.0,
        200.0,
        300.0,
        400.0,
        5000.0,
    ])

    result = average_volume(
        volumes,
        period=4,
        exclude_current=True,
    )

    # Current 5000-share session should NOT
    # contaminate the historical baseline.
    assert result == pytest.approx(250.0)


# ============================================================
# RELATIVE VOLUME
# ============================================================

def test_relative_volume():
    result = relative_volume(
        current_volume=1000.0,
        average_volume_value=250.0,
    )

    assert result == pytest.approx(4.0)


def test_relative_volume_zero_baseline():
    result = relative_volume(
        current_volume=1000.0,
        average_volume_value=0,
    )

    assert result is None


# ============================================================
# RSI
# ============================================================

def test_rsi_all_gains_is_100():
    closes = pd.Series(
        [float(value) for value in range(1, 25)]
    )

    result = rsi(
        closes,
        period=14,
    )

    assert result == pytest.approx(100.0)


def test_rsi_all_losses_is_zero():
    closes = pd.Series(
        [float(value) for value in range(25, 1, -1)]
    )

    result = rsi(
        closes,
        period=14,
    )

    assert result == pytest.approx(0.0)


def test_rsi_stays_between_zero_and_100():
    closes = pd.Series([
        44.0,
        45.0,
        44.5,
        46.0,
        45.5,
        47.0,
        46.0,
        48.0,
        47.5,
        49.0,
        48.0,
        50.0,
        49.0,
        51.0,
        50.0,
        52.0,
        51.0,
        53.0,
        52.0,
        54.0,
    ])

    result = rsi(
        closes,
        period=14,
    )

    assert result is not None
    assert 0 <= result <= 100


# ============================================================
# ATR
# ============================================================

def test_atr_returns_positive_value():
    rows = []

    for value in range(1, 25):
        close = 100.0 + value

        rows.append({
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
        })

    df = pd.DataFrame(rows)

    result = atr(
        df,
        period=14,
    )

    assert result is not None
    assert result > 0


def test_atr_constant_four_point_range():
    rows = []

    for value in range(1, 30):
        close = 100.0 + (value * 0.1)

        rows.append({
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
        })

    df = pd.DataFrame(rows)

    result = atr(
        df,
        period=14,
    )

    # Every day's true range is exactly $4.
    assert result == pytest.approx(
        4.0,
        rel=1e-6,
    )