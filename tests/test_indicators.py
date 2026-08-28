import pandas as pd
import pytest

from scanner.indicators import (
    atr,
    atr_series,
    average_volume,
    distance_from_level_pct,
    percent_change,
    relative_volume,
    remove_incomplete_session,
    return_over_sessions,
    rsi,
    sma,
    wilder_smoothed_series,
)


# ============================================================
# PERCENT CHANGE
# ============================================================

def test_percent_change_gain():
    result = percent_change(
        120,
        100,
    )

    assert result == pytest.approx(20.0)


def test_percent_change_loss():
    result = percent_change(
        80,
        100,
    )

    assert result == pytest.approx(-20.0)


def test_percent_change_zero_previous():
    assert percent_change(
        100,
        0,
    ) is None


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

    assert result == pytest.approx(
        expected
    )


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

    assert result == pytest.approx(
        10.0
    )


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

    # Average of 30, 40, 50.
    assert result == pytest.approx(
        40.0
    )


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

    assert result == pytest.approx(
        20.0
    )


def test_distance_below_level():
    result = distance_from_level_pct(
        price=80.0,
        level=100.0,
    )

    assert result == pytest.approx(
        -20.0
    )


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
    assert result == pytest.approx(
        250.0
    )


# ============================================================
# RELATIVE VOLUME
# ============================================================

def test_relative_volume():
    result = relative_volume(
        current_volume=1000.0,
        average_volume_value=250.0,
    )

    assert result == pytest.approx(
        4.0
    )


def test_relative_volume_zero_baseline():
    result = relative_volume(
        current_volume=1000.0,
        average_volume_value=0,
    )

    assert result is None


# ============================================================
# WILDER SMOOTHING
# ============================================================

def test_wilder_smoothing_uses_arithmetic_seed():
    values = pd.Series([
        1.0,
        2.0,
        3.0,
        6.0,
    ])

    result = wilder_smoothed_series(
        values,
        period=3,
    )

    # Wilder seed:
    # (1 + 2 + 3) / 3 = 2
    assert result.iloc[2] == pytest.approx(
        2.0
    )

    # Next Wilder value:
    # ((2 * 2) + 6) / 3
    # = 10 / 3
    assert result.iloc[3] == pytest.approx(
        10.0 / 3.0
    )


def test_wilder_smoothing_requires_full_seed():
    values = pd.Series([
        1.0,
        2.0,
    ])

    result = wilder_smoothed_series(
        values,
        period=3,
    )

    assert result.isna().all()


def test_wilder_smoothing_rejects_invalid_period():
    values = pd.Series([
        1.0,
        2.0,
        3.0,
    ])

    with pytest.raises(
        ValueError
    ):
        wilder_smoothed_series(
            values,
            period=0,
        )


# ============================================================
# RSI
# ============================================================

def test_rsi_all_gains_is_100():
    closes = pd.Series(
        [
            float(value)
            for value in range(1, 25)
        ]
    )

    result = rsi(
        closes,
        period=14,
    )

    assert result == pytest.approx(
        100.0
    )


def test_rsi_all_losses_is_zero():
    closes = pd.Series(
        [
            float(value)
            for value in range(25, 1, -1)
        ]
    )

    result = rsi(
        closes,
        period=14,
    )

    assert result == pytest.approx(
        0.0
    )


def test_flat_rsi_is_neutral_50():
    closes = pd.Series(
        [100.0] * 20
    )

    result = rsi(
        closes,
        period=14,
    )

    assert result == pytest.approx(
        50.0
    )


def test_rsi_matches_hand_calculated_wilder_example():
    closes = pd.Series([
        10.0,
        11.0,
        12.0,
        11.0,
        13.0,
    ])

    result = rsi(
        closes,
        period=3,
    )

    # Price changes:
    # +1, +1, -1, +2
    #
    # Initial Wilder averages over first 3 changes:
    #
    # Avg gain:
    # (1 + 1 + 0) / 3 = 2/3
    #
    # Avg loss:
    # (0 + 0 + 1) / 3 = 1/3
    #
    # Next averages using final +2 change:
    #
    # Gain:
    # ((2/3 * 2) + 2) / 3 = 10/9
    #
    # Loss:
    # ((1/3 * 2) + 0) / 3 = 2/9
    #
    # RS = (10/9) / (2/9) = 5
    #
    # RSI = 100 - 100 / (1 + 5)
    #     = 83.333333...
    assert result == pytest.approx(
        83.33333333333333,
        rel=1e-10,
    )


def test_rsi_requires_period_plus_one_closes():
    closes = pd.Series([
        10.0,
        11.0,
        12.0,
    ])

    assert rsi(
        closes,
        period=3,
    ) is None


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

    df = pd.DataFrame(
        rows
    )

    result = atr(
        df,
        period=14,
    )

    assert result is not None
    assert result > 0


def test_atr_constant_four_point_range():
    rows = []

    for value in range(1, 30):
        close = (
            100.0
            + (value * 0.1)
        )

        rows.append({
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
        })

    df = pd.DataFrame(
        rows
    )

    result = atr(
        df,
        period=14,
    )

    # Every day's true range is exactly $4.
    assert result == pytest.approx(
        4.0,
        rel=1e-6,
    )


def test_atr_uses_arithmetic_seed():
    df = pd.DataFrame([
        {
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
        },
        {
            "high": 12.0,
            "low": 10.0,
            "close": 11.0,
        },
        {
            "high": 14.0,
            "low": 11.0,
            "close": 13.0,
        },
    ])

    result = atr(
        df,
        period=3,
    )

    # True ranges:
    # 2, 2, 3
    #
    # First Wilder ATR:
    # (2 + 2 + 3) / 3
    assert result == pytest.approx(
        7.0 / 3.0
    )


def test_atr_matches_hand_calculated_wilder_recursion():
    df = pd.DataFrame([
        {
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
        },
        {
            "high": 12.0,
            "low": 10.0,
            "close": 11.0,
        },
        {
            "high": 14.0,
            "low": 11.0,
            "close": 13.0,
        },
        {
            "high": 15.0,
            "low": 12.0,
            "close": 14.0,
        },
        {
            "high": 14.5,
            "low": 10.0,
            "close": 11.0,
        },
    ])

    series = atr_series(
        df,
        period=3,
    )

    # True ranges:
    #
    # Bar 1 = 2
    # Bar 2 = 2
    # Bar 3 = 3
    #
    # Seed ATR:
    # 7/3 = 2.333333...
    #
    # Bar 4 TR = 3
    #
    # ATR:
    # ((7/3 * 2) + 3) / 3
    # = 23/9
    #
    # Bar 5 TR = 4.5
    #
    # ATR:
    # ((23/9 * 2) + 4.5) / 3
    # = 173/54

    assert series.iloc[2] == pytest.approx(
        7.0 / 3.0
    )

    assert series.iloc[3] == pytest.approx(
        23.0 / 9.0
    )

    assert series.iloc[4] == pytest.approx(
        173.0 / 54.0
    )

    assert atr(
        df,
        period=3,
    ) == pytest.approx(
        173.0 / 54.0
    )


def test_atr_requires_full_seed_period():
    df = pd.DataFrame([
        {
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
        },
        {
            "high": 12.0,
            "low": 10.0,
            "close": 11.0,
        },
    ])

    assert atr(
        df,
        period=3,
    ) is None


# ============================================================
# COMPLETED SESSION HANDLING
# ============================================================

def make_session_dataframe() -> pd.DataFrame:
    """
    Create two daily bars whose timestamps represent
    New York trading dates August 27 and August 28, 2026.

    During EDT, 04:00 UTC corresponds to midnight ET.
    """

    return pd.DataFrame([
        {
            "timestamp": pd.Timestamp(
                "2026-08-27T04:00:00Z"
            ),
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 1000.0,
        },
        {
            "timestamp": pd.Timestamp(
                "2026-08-28T04:00:00Z"
            ),
            "open": 101.0,
            "high": 104.0,
            "low": 100.0,
            "close": 103.0,
            "volume": 2000.0,
        },
    ])


def test_incomplete_same_day_session_removed_before_close():
    df = make_session_dataframe()

    now = pd.Timestamp(
        "2026-08-28T15:59:00",
        tz="America/New_York",
    )

    result = remove_incomplete_session(
        df,
        now=now,
    )

    assert len(result) == 1

    assert (
        result.iloc[-1]["close"]
        == pytest.approx(101.0)
    )


def test_same_day_session_kept_at_market_close():
    df = make_session_dataframe()

    now = pd.Timestamp(
        "2026-08-28T16:00:00",
        tz="America/New_York",
    )

    result = remove_incomplete_session(
        df,
        now=now,
    )

    assert len(result) == 2

    assert (
        result.iloc[-1]["close"]
        == pytest.approx(103.0)
    )


def test_same_day_session_kept_after_market_close():
    df = make_session_dataframe()

    now = pd.Timestamp(
        "2026-08-28T18:30:00",
        tz="America/New_York",
    )

    result = remove_incomplete_session(
        df,
        now=now,
    )

    assert len(result) == 2


def test_completed_session_survives_utc_calendar_rollover():
    df = make_session_dataframe()

    # 00:30 UTC on August 29 is still
    # 20:30 ET on August 28.
    #
    # The August 28 regular session is complete and
    # MUST remain available.
    now = pd.Timestamp(
        "2026-08-29T00:30:00Z"
    )

    result = remove_incomplete_session(
        df,
        now=now,
    )

    assert len(result) == 2

    assert (
        result.iloc[-1]["close"]
        == pytest.approx(103.0)
    )


def test_historical_session_is_never_removed():
    df = make_session_dataframe()

    now = pd.Timestamp(
        "2026-09-01T12:00:00",
        tz="America/New_York",
    )

    result = remove_incomplete_session(
        df,
        now=now,
    )

    assert len(result) == 2


def test_remove_incomplete_session_requires_aware_now():
    df = make_session_dataframe()

    now = pd.Timestamp(
        "2026-08-28 15:00:00"
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        remove_incomplete_session(
            df,
            now=now,
        )


def test_single_bar_is_not_removed():
    df = make_session_dataframe().tail(
        1
    ).reset_index(
        drop=True
    )

    now = pd.Timestamp(
        "2026-08-28T15:00:00",
        tz="America/New_York",
    )

    result = remove_incomplete_session(
        df,
        now=now,
    )

    # We intentionally preserve the old safety behavior:
    # with no prior completed bar available, do not turn
    # the dataframe into an empty dataset here.
    assert len(result) == 1