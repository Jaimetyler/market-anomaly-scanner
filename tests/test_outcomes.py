import pandas as pd
import pytest

from scanner.outcomes import (
    calculate_outcomes,
)


def make_future_bars():
    return pd.DataFrame(
        [
            {
                "high": 105.0,
                "low": 95.0,
                "close": 98.0,
            },
            {
                "high": 102.0,
                "low": 90.0,
                "close": 92.0,
            },
            {
                "high": 110.0,
                "low": 88.0,
                "close": 105.0,
            },
            {
                "high": 108.0,
                "low": 80.0,
                "close": 85.0,
            },
            {
                "high": 90.0,
                "low": 75.0,
                "close": 80.0,
            },
        ]
    )


def test_forward_returns():
    bars = make_future_bars()

    result = calculate_outcomes(
        future_bars=bars,
        signal_price=100.0,
    )

    assert result.return_1d == pytest.approx(-2.0)
    assert result.return_2d == pytest.approx(-8.0)
    assert result.return_3d == pytest.approx(5.0)
    assert result.return_5d == pytest.approx(-20.0)


def test_missing_longer_horizons():
    bars = make_future_bars()

    result = calculate_outcomes(
        future_bars=bars,
        signal_price=100.0,
    )

    assert result.return_10d is None
    assert result.return_20d is None


def test_max_future_gain():
    bars = make_future_bars()

    result = calculate_outcomes(
        future_bars=bars,
        signal_price=100.0,
    )

    assert (
        result.max_future_gain_pct
        == pytest.approx(10.0)
    )

    assert (
        result.sessions_to_max_gain
        == 3
    )


def test_max_future_decline():
    bars = make_future_bars()

    result = calculate_outcomes(
        future_bars=bars,
        signal_price=100.0,
    )

    assert (
        result.max_future_decline_pct
        == pytest.approx(-25.0)
    )

    assert (
        result.sessions_to_max_decline
        == 5
    )


def test_short_mfe_and_mae():
    bars = make_future_bars()

    result = calculate_outcomes(
        future_bars=bars,
        signal_price=100.0,
    )

    assert (
        result.mfe_pct
        == pytest.approx(25.0)
    )

    assert (
        result.mae_pct
        == pytest.approx(10.0)
    )


def test_empty_future_data():
    bars = pd.DataFrame(
        columns=[
            "high",
            "low",
            "close",
        ]
    )

    result = calculate_outcomes(
        future_bars=bars,
        signal_price=100.0,
    )

    assert result.return_1d is None
    assert result.mfe_pct is None
    assert result.mae_pct is None