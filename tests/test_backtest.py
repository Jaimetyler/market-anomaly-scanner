from __future__ import annotations

import pandas as pd
import pytest

from scanner.backtest import BacktestConfig, compute_event_outcomes, summarize_outcomes


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "AAA", "date": "2024-01-02", "high": 10.5, "low": 9.5, "close": 10.0},
            {"ticker": "AAA", "date": "2024-01-03", "high": 10.2, "low": 8.8, "close": 9.0},
            {"ticker": "AAA", "date": "2024-01-04", "high": 9.4, "low": 8.0, "close": 8.5},
            {"ticker": "AAA", "date": "2024-01-05", "high": 9.0, "low": 8.2, "close": 8.8},
            {"ticker": "BBB", "date": "2024-01-02", "high": 20.5, "low": 19.5, "close": 20.0},
            {"ticker": "BBB", "date": "2024-01-03", "high": 22.5, "low": 19.8, "close": 22.0},
            {"ticker": "BBB", "date": "2024-01-04", "high": 23.0, "low": 21.5, "close": 22.5},
        ]
    )


def test_upside_anomaly_contrarian_short() -> None:
    events = pd.DataFrame(
        [{"ticker": "AAA", "date": "2024-01-02", "event_direction": 1}]
    )

    result = compute_event_outcomes(
        _prices(),
        events,
        BacktestConfig(horizons=(1, 2)),
    )

    row = result.iloc[0]

    assert row["event_matched"] == True
    assert row["return_1d"] == pytest.approx(-0.10)
    assert row["contrarian_return_1d"] == pytest.approx(0.10)
    assert row["reversed_1d"] == True
    assert row["return_2d"] == pytest.approx(-0.15)
    assert row["contrarian_return_2d"] == pytest.approx(0.15)
    assert row["contrarian_mfe_2d"] == pytest.approx(0.20)
    assert row["contrarian_mae_2d"] == pytest.approx(-0.02)


def test_downside_anomaly_contrarian_long() -> None:
    events = pd.DataFrame(
        [{"ticker": "BBB", "date": "2024-01-02", "event_direction": -1}]
    )

    result = compute_event_outcomes(
        _prices(),
        events,
        BacktestConfig(horizons=(1, 2)),
    )

    row = result.iloc[0]

    assert row["return_1d"] == pytest.approx(0.10)
    assert row["contrarian_return_1d"] == pytest.approx(0.10)
    assert row["reversed_1d"] == True
    assert row["contrarian_mfe_2d"] == pytest.approx(0.15)
    assert row["contrarian_mae_2d"] == pytest.approx(-0.01)


def test_missing_event_price_is_preserved_and_marked() -> None:
    events = pd.DataFrame(
        [{"ticker": "ZZZ", "date": "2024-01-02", "event_direction": 1}]
    )

    result = compute_event_outcomes(
        _prices(),
        events,
        BacktestConfig(horizons=(1,)),
    )

    row = result.iloc[0]

    assert row["event_matched"] == False
    assert row["event_match_error"] == "no_exact_ticker_date_price_row"
    assert row["outcome_available_1d"] == False


def test_end_of_history_marks_unavailable_horizon() -> None:
    events = pd.DataFrame(
        [{"ticker": "AAA", "date": "2024-01-05", "event_direction": 1}]
    )

    result = compute_event_outcomes(
        _prices(),
        events,
        BacktestConfig(horizons=(1,)),
    )

    assert result.iloc[0]["outcome_available_1d"] == False


def test_summary() -> None:
    events = pd.DataFrame(
        [
            {"ticker": "AAA", "date": "2024-01-02", "event_direction": 1},
            {"ticker": "BBB", "date": "2024-01-02", "event_direction": -1},
        ]
    )

    outcomes = compute_event_outcomes(
        _prices(),
        events,
        BacktestConfig(horizons=(1,)),
    )
    summary = summarize_outcomes(outcomes, horizons=(1,))

    assert summary.iloc[0]["n"] == 2
    assert summary.iloc[0]["reversal_rate"] == pytest.approx(1.0)
    assert summary.iloc[0]["contrarian_win_rate"] == pytest.approx(1.0)
