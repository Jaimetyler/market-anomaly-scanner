from datetime import date

import pandas as pd
import pytest

from scanner.miner import (
    HARD_MAX_TICKERS,
    HistoricalMinerConfig,
    build_event_rows,
    normalize_date,
    signal_dates_in_window,
    trading_dates_from_bars,
    validate_miner_config,
)


def timestamp_ms(
    value: str,
) -> int:
    return int(
        pd.Timestamp(
            value,
            tz="UTC",
        ).timestamp()
        * 1000
    )


def make_bar(
    session_date: str,
) -> dict:
    return {
        "t": timestamp_ms(
            session_date
        ),
        "o": 10.0,
        "h": 11.0,
        "l": 9.0,
        "c": 10.5,
        "v": 1_000_000,
    }


def make_research_row(
    *,
    ticker: str,
    signal_date: str,
    primary_setup: str,
    eligible: bool = True,
    executable_return_5d: float = -10.0,
) -> dict:
    return {
        "ticker": ticker,
        "signal_date": signal_date,
        "primary_setup": primary_setup,
        "setup_tags": "",
        "detection_triggers": "TEST_TRIGGER",
        "detection_evidence": "",
        "return_1d": 20.0,
        "return_3d": 30.0,
        "return_5d": 40.0,
        "return_10d": 50.0,
        "return_20d": 60.0,
        "relative_volume": 5.0,
        "rsi_14": 80.0,
        "distance_sma_20_pct": 30.0,
        "distance_sma_50_pct": 40.0,
        "atr_expansion": 2.0,
        "ca_flagged": False,
        "ca_excluded": not eligible,
        "ca_flags": "",
        "research_eligible": eligible,
        "signal_close": 100.0,
        "next_open": 101.0,
        "research_return_1d": -2.0,
        "research_return_2d": -3.0,
        "research_return_3d": -4.0,
        "research_return_5d": -5.0,
        "research_return_10d": -6.0,
        "research_return_20d": -7.0,
        "research_max_future_gain_pct": 10.0,
        "research_max_future_decline_pct": -20.0,
        "research_mfe_pct": 20.0,
        "research_mae_pct": 10.0,
        "research_sessions_to_max_gain": 2,
        "research_sessions_to_max_decline": 4,
        "executable_return_1d": -3.0,
        "executable_return_2d": -4.0,
        "executable_return_3d": -5.0,
        "executable_return_5d": (
            executable_return_5d
        ),
        "executable_return_10d": -8.0,
        "executable_return_20d": -12.0,
        "executable_max_future_gain_pct": 8.0,
        "executable_max_future_decline_pct": -22.0,
        "executable_mfe_pct": 22.0,
        "executable_mae_pct": 8.0,
        "executable_sessions_to_max_gain": 2,
        "executable_sessions_to_max_decline": 4,
    }


def test_normalize_date_string():
    result = normalize_date(
        "2025-06-30"
    )

    assert result == date(
        2025,
        6,
        30,
    )


def test_validate_config_accepts_normal_window():
    config = HistoricalMinerConfig(
        start_date=date(
            2025,
            1,
            1,
        ),
        end_date=date(
            2025,
            3,
            31,
        ),
        max_tickers=25,
    )

    validate_miner_config(
        config
    )


def test_validate_config_rejects_reversed_dates():
    config = HistoricalMinerConfig(
        start_date=date(
            2025,
            3,
            31,
        ),
        end_date=date(
            2025,
            1,
            1,
        ),
        max_tickers=25,
    )

    with pytest.raises(
        ValueError
    ):
        validate_miner_config(
            config
        )


def test_validate_config_rejects_zero_tickers():
    config = HistoricalMinerConfig(
        start_date=date(
            2025,
            1,
            1,
        ),
        end_date=date(
            2025,
            1,
            31,
        ),
        max_tickers=0,
    )

    with pytest.raises(
        ValueError
    ):
        validate_miner_config(
            config
        )


def test_validate_config_enforces_hard_cap():
    config = HistoricalMinerConfig(
        start_date=date(
            2025,
            1,
            1,
        ),
        end_date=date(
            2025,
            1,
            31,
        ),
        max_tickers=(
            HARD_MAX_TICKERS
            + 1
        ),
    )

    with pytest.raises(
        ValueError
    ):
        validate_miner_config(
            config
        )


def test_trading_dates_are_sorted_and_unique():
    bars = [
        make_bar(
            "2025-01-03"
        ),
        make_bar(
            "2025-01-02"
        ),
        make_bar(
            "2025-01-03"
        ),
    ]

    result = trading_dates_from_bars(
        bars
    )

    assert result == [
        "2025-01-02",
        "2025-01-03",
    ]


def test_signal_dates_are_limited_to_window():
    bars = [
        make_bar(
            "2024-12-31"
        ),
        make_bar(
            "2025-01-02"
        ),
        make_bar(
            "2025-01-03"
        ),
        make_bar(
            "2025-01-06"
        ),
        make_bar(
            "2025-01-07"
        ),
    ]

    result = signal_dates_in_window(
        bars=bars,
        start_date=date(
            2025,
            1,
            2,
        ),
        end_date=date(
            2025,
            1,
            6,
        ),
    )

    assert result == [
        "2025-01-02",
        "2025-01-03",
        "2025-01-06",
    ]


def test_event_rows_cluster_consecutive_observations():
    sessions = [
        "2025-03-14",
        "2025-03-17",
        "2025-03-18",
        "2025-03-19",
    ]

    rows = [
        make_research_row(
            ticker="AAOI",
            signal_date="2025-03-14",
            primary_setup="FRESH_SPIKE",
            executable_return_5d=-8.0,
        ),
        make_research_row(
            ticker="AAOI",
            signal_date="2025-03-17",
            primary_setup="UNCLASSIFIED_ANOMALY",
            executable_return_5d=6.0,
        ),
        make_research_row(
            ticker="AAOI",
            signal_date="2025-03-18",
            primary_setup="UNCLASSIFIED_ANOMALY",
            executable_return_5d=7.0,
        ),
    ]

    events = build_event_rows(
        rows=rows,
        session_dates=sessions,
    )

    assert len(
        events
    ) == 1

    event = events[0]

    assert (
        event[
            "observation_count"
        ]
        == 3
    )

    assert (
        event[
            "initial_setup"
        ]
        == "FRESH_SPIKE"
    )

    assert (
        event[
            "final_setup"
        ]
        == "UNCLASSIFIED_ANOMALY"
    )


def test_event_outcome_uses_first_signal_entry():
    sessions = [
        "2025-03-14",
        "2025-03-17",
        "2025-03-18",
    ]

    rows = [
        make_research_row(
            ticker="AAOI",
            signal_date="2025-03-14",
            primary_setup="FRESH_SPIKE",
            executable_return_5d=-8.32,
        ),
        make_research_row(
            ticker="AAOI",
            signal_date="2025-03-17",
            primary_setup="UNCLASSIFIED_ANOMALY",
            executable_return_5d=6.40,
        ),
    ]

    events = build_event_rows(
        rows=rows,
        session_dates=sessions,
    )

    assert len(
        events
    ) == 1

    assert (
        events[0][
            "executable_return_5d"
        ]
        == pytest.approx(
            -8.32
        )
    )


def test_ca_excluded_observation_does_not_enter_event_dataset():
    sessions = [
        "2025-01-02",
        "2025-01-03",
    ]

    rows = [
        make_research_row(
            ticker="TEST",
            signal_date="2025-01-02",
            primary_setup="FRESH_SPIKE",
            eligible=False,
        )
    ]

    events = build_event_rows(
        rows=rows,
        session_dates=sessions,
    )

    assert events == []


def test_different_tickers_create_different_events():
    sessions = [
        "2025-01-02",
        "2025-01-03",
    ]

    rows = [
        make_research_row(
            ticker="AAA",
            signal_date="2025-01-02",
            primary_setup="FRESH_SPIKE",
        ),
        make_research_row(
            ticker="BBB",
            signal_date="2025-01-02",
            primary_setup="FRESH_SPIKE",
        ),
    ]

    events = build_event_rows(
        rows=rows,
        session_dates=sessions,
    )

    assert len(
        events
    ) == 2

    assert {
        event[
            "ticker"
        ]
        for event in events
    } == {
        "AAA",
        "BBB",
    }