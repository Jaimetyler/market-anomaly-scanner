import pandas as pd

from scanner.replay_displacement import (
    compare_baseline_to_policy,
    summarize_periods,
    worst_replacements,
)


def test_replacement_is_detected():
    rows = [
        {
            "policy": "BASELINE",
            "event_id": "A",
            "date": pd.Timestamp(
                "2025-01-01"
            ),
            "horizon_days": 20,
            "top_n": 5,
        },
        {
            "policy": "BASELINE",
            "event_id": "B",
            "date": pd.Timestamp(
                "2025-01-01"
            ),
            "horizon_days": 20,
            "top_n": 5,
        },
        {
            "policy": "PRICE_GE_5",
            "event_id": "A",
            "date": pd.Timestamp(
                "2025-01-01"
            ),
            "horizon_days": 20,
            "top_n": 5,
        },
        {
            "policy": "PRICE_GE_5",
            "event_id": "C",
            "date": pd.Timestamp(
                "2025-01-01"
            ),
            "horizon_days": 20,
            "top_n": 5,
        },
    ]

    df = pd.DataFrame(
        rows
    )

    result = (
        compare_baseline_to_policy(
            df
        )
    )

    flags = dict(
        zip(
            result["event_id"],
            result["is_replacement"],
        )
    )

    assert not flags["A"]
    assert flags["C"]


def test_period_summary_separates_replacements():
    df = pd.DataFrame(
        [
            {
                "event_id": "A",
                "date": pd.Timestamp(
                    "2024-01-01"
                ),
                "test_year": 2024,
                "horizon_days": 20,
                "daily_rank": 4,
                "is_replacement": False,
                "realized_contrarian_return": 10.0,
            },
            {
                "event_id": "B",
                "date": pd.Timestamp(
                    "2024-01-01"
                ),
                "test_year": 2024,
                "horizon_days": 20,
                "daily_rank": 5,
                "is_replacement": True,
                "realized_contrarian_return": -20.0,
            },
        ]
    )

    result = summarize_periods(
        df
    )

    assert set(
        result[
            "is_replacement"
        ]
    ) == {
        False,
        True,
    }


def test_worst_replacements_only_returns_replacements():
    df = pd.DataFrame(
        [
            {
                "event_id": "A",
                "is_replacement": False,
                "realized_contrarian_return": -1000.0,
            },
            {
                "event_id": "B",
                "is_replacement": True,
                "realized_contrarian_return": -50.0,
            },
            {
                "event_id": "C",
                "is_replacement": True,
                "realized_contrarian_return": -100.0,
            },
        ]
    )

    result = worst_replacements(
        df
    )

    assert len(result) == 2

    assert list(
        result["event_id"]
    ) == [
        "C",
        "B",
    ]