import pandas as pd

from scanner.replay_policy import (
    POLICIES,
    apply_policy,
    build_policy_samples,
    profit_factor,
    rank_policy_universe,
    summarize_periods,
)


def policy(
    name: str,
):
    return next(
        item
        for item in POLICIES
        if item.name == name
    )


def test_price_policy_filters_before_ranking():
    df = pd.DataFrame(
        [
            {
                "event_id": "A",
                "date": pd.Timestamp(
                    "2025-01-01"
                ),
                "horizon_days": 20,
                "ranking_score": 100.0,
                "train_n": 100,
                "condition_count": 1,
                "event_close": 1.0,
            },
            {
                "event_id": "B",
                "date": pd.Timestamp(
                    "2025-01-01"
                ),
                "horizon_days": 20,
                "ranking_score": 90.0,
                "train_n": 100,
                "condition_count": 1,
                "event_close": 10.0,
            },
        ]
    )

    ranked = rank_policy_universe(
        df,
        policy(
            "PRICE_GE_5"
        ),
    )

    assert len(ranked) == 1

    assert (
        ranked.iloc[0]["event_id"]
        == "B"
    )

    assert (
        ranked.iloc[0]["daily_rank"]
        == 1
    )


def test_next_eligible_name_moves_into_top_five():
    rows = []

    for i in range(6):
        rows.append(
            {
                "event_id": f"E{i}",
                "date": pd.Timestamp(
                    "2025-01-01"
                ),
                "test_year": 2025,
                "horizon_days": 20,
                "ranking_score": float(
                    100 - i
                ),
                "train_n": 100,
                "condition_count": 1,
                "event_close": (
                    1.0
                    if i == 1
                    else 10.0
                ),
                "realized_contrarian_return": 5.0,
            }
        )

    df = pd.DataFrame(
        rows
    )

    samples = build_policy_samples(
        df
    )

    filtered = samples[
        (
            samples["policy"]
            == "PRICE_GE_5"
        )
        & (
            samples["top_n"]
            == 5
        )
    ]

    ids = set(
        filtered["event_id"]
    )

    assert len(ids) == 5

    assert "E1" not in ids
    assert "E5" in ids


def test_apply_price_five_policy():
    df = pd.DataFrame(
        {
            "event_close": [
                1.0,
                4.99,
                5.0,
                20.0,
            ]
        }
    )

    result = apply_policy(
        df,
        policy(
            "PRICE_GE_5"
        ),
    )

    assert len(result) == 2


def test_baseline_keeps_all():
    df = pd.DataFrame(
        {
            "event_close": [
                0.1,
                1.0,
                100.0,
            ]
        }
    )

    result = apply_policy(
        df,
        policy(
            "BASELINE"
        ),
    )

    assert len(result) == 3


def test_profit_factor():
    values = pd.Series(
        [
            20.0,
            10.0,
            -5.0,
            -5.0,
        ]
    )

    assert (
        profit_factor(
            values
        )
        == 3.0
    )


def test_period_summary_separates_validation():
    df = pd.DataFrame(
        [
            {
                "policy": "PRICE_GE_5",
                "event_id": "A",
                "date": pd.Timestamp(
                    "2024-01-01"
                ),
                "test_year": 2024,
                "horizon_days": 20,
                "top_n": 5,
                "realized_contrarian_return": 10.0,
                "event_close": 10.0,
            },
            {
                "policy": "PRICE_GE_5",
                "event_id": "B",
                "date": pd.Timestamp(
                    "2025-01-01"
                ),
                "test_year": 2025,
                "horizon_days": 20,
                "top_n": 5,
                "realized_contrarian_return": 10.0,
                "event_close": 10.0,
            },
        ]
    )

    summary = summarize_periods(
        df
    )

    assert set(
        summary["period"]
    ) == {
        "DEVELOPMENT_2021_2024",
        "VALIDATION_2025",
    }