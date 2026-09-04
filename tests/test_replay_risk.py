import pandas as pd

from scanner.replay_risk import (
    add_daily_ranks,
    attach_excursions,
    build_top_n_samples,
    summarize_period,
)


def test_daily_ranking_uses_highest_score_first():
    df = pd.DataFrame(
        [
            {
                "event_id": "A",
                "date": pd.Timestamp(
                    "2025-01-01"
                ),
                "horizon_days": 5,
                "ranking_score": 10.0,
            },
            {
                "event_id": "B",
                "date": pd.Timestamp(
                    "2025-01-01"
                ),
                "horizon_days": 5,
                "ranking_score": 30.0,
            },
            {
                "event_id": "C",
                "date": pd.Timestamp(
                    "2025-01-01"
                ),
                "horizon_days": 5,
                "ranking_score": 20.0,
            },
        ]
    )

    ranked = add_daily_ranks(
        df
    )

    ranks = dict(
        zip(
            ranked["event_id"],
            ranked["daily_rank"],
        )
    )

    assert ranks["B"] == 1
    assert ranks["C"] == 2
    assert ranks["A"] == 3


def test_build_top_n_samples():
    df = pd.DataFrame(
        [
            {
                "event_id": f"E{i}",
                "daily_rank": i,
            }
            for i in range(
                1,
                12,
            )
        ]
    )

    samples = build_top_n_samples(
        df
    )

    counts = (
        samples.groupby(
            "top_n"
        )
        .size()
        .to_dict()
    )

    assert counts[1] == 1
    assert counts[3] == 3
    assert counts[5] == 5
    assert counts[10] == 10


def test_attach_excursions_matches_event_and_horizon():
    anchors = pd.DataFrame(
        [
            {
                "event_id": "A",
                "horizon_days": 5,
            },
            {
                "event_id": "A",
                "horizon_days": 20,
            },
        ]
    )

    excursions = pd.DataFrame(
        [
            {
                "event_id": "A",
                "horizon_days": 5,
                "contrarian_mfe": 10.0,
                "contrarian_mae": -4.0,
            },
            {
                "event_id": "A",
                "horizon_days": 20,
                "contrarian_mfe": 25.0,
                "contrarian_mae": -8.0,
            },
        ]
    )

    merged = attach_excursions(
        anchors,
        excursions,
    )

    row_5 = merged[
        merged["horizon_days"] == 5
    ].iloc[0]

    row_20 = merged[
        merged["horizon_days"] == 20
    ].iloc[0]

    assert (
        row_5["contrarian_mfe"]
        == 10.0
    )

    assert (
        row_5["contrarian_mae"]
        == -4.0
    )

    assert (
        row_20["contrarian_mfe"]
        == 25.0
    )

    assert (
        row_20["contrarian_mae"]
        == -8.0
    )


def test_risk_summary_calculates_distribution():
    rows = []

    returns = [
        -20.0,
        -10.0,
        -5.0,
        1.0,
        2.0,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
    ]

    for i, value in enumerate(
        returns
    ):
        rows.append(
            {
                "event_id": f"E{i}",
                "date": pd.Timestamp(
                    "2025-01-01"
                )
                + pd.Timedelta(
                    days=i
                ),
                "test_year": 2025,
                "horizon_days": 5,
                "top_n": 5,
                "realized_contrarian_return": value,
                "contrarian_mfe": (
                    value + 5.0
                ),
                "contrarian_mae": (
                    value - 5.0
                ),
            }
        )

    samples = pd.DataFrame(
        rows
    )

    summary = summarize_period(
        samples
    )

    assert len(summary) == 1

    row = summary.iloc[0]

    assert (
        row["period"]
        == "HOLDOUT_2025"
    )

    assert row["observations"] == 10
    assert row["win_rate"] == 0.7

    assert (
        row["median_return"]
        > 0
    )

    assert (
        row["p05_return"]
        < 0
    )

    assert (
        row["average_loss"]
        < 0
    )

    assert (
        row["profit_factor"]
        > 0
    )


def test_development_and_holdout_are_separate():
    samples = pd.DataFrame(
        [
            {
                "event_id": "A",
                "date": pd.Timestamp(
                    "2024-01-01"
                ),
                "test_year": 2024,
                "horizon_days": 20,
                "top_n": 5,
                "realized_contrarian_return": 10.0,
            },
            {
                "event_id": "B",
                "date": pd.Timestamp(
                    "2025-01-01"
                ),
                "test_year": 2025,
                "horizon_days": 20,
                "top_n": 5,
                "realized_contrarian_return": 20.0,
            },
        ]
    )

    summary = summarize_period(
        samples
    )

    periods = set(
        summary["period"]
    )

    assert (
        "DEVELOPMENT_2021_2024"
        in periods
    )

    assert (
        "HOLDOUT_2025"
        in periods
    )