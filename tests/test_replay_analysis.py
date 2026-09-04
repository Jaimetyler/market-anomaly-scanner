import pandas as pd

from scanner.replay_analysis import (
    RankingMethod,
    add_daily_ranks,
    add_ranking_scores,
    select_anchor_matches,
    summarize_score_deciles,
    wilson_lower_bound,
)


def test_wilson_penalizes_small_sample():
    small = wilson_lower_bound(
        wins=35,
        n=50,
    )

    large = wilson_lower_bound(
        wins=700,
        n=1000,
    )

    assert large > small


def test_add_ranking_scores():
    df = pd.DataFrame(
        [
            {
                "train_edge_score": 100.0,
                "train_win_rate": 0.65,
                "train_median_return": 5.0,
                "train_rank": 2,
                "train_n": 200,
            }
        ]
    )

    scored = add_ranking_scores(df)

    row = scored.iloc[0]

    assert row["score_raw_edge"] == 100.0
    assert row["score_raw_wr"] == 0.65
    assert row["score_raw_median"] == 5.0
    assert row["score_train_rank"] == -2.0
    assert (
        0.0
        < row["score_wilson_wr"]
        < 0.65
    )


def test_anchor_selection_uses_method_score():
    df = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "horizon_days": 20,
                "score": 10.0,
                "train_n": 1000,
                "condition_count": 1,
            },
            {
                "event_id": "E1",
                "horizon_days": 20,
                "score": 20.0,
                "train_n": 50,
                "condition_count": 2,
            },
        ]
    )

    method = RankingMethod(
        name="TEST",
        score_column="score",
    )

    anchors = select_anchor_matches(
        df,
        method,
    )

    assert len(anchors) == 1
    assert (
        anchors.iloc[0]["ranking_score"]
        == 20.0
    )


def test_daily_rank_resets_by_day_and_horizon():
    df = pd.DataFrame(
        [
            {
                "ranking_method": "TEST",
                "date": pd.Timestamp("2025-01-01"),
                "horizon_days": 5,
                "ranking_score": 30.0,
            },
            {
                "ranking_method": "TEST",
                "date": pd.Timestamp("2025-01-01"),
                "horizon_days": 5,
                "ranking_score": 20.0,
            },
            {
                "ranking_method": "TEST",
                "date": pd.Timestamp("2025-01-02"),
                "horizon_days": 5,
                "ranking_score": 10.0,
            },
        ]
    )

    ranked = add_daily_ranks(df)

    assert list(
        ranked["daily_rank"]
    ) == [1, 2, 1]


def test_score_deciles_preserve_group_columns():
    rows = []

    for i in range(20):
        rows.append(
            {
                "event_id": f"E{i}",
                "ranking_method": "TEST",
                "test_year": 2024,
                "horizon_days": 5,
                "ranking_score": float(i),
                "realized_win": i >= 10,
                "realized_contrarian_return": float(
                    i - 10
                ),
                "train_n": 100 + i,
            }
        )

    anchors = pd.DataFrame(rows)

    summary = summarize_score_deciles(
        anchors
    )

    assert "ranking_method" in summary.columns
    assert "test_year" in summary.columns
    assert "horizon_days" in summary.columns
    assert "score_decile" in summary.columns

    assert summary[
        "ranking_method"
    ].eq("TEST").all()

    assert summary[
        "test_year"
    ].eq(2024).all()

    assert summary[
        "horizon_days"
    ].eq(5).all()

    assert (
        summary["score_decile"].min()
        == 1
    )

    assert (
        summary["score_decile"].max()
        == 10
    )