import pandas as pd

from scanner.replay_dataset import (
    build_historical_replay,
)


def _fold(
    *,
    test_year: int,
    horizon: int,
    setup: str,
    edge: float,
) -> dict:
    return {
        "segmentation": "initial_setup",
        "test_year": test_year,
        "train_start_year": test_year - 2,
        "train_end_year": test_year - 1,
        "horizon_days": horizon,
        "train_rank": 1,
        "initial_setup": setup,
        "train_n": 100,
        "train_win_rate": 0.60,
        "train_mean_return": 2.0,
        "train_median_return": 1.5,
        "train_reversal_rate": 0.60,
        "train_edge_score": edge,
        "rvol_bucket": None,
        "sma20_extension_bucket": None,
        "move_1d_bucket": None,
        "move_3d_bucket": None,
        "move_5d_bucket": None,
    }


def test_replay_matches_only_correct_test_year():
    events = pd.DataFrame(
        [
            {
                "event_id": "A",
                "ticker": "AAA",
                "date": "2021-05-01",
                "event_direction": 1,
                "initial_setup": "FRESH_SPIKE",
                "rvol_bucket": "2-3x",
                "sma20_extension_bucket": "+50%+",
                "move_1d_bucket": "+100%+",
                "move_3d_bucket": "+100%+",
                "move_5d_bucket": "+100%+",
                "outcome_available_1d": True,
                "contrarian_return_1d": 5.0,
            },
            {
                "event_id": "B",
                "ticker": "BBB",
                "date": "2022-05-01",
                "event_direction": 1,
                "initial_setup": "FRESH_SPIKE",
                "rvol_bucket": "2-3x",
                "sma20_extension_bucket": "+50%+",
                "move_1d_bucket": "+100%+",
                "move_3d_bucket": "+100%+",
                "move_5d_bucket": "+100%+",
                "outcome_available_1d": True,
                "contrarian_return_1d": -2.0,
            },
        ]
    )

    folds = pd.DataFrame(
        [
            _fold(
                test_year=2021,
                horizon=1,
                setup="FRESH_SPIKE",
                edge=10.0,
            ),
            _fold(
                test_year=2022,
                horizon=1,
                setup="FRESH_SPIKE",
                edge=20.0,
            ),
        ]
    )

    replay = build_historical_replay(
        events,
        folds,
    )

    assert len(replay) == 2

    row_2021 = replay[
        replay["test_year"] == 2021
    ].iloc[0]

    row_2022 = replay[
        replay["test_year"] == 2022
    ].iloc[0]

    assert row_2021["train_edge_score"] == 10.0
    assert row_2022["train_edge_score"] == 20.0


def test_replay_records_realized_outcome():
    events = pd.DataFrame(
        [
            {
                "event_id": "A",
                "ticker": "AAA",
                "date": "2021-05-01",
                "event_direction": 1,
                "initial_setup": "PARABOLIC_EXTENSION",
                "rvol_bucket": "10x+",
                "sma20_extension_bucket": "+50%+",
                "move_1d_bucket": "+100%+",
                "move_3d_bucket": "+100%+",
                "move_5d_bucket": "+100%+",
                "outcome_available_5d": True,
                "contrarian_return_5d": 12.5,
            }
        ]
    )

    folds = pd.DataFrame(
        [
            _fold(
                test_year=2021,
                horizon=5,
                setup="PARABOLIC_EXTENSION",
                edge=50.0,
            )
        ]
    )

    replay = build_historical_replay(
        events,
        folds,
    )

    assert len(replay) == 1
    assert (
        replay.iloc[0][
            "realized_contrarian_return"
        ]
        == 12.5
    )
    assert bool(
        replay.iloc[0]["realized_win"]
    )


def test_replay_does_not_match_wrong_condition():
    events = pd.DataFrame(
        [
            {
                "event_id": "A",
                "ticker": "AAA",
                "date": "2021-05-01",
                "event_direction": 1,
                "initial_setup": "FRESH_SPIKE",
                "rvol_bucket": "2-3x",
                "sma20_extension_bucket": "+50%+",
                "move_1d_bucket": "+100%+",
                "move_3d_bucket": "+100%+",
                "move_5d_bucket": "+100%+",
                "outcome_available_1d": True,
                "contrarian_return_1d": 2.0,
            }
        ]
    )

    folds = pd.DataFrame(
        [
            _fold(
                test_year=2021,
                horizon=1,
                setup="PARABOLIC_EXTENSION",
                edge=50.0,
            )
        ]
    )

    replay = build_historical_replay(
        events,
        folds,
    )

    assert replay.empty