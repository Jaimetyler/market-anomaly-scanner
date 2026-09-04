import pandas as pd

from scanner.evidence_replay import (
    best_match_by_horizon,
    event_features_from_row,
    fold_matches_event,
    match_historical_event,
)


def test_event_features_from_row():
    row = pd.Series(
        {
            "initial_setup": "PARABOLIC_EXTENSION",
            "rvol_bucket": "3-5x",
            "sma20_extension_bucket": "+50%+",
            "move_1d_bucket": None,
            "move_3d_bucket": "+100%+",
            "move_5d_bucket": "+100%+",
        }
    )

    features = event_features_from_row(row)

    assert features["initial_setup"] == "PARABOLIC_EXTENSION"
    assert features["rvol_bucket"] == "3-5x"
    assert features["move_1d_bucket"] is None
    assert features["move_5d_bucket"] == "+100%+"


def test_fold_matches_event_ignores_unused_conditions():
    fold = pd.Series(
        {
            "initial_setup": "PARABOLIC_EXTENSION",
            "rvol_bucket": None,
            "sma20_extension_bucket": "+50%+",
            "move_1d_bucket": None,
            "move_3d_bucket": None,
            "move_5d_bucket": None,
        }
    )

    event = {
        "initial_setup": "PARABOLIC_EXTENSION",
        "rvol_bucket": "3-5x",
        "sma20_extension_bucket": "+50%+",
        "move_1d_bucket": "+30% to +50%",
        "move_3d_bucket": "+100%+",
        "move_5d_bucket": "+100%+",
    }

    assert fold_matches_event(fold, event)


def test_fold_rejects_mismatched_condition():
    fold = pd.Series(
        {
            "initial_setup": "FRESH_SPIKE",
            "rvol_bucket": "10x+",
            "sma20_extension_bucket": None,
            "move_1d_bucket": None,
            "move_3d_bucket": None,
            "move_5d_bucket": None,
        }
    )

    event = {
        "initial_setup": "FRESH_SPIKE",
        "rvol_bucket": "3-5x",
        "sma20_extension_bucket": "+50%+",
        "move_1d_bucket": "+100%+",
        "move_3d_bucket": "+100%+",
        "move_5d_bucket": "+100%+",
    }

    assert not fold_matches_event(fold, event)


def test_match_historical_event_uses_only_same_test_year():
    event = pd.Series(
        {
            "date": "2022-06-15",
            "initial_setup": "PARABOLIC_EXTENSION",
            "rvol_bucket": "10x+",
            "sma20_extension_bucket": "+50%+",
            "move_1d_bucket": "+100%+",
            "move_3d_bucket": "+100%+",
            "move_5d_bucket": "+100%+",
        }
    )

    folds = pd.DataFrame(
        [
            {
                "segmentation": "initial_setup",
                "test_year": 2021,
                "horizon_days": 20,
                "train_rank": 1,
                "train_n": 100,
                "train_win_rate": 0.60,
                "train_mean_return": 5.0,
                "train_median_return": 4.0,
                "train_reversal_rate": 0.60,
                "train_edge_score": 100.0,
                "initial_setup": "PARABOLIC_EXTENSION",
                "rvol_bucket": None,
                "sma20_extension_bucket": None,
                "move_1d_bucket": None,
                "move_3d_bucket": None,
                "move_5d_bucket": None,
            },
            {
                "segmentation": "initial_setup",
                "test_year": 2022,
                "horizon_days": 20,
                "train_rank": 1,
                "train_n": 200,
                "train_win_rate": 0.65,
                "train_mean_return": 7.0,
                "train_median_return": 6.0,
                "train_reversal_rate": 0.65,
                "train_edge_score": 120.0,
                "initial_setup": "PARABOLIC_EXTENSION",
                "rvol_bucket": None,
                "sma20_extension_bucket": None,
                "move_1d_bucket": None,
                "move_3d_bucket": None,
                "move_5d_bucket": None,
            },
        ]
    )

    matches = match_historical_event(event, folds)

    assert len(matches) == 1
    assert matches[0].test_year == 2022
    assert matches[0].train_n == 200
    assert matches[0].train_edge_score == 120.0


def test_match_historical_event_can_limit_horizons():
    event = pd.Series(
        {
            "date": "2023-01-10",
            "initial_setup": "FRESH_SPIKE",
            "rvol_bucket": "2-3x",
            "sma20_extension_bucket": None,
            "move_1d_bucket": "+100%+",
            "move_3d_bucket": "+100%+",
            "move_5d_bucket": "+100%+",
        }
    )

    folds = pd.DataFrame(
        [
            {
                "segmentation": "initial_setup",
                "test_year": 2023,
                "horizon_days": horizon,
                "train_rank": 1,
                "train_n": 1000,
                "train_win_rate": 0.60,
                "train_mean_return": 2.0,
                "train_median_return": 1.5,
                "train_reversal_rate": 0.60,
                "train_edge_score": float(horizon),
                "initial_setup": "FRESH_SPIKE",
                "rvol_bucket": None,
                "sma20_extension_bucket": None,
                "move_1d_bucket": None,
                "move_3d_bucket": None,
                "move_5d_bucket": None,
            }
            for horizon in (1, 3, 5, 10, 20)
        ]
    )

    matches = match_historical_event(
        event,
        folds,
        horizons=(5, 20),
    )

    assert {match.horizon_days for match in matches} == {5, 20}


def test_best_match_by_horizon_selects_highest_edge():
    event = pd.Series(
        {
            "date": "2024-03-01",
            "initial_setup": "PARABOLIC_EXTENSION",
            "rvol_bucket": "10x+",
            "sma20_extension_bucket": "+50%+",
            "move_1d_bucket": "+100%+",
            "move_3d_bucket": "+100%+",
            "move_5d_bucket": "+100%+",
        }
    )

    folds = pd.DataFrame(
        [
            {
                "segmentation": "initial_setup",
                "test_year": 2024,
                "horizon_days": 20,
                "train_rank": 2,
                "train_n": 2000,
                "train_win_rate": 0.62,
                "train_mean_return": 8.0,
                "train_median_return": 7.0,
                "train_reversal_rate": 0.62,
                "train_edge_score": 100.0,
                "initial_setup": "PARABOLIC_EXTENSION",
                "rvol_bucket": None,
                "sma20_extension_bucket": None,
                "move_1d_bucket": None,
                "move_3d_bucket": None,
                "move_5d_bucket": None,
            },
            {
                "segmentation": "setup_x_rvol",
                "test_year": 2024,
                "horizon_days": 20,
                "train_rank": 1,
                "train_n": 800,
                "train_win_rate": 0.69,
                "train_mean_return": 15.0,
                "train_median_return": 12.0,
                "train_reversal_rate": 0.69,
                "train_edge_score": 250.0,
                "initial_setup": "PARABOLIC_EXTENSION",
                "rvol_bucket": "10x+",
                "sma20_extension_bucket": None,
                "move_1d_bucket": None,
                "move_3d_bucket": None,
                "move_5d_bucket": None,
            },
        ]
    )

    matches = match_historical_event(event, folds)
    best = best_match_by_horizon(matches)

    assert best[20].segmentation == "setup_x_rvol"
    assert best[20].train_edge_score == 250.0
    