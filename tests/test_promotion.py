from __future__ import annotations

import pandas as pd

from scanner.promotion import (
    PromotionConfig,
    classify_evidence,
    combine_evidence,
)


GROUPING_MAP = {
    "setup_x_rvol": [
        "initial_setup",
        "rvol_bucket",
    ]
}


def test_classify_promote() -> None:
    row = pd.Series(
        {
            "globally_supported": True,
            "validation_rate": 0.75,
            "folds_tested": 4,
            "total_test_n": 100,
            "weighted_test_win_rate": 0.62,
            "median_test_median_return": 0.05,
        }
    )

    assert (
        classify_evidence(
            row,
            config=PromotionConfig(),
        )
        == "PROMOTE"
    )


def test_classify_watch_when_significant_but_weak_walkforward() -> None:
    row = pd.Series(
        {
            "globally_supported": True,
            "validation_rate": 0.25,
            "folds_tested": 4,
            "total_test_n": 20,
            "weighted_test_win_rate": 0.48,
            "median_test_median_return": -0.01,
        }
    )

    assert (
        classify_evidence(
            row,
            config=PromotionConfig(),
        )
        == "WATCH"
    )


def test_classify_reject() -> None:
    row = pd.Series(
        {
            "globally_supported": False,
            "validation_rate": 0.10,
            "folds_tested": 3,
            "total_test_n": 50,
            "weighted_test_win_rate": 0.45,
            "median_test_median_return": -0.02,
        }
    )

    assert (
        classify_evidence(
            row,
            config=PromotionConfig(),
        )
        == "REJECT"
    )


def test_combine_evidence_matches_same_segment() -> None:
    significance = pd.DataFrame(
        [
            {
                "segmentation": "setup_x_rvol",
                "horizon_days": 5,
                "initial_setup": "FRESH_SPIKE",
                "rvol_bucket": "3-5x",
                "globally_supported": True,
                "confidence_score": 1.2,
                "global_q_value": 0.01,
                "n": 200,
                "win_rate": 0.63,
                "median_contrarian_return": 0.07,
            }
        ]
    )

    persistence = pd.DataFrame(
        [
            {
                "segmentation": "setup_x_rvol",
                "horizon_days": 5,
                "initial_setup": "FRESH_SPIKE",
                "rvol_bucket": "3-5x",
                "folds_tested": 4,
                "validated_folds": 3,
                "validation_rate": 0.75,
                "total_test_n": 80,
                "weighted_test_win_rate": 0.60,
                "median_test_median_return": 0.04,
                "worst_test_median_return": 0.01,
            }
        ]
    )

    result = combine_evidence(
        significance=significance,
        persistence=persistence,
        grouping_map=GROUPING_MAP,
    )

    assert len(result) == 1
    row = result.iloc[0]

    assert row["disposition"] == "PROMOTE"
    assert row["validation_rate"] == 0.75
    assert row["weighted_test_win_rate"] == 0.60


def test_combine_evidence_leaves_unmatched_as_non_promoted() -> None:
    significance = pd.DataFrame(
        [
            {
                "segmentation": "setup_x_rvol",
                "horizon_days": 5,
                "initial_setup": "FRESH_SPIKE",
                "rvol_bucket": "3-5x",
                "globally_supported": True,
                "confidence_score": 1.2,
                "global_q_value": 0.01,
                "n": 200,
                "win_rate": 0.63,
                "median_contrarian_return": 0.07,
            }
        ]
    )

    persistence = pd.DataFrame(
        [
            {
                "segmentation": "setup_x_rvol",
                "horizon_days": 5,
                "initial_setup": "VOLUME_SHOCK",
                "rvol_bucket": "3-5x",
                "folds_tested": 4,
                "validated_folds": 3,
                "validation_rate": 0.75,
                "total_test_n": 80,
                "weighted_test_win_rate": 0.60,
                "median_test_median_return": 0.04,
                "worst_test_median_return": 0.01,
            }
        ]
    )

    result = combine_evidence(
        significance=significance,
        persistence=persistence,
        grouping_map=GROUPING_MAP,
    )

    assert len(result) == 1
    assert result.iloc[0]["disposition"] == "WATCH"


def test_combine_evidence_excludes_incomplete_segment_keys() -> None:
    significance = pd.DataFrame(
        [
            {
                "segmentation": "setup_x_rvol",
                "horizon_days": 5,
                "initial_setup": "FRESH_SPIKE",
                "rvol_bucket": None,
                "globally_supported": True,
                "confidence_score": 1.2,
                "global_q_value": 0.01,
                "n": 200,
                "win_rate": 0.63,
                "median_contrarian_return": 0.07,
            }
        ]
    )
    persistence = pd.DataFrame(
        [
            {
                "segmentation": "setup_x_rvol",
                "horizon_days": 5,
                "initial_setup": "FRESH_SPIKE",
                "rvol_bucket": None,
                "folds_tested": 4,
                "validation_rate": 0.75,
                "total_test_n": 80,
                "weighted_test_win_rate": 0.60,
                "median_test_median_return": 0.04,
            }
        ]
    )

    result = combine_evidence(
        significance=significance,
        persistence=persistence,
        grouping_map=GROUPING_MAP,
    )

    assert result.empty
