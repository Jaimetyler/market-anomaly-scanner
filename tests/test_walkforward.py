from __future__ import annotations

import pandas as pd
import pytest

from scanner.walkforward import (
    WalkForwardConfig,
    summarize_walk_forward,
    walk_forward_validate,
)


def _row(
    *,
    date: str,
    segment: str,
    ret5: float,
) -> dict:
    return {
        "date": date,
        "segment": segment,
        "outcome_available_5d": True,
        "contrarian_return_5d": ret5,
        "reversed_5d": ret5 > 0,
        "contrarian_mfe_5d": float("nan"),
        "contrarian_mae_5d": float("nan"),
    }


def test_walkforward_uses_only_prior_years_for_training() -> None:
    rows = []

    for i in range(6):
        rows.append(
            _row(
                date=f"2019-01-{i + 1:02d}",
                segment="A",
                ret5=0.10,
            )
        )

    for i in range(6):
        rows.append(
            _row(
                date=f"2020-01-{i + 1:02d}",
                segment="A",
                ret5=-0.10,
            )
        )

    frame = pd.DataFrame(rows)

    result = walk_forward_validate(
        frame,
        group_by=["segment"],
        config=WalkForwardConfig(
            horizons=(5,),
            min_train_n=5,
            min_test_n=5,
            top_n_per_fold=5,
        ),
        first_test_year=2020,
    )

    assert len(result) == 1
    row = result.iloc[0]

    assert row["test_year"] == 2020
    assert row["train_start_year"] == 2019
    assert row["train_end_year"] == 2019
    assert row["train_median_return"] == pytest.approx(0.10)
    assert row["test_median_return"] == pytest.approx(-0.10)
    assert bool(row["validated"]) is False


def test_walkforward_validates_persistent_edge() -> None:
    rows = []

    for year in (2019, 2020, 2021):
        for i in range(6):
            rows.append(
                _row(
                    date=f"{year}-01-{i + 1:02d}",
                    segment="A",
                    ret5=0.08,
                )
            )

    frame = pd.DataFrame(rows)

    result = walk_forward_validate(
        frame,
        group_by=["segment"],
        config=WalkForwardConfig(
            horizons=(5,),
            min_train_n=5,
            min_test_n=5,
            top_n_per_fold=5,
        ),
        first_test_year=2020,
    )

    assert len(result) == 2
    assert result["validated"].all()
    assert result["test_year"].tolist() == [2020, 2021]


def test_top_n_is_frozen_from_train_ranking() -> None:
    rows = []

    for i in range(10):
        rows.append(
            _row(
                date=f"2019-01-{i + 1:02d}",
                segment="A",
                ret5=0.12,
            )
        )
        rows.append(
            _row(
                date=f"2019-02-{i + 1:02d}",
                segment="B",
                ret5=0.04,
            )
        )

    for i in range(10):
        rows.append(
            _row(
                date=f"2020-01-{i + 1:02d}",
                segment="A",
                ret5=-0.05,
            )
        )
        rows.append(
            _row(
                date=f"2020-02-{i + 1:02d}",
                segment="B",
                ret5=0.20,
            )
        )

    frame = pd.DataFrame(rows)

    result = walk_forward_validate(
        frame,
        group_by=["segment"],
        config=WalkForwardConfig(
            horizons=(5,),
            min_train_n=5,
            min_test_n=5,
            top_n_per_fold=1,
        ),
        first_test_year=2020,
    )

    assert len(result) == 1
    assert result.iloc[0]["segment"] == "A"


def test_min_test_n_blocks_validation() -> None:
    rows = []

    for i in range(10):
        rows.append(
            _row(
                date=f"2019-01-{i + 1:02d}",
                segment="A",
                ret5=0.10,
            )
        )

    for i in range(3):
        rows.append(
            _row(
                date=f"2020-01-{i + 1:02d}",
                segment="A",
                ret5=0.10,
            )
        )

    frame = pd.DataFrame(rows)

    result = walk_forward_validate(
        frame,
        group_by=["segment"],
        config=WalkForwardConfig(
            horizons=(5,),
            min_train_n=5,
            min_test_n=5,
            top_n_per_fold=5,
        ),
        first_test_year=2020,
    )

    assert result.iloc[0]["test_n"] == 3
    assert bool(result.iloc[0]["validated"]) is False


def test_summarize_walkforward_reports_persistence() -> None:
    frame = pd.DataFrame(
        [
            {
                "segment": "A",
                "horizon_days": 5,
                "test_year": 2020,
                "test_n": 10,
                "test_win_rate": 0.60,
                "test_median_return": 0.05,
                "validated": True,
            },
            {
                "segment": "A",
                "horizon_days": 5,
                "test_year": 2021,
                "test_n": 20,
                "test_win_rate": 0.70,
                "test_median_return": 0.08,
                "validated": True,
            },
            {
                "segment": "A",
                "horizon_days": 5,
                "test_year": 2022,
                "test_n": 10,
                "test_win_rate": 0.40,
                "test_median_return": -0.02,
                "validated": False,
            },
        ]
    )

    result = summarize_walk_forward(
        frame,
        group_by=["segment"],
    )

    row = result.iloc[0]

    assert row["folds_tested"] == 3
    assert row["validated_folds"] == 2
    assert row["validation_rate"] == pytest.approx(2 / 3)
    assert row["total_test_n"] == 40
    assert row["weighted_test_win_rate"] == pytest.approx(0.60)
