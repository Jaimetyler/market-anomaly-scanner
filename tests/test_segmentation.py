from __future__ import annotations

import pandas as pd
import pytest

from scanner.segmentation import (
    SegmentationConfig,
    add_research_buckets,
    rank_best_segments,
    summarize_segments,
)


def test_add_research_buckets() -> None:
    frame = pd.DataFrame(
        [
            {
                "event_close": 8.0,
                "rvol": 3.7,
                "sma20_extension_pct": 0.28,
                "return_5d_event": 0.42,
                "dollar_volume": 12_000_000,
                "market_cap": 900_000_000,
            }
        ]
    )

    result = add_research_buckets(frame)

    row = result.iloc[0]

    assert str(row["price_bucket"]) == "$5-$10"
    assert str(row["rvol_bucket"]) == "3-5x"
    assert str(row["sma20_extension_bucket"]) == "+20% to +30%"
    assert str(row["move_5d_bucket"]) == "+30% to +50%"
    assert str(row["dollar_volume_bucket"]) == "$10M-$25M"
    assert str(row["market_cap_bucket"]) == "$300M-$2B"


def test_custom_column_names() -> None:
    frame = pd.DataFrame(
        [
            {
                "px": 15.0,
                "rv": 2.2,
            }
        ]
    )

    result = add_research_buckets(
        frame,
        SegmentationConfig(
            price_col="px",
            rvol_col="rv",
        ),
    )

    assert str(result.iloc[0]["price_bucket"]) == "$10-$20"
    assert str(result.iloc[0]["rvol_bucket"]) == "2-3x"


def test_summarize_segments_filters_small_samples() -> None:
    frame = pd.DataFrame(
        [
            {
                "rvol_bucket": "3-5x",
                "outcome_available_5d": True,
                "contrarian_return_5d": 0.10,
                "reversed_5d": True,
                "contrarian_mfe_5d": 0.15,
                "contrarian_mae_5d": -0.03,
            },
            {
                "rvol_bucket": "3-5x",
                "outcome_available_5d": True,
                "contrarian_return_5d": 0.05,
                "reversed_5d": True,
                "contrarian_mfe_5d": 0.08,
                "contrarian_mae_5d": -0.02,
            },
            {
                "rvol_bucket": "1-1.5x",
                "outcome_available_5d": True,
                "contrarian_return_5d": -0.01,
                "reversed_5d": False,
                "contrarian_mfe_5d": 0.01,
                "contrarian_mae_5d": -0.04,
            },
        ]
    )

    result = summarize_segments(
        frame,
        group_by=["rvol_bucket"],
        horizons=(5,),
        min_n=2,
    )

    assert len(result) == 1
    assert result.iloc[0]["rvol_bucket"] == "3-5x"
    assert result.iloc[0]["n"] == 2
    assert result.iloc[0]["contrarian_win_rate"] == pytest.approx(1.0)
    assert result.iloc[0]["median_contrarian_return"] == pytest.approx(0.075)


def test_multi_column_segmentation() -> None:
    frame = pd.DataFrame(
        [
            {
                "price_bucket": "$5-$10",
                "rvol_bucket": "3-5x",
                "outcome_available_5d": True,
                "contrarian_return_5d": 0.12,
                "reversed_5d": True,
                "contrarian_mfe_5d": 0.18,
                "contrarian_mae_5d": -0.04,
            },
            {
                "price_bucket": "$5-$10",
                "rvol_bucket": "3-5x",
                "outcome_available_5d": True,
                "contrarian_return_5d": 0.08,
                "reversed_5d": True,
                "contrarian_mfe_5d": 0.15,
                "contrarian_mae_5d": -0.03,
            },
        ]
    )

    result = summarize_segments(
        frame,
        group_by=["price_bucket", "rvol_bucket"],
        horizons=(5,),
        min_n=1,
    )

    assert len(result) == 1
    assert result.iloc[0]["price_bucket"] == "$5-$10"
    assert result.iloc[0]["rvol_bucket"] == "3-5x"


def test_multi_column_segmentation_excludes_incomplete_keys() -> None:
    rows = []
    for rvol_bucket in ("3-5x", None):
        rows.append(
            {
                "initial_setup": "FRESH_SPIKE",
                "rvol_bucket": rvol_bucket,
                "outcome_available_5d": True,
                "contrarian_return_5d": 0.05,
                "reversed_5d": True,
                "contrarian_mfe_5d": 0.08,
                "contrarian_mae_5d": -0.01,
            }
        )

    result = summarize_segments(
        pd.DataFrame(rows),
        group_by=["initial_setup", "rvol_bucket"],
        horizons=(5,),
        min_n=1,
    )

    assert len(result) == 1
    assert result.iloc[0]["rvol_bucket"] == "3-5x"
    assert result.iloc[0]["n"] == 1


def test_rank_best_segments() -> None:
    summary = pd.DataFrame(
        [
            {
                "segment": "A",
                "horizon_days": 5,
                "n": 100,
                "contrarian_win_rate": 0.70,
                "median_contrarian_return": 0.08,
                "edge_score": 0.80,
            },
            {
                "segment": "B",
                "horizon_days": 5,
                "n": 200,
                "contrarian_win_rate": 0.60,
                "median_contrarian_return": 0.04,
                "edge_score": 0.57,
            },
            {
                "segment": "C",
                "horizon_days": 10,
                "n": 150,
                "contrarian_win_rate": 0.80,
                "median_contrarian_return": 0.10,
                "edge_score": 1.22,
            },
        ]
    )

    result = rank_best_segments(
        summary,
        horizon_days=5,
        top_n=10,
        min_win_rate=0.65,
        min_median_return=0.05,
    )

    assert result["segment"].tolist() == ["A"]
