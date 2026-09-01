from __future__ import annotations

import pandas as pd
import pytest

from scanner.significance import (
    SignificanceConfig,
    benjamini_hochberg,
    bootstrap_median_interval,
    evaluate_segment_significance,
    exact_binomial_greater_pvalue,
    wilson_interval,
)


def test_exact_binomial_strong_edge_is_small_pvalue() -> None:
    p = exact_binomial_greater_pvalue(80, 100)
    assert p < 0.001


def test_wilson_interval_for_70_of_100_is_above_half() -> None:
    low, high = wilson_interval(70, 100)

    assert low > 0.50
    assert high < 1.0


def test_bootstrap_positive_median_interval() -> None:
    values = [0.03] * 60 + [0.10] * 40

    low, high = bootstrap_median_interval(
        values,
        samples=1000,
        random_seed=1,
    )

    assert low > 0
    assert high > 0


def test_benjamini_hochberg_flags_small_pvalues() -> None:
    pvalues = pd.Series(
        [0.0001, 0.001, 0.20, 0.80]
    )

    result = benjamini_hochberg(
        pvalues,
        alpha=0.05,
    )

    assert bool(result.iloc[0]["fdr_significant"]) is True
    assert bool(result.iloc[1]["fdr_significant"]) is True
    assert bool(result.iloc[2]["fdr_significant"]) is False
    assert bool(result.iloc[3]["fdr_significant"]) is False


def test_evaluate_segment_significance_supports_real_edge() -> None:
    rows = []

    # Strong segment A: 80 wins, 20 losses, positive median.
    for i in range(80):
        rows.append(
            {
                "segment": "A",
                "outcome_available_5d": True,
                "contrarian_return_5d": 0.08,
            }
        )

    for i in range(20):
        rows.append(
            {
                "segment": "A",
                "outcome_available_5d": True,
                "contrarian_return_5d": -0.03,
            }
        )

    # Noise segment B: 50/50.
    for i in range(50):
        rows.append(
            {
                "segment": "B",
                "outcome_available_5d": True,
                "contrarian_return_5d": 0.02,
            }
        )

    for i in range(50):
        rows.append(
            {
                "segment": "B",
                "outcome_available_5d": True,
                "contrarian_return_5d": -0.02,
            }
        )

    frame = pd.DataFrame(rows)

    result = evaluate_segment_significance(
        frame,
        group_by=["segment"],
        horizons=(5,),
        min_n=30,
        config=SignificanceConfig(
            bootstrap_samples=1000,
            random_seed=7,
        ),
    )

    a = result[result["segment"] == "A"].iloc[0]
    b = result[result["segment"] == "B"].iloc[0]

    assert a["win_rate"] == pytest.approx(0.80)
    assert bool(a["statistically_supported"]) is True
    assert bool(b["statistically_supported"]) is False
