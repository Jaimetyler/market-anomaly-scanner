import pandas as pd

from scanner.consolidation import (
    ConsolidationConfig,
    analyze_incremental_edges,
    assign_archetypes,
    build_overlap_pairs,
    build_signal_definitions,
    build_structural_relationships,
    classify_modifier_relationships,
    match_signal_events,
    overlap_statistics,
    summarize_archetypes,
    summarize_modifiers,
)


def test_overlap_statistics_identical_sets():
    a = {"A", "B", "C"}
    b = {"A", "B", "C"}

    stats = overlap_statistics(a, b)

    assert stats["intersection_n"] == 3
    assert stats["union_n"] == 3
    assert stats["jaccard"] == 1.0
    assert stats["containment"] == 1.0
    assert stats["larger_coverage"] == 1.0


def test_overlap_statistics_nested_sets():
    a = {"A", "B"}
    b = {"A", "B", "C", "D"}

    stats = overlap_statistics(a, b)

    assert stats["intersection_n"] == 2
    assert stats["jaccard"] == 0.5
    assert stats["containment"] == 1.0
    assert stats["larger_coverage"] == 0.5


def test_signal_matching_uses_segmentation_columns():
    events = pd.DataFrame(
        {
            "event_id": ["E1", "E2", "E3"],
            "initial_setup": [
                "FRESH_SPIKE",
                "FRESH_SPIKE",
                "PARABOLIC_EXTENSION",
            ],
            "rvol_bucket": ["10x+", "5-10x", "10x+"],
        }
    )

    signal = pd.Series(
        {
            "signal_id": "S0001",
            "segmentation": "setup_x_rvol",
            "initial_setup": "FRESH_SPIKE",
            "rvol_bucket": "10x+",
        }
    )

    assert match_signal_events(signal, events) == {"E1"}


def test_near_duplicates_collapse_but_nested_subset_does_not():
    signals = pd.DataFrame(
        {
            "signal_id": ["S0001", "S0002", "S0003"],
            "segmentation": [
                "rvol",
                "move_5d",
                "setup_x_rvol_x_move5",
            ],
            "horizon_days": [20, 20, 20],
            "rvol_bucket": ["10x+", None, "10x+"],
            "move_5d_bucket": [None, "+100%+", "+100%+"],
            "initial_setup": [None, None, "FRESH_SPIKE"],
            "promotion_score": [3.0, 2.0, 4.0],
            "n": [100, 100, 50],
            "win_rate": [0.70, 0.70, 0.75],
            "median_contrarian_return": [0.20, 0.20, 0.30],
            "validation_rate": [1.0, 1.0, 1.0],
            "weighted_test_win_rate": [0.70, 0.70, 0.76],
            "median_test_median_return": [0.20, 0.20, 0.31],
            "total_test_n": [80, 80, 45],
        }
    )

    event_sets = {
        "S0001": {f"E{i}" for i in range(100)},
        "S0002": {f"E{i}" for i in range(95)} | {"X1", "X2", "X3", "X4", "X5"},
        "S0003": {f"E{i}" for i in range(50)},
    }

    pairs = build_overlap_pairs(
        signals,
        event_sets,
        config=ConsolidationConfig(
            jaccard_threshold=0.85,
            nested_containment_threshold=0.90,
            nested_jaccard_ceiling=0.85,
        ),
    )

    members = assign_archetypes(signals, pairs)

    a1 = members.loc[
        members["signal_id"] == "S0001",
        "archetype_id",
    ].iloc[0]
    a2 = members.loc[
        members["signal_id"] == "S0002",
        "archetype_id",
    ].iloc[0]
    a3 = members.loc[
        members["signal_id"] == "S0003",
        "archetype_id",
    ].iloc[0]

    assert a1 == a2
    assert a3 != a1

    nested_pair = pairs.loc[
        (
            (pairs["signal_a"] == "S0001")
            & (pairs["signal_b"] == "S0003")
        )
        | (
            (pairs["signal_a"] == "S0003")
            & (pairs["signal_b"] == "S0001")
        )
    ].iloc[0]

    assert bool(nested_pair["nested_relationship"]) is True
    assert bool(nested_pair["near_duplicate"]) is False


def test_build_signal_definitions_filters_promote_only():
    frame = pd.DataFrame(
        {
            "segmentation": ["rvol", "rvol"],
            "rvol_bucket": ["10x+", "5-10x"],
            "horizon_days": [20, 20],
            "disposition": ["PROMOTE", "WATCH"],
        }
    )

    result = build_signal_definitions(frame)

    assert len(result) == 1
    assert result.iloc[0]["signal_id"].startswith("S_")
    assert result.iloc[0]["rvol_bucket"] == "10x+"


def test_archetype_summary_selects_strongest_representative():
    members = pd.DataFrame(
        {
            "signal_id": ["S0001", "S0002"],
            "archetype_id": ["A001", "A001"],
            "segmentation": ["rvol", "rvol"],
            "rvol_bucket": ["10x+", "10x+"],
            "horizon_days": [10, 20],
            "promotion_score": [1.0, 5.0],
            "n": [100, 95],
            "win_rate": [0.60, 0.70],
            "median_contrarian_return": [0.10, 0.20],
            "validation_rate": [0.8, 1.0],
            "weighted_test_win_rate": [0.62, 0.71],
            "median_test_median_return": [0.11, 0.23],
            "total_test_n": [80, 75],
        }
    )

    event_sets = {
        "S0001": {"E1", "E2", "E3"},
        "S0002": {"E1", "E2", "E3"},
    }

    summary = summarize_archetypes(members, event_sets)

    assert len(summary) == 1
    assert summary.iloc[0]["member_count"] == 2
    assert summary.iloc[0]["unique_event_count"] == 3
    assert summary.iloc[0]["representative_signal_id"] == "S0002"
    assert summary.iloc[0]["representative_horizon_days"] == 20


def _incremental_signals() -> pd.DataFrame:
    return build_signal_definitions(
        pd.DataFrame(
            [
                {
                    "segmentation": "initial_setup",
                    "initial_setup": "FRESH_SPIKE",
                    "horizon_days": 5,
                    "disposition": "PROMOTE",
                },
                {
                    "segmentation": "setup_x_rvol",
                    "initial_setup": "FRESH_SPIKE",
                    "rvol_bucket": "10x+",
                    "horizon_days": 5,
                    "disposition": "PROMOTE",
                },
                {
                    "segmentation": "setup_x_rvol",
                    "initial_setup": "FRESH_SPIKE",
                    "rvol_bucket": "10x+",
                    "horizon_days": 10,
                    "disposition": "PROMOTE",
                },
            ]
        )
    )


def test_structural_nesting_enforces_same_horizon_and_builds_remainder():
    signals = _incremental_signals()
    parent, child_5d, child_10d = signals.itertuples(index=False)
    event_sets = {
        parent.signal_id: {"E1", "E2", "E3", "E4"},
        child_5d.signal_id: {"E1", "E2"},
        child_10d.signal_id: {"E1", "E2"},
    }

    relationships = build_structural_relationships(signals, event_sets)

    assert len(relationships) == 1
    row = relationships.iloc[0]
    assert row["parent_signal_id"] == parent.signal_id
    assert row["child_signal_id"] == child_5d.signal_id
    assert row["added_columns"] == "rvol_bucket"
    assert row["parent_event_count"] - row["child_event_count"] == 2
    assert bool(row["integrity_check_passed"]) is True


def test_signal_ids_are_deterministic_across_row_order():
    frame = _incremental_signals().drop(columns="signal_id")
    first = build_signal_definitions(frame)
    second = build_signal_definitions(frame.iloc[::-1])

    first_ids = dict(zip(first["horizon_days"].astype(str) + first["segmentation"], first["signal_id"]))
    second_ids = dict(zip(second["horizon_days"].astype(str) + second["segmentation"], second["signal_id"]))
    assert first_ids == second_ids


def test_signal_definitions_reject_null_required_key():
    frame = pd.DataFrame(
        [{
            "segmentation": "setup_x_rvol",
            "initial_setup": "FRESH_SPIKE",
            "rvol_bucket": None,
            "horizon_days": 5,
            "disposition": "PROMOTE",
        }]
    )

    try:
        build_signal_definitions(frame)
    except ValueError as exc:
        assert "rvol_bucket" in str(exc)
    else:
        raise AssertionError("Expected null grouping key rejection")


def test_incremental_analysis_matched_years_and_worst_year_metrics():
    signals = _incremental_signals().iloc[:2]
    parent, child = signals.itertuples(index=False)
    event_sets = {
        parent.signal_id: {"E1", "E2", "E3", "E4", "E5", "E6"},
        child.signal_id: {"E1", "E2", "E4"},
    }
    relationships = build_structural_relationships(signals, event_sets)
    events = pd.DataFrame(
        {
            "event_id": ["E1", "E2", "E3", "E4", "E5", "E6"],
            "date": ["2021-01-01"] * 3 + ["2022-01-01"] * 3,
            "outcome_available_5d": [True] * 6,
            "contrarian_return_5d": [0.3, 0.2, -0.1, -0.2, 0.1, 0.2],
        }
    )

    aggregate, yearly = analyze_incremental_edges(
        relationships,
        events,
        event_sets,
        config=ConsolidationConfig(
            first_oos_year=2021,
            min_oos_n=1,
            permutation_samples=100,
            bootstrap_samples=100,
        ),
    )

    row = aggregate.iloc[0]
    assert row["historical_remainder_n"] == 3
    assert row["eligible_oos_years"] == 2
    assert row["comparable_oos_years"] == 2
    assert row["worst_oos_child_win_rate"] == 0.0
    assert row["worst_oos_child_median_return"] == -0.2
    assert len(yearly) == 2


def test_incremental_analysis_marks_zero_size_remainder():
    signals = _incremental_signals().iloc[:2]
    parent, child = signals.itertuples(index=False)
    event_sets = {
        parent.signal_id: {"E1", "E2"},
        child.signal_id: {"E1", "E2"},
    }
    relationships = build_structural_relationships(signals, event_sets)
    events = pd.DataFrame(
        {
            "event_id": ["E1", "E2"],
            "date": ["2021-01-01", "2021-01-02"],
            "outcome_available_5d": [True, True],
            "contrarian_return_5d": [0.1, 0.2],
        }
    )

    aggregate, _ = analyze_incremental_edges(
        relationships,
        events,
        event_sets,
        config=ConsolidationConfig(
            permutation_samples=100,
            bootstrap_samples=100,
        ),
    )

    assert bool(aggregate.iloc[0]["zero_size_remainder"]) is True
    assert bool(aggregate.iloc[0]["incrementally_supported"]) is False


def _classification_row(**overrides):
    row = {
        "relationship_depth": 1,
        "added_columns": "rvol_bucket",
        "added_conditions": "rvol_bucket=10x+",
        "parent_segmentation": "initial_setup",
        "parent_condition": "initial_setup=FRESH_SPIKE",
        "horizon_days": 5,
        "historical_child_n": 100,
        "historical_remainder_n": 100,
        "historical_sample_retention": 0.5,
        "comparable_oos_years": 5,
        "win_rate_fdr_significant": True,
        "median_delta_fdr_significant": True,
        "historical_win_rate_delta": 0.05,
        "historical_median_return_delta": 0.5,
        "oos_mean_win_rate_delta": 0.03,
        "oos_median_return_delta": 0.4,
        "oos_win_rate_improvement_consistency": 0.8,
        "oos_median_improvement_consistency": 0.8,
        "worst_oos_win_rate_delta": -0.02,
        "worst_oos_median_return_delta": -0.5,
    }
    row.update(overrides)
    return row


def test_modifier_classification_is_explicit_and_auditable():
    frame = pd.DataFrame(
        [
            _classification_row(),
            _classification_row(historical_child_n=20),
            _classification_row(
                oos_mean_win_rate_delta=-0.03,
                oos_median_return_delta=-0.4,
                oos_win_rate_improvement_consistency=0.2,
                oos_median_improvement_consistency=0.2,
            ),
            _classification_row(
                oos_mean_win_rate_delta=0.005,
                oos_median_return_delta=0.05,
            ),
            _classification_row(
                oos_win_rate_improvement_consistency=0.4,
            ),
            _classification_row(
                oos_mean_win_rate_delta=0.03,
                oos_median_return_delta=-0.1,
            ),
        ]
    )

    result = classify_modifier_relationships(
        frame, config=ConsolidationConfig()
    )

    assert result["modifier_classification"].tolist() == [
        "INFORMATIVE",
        "UNDERPOWERED",
        "DEGRADING",
        "REDUNDANT",
        "FRAGILE",
        "MIXED",
    ]
    assert result["classification_reason"].str.len().gt(0).all()


def test_modifier_summary_groups_contexts_and_horizons():
    immediate = pd.DataFrame(
        [
            {
                **_classification_row(),
                "modifier_classification": "INFORMATIVE",
            },
            {
                **_classification_row(
                    parent_condition="initial_setup=PARABOLIC_EXTENSION",
                    horizon_days=10,
                ),
                "modifier_classification": "REDUNDANT",
            },
        ]
    )

    summary = summarize_modifiers(immediate)

    assert len(summary) == 1
    assert summary.iloc[0]["relationship_count"] == 2
    assert summary.iloc[0]["parent_context_count"] == 2
    assert summary.iloc[0]["horizon_count"] == 2
    assert summary.iloc[0]["informative_rate"] == 0.5
