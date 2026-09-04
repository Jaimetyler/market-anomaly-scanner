import pandas as pd

from scanner.live_research import (
    best_match_per_archetype,
    build_live_features,
    match_live_snapshot,
    move_bucket,
    rvol_bucket,
    sma20_extension_bucket,
)


def test_rvol_bucket_boundaries():
    assert rvol_bucket(0.99) == "<1x"
    assert rvol_bucket(1.0) == "1-1.5x"
    assert rvol_bucket(1.49) == "1-1.5x"
    assert rvol_bucket(1.5) == "1.5-2x"
    assert rvol_bucket(1.99) == "1.5-2x"
    assert rvol_bucket(2.0) == "2-3x"
    assert rvol_bucket(2.99) == "2-3x"
    assert rvol_bucket(3.0) == "3-5x"
    assert rvol_bucket(4.99) == "3-5x"
    assert rvol_bucket(5.0) == "5-10x"
    assert rvol_bucket(9.99) == "5-10x"
    assert rvol_bucket(10.0) == "10x+"


def test_sma20_extreme_buckets_only():
    assert sma20_extension_bucket(-60.0) == "<-50%"
    assert sma20_extension_bucket(-50.0) is None
    assert sma20_extension_bucket(0.0) is None
    assert sma20_extension_bucket(49.9) is None
    assert sma20_extension_bucket(50.0) == "+50%+"


def test_move_extreme_buckets_only():
    assert move_bucket(-60.0) == "<-50%"
    assert move_bucket(-50.0) is None
    assert move_bucket(0.0) is None
    assert move_bucket(99.9) is None
    assert move_bucket(100.0) == "+100%+"


def test_build_live_features():
    snapshot = {
        "price": 12.0,
        "relative_volume": 12.0,
        "distance_sma_20_pct": 60.0,
        "return_1d": 20.0,
        "return_3d": 80.0,
        "return_5d": 120.0,
    }

    features = build_live_features(
        snapshot=snapshot,
        initial_setup="PARABOLIC_EXTENSION",
    )

    assert (
        features["initial_setup"]
        == "PARABOLIC_EXTENSION"
    )
    assert features["rvol_bucket"] == "10x+"
    assert (
        features["sma20_extension_bucket"]
        == "+50%+"
    )
    assert features["move_1d_bucket"] is None
    assert features["move_3d_bucket"] is None
    assert features["move_5d_bucket"] == "+100%+"


def test_parabolic_live_match():
    members = pd.read_csv(
        "data/research/consolidation_reports/"
        "archetype_members.csv"
    )

    snapshot = {
        "price": 10.0,
        "relative_volume": 12.0,
        "distance_sma_20_pct": 60.0,
        "return_1d": 20.0,
        "return_3d": 80.0,
        "return_5d": 120.0,
    }

    matches = match_live_snapshot(
        snapshot=snapshot,
        initial_setup="PARABOLIC_EXTENSION",
        members_df=members,
    )

    archetypes = best_match_per_archetype(
        matches
    )

    ids = {
        match.archetype_id
        for match in archetypes
    }

    assert "A003" in ids
    assert "A006" in ids


def test_fresh_spike_live_match():
    members = pd.read_csv(
        "data/research/consolidation_reports/"
        "archetype_members.csv"
    )

    snapshot = {
        "price": 10.0,
        "relative_volume": 7.0,
        "distance_sma_20_pct": -20.0,
        "return_1d": 5.0,
        "return_3d": -20.0,
        "return_5d": -60.0,
    }

    matches = match_live_snapshot(
        snapshot=snapshot,
        initial_setup="FRESH_SPIKE",
        members_df=members,
    )

    archetypes = best_match_per_archetype(
        matches
    )

    ids = {
        match.archetype_id
        for match in archetypes
    }

    assert "A049" in ids


def test_best_match_per_archetype_collapses_duplicates():
    members = pd.read_csv(
        "data/research/consolidation_reports/"
        "archetype_members.csv"
    )

    snapshot = {
        "price": 10.0,
        "relative_volume": 12.0,
        "distance_sma_20_pct": 60.0,
        "return_1d": 20.0,
        "return_3d": 80.0,
        "return_5d": 120.0,
    }

    matches = match_live_snapshot(
        snapshot=snapshot,
        initial_setup="PARABOLIC_EXTENSION",
        members_df=members,
    )

    collapsed = best_match_per_archetype(
        matches
    )

    ids = [
        match.archetype_id
        for match in collapsed
    ]

    assert len(ids) == len(set(ids))
    assert len(collapsed) < len(matches)