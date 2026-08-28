from scanner.classifier import (
    classify_setup,
)


def base_snapshot():
    return {
        "return_1d": 0.0,
        "return_3d": 0.0,
        "return_5d": 0.0,
        "return_10d": 0.0,
        "return_20d": 0.0,
        "relative_volume": 1.0,
        "rsi_14": 50.0,
        "distance_sma_20_pct": 0.0,
        "distance_sma_50_pct": 0.0,
        "atr_expansion": 1.0,
    }


def test_fresh_spike():
    snapshot = base_snapshot()

    snapshot["return_1d"] = 30.0

    result = classify_setup(
        snapshot
    )

    assert (
        "FRESH_SPIKE"
        in result.tags
    )


def test_volume_shock():
    snapshot = base_snapshot()

    snapshot[
        "relative_volume"
    ] = 8.0

    result = classify_setup(
        snapshot
    )

    assert (
        "VOLUME_SHOCK"
        in result.tags
    )


def test_multi_day_acceleration():
    snapshot = base_snapshot()

    snapshot["return_3d"] = 30.0
    snapshot["return_5d"] = 50.0

    result = classify_setup(
        snapshot
    )

    assert (
        "MULTI_DAY_ACCELERATION"
        in result.tags
    )


def test_parabolic_extension():
    snapshot = base_snapshot()

    snapshot["return_5d"] = 100.0

    snapshot[
        "distance_sma_20_pct"
    ] = 70.0

    result = classify_setup(
        snapshot
    )

    assert (
        "PARABOLIC_EXTENSION"
        in result.tags
    )


def test_post_spike_pullback():
    snapshot = base_snapshot()

    snapshot["return_1d"] = -12.0
    snapshot["return_5d"] = 80.0

    snapshot[
        "distance_sma_20_pct"
    ] = 40.0

    result = classify_setup(
        snapshot
    )

    assert (
        "POST_SPIKE_PULLBACK"
        in result.tags
    )

    assert (
        result.primary_setup
        == "POST_SPIKE_PULLBACK"
    )


def test_extended_trend():
    snapshot = base_snapshot()

    snapshot["return_20d"] = 80.0

    snapshot[
        "distance_sma_20_pct"
    ] = 35.0

    result = classify_setup(
        snapshot
    )

    assert (
        "EXTENDED_TREND"
        in result.tags
    )


def test_unclassified_anomaly():
    result = classify_setup(
        base_snapshot()
    )

    assert (
        result.primary_setup
        == "UNCLASSIFIED_ANOMALY"
    )

    assert result.tags == []