from scanner.detector import detect_anomaly


def make_snapshot(
    **overrides,
):
    snapshot = {
        "price": 10.0,
        "dollar_volume": 5_000_000,

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

    snapshot.update(overrides)

    return snapshot


def test_normal_stock_is_not_candidate():
    result = detect_anomaly(
        make_snapshot()
    )

    assert result.eligible is True
    assert result.candidate is False
    assert result.triggers == []


def test_one_day_spike_triggers_candidate():
    result = detect_anomaly(
        make_snapshot(
            return_1d=18.0,
        )
    )

    assert result.eligible is True
    assert result.candidate is True

    assert any(
        "PRICE_1D_15" in trigger
        for trigger in result.triggers
    )


def test_three_day_spike_triggers_candidate():
    result = detect_anomaly(
        make_snapshot(
            return_3d=31.0,
        )
    )

    assert result.candidate is True

    assert any(
        "PRICE_3D_25" in trigger
        for trigger in result.triggers
    )


def test_five_day_spike_triggers_candidate():
    result = detect_anomaly(
        make_snapshot(
            return_5d=44.0,
        )
    )

    assert result.candidate is True

    assert any(
        "PRICE_5D_35" in trigger
        for trigger in result.triggers
    )


def test_high_rvol_triggers_candidate():
    result = detect_anomaly(
        make_snapshot(
            relative_volume=6.0,
        )
    )

    assert result.candidate is True

    assert any(
        "RVOL_4" in trigger
        for trigger in result.triggers
    )


def test_extreme_sma20_extension_triggers_candidate():
    result = detect_anomaly(
        make_snapshot(
            distance_sma_20_pct=32.0,
        )
    )

    assert result.candidate is True

    assert any(
        "SMA20_EXT_25" in trigger
        for trigger in result.triggers
    )


def test_low_price_fails_eligibility():
    result = detect_anomaly(
        make_snapshot(
            price=0.75,
            return_1d=50.0,
        )
    )

    assert result.eligible is False
    assert result.candidate is False

    assert any(
        "Price below" in failure
        for failure in result.failed_filters
    )


def test_low_dollar_volume_fails_eligibility():
    result = detect_anomaly(
        make_snapshot(
            dollar_volume=500_000,
            return_1d=50.0,
        )
    )

    assert result.eligible is False
    assert result.candidate is False

    assert any(
        "Dollar volume below" in failure
        for failure in result.failed_filters
    )


def test_supporting_evidence_does_not_create_candidate_alone():
    result = detect_anomaly(
        make_snapshot(
            rsi_14=82.0,
            relative_volume=2.5,
            atr_expansion=1.8,
            distance_sma_20_pct=18.0,
        )
    )

    assert result.eligible is True
    assert result.candidate is False

    assert len(result.evidence) >= 4


def test_candidate_can_have_multiple_triggers():
    result = detect_anomaly(
        make_snapshot(
            return_1d=25.0,
            return_3d=40.0,
            return_5d=65.0,
            relative_volume=8.0,
            distance_sma_20_pct=45.0,
        )
    )

    assert result.candidate is True
    assert len(result.triggers) == 5