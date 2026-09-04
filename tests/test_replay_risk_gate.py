import pandas as pd

from scanner.replay_risk_gate import (
    RISK_GATES,
    apply_gate,
    build_gate_summary,
    profit_factor,
    tail_mean,
)


def gate_by_name(
    name: str,
):
    return next(
        gate
        for gate in RISK_GATES
        if gate.name == name
    )


def test_price_gate():
    df = pd.DataFrame(
        {
            "event_close": [
                0.50,
                1.50,
                5.00,
            ]
        }
    )

    result = apply_gate(
        df,
        gate_by_name(
            "PRICE_GE_2"
        ),
    )

    assert len(result) == 1

    assert (
        result.iloc[0][
            "event_close"
        ]
        == 5.0
    )


def test_rvol_gate():
    df = pd.DataFrame(
        {
            "rvol": [
                2.0,
                9.9,
                10.0,
                25.0,
            ]
        }
    )

    result = apply_gate(
        df,
        gate_by_name(
            "RVOL_LT_10"
        ),
    )

    assert len(result) == 2


def test_move5_gate():
    df = pd.DataFrame(
        {
            "return_5d_event": [
                20.0,
                99.0,
                100.0,
                500.0,
            ]
        }
    )

    result = apply_gate(
        df,
        gate_by_name(
            "MOVE5_LT_100"
        ),
    )

    assert len(result) == 2


def test_combined_gate():
    df = pd.DataFrame(
        [
            {
                "event_close": 5.0,
                "rvol": 5.0,
                "return_5d_event": 50.0,
            },
            {
                "event_close": 1.0,
                "rvol": 5.0,
                "return_5d_event": 50.0,
            },
            {
                "event_close": 5.0,
                "rvol": 20.0,
                "return_5d_event": 50.0,
            },
            {
                "event_close": 5.0,
                "rvol": 5.0,
                "return_5d_event": 150.0,
            },
        ]
    )

    result = apply_gate(
        df,
        gate_by_name(
            "PRICE_GE_2_RVOL_LT_10_MOVE5_LT_100"
        ),
    )

    assert len(result) == 1


def test_profit_factor():
    values = pd.Series(
        [
            10.0,
            20.0,
            -5.0,
            -5.0,
        ]
    )

    result = profit_factor(
        values
    )

    assert result == 3.0


def test_tail_mean():
    values = pd.Series(
        range(
            -100,
            100,
        )
    )

    result = tail_mean(
        values,
        0.05,
    )

    assert result < -90


def test_gate_summary_separates_development_and_validation():
    rows = []

    for year in (
        2021,
        2022,
        2023,
        2024,
        2025,
    ):
        rows.append(
            {
                "event_id": f"E{year}",
                "test_year": year,
                "horizon_days": 20,
                "event_close": 10.0,
                "rvol": 2.0,
                "return_5d_event": 50.0,
                "realized_contrarian_return": 10.0,
            }
        )

    df = pd.DataFrame(
        rows
    )

    result = build_gate_summary(
        df
    )

    periods = set(
        result["period"]
    )

    assert (
        "DEVELOPMENT_2021_2024"
        in periods
    )

    assert (
        "VALIDATION_2025"
        in periods
    )


def test_baseline_keeps_everything():
    df = pd.DataFrame(
        [
            {
                "event_close": 0.1,
                "rvol": 100.0,
                "return_5d_event": 500.0,
            },
            {
                "event_close": 50.0,
                "rvol": 1.0,
                "return_5d_event": 5.0,
            },
        ]
    )

    result = apply_gate(
        df,
        gate_by_name(
            "BASELINE"
        ),
    )

    assert len(result) == 2