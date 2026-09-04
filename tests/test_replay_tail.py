import pandas as pd

from scanner.replay_tail import (
    add_tail_flags,
    attach_event_metadata,
    direction_label,
    focus_samples,
    repeated_tail_events,
    summarize_direction,
    worst_events,
)


def test_direction_labels():
    assert (
        direction_label(1)
        == "DIRECTION_POSITIVE"
    )

    assert (
        direction_label(-1)
        == "DIRECTION_NEGATIVE"
    )

    assert (
        direction_label(0)
        == "DIRECTION_ZERO"
    )


def test_attach_event_metadata():
    samples = pd.DataFrame(
        [
            {
                "event_id": "A",
                "ticker": "AAA",
            },
            {
                "event_id": "B",
                "ticker": "BBB",
            },
        ]
    )

    metadata = pd.DataFrame(
        [
            {
                "event_id": "A",
                "event_direction": 1,
                "initial_setup": "FRESH_SPIKE",
            },
            {
                "event_id": "B",
                "event_direction": -1,
                "initial_setup": "POST_SPIKE_PULLBACK",
            },
        ]
    )

    result = attach_event_metadata(
        samples,
        metadata,
    )

    assert (
        result.iloc[0]["event_direction"]
        == 1
    )

    assert (
        result.iloc[0]["initial_setup"]
        == "FRESH_SPIKE"
    )

    assert (
        result.iloc[1]["event_direction"]
        == -1
    )

    assert (
        result.iloc[1]["initial_setup"]
        == "POST_SPIKE_PULLBACK"
    )


def test_attach_event_metadata_preserves_existing_columns():
    samples = pd.DataFrame(
        [
            {
                "event_id": "A",
                "event_direction": 1,
                "initial_setup": "EXISTING_SETUP",
            }
        ]
    )

    metadata = pd.DataFrame(
        [
            {
                "event_id": "A",
                "event_direction": -1,
                "initial_setup": "OTHER_SETUP",
            }
        ]
    )

    result = attach_event_metadata(
        samples,
        metadata,
    )

    assert (
        result.iloc[0]["event_direction"]
        == 1
    )

    assert (
        result.iloc[0]["initial_setup"]
        == "EXISTING_SETUP"
    )


def test_focus_samples_selects_top_five():
    df = pd.DataFrame(
        [
            {
                "top_n": 5,
                "event_direction": 1,
                "realized_contrarian_return": 2.0,
            },
            {
                "top_n": 10,
                "event_direction": 1,
                "realized_contrarian_return": 2.0,
            },
        ]
    )

    result = focus_samples(
        df
    )

    assert len(result) == 1
    assert (
        result.iloc[0]["top_n"]
        == 5
    )


def test_tail_flags():
    df = pd.DataFrame(
        [
            {
                "event_direction": 1,
                "realized_contrarian_return": -250.0,
            },
            {
                "event_direction": -1,
                "realized_contrarian_return": -75.0,
            },
            {
                "event_direction": 1,
                "realized_contrarian_return": 10.0,
            },
        ]
    )

    result = add_tail_flags(
        df
    )

    assert bool(
        result.iloc[0][
            "loss_le_25"
        ]
    )

    assert bool(
        result.iloc[0][
            "loss_le_50"
        ]
    )

    assert bool(
        result.iloc[0][
            "loss_le_100"
        ]
    )

    assert bool(
        result.iloc[0][
            "loss_le_200"
        ]
    )

    assert not bool(
        result.iloc[1][
            "loss_le_100"
        ]
    )

    assert not bool(
        result.iloc[2][
            "loss_le_25"
        ]
    )


def test_direction_summary_separates_sides():
    df = pd.DataFrame(
        [
            {
                "event_id": "A",
                "horizon_days": 5,
                "event_direction": 1,
                "realized_contrarian_return": -100.0,
            },
            {
                "event_id": "B",
                "horizon_days": 5,
                "event_direction": 1,
                "realized_contrarian_return": 10.0,
            },
            {
                "event_id": "C",
                "horizon_days": 5,
                "event_direction": -1,
                "realized_contrarian_return": 20.0,
            },
        ]
    )

    df = add_tail_flags(
        df
    )

    summary = summarize_direction(
        df
    )

    positive = summary[
        summary[
            "direction_label"
        ]
        == "DIRECTION_POSITIVE"
    ].iloc[0]

    negative = summary[
        summary[
            "direction_label"
        ]
        == "DIRECTION_NEGATIVE"
    ].iloc[0]

    assert (
        positive["observations"]
        == 2
    )

    assert (
        negative["observations"]
        == 1
    )

    assert (
        positive["worst_return"]
        == -100.0
    )

    assert (
        negative["worst_return"]
        == 20.0
    )


def test_worst_events_are_sorted():
    df = pd.DataFrame(
        [
            {
                "event_id": "A",
                "ticker": "AAA",
                "horizon_days": 5,
                "realized_contrarian_return": -10.0,
            },
            {
                "event_id": "B",
                "ticker": "BBB",
                "horizon_days": 5,
                "realized_contrarian_return": -100.0,
            },
            {
                "event_id": "C",
                "ticker": "CCC",
                "horizon_days": 5,
                "realized_contrarian_return": -50.0,
            },
        ]
    )

    result = worst_events(
        df,
        limit_per_horizon=3,
    )

    assert list(
        result["ticker"]
    ) == [
        "BBB",
        "CCC",
        "AAA",
    ]


def test_repeated_tail_events():
    df = pd.DataFrame(
        [
            {
                "event_id": "A",
                "ticker": "AAA",
                "date": pd.Timestamp(
                    "2025-01-01"
                ),
                "direction_label": "DIRECTION_POSITIVE",
                "initial_setup": "FRESH_SPIKE",
                "horizon_days": 5,
                "realized_contrarian_return": -60.0,
            },
            {
                "event_id": "A",
                "ticker": "AAA",
                "date": pd.Timestamp(
                    "2025-01-01"
                ),
                "direction_label": "DIRECTION_POSITIVE",
                "initial_setup": "FRESH_SPIKE",
                "horizon_days": 10,
                "realized_contrarian_return": -100.0,
            },
            {
                "event_id": "B",
                "ticker": "BBB",
                "date": pd.Timestamp(
                    "2025-01-01"
                ),
                "direction_label": "DIRECTION_NEGATIVE",
                "initial_setup": "POST_SPIKE_PULLBACK",
                "horizon_days": 5,
                "realized_contrarian_return": 10.0,
            },
        ]
    )

    result = repeated_tail_events(
        df
    )

    assert len(result) == 1

    row = result.iloc[0]

    assert (
        row["event_id"]
        == "A"
    )

    assert (
        row["horizons_hit"]
        == 2
    )

    assert (
        row["worst_return"]
        == -100.0
    )