import pytest

from scanner.events import (
    build_event,
    cluster_signal_events,
    make_signal_observation,
)


def obs(
    signal_date: str,
    setup: str,
    ticker: str = "GME",
):
    return make_signal_observation(
        ticker=ticker,
        signal_date=signal_date,
        primary_setup=setup,
    )


def test_make_signal_observation_normalizes_ticker():
    observation = make_signal_observation(
        ticker="  gme  ",
        signal_date="2021-01-13",
        primary_setup="FRESH_SPIKE",
        tags=["FRESH_SPIKE"],
        signal_price=31.40,
        values={
            "return_1d": 57.4,
        },
    )

    assert observation.ticker == "GME"
    assert (
        observation.signal_date.isoformat()
        == "2021-01-13"
    )
    assert (
        observation.primary_setup
        == "FRESH_SPIKE"
    )
    assert observation.tags == [
        "FRESH_SPIKE"
    ]
    assert observation.signal_price == 31.40
    assert (
        observation.values[
            "return_1d"
        ]
        == 57.4
    )


def test_build_single_observation_event():
    observations = [
        obs(
            "2021-01-13",
            "FRESH_SPIKE",
        )
    ]

    event = build_event(
        observations
    )

    assert event.ticker == "GME"

    assert (
        event.start_date.isoformat()
        == "2021-01-13"
    )

    assert (
        event.end_date.isoformat()
        == "2021-01-13"
    )

    assert event.observation_count == 1

    assert (
        event.initial_setup
        == "FRESH_SPIKE"
    )

    assert (
        event.final_setup
        == "FRESH_SPIKE"
    )

    assert event.setup_path == [
        "FRESH_SPIKE"
    ]

    assert event.transitions == []


def test_consecutive_duplicate_setups_are_compressed():
    observations = [
        obs(
            "2021-01-13",
            "FRESH_SPIKE",
        ),
        obs(
            "2021-01-14",
            "PARABOLIC_EXTENSION",
        ),
        obs(
            "2021-01-15",
            "PARABOLIC_EXTENSION",
        ),
        obs(
            "2021-01-19",
            "POST_SPIKE_PULLBACK",
        ),
        obs(
            "2021-01-20",
            "POST_SPIKE_PULLBACK",
        ),
        obs(
            "2021-01-21",
            "PARABOLIC_EXTENSION",
        ),
    ]

    event = build_event(
        observations
    )

    assert event.setup_path == [
        "FRESH_SPIKE",
        "PARABOLIC_EXTENSION",
        "POST_SPIKE_PULLBACK",
        "PARABOLIC_EXTENSION",
    ]


def test_transition_dates_are_correct():
    observations = [
        obs(
            "2021-01-13",
            "FRESH_SPIKE",
        ),
        obs(
            "2021-01-14",
            "PARABOLIC_EXTENSION",
        ),
        obs(
            "2021-01-15",
            "PARABOLIC_EXTENSION",
        ),
        obs(
            "2021-01-19",
            "POST_SPIKE_PULLBACK",
        ),
    ]

    event = build_event(
        observations
    )

    assert len(
        event.transitions
    ) == 2

    first = event.transitions[0]

    assert (
        first.from_setup
        == "FRESH_SPIKE"
    )

    assert (
        first.to_setup
        == "PARABOLIC_EXTENSION"
    )

    assert (
        first.transition_date.isoformat()
        == "2021-01-14"
    )

    second = event.transitions[1]

    assert (
        second.from_setup
        == "PARABOLIC_EXTENSION"
    )

    assert (
        second.to_setup
        == "POST_SPIKE_PULLBACK"
    )

    assert (
        second.transition_date.isoformat()
        == "2021-01-19"
    )


def test_unique_setups_preserve_first_seen_order():
    observations = [
        obs(
            "2021-01-13",
            "FRESH_SPIKE",
        ),
        obs(
            "2021-01-14",
            "PARABOLIC_EXTENSION",
        ),
        obs(
            "2021-01-15",
            "POST_SPIKE_PULLBACK",
        ),
        obs(
            "2021-01-19",
            "PARABOLIC_EXTENSION",
        ),
    ]

    event = build_event(
        observations
    )

    assert event.unique_setups == [
        "FRESH_SPIKE",
        "PARABOLIC_EXTENSION",
        "POST_SPIKE_PULLBACK",
    ]


def test_large_gap_creates_new_event():
    observations = [
        obs(
            "2021-01-13",
            "FRESH_SPIKE",
        ),
        obs(
            "2021-01-14",
            "PARABOLIC_EXTENSION",
        ),

        # Long quiet period.
        obs(
            "2021-02-10",
            "FRESH_SPIKE",
        ),
    ]

    sessions = [
        "2021-01-13",
        "2021-01-14",
        "2021-01-15",
        "2021-01-19",
        "2021-01-20",
        "2021-01-21",
        "2021-01-22",
        "2021-01-25",
        "2021-01-26",
        "2021-01-27",
        "2021-01-28",
        "2021-01-29",
        "2021-02-01",
        "2021-02-02",
        "2021-02-03",
        "2021-02-04",
        "2021-02-05",
        "2021-02-08",
        "2021-02-09",
        "2021-02-10",
    ]

    events = cluster_signal_events(
        observations=observations,
        session_dates=sessions,
        max_gap_sessions=3,
    )

    assert len(events) == 2

    assert (
        events[0].start_date.isoformat()
        == "2021-01-13"
    )

    assert (
        events[0].end_date.isoformat()
        == "2021-01-14"
    )

    assert (
        events[1].start_date.isoformat()
        == "2021-02-10"
    )


def test_gme_style_sequence_clusters_as_one_event():
    observations = [
        obs(
            "2021-01-13",
            "FRESH_SPIKE",
        ),
        obs(
            "2021-01-14",
            "PARABOLIC_EXTENSION",
        ),
        obs(
            "2021-01-15",
            "POST_SPIKE_PULLBACK",
        ),
        obs(
            "2021-01-19",
            "PARABOLIC_EXTENSION",
        ),
        obs(
            "2021-01-20",
            "PARABOLIC_EXTENSION",
        ),
        obs(
            "2021-01-21",
            "PARABOLIC_EXTENSION",
        ),
        obs(
            "2021-01-22",
            "PARABOLIC_EXTENSION",
        ),
        obs(
            "2021-01-25",
            "PARABOLIC_EXTENSION",
        ),
        obs(
            "2021-01-26",
            "PARABOLIC_EXTENSION",
        ),
        obs(
            "2021-01-27",
            "PARABOLIC_EXTENSION",
        ),
        obs(
            "2021-01-28",
            "POST_SPIKE_PULLBACK",
        ),
        obs(
            "2021-01-29",
            "PARABOLIC_EXTENSION",
        ),
        obs(
            "2021-02-01",
            "POST_SPIKE_PULLBACK",
        ),
    ]

    sessions = [
        "2021-01-13",
        "2021-01-14",
        "2021-01-15",
        "2021-01-19",
        "2021-01-20",
        "2021-01-21",
        "2021-01-22",
        "2021-01-25",
        "2021-01-26",
        "2021-01-27",
        "2021-01-28",
        "2021-01-29",
        "2021-02-01",
    ]

    events = cluster_signal_events(
        observations=observations,
        session_dates=sessions,
        max_gap_sessions=3,
    )

    assert len(events) == 1

    event = events[0]

    assert event.ticker == "GME"

    assert (
        event.start_date.isoformat()
        == "2021-01-13"
    )

    assert (
        event.end_date.isoformat()
        == "2021-02-01"
    )

    assert (
        event.observation_count
        == len(observations)
    )

    assert (
        event.initial_setup
        == "FRESH_SPIKE"
    )

    assert (
        event.final_setup
        == "POST_SPIKE_PULLBACK"
    )

    assert event.setup_path == [
        "FRESH_SPIKE",
        "PARABOLIC_EXTENSION",
        "POST_SPIKE_PULLBACK",
        "PARABOLIC_EXTENSION",
        "POST_SPIKE_PULLBACK",
        "PARABOLIC_EXTENSION",
        "POST_SPIKE_PULLBACK",
    ]


def test_different_tickers_never_share_event():
    observations = [
        obs(
            "2021-01-13",
            "FRESH_SPIKE",
            ticker="GME",
        ),
        obs(
            "2021-01-14",
            "PARABOLIC_EXTENSION",
            ticker="GME",
        ),
        obs(
            "2021-01-13",
            "FRESH_SPIKE",
            ticker="AMC",
        ),
    ]

    sessions = [
        "2021-01-13",
        "2021-01-14",
    ]

    events = cluster_signal_events(
        observations=observations,
        session_dates=sessions,
        max_gap_sessions=3,
    )

    assert len(events) == 2

    tickers = {
        event.ticker
        for event in events
    }

    assert tickers == {
        "GME",
        "AMC",
    }


def test_empty_input_returns_empty_list():
    assert (
        cluster_signal_events(
            observations=[],
            session_dates=[],
        )
        == []
    )


def test_negative_gap_limit_rejected():
    observations = [
        obs(
            "2021-01-13",
            "FRESH_SPIKE",
        )
    ]

    sessions = [
        "2021-01-13",
    ]

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        cluster_signal_events(
            observations=observations,
            session_dates=sessions,
            max_gap_sessions=-1,
        )