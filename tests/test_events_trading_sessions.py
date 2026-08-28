import pytest

from scanner.events import (
    cluster_signal_events,
    make_signal_observation,
    trading_session_gap,
)


def make_obs(
    signal_date: str,
    setup: str = "PARABOLIC_EXTENSION",
    ticker: str = "TEST",
):
    return make_signal_observation(
        ticker=ticker,
        signal_date=signal_date,
        primary_setup=setup,
    )


def test_weekend_does_not_create_fake_gap():
    sessions = [
        "2026-01-02",  # Friday
        "2026-01-05",  # Monday
    ]

    gap = trading_session_gap(
        earlier_date="2026-01-02",
        later_date="2026-01-05",
        session_dates=sessions,
    )

    assert gap == 0


def test_one_missing_trading_session():
    sessions = [
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
    ]

    gap = trading_session_gap(
        earlier_date="2026-01-05",
        later_date="2026-01-07",
        session_dates=sessions,
    )

    assert gap == 1


def test_two_missing_trading_sessions():
    sessions = [
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
    ]

    gap = trading_session_gap(
        earlier_date="2026-01-05",
        later_date="2026-01-08",
        session_dates=sessions,
    )

    assert gap == 2


def test_market_holiday_does_not_count_as_session():
    sessions = [
        "2026-01-16",
        # 2026-01-19 omitted as market holiday
        "2026-01-20",
    ]

    gap = trading_session_gap(
        earlier_date="2026-01-16",
        later_date="2026-01-20",
        session_dates=sessions,
    )

    assert gap == 0


def test_same_event_when_gap_equals_limit():
    sessions = [
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
    ]

    observations = [
        make_obs(
            "2026-01-05",
            "FRESH_SPIKE",
        ),
        make_obs(
            "2026-01-08",
            "PARABOLIC_EXTENSION",
        ),
    ]

    events = cluster_signal_events(
        observations=observations,
        session_dates=sessions,
        max_gap_sessions=2,
    )

    assert len(events) == 1

    assert (
        events[0].observation_count
        == 2
    )


def test_new_event_when_gap_exceeds_limit():
    sessions = [
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
    ]

    observations = [
        make_obs(
            "2026-01-05",
            "FRESH_SPIKE",
        ),
        make_obs(
            "2026-01-08",
            "PARABOLIC_EXTENSION",
        ),
    ]

    events = cluster_signal_events(
        observations=observations,
        session_dates=sessions,
        max_gap_sessions=1,
    )

    assert len(events) == 2


def test_consecutive_sessions_same_event_with_zero_gap():
    sessions = [
        "2026-01-05",
        "2026-01-06",
    ]

    observations = [
        make_obs(
            "2026-01-05",
        ),
        make_obs(
            "2026-01-06",
        ),
    ]

    events = cluster_signal_events(
        observations=observations,
        session_dates=sessions,
        max_gap_sessions=0,
    )

    assert len(events) == 1


def test_missing_signal_date_from_calendar_raises():
    sessions = [
        "2026-01-05",
        "2026-01-06",
    ]

    observations = [
        make_obs(
            "2026-01-07",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="not present in session_dates",
    ):
        cluster_signal_events(
            observations=observations,
            session_dates=sessions,
            max_gap_sessions=3,
        )


def test_negative_gap_limit_rejected():
    observations = [
        make_obs(
            "2026-01-05",
        ),
    ]

    sessions = [
        "2026-01-05",
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


def test_multiple_tickers_create_separate_events():
    sessions = [
        "2026-01-05",
        "2026-01-06",
    ]

    observations = [
        make_obs(
            "2026-01-05",
            ticker="AAA",
        ),
        make_obs(
            "2026-01-06",
            ticker="BBB",
        ),
    ]

    events = cluster_signal_events(
        observations=observations,
        session_dates=sessions,
        max_gap_sessions=3,
    )

    assert len(events) == 2

    assert {
        event.ticker
        for event in events
    } == {
        "AAA",
        "BBB",
    }


def test_setup_path_still_tracks_transitions():
    sessions = [
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
    ]

    observations = [
        make_obs(
            "2026-01-05",
            "FRESH_SPIKE",
        ),
        make_obs(
            "2026-01-06",
            "PARABOLIC_EXTENSION",
        ),
        make_obs(
            "2026-01-07",
            "PARABOLIC_EXTENSION",
        ),
        make_obs(
            "2026-01-08",
            "POST_SPIKE_PULLBACK",
        ),
    ]

    events = cluster_signal_events(
        observations=observations,
        session_dates=sessions,
        max_gap_sessions=0,
    )

    assert len(events) == 1

    event = events[0]

    assert event.setup_path == [
        "FRESH_SPIKE",
        "PARABOLIC_EXTENSION",
        "POST_SPIKE_PULLBACK",
    ]

    assert len(
        event.transitions
    ) == 2

    assert (
        event.transitions[
            0
        ].transition_date.isoformat()
        == "2026-01-06"
    )

    assert (
        event.transitions[
            1
        ].transition_date.isoformat()
        == "2026-01-08"
    )