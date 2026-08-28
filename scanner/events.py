from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable


@dataclass
class SignalObservation:
    ticker: str
    signal_date: date
    primary_setup: str
    tags: list[str] = field(default_factory=list)
    signal_price: float | None = None
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventTransition:
    from_setup: str
    to_setup: str
    transition_date: date


@dataclass
class AnomalyEvent:
    ticker: str
    start_date: date
    end_date: date
    observations: list[SignalObservation]
    initial_setup: str
    final_setup: str
    setup_path: list[str]
    transitions: list[EventTransition]

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def duration_calendar_days(self) -> int:
        return (
            self.end_date - self.start_date
        ).days

    @property
    def unique_setups(self) -> list[str]:
        seen = set()
        ordered = []

        for observation in self.observations:
            setup = observation.primary_setup

            if setup not in seen:
                seen.add(setup)
                ordered.append(setup)

        return ordered


def _normalize_date(
    value: str | date | datetime,
) -> date:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    return date.fromisoformat(value)


def _normalize_session_dates(
    session_dates: Iterable[
        str | date | datetime
    ],
) -> list[date]:
    """
    Normalize a supplied trading-session sequence.

    The caller should supply ACTUAL observed market
    sessions, normally derived directly from the
    historical daily bars being replayed.

    Weekends and market holidays therefore never
    need to be guessed inside this module.
    """

    normalized = {
        _normalize_date(value)
        for value in session_dates
    }

    return sorted(normalized)


def _build_session_index(
    session_dates: Iterable[
        str | date | datetime
    ],
) -> dict[date, int]:
    ordered = _normalize_session_dates(
        session_dates
    )

    if not ordered:
        raise ValueError(
            "session_dates cannot be empty."
        )

    return {
        session_date: index
        for index, session_date in enumerate(
            ordered
        )
    }


def make_signal_observation(
    ticker: str,
    signal_date: str | date | datetime,
    primary_setup: str,
    tags: list[str] | None = None,
    signal_price: float | None = None,
    values: dict[str, Any] | None = None,
) -> SignalObservation:
    return SignalObservation(
        ticker=ticker.upper().strip(),
        signal_date=_normalize_date(
            signal_date
        ),
        primary_setup=primary_setup,
        tags=list(tags or []),
        signal_price=signal_price,
        values=dict(values or {}),
    )


def _compress_setup_path(
    observations: list[
        SignalObservation
    ],
) -> list[str]:
    """
    Collapse consecutive duplicate states.

    Example:

    PARABOLIC
    PARABOLIC
    PARABOLIC
    PULLBACK
    PULLBACK
    PARABOLIC

    becomes:

    PARABOLIC
    PULLBACK
    PARABOLIC
    """

    path = []

    for observation in observations:
        setup = observation.primary_setup

        if (
            not path
            or path[-1] != setup
        ):
            path.append(setup)

    return path


def _build_transitions(
    observations: list[
        SignalObservation
    ],
) -> list[EventTransition]:
    transitions = []

    if len(observations) < 2:
        return transitions

    previous_setup = (
        observations[0].primary_setup
    )

    for observation in observations[1:]:
        current_setup = (
            observation.primary_setup
        )

        if current_setup != previous_setup:
            transitions.append(
                EventTransition(
                    from_setup=previous_setup,
                    to_setup=current_setup,
                    transition_date=(
                        observation.signal_date
                    ),
                )
            )

            previous_setup = current_setup

    return transitions


def build_event(
    observations: list[
        SignalObservation
    ],
) -> AnomalyEvent:
    if not observations:
        raise ValueError(
            "Cannot build an event without "
            "observations."
        )

    ordered = sorted(
        observations,
        key=lambda item: item.signal_date,
    )

    tickers = {
        item.ticker
        for item in ordered
    }

    if len(tickers) != 1:
        raise ValueError(
            "All observations in an event must "
            "belong to the same ticker."
        )

    setup_path = _compress_setup_path(
        ordered
    )

    transitions = _build_transitions(
        ordered
    )

    return AnomalyEvent(
        ticker=ordered[0].ticker,
        start_date=ordered[0].signal_date,
        end_date=ordered[-1].signal_date,
        observations=ordered,
        initial_setup=ordered[
            0
        ].primary_setup,
        final_setup=ordered[
            -1
        ].primary_setup,
        setup_path=setup_path,
        transitions=transitions,
    )


def trading_session_gap(
    earlier_date: str | date | datetime,
    later_date: str | date | datetime,
    session_dates: Iterable[
        str | date | datetime
    ],
) -> int:
    """
    Return the number of actual trading sessions
    BETWEEN two observations.

    Example:

        Friday signal
        Monday signal

    If Friday and Monday are consecutive trading
    sessions, the gap is 0.

    Example:

        Monday signal
        Thursday signal

    If Tuesday and Wednesday were trading sessions,
    the gap is 2.

    This is the quantity used by event clustering.
    """

    earlier = _normalize_date(
        earlier_date
    )

    later = _normalize_date(
        later_date
    )

    if later < earlier:
        raise ValueError(
            "later_date cannot be before "
            "earlier_date."
        )

    session_index = (
        _build_session_index(
            session_dates
        )
    )

    if earlier not in session_index:
        raise ValueError(
            f"{earlier} is not present in "
            "session_dates."
        )

    if later not in session_index:
        raise ValueError(
            f"{later} is not present in "
            "session_dates."
        )

    earlier_index = (
        session_index[earlier]
    )

    later_index = (
        session_index[later]
    )

    return max(
        0,
        later_index
        - earlier_index
        - 1,
    )


def cluster_signal_events(
    observations: list[
        SignalObservation
    ],
    session_dates: Iterable[
        str | date | datetime
    ],
    max_gap_sessions: int = 3,
) -> list[AnomalyEvent]:
    """
    Group daily anomaly observations into
    independent anomaly episodes.

    Event continuity is determined using the
    ACTUAL trading sessions supplied by the caller.

    max_gap_sessions means:

        maximum number of trading sessions with
        NO anomaly observation allowed between
        two consecutive anomaly observations.

    Examples:

    max_gap_sessions = 0

        Friday anomaly
        Monday anomaly

        Same event if Friday and Monday are
        consecutive trading sessions.

    max_gap_sessions = 2

        Monday anomaly
        Thursday anomaly

        Same event if Tuesday and Wednesday were
        the only two sessions between them.

    This intentionally avoids using calendar-day
    approximations. Weekends and exchange holidays
    are naturally handled because they are absent
    from session_dates.
    """

    if max_gap_sessions < 0:
        raise ValueError(
            "max_gap_sessions cannot be "
            "negative."
        )

    if not observations:
        return []

    session_index = (
        _build_session_index(
            session_dates
        )
    )

    by_ticker: dict[
        str,
        list[SignalObservation],
    ] = {}

    for observation in observations:
        if (
            observation.signal_date
            not in session_index
        ):
            raise ValueError(
                f"Signal date "
                f"{observation.signal_date} "
                f"for {observation.ticker} "
                f"is not present in "
                f"session_dates."
            )

        by_ticker.setdefault(
            observation.ticker,
            [],
        ).append(observation)

    events = []

    for (
        ticker,
        ticker_observations,
    ) in by_ticker.items():
        ordered = sorted(
            ticker_observations,
            key=lambda item: (
                item.signal_date
            ),
        )

        current = [
            ordered[0]
        ]

        for observation in ordered[1:]:
            previous = current[-1]

            previous_index = (
                session_index[
                    previous.signal_date
                ]
            )

            current_index = (
                session_index[
                    observation.signal_date
                ]
            )

            if (
                current_index
                <= previous_index
            ):
                raise ValueError(
                    "Trading-session order is "
                    "invalid for observations."
                )

            missing_sessions = (
                current_index
                - previous_index
                - 1
            )

            if (
                missing_sessions
                <= max_gap_sessions
            ):
                current.append(
                    observation
                )

            else:
                events.append(
                    build_event(
                        current
                    )
                )

                current = [
                    observation
                ]

        if current:
            events.append(
                build_event(
                    current
                )
            )

    return sorted(
        events,
        key=lambda event: (
            event.start_date,
            event.ticker,
        ),
    )