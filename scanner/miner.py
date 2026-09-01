from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable

import pandas as pd

from scanner.classifier import classify_setup
from scanner.corporate_actions import (
    check_corporate_action_risk,
    get_stock_splits,
)
from scanner.data import get_daily_bars
from scanner.detector import detect_anomaly
from scanner.events import (
    cluster_signal_events,
    make_signal_observation,
)
from scanner.historical import (
    build_snapshot_as_of,
    calculate_historical_outcomes,
)
from scanner.historical_universe import (
    HistoricalTicker,
    RollingUniverseResult,
    get_rolling_historical_universe,
)
from scanner.universe_quality import (
    build_core_research_universe,
)


HISTORY_LOOKBACK_DAYS = 450
OUTCOME_LOOKFORWARD_DAYS = 45

DEFAULT_MAX_TICKERS = 25
HARD_MAX_TICKERS = 250

MARKET_CALENDAR_TICKER = "SPY"
EVENT_MAX_GAP_SESSIONS = 3

# Historical ticker mining is I/O-heavy because each ticker
# needs market bars and corporate-action data. Keep concurrency
# deliberately bounded so optimization cannot turn into an
# uncontrolled API fan-out.
TICKER_WORKERS = 8


@dataclass(frozen=True)
class HistoricalMinerConfig:
    start_date: date
    end_date: date
    max_tickers: int = DEFAULT_MAX_TICKERS


@dataclass
class HistoricalMinerResult:
    config: HistoricalMinerConfig

    universe_date: date

    raw_universe_count: int
    core_universe_count: int
    selected_ticker_count: int

    tickers_completed: int
    tickers_failed: int

    market_sessions: int
    sessions_tested: int

    anomaly_observations: int
    research_eligible_observations: int
    corporate_action_exclusions: int

    anomaly_events: int

    rows: list[dict[str, Any]]
    event_rows: list[dict[str, Any]]
    errors: list[dict[str, str]]


def normalize_date(
    value: str | date | datetime,
) -> date:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    return date.fromisoformat(value)


def validate_miner_config(
    config: HistoricalMinerConfig,
) -> None:
    if config.end_date < config.start_date:
        raise ValueError(
            "end_date cannot be before start_date."
        )

    if config.max_tickers <= 0:
        raise ValueError(
            "max_tickers must be greater than zero."
        )

    if config.max_tickers > HARD_MAX_TICKERS:
        raise ValueError(
            f"max_tickers cannot exceed "
            f"{HARD_MAX_TICKERS} in the bounded miner."
        )


def trading_dates_from_bars(
    bars: list[dict],
) -> list[str]:
    dates: list[str] = []

    for bar in bars:
        timestamp = bar.get("t")

        if timestamp is None:
            continue

        session_date = (
            pd.to_datetime(
                timestamp,
                unit="ms",
                utc=True,
            )
            .date()
            .isoformat()
        )

        dates.append(
            session_date
        )

    return sorted(
        set(dates)
    )


def signal_dates_in_window(
    bars: list[dict],
    start_date: date,
    end_date: date,
) -> list[str]:
    available_dates = (
        trading_dates_from_bars(
            bars
        )
    )

    return [
        session_date
        for session_date in available_dates
        if (
            start_date
            <= date.fromisoformat(
                session_date
            )
            <= end_date
        )
    ]


def get_market_session_calendar(
    *,
    start_date: date,
    end_date: date,
) -> list[str]:
    """
    Build one common U.S. trading-session calendar.

    IMPORTANT:

    We intentionally do NOT derive the event calendar
    from each individual ticker.

    A ticker can have missing bars because of:
      - IPO/listing timing
      - halts
      - suspensions
      - missing vendor data
      - other ticker-specific conditions

    Those missing bars must not erase elapsed market
    sessions when determining anomaly-event continuity.

    SPY is used here as the reference market calendar.
    """

    bars = get_daily_bars(
        ticker=MARKET_CALENDAR_TICKER,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    session_dates = signal_dates_in_window(
        bars=bars,
        start_date=start_date,
        end_date=end_date,
    )

    if not session_dates:
        raise RuntimeError(
            "Unable to build market-wide trading "
            "calendar from SPY."
        )

    return session_dates


def _safe_attr(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    if value is None:
        return default

    return getattr(
        value,
        name,
        default,
    )


def _string_list(
    values: Iterable[Any] | None,
) -> str:
    if not values:
        return ""

    return "|".join(
        str(value)
        for value in values
    )


def _split_pipe_string(
    value: Any,
) -> tuple[str, ...]:
    if value is None:
        return ()

    text = str(
        value
    ).strip()

    if not text:
        return ()

    return tuple(
        part
        for part in text.split("|")
        if part
    )


def analyze_historical_signal(
    *,
    ticker: str,
    bars: list[dict],
    signal_date: str,
    corporate_actions,
) -> dict[str, Any] | None:
    """
    Reconstruct one historical anomaly observation.

    Snapshot, detection, classification, and
    corporate-action annotation happen before
    forward outcomes are accessed.
    """

    try:
        snapshot = build_snapshot_as_of(
            bars=bars,
            signal_date=signal_date,
        )

    except (
        ValueError,
        IndexError,
        KeyError,
    ):
        return None

    detection = detect_anomaly(
        snapshot
    )

    if not detection.candidate:
        return None

    classification = classify_setup(
        snapshot
    )

    corporate_action_check = (
        check_corporate_action_risk(
            signal_date=signal_date,
            actions=corporate_actions,
            return_1d=snapshot.get(
                "return_1d"
            ),
            return_5d=snapshot.get(
                "return_5d"
            ),
            return_20d=snapshot.get(
                "return_20d"
            ),
        )
    )

    # --------------------------------------------------
    # INFORMATION WALL
    #
    # Nothing above this line has access to future
    # price sessions.
    # --------------------------------------------------

    outcomes = calculate_historical_outcomes(
        bars=bars,
        signal_date=signal_date,
    )

    return {
        "ticker": ticker,
        "signal_date": signal_date,
        "snapshot": snapshot,
        "detection": detection,
        "classification": classification,
        "corporate_action_check": (
            corporate_action_check
        ),
        "outcomes": outcomes,
    }


def build_research_row(
    result: dict[str, Any],
) -> dict[str, Any]:
    snapshot = result[
        "snapshot"
    ]

    detection = result[
        "detection"
    ]

    classification = result[
        "classification"
    ]

    corporate_action_check = result[
        "corporate_action_check"
    ]

    outcomes = result[
        "outcomes"
    ]

    research = outcomes.research
    executable = outcomes.executable

    excluded = bool(
        corporate_action_check.exclude_from_research
    )

    return {
        # ----------------------------------------------
        # IDENTITY
        # ----------------------------------------------
        "ticker": result[
            "ticker"
        ],
        "security_id": result.get(
            "security_id",
            result["ticker"],
        ),
        "signal_date": result[
            "signal_date"
        ],

        # ----------------------------------------------
        # SIGNAL STATE
        # ----------------------------------------------
        "primary_setup": (
            classification.primary_setup
        ),
        "setup_tags": _string_list(
            classification.tags
        ),
        "detection_triggers": _string_list(
            _safe_attr(
                detection,
                "triggers",
                (),
            )
        ),
        "detection_evidence": _string_list(
            _safe_attr(
                detection,
                "evidence",
                (),
            )
        ),

        # ----------------------------------------------
        # FEATURES
        # ----------------------------------------------
        "return_1d": snapshot.get(
            "return_1d"
        ),
        "return_3d": snapshot.get(
            "return_3d"
        ),
        "return_5d": snapshot.get(
            "return_5d"
        ),
        "return_10d": snapshot.get(
            "return_10d"
        ),
        "return_20d": snapshot.get(
            "return_20d"
        ),
        "relative_volume": snapshot.get(
            "relative_volume"
        ),
        "rsi_14": snapshot.get(
            "rsi_14"
        ),
        "distance_sma_20_pct": snapshot.get(
            "distance_sma_20_pct"
        ),
        "distance_sma_50_pct": snapshot.get(
            "distance_sma_50_pct"
        ),
        "atr_expansion": snapshot.get(
            "atr_expansion"
        ),

        # ----------------------------------------------
        # CORPORATE ACTIONS
        # ----------------------------------------------
        "ca_flagged": bool(
            corporate_action_check.flagged
        ),
        "ca_excluded": excluded,
        "ca_flags": _string_list(
            corporate_action_check.flags
        ),
        "research_eligible": (
            not excluded
        ),

        # ----------------------------------------------
        # ENTRY PRICES
        # ----------------------------------------------
        "signal_close": outcomes.signal_close,
        "next_open": outcomes.next_open,

        # ----------------------------------------------
        # SIGNAL-CLOSE RESEARCH OUTCOMES
        # ----------------------------------------------
        "research_return_1d": (
            _safe_attr(
                research,
                "return_1d",
            )
        ),
        "research_return_2d": (
            _safe_attr(
                research,
                "return_2d",
            )
        ),
        "research_return_3d": (
            _safe_attr(
                research,
                "return_3d",
            )
        ),
        "research_return_5d": (
            _safe_attr(
                research,
                "return_5d",
            )
        ),
        "research_return_10d": (
            _safe_attr(
                research,
                "return_10d",
            )
        ),
        "research_return_20d": (
            _safe_attr(
                research,
                "return_20d",
            )
        ),
        "research_max_future_gain_pct": (
            _safe_attr(
                research,
                "max_future_gain_pct",
            )
        ),
        "research_max_future_decline_pct": (
            _safe_attr(
                research,
                "max_future_decline_pct",
            )
        ),
        "research_mfe_pct": (
            _safe_attr(
                research,
                "mfe_pct",
            )
        ),
        "research_mae_pct": (
            _safe_attr(
                research,
                "mae_pct",
            )
        ),
        "research_sessions_to_max_gain": (
            _safe_attr(
                research,
                "sessions_to_max_gain",
            )
        ),
        "research_sessions_to_max_decline": (
            _safe_attr(
                research,
                "sessions_to_max_decline",
            )
        ),

        # ----------------------------------------------
        # NEXT-OPEN EXECUTABLE OUTCOMES
        # ----------------------------------------------
        "executable_return_1d": (
            _safe_attr(
                executable,
                "return_1d",
            )
        ),
        "executable_return_2d": (
            _safe_attr(
                executable,
                "return_2d",
            )
        ),
        "executable_return_3d": (
            _safe_attr(
                executable,
                "return_3d",
            )
        ),
        "executable_return_5d": (
            _safe_attr(
                executable,
                "return_5d",
            )
        ),
        "executable_return_10d": (
            _safe_attr(
                executable,
                "return_10d",
            )
        ),
        "executable_return_20d": (
            _safe_attr(
                executable,
                "return_20d",
            )
        ),
        "executable_max_future_gain_pct": (
            _safe_attr(
                executable,
                "max_future_gain_pct",
            )
        ),
        "executable_max_future_decline_pct": (
            _safe_attr(
                executable,
                "max_future_decline_pct",
            )
        ),
        "executable_mfe_pct": (
            _safe_attr(
                executable,
                "mfe_pct",
            )
        ),
        "executable_mae_pct": (
            _safe_attr(
                executable,
                "mae_pct",
            )
        ),
        "executable_sessions_to_max_gain": (
            _safe_attr(
                executable,
                "sessions_to_max_gain",
            )
        ),
        "executable_sessions_to_max_decline": (
            _safe_attr(
                executable,
                "sessions_to_max_decline",
            )
        ),
    }


def build_event_rows(
    *,
    rows: list[dict[str, Any]],
    session_dates: list[str],
) -> list[dict[str, Any]]:
    """
    Convert research-eligible DAILY observations into
    independent-ish anomaly EPISODES.

    Event outcomes are anchored to the FIRST anomaly
    observation in the event.

    We intentionally preserve both datasets:

      observation rows
      event rows

    Daily rows remain useful for state-transition and
    path analysis.

    Event rows become the primary unit for statistical
    analog research.
    """

    eligible_rows = [
        row
        for row in rows
        if row.get(
            "research_eligible",
            False,
        )
    ]

    if not eligible_rows:
        return []

    # Whole-market history can contain ticker changes/reuse and multiple
    # historical aliases for one stable security identity. Canonicalize rows
    # by stable identity + signal date, then sort them before clustering so
    # every security moves monotonically through the trading-session calendar.
    canonical_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for row in eligible_rows:
        ticker = str(row["ticker"])
        security_id = str(row.get("security_id") or ticker)
        signal_date = str(row["signal_date"])
        key = (security_id, signal_date)

        existing = canonical_by_key.get(key)
        if existing is None:
            canonical_by_key[key] = row
            continue

        # Deterministic alias choice for duplicate identity/date observations.
        if str(row.get("ticker") or "") < str(existing.get("ticker") or ""):
            canonical_by_key[key] = row

    eligible_rows = sorted(
        canonical_by_key.values(),
        key=lambda row: (
            str(row.get("security_id") or row.get("ticker") or ""),
            str(row.get("signal_date") or ""),
            str(row.get("ticker") or ""),
        ),
    )

    observations = []

    row_lookup: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for row in eligible_rows:
        ticker = str(
            row["ticker"]
        )
        security_id = str(
            row.get("security_id")
            or ticker
        )

        signal_date = str(
            row["signal_date"]
        )

        row_lookup[
            (
                security_id,
                signal_date,
            )
        ] = row

        observations.append(
            make_signal_observation(
                # Event clustering is intentionally keyed by
                # stable security identity rather than ticker.
                # The real ticker is restored from entry_row
                # when the event row is emitted below.
                ticker=security_id,
                signal_date=signal_date,
                primary_setup=row[
                    "primary_setup"
                ],
                tags=_split_pipe_string(
                    row.get(
                        "setup_tags"
                    )
                ),
                signal_price=row.get(
                    "signal_close"
                ),
                values={
                    "return_1d": row.get(
                        "return_1d"
                    ),
                    "return_3d": row.get(
                        "return_3d"
                    ),
                    "return_5d": row.get(
                        "return_5d"
                    ),
                    "return_10d": row.get(
                        "return_10d"
                    ),
                    "return_20d": row.get(
                        "return_20d"
                    ),
                    "relative_volume": row.get(
                        "relative_volume"
                    ),
                    "rsi_14": row.get(
                        "rsi_14"
                    ),
                    "distance_sma_20_pct": row.get(
                        "distance_sma_20_pct"
                    ),
                    "distance_sma_50_pct": row.get(
                        "distance_sma_50_pct"
                    ),
                    "atr_expansion": row.get(
                        "atr_expansion"
                    ),
                },
            )
        )

    events = cluster_signal_events(
        observations=observations,
        session_dates=session_dates,
        max_gap_sessions=(
            EVENT_MAX_GAP_SESSIONS
        ),
    )

    event_rows: list[
        dict[str, Any]
    ] = []

    for event_index, event in enumerate(
        events,
        start=1,
    ):
        security_id = str(
            event.ticker
        )

        start_date = str(
            event.start_date
        )

        end_date = str(
            event.end_date
        )

        entry_row = row_lookup[
            (
                security_id,
                start_date,
            )
        ]
        ticker = str(entry_row["ticker"])

        transitions = _string_list(
            (
                f"{transition.transition_date}:"
                f"{transition.from_setup}"
                f"->{transition.to_setup}"
            )
            for transition in event.transitions
        )

        event_rows.append(
            {
                # --------------------------------------
                # EVENT IDENTITY
                # --------------------------------------
                "event_number": event_index,
                "ticker": ticker,
                "security_id": security_id,
                "event_start_date": start_date,
                "event_end_date": end_date,
                "observation_count": (
                    event.observation_count
                ),
                "duration_calendar_days": (
                    event.duration_calendar_days
                ),

                # --------------------------------------
                # STATE PATH
                # --------------------------------------
                "initial_setup": (
                    event.initial_setup
                ),
                "final_setup": (
                    event.final_setup
                ),
                "setup_path": _string_list(
                    event.setup_path
                ),
                "transition_count": len(
                    event.transitions
                ),
                "transitions": transitions,

                # --------------------------------------
                # ENTRY SIGNAL FEATURES
                #
                # These are ONLY from the first
                # observation in the event.
                # --------------------------------------
                "entry_signal_date": (
                    entry_row[
                        "signal_date"
                    ]
                ),
                "entry_signal_close": (
                    entry_row.get(
                        "signal_close"
                    )
                ),
                "entry_next_open": (
                    entry_row.get(
                        "next_open"
                    )
                ),
                "entry_return_1d": (
                    entry_row.get(
                        "return_1d"
                    )
                ),
                "entry_return_3d": (
                    entry_row.get(
                        "return_3d"
                    )
                ),
                "entry_return_5d": (
                    entry_row.get(
                        "return_5d"
                    )
                ),
                "entry_return_10d": (
                    entry_row.get(
                        "return_10d"
                    )
                ),
                "entry_return_20d": (
                    entry_row.get(
                        "return_20d"
                    )
                ),
                "entry_relative_volume": (
                    entry_row.get(
                        "relative_volume"
                    )
                ),
                "entry_rsi_14": (
                    entry_row.get(
                        "rsi_14"
                    )
                ),
                "entry_distance_sma_20_pct": (
                    entry_row.get(
                        "distance_sma_20_pct"
                    )
                ),
                "entry_distance_sma_50_pct": (
                    entry_row.get(
                        "distance_sma_50_pct"
                    )
                ),
                "entry_atr_expansion": (
                    entry_row.get(
                        "atr_expansion"
                    )
                ),
                "entry_setup_tags": (
                    entry_row.get(
                        "setup_tags"
                    )
                ),
                "entry_detection_triggers": (
                    entry_row.get(
                        "detection_triggers"
                    )
                ),
                "entry_detection_evidence": (
                    entry_row.get(
                        "detection_evidence"
                    )
                ),

                # --------------------------------------
                # EVENT OUTCOMES
                #
                # Anchored from first signal close.
                # --------------------------------------
                "research_return_1d": (
                    entry_row.get(
                        "research_return_1d"
                    )
                ),
                "research_return_2d": (
                    entry_row.get(
                        "research_return_2d"
                    )
                ),
                "research_return_3d": (
                    entry_row.get(
                        "research_return_3d"
                    )
                ),
                "research_return_5d": (
                    entry_row.get(
                        "research_return_5d"
                    )
                ),
                "research_return_10d": (
                    entry_row.get(
                        "research_return_10d"
                    )
                ),
                "research_return_20d": (
                    entry_row.get(
                        "research_return_20d"
                    )
                ),
                "research_max_future_gain_pct": (
                    entry_row.get(
                        "research_max_future_gain_pct"
                    )
                ),
                "research_max_future_decline_pct": (
                    entry_row.get(
                        "research_max_future_decline_pct"
                    )
                ),
                "research_mfe_pct": (
                    entry_row.get(
                        "research_mfe_pct"
                    )
                ),
                "research_mae_pct": (
                    entry_row.get(
                        "research_mae_pct"
                    )
                ),
                "research_sessions_to_max_gain": (
                    entry_row.get(
                        "research_sessions_to_max_gain"
                    )
                ),
                "research_sessions_to_max_decline": (
                    entry_row.get(
                        "research_sessions_to_max_decline"
                    )
                ),

                # --------------------------------------
                # EVENT EXECUTABLE OUTCOMES
                #
                # Anchored from next open after first
                # signal.
                # --------------------------------------
                "executable_return_1d": (
                    entry_row.get(
                        "executable_return_1d"
                    )
                ),
                "executable_return_2d": (
                    entry_row.get(
                        "executable_return_2d"
                    )
                ),
                "executable_return_3d": (
                    entry_row.get(
                        "executable_return_3d"
                    )
                ),
                "executable_return_5d": (
                    entry_row.get(
                        "executable_return_5d"
                    )
                ),
                "executable_return_10d": (
                    entry_row.get(
                        "executable_return_10d"
                    )
                ),
                "executable_return_20d": (
                    entry_row.get(
                        "executable_return_20d"
                    )
                ),
                "executable_max_future_gain_pct": (
                    entry_row.get(
                        "executable_max_future_gain_pct"
                    )
                ),
                "executable_max_future_decline_pct": (
                    entry_row.get(
                        "executable_max_future_decline_pct"
                    )
                ),
                "executable_mfe_pct": (
                    entry_row.get(
                        "executable_mfe_pct"
                    )
                ),
                "executable_mae_pct": (
                    entry_row.get(
                        "executable_mae_pct"
                    )
                ),
                "executable_sessions_to_max_gain": (
                    entry_row.get(
                        "executable_sessions_to_max_gain"
                    )
                ),
                "executable_sessions_to_max_decline": (
                    entry_row.get(
                        "executable_sessions_to_max_decline"
                    )
                ),
            }
        )

    return event_rows


def build_rolling_research_selection(
    *,
    session_dates: list[str],
    max_tickers: int,
    force_universe_refresh: bool = False,
) -> tuple[
    list[HistoricalTicker],
    RollingUniverseResult,
    int,
    int,
]:
    """
    Build a bounded ticker cohort from the UNION of
    point-in-time core research universes across the
    requested market sessions.

    Membership is still enforced separately for every
    signal date. Being selected for the bounded cohort
    does NOT make a ticker eligible before it actually
    entered the point-in-time universe or after it left.

    The bounded miner keeps alphabetical selection so
    repeated validation runs remain deterministic.
    """

    rolling = get_rolling_historical_universe(
        session_dates=session_dates,
        force_refresh=force_universe_refresh,
    )

    raw_union: dict[
        tuple[str, str | None],
        HistoricalTicker,
    ] = {}

    core_union: dict[
        tuple[str, str | None],
        HistoricalTicker,
    ] = {}

    for session_date in session_dates:
        snapshot = rolling.snapshots[
            session_date
        ]

        for security in snapshot:
            identity = (
                security.ticker,
                security.composite_figi,
            )

            if identity not in raw_union:
                raw_union[
                    identity
                ] = security

        core_snapshot = (
            build_core_research_universe(
                snapshot
            )
        )

        for security in core_snapshot:
            identity = (
                security.ticker,
                security.composite_figi,
            )

            if identity not in core_union:
                core_union[
                    identity
                ] = security

    core = sorted(
        core_union.values(),
        key=lambda item: (
            item.ticker,
            item.composite_figi
            or "",
        ),
    )

    selected = core[
        :max_tickers
    ]

    return (
        selected,
        rolling,
        len(raw_union),
        len(core_union),
    )


def security_identity(
    security: HistoricalTicker,
) -> str:
    """
    Stable-enough identity for historical research.

    Prefer Composite FIGI, then share-class FIGI, then CIK.
    Ticker is only the last-resort fallback because ticker symbols
    can be reused by different securities across history.
    """

    for namespace, value in (
        ("FIGI", security.composite_figi),
        ("SHARE_CLASS_FIGI", security.share_class_figi),
        ("CIK", security.cik),
    ):
        normalized = str(value or "").strip()
        if normalized:
            return f"{namespace}:{normalized}"

    return f"TICKER:{security.ticker.upper().strip()}"


def membership_dates_for_security(
    *,
    security: HistoricalTicker,
    session_dates: list[str],
    rolling_universe: RollingUniverseResult,
) -> set[str]:
    """
    Return point-in-time membership for one historical security identity.

    If Massive reports the same ticker for multiple distinct identities on
    the same session, that session is ambiguous at ticker-bar granularity.
    We conservatively exclude it from every identity rather than
    double-counting one ticker's price bar as two securities.
    """

    target_ticker = security.ticker.upper().strip()
    target_identity = security_identity(security)
    eligible_dates: set[str] = set()

    for session_date in session_dates:
        snapshot = rolling_universe.snapshots.get(session_date)
        if snapshot is None:
            raise KeyError(
                "No universe snapshot exists for "
                f"{session_date}."
            )

        same_ticker = [
            candidate
            for candidate in snapshot
            if candidate.ticker.upper().strip() == target_ticker
        ]

        identities = {
            security_identity(candidate)
            for candidate in same_ticker
        }

        if len(identities) != 1:
            # 0 = not a member that day.
            # >1 = ambiguous ticker-to-security mapping; skip safely.
            continue

        if target_identity in identities:
            eligible_dates.add(session_date)

    return eligible_dates


def membership_dates_for_ticker(
    *,
    ticker: str,
    session_dates: list[str],
    rolling_universe: RollingUniverseResult,
) -> set[str]:
    """
    Return exactly the supplied market sessions on
    which ticker belonged to the point-in-time
    historical universe.

    Production RollingUniverseResult objects contain a
    precomputed inverse membership index, so this lookup
    no longer rescans ~5,000 securities for every ticker
    on every market session.

    A snapshot-scan fallback is retained for older or
    manually constructed RollingUniverseResult objects.
    """

    target = ticker.upper().strip()

    membership_index = (
        rolling_universe
        .membership_dates_by_ticker
    )

    if membership_index:
        indexed_dates = membership_index.get(
            target,
            frozenset(),
        )

        return {
            session_date
            for session_date in session_dates
            if session_date in indexed_dates
        }

    # Backward-compatible fallback used only when no inverse
    # index is present. Production rolling universes build the
    # index once while the daily snapshots are already being
    # traversed.
    eligible_dates: set[str] = set()

    for session_date in session_dates:
        snapshot = (
            rolling_universe.snapshots[
                session_date
            ]
        )

        if any(
            security.ticker == target
            for security in snapshot
        ):
            eligible_dates.add(
                session_date
            )

    return eligible_dates

def mine_ticker(
    *,
    ticker: str,
    start_date: date,
    end_date: date,
    eligible_session_dates: set[str] | None = None,
    security_id: str | None = None,
) -> tuple[
    list[dict[str, Any]],
    int,
]:
    fetch_start = (
        start_date
        - timedelta(
            days=HISTORY_LOOKBACK_DAYS
        )
    )

    fetch_end = (
        end_date
        + timedelta(
            days=OUTCOME_LOOKFORWARD_DAYS
        )
    )

    bars = get_daily_bars(
        ticker=ticker,
        start_date=fetch_start.isoformat(),
        end_date=fetch_end.isoformat(),
    )

    if not bars:
        return (
            [],
            0,
        )

    signal_dates = signal_dates_in_window(
        bars=bars,
        start_date=start_date,
        end_date=end_date,
    )

    if eligible_session_dates is not None:
        signal_dates = [
            signal_date
            for signal_date in signal_dates
            if signal_date in eligible_session_dates
        ]

    corporate_actions = get_stock_splits(
        ticker=ticker
    )

    rows: list[
        dict[str, Any]
    ] = []

    for signal_date in signal_dates:
        result = analyze_historical_signal(
            ticker=ticker,
            bars=bars,
            signal_date=signal_date,
            corporate_actions=corporate_actions,
        )

        if result is None:
            continue

        if security_id is not None:
            result["security_id"] = security_id

        rows.append(
            build_research_row(
                result
            )
        )

    return (
        rows,
        len(signal_dates),
    )



def _mine_selected_ticker(
    *,
    ticker: str,
    security_id: str,
    start_date: date,
    end_date: date,
    eligible_session_dates: set[str],
) -> tuple[
    list[dict[str, Any]],
    int,
]:
    """
    Worker wrapper for one selected ticker.

    Keeping the worker isolated makes concurrent execution
    side-effect-light: it returns rows/session counts and the
    main thread remains responsible for deterministic merging,
    counters, error collection, and console output.
    """

    return mine_ticker(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        eligible_session_dates=(
            eligible_session_dates
        ),
        security_id=security_id,
    )

def mine_history(
    config: HistoricalMinerConfig,
    *,
    force_universe_refresh: bool = False,
) -> HistoricalMinerResult:
    validate_miner_config(
        config
    )

    universe_date = (
        config.start_date
    )

    print(
        "Building market-wide session calendar..."
    )

    market_session_dates = (
        get_market_session_calendar(
            start_date=config.start_date,
            end_date=config.end_date,
        )
    )

    print(
        f"Market sessions: "
        f"{len(market_session_dates):,}"
    )

    print()
    print(
        "Building rolling point-in-time universe..."
    )

    (
        selected,
        rolling_universe,
        raw_universe_count,
        core_universe_count,
    ) = build_rolling_research_selection(
        session_dates=market_session_dates,
        max_tickers=config.max_tickers,
        force_universe_refresh=(
            force_universe_refresh
        ),
    )

    print(
        f"Universe snapshots from API:   "
        f"{rolling_universe.fetched_snapshots:,}"
    )
    print(
        f"Universe snapshots from cache: "
        f"{rolling_universe.cached_snapshots:,}"
    )
    print()

    rows: list[
        dict[str, Any]
    ] = []

    errors: list[
        dict[str, str]
    ] = []

    sessions_tested = 0
    tickers_completed = 0

    total = len(
        selected
    )

    worker_count = min(
        TICKER_WORKERS,
        total,
    )

    print(
        f"Ticker workers: {worker_count} "
        f"(bounded)"
    )
    print()

    # Process a maximum of TICKER_WORKERS tickers at once.
    # Results are consumed in original ticker order within each
    # batch, preserving deterministic rows, event numbering,
    # error ordering, and console output.
    with ThreadPoolExecutor(
        max_workers=worker_count
    ) as executor:
        for batch_start in range(
            0,
            total,
            worker_count,
        ):
            batch = selected[
                batch_start:
                batch_start + worker_count
            ]

            pending = []

            for batch_offset, security in enumerate(
                batch
            ):
                index = (
                    batch_start
                    + batch_offset
                    + 1
                )

                ticker = security.ticker
                security_id = security_identity(
                    security
                )

                eligible_session_dates = (
                    membership_dates_for_security(
                        security=security,
                        session_dates=(
                            market_session_dates
                        ),
                        rolling_universe=(
                            rolling_universe
                        ),
                    )
                )

                future = executor.submit(
                    _mine_selected_ticker,
                    ticker=ticker,
                    security_id=security_id,
                    start_date=config.start_date,
                    end_date=config.end_date,
                    eligible_session_dates=(
                        eligible_session_dates
                    ),
                )

                pending.append(
                    (
                        index,
                        ticker,
                        future,
                    )
                )

            # Consume in deterministic selection order even
            # though the HTTP/CPU work inside the batch runs
            # concurrently.
            for index, ticker, future in pending:
                print(
                    f"[{index:>3}/{total:<3}] "
                    f"{ticker:<8}",
                    end="",
                    flush=True,
                )

                try:
                    (
                        ticker_rows,
                        ticker_sessions,
                    ) = future.result()

                    rows.extend(
                        ticker_rows
                    )

                    sessions_tested += (
                        ticker_sessions
                    )

                    tickers_completed += 1

                    print(
                        f" sessions={ticker_sessions:>3} "
                        f"anomalies={len(ticker_rows):>3}"
                    )

                except Exception as exc:
                    errors.append(
                        {
                            "ticker": ticker,
                            "error": (
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                        }
                    )

                    print(
                        " FAILED "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

    research_eligible = sum(
        1
        for row in rows
        if row[
            "research_eligible"
        ]
    )

    ca_exclusions = (
        len(rows)
        - research_eligible
    )

    event_rows = build_event_rows(
        rows=rows,
        session_dates=market_session_dates,
    )

    return HistoricalMinerResult(
        config=config,
        universe_date=universe_date,
        raw_universe_count=(
            raw_universe_count
        ),
        core_universe_count=(
            core_universe_count
        ),
        selected_ticker_count=len(
            selected
        ),
        tickers_completed=(
            tickers_completed
        ),
        tickers_failed=len(
            errors
        ),
        market_sessions=len(
            market_session_dates
        ),
        sessions_tested=(
            sessions_tested
        ),
        anomaly_observations=len(
            rows
        ),
        research_eligible_observations=(
            research_eligible
        ),
        corporate_action_exclusions=(
            ca_exclusions
        ),
        anomaly_events=len(
            event_rows
        ),
        rows=rows,
        event_rows=event_rows,
        errors=errors,
    )