import argparse
from datetime import date, timedelta

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


def fmt(value, digits=1):
    if value is None:
        return "N/A"

    return f"{value:+.{digits}f}%"


def analyze_historical_signal(
    ticker: str,
    bars: list[dict],
    signal_date: str,
    corporate_actions,
) -> dict | None:
    """
    Reconstruct one historical scanner decision
    using only information available through
    signal_date.

    Future outcomes are accessed only AFTER
    detection/classification have completed.

    Corporate-action validation is kept separate
    from the anomaly decision so it cannot alter
    the historical signal itself.
    """

    try:
        snapshot = build_snapshot_as_of(
            bars=bars,
            signal_date=signal_date,
        )
    except (ValueError, IndexError):
        return None

    detection = detect_anomaly(
        snapshot
    )

    if not detection.candidate:
        return None

    classification = classify_setup(
        snapshot
    )

    # --------------------------------------------------
    # CORPORATE-ACTION VALIDATION
    #
    # Corporate actions do NOT create or suppress the
    # historical anomaly signal above.
    #
    # They annotate whether the observation should be
    # trusted for research / analog statistics.
    #
    # The look-forward portion of the CA check is for
    # contamination diagnostics only and must NEVER
    # become a scanner feature.
    # --------------------------------------------------

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
    # Nothing below this line may affect:
    #   snapshot
    #   detection
    #   classification
    # --------------------------------------------------

    outcomes = calculate_historical_outcomes(
        bars=bars,
        signal_date=signal_date,
    )

    return {
        "ticker": ticker,
        "signal_date": signal_date,
        "signal_price": outcomes.signal_close,
        "next_open": outcomes.next_open,
        "snapshot": snapshot,
        "detection": detection,
        "classification": classification,
        "corporate_action_check": (
            corporate_action_check
        ),
        "outcomes": outcomes,
    }


def trading_dates_from_bars(
    bars: list[dict],
) -> list[str]:
    """
    Build the actual observed trading-session
    sequence directly from the historical bars.

    This sequence is used by event clustering so
    weekends and market holidays are not guessed.
    """

    dates = []

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


def print_corporate_actions(
    ticker: str,
    actions,
) -> None:
    print()
    print("=" * 100)
    print("CORPORATE ACTION HISTORY")
    print("=" * 100)

    if not actions:
        print(
            f"No split-related corporate actions "
            f"returned for {ticker}."
        )
        return

    print(
        f"Known split-related events: "
        f"{len(actions):,}"
    )

    print()

    for action in actions:
        ratio = ""

        if (
            action.split_from is not None
            and action.split_to is not None
        ):
            ratio = (
                f"{action.split_from:g}"
                f" -> "
                f"{action.split_to:g}"
            )

        factor = (
            f"{action.historical_adjustment_factor:g}"
            if (
                action.historical_adjustment_factor
                is not None
            )
            else "N/A"
        )

        print(
            f"{action.execution_date}  "
            f"{action.event_type:<20} "
            f"{ratio:<15} "
            f"historical factor={factor}"
        )

    print()
    print("=" * 100)


def print_anomaly_events(
    signals: list[dict],
    session_dates: list[str],
) -> None:
    """
    Convert daily historical signals into anomaly
    episodes and print state transitions.

    Event continuity uses the ACTUAL observed
    trading sessions supplied from the historical
    daily bars.

    Signals excluded because of a corporate action
    remain visible in the daily historical output,
    but are not allowed into research-event
    clustering.
    """

    if not signals:
        return

    research_signals = [
        signal
        for signal in signals
        if not signal[
            "corporate_action_check"
        ].exclude_from_research
    ]

    excluded_count = (
        len(signals)
        - len(research_signals)
    )

    if not research_signals:
        print()
        print("=" * 100)
        print("ANOMALY EVENTS")
        print("=" * 100)

        print(
            "No research-eligible observations "
            "remain after corporate-action "
            "validation."
        )

        return

    observations = []

    for signal in research_signals:
        classification = signal[
            "classification"
        ]

        snapshot = signal[
            "snapshot"
        ]

        observations.append(
            make_signal_observation(
                ticker=signal["ticker"],
                signal_date=signal[
                    "signal_date"
                ],
                primary_setup=(
                    classification.primary_setup
                ),
                tags=classification.tags,
                signal_price=signal[
                    "signal_price"
                ],
                values={
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
                    "distance_sma_20_pct": (
                        snapshot.get(
                            "distance_sma_20_pct"
                        )
                    ),
                    "distance_sma_50_pct": (
                        snapshot.get(
                            "distance_sma_50_pct"
                        )
                    ),
                    "atr_expansion": snapshot.get(
                        "atr_expansion"
                    ),
                },
            )
        )

    # --------------------------------------------------
    # TRADING-SESSION-AWARE EVENT CLUSTERING
    #
    # No calendar-day approximation here.
    #
    # max_gap_sessions=3 means up to three ACTUAL
    # trading sessions may occur without another
    # anomaly observation before a new event begins.
    # --------------------------------------------------

    events = cluster_signal_events(
        observations=observations,
        session_dates=session_dates,
        max_gap_sessions=3,
    )

    print()
    print("=" * 100)
    print("ANOMALY EVENTS")
    print("=" * 100)

    print(
        f"Raw daily observations:        "
        f"{len(signals):,}"
    )

    print(
        f"Research-eligible observations:"
        f" {len(observations):,}"
    )

    print(
        f"CA-excluded observations:      "
        f"{excluded_count:,}"
    )

    print(
        f"Clustered events:              "
        f"{len(events):,}"
    )

    for index, event in enumerate(
        events,
        start=1,
    ):
        print()
        print("-" * 100)

        print(
            f"EVENT {index} — "
            f"{event.ticker}"
        )

        print("-" * 100)

        print(
            f"Start:         "
            f"{event.start_date}"
        )

        print(
            f"End:           "
            f"{event.end_date}"
        )

        print(
            f"Observations:  "
            f"{event.observation_count}"
        )

        print(
            f"Calendar span: "
            f"{event.duration_calendar_days} days"
        )

        print(
            f"Initial setup: "
            f"{event.initial_setup}"
        )

        print(
            f"Final setup:   "
            f"{event.final_setup}"
        )

        print()

        print(
            "STATE PATH"
        )

        print(
            "  "
            + " -> ".join(
                event.setup_path
            )
        )

        if event.transitions:
            print()

            print(
                "TRANSITIONS"
            )

            for transition in (
                event.transitions
            ):
                print(
                    f"  "
                    f"{transition.transition_date}  "
                    f"{transition.from_setup}"
                    f" -> "
                    f"{transition.to_setup}"
                )

        print()

        print(
            "DAILY OBSERVATIONS"
        )

        for observation in (
            event.observations
        ):
            print(
                f"  "
                f"{observation.signal_date}  "
                f"{observation.primary_setup}"
            )

    print()
    print("=" * 100)


def print_corporate_action_summary(
    signals: list[dict],
) -> None:
    if not signals:
        return

    flagged = [
        signal
        for signal in signals
        if signal[
            "corporate_action_check"
        ].flagged
    ]

    excluded = [
        signal
        for signal in signals
        if signal[
            "corporate_action_check"
        ].exclude_from_research
    ]

    print()
    print("=" * 100)
    print("CORPORATE-ACTION VALIDATION")
    print("=" * 100)

    print(
        f"Signals checked:    "
        f"{len(signals):,}"
    )

    print(
        f"Signals flagged:    "
        f"{len(flagged):,}"
    )

    print(
        f"Research excluded:  "
        f"{len(excluded):,}"
    )

    if flagged:
        print()

        for signal in flagged:
            check = signal[
                "corporate_action_check"
            ]

            flags = ", ".join(
                check.flags
            )

            disposition = (
                "EXCLUDE"
                if check.exclude_from_research
                else "FLAG ONLY"
            )

            print(
                f"{signal['signal_date']}  "
                f"{signal['ticker']:<7}  "
                f"{disposition:<10}  "
                f"{flags}"
            )

            for action in (
                check.nearby_actions
            ):
                ratio = ""

                if (
                    action.split_from is not None
                    and action.split_to is not None
                ):
                    ratio = (
                        f"{action.split_from:g}"
                        f" -> "
                        f"{action.split_to:g}"
                    )

                print(
                    f"             "
                    f"CA: "
                    f"{action.execution_date} "
                    f"{action.event_type} "
                    f"{ratio}"
                )

    print()
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded historical replay "
            "for one stock."
        )
    )

    parser.add_argument(
        "ticker",
        help="Ticker to replay.",
    )

    parser.add_argument(
        "--start",
        required=True,
        help="First signal date YYYY-MM-DD.",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="Last signal date YYYY-MM-DD.",
    )

    args = parser.parse_args()

    ticker = (
        args.ticker
        .upper()
        .strip()
    )

    start_date = date.fromisoformat(
        args.start
    )

    end_date = date.fromisoformat(
        args.end
    )

    if end_date < start_date:
        raise ValueError(
            "--end cannot be before --start."
        )

    fetch_start = (
        start_date
        - timedelta(days=450)
    )

    fetch_end = (
        end_date
        + timedelta(days=45)
    )

    print()
    print("=" * 100)
    print("BOUNDED HISTORICAL REPLAY")
    print("=" * 100)

    print(
        f"Ticker:        {ticker}"
    )

    print(
        f"Signal window: "
        f"{start_date} -> {end_date}"
    )

    print(
        f"Data window:   "
        f"{fetch_start} -> {fetch_end}"
    )

    print()

    print(
        "Fetching daily history..."
    )

    bars = get_daily_bars(
        ticker=ticker,
        start_date=fetch_start.isoformat(),
        end_date=fetch_end.isoformat(),
    )

    print(
        f"Bars received: {len(bars):,}"
    )

    if not bars:
        print(
            "No history returned."
        )
        return

    # --------------------------------------------------
    # ACTUAL OBSERVED TRADING SESSIONS
    #
    # This becomes the session sequence used by
    # historical replay AND anomaly-event clustering.
    # --------------------------------------------------

    available_dates = (
        trading_dates_from_bars(
            bars
        )
    )

    print()
    print(
        "Loading corporate-action history..."
    )

    # Cached after the first successful request.
    corporate_actions = get_stock_splits(
        ticker=ticker,
    )

    print(
        f"Corporate actions loaded: "
        f"{len(corporate_actions):,}"
    )

    print_corporate_actions(
        ticker=ticker,
        actions=corporate_actions,
    )

    signal_dates = [
        session_date
        for session_date
        in available_dates
        if (
            start_date
            <= date.fromisoformat(
                session_date
            )
            <= end_date
        )
    ]

    print(
        f"Sessions tested: "
        f"{len(signal_dates):,}"
    )

    print()
    print(
        "Replaying scanner..."
    )
    print()

    signals = []

    for signal_date in signal_dates:
        result = analyze_historical_signal(
            ticker=ticker,
            bars=bars,
            signal_date=signal_date,
            corporate_actions=(
                corporate_actions
            ),
        )

        if result is None:
            continue

        signals.append(
            result
        )

        classification = result[
            "classification"
        ]

        snapshot = result[
            "snapshot"
        ]

        ca_check = result[
            "corporate_action_check"
        ]

        if ca_check.exclude_from_research:
            ca_status = "CA:EXCLUDE"

        elif ca_check.flagged:
            ca_status = "CA:FLAG"

        else:
            ca_status = "CA:CLEAN"

        print(
            f"{signal_date}  "
            f"{ticker:<7}  "
            f"{classification.primary_setup:<25} "
            f"1D "
            f"{fmt(snapshot.get('return_1d')):>8}  "
            f"5D "
            f"{fmt(snapshot.get('return_5d')):>8}  "
            f"20D "
            f"{fmt(snapshot.get('return_20d')):>8}  "
            f"{ca_status}"
        )

    print()
    print("=" * 100)
    print("HISTORICAL SIGNALS")
    print("=" * 100)

    if not signals:
        print(
            "No anomaly signals occurred "
            "during this window."
        )
        return

    header = (
        f"{'DATE':<12}"
        f"{'SETUP':<25}"
        f"{'CA':<11}"
        f"{'CLOSE':>10}"
        f"{'NEXT OPEN':>12}"
        f"{'R 1D':>9}"
        f"{'R 5D':>9}"
        f"{'R 20D':>9}"
        f"{'X 1D':>9}"
        f"{'X 5D':>9}"
        f"{'X 20D':>9}"
        f"{'MFE':>9}"
        f"{'MAE':>9}"
    )

    print(header)
    print("-" * 141)

    for result in signals:
        classification = result[
            "classification"
        ]

        outcomes = result[
            "outcomes"
        ]

        ca_check = result[
            "corporate_action_check"
        ]

        if ca_check.exclude_from_research:
            ca_status = "EXCLUDE"

        elif ca_check.flagged:
            ca_status = "FLAG"

        else:
            ca_status = "CLEAN"

        research = outcomes.research
        executable = outcomes.executable

        next_open_text = (
            f"{outcomes.next_open:.2f}"
            if outcomes.next_open
            is not None
            else "N/A"
        )

        executable_1d = (
            executable.return_1d
            if executable
            else None
        )

        executable_5d = (
            executable.return_5d
            if executable
            else None
        )

        executable_20d = (
            executable.return_20d
            if executable
            else None
        )

        executable_mfe = (
            executable.mfe_pct
            if executable
            else None
        )

        executable_mae = (
            executable.mae_pct
            if executable
            else None
        )

        print(
            f"{result['signal_date']:<12}"
            f"{classification.primary_setup:<25}"
            f"{ca_status:<11}"
            f"{outcomes.signal_close:>10.2f}"
            f"{next_open_text:>12}"
            f"{fmt(research.return_1d):>9}"
            f"{fmt(research.return_5d):>9}"
            f"{fmt(research.return_20d):>9}"
            f"{fmt(executable_1d):>9}"
            f"{fmt(executable_5d):>9}"
            f"{fmt(executable_20d):>9}"
            f"{fmt(executable_mfe):>9}"
            f"{fmt(executable_mae):>9}"
        )

    print("=" * 141)

    print()

    print(
        f"Signals generated: "
        f"{len(signals):,}"
    )

    print_corporate_action_summary(
        signals
    )

    # --------------------------------------------------
    # EVENT CLUSTERING
    #
    # We now pass the actual observed trading-session
    # sequence instead of estimating gaps from calendar
    # days.
    # --------------------------------------------------

    print_anomaly_events(
        signals=signals,
        session_dates=available_dates,
    )

    # --------------------------------------------------
    # Research statistics.
    #
    # IMPORTANT:
    # Only corporate-action eligible observations
    # enter aggregate research statistics.
    # --------------------------------------------------

    research_eligible_signals = [
        signal
        for signal in signals
        if not signal[
            "corporate_action_check"
        ].exclude_from_research
    ]

    executable_5d_values = []

    for result in (
        research_eligible_signals
    ):
        executable = (
            result[
                "outcomes"
            ].executable
        )

        if (
            executable is not None
            and executable.return_5d
            is not None
        ):
            executable_5d_values.append(
                executable.return_5d
            )

    if executable_5d_values:
        series = pd.Series(
            executable_5d_values,
            dtype=float,
        )

        print()

        print(
            "NEXT-OPEN -> 5-SESSION "
            "DESCRIPTIVE OUTCOMES"
        )

        print("-" * 60)

        print(
            f"Research observations: "
            f"{len(series):>8,}"
        )

        print(
            f"Mean return:          "
            f"{series.mean():>+8.2f}%"
        )

        print(
            f"Median return:        "
            f"{series.median():>+8.2f}%"
        )

        print(
            f"Finished negative:    "
            f"{(series < 0).mean() * 100:>8.2f}%"
        )

        print(
            f"Down >= 10%:          "
            f"{(series <= -10).mean() * 100:>8.2f}%"
        )

        print(
            f"Down >= 20%:          "
            f"{(series <= -20).mean() * 100:>8.2f}%"
        )

        print(
            f"Up >= 10%:            "
            f"{(series >= 10).mean() * 100:>8.2f}%"
        )

        print(
            f"Up >= 20%:            "
            f"{(series >= 20).mean() * 100:>8.2f}%"
        )

    print()

    print(
        "NOTE: Daily signal observations are "
        "not independent trades."
    )

    print(
        "Corporate-action exclusions are removed "
        "from aggregate research statistics."
    )

    print(
        "Anomaly events group related observations "
        "using actual observed trading sessions."
    )

    print()


if __name__ == "__main__":
    main()