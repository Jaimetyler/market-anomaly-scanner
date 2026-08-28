import argparse
from datetime import date, timedelta

import pandas as pd

from scanner.classifier import classify_setup
from scanner.data import get_daily_bars
from scanner.detector import detect_anomaly
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
) -> dict | None:
    """
    Reconstruct one historical scanner decision
    using only information available through
    signal_date.

    Future information is accessed only AFTER
    detection/classification have completed.
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
        "signal_price": (
            outcomes.signal_close
        ),
        "next_open": (
            outcomes.next_open
        ),
        "snapshot": snapshot,
        "detection": detection,
        "classification": classification,
        "outcomes": outcomes,
    }


def trading_dates_from_bars(
    bars: list[dict],
) -> list[str]:
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

    # Give indicators plenty of history before
    # the first possible signal.

    fetch_start = (
        start_date
        - timedelta(days=450)
    )

    # Need future bars to grade 20-session outcomes.
    # 45 calendar days gives us comfortable room for
    # weekends/holidays without making another call.

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

    available_dates = (
        trading_dates_from_bars(
            bars
        )
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

        print(
            f"{signal_date}  "
            f"{ticker:<7}  "
            f"{classification.primary_setup:<25} "
            f"1D {fmt(snapshot.get('return_1d')):>8}  "
            f"5D {fmt(snapshot.get('return_5d')):>8}  "
            f"20D {fmt(snapshot.get('return_20d')):>8}"
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
    print("-" * 130)

    for result in signals:
        classification = result[
            "classification"
        ]

        outcomes = result[
            "outcomes"
        ]

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

    print("=" * 130)

    print()
    print(
        f"Signals generated: "
        f"{len(signals):,}"
    )

    # --------------------------------------------------
    # Simple descriptive statistics.
    #
    # NOT a trading-performance claim.
    # --------------------------------------------------

    executable_5d_values = []

    for result in signals:
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
            f"Observations:      "
            f"{len(series):>8,}"
        )

        print(
            f"Mean return:       "
            f"{series.mean():>+8.2f}%"
        )

        print(
            f"Median return:     "
            f"{series.median():>+8.2f}%"
        )

        print(
            f"Finished negative: "
            f"{(series < 0).mean() * 100:>8.2f}%"
        )

        print(
            f"Down >= 10%:       "
            f"{(series <= -10).mean() * 100:>8.2f}%"
        )

        print(
            f"Down >= 20%:       "
            f"{(series <= -20).mean() * 100:>8.2f}%"
        )

        print(
            f"Up >= 10%:         "
            f"{(series >= 10).mean() * 100:>8.2f}%"
        )

        print(
            f"Up >= 20%:         "
            f"{(series >= 20).mean() * 100:>8.2f}%"
        )

    print()
    print(
        "NOTE: Consecutive signal days are currently "
        "treated as separate research observations."
    )

    print(
        "These observations are therefore NOT "
        "independent trades."
    )

    print()


if __name__ == "__main__":
    main()