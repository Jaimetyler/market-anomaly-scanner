import argparse
from datetime import date, timedelta

from scanner.classifier import classify_setup
from scanner.data import get_daily_bars
from scanner.detector import detect_anomaly
from scanner.grouped import (
    build_grouped_prefilter,
    get_recent_grouped_sessions,
    grouped_sessions_to_dataframe,
)
from scanner.indicators import build_snapshot
from scanner.universe import get_active_stock_universe


def format_pct(value):
    if value is None:
        return "N/A"

    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def format_x(value):
    if value is None:
        return "N/A"

    return f"{value:.1f}x"


def deep_rank_value(result):
    """
    Temporary ranking function.

    This is NOT our final anomaly score.
    It only sorts the current research output.
    """

    snapshot = result["snapshot"]
    detection = result["detection"]

    trigger_count = len(detection.triggers)
    evidence_count = len(detection.evidence)

    return_1d = snapshot.get("return_1d") or 0
    return_5d = snapshot.get("return_5d") or 0
    return_20d = snapshot.get("return_20d") or 0

    distance_sma20 = (
        snapshot.get("distance_sma_20_pct")
        or 0
    )

    relative_volume = (
        snapshot.get("relative_volume")
        or 0
    )

    atr_expansion = (
        snapshot.get("atr_expansion")
        or 0
    )

    score = 0.0

    score += trigger_count * 1000
    score += evidence_count * 100

    score += max(return_1d, 0)
    score += max(return_5d, 0) * 0.75
    score += max(return_20d, 0) * 0.25

    score += max(distance_sma20, 0)

    score += (
        min(
            max(relative_volume, 0),
            25,
        )
        * 5
    )

    score += (
        min(
            max(atr_expansion, 0),
            10,
        )
        * 10
    )

    return score


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan the U.S. common-stock market "
            "for upside anomalies."
        )
    )

    parser.add_argument(
        "--sessions",
        type=int,
        default=25,
        help=(
            "Recent grouped trading sessions used "
            "for market-wide discovery."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Maximum prefiltered stocks to analyze "
            "deeply. 0 means analyze all candidates."
        ),
    )

    parser.add_argument(
        "--show",
        type=int,
        default=30,
        help=(
            "Maximum detected anomalies to display."
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 150)
    print("MARKET ANOMALY SCANNER")
    print("=" * 150)
    print()

    # --------------------------------------------------
    # 1. Eligible universe
    # --------------------------------------------------

    print(
        "Loading eligible common-stock universe..."
    )

    universe = get_active_stock_universe()

    eligible_symbols = set(
        universe.keys()
    )

    print(
        f"Eligible common stocks: "
        f"{len(eligible_symbols):,}"
    )

    # --------------------------------------------------
    # 2. Recent grouped market history
    # --------------------------------------------------

    print()
    print(
        f"Fetching {args.sessions} recent "
        "grouped market sessions..."
    )

    sessions = get_recent_grouped_sessions(
        trading_sessions=args.sessions,
    )

    if not sessions:
        print(
            "No grouped market sessions were returned."
        )
        return

    session_dates = list(
        sessions.keys()
    )

    print(
        f"Sessions received: "
        f"{len(session_dates)}"
    )

    print(
        f"Oldest session: "
        f"{session_dates[0]}"
    )

    print(
        f"Newest session: "
        f"{session_dates[-1]}"
    )

    # --------------------------------------------------
    # 3. Restrict grouped data to common stocks
    # --------------------------------------------------

    print()
    print(
        "Building common-stock market history..."
    )

    grouped_df = grouped_sessions_to_dataframe(
        sessions=sessions,
        eligible_symbols=eligible_symbols,
    )

    if grouped_df.empty:
        print(
            "No eligible grouped market data remained "
            "after universe filtering."
        )
        return

    grouped_symbols = (
        grouped_df["ticker"].nunique()
    )

    grouped_rows = len(grouped_df)

    print(
        f"Eligible symbols with grouped data: "
        f"{grouped_symbols:,}"
    )

    print(
        f"Grouped rows retained: "
        f"{grouped_rows:,}"
    )

    # --------------------------------------------------
    # 4. Multi-day market-wide prefilter
    # --------------------------------------------------

    print()
    print(
        "Running multi-day market prefilter..."
    )

    prefiltered = build_grouped_prefilter(
        grouped_df
    )

    print(
        f"Broad anomaly candidates: "
        f"{len(prefiltered):,}"
    )

    if not prefiltered:
        print()
        print(
            "No stocks passed the grouped-market "
            "prefilter."
        )
        return

    if args.limit > 0:
        selected = prefiltered[
            :args.limit
        ]
    else:
        selected = prefiltered

    print(
        f"Deep analysis queue: "
        f"{len(selected):,}"
    )

    # --------------------------------------------------
    # 5. Deep historical analysis
    # --------------------------------------------------

    end_date = date.today()

    start_date = (
        end_date
        - timedelta(days=450)
    )

    results = []

    print()
    print(
        "Running deep historical analysis..."
    )
    print()

    for index, item in enumerate(
        selected,
        start=1,
    ):
        ticker = item["ticker"]

        prefilter_triggers = item.get(
            "prefilter_triggers",
            [],
        )

        trigger_text = ",".join(
            prefilter_triggers
        )

        print(
            f"[{index:>3}/{len(selected)}] "
            f"{ticker:<8} "
            f"{trigger_text:<50}",
            end="",
        )

        try:
            bars = get_daily_bars(
                ticker=ticker,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )

            if not bars:
                print(" no history")
                continue

            snapshot = build_snapshot(
                bars
            )

            detection = detect_anomaly(
                snapshot
            )

            if not detection.candidate:
                print(" no deep anomaly")
                continue

            classification = classify_setup(
                snapshot
            )

            print(
                f" CANDIDATE  "
                f"{classification.primary_setup}"
            )

            results.append(
                {
                    "ticker": ticker,
                    "prefilter": item,
                    "snapshot": snapshot,
                    "detection": detection,
                    "classification": classification,
                }
            )

        except Exception as exc:
            print(
                f" ERROR: {exc}"
            )

    # --------------------------------------------------
    # 6. Temporary ranking
    # --------------------------------------------------

    results.sort(
        key=deep_rank_value,
        reverse=True,
    )

    # --------------------------------------------------
    # 7. Display ranked anomalies
    # --------------------------------------------------

    print()
    print("=" * 150)
    print("FINAL MARKET ANOMALIES")
    print("=" * 150)

    if not results:
        print(
            "No deep anomalies detected."
        )
        print("=" * 150)
        return

    header = (
        f"{'TICKER':<8}"
        f"{'1D':>9}"
        f"{'5D':>9}"
        f"{'20D':>9}"
        f"{'RVOL':>9}"
        f"{'RSI':>8}"
        f"{'SMA20':>9}"
        f"{'ATRx':>8}"
        f"{'TRIG':>7}"
        f"{'EVID':>7}"
        f"  {'PRIMARY SETUP':<25}"
    )

    print(header)
    print("-" * 150)

    for result in results[
        :args.show
    ]:
        ticker = result["ticker"]
        snapshot = result["snapshot"]
        detection = result["detection"]
        classification = result[
            "classification"
        ]

        relative_volume = snapshot.get(
            "relative_volume"
        )

        rsi = snapshot.get(
            "rsi_14"
        )

        sma20 = snapshot.get(
            "distance_sma_20_pct"
        )

        atr_expansion = snapshot.get(
            "atr_expansion"
        )

        rsi_text = (
            f"{rsi:.1f}"
            if rsi is not None
            else "N/A"
        )

        print(
            f"{ticker:<8}"
            f"{format_pct(snapshot.get('return_1d')):>9}"
            f"{format_pct(snapshot.get('return_5d')):>9}"
            f"{format_pct(snapshot.get('return_20d')):>9}"
            f"{format_x(relative_volume):>9}"
            f"{rsi_text:>8}"
            f"{format_pct(sma20):>9}"
            f"{format_x(atr_expansion):>8}"
            f"{len(detection.triggers):>7}"
            f"{len(detection.evidence):>7}"
            f"  {classification.primary_setup:<25}"
        )

    print("=" * 150)

    # --------------------------------------------------
    # 8. Setup distribution
    # --------------------------------------------------

    setup_counts = {}

    for result in results:
        setup = (
            result[
                "classification"
            ].primary_setup
        )

        setup_counts[setup] = (
            setup_counts.get(
                setup,
                0,
            )
            + 1
        )

    print()
    print("SETUP DISTRIBUTION")
    print("-" * 50)

    for setup, count in sorted(
        setup_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"{setup:<30} "
            f"{count:>5}"
        )

    # --------------------------------------------------
    # 9. Detailed tags
    # --------------------------------------------------

    print()
    print("TOP ANOMALY TAGS")
    print("-" * 100)

    for result in results[
        :args.show
    ]:
        ticker = result["ticker"]

        classification = result[
            "classification"
        ]

        tags = ", ".join(
            classification.tags
        )

        if not tags:
            tags = (
                "UNCLASSIFIED_ANOMALY"
            )

        print(
            f"{ticker:<8} "
            f"{tags}"
        )

    # --------------------------------------------------
    # 10. Summary
    # --------------------------------------------------

    print()
    print("=" * 70)

    print(
        f"Eligible common stocks:  "
        f"{len(eligible_symbols):>6,}"
    )

    print(
        f"Grouped-market symbols: "
        f"{grouped_symbols:>6,}"
    )

    print(
        f"Broad candidates:       "
        f"{len(prefiltered):>6,}"
    )

    print(
        f"Deep analyzed:          "
        f"{len(selected):>6,}"
    )

    print(
        f"Final anomalies:        "
        f"{len(results):>6,}"
    )

    print("=" * 70)

    print()
    print(
        "Full single-stock analysis:"
    )

    print(
        "python analyze.py TICKER"
    )

    print()


if __name__ == "__main__":
    main()