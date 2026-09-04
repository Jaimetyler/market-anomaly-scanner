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
from scanner.live_research import (
    best_match_per_archetype,
    load_promoted_archetype_members,
    match_live_snapshot,
)
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


def format_rate(value):
    if value is None:
        return "N/A"

    return f"{value:.1%}"


def research_rank_value(result):
    """
    Rank live anomalies primarily by validated research evidence.

    This is still a research ranking, not a trading recommendation
    or final production score.

    Priority:
    1. Has a validated archetype match.
    2. Promotion score.
    3. Out-of-sample sample size.
    4. Out-of-sample win rate.
    5. Out-of-sample median contrarian return.
    6. Current anomaly intensity as a final tie-breaker.
    """

    best_match = result.get("best_research_match")

    snapshot = result["snapshot"]
    detection = result["detection"]

    if best_match is None:
        research_flag = 0
        promotion_score = 0.0
        oos_n = 0
        oos_win_rate = 0.0
        oos_median_return = 0.0
    else:
        research_flag = 1
        promotion_score = best_match.promotion_score
        oos_n = best_match.oos_n
        oos_win_rate = best_match.oos_win_rate
        oos_median_return = best_match.oos_median_return

    trigger_count = len(detection.triggers)
    evidence_count = len(detection.evidence)

    return_1d = snapshot.get("return_1d") or 0.0
    return_5d = snapshot.get("return_5d") or 0.0

    return (
        research_flag,
        promotion_score,
        oos_n,
        oos_win_rate,
        oos_median_return,
        trigger_count,
        evidence_count,
        return_5d,
        return_1d,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan the U.S. common-stock market "
            "for upside anomalies and validated "
            "historical contrarian setups."
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
    print("=" * 170)
    print("MARKET ANOMALY SCANNER")
    print("=" * 170)
    print()

    # --------------------------------------------------
    # 1. Load promoted research evidence
    # --------------------------------------------------

    print(
        "Loading validated research archetypes..."
    )

    research_members = (
        load_promoted_archetype_members()
    )

    research_archetype_count = (
        research_members["archetype_id"]
        .nunique()
    )

    print(
        f"Promoted research definitions: "
        f"{len(research_members):,}"
    )

    print(
        f"Validated archetypes:           "
        f"{research_archetype_count:,}"
    )

    # --------------------------------------------------
    # 2. Eligible universe
    # --------------------------------------------------

    print()
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
    # 3. Recent grouped market history
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
    # 4. Restrict grouped data to common stocks
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
    # 5. Multi-day market-wide prefilter
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
    # 6. Deep analysis + research matching
    # --------------------------------------------------

    end_date = date.today()

    start_date = (
        end_date
        - timedelta(days=450)
    )

    results = []

    print()
    print(
        "Running deep historical analysis "
        "and archetype matching..."
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

            raw_matches = match_live_snapshot(
                snapshot=snapshot,
                initial_setup=(
                    classification.primary_setup
                ),
                members_df=research_members,
            )

            archetype_matches = (
                best_match_per_archetype(
                    raw_matches
                )
            )

            best_research_match = (
                archetype_matches[0]
                if archetype_matches
                else None
            )

            if best_research_match is None:
                research_text = "NO RESEARCH MATCH"
            else:
                research_text = (
                    f"{best_research_match.archetype_id} "
                    f"{best_research_match.horizon_days}D "
                    f"{best_research_match.oos_win_rate:.1%}"
                )

            print(
                f" CANDIDATE  "
                f"{classification.primary_setup:<24} "
                f"{research_text}"
            )

            results.append(
                {
                    "ticker": ticker,
                    "prefilter": item,
                    "snapshot": snapshot,
                    "detection": detection,
                    "classification": classification,
                    "raw_research_matches": raw_matches,
                    "research_matches": archetype_matches,
                    "best_research_match": (
                        best_research_match
                    ),
                }
            )

        except Exception as exc:
            print(
                f" ERROR: {exc}"
            )

    # --------------------------------------------------
    # 7. Research-aware ranking
    # --------------------------------------------------

    results.sort(
        key=research_rank_value,
        reverse=True,
    )

    # --------------------------------------------------
    # 8. Display ranked anomalies
    # --------------------------------------------------

    print()
    print("=" * 170)
    print("VALIDATED LIVE ANOMALIES")
    print("=" * 170)

    if not results:
        print(
            "No deep anomalies detected."
        )
        print("=" * 170)
        return

    header = (
        f"{'TICKER':<8}"
        f"{'SETUP':<25}"
        f"{'ARCH':<7}"
        f"{'HZN':>5}"
        f"{'OOS N':>9}"
        f"{'OOS WR':>9}"
        f"{'OOS MED':>10}"
        f"{'WORST':>10}"
        f"{'1D':>9}"
        f"{'5D':>9}"
        f"{'RVOL':>9}"
        f"{'SMA20':>9}"
        f"{'MATCHES':>9}"
    )

    print(header)
    print("-" * 170)

    for result in results[
        :args.show
    ]:
        ticker = result["ticker"]

        snapshot = result["snapshot"]

        classification = result[
            "classification"
        ]

        best_match = result.get(
            "best_research_match"
        )

        research_matches = result.get(
            "research_matches",
            [],
        )

        if best_match is None:
            archetype = "-"
            horizon = "-"
            oos_n = "-"
            oos_wr = "-"
            oos_med = "-"
            worst = "-"
        else:
            archetype = (
                best_match.archetype_id
            )

            horizon = (
                f"{best_match.horizon_days}D"
            )

            oos_n = (
                f"{best_match.oos_n:,}"
            )

            oos_wr = (
                f"{best_match.oos_win_rate:.1%}"
            )

            oos_med = (
                f"{best_match.oos_median_return:+.2f}%"
            )

            worst = (
                f"{best_match.worst_oos_median_return:+.2f}%"
            )

        print(
            f"{ticker:<8}"
            f"{classification.primary_setup:<25}"
            f"{archetype:<7}"
            f"{horizon:>5}"
            f"{oos_n:>9}"
            f"{oos_wr:>9}"
            f"{oos_med:>10}"
            f"{worst:>10}"
            f"{format_pct(snapshot.get('return_1d')):>9}"
            f"{format_pct(snapshot.get('return_5d')):>9}"
            f"{format_x(snapshot.get('relative_volume')):>9}"
            f"{format_pct(snapshot.get('distance_sma_20_pct')):>9}"
            f"{len(research_matches):>9}"
        )

    print("=" * 170)

    # --------------------------------------------------
    # 9. Detailed research evidence
    # --------------------------------------------------

    print()
    print("TOP RESEARCH EVIDENCE")
    print("=" * 120)

    displayed = 0

    for result in results[
        :args.show
    ]:
        best_match = result.get(
            "best_research_match"
        )

        if best_match is None:
            continue

        displayed += 1

        ticker = result["ticker"]

        classification = result[
            "classification"
        ]

        print()
        print(
            f"{ticker} — "
            f"{classification.primary_setup}"
        )

        print(
            f"  Archetype:           "
            f"{best_match.archetype_id}"
        )

        print(
            f"  Condition:           "
            f"{best_match.condition}"
        )

        print(
            f"  Horizon:             "
            f"{best_match.horizon_days} trading days"
        )

        print(
            f"  Historical N:        "
            f"{best_match.historical_n:,}"
        )

        print(
            f"  Historical win rate: "
            f"{best_match.historical_win_rate:.1%}"
        )

        print(
            f"  Historical median:   "
            f"{best_match.historical_median_return:+.2f}%"
        )

        print(
            f"  OOS N:               "
            f"{best_match.oos_n:,}"
        )

        print(
            f"  OOS win rate:        "
            f"{best_match.oos_win_rate:.1%}"
        )

        print(
            f"  OOS median return:   "
            f"{best_match.oos_median_return:+.2f}%"
        )

        print(
            f"  Worst OOS median:    "
            f"{best_match.worst_oos_median_return:+.2f}%"
        )

        print(
            f"  Walk-forward folds:  "
            f"{best_match.validated_folds}/"
            f"{best_match.folds_tested}"
        )

        print(
            f"  Validation rate:     "
            f"{best_match.validation_rate:.1%}"
        )

        print(
            f"  Promotion score:     "
            f"{best_match.promotion_score:.2f}"
        )

    if displayed == 0:
        print()
        print(
            "No displayed anomalies matched a "
            "promoted research archetype."
        )

    # --------------------------------------------------
    # 10. Setup distribution
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
    print("-" * 60)

    for setup, count in sorted(
        setup_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"{setup:<35} "
            f"{count:>5}"
        )

    # --------------------------------------------------
    # 11. Summary
    # --------------------------------------------------

    research_matched = sum(
        1
        for result in results
        if result.get(
            "best_research_match"
        )
        is not None
    )

    print()
    print("=" * 80)

    print(
        f"Eligible common stocks:    "
        f"{len(eligible_symbols):>8,}"
    )

    print(
        f"Grouped-market symbols:   "
        f"{grouped_symbols:>8,}"
    )

    print(
        f"Broad candidates:         "
        f"{len(prefiltered):>8,}"
    )

    print(
        f"Deep analyzed:            "
        f"{len(selected):>8,}"
    )

    print(
        f"Final anomalies:          "
        f"{len(results):>8,}"
    )

    print(
        f"Research-matched:         "
        f"{research_matched:>8,}"
    )

    print(
        f"Validated archetypes:     "
        f"{research_archetype_count:>8,}"
    )

    print("=" * 80)

    print()
    print(
        "OOS metrics shown above are historical "
        "contrarian research results."
    )

    print(
        "They are evidence summaries, not forecasts "
        "or trading recommendations."
    )

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