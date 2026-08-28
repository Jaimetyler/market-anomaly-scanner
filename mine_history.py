from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from scanner.miner import (
    DEFAULT_MAX_TICKERS,
    HARD_MAX_TICKERS,
    HistoricalMinerConfig,
    mine_history,
)


OUTPUT_DIR = Path("data/research")


def print_outcome_summary(
    dataframe: pd.DataFrame,
    *,
    title: str,
) -> None:
    if dataframe.empty:
        return

    if "executable_return_5d" not in dataframe.columns:
        return

    five_day = (
        pd.to_numeric(
            dataframe["executable_return_5d"],
            errors="coerce",
        )
        .dropna()
    )

    if five_day.empty:
        return

    print()
    print(title)
    print("-" * 60)

    print(
        f"Observations/events:   "
        f"{len(five_day):>8,}"
    )

    print(
        f"Mean:                 "
        f"{five_day.mean():>+8.2f}%"
    )

    print(
        f"Median:               "
        f"{five_day.median():>+8.2f}%"
    )

    print(
        f"Finished negative:    "
        f"{(five_day < 0).mean() * 100:>8.2f}%"
    )

    print(
        f"Down >= 10%:          "
        f"{(five_day <= -10).mean() * 100:>8.2f}%"
    )

    print(
        f"Down >= 20%:          "
        f"{(five_day <= -20).mean() * 100:>8.2f}%"
    )

    print(
        f"Up >= 10%:            "
        f"{(five_day >= 10).mean() * 100:>8.2f}%"
    )

    print(
        f"Up >= 20%:            "
        f"{(five_day >= 20).mean() * 100:>8.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded whole-market historical "
            "anomaly mining experiment."
        )
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

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_MAX_TICKERS,
        help=(
            "Maximum number of securities selected "
            "from the rolling point-in-time core "
            "universe union. "
            f"Hard cap: {HARD_MAX_TICKERS}."
        ),
    )

    parser.add_argument(
        "--refresh-universe",
        action="store_true",
        help=(
            "Ignore cached point-in-time universe "
            "snapshots and refetch them from Massive."
        ),
    )

    parser.add_argument(
        "--output",
        help="Optional observation CSV output path.",
    )

    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    config = HistoricalMinerConfig(
        start_date=start_date,
        end_date=end_date,
        max_tickers=args.limit,
    )

    print()
    print("=" * 100)
    print("BOUNDED WHOLE-MARKET HISTORICAL MINER")
    print("=" * 100)

    print(
        f"Signal window:    "
        f"{start_date} -> {end_date}"
    )

    print(
        "Universe mode:    "
        "rolling point-in-time"
    )

    print(
        f"Universe window:  "
        f"{start_date} -> {end_date}"
    )

    print(
        f"Ticker limit:     "
        f"{args.limit:,}"
    )

    print()

    print(
        "NOTE: Universe membership is evaluated "
        "separately for each market session."
    )

    print(
        "A ticker is only eligible on sessions where "
        "it belongs to that historical universe."
    )

    print(
        "Bounded ticker selection is deterministic "
        "from the union of core point-in-time snapshots."
    )

    print(
        "Event clustering uses a common SPY "
        "trading-session calendar."
    )

    print()
    print("Mining...")
    print()

    result = mine_history(
        config,
        force_universe_refresh=(
            args.refresh_universe
        ),
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        f"{start_date.isoformat()}_"
        f"{end_date.isoformat()}_"
        f"{args.limit}"
    )

    if args.output:
        observation_path = Path(args.output)

        event_path = (
            observation_path.parent
            / (
                f"{observation_path.stem}_events"
                f"{observation_path.suffix}"
            )
        )

    else:
        observation_path = (
            OUTPUT_DIR
            / (
                "historical_observations_"
                f"{suffix}.csv"
            )
        )

        event_path = (
            OUTPUT_DIR
            / (
                "historical_events_"
                f"{suffix}.csv"
            )
        )

    observation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    event_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    observation_df = pd.DataFrame(
        result.rows
    )

    event_df = pd.DataFrame(
        result.event_rows
    )

    observation_df.to_csv(
        observation_path,
        index=False,
    )

    event_df.to_csv(
        event_path,
        index=False,
    )

    print()
    print("=" * 100)
    print("MINER SUMMARY")
    print("=" * 100)

    print(
        f"Raw universe union:           "
        f"{result.raw_universe_count:>8,}"
    )

    print(
        f"Core research universe union: "
        f"{result.core_universe_count:>8,}"
    )

    print(
        f"Tickers selected:             "
        f"{result.selected_ticker_count:>8,}"
    )

    print(
        f"Tickers completed:            "
        f"{result.tickers_completed:>8,}"
    )

    print(
        f"Tickers failed:               "
        f"{result.tickers_failed:>8,}"
    )

    print(
        f"Market sessions:              "
        f"{result.market_sessions:>8,}"
    )

    print(
        f"Ticker sessions tested:       "
        f"{result.sessions_tested:>8,}"
    )

    print(
        f"Daily anomaly observations:   "
        f"{result.anomaly_observations:>8,}"
    )

    print(
        f"Research eligible:            "
        f"{result.research_eligible_observations:>8,}"
    )

    print(
        f"Corporate-action exclusions:  "
        f"{result.corporate_action_exclusions:>8,}"
    )

    print(
        f"Clustered anomaly events:     "
        f"{result.anomaly_events:>8,}"
    )

    if not observation_df.empty:
        print()
        print("DAILY SETUP DISTRIBUTION")
        print("-" * 60)

        setup_counts = (
            observation_df["primary_setup"]
            .value_counts()
        )

        for setup, count in setup_counts.items():
            print(
                f"{setup:<30} "
                f"{count:>8,}"
            )

        eligible_observations = (
            observation_df[
                observation_df[
                    "research_eligible"
                ]
                == True
            ]
        )

        print_outcome_summary(
            eligible_observations,
            title=(
                "DAILY OBSERVATION "
                "NEXT-OPEN -> 5-SESSION OUTCOMES"
            ),
        )

    if not event_df.empty:
        print()
        print("EVENT INITIAL-SETUP DISTRIBUTION")
        print("-" * 60)

        setup_counts = (
            event_df["initial_setup"]
            .value_counts()
        )

        for setup, count in setup_counts.items():
            print(
                f"{setup:<30} "
                f"{count:>8,}"
            )

        print_outcome_summary(
            event_df,
            title=(
                "EVENT-ENTRY "
                "NEXT-OPEN -> 5-SESSION OUTCOMES"
            ),
        )

    if result.errors:
        print()
        print("ERRORS")
        print("-" * 60)

        for error in result.errors:
            print(
                f"{error['ticker']:<8} "
                f"{error['error']}"
            )

    print()

    print(
        f"Observation CSV: "
        f"{observation_path.resolve()}"
    )

    print(
        f"Event CSV:       "
        f"{event_path.resolve()}"
    )

    print()

    print(
        "Daily observations remain available for "
        "state/path research."
    )

    print(
        "Event rows are anchored to the FIRST signal "
        "in each anomaly episode."
    )

    print(
        "Event rows are now the preferred unit for "
        "historical analog statistics."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()