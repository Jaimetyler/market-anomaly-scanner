import argparse
from collections import Counter
from datetime import date

from scanner.historical_universe import (
    get_historical_universe,
)
from scanner.universe_quality import (
    classify_universe,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the point-in-time "
            "historical stock universe."
        )
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Historical universe date YYYY-MM-DD.",
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Ignore local cache and "
            "refetch from Massive."
        ),
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=25,
        help="Number of core tickers to print.",
    )

    parser.add_argument(
        "--excluded-sample",
        type=int,
        default=20,
        help=(
            "Number of excluded securities "
            "to print."
        ),
    )

    args = parser.parse_args()

    as_of_date = date.fromisoformat(
        args.date
    )

    print()
    print("=" * 100)
    print(
        "POINT-IN-TIME HISTORICAL UNIVERSE"
    )
    print("=" * 100)

    print(
        f"As-of date:      {as_of_date}"
    )

    print(
        f"Force refresh:   {args.refresh}"
    )

    print()
    print("Loading universe...")

    result = get_historical_universe(
        as_of_date=as_of_date,
        force_refresh=args.refresh,
    )

    classifications = classify_universe(
        result.tickers
    )

    core = [
        item
        for item in classifications
        if item.include_in_core_research
    ]

    excluded = [
        item
        for item in classifications
        if not item.include_in_core_research
    ]

    bucket_counts = Counter(
        item.bucket.value
        for item in classifications
    )

    print()

    print(
        f"Source:          "
        f"{'CACHE' if result.fetched_from_cache else 'MASSIVE API'}"
    )

    print(
        f"Raw eligible:    "
        f"{result.total_count:,}"
    )

    print(
        f"Core research:   "
        f"{len(core):,}"
    )

    print(
        f"Excluded/tagged: "
        f"{len(excluded):,}"
    )

    print()

    print("BUCKET COUNTS")
    print("-" * 100)

    for bucket, count in sorted(
        bucket_counts.items()
    ):
        print(
            f"{bucket:<25} "
            f"{count:>8,}"
        )

    print()
    print("CORE RESEARCH SAMPLE")
    print("-" * 100)

    sample_count = max(
        0,
        args.sample,
    )

    for item in core[:sample_count]:
        ticker = item.ticker

        print(
            f"{ticker.ticker:<10} "
            f"{str(ticker.type):<5} "
            f"{str(ticker.primary_exchange):<6} "
            f"{ticker.name or ''}"
        )

    print()
    print("EXCLUDED / TAGGED SAMPLE")
    print("-" * 100)

    excluded_sample = max(
        0,
        args.excluded_sample,
    )

    for item in excluded[
        :excluded_sample
    ]:
        ticker = item.ticker

        reasons = (
            ", ".join(
                item.reasons
            )
            or "-"
        )

        print(
            f"{ticker.ticker:<10} "
            f"{item.bucket.value:<20} "
            f"{ticker.name or ''}"
        )

        print(
            f"{'':10} "
            f"reason: {reasons}"
        )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()