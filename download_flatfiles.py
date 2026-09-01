from __future__ import annotations

import argparse
import sys

from scanner.flatfiles import (
    download_day_aggregate_range,
    validate_day_aggregate_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Massive U.S. stocks day-aggregate flat files."
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    print("=" * 88)
    print("MASSIVE STOCKS DAY-AGGREGATE FLAT FILE DOWNLOADER")
    print("=" * 88)
    print(f"Date range: {args.start} -> {args.end}")
    print("Dataset:    us_stocks_sip/day_aggs_v1")
    print("Local dir:  data/flatfiles/day_aggs")
    print()

    try:
        paths = download_day_aggregate_range(
            args.start, args.end, force=args.force, progress=True
        )

        print()
        print(f"Files ready: {len(paths)}")

        if args.validate:
            print()
            print("Validating...")
            total_rows = 0
            for path in paths:
                result = validate_day_aggregate_file(path)
                total_rows += int(result["rows"])
                print(
                    f"  OK {path.name}: {result['rows']:,} rows "
                    f"sample={','.join(result['sample_tickers'])}"
                )
            print()
            print(f"Validated rows across files: {total_rows:,}")

        print()
        print("Done.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
