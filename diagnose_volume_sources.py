from __future__ import annotations

import argparse
from datetime import datetime, timezone
from statistics import mean, median

from scanner.corporate_actions import get_stock_splits
from scanner.data import get_daily_bars
from scanner.flatfile_adjustments import adjust_flatfile_history_for_splits
from scanner.flatfile_history import read_ticker_history


def _session_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc,
    ).date().isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure Massive flat-file vs REST adjusted=true volume differences."
    )
    parser.add_argument("ticker")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--show", type=int, default=15)
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()

    raw_flat = read_ticker_history(ticker, args.start, args.end)
    events = get_stock_splits(ticker=ticker)
    flat = adjust_flatfile_history_for_splits(raw_flat, events)
    rest = get_daily_bars(ticker, args.start, args.end)

    flat_by_date = {bar.session_date: bar for bar in flat}
    rest_by_date = {
        _session_from_ms(int(bar["t"])): bar
        for bar in rest
    }

    common = sorted(set(flat_by_date) & set(rest_by_date))
    rows = []

    for session in common:
        f = float(flat_by_date[session].volume)
        r = float(rest_by_date[session]["v"])

        if r == 0:
            pct = 0.0 if f == 0 else float("inf")
        else:
            pct = ((f / r) - 1.0) * 100.0

        rows.append((session, f, r, pct, abs(pct)))

    finite = [row for row in rows if row[4] != float("inf")]
    abs_pcts = [row[4] for row in finite]
    signed_pcts = [row[3] for row in finite]

    mismatches = [row for row in finite if row[4] > 1e-12]
    over_01 = [row for row in finite if row[4] >= 0.1]
    over_05 = [row for row in finite if row[4] >= 0.5]
    over_1 = [row for row in finite if row[4] >= 1.0]
    over_2 = [row for row in finite if row[4] >= 2.0]
    over_5 = [row for row in finite if row[4] >= 5.0]

    print("=" * 104)
    print("FLAT FILE vs REST VOLUME DIAGNOSTIC")
    print("=" * 104)
    print(f"Ticker:                 {ticker}")
    print(f"Date range:             {args.start} -> {args.end}")
    print(f"Common sessions:        {len(common)}")
    print(f"Volume mismatches:      {len(mismatches)}")
    print()

    if finite:
        print(f"Mean signed difference: {mean(signed_pcts):+.4f}%")
        print(f"Median signed diff:     {median(signed_pcts):+.4f}%")
        print(f"Mean absolute diff:     {mean(abs_pcts):.4f}%")
        print(f"Median absolute diff:   {median(abs_pcts):.4f}%")
        print(f"Max absolute diff:      {max(abs_pcts):.4f}%")
        print()
        print(f"Sessions >= 0.1% diff:  {len(over_01):>4}/{len(finite)}")
        print(f"Sessions >= 0.5% diff:  {len(over_05):>4}/{len(finite)}")
        print(f"Sessions >= 1.0% diff:  {len(over_1):>4}/{len(finite)}")
        print(f"Sessions >= 2.0% diff:  {len(over_2):>4}/{len(finite)}")
        print(f"Sessions >= 5.0% diff:  {len(over_5):>4}/{len(finite)}")

    if finite:
        print()
        print("LARGEST ABSOLUTE VOLUME DIFFERENCES")
        print("-" * 104)
        print(f"{'DATE':<12} {'FLAT':>18} {'REST':>18} {'DIFF %':>12}")
        for session, f, r, pct, _ in sorted(
            finite,
            key=lambda row: row[4],
            reverse=True,
        )[: args.show]:
            print(
                f"{session:<12} {f:>18,.0f} {r:>18,.0f} {pct:>+11.4f}%"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())