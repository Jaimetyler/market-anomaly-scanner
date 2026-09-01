from __future__ import annotations

import argparse
from datetime import datetime, timezone
from math import isclose

from scanner.corporate_actions import get_stock_splits
from scanner.data import get_daily_bars
from scanner.flatfile_adjustments import adjust_flatfile_history_for_splits
from scanner.flatfile_history import read_ticker_history


PRICE_REL_TOL = 1e-7
PRICE_ABS_TOL = 1e-7
VOLUME_REL_TOL = 1e-9
VOLUME_ABS_TOL = 1e-9


def _session_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc,
    ).date().isoformat()


def _index_rest(bars):
    return {_session_from_ms(int(bar["t"])): bar for bar in bars}


def _index_flat(bars):
    return {bar.session_date: bar for bar in bars}


def _price_match(a: float, b: float) -> bool:
    return isclose(
        float(a),
        float(b),
        rel_tol=PRICE_REL_TOL,
        abs_tol=PRICE_ABS_TOL,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare split-adjusted local flat files with Massive REST adjusted=true."
    )
    parser.add_argument("ticker")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--show", type=int, default=15)
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()

    raw_flat = read_ticker_history(ticker, args.start, args.end)
    events = get_stock_splits(ticker=ticker)
    adjusted_flat = adjust_flatfile_history_for_splits(raw_flat, events)
    rest = get_daily_bars(ticker, args.start, args.end)

    flat_by_date = _index_flat(adjusted_flat)
    rest_by_date = _index_rest(rest)

    flat_dates = set(flat_by_date)
    rest_dates = set(rest_by_date)
    common = sorted(flat_dates & rest_dates)

    price_mismatches = []
    volume_mismatches = []

    for session in common:
        f = flat_by_date[session]
        r = rest_by_date[session]

        field_diffs = []
        for name, key in (
            ("open", "o"),
            ("high", "h"),
            ("low", "l"),
            ("close", "c"),
        ):
            flat_value = float(getattr(f, name))
            rest_value = float(r[key])
            if not _price_match(flat_value, rest_value):
                field_diffs.append(
                    (name, flat_value, rest_value, flat_value - rest_value)
                )

        if field_diffs:
            price_mismatches.append((session, field_diffs))

        if not isclose(
            f.volume,
            float(r["v"]),
            rel_tol=VOLUME_REL_TOL,
            abs_tol=VOLUME_ABS_TOL,
        ):
            volume_mismatches.append(
                (session, f.volume, float(r["v"]))
            )

    print("=" * 100)
    print("SPLIT-ADJUSTED FLAT FILE vs REST adjusted=true")
    print("=" * 100)
    print(f"Ticker:                 {ticker}")
    print(f"Date range:             {args.start} -> {args.end}")
    print(f"Split events loaded:    {len(events)}")
    print(f"Flat sessions:          {len(flat_by_date)}")
    print(f"REST sessions:          {len(rest_by_date)}")
    print(f"Common sessions:        {len(common)}")
    print(f"Flat-only sessions:     {len(flat_dates - rest_dates)}")
    print(f"REST-only sessions:     {len(rest_dates - flat_dates)}")
    print(f"Price mismatches:       {len(price_mismatches)}")
    print(f"Volume mismatches:      {len(volume_mismatches)}")
    print()

    if price_mismatches:
        print("FIRST TRUE PRICE MISMATCHES")
        print("-" * 100)
        for session, diffs in price_mismatches[: args.show]:
            print(session)
            for field, flat_value, rest_value, delta in diffs:
                print(
                    f"  {field:<5} "
                    f"flat={flat_value:.12f} "
                    f"rest={rest_value:.12f} "
                    f"delta={delta:.12g}"
                )
        print()

    if volume_mismatches:
        print("FIRST VOLUME MISMATCHES")
        print("-" * 100)
        for session, flat_volume, rest_volume in volume_mismatches[: args.show]:
            print(
                f"{session} flat={flat_volume:.4f} rest={rest_volume:.4f}"
            )
        print()

    if not price_mismatches:
        print("PRICE RESULT: MATCH")
    else:
        print("PRICE RESULT: MISMATCH")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())