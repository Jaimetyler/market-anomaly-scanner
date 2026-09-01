from __future__ import annotations

import argparse
from datetime import datetime, timezone
from math import isclose

from scanner.data import get_daily_bars
from scanner.flatfile_history import get_flatfile_daily_bars


def _session_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc,
    ).date().isoformat()


def _index_by_session(bars: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for bar in bars:
        timestamp = int(bar["t"])
        result[_session_from_ms(timestamp)] = bar
    return result


def _pct_diff(flat: float, rest: float) -> float:
    if rest == 0:
        return 0.0 if flat == 0 else float("inf")
    return ((flat / rest) - 1.0) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare local Massive flat-file bars with adjusted REST bars."
    )
    parser.add_argument("ticker")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--show",
        type=int,
        default=15,
        help="Maximum mismatched sessions to print.",
    )
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()

    print("=" * 100)
    print("FLAT FILE vs ADJUSTED REST BAR COMPARISON")
    print("=" * 100)
    print(f"Ticker:      {ticker}")
    print(f"Date range:  {args.start} -> {args.end}")
    print("Flat files:  unadjusted Massive day aggregates")
    print("REST bars:   adjusted=true")
    print()

    flat = get_flatfile_daily_bars(ticker, args.start, args.end)
    rest = get_daily_bars(ticker, args.start, args.end)

    flat_by_date = _index_by_session(flat)
    rest_by_date = _index_by_session(rest)

    flat_dates = set(flat_by_date)
    rest_dates = set(rest_by_date)
    common = sorted(flat_dates & rest_dates)

    print(f"Flat-file sessions: {len(flat_dates):,}")
    print(f"REST sessions:      {len(rest_dates):,}")
    print(f"Common sessions:    {len(common):,}")
    print(f"Flat only:          {len(flat_dates - rest_dates):,}")
    print(f"REST only:          {len(rest_dates - flat_dates):,}")
    print()

    exact_ohlc = 0
    price_mismatches: list[tuple[str, float, float, float]] = []
    volume_mismatches = 0

    for session in common:
        f = flat_by_date[session]
        r = rest_by_date[session]

        prices_match = all(
            isclose(float(f[key]), float(r[key]), rel_tol=1e-10, abs_tol=1e-10)
            for key in ("o", "h", "l", "c")
        )

        if prices_match:
            exact_ohlc += 1
        else:
            price_mismatches.append(
                (
                    session,
                    float(f["c"]),
                    float(r["c"]),
                    _pct_diff(float(f["c"]), float(r["c"])),
                )
            )

        if not isclose(
            float(f["v"]),
            float(r["v"]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            volume_mismatches += 1

    print(f"Exact OHLC sessions:      {exact_ohlc:,}/{len(common):,}")
    print(f"Price mismatch sessions:  {len(price_mismatches):,}")
    print(f"Volume mismatch sessions: {volume_mismatches:,}")

    if price_mismatches:
        print()
        print("FIRST PRICE MISMATCHES")
        print("-" * 100)
        print(f"{'DATE':<12} {'FLAT CLOSE':>14} {'REST CLOSE':>14} {'DIFF %':>12}")
        for session, flat_close, rest_close, diff_pct in price_mismatches[: args.show]:
            print(
                f"{session:<12} {flat_close:>14.6f} "
                f"{rest_close:>14.6f} {diff_pct:>11.4f}%"
            )

    if flat_dates - rest_dates:
        print()
        print("Flat-only sessions:")
        print(", ".join(sorted(flat_dates - rest_dates)[: args.show]))

    if rest_dates - flat_dates:
        print()
        print("REST-only sessions:")
        print(", ".join(sorted(rest_dates - flat_dates)[: args.show]))

    print()
    if not price_mismatches and flat_dates == rest_dates:
        print("RESULT: price histories match over this window.")
    elif price_mismatches:
        print(
            "RESULT: price histories differ. For split-affected names this is expected "
            "because flat files are unadjusted and REST is adjusted."
        )
    else:
        print("RESULT: prices match where sessions overlap, but session coverage differs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())