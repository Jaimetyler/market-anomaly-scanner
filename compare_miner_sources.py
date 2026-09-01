from __future__ import annotations

# DESTINATION: repo root / compare_miner_sources.py
#
# Runs the same bounded historical benchmark through:
#   1) existing REST miner
#   2) local split-adjusted flat-file miner
#
# Then reports ONLY meaningful discrepancies:
#   - signal rows present in one source but not the other
#   - research eligibility / CA differences
#   - feature deltas on differing rows
#   - event-entry differences
#
# Example:
#   ./.venv/Scripts/python.exe compare_miner_sources.py \
#       --start 2025-01-02 --end 2025-03-31 --limit 250

import argparse
from dataclasses import asdict, is_dataclass
from datetime import date
import math
import time
from typing import Any

from mine_history_ab import mine_history_flatfiles
from scanner.miner import HistoricalMinerConfig, mine_history as mine_history_rest


FEATURE_FIELDS = (
    "close",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "rvol",
    "sma20_extension_pct",
    "atr14_pct",
    "dollar_volume",
)

META_FIELDS = (
    "research_eligible",
    "corporate_action_excluded",
    "corporate_action_flagged",
    "setup_type",
)


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("ticker")), str(row.get("signal_date"))


def _event_key(row: dict[str, Any]) -> tuple[str, str]:
    ticker = str(row.get("ticker"))
    # Current event rows are built from event entry/signal date. Be tolerant
    # of either naming convention.
    session = (
        row.get("event_entry_date")
        or row.get("entry_date")
        or row.get("signal_date")
        or row.get("event_start_date")
    )
    return ticker, str(session)


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _fmt(value: Any) -> str:
    number = _num(value)
    if number is not None:
        return f"{number:.12g}"
    return repr(value)


def _delta(a: Any, b: Any) -> str:
    left = _num(a)
    right = _num(b)
    if left is None or right is None:
        return "N/A"
    return f"{right - left:+.12g}"


def _row_value(row: dict[str, Any], field: str) -> Any:
    if field in row:
        return row.get(field)

    snapshot = row.get("snapshot")
    if isinstance(snapshot, dict) and field in snapshot:
        return snapshot.get(field)

    return None


def _meaningfully_different(a: Any, b: Any) -> bool:
    left = _num(a)
    right = _num(b)

    if left is not None and right is not None:
        return not math.isclose(
            left,
            right,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    return a != b


def _print_row(title: str, row: dict[str, Any]) -> None:
    print()
    print(title)
    print("-" * 100)
    print(f"ticker={row.get('ticker')} signal_date={row.get('signal_date')}")
    for field in META_FIELDS:
        if field in row:
            print(f"{field:28} {_fmt(row.get(field))}")

    for field in FEATURE_FIELDS:
        value = _row_value(row, field)
        if value is not None:
            print(f"{field:28} {_fmt(value)}")


def _print_pair(
    rest_row: dict[str, Any],
    flat_row: dict[str, Any],
) -> None:
    ticker, session = _key(rest_row)
    print()
    print(f"{ticker} {session}")
    print("-" * 100)
    print(f"{'FIELD':28} {'REST':>20} {'FLAT':>20} {'FLAT-REST':>20}")

    fields = META_FIELDS + FEATURE_FIELDS

    for field in fields:
        rest_value = _row_value(rest_row, field)
        flat_value = _row_value(flat_row, field)

        if not _meaningfully_different(rest_value, flat_value):
            continue

        print(
            f"{field:28} "
            f"{_fmt(rest_value):>20} "
            f"{_fmt(flat_value):>20} "
            f"{_delta(rest_value, flat_value):>20}"
        )


def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {_key(row): row for row in rows}


def _event_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {_event_key(row): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare REST and split-adjusted flat-file miner outputs."
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument(
        "--force-universe-refresh",
        action="store_true",
    )
    args = parser.parse_args()

    config = HistoricalMinerConfig(
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end),
        max_tickers=args.limit,
    )

    print("=" * 100)
    print("REST vs FLAT-FILE MINER DISCREPANCY FINDER")
    print("=" * 100)
    print(f"Signal window: {args.start} -> {args.end}")
    print(f"Ticker limit:  {args.limit}")
    print()

    started = time.perf_counter()

    print("STEP 1/2 — REST CONTROL")
    print("=" * 100)
    rest = mine_history_rest(
        config,
        force_universe_refresh=args.force_universe_refresh,
    )

    print()
    print("STEP 2/2 — FLAT-FILE EXPERIMENT")
    print("=" * 100)
    flat = mine_history_flatfiles(
        config,
        force_universe_refresh=args.force_universe_refresh,
    )

    rest_rows = _index(rest.rows)
    flat_rows = _index(flat.rows)

    rest_keys = set(rest_rows)
    flat_keys = set(flat_rows)

    only_rest = sorted(rest_keys - flat_keys)
    only_flat = sorted(flat_keys - rest_keys)
    common = sorted(rest_keys & flat_keys)

    meta_diffs = []
    feature_diffs = []

    for key in common:
        rr = rest_rows[key]
        fr = flat_rows[key]

        if any(
            _meaningfully_different(
                _row_value(rr, field),
                _row_value(fr, field),
            )
            for field in META_FIELDS
        ):
            meta_diffs.append(key)

        if any(
            _meaningfully_different(
                _row_value(rr, field),
                _row_value(fr, field),
            )
            for field in FEATURE_FIELDS
        ):
            feature_diffs.append(key)

    rest_events = _event_index(rest.event_rows)
    flat_events = _event_index(flat.event_rows)

    rest_event_keys = set(rest_events)
    flat_event_keys = set(flat_events)

    only_rest_events = sorted(rest_event_keys - flat_event_keys)
    only_flat_events = sorted(flat_event_keys - rest_event_keys)

    print()
    print("=" * 100)
    print("DISCREPANCY SUMMARY")
    print("=" * 100)
    print(f"REST daily anomaly rows:       {len(rest_rows):>6}")
    print(f"FLAT daily anomaly rows:       {len(flat_rows):>6}")
    print(f"Common daily anomaly rows:     {len(common):>6}")
    print(f"REST-only anomaly rows:        {len(only_rest):>6}")
    print(f"FLAT-only anomaly rows:        {len(only_flat):>6}")
    print(f"Common rows with meta diffs:   {len(meta_diffs):>6}")
    print(f"Common rows with feature diffs:{len(feature_diffs):>6}")
    print()
    print(f"REST events:                   {len(rest_events):>6}")
    print(f"FLAT events:                   {len(flat_events):>6}")
    print(f"REST-only event entries:       {len(only_rest_events):>6}")
    print(f"FLAT-only event entries:       {len(only_flat_events):>6}")

    if only_flat:
        print()
        print("=" * 100)
        print("FLAT-ONLY DAILY ANOMALIES")
        print("=" * 100)
        for key in only_flat:
            _print_row("FLAT ONLY", flat_rows[key])

    if only_rest:
        print()
        print("=" * 100)
        print("REST-ONLY DAILY ANOMALIES")
        print("=" * 100)
        for key in only_rest:
            _print_row("REST ONLY", rest_rows[key])

    interesting_common = sorted(set(meta_diffs) | set(feature_diffs))

    if interesting_common:
        print()
        print("=" * 100)
        print("COMMON ANOMALIES WITH SOURCE DIFFERENCES")
        print("=" * 100)
        for key in interesting_common:
            _print_pair(rest_rows[key], flat_rows[key])

    if only_flat_events:
        print()
        print("=" * 100)
        print("FLAT-ONLY EVENT ENTRIES")
        print("=" * 100)
        for key in only_flat_events:
            print(key)

    if only_rest_events:
        print()
        print("=" * 100)
        print("REST-ONLY EVENT ENTRIES")
        print("=" * 100)
        for key in only_rest_events:
            print(key)

    print()
    print("=" * 100)

    if (
        not only_rest
        and not only_flat
        and not meta_diffs
        and not only_rest_events
        and not only_flat_events
    ):
        print("RESULT: STRUCTURAL MATCH")
    else:
        print("RESULT: DISCREPANCIES IDENTIFIED ABOVE")

    print("=" * 100)
    print(f"Elapsed: {time.perf_counter() - started:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())