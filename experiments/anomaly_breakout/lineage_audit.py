"""Offline security-lineage boundary audit. No API calls, no P&L."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from scanner.flatfile_history import read_many_ticker_histories
from scanner.turtle_history import CONFIRMED_LINEAGE_BREAKS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--flatfiles-root", type=Path, required=True)
    p.add_argument("--tickers", nargs="+", required=True)
    args = p.parse_args()
    wanted = set(args.tickers)
    ids = {t: {} for t in wanted}
    with args.observations.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t, day = row["ticker"], row["signal_date"]
            if t in wanted and "2019-01-01" <= day <= "2023-12-31":
                ids[t].setdefault(row["security_id"], [day, day])
                dates = ids[t][row["security_id"]]
                dates[0] = min(dates[0], day)
                dates[1] = max(dates[1], day)
    histories = read_many_ticker_histories(
        args.tickers, "2018-01-01", "2023-12-31", root=args.flatfiles_root
    )
    for t in args.tickers:
        bars = histories[t]
        print(f"\n{t}: {len(bars)} bars")
        for identity, dates in sorted(ids[t].items()):
            print(f"  security_id={identity!r} observation_range={dates[0]}..{dates[1]}")
        for boundary in sorted(CONFIRMED_LINEAGE_BREAKS.get(t, ())):
            before = [b for b in bars if b.session_date < boundary]
            after = [b for b in bars if b.session_date >= boundary]
            print(f"  confirmed lineage boundary={boundary}; before={len(before)} bars; after={len(after)} bars")
        if len(ids[t]) != 1 or t in CONFIRMED_LINEAGE_BREAKS:
            print("  STATUS=REVIEW_REQUIRED: do not join price bars across identities/boundaries")
        else:
            print("  STATUS=PRELIMINARY_ONLY: split, missing-session and universe checks outstanding")


if __name__ == "__main__":
    main()
