"""Offline research data-quality preflight; no network and no P&L.

Run against the local maturity rebuild and the existing day-aggregate archive.
This preflight never assumes a raw ticker is a permanent security identity.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from scanner.flatfile_history import read_many_ticker_histories


def audit(observations: Path, root: Path, tickers: list[str]) -> dict:
    if not observations.is_file() or not root.is_dir():
        raise FileNotFoundError("Missing observations CSV or day_aggs root")
    identities = defaultdict(set)
    eligible = defaultdict(int)
    excluded = defaultdict(int)
    with observations.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"ticker", "security_id", "signal_date", "research_eligible", "ca_excluded"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing required observations columns")
        for row in reader:
            if row["ticker"] not in tickers or not ("2019-01-01" <= row["signal_date"] <= "2023-12-31"):
                continue
            t = row["ticker"]
            identities[t].add(row["security_id"])
            if row["research_eligible"].strip().lower() == "true" and row["ca_excluded"].strip().lower() != "true":
                eligible[t] += 1
            else:
                excluded[t] += 1
    histories = read_many_ticker_histories([*tickers, "SPY"], "2018-01-01", "2023-12-31", root=root)
    report = {}
    for ticker in [*tickers, "SPY"]:
        bars = histories.get(ticker, [])
        dates = [b.session_date for b in bars]
        invalid = sum(
            not (b.low > 0 and b.low <= min(b.open, b.close) and b.high >= max(b.open, b.close) and b.volume >= 0)
            for b in bars
        )
        report[ticker] = {
            "bars": len(bars),
            "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None,
            "duplicate_dates": len(dates) - len(set(dates)),
            "invalid_ohlcv": invalid,
            "security_ids": sorted(identities[ticker]),
            "identity_ambiguous": len(identities[ticker]) > 1 or "" in identities[ticker],
            "eligible_anomalies": eligible[ticker],
            "excluded_anomalies": excluded[ticker],
            "status": "REVIEW_REQUIRED",
        }
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--observations", required=True, type=Path)
    p.add_argument("--flatfiles-root", required=True, type=Path)
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--output", type=Path, default=Path("data/research/anomaly_breakout/preflight.json"))
    args = p.parse_args()
    if len(args.tickers) != len(set(args.tickers)) or "SPY" in args.tickers:
        p.error("Use distinct non-SPY tickers")
    report = audit(args.observations, args.flatfiles_root, args.tickers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for ticker, row in report.items():
        print(f"{ticker}: {row['bars']} bars; {len(row['security_ids'])} security IDs; "
              f"{row['duplicate_dates']} duplicate dates; {row['invalid_ohlcv']} invalid bars; "
              f"identity_ambiguous={row['identity_ambiguous']}")
    print(f"Saved {args.output}")
    print("REVIEW ONLY: not a corporate-action, lineage, completeness or profitability certification")


if __name__ == "__main__":
    main()
