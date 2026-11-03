"""Read-only local diagnostics for the 2019-2023 breakout development window.

This command deliberately does NOT produce P&L: raw Massive flatfile bars
require split, security-lineage and point-in-time-universe verification.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from experiments.anomaly_breakout.local_adapter import raw_development_smoke


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--observations", default="data/research/maturity_rebuild_v1/master_observations_2019_2025.csv")
    p.add_argument("--flatfiles-root", default=None, help="Existing local day_aggs directory")
    p.add_argument("--tickers", nargs="+", required=True, help="Exact case-sensitive tickers")
    p.add_argument("--output", default="data/research/anomaly_breakout/raw_development_diagnostic.csv")
    args = p.parse_args()
    signals = raw_development_smoke(args.observations, tickers=args.tickers, root=args.flatfiles_root)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticker", "signal_date", "entry_date", "stop_raw", "anomaly_5_sessions"])
        writer.writerows((s.ticker, s.signal_date, s.entry_date, s.stop, s.anomaly) for s in signals)
    print(f"RAW DIAGNOSTIC ONLY: {len(signals)} signals; {sum(s.anomaly for s in signals)} anomaly-tagged")
    print(f"Saved {target}")
    print("NO BACKTEST / NO PROFITABILITY CLAIM: corporate-action, identity and universe checks outstanding.")


if __name__ == "__main__":
    main()
