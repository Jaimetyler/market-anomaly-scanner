"""Offline portfolio report from pre-existing paper plans and grouped bars."""
import argparse
import json
from pathlib import Path

from scanner.paper_portfolio import save_report, simulate_portfolio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, default=Path("data/research/paper_watchlist/ledger.json"))
    parser.add_argument("--sessions", type=Path, required=True,
                        help="JSON mapping ISO dates to grouped OHLC bar arrays")
    parser.add_argument("--output", type=Path, default=Path("data/research/paper_watchlist/portfolio_report.json"))
    args = parser.parse_args()
    plans = json.loads(args.plans.read_text(encoding="utf-8"))
    sessions = json.loads(args.sessions.read_text(encoding="utf-8"))
    report = simulate_portfolio(plans, sessions)
    save_report(report, args.output)
    print(f"Report: {args.output}")
    print(f"Closed trades: {report['closed_trades']}")
    print(f"Realized P&L: ${report['realized_pnl']:,.2f}")
    print(f"Ending equity: {report['ending_equity']}")
    print("WARNING: Offline simulation, not broker fills or verified historical signal catch-up.")


if __name__ == "__main__":
    main()
