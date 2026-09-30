"""Local Faber paper account commands; no broker connection."""

import argparse
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

from scanner.gtaa_signals import NY, fetch_daily
from scanner.paper_ledger import (
    SYMBOLS, add_income, cancel_plan, create_plan, export_account, fill_plan,
    initialize, mark_account, read_prices, snapshot, validate_prices, write_csv,
)

DEFAULT_FOLDER = Path("data/research/faber_paper")


def today():
    return datetime.now(NY).date()


def fetch_prices(as_of, destination):
    if as_of > today():
        raise ValueError("Cannot fetch future prices")
    with ThreadPoolExecutor(max_workers=5) as pool:
        series = list(pool.map(lambda ticker: fetch_daily(ticker, as_of), SYMBOLS))
    quotes = []
    for ticker, bars in zip(SYMBOLS, series):
        if not bars:
            raise ValueError(f"{ticker}: no completed daily prices")
        quote = max(bars, key=lambda r: r["date"])
        quotes.append({"ticker": ticker, "date": quote["date"], "close": quote["close"]})
    days = {date.fromisoformat(q["date"]) for q in quotes}
    if len(days) != 1:
        raise ValueError("Latest daily price dates disagree across ETFs")
    day = next(iter(days))
    validate_prices({q["ticker"]: q["close"] for q in quotes}, day, as_of)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, quotes, ["ticker", "date", "close"])
    print(f"Saved raw closing prices for {day} to {path.resolve()}")


def pending_id(path):
    pending = [p for p in snapshot(path)["plans"] if p["status"] == "PENDING"]
    if not pending:
        raise ValueError("No pending plan; run paper plan first")
    return pending[-1]["id"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_FOLDER / "account.sqlite"))
    parser.add_argument("--output-dir", help="default: the account database's folder")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="initialize cash-only account without overwriting an existing one")
    init.add_argument("--equity", default="100000")
    plan = commands.add_parser("plan", help="save the latest confirmed monthly target plan")
    plan.add_argument("--signals", default="data/research/faber_current_signals/signals.json")
    plan.add_argument("--fee", default="0", help="dollars charged per filled order")
    plan.add_argument("--slippage-bps", default="0", help="adverse basis points applied to each fill")
    prices = commands.add_parser("prices", help="fetch five ETF completed daily raw closes")
    prices.add_argument("--as-of", type=date.fromisoformat)
    prices.add_argument("--output-csv")
    for name in ("fill", "mark"):
        command = commands.add_parser(name, help="record pending plan fills" if name == "fill" else "record account valuation")
        command.add_argument("--prices-csv")
        if name == "fill":
            command.add_argument("--plan-id", help="default: most recent pending plan")
    commands.add_parser("status", help="print/export current ledger and recorded valuations")
    cancel = commands.add_parser("cancel", help="cancel a pending plan; never changes recorded fills")
    cancel.add_argument("--plan-id", help="default: most recent pending plan")
    income = commands.add_parser("income", help="record a dividend or cash interest credit")
    income.add_argument("--type", choices=("dividend", "interest"), required=True)
    income.add_argument("--amount", required=True)
    income.add_argument("--date", type=date.fromisoformat, required=True)
    income.add_argument("--reference", required=True, help="unique credit reference prevents duplicates")
    args = parser.parse_args(argv)
    now = today()
    folder = Path(args.output_dir) if args.output_dir else Path(args.db).parent
    try:
        if args.command == "prices":
            fetch_prices(args.as_of or now, args.output_csv or folder / "prices.csv")
            return
        if args.command == "init":
            initialize(args.db, args.equity, now)
            print("Paper account initialized. No positions or fills recorded.")
        elif args.command == "plan":
            report = json.loads(Path(args.signals).read_text(encoding="utf-8"))
            folder.mkdir(parents=True, exist_ok=True)
            result = create_plan(args.db, report, now, args.fee, args.slippage_bps)
            (folder / "plan.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Pending plan {result['id']}; signal {result['signal_date']}; created {result['created_on']}.")
            for order in result["orders"]:
                print(f"  {order['side']} {order['shares']} {order['ticker']}")
            print("No fills recorded. Fill prices must be from a later completed session.")
        elif args.command in ("fill", "mark"):
            day, values = read_prices(args.prices_csv or folder / "prices.csv", now)
            if args.command == "fill":
                plan_id = args.plan_id or pending_id(args.db)
                fills = fill_plan(args.db, plan_id, values, day, now)
                print(f"Recorded {len(fills)} simulated fills for {day}; plan {plan_id}.")
            else:
                mark_account(args.db, values, day, now)
        elif args.command == "cancel":
            cancel_plan(args.db, args.plan_id or pending_id(args.db))
            print("Pending plan cancelled; cash and holdings unchanged.")
        elif args.command == "income":
            add_income(args.db, args.amount, args.date, now, args.type, args.reference)
        export_account(args.db, folder)
    except (ValueError, KeyError, IndexError, TypeError, ArithmeticError, OSError, sqlite3.Error) as error:
        parser.exit(2, f"Paper command unavailable: {error}\n")


if __name__ == "__main__":
    main()
