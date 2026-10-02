"""Separate GEM paper account: init, daily and status. No broker connection."""

import argparse
import csv
import hashlib
import html
import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

from scanner.gem_execution_data import NY, sessions
from scanner.gem_paper_data import fetch_snapshot, next_session, signal_at, validate_snapshot
from scanner.paper_ledger import cents, decimal, positive_price

SYMBOLS = ("SPY", "VEU", "AGG")
DEFAULT_FOLDER = Path("data/research/gem_paper")
STRATEGY = "gem-paper-v1"


@contextmanager
def transaction(path, initializing=False):
    path = Path(path)
    if not initializing and not path.is_file():
        raise ValueError("GEM account missing; run python -m scanner.gem_paper init")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    try:
        connection.execute("BEGIN IMMEDIATE")
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables and tables != {"gem_account"}:
            raise ValueError("This is not a GEM database; choose a separate account path")
        if initializing:
            connection.execute("CREATE TABLE IF NOT EXISTS gem_account (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)")
        elif not tables:
            raise ValueError("GEM account has not been initialized")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def load_state(connection):
    row = connection.execute("SELECT payload FROM gem_account WHERE id=1").fetchone()
    if not row:
        raise ValueError("GEM account has not been initialized")
    state = json.loads(row[0])
    if state.get("strategy") != STRATEGY:
        raise ValueError("Unsupported GEM account version")
    return state


def snapshot(path):
    with transaction(path) as connection:
        return load_state(connection)


def held_ticker(state):
    held = [t for t, p in state["positions"].items() if p["shares"]]
    if len(held) > 1:
        raise ValueError("GEM account unexpectedly holds multiple ETFs")
    return held[0] if held else "CASH"


def make_plan(state, signal, created_on, quote_day, indexed, origin):
    for plan in state["plans"]:
        if plan["status"] == "PENDING":
            plan["status"] = "SUPERSEDED"
    target = signal["ticker"]
    if target not in SYMBOLS:
        raise ValueError("GEM must choose SPY, VEU or AGG")
    equity = state["cash_cents"] + sum(cents(positive_price(indexed[t][quote_day]["close"])*p["shares"])
                                            for t, p in state["positions"].items())
    rate = decimal(state["cost_bps"])/10000
    price = positive_price(indexed[target][quote_day]["close"])
    estimate = int(decimal(equity)/100 / (price*(1+rate)))
    unchanged = held_ticker(state) == target
    plan = {"id": signal["signal_date"] + ":" + target, **signal,
            "created_on": created_on, "eligible_on": next_session(date.fromisoformat(created_on)),
            "quote_date": quote_day, "estimated_target_shares": estimate,
            "status": "NO_TRADE" if unchanged else "PENDING", "origin": origin,
            "from_ticker": held_ticker(state), "sizing": "maximum affordable whole shares at eligible open"}
    state["plans"].append(plan)
    return plan


def initialize(path, data, as_of, equity="100000", cost_bps="10"):
    initial, rate = cents(equity), decimal(cost_bps)
    if decimal(equity) <= 0 or initial <= 0 or not 0 <= rate < 5000:
        raise ValueError("Equity must be positive; cost-bps must be between 0 and 5000 (exclusive)")
    monthly, indexed, _, quote_day = validate_snapshot(data, as_of)
    state = {"strategy": STRATEGY, "opened_on": as_of.isoformat(),
             "recorded_at": datetime.now(NY).isoformat(), "initial_cents": initial,
             "cash_cents": initial, "fees_cents": 0, "income_cents": 0,
             "realized_cents": 0, "cost_bps": str(rate),
             "last_processed": (as_of-timedelta(days=1)).isoformat(),
             "positions": {t: {"shares": 0, "cost_cents": 0} for t in SYMBOLS},
             "plans": [], "fills": [], "income": [], "valuations": [],
             "actions_seen": [], "last_run": None}
    make_plan(state, signal_at(monthly, quote_day), as_of.isoformat(), quote_day, indexed, "account initialization")
    with transaction(path, initializing=True) as connection:
        if connection.execute("SELECT 1 FROM gem_account").fetchone():
            raise ValueError("GEM account already exists; initialization never overwrites it")
        connection.execute("INSERT INTO gem_account VALUES(1,?)", (json.dumps(state),))
    return state


def record_fill(state, plan, ticker, side, shares, price, day):
    gross = cents(price*shares)
    fee = cents(price*shares*decimal(state["cost_bps"])/10000)
    position = state["positions"][ticker]
    realized = 0
    if side == "SELL":
        if shares != position["shares"]:
            raise ValueError("GEM switches sell the entire old position")
        realized = gross-fee-position["cost_cents"]
        state["cash_cents"] += gross-fee
        state["realized_cents"] += realized
        position.update(shares=0, cost_cents=0)
    else:
        if position["shares"] or gross+fee > state["cash_cents"]:
            raise ValueError("Buy would exceed cash or duplicate a position")
        state["cash_cents"] -= gross+fee
        position.update(shares=shares, cost_cents=gross+fee)
    state["fees_cents"] += fee
    state["fills"].append({"date": day, "plan_id": plan["id"], "signal_date": plan["signal_date"],
                           "ticker": ticker, "side": side, "shares": shares,
                           "reference_open": str(price), "gross_cents": gross,
                           "cost_cents": fee, "realized_cents": realized,
                           "cash_after_cents": state["cash_cents"],
                           "method": "completed-session opening price; cost charged separately"})


def execute_plan(state, plan, indexed, day):
    if day != plan["eligible_on"] or day <= plan["created_on"] or day <= plan["signal_date"]:
        raise ValueError("Fill must use exactly the first eligible session after the plan")
    old, target = held_ticker(state), plan["ticker"]
    if old == target:
        plan["status"] = "NO_TRADE"
        return
    if old != "CASH":
        record_fill(state, plan, old, "SELL", state["positions"][old]["shares"],
                    positive_price(indexed[old][day]["open"]), day)
    price = positive_price(indexed[target][day]["open"])
    rate = decimal(state["cost_bps"])/10000
    shares = int(decimal(state["cash_cents"])/100 / (price*(1+rate)))
    # Cash and fees round independently to cents; never permit an overdraft.
    while shares and cents(price*shares)+cents(price*shares*rate) > state["cash_cents"]:
        shares -= 1
    if shares < 1:
        raise ValueError("Insufficient cash for one whole share; no changes recorded")
    record_fill(state, plan, target, "BUY", shares, price, day)
    plan.update(status="FILLED", fill_date=day, filled_target_shares=shares)


def advance_signal(state, monthly, cutoff, indexed):
    signal = signal_at(monthly, cutoff)
    if signal["signal_date"] > state["plans"][-1]["signal_date"]:
        # The initial mandate is recorded at init. Future month-end plans are
        # generated chronologically under that standing rule, even on catch-up.
        if signal["signal_date"] < state["opened_on"]:
            raise ValueError("Cannot invent a historical plan before account opening")
        make_plan(state, signal, signal["signal_date"], signal["signal_date"], indexed,
                  "standing monthly GEM rule")


def run_daily(path, data, as_of):
    monthly, indexed, actions, end = validate_snapshot(data, as_of)
    with transaction(path) as connection:
        state = load_state(connection)
        if as_of.isoformat() < state["opened_on"] or (state["valuations"] and end < state["last_processed"]):
            raise ValueError("Run precedes recorded account history")
        relevant = [a for a in actions if a["ticker"] in SYMBOLS and a["date"] >= state["opened_on"]]
        if any(a["kind"] == "SPLIT" for a in relevant):
            raise ValueError("Split detected since account opening: manual review required; ledger unchanged. Yahoo historical raw prices may be restated by splits.")
        prior = [a for a in relevant if a["date"] <= state["last_processed"]]
        if prior != state["actions_seen"]:
            raise ValueError("Previously processed corporate actions changed; review required, ledger unchanged")
        count = len(state["fills"])
        days = sessions(date.fromisoformat(state["last_processed"])+timedelta(days=1), date.fromisoformat(end))
        for day in days:
            # Only signals known BEFORE this session can affect its opening.
            advance_signal(state, monthly, (date.fromisoformat(day)-timedelta(days=1)).isoformat(), indexed)
            for action in relevant:
                if action["date"] != day:
                    continue
                shares = state["positions"][action["ticker"]]["shares"]
                if shares:
                    amount = cents(decimal(action["amount"])*shares)
                    state["cash_cents"] += amount
                    state["income_cents"] += amount
                    state["income"].append({**action, "shares_entitled": shares, "amount_cents": amount,
                                            "method": "modeled ex-dividend-date cash credit; not actual payment date"})
            for plan in state["plans"]:
                if plan["status"] == "PENDING":
                    if plan["eligible_on"] < day:
                        raise ValueError("Missed eligible session; refusing to substitute a later fill")
                    if plan["eligible_on"] == day:
                        execute_plan(state, plan, indexed, day)
            holdings = [{"ticker": t, **p, "close": str(indexed[t][day]["close"]),
                         "value_cents": cents(positive_price(indexed[t][day]["close"])*p["shares"])}
                        for t, p in state["positions"].items()]
            equity = state["cash_cents"]+sum(p["value_cents"] for p in holdings)
            state["valuations"].append({"date": day, "equity_cents": equity,
                                        "cash_cents": state["cash_cents"], "pnl_cents": equity-state["initial_cents"],
                                        "holdings": holdings})
            state["last_processed"] = day
        # A month can end on a weekend. Its newly confirmed signal still needs
        # a plan even if there has been no new exchange session since last run.
        advance_signal(state, monthly, end, indexed)
        state["actions_seen"] = [a for a in relevant if a["date"] <= state["last_processed"]]
        pending = [p for p in state["plans"] if p["status"] == "PENDING"]
        result = {"as_of": as_of.isoformat(), "price_date": end, "sessions_processed": len(days),
                  "new_fills": len(state["fills"])-count,
                  "status": "WAITING" if pending else "OK",
                  "next_fill_date": pending[0]["eligible_on"] if pending else None,
                  "snapshot_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()}
        state["last_run"] = result
        connection.execute("UPDATE gem_account SET payload=? WHERE id=1", (json.dumps(state),))
    return state


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_account(state, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"ledger.json").write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
    plan = state["plans"][-1]
    (folder/"plan.json").write_text(json.dumps(plan, indent=2)+"\n", encoding="utf-8")
    write_csv(folder/"fills.csv", state["fills"], ["date", "signal_date", "plan_id", "ticker", "side", "shares", "reference_open", "gross_cents", "cost_cents", "cash_after_cents"])
    write_csv(folder/"income.csv", state["income"], ["date", "ticker", "shares_entitled", "amount", "amount_cents", "method"])
    write_csv(folder/"equity.csv", state["valuations"], ["date", "equity_cents", "cash_cents", "pnl_cents"])
    write_csv(folder/"holdings.csv", [{"ticker": t, **p} for t, p in state["positions"].items()], ["ticker", "shares", "cost_cents"])
    last = state["valuations"][-1] if state["valuations"] else None
    value = last["equity_cents"] if last else state["cash_cents"]
    lines = ["GEM paper account", f"Opened: {state['opened_on']}; starting cash: ${state['initial_cents']/100:,.2f}",
             f"Cash: ${state['cash_cents']/100:,.2f}; account value: ${value/100:,.2f}",
             f"Marked through: {last['date'] if last else 'not marked yet'}; held: {held_ticker(state)}",
             f"Confirmed signal: {plan['signal_date']} -> {plan['ticker']}. {plan['reason']}",
             f"Plan: {plan['status']}; eligible opening session: {plan['eligible_on']}",
             f"Costs: {state['cost_bps']} bps per traded dollar; recorded costs: ${state['fees_cents']/100:,.2f}",
             f"Modeled dividend income: ${state['income_cents']/100:,.2f}"]
    if plan["status"] == "PENDING":
        lines.append(f"Estimated target: {plan['estimated_target_shares']} {plan['ticker']} shares; final whole-share quantity uses the eligible opening price and available cash.")
    if state["last_run"]:
        run = state["last_run"]
        lines.append(f"Daily run: {run['sessions_processed']} sessions processed; {run['new_fills']} new fills; prices through {run['price_date']}.")
    lines.extend(["", "Holdings:", *[f"  {t}: {p['shares']} shares" for t, p in state["positions"].items()], "",
                  "Simulation: fills are recorded after the session using its opening price. Current-day bars are excluded.",
                  "Dividends are modeled as cash on the ex-dividend date, earlier than actual payment; cash earns no interest.",
                  "A detected split or revised past dividend stops the run for review. Adjusted levels are used only for signals.",
                  "The standing monthly rule catches up missed sessions. Unchanged ETF signals do not rebalance or reinvest cash.",
                  "Whole shares, cash residuals and ex-date income differ from the fractional total-return research audit."])
    report = "\n".join(lines)+"\n"
    (folder/"account.txt").write_text(report, encoding="utf-8")
    rows = "".join("<tr>"+"".join(f"<td>{html.escape(str(e[k]))}</td>" for k in ("date", "side", "ticker", "shares", "reference_open"))+"</tr>" for e in reversed(state["fills"]))
    dashboard = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                 '<title>GEM paper account</title><style>body{font:16px system-ui;background:#101925;color:#e6eef8;max-width:1050px;margin:40px auto;padding:20px}pre{white-space:pre-wrap;line-height:1.6;font:inherit}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #425268}h1{color:#79d9ba}</style>'
                 '<h1>GEM paper account</h1><pre>'+html.escape(report)+'</pre><h2>Recorded simulated fills</h2><table><tr><th>Date</th><th>Side</th><th>ETF</th><th>Shares</th><th>Opening price</th></tr>'+rows+'</table></html>')
    (folder/"dashboard.html").write_text(dashboard, encoding="utf-8")
    print(report)
    print(f"Dashboard: {(folder/'dashboard.html').resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_FOLDER/"account.sqlite"))
    parser.add_argument("--output-dir", help="default: account database folder")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="create a separate GEM account and initial pending plan")
    init.add_argument("--equity", default="100000")
    init.add_argument("--cost-bps", default="10")
    init.add_argument("--snapshot", help="offline paper snapshot matching today's New York date")
    daily = sub.add_parser("daily", help="refresh data and process all eligible completed sessions")
    daily.add_argument("--snapshot", help="offline paper snapshot matching today's New York date")
    sub.add_parser("status", help="display/export ledger without fetching or trading")
    args = parser.parse_args(argv)
    folder = Path(args.output_dir) if args.output_dir else Path(args.db).parent
    try:
        if args.command == "status":
            state = snapshot(args.db)
        else:
            now = datetime.now(NY).date()
            if args.command == "init" and Path(args.db).exists():
                raise ValueError("Account file already exists; initialization never overwrites it")
            if args.command == "daily":
                snapshot(args.db)  # Fail before fetching if account is missing/wrong.
            print("Loading GEM prices and corporate actions...", flush=True)
            data = json.loads(Path(args.snapshot).read_text(encoding="utf-8-sig")) if args.snapshot else fetch_snapshot(now)
            state = initialize(args.db, data, now, args.equity, args.cost_bps) if args.command == "init" else run_daily(args.db, data, now)
            folder.mkdir(parents=True, exist_ok=True)
            (folder/"daily_snapshot.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
        export_account(state, folder)
    except (ValueError, KeyError, TypeError, IndexError, ArithmeticError, OSError, sqlite3.Error) as error:
        parser.exit(2, f"GEM paper account unavailable: {error}\n")


if __name__ == "__main__":
    main()
