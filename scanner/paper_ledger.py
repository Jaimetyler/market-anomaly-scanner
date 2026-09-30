"""Transactional, local paper account. Money is stored as integer cents."""

import csv
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from scanner.gtaa_signals import TICKERS, latest_completed_month

SYMBOLS = tuple(TICKERS.values())


def decimal(value):
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("Numbers must be finite")
    return number


def cents(value):
    return int((decimal(value) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def positive_price(value):
    number = decimal(value)
    if number <= 0:
        raise ValueError("Prices must be positive")
    return number


@contextmanager
def transaction(path, initialize=False):
    path = Path(path)
    if not initialize and not path.is_file():
        raise ValueError("Paper account missing; run paper init first")
    if initialize:
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    try:
        connection.execute("BEGIN IMMEDIATE")
        if initialize:
            connection.execute("CREATE TABLE IF NOT EXISTS account (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS plans (id TEXT PRIMARY KEY, month TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL)")
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_monthly_plan ON plans(month) WHERE status != 'CANCELLED'")
            connection.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, kind TEXT NOT NULL, day TEXT NOT NULL, payload TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS valuations (day TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def account(connection):
    row = connection.execute("SELECT payload FROM account WHERE id=1").fetchone()
    if not row:
        raise ValueError("Paper account has not been initialized")
    result = json.loads(row[0])
    if result.get("schema_version") != 1:
        raise ValueError("Unsupported paper account schema")
    return result


def save_account(connection, state):
    connection.execute("UPDATE account SET payload=? WHERE id=1", (json.dumps(state),))


def initialize(path, equity, opened_on):
    equity = decimal(equity)
    if equity <= 0 or cents(equity) <= 0:
        raise ValueError("Starting equity must be positive")
    state = {"schema_version": 1, "opened_on": opened_on.isoformat(),
             "initial_cents": cents(equity), "cash_cents": cents(equity),
             "realized_cents": 0, "income_cents": 0, "fees_cents": 0, "revision": 0,
             "positions": {t: {"shares": 0, "cost_cents": 0} for t in SYMBOLS}}
    with transaction(path, initialize=True) as connection:
        if connection.execute("SELECT 1 FROM account").fetchone():
            raise ValueError("Account already exists; initialization never overwrites it")
        connection.execute("INSERT INTO account VALUES(1,?)", (json.dumps(state),))
    return state


def validate_prices(prices, day, as_of):
    if day >= as_of:
        raise ValueError("Price date must be a completed day before --as-of")
    if (as_of - day).days > 7:
        raise ValueError("Prices are stale; maximum age is seven calendar days")
    if set(prices) != set(SYMBOLS):
        raise ValueError("Prices must include exactly SPY, EFA, IEF, VNQ, GSG")
    return {t: positive_price(prices[t]) for t in SYMBOLS}


def read_prices(path, as_of):
    prices, days = {}, set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            ticker = row["ticker"].strip().upper()
            if ticker in prices:
                raise ValueError(f"Duplicate price ticker: {ticker}")
            prices[ticker] = row["close"]
            days.add(date.fromisoformat(row["date"]))
    if len(days) != 1:
        raise ValueError("All five price dates must agree")
    day = next(iter(days))
    return day, validate_prices(prices, day, as_of)


def equity_cents(state, prices):
    return state["cash_cents"] + sum(cents(prices[t] * p["shares"])
                                     for t, p in state["positions"].items())


def latest_recorded_day(connection, state):
    days = [state["opened_on"]]
    for table in ("events", "valuations"):
        day = connection.execute(f"SELECT MAX(day) FROM {table}").fetchone()[0]
        if day:
            days.append(day)
    return date.fromisoformat(max(days))


def create_plan(path, report, as_of, fee=0, slippage_bps=0):
    fee_cents = cents(fee)
    slippage = decimal(slippage_bps)
    if decimal(fee) < 0 or slippage < 0 or slippage >= 10000:
        raise ValueError("Fees must be nonnegative; slippage must be between 0 and 10000 bps")
    signal_day = date.fromisoformat(report["signal_date"])
    signal_month = latest_completed_month(as_of)
    if signal_day.strftime("%Y-%m") != signal_month:
        raise ValueError("Report must contain the latest completed month's signal")
    report_day = date.fromisoformat(report["as_of"])
    if report_day > as_of or (as_of - report_day).days > 7:
        raise ValueError("Signal report is future-dated or stale; regenerate it")
    slots = report["slots"]
    if len(slots) != 5 or {s["ticker"] for s in slots} != set(SYMBOLS):
        raise ValueError("Signal report must contain exactly five unique ETFs")
    if any(s["signal"] not in ("ASSET", "TBILL") for s in slots):
        raise ValueError("Unknown ETF signal")
    if any(decimal(s["target_weight"]) != (Decimal("0.2") if s["signal"] == "ASSET" else Decimal(0)) for s in slots):
        raise ValueError("Each signal must have its own 20% or 0% allocation")
    quote_days = {date.fromisoformat(s["quote_date"]) for s in slots}
    if len(quote_days) != 1:
        raise ValueError("Signal quote dates disagree")
    quote_day = next(iter(quote_days))
    prices = validate_prices({s["ticker"]: s["quote_close"] for s in slots}, quote_day, as_of)
    if quote_day < signal_day:
        raise ValueError("Planning prices precede the confirmed signal")
    with transaction(path) as connection:
        state = account(connection)
        if as_of < latest_recorded_day(connection, state):
            raise ValueError("Cannot create a plan before recorded account history")
        if quote_day < latest_recorded_day(connection, state) and any(p["shares"] for p in state["positions"].values()):
            raise ValueError("Planning quotes precede recorded account history")
        existing = connection.execute("SELECT id,status FROM plans WHERE month=? AND status!='CANCELLED'", (signal_month,)).fetchone()
        if existing:
            raise ValueError(f"Monthly plan already exists: {existing[0]} ({existing[1]}). Cancel a pending plan before replacing it")
        equity = equity_cents(state, prices)
        orders = []
        targets = {}
        for slot in slots:
            ticker = slot["ticker"]
            weight = decimal(slot["target_weight"])
            targets[ticker] = int((Decimal(equity) / 100 * weight / prices[ticker]).to_integral_value(rounding="ROUND_FLOOR"))
            change = targets[ticker] - state["positions"][ticker]["shares"]
            if change:
                orders.append({"ticker": ticker, "side": "BUY" if change > 0 else "SELL", "shares": abs(change)})
        orders.sort(key=lambda o: (o["side"] != "SELL", o["ticker"]))
        plan = {"id": uuid.uuid4().hex[:16], "signal_date": signal_day.isoformat(),
                "signal_month": signal_month, "created_on": as_of.isoformat(),
                "quote_date": quote_day.isoformat(), "planning_equity_cents": equity,
                "prices": {t: str(p) for t, p in prices.items()}, "targets": targets,
                "orders": orders, "fee_cents": fee_cents, "slippage_bps": str(slippage),
                "account_revision": state["revision"]}
        connection.execute("INSERT INTO plans VALUES(?,?,?,?)", (plan["id"], signal_month, "PENDING", json.dumps(plan)))
    return plan


def cancel_plan(path, plan_id):
    with transaction(path) as connection:
        changed = connection.execute("UPDATE plans SET status='CANCELLED' WHERE id=? AND status='PENDING'", (plan_id,)).rowcount
        if not changed:
            raise ValueError("Only an existing pending plan can be cancelled")


def fill_plan(path, plan_id, prices, day, as_of):
    prices = validate_prices(prices, day, as_of)
    with transaction(path) as connection:
        state = account(connection)
        row = connection.execute("SELECT status,payload FROM plans WHERE id=?", (plan_id,)).fetchone()
        if not row or row[0] != "PENDING":
            raise ValueError("Plan is missing, cancelled, or already filled")
        plan = json.loads(row[1])
        if day <= date.fromisoformat(plan["created_on"]) or day <= date.fromisoformat(plan["signal_date"]):
            raise ValueError("Fill prices must come from a session after plan creation and signal date")
        if latest_completed_month(day) != plan["signal_month"]:
            raise ValueError("Plan is for an earlier allocation month; generate current signals and replace it")
        if day < latest_recorded_day(connection, state):
            raise ValueError("Cannot backdate fills before recorded account history")
        if connection.execute("SELECT 1 FROM valuations WHERE day=?", (day.isoformat(),)).fetchone():
            raise ValueError("That session is already marked; fills must precede its valuation")
        if state["revision"] != plan["account_revision"]:
            raise ValueError("Account changed after planning; cancel and replace the pending plan")
        fills = []
        slip = decimal(plan["slippage_bps"]) / 10000
        for order in plan["orders"]:
            ticker, shares, side = order["ticker"], order["shares"], order["side"]
            price = prices[ticker] * (1 + slip if side == "BUY" else 1 - slip)
            gross = cents(price * shares)
            fee = plan["fee_cents"]
            position = state["positions"][ticker]
            realized = 0
            if side == "SELL":
                if shares > position["shares"]:
                    raise ValueError("Paper account cannot sell shares it does not hold")
                basis = (position["cost_cents"] if shares == position["shares"] else
                         int((Decimal(position["cost_cents"]) * shares / position["shares"]).quantize(Decimal("1"), rounding=ROUND_HALF_UP)))
                position["shares"] -= shares
                position["cost_cents"] -= basis
                state["cash_cents"] += gross - fee
                realized = gross - fee - basis
                state["realized_cents"] += realized
            else:
                cost = gross + fee
                if cost > state["cash_cents"]:
                    raise ValueError("Insufficient cash at fill prices; no fills recorded. Cancel and replan with current prices")
                state["cash_cents"] -= cost
                position["shares"] += shares
                position["cost_cents"] += cost
            if state["cash_cents"] < 0:
                raise ValueError("Fees exceed available cash; no fills recorded")
            state["fees_cents"] += fee
            event = {"plan_id": plan_id, "ticker": ticker, "side": side, "shares": shares,
                     "reference_close": str(prices[ticker]), "fill_price": str(price),
                     "gross_cents": gross, "fee_cents": fee, "realized_cents": realized,
                     "cash_after_cents": state["cash_cents"], "date": day.isoformat(),
                     "method": "User-supplied completed-session close with configured slippage"}
            connection.execute("INSERT INTO events VALUES(?,?,?,?)", (uuid.uuid4().hex, "FILL", day.isoformat(), json.dumps(event)))
            fills.append(event)
        state["revision"] += 1
        save_account(connection, state)
        connection.execute("UPDATE plans SET status='FILLED' WHERE id=?", (plan_id,))
        save_valuation(connection, state, prices, day)
    return fills


def save_valuation(connection, state, prices, day):
    holdings = []
    for ticker, position in state["positions"].items():
        value = cents(prices[ticker] * position["shares"])
        holdings.append({"ticker": ticker, "shares": position["shares"], "price": str(prices[ticker]),
                         "value_cents": value, "cost_cents": position["cost_cents"],
                         "unrealized_cents": value - position["cost_cents"]})
    value = equity_cents(state, prices)
    result = {"date": day.isoformat(), "equity_cents": value,
              "cash_cents": state["cash_cents"], "pnl_cents": value - state["initial_cents"],
              "realized_cents": state["realized_cents"], "income_cents": state["income_cents"],
              "fees_cents": state["fees_cents"], "account_revision": state["revision"], "holdings": holdings}
    existing = connection.execute("SELECT payload FROM valuations WHERE day=?", (day.isoformat(),)).fetchone()
    if existing:
        if json.loads(existing[0]) != result:
            raise ValueError("Recorded session valuation differs; history cannot be silently overwritten")
    else:
        connection.execute("INSERT INTO valuations VALUES(?,?)", (day.isoformat(), json.dumps(result)))
    return result


def mark_account(path, prices, day, as_of):
    prices = validate_prices(prices, day, as_of)
    with transaction(path) as connection:
        state = account(connection)
        if day < latest_recorded_day(connection, state):
            raise ValueError("Cannot mark prices before recorded account history")
        result = save_valuation(connection, state, prices, day)
    return result


def add_income(path, amount, day, as_of, kind, reference):
    if kind not in ("dividend", "interest") or not reference.strip():
        raise ValueError("Income needs a dividend/interest type and unique reference")
    amount_cents = cents(amount)
    if decimal(amount) <= 0 or amount_cents <= 0:
        raise ValueError("Income amount must be positive")
    if day >= as_of:
        raise ValueError("Income date must be a completed day before --as-of")
    with transaction(path) as connection:
        state = account(connection)
        if day < latest_recorded_day(connection, state):
            raise ValueError("Cannot backdate income before recorded account history")
        if connection.execute("SELECT 1 FROM valuations WHERE day=?", (day.isoformat(),)).fetchone():
            raise ValueError("Record income before marking that session")
        if connection.execute("SELECT 1 FROM events WHERE id=?", ("income:" + reference,)).fetchone():
            raise ValueError("Income reference already recorded")
        event = {"type": kind, "amount_cents": amount_cents, "reference": reference, "date": day.isoformat()}
        connection.execute("INSERT INTO events VALUES(?,?,?,?)", ("income:" + reference, "INCOME", day.isoformat(), json.dumps(event)))
        state["cash_cents"] += amount_cents
        state["income_cents"] += amount_cents
        state["revision"] += 1
        save_account(connection, state)
    return event


def snapshot(path):
    with transaction(path) as connection:
        state = account(connection)
        plans = [{"status": status, **json.loads(payload)} for status, payload in connection.execute("SELECT status,payload FROM plans ORDER BY rowid")]
        events = [{"kind": kind, **json.loads(payload)} for kind, payload in connection.execute("SELECT kind,payload FROM events ORDER BY rowid")]
        valuations = [json.loads(payload) for (payload,) in connection.execute("SELECT payload FROM valuations ORDER BY day")]
    return {"account": state, "plans": plans, "events": events, "valuations": valuations}


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_account(path, folder):
    data = snapshot(path)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    state = data["account"]
    last = data["valuations"][-1] if data["valuations"] else None
    latest_day = max([state["opened_on"], *[e["date"] for e in data["events"]]])
    current = last is not None and last["account_revision"] == state["revision"]
    text = ["# Faber paper account", "", f"Opened: **{state['opened_on']}** · starting cash: **${state['initial_cents']/100:,.2f}**.",
            f"Current cash: **${state['cash_cents']/100:,.2f}** · realized P&L: **${state['realized_cents']/100:,.2f}** · recorded income: **${state['income_cents']/100:,.2f}**."]
    if current:
        text.extend([f"Marked through **{last['date']}** · account value: **${last['equity_cents']/100:,.2f}** · P&L: **${last['pnl_cents']/100:,.2f}** ({last['pnl_cents']/state['initial_cents']:.2%}).",
                     "These are recorded marks only; market prices may have changed since the stated date."])
    elif not any(p["shares"] for p in state["positions"].values()):
        text.append(f"Cash-only account value: **${state['cash_cents']/100:,.2f}** as of recorded events through **{latest_day}**.")
    else:
        text.append("Account changed since its latest mark; use paper mark for current account value.")
    if current:
        text.extend(["", "| ETF | Shares | Cost basis | Market value | Unrealized P&L |", "| --- | ---: | ---: | ---: | ---: |"])
        for p in last["holdings"]:
            text.append(f"| {p['ticker']} | {p['shares']} | ${p['cost_cents']/100:,.2f} | ${p['value_cents']/100:,.2f} | ${p['unrealized_cents']/100:,.2f} |")
    else:
        text.extend(["", "| ETF | Shares | Cost basis |", "| --- | ---: | ---: |"])
        for t, p in state["positions"].items():
            text.append(f"| {t} | {p['shares']} | ${p['cost_cents']/100:,.2f} |")
    text.extend(["", "Plans:", ""])
    for p in data["plans"]:
        text.append(f"- {p['id']}: {p['status']}, signal {p['signal_date']}, created {p['created_on']}; {len(p['orders'])} orders.")
    text.extend(["", "Paper fills use specified completed-session closes and configured fees/slippage. This is an execution assumption, not broker fill evidence.",
                 "Price marks omit unrecorded dividends, interest and corporate actions. Record income separately. Account P&L covers this ledger's dates, not the historical strategy backtest.",
                 "Historical marks are sparse recorded observations, not a complete daily curve or reliable maximum-drawdown series.", ""])
    (folder / "account.md").write_text("\n".join(text), encoding="utf-8")
    (folder / "ledger.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    positions = [{"ticker": t, "shares": p["shares"]} for t, p in state["positions"].items()]
    write_csv(folder / "holdings.csv", positions, ["ticker", "shares"])
    fields = ["date", "plan_id", "ticker", "side", "shares", "reference_close", "fill_price", "gross_cents", "fee_cents", "realized_cents", "cash_after_cents", "method"]
    write_csv(folder / "fills.csv", [e for e in data["events"] if e["kind"] == "FILL"], fields)
    write_csv(folder / "equity.csv", data["valuations"], ["date", "equity_cents", "cash_cents", "pnl_cents", "realized_cents", "income_cents", "fees_cents"])
    print("\n".join(text))
    return data
