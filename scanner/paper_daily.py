"""One daily paper-account cycle. No scheduler installation or broker orders."""

import argparse
import io
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
from datetime import date, datetime, timezone
from pathlib import Path

from scanner.gtaa_signals import (
    NY, fetch_daily, latest_completed_month, make_report, monthly_input, render_report,
)
from scanner.paper_ledger import (
    SYMBOLS, cancel_plan, create_plan, decimal, equity_cents, export_account,
    fill_plan, mark_account, snapshot, validate_prices, write_csv,
)


@contextmanager
def runner_lock(db):
    """OS releases this advisory lock if the process exits, including crashes."""
    lock_path = Path(str(db) + ".daily.lock")
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        locked = False
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise ValueError("Another daily runner is using this account; wait for it to finish") from error
        try:
            yield
        finally:
            if locked:
                handle.seek(0)
                if sys.platform == "win32":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def last_recorded_day(data):
    days = [data["account"]["opened_on"]]
    days.extend(e["date"] for e in data["events"])
    days.extend(v["date"] for v in data["valuations"])
    return date.fromisoformat(max(days))


def fetch_histories(as_of):
    with ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(lambda ticker: fetch_daily(ticker, as_of), SYMBOLS))
    return dict(zip(SYMBOLS, records))


def save_signals(folder, report, histories):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "signals.md").write_text(render_report(report), encoding="utf-8")
    (folder / "signals.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(folder / "signals.csv", report["slots"], list(report["slots"][0]))
    (folder / "daily_snapshot.json").write_text(json.dumps(histories, indent=2) + "\n", encoding="utf-8")


def run_daily(db, output_dir, as_of, histories=None, fee=None, slippage_bps=None):
    db, folder = Path(db), Path(output_dir)
    if not db.is_file():
        raise ValueError("Paper account missing; run python -m scanner.paper_cli init first")
    if fee is not None and decimal(fee) < 0:
        raise ValueError("Fee must be nonnegative")
    if slippage_bps is not None and not 0 <= decimal(slippage_bps) < 10000:
        raise ValueError("Slippage must be between 0 and 10000 bps")
    with runner_lock(db):
        data = snapshot(db)
        if as_of < last_recorded_day(data):
            raise ValueError("Runner date precedes recorded account history")
        if histories is None:
            histories = fetch_histories(as_of)
        monthly = monthly_input(histories, as_of)
        # Validate complete signals and prices before touching the ledger.
        report = make_report(monthly, as_of, histories=histories)
        quote_day = date.fromisoformat(report["slots"][0]["quote_date"])
        prices = validate_prices({s["ticker"]: s["quote_close"] for s in report["slots"]}, quote_day, as_of)
        messages, review, fills_count = [], False, 0
        messages.append(f"Latest completed price date: {quote_day}; confirmed signal: {report['signal_date']}.")
        for plan in data["plans"]:
            if plan["status"] != "PENDING":
                continue
            fill_month = latest_completed_month(quote_day)
            if fill_month > plan["signal_month"]:
                cancel_plan(db, plan["id"])
                messages.append(f"Cancelled expired plan {plan['id']} for signal month {plan['signal_month']}.")
                continue
            if quote_day <= date.fromisoformat(plan["created_on"]):
                messages.append(f"WAITING: plan {plan['id']} was created {plan['created_on']}; need a completed session after that date.")
                continue
            if fill_month < plan["signal_month"]:
                messages.append(f"WAITING: current prices precede plan {plan['id']}'s allocation month.")
                continue
            current = snapshot(db)
            if quote_day < last_recorded_day(current):
                messages.append("WAITING: price date is older than recorded account history.")
                continue
            if any(v["date"] == quote_day.isoformat() for v in current["valuations"]):
                messages.append(f"WAITING: {quote_day} is already marked; pending fills need a later session.")
                continue
            try:
                fills = fill_plan(db, plan["id"], prices, quote_day, as_of)
                fills_count += len(fills)
                messages.append(f"Filled plan {plan['id']}: {len(fills)} simulated orders using {quote_day} closes.")
            except ValueError as error:
                messages.append(f"REVIEW: plan {plan['id']} remains pending: {error}")
                review = True
        data = snapshot(db)
        if quote_day >= last_recorded_day(data):
            try:
                mark_account(db, prices, quote_day, as_of)
                messages.append(f"Account marked through {quote_day}.")
            except ValueError as error:
                messages.append(f"REVIEW: mark unavailable: {error}")
                review = True
        else:
            messages.append("WAITING: latest quote predates account history; account has not been marked backward.")
        data = snapshot(db)
        signal_month = latest_completed_month(as_of)
        current_plans = [p for p in data["plans"] if p["signal_month"] == signal_month and p["status"] != "CANCELLED"]
        account_equity = equity_cents(data["account"], prices)
        report = make_report(monthly, as_of, equity=account_equity / 100, histories=histories)
        can_plan = quote_day >= last_recorded_day(data) or not any(p["shares"] for p in data["account"]["positions"].values())
        if not current_plans and not can_plan:
            messages.append("WAITING: newer prices are needed before planning against held positions.")
        elif not current_plans and not review:
            previous = data["plans"]
            assumptions = previous[-1] if previous else {}
            chosen_fee = fee if fee is not None else str(decimal(assumptions.get("fee_cents", 0)) / 100)
            chosen_slip = slippage_bps if slippage_bps is not None else assumptions.get("slippage_bps", "0")
            try:
                plan = create_plan(db, report, as_of, chosen_fee, chosen_slip)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
                messages.append(f"Created monthly plan {plan['id']}: {len(plan['orders'])} orders; waits for a later completed session.")
            except ValueError as error:
                messages.append(f"REVIEW: could not create current plan: {error}")
                review = True
        elif current_plans:
            plan = current_plans[-1]
            messages.append(f"Current monthly plan {plan['id']}: {plan['status']}; no duplicate plan created.")
        save_signals(folder / "signals", report, histories)
        write_csv(folder / "prices.csv", [{"ticker": t, "date": quote_day.isoformat(), "close": str(prices[t])} for t in SYMBOLS], ["ticker", "date", "close"])
        with redirect_stdout(io.StringIO()):
            data = export_account(db, folder)
        result = {"ran_at": datetime.now(timezone.utc).isoformat(), "as_of": as_of.isoformat(),
                  "price_date": quote_day.isoformat(), "status": "REVIEW" if review else "OK",
                  "fills_recorded": fills_count, "messages": messages}
        (folder / "daily_status.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        with (folder / "daily_runs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
        from scanner.paper_dashboard import write_dashboard
        write_dashboard(folder / "dashboard.html", data, report, result)
        print("\nDaily runner:")
        print(f"  Recorded cash: ${data['account']['cash_cents']/100:,.2f}; new simulated fills: {fills_count}.")
        for message in messages:
            print(f"  {message}")
        print(f"Dashboard: {(folder / 'dashboard.html').resolve()}")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/research/faber_paper/account.sqlite")
    parser.add_argument("--output-dir", help="default: account database folder")
    parser.add_argument("--snapshot-json", help="use saved daily history instead of fetching; date/freshness checks still apply")
    parser.add_argument("--fee", help="fee dollars for newly created plans; existing plans retain their assumptions")
    parser.add_argument("--slippage-bps", help="slippage for newly created plans")
    args = parser.parse_args(argv)
    try:
        histories = json.loads(Path(args.snapshot_json).read_text(encoding="utf-8")) if args.snapshot_json else None
        result = run_daily(args.db, args.output_dir or Path(args.db).parent,
                           datetime.now(NY).date(), histories, args.fee, args.slippage_bps)
        if result["status"] == "REVIEW":
            parser.exit(2, "Daily run needs review; see daily_status.json.\n")
    except (ValueError, KeyError, IndexError, TypeError, ArithmeticError, OSError, sqlite3.Error) as error:
        parser.exit(2, f"Daily runner unavailable: {error}\n")


if __name__ == "__main__":
    main()
