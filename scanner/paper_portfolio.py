"""Session-by-session, capital-constrained anomaly short paper portfolio.

Uses previously generated signal plans; does NOT regenerate historical signals.
Short proceeds are restricted and never counted as buying power.
Daily OHLC cannot resolve same-bar entry/stop ordering.
"""
import json
import math
from pathlib import Path

from scanner.paper_replay import finite_positive


def simulate_portfolio(plans, sessions, starting_equity=100000.0,
                       risk_fraction=0.005, position_limit=0.10,
                       gross_limit=0.50, hold_sessions=5):
    if starting_equity <= 0 or hold_sessions < 1:
        raise ValueError("Invalid account configuration")
    if not all(0 < x <= 1 for x in (risk_fraction, position_limit, gross_limit)):
        raise ValueError("Risk and exposure limits must be in (0,1]")
    dates = sorted(sessions)
    indexed = {d: {b.get("T"): b for b in sessions[d]} for d in dates}
    cash = float(starting_equity)
    open_positions = {}
    records = {}
    equity_curve = []
    signal_items = sorted(plans.items(), key=lambda kv: (kv[1]["signal_date"], kv[0]))
    for key, p in signal_items:
        records[key] = {"id": key, "ticker": p.get("ticker"),
                        "signal_date": p.get("signal_date"), "status": "PENDING"}
    for day_index, day in enumerate(dates):
        bars = indexed[day]
        # Process exits first, but do not reuse same-day proceeds for entries.
        for key, pos in list(open_positions.items()):
            if pos["entry_date"] == day:
                continue
            bar = bars.get(pos["ticker"])
            if bar is None:
                records[key]["mark_status"] = "MISSING_BAR"
                continue
            op = finite_positive(bar.get("o"))
            high = finite_positive(bar.get("h"))
            close = finite_positive(bar.get("c"))
            if not all((op, high, close)) or high < max(op, close):
                records[key]["mark_status"] = "INVALID_BAR"
                continue
            held = day_index - pos["entry_index"]
            if op >= pos["stop"]:
                price, reason = op, "STOP_GAP"
            elif high >= pos["stop"]:
                price, reason = pos["stop"], "STOP"
            elif held >= hold_sessions:
                price, reason = close, "TIME_EXIT"
            else:
                continue
            pnl = round((pos["entry"] - price) * pos["shares"], 2)
            cash += pnl
            records[key].update(status="CLOSED", exit_date=day, exit_price=price,
                                exit_reason=reason, realized_pnl=pnl)
            del open_positions[key]

        # Compute exposure against start-of-session equity; short proceeds restricted.
        def mark_value(pos):
            b = bars.get(pos["ticker"])
            c = finite_positive(b.get("c")) if b else None
            return c

        unrealized = 0.0
        exposure = 0.0
        marks_complete = True
        for pos in open_positions.values():
            close = mark_value(pos)
            if close is None:
                marks_complete = False
                continue
            unrealized += (pos["entry"] - close) * pos["shares"]
            exposure += close * pos["shares"]
        equity = cash + unrealized
        # Missing marks make exposure and equity uncertain; halt new entries.
        eligible = [(key, p) for key, p in signal_items
                    if p.get("signal_date") in dates
                    and dates.index(p["signal_date"]) == day_index - 1]
        for key, plan in eligible:
            rec = records[key]
            ticker = plan.get("ticker")
            entry = finite_positive(plan.get("entry_below"))
            stop = finite_positive(plan.get("stop_above"))
            if plan.get("side") != "short" or not ticker or not entry or not stop or stop <= entry:
                rec["status"] = "INVALID_PLAN"
                continue
            if plan.get("borrow") != "available":
                rec["status"] = "BORROW_NOT_VERIFIED"
                continue
            if any(pos["ticker"] == ticker for pos in open_positions.values()):
                rec["status"] = "DUPLICATE_OPEN_TICKER"
                continue
            b = bars.get(ticker)
            if b is None:
                rec["status"] = "MISSING_ENTRY_BAR"
                continue
            o, lo, hi = (finite_positive(b.get(k)) for k in ("o", "l", "h"))
            if not all((o, lo, hi)) or not lo <= o <= hi:
                rec["status"] = "INVALID_ENTRY_BAR"
                continue
            if o <= entry or (lo < entry and hi >= stop):
                rec["status"] = "AMBIGUOUS_ENTRY_OR_STOP"
                continue
            if lo >= entry:
                rec["status"] = "NOT_TRIGGERED"
                continue
            if not marks_complete or equity <= 0:
                rec["status"] = "PORTFOLIO_MARK_UNAVAILABLE"
                continue
            shares = math.floor(min(equity * risk_fraction / (stop - entry),
                                    equity * position_limit / entry,
                                    max(0, equity * gross_limit - exposure) / entry))
            if shares < 1:
                rec["status"] = "EXPOSURE_OR_RISK_LIMIT"
                continue
            pos = {"ticker": ticker, "entry": entry, "stop": stop,
                   "shares": shares, "entry_date": day, "entry_index": day_index}
            open_positions[key] = pos
            exposure += shares * entry
            rec.update(status="OPEN", entry_date=day, entry_price=entry,
                       shares=shares, stop_price=stop)
        # End-of-day mark. Never silently substitute old prices for missing bars.
        unrealized, gross, marks_complete = 0.0, 0.0, True
        for key, pos in open_positions.items():
            close = mark_value(pos)
            if close is None:
                marks_complete = False
                records[key]["mark_status"] = "MISSING_CLOSE"
                continue
            unrealized += (pos["entry"] - close) * pos["shares"]
            gross += close * pos["shares"]
        equity_curve.append({"date": day, "realized_balance": round(cash, 2),
                             "unrealized_pnl": round(unrealized, 2) if marks_complete else None,
                             "equity": round(cash + unrealized, 2) if marks_complete else None,
                             "gross_exposure": round(gross, 2) if marks_complete else None,
                             "open_positions": len(open_positions)})
    for key, rec in records.items():
        if rec["status"] == "PENDING":
            rec["status"] = "AWAITING_NEXT_SESSION" if rec["signal_date"] in dates else "SIGNAL_OUTSIDE_HISTORY"
    peak, drawdown = starting_equity, 0.0
    for row in equity_curve:
        if row["equity"] is not None:
            peak = max(peak, row["equity"])
            drawdown = min(drawdown, (row["equity"] / peak - 1) * 100)
    closed = [r for r in records.values() if r["status"] == "CLOSED"]
    return {"starting_equity": starting_equity,
            "realized_pnl": round(cash - starting_equity, 2),
            "ending_equity": equity_curve[-1]["equity"] if equity_curve else starting_equity,
            "max_drawdown_pct": round(drawdown, 3),
            "closed_trades": len(closed),
            "wins": sum(r["realized_pnl"] > 0 for r in closed),
            "win_rate": round(sum(r["realized_pnl"] > 0 for r in closed) / len(closed), 4) if closed else None,
            "trades": list(records.values()), "equity_curve": equity_curve,
            "limitations": "OHLC-only model. No borrow fees, hard-to-borrow fees, commissions, "
                           "slippage, locate validation or forced buy-ins. Not broker-executable."}


def save_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
