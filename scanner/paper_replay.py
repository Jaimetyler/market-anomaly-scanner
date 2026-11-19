"""Conservative, reproducible paper replay for short anomaly plans.

Input: existing paper_watchlist ledger JSON and grouped daily bars JSON mapping
ISO session dates to Massive-style bars. No broker orders are placed.
"""
import argparse
import json
import math
from pathlib import Path


def finite_positive(value):
    try:
        n = float(value)
        return n if math.isfinite(n) and n > 0 else None
    except (TypeError, ValueError):
        return None


def replay(plans, sessions, starting_cash=100000.0, risk_fraction=0.005,
           max_notional_fraction=0.10, holding_sessions=5):
    """Replay in date order. Never infer intraday event ordering from OHLC."""
    if not 0 < risk_fraction <= 1 or not 0 < max_notional_fraction <= 1:
        raise ValueError("Risk and notional fractions must be in (0,1].")
    if holding_sessions < 1 or starting_cash <= 0:
        raise ValueError("Invalid account or holding period")
    dates = sorted(sessions)
    orders = sorted(plans.items(), key=lambda item: (item[1]["signal_date"], item[0]))
    records = []
    for key, plan in orders:
        record = {"id": key, "ticker": plan["ticker"],
                  "signal_date": plan["signal_date"], "side": plan.get("side"),
                  "status": "NOT_EVALUATED"}
        records.append(record)
        if plan.get("side") != "short":
            record["status"] = "UNSUPPORTED_SIDE"
            continue
        if plan.get("borrow") != "available":
            record["status"] = "BORROW_NOT_VERIFIED"
            continue
        entry = finite_positive(plan.get("entry_below"))
        stop = finite_positive(plan.get("stop_above"))
        if not entry or not stop or stop <= entry:
            record["status"] = "INVALID_PLAN"
            continue
        future = [d for d in dates if d > plan["signal_date"]]
        if not future:
            record["status"] = "AWAITING_NEXT_SESSION"
            continue
        day = future[0]
        bars = sessions[day]
        bar = next((b for b in bars if b.get("T") == plan["ticker"]), None)
        if bar is None:
            record["status"] = "MISSING_ENTRY_BAR"
            continue
        op = finite_positive(bar.get("o"))
        low = finite_positive(bar.get("l"))
        high = finite_positive(bar.get("h"))
        if not all((op, low, high)) or low > high or not low <= op <= high:
            record["status"] = "INVALID_ENTRY_BAR"
            continue
        if op <= entry or (low < entry and high >= stop):
            record["status"] = "AMBIGUOUS_ENTRY_OR_STOP"
            continue
        if low >= entry:
            record["status"] = "NOT_TRIGGERED"
            continue
        # Conservative trigger fill at the threshold, never the favorable low.
        risk_per_share = stop - entry
        shares = math.floor(min(starting_cash * risk_fraction / risk_per_share,
                                starting_cash * max_notional_fraction / entry))
        if shares < 1:
            record["status"] = "INSUFFICIENT_SIZE"
            continue
        record.update(entry_date=day, entry_price=entry, shares=shares,
                      stop_price=stop, status="OPEN")
        later = [d for d in dates if d > day]
        for idx, exit_day in enumerate(later):
            exit_bar = next((b for b in sessions[exit_day]
                             if b.get("T") == plan["ticker"]), None)
            if not exit_bar:
                record["status"] = "MISSING_EXIT_BAR"
                break
            o = finite_positive(exit_bar.get("o"))
            h = finite_positive(exit_bar.get("h"))
            c = finite_positive(exit_bar.get("c"))
            if not all((o, h, c)):
                record["status"] = "INVALID_EXIT_BAR"
                break
            if o >= stop:
                exit_price, reason = o, "STOP_GAP"
            elif h >= stop:
                exit_price, reason = stop, "STOP"
            elif idx + 1 >= holding_sessions:
                exit_price, reason = c, "TIME_EXIT"
            else:
                continue
            record.update(exit_date=exit_day, exit_price=exit_price,
                          pnl=round((entry - exit_price) * shares, 2),
                          status="CLOSED", exit_reason=reason)
            break
    realized = round(sum(r.get("pnl", 0) for r in records), 2)
    return {"starting_cash": starting_cash, "realized_pnl": realized,
            "indicative_equity": round(starting_cash + realized, 2),
            "warning": "Independent signal replay; not a capital-constrained portfolio. "
                       "No borrow fees, slippage, commissions or intraday sequencing.",
            "trades": records}


def main():
    parser = argparse.ArgumentParser(description="Offline anomaly short paper replay")
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plans = json.loads(args.plans.read_text(encoding="utf-8"))
    sessions = json.loads(args.sessions.read_text(encoding="utf-8"))
    report = replay(plans, sessions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Reviewed {len(report['trades'])} plans; realized P&L: "
          f"${report['realized_pnl']:.2f}")
    print(report["warning"])


if __name__ == "__main__":
    main()
