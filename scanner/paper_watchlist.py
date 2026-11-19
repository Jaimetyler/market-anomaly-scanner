"""Record next-session short setup observations from daily market bars."""

import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def price_range(bar):
    try:
        low, high = float(bar["l"]), float(bar["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if math.isfinite(low) and math.isfinite(high) and 0 < low < high:
        return low, high
    return None


def next_session_status(plan, bar):
    if plan["borrow"] == "unavailable":
        return "Skipped: borrow unavailable"
    prices = price_range(bar)
    if prices is None:
        return "Review: next-session bar unavailable"
    low, high = prices
    if low >= plan["entry_below"]:
        return "Expired"
    if high > plan["stop_above"]:
        return "Review: entry and stop both crossed"
    if plan["borrow"] != "available":
        return "Trigger seen: borrow unverified"
    return "Triggered"


def update_paper_watchlist(path: Path, sessions, ranked_results):
    """Maintain one plan per ticker and signal date without inventing fills."""
    plans = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    dates = sorted(sessions)
    latest = dates[-1]
    next_dates = dict(zip(dates, dates[1:]))
    bars_by_date = {
        day: {bar.get("T"): bar for bar in bars} for day, bars in sessions.items()
    }
    shown = {}

    for key, plan in plans.items():
        signal_date = plan["signal_date"]
        next_date = plan.get("review_date") or next_dates.get(signal_date)
        if next_date in bars_by_date:
            bar = bars_by_date[next_date].get(plan["ticker"], {})
            plan["review_date"] = next_date
            plan["status"] = next_session_status(plan, bar)
            if next_date == latest:
                shown[key] = plan
        elif signal_date < dates[0] and plan["status"] == "Watching":
            plan["status"] = "Review: signal outside scan window"
            shown[key] = plan

    for result in ranked_results:
        if result["classification"].primary_setup != "PARABOLIC_EXTENSION":
            continue
        ticker = result["ticker"]
        prices = price_range(bars_by_date[latest].get(ticker, {}))
        if prices is None:
            print(f"Paper plan skipped for {ticker}: signal-session range unavailable.")
            continue
        low, high = prices
        key = f"{latest}:{ticker}"
        shown[key] = plans.setdefault(key, {
            "ticker": ticker,
            "signal_date": latest,
            "side": "short",
            "entry_below": low,
            "stop_above": high,
            "reference_risk_per_share": round(high - low, 8),
            "borrow": "unknown",
            "status": "Watching",
        })

    if plans:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            try:
                json.dump(plans, output, indent=2, sort_keys=True, allow_nan=False)
                output.write("\n")
                output.close()
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    print("\nPAPER SHORT PLAN")
    print(f"{'DATE':<12}{'TICKER':<9}{'ENTRY BELOW':>14}{'STOP ABOVE':>14}"
          f"{'REF RISK/SH':>14}  STATUS")
    for plan in shown.values():
        print(f"{plan['signal_date']:<12}{plan['ticker']:<9}"
              f"{plan['entry_below']:>14.8g}{plan['stop_above']:>14.8g}"
              f"{plan['reference_risk_per_share']:>14.8g}  {plan['status']}")
    if not shown:
        print("No paper setups or next-session updates.")
    if plans:
        print(f"Paper ledger: {path}")
