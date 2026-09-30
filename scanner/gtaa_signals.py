"""Current monthly Faber ETF proxy targets; never submits broker orders."""

import calendar
import csv
import json
import math
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from scanner.published_faber_gtaa import ASSETS, backtest

TICKERS = dict(zip(ASSETS, ("SPY", "EFA", "IEF", "VNQ", "GSG")))
LABELS = ("US stocks", "Foreign stocks", "US Treasury bonds", "Real estate", "Commodities")
NY = ZoneInfo("America/New_York")


def latest_completed_month(as_of):
    return (as_of.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")


def fetch_daily(ticker, as_of):
    """Fetch adjusted levels for signals and unadjusted closes for sizing.

    Exclude the as-of day's bar even after its close, avoiding partial bars.
    Yahoo adjusted closes are a proxy for total-return index levels.
    """
    end = int(datetime.combine(as_of, datetime.min.time(), NY).timestamp())
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
           + urllib.parse.quote(ticker)
           + f"?period1=1167609600&period2={end}&interval=1d")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        payload = json.load(response)
    chart = payload["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"{ticker}: data provider returned no history")
    data = chart["result"][0]
    stamps = data["timestamp"]
    adjusted = data["indicators"]["adjclose"][0]["adjclose"]
    closes = data["indicators"]["quote"][0]["close"]
    if not len(stamps) == len(adjusted) == len(closes):
        raise ValueError(f"{ticker}: inconsistent data arrays")
    records = []
    for stamp, level, close in zip(stamps, adjusted, closes):
        day = datetime.fromtimestamp(stamp, NY).date()
        if day >= as_of or level is None or close is None:
            continue
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in (level, close)):
            raise ValueError(f"{ticker}: invalid price on {day}")
        records.append({"date": day.isoformat(), "adjusted_close": float(level),
                        "close": float(close)})
    return records


def monthly_input(histories, as_of):
    """Require aligned month-end observations for all five ETFs."""
    cutoff = latest_completed_month(as_of)
    series = {}
    for asset in ASSETS:
        series[asset] = {}
        seen = set()
        for record in histories[TICKERS[asset]]:
            day = date.fromisoformat(record["date"])
            if day in seen:
                raise ValueError(f"{TICKERS[asset]}: duplicate daily observation")
            seen.add(day)
            month = day.strftime("%Y-%m")
            if day >= as_of or month > cutoff:
                continue
            old = series[asset].get(month)
            if old is None or day.isoformat() > old["date"]:
                series[asset][month] = record
    months = sorted(set.union(*(set(s) for s in series.values())))
    if not months or months[-1] != cutoff:
        raise ValueError(f"Stale data: require completed month {cutoff}")
    rows = []
    for month in months:
        if any(month not in series[a] for a in ASSETS):
            raise ValueError(f"Missing ETF data for {month}")
        days = {series[a][month]["date"] for a in ASSETS}
        if len(days) != 1:
            raise ValueError(f"ETF month-end dates disagree for {month}")
        day = date.fromisoformat(next(iter(days)))
        if calendar.monthrange(day.year, day.month)[1] - day.day > 5:
            raise ValueError(f"Incomplete month-end data for {month}")
        rows.append({"month": day.isoformat(), "tbill_return": 0.0,
                     **{a: series[a][month]["adjusted_close"] for a in ASSETS}})
    return rows


def make_report(rows, as_of, equity=100_000, histories=None, holdings=None):
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError("equity must be positive and finite")
    if holdings is not None:
        if set(holdings) - set(TICKERS.values()):
            raise ValueError("holdings may contain only SPY, EFA, IEF, VNQ, GSG")
        if any(not math.isfinite(v) or v < 0 for v in holdings.values()):
            raise ValueError("holdings shares must be nonnegative and finite")
        if histories is None:
            raise ValueError("holdings sizing requires daily prices; omit --input-csv")
    cutoff = latest_completed_month(as_of)
    confirmed = [r for r in rows if date.fromisoformat(r["month"]).strftime("%Y-%m") <= cutoff]
    if not confirmed or confirmed[-1]["month"][:7] != cutoff:
        raise ValueError(f"Stale input: latest completed month must be {cutoff}")
    calculated = backtest(confirmed)
    last = calculated[-1]
    if last["month"][:7] != cutoff:
        raise ValueError(f"Stale input: latest completed month must be {cutoff}")
    # Historical input may use calendar month-end dates; reject obvious truncation.
    signal_day = date.fromisoformat(last["month"])
    if calendar.monthrange(signal_day.year, signal_day.month)[1] - signal_day.day > 5:
        raise ValueError("Latest monthly observation is too early to be month-end")
    slots = []
    quote_dates = set()
    for asset, label in zip(ASSETS, LABELS):
        ticker = TICKERS[asset]
        target = last[f"{asset}_next"]
        prior = last[f"{asset}_held"]
        weight = 0.2 if target == "ASSET" else 0.0
        slot = {"as_of": as_of.isoformat(), "signal_date": last["month"],
                "allocation_month": as_of.strftime("%Y-%m"),
                "asset": label, "ticker": ticker, "signal": target,
                "adjusted_level": last[f"{asset}_index"], "sma10": last[f"{asset}_sma10"],
                "distance_pct": (last[f"{asset}_index"] / last[f"{asset}_sma10"] - 1) * 100,
                "target_weight": weight, "target_dollars": equity * weight,
                "model_change": ("ENTER" if target == "ASSET" else "EXIT") if prior != target
                                else ("HOLD" if target == "ASSET" else "CASH"),
                "quote_date": None, "quote_close": None, "estimated_target_shares": None,
                "current_shares": None, "share_change": None, "action": None,
                "developing_signal": None}
        if histories is not None:
            bars = [r for r in histories[ticker] if date.fromisoformat(r["date"]) < as_of]
            if not bars:
                raise ValueError(f"{ticker}: no completed daily prices")
            quote = max(bars, key=lambda r: r["date"])
            quote_day = date.fromisoformat(quote["date"])
            if (as_of - quote_day).days > 7:
                raise ValueError(f"{ticker}: stale daily price {quote_day}")
            close = float(quote["close"])
            level = float(quote["adjusted_close"])
            if any(not math.isfinite(v) or v <= 0 for v in (close, level)):
                raise ValueError(f"{ticker}: invalid daily price")
            quote_dates.add(quote_day)
            shares = math.floor(slot["target_dollars"] / close)
            slot.update(quote_date=quote["date"], quote_close=close, estimated_target_shares=shares)
            if holdings is not None:
                current = holdings.get(ticker, 0.0)
                change = shares - current
                slot.update(current_shares=current, share_change=change,
                            action="BUY" if change > 0 else "SELL" if change < 0 else "HOLD")
            if quote["date"][:7] > cutoff:
                average = (sum(float(r[asset]) for r in confirmed[-9:]) + level) / 10
                slot["developing_signal"] = ("ASSET" if level > average else "TBILL"
                                              if level < average else target)
        slots.append(slot)
    if len(quote_dates) > 1:
        raise ValueError("Latest completed daily price dates disagree across ETFs")
    cash = equity - sum(s["estimated_target_shares"] * s["quote_close"] for s in slots) if histories else None
    return {"strategy": "faber-gtaa5 ETF proxy", "as_of": as_of.isoformat(),
            "signal_date": last["month"], "allocation_month": as_of.strftime("%Y-%m"),
            "equity": equity, "cash_weight": 1 - sum(s["target_weight"] for s in slots),
            "estimated_cash_after_rounding": cash, "slots": slots,
            "source": "Yahoo daily adjusted closes" if histories else "User monthly index CSV",
            "holdings_supplied": holdings is not None}


def render_report(report):
    lines = ["# Faber GTAA5 signals", "",
             f"As of **{report['as_of']}** · confirmed signal **{report['signal_date']}** · allocation for **{report['allocation_month']}**.",
             f"Portfolio equity: **${report['equity']:,.2f}** · target cash: **{report['cash_weight']:.0%}**.", "",
             "| ETF | Asset | Adjusted level | 10-month average | Above/below | Target | Model change |",
             "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for s in report["slots"]:
        lines.append(f"| {s['ticker']} | {s['asset']} | {s['adjusted_level']:.3f} | {s['sma10']:.3f} | {s['distance_pct']:+.2f}% | {s['target_weight']:.0%} | {s['model_change']} |")
    if report["slots"][0]["quote_close"] is not None:
        lines.extend(["", "Indicative sizing using completed daily closes; refresh quotes before paper fills.", "",
                      "| ETF | Quote date | Raw close | Target dollars | Whole shares |",
                      "| --- | --- | ---: | ---: | ---: |"])
        for s in report["slots"]:
            lines.append(f"| {s['ticker']} | {s['quote_date']} | ${s['quote_close']:.2f} | ${s['target_dollars']:,.2f} | {s['estimated_target_shares']} |")
        lines.extend(["", f"Estimated cash including rounding: **${report['estimated_cash_after_rounding']:,.2f}**."])
    if report["holdings_supplied"]:
        lines.extend(["", "Paper rebalance against supplied holdings (equity includes cash):", "",
                      "| ETF | Current shares | Target shares | Action | Share change |",
                      "| --- | ---: | ---: | --- | ---: |"])
        for s in report["slots"]:
            lines.append(f"| {s['ticker']} | {s['current_shares']:g} | {s['estimated_target_shares']} | {s['action']} | {s['share_change']:+g} |")
    else:
        lines.extend(["", "Holdings were not supplied. Model changes compare monthly signals; share counts are target positions, not buy/sell orders."])
    preview = [s for s in report["slots"] if s["developing_signal"] is not None]
    if preview:
        lines.extend(["", "Developing month preview — wait for month-end confirmation:", ""])
        lines.extend(f"- {s['ticker']}: {s['developing_signal']}" for s in preview)
    lines.extend(["", f"Source: {report['source']}.",
                  "Signals use adjusted levels; sizing uses raw closes. Each sleeve keeps its own 20% budget; a cash sleeve is not redistributed.",
                  "The current calendar month is excluded from confirmed signals. Fetches exclude the as-of day's daily bar; run the next day to include it.",
                  "Month-end signals become available after the close; paper fills occur in the next session. Midmonth runs display the existing monthly target, not a new entry trigger.",
                  "Equality retains the previous model signal. ETF proxies differ from the paper's original indices. No broker orders, performance projection, or cash-return estimate is produced.", ""])
    return "\n".join(lines)


def run_signals(as_of, output_dir, equity=100_000, input_csv=None, holdings_csv=None):
    histories = None
    if input_csv:
        with Path(input_csv).open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    else:
        with ThreadPoolExecutor(max_workers=5) as pool:
            values = list(pool.map(lambda ticker: fetch_daily(ticker, as_of), TICKERS.values()))
        histories = dict(zip(TICKERS.values(), values))
        rows = monthly_input(histories, as_of)
    holdings = None
    if holdings_csv:
        holdings = {}
        with Path(holdings_csv).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                ticker = row["ticker"].strip().upper()
                if ticker in holdings:
                    raise ValueError(f"Duplicate holdings ticker: {ticker}")
                holdings[ticker] = float(row["shares"])
    report = make_report(rows, as_of, equity, histories, holdings)
    text = render_report(report)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "signals.md").write_text(text, encoding="utf-8")
    (folder / "signals.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (folder / "signals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["slots"][0]))
        writer.writeheader()
        writer.writerows(report["slots"])
    if histories:
        (folder / "daily_snapshot.json").write_text(json.dumps(histories, indent=2) + "\n", encoding="utf-8")
    print(text)
    print(f"Reports written to {folder.resolve()}")
    return report
