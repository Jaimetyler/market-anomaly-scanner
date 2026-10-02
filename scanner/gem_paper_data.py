"""Current GEM data, including corporate actions, for the paper account."""

import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

from scanner.gem_execution_data import MAPPING, NY, TICKERS, prepare, sessions
from scanner.paper_ledger import positive_price
from scanner.published_gem import decision


def next_session(day):
    return sessions(day + timedelta(days=1), day + timedelta(days=14))[0]


def previous_session(day):
    return sessions(day - timedelta(days=14), day - timedelta(days=1))[-1]


def parse_chart(chart, ticker, as_of):
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"{ticker}: provider returned no prices")
    data = chart["result"][0]
    quotes = data["indicators"]["quote"][0]
    adjusted = data["indicators"]["adjclose"][0]["adjclose"]
    stamps = data["timestamp"]
    if len({len(stamps), len(adjusted), len(quotes["open"]), len(quotes["close"])}) != 1:
        raise ValueError(f"{ticker}: mismatched price arrays")
    bars = []
    for stamp, opening, close, level in zip(stamps, quotes["open"], quotes["close"], adjusted):
        day = datetime.fromtimestamp(stamp, NY).date()
        if day < as_of:
            bars.append({"date": day.isoformat(), "open": opening,
                         "close": close, "adjusted_close": level})
    actions = []
    for kind, field in (("DIVIDEND", "dividends"), ("SPLIT", "splits")):
        for event in data.get("events", {}).get(field, {}).values():
            day = datetime.fromtimestamp(event["date"], NY).date()
            if day >= as_of:
                continue
            action = {"date": day.isoformat(), "kind": kind, "ticker": ticker}
            if kind == "DIVIDEND":
                action["amount"] = str(positive_price(event["amount"]))
            else:
                action["ratio"] = str(positive_price(event["numerator"]) / positive_price(event["denominator"]))
            actions.append(action)
    return bars, actions


def fetch_snapshot(as_of):
    def fetch(ticker):
        end = int(datetime.combine(as_of, datetime.min.time(), NY).timestamp())
        url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
               + urllib.parse.quote(ticker)
               + f"?period1=1167609600&period2={end}&interval=1d&events=div%2Csplits")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=40) as response:
            return parse_chart(json.load(response)["chart"], ticker, as_of)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, TICKERS))
    return {"schema_version": 1, "as_of": as_of.isoformat(),
            "source": "Yahoo daily chart with dividend and split events",
            "corporate_actions_included": True,
            "histories": {t: r[0] for t, r in zip(TICKERS, results)},
            "actions": [a for r in results for a in r[1]]}


def validate_snapshot(snapshot, as_of):
    # Reuse the audited monthly preparation and calendar, then validate the
    # current month's daily bars too (the research audit stops at month end).
    monthly, _ = prepare(snapshot, as_of)
    if snapshot.get("corporate_actions_included") is not True or "actions" not in snapshot:
        raise ValueError("Paper snapshot needs dividend and split events; an audit price-only snapshot is insufficient")
    indexed = {}
    for ticker in TICKERS:
        indexed[ticker] = {}
        for row in snapshot["histories"][ticker]:
            indexed[ticker][row["date"]] = row
    start = date.fromisoformat(monthly[0]["month"])
    end = date.fromisoformat(previous_session(as_of))
    expected = sessions(start, end)
    for ticker, rows in indexed.items():
        present = {d for d in rows if d >= start.isoformat()}
        if present != set(expected):
            raise ValueError(f"{ticker}: missing/extra sessions: {sorted(set(expected)-present)[:5]} / {sorted(present-set(expected))[:5]}")
        for day in expected:
            for field in ("open", "close", "adjusted_close"):
                positive_price(rows[day][field])
    actions, seen = [], set()
    for raw in snapshot["actions"]:
        action = dict(raw)
        day = date.fromisoformat(action["date"])
        ticker, kind = action["ticker"], action["kind"]
        if ticker not in TICKERS or kind not in ("DIVIDEND", "SPLIT"):
            raise ValueError("Invalid corporate action")
        if day >= as_of or not sessions(day, day):
            raise ValueError("Corporate action must be on a completed exchange session")
        key = (ticker, kind, action["date"])
        if key in seen:
            raise ValueError("Duplicate corporate action")
        seen.add(key)
        field = "amount" if kind == "DIVIDEND" else "ratio"
        action[field] = str(positive_price(action[field]).normalize())
        actions.append(action)
    actions.sort(key=lambda a: (a["date"], a["ticker"], a["kind"]))
    return monthly, indexed, actions, end.isoformat()


def signal_at(monthly, cutoff):
    rows = [r for r in monthly if r["month"] <= cutoff]
    if len(rows) < 13:
        raise ValueError("GEM needs 13 completed month-end levels")
    selected, momentum, reason = decision(rows, len(rows)-1)
    return {"signal_date": rows[-1]["month"], "ticker": MAPPING[selected],
            "reason": reason, "momentum_12m": {MAPPING[k]: v for k, v in momentum.items()}}
