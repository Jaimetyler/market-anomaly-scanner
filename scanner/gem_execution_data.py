"""Daily prices and US equity session validation for the GEM timing audit."""

import calendar
import json
import math
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

from scanner.gem_data import month_end_session
from scanner.gtaa_signals import latest_completed_month
from scanner.published_gem import validate_rows

NY = ZoneInfo("America/New_York")
MAPPING = {"us_stocks": "SPY", "foreign_stocks": "VEU",
           "aggregate_bonds": "AGG", "tbill_index": "BIL"}
TICKERS = tuple(MAPPING.values())
EXTRA_CLOSURES = {date(2007, 1, 2), date(2012, 10, 29), date(2012, 10, 30),
                  date(2018, 12, 5), date(2025, 1, 9)}


@lru_cache(maxsize=32)
def holidays(year):
    """Regular NYSE holiday dates for 2007–2028, plus known full closures.

    New Year's Day on Saturday is NOT observed on the prior Friday at NYSE.
    Half-day sessions remain valid sessions. See docs/gem_execution.md.
    """
    if not 2007 <= year <= 2028:
        raise ValueError("Execution calendar supports 2007–2028; update it for other years")
    def nth(month, weekday, n):
        first = date(year, month, 1)
        return first + timedelta(days=(weekday-first.weekday()) % 7 + 7*(n-1))
    def observed(day):
        return day + timedelta(days=-1 if day.weekday() == 5 else 1 if day.weekday() == 6 else 0)
    # Gregorian Easter, used only for Good Friday.
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b+8)//25
    g = (b-f+1)//3
    h = (19*a+b-d-g+15) % 30
    i, k = c//4, c % 4
    l = (32+2*e+2*i-h-k) % 7
    m = (a+11*h+22*l)//451
    easter = date(year, (h+l-7*m+114)//31, (h+l-7*m+114) % 31+1)
    memorial = date(year, 5, 31)
    memorial -= timedelta(days=memorial.weekday())
    new_year = date(year, 1, 1)
    if new_year.weekday() == 6:
        new_year += timedelta(days=1)
    days = {new_year, nth(1, 0, 3), nth(2, 0, 3), easter-timedelta(days=2),
            memorial, observed(date(year, 7, 4)), nth(9, 0, 1), nth(11, 3, 4),
            observed(date(year, 12, 25))}
    if year >= 2022:
        days.add(observed(date(year, 6, 19)))
    return frozenset(days | {d for d in EXTRA_CLOSURES if d.year == year})


def sessions(start, end):
    result = []
    day = start
    while day <= end:
        if day.weekday() < 5 and day not in holidays(day.year):
            result.append(day.isoformat())
        day += timedelta(days=1)
    return result


def fetch_ohlc(ticker, as_of):
    end = int(datetime.combine(as_of, datetime.min.time(), NY).timestamp())
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
           + urllib.parse.quote(ticker)
           + f"?period1=1167609600&period2={end}&interval=1d")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        chart = json.load(response)["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"{ticker}: provider returned no prices")
    data = chart["result"][0]
    quotes = data["indicators"]["quote"][0]
    adjusted = data["indicators"]["adjclose"][0]["adjclose"]
    stamps = data["timestamp"]
    if not len(stamps) == len(adjusted) == len(quotes["open"]) == len(quotes["close"]):
        raise ValueError(f"{ticker}: mismatched price-array lengths")
    rows = []
    for stamp, opening, close, level in zip(stamps, quotes["open"], quotes["close"], adjusted):
        day = datetime.fromtimestamp(stamp, NY).date()
        if day >= as_of:
            continue
        # Retain nulls so validation reports the actual missing observation.
        rows.append({"date": day.isoformat(), "open": opening, "close": close,
                     "adjusted_close": level})
    return rows


def fetch_snapshot(as_of):
    with ThreadPoolExecutor(max_workers=4) as pool:
        histories = dict(zip(TICKERS, pool.map(lambda t: fetch_ohlc(t, as_of), TICKERS)))
    return {"schema_version": 1, "as_of": as_of.isoformat(),
            "source": "Yahoo daily chart: open, close, adjusted close",
            "histories": histories}


def prepare(snapshot, as_of):
    """Validate full daily sessions; build fresh monthly signals from ONE snapshot."""
    if snapshot.get("schema_version") != 1 or snapshot.get("as_of") != as_of.isoformat():
        raise ValueError("Snapshot schema/as-of mismatch; use its original --as-of or --refresh")
    histories = snapshot["histories"]
    if set(histories) != set(TICKERS):
        raise ValueError("Snapshot must contain exactly SPY, VEU, AGG and BIL")
    cutoff = latest_completed_month(as_of)
    year, month = map(int, cutoff.split("-"))
    end = month_end_session(year, month)
    indexed = {}
    first_months = []
    for ticker in TICKERS:
        by_day = {}
        previous = None
        for raw in histories[ticker]:
            day = date.fromisoformat(raw["date"])
            if previous is not None and day <= previous:
                raise ValueError(f"{ticker}: dates must be ordered and unique")
            previous = day
            if day >= as_of:
                raise ValueError(f"{ticker}: snapshot includes an unfinished/future day {day}")
            if day > end:
                continue
            try:
                values = {k: float(raw[k]) for k in ("open", "close", "adjusted_close")}
            except (TypeError, ValueError) as error:
                raise ValueError(f"{ticker}: missing daily price on {day}") from error
            if any(not math.isfinite(v) or v <= 0 for v in values.values()):
                raise ValueError(f"{ticker}: invalid price on {day}")
            ratio = values["adjusted_close"] / values["close"]
            adjusted_open = values["open"] * ratio
            if not math.isfinite(adjusted_open) or adjusted_open <= 0:
                raise ValueError(f"{ticker}: invalid adjusted open on {day}")
            by_day[day.isoformat()] = {**values, "adjusted_open": adjusted_open}
        if not by_day:
            raise ValueError(f"{ticker}: no completed history")
        indexed[ticker] = by_day
        first_months.append(min(by_day)[:7])
    # Match the existing study's conservative exclusion of inception months.
    first = date.fromisoformat(max(first_months)+"-01")
    start = (first.replace(day=28)+timedelta(days=4)).replace(day=1)
    expected = sessions(start, end)
    if not expected:
        raise ValueError("No common full-month daily history")
    expected_set = set(expected)
    for ticker in TICKERS:
        present = {d for d in indexed[ticker] if start.isoformat() <= d <= end.isoformat()}
        missing, extra = expected_set-present, present-expected_set
        if missing or extra:
            raise ValueError(f"{ticker}: session mismatch; missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")
    months = sorted({d[:7] for d in expected})
    monthly = []
    for key in months:
        y, m = map(int, key.split("-"))
        day = month_end_session(y, m).isoformat()
        monthly.append({"month": day,
                        **{a: indexed[t][day]["adjusted_close"] for a, t in MAPPING.items()}})
    monthly = validate_rows(monthly)
    baseline = monthly[12]["month"]
    daily = [{"date": d, "assets": {t: indexed[t][d] for t in TICKERS}}
             for d in expected if d >= baseline]
    return monthly, daily
