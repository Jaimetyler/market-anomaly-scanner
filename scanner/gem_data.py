"""Build an aligned monthly ETF panel using the existing daily data fetcher."""

import calendar
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from scanner.gtaa_signals import fetch_daily, latest_completed_month
from scanner.published_gem import validate_rows

TICKERS = ("SPY", "VEU", "AGG", "BIL", "EFA", "IEF", "VNQ", "GSG")


def month_end_session(year, month):
    """Last regular US equity session (covers the historical ETF sample).

    Only Memorial Day and Good Friday can close a month's last weekday.
    Reject unsupported pre-2007 data; exceptional future closures may require
    a calendar update, rather than accepting an earlier stale snapshot.
    """
    if year < 2007:
        raise ValueError("ETF panel supports 2007 onward")
    # Gregorian Easter (Meeus/Jones/Butcher), then Good Friday.
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    easter = date(year, (h + l - 7 * m + 114) // 31,
                  (h + l - 7 * m + 114) % 31 + 1)
    good_friday = easter - timedelta(days=2)
    day = date(year, month, calendar.monthrange(year, month)[1])
    while (day.weekday() >= 5 or day == good_friday
           or (day.month == 5 and day.weekday() == 0 and day.day >= 25)):
        day -= timedelta(days=1)
    return day


def validate_month_end(day):
    expected = month_end_session(day.year, day.month)
    # Offline index files may label a genuine month-end level with calendar
    # month-end. Daily fetches are checked against the trading date separately.
    calendar_end = date(day.year, day.month, calendar.monthrange(day.year, day.month)[1])
    if day not in (expected, calendar_end):
        raise ValueError(f"Not a completed month-end observation: {day}; expected {expected}")


def monthly_panel(histories, as_of):
    cutoff = latest_completed_month(as_of)
    series = {}
    starts = []
    for ticker in TICKERS:
        records = histories[ticker]
        seen = set()
        monthly = {}
        first = None
        for record in records:
            day = date.fromisoformat(record["date"])
            if day in seen:
                raise ValueError(f"{ticker}: duplicate daily date {day}")
            seen.add(day)
            if day >= as_of or day.strftime("%Y-%m") > cutoff:
                continue
            value = float(record["adjusted_close"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{ticker}: invalid adjusted close on {day}")
            first = min(first, day) if first else day
            month = day.strftime("%Y-%m")
            if month not in monthly or day > monthly[month][0]:
                monthly[month] = (day, value)
        if first is None:
            raise ValueError(f"{ticker}: no completed-month history")
        # Conservatively exclude each series' first observed month, which may
        # be an inception/fetch truncation month. Never backfill ETF histories.
        starts.append(first.strftime("%Y-%m"))
        series[ticker] = monthly
    start = max(starts)
    months = sorted({m for s in series.values() for m in s if start < m <= cutoff})
    if not months or months[-1] != cutoff:
        raise ValueError(f"Stale data: require completed month {cutoff}")
    panel = []
    for month in months:
        missing = [t for t in TICKERS if month not in series[t]]
        if missing:
            raise ValueError(f"Missing {month} data for {', '.join(missing)}")
        dates = {series[t][month][0] for t in TICKERS}
        if len(dates) != 1:
            raise ValueError(f"ETF month-end dates disagree for {month}")
        day = next(iter(dates))
        if day != month_end_session(day.year, day.month):
            raise ValueError(f"Incomplete month-end observation for {month}: {day}")
        panel.append({"month": day.isoformat(), **{t: series[t][month][1] for t in TICKERS}})
    return validate_panel(panel, as_of)


def validate_panel(rows, as_of):
    panel = validate_rows(rows, TICKERS)
    cutoff = latest_completed_month(as_of)
    if panel[-1]["month"][:7] != cutoff:
        raise ValueError(f"Input must end at completed month {cutoff}; use matching --as-of or --refresh")
    for row in panel:
        day = date.fromisoformat(row["month"])
        validate_month_end(day)
    return panel


def fetch_panel(as_of):
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda t: fetch_daily(t, as_of), TICKERS))
    histories = dict(zip(TICKERS, values))
    return monthly_panel(histories, as_of), histories
