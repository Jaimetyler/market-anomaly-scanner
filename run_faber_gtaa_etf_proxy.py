"""Independent ETF proxy study of Faber's five-asset rule (not paper replication)."""

import calendar
import argparse
import csv
import datetime as dt
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

from scanner.published_faber_gtaa import ASSETS, backtest


TICKERS = dict(zip(ASSETS, ("SPY", "EFA", "IEF", "VNQ", "GSG")))
START, END = "2007-01", "2026-08"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read()


def monthly_adjusted(ticker):
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
           + urllib.parse.quote(ticker) + "?period1=1167609600&period2=1790812800&interval=1d")
    chart = json.loads(fetch(url))["chart"]["result"][0]
    adjusted = chart["indicators"]["adjclose"][0]["adjclose"]
    result = {}
    for timestamp, level in zip(chart["timestamp"], adjusted):
        day = dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).date()
        month = day.strftime("%Y-%m")
        if START <= month <= END and level is not None:
            if month not in result or day > result[month][0]:
                result[month] = (day, float(level))
    return result


def build():
    data = {asset: monthly_adjusted(ticker) for asset, ticker in TICKERS.items()}
    rates = {}
    raw = fetch("https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS")
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        if row["TB3MS"] not in ("", "."):
            rates[row["observation_date"][:7]] = float(row["TB3MS"]) / 100
    months = sorted(set.intersection(*(set(series) for series in data.values())))
    expected = [f"{year:04d}-{month:02d}" for year in range(2007, 2027)
                for month in range(1, 13) if START <= f"{year:04d}-{month:02d}" <= END]
    if months != expected:
        raise ValueError(f"ETF months missing: {sorted(set(expected) - set(months))}")
    result = []
    for i, month in enumerate(months):
        year, number = map(int, month.split("-"))
        days = calendar.monthrange(year, number)[1]
        # Estimate monthly bill growth from prior month's annualized bank-discount
        # quote. This is not a realized 90-day Treasury-bill return series.
        if i:
            prior = months[i - 1]
            rate = rates[prior]
            purchase_price = 1 - rate * 91 / 360
            cash = (1 / purchase_price) ** (days / 91) - 1
        else:
            cash = 0.0
        result.append({"month": max(data[asset][month][0] for asset in ASSETS).isoformat(),
                       **{asset: data[asset][month][1] for asset in ASSETS},
                       "tbill_return": cash})
    return result


def write_csv(path, records):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="fetch fresh Yahoo and FRED data")
    args = parser.parse_args()
    input_path = Path("faber_gtaa_etf_proxy_input.csv")
    if args.refresh:
        observations = build()
        write_csv(input_path, observations)
    else:
        if not input_path.exists():
            parser.error(f"{input_path} missing; extract it from the research ZIP or use --refresh")
        with input_path.open(newline="", encoding="utf-8-sig") as handle:
            observations = list(csv.DictReader(handle))
    result = backtest(observations)
    write_csv("faber_gtaa_etf_proxy_results.csv", result)
    months = len(result) - 1
    print(f"Invested months: {months}; {result[1]['month']} through {result[-1]['month']}")
    for label, field in (("GTAA ETF proxy", "portfolio_equity"),
                         ("Equal-weight ETF proxy", "equal_weight_equity")):
        value = result[-1][field]
        print(f"{label}: $100,000 -> ${value:,.2f}; annualized {(value / 100000) ** (12 / months) - 1:.2%}")
