"""Read-only, local-data research adapter. No API calls and no holdout reads."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from experiments.anomaly_breakout.engine import Bar, detect_signals
from scanner.flatfile_history import read_many_ticker_histories

DEVELOPMENT_END = "2023-12-31"


def read_anomaly_dates(path: str | Path, *, tickers: list[str] | None = None) -> dict[str, set[str]]:
    """Eligible observations only. Exact case-sensitive provider ticker retained.

    The observation's security_id is also tracked; any ticker with multiple
    identities is rejected until a date-specific identity mapping is audited.
    """
    selected = set(tickers) if tickers is not None else None
    dates: dict[str, set[str]] = defaultdict(set)
    identities: dict[str, set[str]] = defaultdict(set)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"ticker", "security_id", "signal_date", "research_eligible", "ca_excluded"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing anomaly columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            day = row["signal_date"]
            if not ("2019-01-01" <= day <= DEVELOPMENT_END):
                continue
            ticker = row["ticker"].strip()
            if selected is not None and ticker not in selected:
                continue
            identities[ticker].add(row["security_id"].strip())
            if row["research_eligible"].strip().lower() == "true" and row["ca_excluded"].strip().lower() != "true":
                dates[ticker].add(day)
    ambiguous = sorted(t for t, ids in identities.items() if len(ids) != 1 or "" in ids)
    if ambiguous:
        raise ValueError(f"Ticker identity ambiguity (sample): {ambiguous[:20]}")
    return dict(dates)


def regime_from_spy(bars: list[Bar]) -> dict[str, bool]:
    """Signal-day SPY close vs same-day 200-session SMA; no future prices."""
    result: dict[str, bool] = {}
    for i in range(199, len(bars)):
        sample = bars[i - 199:i + 1]
        result[bars[i].date] = bars[i].close > sum(x.close for x in sample) / 200
    return result


def raw_development_smoke(
    observation_csv: str | Path,
    *,
    tickers: list[str],
    start: str = "2018-01-01",
    root: str | Path | None = None,
) -> list:
    """Generate **unadjusted** diagnostic signals only.

    Raw flatfiles are not safe for profitability analysis. The caller must
    replace this loader with verified split-adjusted, lineage-continuous,
    point-in-time eligible bars before portfolio simulation.
    """
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Provide unique ticker symbols")
    anomaly = read_anomaly_dates(observation_csv, tickers=tickers)
    histories = read_many_ticker_histories(
        [*tickers, "SPY"], start, DEVELOPMENT_END, root=root, progress=False
    )
    def convert(items):
        return [
            Bar(x.session_date, x.open, x.high, x.low, x.close, x.volume)
            for x in items
        ]
    spy = convert(histories.get("SPY", []))
    if len(spy) < 200:
        raise ValueError("Insufficient SPY history; refusing unfiltered signals")
    regime = regime_from_spy(spy)
    signals = []
    for ticker in tickers:
        bars = convert(histories[ticker])
        if len(bars) < 22:
            continue
        signals.extend(detect_signals(ticker, bars, anomaly.get(ticker, set()), regime))
    return sorted(signals, key=lambda x: (x.signal_date, x.ticker))
