from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from scanner.flatfiles import LOCAL_DAY_AGGS_DIR


@dataclass(frozen=True)
class FlatFileBar:
    ticker: str
    session_date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    transactions: int | None
    window_start_ns: int
    vwap: float | None = None

    def as_massive_aggregate(self) -> dict[str, object]:
        result: dict[str, object] = {
            "T": self.ticker,
            "o": self.open,
            "h": self.high,
            "l": self.low,
            "c": self.close,
            "v": self.volume,
            "t": self.window_start_ns // 1_000_000,
        }
        if self.transactions is not None:
            result["n"] = self.transactions
        if self.vwap is not None:
            result["vw"] = self.vwap
        return result


def _parse_date(value: str | date) -> date:
    return date.fromisoformat(value) if isinstance(value, str) else value


def _date_from_path(path: Path) -> date:
    name = path.name
    if not name.endswith(".csv.gz"):
        raise ValueError(f"Expected Massive .csv.gz file: {path}")
    return date.fromisoformat(name[:-7])


def _ticker_from_row(row: dict[str, str]) -> str:
    """
    Return the provider ticker exactly as supplied, aside from whitespace.

    Massive flat files can contain distinct ticker identifiers that differ only
    by letter case. Ticker identity therefore MUST remain case-sensitive here.

    Example:
        BCpC != BCPC

    Uppercasing provider symbols would collapse those two securities and can
    cause one security's OHLCV history to be assigned to another.
    """
    ticker = (row.get("ticker") or "").strip()
    if not ticker:
        raise ValueError("Missing ticker")
    return ticker


def iter_local_day_aggregate_files(
    start_date: str | date,
    end_date: str | date,
    *,
    root: str | Path | None = None,
) -> list[Path]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    root_path = Path(root) if root is not None else LOCAL_DAY_AGGS_DIR
    if not root_path.exists():
        return []

    paths: list[Path] = []
    for path in root_path.rglob("*.csv.gz"):
        try:
            session = _date_from_path(path)
        except ValueError:
            continue
        if start <= session <= end:
            paths.append(path)

    paths.sort(key=lambda p: (_date_from_path(p), str(p)))
    return paths


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required numeric field {key}")
    return float(value)


def _int_or_none(row: dict[str, str], key: str) -> int | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return int(value)


def _float_or_none(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _parse_row(row: dict[str, str], session: date) -> FlatFileBar:
    ticker = _ticker_from_row(row)

    window_start = row.get("window_start")
    if window_start is None or window_start == "":
        raise ValueError("Missing window_start")

    return FlatFileBar(
        ticker=ticker,
        session_date=session.isoformat(),
        open=_float(row, "open"),
        high=_float(row, "high"),
        low=_float(row, "low"),
        close=_float(row, "close"),
        volume=_float(row, "volume"),
        transactions=_int_or_none(row, "transactions"),
        window_start_ns=int(window_start),
        vwap=_float_or_none(row, "vwap"),
    )


def read_ticker_history(
    ticker: str,
    start_date: str | date,
    end_date: str | date,
    *,
    root: str | Path | None = None,
) -> list[FlatFileBar]:
    symbol = ticker.strip()
    if not symbol:
        raise ValueError("ticker must not be empty")

    bars: list[FlatFileBar] = []

    for path in iter_local_day_aggregate_files(
        start_date,
        end_date,
        root=root,
    ):
        session = _date_from_path(path)

        with gzip.open(
            path,
            "rt",
            newline="",
            encoding="utf-8",
        ) as handle:
            reader = csv.DictReader(handle)

            for row in reader:
                row_ticker = (row.get("ticker") or "").strip()

                # Provider ticker identity is case-sensitive.
                if row_ticker == symbol:
                    bars.append(_parse_row(row, session))
                    break

    return bars


def read_many_ticker_histories(
    tickers: Sequence[str],
    start_date: str | date,
    end_date: str | date,
    *,
    root: str | Path | None = None,
    progress: bool = False,
) -> dict[str, list[FlatFileBar]]:
    wanted = {
        ticker.strip()
        for ticker in tickers
        if ticker.strip()
    }

    if not wanted:
        return {}

    result: dict[str, list[FlatFileBar]] = {
        ticker: []
        for ticker in sorted(wanted)
    }

    paths = iter_local_day_aggregate_files(
        start_date,
        end_date,
        root=root,
    )
    total = len(paths)

    for i, path in enumerate(paths, start=1):
        session = _date_from_path(path)
        found: set[str] = set()

        with gzip.open(
            path,
            "rt",
            newline="",
            encoding="utf-8",
        ) as handle:
            reader = csv.DictReader(handle)

            for row in reader:
                ticker = (row.get("ticker") or "").strip()

                if ticker in wanted and ticker not in found:
                    result[ticker].append(
                        _parse_row(row, session)
                    )
                    found.add(ticker)

                    if len(found) == len(wanted):
                        break

        if progress:
            print(
                f"[{i:>4}/{total:<4}] {session.isoformat()} "
                f"matched={len(found):>4}/{len(wanted)}"
            )

    return result


def get_flatfile_daily_bars(
    ticker: str,
    start_date: str | date,
    end_date: str | date,
    *,
    root: str | Path | None = None,
) -> list[dict[str, object]]:
    return [
        bar.as_massive_aggregate()
        for bar in read_ticker_history(
            ticker,
            start_date,
            end_date,
            root=root,
        )
    ]
