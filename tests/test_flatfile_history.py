from __future__ import annotations

import csv
import gzip
from pathlib import Path

import scanner.flatfile_history as fh


FIELDS = [
    "ticker",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "window_start",
    "transactions",
    "vwap",
]


def _write_day(root: Path, session: str, rows: list[dict[str, str]]) -> Path:
    year, month, _ = session.split("-")
    path = root / year / month / f"{session}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return path


def _row(ticker: str, close: str, ts: str) -> dict[str, str]:
    return {
        "ticker": ticker,
        "volume": "1000",
        "open": "10",
        "close": close,
        "high": "12",
        "low": "9",
        "window_start": ts,
        "transactions": "50",
        "vwap": "10.5",
    }


def test_iter_files_filters_and_sorts(tmp_path):
    _write_day(tmp_path, "2025-01-03", [_row("AAPL", "11", "1735862400000000000")])
    _write_day(tmp_path, "2025-01-02", [_row("AAPL", "10", "1735776000000000000")])
    _write_day(tmp_path, "2025-01-06", [_row("AAPL", "12", "1736121600000000000")])

    result = fh.iter_local_day_aggregate_files(
        "2025-01-02", "2025-01-03", root=tmp_path
    )

    assert [p.name for p in result] == [
        "2025-01-02.csv.gz",
        "2025-01-03.csv.gz",
    ]


def test_read_ticker_history(tmp_path):
    _write_day(
        tmp_path,
        "2025-01-02",
        [
            _row("AA", "20", "1735776000000000000"),
            _row("AAPL", "10", "1735776000000000000"),
        ],
    )
    _write_day(
        tmp_path,
        "2025-01-03",
        [_row("AAPL", "11", "1735862400000000000")],
    )

    bars = fh.read_ticker_history(
        "aapl", "2025-01-02", "2025-01-03", root=tmp_path
    )

    assert len(bars) == 2
    assert [bar.close for bar in bars] == [10.0, 11.0]
    assert [bar.session_date for bar in bars] == ["2025-01-02", "2025-01-03"]


def test_read_many_scans_multiple_tickers(tmp_path):
    _write_day(
        tmp_path,
        "2025-01-02",
        [
            _row("AA", "20", "1735776000000000000"),
            _row("AAPL", "10", "1735776000000000000"),
            _row("MSFT", "30", "1735776000000000000"),
        ],
    )

    result = fh.read_many_ticker_histories(
        ["AAPL", "MSFT"], "2025-01-02", "2025-01-02", root=tmp_path
    )

    assert result["AAPL"][0].close == 10.0
    assert result["MSFT"][0].close == 30.0
    assert "AA" not in result


def test_massive_shape_normalizes_nanoseconds_to_milliseconds(tmp_path):
    _write_day(
        tmp_path,
        "2025-01-02",
        [_row("AAPL", "10", "1735776000000000000")],
    )

    bars = fh.get_flatfile_daily_bars(
        "AAPL", "2025-01-02", "2025-01-02", root=tmp_path
    )

    assert bars == [{
        "T": "AAPL",
        "o": 10.0,
        "h": 12.0,
        "l": 9.0,
        "c": 10.0,
        "v": 1000.0,
        "t": 1735776000000,
        "n": 50,
        "vw": 10.5,
    }]


def test_empty_ticker_history_returns_empty(tmp_path):
    _write_day(
        tmp_path,
        "2025-01-02",
        [_row("AAPL", "10", "1735776000000000000")],
    )

    assert fh.read_ticker_history(
        "MSFT", "2025-01-02", "2025-01-02", root=tmp_path
    ) == []


def test_many_empty_request_returns_empty(tmp_path):
    assert fh.read_many_ticker_histories(
        [], "2025-01-02", "2025-01-03", root=tmp_path
    ) == {}