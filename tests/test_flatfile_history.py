from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

import scanner.flatfile_history as fh


FIELDNAMES = [
    "ticker",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "window_start",
    "transactions",
]


def _row(
    ticker: str,
    close: str,
    window_start: str,
) -> dict[str, str]:
    close_value = float(close)

    return {
        "ticker": ticker,
        "volume": "1000",
        "open": str(close_value - 1.0),
        "close": close,
        "high": str(close_value + 1.0),
        "low": str(close_value - 2.0),
        "window_start": window_start,
        "transactions": "100",
    }


def _write_day(
    root: Path,
    session: str,
    rows: list[dict[str, str]],
) -> None:
    year, month, _ = session.split("-")

    path = (
        root
        / year
        / month
        / f"{session}.csv.gz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(
        path,
        "wt",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)


def test_iter_local_day_aggregate_files(tmp_path):
    _write_day(
        tmp_path,
        "2025-01-02",
        [_row("AAPL", "10", "1735776000000000000")],
    )
    _write_day(
        tmp_path,
        "2025-01-03",
        [_row("AAPL", "11", "1735862400000000000")],
    )
    _write_day(
        tmp_path,
        "2025-01-06",
        [_row("AAPL", "12", "1736121600000000000")],
    )

    paths = fh.iter_local_day_aggregate_files(
        "2025-01-02",
        "2025-01-03",
        root=tmp_path,
    )

    assert [path.name for path in paths] == [
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
        "AAPL",
        "2025-01-02",
        "2025-01-03",
        root=tmp_path,
    )

    assert len(bars) == 2
    assert bars[0].ticker == "AAPL"
    assert bars[0].close == 10.0
    assert bars[1].close == 11.0


def test_read_ticker_history_is_case_sensitive(tmp_path):
    _write_day(
        tmp_path,
        "2025-01-02",
        [
            _row("Aapl", "20", "1735776000000000000"),
            _row("AAPL", "10", "1735776000000000000"),
        ],
    )

    uppercase = fh.read_ticker_history(
        "AAPL",
        "2025-01-02",
        "2025-01-02",
        root=tmp_path,
    )

    mixed_case = fh.read_ticker_history(
        "Aapl",
        "2025-01-02",
        "2025-01-02",
        root=tmp_path,
    )

    lowercase = fh.read_ticker_history(
        "aapl",
        "2025-01-02",
        "2025-01-02",
        root=tmp_path,
    )

    assert len(uppercase) == 1
    assert uppercase[0].ticker == "AAPL"
    assert uppercase[0].close == 10.0

    assert len(mixed_case) == 1
    assert mixed_case[0].ticker == "Aapl"
    assert mixed_case[0].close == 20.0

    assert lowercase == []


def test_read_many_ticker_histories(tmp_path):
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
        [
            _row("AA", "21", "1735862400000000000"),
            _row("AAPL", "11", "1735862400000000000"),
        ],
    )

    histories = fh.read_many_ticker_histories(
        ["AA", "AAPL"],
        "2025-01-02",
        "2025-01-03",
        root=tmp_path,
    )

    assert set(histories) == {"AA", "AAPL"}

    assert [bar.close for bar in histories["AA"]] == [
        20.0,
        21.0,
    ]
    assert [bar.close for bar in histories["AAPL"]] == [
        10.0,
        11.0,
    ]


def test_read_many_preserves_case_distinct_tickers(tmp_path):
    _write_day(
        tmp_path,
        "2025-01-02",
        [
            _row("BCpC", "24.25", "1735776000000000000"),
            _row("BCPC", "182.44", "1735776000000000000"),
        ],
    )

    histories = fh.read_many_ticker_histories(
        ["BCpC", "BCPC"],
        "2025-01-02",
        "2025-01-02",
        root=tmp_path,
    )

    assert histories["BCpC"][0].ticker == "BCpC"
    assert histories["BCpC"][0].close == 24.25

    assert histories["BCPC"][0].ticker == "BCPC"
    assert histories["BCPC"][0].close == 182.44


def test_get_flatfile_daily_bars(tmp_path):
    _write_day(
        tmp_path,
        "2025-01-02",
        [_row("AAPL", "10", "1735776000000000000")],
    )

    bars = fh.get_flatfile_daily_bars(
        "AAPL",
        "2025-01-02",
        "2025-01-02",
        root=tmp_path,
    )

    assert len(bars) == 1

    bar = bars[0]

    assert bar["T"] == "AAPL"
    assert bar["c"] == 10.0
    assert bar["t"] == 1735776000000


def test_invalid_date_range_raises(tmp_path):
    with pytest.raises(
        ValueError,
        match="end_date must be on or after start_date",
    ):
        fh.iter_local_day_aggregate_files(
            "2025-01-03",
            "2025-01-02",
            root=tmp_path,
        )
