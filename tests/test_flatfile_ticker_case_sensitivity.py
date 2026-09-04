from __future__ import annotations

import csv
import gzip
from pathlib import Path

from scanner.flatfile_history import (
    read_many_ticker_histories,
    read_ticker_history,
)


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


def _write_day_file(root: Path) -> None:
    path = root / "2024" / "11" / "2024-11-22.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "ticker": "BCpC",
            "volume": "10035",
            "open": "24.23",
            "close": "24.25",
            "high": "24.3899",
            "low": "24.23",
            "window_start": "1732251600000000000",
            "transactions": "81",
        },
        {
            "ticker": "BCPC",
            "volume": "130425",
            "open": "181.22",
            "close": "182.44",
            "high": "182.82",
            "low": "180.085",
            "window_start": "1732251600000000000",
            "transactions": "4688",
        },
    ]

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


def test_read_ticker_history_does_not_collapse_case_distinct_tickers(
    tmp_path: Path,
) -> None:
    _write_day_file(tmp_path)

    bars = read_ticker_history(
        "BCPC",
        "2024-11-22",
        "2024-11-22",
        root=tmp_path,
    )

    assert len(bars) == 1

    bar = bars[0]

    assert bar.ticker == "BCPC"
    assert bar.open == 181.22
    assert bar.close == 182.44
    assert bar.volume == 130425
    assert bar.transactions == 4688


def test_read_many_keeps_case_distinct_tickers_separate(
    tmp_path: Path,
) -> None:
    _write_day_file(tmp_path)

    histories = read_many_ticker_histories(
        ["BCpC", "BCPC"],
        "2024-11-22",
        "2024-11-22",
        root=tmp_path,
    )

    assert histories["BCpC"][0].close == 24.25
    assert histories["BCPC"][0].close == 182.44

    assert histories["BCpC"][0].ticker == "BCpC"
    assert histories["BCPC"][0].ticker == "BCPC"


def test_uppercase_ticker_does_not_match_mixed_case_ticker(
    tmp_path: Path,
) -> None:
    _write_day_file(tmp_path)

    bars = read_ticker_history(
        "BCPC",
        "2024-11-22",
        "2024-11-22",
        root=tmp_path,
    )

    assert len(bars) == 1
    assert bars[0].close != 24.25
    assert bars[0].close == 182.44
