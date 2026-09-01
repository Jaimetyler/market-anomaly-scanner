from __future__ import annotations

import csv
import gzip
from datetime import datetime, timezone
from pathlib import Path

import pytest
import scanner.flatfiles as ff


class FakeListClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def list_objects_v2(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeDownloadClient:
    def __init__(self, payload: bytes = b"hello"):
        self.payload = payload
        self.calls = 0

    def download_file(self, bucket, key, filename):
        self.calls += 1
        Path(filename).write_bytes(self.payload)


def test_month_prefix():
    assert ff.month_prefix(2025, 1) == "us_stocks_sip/day_aggs_v1/2025/01/"


def test_local_path_for_key(monkeypatch, tmp_path):
    monkeypatch.setattr(ff, "LOCAL_DAY_AGGS_DIR", tmp_path)
    path = ff.local_path_for_key(
        "us_stocks_sip/day_aggs_v1/2025/01/2025-01-03.csv.gz"
    )
    assert path == tmp_path / "2025/01/2025-01-03.csv.gz"


def test_list_filters_requested_date():
    client = FakeListClient([{
        "Contents": [
            {"Key": "us_stocks_sip/day_aggs_v1/2025/01/2025-01-02.csv.gz", "Size": 100},
            {"Key": "us_stocks_sip/day_aggs_v1/2025/01/2025-01-03.csv.gz", "Size": 200},
        ],
        "IsTruncated": False,
    }])
    result = ff.list_day_aggregate_files(
        "2025-01-03", "2025-01-03", client=client
    )
    assert [x.session_date.isoformat() for x in result] == ["2025-01-03"]
    assert result[0].size == 200


def test_list_handles_pagination():
    client = FakeListClient([
        {
            "Contents": [
                {"Key": "us_stocks_sip/day_aggs_v1/2025/01/2025-01-02.csv.gz", "Size": 1}
            ],
            "IsTruncated": True,
            "NextContinuationToken": "next",
        },
        {
            "Contents": [
                {"Key": "us_stocks_sip/day_aggs_v1/2025/01/2025-01-03.csv.gz", "Size": 1}
            ],
            "IsTruncated": False,
        },
    ])
    result = ff.list_day_aggregate_files(
        "2025-01-02", "2025-01-03", client=client
    )
    assert len(result) == 2
    assert client.calls[1]["ContinuationToken"] == "next"


def test_download_uses_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(ff, "LOCAL_DAY_AGGS_DIR", tmp_path)
    obj = ff.FlatFileObject(
        "us_stocks_sip/day_aggs_v1/2025/01/2025-01-03.csv.gz", 100
    )
    destination = ff.local_path_for_key(obj.key)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"cached")

    client = FakeDownloadClient(b"new")
    result = ff.download_day_aggregate_file(obj, client=client)
    assert result.read_bytes() == b"cached"
    assert client.calls == 0


def test_download_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(ff, "LOCAL_DAY_AGGS_DIR", tmp_path)
    obj = ff.FlatFileObject(
        "us_stocks_sip/day_aggs_v1/2025/01/2025-01-03.csv.gz", 100
    )
    client = FakeDownloadClient(b"new data")
    result = ff.download_day_aggregate_file(obj, client=client)
    assert result.read_bytes() == b"new data"
    assert client.calls == 1


def test_validate_day_aggregate_file(tmp_path):
    path = tmp_path / "2025-01-03.csv.gz"
    fields = [
        "ticker", "volume", "open", "close", "high", "low",
        "window_start", "transactions",
    ]
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "ticker": "AAPL", "volume": "100", "open": "10", "close": "11",
            "high": "12", "low": "9", "window_start": "1735862400000000000",
            "transactions": "50",
        })

    result = ff.validate_day_aggregate_file(path)
    assert result["rows"] == 1
    assert result["sample_tickers"] == ["AAPL"]


def test_validate_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv.gz"
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        handle.write("ticker,close\nAAPL,10\n")

    with pytest.raises(ValueError, match="missing required columns"):
        ff.validate_day_aggregate_file(path)
