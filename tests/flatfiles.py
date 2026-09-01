from __future__ import annotations

import csv
import gzip
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

MASSIVE_S3_ENDPOINT = "https://files.massive.com"
MASSIVE_S3_BUCKET = "flatfiles"
MASSIVE_DAY_AGGS_PREFIX = "us_stocks_sip/day_aggs_v1"
LOCAL_DAY_AGGS_DIR = Path("data/flatfiles/day_aggs")

REQUIRED_COLUMNS = {
    "ticker", "volume", "open", "close", "high", "low",
    "window_start", "transactions",
}


@dataclass(frozen=True)
class FlatFileObject:
    key: str
    size: int
    last_modified: datetime | None = None

    @property
    def filename(self) -> str:
        return self.key.rsplit("/", 1)[-1]

    @property
    def session_date(self) -> date:
        name = self.filename
        if name.endswith(".csv.gz"):
            name = name[:-7]
        return date.fromisoformat(name)


def _load_boto3():
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required. Install it with: pip install boto3"
        ) from exc
    return boto3, Config


def _credentials() -> tuple[str, str]:
    load_dotenv()
    access_key = (
        os.getenv("MASSIVE_S3_ACCESS_KEY")
        or os.getenv("AWS_ACCESS_KEY_ID")
        or ""
    ).strip()
    secret_key = (
        os.getenv("MASSIVE_S3_SECRET_KEY")
        or os.getenv("AWS_SECRET_ACCESS_KEY")
        or ""
    ).strip()

    if not access_key or not secret_key:
        raise RuntimeError(
            "Missing Massive S3 credentials. Add these to .env:\n"
            "MASSIVE_S3_ACCESS_KEY=...\n"
            "MASSIVE_S3_SECRET_KEY=...\n\n"
            "Use the Flat Files/S3 credentials from the Massive dashboard, "
            "not MASSIVE_API_KEY."
        )
    return access_key, secret_key


def build_s3_client():
    boto3, Config = _load_boto3()
    access_key, secret_key = _credentials()

    return boto3.client(
        "s3",
        endpoint_url=MASSIVE_S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
        config=Config(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=15,
            read_timeout=60,
            signature_version="s3v4",
        ),
    )


def _month_starts(start: date, end: date) -> Iterable[date]:
    if end < start:
        raise ValueError("end must be on or after start")

    current = date(start.year, start.month, 1)
    while current <= end:
        yield current
        current = (
            date(current.year + 1, 1, 1)
            if current.month == 12
            else date(current.year, current.month + 1, 1)
        )


def month_prefix(year: int, month: int) -> str:
    return f"{MASSIVE_DAY_AGGS_PREFIX}/{year:04d}/{month:02d}/"


def local_path_for_key(key: str) -> Path:
    expected = f"{MASSIVE_DAY_AGGS_PREFIX}/"
    if not key.startswith(expected):
        raise ValueError(f"Unexpected Massive key: {key}")
    return LOCAL_DAY_AGGS_DIR / key[len(expected):]


def list_day_aggregate_files(
    start_date: str | date,
    end_date: str | date,
    *,
    client=None,
) -> list[FlatFileObject]:
    start = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    end = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    client = client or build_s3_client()
    found: list[FlatFileObject] = []

    for month_start in _month_starts(start, end):
        prefix = month_prefix(month_start.year, month_start.month)
        token = None

        while True:
            kwargs = {
                "Bucket": MASSIVE_S3_BUCKET,
                "Prefix": prefix,
                "MaxKeys": 1000,
            }
            if token:
                kwargs["ContinuationToken"] = token

            response = client.list_objects_v2(**kwargs)

            for item in response.get("Contents", []):
                key = item["Key"]
                name = key.rsplit("/", 1)[-1]
                if not name.endswith(".csv.gz"):
                    continue
                try:
                    session = date.fromisoformat(name[:-7])
                except ValueError:
                    continue
                if start <= session <= end:
                    found.append(
                        FlatFileObject(
                            key=key,
                            size=int(item.get("Size", 0)),
                            last_modified=item.get("LastModified"),
                        )
                    )

            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
            if not token:
                raise RuntimeError(
                    f"Truncated Massive listing had no continuation token: {prefix}"
                )

    found.sort(key=lambda item: (item.session_date, item.key))
    return found


def download_day_aggregate_file(
    obj: FlatFileObject,
    *,
    client=None,
    force: bool = False,
) -> Path:
    client = client or build_s3_client()
    destination = local_path_for_key(obj.key)

    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".part")

    try:
        client.download_file(MASSIVE_S3_BUCKET, obj.key, str(temp_path))
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded empty file: {obj.key}")
        os.replace(temp_path, destination)
        return destination
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def download_day_aggregate_range(
    start_date: str | date,
    end_date: str | date,
    *,
    force: bool = False,
    client=None,
    progress: bool = True,
) -> list[Path]:
    client = client or build_s3_client()
    objects = list_day_aggregate_files(start_date, end_date, client=client)

    if not objects:
        raise RuntimeError(
            f"No Massive day-aggregate files found for {start_date} -> {end_date}"
        )

    paths: list[Path] = []
    total = len(objects)

    for i, obj in enumerate(objects, start=1):
        destination = local_path_for_key(obj.key)
        cached = destination.exists() and destination.stat().st_size > 0 and not force
        path = download_day_aggregate_file(obj, client=client, force=force)
        paths.append(path)

        if progress:
            source = "CACHE" if cached else "S3"
            print(
                f"[{i:>4}/{total:<4}] {obj.session_date.isoformat()} "
                f"{source:<5} {path.stat().st_size / (1024 * 1024):>6.2f} MB"
            )

    return paths


def validate_day_aggregate_file(path: str | Path) -> dict[str, object]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffixes[-2:] != [".csv", ".gz"]:
        raise ValueError(f"Expected .csv.gz file: {path}")

    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

        row_count = 0
        sample_tickers: list[str] = []
        for row in reader:
            row_count += 1
            ticker = row.get("ticker")
            if ticker and len(sample_tickers) < 5:
                sample_tickers.append(ticker)

    if row_count == 0:
        raise ValueError(f"{path} contains no rows")

    return {
        "path": str(path),
        "rows": row_count,
        "columns": sorted(columns),
        "sample_tickers": sample_tickers,
    }
