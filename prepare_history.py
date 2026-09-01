#!/usr/bin/env python3
"""
prepare_history.py

Resume-safe Massive daily-aggregate flat-file downloader + local coverage validator.

Destination:
    repo root / prepare_history.py

Examples:
    python prepare_history.py --start 2019-01-01 --end 2025-12-31
    python prepare_history.py --start 2019-01-01 --end 2025-12-31 --validate-only

Environment:
    Reads Massive S3 credentials from .env. It accepts several common variable
    names so it can coexist with the project's existing flat-file setup.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

BUCKET = "flatfiles"
ENDPOINT_URL = "https://files.massive.com"
PREFIX = "us_stocks_sip/day_aggs_v1"
DEFAULT_ROOT = Path("data/flatfiles/day_aggs")

# Massive day-aggregate object names contain YYYY-MM-DD.csv.gz.
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.csv\.gz$")


@dataclass(frozen=True)
class RemoteFile:
    session_date: date
    key: str
    size: int


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


def env_first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def make_s3_client():
    access_key = env_first(
        "MASSIVE_S3_ACCESS_KEY",
        "MASSIVE_FLATFILES_ACCESS_KEY",
        "MASSIVE_ACCESS_KEY_ID",
        "AWS_ACCESS_KEY_ID",
    )
    secret_key = env_first(
        "MASSIVE_S3_SECRET_KEY",
        "MASSIVE_FLATFILES_SECRET_KEY",
        "MASSIVE_SECRET_ACCESS_KEY",
        "AWS_SECRET_ACCESS_KEY",
    )

    if not access_key or not secret_key:
        raise RuntimeError(
            "Massive flat-file S3 credentials were not found in .env. "
            "Keep using the same credential variables as your existing downloader, "
            "or add MASSIVE_S3_ACCESS_KEY and MASSIVE_S3_SECRET_KEY."
        )

    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 8, "mode": "adaptive"},
            connect_timeout=20,
            read_timeout=120,
        ),
    )


def extract_date_from_key(key: str) -> date | None:
    match = DATE_RE.search(key)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def list_remote_files(s3, start: date, end: date) -> list[RemoteFile]:
    """
    List only year prefixes needed for the requested range.

    Massive keys are typically:
      us_stocks_sip/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz
    """
    found: dict[date, RemoteFile] = {}

    for year in range(start.year, end.year + 1):
        prefix = f"{PREFIX}/{year}/"
        token = None

        while True:
            kwargs = {
                "Bucket": BUCKET,
                "Prefix": prefix,
                "MaxKeys": 1000,
            }
            if token:
                kwargs["ContinuationToken"] = token

            response = s3.list_objects_v2(**kwargs)

            for obj in response.get("Contents", []):
                key = obj["Key"]
                session_date = extract_date_from_key(key)
                if session_date is None or not (start <= session_date <= end):
                    continue
                found[session_date] = RemoteFile(
                    session_date=session_date,
                    key=key,
                    size=int(obj.get("Size", 0)),
                )

            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
            if not token:
                break

    return [found[d] for d in sorted(found)]


def local_path_for(remote: RemoteFile, root: Path) -> Path:
    # Preserve year/month structure to avoid a giant single directory.
    return (
        root
        / f"{remote.session_date.year:04d}"
        / f"{remote.session_date.month:02d}"
        / f"{remote.session_date.isoformat()}.csv.gz"
    )


def file_looks_complete(path: Path, expected_size: int) -> bool:
    if not path.is_file():
        return False

    # Exact compressed byte size is our cheap first-line resume check.
    if expected_size > 0 and path.stat().st_size != expected_size:
        return False

    # Confirm gzip footer/stream can be read. Reading one row is cheap and catches
    # obviously corrupt/truncated files without fully decompressing every file.
    try:
        with gzip.open(path, "rt", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            next(reader, None)
        return True
    except (OSError, EOFError):
        return False


def download_one(s3, remote: RemoteFile, root: Path, retries: int = 4) -> str:
    destination = local_path_for(remote, root)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if file_looks_complete(destination, remote.size):
        return "skip"

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)

    for attempt in range(1, retries + 1):
        try:
            s3.download_file(BUCKET, remote.key, str(partial))

            if remote.size > 0 and partial.stat().st_size != remote.size:
                raise IOError(
                    f"size mismatch: expected {remote.size:,}, "
                    f"got {partial.stat().st_size:,}"
                )

            # Verify gzip before atomically promoting the file.
            with gzip.open(partial, "rb") as fh:
                fh.read(4096)

            partial.replace(destination)
            return "download"
        except (ClientError, OSError, IOError) as exc:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(
                    f"Failed {remote.session_date} after {retries} attempts: {exc}"
                ) from exc
            time.sleep(min(2 ** attempt, 10))

    raise AssertionError("unreachable")


def discover_local_dates(root: Path, start: date, end: date) -> set[date]:
    dates: set[date] = set()
    if not root.exists():
        return dates

    for path in root.rglob("*.csv.gz"):
        session_date = extract_date_from_key(path.name)
        if session_date and start <= session_date <= end:
            dates.add(session_date)
    return dates


def validate_coverage(
    remote_files: Iterable[RemoteFile],
    root: Path,
    start: date,
    end: date,
) -> tuple[list[date], list[date], list[date]]:
    remote_by_date = {x.session_date: x for x in remote_files}
    remote_dates = set(remote_by_date)
    local_dates = discover_local_dates(root, start, end)

    missing = sorted(remote_dates - local_dates)
    extra = sorted(local_dates - remote_dates)

    bad: list[date] = []
    for session_date in sorted(remote_dates & local_dates):
        remote = remote_by_date[session_date]
        path = local_path_for(remote, root)

        # Compatibility with older downloader layouts: if canonical path is absent,
        # locate an existing same-date file anywhere under root.
        if not path.exists():
            matches = list(root.rglob(f"{session_date.isoformat()}.csv.gz"))
            if len(matches) == 1:
                path = matches[0]

        if not file_looks_complete(path, remote.size):
            bad.append(session_date)

    return missing, extra, bad


def fmt_dates(values: list[date], limit: int = 20) -> str:
    if not values:
        return "none"
    shown = ", ".join(x.isoformat() for x in values[:limit])
    if len(values) > limit:
        shown += f", ... (+{len(values) - limit} more)"
    return shown


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resume-safe Massive day-aggregate history preparation."
    )
    parser.add_argument("--start", required=True, type=parse_date)
    parser.add_argument("--end", required=True, type=parse_date)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Local flat-file root (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Do not download; only compare local coverage with Massive.",
    )
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must be on or after --start")

    load_dotenv()
    s3 = make_s3_client()

    print("=" * 96)
    print("MASSIVE HISTORICAL FLAT-FILE PREPARATION")
    print("=" * 96)
    print(f"Requested: {args.start} -> {args.end}")
    print(f"Local root: {args.root.resolve()}")
    print("Listing Massive day-aggregate objects...")

    remote_files = list_remote_files(s3, args.start, args.end)
    if not remote_files:
        print("ERROR: Massive returned no daily aggregate files for this range.")
        return 2

    total_bytes = sum(x.size for x in remote_files)
    print(f"Remote trading sessions: {len(remote_files):,}")
    print(f"Remote compressed size:  {total_bytes / (1024**3):.2f} GiB")

    if not args.validate_only:
        print()
        print("Resume-safe download...")
        downloaded = 0
        skipped = 0
        failures: list[tuple[date, str]] = []

        for i, remote in enumerate(remote_files, 1):
            try:
                result = download_one(s3, remote, args.root)
                if result == "skip":
                    skipped += 1
                    label = "SKIP"
                else:
                    downloaded += 1
                    label = "DOWNLOADED"
                print(
                    f"[{i:4d}/{len(remote_files)}] "
                    f"{remote.session_date} {label:10s} "
                    f"{remote.size / (1024**2):7.2f} MiB"
                )
            except Exception as exc:
                failures.append((remote.session_date, str(exc)))
                print(
                    f"[{i:4d}/{len(remote_files)}] "
                    f"{remote.session_date} FAILED — {exc}",
                    file=sys.stderr,
                )

        print()
        print(f"Downloaded: {downloaded:,}")
        print(f"Skipped:    {skipped:,}")
        print(f"Failures:   {len(failures):,}")

        if failures:
            print("Failed sessions:")
            for session_date, message in failures[:20]:
                print(f"  {session_date}: {message}")
            if len(failures) > 20:
                print(f"  ... +{len(failures) - 20} more")
            print("Re-run the same command; completed files will be skipped.")

    print()
    print("Validating local coverage...")
    missing, extra, bad = validate_coverage(
        remote_files, args.root, args.start, args.end
    )

    print(f"Expected Massive sessions: {len(remote_files):,}")
    print(f"Missing local sessions:    {len(missing):,}")
    print(f"Bad/incomplete sessions:   {len(bad):,}")
    print(f"Unexpected local sessions: {len(extra):,}")

    if missing:
        print(f"Missing: {fmt_dates(missing)}")
    if bad:
        print(f"Bad:     {fmt_dates(bad)}")
    if extra:
        print(f"Extra:   {fmt_dates(extra)}")

    if missing or bad:
        print()
        print("RESULT: INCOMPLETE — safe to re-run the same command.")
        return 1

    print()
    print("RESULT: COVERAGE VALIDATED")
    print(
        "The requested Massive trading sessions are present locally and passed "
        "compressed-file checks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())