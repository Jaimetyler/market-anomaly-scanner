from __future__ import annotations

# DESTINATION: repo root / build_master_dataset.py
#
# Production historical research runner.
#
# Design goals:
#   - Flat files are the canonical bulk-history source.
#   - Mine one signal-year at a time so RAM stays bounded.
#   - Checkpoint each completed year as JSONL.
#   - Safe to resume after sleep/reboot/interruption.
#   - Rebuild events globally at the end so events that cross
#     Dec/Jan are clustered correctly.
#   - Preserve Python types in checkpoints (important for booleans).
#
# Default research window:
#   2019-01-01 -> 2025-12-31
#
# Required local flat-file buffer is already expected to cover the
# miner's HISTORY_LOOKBACK_DAYS and OUTCOME_LOOKFORWARD_DAYS.
#
# Usage:
#   ./.venv/Scripts/python.exe build_master_dataset.py
#
# Resume (default behavior):
#   ./.venv/Scripts/python.exe build_master_dataset.py
#
# Re-run one or more years:
#   ./.venv/Scripts/python.exe build_master_dataset.py --redo-year 2022
#
# Rebuild final CSVs only from completed checkpoints:
#   ./.venv/Scripts/python.exe build_master_dataset.py --finalize-only

import argparse
import csv
import json
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path
import time
from typing import Any, Iterable

from mine_history_ab import mine_history_flatfiles
from scanner.flatfile_history import read_many_ticker_histories
from scanner.miner import (
    MARKET_CALENDAR_TICKER,
    HistoricalMinerConfig,
    build_event_rows,
    trading_dates_from_bars,
)


DEFAULT_START = date(2019, 1, 1)
DEFAULT_END = date(2025, 12, 31)

# The flat-file miner itself intentionally does not call the REST miner's
# bounded validation. A very large limit means "take the whole rolling
# point-in-time core universe" while preserving its deterministic ordering.
FULL_UNIVERSE_LIMIT = 100_000

RESEARCH_DIR = Path("data/research")
CHECKPOINT_DIR = RESEARCH_DIR / "master_checkpoints"

MASTER_OBSERVATIONS = RESEARCH_DIR / "master_observations_2019_2025.csv"
MASTER_EVENTS = RESEARCH_DIR / "master_anomalies_2019_2025.csv"
MASTER_ERRORS = RESEARCH_DIR / "master_errors_2019_2025.csv"
MASTER_MANIFEST = RESEARCH_DIR / "master_manifest_2019_2025.json"


def _year_bounds(year: int, start: date, end: date) -> tuple[date, date]:
    return max(start, date(year, 1, 1)), min(end, date(year, 12, 31))


def _checkpoint_rows_path(year: int) -> Path:
    return CHECKPOINT_DIR / f"observations_{year}.jsonl"


def _checkpoint_meta_path(year: int) -> Path:
    return CHECKPOINT_DIR / f"summary_{year}.json"


def _checkpoint_done_path(year: int) -> Path:
    return CHECKPOINT_DIR / f"DONE_{year}"


def _checkpoint_errors_path(year: int) -> Path:
    return CHECKPOINT_DIR / f"errors_{year}.json"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_json_atomic(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
    )


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid checkpoint JSONL: {path} line {line_no}"
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"Checkpoint row is not an object: {path} line {line_no}"
                )
            rows.append(value)
    return rows


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "security_id",
        "ticker",
        "signal_date",
        "event_id",
        "event_start_date",
        "event_end_date",
        "event_observation_count",
        "primary_setup",
        "setup_tags",
        "detection_triggers",
        "detection_evidence",
        "research_eligible",
        "ca_flagged",
        "ca_excluded",
        "ca_flags",
        "signal_close",
        "next_open",
        "return_1d",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "relative_volume",
        "rsi_14",
        "distance_sma_20_pct",
        "distance_sma_50_pct",
        "atr_expansion",
        "research_return_1d",
        "research_return_2d",
        "research_return_3d",
        "research_return_5d",
        "research_return_10d",
        "research_return_20d",
        "research_max_future_gain_pct",
        "research_max_future_decline_pct",
        "research_mfe_pct",
        "research_mae_pct",
        "research_sessions_to_max_gain",
        "research_sessions_to_max_decline",
        "executable_return_1d",
        "executable_return_2d",
        "executable_return_3d",
        "executable_return_5d",
        "executable_return_10d",
        "executable_return_20d",
        "executable_max_future_gain_pct",
        "executable_max_future_decline_pct",
        "executable_mfe_pct",
        "executable_mae_pct",
        "executable_sessions_to_max_gain",
        "executable_sessions_to_max_decline",
    ]

    seen: set[str] = set()
    all_keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    ordered = [key for key in preferred if key in seen]
    ordered_set = set(ordered)
    ordered.extend(sorted(key for key in all_keys if key not in ordered_set))
    return ordered


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    if not rows:
        _atomic_text(path, "")
        return

    fields = _fieldnames(rows)

    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp, path)


def _security_key(row: dict[str, Any]) -> tuple[str, str, str]:
    # security_id is the preferred identity. Ticker is retained in the key
    # as a guard against malformed/legacy rows, and date makes observations
    # unique.
    return (
        str(row.get("security_id") or row.get("ticker") or ""),
        str(row.get("ticker") or ""),
        str(row.get("signal_date") or ""),
    )


def _dedupe_sort(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _security_key(row)
        if not key[2]:
            raise RuntimeError(f"Observation is missing signal_date: {row!r}")
        if key in by_key and by_key[key] != row:
            raise RuntimeError(
                "Conflicting duplicate observation encountered for "
                f"{key}. Refusing to silently choose one."
            )
        by_key[key] = row

    return [
        by_key[key]
        for key in sorted(by_key, key=lambda k: (k[2], k[1], k[0]))
    ]


def _market_sessions(start: date, end: date) -> list[str]:
    spy_map = read_many_ticker_histories(
        [MARKET_CALENDAR_TICKER],
        start,
        end,
        progress=False,
    )
    flat_bars = spy_map.get(MARKET_CALENDAR_TICKER, [])
    bars = [bar.as_massive_aggregate() for bar in flat_bars]
    sessions = [
        session
        for session in trading_dates_from_bars(bars)
        if start <= date.fromisoformat(session) <= end
    ]
    if not sessions:
        raise RuntimeError(
            "No local SPY sessions found for the master signal window."
        )
    return sessions


def _year_is_complete(year: int) -> bool:
    rows_path = _checkpoint_rows_path(year)
    meta_path = _checkpoint_meta_path(year)
    done_path = _checkpoint_done_path(year)

    if not (rows_path.exists() and meta_path.exists() and done_path.exists()):
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = int(meta["anomaly_observations"])
        actual = len(_read_jsonl(rows_path))
    except Exception:
        return False

    return expected == actual


def _remove_year_checkpoint(year: int) -> None:
    for path in (
        _checkpoint_rows_path(year),
        _checkpoint_meta_path(year),
        _checkpoint_done_path(year),
        _checkpoint_errors_path(year),
    ):
        if path.exists():
            path.unlink()


def _summary_dict(result, elapsed: float, year: int) -> dict[str, Any]:
    return {
        "year": year,
        "signal_start": result.config.start_date.isoformat(),
        "signal_end": result.config.end_date.isoformat(),
        "raw_universe_count": result.raw_universe_count,
        "core_universe_count": result.core_universe_count,
        "selected_security_count": result.selected_ticker_count,
        "securities_completed": result.tickers_completed,
        "securities_failed": result.tickers_failed,
        "market_sessions": result.market_sessions,
        "security_sessions_tested": result.sessions_tested,
        "anomaly_observations": result.anomaly_observations,
        "research_eligible_observations": (
            result.research_eligible_observations
        ),
        "corporate_action_exclusions": result.corporate_action_exclusions,
        "year_local_event_count": None,
        "elapsed_seconds": elapsed,
    }


def mine_year(
    year: int,
    *,
    start: date,
    end: date,
    force_universe_refresh: bool,
) -> dict[str, Any]:
    year_start, year_end = _year_bounds(year, start, end)

    print()
    print("=" * 100)
    print(f"MASTER DATASET — MINING {year}")
    print("=" * 100)
    print(f"Signal window: {year_start} -> {year_end}")
    print("Source:        local Massive daily aggregate flat files")
    print("Universe:      full rolling point-in-time core universe")
    print()

    config = HistoricalMinerConfig(
        start_date=year_start,
        end_date=year_end,
        max_tickers=FULL_UNIVERSE_LIMIT,
    )

    started = time.perf_counter()
    result = mine_history_flatfiles(
        config,
        force_universe_refresh=force_universe_refresh,
        build_events=False,
    )
    elapsed = time.perf_counter() - started

    if result.errors:
        # Preserve everything for diagnosis, but DO NOT mark the year done.
        _write_json_atomic(
            _checkpoint_errors_path(year),
            result.errors,
        )
        _write_jsonl_atomic(
            _checkpoint_rows_path(year),
            result.rows,
        )
        summary = _summary_dict(result, elapsed, year)
        _write_json_atomic(_checkpoint_meta_path(year), summary)

        raise RuntimeError(
            f"{year} finished with {len(result.errors)} failed securities. "
            f"Partial rows/errors were checkpointed, but the year was NOT "
            f"marked complete. Fix/retry {year} before finalizing."
        )

    written = _write_jsonl_atomic(
        _checkpoint_rows_path(year),
        result.rows,
    )

    if written != result.anomaly_observations:
        raise RuntimeError(
            f"{year} checkpoint count mismatch: wrote {written}, "
            f"miner reported {result.anomaly_observations}."
        )

    summary = _summary_dict(result, elapsed, year)
    _write_json_atomic(_checkpoint_meta_path(year), summary)

    # DONE is written last. Its existence is the transaction commit.
    _atomic_text(
        _checkpoint_done_path(year),
        f"{year} complete\nrows={written}\n",
    )

    print()
    print(
        f"{year} CHECKPOINT COMPLETE — "
        f"{written:,} anomaly observations in {elapsed:.2f}s"
    )
    return summary


def finalize(
    *,
    start: date,
    end: date,
) -> dict[str, Any]:
    years = list(range(start.year, end.year + 1))

    missing = [
        year
        for year in years
        if not _year_is_complete(year)
    ]
    if missing:
        raise RuntimeError(
            "Cannot finalize. Missing/incomplete yearly checkpoints: "
            + ", ".join(str(year) for year in missing)
        )

    print()
    print("=" * 100)
    print("FINALIZING MASTER DATASET")
    print("=" * 100)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for year in years:
        year_rows = _read_jsonl(_checkpoint_rows_path(year))
        all_rows.extend(year_rows)
        summaries.append(
            json.loads(
                _checkpoint_meta_path(year).read_text(encoding="utf-8")
            )
        )
        print(f"Loaded {year}: {len(year_rows):,} observations")

    rows = _dedupe_sort(all_rows)

    print()
    print(f"Unique daily anomaly observations: {len(rows):,}")
    print("Building one global trading-session calendar...")
    sessions = _market_sessions(start, end)
    print(f"Master market sessions: {len(sessions):,}")

    print("Clustering events globally across year boundaries...")
    event_rows = build_event_rows(
        rows=rows,
        session_dates=sessions,
    )

    # Deterministic final order.
    event_rows = sorted(
        event_rows,
        key=lambda row: (
            str(
                row.get("event_start_date")
                or row.get("signal_date")
                or ""
            ),
            str(row.get("ticker") or ""),
            str(row.get("security_id") or ""),
        ),
    )

    _write_csv_atomic(MASTER_OBSERVATIONS, rows)
    _write_csv_atomic(MASTER_EVENTS, event_rows)

    error_rows: list[dict[str, Any]] = []
    for year in years:
        error_path = _checkpoint_errors_path(year)
        if error_path.exists():
            errors = json.loads(error_path.read_text(encoding="utf-8"))
            for error in errors:
                error_rows.append({"year": year, **error})
    _write_csv_atomic(MASTER_ERRORS, error_rows)

    eligible = sum(
        1 for row in rows if row.get("research_eligible") is True
    )
    excluded = len(rows) - eligible

    manifest = {
        "dataset_version": 1,
        "source": "Massive daily aggregate flat files",
        "signal_start": start.isoformat(),
        "signal_end": end.isoformat(),
        "years": years,
        "daily_observations": len(rows),
        "research_eligible_observations": eligible,
        "corporate_action_exclusions": excluded,
        "global_events": len(event_rows),
        "market_sessions": len(sessions),
        "observation_file": str(MASTER_OBSERVATIONS),
        "event_file": str(MASTER_EVENTS),
        "error_file": str(MASTER_ERRORS),
        "year_summaries": summaries,
    }
    _write_json_atomic(MASTER_MANIFEST, manifest)

    print()
    print("=" * 100)
    print("MASTER DATASET COMPLETE")
    print("=" * 100)
    print(f"Daily observations: {len(rows):>10,}")
    print(f"Research eligible:  {eligible:>10,}")
    print(f"CA exclusions:      {excluded:>10,}")
    print(f"Global events:      {len(event_rows):>10,}")
    print()
    print(f"Observations: {MASTER_OBSERVATIONS}")
    print(f"Events:       {MASTER_EVENTS}")
    print(f"Manifest:     {MASTER_MANIFEST}")

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the resume-safe 2019-2025 master anomaly research "
            "dataset from local Massive daily aggregate flat files."
        )
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START.isoformat(),
        help="First signal date (default: 2019-01-01)",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END.isoformat(),
        help="Last signal date (default: 2025-12-31)",
    )
    parser.add_argument(
        "--redo-year",
        type=int,
        action="append",
        default=[],
        help="Delete and rerun this year's checkpoint; repeatable.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Do not mine; rebuild final CSVs from completed checkpoints.",
    )
    parser.add_argument(
        "--force-universe-refresh",
        action="store_true",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if end < start:
        raise SystemExit("--end cannot be before --start")

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    years = list(range(start.year, end.year + 1))

    invalid_redo = sorted(set(args.redo_year) - set(years))
    if invalid_redo:
        raise SystemExit(
            "--redo-year outside requested range: "
            + ", ".join(str(year) for year in invalid_redo)
        )

    for year in args.redo_year:
        print(f"Removing checkpoint for {year}...")
        _remove_year_checkpoint(year)

    if not args.finalize_only:
        for year in years:
            if _year_is_complete(year):
                meta = json.loads(
                    _checkpoint_meta_path(year).read_text(encoding="utf-8")
                )
                print(
                    f"{year}: checkpoint already complete "
                    f"({meta['anomaly_observations']:,} observations) — SKIP"
                )
                continue

            mine_year(
                year,
                start=start,
                end=end,
                force_universe_refresh=args.force_universe_refresh,
            )

    finalize(start=start, end=end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())