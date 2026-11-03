"""One-command, offline development review with a small shareable results ZIP.

Uses existing local inputs only. Never fetches, refreshes caches, reads credentials,
or simulates portfolios with uncertified inputs. 2024-2025 prices/universe snapshots
are outside this diagnostic. Legacy split-cache presence is not proof of coverage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime, timezone
from math import isfinite
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.anomaly_breakout.engine import Bar, detect_signals
from experiments.anomaly_breakout.lineage import CONFIRMED_LINEAGE_BREAKS, lineage_segment
from experiments.anomaly_breakout.local_adapter import read_anomaly_dates, regime_from_spy
from scanner.flatfile_adjustments import adjust_flatfile_history_for_splits
from scanner.flatfile_history import read_many_ticker_histories

START = "2018-01-01"
SIGNAL_START = "2019-01-01"
END = "2023-12-31"
DEFAULT_TICKERS = ["A", "AA", "AAL", "AAC", "AAN"]
OBSERVATIONS = Path("research/maturity_rebuild_v1/master_observations_2019_2025.csv")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def data_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    repo = Path(__file__).resolve().parents[2]
    candidates = [repo / "data", repo.parent / "market-anomaly-scanner" / "data"]
    for candidate in candidates:
        if (candidate / OBSERVATIONS).is_file() and (candidate / "flatfiles/day_aggs").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate master observations and day_aggs; use --data-root PATH")


def canonical_day(value) -> str:
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Expected an ISO YYYY-MM-DD date")
    return value


def split_cache(root: Path, ticker: str):
    """Return validated development events plus evidence, never call an API.

    Legacy cache names collapse letter case. Reject mixed-case symbols even if
    an uppercase-named file exists. Empty legacy lists remain unverified.
    """
    path = root / f"{ticker.upper()}_splits.json"
    evidence = {"file": path.name, "status": "missing", "events": 0}
    if ticker != ticker.upper():
        evidence.update(status="rejected", reason="Legacy uppercase cache cannot establish exact ticker identity")
        return None, evidence
    if not path.is_file():
        return None, evidence
    try:
        raw_bytes = path.read_bytes()
        evidence["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
        payload = json.loads(raw_bytes)
        if not isinstance(payload, list):
            raise ValueError("Expected the legacy list of split events")
        events = []
        seen = set()
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError("Split event is not an object")
            day = canonical_day(row.get("execution_date") or row.get("ex_date") or row.get("date"))
            if not START <= day <= END:
                continue
            if row.get("ticker") != ticker:
                raise ValueError("Split event ticker is missing or does not exactly match")
            values = [row.get("split_from"), row.get("split_to")]
            if any(isinstance(v, bool) for v in values):
                raise ValueError("Invalid split ratio")
            before, after = map(float, values)
            if not all(isfinite(v) and v > 0 for v in (before, after)):
                raise ValueError("Invalid split ratio")
            if day in seen:
                raise ValueError("Duplicate or conflicting split execution dates")
            seen.add(day)
            events.append(SimpleNamespace(execution_date=day, split_from=before, split_to=after))
        evidence.update(status="available_unverified", events=len(events),
                        coverage="Legacy lists contain no query coverage/pagination evidence")
        return events, evidence
    except (OSError, ValueError, TypeError) as error:
        evidence.update(status="rejected", reason=str(error))
        return None, evidence


def convert(bars):
    return [Bar(b.session_date, b.open, b.high, b.low, b.close, b.volume) for b in bars]


def segments(ticker, bars):
    result = defaultdict(list)
    for bar in bars:
        result[lineage_segment(ticker, bar.session_date)].append(bar)
    return result


def adjusted(ticker, bars, events):
    """A later company's splits must never adjust an earlier company's bars."""
    output = []
    for segment, items in segments(ticker, bars).items():
        own_events = [e for e in events if lineage_segment(ticker, e.execution_date) == segment]
        output.extend(adjust_flatfile_history_for_splits(items, own_events))
    return output


def signal_rows(ticker, bars, anomalies, regime, basis):
    rows = []
    # Validate the whole input too: partitioning must not conceal out-of-order
    # dates, duplicates or invalid bars from the detector's normal checks.
    detect_signals(ticker, convert(bars), set(), {})
    for segment, items in segments(ticker, bars).items():
        own_anomalies = {day for day in anomalies if lineage_segment(ticker, day) == segment}
        for signal in detect_signals(ticker, convert(items), own_anomalies, regime):
            if SIGNAL_START <= signal.signal_date <= END:
                rows.append({"basis": basis, **asdict(signal)})
    return rows


def price_review(bars, spy_dates):
    dates = [b.session_date for b in bars]
    date_set = set(dates)
    first, last = (min(dates), max(dates)) if dates else (None, None)
    valid = all(all(isfinite(v) for v in (b.open, b.high, b.low, b.close, b.volume))
                and 0 < b.low <= min(b.open, b.close)
                and b.high >= max(b.open, b.close) and b.volume >= 0 for b in bars)
    missing = sorted(d for d in spy_dates if dates and first <= d <= last and d not in date_set)
    return {"bars": len(bars), "first": first, "last": last,
            "valid_ohlcv": valid, "sorted_unique_dates": dates == sorted(set(dates)),
            "missing_vs_spy_within_observed_span": len(missing), "missing_sample": missing[:20]}


def universe_review(root: Path, expected_days: set[str], tickers: list[str]):
    """Inspect exact-date snapshots; never fill gaps with a future snapshot."""
    inventory, membership, union = [], {}, set()
    for path in sorted(root.glob("*.json")):
        try:
            day = canonical_day(path.stem)
        except (ValueError, TypeError):
            continue
        if not SIGNAL_START <= day <= END:
            continue
        evidence = {"date": day, "file": path.name, "status": "rejected"}
        try:
            raw_bytes = path.read_bytes()
            evidence["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
            payload = json.loads(raw_bytes)
            if not isinstance(payload, dict) or payload.get("as_of_date") != day or payload.get("active_only") is not True:
                raise ValueError("Missing exact as_of_date or active_only=true")
            records = payload.get("tickers")
            if not isinstance(records, list) or not records:
                raise ValueError("Missing or empty ticker snapshot")
            eligible = {}
            missing_ids = 0
            for row in records:
                if not isinstance(row, dict) or not isinstance(row.get("ticker"), str) or not row["ticker"].strip():
                    raise ValueError("Malformed ticker record")
                if row.get("type") != "CS" or row.get("primary_exchange") not in {"XNYS", "XNAS", "XASE"}:
                    continue
                if row.get("active") is not True:
                    raise ValueError("Eligible ticker lacks active=true")
                ticker = row["ticker"].strip()  # retain provider letter case
                identity = row.get("composite_figi")
                if identity is not None and not isinstance(identity, str):
                    raise ValueError("Invalid composite FIGI")
                if ticker in eligible:
                    raise ValueError("Duplicate ticker in one dated snapshot")
                eligible[ticker] = identity
                missing_ids += not bool(identity)
            if not eligible:
                raise ValueError("No eligible common stocks in snapshot")
            membership[day] = {ticker: eligible[ticker] for ticker in tickers if ticker in eligible}
            union.update(eligible)
            evidence.update(status="available_unverified", eligible=len(eligible), missing_figi=missing_ids)
        except (OSError, ValueError, TypeError) as error:
            evidence["reason"] = str(error)
        inventory.append(evidence)
    missing = sorted(expected_days - set(membership))
    return {"snapshots": len(inventory), "usable_snapshots": len(membership),
            "expected_spy_development_sessions": len(expected_days),
            "missing_sessions": len(missing), "missing_sample": missing[:25],
            "distinct_eligible_symbols": len(union),
            "coverage": "Legacy snapshots do not certify pagination or security lineage",
            "inventory": inventory}, membership


def membership_status(membership, day, ticker):
    if day not in membership:
        return "unknown_no_valid_snapshot"
    if ticker not in membership[day]:
        return "not_eligible"
    return "listed_with_figi" if membership[day][ticker] else "listed_missing_figi"


def run(root: Path, tickers: list[str], progress=print):
    observations = root / OBSERVATIONS
    if not observations.is_file() or not (root / "flatfiles/day_aggs").is_dir():
        raise FileNotFoundError("Missing master observations CSV or day_aggs folder")
    report = {"kind": "OFFLINE_INPUT_REVIEW_AND_SIGNAL_DIAGNOSTIC", "performance_ready": False,
              "development": [SIGNAL_START, END], "warmup_start": START, "tickers": tickers,
              "observations_sha256": digest(observations),
              "lineage_boundaries": {t: sorted(CONFIRMED_LINEAGE_BREAKS.get(t, ())) for t in tickers},
              "blockers": ["Selected-ticker diagnostic is not a full-market baseline",
                           "Legacy cache coverage and identity mapping require verification",
                           "Missing-session and delisting/terminal-value policy require review",
                           "No portfolio comparison or profitability claim in this report"]}
    progress("Reading eligible anomaly dates (2019-2023 only)...", flush=True)
    try:
        anomalies = read_anomaly_dates(observations, tickers=tickers, lineage_aware=True)
    except (OSError, ValueError, TypeError) as error:
        report["observation_error"] = str(error)
        report["status"] = "BLOCKED"
        return report, []
    progress("Reading 2018-2023 daily archives once; this may take a few minutes...", flush=True)
    histories = read_many_ticker_histories([*tickers, "SPY"], START, END, root=root / "flatfiles/day_aggs")
    spy = histories.get("SPY", [])
    spy_dates = {b.session_date for b in spy}
    report["prices"] = {t: price_review(histories.get(t, []), spy_dates) for t in [*tickers, "SPY"]}
    # Fingerprint the exact selected bars, not the multi-hundred-MB archives.
    report["selected_bars_sha256"] = hashlib.sha256(json.dumps(
        {t: [asdict(b) for b in bars] for t, bars in sorted(histories.items())},
        sort_keys=True, allow_nan=True).encode()).hexdigest()
    progress("Checking split caches and exact-date universe snapshots...", flush=True)
    caches, evidence = {}, {}
    for ticker in [*tickers, "SPY"]:
        caches[ticker], evidence[ticker] = split_cache(root / "corporate_actions", ticker)
    report["splits"] = evidence
    report["universe"], membership = universe_review(
        root / "historical_universe", {d for d in spy_dates if SIGNAL_START <= d <= END}, tickers)
    rows = []
    report["signal_errors"] = {}
    spy_review = report["prices"]["SPY"]
    if len(spy) < 200 or not spy_review["valid_ohlcv"] or not spy_review["sorted_unique_dates"]:
        report["status"] = "BLOCKED"
        report["signal_errors"]["SPY"] = "Insufficient or invalid SPY history"
        return report, rows
    for basis in ["raw", "split_adjusted_diagnostic"]:
        if basis != "raw" and caches["SPY"] is None:
            report["signal_errors"][basis] = "SPY split cache missing or rejected; adjusted market filter unavailable"
            continue
        try:
            market = spy if basis == "raw" else adjusted("SPY", spy, caches["SPY"])
            detect_signals("SPY", convert(market), set(), {})
            regime = regime_from_spy(convert(market))
        except ValueError as error:
            report["signal_errors"][basis] = str(error)
            continue
        for ticker in tickers:
            bars = histories.get(ticker, [])
            if len(bars) < 22:
                report["signal_errors"][f"{basis}:{ticker}"] = "Insufficient price history"
                continue
            if basis != "raw" and caches[ticker] is None:
                report["signal_errors"][f"{basis}:{ticker}"] = "Split cache missing or rejected"
                continue
            try:
                prices = bars if basis == "raw" else adjusted(ticker, bars, caches[ticker])
                rows.extend(signal_rows(ticker, prices, anomalies.get(ticker, set()), regime, basis))
            except (ValueError, OverflowError) as error:
                report["signal_errors"][f"{basis}:{ticker}"] = str(error)
    for row in rows:
        row["signal_universe_status"] = membership_status(membership, row["signal_date"], row["ticker"])
        row["entry_universe_status"] = membership_status(membership, row["entry_date"], row["ticker"])
    rows.sort(key=lambda row: (row["basis"], row["signal_date"], row["ticker"]))
    report["signals"] = {basis: {"signals": sum(r["basis"] == basis for r in rows),
                                "anomaly_tagged": sum(r["basis"] == basis and r["anomaly"] for r in rows)}
                         for basis in ["raw", "split_adjusted_diagnostic"]}
    report["status"] = "REVIEW_REQUIRED"
    return report, rows


def write_bundle(report, rows, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    signals = io.StringIO(newline="")
    writer = csv.DictWriter(signals, fieldnames=["basis", "ticker", "signal_date", "entry_date", "stop", "anomaly",
                                               "signal_universe_status", "entry_universe_status"])
    writer.writeheader()
    writer.writerows(rows)
    code = Path(__file__).resolve()
    report["runner_sha256"] = digest(code)
    report["code_sha256"] = {p.name: digest(p) for p in [code.with_name(n) for n in
        ("engine.py", "local_adapter.py", "lineage.py")] + [code.parents[2] / "scanner/flatfile_adjustments.py"]}
    report["generated_utc"] = datetime.now(timezone.utc).isoformat()
    with ZipFile(output, "x", ZIP_DEFLATED) as archive:
        archive.writestr("summary.json", json.dumps(report, indent=2, allow_nan=False))
        archive.writestr("signals.csv", signals.getvalue())
        archive.writestr("README.txt", "Offline 2019-2023 input review and signal diagnostics only.\n"
                         "No source dataset, credentials or paper accounts included.\n"
                         "Split-adjusted diagnostics use available legacy caches; completeness is unverified.\n"
                         "No full-universe portfolio backtest or profitability claim.\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Main scanner data folder; auto-detected by default")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--output", type=Path, help="Optional results ZIP path")
    args = parser.parse_args(argv)
    if (not args.tickers or len(args.tickers) != len(set(args.tickers)) or "SPY" in args.tickers
            or any(not t or any(c in t for c in "/\\\0") or t != t.strip() for t in args.tickers)):
        parser.error("Use distinct non-SPY ticker symbols without path separators")
    output = args.output or Path("data/research/anomaly_breakout/local_runs") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_scanner_results.zip")
    try:
        root = data_root(args.data_root)
        print(f"Using data folder: {root}", flush=True)
        report, rows = run(root, args.tickers)
    except (OSError, ValueError, TypeError) as error:
        report, rows = {"status": "BLOCKED", "performance_ready": False, "error": str(error)}, []
    write_bundle(report, rows, output)
    for basis, counts in report.get("signals", {}).items():
        print(f"{basis}: {counts['signals']} signals; {counts['anomaly_tagged']} anomaly-tagged")
    if report.get("error") or report.get("observation_error"):
        print(report.get("error") or report["observation_error"])
    print(f"Status: {report['status']}. Diagnostic only; no performance claim.")
    print(f"Upload this small results file: {output.resolve()}")
    print(f"Size: {output.stat().st_size / 1024:.1f} KB")
    return 2 if report["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
