"""Offline research runner: real files, no API key, compact export, strict caches."""
import csv
import gzip
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from experiments.anomaly_breakout import run_local as runner
from scanner.flatfile_history import FlatFileBar


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def bar(day, price=100):
    return FlatFileBar("AAC", day, price, price, price, price, 1000, 1, 1)


def test_split_adjustment_never_crosses_company_boundary():
    bars = [bar("2019-01-01"), bar("2021-03-25"), bar("2021-04-01")]
    events = [SimpleNamespace(execution_date="2021-04-01", split_from=1, split_to=10)]
    output = runner.adjusted("AAC", bars, events)
    assert [b.close for b in output] == [100, 10, 100]
    assert [b.volume for b in output] == [1000, 10000, 1000]
    assert bars[1].close == 100  # input objects remain unchanged


@pytest.mark.parametrize("bad", [0, -1, "NaN", "Infinity", None, True])
def test_bad_split_ratio_is_rejected(tmp_path, bad):
    save(tmp_path / "XYZ_splits.json", [{"ticker": "XYZ", "execution_date": "2020-01-02",
                                          "split_from": bad, "split_to": 1}])
    events, report = runner.split_cache(tmp_path, "XYZ")
    assert events is None
    assert report["status"] == "rejected"


def test_missing_empty_case_collision_and_duplicates(tmp_path):
    assert runner.split_cache(tmp_path, "XYZ")[1]["status"] == "missing"
    save(tmp_path / "XYZ_splits.json", [])
    events, report = runner.split_cache(tmp_path, "XYZ")
    assert events == [] and report["status"] == "available_unverified"
    assert runner.split_cache(tmp_path, "xyz")[0] is None
    row = {"ticker": "XYZ", "execution_date": "2020-01-02", "split_from": 1, "split_to": 2}
    save(tmp_path / "XYZ_splits.json", [row, row])
    assert runner.split_cache(tmp_path, "XYZ")[1]["status"] == "rejected"
    row["ticker"] = "xyz"
    save(tmp_path / "XYZ_splits.json", [row])
    assert runner.split_cache(tmp_path, "XYZ")[0] is None


def snapshot(day, **changes):
    return {"as_of_date": day, "active_only": True, "tickers": [
        {"ticker": "XYZ", "type": "CS", "primary_exchange": "XNAS", "active": True,
         "composite_figi": "BBG1"}], **changes}


def test_exact_date_universe_never_uses_future_snapshot(tmp_path):
    save(tmp_path / "2020-01-03.json", snapshot("2020-01-03"))
    # Deliberately invalid holdout bytes must not be opened/parsed.
    (tmp_path / "2024-01-01.json").write_text("not json")
    report, membership = runner.universe_review(tmp_path, {"2020-01-02", "2020-01-03"}, ["XYZ"])
    assert report["snapshots"] == 1 and report["missing_sessions"] == 1
    assert runner.membership_status(membership, "2020-01-02", "XYZ") == "unknown_no_valid_snapshot"
    assert runner.membership_status(membership, "2020-01-03", "XYZ") == "listed_with_figi"
    assert runner.membership_status(membership, "2020-01-03", "OTHER") == "not_eligible"


@pytest.mark.parametrize("changes", [{"active_only": False}, {"as_of_date": "2020-01-01"}, {"tickers": []}])
def test_unusable_universe_snapshot_is_not_coverage(tmp_path, changes):
    save(tmp_path / "2020-01-03.json", snapshot("2020-01-03", **changes))
    report, membership = runner.universe_review(tmp_path, {"2020-01-03"}, ["XYZ"])
    assert membership == {} and report["missing_sessions"] == 1
    assert report["inventory"][0]["status"] == "rejected"


def create_inputs(root):
    observation = root / runner.OBSERVATIONS
    observation.parent.mkdir(parents=True)
    with observation.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticker", "security_id", "signal_date", "research_eligible", "ca_excluded", "research_return_future"])
        writer.writerow(["XYZ", "FIGI:BBG1", "2019-01-20", "True", "False", "DO_NOT_USE"])
        writer.writerow(["XYZ", "future_conflict", "2024-01-01", "True", "False", "DO_NOT_USE"])
    # Enough real file-backed SPY warmup for the market filter.
    dates = [(date(2018, 6, 1) + timedelta(days=i)).isoformat() for i in range(213)]
    dates += [(date(2019, 1, 1) + timedelta(days=i)).isoformat() for i in range(22)]
    for index, day in enumerate(dates):
        path = root / "flatfiles/day_aggs" / (day + ".csv.gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ticker", "open", "high", "low", "close", "volume", "window_start", "transactions"])
            price = 100 + index
            writer.writerow(["SPY", price, price, price, price, 1000, 1, 1])
            if day >= "2019-01-01":
                i = int(day[-2:]) - 1
                high, low, close, volume = (18, 6, 10, 100) if i < 15 else (12, 9, 10, 100)
                if i >= 20:
                    high, low, close, volume = 25, 10, 24, 200
                writer.writerow(["XYZ", close, high, low, close, volume, 1, 1])
    save(root / "corporate_actions/XYZ_splits.json", [])
    save(root / "corporate_actions/SPY_splits.json", [])
    save(root / "historical_universe/2019-01-21.json", snapshot("2019-01-21"))
    save(root / "historical_universe/2019-01-22.json", snapshot("2019-01-22"))


def invoke(root, output):
    env = {key: value for key, value in os.environ.items() if key != "MASSIVE_API_KEY"}
    return subprocess.run([sys.executable, "-m", "experiments.anomaly_breakout.run_local",
                           "--data-root", str(root), "--tickers", "XYZ", "--output", str(output)],
                          env=env, capture_output=True, text=True, timeout=30)


def test_file_backed_cli_without_api_key_exports_only_small_results(tmp_path):
    root = tmp_path / "data"
    create_inputs(root)
    # A secret file is deliberately outside all allowed export paths.
    (root / ".env").write_text("MASSIVE_API_KEY=NEVER_EXPORT_THIS")
    output = tmp_path / "results.zip"
    result = invoke(root, output)
    assert result.returncode == 0, result.stderr
    assert "Upload this small results file" in result.stdout
    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {"summary.json", "signals.csv", "README.txt"}
        summary = json.loads(archive.read("summary.json"))
        assert summary["performance_ready"] is False
        assert summary["signals"]["raw"] == {"signals": 1, "anomaly_tagged": 1}
        assert summary["signals"]["split_adjusted_diagnostic"] == {"signals": 1, "anomaly_tagged": 1}
        signals = archive.read("signals.csv").decode()
        assert "listed_with_figi" in signals
        assert "DO_NOT_USE" not in signals
        assert all(b"NEVER_EXPORT_THIS" not in archive.read(name) for name in archive.namelist())
    assert output.stat().st_size < 100_000
    assert (root / ".env").read_text() == "MASSIVE_API_KEY=NEVER_EXPORT_THIS"


def test_missing_data_still_exports_actionable_results(tmp_path):
    output = tmp_path / "blocked.zip"
    result = invoke(tmp_path / "nonexistent", output)
    assert result.returncode == 2
    with ZipFile(output) as archive:
        summary = json.loads(archive.read("summary.json"))
    assert summary["status"] == "BLOCKED" and "Missing master" in summary["error"]


def test_input_files_not_overwritten_by_output(tmp_path):
    path = tmp_path / "existing.zip"
    path.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        runner.write_bundle({}, [], path)
    assert path.read_bytes() == b"preserve"


def test_automatic_sibling_data_discovery(tmp_path, monkeypatch):
    checkout = tmp_path / "market-anomaly-breakout"
    module_path = checkout / "experiments/anomaly_breakout/run_local.py"
    root = tmp_path / "market-anomaly-scanner/data"
    (root / runner.OBSERVATIONS).parent.mkdir(parents=True)
    (root / runner.OBSERVATIONS).touch()
    (root / "flatfiles/day_aggs").mkdir(parents=True)
    monkeypatch.setattr(runner, "__file__", str(module_path))
    assert runner.data_root(None) == root.resolve()


def test_missing_split_caches_preserve_raw_results_and_report_block(tmp_path):
    root = tmp_path / "data"
    create_inputs(root)
    (root / "corporate_actions/SPY_splits.json").unlink()
    report, rows = runner.run(root, ["XYZ"], progress=lambda *a, **k: None)
    assert report["signals"]["raw"]["signals"] == 1
    assert report["signals"]["split_adjusted_diagnostic"]["signals"] == 0
    assert "SPY split cache" in report["signal_errors"]["split_adjusted_diagnostic"]
    assert report["performance_ready"] is False
