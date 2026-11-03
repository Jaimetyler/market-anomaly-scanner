import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from zipfile import ZipFile

import pytest

from experiments.anomaly_breakout import run_local as runner
from experiments.anomaly_breakout.local_adapter import read_anomaly_dates
from scanner.flatfile_history import FlatFileBar
from tests.test_anomaly_breakout_local_runner import create_inputs


def test_sample_is_fixed_case_sensitive_and_independent_of_observations(tmp_path):
    root = tmp_path / "historical_universe"
    root.mkdir()
    records = [{"ticker": t, "type": "CS", "active": True, "primary_exchange": "XNAS",
                "composite_figi": "FIGI_" + t} for t in ["ABC", "abc", "XYZ", "DEF", "SPY"]]
    snapshot = {"as_of_date": "2020-01-02", "active_only": True, "tickers": records}
    (root / "2020-01-02.json").write_text(json.dumps(snapshot))
    # No master observations file exists. Selection still succeeds.
    selected, evidence = runner.select_universe_sample(tmp_path, 3)
    snapshot["tickers"] = list(reversed(records))
    (root / "2020-01-02.json").write_text(json.dumps(snapshot))
    assert runner.select_universe_sample(tmp_path, 3) == (selected, evidence)
    assert "SPY" not in selected and len(selected) == 3
    assert evidence["source_symbols"] == 4
    assert evidence["not_conditioned_on_anomalies_or_outcomes"] is True
    assert set(runner.select_universe_sample(tmp_path, 250)[0]) == {"ABC", "abc", "XYZ", "DEF"}


def test_empty_or_invalid_only_universe_cannot_be_sampled(tmp_path):
    with pytest.raises(ValueError, match="No usable"):
        runner.select_universe_sample(tmp_path, 250)


def test_identity_conflict_report_does_not_drop_unrelated_baseline_symbols(tmp_path):
    path = tmp_path / "obs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticker", "security_id", "signal_date", "research_eligible", "ca_excluded"])
        writer.writerows([["BAD", "one", "2020-01-02", "True", "False"],
                          ["BAD", "two", "2020-01-03", "True", "False"],
                          ["GOOD", "one", "2020-01-02", "True", "False"]])
    errors = {}
    assert read_anomaly_dates(path, tickers=["BAD", "GOOD", "NO_ANOMALIES"],
                              lineage_aware=True, identity_errors=errors) == {"GOOD": {"2020-01-02"}}
    assert set(errors) == {"BAD"}
    with pytest.raises(ValueError, match="BAD"):
        read_anomaly_dates(path)


def test_streamed_fingerprint_matches_existing_report_format():
    h = {"b": [FlatFileBar("b", "2020-01-01", 1, 2, 1, 2, 100, None, 1)], "A": []}
    legacy = json.dumps({t: [asdict(b) for b in bars] for t, bars in sorted(h.items())},
                        sort_keys=True, allow_nan=True).encode()
    assert runner.bars_digest(h) == hashlib.sha256(legacy).hexdigest()


def test_mutually_exclusive_selection_arguments():
    with pytest.raises(SystemExit) as error:
        runner.main(["--tickers", "A", "--universe-sample", "250"])
    assert error.value.code == 2


def test_sample_cli_writes_selection_and_outcomes_in_same_small_bundle(tmp_path):
    root, output = tmp_path / "data", tmp_path / "sample.zip"
    create_inputs(root)
    env = {k: v for k, v in os.environ.items() if k != "MASSIVE_API_KEY"}
    result = subprocess.run([sys.executable, "-m", "experiments.anomaly_breakout.run_local",
                             "--universe-sample", "250", "--data-root", str(root), "--output", str(output)],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    with ZipFile(output) as archive:
        summary = json.loads(archive.read("summary.json"))
    assert summary["tickers"] == ["XYZ"]
    assert summary["sample_selection"]["requested"] == 250
    assert summary["sample_selection"]["selected"] == 1
    assert summary["candidate_review"]["groups"]["anomaly_subset"]["censored"] == 1
    assert summary["missing_split_fetches"] == []
    assert summary["performance_ready"] is False
