from pathlib import Path
import csv
import pytest
from experiments.anomaly_breakout.local_adapter import read_anomaly_dates


def test_identity_audit_scoped_to_requested_tickers(tmp_path: Path):
    path = tmp_path / "observations.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticker", "security_id", "signal_date", "research_eligible", "ca_excluded"])
        writer.writerow(["A", "FIGI:1", "2020-01-02", "True", "False"])
        writer.writerow(["A", "FIGI:1", "2020-01-03", "True", "False"])
        writer.writerow(["AAN", "FIGI:2", "2020-01-02", "True", "False"])
        writer.writerow(["AAN", "FIGI:3", "2020-01-03", "True", "False"])
    assert read_anomaly_dates(path, tickers=["A"]) == {"A": {"2020-01-02", "2020-01-03"}}
    with pytest.raises(ValueError, match="AAN"):
        read_anomaly_dates(path, tickers=["AAN"])
