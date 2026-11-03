"""Regression coverage for ticker reuse in the offline diagnostic."""
import csv
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from experiments.anomaly_breakout import local_adapter as adapter
from experiments.anomaly_breakout.engine import Bar, detect_signals
from experiments.anomaly_breakout.lineage import lineage_segment


def observations(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticker", "security_id", "signal_date", "research_eligible", "ca_excluded"])
        writer.writerows(rows)
    return path


def test_reviewed_boundary_allows_separate_ids_but_not_conflicts(tmp_path):
    path = observations(tmp_path / "obs.csv", [
        ["AAN", "old", "2020-11-30", "True", "False"],
        ["AAN", "new", "2020-12-01", "True", "False"],
    ])
    assert adapter.read_anomaly_dates(path, tickers=["AAN"], lineage_aware=True) == {
        "AAN": {"2020-11-30", "2020-12-01"}
    }
    with pytest.raises(ValueError, match="AAN"):
        adapter.read_anomaly_dates(path, tickers=["AAN"])
    observations(path, [
        ["AAN", "old", "2020-12-01", "True", "False"],
        ["AAN", "new", "2020-12-02", "False", "True"],
    ])
    with pytest.raises(ValueError, match="AAN"):
        adapter.read_anomaly_dates(path, tickers=["AAN"], lineage_aware=True)


@pytest.mark.parametrize("identity", ["", "other"])
def test_unknown_or_missing_identity_still_rejected(tmp_path, identity):
    path = observations(tmp_path / "obs.csv", [
        ["XYZ", "one", "2020-01-01", "True", "False"],
        ["XYZ", identity, "2020-02-01", "True", "False"],
    ])
    with pytest.raises(ValueError, match="XYZ"):
        adapter.read_anomaly_dates(path, lineage_aware=True)


def test_boundary_is_inclusive_and_case_sensitive():
    assert lineage_segment("AAC", "2021-03-24") == 0
    assert lineage_segment("AAC", "2021-03-25") == 1
    assert lineage_segment("aac", "2021-03-25") == 0


def fixture_bars(start):
    out = []
    for i in range(22):
        day = (start + timedelta(days=i)).isoformat()
        high, low, close, volume = (18, 6, 10, 100) if i < 15 else (12, 9, 10, 100)
        if i >= 20:
            high, low, close, volume = 25, 10, 24, 200
        out.append(Bar(day, close, high, low, close, volume))
    return out


@pytest.mark.parametrize("offset", [1, 10, 21])
def test_no_lookback_or_entry_crosses_boundary(tmp_path, monkeypatch, offset):
    # Unsegmented bars produce a breakout. The reviewed boundary cuts either
    # its lookback or its next-open entry, so the diagnostic must emit none.
    start = date(2021, 3, 25) - timedelta(days=offset)
    bars = fixture_bars(start)
    regime = {b.date: True for b in bars}
    assert len(detect_signals("AAC", bars, {bars[0].date}, regime)) == 1
    path = observations(tmp_path / "obs.csv", [
        ["AAC", "old", bars[0].date, "True", "False"],
    ])
    flat = [SimpleNamespace(session_date=b.date, open=b.open, high=b.high,
                            low=b.low, close=b.close, volume=b.volume) for b in bars]
    monkeypatch.setattr(adapter, "read_many_ticker_histories", lambda *a, **k: {"AAC": flat, "SPY": flat * 10})
    monkeypatch.setattr(adapter, "regime_from_spy", lambda _: regime)
    assert adapter.raw_development_smoke(path, tickers=["AAC"]) == []


def test_segment_warms_up_and_retains_own_anomaly(tmp_path, monkeypatch):
    bars = fixture_bars(date(2021, 3, 25))
    regime = {b.date: True for b in bars}
    path = observations(tmp_path / "obs.csv", [
        ["AAC", "old", "2019-01-14", "True", "False"],
        ["AAC", "new", bars[19].date, "True", "False"],
        ["AAC", "holdout", "2024-01-01", "True", "False"],
    ])
    flat = [SimpleNamespace(session_date=b.date, open=b.open, high=b.high,
                            low=b.low, close=b.close, volume=b.volume) for b in bars]
    monkeypatch.setattr(adapter, "read_many_ticker_histories", lambda *a, **k: {"AAC": flat, "SPY": flat * 10})
    monkeypatch.setattr(adapter, "regime_from_spy", lambda _: regime)
    signals = adapter.raw_development_smoke(path, tickers=["AAC"])
    assert len(signals) == 1
    assert signals[0].anomaly is True
