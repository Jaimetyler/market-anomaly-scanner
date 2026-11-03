from datetime import date, timedelta

import pytest

from experiments.anomaly_breakout.candidate_outcomes import evaluate, summarize
from experiments.anomaly_breakout.engine import Bar


def fixture():
    bars = [Bar((date(2020, 1, 1) + timedelta(days=i)).isoformat(), 100, 110, 99, 105, 1000)
            for i in range(22)]
    signal = {"ticker": "XYZ", "signal_date": bars[0].date, "entry_date": bars[1].date,
              "stop": 95, "anomaly": True, "signal_universe_status": "listed_with_figi",
              "entry_universe_status": "listed_with_figi"}
    return signal, bars, {b.date for b in bars}


def test_twentieth_held_session_exit_and_costs():
    signal, bars, sessions = fixture()
    out = evaluate(signal, bars, sessions)
    assert out["status"] == "closed" and out["exit_date"] == bars[20].date
    assert out["held_sessions"] == 20
    assert out["entry_fill"] == pytest.approx(100.1)
    assert out["exit_fill"] == pytest.approx(104.895)
    assert out["net_pnl_per_share"] == pytest.approx(4.785)


def test_entry_day_stop_and_gap_stop():
    signal, bars, sessions = fixture()
    bars[1] = Bar(bars[1].date, 100, 110, 90, 100, 1000)
    out = evaluate(signal, bars, sessions)
    assert out["reason"] == "stop" and out["held_sessions"] == 1 and out["net_r"] < -1
    signal, bars, sessions = fixture()
    bars[2] = Bar(bars[2].date, 90, 92, 80, 85, 1000)
    out = evaluate(signal, bars, sessions)
    assert out["reason"] == "gap_stop" and out["exit_fill"] == pytest.approx(89.91)


def test_missing_or_truncated_prices_do_not_become_completed_trades():
    signal, bars, sessions = fixture()
    assert evaluate(signal, bars[:10], sessions)["status"] == "censored"
    missing = bars[:5] + bars[6:]
    out = evaluate(signal, missing, sessions)
    assert out["status"] == "censored" and out["reason"] == "missing_price_session"


def test_cannot_trade_across_lineage_boundary():
    signal, bars, sessions = fixture()
    bars = [Bar((date(2021, 3, 23) + timedelta(days=i)).isoformat(), b.open, b.high, b.low, b.close, b.volume)
            for i, b in enumerate(bars)]
    signal.update(ticker="AAC", signal_date=bars[0].date, entry_date=bars[1].date)
    out = evaluate(signal, bars, {b.date for b in bars})
    assert out["status"] == "censored" and out["reason"] == "lineage_boundary"


def test_identity_exclusion_and_unevaluable_denominators():
    signal, bars, sessions = fixture()
    good = evaluate(signal, bars, sessions)
    signal["entry_universe_status"] = "listed_missing_figi"
    bad = evaluate(signal, bars, sessions)
    assert bad["status"] == "excluded"
    groups = summarize([good, bad])["groups"]
    assert groups["anomaly_subset"]["closed"] == 1
    assert groups["anomaly_subset"]["excluded"] == 1
    assert groups["anomaly_subset"]["win_rate_pct"] == 100
    assert groups["without_anomaly"]["mean_net_r"] is None


def test_invalid_entry_is_excluded():
    signal, bars, sessions = fixture()
    signal["stop"] = 100
    assert evaluate(signal, bars, sessions)["status"] == "excluded"


def test_missing_session_between_signal_and_entry_is_censored():
    signal, bars, sessions = fixture()
    bars = bars[:1] + bars[2:]
    signal["entry_date"] = bars[1].date
    result = evaluate(signal, bars, sessions)
    assert result["status"] == "censored" and result["reason"] == "missing_price_session"
