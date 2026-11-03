"""Independent candidate outcomes, never a portfolio or strategy validation.

Frozen diagnostic exit: fixed signal stop or the close of the 20th held session,
including entry day. Enter next-session open; charge 10 bps each way plus $0.005
per share each way. Missing sessions, lineage changes and truncated data censor
the candidate instead of supplying a favorable or fabricated terminal value.
"""
from bisect import bisect_right
from statistics import mean, median

from experiments.anomaly_breakout.lineage import lineage_segment


def evaluate(signal, bars, sessions, *, hold_sessions=20, slippage_bps=10.0, commission=0.005):
    if hold_sessions < 1 or not 0 <= slippage_bps < 10000 or commission < 0:
        raise ValueError("Invalid outcome policy")
    ticker, day = signal["ticker"], signal["signal_date"]
    result = {"ticker": ticker, "signal_date": day, "entry_date": signal["entry_date"],
              "anomaly": signal["anomaly"], "status": "censored", "reason": "end_of_development_data"}
    if signal.get("signal_universe_status") != "listed_with_figi" or signal.get("entry_universe_status") != "listed_with_figi":
        return {**result, "status": "excluded", "reason": "unresolved_universe_identity"}
    dates = [b.date for b in bars]
    if dates != sorted(set(dates)):
        raise ValueError("Unsorted or duplicate candidate prices")
    start = bisect_right(dates, day)
    if start == 0 or dates[start - 1] != day or start >= len(bars) or dates[start] != signal["entry_date"]:
        return {**result, "status": "excluded", "reason": "missing_signal_or_next_open"}
    after_signal = sorted(d for d in sessions if d > day)
    if not after_signal or after_signal[0] != signal["entry_date"]:
        return {**result, "reason": "missing_price_session"}
    scheduled = sorted(d for d in sessions if d >= signal["entry_date"])
    if not scheduled or scheduled[0] != signal["entry_date"]:
        return {**result, "reason": "missing_market_calendar"}
    stop = signal["stop"]
    entry = bars[start].open * (1 + slippage_bps / 10000)
    if bars[start].open <= stop:
        return {**result, "status": "excluded", "reason": "entry_open_at_or_below_stop"}
    own_segment = lineage_segment(ticker, day)
    for held in range(hold_sessions):
        index = start + held
        if index >= len(bars) or held >= len(scheduled):
            return result
        bar = bars[index]
        if lineage_segment(ticker, bar.date) != own_segment:
            return {**result, "reason": "lineage_boundary"}
        if bar.date != scheduled[held]:
            return {**result, "reason": "missing_price_session"}
        if bar.open <= stop:
            fill, reason = bar.open, "gap_stop"
        elif bar.low <= stop:
            fill, reason = stop, "stop"
        elif held + 1 == hold_sessions:
            fill, reason = bar.close, "time"
        else:
            continue
        exit_price = fill * (1 - slippage_bps / 10000)
        pnl = exit_price - entry - 2 * commission
        return {**result, "status": "closed", "reason": reason, "exit_date": bar.date,
                "held_sessions": held + 1, "entry_fill": entry, "exit_fill": exit_price,
                "net_pnl_per_share": pnl, "net_return_pct": 100 * pnl / (entry + commission),
                "net_r": pnl / (entry - stop)}
    return result


def summarize(outcomes):
    groups = {}
    for name, selected in [("all_sample_breakouts", outcomes),
                           ("anomaly_subset", [r for r in outcomes if r["anomaly"]]),
                           ("without_anomaly", [r for r in outcomes if not r["anomaly"]])]:
        closed = [r for r in selected if r["status"] == "closed"]
        groups[name] = {"candidates": len(selected), "closed": len(closed),
                        "censored": sum(r["status"] == "censored" for r in selected),
                        "excluded": sum(r["status"] == "excluded" for r in selected),
                        "wins": sum(r["net_r"] > 0 for r in closed),
                        "win_rate_pct": 100 * sum(r["net_r"] > 0 for r in closed) / len(closed) if closed else None,
                        "mean_net_r": mean(r["net_r"] for r in closed) if closed else None,
                        "median_net_r": median(r["net_r"] for r in closed) if closed else None,
                        "mean_net_return_pct": mean(r["net_return_pct"] for r in closed) if closed else None}
    return {"kind": "EXPLORATORY_INDEPENDENT_CANDIDATES_NOT_PORTFOLIO",
            "policy": {"hold_sessions_including_entry": 20, "slippage_bps_each_way": 10,
                       "commission_per_share_each_way": 0.005, "stop": "fixed_signal_stop"},
            "limitations": "Selected tickers, overlapping candidates, unverified legacy split coverage and security mapping; no holdout evaluation or portfolio performance claim",
            "groups": groups}
