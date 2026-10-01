"""GEM month-end-close versus next-session-open execution timing audit."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import date, datetime
from pathlib import Path

from scanner.gem_execution_data import MAPPING, NY, fetch_snapshot, prepare
from scanner.gem_research import metrics
from scanner.published_gem import backtest, write_csv


def simulate(daily, signals, mode="next_open", starting_equity=100_000.0, cost_bps=10.0):
    """Trade fractional total-return index units, not actual ETF shares.

    A next-open switch keeps the OLD holding through its overnight gap. The
    new holding earns its own open-to-close return. Unchanged signals do not
    trade. Close-reference changes happen after that day's closing valuation,
    preserving the monthly reference's before-next-trade equity convention.
    """
    if mode not in ("next_open", "signal_close"):
        raise ValueError("Unknown execution mode")
    if not math.isfinite(starting_equity) or starting_equity <= 0:
        raise ValueError("starting_equity must be positive and finite")
    if not math.isfinite(cost_bps) or not 0 <= cost_bps < 5000:
        raise ValueError("cost_bps must be finite and between 0 and 5000 (exclusive)")
    days = [r["date"] for r in daily]
    if len(days) < 2 or days != sorted(set(days)):
        raise ValueError("Daily dates must be unique, sorted, and include a return session")
    if days[0] not in signals or set(signals)-set(days):
        raise ValueError("Signals must align with daily dates and include the baseline")
    if any(s["ticker"] not in ("SPY", "VEU", "AGG") for s in signals.values()):
        raise ValueError("GEM must select SPY, VEU or AGG")
    for row in daily:
        date.fromisoformat(row["date"])
        for ticker in ("SPY", "VEU", "AGG", "BIL"):
            for field in ("adjusted_open", "adjusted_close"):
                value = row["assets"][ticker][field]
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Invalid daily adjusted price")
    held, units, cash = "CASH", 0.0, starting_equity
    path, trades = [], []
    pending = None
    cost_since_mark = 0.0

    def trade(row, signal_day, target, field):
        nonlocal held, units, cash
        if target == held:
            return 0.0
        if mode == "next_open" and row["date"] <= signal_day:
            raise ValueError("Next-open fill must follow its signal date")
        before = cash if held == "CASH" else units * row["assets"][held][field]
        sides = 1 if held == "CASH" else 2
        fees = before * sides * cost_bps / 10000
        level = row["assets"][target][field]
        trades.append({"signal_date": signal_day, "fill_date": row["date"],
                       "mode": mode, "from_ticker": held, "to_ticker": target,
                       "sell_adjusted_level": None if held == "CASH" else row["assets"][held][field],
                       "buy_adjusted_level": level, "equity_before_trade": before,
                       "gross_traded_weight": sides, "cost_dollars": fees,
                       "equity_after_trade": before-fees})
        units, cash, held = (before-fees)/level, 0.0, target
        return fees

    for i, row in enumerate(daily):
        day = row["date"]
        if mode == "next_open" and pending:
            signal_day, target = pending
            cost_since_mark += trade(row, signal_day, target, "adjusted_open")
            pending = None
        equity = cash if held == "CASH" else units * row["assets"][held]["adjusted_close"]
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("Invalid account equity")
        path.append({"date": day, "equity": equity, "held_at_close": held,
                     "daily_return": equity/path[-1]["equity"]-1 if path else None,
                     "cost_dollars_since_previous_mark": cost_since_mark})
        cost_since_mark = 0.0
        if day in signals and i < len(daily)-1:
            target = signals[day]["ticker"]
            if mode == "next_open":
                pending = (day, target)
            else:
                cost_since_mark += trade(row, day, target, "adjusted_close")
    return {"path": path, "trades": trades}


def daily_drawdown(path):
    peak = path[0]["equity"]
    peak_date = path[0]["date"]
    worst, worst_peak, trough = 0.0, peak_date, peak_date
    for row in path:
        if row["equity"] > peak:
            peak, peak_date = row["equity"], row["date"]
        drawdown = row["equity"]/peak-1
        if drawdown < worst:
            worst, worst_peak, trough = drawdown, peak_date, row["date"]
    return {"max_drawdown_daily_close": worst, "drawdown_peak_date": worst_peak,
            "drawdown_trough_date": trough}


def summarize(simulation, monthly, daily):
    by_day = {r["date"]: r for r in simulation["path"]}
    marks = [by_day[r["month"]] for r in monthly[12:]]
    month_path = [{"month": r["date"], "equity": r["equity"],
                   "net_return": r["equity"]/marks[i-1]["equity"]-1 if i else None,
                   "turnover": 0.0, "cost_dollars": 0.0} for i, r in enumerate(marks)]
    cash_returns = [monthly[i]["tbill_index"]/monthly[i-1]["tbill_index"]-1
                    for i in range(13, len(monthly))]
    result = metrics(month_path, cash_returns)
    result.update(daily_drawdown(simulation["path"]))
    returns = [r["daily_return"] for r in simulation["path"][1:]]
    result["annualized_volatility_daily"] = statistics.stdev(returns)*math.sqrt(252) if len(returns)>1 else None
    result["cost_dollars"] = sum(t["cost_dollars"] for t in simulation["trades"])
    result["trade_events"] = len(simulation["trades"])
    result["holding_switches"] = sum(t["from_ticker"] != "CASH" for t in simulation["trades"])
    result["gross_traded_weight"] = sum(t["gross_traded_weight"] for t in simulation["trades"])
    result["worst_day"] = min(simulation["path"][1:], key=lambda r:r["daily_return"])["date"]
    result["worst_day_return"] = min(returns)
    return result, month_path


def run_study(monthly, daily, starting_equity=100_000.0, cost_bps=10.0):
    reference = backtest(monthly, starting_equity, cost_bps)
    signals = {r["month"]: {"ticker": MAPPING[r["next_month_allocation"]], "reason": r["reason"]}
               for r in reference}
    simulations, summaries, monthly_paths = {}, {}, {}
    for name, mode, chosen in (("gem_signal_close", "signal_close", signals),
                              ("gem_next_open", "next_open", signals),
                              ("spy_next_open", "next_open", {daily[0]["date"]:{"ticker":"SPY"}})):
        sim = simulate(daily, chosen, mode, starting_equity, cost_bps)
        simulations[name] = sim
        summaries[name], monthly_paths[name] = summarize(sim, monthly, daily)
    # Reconcile EVERY month, so the audit isolates execution timing instead
    # of accidentally introducing a second implementation of GEM rules.
    differences = [abs(r["equity"]-m["equity"])
                   for r,m in zip(reference, monthly_paths["gem_signal_close"])]
    max_difference = max(differences)
    if max_difference > max(.01, starting_equity*1e-9):
        raise ValueError(f"Daily close path does not reconcile with monthly reference: {max_difference}")
    trades_close = simulations["gem_signal_close"]["trades"]
    trades_open = simulations["gem_next_open"]["trades"]
    if len(trades_close) != len(trades_open):
        raise ValueError("Close/open versions produced different number of trade events")
    timing = []
    for a,b in zip(trades_close,trades_open):
        if (a["signal_date"],a["from_ticker"],a["to_ticker"]) != (b["signal_date"],b["from_ticker"],b["to_ticker"]):
            raise ValueError("Close/open versions disagree on signal sequence")
        previous_growth = (b["sell_adjusted_level"]/a["sell_adjusted_level"]
                           if a["from_ticker"] != "CASH" else 1.0)
        new_growth = b["buy_adjusted_level"]/a["buy_adjusted_level"]
        timing.append({"signal_date":a["signal_date"], "fill_date":b["fill_date"],
                       "from_ticker":a["from_ticker"], "to_ticker":a["to_ticker"],
                       "old_holding_overnight_return":previous_growth-1,
                       "new_holding_overnight_return":new_growth-1,
                       "next_open_relative_wealth_factor":previous_growth/new_growth})
    ratio = summaries["gem_next_open"]["ending_equity"]/summaries["gem_signal_close"]["ending_equity"]
    if not math.isclose(math.prod(t["next_open_relative_wealth_factor"] for t in timing), ratio, rel_tol=1e-9):
        raise ValueError("Overnight-gap attribution does not reconcile")
    return {"baseline_date":daily[0]["date"], "first_fill_date":trades_open[0]["fill_date"],
            "end_date":daily[-1]["date"], "daily_return_sessions":len(daily)-1,
            "return_months":len(reference)-1, "starting_equity":starting_equity,
            "cost_bps_per_side":cost_bps, "summary":summaries,
            "next_open_minus_close_dollars":summaries["gem_next_open"]["ending_equity"]-summaries["gem_signal_close"]["ending_equity"],
            "next_open_relative_ending_wealth":ratio-1,
            "max_monthly_reference_reconciliation_error":max_difference,
            "latest_unexecuted_signal":{"date":reference[-1]["month"], **signals[reference[-1]["month"]]},
            "simulations":simulations, "monthly_paths":monthly_paths, "timing":timing,
            "signals":[{"date":d,**s} for d,s in signals.items()]}


def render_report(result, as_of):
    labels = {"gem_signal_close":"GEM signal-close reference", "gem_next_open":"GEM next-session open",
              "spy_next_open":"SPY buy/hold from same next open"}
    lines = ["# GEM execution timing audit", "",
             f"As of {as_of}; baseline {result['baseline_date']}; first opening fill {result['first_fill_date']}; end {result['end_date']}.",
             f"{result['daily_return_sessions']} daily return sessions; {result['return_months']} months. "
             f"Starting capital ${result['starting_equity']:,.2f}; {result['cost_bps_per_side']:g} bps per buy/sell dollar.", "",
             "| Model | Net ending | CAGR | Worst daily-close drawdown | Worst month-end drawdown | Daily annualized volatility |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key,s in result["summary"].items():
        vol = "n/a" if s["annualized_volatility_daily"] is None else f"{s['annualized_volatility_daily']:.2%}"
        lines.append(f"| {labels[key]} | ${s['ending_equity']:,.2f} | {s['cagr']:.2%} | "
                     f"{s['max_drawdown_daily_close']:.2%} | {s['max_drawdown_monthly']:.2%} | {vol} |")
    actual = result["summary"]["gem_next_open"]
    lines += ["", f"Next-open ending value minus close reference: **${result['next_open_minus_close_dollars']:+,.2f}** "
              f"({result['next_open_relative_ending_wealth']:+.2%} relative ending wealth).",
              f"GEM: {actual['holding_switches']} switches plus initial entry. "
              f"Worst daily-close drawdown peak {actual['drawdown_peak_date']}, trough {actual['drawdown_trough_date']}.",
              f"Daily signal-close path matched every monthly reference mark; maximum error "
              f"${result['max_monthly_reference_reconciliation_error']:.8f}.", "",
              "## What this tests", "",
              "Signals use completed month-end data and the locked GEM rules. Next-open trades occur in the first "
              "following exchange session. On a switch, the old position bears its overnight move before sale; "
              "the new position begins earning returns at its opening price. An unchanged monthly signal does not trade.",
              "Prices remain total-return proxies: adjusted open = provider open × adjusted close / provider close. "
              "Units are fractional return-index units, not ETF shares. Dividend/split adjustments are represented through "
              "the provider's price factors; actual dividend payment dates, cash entitlements and whole-share rounding are not modeled. "
              "This adjustment convention can affect ex-dividend overnight attribution.",
              "All paths use one saved price snapshot and the same cost convention. Initial cash earns zero until entry; "
              "GEM then stays fully allocated to one ETF. Costs approximate 10 bps by default per traded side, with a full switch "
              "charged on 200% of pretrade equity. Actual auction liquidity, spreads, partial fills, taxes and settlement are not simulated.",
              "Daily drawdown uses closing valuations; intraday losses may be deeper. The signal-close reference records "
              "month-end equity before that close's new trade cost, matching the monthly study. The final signal is reported but not filled.",
              "Fresh snapshots can differ slightly from earlier downloads as the provider revises adjusted history. "
              "This run's monthly reference is recomputed from this same snapshot, not mixed with an older download.",
              "This is a historical timing audit. It does not open a paper account or submit any orders.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--as-of", type=date.fromisoformat, default=datetime.now(NY).date())
    parser.add_argument("--snapshot", help="offline daily OHLC snapshot JSON")
    parser.add_argument("--output-dir", default="data/research/gem_execution")
    parser.add_argument("--starting-equity", type=float, default=100_000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.refresh and args.snapshot:
        parser.error("choose --refresh or --snapshot")
    if args.as_of > datetime.now(NY).date():
        parser.error("--as-of cannot be in the future")
    folder = Path(args.output_dir)
    source = Path(args.snapshot) if args.snapshot else folder/"daily_ohlc_snapshot.json"
    try:
        if args.refresh:
            print("Fetching daily opens and closes for SPY, VEU, AGG, BIL...", flush=True)
            snapshot = fetch_snapshot(args.as_of)
        else:
            if not source.exists():
                raise ValueError(f"{source} missing; use --refresh first")
            snapshot = json.loads(source.read_text(encoding="utf-8-sig"))
        monthly, daily = prepare(snapshot, args.as_of)
        result = run_study(monthly, daily, args.starting_equity, args.cost_bps)
        folder.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(snapshot, indent=2, allow_nan=False)+"\n"
        (folder/"daily_ohlc_snapshot.json").write_text(raw, encoding="utf-8")
        write_csv(folder/"monthly_input.csv", monthly)
        write_csv(folder/"signals.csv", result.pop("signals"))
        write_csv(folder/"switch_timing.csv", result.pop("timing"))
        for name, sim in result.pop("simulations").items():
            write_csv(folder/f"{name}_daily.csv", sim["path"])
            write_csv(folder/f"{name}_trades.csv", sim["trades"])
        for name, path in result.pop("monthly_paths").items():
            write_csv(folder/f"{name}_monthly.csv", path)
        result["data"] = {"as_of":args.as_of.isoformat(), "snapshot_sha256":hashlib.sha256(raw.encode()).hexdigest(),
                          "source":snapshot.get("source", "unspecified offline input")}
        (folder/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        report = render_report(result, args.as_of)
        (folder/"report.md").write_text(report, encoding="utf-8")
    except (ValueError, KeyError, TypeError, IndexError, OSError, ArithmeticError) as error:
        parser.exit(2, f"GEM execution audit unavailable: {error}\n")
    print(report)
    print(f"Reports: {folder.resolve()}")


if __name__ == "__main__":
    main()
