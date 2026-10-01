"""Reproducible GEM ETF study and common-window strategy comparison."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scanner.gem_data import TICKERS, fetch_panel, validate_panel
from scanner.published_faber_gtaa import ASSETS as FABER_ASSETS, backtest as faber_backtest
from scanner.published_gem import backtest, portfolio_path, write_csv

GEM_MAP = {"us_stocks": "SPY", "foreign_stocks": "VEU",
           "aggregate_bonds": "AGG", "tbill_index": "BIL"}
FABER_MAP = dict(zip(FABER_ASSETS, ("SPY", "EFA", "IEF", "VNQ", "GSG")))
LABELS = {"gem": "GEM ETF proxy", "faber": "Faber GTAA5 (BIL cash proxy)",
          "faber_equal_weight": "Five-ETF equal weight, monthly rebalance",
          "gem_benchmark": "GEM 45/28/27 benchmark, monthly rebalance",
          "spy_buy_hold": "SPY buy and hold"}


def metrics(path, cash_returns):
    returns = [r["net_return"] for r in path[1:]]
    if len(returns) != len(cash_returns) or not returns:
        raise ValueError("metrics require aligned portfolio and cash returns")
    start, end = path[0]["equity"], path[-1]["equity"]
    peak, max_dd = start, 0.0
    for r in path:
        peak = max(peak, r["equity"])
        max_dd = min(max_dd, r["equity"] / peak - 1)
    volatility = statistics.stdev(returns) * math.sqrt(12) if len(returns) > 1 else None
    excess = [r - c for r, c in zip(returns, cash_returns)]
    excess_sd = statistics.stdev(excess) if len(excess) > 1 else 0.0
    years = {}
    for row in path[1:]:
        year = row["month"][:4]
        years.setdefault(year, []).append(row)
    full_years = {y: math.prod(1 + r["net_return"] for r in rows) - 1
                  for y, rows in years.items() if len(rows) == 12}
    worst = min(full_years, key=full_years.get) if full_years else None
    return {"ending_equity": end, "total_return": end / start - 1,
            "cagr": (end / start) ** (12 / len(returns)) - 1,
            "max_drawdown_monthly": max_dd, "annualized_volatility": volatility,
            "sharpe_vs_bil": statistics.mean(excess) / excess_sd * math.sqrt(12) if excess_sd > 1e-15 else None,
            "worst_full_year": worst,
            "worst_full_year_return": full_years[worst] if worst else None,
            "gross_traded_weight": sum(r["turnover"] for r in path),
            "cost_dollars": sum(r["cost_dollars"] for r in path)}


def compare(panel, starting_equity=100_000.0, cost_bps=10.0):
    gem_input = [{"month": r["month"], **{a: r[t] for a, t in GEM_MAP.items()}} for r in panel]
    gem = backtest(gem_input, starting_equity, cost_bps)
    faber_input = [{"month": r["month"], **{a: r[t] for a, t in FABER_MAP.items()},
                    "tbill_return": r["BIL"] / panel[i-1]["BIL"] - 1 if i else 0.0}
                   for i, r in enumerate(panel)]
    faber = {r["month"]: r for r in faber_backtest(faber_input, starting_equity)}
    common = panel[12:]
    targets = {k: [] for k in LABELS}
    for row, g in zip(common, gem):
        targets["gem"].append({GEM_MAP[g["next_month_allocation"]]: 1.0})
        weights = {}
        f = faber[row["month"]]
        for asset, ticker in FABER_MAP.items():
            chosen = ticker if f[f"{asset}_next"] == "ASSET" else "BIL"
            weights[chosen] = weights.get(chosen, 0.0) + 0.2
        targets["faber"].append(weights)
        targets["faber_equal_weight"].append({t: 0.2 for t in FABER_MAP.values()})
        targets["gem_benchmark"].append({"SPY": 0.45, "VEU": 0.28, "AGG": 0.27})
        targets["spy_buy_hold"].append({"SPY": 1.0})
    cash = [common[i]["BIL"] / common[i-1]["BIL"] - 1 for i in range(1, len(common))]
    paths, summary = {}, {}
    for name, weights in targets.items():
        path = portfolio_path(common, weights, starting_equity, cost_bps)
        gross = portfolio_path(common, weights, starting_equity, 0.0)
        for i, record in enumerate(path):
            record["next_month_weights"] = json.dumps(weights[i], sort_keys=True)
        paths[name] = path
        summary[name] = {"label": LABELS[name], **metrics(path, cash),
                         "gross_ending_equity": gross[-1]["equity"],
                         "gross_cagr": (gross[-1]["equity"] / starting_equity) ** (12 / len(cash)) - 1}
    return {"baseline_date": common[0]["month"], "first_return_month": common[1]["month"],
            "end_date": common[-1]["month"], "invested_months": len(cash),
            "starting_equity": starting_equity, "cost_bps_per_side": cost_bps,
            "summary": summary, "paths": paths, "gem_input": gem_input, "gem": gem,
            "latest_signal": {"date": gem[-1]["month"],
                              "next_ticker": GEM_MAP[gem[-1]["next_month_allocation"]],
                              "reason": gem[-1]["reason"],
                              "momentum_12m": {t: gem[-1][f"{a}_return_12m"] for a, t in GEM_MAP.items()}}}


def render_report(result, as_of):
    def pct(value):
        return "n/a" if value is None else f"{value:.2%}"
    lines = ["# GEM versus Faber — ETF proxy research", "",
             f"As of: {as_of}. Baseline: {result['baseline_date']}; return months: "
             f"{result['first_return_month']} through {result['end_date']} ({result['invested_months']} months).",
             f"Starting capital: ${result['starting_equity']:,.2f}. Assumed trading cost: "
             f"{result['cost_bps_per_side']:g} basis points per buy/sell dollar, including initial entry.", "",
             "| Strategy | Gross ending | Net ending | Net CAGR | Monthly max drawdown | Annual volatility | Sharpe vs BIL | Worst full year |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for s in result["summary"].values():
        sharpe = "n/a" if s["sharpe_vs_bil"] is None else f"{s['sharpe_vs_bil']:.2f}"
        worst = "n/a" if s["worst_full_year"] is None else f"{s['worst_full_year']}: {pct(s['worst_full_year_return'])}"
        lines.append(f"| {s['label']} | ${s['gross_ending_equity']:,.2f} | ${s['ending_equity']:,.2f} | "
                     f"{pct(s['cagr'])} | {pct(s['max_drawdown_monthly'])} | {pct(s['annualized_volatility'])} | {sharpe} | {worst} |")
    signal = result["latest_signal"]
    lines += ["", f"Latest completed month signal ({signal['date']}): **{signal['next_ticker']}** for the following month.",
              signal["reason"] + ".", "", "Trailing 12-month total-return proxies:", ""]
    lines += [f"- {t}: {v:.2%}" for t, v in signal["momentum_12m"].items()]
    lines += ["", "## Interpretation and limits", "",
              "These are hypothetical monthly reference results, not actual or paper-account returns. "
              "A signal formed at a month-end close earns the following close-to-close monthly return. "
              "This idealizes execution at the signal close; it does not model next-session open fills or overnight gaps.",
              "Drawdowns use month-end equity only and can understate losses experienced inside a month. "
              "Volatility uses sample monthly standard deviation; Sharpe uses monthly excess returns over BIL, annualized by sqrt(12). "
              "Worst year excludes partial calendar years.",
              "GEM uses SPY, VEU, AGG and BIL adjusted closes as total-return proxies. "
              "VEU tracks a different ex-US index from original GEM; BIL has fund expenses/tracking differences from Treasury-bill indexes. "
              "This is not an exact reproduction of Antonacci's index results.",
              "Faber reuses the existing five-asset signals with SPY/EFA/IEF/VNQ/GSG. "
              "Its defensive sleeves use BIL here, replacing the previous study's estimated T-bill returns. "
              "All comparisons reset capital at the same baseline after GEM's 12-month warm-up. "
              "Consequently these results need not match the earlier Faber report.",
              "The 45/28/27 and five-ETF benchmarks rebalance monthly; SPY is buy and hold. "
              "Costs are a user-selected estimate on gross ETF turnover (a full switch trades 200%). "
              "Weights are measured before fees and the remaining capital is allocated proportionally. "
              "No ending liquidation cost, taxes, management fees beyond ETF returns, or market impact is modeled.",
              "Signals use all 12 trailing monthly intervals, with no skipped month. "
              "US equal to bills selects bonds; equal US/foreign momentum selects US. These tie conventions are explicit implementation choices.",
              "Data includes only completed calendar months, excludes the as-of day's daily bar, and is saved with a SHA-256 digest. "
              "The latest target is informational: no account, paper fill, or broker order is created.",
              "See docs/gem_dual_momentum.md for locked rules, proxy choices and original sources.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="fetch a new daily ETF snapshot")
    parser.add_argument("--as-of", type=date.fromisoformat,
                        default=datetime.now(ZoneInfo("America/New_York")).date())
    parser.add_argument("--input-csv", help="saved monthly panel with month and eight ticker columns")
    parser.add_argument("--output-dir", default="data/research/gem_etf_proxy")
    parser.add_argument("--starting-equity", type=float, default=100_000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.refresh and args.input_csv:
        parser.error("choose --refresh or --input-csv, not both")
    if args.as_of > datetime.now(ZoneInfo("America/New_York")).date():
        parser.error("--as-of cannot be a future date")
    folder = Path(args.output_dir)
    source = Path(args.input_csv) if args.input_csv else folder / "monthly_panel.csv"
    histories = None
    try:
        if args.refresh:
            print("Fetching SPY, VEU, AGG, BIL and Faber comparison ETFs...", flush=True)
            panel, histories = fetch_panel(args.as_of)
        else:
            if not source.exists():
                raise ValueError(f"{source} is missing; first run with --refresh")
            with source.open(newline="", encoding="utf-8-sig") as handle:
                panel = validate_panel(csv.DictReader(handle), args.as_of)
        result = compare(panel, args.starting_equity, args.cost_bps)
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(folder / "monthly_panel.csv", panel)
        write_csv(folder / "gem_input.csv", result.pop("gem_input"))
        write_csv(folder / "gem_results.csv", result.pop("gem"))
        for name, path in result.pop("paths").items():
            write_csv(folder / f"{name}_path.csv", path)
        source_bytes = (folder / "monthly_panel.csv").read_bytes()
        result["data"] = {"as_of": args.as_of.isoformat(), "tickers": list(TICKERS),
                          "monthly_panel_sha256": hashlib.sha256(source_bytes).hexdigest(),
                          "source": "Yahoo adjusted closes" if histories is not None else "Offline input CSV (provenance not independently verified)"}
        if histories is not None:
            (folder / "daily_snapshot.json").write_text(json.dumps(histories, indent=2) + "\n", encoding="utf-8")
        (folder / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report = render_report(result, args.as_of)
        (folder / "report.md").write_text(report, encoding="utf-8")
    except (ValueError, KeyError, TypeError, IndexError, OSError, ArithmeticError) as error:
        parser.exit(2, f"GEM research unavailable: {error}\n")
    print(report)
    print(f"Reports: {folder.resolve()}")


if __name__ == "__main__":
    main()
