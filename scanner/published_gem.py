"""GEM monthly reference: absolute momentum first, then relative momentum.

Inputs are month-end total-return levels, not raw closes or annualized yields.
The signal at t earns returns only in t+1. See docs/gem_dual_momentum.md.
"""

import csv
import math
from datetime import date
from pathlib import Path

ASSETS = ("us_stocks", "foreign_stocks", "aggregate_bonds", "tbill_index")
LOOKBACK = 12


def validate_rows(rows, assets=ASSETS, minimum=14):
    result = []
    previous = None
    for row in rows:
        day = date.fromisoformat(row["month"])
        key = day.year * 12 + day.month
        if previous is not None and key != previous + 1:
            raise ValueError("months must be consecutive, ordered and unique")
        previous = key
        values = {a: float(row[a]) for a in assets}
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("total-return levels must be positive and finite")
        result.append({"month": day.isoformat(), **values})
    if len(result) < minimum:
        raise ValueError(f"at least {minimum} consecutive monthly levels are required")
    return result


def decision(rows, i):
    if i < LOOKBACK:
        raise ValueError("12-month momentum needs 13 month-end levels")
    momentum = {a: rows[i][a] / rows[i - LOOKBACK][a] - 1 for a in ASSETS}
    # The US absolute test gates BOTH stock markets, even if foreign is stronger.
    if momentum["us_stocks"] <= momentum["tbill_index"]:
        selected, reason = "aggregate_bonds", "US did not beat Treasury bills"
    elif momentum["us_stocks"] >= momentum["foreign_stocks"]:
        selected, reason = "us_stocks", "US passed absolute test and leads/ties foreign"
    else:
        selected, reason = "foreign_stocks", "US passed absolute test; foreign leads"
    return selected, momentum, reason


def portfolio_path(rows, targets, starting_equity=100_000.0, cost_bps=0.0):
    """Reference monthly path from a baseline row plus subsequent return months.

    Each targets[i] is formed at rows[i]'s close and held during rows[i+1].
    Cost uses gross traded ETF weight (buy + sell), measured against pretrade
    equity, charged before the next month's return. This is a disclosed cost
    approximation; monthly close execution is not an executable fill model.
    """
    if not math.isfinite(starting_equity) or starting_equity <= 0:
        raise ValueError("starting_equity must be positive and finite")
    if not math.isfinite(cost_bps) or not 0 <= cost_bps < 5000:
        raise ValueError("cost_bps must be finite and between 0 and 5000 (exclusive)")
    if len(rows) < 2 or len(rows) != len(targets):
        raise ValueError("rows and targets must align and include a return month")
    for weights in targets:
        if (not weights or any(not math.isfinite(w) or w < 0 for w in weights.values())
                or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-10)):
            raise ValueError("target weights must be nonnegative and sum to one")
    equity = starting_equity
    drifted = {}  # Initial funding is uninvested cash; first ETF purchase costs 1x.
    path = [{"month": rows[0]["month"], "gross_return": None,
             "net_return": None, "equity": equity, "turnover": 0.0, "cost_dollars": 0.0}]
    for i in range(1, len(rows)):
        weights = targets[i - 1]
        traded = sum(abs(weights.get(a, 0) - drifted.get(a, 0))
                     for a in weights.keys() | drifted.keys())
        fraction = traded * cost_bps / 10000
        growth = {a: rows[i][a] / rows[i - 1][a] for a in weights}
        gross_growth = sum(weights[a] * growth[a] for a in weights)
        net_return = (1 - fraction) * gross_growth - 1
        cost = equity * fraction
        equity *= 1 + net_return
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("invalid portfolio equity; inspect input levels")
        drifted = {a: weights[a] * growth[a] / gross_growth for a in weights}
        path.append({"month": rows[i]["month"], "gross_return": gross_growth - 1,
                     "net_return": net_return, "equity": equity,
                     "turnover": traded, "cost_dollars": cost})
    return path


def backtest(rows, starting_equity=100_000.0, cost_bps=0.0):
    observations = validate_rows(rows)
    decisions = [decision(observations, i) for i in range(LOOKBACK, len(observations))]
    path = portfolio_path(observations[LOOKBACK:], [{d[0]: 1.0} for d in decisions],
                          starting_equity, cost_bps)
    result = []
    for i, (record, (selected, momentum, reason)) in enumerate(zip(path, decisions)):
        result.append({**record, "held_this_month": decisions[i-1][0] if i else "WARMUP",
                       "next_month_allocation": selected, "reason": reason,
                       **{f"{a}_return_12m": momentum[a] for a in ASSETS}})
    return result


def write_csv(path, records):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(source, destination):
    with Path(source).open(newline="", encoding="utf-8-sig") as handle:
        result = backtest(csv.DictReader(handle))
    write_csv(destination, result)
