"""Faber's five-asset GTAA calculation on monthly total-return index levels."""

import csv
import math
import sys
from datetime import date
from pathlib import Path


ASSETS = ("us_stocks", "foreign_stocks", "us_10y_bonds", "real_estate", "commodities")


def backtest(rows, starting_equity=100_000.0):
    """Apply each asset's 10-month signal to the next month's 20% sleeve."""
    if not math.isfinite(starting_equity) or starting_equity <= 0:
        raise ValueError("starting_equity must be positive and finite")
    observations = []
    previous_month = None
    for row in rows:
        month = date.fromisoformat(row["month"])
        key = month.year * 12 + month.month
        if previous_month is not None and key != previous_month + 1:
            raise ValueError("months must be consecutive and unique")
        previous_month = key
        levels = {asset: float(row[asset]) for asset in ASSETS}
        cash = float(row["tbill_return"])
        if any(not math.isfinite(v) or v <= 0 for v in levels.values()):
            raise ValueError("asset index levels must be positive and finite")
        if not math.isfinite(cash) or cash <= -1:
            raise ValueError("invalid T-bill return")
        observations.append((month.isoformat(), levels, cash))
    if len(observations) < 11:
        raise ValueError("at least 11 consecutive months are required")

    equity = starting_equity
    benchmark = starting_equity
    allocation = {asset: None for asset in ASSETS}
    result = []
    for i in range(9, len(observations)):
        month, levels, cash = observations[i]
        held = allocation.copy()
        asset_returns = ({asset: levels[asset] / observations[i - 1][1][asset] - 1
                          for asset in ASSETS} if i > 9 else None)
        portfolio_return = benchmark_return = None
        if asset_returns is not None:
            # Rebalance all five sleeves to 20% at each month boundary.
            portfolio_return = sum(asset_returns[a] if held[a] == "ASSET" else cash
                                   for a in ASSETS) / len(ASSETS)
            benchmark_return = sum(asset_returns.values()) / len(ASSETS)
            equity *= 1 + portfolio_return
            benchmark *= 1 + benchmark_return
        averages = {asset: sum(observations[j][1][asset] for j in range(i - 9, i + 1)) / 10
                    for asset in ASSETS}
        for asset in ASSETS:
            if levels[asset] > averages[asset]:
                allocation[asset] = "ASSET"
            elif levels[asset] < averages[asset] or allocation[asset] is None:
                allocation[asset] = "TBILL"
            # Equality retains the prior position; initial equality takes cash.
        record = {"month": month, "portfolio_return": portfolio_return,
                  "portfolio_equity": equity, "equal_weight_return": benchmark_return,
                  "equal_weight_equity": benchmark, "tbill_return": cash}
        for asset in ASSETS:
            record[f"{asset}_index"] = levels[asset]
            record[f"{asset}_sma10"] = averages[asset]
            record[f"{asset}_held"] = held[asset] or "WARMUP"
            record[f"{asset}_next"] = allocation[asset]
        result.append(record)
    return result


def main(source, destination):
    with Path(source).open(newline="", encoding="utf-8-sig") as handle:
        result = backtest(csv.DictReader(handle))
    with Path(destination).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m scanner.published_faber_gtaa monthly.csv output.csv")
    main(sys.argv[1], sys.argv[2])
