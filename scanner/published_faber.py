"""Reference calculation for Faber's published single-asset monthly timing rule."""

import csv
import math
import sys
from datetime import date
from pathlib import Path


def backtest(rows, starting_equity=100_000.0):
    if not math.isfinite(starting_equity) or starting_equity <= 0:
        raise ValueError("starting_equity must be positive and finite")
    observations = []
    previous_month = None
    for row in rows:
        month = date.fromisoformat(row["month"])
        month_key = month.year * 12 + month.month
        if previous_month is not None and month_key != previous_month + 1:
            raise ValueError("months must be consecutive and unique")
        previous_month = month_key
        level = float(row["total_return_index"])
        cash_return = float(row["tbill_return"])
        if not math.isfinite(level) or level <= 0 or not math.isfinite(cash_return) or cash_return <= -1:
            raise ValueError("invalid index level or T-bill return")
        observations.append((month.isoformat(), level, cash_return))
    if len(observations) < 11:
        raise ValueError("at least 11 consecutive months are required")

    equity = starting_equity
    allocation = None
    result = []
    for i in range(9, len(observations)):
        month, level, cash_return = observations[i]
        asset_return = None if i == 9 else level / observations[i - 1][1] - 1
        held = allocation
        if held is not None:
            selected_return = asset_return if held == "ASSET" else cash_return
            equity *= 1 + selected_return
        sma = sum(item[1] for item in observations[i - 9:i + 1]) / 10
        if level > sma:
            allocation = "ASSET"
        elif level < sma:
            allocation = "TBILL"
        elif allocation is None:
            allocation = "TBILL"
        # If exactly equal, otherwise retain the previous allocation.
        result.append({"month": month, "index": level, "sma10": sma,
                       "held_this_month": held or "WARMUP", "asset_return": asset_return,
                       "tbill_return": cash_return, "equity": equity,
                       "next_month_allocation": allocation or "UNDECIDED"})
    return result


def main(source, destination):
    with Path(source).open(newline="", encoding="utf-8") as handle:
        rows = backtest(csv.DictReader(handle))
    with Path(destination).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python faber_10month.py monthly.csv output.csv")
    main(sys.argv[1], sys.argv[2])
