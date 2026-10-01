import contextlib
import copy
import csv
import io
import math
import tempfile
import unittest
from pathlib import Path

from scanner.published_gem import ASSETS, backtest, decision, portfolio_path, validate_rows, write_csv
from scanner.strategy_cli import main as strategy_main


def sample(n=15):
    rows = []
    for i in range(n):
        y, m = divmod(2020 * 12 + i, 12)
        rows.append({"month": f"{y}-{m+1:02d}-28", **{a: 100.0 for a in ASSETS}})
    return rows


class GemTests(unittest.TestCase):
    def test_us_absolute_filter_precedes_foreign_strength(self):
        rows = sample()
        rows[12].update(us_stocks=99, foreign_stocks=150, tbill_index=101)
        self.assertEqual(decision(rows, 12)[0], "aggregate_bonds")

    def test_positive_us_return_still_must_beat_bills(self):
        rows = sample()
        rows[12].update(us_stocks=103, foreign_stocks=150, tbill_index=105)
        self.assertEqual(decision(rows, 12)[0], "aggregate_bonds")

    def test_relative_momentum_selects_foreign_only_after_us_passes(self):
        rows = sample()
        rows[12].update(us_stocks=110, foreign_stocks=120, tbill_index=101)
        self.assertEqual(decision(rows, 12)[0], "foreign_stocks")
        rows[12]["us_stocks"] = 130
        self.assertEqual(decision(rows, 12)[0], "us_stocks")

    def test_ties_are_explicit(self):
        rows = sample()
        self.assertEqual(decision(rows, 12)[0], "aggregate_bonds")
        rows[12].update(us_stocks=120, foreign_stocks=120)
        self.assertEqual(decision(rows, 12)[0], "us_stocks")

    def test_exactly_twelve_intervals_no_skipped_month(self):
        rows = sample()
        rows[0]["us_stocks"] = 50
        rows[11]["us_stocks"] = 500
        rows[12]["us_stocks"] = 100
        selected, momentum, _ = decision(rows, 12)
        self.assertEqual(momentum["us_stocks"], 1.0)
        self.assertEqual(selected, "us_stocks")

    def test_signal_t_does_not_earn_t_return(self):
        rows = sample()
        rows[12]["us_stocks"] = 120
        rows[13]["us_stocks"] = 60
        rows[14].update(us_stocks=120, aggregate_bonds=110)
        result = backtest(rows)
        self.assertEqual(result[0]["equity"], 100000)
        self.assertEqual(result[1]["held_this_month"], "us_stocks")
        self.assertEqual(result[1]["next_month_allocation"], "aggregate_bonds")
        self.assertAlmostEqual(result[1]["equity"], 50000)
        self.assertAlmostEqual(result[2]["equity"], 55000)

    def test_future_changes_do_not_rewrite_past(self):
        rows = sample(20)
        before = backtest(rows)
        rows[-1]["us_stocks"] *= 10
        self.assertEqual(before[:-1], backtest(rows)[:-1])

    def test_bond_fallback_can_lose_money(self):
        rows = sample(14)
        rows[-1]["aggregate_bonds"] = 90
        self.assertAlmostEqual(backtest(rows)[-1]["equity"], 90000)

    def test_cost_entry_and_full_switch_count_both_sides(self):
        rows = sample()
        path = portfolio_path(rows[:3], [{"us_stocks": 1}, {"aggregate_bonds": 1},
                                         {"foreign_stocks": 1}], cost_bps=10)
        self.assertEqual([r["turnover"] for r in path], [0, 1, 2])
        self.assertAlmostEqual(path[-1]["equity"], 100000 * .999 * .998)
        # The last signal is not executed until another return month exists.
        self.assertAlmostEqual(sum(r["cost_dollars"] for r in path), 100 + 199.8)

    def test_unchanged_one_asset_has_no_recurring_cost(self):
        rows = sample()
        path = portfolio_path(rows[:3], [{"us_stocks": 1}] * 3, cost_bps=10)
        self.assertEqual(path[-1]["turnover"], 0)

    def test_multi_asset_rebalance_uses_drifted_weights(self):
        rows = sample(3)
        rows[1]["us_stocks"] = rows[2]["us_stocks"] = 200
        path = portfolio_path(rows, [{"us_stocks": .5, "aggregate_bonds": .5}] * 3)
        self.assertAlmostEqual(path[-1]["turnover"], 1/3)

    def test_rejects_bad_rows_and_parameters(self):
        rows = sample()
        for bad in (rows[:13], rows[:4] + rows[5:], rows[::-1], rows[:4] + rows[3:]):
            with self.subTest(bad=bad[0]["month"]):
                with self.assertRaises(ValueError):
                    backtest(bad)
        for value in (0, -1, "nan", "inf"):
            bad = copy.deepcopy(rows)
            bad[5]["tbill_index"] = value
            with self.assertRaises(ValueError):
                backtest(bad)
        for kwargs in ({"starting_equity": 0}, {"cost_bps": -1}, {"cost_bps": math.nan}):
            with self.assertRaises(ValueError):
                backtest(rows, **kwargs)

    def test_selector_dispatch_and_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder)/"input.csv", Path(folder)/"out.csv"
            write_csv(source, sample())
            with contextlib.redirect_stdout(io.StringIO()):
                strategy_main(["run", "gem", str(source), str(out)])
            with out.open() as handle:
                result = list(csv.DictReader(handle))
            self.assertEqual(result[-1]["next_month_allocation"], "aggregate_bonds")
            self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
