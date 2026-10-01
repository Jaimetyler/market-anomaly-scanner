import calendar
import contextlib
import copy
import csv
import io
import json
import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from scanner.gem_data import TICKERS, month_end_session, monthly_panel, validate_panel
from scanner.gem_research import compare, main, metrics
from scanner.published_gem import write_csv


def panel_sample(n=30):
    rows = []
    for i in range(n):
        y, m = divmod(2020 * 12 + i, 12)
        rows.append({"month": month_end_session(y, m+1).isoformat(),
                     **{t: 100 * (1 + (j+1)/1000) ** i for j, t in enumerate(TICKERS)}})
    return rows


def daily_sample():
    return {t: [{"date": r["month"], "adjusted_close": r[t], "close": r[t]}
                for r in panel_sample()] for t in TICKERS}


class GemResearchTests(unittest.TestCase):
    def test_common_window_and_equal_starting_capital(self):
        panel = panel_sample()
        result = compare(panel, cost_bps=0)
        self.assertEqual(result["baseline_date"], panel[12]["month"])
        self.assertEqual(result["invested_months"], 17)
        for path in result["paths"].values():
            self.assertEqual(path[0]["equity"], 100000)
            self.assertEqual(path[0]["month"], panel[12]["month"])
            self.assertEqual(path[-1]["month"], panel[-1]["month"])
        self.assertAlmostEqual(result["summary"]["spy_buy_hold"]["ending_equity"],
                               100000 * panel[-1]["SPY"] / panel[12]["SPY"])

    def test_gem_research_and_standalone_paths_agree(self):
        result = compare(panel_sample())
        self.assertEqual([r["equity"] for r in result["gem"]],
                         [r["equity"] for r in result["paths"]["gem"]])

    def test_comparison_is_causal(self):
        panel = panel_sample()
        before = compare(panel)
        panel[-1]["SPY"] *= 3
        after = compare(panel)
        for name in before["paths"]:
            self.assertEqual(before["paths"][name][:-1], after["paths"][name][:-1])

    def test_faber_sleeves_use_bil_cash(self):
        panel = panel_sample()
        for row in panel:
            row["GSG"] = 10000 / row["GSG"]
        weights = json.loads(compare(panel)["paths"]["faber"][0]["next_month_weights"])
        self.assertAlmostEqual(weights["BIL"], .2)
        self.assertNotIn("GSG", weights)
        self.assertAlmostEqual(sum(weights.values()), 1)

    def test_costs_reduce_all_portfolios_and_are_reported(self):
        result = compare(panel_sample(), cost_bps=10)
        for s in result["summary"].values():
            self.assertLess(s["ending_equity"], s["gross_ending_equity"])
            self.assertGreaterEqual(s["cost_dollars"], 100)

    def test_metrics_include_initial_drawdown_and_exclude_partial_year(self):
        path = [{"month": "2020-12-31", "equity": 100, "net_return": None, "turnover": 0, "cost_dollars": 0},
                {"month": "2021-01-29", "equity": 50, "net_return": -.5, "turnover": 1, "cost_dollars": 0},
                {"month": "2021-02-26", "equity": 60, "net_return": .2, "turnover": 0, "cost_dollars": 0}]
        m = metrics(path, [0, 0])
        self.assertAlmostEqual(m["max_drawdown_monthly"], -.5)
        self.assertAlmostEqual(m["cagr"], .6 ** 6 - 1)
        self.assertIsNone(m["worst_full_year"])
        self.assertAlmostEqual(m["annualized_volatility"], math.sqrt(.245) * math.sqrt(12))

    def test_worst_full_year_calculation(self):
        path = compare(panel_sample(), cost_bps=0)["paths"]["spy_buy_hold"]
        # Baseline Jan 2021: first year has only 11 returns, 2022 has 6.
        self.assertIsNone(metrics(path, [0] * (len(path)-1))["worst_full_year"])
        path = compare(panel_sample(48), cost_bps=0)["paths"]["spy_buy_hold"]
        m = metrics(path, [0] * (len(path)-1))
        self.assertIn(m["worst_full_year"], ("2022", "2023"))
        self.assertAlmostEqual(m["worst_full_year_return"], 1.001**12 - 1)

    def test_trading_month_end_holidays(self):
        self.assertEqual(month_end_session(2024, 3), date(2024, 3, 28))
        self.assertEqual(month_end_session(2021, 5), date(2021, 5, 28))
        self.assertEqual(month_end_session(2021, 12), date(2021, 12, 31))
        self.assertEqual(month_end_session(2026, 9), date(2026, 9, 30))

    def test_daily_panel_alignment_and_inception_month_exclusion(self):
        panel = monthly_panel(daily_sample(), date(2022, 7, 1))
        self.assertEqual(len(panel), 29)
        self.assertEqual(panel[0]["month"], "2020-02-28")
        self.assertEqual(panel[-1]["month"], "2022-06-30")

    def test_rejects_provider_stale_one_day_before_month_end(self):
        histories = daily_sample()
        for records in histories.values():
            records[-1]["date"] = "2022-06-29"
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            monthly_panel(histories, date(2022, 7, 1))

    def test_rejects_missing_duplicate_and_disagreeing_daily_data(self):
        histories = daily_sample()
        histories["VEU"].pop(10)
        with self.assertRaisesRegex(ValueError, "Missing"):
            monthly_panel(histories, date(2022, 7, 1))
        histories = daily_sample()
        histories["BIL"].append(histories["BIL"][10])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            monthly_panel(histories, date(2022, 7, 1))
        histories = daily_sample()
        histories["AGG"][-1]["date"] = "2022-06-29"
        with self.assertRaisesRegex(ValueError, "disagree"):
            monthly_panel(histories, date(2022, 7, 1))

    def test_current_month_is_excluded(self):
        histories = daily_sample()
        for records in histories.values():
            records.append({"date": "2022-07-15", "adjusted_close": 99999, "close": 99999})
        panel = monthly_panel(histories, date(2022, 7, 20))
        self.assertEqual(panel[-1]["month"], "2022-06-30")

    def test_offline_rejects_stale_future_and_missing_months(self):
        for as_of in (date(2022, 8, 1), date(2022, 6, 1)):
            with self.assertRaises(ValueError):
                validate_panel(panel_sample(), as_of)
        rows = panel_sample()
        del rows[15]
        with self.assertRaises(ValueError):
            validate_panel(rows, date(2022, 7, 1))

    def test_offline_cli_creates_reports_and_repeats_deterministically(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root/"source.csv", root/"results"
            write_csv(source, panel_sample())
            args = ["--input-csv", str(source), "--output-dir", str(output), "--as-of", "2022-07-01"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
                first = (output/"summary.json").read_bytes()
                main(args)
            self.assertEqual(first, (output/"summary.json").read_bytes())
            report = (output/"report.md").read_text()
            self.assertIn("not actual or paper-account returns", report)
            self.assertIn("VEU", report)
            self.assertEqual(len(list(output.glob("*_path.csv"))), 5)
            self.assertFalse((root/"faber_paper").exists())

    def test_refresh_cli_with_mocked_provider(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("scanner.gem_research.fetch_panel", return_value=(panel_sample(), daily_sample())):
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["--refresh", "--output-dir", folder, "--as-of", "2022-07-01"])
            self.assertTrue((Path(folder)/"daily_snapshot.json").exists())
            report = json.loads((Path(folder)/"summary.json").read_text())
            self.assertEqual(report["data"]["source"], "Yahoo adjusted closes")


if __name__ == "__main__":
    unittest.main()
