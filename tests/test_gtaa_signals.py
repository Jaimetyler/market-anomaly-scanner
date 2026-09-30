import contextlib
import csv
import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from scanner.gtaa_signals import TICKERS, make_report, monthly_input, fetch_daily
from scanner.published_faber_gtaa import ASSETS, backtest
from scanner.strategy_cli import main


def monthly():
    dates = ["2025-09-30", "2025-10-31", "2025-11-28", "2025-12-31", "2026-01-30", "2026-02-27",
             "2026-03-31", "2026-04-30", "2026-05-29", "2026-06-30", "2026-07-31", "2026-08-31"]
    return [{"month": d, "tbill_return": 0, **{a: 100 + i for a in ASSETS}}
            for i, d in enumerate(dates)]


def histories():
    return {t: [{"date": r["month"], "adjusted_close": r[a], "close": r[a] + 10}
                for r in monthly()] + [{"date": "2026-09-28", "adjusted_close": 40, "close": 50}]
            for a, t in TICKERS.items()}


class SignalsTests(unittest.TestCase):
    def test_completed_month_matches_reference_and_ignores_partial(self):
        rows = monthly()
        rows.append({"month": "2026-09-28", "tbill_return": 0, **{a: 1 for a in ASSETS}})
        report = make_report(rows, date(2026, 9, 29), histories=histories())
        reference = backtest(monthly())[-1]
        self.assertEqual(report["signal_date"], "2026-08-31")
        self.assertEqual(report["cash_weight"], 0)
        for asset, slot in zip(ASSETS, report["slots"]):
            self.assertEqual(slot["signal"], reference[f"{asset}_next"])
            self.assertEqual(slot["estimated_target_shares"], 400)
            self.assertEqual(slot["developing_signal"], "TBILL")
            self.assertIsNone(slot["action"])

    def test_cash_not_redistributed_and_real_holdings_actions(self):
        rows = monthly()
        rows[-1]["commodities"] = 10
        report = make_report(rows, date(2026, 9, 29), histories=histories(),
                             holdings={"SPY": 450, "GSG": 10})
        self.assertAlmostEqual(report["cash_weight"], 0.2)
        self.assertEqual(report["slots"][0]["action"], "SELL")
        self.assertEqual(report["slots"][0]["share_change"], -50)
        self.assertEqual(report["slots"][1]["action"], "BUY")
        self.assertEqual(report["slots"][-1]["estimated_target_shares"], 0)
        self.assertEqual(report["slots"][-1]["share_change"], -10)

    def test_month_end_waits_until_next_calendar_day(self):
        report = make_report(monthly(), date(2026, 8, 31))
        self.assertEqual(report["signal_date"], "2026-07-31")
        self.assertEqual(make_report(monthly(), date(2026, 9, 1))["signal_date"], "2026-08-31")

    def test_rejects_stale_missing_or_misaligned_data(self):
        with self.assertRaisesRegex(ValueError, "Stale"):
            make_report(monthly()[:-1], date(2026, 9, 29))
        with self.assertRaisesRegex(ValueError, "consecutive"):
            make_report(monthly()[:3] + monthly()[4:], date(2026, 9, 29))
        data = histories()
        data["SPY"][-2]["date"] = "2026-08-28"
        with self.assertRaisesRegex(ValueError, "disagree"):
            monthly_input(data, date(2026, 9, 29))
        data = histories()
        data["SPY"][-1]["date"] = "2026-09-27"
        with self.assertRaisesRegex(ValueError, "disagree"):
            make_report(monthly(), date(2026, 9, 29), histories=data)

    def test_tie_preserves_previous_allocation(self):
        rows = monthly()
        # Month 11 equals its ten-point SMA and retains month 10's long signal.
        rows[-1]["us_stocks"] = sum(r["us_stocks"] for r in rows[-10:-1]) / 9
        result = make_report(rows, date(2026, 9, 29))
        self.assertEqual(result["slots"][0]["signal"], "ASSET")

    def test_rejects_invalid_sizing_and_stale_quotes(self):
        with self.assertRaises(ValueError):
            make_report(monthly(), date(2026, 9, 29), equity=float("nan"))
        with self.assertRaises(ValueError):
            make_report(monthly(), date(2026, 9, 29), histories=histories(), holdings={"SPY": -1})
        data = histories()
        data["SPY"][-1]["date"] = "2026-09-01"
        with self.assertRaisesRegex(ValueError, "stale daily"):
            make_report(monthly(), date(2026, 9, 29), histories=data)
        rows = monthly()
        rows[-1]["month"] = "2026-08-20"
        with self.assertRaisesRegex(ValueError, "too early"):
            make_report(rows, date(2026, 9, 29))

    def test_live_cli_saves_snapshot_and_dated_csv(self):
        data = histories()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            with patch("scanner.gtaa_signals.fetch_daily", side_effect=lambda ticker, as_of: data[ticker]):
                main(["signals", "faber-gtaa5", "--as-of", "2026-09-29", "--output-dir", temp])
            self.assertEqual(json.loads((Path(temp) / "daily_snapshot.json").read_text()), data)
            with (Path(temp) / "signals.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["signal_date"], "2026-08-31")
            self.assertEqual(rows[0]["as_of"], "2026-09-29")

    def test_fetch_excludes_as_of_day_and_does_not_fallback_to_raw(self):
        stamps = [1788183000, 1788269400]  # Aug 31 and Sep 1, 2026 market opens.
        payload = {"chart": {"result": [{"timestamp": stamps,
                   "indicators": {"adjclose": [{"adjclose": [101, 102]}],
                                  "quote": [{"close": [111, 112]}]}}], "error": None}}
        with patch("scanner.gtaa_signals.urllib.request.urlopen") as opened:
            opened.return_value.__enter__.return_value = io.StringIO(json.dumps(payload))
            bars = fetch_daily("SPY", date(2026, 9, 1))
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0]["close"], 111)
        self.assertEqual(bars[0]["adjusted_close"], 101)

    def test_cli_offline_writes_three_reports_and_failure_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.csv"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["month", "tbill_return", *ASSETS])
                writer.writeheader()
                writer.writerows(monthly())
            folder = Path(temp) / "report"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["signals", "faber-gtaa5", "--as-of", "2026-09-29", "--input-csv", str(source),
                      "--output-dir", str(folder)])
            self.assertEqual(len(list(folder.iterdir())), 3)
            self.assertIn("2026-08-31", (folder / "signals.md").read_text())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["signals", "faber-gtaa5", "--as-of", "2026-10-01", "--input-csv", str(source),
                      "--output-dir", str(Path(temp) / "failed")])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((Path(temp) / "failed").exists())


if __name__ == "__main__":
    unittest.main()
