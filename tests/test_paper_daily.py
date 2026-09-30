import contextlib
import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from scanner.paper_daily import main, run_daily, runner_lock
from scanner.paper_dashboard import write_dashboard
from scanner.paper_ledger import SYMBOLS, add_income, create_plan, initialize, snapshot


def histories():
    monthly_dates = ["2025-09-30", "2025-10-31", "2025-11-28", "2025-12-31", "2026-01-30",
                     "2026-02-27", "2026-03-31", "2026-04-30", "2026-05-29", "2026-06-30",
                     "2026-07-31", "2026-08-31"]
    result = {}
    for ticker in SYMBOLS:
        bars = [{"date": d, "adjusted_close": 200 - i if ticker == "IEF" else 100 + i,
                 "close": 100} for i, d in enumerate(monthly_dates)]
        for d in ("2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"):
            level = 180 if ticker == "IEF" else 80 if ticker == "VNQ" else 112
            bars.append({"date": d, "adjusted_close": level, "close": 100})
        result[ticker] = bars
    return result


def initial_report():
    return {"as_of": "2026-09-29", "signal_date": "2026-08-31",
            "slots": [{"ticker": t, "signal": "TBILL" if t == "IEF" else "ASSET",
                       "target_weight": 0 if t == "IEF" else 0.2,
                       "quote_date": "2026-09-28", "quote_close": 100} for t in SYMBOLS]}


class DailyRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "account.sqlite"
        self.folder = Path(self.temp.name) / "reports"
        self.opened = date(2026, 9, 29)
        initialize(self.db, 100000, self.opened)

    def tearDown(self):
        self.temp.cleanup()

    def run_day(self, day, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_daily(self.db, self.folder, day, histories=kwargs.pop("histories", histories()), **kwargs)

    def test_waiting_is_success_and_preserves_existing_plan(self):
        plan = create_plan(self.db, initial_report(), self.opened)
        before = snapshot(self.db)
        result = self.run_day(self.opened)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["fills_recorded"], 0)
        self.assertTrue(any(m.startswith("WAITING:") for m in result["messages"]))
        self.assertEqual(snapshot(self.db), before)
        self.assertEqual(snapshot(self.db)["plans"][0]["id"], plan["id"])
        self.assertTrue((self.folder / "dashboard.html").exists())

    def test_creates_initial_plan_without_spending_cash(self):
        result = self.run_day(self.opened)
        data = snapshot(self.db)
        self.assertEqual(len(data["plans"]), 1)
        self.assertEqual(data["plans"][0]["created_on"], "2026-09-29")
        self.assertEqual(data["events"], [])
        self.assertEqual(data["account"]["cash_cents"], 10000000)
        self.assertEqual(result["fills_recorded"], 0)

    def test_september_fills_before_october_plan_and_repeats_are_idempotent(self):
        first = create_plan(self.db, initial_report(), self.opened)
        result = self.run_day(date(2026, 10, 1))
        data = snapshot(self.db)
        self.assertEqual(result["fills_recorded"], 4)
        self.assertEqual(data["plans"][0]["id"], first["id"])
        self.assertEqual(data["plans"][0]["status"], "FILLED")
        self.assertEqual(data["plans"][1]["signal_date"], "2026-09-30")
        self.assertEqual(data["plans"][1]["created_on"], "2026-10-01")
        self.assertEqual(data["plans"][1]["status"], "PENDING")
        self.assertEqual(data["plans"][1]["orders"], [{"ticker": "VNQ", "side": "SELL", "shares": 200}])
        self.assertEqual({e["date"] for e in data["events"]}, {"2026-09-30"})
        again = self.run_day(date(2026, 10, 1))
        self.assertEqual(again["fills_recorded"], 0)
        self.assertEqual(snapshot(self.db), data)
        self.assertEqual(len((self.folder / "daily_runs.jsonl").read_text().splitlines()), 2)

    def test_october_plan_waits_for_later_price_date_then_exits_vnq(self):
        create_plan(self.db, initial_report(), self.opened)
        self.run_day(date(2026, 10, 1))
        waiting = self.run_day(date(2026, 10, 2))
        self.assertEqual(waiting["fills_recorded"], 0)
        filled = self.run_day(date(2026, 10, 3))
        self.assertEqual(filled["fills_recorded"], 1)
        data = snapshot(self.db)
        self.assertEqual(data["account"]["positions"]["VNQ"]["shares"], 0)
        self.assertEqual(data["account"]["cash_cents"], 4000000)
        self.assertEqual(data["valuations"][-1]["equity_cents"], 10000000)

    def test_missed_september_plan_is_cancelled_before_october_planning(self):
        create_plan(self.db, initial_report(), self.opened)
        result = self.run_day(date(2026, 10, 2))
        data = snapshot(self.db)
        self.assertEqual(data["plans"][0]["status"], "CANCELLED")
        self.assertEqual(data["plans"][1]["signal_month"], "2026-09")
        self.assertEqual(result["fills_recorded"], 0)
        self.assertEqual(data["events"], [])

    def test_cash_failure_keeps_pending_plan_and_returns_review(self):
        create_plan(self.db, initial_report(), self.opened)
        bars = histories()
        for ticker in SYMBOLS:
            for bar in bars[ticker]:
                if bar["date"] == "2026-09-30":
                    bar["close"] = 200
        result = self.run_day(date(2026, 10, 1), histories=bars)
        data = snapshot(self.db)
        self.assertEqual(result["status"], "REVIEW")
        self.assertEqual(data["plans"][0]["status"], "PENDING")
        self.assertEqual(data["events"], [])
        self.assertEqual(data["account"]["cash_cents"], 10000000)
        self.assertEqual(len(data["plans"]), 1)

    def test_account_change_requires_review_not_silent_replanning(self):
        create_plan(self.db, initial_report(), self.opened)
        add_income(self.db, 10, self.opened, date(2026, 9, 30), "interest", "credit")
        result = self.run_day(date(2026, 10, 1))
        self.assertEqual(result["status"], "REVIEW")
        self.assertEqual(snapshot(self.db)["plans"][0]["status"], "PENDING")

    def test_bad_feed_never_changes_ledger(self):
        create_plan(self.db, initial_report(), self.opened)
        before = snapshot(self.db)
        bars = histories()
        bars["SPY"] = [r for r in bars["SPY"] if r["date"] != "2026-08-31"]
        with self.assertRaises(ValueError):
            self.run_day(self.opened, histories=bars)
        self.assertEqual(snapshot(self.db), before)

    def test_concurrent_runner_lock_rejects_and_releases(self):
        with runner_lock(self.db):
            with self.assertRaisesRegex(ValueError, "Another daily runner"):
                self.run_day(self.opened)
        self.run_day(self.opened)

    def test_new_plans_inherit_previous_assumptions(self):
        create_plan(self.db, initial_report(), self.opened, fee=1, slippage_bps=5)
        self.run_day(date(2026, 10, 1))
        plan = snapshot(self.db)["plans"][-1]
        self.assertEqual(plan["fee_cents"], 100)
        self.assertEqual(plan["slippage_bps"], "5")

    def test_expired_plan_keeps_cost_assumptions_for_its_replacement(self):
        create_plan(self.db, initial_report(), self.opened, fee=2, slippage_bps=10)
        self.run_day(date(2026, 10, 2))
        plan = snapshot(self.db)["plans"][-1]
        self.assertEqual(plan["fee_cents"], 200)
        self.assertEqual(plan["slippage_bps"], "10")

    def test_outputs_include_account_signals_and_status(self):
        self.run_day(self.opened)
        self.assertTrue((self.folder / "signals/signals.json").exists())
        self.assertTrue((self.folder / "signals/daily_snapshot.json").exists())
        self.assertTrue((self.folder / "account.md").exists())
        self.assertEqual(json.loads((self.folder / "daily_status.json").read_text())["status"], "OK")

    def test_dashboard_embedded_json_cannot_inject_script(self):
        self.run_day(self.opened)
        data = snapshot(self.db)
        run = {"as_of": "2026-09-29", "price_date": "2026-09-28", "status": "OK",
               "messages": ["</script><script>alert('x')</script>"]}
        signals = json.loads((self.folder / "signals/signals.json").read_text())
        path = self.folder / "injection.html"
        write_dashboard(path, data, signals, run)
        html = path.read_text()
        self.assertNotIn("</script><script>alert", html)
        self.assertIn("\\u003c/script\\u003e", html)

    def test_cli_offline_snapshot_waits_successfully(self):
        source = Path(self.temp.name) / "snapshot.json"
        source.write_text(json.dumps(histories()))
        with patch("scanner.paper_daily.datetime") as clock, contextlib.redirect_stdout(io.StringIO()):
            clock.now.return_value = datetime_for_test()
            main(["--db", str(self.db), "--output-dir", str(self.folder), "--snapshot-json", str(source)])
        self.assertEqual(len(snapshot(self.db)["plans"]), 1)


def datetime_for_test():
    from datetime import datetime
    return datetime(2026, 9, 29, 22, 0)


if __name__ == "__main__":
    unittest.main()
