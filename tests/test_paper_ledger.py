import contextlib
import csv
import io
import json
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from scanner.paper_cli import main
from scanner.paper_ledger import (
    SYMBOLS, add_income, cancel_plan, create_plan, export_account, fill_plan,
    initialize, mark_account, read_prices, snapshot,
)


def report(signal_date="2026-08-31", as_of="2026-09-29", quote="2026-09-28", inactive=("IEF",)):
    return {"signal_date": signal_date, "as_of": as_of, "equity": 99999999,
            "slots": [{"ticker": t, "signal": "TBILL" if t in inactive else "ASSET",
                       "target_weight": 0 if t in inactive else 0.2,
                       "quote_date": quote, "quote_close": 100} for t in SYMBOLS]}


def prices(value=100):
    return {t: Decimal(str(value)) for t in SYMBOLS}


class PaperLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "account.sqlite"
        self.opened = date(2026, 9, 29)
        self.fill_day, self.fill_record_day = date(2026, 9, 30), date(2026, 10, 1)
        initialize(self.db, 100000, self.opened)

    def tearDown(self):
        self.temp.cleanup()

    def plan(self, **kwargs):
        return create_plan(self.db, report(), self.opened, **kwargs)

    def fill(self, plan, values=None):
        return fill_plan(self.db, plan["id"], values or prices(), self.fill_day, self.fill_record_day)

    def test_initialization_is_cash_only_and_refuses_overwrite(self):
        state = snapshot(self.db)["account"]
        self.assertEqual(state["cash_cents"], 10000000)
        self.assertTrue(all(p["shares"] == 0 for p in state["positions"].values()))
        with self.assertRaisesRegex(ValueError, "already exists"):
            initialize(self.db, 1, self.opened)
        self.assertEqual(snapshot(self.db)["account"], state)

    def test_plan_uses_account_equity_and_records_no_fills(self):
        plan = self.plan()
        self.assertEqual(plan["targets"]["SPY"], 200)
        self.assertEqual(plan["targets"]["IEF"], 0)
        self.assertEqual(len(plan["orders"]), 4)
        self.assertEqual(snapshot(self.db)["events"], [])
        self.assertEqual(snapshot(self.db)["account"]["cash_cents"], 10000000)

    def test_rejects_duplicate_plan_and_duplicate_fill(self):
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.plan()
        self.fill(plan)
        original = snapshot(self.db)
        with self.assertRaisesRegex(ValueError, "already filled"):
            self.fill(plan)
        self.assertEqual(snapshot(self.db), original)

    def test_no_backdated_same_day_or_partial_session_fills(self):
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, "after plan"):
            fill_plan(self.db, plan["id"], prices(), self.opened, self.fill_day)
        with self.assertRaisesRegex(ValueError, "completed day"):
            fill_plan(self.db, plan["id"], prices(), self.fill_day, self.fill_day)
        self.assertEqual(snapshot(self.db)["events"], [])

    def test_stale_allocation_month_is_rejected(self):
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, "earlier allocation"):
            fill_plan(self.db, plan["id"], prices(), date(2026, 10, 1), date(2026, 10, 2))

    def test_fees_slippage_and_cost_basis_reconcile(self):
        plan = self.plan(fee=1, slippage_bps=5)
        fills = self.fill(plan)
        state = snapshot(self.db)["account"]
        mark = snapshot(self.db)["valuations"][-1]
        self.assertEqual(len(fills), 4)
        self.assertEqual(state["cash_cents"], 1995600)
        self.assertEqual(state["fees_cents"], 400)
        self.assertEqual(state["positions"]["SPY"]["cost_cents"], 2001100)
        self.assertEqual(mark["equity_cents"], 9995600)
        self.assertEqual(mark["pnl_cents"], -4400)
        self.assertEqual(mark["pnl_cents"], sum(p["unrealized_cents"] for p in mark["holdings"]))

    def test_cash_failure_rolls_back_entire_batch(self):
        plan = create_plan(self.db, report(inactive=()), self.opened)
        original = snapshot(self.db)
        with self.assertRaisesRegex(ValueError, "Insufficient cash"):
            self.fill(plan, prices(110))
        self.assertEqual(snapshot(self.db), original)

    def test_new_month_sells_before_buys_and_realized_pnl_reconciles(self):
        first = self.plan()
        self.fill(first)
        next_report = report("2026-09-30", "2026-10-01", "2026-09-30", ("VNQ",))
        next_plan = create_plan(self.db, next_report, date(2026, 10, 1))
        # IEF enters and VNQ exits; the sale funds the buy.
        fills = fill_plan(self.db, next_plan["id"], prices(90), date(2026, 10, 2), date(2026, 10, 3))
        self.assertEqual([f["side"] for f in fills], ["SELL", "BUY"])
        self.assertEqual(fills[0]["ticker"], "VNQ")
        self.assertEqual(fills[0]["realized_cents"], -200000)
        data = snapshot(self.db)
        latest = data["valuations"][-1]
        self.assertEqual(latest["pnl_cents"], data["account"]["realized_cents"] +
                         sum(h["unrealized_cents"] for h in latest["holdings"]))
        self.assertEqual(data["account"]["positions"]["VNQ"]["shares"], 0)
        self.assertEqual(data["account"]["positions"]["VNQ"]["cost_cents"], 0)

    def test_partial_sale_retains_basis_and_same_signal_rebalances(self):
        first = self.plan()
        self.fill(first)
        next_report = report("2026-09-30", "2026-10-01", "2026-09-30")
        next_report["slots"][0]["quote_close"] = 200  # SPY doubled, other ETFs unchanged.
        next_plan = create_plan(self.db, next_report, date(2026, 10, 1))
        values = prices()
        values["SPY"] = Decimal(200)
        fills = fill_plan(self.db, next_plan["id"], values, date(2026, 10, 2), date(2026, 10, 3))
        self.assertEqual(fills[0]["ticker"], "SPY")
        self.assertEqual(fills[0]["side"], "SELL")
        state = snapshot(self.db)["account"]
        self.assertEqual(state["positions"]["SPY"]["shares"], 120)
        self.assertEqual(state["positions"]["SPY"]["cost_cents"], 1200000)
        self.assertEqual(state["realized_cents"], 800000)

    def test_cancel_allows_replan_but_cannot_cancel_filled(self):
        first = self.plan()
        cancel_plan(self.db, first["id"])
        second = self.plan()
        self.assertNotEqual(first["id"], second["id"])
        self.fill(second)
        with self.assertRaisesRegex(ValueError, "pending"):
            cancel_plan(self.db, second["id"])

    def test_income_is_idempotent_and_invalidates_pending_plan(self):
        plan = self.plan()
        add_income(self.db, "25.50", self.opened, self.fill_day, "dividend", "unique-dividend")
        state = snapshot(self.db)["account"]
        self.assertEqual(state["cash_cents"], 10002550)
        with self.assertRaisesRegex(ValueError, "already recorded"):
            add_income(self.db, "25.50", self.opened, self.fill_day, "dividend", "unique-dividend")
        with self.assertRaisesRegex(ValueError, "Account changed"):
            self.fill(plan)
        mark = mark_account(self.db, prices(), self.fill_day, self.fill_record_day)
        self.assertEqual(mark["pnl_cents"], 2550)

    def test_marks_idempotent_immutable_and_chronological(self):
        plan = self.plan()
        self.fill(plan)
        original = snapshot(self.db)
        mark_account(self.db, prices(), self.fill_day, self.fill_record_day)
        self.assertEqual(snapshot(self.db), original)
        with self.assertRaisesRegex(ValueError, "overwritten"):
            mark_account(self.db, prices(110), self.fill_day, self.fill_record_day)
        with self.assertRaisesRegex(ValueError, "before recorded"):
            mark_account(self.db, prices(), self.opened, self.fill_day)

    def test_bad_inputs_and_stale_prices_are_rejected(self):
        for equity in (0, -1, "NaN"):
            with self.assertRaises(ValueError):
                initialize(Path(self.temp.name) / "bad.sqlite", equity, self.opened)
        for fee, slip in ((-1, 0), (0, -1), (0, 10000)):
            with self.assertRaises(ValueError):
                self.plan(fee=fee, slippage_bps=slip)
        bad = report()
        bad["slots"][0]["target_weight"] = 0.4
        with self.assertRaisesRegex(ValueError, "20%"):
            create_plan(self.db, bad, self.opened)
        bad = report()
        bad["slots"][0]["quote_close"] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            create_plan(self.db, bad, self.opened)
        bad = report(quote="2026-09-01")
        with self.assertRaisesRegex(ValueError, "stale"):
            create_plan(self.db, bad, self.opened)

    def test_price_csv_requires_all_five_aligned_unique_dates(self):
        source = Path(self.temp.name) / "prices.csv"
        source.write_text("ticker,date,close\nSPY,2026-09-30,100\nSPY,2026-09-30,100\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_prices(source, self.fill_record_day)
        source.write_text("ticker,date,close\nSPY,2026-09-30,100\nEFA,2026-09-29,100\n")
        with self.assertRaisesRegex(ValueError, "agree"):
            read_prices(source, self.fill_record_day)

    def test_export_offline_and_cli_plan_preserves_cash(self):
        signals = Path(self.temp.name) / "signals.json"
        signals.write_text(json.dumps(report()))
        folder = Path(self.temp.name) / "exports"
        with patch("scanner.paper_cli.today", return_value=self.opened), contextlib.redirect_stdout(io.StringIO()):
            main(["--db", str(self.db), "--output-dir", str(folder), "plan", "--signals", str(signals)])
        self.assertEqual(len(snapshot(self.db)["plans"]), 1)
        self.assertEqual(snapshot(self.db)["account"]["cash_cents"], 10000000)
        self.assertIn("Cash-only", (folder / "account.md").read_text())
        self.assertTrue((folder / "plan.json").exists())
        with (folder / "holdings.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 5)

    def test_live_prices_cli_uses_raw_close_and_does_not_modify_account(self):
        folder = Path(self.temp.name) / "exports"
        with patch("scanner.paper_cli.today", return_value=self.fill_record_day), contextlib.redirect_stdout(io.StringIO()):
            with patch("scanner.paper_cli.fetch_daily", return_value=[{"date": "2026-09-30", "close": 100, "adjusted_close": 70}]):
                main(["--db", str(self.db), "--output-dir", str(folder), "prices"])
        day, values = read_prices(folder / "prices.csv", self.fill_record_day)
        self.assertEqual(day, self.fill_day)
        self.assertTrue(all(p == 100 for p in values.values()))
        self.assertEqual(snapshot(self.db)["events"], [])


if __name__ == "__main__":
    unittest.main()
