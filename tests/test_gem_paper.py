import contextlib
import io
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from scanner.gem_execution_data import NY, TICKERS, sessions
from scanner.gem_paper import initialize, main, run_daily, snapshot
from scanner.gem_paper_data import parse_chart, validate_snapshot
from scanner.paper_ledger import initialize as initialize_faber


def fixture(as_of, switches=False):
    days = sessions(date(2024, 1, 2), as_of-timedelta(days=1))
    histories = {}
    for ticker in TICKERS:
        rows = []
        for day in days:
            d = date.fromisoformat(day)
            month = (d.year-2024)*12+d.month
            drift = {"SPY": 2, "VEU": 1, "AGG": .2, "BIL": .1}[ticker]
            adjusted = 100+month*drift
            if switches and ticker == "SPY" and day >= "2025-04-30":
                adjusted = 90
            rows.append({"date": day, "open": 100, "close": 100, "adjusted_close": adjusted})
        histories[ticker] = rows
    return {"schema_version": 1, "as_of": as_of.isoformat(), "source": "synthetic test fixture",
            "histories": histories, "corporate_actions_included": True, "actions": []}


def set_price(data, ticker, day, **fields):
    next(r for r in data["histories"][ticker] if r["date"] == day).update(fields)


class GemPaperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root/"gem.sqlite"
        self.opened = date(2025, 4, 1)

    def tearDown(self):
        self.temp.cleanup()

    def init(self, equity="100000", cost="0", day=None):
        day = day or self.opened
        return initialize(self.db, fixture(day), day, equity, cost)

    def test_initial_plan_has_no_backdated_fill(self):
        state = self.init()
        self.assertEqual(state["fills"], [])
        self.assertEqual(state["plans"][-1]["ticker"], "SPY")
        self.assertEqual(state["plans"][-1]["eligible_on"], "2025-04-02")
        state = run_daily(self.db, fixture(date(2025, 4, 2)), date(2025, 4, 2))
        self.assertEqual(state["fills"], [])
        self.assertEqual(state["last_run"]["status"], "WAITING")

    def test_open_fill_uses_whole_shares_and_marks_at_close(self):
        self.init()
        data = fixture(date(2025, 4, 3))
        set_price(data, "SPY", "2025-04-02", open=110, close=121)
        state = run_daily(self.db, data, date(2025, 4, 3))
        self.assertEqual(state["positions"]["SPY"]["shares"], 909)
        self.assertEqual(state["cash_cents"], 1000)
        self.assertEqual(state["valuations"][-1]["equity_cents"], 110000*100-100)
        self.assertEqual(state["fills"][0]["reference_open"], "110")

    def test_duplicate_run_is_idempotent(self):
        self.init()
        data = fixture(date(2025, 4, 4))
        first = run_daily(self.db, data, date(2025, 4, 4))
        second = run_daily(self.db, data, date(2025, 4, 4))
        for key in ("fills", "positions", "cash_cents", "valuations", "plans", "income"):
            self.assertEqual(first[key], second[key])
        self.assertEqual(second["last_run"]["new_fills"], 0)

    def test_catchup_uses_original_open_not_latest_open(self):
        self.init()
        data = fixture(date(2025, 4, 10))
        set_price(data, "SPY", "2025-04-02", open=125)
        set_price(data, "SPY", "2025-04-09", open=500)
        state = run_daily(self.db, data, date(2025, 4, 10))
        self.assertEqual(state["fills"][0]["date"], "2025-04-02")
        self.assertEqual(state["fills"][0]["shares"], 800)
        self.assertEqual(len(state["valuations"]), 7)

    def test_missing_session_leaves_entire_ledger_unchanged(self):
        self.init()
        before = snapshot(self.db)
        data = fixture(date(2025, 4, 10))
        for t in TICKERS:
            data["histories"][t] = [r for r in data["histories"][t] if r["date"] != "2025-04-02"]
        with self.assertRaisesRegex(ValueError, "session"):
            run_daily(self.db, data, date(2025, 4, 10))
        self.assertEqual(before, snapshot(self.db))

    def test_monthly_switch_old_asset_gets_overnight_gap(self):
        self.init()
        data = fixture(date(2025, 5, 2), switches=True)
        set_price(data, "SPY", "2025-05-01", open=90, close=60)
        set_price(data, "AGG", "2025-05-01", open=110, close=121)
        state = run_daily(self.db, data, date(2025, 5, 2))
        self.assertEqual([(r["ticker"], r["side"], r["date"]) for r in state["fills"]],
                         [("SPY", "BUY", "2025-04-02"), ("SPY", "SELL", "2025-05-01"), ("AGG", "BUY", "2025-05-01")])
        self.assertEqual(state["positions"]["AGG"]["shares"], 818)
        self.assertEqual(state["cash_cents"], 2000)
        self.assertEqual(state["valuations"][-1]["equity_cents"], 98998*100)
        self.assertEqual(state["realized_cents"], -10000*100)

    def test_unchanged_signal_does_not_rebalance(self):
        self.init()
        state = run_daily(self.db, fixture(date(2025, 5, 2)), date(2025, 5, 2))
        self.assertEqual(len(state["fills"]), 1)
        self.assertEqual(state["plans"][-1]["status"], "NO_TRADE")

    def test_buy_costs_reduce_affordable_shares(self):
        self.init(cost="10")
        state = run_daily(self.db, fixture(date(2025, 4, 3)), date(2025, 4, 3))
        self.assertEqual(state["positions"]["SPY"]["shares"], 999)
        self.assertEqual(state["fees_cents"], 9990)
        self.assertEqual(state["cash_cents"], 10)
        self.assertEqual(state["valuations"][-1]["equity_cents"], 10000000-9990)

    def test_switch_costs_reconcile_to_cash_and_basis(self):
        self.init(cost="10")
        state = run_daily(self.db, fixture(date(2025, 5, 2), switches=True), date(2025, 5, 2))
        # Buy 999 at $100 + $99.90, sell 999 - $99.90,
        # then buy 997 + $99.70; $0.50 remains in cash.
        self.assertEqual(state["fees_cents"], 29950)
        self.assertEqual(state["cash_cents"], 50)
        self.assertEqual(state["positions"]["AGG"], {"shares": 997, "cost_cents": 9979970})
        self.assertEqual(state["valuations"][-1]["equity_cents"], 10000000-29950)

    def test_future_prices_do_not_change_earlier_fills(self):
        self.init()
        data = fixture(date(2025, 4, 10))
        first = run_daily(self.db, data, date(2025, 4, 10))
        other = self.root/"other.sqlite"
        initialize(other, fixture(self.opened), self.opened, cost_bps="0")
        set_price(data, "SPY", "2025-04-09", open=999, close=999, adjusted_close=999)
        second = run_daily(other, data, date(2025, 4, 10))
        self.assertEqual(first["fills"], second["fills"])

    def test_new_month_does_not_rewrite_recorded_monthly_signal(self):
        self.init()
        run_daily(self.db, fixture(date(2025, 4, 3)), date(2025, 4, 3))
        state = run_daily(self.db, fixture(date(2025, 5, 1), switches=True), date(2025, 5, 1))
        self.assertEqual(state["plans"][-1]["ticker"], "AGG")
        # Refreshed historical adjusted data changes the recomputed signal;
        # the already-recorded standing plan is still authoritative.
        state = run_daily(self.db, fixture(date(2025, 5, 2)), date(2025, 5, 2))
        self.assertEqual(state["fills"][-1]["ticker"], "AGG")

    def test_dividend_uses_prior_holding_before_switch(self):
        self.init()
        data = fixture(date(2025, 5, 2), switches=True)
        data["actions"] = [{"kind": "DIVIDEND", "ticker": t, "date": "2025-05-01", "amount": "1"} for t in ("SPY", "AGG")]
        state = run_daily(self.db, data, date(2025, 5, 2))
        self.assertEqual(state["income_cents"], 100000)
        self.assertEqual(state["income"][0]["ticker"], "SPY")
        self.assertEqual(state["positions"]["AGG"]["shares"], 1010)
        again = run_daily(self.db, data, date(2025, 5, 2))
        self.assertEqual(again["income"], state["income"])

    def test_same_day_purchase_not_entitled_to_dividend(self):
        self.init()
        data = fixture(date(2025, 4, 3))
        data["actions"] = [{"kind": "DIVIDEND", "ticker": "SPY", "date": "2025-04-02", "amount": "1"}]
        state = run_daily(self.db, data, date(2025, 4, 3))
        self.assertEqual(state["income_cents"], 0)

    def test_split_stops_before_any_fill_or_mark(self):
        self.init()
        before = snapshot(self.db)
        data = fixture(date(2025, 4, 4))
        data["actions"] = [{"kind": "SPLIT", "ticker": "SPY", "date": "2025-04-03", "ratio": "2"}]
        with self.assertRaisesRegex(ValueError, "Split detected"):
            run_daily(self.db, data, date(2025, 4, 4))
        self.assertEqual(before, snapshot(self.db))

    def test_late_dividend_correction_requires_review(self):
        self.init()
        data = fixture(date(2025, 4, 4))
        before = run_daily(self.db, data, date(2025, 4, 4))
        data["actions"] = [{"kind": "DIVIDEND", "ticker": "SPY", "date": "2025-04-03", "amount": "1"}]
        with self.assertRaisesRegex(ValueError, "corporate actions changed"):
            run_daily(self.db, data, date(2025, 4, 4))
        self.assertEqual(before, snapshot(self.db))

    def test_cannot_reinitialize_or_use_faber_account(self):
        self.init()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.init()
        other = self.root/"faber.sqlite"
        initialize_faber(other, "100000", self.opened)
        before = other.read_bytes()
        with self.assertRaisesRegex(ValueError, "not a GEM database"):
            initialize(other, fixture(self.opened), self.opened)
        self.assertEqual(before, other.read_bytes())

    def test_weekend_initialization_can_wait_without_error(self):
        day = date(2025, 4, 6)
        self.init(day=day)
        state = run_daily(self.db, fixture(day), day)
        self.assertEqual(state["plans"][-1]["eligible_on"], "2025-04-07")
        self.assertEqual(state["fills"], [])

    def test_month_end_initialization_refreshes_before_first_fill(self):
        day = date(2025, 4, 30)
        self.init(day=day)
        state = run_daily(self.db, fixture(date(2025, 5, 2), switches=True), date(2025, 5, 2))
        self.assertEqual(len(state["fills"]), 1)
        self.assertEqual(state["fills"][0]["ticker"], "AGG")
        self.assertEqual(state["plans"][0]["status"], "SUPERSEDED")

    def test_weekend_month_end_confirms_without_new_session(self):
        self.init(day=date(2025, 8, 1))
        run_daily(self.db, fixture(date(2025, 8, 30)), date(2025, 8, 30))
        state = run_daily(self.db, fixture(date(2025, 9, 1), switches=True), date(2025, 9, 1))
        self.assertEqual(state["last_run"]["sessions_processed"], 0)
        self.assertEqual(state["plans"][-1]["signal_date"], "2025-08-29")
        self.assertEqual(state["plans"][-1]["eligible_on"], "2025-09-02")

    def test_insufficient_cash_rolls_back_entire_run(self):
        self.init(equity="50")
        before = snapshot(self.db)
        with self.assertRaisesRegex(ValueError, "one whole share"):
            run_daily(self.db, fixture(date(2025, 4, 3)), date(2025, 4, 3))
        self.assertEqual(before, snapshot(self.db))

    def test_price_only_snapshot_rejected(self):
        data = fixture(self.opened)
        del data["corporate_actions_included"]
        with self.assertRaisesRegex(ValueError, "dividend and split"):
            initialize(self.db, data, self.opened)
        self.assertFalse(self.db.exists())

    def test_future_and_invalid_prices_rejected(self):
        for value in (None, 0, -1, float("nan"), float("inf")):
            data = fixture(date(2025, 4, 4))
            set_price(data, "VEU", "2025-04-03", open=value)
            with self.assertRaises((ValueError, ArithmeticError)):
                validate_snapshot(data, date(2025, 4, 4))
        data = fixture(date(2025, 4, 4))
        data["histories"]["SPY"].append({"date": "2025-04-04", "open": 100, "close": 100, "adjusted_close": 100})
        with self.assertRaisesRegex(ValueError, "future"):
            validate_snapshot(data, date(2025, 4, 4))

    def test_offline_cli_and_dashboard(self):
        data = self.root/"prices.json"
        data.write_text(json.dumps(fixture(self.opened)))
        with patch("scanner.gem_paper.datetime") as clock, contextlib.redirect_stdout(io.StringIO()):
            clock.now.return_value = datetime(2025, 4, 1, 22, tzinfo=NY)
            main(["--db", str(self.db), "init", "--snapshot", str(data)])
            main(["--db", str(self.db), "status"])
        self.assertTrue((self.root/"dashboard.html").exists())
        self.assertTrue((self.root/"fills.csv").exists())
        self.assertIn("ex-dividend", (self.root/"account.txt").read_text())

    def test_provider_event_parser(self):
        stamp = int(datetime(2025, 4, 2, 9, 30, tzinfo=NY).timestamp())
        chart = {"result": [{"timestamp": [stamp], "indicators": {"quote": [{"open": [100], "close": [101]}], "adjclose": [{"adjclose": [99]}]},
                              "events": {"dividends": {"a": {"date": stamp, "amount": 1.25}},
                                         "splits": {"b": {"date": stamp, "numerator": 2, "denominator": 1}}}}]}
        bars, actions = parse_chart(chart, "SPY", date(2025, 4, 3))
        self.assertEqual(bars[0]["open"], 100)
        self.assertEqual([a["kind"] for a in actions], ["DIVIDEND", "SPLIT"])
        self.assertEqual(parse_chart(chart, "SPY", date(2025, 4, 2)), ([], []))


if __name__ == "__main__":
    unittest.main()
