import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from datetime import date
from pathlib import Path

from scanner.gem_execution import daily_drawdown, main, run_study, simulate
from scanner.gem_execution_data import TICKERS, prepare, sessions


def row(day, **prices):
    assets = {}
    for t in TICKERS:
        opening, close = prices.get(t, (100.0, 100.0))
        assets[t] = {"open": opening, "close": close,
                     "adjusted_open": opening, "adjusted_close": close}
    return {"date": day, "assets": assets}


def snapshot_fixture():
    days = sessions(date(2007, 5, 1), date(2009, 7, 31))
    histories = {}
    for j,t in enumerate(TICKERS):
        records = []
        for i,d in enumerate(days):
            # Known finite data with different drift and deterministic cycles;
            # dividends are represented by the supplied adjustment factor.
            level = 100 * math.exp(i*(.0001+.0002*(3-j)) + .1*math.sin(i/30+j))
            records.append({"date": d, "open": level*1.01, "close": level,
                            "adjusted_close": level*.8})
        histories[t] = records
    return {"schema_version":1,"as_of":"2009-08-01", "source":"synthetic fixture",
            "histories":histories}


class GemExecutionTests(unittest.TestCase):
    def test_first_entry_waits_for_next_open_and_misses_initial_gap(self):
        daily = [row("2020-06-30"), row("2020-07-01", SPY=(110,121)),
                 row("2020-07-02", SPY=(121,121))]
        signals = {"2020-06-30":{"ticker":"SPY"}}
        opened = simulate(daily,signals,cost_bps=0)
        closed = simulate(daily,signals,"signal_close",cost_bps=0)
        self.assertEqual(opened["path"][0]["held_at_close"],"CASH")
        self.assertEqual(opened["trades"][0]["fill_date"],"2020-07-01")
        self.assertAlmostEqual(opened["path"][-1]["equity"],110000)
        self.assertAlmostEqual(closed["path"][-1]["equity"],121000)

    def test_old_asset_bears_overnight_move_before_switch(self):
        daily = [row("2020-06-30"), row("2020-07-01",SPY=(100,110)),
                 row("2020-07-31",SPY=(110,120)),
                 row("2020-08-03",SPY=(90,60),AGG=(110,121))]
        signals = {"2020-06-30":{"ticker":"SPY"},"2020-07-31":{"ticker":"AGG"}}
        result = simulate(daily,signals,cost_bps=0)
        self.assertAlmostEqual(result["trades"][1]["equity_before_trade"],90000)
        self.assertAlmostEqual(result["path"][-1]["equity"],99000)
        self.assertEqual(result["trades"][1]["signal_date"],"2020-07-31")
        self.assertEqual(result["trades"][1]["fill_date"],"2020-08-03")

    def test_entry_and_switch_costs_use_fill_time_equity(self):
        daily = [row("2020-06-30"),row("2020-07-01"),row("2020-07-31"),row("2020-08-03")]
        signals = {"2020-06-30":{"ticker":"SPY"},"2020-07-31":{"ticker":"AGG"}}
        result = simulate(daily,signals,cost_bps=10)
        self.assertAlmostEqual(result["path"][-1]["equity"],100000*.999*.998)
        self.assertEqual([t["gross_traded_weight"] for t in result["trades"]],[1,2])
        self.assertAlmostEqual(result["trades"][1]["cost_dollars"],199.8)

    def test_unchanged_signal_does_not_retrade(self):
        daily = [row("2020-06-30"),row("2020-07-01"),row("2020-07-31"),row("2020-08-03")]
        signals = {"2020-06-30":{"ticker":"SPY"},"2020-07-31":{"ticker":"SPY"}}
        result = simulate(daily,signals)
        self.assertEqual(len(result["trades"]),1)
        self.assertAlmostEqual(result["path"][-1]["equity"],99900)

    def test_final_signal_has_no_fill_without_following_session(self):
        daily = [row("2020-06-30"),row("2020-07-01"),row("2020-07-31")]
        signals = {"2020-06-30":{"ticker":"SPY"},"2020-07-31":{"ticker":"AGG"}}
        for mode in ("next_open","signal_close"):
            self.assertEqual(len(simulate(daily,signals,mode)["trades"]),1)

    def test_future_price_changes_do_not_change_past_fills(self):
        daily = [row("2020-06-30"),row("2020-07-01"),row("2020-07-02")]
        signals = {"2020-06-30":{"ticker":"SPY"}}
        before = simulate(daily,signals)
        daily[-1]["assets"]["SPY"]["adjusted_close"] = 400
        after = simulate(daily,signals)
        self.assertEqual(before["trades"],after["trades"])
        self.assertEqual(before["path"][:-1],after["path"][:-1])

    def test_daily_drawdown_captures_loss_hidden_at_month_end(self):
        path = [{"date":"2020-06-30","equity":100},
                {"date":"2020-07-10","equity":60},
                {"date":"2020-07-31","equity":110}]
        result = daily_drawdown(path)
        self.assertAlmostEqual(result["max_drawdown_daily_close"],-.4)
        self.assertEqual(result["drawdown_peak_date"],"2020-06-30")
        self.assertEqual(result["drawdown_trough_date"],"2020-07-10")

    def test_invalid_simulation_inputs_rejected(self):
        daily = [row("2020-06-30"),row("2020-07-01")]
        signals = {"2020-06-30":{"ticker":"SPY"}}
        for kwargs in ({"mode":"unknown"},{"cost_bps":math.nan},{"starting_equity":0},{"cost_bps":-1}):
            with self.assertRaises(ValueError):
                simulate(daily,signals,**kwargs)
        with self.assertRaises(ValueError):
            simulate(daily[::-1],signals)
        with self.assertRaises(ValueError):
            simulate(daily,{"2020-06-30":{"ticker":"BIL"}})
        with self.assertRaises(ValueError):
            simulate(daily,{"2020-07-01":{"ticker":"SPY"}})

    def test_daily_reference_reconciles_every_month_and_gap_attribution(self):
        monthly,daily = prepare(snapshot_fixture(),date(2009,8,1))
        result = run_study(monthly,daily)
        self.assertLess(result["max_monthly_reference_reconciliation_error"],.00001)
        ratio = math.prod(t["next_open_relative_wealth_factor"] for t in result["timing"])
        self.assertAlmostEqual(ratio,1+result["next_open_relative_ending_wealth"])
        for t in result["simulations"]["gem_next_open"]["trades"]:
            self.assertGreater(t["fill_date"],t["signal_date"])

    def test_offline_command_writes_reproducible_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/"input.json"
            source.write_text(json.dumps(snapshot_fixture()))
            output=root/"audit"
            args=["--snapshot",str(source),"--output-dir",str(output),"--as-of","2009-08-01"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
                first=(output/"summary.json").read_bytes()
                main(args)
            self.assertEqual(first,(output/"summary.json").read_bytes())
            self.assertTrue((output/"gem_next_open_trades.csv").exists())
            self.assertTrue((output/"switch_timing.csv").exists())
            self.assertIn("not ETF shares",(output/"report.md").read_text())


class GemExecutionDataTests(unittest.TestCase):
    def test_adjusted_open_and_common_baseline(self):
        snap=snapshot_fixture()
        monthly,daily=prepare(snap,date(2009,8,1))
        self.assertEqual(daily[0]["date"],monthly[12]["month"])
        first=daily[0]["assets"]["SPY"]
        self.assertAlmostEqual(first["adjusted_open"],first["open"]*.8)
        self.assertAlmostEqual(first["adjusted_close"]/first["adjusted_open"],first["close"]/first["open"])

    def test_one_missing_session_fails_instead_of_delaying_a_trade(self):
        snap=snapshot_fixture()
        snap["histories"]["VEU"]=[r for r in snap["histories"]["VEU"] if r["date"]!="2008-07-01"]
        with self.assertRaisesRegex(ValueError,"2008-07-01"):
            prepare(snap,date(2009,8,1))

    def test_session_missing_from_every_ticker_is_detected(self):
        snap=snapshot_fixture()
        for t in TICKERS:
            snap["histories"][t]=[r for r in snap["histories"][t] if r["date"]!="2008-07-01"]
        with self.assertRaisesRegex(ValueError,"2008-07-01"):
            prepare(snap,date(2009,8,1))

    def test_invalid_open_duplicate_and_null_prices_rejected(self):
        for value in (0,-1,None,math.nan,math.inf):
            snap=snapshot_fixture()
            snap["histories"]["SPY"][40]["open"]=value
            with self.assertRaises(ValueError):
                prepare(snap,date(2009,8,1))
        snap=snapshot_fixture()
        snap["histories"]["SPY"].insert(50,snap["histories"]["SPY"][49])
        with self.assertRaisesRegex(ValueError,"unique"):
            prepare(snap,date(2009,8,1))

    def test_rejects_stale_snapshot_asof(self):
        with self.assertRaisesRegex(ValueError,"as-of"):
            prepare(snapshot_fixture(),date(2009,9,1))

    def test_current_calendar_month_excluded_from_confirmed_signal(self):
        snap=snapshot_fixture()
        snap["as_of"]="2009-08-20"
        for t in TICKERS:
            snap["histories"][t].append({"date":"2009-08-19","open":9999,"close":9999,"adjusted_close":9999})
        monthly,daily=prepare(snap,date(2009,8,20))
        self.assertEqual(monthly[-1]["month"],"2009-07-31")
        self.assertEqual(daily[-1]["date"],"2009-07-31")

    def test_known_holidays_and_exceptional_closures(self):
        self.assertEqual(sessions(date(2012,10,26),date(2012,10,31)),["2012-10-26","2012-10-31"])
        self.assertNotIn("2018-12-05",sessions(date(2018,12,3),date(2018,12,7)))
        self.assertNotIn("2025-01-09",sessions(date(2025,1,8),date(2025,1,10)))
        self.assertIn("2021-12-31",sessions(date(2021,12,30),date(2022,1,4)))
        self.assertNotIn("2022-06-20",sessions(date(2022,6,17),date(2022,6,21)))
        self.assertNotIn("2024-03-29",sessions(date(2024,3,28),date(2024,4,1)))
        self.assertEqual(sessions(date(2007,1,1),date(2007,1,3)),["2007-01-03"])
        self.assertIn("2026-11-27",sessions(date(2026,11,26),date(2026,11,27)))


if __name__=="__main__":
    unittest.main()
