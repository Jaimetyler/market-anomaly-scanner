import unittest

from scanner.paper_portfolio import simulate_portfolio


def bar(ticker, o, l, h, c):
    return {"T": ticker, "o": o, "l": l, "h": h, "c": c}


def plan(ticker, entry=10, stop=12, borrow="available"):
    return {"ticker": ticker, "signal_date": "2026-11-11",
            "side": "short", "entry_below": entry,
            "stop_above": stop, "borrow": borrow}


class PortfolioTests(unittest.TestCase):
    def test_exposure_cap_restricts_simultaneous_shorts(self):
        plans = {f"2026-11-11:{t}": plan(t) for t in ("AAA", "BBB")}
        sessions = {"2026-11-11": [],
                    "2026-11-12": [bar(t, 11, 9, 11, 10) for t in ("AAA", "BBB")]}
        report = simulate_portfolio(plans, sessions, position_limit=.1, gross_limit=.1)
        self.assertEqual(sum(r.get("shares", 0) for r in report["trades"]), 1000)
        self.assertEqual(report["equity_curve"][-1]["gross_exposure"], 10000)

    def test_stop_gap_realized_and_drawdown(self):
        plans = {"2026-11-11:AAA": plan("AAA")}
        sessions = {"2026-11-11": [], "2026-11-12": [bar("AAA", 11, 9, 11, 10)],
                    "2026-11-13": [bar("AAA", 14, 13, 15, 14)]}
        report = simulate_portfolio(plans, sessions)
        self.assertEqual(report["trades"][0]["exit_reason"], "STOP_GAP")
        self.assertEqual(report["realized_pnl"], -1000)
        self.assertEqual(report["ending_equity"], 99000)
        self.assertLess(report["max_drawdown_pct"], 0)

    def test_unknown_borrow_never_opens(self):
        plans = {"2026-11-11:AAA": plan("AAA", borrow="unknown")}
        sessions = {"2026-11-11": [], "2026-11-12": [bar("AAA", 11, 9, 11, 10)]}
        report = simulate_portfolio(plans, sessions)
        self.assertEqual(report["trades"][0]["status"], "BORROW_NOT_VERIFIED")
        self.assertEqual(report["ending_equity"], 100000)

    def test_ambiguous_bar_not_filled(self):
        plans = {"2026-11-11:AAA": plan("AAA")}
        sessions = {"2026-11-11": [], "2026-11-12": [bar("AAA", 11, 9, 13, 10)]}
        report = simulate_portfolio(plans, sessions)
        self.assertEqual(report["trades"][0]["status"], "AMBIGUOUS_ENTRY_OR_STOP")


if __name__ == "__main__":
    unittest.main()
