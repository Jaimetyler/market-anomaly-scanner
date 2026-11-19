import unittest
from scanner.paper_replay import replay


def bar(ticker, o, l, h, c):
    return {"T": ticker, "o": o, "l": l, "h": h, "c": c}


class PaperReplayTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"2026-11-11:TEST": {
            "ticker": "TEST", "signal_date": "2026-11-11",
            "side": "short", "entry_below": 10, "stop_above": 12,
            "borrow": "available"}}

    def test_ambiguous_intraday_entry_and_stop_not_filled(self):
        sessions = {"2026-11-12": [bar("TEST", 11, 9, 13, 10)]}
        trade = replay(self.plan, sessions)["trades"][0]
        self.assertEqual(trade["status"], "AMBIGUOUS_ENTRY_OR_STOP")

    def test_borrow_unknown_prevents_trade(self):
        self.plan["2026-11-11:TEST"]["borrow"] = "unknown"
        result = replay(self.plan, {"2026-11-12": [bar("TEST", 11, 9, 11, 10)]})
        self.assertEqual(result["trades"][0]["status"], "BORROW_NOT_VERIFIED")

    def test_stop_gap_uses_open_not_stop(self):
        sessions = {
            "2026-11-12": [bar("TEST", 11, 9, 11, 10)],
            "2026-11-13": [bar("TEST", 14, 13, 15, 14)],
        }
        trade = replay(self.plan, sessions)["trades"][0]
        self.assertEqual(trade["status"], "CLOSED")
        self.assertEqual(trade["exit_reason"], "STOP_GAP")
        self.assertEqual(trade["exit_price"], 14)

    def test_no_future_session(self):
        result = replay(self.plan, {"2026-11-11": [bar("TEST", 10, 9, 12, 11)]})
        self.assertEqual(result["trades"][0]["status"], "AWAITING_NEXT_SESSION")


if __name__ == "__main__":
    unittest.main()
