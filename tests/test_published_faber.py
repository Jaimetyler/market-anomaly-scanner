import unittest

from scanner.published_faber import backtest


class FaberReferenceTests(unittest.TestCase):
    def test_signal_is_applied_to_following_month(self):
        rows = [{"month": f"2020-{m:02d}-28", "total_return_index": str(100 + m),
                 "tbill_return": "0.01"} for m in range(1, 13)]
        result = backtest(rows)
        self.assertEqual(result[0]["held_this_month"], "WARMUP")
        self.assertEqual(result[0]["next_month_allocation"], "ASSET")
        self.assertEqual(result[1]["held_this_month"], "ASSET")
        self.assertAlmostEqual(result[1]["equity"], 100000 * 111 / 110)

    def test_cash_uses_months_return_after_prior_signal(self):
        rows = [{"month": f"2020-{m:02d}-28", "total_return_index": str(200 - m),
                 "tbill_return": "0.01"} for m in range(1, 13)]
        result = backtest(rows)
        self.assertEqual(result[0]["next_month_allocation"], "TBILL")
        self.assertEqual(result[1]["held_this_month"], "TBILL")
        self.assertAlmostEqual(result[1]["equity"], 101000)

    def test_rejects_missing_month(self):
        rows = [{"month": f"2020-{m:02d}-28", "total_return_index": "100",
                 "tbill_return": "0"} for m in range(1, 13) if m != 5]
        with self.assertRaises(ValueError):
            backtest(rows)

    def test_switch_to_cash_takes_effect_next_month(self):
        levels = [100 + m for m in range(1, 11)] + [80, 40]
        rows = [{"month": f"2020-{m:02d}-28", "total_return_index": str(level),
                 "tbill_return": "0.01"} for m, level in enumerate(levels, 1)]
        result = backtest(rows)
        self.assertEqual(result[1]["held_this_month"], "ASSET")
        self.assertEqual(result[1]["next_month_allocation"], "TBILL")
        self.assertAlmostEqual(result[1]["equity"], 100000 * 80 / 110)
        self.assertEqual(result[2]["held_this_month"], "TBILL")
        self.assertAlmostEqual(result[2]["equity"], 100000 * 80 / 110 * 1.01)


if __name__ == "__main__":
    unittest.main()
