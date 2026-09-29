import unittest

from scanner.published_faber_gtaa import ASSETS, backtest


def sample(levels, cash=0.01):
    return [{"month": f"2020-{m:02d}-28", "tbill_return": str(cash),
             **{asset: str(series[m - 1]) for asset, series in levels.items()}}
            for m in range(1, 13)]


class FiveAssetTests(unittest.TestCase):
    def test_independent_signals_and_monthly_equal_weight(self):
        levels = {asset: [100 + m for m in range(1, 13)] for asset in ASSETS}
        levels["commodities"] = [200 - m for m in range(1, 13)]
        result = backtest(sample(levels))
        self.assertEqual(result[0]["commodities_next"], "TBILL")
        self.assertTrue(all(result[0][f"{a}_next"] == "ASSET" for a in ASSETS[:-1]))
        self.assertEqual(result[1]["commodities_held"], "TBILL")
        expected = (4 * (111 / 110 - 1) + 0.01) / 5
        benchmark = (4 * (111 / 110 - 1) + (189 / 190 - 1)) / 5
        self.assertAlmostEqual(result[1]["portfolio_equity"], 100000 * (1 + expected))
        self.assertAlmostEqual(result[1]["equal_weight_equity"], 100000 * (1 + benchmark))

    def test_month_end_change_affects_next_month_only(self):
        levels = {asset: [100 + m for m in range(1, 13)] for asset in ASSETS}
        levels["us_stocks"] = list(range(101, 111)) + [80, 40]
        result = backtest(sample(levels))
        self.assertEqual(result[1]["us_stocks_held"], "ASSET")
        self.assertEqual(result[1]["us_stocks_next"], "TBILL")
        self.assertEqual(result[2]["us_stocks_held"], "TBILL")
        expected = (0.01 + 4 * (112 / 111 - 1)) / 5
        self.assertAlmostEqual(result[2]["portfolio_return"], expected)

    def test_rejects_missing_month_and_bad_level(self):
        levels = {asset: [100 + m for m in range(1, 13)] for asset in ASSETS}
        rows = sample(levels)
        with self.assertRaises(ValueError):
            backtest(rows[:4] + rows[5:])
        rows[4]["real_estate"] = "nan"
        with self.assertRaises(ValueError):
            backtest(rows)


if __name__ == "__main__":
    unittest.main()
