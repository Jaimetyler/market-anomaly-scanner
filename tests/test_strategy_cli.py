import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from scanner.strategy_cli import main


class StrategyCliTests(unittest.TestCase):
    def test_catalog_and_single_asset_dispatch(self):
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            main(["list"])
        self.assertIn("faber-10month", text.getvalue())
        self.assertIn("faber-gtaa5", text.getvalue())
        self.assertIn("experimental", text.getvalue())
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder) / "input.csv", Path(folder) / "output.csv"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["month", "total_return_index", "tbill_return"])
                writer.writeheader()
                writer.writerows({"month": f"2020-{m:02d}-28", "total_return_index": 100 + m,
                                  "tbill_return": 0.01} for m in range(1, 12))
            with contextlib.redirect_stdout(io.StringIO()):
                main(["run", "faber-10month", str(source), str(destination)])
            with destination.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["held_this_month"], "ASSET")

    def test_five_asset_dispatch(self):
        from scanner.published_faber_gtaa import ASSETS
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder) / "input.csv", Path(folder) / "output.csv"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["month", *ASSETS, "tbill_return"])
                writer.writeheader()
                writer.writerows({"month": f"2020-{m:02d}-28", **{a: 100 + m for a in ASSETS},
                                  "tbill_return": 0.01} for m in range(1, 12))
            with contextlib.redirect_stdout(io.StringIO()):
                main(["run", "faber-gtaa5", str(source), str(destination)])
            with destination.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["us_stocks_held"], "ASSET")


if __name__ == "__main__":
    unittest.main()
