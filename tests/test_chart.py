import csv
import tempfile
import unittest
from pathlib import Path

from ghg_model.chart import create_chart


class ChartTests(unittest.TestCase):
    def test_create_chart_writes_non_empty_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            results_path = folder / "results.csv"
            chart_path = folder / "scenario_totals.png"

            with results_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["scenario", "emissions_kg_co2e"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"scenario": "baseline", "emissions_kg_co2e": "1000"},
                        {"scenario": "transition", "emissions_kg_co2e": "600"},
                    ]
                )

            create_chart(results_path, chart_path)

            self.assertTrue(chart_path.exists())
            self.assertGreater(chart_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
