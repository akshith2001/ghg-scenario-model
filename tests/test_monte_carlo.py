import tempfile
import unittest
from pathlib import Path

from ghg_model.monte_carlo import (
    run_monte_carlo,
    write_histogram_svg,
    write_samples_csv,
)


class MonteCarloTests(unittest.TestCase):
    def test_fixed_seed_is_reproducible(self) -> None:
        first_samples, first_summary = run_monte_carlo(200, seed=7)
        second_samples, second_summary = run_monte_carlo(200, seed=7)
        self.assertEqual(first_samples, second_samples)
        self.assertEqual(first_summary, second_summary)

    def test_summary_percentiles_are_ordered(self) -> None:
        _, summary = run_monte_carlo(500, seed=11)
        self.assertLess(summary.minimum_saving_kg_co2e, summary.p05_saving_kg_co2e)
        self.assertLess(summary.p05_saving_kg_co2e, summary.median_saving_kg_co2e)
        self.assertLess(summary.median_saving_kg_co2e, summary.p95_saving_kg_co2e)
        self.assertLess(summary.p95_saving_kg_co2e, summary.maximum_saving_kg_co2e)

    def test_too_few_samples_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_monte_carlo(99)

    def test_outputs_are_written(self) -> None:
        samples, summary = run_monte_carlo(200, seed=3)
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "samples.csv"
            svg_path = Path(directory) / "histogram.svg"
            write_samples_csv(samples, csv_path)
            write_histogram_svg(samples, summary, svg_path)
            self.assertIn("sample_id", csv_path.read_text(encoding="utf-8"))
            self.assertIn("<svg", svg_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
