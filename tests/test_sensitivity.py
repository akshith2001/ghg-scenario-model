import tempfile
import unittest
from pathlib import Path

from ghg_model.sensitivity import (
    analyse_sensitivity,
    write_results_csv,
    write_tornado_svg,
)


class SensitivityTests(unittest.TestCase):
    def test_analysis_returns_ranked_parameter_results(self) -> None:
        results = analyse_sensitivity()
        self.assertEqual(len(results), 8)
        scores = [result.sensitivity_score_pct for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_linear_flow_change_produces_ten_percent_output_change(self) -> None:
        result = next(
            item
            for item in analyse_sensitivity(variation_pct=10)
            if item.parameter == "flow_litres_per_minute"
        )
        self.assertAlmostEqual(result.low_change_pct, -10.0)
        self.assertAlmostEqual(result.high_change_pct, 10.0)

    def test_invalid_variation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analyse_sensitivity(variation_pct=0)

    def test_csv_and_svg_outputs_are_written(self) -> None:
        results = analyse_sensitivity()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "results.csv"
            svg_path = Path(directory) / "chart.svg"
            write_results_csv(results, csv_path)
            write_tornado_svg(results, svg_path)
            self.assertIn("sensitivity_score_pct", csv_path.read_text(encoding="utf-8"))
            self.assertIn("<svg", svg_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
