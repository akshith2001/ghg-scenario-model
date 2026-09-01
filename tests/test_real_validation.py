import csv
import tempfile
import unittest
from pathlib import Path

from ghg_model.real_validation import load_egrid_csv, run_validation


class RealDataValidationTests(unittest.TestCase):
    def _fixture(self, directory: str, count: int = 75) -> Path:
        path = Path(directory) / "plants.csv"
        fields = ["YEAR", "ORISPL", "PNAME", "PSTATABB", "PLPRMFL", "PLHTIAN", "PLNGENAN", "NAMEPCAP", "CAPFAC", "PLCO2AN"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(1, count + 1):
                heat = 10_000.0 + index * 1_000.0
                writer.writerow(
                    {
                        "YEAR": 2018,
                        "ORISPL": index,
                        "PNAME": f"Plant {index}",
                        "PSTATABB": "XX",
                        "PLPRMFL": "NG",
                        "PLHTIAN": heat,
                        "PLNGENAN": heat / 8.0,
                        "NAMEPCAP": 10.0 + index,
                        "CAPFAC": 0.5,
                        "PLCO2AN": heat * 53.06 / 907.18474,
                    }
                )
        return path

    def test_validation_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._fixture(directory)
            first, _ = run_validation(path)
            second, _ = run_validation(path)
            self.assertEqual(first, second)

    def test_external_factor_recovers_physical_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, _ = run_validation(self._fixture(directory))
            self.assertLess(result.external_factor_physics.rmse_short_tons, 1e-9)

    def test_missing_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("YEAR,ORISPL\n2018,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_egrid_csv(path)

    def test_target_derived_rate_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._fixture(directory)
            content = path.read_text(encoding="utf-8")
            path.write_text(content.replace("PLCO2AN", "PLCO2AN,PLCO2RA", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target-derived"):
                load_egrid_csv(path)


if __name__ == "__main__":
    unittest.main()
