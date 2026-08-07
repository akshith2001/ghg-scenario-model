import unittest

from ghg_model.model import Activity, apply_scenario, calculate_emissions


def sample_activity() -> Activity:
    return Activity(
        activity_id="E001",
        sector="electricity",
        activity_name="test electricity",
        activity_value=100.0,
        activity_unit="kWh",
        emission_factor=0.2,
        factor_unit="kg CO2e/kWh",
        factor_year="illustrative",
        factor_source="test",
        activity_uncertainty_pct=3.0,
        factor_uncertainty_pct=4.0,
    )


class ModelTests(unittest.TestCase):
    def test_baseline_calculation_and_uncertainty(self) -> None:
        result = calculate_emissions(sample_activity())
        self.assertAlmostEqual(result.emissions_kg_co2e, 20.0)
        self.assertAlmostEqual(result.uncertainty_kg_co2e, 1.0)
        self.assertAlmostEqual(result.lower_kg_co2e, 19.0)
        self.assertAlmostEqual(result.upper_kg_co2e, 21.0)

    def test_scenario_changes_activity_and_factor(self) -> None:
        result = apply_scenario(
            sample_activity(),
            "transition",
            {"electricity": {"activity_change_pct": -10, "factor_change_pct": -50}},
        )
        self.assertAlmostEqual(result.activity_value, 90.0)
        self.assertAlmostEqual(result.emission_factor, 0.1)
        self.assertAlmostEqual(result.emissions_kg_co2e, 9.0)

    def test_unmatched_sector_is_unchanged(self) -> None:
        result = apply_scenario(
            sample_activity(), "other", {"transport": {"activity_change_pct": -20}}
        )
        self.assertAlmostEqual(result.emissions_kg_co2e, 20.0)

    def test_negative_resulting_activity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_scenario(
                sample_activity(), "invalid", {"electricity": {"activity_change_pct": -101}}
            )


if __name__ == "__main__":
    unittest.main()
