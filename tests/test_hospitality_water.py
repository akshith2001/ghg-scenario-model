import unittest

from ghg_model.hospitality_water import WaterCaseInputs, estimate_water_case


class HospitalityWaterCaseTests(unittest.TestCase):
    def test_default_water_volume_and_savings(self) -> None:
        result = estimate_water_case(WaterCaseInputs())
        self.assertAlmostEqual(result.annual_water_m3, 810.0)
        self.assertAlmostEqual(result.annual_water_saved_m3, 607.5)
        self.assertAlmostEqual(result.water_after_intervention_m3, 202.5)

    def test_emissions_are_reduced_in_same_proportion(self) -> None:
        result = estimate_water_case(WaterCaseInputs(avoidable_use_pct=60))
        self.assertAlmostEqual(
            result.emissions_after_intervention_kg_co2e,
            result.baseline_emissions_kg_co2e * 0.4,
        )

    def test_invalid_avoidable_percentage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_water_case(WaterCaseInputs(avoidable_use_pct=101))

    def test_zero_flow_produces_zero_result(self) -> None:
        result = estimate_water_case(WaterCaseInputs(flow_litres_per_minute=0))
        self.assertEqual(result.annual_water_m3, 0)
        self.assertEqual(result.baseline_emissions_kg_co2e, 0)


if __name__ == "__main__":
    unittest.main()
