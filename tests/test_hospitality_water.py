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
        self.assertAlmostEqual(
            result.total_emissions_after_intervention_kg_co2e,
            result.baseline_total_emissions_kg_co2e * 0.4,
        )

    def test_hot_water_energy_uses_physical_heat_balance(self) -> None:
        inputs = WaterCaseInputs(
            flow_litres_per_minute=1,
            running_hours_per_day=1,
            operating_days_per_year=1,
            avoidable_use_pct=0,
            hot_water_share_pct=100,
            inlet_temperature_c=10,
            outlet_temperature_c=40,
            heater_efficiency_pct=100,
            heating_emission_factor_kg_per_kwh=0,
        )
        result = estimate_water_case(inputs)
        expected_kwh = 60 * 4.186 * 30 / 3600
        self.assertAlmostEqual(result.baseline_heating_energy_kwh, expected_kwh)

    def test_cold_water_case_has_no_heating_energy(self) -> None:
        result = estimate_water_case(WaterCaseInputs(hot_water_share_pct=0))
        self.assertEqual(result.baseline_heating_energy_kwh, 0)
        self.assertEqual(result.baseline_heating_emissions_kg_co2e, 0)

    def test_invalid_avoidable_percentage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_water_case(WaterCaseInputs(avoidable_use_pct=101))

    def test_zero_flow_produces_zero_result(self) -> None:
        result = estimate_water_case(WaterCaseInputs(flow_litres_per_minute=0))
        self.assertEqual(result.annual_water_m3, 0)
        self.assertEqual(result.baseline_emissions_kg_co2e, 0)

    def test_invalid_temperature_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_water_case(
                WaterCaseInputs(inlet_temperature_c=50, outlet_temperature_c=40)
            )


if __name__ == "__main__":
    unittest.main()
