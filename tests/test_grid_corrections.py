import unittest
import numpy as np
from ghg_model import synthetic_grid as sg


class CorrectionTests(unittest.TestCase):
    def test_missing_voltages_do_not_hide_invalid_loads(self):
        for data in (np.full((1, 192), np.nan), np.empty((0, 192)), np.zeros((1, 8))):
            with self.assertRaises(ValueError):
                sg.pf_residual(data, None)

    def test_nonconvergence_rejected(self):
        with self.assertRaises(RuntimeError):
            sg.load_flow(np.full((1, 8), .3), iters=1)

    def test_import_matches_independent_resistive_loss_balance(self):
        loads = np.full((1, 8), .3)
        v, currents, converged = sg._sweep(loads, sg.TAN_PHI, 60)
        self.assertTrue(converged)
        losses = (np.abs(currents[:, 1:])**2 * sg.Z[1:].real).sum(1)
        expected = (loads.sum(1) + losses) * sg.KWH_PER_PU_HOUR * 24
        actual = sg.feeder_import_kwh(loads, t=1).sum(1)
        np.testing.assert_allclose(actual, expected, rtol=1e-8)
        self.assertGreater(actual[0], loads.sum()*sg.KWH_PER_PU_HOUR*24)

    def test_daily_and_constant_hourly_profiles_agree(self):
        daily = np.full((2, 8), .3)
        hourly = np.repeat(daily, 24, axis=1)
        np.testing.assert_allclose(sg.daily_emissions_kg(daily, t=1), sg.daily_emissions_kg(hourly))
        self.assertEqual(sg.emissions_error(daily, daily, t=1), 0)
        self.assertGreater(sg.electricity_activity(daily, t=1).activity_value, 0)

    def test_daily_data_reject_hourly_factors(self):
        with self.assertRaises(ValueError):
            sg.daily_emissions_kg(np.ones((1, 8))*.1, time_of_use=True, t=1)

    def test_reported_voltages_require_exact_shape(self):
        with self.assertRaises(ValueError):
            sg.pf_residual(np.ones((1,192))*.1, np.zeros((1,1)))

    def test_overlapping_windows_have_no_paired_intervals(self):
        rows = [dict(seed=i, arm='GAUSS', w1_excess=float(i)) for i in range(5)]
        result = sg.summarise(rows, metrics=['w1_excess'], arms=['GAUSS'])
        self.assertIn('descriptive', result)
        self.assertNotIn('95% t-interval', result)

    def test_zero_reference_error_is_explicit(self):
        with self.assertRaises(ValueError):
            sg.emissions_error(np.zeros((1,192)), np.zeros((1,192)))
