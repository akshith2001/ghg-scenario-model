import unittest
from unittest.mock import patch
import numpy as np
from ghg_model import synthetic_grid as sg


class ReviewedPhysicsTests(unittest.TestCase):
    def test_independent_nodal_balance_and_voltage_limits(self):
        p = np.full((2, 8), 0.3)
        result = sg.physics_diagnostics(p, t=1)
        self.assertLess(result['nodal_balance_max_pu'], 1e-8)
        self.assertGreaterEqual(result['voltage_limit_violation_share'], 0)
        self.assertLessEqual(result['voltage_limit_violation_share'], 1)

    def test_no_reported_voltages_does_not_skip_solver(self):
        with patch.object(sg, 'day_voltages', side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):
                sg.pf_residual(np.ones((1, 8)), None, t=1)

    def test_unconverged_solver_is_rejected(self):
        with self.assertRaises(RuntimeError):
            sg.load_flow(np.full((1, 8), .3), iters=1)

    def test_invalid_injections_rejected(self):
        for p in (np.ones((1, 7)), np.full((1, 8), np.nan), np.empty((0, 8))):
            with self.assertRaises(ValueError):
                sg.load_flow(p)

    def test_import_includes_losses_and_daily_units(self):
        p = np.full((2, 8), .2)
        daily = sg.feeder_import_kwh(p, t=1).sum(1)
        self.assertTrue(np.all(daily > p.sum(1) * 24000))
        hourly = np.repeat(p[:, :, None], 24, axis=2).reshape(2, -1)
        np.testing.assert_allclose(daily, sg.feeder_import_kwh(hourly).sum(1))
        np.testing.assert_allclose(sg.daily_emissions_kg(p, .2, t=1), daily * .2)

    def test_exports_not_credited_and_hourly_factor_guard(self):
        self.assertEqual(sg.feeder_import_kwh(np.full((1, 8), -.01), t=1).sum(), 0)
        with self.assertRaises(ValueError):
            sg.daily_emissions_kg(np.ones((1, 8)), time_of_use=True, t=1)

    def test_rolling_summary_does_not_claim_independent_intervals(self):
        rows = [dict(seed=i, arm='A', w1=float(i)) for i in range(5)]
        report = sg.summarise(rows, metrics=['w1'], arms=['A'], intervals=False)
        self.assertIn('dependent', report)
        self.assertNotIn('95%', report)
