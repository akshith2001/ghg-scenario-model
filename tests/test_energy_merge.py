import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from ghg_model.hourly_dispatch import SystemParameters, solve, export_price
from ghg_model.luxembourg_profiles import HourlyProfiles
from ghg_model import synthetic_grid as sg, price_feedback as pf


class EnergyMergeTests(unittest.TestCase):
    def test_negative_prices_do_not_pay_for_circular_trade(self):
        zero = np.zeros(24)
        prof = HourlyProfiles(zero, zero, zero, zero, np.full(24, -100.), zero, zero)
        params = SystemParameters(existing_solar_mw=0, solar_tier_mw=(0.,),
                                  solar_tier_cost_multiplier=(1.,), max_battery_mwh=0,
                                  carbon_price_eur_per_t=0)
        for rule in ('average', 'marginal'):
            result = solve(prof, params, rule)
            self.assertAlmostEqual(result.imports_gwh, 0)
            self.assertAlmostEqual(result.exports_gwh, 0)
            self.assertAlmostEqual(result.energy_cost_eur, 0)

    def test_export_spread_on_both_sides_of_zero(self):
        np.testing.assert_allclose(export_price(np.array([-100., 0., 100.]), .9), [-110, 0, 90])

    def test_hourly_shape_keeps_losses_and_rejects_invalid_inputs(self):
        days = np.full((2, 8 * 24), .1)
        np.testing.assert_allclose(sg.daily_emissions_kg(days, shape=np.ones(24)),
                                  sg.feeder_import_kwh(days).sum(1) * sg.DEFAULT_FACTOR)
        for shape in [np.zeros(24), np.full(24, np.nan), -np.ones(24), np.ones(12)]:
            with self.assertRaises(ValueError):
                sg.daily_emissions_kg(days, shape=shape)
        with self.assertRaises(ValueError):
            sg.daily_emissions_kg(np.ones((2, 8)), t=1, shape=np.ones(24))

    def test_failed_fleet_dispatch_is_not_silently_zero(self):
        curve = pf.MeritCurve(np.array([0., 100.]), np.array([10., 20.]))
        with patch.object(pf, 'linprog', return_value=SimpleNamespace(status=2, message='infeasible')):
            with self.assertRaises(RuntimeError):
                pf.fleet_dispatch(np.ones(24), curve, 10, 5)

    def test_unconverged_price_iteration_is_flagged(self):
        d = {k: np.ones(24) for k in ('de_load', 'de_solar', 'de_wind', 'price')}
        z = np.zeros(24)
        prof = HourlyProfiles(z,z,z,z,z,z,z)
        curve = pf.MeritCurve(np.array([0., 100.]), np.array([10., 20.]))
        with patch.object(pf, 'fleet_dispatch', return_value=z), patch.object(
                pf, 'solve', return_value=SimpleNamespace(battery_mwh=1000., new_solar_mw=0.)):
            result = pf.equilibrium_plan(prof, d, SystemParameters(), 'average', 0, iterations=1, curve=curve)
        self.assertFalse(result['converged'])
        self.assertEqual(result['fixed_point_residual_mwh'], 1000.)
