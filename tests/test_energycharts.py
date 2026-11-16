import tempfile
import unittest
from pathlib import Path

import numpy as np

from ghg_model.energycharts import COLUMNS, build_table, ilr_specific_yield_cf, profiles_from_csv, write_table


def _fake_table(seed=0):
    rng = np.random.default_rng(seed)
    h = np.arange(8760)
    hod = h % 24
    solar = np.clip(np.sin((hod - 6) / 12 * np.pi), 0, None) * 300
    de_res = 40000 + 10000 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 2000, 8760)
    return {
        "lu_load": 550 + 100 * np.sin(h / 24 * 2 * np.pi), "lu_solar": solar, "lu_wind": np.full(8760, 50.0),
        "lu_other": np.full(8760, 50.0), "price": 90 - 0.2 * solar, "flow_de": np.full(8760, 300.0),
        "flow_be": np.full(8760, 100.0), "de_load": de_res + 20000, "de_solar": np.full(8760, 8000.0),
        "de_wind": np.full(8760, 12000.0), "de_emis_t": 0.5 * de_res, "int_de": np.full(8760, 300.0),
        "int_fr": np.full(8760, 15.0), "int_be": np.full(8760, 100.0),
    }


class TestEnergyCharts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ec.csv"
        write_table(_fake_table(), self.path, 2025)

    def tearDown(self):
        self.tmp.cleanup()

    def test_profiles_built_with_measured_shapes(self):
        prof = profiles_from_csv(self.path)
        self.assertEqual(prof.n_hours, 8760)
        self.assertAlmostEqual(prof.border_shares["DE"], 0.75, places=6)
        np.testing.assert_allclose(prof.import_avg_g_per_kwh, 0.75 * 300 + 0.25 * 100)
        self.assertAlmostEqual(prof.solar_cf.mean(), ilr_specific_yield_cf(), places=4)
        self.assertIsNotNone(prof.zone_residual_mw)

    def test_estimated_marginal_recovers_linear_slope(self):
        # emissions = 0.5 t per MW of residual load -> 500 kg/MWh everywhere
        prof = profiles_from_csv(self.path)
        np.testing.assert_allclose(prof.marginal_g_per_kwh, 500.0, rtol=1e-3)  # CSV rounds to 0.1

    def test_existing_solar_reproduces_metered_output(self):
        prof = profiles_from_csv(self.path)
        metered = _fake_table()["lu_solar"].sum()
        self.assertAlmostEqual((prof.solar_cf * prof.metadata["existing_solar_mw"]).sum(), metered, delta=metered * 1e-3)

    def test_build_table_averages_quarter_hours(self):
        def doc(values):
            return {"data": [{"timestamp": f"2025-01-01T00:{m:02d}:00+00:00", "values": v}
                             for m, v in zip((0, 15, 30, 45), values)]}
        power = doc([{"load": x, "solar": 0, "wind_onshore": 0, "fossil_gas": x} for x in (10, 20, 30, 40)])
        raw = {k: power for k in ("lu", "de", "fr", "be")}
        raw["cbpf"] = doc([{"germany": 0.1, "belgium": 0.0}] * 4)
        raw["price"] = doc([{"day_ahead_price": x} for x in (10, 20, 30, 40)])
        table = build_table(raw)
        self.assertAlmostEqual(table["lu_load"][0], 25.0)
        self.assertAlmostEqual(table["flow_de"][0], 100.0)
        self.assertAlmostEqual(table["int_de"][0], 370.0)
        self.assertEqual(set(table), set(COLUMNS))


class TestMarginalMethods(unittest.TestCase):
    def test_dispatchable_method_scales_by_thermal_share(self):
        # Emissions follow dispatchable output at 500 kg/MWh; half of each residual-load change
        # is met by trade. German-plants regression then sees ~250, dispatchable ~500.
        from ghg_model.energycharts import estimate_marginal
        rng = np.random.default_rng(1)
        n = 8760
        resid = 40000 + 8000 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(0, 1500, n)
        trade = 0.5 * (resid - resid.mean())
        disp = resid - trade
        d = {"de_load": resid + 20000, "de_solar": np.full(n, 8000.0), "de_wind": np.full(n, 12000.0),
             "de_emis_t": 0.5 * disp, "de_trade": trade, "de_pumped": np.zeros(n)}
        low, _ = estimate_marginal(d, method="german_plants")
        high, _ = estimate_marginal(d, method="dispatchable")
        self.assertAlmostEqual(float(low.mean()), 250.0, delta=5)
        self.assertAlmostEqual(float(high.mean()), 500.0, delta=5)

    def test_dispatchable_needs_trade_columns(self):
        from ghg_model.energycharts import estimate_marginal
        with self.assertRaises(ValueError):
            estimate_marginal({k: np.ones(10) for k in ("de_load", "de_solar", "de_wind", "de_emis_t")},
                              method="dispatchable")


class TestTouBridge(unittest.TestCase):
    def test_measured_shape_feeds_synthetic_grid(self):
        from ghg_model import synthetic_grid as sg
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ec.csv"
            write_table(_fake_table(), path, 2025)
            from ghg_model.energycharts import tou_shape
            shape = tou_shape(path, kind="average")
        self.assertEqual(shape.shape, (24,))
        self.assertAlmostEqual(shape.mean(), 1.0, places=9)
        days = sg.simulate(5, seed=0)
        flat = sg.daily_emissions_kg(days)
        shaped = sg.daily_emissions_kg(days, shape=np.ones(24))
        np.testing.assert_allclose(flat, shaped)
        with self.assertRaises(ValueError):
            sg.daily_emissions_kg(days, shape=np.ones(12))


if __name__ == "__main__":
    unittest.main()
