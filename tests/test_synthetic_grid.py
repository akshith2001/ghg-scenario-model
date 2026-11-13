import unittest

import numpy as np

from ghg_model import synthetic_grid as sg
from ghg_model.model import calculate_emissions

try:
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    HAVE_SKLEARN = False


class LoadFlowTests(unittest.TestCase):
    def test_zero_load_is_flat(self) -> None:
        self.assertTrue(np.allclose(sg.load_flow(np.zeros((3, 8))), 1.0))

    def test_voltage_drops_along_feeder(self) -> None:
        v = sg.load_flow(np.full((1, 8), 0.3))[0]
        for i in range(1, sg.N_BUS):
            self.assertLess(v[i], v[sg.PARENT[i]] + 1e-12)

    def test_losses_are_positive_and_small(self) -> None:
        p = np.random.default_rng(1).uniform(0.05, 0.4, (4, 8))
        slack = sg.slack_power(p)
        self.assertTrue(np.all(slack > p.sum(1)))
        self.assertTrue(np.all(slack < p.sum(1) * 1.2))

    def test_pv_raises_voltage(self) -> None:
        base = np.full((1, 8), 0.3)
        pv = base.copy()
        pv[0, 4] = -0.3
        self.assertGreater(sg.load_flow(pv)[0, 5], sg.load_flow(base)[0, 5])


class SimulatorAndModelTests(unittest.TestCase):
    def test_simulator_deterministic(self) -> None:
        a, b = sg.simulate(10, 3), sg.simulate(10, 3)
        self.assertEqual(a.shape, (10, 8 * sg.T))
        self.assertTrue(np.allclose(a, b))

    def test_vae_gradients_match_finite_differences(self) -> None:
        rng = np.random.default_rng(0)
        d = 6
        A = rng.normal(size=(3, 3)) * 0.3
        b = rng.normal(size=3) * 0.1
        for lam in (0.0, 1.0):
            vae = sg.VAE(d, hidden=5, latent=2, seed=1, lam=lam, n_p=3, surrogate=(A, b))
            x = rng.normal(size=(4, d))
            eps = rng.normal(size=(4, 2))
            _, grads = vae.loss_and_grads(x, eps)
            for key in ("W1", "Wm", "Wl", "W2", "W3", "b3"):
                idx = tuple(0 for _ in vae.p[key].shape)
                h = 1e-6
                vae.p[key][idx] += h
                lp, _ = vae.loss_and_grads(x, eps)
                vae.p[key][idx] -= 2 * h
                lm, _ = vae.loss_and_grads(x, eps)
                vae.p[key][idx] += h
                self.assertAlmostEqual((lp - lm) / (2 * h), grads[key][idx], delta=1e-5)

    def test_hard_arm_has_zero_residual_free_arm_does_not(self) -> None:
        tr, va, _ = sg.split(0)
        p, v = sg.run_arm("VAE-HARD", tr, va, 50, 0, epochs=3)
        self.assertEqual(sg.pf_residual(p, v), 0.0)
        p2, v2 = sg.run_arm("VAE-FREE", tr, va, 50, 0, epochs=3)
        self.assertGreater(sg.pf_residual(p2, v2), 0.0)

    def test_metrics_zero_for_identical_sets(self) -> None:
        _, _, te = sg.split(0)
        w1, corr = sg.fidelity(te, te)
        self.assertLess(w1, 1e-9)
        self.assertLess(corr, 1e-9)
        self.assertEqual(sg.emissions_error(te, te), 0.0)

    @unittest.skipUnless(HAVE_SKLEARN, "scikit-learn not installed")
    def test_tstr_runs(self) -> None:
        tr, _, te = sg.split(0)
        self.assertGreater(sg.tstr_mae(tr, te), 0.0)


class GhgBridgeTests(unittest.TestCase):
    def test_tou_shape_has_unit_mean(self) -> None:
        self.assertAlmostEqual(float(sg.TOU_SHAPE.mean()), 1.0, places=9)

    def test_flat_emissions_equal_import_times_factor(self) -> None:
        _, _, te = sg.split(0)
        kwh = sg.feeder_import_kwh(te).sum(axis=1)
        self.assertTrue(np.allclose(sg.daily_emissions_kg(te, 0.2), kwh * 0.2))

    def test_activity_record_feeds_the_ghg_model(self) -> None:
        _, _, te = sg.split(0)
        act = sg.electricity_activity(te)
        result = calculate_emissions(act)
        self.assertGreater(act.activity_value, 0.0)
        self.assertGreater(act.activity_uncertainty_pct, 0.0)
        self.assertAlmostEqual(result.emissions_kg_co2e, act.activity_value * act.emission_factor)
        self.assertEqual(act.factor_year, "illustrative")


class RealBuildingStudyTests(unittest.TestCase):
    DATA = "data/real/bdg2_hospitality_daily.csv"

    def test_loader_keeps_only_complete_days_and_sorts_venues(self) -> None:
        import csv
        import tempfile
        from pathlib import Path
        rows = [("B", "2016-01-01", 24.0), ("A", "2016-01-01", 48.0), ("A", "2016-01-02", 72.0)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                w = csv.writer(handle)
                w.writerow(["venue_id", "utc_date", "electricity_kwh"])
                w.writerows(rows)
            venues, dates, kwh = sg.load_real_days(path)
        self.assertEqual(venues, ["A", "B"])
        self.assertEqual(dates, ["2016-01-01"])
        self.assertTrue(np.allclose(kwh, [[48.0, 24.0]]))

    def test_calibration_hits_target_violation_share(self) -> None:
        _, tr, _, _, scale = sg.real_split(self.DATA)
        self.assertGreater(scale, 0.0)
        self.assertAlmostEqual(sg.violation_days(sg.day_voltages(tr, 1)), 0.10, delta=0.02)

    def test_split_follows_protocol_and_is_chronological(self) -> None:
        venues, tr, va, te, _ = sg.real_split(self.DATA)
        self.assertEqual((tr.shape, va.shape, te.shape), ((282, 8), (50, 8), (313, 8)))
        self.assertEqual(venues[1], sg.REAL_TARGET_VENUE)

    def test_real_arms_run_with_daily_resolution(self) -> None:
        _, tr, va, te, _ = sg.real_split(self.DATA)
        p, v = sg.run_arm("VAE-FREE", tr, va, 30, 0, epochs=3, **sg.REAL_ARM_KWARGS)
        self.assertEqual(p.shape, (30, 8))
        self.assertGreater(sg.pf_residual(p, v, 1), 0.0)
        p2, v2 = sg.run_arm("VAE-HARD", tr, va, 30, 0, epochs=3, **sg.REAL_ARM_KWARGS)
        self.assertEqual(sg.pf_residual(p2, v2, 1), 0.0)


class RollingStudyTests(unittest.TestCase):
    DATA = "data/real/bdg2_hospitality_daily.csv"

    def test_windows_have_protocol_sizes_and_move_forward(self) -> None:
        _, _, kwh = sg.load_real_days(self.DATA)
        starts = []
        for origin in sg.ROLL_ORIGINS:
            tr, va, te, scale = sg.rolling_window(kwh, origin)
            self.assertEqual((tr.shape, va.shape, te.shape), ((240, 8), (40, 8), (120, 8)))
            self.assertGreater(scale, 0.0)
            starts.append(tr[0, 0])
        self.assertGreater(len(set(starts)), 1)

    def test_fidelity_with_reference_survives_a_flat_meter(self) -> None:
        rng = np.random.default_rng(0)
        train = rng.normal(1.0, 0.2, (100, 4))
        real = train[:60].copy()
        real[:, 2] = 0.01                                   # a flat meter in the test window
        syn = rng.normal(1.0, 0.2, (80, 4))
        w1, corr = sg.fidelity(syn, real, ref=train)
        self.assertTrue(np.isfinite(w1) and np.isfinite(corr))
        self.assertLess(w1, 10.0)
        w1_old, _ = sg.fidelity(syn, real)                  # the locked Study 1 and 2 definition
        self.assertGreater(w1_old, 1e3)                     # shows why the Study 3 change was needed

    def test_gmm_arm_runs_and_uses_exact_voltages(self) -> None:
        _, _, kwh = sg.load_real_days(self.DATA)
        tr, va, _, _ = sg.rolling_window(kwh, 0)
        p, v = sg.run_arm("GMM", tr, va, 40, 0, epochs=1, **sg.REAL_ARM_KWARGS)
        self.assertEqual(p.shape, (40, 8))
        self.assertIsNone(v)


if __name__ == "__main__":
    unittest.main()
