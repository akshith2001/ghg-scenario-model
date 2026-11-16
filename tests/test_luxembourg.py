import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ghg_model.entsoe import fill_gaps, parse_timeseries
from ghg_model.hourly_dispatch import (
    SystemParameters,
    accounting_factors,
    cross_evaluate,
    marginal_intensity,
    objective_under,
    solve,
    with_overrides,
)
from ghg_model.foresight import battery_value_retention, derated_parameters, forecast_prices
from ghg_model.hourly_dispatch import import_channels
from ghg_model.luxembourg_case import MarginalCache, monte_carlo
from ghg_model.marginal_emissions import (
    MeritOrderSettings,
    estimate_marginal_factors,
    german_residual_load_index,
    merit_order_marginal,
    synthetic_zone_emissions,
)
from ghg_model.luxembourg_profiles import (
    HourlyProfiles,
    calibration_report,
    load_profiles_csv,
    representative_days,
    synthesise_profiles,
    write_profiles_csv,
)


def _toy_profiles(hours=48, price=80.0, solar_peak=0.8, load=500.0):
    hod = np.arange(hours) % 24
    solar = np.clip(np.sin((hod - 6) / 12 * np.pi), 0, None) * solar_peak
    return HourlyProfiles(
        load_mw=np.full(hours, load),
        solar_cf=solar,
        wind_mw=np.zeros(hours),
        other_domestic_mw=np.zeros(hours),
        price_eur_per_mwh=np.full(hours, price),
        import_avg_g_per_kwh=np.full(hours, 200.0),
        solar_index=solar / solar.max(),
        hour_weights=np.full(hours, 8760 / hours),
    )


class TestProfiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.year = synthesise_profiles()

    def test_annual_totals_match_published_statistics(self):
        report = {r["statistic"]: r for r in calibration_report(self.year)}
        for key in ("Annual consumption (GWh)", "Solar generation (GWh)", "Wind generation (GWh)",
                    "Mean day-ahead price (EUR/MWh)", "Import-weighted average intensity (g/kWh)"):
            self.assertLess(abs(report[key]["relative_error"]), 0.005, key)

    def test_shape_statistics_within_twenty_percent(self):
        for row in calibration_report(self.year):
            if row["statistic"] == "Negative-price hours":
                continue  # documented as the weakest fit
            self.assertLess(abs(row["relative_error"]), 0.2, row["statistic"])

    def test_synthesis_is_reproducible(self):
        again = synthesise_profiles()
        np.testing.assert_allclose(again.load_mw, self.year.load_mw)
        np.testing.assert_allclose(again.price_eur_per_mwh, self.year.price_eur_per_mwh)

    def test_solar_is_zero_at_night(self):
        hod = np.arange(self.year.n_hours) % 24
        self.assertEqual(float(self.year.solar_cf[(hod < 3) | (hod > 21)].max()), 0.0)

    def test_representative_days_preserve_annual_energy(self):
        rep = representative_days(self.year, n_days=24)
        self.assertEqual(rep.n_hours, 24 * 24)
        self.assertAlmostEqual(rep.hour_weights.sum(), 8760, delta=1)
        annual = np.sum(rep.hour_weights * rep.load_mw)
        self.assertAlmostEqual(annual / self.year.load_mw.sum(), 1.0, delta=0.03)

    def test_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv"
            write_profiles_csv(self.year, path)
            back = load_profiles_csv(path, existing_solar_mw=746)
        np.testing.assert_allclose(back.load_mw, self.year.load_mw, rtol=1e-5)
        self.assertEqual(back.metadata["existing_solar_mw"], 746)

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            HourlyProfiles(np.ones(3), np.zeros(3), np.zeros(3), np.zeros(3), np.ones(3), np.ones(2), np.zeros(3))


class TestDispatch(unittest.TestCase):
    def test_energy_balance_holds(self):
        prof = _toy_profiles()
        params = SystemParameters(existing_solar_mw=200, solar_tier_mw=(0.0,), solar_tier_cost_multiplier=(1.0,),
                                  max_battery_mwh=0)
        res = solve(prof, params, "average")
        supplied = res.imports_gwh + np.sum(prof.hour_weights * 200 * prof.solar_cf) / 1000
        demand = np.sum(prof.hour_weights * prof.load_mw) / 1000 + res.exports_gwh + res.curtailment_gwh
        self.assertAlmostEqual(supplied, demand, places=3)

    def test_no_investment_when_solar_is_expensive(self):
        params = SystemParameters(existing_solar_mw=0, solar_lcoe_eur_per_mwh=500, battery_capex_eur_per_kwh=5000)
        res = solve(_toy_profiles(), params, "average")
        self.assertAlmostEqual(res.new_solar_mw, 0.0, places=3)
        self.assertAlmostEqual(res.battery_mwh, 0.0, places=3)

    def test_cheap_solar_is_built(self):
        params = SystemParameters(existing_solar_mw=0, solar_lcoe_eur_per_mwh=5, battery_capex_eur_per_kwh=5000)
        self.assertGreater(solve(_toy_profiles(), params, "average").new_solar_mw, 100)

    def test_marginal_rule_credits_exports(self):
        prof = _toy_profiles()
        params = SystemParameters(existing_solar_mw=0, marginal_solar_dip=0.0)
        f, h = accounting_factors(prof, params, "marginal")
        np.testing.assert_allclose(f[0], h)
        f_avg, h_avg = accounting_factors(prof, params, "average")
        self.assertEqual(float(h_avg.max()), 0.0)

    def test_marginal_dip_lowers_midday_intensity(self):
        prof = _toy_profiles()
        m = marginal_intensity(prof, SystemParameters(marginal_intensity_g_per_kwh=500, marginal_solar_dip=0.5))
        self.assertAlmostEqual(float(m.max()), 500.0)
        self.assertAlmostEqual(float(m.min()), 250.0)

    def test_fixed_capacities_are_respected(self):
        res = solve(_toy_profiles(), SystemParameters(existing_solar_mw=0), "marginal",
                    fixed_solar_mw=123.0, fixed_battery_mwh=45.0)
        self.assertAlmostEqual(res.new_solar_mw, 123.0, places=4)
        self.assertAlmostEqual(res.battery_mwh, 45.0, places=4)

    def test_unknown_rule_rejected(self):
        with self.assertRaises(ValueError):
            accounting_factors(_toy_profiles(), SystemParameters(), "consumption")

    def test_regret_is_non_negative(self):
        # The marginal-rule plan is optimal under the marginal objective, so the
        # average-rule plan can only tie or do worse when judged that way.
        params = SystemParameters(existing_solar_mw=0, solar_lcoe_eur_per_mwh=70, carbon_price_eur_per_t=120)
        ce = cross_evaluate(_toy_profiles(), params)
        self.assertGreaterEqual(ce["regret_eur_per_year"], -1e-3)
        judged = ce["judged_under_marginal"]["marginal"]
        self.assertAlmostEqual(objective_under(judged, params, "marginal"),
                               judged.objective_eur, delta=1.0)

    def test_zero_carbon_price_makes_rules_agree(self):
        params = SystemParameters(existing_solar_mw=0, carbon_price_eur_per_t=0.0)
        ce = cross_evaluate(_toy_profiles(), params)
        self.assertAlmostEqual(ce["regret_eur_per_year"], 0.0, delta=1.0)


class TestModelExtensions(unittest.TestCase):
    def test_solar_tiers_give_interior_solution(self):
        # Tier 1 is cheap, tier 2 is prohibitively expensive: build exactly tier 1.
        params = SystemParameters(existing_solar_mw=0, solar_tier_mw=(100.0, 100.0),
                                  solar_tier_cost_multiplier=(0.1, 50.0), battery_capex_eur_per_kwh=5000)
        res = solve(_toy_profiles(), params, "average")
        self.assertAlmostEqual(res.new_solar_by_tier_mw[0], 100.0, places=3)
        self.assertAlmostEqual(res.new_solar_by_tier_mw[1], 0.0, places=3)

    def test_fixed_solar_fills_cheapest_tier_first(self):
        params = SystemParameters(existing_solar_mw=0)
        res = solve(_toy_profiles(), params, "average", fixed_solar_mw=700.0, fixed_battery_mwh=0.0)
        self.assertEqual([round(x) for x in res.new_solar_by_tier_mw], [500, 200, 0])

    def test_cycle_limit_caps_daily_throughput(self):
        prof = _toy_profiles()
        prof.price_eur_per_mwh[:] = np.where(np.arange(48) % 2, 200.0, 0.0)  # extreme hourly spread
        params = SystemParameters(existing_solar_mw=0, solar_tier_mw=(0.0,), solar_tier_cost_multiplier=(1.0,),
                                  max_cycles_per_day=1.0, battery_capex_eur_per_kwh=1)
        res = solve(prof, params, "average", fixed_battery_mwh=100.0)
        self.assertLessEqual(res.battery_cycles_per_year, 365.0 + 1e-6)

    def test_wear_cost_reduces_battery_use(self):
        prof = _toy_profiles()
        prof.price_eur_per_mwh[:] = 80 + 30 * np.sin(np.arange(48) / 24 * 2 * np.pi)
        cheap = SystemParameters(existing_solar_mw=0, battery_cycle_life=1e9)
        costly = SystemParameters(existing_solar_mw=0, battery_cycle_life=500)
        a = solve(prof, cheap, "average", fixed_solar_mw=0, fixed_battery_mwh=200)
        b = solve(prof, costly, "average", fixed_solar_mw=0, fixed_battery_mwh=200)
        self.assertLess(b.battery_cycles_per_year, a.battery_cycles_per_year)

    def test_flow_shares_capacity_is_tightest_border(self):
        prof = representative_days(synthesise_profiles(), n_days=4)
        params = SystemParameters(border_mode="flow_shares",
                                  border_import_cap_mw={"DE": 1000.0, "BE": 100.0, "FR": 1000.0})
        (_, _, cap), = import_channels(prof, params)
        self.assertAlmostEqual(cap, 100.0 / prof.border_shares["BE"], places=6)

    def test_free_borders_respect_caps_and_favour_clean_imports(self):
        prof = representative_days(synthesise_profiles(), n_days=4)
        params = SystemParameters(border_mode="free", carbon_price_eur_per_t=150)
        res = solve(prof, params, "average", fixed_solar_mw=0, fixed_battery_mwh=0)
        hours = np.sum(prof.hour_weights)
        for zone, gwh in res.imports_by_channel_gwh.items():
            self.assertLessEqual(gwh * 1000, params.border_import_cap_mw[zone] * hours + 1e-3)
        self.assertGreater(res.imports_by_channel_gwh["FR"], res.imports_by_channel_gwh["DE"])

    def test_profile_marginal_source_requires_series(self):
        with self.assertRaises(ValueError):
            solve(_toy_profiles(), SystemParameters(marginal_source="profile"), "marginal")


class TestMarginalEmissions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.year = synthesise_profiles()
        cls.residual = german_residual_load_index(cls.year)

    def test_marginal_rises_with_residual_load_band(self):
        m = merit_order_marginal(self.residual)
        low = m[self.residual < np.quantile(self.residual, 0.05)].mean()
        mid = m[(self.residual > np.quantile(self.residual, 0.4)) & (self.residual < np.quantile(self.residual, 0.6))].mean()
        self.assertLess(low, mid)

    def test_regimes_differ_in_middle_of_stack(self):
        coal = merit_order_marginal(self.residual, MeritOrderSettings(regime="coal_cheaper"))
        gas = merit_order_marginal(self.residual, MeritOrderSettings(regime="gas_cheaper"))
        self.assertGreater(coal.mean(), gas.mean())

    def test_unknown_regime_rejected(self):
        with self.assertRaises(ValueError):
            merit_order_marginal(self.residual, MeritOrderSettings(regime="nuclear_cheaper"))

    def test_estimator_recovers_known_marginal_factors(self):
        emissions = synthetic_zone_emissions(self.residual)
        est = estimate_marginal_factors(emissions, self.residual)
        truth = merit_order_marginal(self.residual)
        self.assertGreater(np.corrcoef(est["hourly"], truth)[0, 1], 0.8)
        self.assertLess(abs(est["hourly"].mean() / truth.mean() - 1), 0.1)

    def test_estimator_exact_for_linear_system(self):
        demand = np.linspace(100, 200, 500) + np.sin(np.arange(500))
        est = estimate_marginal_factors(400.0 * demand + 7.0, demand, n_bins=4)
        np.testing.assert_allclose(est["slopes"], 400.0, rtol=1e-9)


class TestForesight(unittest.TestCase):
    def test_retention_is_a_share_and_derating_raises_battery_cost(self):
        rep = representative_days(synthesise_profiles(), n_days=4)
        rep = rep.with_marginal(np.full(rep.n_hours, 500.0), "flat")
        params = SystemParameters(marginal_source="profile")
        ret = battery_value_retention(rep, params, solar_mw=0.0, battery_mwh=300.0, seeds=(1,))
        self.assertTrue(0.0 <= ret["retention"] <= 1.0)
        self.assertLessEqual(ret["battery_value_forecast_eur"], ret["battery_value_perfect_eur"] + 1.0)
        derated = derated_parameters(params, 0.5)
        self.assertAlmostEqual(derated.battery_annual_cost_per_mwh(), 2 * params.battery_annual_cost_per_mwh())

    def test_forecast_is_reproducible_and_noisy(self):
        prof = _toy_profiles()
        a, b = forecast_prices(prof, seed=1), forecast_prices(prof, seed=1)
        np.testing.assert_allclose(a, b)
        self.assertGreater(np.abs(a - prof.price_eur_per_mwh).mean(), 0)


class TestCaseStudy(unittest.TestCase):
    def test_monte_carlo_reports_shares_and_drivers(self):
        year = synthesise_profiles()
        rep = representative_days(year, n_days=6)
        cache = MarginalCache(year, rep)
        base = SystemParameters(marginal_source="profile", border_mode="flow_shares")
        mc = monte_carlo(cache, base, n_draws=4, seed=1)
        self.assertEqual(len(mc["records"]), 4)
        self.assertTrue(0.0 <= mc["solar_decision_flip_share"] <= 1.0)
        self.assertIn("carbon_price_eur_per_t", mc["spearman_with_regret"])

    def test_cache_subsets_full_year_series(self):
        year = synthesise_profiles()
        rep = representative_days(year, n_days=3)
        cache = MarginalCache(year, rep)
        prof = cache.profiles(MeritOrderSettings())
        full = merit_order_marginal(cache.residual, MeritOrderSettings())
        np.testing.assert_allclose(prof.marginal_g_per_kwh, full[np.array(rep.metadata["hours"])])


ENTSOE_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <TimeSeries>
    <MktPSRType><psrType>B16</psrType></MktPSRType>
    <Period>
      <timeInterval><start>2025-01-01T00:00Z</start><end>2025-01-01T02:00Z</end></timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><quantity>10</quantity></Point>
      <Point><position>2</position><quantity>20</quantity></Point>
      <Point><position>3</position><quantity>30</quantity></Point>
      <Point><position>4</position><quantity>40</quantity></Point>
      <Point><position>5</position><quantity>100</quantity></Point>
    </Period>
  </TimeSeries>
  <TimeSeries>
    <MktPSRType><psrType>B19</psrType></MktPSRType>
    <Period>
      <timeInterval><start>2025-01-01T00:00Z</start><end>2025-01-01T01:00Z</end></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>999</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""


class TestEntsoeParser(unittest.TestCase):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def test_quarter_hours_are_averaged_and_psr_filtered(self):
        solar = parse_timeseries(ENTSOE_FIXTURE, self.start, 3, psr_type="B16")
        self.assertAlmostEqual(solar[0], 25.0)
        self.assertAlmostEqual(solar[1], 100.0)
        self.assertTrue(np.isnan(solar[2]))
        wind = parse_timeseries(ENTSOE_FIXTURE, self.start, 3, psr_type="B19")
        self.assertAlmostEqual(wind[0], 999.0)

    def test_fill_gaps_interpolates(self):
        np.testing.assert_allclose(fill_gaps(np.array([1.0, np.nan, 3.0, np.nan])), [1, 2, 3, 3])


if __name__ == "__main__":
    unittest.main()
