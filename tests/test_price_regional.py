import numpy as np

from ghg_model import price_feedback as pf
from ghg_model.regional_marginal import NEIGHBOURS, drop_leap_day, regional_estimate


def _year(seed=0, n_days=60):
    rng = np.random.default_rng(seed)
    h = np.arange(n_days * 24)
    solar = np.clip(np.sin((h % 24 - 6) / 12 * np.pi), 0, None) * 30000
    wind = 10000 + 8000 * np.sin(h / 97.0) + rng.normal(0, 1500, len(h))
    load = 55000 + 8000 * np.sin((h % 24 - 9) / 24 * 2 * np.pi)
    return h, load, solar, wind, rng


def test_merit_curve_is_monotone_and_recovers_slope():
    _, load, solar, wind, rng = _year()
    rl = load - solar - wind
    price = 20 + 0.002 * rl + rng.normal(0, 8, len(rl))
    c = pf.merit_curve(rl, price)
    assert np.all(np.diff(c.p_eur) > 0)
    assert abs(c.slope_at(np.median(rl)) - 0.002) < 0.0008


def test_fleet_flattens_prices_and_conserves_energy():
    _, load, solar, wind, rng = _year()
    rl = load - solar - wind
    price = 20 + 0.002 * rl
    c = pf.merit_curve(rl, price)
    delta = pf.fleet_dispatch(rl, c, energy_mwh=20000, power_mw=10000, efficiency=0.9)
    shifted = pf.shifted_prices(price, rl, delta, c)
    assert pf.spread_stats(shifted)["mean_daily_spread_eur"] < pf.spread_stats(price)["mean_daily_spread_eur"]
    # charge exceeds discharge by the losses only
    charge, discharge = delta.clip(min=0).sum(), (-delta).clip(min=0).sum()
    assert discharge <= charge + 1e-6 and discharge >= 0.85 * charge
    assert np.all(pf.fleet_dispatch(rl, c, 0, 0) == 0)


def test_regional_estimate_recovers_cross_border_response():
    h, load, solar, wind, rng = _year(1, 120)
    n = len(h)
    d = {"de_load": load, "de_solar": solar, "de_wind": wind}
    rl_de = load - solar - wind
    nb = {}
    # Germany meets 40% at 800 kg/MWh; Poland 30% at 900; France 30% at 50
    share_ef = {"pl": (0.3, 900), "fr": (0.3, 50)}
    d["de_emis_t"] = 0.4 * rl_de * 0.8 + rng.normal(0, 50, n)
    for c in NEIGHBOURS:
        c_load = 10000 + rng.normal(0, 300, n)
        c_wind = 2000 + 0.3 * (wind - wind.mean()) + rng.normal(0, 300, n)  # correlated weather
        c_rl = c_load - c_wind
        s, ef = share_ef.get(c, (0.0, 400))
        gen = c_rl + s * rl_de
        nb.update({f"{c}_load": c_load, f"{c}_solar": np.zeros(n), f"{c}_wind": c_wind,
                   f"{c}_gen": gen, f"{c}_trade": np.zeros(n),
                   f"{c}_emis_t": (0.5 * c_rl * 0.4 + s * rl_de * ef / 1000) + rng.normal(0, 20, n)})
    _, fit = regional_estimate(d, nb, n_boot=30)
    expected = 0.4 * 800 + 0.3 * 900 + 0.3 * 50
    assert abs(fit["overall_kg_per_mwh"] - expected) < 25
    assert abs(fit["within_hour"]["kg_per_mwh"] - expected) < 25
    assert fit["range_kg_per_mwh"][0] < fit["overall_kg_per_mwh"] < fit["range_kg_per_mwh"][1]
    assert abs(fit["within_hour"]["by_country_kg_per_mwh"]["pl"] - 270) < 20
    assert abs(fit["daily_day_to_day"]["coverage_mw_per_mw"]["pl"] - 0.3) < 0.03


def test_drop_leap_day():
    assert len(drop_leap_day(np.arange(8784))) == 8760
    assert drop_leap_day(np.arange(8784))[1416] == 1440


def load_tests(loader, tests, pattern):
    # Include module-level tests in the project's unittest-based CI.
    import unittest
    tests.addTests(unittest.FunctionTestCase(fn) for name, fn in globals().items()
                   if name.startswith("test_") and callable(fn))
    return tests
