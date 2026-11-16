"""How much battery value survives imperfect price foresight?

The planning LP sees every hourly price in advance, which overstates what a
battery operator earns. This module re-operates a fixed plan on *forecast*
prices (actual price plus autocorrelated error) and settles at *actual*
prices. The retained share of battery value is

    retention = (cost without battery - cost with battery on forecasts)
                / (cost without battery - cost with battery, perfect foresight)

A retention of 0.7 means a real operator captures about 70% of the
arbitrage and carbon value the LP assumes. Planning can then scale battery
cost by 1 / retention (``battery_cost_multiplier``) so storage is sized
against value that is actually reachable.

The forecast-error level is an assumption (day-ahead price forecasts in
liquid European markets typically miss by roughly 10-20% of the mean price).
"""

from __future__ import annotations

import numpy as np

from .hourly_dispatch import SystemParameters, solve, with_overrides
from .luxembourg_profiles import HourlyProfiles, _ar1


def forecast_prices(profiles: HourlyProfiles, error_share: float = 0.15, rho: float = 0.8,
                    seed: int = 3) -> np.ndarray:
    """Actual prices plus AR(1) errors scaled to ``error_share`` x mean |price|."""
    rng = np.random.default_rng(seed)
    scale = error_share * float(np.mean(np.abs(profiles.price_eur_per_mwh)))
    return profiles.price_eur_per_mwh + _ar1(rng, profiles.n_hours, rho, scale)


def battery_value_retention(profiles: HourlyProfiles, params: SystemParameters, solar_mw: float,
                            battery_mwh: float, accounting: str = "marginal", error_share: float = 0.15,
                            seeds: tuple = (3, 4, 5)) -> dict:
    """Share of battery value retained when operating on price forecasts."""
    if battery_mwh <= 1e-6:
        return {"retention": 1.0, "battery_value_perfect_eur": 0.0, "battery_value_forecast_eur": 0.0}

    def operating_cost(res):
        emissions = res.emissions_average_t if accounting == "average" else res.emissions_marginal_t
        return res.energy_cost_eur + res.wear_cost_eur + params.carbon_price_eur_per_t * emissions

    none = solve(profiles, params, accounting, fixed_solar_mw=solar_mw, fixed_battery_mwh=0.0)
    perfect = solve(profiles, params, accounting, fixed_solar_mw=solar_mw, fixed_battery_mwh=battery_mwh)
    v_perfect = operating_cost(none) - operating_cost(perfect)
    v_forecast = []
    for seed in seeds:
        fc = forecast_prices(profiles, error_share=error_share, seed=seed)
        res = solve(profiles, params, accounting, fixed_solar_mw=solar_mw, fixed_battery_mwh=battery_mwh,
                    price_override=fc)
        v_forecast.append(operating_cost(none) - operating_cost(res))
    v_fc = float(np.mean(v_forecast))
    retention = v_fc / v_perfect if v_perfect > 1e-6 else 1.0
    return {
        "retention": float(np.clip(retention, 0.0, 1.0)),
        "battery_value_perfect_eur": float(v_perfect),
        "battery_value_forecast_eur": v_fc,
        "error_share": error_share,
        "seeds": list(seeds),
    }


def derated_parameters(params: SystemParameters, retention: float, floor: float = 0.2) -> SystemParameters:
    """Scale battery cost by 1 / retention so planning sizes storage on reachable value."""
    return with_overrides(params, battery_cost_multiplier=params.battery_cost_multiplier / max(retention, floor))
