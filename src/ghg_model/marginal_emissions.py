"""Hourly marginal emission factors for the DE-LU zone.

Two complementary routes replace a single assumed marginal factor:

1. **Merit-order model** (:func:`merit_order_marginal`). The plant that sets
   the margin depends on residual load (demand minus wind and solar) in the
   neighbouring German system. Residual load is ranked into percentiles and
   mapped onto a stylised supply stack whose order depends on the fuel-price
   regime: when gas is cheaper than hard coal, combined-cycle gas sits below
   coal in the stack; otherwise above it.

2. **Statistical estimator** (:func:`estimate_marginal_factors`). Following
   the regression approach of Hawkes (2010, Energy Policy 38, 5977-5987),
   the marginal factor is the slope of hour-to-hour changes in system
   emissions against changes in system demand, estimated within bins of
   demand level. Given measured hourly emissions and load (for example German
   generation by type from ENTSO-E multiplied by technology factors), it
   estimates the marginal series from data instead of assuming it.

Technology intensities are typical direct (combustion) emissions in kg/MWh.
They and the stack shares are assumptions, varied in the Monte Carlo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .luxembourg_profiles import HourlyProfiles, _ar1, _calendar

# Typical direct-emission intensities (kg CO2/MWh electricity); assumptions.
TECH_INTENSITY = {
    "renewable surplus": 0.0,
    "lignite": 1100.0,
    "hard coal": 850.0,
    "CCGT": 370.0,
    "OCGT": 550.0,
}

# (technology, upper residual-load percentile) in merit order.
STACKS = {
    "gas_cheaper": [("renewable surplus", 0.08), ("lignite", 0.35), ("CCGT", 0.70),
                    ("hard coal", 0.92), ("OCGT", 1.00)],
    "coal_cheaper": [("renewable surplus", 0.08), ("lignite", 0.35), ("hard coal", 0.62),
                     ("CCGT", 0.92), ("OCGT", 1.00)],
}


@dataclass(frozen=True)
class MeritOrderSettings:
    regime: str = "coal_cheaper"
    blend_width: float = 0.04          # percentile width over which adjacent techs blend
    lignite_must_run: float = 0.5      # share of lignite hours where lignite, not the next unit, sets the margin
    intensity_scale: float = 1.0       # scales every fossil intensity (fuel quality, part-load efficiency)


def german_residual_load_index(profiles: HourlyProfiles, seed: int = 11) -> np.ndarray:
    """Synthetic German residual load (0-1 scale), correlated with Luxembourg weather.

    Germany shares Luxembourg's weather and daily rhythm, so its load, solar and
    wind are built from the Luxembourg shapes plus independent noise. German
    solar and wind are scaled so wind plus solar cover about 45% of annual load,
    close to recent German shares.
    """
    rng = np.random.default_rng(seed)
    n = profiles.n_hours
    load = profiles.load_mw / profiles.load_mw.mean()
    load = load * np.exp(_ar1(rng, n, 0.9, 0.03))
    solar = np.clip(profiles.solar_index * np.exp(_ar1(rng, n, 0.8, 0.15)), 0, None)
    wind = profiles.wind_mw / max(profiles.wind_mw.mean(), 1e-9)
    wind = wind * np.exp(_ar1(rng, n, 0.95, 0.25))
    solar = solar / solar.mean() * 0.15
    wind = wind / wind.mean() * 0.30
    residual = load - solar - wind
    return residual


def _percentile_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return (order + 0.5) / len(x)


def merit_order_marginal(residual: np.ndarray, settings: MeritOrderSettings = MeritOrderSettings()) -> np.ndarray:
    """Map residual load onto the merit order and return marginal intensity (kg/MWh)."""
    if settings.regime not in STACKS:
        raise ValueError(f"unknown regime {settings.regime!r}; use one of {sorted(STACKS)}")
    stack = STACKS[settings.regime]
    q = _percentile_rank(residual)
    intensities = np.array([TECH_INTENSITY[name] for name, _ in stack], dtype=float)
    intensities[1:] *= settings.intensity_scale
    uppers = np.array([u for _, u in stack])
    # Inside the lignite band, part of the time the next unit is already marginal.
    lig = [i for i, (name, _) in enumerate(stack) if name == "lignite"][0]
    blended = intensities.copy()
    blended[lig] = settings.lignite_must_run * intensities[lig] + (1 - settings.lignite_must_run) * intensities[lig + 1]
    # Smooth step function: logistic blend at each boundary.
    out = np.full_like(q, blended[0])
    w = max(settings.blend_width, 1e-6)
    for i in range(1, len(stack)):
        step = 1.0 / (1.0 + np.exp(-(q - uppers[i - 1]) / (w / 4)))
        out = out + step * (blended[i] - blended[i - 1])
    return out


def synthetic_zone_emissions(residual: np.ndarray, settings: MeritOrderSettings = MeritOrderSettings(),
                             base_emissions: float = 0.0) -> np.ndarray:
    """Total emissions of a zone whose marginal plant follows the merit order.

    Emissions are the integral of marginal intensity over residual load, so the
    true marginal factor at each hour is known; used to validate the estimator.
    """
    order = np.argsort(residual)
    r_sorted = residual[order]
    m_sorted = merit_order_marginal(residual, settings)[order]
    increments = np.concatenate([[0.0], np.diff(r_sorted) * 0.5 * (m_sorted[1:] + m_sorted[:-1])])
    cumulative = np.cumsum(increments)
    out = np.empty_like(residual)
    out[order] = base_emissions + cumulative
    return out


def estimate_marginal_factors(emissions: np.ndarray, demand: np.ndarray, n_bins: int = 8,
                              min_obs: int = 30) -> dict:
    """Regression estimate of marginal factors by demand level (Hawkes-style).

    Regress hour-to-hour changes in emissions on changes in demand within bins
    of demand level. Returns bin edges, slopes (kg per MWh of demand change)
    and an hourly series obtained by assigning each hour its bin's slope.
    """
    emissions = np.asarray(emissions, dtype=float)
    demand = np.asarray(demand, dtype=float)
    if emissions.shape != demand.shape:
        raise ValueError("emissions and demand must have the same length")
    d_e = np.diff(emissions)
    d_l = np.diff(demand)
    level = 0.5 * (demand[1:] + demand[:-1])
    edges = np.quantile(level, np.linspace(0, 1, n_bins + 1))
    slopes, counts, r2 = [], [], []
    overall = float(np.sum(d_e * d_l) / np.sum(d_l * d_l))
    for b in range(n_bins):
        hi = level <= edges[b + 1] if b == n_bins - 1 else level < edges[b + 1]
        mask = (level >= edges[b]) & hi & (np.abs(d_l) > 1e-9)
        counts.append(int(mask.sum()))
        if mask.sum() < min_obs:
            slopes.append(overall)
            r2.append(float("nan"))
            continue
        x, y = d_l[mask], d_e[mask]
        slope = float(np.sum(x * y) / np.sum(x * x))
        resid = y - slope * x
        r2.append(float(1 - np.sum(resid**2) / max(np.sum((y - y.mean()) ** 2), 1e-12)))
        slopes.append(slope)
    bin_of_hour = np.clip(np.searchsorted(edges, demand, side="right") - 1, 0, n_bins - 1)
    return {
        "edges": edges.tolist(),
        "slopes": slopes,
        "counts": counts,
        "r2": r2,
        "overall_slope": overall,
        "hourly": np.array(slopes)[bin_of_hour],
    }


def marginal_profile(profiles: HourlyProfiles, settings: MeritOrderSettings = MeritOrderSettings(),
                     seed: int = 11) -> np.ndarray:
    """Hourly marginal import intensity for Luxembourg from the DE merit order."""
    return merit_order_marginal(german_residual_load_index(profiles, seed=seed), settings)


def describe(series: np.ndarray, profiles: HourlyProfiles) -> dict:
    _, hod, _ = _calendar(len(series))
    by_hour = [float(series[hod == h].mean()) for h in range(24)]
    return {
        "mean": float(series.mean()),
        "p10": float(np.percentile(series, 10)),
        "p90": float(np.percentile(series, 90)),
        "hour_of_day_mean": by_hour,
    }
