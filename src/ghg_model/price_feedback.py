"""Price feedback: storage that flattens the DE-LU price spread it earns from.

The hourly LP treats day-ahead prices as given. That is fine for one small
plant, but storage at gigawatt scale - Luxembourg's own optimum plus the fleet
being built across the German-Luxembourg bidding zone - lowers peak prices and
raises off-peak prices. This module puts that feedback in, using measured data
only:

1. ``merit_curve`` estimates the zone supply curve: the median day-ahead price
   as a monotone function of German residual load (load minus wind and solar),
   from the same year's hourly data.
2. ``fleet_dispatch`` dispatches a zone storage fleet of given energy and power
   the way a competitive fleet would in equilibrium: it minimises the area under
   the supply curve, which flattens prices within each day. The curve is
   piecewise linear, so each day is a small LP.
3. ``shifted_prices`` moves every hour along the curve by the fleet's net
   charging, keeping each hour's deviation from the curve (fuel prices, outages,
   neighbours) unchanged.
4. ``equilibrium_plan`` re-plans Luxembourg against the compressed prices and
   counts Luxembourg's own battery in the fleet, iterating to a fixed point, so
   Luxembourg is no longer a price-taker.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .hourly_dispatch import SystemParameters, solve
from .luxembourg_profiles import HourlyProfiles


@dataclass(frozen=True)
class MeritCurve:
    """Monotone piecewise-linear price (EUR/MWh) as a function of residual load (MW)."""

    x_mw: np.ndarray
    p_eur: np.ndarray

    def __call__(self, rl_mw):
        rl = np.asarray(rl_mw, dtype=float)
        x, p = self.x_mw, self.p_eur
        lo = (p[1] - p[0]) / (x[1] - x[0])
        hi = (p[-1] - p[-2]) / (x[-1] - x[-2])
        y = np.interp(rl, x, p)
        y = np.where(rl < x[0], p[0] + lo * (rl - x[0]), y)
        return np.where(rl > x[-1], p[-1] + hi * (rl - x[-1]), y)

    def slope_at(self, rl_mw: float, width_mw: float = 2000.0) -> float:
        """EUR/MWh per MW around a residual load."""
        return float((self(rl_mw + width_mw) - self(rl_mw - width_mw)) / (2 * width_mw))


def merit_curve(rl_mw: np.ndarray, price: np.ndarray, n_bins: int = 30) -> MeritCurve:
    """Binned-median supply curve, made non-decreasing by pooling adjacent violators."""
    rl_mw, price = np.asarray(rl_mw, float), np.asarray(price, float)
    edges = np.quantile(rl_mw, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(rl_mw, edges[1:-1]), 0, n_bins - 1)
    x = np.array([np.median(rl_mw[idx == i]) for i in range(n_bins)])
    y = np.array([np.median(price[idx == i]) for i in range(n_bins)])
    w = np.array([(idx == i).sum() for i in range(n_bins)], dtype=float)
    # pool-adjacent-violators for a non-decreasing fit
    blocks = [[y[i], w[i], [i]] for i in range(n_bins)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1], blocks[i][2] + blocks[i + 1][2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fit = np.empty(n_bins)
    for v, _, members in blocks:
        fit[members] = v
    # strictly increasing x needed for interpolation; tiny positive slope keeps F strictly convex
    fit = fit + np.arange(n_bins) * 1e-3
    return MeritCurve(x, fit)


def shifted_prices(price: np.ndarray, rl_mw: np.ndarray, delta_mw: np.ndarray, curve: MeritCurve) -> np.ndarray:
    """Prices after net load changes by ``delta_mw`` (charging positive)."""
    return np.asarray(price) + curve(np.asarray(rl_mw) + delta_mw) - curve(rl_mw)


def _segments(curve: MeritCurve, lo: float, hi: float, n: int = 60):
    """Tangent lines z >= a_k + b_k * x of F(x) = integral of the curve, on [lo, hi]."""
    xs = np.linspace(lo, hi, n + 1)
    mids = 0.5 * (xs[:-1] + xs[1:])
    # F up to a constant, by trapezoid on a fine grid
    grid = np.linspace(lo - 1, hi + 1, 4000)
    Fg = np.concatenate([[0.0], np.cumsum(0.5 * (curve(grid[1:]) + curve(grid[:-1])) * np.diff(grid))])
    F = np.interp(mids, grid, Fg)
    b = curve(mids)
    return F - b * mids, b


def fleet_dispatch(rl_mw: np.ndarray, curve: MeritCurve, energy_mwh: float, power_mw: float,
                   efficiency: float = 0.9) -> np.ndarray:
    """Net charging (MW, charge positive) of a zone storage fleet, day by day.

    Minimises sum_h F(rl_h + charge_h - discharge_h), the variable generation cost
    implied by the supply curve, with a state of charge that returns to its start
    each day. Returns zeros for an empty fleet.
    """
    rl_mw = np.asarray(rl_mw, float)
    n_h = len(rl_mw)
    if energy_mwh <= 0 or power_mw <= 0:
        return np.zeros(n_h)
    eta = np.sqrt(efficiency)
    a, b = _segments(curve, rl_mw.min() - power_mw, rl_mw.max() + power_mw)
    K = len(a)
    out = np.zeros(n_h)
    T = 24
    # variables per day: c(T), d(T), s(T), z(T)
    nv = 4 * T
    cost = np.concatenate([np.zeros(3 * T), np.ones(T)])
    bounds = [(0, power_mw)] * T + [(0, power_mw)] * T + [(0, energy_mwh)] * T + [(None, None)] * T
    A_eq = lil_matrix((T, nv))
    for t in range(T):  # s_t - s_{t-1} - eta c_t + d_t/eta = 0, cyclic
        A_eq[t, 2 * T + t] = 1.0
        A_eq[t, 2 * T + (t - 1) % T] = -1.0
        A_eq[t, t] = -eta
        A_eq[t, T + t] = 1.0 / eta
    A_eq = A_eq.tocsr()
    A_ub = lil_matrix((T * K, nv))
    for t in range(T):  # a_k + b_k (rl + c - d) - z <= 0
        for k in range(K):
            r = t * K + k
            A_ub[r, t] = b[k]
            A_ub[r, T + t] = -b[k]
            A_ub[r, 3 * T + t] = -1.0
    A_ub = A_ub.tocsr()
    for day in range(n_h // T):
        rl = rl_mw[day * T:(day + 1) * T]
        b_ub = -(a[None, :] + b[None, :] * rl[:, None]).ravel()
        res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(T), bounds=bounds, method="highs")
        if res.status != 0:
            raise RuntimeError(f"fleet dispatch failed on day {day}: {res.message}")
        out[day * T:(day + 1) * T] = res.x[:T] - res.x[T:2 * T]
    return out


def spread_stats(price: np.ndarray) -> dict:
    P = np.asarray(price)[: len(price) // 24 * 24].reshape(-1, 24)
    return {"mean_daily_spread_eur": float((P.max(1) - P.min(1)).mean()),
            "mean_price_eur": float(np.mean(price)),
            "negative_hours": int((np.asarray(price) < 0).sum())}


def zone_prices(d: dict, curve: MeritCurve, fleet_mwh: float, duration_h: float = 2.0,
                efficiency: float = 0.9) -> np.ndarray:
    """Year of DE-LU prices with an added zone storage fleet of ``fleet_mwh``."""
    rl = d["de_load"] - d["de_solar"] - d["de_wind"]
    delta = fleet_dispatch(rl, curve, fleet_mwh, fleet_mwh / duration_h, efficiency)
    return shifted_prices(d["price"], rl, delta, curve)


def equilibrium_plan(prof: HourlyProfiles, d: dict, params: SystemParameters, accounting: str,
                     zone_fleet_mwh: float, duration_h: float = 2.0, iterations: int = 6,
                     damping: float = 0.5, curve: MeritCurve | None = None) -> dict:
    """Luxembourg plan when prices respond to the zone fleet plus Luxembourg's own battery.

    Fixed point: prices depend on the total fleet (zone fleet + Luxembourg battery
    at the LU model's duration); the LU battery is re-optimised against those
    prices; the battery size is damped between iterations.
    """
    if iterations < 1 or not 0 <= damping < 1:
        raise ValueError("iterations must be positive and damping in [0, 1)")
    rl = d["de_load"] - d["de_solar"] - d["de_wind"]
    curve = curve or merit_curve(rl, d["price"])
    lu_dur = params.battery_duration_h
    history = []
    b_lu = 0.0
    res = None
    converged = False
    for it in range(iterations):
        delta = fleet_dispatch(rl, curve, zone_fleet_mwh, zone_fleet_mwh / duration_h)
        if b_lu > 0:
            delta = delta + fleet_dispatch(rl + delta, curve, b_lu, b_lu / lu_dur)
        price = shifted_prices(d["price"], rl, delta, curve)
        res = solve(replace(prof, price_eur_per_mwh=price), params, accounting)
        history.append({"iteration": it, "lu_battery_assumed_mwh": round(b_lu),
                        "lu_battery_chosen_mwh": round(res.battery_mwh), "new_solar_mw": round(res.new_solar_mw),
                        **{k: round(v, 1) for k, v in spread_stats(price).items()}})
        new_b = res.battery_mwh if it == 0 else damping * b_lu + (1 - damping) * res.battery_mwh
        residual = abs(res.battery_mwh - b_lu)
        if residual < 50:
            converged = True
            b_lu = new_b
            break
        b_lu = new_b
    return {"zone_fleet_mwh": zone_fleet_mwh, "battery_mwh": round(res.battery_mwh), "new_solar_mw": round(res.new_solar_mw),
            "spread": spread_stats(price), "history": history, "_price": price, "_result": res,
            "converged": converged, "fixed_point_residual_mwh": float(residual)}


ZONE_FLEETS_MWH = (0.0, 10000.0, 25000.0, 50000.0, 100000.0)


def price_feedback_study(prof: HourlyProfiles, d: dict, params: SystemParameters,
                         fleets=ZONE_FLEETS_MWH, duration_h: float = 2.0, log=print) -> dict:
    """Plans under each accounting rule as zone-wide storage compresses the spread.

    ``fleets`` is storage added across DE-LU beyond what is already reflected
    in the measured prices. For each fleet both rules are solved to their price
    equilibrium; the average-rule plan is then judged under marginal accounting
    at the marginal-rule equilibrium prices (regret).
    """
    from .hourly_dispatch import objective_under
    rl = d["de_load"] - d["de_solar"] - d["de_wind"]
    curve = merit_curve(rl, d["price"])
    hod_slope = curve.slope_at(float(np.median(rl))) * 1000.0
    rows = []
    for E in fleets:
        eq = {rule: equilibrium_plan(prof, d, params, rule, E, duration_h, curve=curve) for rule in ("average", "marginal")}
        p_m = replace(prof, price_eur_per_mwh=eq["marginal"]["_price"])
        judged = {rule: solve(p_m, params, "marginal", fixed_solar_mw=eq[rule]["_result"].new_solar_mw,
                              fixed_battery_mwh=eq[rule]["_result"].battery_mwh) for rule in eq}
        obj = {rule: objective_under(judged[rule], params, "marginal") for rule in judged}
        row = {"zone_fleet_gwh": E / 1000.0,
               "mean_daily_spread_eur": round(eq["marginal"]["spread"]["mean_daily_spread_eur"], 1),
               "negative_price_hours": eq["marginal"]["spread"]["negative_hours"],
               "solar_average_mw": eq["average"]["new_solar_mw"], "solar_marginal_mw": eq["marginal"]["new_solar_mw"],
               "battery_average_mwh": eq["average"]["battery_mwh"], "battery_marginal_mwh": eq["marginal"]["battery_mwh"],
               "regret_share": round((obj["average"] - obj["marginal"]) / abs(obj["marginal"]), 4),
               "iterations": {rule: len(eq[rule]["history"]) for rule in eq},
               "converged": {rule: eq[rule]["converged"] for rule in eq},
               "fixed_point_residual_mwh": {rule: eq[rule]["fixed_point_residual_mwh"] for rule in eq}}
        rows.append(row)
        log(f"  zone fleet {E / 1000:.0f} GWh: spread {row['mean_daily_spread_eur']}, solar {row['solar_average_mw']} vs "
            f"{row['solar_marginal_mw']} MW, battery {row['battery_average_mwh']} vs {row['battery_marginal_mwh']} MWh, "
            f"regret {row['regret_share']:.2%}")
    base = spread_stats(d["price"])
    return {"merit_curve": {"residual_load_mw": curve.x_mw.round(0).tolist(), "price_eur": curve.p_eur.round(2).tolist(),
                            "slope_eur_per_mwh_per_gw_at_median": round(hod_slope, 2)},
            "measured_spread": base, "zone_fleet_duration_h": duration_h, "rows": rows}
