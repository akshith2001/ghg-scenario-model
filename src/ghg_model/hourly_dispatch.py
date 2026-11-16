"""Hourly capacity-expansion and dispatch model for an import-dependent grid.

The model chooses additional solar capacity (in cost tiers) and battery energy
capacity, together with hourly imports per channel, exports, curtailment and
battery operation, to minimise annual cost plus a carbon price times
emissions.

The question it is built to test is whether the *carbon accounting rule*
changes the plan:

``"average"`` (attributional)
    Imports carry the hourly *average* intensity of the exporting grid(s).
    Exports earn no emission credit. This mirrors inventory-style reporting.

``"marginal"`` (consequential)
    Imports carry an hourly *marginal* intensity: the emissions of the plant
    that responds to Luxembourg's extra demand. Exports displace the same
    marginal plant and earn a credit at that intensity.

Features that relax the simplest assumptions
-------------------------------------------
* **Solar cost supply curve** - new solar is split into tiers with rising cost
  (large roofs and ground-mount first, small roofs last), so the optimum can
  be interior rather than all-or-nothing.
* **Battery wear** - each MWh discharged costs capex / cycle life, and daily
  throughput is capped at ``max_cycles_per_day`` x capacity.
* **Imports by border** - ``border_mode`` selects ``"aggregate"`` (one import
  channel), ``"flow_shares"`` (physical flows split across DE, FR and BE in
  fixed shares, so the tightest interconnector limits the total) or
  ``"free"`` (each border chosen independently - an upper bound that shows
  how much average accounting rewards contracting the cleanest neighbour).
* **Marginal factor source** - ``marginal_source`` selects a flat factor with
  a midday dip (``"parametric"``) or an hourly series on the profiles, such
  as the DE merit-order model or an estimate from measured data
  (``"profile"``).
* **Imperfect foresight** - ``battery_cost_multiplier`` lets the planner
  scale battery cost by the inverse of the value retained when batteries are
  operated on forecasts (see ``foresight.py``).

All energy is in MWh, intensities in kg/MWh (= g/kWh), emissions in tonnes,
money in EUR per year. Solved with SciPy's HiGHS LP solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .luxembourg_profiles import HourlyProfiles

ACCOUNTING_RULES = ("average", "marginal")
BORDER_MODES = ("aggregate", "flow_shares", "free")


@dataclass(frozen=True)
class SystemParameters:
    """Economic and technical assumptions. Every default is an assumption."""

    existing_solar_mw: float = 746.0
    solar_tier_mw: tuple = (500.0, 500.0, 500.0)
    solar_tier_cost_multiplier: tuple = (1.0, 1.3, 1.7)
    max_battery_mwh: float = 6000.0
    solar_lcoe_eur_per_mwh: float = 60.0
    battery_capex_eur_per_kwh: float = 150.0
    battery_lifetime_years: float = 15.0
    battery_fixed_om_share: float = 0.02
    battery_cycle_life: float = 6000.0
    max_cycles_per_day: float = 1.5
    battery_cost_multiplier: float = 1.0
    discount_rate: float = 0.06
    battery_duration_h: float = 4.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    export_cap_mw: float = 300.0
    import_cap_mw: float = 1800.0
    border_mode: str = "aggregate"
    border_import_cap_mw: dict = field(default_factory=lambda: {"DE": 1000.0, "BE": 650.0, "FR": 400.0})
    export_value_share: float = 0.9
    carbon_price_eur_per_t: float = 80.0
    marginal_source: str = "parametric"
    marginal_intensity_g_per_kwh: float = 550.0
    marginal_solar_dip: float = 0.3
    marginal_scale: float = 1.0
    price_scale: float = 1.0

    @property
    def max_new_solar_mw(self) -> float:
        return float(sum(self.solar_tier_mw))

    def solar_annual_cost_per_mw(self, mean_cf: float, tier: int = 0) -> float:
        return self.solar_lcoe_eur_per_mwh * self.solar_tier_cost_multiplier[tier] * mean_cf * 8760.0

    def battery_annual_cost_per_mwh(self) -> float:
        r, n = self.discount_rate, self.battery_lifetime_years
        crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
        capex_per_mwh = self.battery_capex_eur_per_kwh * 1000.0
        return capex_per_mwh * (crf + self.battery_fixed_om_share) * self.battery_cost_multiplier

    def battery_wear_cost_per_mwh(self) -> float:
        return self.battery_capex_eur_per_kwh * 1000.0 / self.battery_cycle_life


@dataclass
class DispatchResult:
    accounting: str
    new_solar_mw: float
    new_solar_by_tier_mw: list
    battery_mwh: float
    annual_cost_eur: float
    investment_cost_eur: float
    energy_cost_eur: float
    wear_cost_eur: float
    imports_gwh: float
    imports_by_channel_gwh: dict
    exports_gwh: float
    curtailment_gwh: float
    battery_cycles_per_year: float
    emissions_average_t: float
    emissions_marginal_t: float
    objective_eur: float
    status: str

    def to_dict(self) -> dict:
        def r(v):
            if isinstance(v, float):
                return round(v, 3)
            if isinstance(v, list):
                return [r(x) for x in v]
            if isinstance(v, dict):
                return {k: r(x) for k, x in v.items()}
            return v
        return {k: r(v) for k, v in asdict(self).items()}


def marginal_intensity(profiles: HourlyProfiles, params: SystemParameters) -> np.ndarray:
    """Hourly marginal import intensity (kg/MWh) from the chosen source."""
    if params.marginal_source == "profile":
        if profiles.marginal_g_per_kwh is None:
            raise ValueError("marginal_source='profile' needs profiles.marginal_g_per_kwh")
        return profiles.marginal_g_per_kwh * params.marginal_scale
    if params.marginal_source != "parametric":
        raise ValueError(f"unknown marginal_source {params.marginal_source!r}")
    dip = np.clip(params.marginal_solar_dip, 0.0, 1.0)
    return params.marginal_intensity_g_per_kwh * (1.0 - dip * profiles.solar_index) * params.marginal_scale


def import_channels(profiles: HourlyProfiles, params: SystemParameters) -> list[tuple[str, np.ndarray, float]]:
    """Return (name, average intensity series, capacity MW) for each import channel."""
    mode = params.border_mode
    if mode not in BORDER_MODES:
        raise ValueError(f"unknown border_mode {mode!r}; use one of {BORDER_MODES}")
    if mode == "aggregate" or profiles.border_intensity is None:
        return [("all", profiles.import_avg_g_per_kwh, params.import_cap_mw)]
    caps = params.border_import_cap_mw
    zones = [z for z in profiles.border_intensity if z in caps]
    if mode == "flow_shares":
        shares = profiles.border_shares or {z: 1 / len(zones) for z in zones}
        total = sum(shares[z] for z in zones)
        s = {z: shares[z] / total for z in zones}
        cap = min(caps[z] / s[z] for z in zones if s[z] > 0)
        factor = sum(s[z] * profiles.border_intensity[z] for z in zones)
        return [("all (flow shares)", factor, float(cap))]
    return [(z, profiles.border_intensity[z], float(caps[z])) for z in zones]


def accounting_factors(profiles: HourlyProfiles, params: SystemParameters, rule: str):
    """Return (list of import factors per channel, export credit) for a rule."""
    channels = import_channels(profiles, params)
    if rule == "average":
        return [f for _, f, _ in channels], np.zeros(profiles.n_hours)
    if rule == "marginal":
        m = marginal_intensity(profiles, params)
        return [m for _ in channels], m
    raise ValueError(f"unknown accounting rule {rule!r}; use one of {ACCOUNTING_RULES}")


def _blocks(profiles: HourlyProfiles) -> list[tuple[int, int]]:
    """Battery state of charge is cyclic within each block."""
    n = profiles.n_hours
    if n % 24 != 0:
        return [(0, n)]
    if np.allclose(profiles.hour_weights, 1.0) and n >= 8760:
        return [(0, n)]
    return [(i, i + 24) for i in range(0, n, 24)]


def export_price(price: np.ndarray, value_share: float) -> np.ndarray:
    """Export settlement with a nonnegative spread on either side of zero.

    Positive prices retain the original discount. At negative prices exporting
    costs more than importing earns, preventing same-hour circular trade.
    """
    if not np.isfinite(value_share) or not 0 <= value_share <= 1:
        raise ValueError("export_value_share must lie in [0, 1]")
    price = np.asarray(price, dtype=float)
    return price - (1 - value_share) * np.abs(price)


def solve(
    profiles: HourlyProfiles,
    params: SystemParameters,
    accounting: str = "average",
    fixed_solar_mw: float | None = None,
    fixed_battery_mwh: float | None = None,
    price_override: np.ndarray | None = None,
) -> DispatchResult:
    """Optimise investment and dispatch (or dispatch only, if capacities are fixed).

    ``price_override`` replaces the price series in the objective (used to
    dispatch on a forecast); reported energy cost always uses actual prices.
    """
    n = profiles.n_hours
    w = profiles.hour_weights
    actual_price = profiles.price_eur_per_mwh * params.price_scale
    price = actual_price if price_override is None else np.asarray(price_override) * params.price_scale
    sell_price = export_price(price, params.export_value_share)
    actual_sell_price = export_price(actual_price, params.export_value_share)
    channels = import_channels(profiles, params)
    f_imp, h_exp = accounting_factors(profiles, params, accounting)
    cf = profiles.solar_cf
    eta_c, eta_d = params.charge_efficiency, params.discharge_efficiency
    n_tiers = len(params.solar_tier_mw)
    n_ch = len(channels)

    # Variable layout: [S_1..S_T, B, g_1(n)..g_C(n), x(n), k(n), ch(n), dis(n), soc(n)]
    iS = list(range(n_tiers))
    iB = n_tiers
    base = n_tiers + 1
    g_off = [base + c * n for c in range(n_ch)]
    after = base + n_ch * n
    off = {name: after + j * n for j, name in enumerate(("x", "k", "ch", "dis", "soc"))}
    nvar = after + 5 * n
    idx = np.arange(n)

    mean_cf = float(np.average(cf, weights=w))
    a_S = [params.solar_annual_cost_per_mw(mean_cf, t) for t in range(n_tiers)]
    a_B = params.battery_annual_cost_per_mwh()
    wear = params.battery_wear_cost_per_mwh()
    c = np.zeros(nvar)
    c[iS] = a_S
    c[iB] = a_B
    for ci in range(n_ch):
        c[g_off[ci] + idx] = w * (price + params.carbon_price_eur_per_t * f_imp[ci] / 1000.0)
    c[off["x"] + idx] = -w * (sell_price + params.carbon_price_eur_per_t * h_exp / 1000.0)
    c[off["k"] + idx] = 1e-3 * w
    c[off["dis"] + idx] = w * wear

    rows, cols, vals = [], [], []

    def add(r, cidx, v):
        rows.extend(np.atleast_1d(r)); cols.extend(np.atleast_1d(cidx)); vals.extend(np.atleast_1d(v))

    # Energy balance
    for ci in range(n_ch):
        add(idx, g_off[ci] + idx, np.ones(n))
    add(idx, off["dis"] + idx, np.ones(n))
    add(idx, off["ch"] + idx, -np.ones(n))
    add(idx, off["x"] + idx, -np.ones(n))
    add(idx, off["k"] + idx, -np.ones(n))
    for t_ in iS:
        add(idx, np.full(n, t_), cf)
    b_eq = profiles.load_mw - params.existing_solar_mw * cf - profiles.wind_mw - profiles.other_domestic_mw

    # State of charge, cyclic per block
    blocks = _blocks(profiles)
    for start, end in blocks:
        t = np.arange(start, end)
        prev = np.concatenate([[end - 1], t[:-1]])
        r = n + t
        add(r, off["soc"] + t, np.ones(len(t)))
        add(r, off["soc"] + prev, -np.ones(len(t)))
        add(r, off["ch"] + t, np.full(len(t), -eta_c))
        add(r, off["dis"] + t, np.full(len(t), 1.0 / eta_d))
    A_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(2 * n, nvar))
    b_eq = np.concatenate([b_eq, np.zeros(n)])

    # Inequalities
    rows, cols, vals = [], [], []
    for j, name in enumerate(("ch", "dis")):
        r = j * n + idx
        add(r, off[name] + idx, np.ones(n))
        add(r, np.full(n, iB), np.full(n, -1.0 / params.battery_duration_h))
    r = 2 * n + idx
    add(r, off["soc"] + idx, np.ones(n))
    add(r, np.full(n, iB), -np.ones(n))
    r = 3 * n + idx
    add(r, off["k"] + idx, np.ones(n))
    for t_ in iS:
        add(r, np.full(n, t_), -cf)
    b_ub = [np.zeros(3 * n), params.existing_solar_mw * cf + profiles.wind_mw]
    # Daily throughput limit: sum(dis over each 24 h) <= cycles * B
    day_rows = 0
    if params.max_cycles_per_day is not None and n % 24 == 0:
        for d, start in enumerate(range(0, n, 24)):
            rr = 4 * n + d
            t = np.arange(start, start + 24)
            add(np.full(24, rr), off["dis"] + t, np.ones(24))
            add(rr, iB, -params.max_cycles_per_day)
            day_rows += 1
        b_ub.append(np.zeros(day_rows))
    A_ub = sparse.csr_matrix((vals, (rows, cols)), shape=(4 * n + day_rows, nvar))
    b_ub = np.concatenate(b_ub)

    bounds = [(0, None)] * nvar
    if fixed_solar_mw is not None:
        remaining = fixed_solar_mw
        for t_, cap in zip(iS, params.solar_tier_mw):  # fill cheapest tiers first
            take = min(cap, max(remaining, 0.0))
            bounds[t_] = (take, take)
            remaining -= take
    else:
        for t_, cap in zip(iS, params.solar_tier_mw):
            bounds[t_] = (0, cap)
    bounds[iB] = (fixed_battery_mwh, fixed_battery_mwh) if fixed_battery_mwh is not None else (0, params.max_battery_mwh)
    for ci, (_, _, cap) in enumerate(channels):
        for t in range(n):
            bounds[g_off[ci] + t] = (0, cap)
    for t in range(n):
        bounds[off["x"] + t] = (0, params.export_cap_mw)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"dispatch LP failed: {res.message}")
    z = res.x
    S_tiers = [float(z[t_]) for t_ in iS]
    B = float(z[iB])
    g = [z[g_off[ci] + idx] for ci in range(n_ch)]
    g_total = np.sum(g, axis=0)
    x, k, dis = (z[off[n_] + idx] for n_ in ("x", "k", "dis"))
    f_avg, _ = accounting_factors(profiles, params, "average")
    f_mar, h_mar = accounting_factors(profiles, params, "marginal")
    investment = float(sum(a * s for a, s in zip(a_S, S_tiers)) + a_B * B)
    energy = float(np.sum(w * (actual_price * g_total - actual_sell_price * x)))
    wear_cost = float(np.sum(w * wear * dis))
    em_avg = float(sum(np.sum(w * f_avg[ci] * g[ci]) for ci in range(n_ch)) / 1000)
    em_mar = float((np.sum(w * f_mar[0] * g_total) - np.sum(w * h_mar * x)) / 1000)
    return DispatchResult(
        accounting=accounting,
        new_solar_mw=float(sum(S_tiers)),
        new_solar_by_tier_mw=S_tiers,
        battery_mwh=B,
        annual_cost_eur=investment + energy + wear_cost,
        investment_cost_eur=investment,
        energy_cost_eur=energy,
        wear_cost_eur=wear_cost,
        imports_gwh=float(np.sum(w * g_total) / 1000),
        imports_by_channel_gwh={channels[ci][0]: float(np.sum(w * g[ci]) / 1000) for ci in range(n_ch)},
        exports_gwh=float(np.sum(w * x) / 1000),
        curtailment_gwh=float(np.sum(w * k) / 1000),
        battery_cycles_per_year=float(np.sum(w * dis) / B) if B > 1e-6 else 0.0,
        emissions_average_t=em_avg,
        emissions_marginal_t=em_mar,
        objective_eur=float(res.fun),
        status=res.message,
    )


def objective_under(result: DispatchResult, params: SystemParameters, rule: str) -> float:
    """Annual cost plus carbon cost of a solved plan under an accounting rule."""
    emissions = result.emissions_average_t if rule == "average" else result.emissions_marginal_t
    return result.annual_cost_eur + params.carbon_price_eur_per_t * emissions


def cross_evaluate(profiles: HourlyProfiles, params: SystemParameters) -> dict:
    """Plan under each rule, then judge each plan under the marginal rule.

    Regret = how much worse (EUR/year, cost plus carbon cost under marginal
    accounting) the average-rule plan is than the marginal-rule plan, after
    re-dispatching both with their capacities fixed.
    """
    plans = {rule: solve(profiles, params, rule) for rule in ACCOUNTING_RULES}
    judged = {
        rule: solve(profiles, params, "marginal",
                    fixed_solar_mw=plans[rule].new_solar_mw,
                    fixed_battery_mwh=plans[rule].battery_mwh)
        for rule in ACCOUNTING_RULES
    }
    obj = {rule: objective_under(judged[rule], params, "marginal") for rule in ACCOUNTING_RULES}
    return {
        "plans": plans,
        "judged_under_marginal": judged,
        "regret_eur_per_year": obj["average"] - obj["marginal"],
        "regret_share": (obj["average"] - obj["marginal"]) / abs(obj["marginal"]),
    }


def with_overrides(params: SystemParameters, **overrides) -> SystemParameters:
    return replace(params, **overrides)
