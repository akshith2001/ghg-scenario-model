"""Luxembourg case study: does the carbon-accounting rule change the solar plan?

Research question
-----------------
Luxembourg imports about three quarters of its electricity. If imported
electricity is valued at the *average* intensity of the exporting grids, does
an hourly cost-plus-carbon optimisation choose a different solar and storage
build than when imports are valued at their *marginal* intensity - and how
much does the difference cost, judged under marginal accounting?

Central configuration
---------------------
* calibrated hourly 2025 year compressed to representative days;
* hourly marginal intensity from a German merit-order model
  (``marginal_emissions``), or from a measured series if supplied;
* imports split across DE, FR and BE in fixed physical shares with assumed
  interconnector limits (``border_mode="flow_shares"``);
* tiered solar costs, battery wear and a daily cycle limit;
* battery cost scaled by the value retained under imperfect price foresight
  (``foresight``).

The Monte Carlo varies the fuel-price regime, merit-order parameters, costs,
carbon price, price level and interconnector limits. A robustness table
re-runs the central case under alternative modelling choices.

Results are conditional on the synthetic hourly shapes and the stated
assumptions. They are not forecasts or policy advice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .foresight import battery_value_retention, derated_parameters
from .hourly_dispatch import ACCOUNTING_RULES, SystemParameters, cross_evaluate, with_overrides
from .luxembourg_profiles import (
    HourlyProfiles,
    calibration_report,
    load_profiles_csv,
    representative_days,
    synthesise_profiles,
)
from .marginal_emissions import (
    MeritOrderSettings,
    describe,
    german_residual_load_index,
    merit_order_marginal,
)

# Uniform ranges for the Monte Carlo (all assumptions; see docs/luxembourg_case_study.md)
UNCERTAIN_RANGES = {
    "gas_cheaper": (0.0, 1.0),               # Bernoulli: fuel-price regime
    "lignite_must_run": (0.2, 0.8),
    "marginal_scale": (0.8, 1.2),
    "solar_lcoe_eur_per_mwh": (45.0, 90.0),
    "battery_capex_eur_per_kwh": (110.0, 250.0),
    "carbon_price_eur_per_t": (40.0, 200.0),
    "price_scale": (0.75, 1.25),
    "interconnector_scale": (0.7, 1.3),
}
FLIP_THRESHOLD_MW = 100.0
CENTRAL_PARAMS = SystemParameters(marginal_source="profile", border_mode="flow_shares")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


class MarginalCache:
    """Merit-order marginal series computed on the full year, subset to representative hours."""

    def __init__(self, year: HourlyProfiles, rep: HourlyProfiles):
        self.residual = (year.zone_residual_mw if year.zone_residual_mw is not None
                         else german_residual_load_index(year))
        self.hours = np.array(rep.metadata.get("hours", np.arange(rep.n_hours)))
        self.rep = rep
        self._cache: dict = {}
        # A measured or estimated series supplied with the profiles overrides the
        # merit-order model for every draw (only marginal_scale then varies it).
        self.supplied = (None if year.marginal_g_per_kwh is None
                         else rep.with_marginal(year.marginal_g_per_kwh[self.hours], "supplied"))

    def profiles(self, settings: MeritOrderSettings) -> HourlyProfiles:
        if self.supplied is not None:
            return self.supplied
        key = (settings.regime, round(settings.lignite_must_run, 4))
        if key not in self._cache:
            series = merit_order_marginal(self.residual, settings)[self.hours]
            self._cache[key] = self.rep.with_marginal(series, f"merit order ({settings.regime})")
        return self._cache[key]


def summarise_cross_evaluation(ce: dict) -> dict:
    return {
        "plans": {rule: ce["plans"][rule].to_dict() for rule in ACCOUNTING_RULES},
        "plans_redispatched_under_marginal": {rule: ce["judged_under_marginal"][rule].to_dict()
                                              for rule in ACCOUNTING_RULES},
        "regret_eur_per_year": round(ce["regret_eur_per_year"], 0),
        "regret_share": round(ce["regret_share"], 5),
    }


def _draw_params(base: SystemParameters, row: dict) -> tuple[SystemParameters, MeritOrderSettings]:
    caps = {z: v * row["interconnector_scale"] for z, v in base.border_import_cap_mw.items()}
    params = with_overrides(
        base,
        marginal_scale=row["marginal_scale"],
        solar_lcoe_eur_per_mwh=row["solar_lcoe_eur_per_mwh"],
        battery_capex_eur_per_kwh=row["battery_capex_eur_per_kwh"],
        carbon_price_eur_per_t=row["carbon_price_eur_per_t"],
        price_scale=row["price_scale"],
        border_import_cap_mw=caps,
        import_cap_mw=base.import_cap_mw * row["interconnector_scale"],
    )
    settings = MeritOrderSettings(regime="gas_cheaper" if row["gas_cheaper"] >= 0.5 else "coal_cheaper",
                                  lignite_must_run=row["lignite_must_run"])
    return params, settings


def monte_carlo(cache: MarginalCache, base: SystemParameters, n_draws: int = 300, seed: int = 7) -> dict:
    """Sample uncertain inputs and cross-evaluate the two accounting rules."""
    rng = np.random.default_rng(seed)
    names = list(UNCERTAIN_RANGES)
    records = []
    for _ in range(n_draws):
        row = {k: float(rng.uniform(*UNCERTAIN_RANGES[k])) for k in names}
        row["gas_cheaper"] = float(row["gas_cheaper"] >= 0.5)
        params, settings = _draw_params(base, row)
        prof = cache.profiles(settings)
        ce = cross_evaluate(prof, params)
        mean_marginal = float(np.average(prof.marginal_g_per_kwh * params.marginal_scale, weights=prof.hour_weights))
        records.append({
            **row,
            "mean_marginal_g_per_kwh": mean_marginal,
            "solar_average_mw": ce["plans"]["average"].new_solar_mw,
            "solar_marginal_mw": ce["plans"]["marginal"].new_solar_mw,
            "battery_average_mwh": ce["plans"]["average"].battery_mwh,
            "battery_marginal_mwh": ce["plans"]["marginal"].battery_mwh,
            "regret_eur_per_year": ce["regret_eur_per_year"],
            "regret_share": ce["regret_share"],
        })
    regret = np.array([r["regret_eur_per_year"] for r in records])
    share = np.array([r["regret_share"] for r in records])
    gap = np.array([r["solar_marginal_mw"] - r["solar_average_mw"] for r in records])
    drivers = {k: round(_spearman(np.array([r[k] for r in records]), regret), 3) for k in names}
    return {
        "n_draws": n_draws,
        "seed": seed,
        "ranges": UNCERTAIN_RANGES,
        "solar_decision_flip_share": float(np.mean(np.abs(gap) > FLIP_THRESHOLD_MW)),
        "marginal_builds_more_solar_share": float(np.mean(gap > FLIP_THRESHOLD_MW)),
        "average_builds_more_solar_share": float(np.mean(gap < -FLIP_THRESHOLD_MW)),
        "solar_gap_mw": {"median": float(np.median(gap)), "p05": float(np.percentile(gap, 5)),
                         "p95": float(np.percentile(gap, 95))},
        "regret_eur_per_year": {"median": float(np.median(regret)), "p05": float(np.percentile(regret, 5)),
                                "p95": float(np.percentile(regret, 95))},
        "regret_share": {"median": float(np.median(share)), "p05": float(np.percentile(share, 5)),
                         "p95": float(np.percentile(share, 95))},
        "spearman_with_regret": drivers,
        "records": records,
    }


def robustness(year: HourlyProfiles, rep: HourlyProfiles, cache: MarginalCache, base: SystemParameters,
               n_days: int) -> list[dict]:
    """Re-run the central case under alternative modelling choices."""
    def row(label, ce):
        p = ce["plans"]
        return {"check": label,
                "solar_average_mw": round(p["average"].new_solar_mw),
                "solar_marginal_mw": round(p["marginal"].new_solar_mw),
                "battery_average_mwh": round(p["average"].battery_mwh),
                "battery_marginal_mwh": round(p["marginal"].battery_mwh),
                "regret_share": round(ce["regret_share"], 4)}

    central_prof = cache.profiles(MeritOrderSettings())
    out = [row("Central configuration", cross_evaluate(central_prof, base))]
    full = year.with_marginal(merit_order_marginal(cache.residual, MeritOrderSettings()), "merit order")
    out.append(row("Full 8,760 hours (no representative days)", cross_evaluate(full, base)))
    for seed in (1, 2, 3, 4):
        y = synthesise_profiles(seed=seed)
        r = representative_days(y, n_days=n_days)
        c = MarginalCache(y, r)
        out.append(row(f"Synthetic weather year, seed {seed}", cross_evaluate(c.profiles(MeritOrderSettings()), base)))
    out.append(row("Gas cheaper than coal (merit-order regime)",
                   cross_evaluate(cache.profiles(MeritOrderSettings(regime="gas_cheaper")), base)))
    out.append(row("Flat marginal factor with midday dip (old method)",
                   cross_evaluate(central_prof, with_overrides(base, marginal_source="parametric"))))
    out.append(row("One aggregated import, no border limits",
                   cross_evaluate(central_prof, with_overrides(base, border_mode="aggregate"))))
    out.append(row("Borders chosen freely (contract-based accounting)",
                   cross_evaluate(central_prof, with_overrides(base, border_mode="free"))))
    out.append(row("No battery allowed", cross_evaluate(central_prof, with_overrides(base, max_battery_mwh=0.0))))
    out.append(row("Battery at perfect foresight (no derating)",
                   cross_evaluate(central_prof, with_overrides(base, battery_cost_multiplier=1.0))))
    out.append(row("Single solar cost tier (linear costs)",
                   cross_evaluate(central_prof, with_overrides(base, solar_tier_mw=(1500.0,),
                                                               solar_tier_cost_multiplier=(1.0,)))))
    return out


def contract_loophole(cache: MarginalCache, base: SystemParameters) -> dict:
    """Paper versus marginal emissions when borders can be chosen freely."""
    prof = cache.profiles(MeritOrderSettings())
    flow = cross_evaluate(prof, base)["plans"]["average"]
    free = cross_evaluate(prof, with_overrides(base, border_mode="free"))["plans"]["average"]
    return {
        "flow_shares": {"emissions_average_t": flow.emissions_average_t,
                        "emissions_marginal_t": flow.emissions_marginal_t,
                        "imports_by_channel_gwh": flow.imports_by_channel_gwh},
        "free_borders": {"emissions_average_t": free.emissions_average_t,
                         "emissions_marginal_t": free.emissions_marginal_t,
                         "imports_by_channel_gwh": free.imports_by_channel_gwh},
    }


# --------------------------------------------------------------------------- figures
SERIES = {"average": "#2a78d6", "marginal": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEITHER, BOTH = "#cfcdc6", "#6f6d66"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_plans(summary: dict, path: Path) -> None:
    plt = _plt()
    plans = summary["plans"]
    metrics = [
        ("New solar (MW)", "new_solar_mw", 1),
        ("Battery (GWh)", "battery_mwh", 1e-3),
        ("Emissions, average rule (kt CO2e)", "emissions_average_t", 1e-3),
        ("Emissions, marginal rule (kt CO2e)", "emissions_marginal_t", 1e-3),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, (title, key, scale) in zip(axes, metrics):
        vals = [plans[r][key] * scale for r in ACCOUNTING_RULES]
        bars = ax.bar([0, 1], vals, width=0.55, color=[SERIES[r] for r in ACCOUNTING_RULES],
                      edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            label = f"{v:,.1f}" if scale < 1 and v < 20 else f"{v:,.0f}"
            ax.text(b.get_x() + b.get_width() / 2, v, label, ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks([0, 1], ["Planned with\naverage rule", "Planned with\nmarginal rule"], fontsize=8.5, color=INK2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        _style(ax)
        ax.margins(y=0.15)
    fig.suptitle("The accounting rule changes the build: central configuration, calibrated 2025 year",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_monte_carlo(mc: dict, path: Path) -> None:
    plt = _plt()
    rec = mc["records"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    carbon_value = np.array([r["carbon_price_eur_per_t"] * r["mean_marginal_g_per_kwh"] / 1000 for r in rec])
    lcoe = np.array([r["solar_lcoe_eur_per_mwh"] for r in rec])
    gap = np.array([r["solar_marginal_mw"] - r["solar_average_mw"] for r in rec])
    flip = gap > FLIP_THRESHOLD_MW
    avg_builds = np.array([r["solar_average_mw"] for r in rec]) > FLIP_THRESHOLD_MW
    groups = [
        (~flip & ~avg_builds, NEITHER, "No new solar under either rule"),
        (~flip & avg_builds, BOTH, "Same solar under both rules"),
        (flip, SERIES["marginal"], f"Marginal rule builds >{FLIP_THRESHOLD_MW:.0f} MW more solar"),
    ]
    for mask, colour, label in groups:
        if not mask.any():
            continue
        ax1.scatter(carbon_value[mask], lcoe[mask], s=26, color=colour, edgecolor="white", linewidth=0.8,
                    label=f"{label} ({mask.mean():.0%} of draws)")
    ax1.set_xlabel("Carbon value of displaced marginal imports (EUR/MWh)\n= carbon price x mean marginal intensity",
                   fontsize=9, color=INK2)
    ax1.set_ylabel("Solar LCOE, cheapest tier (EUR/MWh)", fontsize=9, color=INK2)
    ax1.set_title("Where the accounting rule changes the solar decision", fontsize=10, color=INK, loc="left")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, -0.34), ncol=1)
    _style(ax1)

    share = np.array([r["regret_share"] for r in rec]) * 100
    ax2.hist(share, bins=25, color=SERIES["average"], edgecolor="white", linewidth=1.5)
    med = np.median(share)
    ax2.axvline(med, color=INK, linewidth=1.2, linestyle="--")
    ax2.text(med, ax2.get_ylim()[1] * 0.95, f" median {med:.1f}%", fontsize=9, color=INK, va="top")
    ax2.set_xlabel("Regret of the average-rule plan (% of cost + carbon cost, marginal rule)", fontsize=9, color=INK2)
    ax2.set_ylabel("Draws", fontsize=9, color=INK2)
    ax2.set_title("How much the average-rule plan loses", fontsize=10, color=INK, loc="left")
    _style(ax2)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_intensity_profiles(year: HourlyProfiles, residual: np.ndarray, path: Path) -> None:
    """Average vs marginal import intensity by hour of day: the mechanism behind the result."""
    plt = _plt()
    hod = np.arange(year.n_hours) % 24
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    avg = [year.import_avg_g_per_kwh[hod == h].mean() for h in hours]
    ax.plot(hours, avg, color=SERIES["average"], linewidth=2, label="Average intensity of imports")
    styles = {"coal_cheaper": "-", "gas_cheaper": "--"}
    for regime, ls in styles.items():
        m = merit_order_marginal(residual, MeritOrderSettings(regime=regime))
        ax.plot(hours, [m[hod == h].mean() for h in hours], color=SERIES["marginal"], linewidth=2, linestyle=ls,
                label=f"Marginal intensity, {regime.replace('_', ' ')}")
    ax.set_xlabel("Hour of day", fontsize=9, color=INK2)
    ax.set_ylabel("g CO2 per kWh imported", fontsize=9, color=INK2)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, 24, 3))
    ax.set_title("Marginal imports are 2-3 times dirtier than the average, at every hour", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- run
def run(
    profiles_csv: str | None = None,
    n_days: int = 24,
    n_draws: int = 300,
    seed: int = 7,
    output: str = "outputs/luxembourg_case_results.json",
    figures_dir: str = "figures",
    with_robustness: bool = True,
) -> dict:
    if profiles_csv:
        year = load_profiles_csv(profiles_csv, existing_solar_mw=CENTRAL_PARAMS.existing_solar_mw)
    else:
        year = synthesise_profiles()
    rep = representative_days(year, n_days=n_days)
    cache = MarginalCache(year, rep)

    # Imperfect foresight: derate battery cost by the value retained on forecasts.
    central_prof = cache.profiles(MeritOrderSettings())
    first = cross_evaluate(central_prof, CENTRAL_PARAMS)["plans"]["marginal"]
    retention = battery_value_retention(central_prof, CENTRAL_PARAMS, first.new_solar_mw, first.battery_mwh)
    base = derated_parameters(CENTRAL_PARAMS, retention["retention"])

    central = summarise_cross_evaluation(cross_evaluate(central_prof, base))
    mc = monte_carlo(cache, base, n_draws=n_draws, seed=seed)
    full_marginal = merit_order_marginal(cache.residual, MeritOrderSettings())
    result = {
        "profile_source": year.source,
        "representative_days": n_days,
        "calibration": calibration_report(year) if not profiles_csv else None,
        "marginal_intensity_summary": {k: v for k, v in describe(full_marginal, year).items()},
        "foresight": retention,
        "central_parameters": asdict(base),
        "central": central,
        "monte_carlo": {k: v for k, v in mc.items() if k != "records"},
        "contract_loophole": contract_loophole(cache, base),
        "robustness": robustness(year, rep, cache, base, n_days) if with_robustness else None,
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out.with_name(out.stem + "_draws.json").write_text(json.dumps(mc["records"], indent=2), encoding="utf-8")
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_plans(central, fig_dir / "luxembourg_plans.svg")
    plot_monte_carlo(mc, fig_dir / "luxembourg_monte_carlo.svg")
    plot_intensity_profiles(year, cache.residual, fig_dir / "luxembourg_intensity_profiles.svg")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--profiles-csv", help="measured hourly profiles (e.g. from ghg-entsoe)")
    parser.add_argument("--days", type=int, default=24, help="number of representative days")
    parser.add_argument("--draws", type=int, default=300, help="Monte Carlo draws")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="outputs/luxembourg_case_results.json")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.profiles_csv, args.days, args.draws, args.seed, args.output, args.figures_dir,
                 not args.skip_robustness)
    c, mc = result["central"], result["monte_carlo"]
    print(f"Profiles: {result['profile_source']}")
    print(f"Battery value retained under forecast errors: {result['foresight']['retention']:.0%}")
    for rule in ACCOUNTING_RULES:
        p = c["plans"][rule]
        print(f"  {rule:8s} plan: +{p['new_solar_mw']:.0f} MW solar, {p['battery_mwh']:.0f} MWh battery")
    print(f"  regret of average-rule plan: EUR {c['regret_eur_per_year']:,.0f}/yr ({c['regret_share']:.1%})")
    print(f"Monte Carlo ({mc['n_draws']} draws): marginal rule builds more solar in "
          f"{mc['marginal_builds_more_solar_share']:.0%}, less in {mc['average_builds_more_solar_share']:.0%};"
          f" median regret {mc['regret_share']['median']:.1%}")
    if result["robustness"]:
        for r in result["robustness"]:
            print(f"  {r['check']}: solar {r['solar_average_mw']} vs {r['solar_marginal_mw']} MW, "
                  f"regret {r['regret_share']:.1%}")


if __name__ == "__main__":
    main()
