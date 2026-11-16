"""Luxembourg case study on measured hourly data (Energy-Charts 2024 and 2025).

Differences from the synthetic run in ``luxembourg_case``:

* load, solar shape, wind, other generation, prices, physical import flows
  (Germany, Belgium) and neighbour intensities are measured;
* the marginal factor is estimated from German generation data by regression
  (``energycharts.estimate_marginal``); the merit-order model, driven by the
  *measured* German residual load, is kept as an alternative;
* the central plan and the battery size are solved on the **full 8,760-hour
  year**, so storage is no longer sensitive to which representative days are
  picked;
* a plan made on one year is tested on the other year (out-of-sample).

The Monte Carlo still uses representative days for speed; it reports the solar
decision and regret, not battery size.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .energycharts import data_summary, estimate_marginal, profiles_from_csv, read_csv
from .foresight import battery_value_retention, derated_parameters
from .hourly_dispatch import ACCOUNTING_RULES, SystemParameters, cross_evaluate, objective_under, solve, with_overrides
from .luxembourg_case import MarginalCache, monte_carlo, plot_monte_carlo, plot_plans, summarise_cross_evaluation
from .luxembourg_profiles import HourlyProfiles, representative_days
from .marginal_emissions import MeritOrderSettings, merit_order_marginal
from .price_feedback import price_feedback_study
from .regional_marginal import read_neighbours, regional_estimate

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BORDER_CAPS = {"DE": 1000.0, "BE": 650.0}  # assumptions; measured 2025 peaks: DE 835 MW, BE 284 MW


def base_parameters(prof: HourlyProfiles) -> SystemParameters:
    return SystemParameters(
        existing_solar_mw=prof.metadata["existing_solar_mw"],
        marginal_source="profile",
        border_mode="flow_shares",
        border_import_cap_mw=dict(BORDER_CAPS),
        import_cap_mw=sum(BORDER_CAPS.values()),
    )


def plan_summary(ce: dict) -> dict:
    return summarise_cross_evaluation(ce)


def battery_curve(prof: HourlyProfiles, params: SystemParameters, capex_values=(80, 110, 150, 200, 250)) -> list[dict]:
    """Full-year battery size chosen by each rule as battery capex varies."""
    rows = []
    for capex in capex_values:
        p = with_overrides(params, battery_capex_eur_per_kwh=float(capex))
        row = {"battery_capex_eur_per_kwh": capex}
        for rule in ACCOUNTING_RULES:
            res = solve(prof, p, rule)
            row[f"battery_{rule}_mwh"] = round(res.battery_mwh)
            row[f"solar_{rule}_mw"] = round(res.new_solar_mw)
        rows.append(row)
    return rows


def out_of_sample(plan_prof: HourlyProfiles, test_prof: HourlyProfiles, params_plan: SystemParameters,
                  params_test: SystemParameters) -> dict:
    """Plan on one year, then operate both plans on another year and judge under marginal accounting."""
    plans = {rule: solve(plan_prof, params_plan, rule) for rule in ACCOUNTING_RULES}
    judged = {rule: solve(test_prof, params_test, "marginal", fixed_solar_mw=plans[rule].new_solar_mw,
                          fixed_battery_mwh=plans[rule].battery_mwh) for rule in ACCOUNTING_RULES}
    obj = {rule: objective_under(judged[rule], params_test, "marginal") for rule in ACCOUNTING_RULES}
    return {
        "plans": {rule: {"new_solar_mw": round(plans[rule].new_solar_mw), "battery_mwh": round(plans[rule].battery_mwh)}
                  for rule in ACCOUNTING_RULES},
        "regret_eur_per_year": round(obj["average"] - obj["marginal"]),
        "regret_share": round((obj["average"] - obj["marginal"]) / abs(obj["marginal"]), 5),
    }


def marginal_comparison(prof: HourlyProfiles) -> dict:
    """Estimated (regression) vs merit-order marginal factors on measured German residual load."""
    est = prof.marginal_g_per_kwh
    mo = {reg: merit_order_marginal(prof.zone_residual_mw, MeritOrderSettings(regime=reg))
          for reg in ("coal_cheaper", "gas_cheaper")}
    hod = np.arange(prof.n_hours) % 24
    def by_hour(x):
        return [round(float(x[hod == h].mean()), 1) for h in range(24)]
    return {
        "estimated": {"mean": float(est.mean()), "p10": float(np.percentile(est, 10)),
                      "p90": float(np.percentile(est, 90)), "hour_of_day": by_hour(est)},
        **{f"merit_order_{k}": {"mean": float(v.mean()), "corr_with_estimated": float(np.corrcoef(v, est)[0, 1])}
           for k, v in mo.items()},
        "average_import_intensity_mean": float(prof.import_avg_g_per_kwh.mean()),
        "average_import_by_hour": by_hour(prof.import_avg_g_per_kwh),
        "fit": prof.metadata.get("marginal_fit"),
    }


MARGINAL_VARIANTS = {
    "german_plants": "Lower bound: German plants only (trade and storage counted as zero-carbon)",
    "dispatchable": "Upper data-based estimate: emissions per MWh of German dispatchable response",
    "merit_order": "Stylised merit order on measured German residual load",
    "regional": "Central: regional (DE + 8 neighbours) emissions per MWh of German-zone demand, daily level, hourly shape",
}
MC_RANGES = {
    "marginal_level": None,                   # set per year to the regional estimate's range (g/kWh)
    "solar_lcoe_eur_per_mwh": (45.0, 90.0),
    "battery_capex_eur_per_kwh": (110.0, 250.0),
    "carbon_price_eur_per_t": (40.0, 200.0),
    "price_scale": (0.75, 1.25),
    "interconnector_scale": (0.7, 1.3),
}


def neighbour_csv(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.name.replace("energycharts_lu_de_", "energycharts_neighbours_"))


def variant_profiles(csv_path: Path) -> dict[str, HourlyProfiles]:
    lower = profiles_from_csv(csv_path, method="german_plants")
    upper = profiles_from_csv(csv_path, method="dispatchable")
    mo = lower.with_marginal(merit_order_marginal(lower.zone_residual_mw), "merit order on measured residual")
    out = {"german_plants": lower, "dispatchable": upper, "merit_order": mo}
    nb_path = neighbour_csv(csv_path)
    if nb_path.exists():
        series, fit = regional_estimate(read_csv(csv_path), read_neighbours(nb_path))
        reg = lower.with_marginal(series, "regional estimate (DE + 8 neighbours)")
        reg.metadata["marginal_fit"] = fit
        reg.metadata["marginal_method"] = "regional"
        out["regional"] = reg
    return out


def level_monte_carlo(rep: HourlyProfiles, level_range: tuple[float, float], params: SystemParameters,
                      n_draws: int, seed: int) -> dict:
    """Monte Carlo over the regional marginal level (its range across specifications) and cost/policy inputs."""
    from .luxembourg_case import _spearman
    rng = np.random.default_rng(seed)
    ranges = dict(MC_RANGES, marginal_level=tuple(float(x) for x in level_range))
    names = list(ranges)
    base_mean = float(np.average(rep.marginal_g_per_kwh, weights=rep.hour_weights))
    records = []
    for _ in range(n_draws):
        row = {k: float(rng.uniform(*ranges[k])) for k in names}
        prof = rep.with_marginal(rep.marginal_g_per_kwh * row["marginal_level"] / base_mean,
                                 f"regional level {row['marginal_level']:.0f}")
        caps = {z: v * row["interconnector_scale"] for z, v in params.border_import_cap_mw.items()}
        p = with_overrides(params, solar_lcoe_eur_per_mwh=row["solar_lcoe_eur_per_mwh"],
                           battery_capex_eur_per_kwh=row["battery_capex_eur_per_kwh"],
                           carbon_price_eur_per_t=row["carbon_price_eur_per_t"], price_scale=row["price_scale"],
                           border_import_cap_mw=caps, import_cap_mw=params.import_cap_mw * row["interconnector_scale"])
        ce = cross_evaluate(prof, p)
        records.append({**row,
                        "mean_marginal_g_per_kwh": float(np.average(prof.marginal_g_per_kwh, weights=prof.hour_weights)),
                        "mean_average_g_per_kwh": float(np.average(prof.import_avg_g_per_kwh, weights=prof.hour_weights)),
                        "solar_average_mw": ce["plans"]["average"].new_solar_mw,
                        "solar_marginal_mw": ce["plans"]["marginal"].new_solar_mw,
                        "regret_eur_per_year": ce["regret_eur_per_year"], "regret_share": ce["regret_share"]})
    gap = np.array([r["solar_marginal_mw"] - r["solar_average_mw"] for r in records])
    share = np.array([r["regret_share"] for r in records])
    ratio = np.array([r["mean_marginal_g_per_kwh"] / r["mean_average_g_per_kwh"] for r in records])
    # Blend level at which the marginal rule starts building more solar (logistic-free: median split)
    more = gap > 100
    less = gap < -100
    return {
        "n_draws": n_draws, "seed": seed, "ranges": ranges,
        "marginal_builds_more_solar_share": float(np.mean(more)),
        "marginal_builds_less_solar_share": float(np.mean(less)),
        "same_solar_share": float(np.mean(~more & ~less)),
        "more_share_when_marginal_above_average": float(np.mean(more[ratio > 1])) if np.any(ratio > 1) else None,
        "less_share_when_marginal_below_average": float(np.mean(less[ratio < 1])) if np.any(ratio < 1) else None,
        "regret_share": {"median": float(np.median(share)), "p05": float(np.percentile(share, 5)),
                         "p95": float(np.percentile(share, 95))},
        "spearman_with_solar_gap": {k: round(_spearman(np.array([r[k] for r in records]), gap), 3) for k in names},
        "records": records,
    }


def ROBUST_VARIANTS(variants):
    return ("regional",) if "regional" in variants else ("german_plants", "dispatchable")


def _row(label, ce):
    p = ce["plans"]
    return {"check": label, "solar_average_mw": round(p["average"].new_solar_mw),
            "solar_marginal_mw": round(p["marginal"].new_solar_mw),
            "battery_average_mwh": round(p["average"].battery_mwh),
            "battery_marginal_mwh": round(p["marginal"].battery_mwh),
            "regret_share": round(ce["regret_share"], 4)}


def run_year(year: int, csv_path: Path, n_days: int, n_draws: int, seed: int, fig_dir: Path | None,
             log=print) -> tuple[dict, dict, SystemParameters]:
    variants = variant_profiles(csv_path)
    central = variants.get("regional", variants["dispatchable"])
    upper = central
    params0 = base_parameters(upper)
    first = solve(upper, params0, "marginal")
    retention = battery_value_retention(upper, params0, first.new_solar_mw, max(first.battery_mwh, 50.0), seeds=(3,))
    params = derated_parameters(params0, retention["retention"])
    log(f"{year}: battery value retained under forecasts {retention['retention']:.0%}")
    by_variant = {}
    for name, prof in variants.items():
        ce = cross_evaluate(prof, params)
        by_variant[name] = {"description": MARGINAL_VARIANTS[name],
                            "mean_marginal_g_per_kwh": float(prof.marginal_g_per_kwh.mean()),
                            **plan_summary(ce)}
        pl = ce["plans"]
        log(f"{year} {name}: solar {pl['average'].new_solar_mw:.0f} vs {pl['marginal'].new_solar_mw:.0f} MW, "
            f"battery {pl['average'].battery_mwh:.0f} vs {pl['marginal'].battery_mwh:.0f} MWh, regret {ce['regret_share']:.2%}")
    robustness = []
    for label, p in [("No battery allowed", with_overrides(params, max_battery_mwh=0.0)),
                     ("Carbon price EUR 40/t", with_overrides(params, carbon_price_eur_per_t=40.0)),
                     ("Carbon price EUR 150/t", with_overrides(params, carbon_price_eur_per_t=150.0)),
                     ("One aggregated import, no border limits", with_overrides(params, border_mode="aggregate")),
                     ("Borders chosen freely (DE or BE)", with_overrides(params, border_mode="free"))]:
        for name in ROBUST_VARIANTS(variants):
            robustness.append(_row(f"{label} - {name}", cross_evaluate(variants[name], p)))
    rep = representative_days(central, n_days=n_days)
    for name, rp in (("regional" if "regional" in variants else "dispatchable", rep),):
        robustness.append(_row(f"{n_days} representative days instead of full year - {name}", cross_evaluate(rp, params)))
    log(f"{year}: robustness done")
    curve = battery_curve(upper, params)
    log(f"{year}: battery curve done")
    fit = central.metadata.get("marginal_fit", {})
    level_range = fit.get("range_kg_per_mwh") or [float(variants["german_plants"].marginal_g_per_kwh.mean()),
                                                  float(variants["dispatchable"].marginal_g_per_kwh.mean())]
    mc = level_monte_carlo(rep, level_range, params, n_draws=n_draws, seed=seed)
    log(f"{year}: Monte Carlo done")
    feedback = price_feedback_study(central, read_csv(csv_path), params, log=log)
    log(f"{year}: price feedback done")
    result = {
        "year": year,
        "data_summary": data_summary(csv_path),
        "marginal_factors": {name: marginal_comparison(prof) for name, prof in variants.items()},
        "foresight": retention,
        "central_parameters": asdict(params),
        "by_marginal_estimate_full_year": by_variant,
        "battery_curve_full_year": curve,
        "monte_carlo": {k: v for k, v in mc.items() if k != "records"},
        "robustness": robustness,
        "price_feedback": feedback,
    }
    if fig_dir is not None:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"luxembourg_real_{year}_draws.json").write_text(json.dumps(mc["records"]), encoding="utf-8")
    return result, variants, params


def plot_real_intensity(results: dict, path: Path) -> None:
    from .luxembourg_case import INK, INK2, SERIES, _plt, _style
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    hours = np.arange(24)
    styles = {2024: "--", 2025: "-"}
    year = max(results)
    mf = results[year]["marginal_factors"]
    ax.plot(hours, mf["dispatchable"]["average_import_by_hour"], color=SERIES["average"], linewidth=2,
            label="Average intensity of imports")
    ax.plot(hours, mf["german_plants"]["estimated"]["hour_of_day"], color="#b5b3ac", linewidth=1.4,
            linestyle="--", label="Old lower bound: German plants only")
    ax.plot(hours, mf["dispatchable"]["estimated"]["hour_of_day"], color="#b5b3ac", linewidth=1.4,
            linestyle=":", label="Old upper bound: German dispatchable response")
    if "regional" in mf:
        reg = np.array(mf["regional"]["estimated"]["hour_of_day"])
        lo, hi = mf["regional"]["fit"]["range_kg_per_mwh"]
        mean = mf["regional"]["estimated"]["mean"]
        ax.fill_between(hours, reg * lo / mean, reg * hi / mean, color=SERIES["marginal"], alpha=0.18, linewidth=0,
                        label="Regional estimate, range across methods")
        ax.plot(hours, reg, color=SERIES["marginal"], linewidth=2.2,
                label="Marginal: regional estimate (Germany + 8 neighbours)")
    ax.set_xlabel("Hour of day (UTC)", fontsize=9, color=INK2)
    ax.set_ylabel("g CO2 per kWh", fontsize=9, color=INK2)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, 24, 3))
    ax.set_title(f"Measured {year}: following German demand across borders narrows the marginal factor",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def run(years=(2024, 2025), n_days: int = 36, n_draws: int = 200, seed: int = 7,
        output: str = "outputs/luxembourg_real_results.json", figures_dir: str = "figures") -> dict:
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results, profs, params = {}, {}, {}
    for year in years:
        path = DATA_DIR / f"energycharts_lu_de_{year}_hourly.csv"
        if not path.exists():
            continue
        results[year], profs[year], params[year] = run_year(year, path, n_days, n_draws, seed, fig_dir)
    cross = {}
    ys = sorted(profs)
    for a in ys:
        for b in ys:
            if a != b:
                for name in ROBUST_VARIANTS(profs[a]):
                    cross[f"plan_{a}_test_{b}_{name}"] = out_of_sample(profs[a][name], profs[b][name], params[a], params[b])
                    print(f"out-of-sample {a}->{b} {name}: {cross[f'plan_{a}_test_{b}_{name}']}", flush=True)
    out = {"years": {str(k): v for k, v in results.items()}, "out_of_sample": cross}
    if results:
        plot_real_intensity(results, fig_dir / "luxembourg_real_intensity.svg")
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Luxembourg case on measured Energy-Charts data")
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--days", type=int, default=36)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--output", default="outputs/luxembourg_real_results.json")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args(argv)
    out = run(tuple(args.years), args.days, args.draws, output=args.output, figures_dir=args.figures_dir)
    redraw_figures(args.output, args.figures_dir)
    for year, res in out["years"].items():
        mc = res["monte_carlo"]
        print(f"{year} MC: marginal rule builds more solar in {mc['marginal_builds_more_solar_share']:.0%}, "
              f"less in {mc['marginal_builds_less_solar_share']:.0%}; median regret {mc['regret_share']['median']:.1%}")


def plot_real_figures(results_path: str = "outputs/luxembourg_real_results.json", figures_dir: str = "figures") -> None:
    """Battery break-even curve and Monte Carlo decision map from saved real-data results."""
    from .luxembourg_case import INK, INK2, SERIES, _plt, _style
    plt = _plt()
    res = json.loads(Path(results_path).read_text(encoding="utf-8"))
    fig_dir = Path(figures_dir)

    # Battery size vs capex, both years (average-rule plan; marginal-rule plan is close)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    styles = {"2024": "--", "2025": "-"}
    for year, r in sorted(res["years"].items()):
        curve = r["battery_curve_full_year"]
        x = [c["battery_capex_eur_per_kwh"] for c in curve]
        ax.plot(x, [c["battery_average_mwh"] / 1000 for c in curve], color=SERIES["average"], linestyle=styles[year],
                linewidth=2, marker="o", markersize=5, label=f"Planned with average rule, {year}")
        ax.plot(x, [c["battery_marginal_mwh"] / 1000 for c in curve], color=SERIES["marginal"], linestyle=styles[year],
                linewidth=2, marker="o", markersize=5, label=f"Planned with marginal rule, {year}")
    ax.set_xlabel("Battery installed cost (EUR/kWh)", fontsize=9, color=INK2)
    ax.set_ylabel("Battery chosen (GWh)", fontsize=9, color=INK2)
    ax.set_title("Battery size is set by its cost, not by the accounting rule (full-year plans)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "luxembourg_real_battery_curve.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    # Monte Carlo decision map, both years
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (year, _) in zip(axes, sorted(res["years"].items())):
        rec = json.loads((Path("outputs") / f"luxembourg_real_{year}_draws.json").read_text(encoding="utf-8"))
        ratio = np.array([d["mean_marginal_g_per_kwh"] / d["mean_average_g_per_kwh"] for d in rec])
        cp = np.array([d["carbon_price_eur_per_t"] for d in rec])
        gap = np.array([d["solar_marginal_mw"] - d["solar_average_mw"] for d in rec])
        more = gap > 100
        ax.scatter(ratio[~more], cp[~more], s=24, color="#b5b3ac", edgecolor="white", linewidth=0.7,
                   label=f"Same solar under both rules ({(~more).mean():.0%})")
        ax.scatter(ratio[more], cp[more], s=24, color=SERIES["marginal"], edgecolor="white", linewidth=0.7,
                   label=f"Marginal rule builds >100 MW more ({more.mean():.0%})")
        ax.axvline(1.0, color=INK2, linewidth=1, linestyle=":")
        ax.text(1.01, 195, "marginal = average", fontsize=8, color=INK2, va="top")
        ax.set_title(f"{year} (measured data)", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Marginal / average intensity of imports", fontsize=9, color=INK2)
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.22))
        _style(ax)
    axes[0].set_ylabel("Carbon price (EUR/t)", fontsize=9, color=INK2)
    fig.suptitle("With the regional marginal factor, imports at the margin are 7-55% dirtier than average: the marginal rule often adds solar, never removes it",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "luxembourg_real_monte_carlo.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_price_feedback(res: dict, path: Path) -> None:
    """Spread, Luxembourg battery and solar as zone-wide storage grows."""
    from .luxembourg_case import INK, INK2, SERIES, _plt, _style
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    styles = {"2024": "--", "2025": "-"}
    for year, r in sorted(res["years"].items()):
        rows = r["price_feedback"]["rows"]
        x = [row["zone_fleet_gwh"] for row in rows]
        axes[0].plot(x, [row["mean_daily_spread_eur"] for row in rows], color=INK2, linestyle=styles[year],
                     marker="o", markersize=4, linewidth=1.8, label=year)
        for rule in ("average", "marginal"):
            axes[1].plot(x, [row[f"battery_{rule}_mwh"] / 1000 for row in rows], color=SERIES[rule],
                         linestyle=styles[year], marker="o", markersize=4, linewidth=1.8, label=f"{rule.capitalize()} rule, {year}")
            axes[2].plot(x, [row[f"solar_{rule}_mw"] for row in rows], color=SERIES[rule],
                         linestyle=styles[year], marker="o", markersize=4, linewidth=1.8, label=f"{rule.capitalize()} rule, {year}")
    axes[0].set_title("Mean daily price spread (EUR/MWh)", fontsize=10, color=INK, loc="left")
    axes[1].set_title("Luxembourg battery chosen (GWh)", fontsize=10, color=INK, loc="left")
    axes[2].set_title("New Luxembourg solar (MW)", fontsize=10, color=INK, loc="left")
    for ax in axes:
        ax.set_xlabel("Storage added across DE-LU (GWh)", fontsize=9, color=INK2)
        ax.set_ylim(bottom=0)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("Zone-wide storage compresses the spread: Luxembourg's own battery stops paying, solar gains",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def redraw_figures(results_path: str = "outputs/luxembourg_real_results.json", figures_dir: str = "figures") -> None:
    res = json.loads(Path(results_path).read_text(encoding="utf-8"))
    fig_dir = Path(figures_dir)
    plot_real_intensity({int(k): v for k, v in res["years"].items()}, fig_dir / "luxembourg_real_intensity.svg")
    plot_real_figures(results_path, figures_dir)
    if all("price_feedback" in v for v in res["years"].values()):
        plot_price_feedback(res, fig_dir / "luxembourg_real_price_feedback.svg")


if __name__ == "__main__":
    main()
