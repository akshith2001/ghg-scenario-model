"""Regional marginal emission factor: follow German demand across the border.

``energycharts.estimate_marginal`` brackets the consequential factor for
DE-LU demand between two estimates. The lower bound counts only German plants,
although 72% of a change in German residual load is met through trade and
pumped storage. The upper estimate assumes that the part met abroad is as
carbon-intensive as German thermal plants.

This module closes the gap with data from the eight neighbours that trade most
with Germany (FR, BE, NL, AT, CH, CZ, PL, DK). It regresses the hour-to-hour
change in **regional** emissions (Germany plus neighbours) on the change in
German residual load and **controls for each neighbour's own residual load**,
because wind and sun move together across borders. The coefficient on German
residual load is the extra regional CO2 per extra MWh of German-zone demand,
wherever it is generated. The same regression on each country's emissions
splits that factor by country, and on each country's generation shows how much
of the response the eight neighbours cover.

Pumped storage is left as a response: it shifts energy in time, and the
emissions of refilling it appear in other hours. Countries beyond the ring
(Nordic hydro, Italy, the Iberian peninsula) are not observed, so the share of
the response they meet is counted as zero-carbon. The coverage diagnostic says
how large that share is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NEIGHBOURS = ("fr", "be", "nl", "at", "ch", "cz", "pl", "dk")
FIELDS = ("load", "solar", "wind", "emis_t", "gen", "trade")


def read_neighbours(path: str | Path) -> dict[str, np.ndarray]:
    """Columns ``<country>_<field>``, 8,760 rows, as written by ``write_neighbours``."""
    with Path(path).open(encoding="utf-8") as handle:
        lines = [ln for ln in handle if not ln.startswith("#")]
    header = lines[0].strip().split(",")
    rows = np.array([[float(x) for x in ln.strip().split(",")] for ln in lines[1:] if ln.strip()])
    return {name: rows[:, i] for i, name in enumerate(header) if name != "hour"}


def write_neighbours(tables: dict[str, dict[str, np.ndarray]], path: str | Path, year: int) -> None:
    """Write per-country hourly tables (dropping 29 February in leap years) to one CSV."""
    cols = [f"{c}_{f}" for c in NEIGHBOURS for f in FIELDS]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# Source: energy-charts.info (Fraunhofer ISE), CC BY 4.0; public_power hourly means, UTC, {year}; "
                     "emis_t from direct combustion factors as in energycharts.FACTORS.\n")
        handle.write("hour," + ",".join(cols) + "\n")
        for h in range(8760):
            vals = [tables[c][f][h] for c in NEIGHBOURS for f in FIELDS]
            handle.write(f"{h}," + ",".join(f"{round(float(v) * 10) / 10:g}" for v in vals) + "\n")


def drop_leap_day(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.concatenate([x[:1416], x[1440:]]) if len(x) == 8784 else x


def _design(d: dict, nb: dict, countries=NEIGHBOURS) -> tuple[np.ndarray, np.ndarray]:
    rl_de = d["de_load"] - d["de_solar"] - d["de_wind"]
    own = [nb[f"{c}_load"] - nb[f"{c}_solar"] - nb[f"{c}_wind"] for c in countries]
    X = np.column_stack([rl_de] + own)
    return np.diff(X, axis=0), 0.5 * (rl_de[1:] + rl_de[:-1])


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _daily_fit(d: dict, nb: dict, countries, lag: int, n_boot: int, seed: int) -> dict:
    D = lambda x: np.asarray(x)[: len(x) // 24 * 24].reshape(-1, 24).sum(1)  # noqa: E731
    rl_de = d["de_load"] - d["de_solar"] - d["de_wind"]
    X = np.column_stack([D(rl_de)] + [D(nb[f"{c}_load"] - nb[f"{c}_solar"] - nb[f"{c}_wind"]) for c in countries])
    dX = X[lag:] - X[:-lag]
    diff = lambda v: D(v)[lag:] - D(v)[:-lag]  # noqa: E731
    y = diff(d["de_emis_t"] + sum(nb[f"{c}_emis_t"] for c in countries)) * 1000.0
    beta = _ols(dX, y)[0]
    rng = np.random.default_rng(seed)
    boot = [_ols(dX[i], y[i])[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    coverage = {}
    if "de_trade" in d and "de_pumped" in d:
        coverage["de_dispatchable"] = float(_ols(dX, diff(rl_de - d["de_trade"] - d["de_pumped"]))[0])
        coverage["de_pumped_storage"] = float(_ols(dX, diff(d["de_pumped"]))[0])
    for c in countries:
        coverage[c] = float(_ols(dX, diff(nb[f"{c}_gen"]))[0])
    coverage["observed_total"] = float(sum(coverage.values()))
    return {"kg_per_mwh": float(beta), "ci95_kg_per_mwh": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "de_only_kg_per_mwh": float(_ols(dX, diff(d["de_emis_t"]) * 1000.0)[0]),
            "by_country_kg_per_mwh": {c: float(_ols(dX, diff(nb[f"{c}_emis_t"]) * 1000.0)[0]) for c in countries},
            "coverage_mw_per_mw": coverage}


def regional_estimate(d: dict, nb: dict, n_bins: int = 10, n_boot: int = 200, seed: int = 1,
                      countries=NEIGHBOURS) -> tuple[np.ndarray, dict]:
    """Hourly regional marginal factor (kg/MWh) for DE-LU demand, with diagnostics.

    The **level** comes from daily totals (day-to-day differences): within a day,
    pumped storage and reservoir hydro move energy between hours, so an hourly
    regression counts their response as zero-carbon and misses the emissions of
    refilling them later. The **hour-to-hour shape** (which hours have dirtier
    marginal supply) comes from an hourly regression by German residual-load
    bin, rescaled to the daily level. The reported range spans both daily
    specifications (day-to-day and same-weekday 7-day differences) with
    bootstrap 95% intervals.
    """
    dX, level = _design(d, nb, countries)
    e_region = d["de_emis_t"] + sum(nb[f"{c}_emis_t"] for c in countries)
    dy = np.diff(e_region) * 1000.0  # kg
    within_hour = _ols(dX, dy)[0]

    by_country_h = {"de": float(_ols(dX, np.diff(d["de_emis_t"]) * 1000.0)[0])}
    for c in countries:
        by_country_h[c] = float(_ols(dX, np.diff(nb[f"{c}_emis_t"]) * 1000.0)[0])
    rl_de = d["de_load"] - d["de_solar"] - d["de_wind"]
    coverage_h = {}
    if "de_trade" in d and "de_pumped" in d:
        coverage_h["de_dispatchable"] = float(_ols(dX, np.diff(rl_de - d["de_trade"] - d["de_pumped"]))[0])
        coverage_h["de_pumped_storage"] = float(_ols(dX, np.diff(d["de_pumped"]))[0])
    for c in countries:
        coverage_h[c] = float(_ols(dX, np.diff(nb[f"{c}_gen"]))[0])
    coverage_h["observed_total"] = float(sum(coverage_h.values()))

    edges = np.quantile(level, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, level, side="right") - 1, 0, n_bins - 1)
    slopes = np.array([float(_ols(dX[bins == b], dy[bins == b])[0]) if (bins == b).sum() > 50 else float(within_hour)
                       for b in range(n_bins)])
    slopes = np.clip(slopes, 0.0, None)

    day1 = _daily_fit(d, nb, countries, 1, n_boot, seed)
    day7 = _daily_fit(d, nb, countries, 7, n_boot, seed + 1)
    level_kg = day1["kg_per_mwh"]
    bin_of_hour = np.clip(np.searchsorted(edges, rl_de, side="right") - 1, 0, n_bins - 1)
    shape = slopes[bin_of_hour]
    hourly = shape * (level_kg / shape.mean()) if shape.mean() > 0 else np.full(len(rl_de), level_kg)
    lo = min(day1["ci95_kg_per_mwh"][0], day7["ci95_kg_per_mwh"][0])
    hi = max(day1["ci95_kg_per_mwh"][1], day7["ci95_kg_per_mwh"][1])
    return hourly, {
        "overall_kg_per_mwh": float(level_kg),
        "range_kg_per_mwh": [float(lo), float(hi)],
        "daily_day_to_day": day1,
        "daily_7_day": day7,
        "within_hour": {"kg_per_mwh": float(within_hour), "by_country_kg_per_mwh": by_country_h,
                        "coverage_mw_per_mw": coverage_h,
                        "bin_edges_mw": edges.tolist(), "bin_slopes_kg_per_mwh": slopes.tolist()},
        "countries": list(countries),
    }
