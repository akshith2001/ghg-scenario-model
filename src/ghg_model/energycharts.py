"""Load measured 2025 hourly data from the Energy-Charts API (Fraunhofer ISE).

The CSV is built from public, key-free Energy-Charts endpoints (licence
CC BY 4.0, attribution energy-charts.info):

* ``/v2/public_power?country=lu|de|fr|be`` - load and generation by type
* ``/v2/cbpf?country=lu`` - physical cross-border flows (Luxembourg's public
  grid connects physically to Germany and Belgium only)
* ``/v2/price?bzn=DE-LU`` - day-ahead price

Fifteen-minute values are averaged to UTC hours. ``scripts/energycharts_browser.js``
reproduces the CSV in a web browser; :func:`fetch_csv` does the same with
the standard library where the API is reachable.

Scope note
----------
Energy-Charts load for Luxembourg is the public (Creos) grid: 4.84 TWh in
2025, against 6.53 TWh gross consumption reported by ILR. The difference is
most likely the separate industrial network that supplies heavy industry
and is operated with the Belgian grid. Measured imports from Germany and
Belgium match the measured trade balance, so the public grid is internally
consistent and is what this module models.

Metered solar (429 GWh) is below ILR's 627 GWh, consistent with unmetered
self-consumption. New solar therefore uses the measured hourly *shape*,
scaled to ILR's specific yield, and existing solar is represented by its
metered output.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from .luxembourg_profiles import HourlyProfiles, load_inputs
from .marginal_emissions import estimate_marginal_factors

COLUMNS = ("lu_load", "lu_solar", "lu_wind", "lu_other", "price", "flow_de", "flow_be",
           "de_load", "de_solar", "de_wind", "de_emis_t", "int_de", "int_fr", "int_be")
OPTIONAL_COLUMNS = ("de_trade", "de_pumped")  # German net imports (+) and pumped-storage net output, MW
DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "energycharts_lu_de_2025_hourly.csv"


def read_csv(path: str | Path = DEFAULT_CSV) -> dict[str, np.ndarray]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [line for line in handle if not line.startswith("#")]
    reader = csv.DictReader(rows)
    present = [c for c in OPTIONAL_COLUMNS if c in (reader.fieldnames or [])]
    data = {c: [] for c in COLUMNS + tuple(present)}
    for r in reader:
        for c in data:
            data[c].append(float(r[c]))
    out = {c: np.array(v) for c, v in data.items()}
    if len(out["lu_load"]) != 8760:
        raise ValueError(f"expected 8760 hours, got {len(out['lu_load'])}")
    return out


def ilr_specific_yield_cf(inputs: dict | None = None) -> float:
    """Annual solar capacity factor implied by ILR 2025 generation and mean capacity."""
    v = inputs or load_inputs()
    cap_end = v["solar_capacity_end_year_mw"]
    cap_mean = 0.5 * (cap_end + cap_end / 1.358)
    return v["solar_generation_gwh"] * 1000.0 / (cap_mean * 8760.0)


MARGINAL_METHODS = ("german_plants", "dispatchable")


def estimate_marginal(d: dict[str, np.ndarray], n_bins: int = 10, method: str = "german_plants") -> tuple[np.ndarray, dict]:
    """Hourly marginal factors (kg/MWh) for DE-LU from measured German data.

    ``"german_plants"`` regresses German emissions on German residual load
    (load - wind - solar). Part of each change in residual load is met by
    cross-border trade and pumped storage, which this method counts as
    zero-carbon, so it is a **lower bound** on the consequential factor.

    ``"dispatchable"`` regresses German emissions on the change in German
    dispatchable output (residual load - net imports - pumped-storage output).
    It gives the emissions per MWh of thermal response, and equals the
    consequential factor if the response met abroad is as carbon-intensive as
    Germany's - an **upper data-based estimate**. Needs ``de_trade`` and ``de_pumped``.
    """
    residual = d["de_load"] - d["de_solar"] - d["de_wind"]
    if method == "german_plants":
        x = residual
    elif method == "dispatchable":
        if "de_trade" not in d or "de_pumped" not in d:
            raise ValueError("method 'dispatchable' needs de_trade and de_pumped columns")
        x = residual - d["de_trade"] - d["de_pumped"]
    else:
        raise ValueError(f"unknown method {method!r}; use one of {MARGINAL_METHODS}")
    est = estimate_marginal_factors(d["de_emis_t"] * 1000.0, x, n_bins=n_bins)
    return est["hourly"], {k: v for k, v in est.items() if k != "hourly"}


def profiles_from_csv(path: str | Path = DEFAULT_CSV, marginal: str = "estimated",
                      method: str = "german_plants") -> HourlyProfiles:
    """Build model profiles from the Energy-Charts CSV.

    ``marginal`` = ``"estimated"`` attaches the regression estimate from German
    data; ``"none"`` leaves it for the merit-order model (which then uses the
    measured German residual load).
    """
    d = read_csv(path)
    cf_target = ilr_specific_yield_cf()
    solar_shape = d["lu_solar"]
    equivalent_mw = solar_shape.sum() / (cf_target * 8760.0)   # metered fleet expressed in MW at ILR yield
    solar_cf = np.clip(solar_shape / equivalent_mw, 0.0, 1.0)
    imp_de = np.clip(d["flow_de"], 0, None)
    imp_be = np.clip(d["flow_be"], 0, None)
    total = imp_de + imp_be
    mix = np.where(total > 1e-6, (imp_de * d["int_de"] + imp_be * d["int_be"]) / np.maximum(total, 1e-6),
                   0.5 * (d["int_de"] + d["int_be"]))
    shares = {"DE": float(imp_de.sum() / total.sum()), "BE": float(imp_be.sum() / total.sum())}
    residual = d["de_load"] - d["de_solar"] - d["de_wind"]
    prof = HourlyProfiles(
        load_mw=d["lu_load"],
        solar_cf=solar_cf,
        wind_mw=d["lu_wind"],
        other_domestic_mw=d["lu_other"],
        price_eur_per_mwh=d["price"],
        import_avg_g_per_kwh=mix,
        solar_index=solar_shape / solar_shape.max(),
        source="Energy-Charts 2025 (measured, CC BY 4.0)",
        metadata={"existing_solar_mw": float(equivalent_mw), "solar_cf_target": cf_target},
        border_intensity={"DE": d["int_de"], "BE": d["int_be"]},
        border_shares=shares,
        zone_residual_mw=residual,
    )
    if marginal == "estimated":
        series, fit = estimate_marginal(d, method=method)
        prof = prof.with_marginal(series, f"estimated from Energy-Charts German data ({method})")
        prof.metadata["marginal_fit"] = fit
        prof.metadata["marginal_method"] = method
    return prof


def tou_shape(path: str | Path = DEFAULT_CSV, kind: str = "marginal", utc_offset_h: int = 1) -> np.ndarray:
    """Measured time-of-day factor shape (24 values, mean 1) in local standard time.

    ``kind`` = ``"marginal"`` uses the regression-estimated marginal factor;
    ``"average"`` uses the flow-weighted average intensity of imports. Intended for
    ``synthetic_grid.daily_emissions_kg(..., shape=...)``.
    """
    prof = profiles_from_csv(path, marginal="estimated" if kind == "marginal" else "none")
    series = prof.marginal_g_per_kwh if kind == "marginal" else prof.import_avg_g_per_kwh
    hod = (np.arange(prof.n_hours) + utc_offset_h) % 24
    by_hour = np.array([series[hod == h].mean() for h in range(24)])
    return by_hour / by_hour.mean()


def data_summary(path: str | Path = DEFAULT_CSV) -> list[dict]:
    """Compare the measured series with published annual statistics."""
    d = read_csv(path)
    v = load_inputs()
    imports = np.clip(d["flow_de"], 0, None) + np.clip(d["flow_be"], 0, None)
    rows = [
        ("Luxembourg public-grid load (GWh)", d["lu_load"].sum() / 1000, v["gross_consumption_gwh"], "ILR gross consumption, incl. industrial network"),
        ("Solar, metered (GWh)", d["lu_solar"].sum() / 1000, v["solar_generation_gwh"], "ILR, incl. self-consumption"),
        ("Wind (GWh)", d["lu_wind"].sum() / 1000, v["wind_generation_gwh"], "ILR"),
        ("Physical imports from DE (GWh)", np.clip(d["flow_de"], 0, None).sum() / 1000,
         v["imports_by_neighbour_twh"]["DE"] * 1000, "Electricity Maps, commercial"),
        ("Physical imports from BE (GWh)", np.clip(d["flow_be"], 0, None).sum() / 1000,
         v["imports_by_neighbour_twh"]["BE"] * 1000, "Electricity Maps, commercial"),
        ("Mean day-ahead price (EUR/MWh)", d["price"].mean(), v["day_ahead_price_eur_per_mwh"]["mean"], "Electricity Maps"),
        ("Negative-price hours", float((d["price"] < 0).sum()), v["day_ahead_price_eur_per_mwh"]["negative_price_hours"], "Electricity Maps"),
        ("German average intensity, direct (g/kWh)", float(np.average(d["int_de"])), v["neighbour_average_intensity_g_per_kwh"]["DE"], "Electricity Maps, life-cycle"),
        ("Belgian average intensity, direct (g/kWh)", float(np.average(d["int_be"])), v["neighbour_average_intensity_g_per_kwh"]["BE"], "Electricity Maps, life-cycle"),
        ("French average intensity, direct (g/kWh)", float(np.average(d["int_fr"])), v["neighbour_average_intensity_g_per_kwh"]["FR"], "Electricity Maps, life-cycle"),
        ("Peak public-grid load (MW)", d["lu_load"].max(), None, ""),
        ("Peak import (MW)", imports.max(), None, ""),
    ]
    return [{"statistic": a, "measured": round(float(b), 1), "published": c, "published_basis": e} for a, b, c, e in rows]


def fetch_csv(output: str | Path = DEFAULT_CSV, year: int = 2025, pause_s: float = 35.0) -> Path:
    """Download the CSV directly (standard library). Respects ~2 requests/minute."""
    base = "https://api.energy-charts.info"
    span = f"start={year}-01-01T00:00Z&end={year + 1}-01-01T00:00Z"
    urls = {
        "lu": f"{base}/v2/public_power?country=lu&{span}", "de": f"{base}/v2/public_power?country=de&{span}",
        "fr": f"{base}/v2/public_power?country=fr&{span}", "be": f"{base}/v2/public_power?country=be&{span}",
        "cbpf": f"{base}/v2/cbpf?country=lu&{span}", "price": f"{base}/v2/price?bzn=DE-LU&{span}",
    }
    raw = {}
    for i, (key, url) in enumerate(urls.items()):
        if i:
            time.sleep(pause_s)
        with urllib.request.urlopen(url, timeout=300) as resp:
            raw[key] = json.loads(resp.read().decode("utf-8"))
    table = build_table(raw, year)
    write_table(table, output, year)
    return Path(output)


FACTORS = {"fossil_brown_coal_lignite": 1100, "fossil_hard_coal": 850, "fossil_coal_derived_gas": 1000,
           "fossil_gas": 370, "fossil_oil": 750, "waste": 330, "others": 400}
NOT_GENERATION = {"load", "residual_load", "cross_border_electricity_trading", "renewable_share_of_load",
                  "renewable_share_of_generation", "hydro_pumped_storage_consumption", "battery_consumption"}


def _hourly(doc: dict, fn, year: int) -> np.ndarray:
    from datetime import datetime, timezone
    t0 = datetime(year, 1, 1, tzinfo=timezone.utc).timestamp()
    s, c = np.zeros(8760), np.zeros(8760)
    for row in doc["data"]:
        h = int((datetime.fromisoformat(row["timestamp"]).timestamp() - t0) // 3600)
        if 0 <= h < 8760:
            v = fn(row["values"])
            if v is not None and np.isfinite(v):
                s[h] += v
                c[h] += 1
    out = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    for h in range(8760):  # carry forward across gaps, as in the browser script
        if np.isnan(out[h]):
            out[h] = out[h - 1] if h else 0.0
    return out


def _gen(v):
    return sum(x for k, x in v.items() if x is not None and k not in NOT_GENERATION and x > 0)


def _emis(v):
    return sum(v[k] * f for k, f in FACTORS.items() if v.get(k) is not None and v[k] > 0) / 1000.0


def _int(v):
    g = _gen(v)
    return _emis(v) * 1000.0 / g if g > 0 else None


def build_table(raw: dict, year: int = 2025) -> dict[str, np.ndarray]:
    g = lambda key, fn: _hourly(raw[key], fn, year)  # noqa: E731
    return {
        "lu_load": g("lu", lambda v: v.get("load")), "lu_solar": g("lu", lambda v: v.get("solar")),
        "lu_wind": g("lu", lambda v: v.get("wind_onshore")),
        "lu_other": g("lu", lambda v: sum(v.get(k) or 0 for k in ("hydro_run_of_river", "biomass", "fossil_gas",
                                                                  "waste", "hydro_water_reservoir"))),
        "price": g("price", lambda v: v.get("day_ahead_price")),
        "flow_de": g("cbpf", lambda v: None if v.get("germany") is None else v["germany"] * 1000),
        "flow_be": g("cbpf", lambda v: None if v.get("belgium") is None else v["belgium"] * 1000),
        "de_load": g("de", lambda v: v.get("load")), "de_solar": g("de", lambda v: v.get("solar")),
        "de_wind": g("de", lambda v: (v.get("wind_onshore") or 0) + (v.get("wind_offshore") or 0)),
        "de_emis_t": g("de", _emis), "int_de": g("de", _int), "int_fr": g("fr", _int), "int_be": g("be", _int),
    }


def write_table(table: dict, output: str | Path, year: int) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# Source: energy-charts.info (Fraunhofer ISE), CC BY 4.0; hourly means of 15-min data, UTC, {year}.\n")
        handle.write("hour," + ",".join(COLUMNS) + "\n")
        for h in range(8760):
            handle.write(f"{h}," + ",".join(f"{round(float(table[c][h]) * 10) / 10:g}" for c in COLUMNS) + "\n")


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Fetch Energy-Charts hourly data for the Luxembourg case")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--output", default=str(DEFAULT_CSV))
    args = parser.parse_args(argv)
    print(f"Wrote {fetch_csv(args.output, args.year)}")


if __name__ == "__main__":
    main()
