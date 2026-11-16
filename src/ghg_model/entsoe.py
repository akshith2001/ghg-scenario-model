"""Fetch measured hourly Luxembourg data from the ENTSO-E Transparency Platform.

Requires a free API token (register at https://transparency.entsoe.eu/ and
request "Restful API access"), supplied via the ``ENTSOE_API_TOKEN``
environment variable. Uses only the Python standard library.

Fetched series
--------------
* actual total load, Luxembourg (document A65, process A16)
* actual generation per type, Luxembourg: solar (B16) and onshore wind (B19)
  (document A75, process A16)
* day-ahead prices, DE-LU bidding zone (document A44)

The output CSV matches :func:`ghg_model.luxembourg_profiles.load_profiles_csv`.
ENTSO-E does not publish the carbon intensity of imports, so
``import_avg_g_per_kwh`` is either read from an hourly CSV you supply
(``--intensity-csv``, columns ``hour,import_avg_g_per_kwh``, e.g. exported
from Electricity Maps) or set to the published 2025 import-weighted annual
average. Non-solar, non-wind domestic generation is set to its published
annual average, spread evenly.
"""

from __future__ import annotations

import argparse
import csv
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .luxembourg_profiles import HourlyProfiles, load_inputs, write_profiles_csv
from .marginal_emissions import TECH_INTENSITY, estimate_marginal_factors

API_URL = "https://web-api.tp.entsoe.eu/api"
AREA_LU = "10YLU-CEGEDEL-NQ"
BZN_DE_LU = "10Y1001A1001A82H"
PSR_SOLAR, PSR_WIND_ONSHORE = "B16", "B19"
AREA_DE = "10Y1001A1001A83F"
# ENTSO-E production types with direct-emission intensities (kg/MWh; assumptions)
FOSSIL_PSR = {"B02": TECH_INTENSITY["lignite"], "B05": TECH_INTENSITY["hard coal"],
              "B04": TECH_INTENSITY["CCGT"], "B06": 750.0, "B03": 600.0}
VARIABLE_RES_PSR = ("B16", "B18", "B19")


def _resolution_minutes(text: str) -> int:
    if text == "PT60M":
        return 60
    if text == "PT30M":
        return 30
    if text == "PT15M":
        return 15
    raise ValueError(f"unsupported resolution {text!r}")


def parse_timeseries(xml_text: str, start: datetime, n_hours: int, value_tag: str = "quantity",
                     psr_type: str | None = None) -> np.ndarray:
    """Parse an ENTSO-E market document into an hourly mean series.

    Points finer than an hour are averaged into the hour they fall in. Hours
    with no data are NaN so gaps stay visible.
    """
    root = ET.fromstring(xml_text)
    for el in root.iter():  # drop XML namespaces so paths stay readable
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    sums = np.zeros(n_hours)
    counts = np.zeros(n_hours)
    for ts in root.findall("TimeSeries"):
        if psr_type is not None:
            psr = ts.find("MktPSRType/psrType")
            if psr is None or psr.text != psr_type:
                continue
        for period in ts.findall("Period"):
            p_start = datetime.strptime(period.find("timeInterval/start").text,
                                        "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
            step = _resolution_minutes(period.find("resolution").text)
            for point in period.findall("Point"):
                pos = int(point.find("position").text)
                t = p_start + timedelta(minutes=step * (pos - 1))
                h = int((t - start).total_seconds() // 3600)
                if 0 <= h < n_hours:
                    sums[h] += float(point.find(value_tag).text)
                    counts[h] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def _get(params: dict, token: str) -> str:
    query = urllib.parse.urlencode({"securityToken": token, **params})
    with urllib.request.urlopen(f"{API_URL}?{query}", timeout=120) as resp:
        return resp.read().decode("utf-8")


def _fmt(t: datetime) -> str:
    return t.strftime("%Y%m%d%H%M")


def fill_gaps(series: np.ndarray) -> np.ndarray:
    """Linear interpolation across NaN gaps (edges take the nearest value)."""
    s = np.asarray(series, dtype=float).copy()
    bad = np.isnan(s)
    if bad.all():
        raise ValueError("series has no data")
    if bad.any():
        idx = np.arange(len(s))
        s[bad] = np.interp(idx[bad], idx[~bad], s[~bad])
    return s


def fetch_year(year: int, token: str, solar_capacity_mw: float, intensity_csv: str | None = None) -> HourlyProfiles:
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    n = int((end - start).total_seconds() // 3600)
    span = {"periodStart": _fmt(start), "periodEnd": _fmt(end)}
    load = parse_timeseries(_get({"documentType": "A65", "processType": "A16",
                                  "outBiddingZone_Domain": AREA_LU, **span}, token), start, n)
    gen_xml = _get({"documentType": "A75", "processType": "A16", "in_Domain": AREA_LU, **span}, token)
    solar = parse_timeseries(gen_xml, start, n, psr_type=PSR_SOLAR)
    wind = parse_timeseries(gen_xml, start, n, psr_type=PSR_WIND_ONSHORE)
    price = parse_timeseries(_get({"documentType": "A44", "in_Domain": BZN_DE_LU, "out_Domain": BZN_DE_LU, **span},
                                  token), start, n, value_tag="price.amount")
    v = load_inputs()
    if intensity_csv:
        with open(intensity_csv, newline="", encoding="utf-8") as handle:
            intensity = np.array([float(r["import_avg_g_per_kwh"]) for r in csv.DictReader(handle)])[:n]
    else:
        shares, factors = v["imports_by_neighbour_twh"], v["neighbour_average_intensity_g_per_kwh"]
        intensity = np.full(n, sum(shares[z] * factors[z] for z in shares) / sum(shares.values()))
    other_gwh = v["domestic_generation_gwh"] - v["solar_generation_gwh"] - v["wind_generation_gwh"]
    solar = fill_gaps(solar)
    solar_cf = np.clip(solar / solar_capacity_mw, 0.0, 1.0)
    return HourlyProfiles(
        load_mw=fill_gaps(load),
        solar_cf=solar_cf,
        wind_mw=fill_gaps(wind),
        other_domestic_mw=np.full(n, other_gwh * 1000.0 / n),
        price_eur_per_mwh=fill_gaps(price),
        import_avg_g_per_kwh=intensity,
        solar_index=solar_cf / max(solar_cf.max(), 1e-9),
        source=f"ENTSO-E {year}",
        metadata={"existing_solar_mw": solar_capacity_mw},
    )


def german_marginal_series(year: int, token: str, n_bins: int = 8) -> tuple[np.ndarray, dict]:
    """Estimate hourly marginal emission factors for Germany from ENTSO-E data.

    Emissions = sum over fossil production types of generation x intensity.
    Residual load = total load - solar - wind. The marginal factor is the
    Hawkes-style regression slope of emissions on residual load by bin.
    """
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    n = int((end - start).total_seconds() // 3600)
    span = {"periodStart": _fmt(start), "periodEnd": _fmt(end)}
    load = fill_gaps(parse_timeseries(_get({"documentType": "A65", "processType": "A16",
                                            "outBiddingZone_Domain": AREA_DE, **span}, token), start, n))
    gen_xml = _get({"documentType": "A75", "processType": "A16", "in_Domain": AREA_DE, **span}, token)
    emissions = np.zeros(n)
    for psr, factor in FOSSIL_PSR.items():
        series = parse_timeseries(gen_xml, start, n, psr_type=psr)
        if not np.all(np.isnan(series)):
            emissions += fill_gaps(series) * factor
    vres = np.zeros(n)
    for psr in VARIABLE_RES_PSR:
        series = parse_timeseries(gen_xml, start, n, psr_type=psr)
        if not np.all(np.isnan(series)):
            vres += fill_gaps(series)
    est = estimate_marginal_factors(emissions, load - vres, n_bins=n_bins)
    return est["hourly"], {k: v for k, v in est.items() if k != "hourly"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch hourly Luxembourg profiles from ENTSO-E")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--solar-capacity-mw", type=float, default=746.0,
                        help="installed solar used to convert generation to capacity factor")
    parser.add_argument("--intensity-csv", help="optional hourly import intensity (hour,import_avg_g_per_kwh)")
    parser.add_argument("--estimate-marginal", action="store_true",
                        help="also estimate hourly marginal factors from German generation data")
    parser.add_argument("--output", default="data/luxembourg_entsoe_hourly.csv")
    args = parser.parse_args(argv)
    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        raise SystemExit("Set ENTSOE_API_TOKEN to your ENTSO-E Transparency Platform API token.")
    profiles = fetch_year(args.year, token, args.solar_capacity_mw, args.intensity_csv)
    if args.estimate_marginal:
        series, fit = german_marginal_series(args.year, token)
        profiles = profiles.with_marginal(series[: profiles.n_hours], "estimated from ENTSO-E (DE)")
        print("Marginal-factor bins (kg/MWh):", [round(x) for x in fit["slopes"]])
    write_profiles_csv(profiles, Path(args.output))
    print(f"Wrote {profiles.n_hours} hours to {args.output}")


if __name__ == "__main__":
    main()
