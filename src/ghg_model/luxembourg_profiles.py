"""Hourly input profiles for the Luxembourg electricity case study.

Two routes produce the same :class:`HourlyProfiles` object:

* :func:`synthesise_profiles` builds a reproducible 8,760-hour year whose
  *annual totals and published summary statistics* match 2025 figures in
  ``data/luxembourg_2025_inputs.json`` (ILR and Electricity Maps). The hourly
  *shapes* are parametric (solar geometry, diurnal load and price curves,
  autocorrelated weather noise). They are calibrated, not measured.
* :func:`load_profiles_csv` reads measured hourly data, for example the output
  of ``scripts/fetch_entsoe_luxembourg.py``, so the synthetic shapes can be
  replaced with ENTSO-E observations without changing the model.

Every result built on synthetic profiles must be described as conditional on
those shapes. The calibration report returned by :func:`calibration_report`
documents how closely the synthetic year reproduces each published target.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HOURS_PER_YEAR = 8760
LATITUDE_DEG = 49.61  # Luxembourg City
DEFAULT_INPUTS = Path(__file__).resolve().parents[2] / "data" / "luxembourg_2025_inputs.json"

PROFILE_COLUMNS = (
    "load_mw",
    "solar_cf",
    "wind_mw",
    "other_domestic_mw",
    "price_eur_per_mwh",
    "import_avg_g_per_kwh",
    "solar_index",
)


@dataclass
class HourlyProfiles:
    """Aligned hourly series for one year (or any number of hours).

    ``solar_index`` (0-1) is a regional solar-availability signal used to
    shape marginal import intensity; it equals the normalised domestic solar
    capacity factor for synthetic data.
    """

    load_mw: np.ndarray
    solar_cf: np.ndarray
    wind_mw: np.ndarray
    other_domestic_mw: np.ndarray
    price_eur_per_mwh: np.ndarray
    import_avg_g_per_kwh: np.ndarray
    solar_index: np.ndarray
    hour_weights: np.ndarray | None = None
    source: str = "synthetic"
    metadata: dict = field(default_factory=dict)
    border_intensity: dict[str, np.ndarray] | None = None
    border_shares: dict[str, float] | None = None
    marginal_g_per_kwh: np.ndarray | None = None
    zone_residual_mw: np.ndarray | None = None

    def __post_init__(self) -> None:
        lengths = {len(getattr(self, name)) for name in PROFILE_COLUMNS}
        if len(lengths) != 1:
            raise ValueError(f"profile series have different lengths: {sorted(lengths)}")
        if self.hour_weights is None:
            self.hour_weights = np.ones(self.n_hours)
        elif len(self.hour_weights) != self.n_hours:
            raise ValueError("hour_weights must match the number of hours")
        if np.any(self.solar_cf < 0) or np.any(self.solar_cf > 1):
            raise ValueError("solar_cf must lie in [0, 1]")
        if np.any(self.load_mw < 0):
            raise ValueError("load_mw must be non-negative")
        if self.border_intensity is not None:
            for zone, series in self.border_intensity.items():
                if len(series) != self.n_hours:
                    raise ValueError(f"border_intensity[{zone}] has the wrong length")
        if self.marginal_g_per_kwh is not None and len(self.marginal_g_per_kwh) != self.n_hours:
            raise ValueError("marginal_g_per_kwh has the wrong length")

    @property
    def n_hours(self) -> int:
        return len(self.load_mw)

    def subset(self, hours: np.ndarray, weights: np.ndarray) -> "HourlyProfiles":
        """Return selected hours with explicit weights (e.g. representative days)."""
        hours = np.asarray(hours, dtype=int)
        return HourlyProfiles(
            **{name: np.asarray(getattr(self, name))[hours] for name in PROFILE_COLUMNS},
            hour_weights=np.asarray(weights, dtype=float),
            source=f"{self.source} (subset of {len(hours)} hours)",
            metadata=dict(self.metadata, hours=hours.tolist()),
            border_intensity=(None if self.border_intensity is None
                              else {z: np.asarray(v)[hours] for z, v in self.border_intensity.items()}),
            border_shares=None if self.border_shares is None else dict(self.border_shares),
            marginal_g_per_kwh=None if self.marginal_g_per_kwh is None else np.asarray(self.marginal_g_per_kwh)[hours],
            zone_residual_mw=None if self.zone_residual_mw is None else np.asarray(self.zone_residual_mw)[hours],
        )

    def with_marginal(self, series: np.ndarray, label: str) -> "HourlyProfiles":
        """Return a copy carrying an hourly marginal-intensity series (kg/MWh)."""
        from dataclasses import replace
        meta = dict(self.metadata, marginal_source=label)
        return replace(self, marginal_g_per_kwh=np.asarray(series, dtype=float), metadata=meta)


def load_inputs(path: str | Path = DEFAULT_INPUTS) -> dict:
    """Return the ``values`` mapping with plain numbers (sources stripped)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: entry["value"] for key, entry in raw["values"].items()}


def _calendar(n_hours: int = HOURS_PER_YEAR):
    hour_of_year = np.arange(n_hours)
    day_of_year = hour_of_year // 24  # 0-based; 2025 starts on a Wednesday
    hour_of_day = hour_of_year % 24
    weekday = (day_of_year + 2) % 7  # 0 = Monday
    is_weekend = weekday >= 5
    return day_of_year, hour_of_day, is_weekend


def _ar1(rng: np.random.Generator, n: int, rho: float, sigma: float) -> np.ndarray:
    noise = rng.normal(0.0, sigma * np.sqrt(1 - rho**2), n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + noise[i]
    return out


def _clear_sky_shape(day_of_year: np.ndarray, hour_of_day: np.ndarray) -> np.ndarray:
    """Relative clear-sky output from solar elevation (local solar time)."""
    lat = np.radians(LATITUDE_DEG)
    declination = np.radians(23.44) * np.sin(2 * np.pi * (284 + day_of_year + 1) / 365)
    hour_angle = np.radians(15.0 * (hour_of_day + 0.5 - 12.0))
    sin_elev = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    return np.clip(sin_elev, 0.0, None) ** 1.2


def synthesise_profiles(
    seed: int = 2025,
    inputs_path: str | Path = DEFAULT_INPUTS,
) -> HourlyProfiles:
    """Build a calibrated synthetic 2025 hourly year for Luxembourg."""
    v = load_inputs(inputs_path)
    rng = np.random.default_rng(seed)
    day, hod, weekend = _calendar()
    season = np.cos(2 * np.pi * (day - 15) / 365)  # +1 mid-January, -1 mid-July

    # --- Solar capacity factor ------------------------------------------------
    cap_end = v["solar_capacity_end_year_mw"]
    cap_mean = 0.5 * (cap_end + cap_end / 1.358)  # capacity grew 35.8% during 2025
    target_cf = v["solar_generation_gwh"] * 1000.0 / (cap_mean * HOURS_PER_YEAR)
    daily_cloud = np.repeat(_ar1(rng, 365, rho=0.6, sigma=1.0), 24)
    clearness = 1.0 / (1.0 + np.exp(-(daily_cloud + 0.4 - 0.5 * season)))
    raw_solar = _clear_sky_shape(day, hod) * clearness
    solar_cf = np.clip(raw_solar * target_cf / raw_solar.mean(), 0.0, 1.0)
    solar_index = solar_cf / solar_cf.max()

    # --- Load -----------------------------------------------------------------
    diurnal = (
        0.70
        + 0.22 * np.exp(-0.5 * ((hod - 10.0) / 3.2) ** 2)
        + 0.10 * np.exp(-0.5 * ((hod - 18.0) / 2.2) ** 2)
    )
    weekly = np.where(weekend, 0.6 / 0.7, 1.0)
    seasonal_load = 1.0 + 0.14 * season
    load_noise = np.exp(_ar1(rng, HOURS_PER_YEAR, rho=0.9, sigma=0.04))
    load_shape = diurnal * weekly * seasonal_load * load_noise
    load_mw = load_shape * v["gross_consumption_gwh"] * 1000.0 / load_shape.sum()

    # --- Wind and other domestic generation -----------------------------------
    wind_shape = (1.0 + 0.45 * season) * np.exp(_ar1(rng, HOURS_PER_YEAR, rho=0.97, sigma=0.7))
    wind_mw = wind_shape * v["wind_generation_gwh"] * 1000.0 / wind_shape.sum()
    other_gwh = v["domestic_generation_gwh"] - v["solar_generation_gwh"] - v["wind_generation_gwh"]
    other_domestic_mw = np.full(HOURS_PER_YEAR, other_gwh * 1000.0 / HOURS_PER_YEAR)

    # --- Day-ahead price (DE-LU zone) -----------------------------------------
    price_cfg = v["day_ahead_price_eur_per_mwh"]
    evening = np.exp(-0.5 * ((hod - 18.5) / 2.0) ** 2)
    morning = np.exp(-0.5 * ((hod - 8.0) / 1.5) ** 2)
    price_shape = (
        1.0 + 0.55 * evening + 0.15 * morning + 0.18 * season
        - 1.45 * solar_index
        + _ar1(rng, HOURS_PER_YEAR, rho=0.95, sigma=0.25)
    )
    price = price_shape * price_cfg["mean"] / price_shape.mean()

    # --- Average carbon intensity of imports ----------------------------------
    shares = v["imports_by_neighbour_twh"]
    intensity = v["neighbour_average_intensity_g_per_kwh"]
    total_import = sum(shares.values())
    modulation = {"DE": 0.35, "BE": 0.15, "FR": 0.0}
    import_avg = np.zeros(HOURS_PER_YEAR)
    border_series = {}
    for zone, twh in shares.items():
        shape = 1.0 - modulation[zone] * solar_index + 0.08 * evening * (modulation[zone] > 0)
        series = shape * intensity[zone] / shape.mean()
        border_series[zone] = series
        import_avg += (twh / total_import) * series

    return HourlyProfiles(
        load_mw=load_mw,
        solar_cf=solar_cf,
        wind_mw=wind_mw,
        other_domestic_mw=other_domestic_mw,
        price_eur_per_mwh=price,
        import_avg_g_per_kwh=import_avg,
        solar_index=solar_index,
        source=f"synthetic, calibrated to 2025 published statistics (seed {seed})",
        metadata={"existing_solar_mw": float(cap_end), "seed": seed},
        border_intensity=border_series,
        border_shares={z: twh / total_import for z, twh in shares.items()},
    )


def calibration_report(profiles: HourlyProfiles, inputs_path: str | Path = DEFAULT_INPUTS) -> list[dict]:
    """Compare a profile year with the published 2025 targets."""
    v = load_inputs(inputs_path)
    day, hod, weekend = _calendar(profiles.n_hours)
    hourly_mean = np.array([profiles.load_mw[hod == h].mean() for h in range(24)])
    price_hourly = np.array([profiles.price_eur_per_mwh[hod == h].mean() for h in range(24)])
    cap_end = profiles.metadata.get("existing_solar_mw", v["solar_capacity_end_year_mw"])
    cap_mean = 0.5 * (cap_end + cap_end / 1.358)
    targets = v["load_profile_targets_gw"]
    price = v["day_ahead_price_eur_per_mwh"]
    shares = v["imports_by_neighbour_twh"]
    intensity = v["neighbour_average_intensity_g_per_kwh"]
    import_mean = sum(shares[z] * intensity[z] for z in shares) / sum(shares.values())
    rows = [
        ("Annual consumption (GWh)", v["gross_consumption_gwh"], profiles.load_mw.sum() / 1000),
        ("Solar generation (GWh)", v["solar_generation_gwh"], (profiles.solar_cf * cap_mean).sum() / 1000),
        ("Wind generation (GWh)", v["wind_generation_gwh"], profiles.wind_mw.sum() / 1000),
        ("Peak load (GW)", targets["peak"], profiles.load_mw.max() / 1000),
        ("Minimum load (GW)", targets["minimum"], profiles.load_mw.min() / 1000),
        ("Highest hour-of-day mean load (GW)", targets["hourly_mean_max"], hourly_mean.max() / 1000),
        ("Lowest hour-of-day mean load (GW)", targets["hourly_mean_min"], hourly_mean.min() / 1000),
        ("Weekday mean load (GW)", targets["weekday_mean"], profiles.load_mw[~weekend].mean() / 1000),
        ("Weekend mean load (GW)", targets["weekend_mean"], profiles.load_mw[weekend].mean() / 1000),
        ("Mean day-ahead price (EUR/MWh)", price["mean"], profiles.price_eur_per_mwh.mean()),
        ("Highest hour-of-day mean price (EUR/MWh)", price["hourly_profile_high"], price_hourly.max()),
        ("Lowest hour-of-day mean price (EUR/MWh)", price["hourly_profile_low"], price_hourly.min()),
        ("Negative-price hours", price["negative_price_hours"], float((profiles.price_eur_per_mwh < 0).sum())),
        ("Import-weighted average intensity (g/kWh)", round(import_mean, 1), profiles.import_avg_g_per_kwh.mean()),
    ]
    return [
        {
            "statistic": name,
            "published": float(target),
            "model": round(float(model), 3),
            "relative_error": round(float((model - target) / target), 3) if target else None,
        }
        for name, target, model in rows
    ]


def load_profiles_csv(path: str | Path, existing_solar_mw: float | None = None) -> HourlyProfiles:
    """Load measured hourly profiles with columns named as in ``PROFILE_COLUMNS``.

    ``solar_index`` is optional; if missing it is derived from ``solar_cf``.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no rows")
    required = set(PROFILE_COLUMNS) - {"solar_index"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    data = {name: np.array([float(r[name]) for r in rows]) for name in required}
    if "solar_index" in rows[0]:
        data["solar_index"] = np.array([float(r["solar_index"]) for r in rows])
    else:
        peak = data["solar_cf"].max()
        data["solar_index"] = data["solar_cf"] / peak if peak > 0 else data["solar_cf"]
    metadata = {}
    if existing_solar_mw is not None:
        metadata["existing_solar_mw"] = float(existing_solar_mw)
    borders = {c[len("import_avg_"):]: np.array([float(r[c]) for r in rows])
               for c in rows[0] if c.startswith("import_avg_") and c != "import_avg_g_per_kwh"}
    marginal = (np.array([float(r["marginal_g_per_kwh"]) for r in rows])
                if "marginal_g_per_kwh" in rows[0] else None)
    return HourlyProfiles(**data, source=f"measured ({Path(path).name})", metadata=metadata,
                          border_intensity=borders or None, marginal_g_per_kwh=marginal)


def write_profiles_csv(profiles: HourlyProfiles, path: str | Path) -> None:
    """Write profiles to CSV in the format read by :func:`load_profiles_csv`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        borders = sorted(profiles.border_intensity or {})
        extra = [f"import_avg_{z}" for z in borders]
        if profiles.marginal_g_per_kwh is not None:
            extra.append("marginal_g_per_kwh")
        writer.writerow(("hour",) + PROFILE_COLUMNS + tuple(extra))
        for i in range(profiles.n_hours):
            row = [i] + [f"{getattr(profiles, c)[i]:.6g}" for c in PROFILE_COLUMNS]
            row += [f"{profiles.border_intensity[z][i]:.6g}" for z in borders]
            if profiles.marginal_g_per_kwh is not None:
                row.append(f"{profiles.marginal_g_per_kwh[i]:.6g}")
            writer.writerow(row)


def representative_days(profiles: HourlyProfiles, n_days: int = 12, seed: int = 0) -> HourlyProfiles:
    """Pick representative days by k-means on daily (load, solar, price) shapes.

    Each chosen day is weighted by the number of days in its cluster, so annual
    totals are preserved approximately while the LP shrinks ~30x.
    """
    n_full_days = profiles.n_hours // 24
    if n_days >= n_full_days:
        return profiles
    def daily(x):
        return np.asarray(x[: n_full_days * 24]).reshape(n_full_days, 24)
    feats = np.hstack([
        daily(profiles.load_mw) / profiles.load_mw.max(),
        daily(profiles.solar_cf) / max(profiles.solar_cf.max(), 1e-9),
        daily(profiles.price_eur_per_mwh) / np.abs(profiles.price_eur_per_mwh).max(),
        daily(profiles.wind_mw) / max(profiles.wind_mw.max(), 1e-9),
    ])
    rng = np.random.default_rng(seed)
    centres = feats[rng.choice(n_full_days, n_days, replace=False)]
    for _ in range(50):
        dist = ((feats[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        new = np.array([feats[labels == k].mean(axis=0) if np.any(labels == k) else centres[k] for k in range(n_days)])
        if np.allclose(new, centres):
            break
        centres = new
    hours, weights = [], []
    for k in range(n_days):
        members = np.flatnonzero(labels == k)
        if len(members) == 0:
            continue
        medoid = members[((feats[members] - centres[k]) ** 2).sum(axis=1).argmin()]
        hours.extend(range(medoid * 24, medoid * 24 + 24))
        weights.extend([len(members) * HOURS_PER_YEAR / (n_full_days * 24)] * 24)
    return profiles.subset(np.array(hours), np.array(weights))
