"""Held-out validation against reported EPA eGRID plant emissions.

The benchmark deliberately excludes target-derived eGRID emission-rate fields.
It compares a data-only ridge model, an external-factor physical estimate and a
physics-guided residual model on plants that are never used for fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


NATURAL_GAS_CO2_KG_PER_MMBTU = 53.06
KG_PER_SHORT_TON = 907.18474
REQUIRED_COLUMNS = {
    "YEAR",
    "ORISPL",
    "PNAME",
    "PSTATABB",
    "PLPRMFL",
    "PLHTIAN",
    "PLNGENAN",
    "NAMEPCAP",
    "CAPFAC",
    "PLCO2AN",
}
FORBIDDEN_FEATURES = {"PLCO2RTA", "PLCO2RA", "PLCO2CRT", "PLC2ERTA", "PLC2ERA"}


@dataclass(frozen=True)
class PlantRecord:
    year: int
    plant_id: int
    plant_name: str
    state: str
    fuel: str
    heat_input_mmbtu: float
    net_generation_mwh: float | None
    nameplate_capacity_mw: float | None
    capacity_factor: float | None
    reported_co2_short_tons: float


@dataclass(frozen=True)
class ValidationMetrics:
    rmse_short_tons: float
    mae_short_tons: float
    median_absolute_percentage_error: float


@dataclass(frozen=True)
class ValidationResult:
    data_year: int
    source_records: int
    eligible_records: int
    training_records: int
    held_out_records: int
    split_rule: str
    external_factor_kg_co2_per_mmbtu: float
    data_only: ValidationMetrics
    external_factor_physics: ValidationMetrics
    physics_guided_residual: ValidationMetrics


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def load_egrid_csv(path: Path) -> tuple[list[PlantRecord], int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"missing required eGRID columns: {sorted(missing)}")
        leaked = FORBIDDEN_FEATURES & columns
        if leaked:
            raise ValueError(
                "target-derived emission-rate fields must be removed before validation: "
                f"{sorted(leaked)}"
            )
        rows = list(reader)

    records: list[PlantRecord] = []
    for row in rows:
        heat = _optional_float(row["PLHTIAN"])
        reported = _optional_float(row["PLCO2AN"])
        if heat is None or reported is None or heat <= 0 or reported <= 0:
            continue
        records.append(
            PlantRecord(
                year=int(row["YEAR"]),
                plant_id=int(row["ORISPL"]),
                plant_name=row["PNAME"],
                state=row["PSTATABB"],
                fuel=row["PLPRMFL"],
                heat_input_mmbtu=heat,
                net_generation_mwh=_optional_float(row["PLNGENAN"]),
                nameplate_capacity_mw=_optional_float(row["NAMEPCAP"]),
                capacity_factor=_optional_float(row["CAPFAC"]),
                reported_co2_short_tons=reported,
            )
        )
    if len(records) < 50:
        raise ValueError("at least 50 eligible plant records are required")
    return records, len(rows)


def _split(records: list[PlantRecord]) -> tuple[list[PlantRecord], list[PlantRecord]]:
    training = [record for record in records if record.plant_id % 5 != 0]
    held_out = [record for record in records if record.plant_id % 5 == 0]
    if not training or not held_out:
        raise ValueError("deterministic plant split produced an empty partition")
    return training, held_out


def _raw_features(records: list[PlantRecord]) -> np.ndarray:
    return np.asarray(
        [
            [
                record.heat_input_mmbtu,
                np.nan if record.net_generation_mwh is None else max(record.net_generation_mwh, 0.0),
                np.nan if record.nameplate_capacity_mw is None else max(record.nameplate_capacity_mw, 0.0),
                np.nan if record.capacity_factor is None else max(record.capacity_factor, 0.0),
            ]
            for record in records
        ],
        dtype=float,
    )


def _prepare_features(
    records: list[PlantRecord], medians: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    raw = _raw_features(records)
    if medians is None:
        medians = np.nanmedian(raw, axis=0)
    filled = np.where(np.isnan(raw), medians, raw)
    return np.log1p(filled), medians


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    design = np.column_stack((np.ones(len(x)), (x - mean) / std))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return coefficients, mean, std


def _predict_ridge(x: np.ndarray, fitted: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    coefficients, mean, std = fitted
    design = np.column_stack((np.ones(len(x)), (x - mean) / std))
    return design @ coefficients


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> ValidationMetrics:
    error = prediction - truth
    percentage = np.abs(error) / truth * 100.0
    return ValidationMetrics(
        rmse_short_tons=float(np.sqrt(np.mean(error**2))),
        mae_short_tons=float(np.mean(np.abs(error))),
        median_absolute_percentage_error=float(np.median(percentage)),
    )


def run_validation(path: Path) -> tuple[ValidationResult, dict[str, np.ndarray]]:
    records, source_count = load_egrid_csv(path)
    training, held_out = _split(records)
    train_x, medians = _prepare_features(training)
    test_x, _ = _prepare_features(held_out, medians)
    train_y = np.asarray([record.reported_co2_short_tons for record in training])
    test_y = np.asarray([record.reported_co2_short_tons for record in held_out])

    data_only_fit = _fit_ridge(train_x, np.log1p(train_y))
    data_only_prediction = np.maximum(np.expm1(_predict_ridge(test_x, data_only_fit)), 0.0)

    train_heat = np.asarray([record.heat_input_mmbtu for record in training])
    test_heat = np.asarray([record.heat_input_mmbtu for record in held_out])
    train_physics = train_heat * NATURAL_GAS_CO2_KG_PER_MMBTU / KG_PER_SHORT_TON
    test_physics = test_heat * NATURAL_GAS_CO2_KG_PER_MMBTU / KG_PER_SHORT_TON

    residual_target = np.log((train_y + 1.0) / (train_physics + 1.0))
    residual_fit = _fit_ridge(train_x[:, 1:], residual_target)
    correction = np.exp(_predict_ridge(test_x[:, 1:], residual_fit))
    guided_prediction = np.maximum(test_physics * correction, 0.0)

    years = {record.year for record in records}
    if len(years) != 1:
        raise ValueError(f"expected one eGRID data year, found {sorted(years)}")
    result = ValidationResult(
        data_year=years.pop(),
        source_records=source_count,
        eligible_records=len(records),
        training_records=len(training),
        held_out_records=len(held_out),
        split_rule="held out when ORISPL plant identifier modulo 5 equals zero",
        external_factor_kg_co2_per_mmbtu=NATURAL_GAS_CO2_KG_PER_MMBTU,
        data_only=_metrics(data_only_prediction, test_y),
        external_factor_physics=_metrics(test_physics, test_y),
        physics_guided_residual=_metrics(guided_prediction, test_y),
    )
    arrays = {
        "truth": test_y,
        "data_only": data_only_prediction,
        "external_factor_physics": test_physics,
        "physics_guided_residual": guided_prediction,
    }
    return result, arrays


def _plot(result: ValidationResult, path: Path) -> None:
    labels = ("Data only", "External factor", "Physics guided")
    values = (
        result.data_only.median_absolute_percentage_error,
        result.external_factor_physics.median_absolute_percentage_error,
        result.physics_guided_residual.median_absolute_percentage_error,
    )
    colors = ("#D97706", "#64748B", "#17638F")
    width, height = 850, 470
    chart_top, chart_bottom = 75, 365
    maximum = max(values) * 1.15
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#173B5E}.axis{stroke:#667085;stroke-width:1}</style>',
        '<text x="425" y="30" text-anchor="middle" font-size="20" font-weight="700">Held-out EPA eGRID natural-gas plants</text>',
        '<text x="425" y="53" text-anchor="middle" font-size="13">Median absolute percentage error; lower is better</text>',
        f'<line class="axis" x1="75" y1="{chart_bottom}" x2="800" y2="{chart_bottom}"/>',
    ]
    for index, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        x = 125 + index * 235
        bar_height = value / maximum * (chart_bottom - chart_top)
        y = chart_bottom - bar_height
        parts.extend(
            [
                f'<rect x="{x}" y="{y:.2f}" width="145" height="{bar_height:.2f}" rx="4" fill="{color}"/>',
                f'<text x="{x + 72.5}" y="{y - 10:.2f}" text-anchor="middle" font-size="15" font-weight="700">{value:.2f}%</text>',
                f'<text x="{x + 72.5}" y="395" text-anchor="middle" font-size="14">{label}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="425" y="438" text-anchor="middle" font-size="12">{result.held_out_records} plants held out by identifier; 2018 data</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/egrid_natural_gas_2018.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/egrid_validation.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/egrid_validation.svg"))
    args = parser.parse_args()
    result, _ = run_validation(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    _plot(result, args.figure)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
