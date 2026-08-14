"""One-at-a-time sensitivity analysis for the hospitality water case."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

from .hospitality_water import WaterCaseInputs, estimate_water_case


@dataclass(frozen=True)
class SensitivityResult:
    parameter: str
    low_value: float
    base_value: float
    high_value: float
    low_saving_kg_co2e: float
    base_saving_kg_co2e: float
    high_saving_kg_co2e: float
    low_change_pct: float
    high_change_pct: float
    sensitivity_score_pct: float


PARAMETER_LABELS = {
    "flow_litres_per_minute": "Tap flow",
    "running_hours_per_day": "Running time",
    "operating_days_per_year": "Operating days",
    "avoidable_use_pct": "Avoidable-use reduction",
    "hot_water_share_pct": "Hot-water share",
    "temperature_rise_c": "Temperature rise",
    "heater_efficiency_pct": "Heater efficiency",
    "heating_emission_factor_kg_per_kwh": "Heating emission factor",
}


def _scaled_inputs(
    base: WaterCaseInputs, parameter: str, multiplier: float
) -> tuple[WaterCaseInputs, float]:
    if parameter == "temperature_rise_c":
        base_value = base.outlet_temperature_c - base.inlet_temperature_c
        varied_value = base_value * multiplier
        return replace(
            base, outlet_temperature_c=base.inlet_temperature_c + varied_value
        ), varied_value

    base_value = float(getattr(base, parameter))
    varied_value = base_value * multiplier
    if parameter == "operating_days_per_year":
        varied_value = round(varied_value)
    return replace(base, **{parameter: varied_value}), float(varied_value)


def analyse_sensitivity(
    base: WaterCaseInputs | None = None, variation_pct: float = 10.0
) -> list[SensitivityResult]:
    """Rank OAT effects on potential total emissions savings."""
    if not 0 < variation_pct < 100:
        raise ValueError("Variation percentage must be above 0 and below 100")
    base = base or WaterCaseInputs()
    base.validate()
    base_saving = estimate_water_case(base).total_emissions_saved_kg_co2e
    if base_saving == 0:
        raise ValueError("Base case must have non-zero emissions savings")

    parameters = list(PARAMETER_LABELS)
    low_multiplier = 1 - variation_pct / 100
    high_multiplier = 1 + variation_pct / 100
    results: list[SensitivityResult] = []
    for parameter in parameters:
        low_inputs, low_value = _scaled_inputs(base, parameter, low_multiplier)
        high_inputs, high_value = _scaled_inputs(base, parameter, high_multiplier)
        low_saving = estimate_water_case(low_inputs).total_emissions_saved_kg_co2e
        high_saving = estimate_water_case(high_inputs).total_emissions_saved_kg_co2e
        low_change = (low_saving / base_saving - 1) * 100
        high_change = (high_saving / base_saving - 1) * 100
        if parameter == "temperature_rise_c":
            base_value = base.outlet_temperature_c - base.inlet_temperature_c
        else:
            base_value = float(getattr(base, parameter))
        results.append(
            SensitivityResult(
                parameter=parameter,
                low_value=low_value,
                base_value=base_value,
                high_value=high_value,
                low_saving_kg_co2e=low_saving,
                base_saving_kg_co2e=base_saving,
                high_saving_kg_co2e=high_saving,
                low_change_pct=low_change,
                high_change_pct=high_change,
                sensitivity_score_pct=max(abs(low_change), abs(high_change)),
            )
        )
    return sorted(results, key=lambda item: item.sensitivity_score_pct, reverse=True)


def write_results_csv(results: list[SensitivityResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SensitivityResult.__annotations__))
        writer.writeheader()
        writer.writerows(result.__dict__ for result in results)


def write_tornado_svg(results: list[SensitivityResult], output_path: Path) -> None:
    """Create a dependency-free sensitivity chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 120 + 52 * len(results)
    centre, scale = 520, 24
    rows = []
    for index, result in enumerate(results):
        y = 98 + index * 52
        low_x = centre + result.low_change_pct * scale
        high_x = centre + result.high_change_pct * scale
        left, right = min(low_x, high_x), max(low_x, high_x)
        rows.append(
            f'<text x="300" y="{y + 5}" text-anchor="end" font-family="Arial" font-size="15" fill="#263238">{PARAMETER_LABELS[result.parameter]}</text>'
            f'<rect x="{left:.1f}" y="{y - 13}" width="{max(2, right-left):.1f}" height="24" rx="4" fill="#2A7F62"/>'
            f'<text x="{left - 8:.1f}" y="{y + 5}" text-anchor="end" font-family="Arial" font-size="12" fill="#444">{result.low_change_pct:+.1f}%</text>'
            f'<text x="{right + 8:.1f}" y="{y + 5}" font-family="Arial" font-size="12" fill="#444">{result.high_change_pct:+.1f}%</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">One-at-a-time sensitivity of annual emissions savings</title>
  <desc id="desc">Parameters ranked by the maximum percentage change in calculated emissions savings after a plus or minus ten percent input change.</desc>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="450" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold" fill="#193043">Sensitivity of annual emissions savings</text>
  <text x="450" y="59" text-anchor="middle" font-family="Arial" font-size="13" fill="#555">One input varied at a time; default range ±10%</text>
  <line x1="{centre}" y1="76" x2="{centre}" y2="{height-28}" stroke="#455A64" stroke-width="1"/>
  {''.join(rows)}
  <text x="{centre}" y="{height-8}" text-anchor="middle" font-family="Arial" font-size="12" fill="#555">Percentage change from base-case emissions saving</text>
</svg>'''
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one-at-a-time sensitivity analysis for the water case"
    )
    parser.add_argument("--variation-pct", type=float, default=10.0)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/hospitality_sensitivity.csv")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("figures/hospitality_sensitivity.svg")
    )
    args = parser.parse_args()
    results = analyse_sensitivity(variation_pct=args.variation_pct)
    write_results_csv(results, args.output)
    write_tornado_svg(results, args.figure)
    print("Ranked one-at-a-time sensitivity")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {PARAMETER_LABELS[result.parameter]}: "
            f"{result.sensitivity_score_pct:.2f}% maximum output change"
        )
    print(f"Results written to {args.output}")
    print(f"Chart written to {args.figure}")


if __name__ == "__main__":
    main()
