"""Hospitality water-use case study with transparent assumptions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


# UK Government GHG Conversion Factors 2026 (kg CO2e per cubic metre).
# Water factors remain based on UK water-company Carbon Accounting Workbooks.
WATER_SUPPLY_FACTOR = 0.15311
WATER_TREATMENT_FACTOR = 0.17088
NATURAL_GAS_FACTOR = 0.18231  # kg CO2e/kWh gross calorific value, illustrative default
WATER_SPECIFIC_HEAT_KJ_PER_KG_K = 4.186
WATER_DENSITY_KG_PER_LITRE = 1.0
FACTOR_SOURCE = (
    "https://www.gov.uk/government/publications/"
    "greenhouse-gas-reporting-conversion-factors-2026"
)


@dataclass(frozen=True)
class WaterCaseInputs:
    flow_litres_per_minute: float = 6.0
    running_hours_per_day: float = 7.5
    operating_days_per_year: int = 300
    avoidable_use_pct: float = 75.0
    hot_water_share_pct: float = 50.0
    inlet_temperature_c: float = 12.0
    outlet_temperature_c: float = 45.0
    heater_efficiency_pct: float = 90.0
    heating_emission_factor_kg_per_kwh: float = NATURAL_GAS_FACTOR

    def validate(self) -> None:
        if self.flow_litres_per_minute < 0 or self.running_hours_per_day < 0:
            raise ValueError("Flow and running time must be non-negative")
        if self.operating_days_per_year < 0:
            raise ValueError("Operating days must be non-negative")
        if not 0 <= self.avoidable_use_pct <= 100:
            raise ValueError("Avoidable-use percentage must be between 0 and 100")
        if not 0 <= self.hot_water_share_pct <= 100:
            raise ValueError("Hot-water percentage must be between 0 and 100")
        if self.outlet_temperature_c < self.inlet_temperature_c:
            raise ValueError("Outlet temperature cannot be below inlet temperature")
        if not 0 < self.heater_efficiency_pct <= 100:
            raise ValueError("Heater efficiency must be above 0 and at most 100")
        if self.heating_emission_factor_kg_per_kwh < 0:
            raise ValueError("Heating emission factor must be non-negative")


@dataclass(frozen=True)
class WaterCaseResult:
    annual_water_m3: float
    water_after_intervention_m3: float
    annual_water_saved_m3: float
    baseline_emissions_kg_co2e: float
    emissions_after_intervention_kg_co2e: float
    emissions_saved_kg_co2e: float
    baseline_heating_energy_kwh: float
    heating_energy_after_intervention_kwh: float
    heating_energy_saved_kwh: float
    baseline_heating_emissions_kg_co2e: float
    heating_emissions_after_intervention_kg_co2e: float
    heating_emissions_saved_kg_co2e: float
    baseline_total_emissions_kg_co2e: float
    total_emissions_after_intervention_kg_co2e: float
    total_emissions_saved_kg_co2e: float


def estimate_water_case(inputs: WaterCaseInputs) -> WaterCaseResult:
    """Estimate annual water and supply/treatment emissions for one tap."""
    inputs.validate()
    annual_water_m3 = (
        inputs.flow_litres_per_minute
        * 60
        * inputs.running_hours_per_day
        * inputs.operating_days_per_year
        / 1000
    )
    remaining_share = 1 - inputs.avoidable_use_pct / 100
    water_after = annual_water_m3 * remaining_share
    factor = WATER_SUPPLY_FACTOR + WATER_TREATMENT_FACTOR
    baseline_emissions = annual_water_m3 * factor
    emissions_after = water_after * factor
    hot_water_litres = annual_water_m3 * 1000 * inputs.hot_water_share_pct / 100
    temperature_rise_k = inputs.outlet_temperature_c - inputs.inlet_temperature_c
    useful_heat_kwh = (
        hot_water_litres
        * WATER_DENSITY_KG_PER_LITRE
        * WATER_SPECIFIC_HEAT_KJ_PER_KG_K
        * temperature_rise_k
        / 3600
    )
    baseline_heating_energy = useful_heat_kwh / (inputs.heater_efficiency_pct / 100)
    heating_energy_after = baseline_heating_energy * remaining_share
    baseline_heating_emissions = (
        baseline_heating_energy * inputs.heating_emission_factor_kg_per_kwh
    )
    heating_emissions_after = (
        heating_energy_after * inputs.heating_emission_factor_kg_per_kwh
    )
    return WaterCaseResult(
        annual_water_m3=annual_water_m3,
        water_after_intervention_m3=water_after,
        annual_water_saved_m3=annual_water_m3 - water_after,
        baseline_emissions_kg_co2e=baseline_emissions,
        emissions_after_intervention_kg_co2e=emissions_after,
        emissions_saved_kg_co2e=baseline_emissions - emissions_after,
        baseline_heating_energy_kwh=baseline_heating_energy,
        heating_energy_after_intervention_kwh=heating_energy_after,
        heating_energy_saved_kwh=baseline_heating_energy - heating_energy_after,
        baseline_heating_emissions_kg_co2e=baseline_heating_emissions,
        heating_emissions_after_intervention_kg_co2e=heating_emissions_after,
        heating_emissions_saved_kg_co2e=(
            baseline_heating_emissions - heating_emissions_after
        ),
        baseline_total_emissions_kg_co2e=(
            baseline_emissions + baseline_heating_emissions
        ),
        total_emissions_after_intervention_kg_co2e=(
            emissions_after + heating_emissions_after
        ),
        total_emissions_saved_kg_co2e=(
            baseline_emissions
            + baseline_heating_emissions
            - emissions_after
            - heating_emissions_after
        ),
    )


def write_chart(result: WaterCaseResult, output_path: Path) -> None:
    """Write an accessible SVG comparison chart without third-party packages."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = [
        result.baseline_total_emissions_kg_co2e,
        result.total_emissions_after_intervention_kg_co2e,
    ]
    maximum = max(values) or 1
    heights = [value / maximum * 220 for value in values]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="460" viewBox="0 0 760 460" role="img" aria-labelledby="title desc">
  <title id="title">Illustrative hospitality tap water case</title>
  <desc id="desc">Baseline emissions are {values[0]:.1f} kilograms carbon dioxide equivalent and reduced-running emissions are {values[1]:.1f} kilograms.</desc>
  <rect width="760" height="460" fill="#ffffff"/>
  <text x="380" y="38" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold" fill="#193043">Illustrative hospitality tap water case</text>
  <text x="24" y="78" font-family="Arial" font-size="14" fill="#333333">Annual emissions (kg CO2e)</text>
  <line x1="90" y1="350" x2="700" y2="350" stroke="#666666" stroke-width="1"/>
  <rect x="190" y="{350-heights[0]:.1f}" width="130" height="{heights[0]:.1f}" rx="5" fill="#9D5C63"/>
  <rect x="450" y="{350-heights[1]:.1f}" width="130" height="{heights[1]:.1f}" rx="5" fill="#2A7F62"/>
  <text x="255" y="{338-heights[0]:.1f}" text-anchor="middle" font-family="Arial" font-size="17" font-weight="bold" fill="#333333">{values[0]:.1f}</text>
  <text x="515" y="{338-heights[1]:.1f}" text-anchor="middle" font-family="Arial" font-size="17" font-weight="bold" fill="#333333">{values[1]:.1f}</text>
  <text x="255" y="378" text-anchor="middle" font-family="Arial" font-size="15" fill="#333333">Observed-practice scenario</text>
  <text x="515" y="378" text-anchor="middle" font-family="Arial" font-size="15" fill="#333333">Reduced-running scenario</text>
  <text x="380" y="422" text-anchor="middle" font-family="Arial" font-size="13" fill="#555555">Includes water supply, wastewater treatment and parameterised water-heating emissions.</text>
</svg>'''
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate an illustrative hospitality tap water-use case"
    )
    parser.add_argument("--flow-lpm", type=float, default=6.0)
    parser.add_argument("--hours-per-day", type=float, default=7.5)
    parser.add_argument("--days-per-year", type=int, default=300)
    parser.add_argument("--avoidable-use-pct", type=float, default=75.0)
    parser.add_argument("--hot-water-share-pct", type=float, default=50.0)
    parser.add_argument("--inlet-temperature-c", type=float, default=12.0)
    parser.add_argument("--outlet-temperature-c", type=float, default=45.0)
    parser.add_argument("--heater-efficiency-pct", type=float, default=90.0)
    parser.add_argument(
        "--heating-factor-kg-per-kwh", type=float, default=NATURAL_GAS_FACTOR
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("figures/hospitality_water_case.svg")
    )
    args = parser.parse_args()
    inputs = WaterCaseInputs(
        flow_litres_per_minute=args.flow_lpm,
        running_hours_per_day=args.hours_per_day,
        operating_days_per_year=args.days_per_year,
        avoidable_use_pct=args.avoidable_use_pct,
        hot_water_share_pct=args.hot_water_share_pct,
        inlet_temperature_c=args.inlet_temperature_c,
        outlet_temperature_c=args.outlet_temperature_c,
        heater_efficiency_pct=args.heater_efficiency_pct,
        heating_emission_factor_kg_per_kwh=args.heating_factor_kg_per_kwh,
    )
    result = estimate_water_case(inputs)
    write_chart(result, args.figure)
    print("Illustrative hospitality water-use case")
    print(f"Annual water use: {result.annual_water_m3:,.1f} m3")
    print(f"Potential water saving: {result.annual_water_saved_m3:,.1f} m3")
    print(f"Baseline water-system emissions: {result.baseline_emissions_kg_co2e:,.1f} kg CO2e")
    print(f"Baseline water-heating energy: {result.baseline_heating_energy_kwh:,.1f} kWh")
    print(f"Baseline heating emissions: {result.baseline_heating_emissions_kg_co2e:,.1f} kg CO2e")
    print(f"Baseline total emissions: {result.baseline_total_emissions_kg_co2e:,.1f} kg CO2e")
    print(f"Potential total emissions saving: {result.total_emissions_saved_kg_co2e:,.1f} kg CO2e")
    print(f"Chart written to {args.figure}")
    print(f"Factor source: {FACTOR_SOURCE}")


if __name__ == "__main__":
    main()
