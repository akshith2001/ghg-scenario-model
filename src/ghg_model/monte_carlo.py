"""Reproducible Monte Carlo uncertainty analysis for the water case."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median

from .hospitality_water import WaterCaseInputs, estimate_water_case


@dataclass(frozen=True)
class TriangularRange:
    low: float
    mode: float
    high: float

    def validate(self) -> None:
        if not self.low <= self.mode <= self.high:
            raise ValueError("Triangular ranges require low <= mode <= high")


DEFAULT_RANGES = {
    "flow_litres_per_minute": TriangularRange(4.0, 6.0, 8.0),
    "running_hours_per_day": TriangularRange(7.0, 7.5, 8.0),
    "operating_days_per_year": TriangularRange(250.0, 300.0, 350.0),
    "avoidable_use_pct": TriangularRange(50.0, 75.0, 90.0),
    "hot_water_share_pct": TriangularRange(25.0, 50.0, 75.0),
    "inlet_temperature_c": TriangularRange(8.0, 12.0, 16.0),
    "outlet_temperature_c": TriangularRange(40.0, 45.0, 50.0),
    "heater_efficiency_pct": TriangularRange(80.0, 90.0, 98.0),
    "heating_emission_factor_kg_per_kwh": TriangularRange(0.15, 0.18231, 0.22),
}


@dataclass(frozen=True)
class MonteCarloSample:
    sample_id: int
    annual_water_saved_m3: float
    heating_energy_saved_kwh: float
    total_emissions_saved_kg_co2e: float


@dataclass(frozen=True)
class MonteCarloSummary:
    samples: int
    seed: int
    mean_saving_kg_co2e: float
    median_saving_kg_co2e: float
    p05_saving_kg_co2e: float
    p95_saving_kg_co2e: float
    minimum_saving_kg_co2e: float
    maximum_saving_kg_co2e: float


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile from no values")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def run_monte_carlo(
    sample_count: int = 10_000,
    seed: int = 2026,
    ranges: dict[str, TriangularRange] | None = None,
) -> tuple[list[MonteCarloSample], MonteCarloSummary]:
    """Sample independent triangular assumptions and summarise emissions savings."""
    if sample_count < 100:
        raise ValueError("Use at least 100 samples for a stable demonstration")
    ranges = ranges or DEFAULT_RANGES
    missing = set(DEFAULT_RANGES) - set(ranges)
    if missing:
        raise ValueError(f"Missing ranges: {sorted(missing)}")
    for value_range in ranges.values():
        value_range.validate()

    generator = random.Random(seed)
    samples: list[MonteCarloSample] = []
    for sample_id in range(1, sample_count + 1):
        values = {
            name: generator.triangular(value.low, value.high, value.mode)
            for name, value in ranges.items()
        }
        values["operating_days_per_year"] = round(values["operating_days_per_year"])
        result = estimate_water_case(WaterCaseInputs(**values))
        samples.append(
            MonteCarloSample(
                sample_id=sample_id,
                annual_water_saved_m3=result.annual_water_saved_m3,
                heating_energy_saved_kwh=result.heating_energy_saved_kwh,
                total_emissions_saved_kg_co2e=result.total_emissions_saved_kg_co2e,
            )
        )

    savings = sorted(sample.total_emissions_saved_kg_co2e for sample in samples)
    summary = MonteCarloSummary(
        samples=sample_count,
        seed=seed,
        mean_saving_kg_co2e=fmean(savings),
        median_saving_kg_co2e=median(savings),
        p05_saving_kg_co2e=_percentile(savings, 0.05),
        p95_saving_kg_co2e=_percentile(savings, 0.95),
        minimum_saving_kg_co2e=savings[0],
        maximum_saving_kg_co2e=savings[-1],
    )
    return samples, summary


def write_samples_csv(samples: list[MonteCarloSample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MonteCarloSample.__annotations__))
        writer.writeheader()
        writer.writerows(sample.__dict__ for sample in samples)


def write_histogram_svg(
    samples: list[MonteCarloSample], summary: MonteCarloSummary, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = [sample.total_emissions_saved_kg_co2e for sample in samples]
    minimum, maximum = min(values), max(values)
    bin_count = 24
    value_span = maximum - minimum
    bin_width = value_span / bin_count or 1
    counts = [0] * bin_count
    for value in values:
        index = min(int((value - minimum) / bin_width), bin_count - 1)
        counts[index] += 1
    max_count = max(counts) or 1
    chart_left, chart_top, chart_width, chart_height = 75, 85, 760, 280
    bar_width = chart_width / bin_count
    bars = []
    for index, count in enumerate(counts):
        height = count / max_count * chart_height
        x = chart_left + index * bar_width
        y = chart_top + chart_height - height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width-1:.1f}" height="{height:.1f}" fill="#2A7F62"/>'
        )
    plot_span = value_span or 1
    p05_x = chart_left + (summary.p05_saving_kg_co2e - minimum) / plot_span * chart_width
    p95_x = chart_left + (summary.p95_saving_kg_co2e - minimum) / plot_span * chart_width
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500" role="img" aria-labelledby="title desc">
  <title id="title">Monte Carlo distribution of potential annual emissions savings</title>
  <desc id="desc">Distribution from {summary.samples} reproducible samples. The conditional fifth to ninety-fifth percentile range is {summary.p05_saving_kg_co2e:.0f} to {summary.p95_saving_kg_co2e:.0f} kilograms carbon dioxide equivalent.</desc>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="450" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold" fill="#193043">Uncertainty in potential annual emissions savings</text>
  <text x="450" y="59" text-anchor="middle" font-family="Arial" font-size="13" fill="#555">{summary.samples:,} independent triangular samples; seed {summary.seed}</text>
  {''.join(bars)}
  <line x1="{p05_x:.1f}" y1="78" x2="{p05_x:.1f}" y2="375" stroke="#9D5C63" stroke-width="3"/>
  <line x1="{p95_x:.1f}" y1="78" x2="{p95_x:.1f}" y2="375" stroke="#9D5C63" stroke-width="3"/>
  <line x1="{chart_left}" y1="{chart_top+chart_height}" x2="{chart_left+chart_width}" y2="{chart_top+chart_height}" stroke="#455A64"/>
  <text x="{p05_x:.1f}" y="398" text-anchor="middle" font-family="Arial" font-size="12" fill="#7A3F45">P5 {summary.p05_saving_kg_co2e:.0f}</text>
  <text x="{p95_x:.1f}" y="398" text-anchor="middle" font-family="Arial" font-size="12" fill="#7A3F45">P95 {summary.p95_saving_kg_co2e:.0f}</text>
  <text x="450" y="435" text-anchor="middle" font-family="Arial" font-size="13" fill="#333">Potential annual emissions saving (kg CO2e)</text>
  <text x="450" y="466" text-anchor="middle" font-family="Arial" font-size="12" fill="#555">Conditional range reflects assumed inputs; it is not a measured confidence interval.</text>
</svg>'''
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible Monte Carlo analysis for the water case"
    )
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/hospitality_monte_carlo.csv")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("figures/hospitality_monte_carlo.svg")
    )
    args = parser.parse_args()
    samples, summary = run_monte_carlo(args.samples, args.seed)
    write_samples_csv(samples, args.output)
    write_histogram_svg(samples, summary, args.figure)
    print("Monte Carlo uncertainty summary")
    print(f"Samples: {summary.samples:,}; seed: {summary.seed}")
    print(f"Mean saving: {summary.mean_saving_kg_co2e:,.1f} kg CO2e/year")
    print(f"Median saving: {summary.median_saving_kg_co2e:,.1f} kg CO2e/year")
    print(
        f"Conditional P5-P95 range: {summary.p05_saving_kg_co2e:,.1f} to "
        f"{summary.p95_saving_kg_co2e:,.1f} kg CO2e/year"
    )
    print(f"Samples written to {args.output}")
    print(f"Chart written to {args.figure}")


if __name__ == "__main__":
    main()
