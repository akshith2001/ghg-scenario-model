"""Command-line interface for running scenario calculations."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .model import Activity, apply_scenario


def load_activities(path: Path) -> list[Activity]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return [
            Activity(
                activity_id=row["activity_id"],
                sector=row["sector"],
                activity_name=row["activity_name"],
                activity_value=float(row["activity_value"]),
                activity_unit=row["activity_unit"],
                emission_factor=float(row["emission_factor"]),
                factor_unit=row["factor_unit"],
                factor_year=row["factor_year"],
                factor_source=row["factor_source"],
                activity_uncertainty_pct=float(row["activity_uncertainty_pct"]),
                factor_uncertainty_pct=float(row["factor_uncertainty_pct"]),
            )
            for row in rows
        ]


def run(activities_path: Path, scenarios_path: Path, output_path: Path) -> None:
    activities = load_activities(activities_path)
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    results = [
        apply_scenario(activity, scenario_name, sector_changes)
        for scenario_name, sector_changes in scenarios.items()
        for activity in activities
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    totals: dict[str, float] = {}
    for result in results:
        totals[result.scenario] = totals.get(result.scenario, 0.0) + result.emissions_kg_co2e
    baseline = totals.get("baseline")
    print("Scenario totals")
    for name, total in totals.items():
        reduction = 0.0 if not baseline else (1 - total / baseline) * 100
        print(f"{name}: {total / 1000:.2f} tCO2e ({reduction:.1f}% vs baseline)")
    print(f"Detailed results written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-sector GHG scenarios")
    parser.add_argument("--activities", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.activities, args.scenarios, args.output)


if __name__ == "__main__":
    main()
