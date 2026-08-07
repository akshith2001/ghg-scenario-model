"""Create a bar chart comparing total emissions by scenario."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def create_chart(results_path: Path, chart_path: Path) -> None:
    """Read detailed results and save scenario totals as a PNG chart."""
    totals_kg: dict[str, float] = defaultdict(float)

    with results_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            totals_kg[row["scenario"]] += float(row["emissions_kg_co2e"])

    scenarios = list(totals_kg)
    totals_tonnes = [totals_kg[name] / 1000 for name in scenarios]
    labels = [name.replace("_", " ").title() for name in scenarios]

    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    bars = axis.bar(labels, totals_tonnes, color=["#64748b", "#0f766e", "#15803d"])
    axis.bar_label(bars, fmt="%.2f tCO2e", padding=3)
    axis.set_title("Annual Greenhouse-Gas Emissions by Scenario")
    axis.set_ylabel("Emissions (tonnes CO2e)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(0, max(totals_tonnes) * 1.15)

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chart_path, dpi=180)
    plt.close(figure)
    print(f"Chart written to {chart_path}")


def main() -> None:
    create_chart(Path("outputs/results.csv"), Path("figures/scenario_totals.png"))


if __name__ == "__main__":
    main()
