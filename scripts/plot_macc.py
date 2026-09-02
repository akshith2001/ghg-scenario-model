"""Generate a marginal abatement cost curve (MACC) figure from an options file.

Usage:
    python scripts/plot_macc.py --options data/abatement_options.json \
        --output figures/macc_curve.svg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ghg_model.optimization import load_options  # noqa: E402


def plot_macc(options_path: str, output_path: str) -> None:
    options = load_options(options_path)
    # Sort by cost per unit abatement (GBP per kg CO2e), ascending
    ranked = sorted(
        options, key=lambda o: o.annual_cost_gbp / max(o.abatement_kgco2e_per_year, 1e-9)
    )

    cumulative = 0.0
    widths, lefts, heights, labels, colors = [], [], [], [], []
    for o in ranked:
        cost_per_unit = o.annual_cost_gbp / max(o.abatement_kgco2e_per_year, 1e-9)
        widths.append(o.abatement_kgco2e_per_year)
        lefts.append(cumulative)
        heights.append(cost_per_unit)
        labels.append(o.name)
        colors.append("#2a9d8f" if cost_per_unit < 0 else "#e76f51")
        cumulative += o.abatement_kgco2e_per_year

    fig, ax = plt.subplots(figsize=(9, 5))
    for left, width, height, label, color in zip(lefts, widths, heights, labels, colors):
        ax.bar(
            left + width / 2, height, width=width, color=color,
            edgecolor="white", align="center",
        )
        ax.text(
            left + width / 2, height + (0.002 if height >= 0 else -0.006),
            label, rotation=90, ha="center",
            va="bottom" if height >= 0 else "top", fontsize=7,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cumulative abatement potential (kg CO2e / year)")
    ax.set_ylabel("Cost per unit abatement (GBP / kg CO2e)")
    ax.set_title(
        "Illustrative marginal abatement cost curve\n"
        "(measures ranked by cost-effectiveness; not a real-world assessment)"
    )
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--options", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot_macc(args.options, args.output)
