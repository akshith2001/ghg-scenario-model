"""Regenerate the synthetic-grid figures from the committed results.

Usage: python scripts/plot_synthetic_grid.py [simulated|real|rolling]
"""

from __future__ import annotations

import csv
import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["GAUSS", "VAE-FREE", "VAE-SOFT", "VAE-HARD"]
COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
ROLLING_PANELS = [("w1_excess", "Wasserstein minus reference"),
                  ("corr_err_excess", "Correlation error minus reference"),
                  ("tstr_ratio", "TSTR MAE ratio (1.0 = real-trained)")]
PANELS = [("w1", "Mean 1-D Wasserstein"), ("corr_err", "Correlation-matrix error"),
          ("violation_gap", "Voltage-violation-rate gap")]


def drift_floor(summary: Path) -> dict[str, float]:
    """Read the reference line (real 2016 days scored against 2017) from a Study 2 summary."""
    text = summary.read_text(encoding="utf-8")
    return {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.]+)", text.split("Reference", 1)[1])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', nargs='?', choices=['simulated','real','rolling'], default='simulated')
    parser.add_argument('--input', type=Path, default=ROOT / 'output/synthetic_grid_reviewed')
    args = parser.parse_args()
    study = args.study
    if study not in ("simulated", "real", "rolling"):
        raise ValueError("study must be simulated, real or rolling")
    folder = args.input / study
    arms = ARMS + ["GMM"] if study == "rolling" else ARMS
    panels = ROLLING_PANELS if study == "rolling" else PANELS
    with (folder / "raw.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    floor = drift_floor(folder / "summary.txt") if study == "real" else {}
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, (metric, title) in zip(axes, panels):
        for i, arm in enumerate(arms):
            vals = [float(r[metric]) for r in rows if r["arm"] == arm]
            xs = [i + (j - (len(vals) - 1) / 2) * 0.06 for j in range(len(vals))]
            ax.scatter(xs, vals, color=COLOURS[i], s=14, zorder=3)
            mean = sum(vals) / len(vals)
            ax.plot([i - 0.25, i + 0.25], [mean, mean], color="black", lw=1.5, zorder=4)
        if metric in floor:
            ax.axhline(floor[metric], color="grey", ls="--", lw=1, zorder=2)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a.replace("VAE-", "V-") for a in arms], fontsize=8)
        ax.set_title(title, fontsize=9)
        if study != "rolling":
            ax.set_ylim(bottom=0)
        else:
            ax.axhline(1.0 if metric == "tstr_ratio" else 0.0, color="grey", ls="--", lw=1, zorder=2)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    if study == "rolling":
        fig.text(.5, .02, "Dots: overlapping origins; bars: means. Descriptive comparison, not independent trials.", ha="center", fontsize=7)
    if floor:
        fig.text(0.99, 0.01, "dashed line: real 2016 days scored against 2017 (distribution-shift reference)", ha="right", fontsize=7, color="grey")
    out = ROOT / "figures" / ("synthetic_grid_reviewed_" + study + ".png")
    fig.savefig(out, dpi=200)
    print(f"Figure written to {out}")


if __name__ == "__main__":
    main()
