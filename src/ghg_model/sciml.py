"""Small, auditable scientific-ML benchmark for the emissions equation.

This module intentionally implements a compact neural network with NumPy so the
physics-informed objective and its gradients remain inspectable. It is an
educational benchmark on synthetic data, not evidence of industrial validity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Metrics:
    rmse_kg_co2e: float
    mae_kg_co2e: float
    mean_physics_violation_kg_co2e: float


@dataclass(frozen=True)
class ExperimentResult:
    seed: int
    samples: int
    noise_fraction: float
    physics_weight: float
    data_only_interpolation: Metrics
    physics_informed_interpolation: Metrics
    data_only_extrapolation: Metrics
    physics_informed_extrapolation: Metrics


class TinyRegressor:
    """One-hidden-layer neural regressor trained by deterministic gradient descent."""

    def __init__(self, input_mean: np.ndarray, input_std: np.ndarray, target_scale: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.input_mean = input_mean
        self.input_std = input_std
        self.target_scale = target_scale
        self.w1 = rng.normal(0.0, 0.25, size=(2, 24))
        self.b1 = np.zeros(24)
        self.w2 = rng.normal(0.0, 0.15, size=(24, 1))
        self.b2 = np.zeros(1)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xn = (x - self.input_mean) / self.input_std
        hidden = np.tanh(xn @ self.w1 + self.b1)
        prediction = hidden @ self.w2 + self.b2
        return xn, hidden, prediction[:, 0]

    def fit(self, x: np.ndarray, observed_y: np.ndarray, *, physics_weight: float, epochs: int = 2500, learning_rate: float = 0.018) -> None:
        observed = observed_y / self.target_scale
        physical = (x[:, 0] * x[:, 1]) / self.target_scale
        n = len(x)
        for _ in range(epochs):
            xn, hidden, pred = self._forward(x)
            grad_pred = (2.0 / n) * ((pred - observed) + physics_weight * (pred - physical))
            grad_w2 = hidden.T @ grad_pred[:, None]
            grad_b2 = np.array([grad_pred.sum()])
            grad_hidden = grad_pred[:, None] @ self.w2.T
            grad_z1 = grad_hidden * (1.0 - hidden**2)
            grad_w1 = xn.T @ grad_z1
            grad_b1 = grad_z1.sum(axis=0)
            for parameter, gradient in ((self.w1, grad_w1), (self.b1, grad_b1), (self.w2, grad_w2), (self.b2, grad_b2)):
                np.clip(gradient, -5.0, 5.0, out=gradient)
                parameter -= learning_rate * gradient

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[2] * self.target_scale


def _sample(rng: np.random.Generator, n: int, activity_range: tuple[float, float], factor_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    activity = rng.uniform(*activity_range, size=n)
    factor = rng.uniform(*factor_range, size=n)
    x = np.column_stack((activity, factor))
    return x, activity * factor


def _metrics(prediction: np.ndarray, truth: np.ndarray, x: np.ndarray) -> Metrics:
    error = prediction - truth
    violation = prediction - (x[:, 0] * x[:, 1])
    return Metrics(
        rmse_kg_co2e=float(np.sqrt(np.mean(error**2))),
        mae_kg_co2e=float(np.mean(np.abs(error))),
        mean_physics_violation_kg_co2e=float(np.mean(np.abs(violation))),
    )


def run_experiment(*, seed: int = 2026, samples: int = 600, noise_fraction: float = 0.15, physics_weight: float = 1.5) -> tuple[ExperimentResult, dict[str, np.ndarray]]:
    if samples < 100:
        raise ValueError("samples must be at least 100")
    if not 0 <= noise_fraction <= 1:
        raise ValueError("noise_fraction must be between 0 and 1")
    if physics_weight < 0:
        raise ValueError("physics_weight must be non-negative")
    rng = np.random.default_rng(seed)
    train_x, train_truth = _sample(rng, samples, (100.0, 1000.0), (0.05, 0.80))
    observed = train_truth + rng.normal(0.0, noise_fraction * np.maximum(train_truth, 20.0))
    interpolation_x, interpolation_truth = _sample(rng, 250, (100.0, 1000.0), (0.05, 0.80))
    extrapolation_x, extrapolation_truth = _sample(rng, 250, (1000.0, 1500.0), (0.80, 1.20))
    mean, std = train_x.mean(axis=0), train_x.std(axis=0)
    scale = float(train_truth.std())
    data_only = TinyRegressor(mean, std, scale, seed)
    physics_informed = TinyRegressor(mean, std, scale, seed)
    data_only.fit(train_x, observed, physics_weight=0.0)
    physics_informed.fit(train_x, observed, physics_weight=physics_weight)
    di, pi = data_only.predict(interpolation_x), physics_informed.predict(interpolation_x)
    de, pe = data_only.predict(extrapolation_x), physics_informed.predict(extrapolation_x)
    result = ExperimentResult(
        seed=seed, samples=samples, noise_fraction=noise_fraction, physics_weight=physics_weight,
        data_only_interpolation=_metrics(di, interpolation_truth, interpolation_x),
        physics_informed_interpolation=_metrics(pi, interpolation_truth, interpolation_x),
        data_only_extrapolation=_metrics(de, extrapolation_truth, extrapolation_x),
        physics_informed_extrapolation=_metrics(pe, extrapolation_truth, extrapolation_x),
    )
    arrays = {"interpolation_truth": interpolation_truth, "data_only_interpolation": di, "physics_informed_interpolation": pi,
              "extrapolation_truth": extrapolation_truth, "data_only_extrapolation": de, "physics_informed_extrapolation": pe}
    return result, arrays


def _plot(arrays: dict[str, np.ndarray], path: Path) -> None:
    width, height, pad, gap = 1100, 480, 70, 70
    panel_w = (width - 2 * pad - gap) / 2
    panel_h = 310
    colors = {"data_only": "#D97706", "physics_informed": "#17638F"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#173B5E}.axis{stroke:#667085;stroke-width:1}.ideal{stroke:#111827;stroke-dasharray:6 5}.pt{opacity:.48}</style>',
             '<text x="550" y="28" text-anchor="middle" font-size="20" font-weight="700">Synthetic scientific-ML benchmark</text>']
    for index, split in enumerate(("interpolation", "extrapolation")):
        x0 = pad + index * (panel_w + gap); y0 = 65
        truth = arrays[f"{split}_truth"]
        predictions = np.concatenate((arrays[f"data_only_{split}"], arrays[f"physics_informed_{split}"]))
        lo, hi = float(min(truth.min(), predictions.min())), float(max(truth.max(), predictions.max()))
        span = max(hi - lo, 1.0); lo -= span * 0.05; hi += span * 0.05
        sx = lambda value: x0 + (float(value) - lo) / (hi - lo) * panel_w
        sy = lambda value: y0 + panel_h - (float(value) - lo) / (hi - lo) * panel_h
        parts += [f'<line class="axis" x1="{x0}" y1="{y0+panel_h}" x2="{x0+panel_w}" y2="{y0+panel_h}"/>',
                  f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+panel_h}"/>',
                  f'<line class="ideal" x1="{sx(lo)}" y1="{sy(lo)}" x2="{sx(hi)}" y2="{sy(hi)}"/>',
                  f'<text x="{x0+panel_w/2}" y="52" text-anchor="middle" font-size="16" font-weight="700">{split.title()}</text>',
                  f'<text x="{x0+panel_w/2}" y="414" text-anchor="middle" font-size="13">Physical emissions (kg CO2e)</text>']
        for model in ("data_only", "physics_informed"):
            for x_value, y_value in zip(truth, arrays[f"{model}_{split}"], strict=True):
                parts.append(f'<circle class="pt" cx="{sx(x_value):.2f}" cy="{sy(y_value):.2f}" r="2.2" fill="{colors[model]}"/>')
    parts += ['<circle cx="385" cy="452" r="5" fill="#D97706"/><text x="397" y="457" font-size="13">Data only</text>',
              '<circle cx="515" cy="452" r="5" fill="#17638F"/><text x="527" y="457" font-size="13">Physics informed</text>',
              '<line class="ideal" x1="690" y1="452" x2="720" y2="452"/><text x="730" y="457" font-size="13">Ideal</text>', '</svg>']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--noise-fraction", type=float, default=0.15)
    parser.add_argument("--physics-weight", type=float, default=1.5)
    parser.add_argument("--output", type=Path, default=Path("outputs/sciml_benchmark.json"))
    parser.add_argument("--figure", type=Path, default=Path("figures/sciml_benchmark.svg"))
    args = parser.parse_args()
    result, arrays = run_experiment(seed=args.seed, samples=args.samples, noise_fraction=args.noise_fraction, physics_weight=args.physics_weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    _plot(arrays, args.figure)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
