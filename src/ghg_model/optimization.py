"""Least-cost abatement pathway optimisation.

This module extends the scenario model with a discrete optimisation layer:
given a portfolio of candidate abatement measures (each with a cost and an
emissions-reduction potential), find the lowest-cost combination that meets
or exceeds a target total abatement.

Method
------
Each measure is treated as an indivisible ("0/1 knapsack") decision: either
adopted in full or not adopted, which better reflects real interventions
(replace a boiler, retrofit lighting) than allowing arbitrary fractional
adoption. The problem is solved as a mixed-integer linear program using
``scipy.optimize.milp``:

    minimise   sum(cost_i * x_i)
    subject to sum(abatement_i * x_i) >= target_abatement
               x_i in {0, 1}

A continuous relaxation (``allow_partial=True``) is also provided for
comparison; it will generally suggest a lower cost than is achievable in
practice, because it allows partially adopting a measure.

Robustness
----------
Costs and abatement potentials are rarely known precisely. ``rank_reversal_analysis``
repeatedly perturbs every measure's cost and abatement within a stated
relative uncertainty, re-solves the optimisation, and reports how often each
measure appears in the optimal set. A measure selected in the base case but
included in, say, only 40% of perturbed solutions is a fragile choice, not a
robust one — this directly extends the "test whether intervention rankings
reverse across plausible input distributions" item already listed in the
project's roadmap.

This module does not claim to find a real-world optimum. It is a
transparent, reproducible demonstration of the method; see
``docs/optimization_pathway.md`` for the full interpretation boundary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, milp, Bounds, linprog


@dataclass(frozen=True)
class AbatementOption:
    """A single candidate intervention.

    Parameters
    ----------
    name:
        Short identifier, e.g. "LED retrofit - kitchen".
    sector:
        One of the model's existing sectors (electricity, heating, transport,
        waste, industrial) so results can be cross-referenced with the
        scenario model.
    annual_cost_gbp:
        Illustrative annualised net cost in GBP. Positive values are a net
        cost; negative values represent a net saving (e.g. reduced utility
        bills exceed the amortised capital cost).
    abatement_kgco2e_per_year:
        Illustrative annual emissions reduction if fully adopted.
    cost_uncertainty_pct / abatement_uncertainty_pct:
        Relative uncertainty (as a fraction, e.g. 0.2 for +/-20%) used only
        by the robustness analysis. These are illustrative placeholders and
        must be replaced with evidence-based ranges before any real
        decision use.
    """

    name: str
    sector: str
    annual_cost_gbp: float
    abatement_kgco2e_per_year: float
    cost_uncertainty_pct: float = 0.2
    abatement_uncertainty_pct: float = 0.2


@dataclass
class OptimizationResult:
    selected: list[str]
    total_cost_gbp: float
    total_abatement_kgco2e: float
    target_abatement_kgco2e: float
    feasible: bool
    relaxed: bool = False
    fractional_adoption: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "total_cost_gbp": round(self.total_cost_gbp, 2),
            "total_abatement_kgco2e": round(self.total_abatement_kgco2e, 2),
            "target_abatement_kgco2e": round(self.target_abatement_kgco2e, 2),
            "feasible": self.feasible,
            "relaxed_lp_lower_bound": self.relaxed,
            "fractional_adoption": {
                k: round(v, 4) for k, v in self.fractional_adoption.items()
            },
        }


def load_options(path: str | Path) -> list[AbatementOption]:
    """Load a JSON list of abatement options from disk."""
    data = json.loads(Path(path).read_text())
    return [AbatementOption(**row) for row in data]


def solve_least_cost_pathway(
    options: list[AbatementOption],
    target_abatement_kgco2e: float,
    allow_partial: bool = False,
) -> OptimizationResult:
    """Solve for the least-cost set of measures meeting the target.

    ``allow_partial=False`` (default) solves the realistic 0/1 problem via
    MILP. ``allow_partial=True`` solves the continuous relaxation via LP and
    returns a fractional-adoption lower bound on cost; it is reported
    separately and must not be presented as an achievable real-world plan.
    """
    if not options:
        raise ValueError("options must be a non-empty list")
    if target_abatement_kgco2e < 0:
        raise ValueError("target_abatement_kgco2e must be non-negative")

    costs = np.array([o.annual_cost_gbp for o in options], dtype=float)
    abatement = np.array([o.abatement_kgco2e_per_year for o in options], dtype=float)
    names = [o.name for o in options]

    if target_abatement_kgco2e == 0:
        return OptimizationResult(
            selected=[], total_cost_gbp=0.0, total_abatement_kgco2e=0.0,
            target_abatement_kgco2e=0.0, feasible=True, relaxed=allow_partial,
        )

    max_possible = abatement.clip(min=0).sum()
    feasible_target = target_abatement_kgco2e <= max_possible + 1e-9

    # sum(abatement_i * x_i) >= target  <=>  -abatement . x <= -target
    constraint = LinearConstraint(-abatement, -np.inf, -target_abatement_kgco2e)

    if allow_partial:
        # LP relaxation: 0 <= x_i <= 1, minimise cost . x
        bounds = [(0.0, 1.0)] * len(options)
        res = linprog(
            c=costs, A_ub=-abatement.reshape(1, -1),
            b_ub=[-target_abatement_kgco2e], bounds=bounds, method="highs",
        )
        if not res.success or not feasible_target:
            return OptimizationResult(
                selected=[], total_cost_gbp=float("nan"),
                total_abatement_kgco2e=0.0,
                target_abatement_kgco2e=target_abatement_kgco2e,
                feasible=False, relaxed=True,
            )
        x = res.x
        selected = [n for n, xi in zip(names, x) if xi > 1e-6]
        fractional = {n: float(xi) for n, xi in zip(names, x) if xi > 1e-6}
        return OptimizationResult(
            selected=selected,
            total_cost_gbp=float(costs @ x),
            total_abatement_kgco2e=float(abatement @ x),
            target_abatement_kgco2e=target_abatement_kgco2e,
            feasible=True, relaxed=True, fractional_adoption=fractional,
        )

    if not feasible_target:
        return OptimizationResult(
            selected=[], total_cost_gbp=float("nan"), total_abatement_kgco2e=0.0,
            target_abatement_kgco2e=target_abatement_kgco2e, feasible=False,
        )

    n = len(options)
    result = milp(
        c=costs,
        constraints=[constraint],
        integrality=np.ones(n),
        bounds=Bounds(lb=np.zeros(n), ub=np.ones(n)),
        options={"disp": False},
    )
    if not result.success:
        return OptimizationResult(
            selected=[], total_cost_gbp=float("nan"), total_abatement_kgco2e=0.0,
            target_abatement_kgco2e=target_abatement_kgco2e, feasible=False,
        )

    x = np.round(result.x).astype(int)
    selected = [n for n, xi in zip(names, x) if xi == 1]
    return OptimizationResult(
        selected=selected,
        total_cost_gbp=float(costs @ x),
        total_abatement_kgco2e=float(abatement @ x),
        target_abatement_kgco2e=target_abatement_kgco2e,
        feasible=True,
    )


def rank_reversal_analysis(
    options: list[AbatementOption],
    target_abatement_kgco2e: float,
    samples: int = 500,
    seed: int = 2026,
) -> dict:
    """Perturb costs/abatement within their stated uncertainty and re-solve.

    Returns, for each option, the fraction of perturbed trials in which it
    appeared in the optimal (least-cost, 0/1) solution, plus the fraction of
    trials that were feasible at all. A high selection frequency indicates a
    robust choice; a low one indicates the base-case result is fragile to
    plausible input error.
    """
    rng = np.random.default_rng(seed)
    names = [o.name for o in options]
    selection_counts = {n: 0 for n in names}
    feasible_trials = 0

    for _ in range(samples):
        perturbed = []
        for o in options:
            cost_factor = 1 + rng.uniform(-o.cost_uncertainty_pct, o.cost_uncertainty_pct)
            abate_factor = 1 + rng.uniform(
                -o.abatement_uncertainty_pct, o.abatement_uncertainty_pct
            )
            perturbed.append(
                AbatementOption(
                    name=o.name,
                    sector=o.sector,
                    annual_cost_gbp=o.annual_cost_gbp * cost_factor,
                    abatement_kgco2e_per_year=max(
                        0.0, o.abatement_kgco2e_per_year * abate_factor
                    ),
                )
            )
        res = solve_least_cost_pathway(perturbed, target_abatement_kgco2e)
        if res.feasible:
            feasible_trials += 1
            for n in res.selected:
                selection_counts[n] += 1

    denom = max(feasible_trials, 1)
    return {
        "samples": samples,
        "feasible_trials": feasible_trials,
        "selection_frequency": {
            n: round(c / denom, 3) for n, c in selection_counts.items()
        },
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Solve for a least-cost abatement pathway meeting a target."
    )
    parser.add_argument("--options", required=True, help="Path to a JSON options file")
    parser.add_argument(
        "--target-kgco2e", type=float, required=True,
        help="Target annual abatement in kg CO2e",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Also report the continuous LP relaxation lower bound",
    )
    parser.add_argument(
        "--rank-reversal", action="store_true",
        help="Run the Monte Carlo rank-reversal robustness check",
    )
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    options = load_options(args.options)

    result = solve_least_cost_pathway(options, args.target_kgco2e)
    print("=== Least-cost pathway (0/1 adoption) ===")
    print(json.dumps(result.to_dict(), indent=2))

    if args.allow_partial:
        relaxed = solve_least_cost_pathway(
            options, args.target_kgco2e, allow_partial=True
        )
        print("\n=== Continuous LP relaxation (lower bound only) ===")
        print(json.dumps(relaxed.to_dict(), indent=2))

    if args.rank_reversal:
        summary = rank_reversal_analysis(
            options, args.target_kgco2e, samples=args.samples, seed=args.seed
        )
        print("\n=== Rank-reversal robustness check ===")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
