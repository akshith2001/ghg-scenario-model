"""Core calculation functions for the GHG scenario model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
from typing import Mapping


@dataclass(frozen=True)
class Activity:
    activity_id: str
    sector: str
    activity_name: str
    activity_value: float
    activity_unit: str
    emission_factor: float
    factor_unit: str
    factor_year: str
    factor_source: str
    activity_uncertainty_pct: float = 0.0
    factor_uncertainty_pct: float = 0.0

    def validate(self) -> None:
        if self.activity_value < 0 or self.emission_factor < 0:
            raise ValueError("Activity values and emission factors must be non-negative")
        if self.activity_uncertainty_pct < 0 or self.factor_uncertainty_pct < 0:
            raise ValueError("Uncertainty percentages must be non-negative")


@dataclass(frozen=True)
class ModelResult:
    activity_id: str
    sector: str
    activity_name: str
    scenario: str
    activity_value: float
    emission_factor: float
    emissions_kg_co2e: float
    uncertainty_kg_co2e: float
    lower_kg_co2e: float
    upper_kg_co2e: float


def calculate_emissions(activity: Activity, scenario_name: str = "baseline") -> ModelResult:
    activity.validate()
    emissions = activity.activity_value * activity.emission_factor
    relative_uncertainty = sqrt(
        activity.activity_uncertainty_pct**2 + activity.factor_uncertainty_pct**2
    ) / 100
    uncertainty = emissions * relative_uncertainty
    return ModelResult(
        activity_id=activity.activity_id,
        sector=activity.sector,
        activity_name=activity.activity_name,
        scenario=scenario_name,
        activity_value=activity.activity_value,
        emission_factor=activity.emission_factor,
        emissions_kg_co2e=emissions,
        uncertainty_kg_co2e=uncertainty,
        lower_kg_co2e=max(0.0, emissions - uncertainty),
        upper_kg_co2e=emissions + uncertainty,
    )


def apply_scenario(
    activity: Activity,
    scenario_name: str,
    sector_changes: Mapping[str, Mapping[str, float]],
) -> ModelResult:
    changes = sector_changes.get(activity.sector, {})
    activity_multiplier = 1 + changes.get("activity_change_pct", 0.0) / 100
    factor_multiplier = 1 + changes.get("factor_change_pct", 0.0) / 100
    if activity_multiplier < 0 or factor_multiplier < 0:
        raise ValueError("Scenario changes cannot make activity or factors negative")
    adjusted = replace(
        activity,
        activity_value=activity.activity_value * activity_multiplier,
        emission_factor=activity.emission_factor * factor_multiplier,
    )
    return calculate_emissions(adjusted, scenario_name)
