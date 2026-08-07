# Cross-Sector Greenhouse Gas Scenario Model

A small, transparent Python model for estimating greenhouse-gas emissions and testing reduction scenarios across electricity, heating, transport, waste, and industrial activity.

This portfolio project was created to demonstrate reproducible modelling, scenario design, uncertainty analysis, documentation, and version-control-ready research practice. It is a prototype, not a regulatory carbon-accounting tool.

## Research question

How do different sector-level interventions change total annual greenhouse-gas emissions, and how sensitive are the results to uncertainty in activity data and emission factors?

## Method

For each activity record:

`emissions (kg CO2e) = activity × emission factor`

A scenario changes activity, the emission factor, or both. Independent relative uncertainties are combined using root-sum-of-squares propagation.

## Included scenarios

- `baseline`: no intervention
- `balanced_transition`: lower electricity demand, grid decarbonisation, heating efficiency, transport electrification, waste reduction, and industrial efficiency
- `rapid_decarbonisation`: more ambitious reductions across the same sectors

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
ghg-model --activities data/sample_activities.csv --scenarios data/scenarios.json --output outputs/results.csv
ghg-chart
python -m unittest discover -s tests -v
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

## Example output

![Scenario emissions comparison](figures/scenario_totals.png)

## Data and provenance

`data/sample_activities.csv` contains a synthetic demonstration dataset. Its emission factors are illustrative and must not be used for formal reporting. The model keeps factor year, source, unit, and uncertainty alongside every record so authoritative factors can be substituted without changing the code.

For a real study, use the latest UK Government greenhouse-gas conversion-factor flat file or another jurisdiction-appropriate inventory source, document all mapping decisions, and retain the original source version. Relevant methodological foundations include:

- UK Government greenhouse-gas reporting conversion factors: https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting
- 2006 IPCC Guidelines for National Greenhouse Gas Inventories: https://www.ipcc-nggip.iges.or.jp/public/2006gl/

## Repository structure

```text
data/                  Synthetic activities and scenario assumptions
src/ghg_model/         Model and command-line interface
tests/                 Unit tests
outputs/               Generated results (created when the model runs)
```

## Limitations and next steps

- Replace illustrative factors with an automated, versioned import from an authoritative source.
- Add direct and indirect emissions with explicit Scope 1, 2, and 3 treatment.
- Add spatial and hourly resolution for integration with an energy-system model.
- Model correlations and probability distributions using Monte Carlo simulation.
- Validate sector mappings and scenario assumptions with domain experts.
- Add optimisation to identify least-cost pathways subject to an emissions target.

## Ethical and reproducibility note

Results are only as reliable as their boundaries, data, factors, and assumptions. Scenario outputs should be reported with uncertainty and should not be presented as forecasts.
