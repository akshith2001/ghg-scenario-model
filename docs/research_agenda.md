# GHG Scenario Model research agenda

## Working title

**Transparent Cross-Sector Emissions Scenarios Under Parameter Uncertainty**

## Research proposition

Scenario modelling is most useful when assumptions, system boundaries,
emission factors and uncertainty are visible enough for another researcher to
reproduce and challenge the result.

The current software is a research prototype. Its outputs are conditional
scenario estimates, not measurements, causal effects or forecasts.

## Research gap

Small organisations and researchers need models that are easier to audit than
large integrated assessment systems but more rigorous than spreadsheet totals.
The open research gap is a compact cross-sector workflow that traces every
estimate to activity data and a factor, propagates uncertainty and reports when
mitigation rankings are unstable.

## Primary research questions

1. Which input assumptions dominate uncertainty in cross-sector baseline
   emissions and reduction scenarios?
2. How stable are sector and intervention rankings across plausible
   emission-factor distributions and activity-data ranges?
3. How do alternative system boundaries change the apparent value of
   electricity, heating, transport, waste and water-related interventions?
4. Can provenance-first reporting improve reproducibility and prevent scenario
   results from being mistaken for forecasts?

## Working hypotheses

- A small subset of activity and factor inputs explains most output variance.
- Some intervention rankings reverse under plausible uncertainty, making robust
  ranges more useful than point estimates.
- Explicit provenance and boundary labels reduce unsupported comparison between
  scenarios built from incompatible assumptions.

## Proposed study

1. Freeze the functional unit, reporting year, geography, gases, scopes and
   lifecycle boundary before calculation.
2. Create a factor registry containing source, version, unit, geography, valid
   year, uncertainty and conversion logic.
3. Compute the baseline with unit-checked activity-by-factor equations and a
   complete sector contribution table.
4. Define each mitigation scenario as transparent changes to named inputs.
5. Run one-at-a-time sensitivity and Monte Carlo analysis with justified input
   distributions and correlations where evidence supports them.
6. Report medians, intervals, variance contributions and rank-reversal
   frequency instead of presenting a single scenario total as a prediction.
7. Reconcile the model against at least one independently prepared inventory
   and document every material difference.

## Hospitality case-study boundary

The hospitality water-use example is motivated by observation of restaurant
operations, but its input values are editable and partly illustrative. It can
become research evidence only after flow, operating time, temperature, energy
source and service constraints are measured under a documented protocol.

## Publication boundary

The strongest claim available today is that the repository provides an open,
inspectable implementation of cross-sector scenario and uncertainty methods.
It does not yet demonstrate a measured real-world emissions reduction or a
validated forecast.
