# Changelog

## 0.5.1 - 2026-11-16

- Integrated Luxembourg planning and measured-data modules onto the corrected published synthetic-grid implementation.
- Preserved line-loss accounting, daily input support, convergence validation, independent nodal diagnostics and descriptive rolling summaries.
- Added validated measured hourly factor shapes without changing the existing daily-resolution API.
- Applied a sign-aware export spread to eliminate profitable circular trade at negative prices.
- Included all four module-level regional/price tests in unittest discovery; added five merge regression tests.
- Fleet solver failures now raise; price iterations report convergence and the undamped fixed-point residual.
- Imported numerical claims and figures are explicitly historical pending full reproduction.

## Imported source history

The entries below describe the supplied v0.5 archive, not the chronology or validation status of this repository.


## 0.5.0 - 2026-11-16

### Added
- `regional_marginal`: marginal factor for DE-LU demand estimated from emissions in
  Germany and eight neighbours (FR, BE, NL, AT, CH, CZ, PL, DK). It controls for
  neighbours' own residual load, takes the level from daily totals and the hourly shape
  from hourly bins, and reports bootstrap ranges and the response split by country.
  Result: 317 (2024) and 333 (2025) g/kWh, range 276-394, replacing the 200-480 bracket.
- `price_feedback`: supply curve estimated from data, competitive zone storage
  dispatch, prices shifted along the curve, and a fixed-point Luxembourg plan with
  price feedback. Scenarios add 0-100 GWh of storage across DE-LU.
- Neighbour hourly datasets for 2024 and 2025, and a price-feedback figure.

### Changed
- The regional estimate is now the central case. The Monte Carlo samples its level
  instead of blending the old bounds.

## 0.4.0 - 2026-11-16

### Changed
- The Luxembourg case now runs on measured hourly data for 2024 and 2025 (Energy-Charts,
  CC BY 4.0). Marginal factors are estimated from data and reported as a range: a lower
  bound from German plants only (about 200 g/kWh) and an upper estimate from German
  dispatchable response (about 470-480 g/kWh). The average import intensity is about
  256 g/kWh. Result: the marginal rule never builds less solar, and builds up to 1,000 MW
  more near the upper estimate (regret 0.6-2.8%). At the lower bound the two rules agree.
- Battery size comes from full-year (8,760 h) plans and is now reported as a break-even
  cost curve: about 1.6-3.0 GWh at EUR 150/kWh, none at EUR 200/kWh or more.

### Added
- `real_case` (`ghg-luxembourg-real`): full-year plans per marginal estimate, a Monte
  Carlo over position in the marginal range and the cost and policy inputs,
  out-of-sample plan tests across years, robustness checks and three figures.
- `energycharts.estimate_marginal` with `german_plants` and `dispatchable` methods;
  German cross-border trade and pumped storage columns; the 2024 hourly dataset.
- `synthetic_grid` emissions functions accept a time-of-use shape taken from measured data.

## Earlier unreleased work, included in 0.4.0

### Added
- `synthetic_grid` merged from the physics-informed synthetic energy project: simulated
  9-bus radial feeder with exact AC load flow (forward-backward sweep), Gaussian, GMM and
  NumPy VAE generators (unconstrained, soft penalty, hard physics layer), three locked
  studies (simulated, real BDG2 buildings, rolling origins), results, figures and a bridge
  to `Activity` records. Study 1 reruns match the committed results to within 1e-12
  relative difference.
- `energycharts`: loader and fetcher for measured 2025 hourly data from the public
  Energy-Charts API (Fraunhofer ISE, CC BY 4.0) - Luxembourg load, generation, physical
  flows with Germany and Belgium, DE-LU prices, and German generation for estimating
  marginal emission factors.

## 0.3.0 - 2026-11-16

### Added
- Solar cost supply curve (three tiers), so plans can stop between zero and the cap.
- Battery wear cost (capex / cycle life) and a daily cycle limit.
- Imports by border: fixed physical flow shares with interconnector limits, and a
  free-choice mode that exposes contract-based accounting.
- `marginal_emissions`: German merit-order marginal factors by fuel-price regime and a
  Hawkes-style regression estimator, validated on a synthetic zone with known answers.
- `foresight`: battery value retained under price-forecast errors and a derating step.
- `ghg-entsoe --estimate-marginal`: marginal factors from measured German generation.
- Robustness table and contract-loophole analysis in the case study; a third figure.
- 15 further tests (73 in total).

### Changed
- The case study's central configuration now uses merit-order marginal factors,
  flow-share imports, tiered solar, battery wear and foresight derating.

## 0.2.0 - 2026-11-16

### Added
- Hourly capacity-expansion and dispatch linear programme (`ghg_model.hourly_dispatch`):
  solar, battery storage, imports, exports and curtailment, solved with SciPy HiGHS,
  with average and marginal carbon-accounting rules and plan cross-evaluation.
- Luxembourg 2025 case study (`ghg-luxembourg`): sourced inputs, calibrated synthetic
  hourly year, representative days, Monte Carlo decision-flip and regret analysis,
  two figures and `docs/luxembourg_case_study.md`.
- ENTSO-E Transparency Platform client (`ghg-entsoe`) to replace synthetic hourly
  shapes with measured load, solar, wind and price data.
- 19 tests for profiles, dispatch, case study and the ENTSO-E parser.

### Changed
- README: new case-study section, updated limitations, and the Zenodo DOI is now live.

## 0.1.0 - 2026-09-09

- Initial research release: cross-sector scenario model, sensitivity and Monte Carlo
  analysis, least-cost abatement optimisation, scientific-ML benchmark and EPA eGRID
  validation.
