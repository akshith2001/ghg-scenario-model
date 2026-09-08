# First real-data validation: EPA eGRID natural-gas plants

## Purpose

This study tests the core `activity × emission factor` calculation against
reported plant emissions. It is intentionally narrower than the synthetic
scientific-ML experiment and should not be described as contemporary,
European or multi-sector validation.

## Data and locked design

- Source: US EPA ArcGIS eGRID-derived natural-gas plant layer.
- Data year: 2018.
- Records retrieved: 1,593.
- Eligible records: positive annual combustion heat input and reported CO2.
- Target: `PLCO2AN`, reported annual plant CO2 emissions in short tons.
- External physical factor: 53.06 kg CO2/MMBtu from EPA's 2025 GHG Emission
  Factors Hub.
- Held-out rule: plants whose ORISPL identifier is divisible by five are never
  used for fitting.

The physical prediction is:

`CO2 short tons = heat input MMBtu × 53.06 kg CO2/MMBtu ÷ 907.18474`

The benchmark compares a data-only ridge model, the external-factor equation,
and a ridge residual correction anchored to the physical estimate.

## Locked result

The deterministic split retained 1,261 plants for fitting and held out 320
plants. All three approaches were evaluated on exactly the same held-out set.

| Model | RMSE (short tons) | MAE (short tons) | Median absolute percentage error |
|---|---:|---:|---:|
| Data-only ridge | 32,058.72 | 9,694.49 | 1.88% |
| External-factor equation | 32,541.30 | 9,920.03 | 1.58% |
| Physics-guided residual | 32,079.37 | 9,579.33 | 1.43% |

The physics-guided model reduced median percentage error by 23.9% relative to
the data-only model and reduced MAE by 1.2%. It did **not** reduce RMSE: its
RMSE was 0.06% higher than the data-only result. A small number of very large
plants dominate squared error, so this result supports a limited claim about
typical relative error, not universal superiority.

### Paired uncertainty and failure slices

The benchmark now uses 2,000 paired bootstrap resamples of the same 320 held-out
plants. The estimand is the physics-guided median absolute percentage error
minus the data-only value; negative values favour the physics-guided model.
The JSON artifact records the locked seed, percentile interval and the share of
plants on which the guided model has lower absolute error.

The observed paired difference was -0.45 percentage points with a 95%
bootstrap interval from -0.64 to -0.25 percentage points. The guided model had
lower absolute error on 57.5% of held-out plants. This interval quantifies
sampling variation within the fixed 2018 holdout; it does not address temporal
or dataset shift.

Results are also reported across quartiles of independently reported plant CO2.
These slices are diagnostic and were added after the original aggregate result;
they must not be used to retune the locked models or relabel the first test as a
preregistered analysis.

Median percentage error was lower for the guided model in all four post-hoc
size quartiles. The largest change occurred in the smallest-emissions quartile
(3.29% to 1.64%); this diagnostic observation requires replication on a new
year before it can support a general claim.

## Leakage controls

No eGRID emission-rate field is supplied as a feature. In particular,
`PLCO2RTA`, `PLCO2RA` and related target-derived rate fields are excluded.
The reported `PLCO2AN` value is used only as the evaluation target and, for
the two fitted models, in the training partition.

## Interpretation boundaries

- The split tests unseen plants in one historical year, not future years.
- Both activity and target originate in the same EPA data product, although
  the physical factor is external to the eGRID target.
- Natural-gas composition, CHP adjustments, biomass adjustments and reporting
  methods can create legitimate deviations from the fixed factor.
- The physics-guided residual model is a transparent statistical hybrid, not
  a physics-informed neural network.
- This result is a foundation for, not a substitute for, a temporal eGRID
  benchmark and an EU ETS replication.

## Reproduce

```bash
python scripts/fetch_egrid_validation_data.py
ghg-real-validation
```

The machine-readable result is written to `outputs/egrid_validation.json` and
the checked-in figure to `figures/egrid_validation.svg`.
