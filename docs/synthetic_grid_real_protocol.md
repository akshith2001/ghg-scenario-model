> HISTORICAL RECORD: predates the validation/accounting correction. Numerical results and inferential claims below are not current validated findings. See [correction notes](synthetic_grid_corrections.md).

# Locked protocol, Study 2: real-building replication

Status: written before any model was trained on the real data and before any real-data result
was seen. Changes after the first run are logged under "Deviations" and never edited above.

## Why a second study
Study 1 ([`synthetic_grid_protocol.md`](synthetic_grid_protocol.md)) used data from the author's own
simulator, so a reviewer can fairly say the generator and the models may share structure. Study 2
replaces the simulated loads with REAL building electricity data and keeps only the network physics
simulated. It asks whether the Study 1 conclusions survive.

## Data (fixed)
- `data/real/bdg2_hospitality_daily.csv`: 8 real hospitality buildings (6 food service, 2 lodging)
  from the open Building Data Genome Project 2 (BDG2 v1.0), daily electricity use in kWh, 2016-2017.
  The file is the extract produced by the author's `hospitality-sustainability-ai` importer.
- A sample is one calendar day on which ALL 8 buildings have a reading (645 such days).
- Temporal split: train = the first 282 complete days of 2016; validation = the remaining 50
  complete days of 2016 (chronological); real-test = all complete days of 2017 (313).
- Buildings are mapped to feeder buses 1-8 in alphabetical order of venue id (fixed before seeing
  any model output).

## What is real and what is not
Real: the building loads and their joint day-to-day variation, and the temporal shift between 2016
and 2017. NOT real: the feeder. The eight buildings are at different sites, so they do not share a
real network; the network, its impedances and the load-flow voltages are simulated. Only daily totals
exist, so each sample is a daily-mean operating point (kWh per day divided by 24 gives mean kW), not an
hourly profile. A single scale factor `s` converts mean kW to per-unit; it stands for `s` similar
premises per bus and is calibrated on TRAIN days only so that 10% of train days have any bus below
0.95 pu (bisection, before any model is trained). No PV exists in this data.

## Arms (fixed; same logic as Study 1, sizes adapted to 8 features)
GAUSS (full-covariance Gaussian on the 8 injections; voltages from the solver), VAE-FREE (joint
injections and voltages), VAE-SOFT (plus linear-surrogate penalty, weight 1.0), VAE-HARD (injections
only; voltages from the exact solver). VAEs: hidden 32, latent 3, Adam, 300 epochs, batch 64,
selection on validation loss. Five seeds (0-4). Data are fixed across seeds, so seed intervals reflect
training and sampling randomness only, not data-sampling variation; this is stated in the write-up.

## Metrics (fixed)
Power-flow residual; voltage-violation-rate gap; mean per-building 1-D Wasserstein distance on
standardised loads; 8x8 correlation-matrix error; TSTR (HistGradientBoosting trained on synthetic,
tested on real-test 2017) predicting the load of `Eagle_food_Kay` from the other seven buildings,
reported as a ratio to a model trained on real-train (the signal is weak because buildings are at
different sites, so this is a sanity check, not a strong utility test); memorisation share
(synthetic days closer to a train day than the 5th percentile of train-train nearest-neighbour
distances); exploratory: relative error of mean total daily energy (the emissions error under a flat
factor). Paired 95% t-intervals use the five seeds.

Reference line (added before the first run): the real 2016 train days scored against 2017 with the same
metrics. It shows how much of each metric is temporal drift that no model trained on 2016 could remove,
so arms are read against it and not against zero.

## Predeclared outcomes
- R1 (replication of the physics result): hard layer residual < 1e-6 and soft residual below the
  unconstrained VAE's in at least 4 of 5 seeds.
- R2 (replication of the baseline result): GAUSS has lower Wasserstein distance than VAE-HARD in at
  least 4 of 5 seeds.
- R3 (fidelity cost of physics): "supported" only if VAE-HARD or VAE-SOFT is worse than VAE-FREE on
  Wasserstein with a seed-paired interval that excludes zero.
- Null or contrary results are reported as such. Anything added after the first run is exploratory.

## Known limits to state
Eight buildings from different sites, 645 days, daily resolution, no real feeder, a calibrated scale
factor, very low dimension (8 features), a temporal split on two years only, and data fixed across seeds.
BDG2 was collected by third parties and is not hospitality-operations data in the full sense.

## Deviations
- Before the first run: the drift-floor reference line was added and the scale search changed to bracketing
  plus bisection that treats a non-converged load flow as a violation (the flow stops converging for far larger loads). No real-data model result had been seen.
