> HISTORICAL RECORD: predates the validation/accounting correction. Numerical results and inferential claims below are not current validated findings. See [correction notes](synthetic_grid_corrections.md).

# Locked protocol: physics-informed generative models for synthetic feeder energy data (pilot)

Status: written before any model was trained or any result seen. Changes after the first run are
logged under "Deviations" at the bottom and never edited above.

## Scope and honesty statement
A small pilot on SIMULATED data (no real grid data is accessible from this workspace). It tests
whether a physics-aware design helps a generative model for load time series on a radial network.
It does not claim anything about real distribution networks, and it uses a pure-NumPy VAE because
no deep-learning framework is installed. Results are a method study, not a benchmark.

## Questions
- Q1 (physical consistency): Does a generative model that outputs injections AND voltages jointly
  break the power-flow relation, and do a soft penalty or a hard physics layer fix it?
- Q2 (fidelity cost): Does enforcing physics cost statistical fidelity?
- Q3 (utility and privacy): Is the synthetic data useful for a downstream task (train-synthetic-
  test-real, TSTR) and does it avoid copying training days?

## Data (simulated, fixed)
- Radial 9-bus feeder (slack + 8 load buses), fixed topology and impedances, nonlinear AC power
  flow solved by forward-backward sweep (pure NumPy, vectorised).
- One sample = one day, 8 buses x 24 hourly net injections (kW, positive = consumption). Three
  buses carry PV (negative injection by day).
- Shared weather drivers (temperature, cloudiness), residential and commercial shapes, weekday
  effect, bus-level noise. 1500 days per seed: 800 train, 100 validation, 600 real-test.
- Five seeds (0-4) vary the simulated data and the model initialisation.

## Arms (fixed)
1. GAUSS: multivariate Gaussian (PCA, 24 components) on injections; voltages from the load-flow solver.
2. VAE-FREE: VAE on [injections, voltages] jointly, no physics term.
3. VAE-SOFT: as VAE-FREE plus penalty lambda * ||v - (A p + b)||^2, with a LINEAR surrogate of the
   power flow in the loss (lambda fixed at 1.0 on standardised features, not tuned on test).
4. VAE-HARD: VAE on injections only; voltages from the exact load-flow solver (physics layer).
VAE: one hidden layer of 128 tanh units, latent dim 8, Adam, 300 epochs, batch 64, KL weight 1.0,
early selection on validation loss. Same settings for all VAE arms.

## Metrics (fixed; all computed against the real-test set)
- PF residual: RMSE between a sample's generated voltages and the exact load-flow voltages of its
  generated injections (pu). Lower is better.
- Voltage violation gap: absolute difference between the share of days with any bus below 0.95 pu
  in synthetic vs real (voltages from the exact solver on injections). Lower is better.
- Fidelity: (a) mean per-feature 1-D Wasserstein distance on standardised injections; (b) Frobenius
  error between correlation matrices of the injections.
- TSTR: HistGradientBoosting trained on synthetic, tested on real-test, predicting the 17:00-21:00
  total feeder load from hours 0-11; report MAE ratio vs a model trained on real-train (TRTR).
- Memorisation: share of synthetic days whose nearest real-train day (Euclidean, standardised) is
  closer than the 5th percentile of real-train-to-real-train nearest-neighbour distances.
- Uncertainty: mean and sd over 5 seeds; paired differences with a 95% interval from seed-level
  differences (t-interval, n=5).

## Predeclared outcomes
- Q1 supported if VAE-HARD has PF residual ~0 (< 1e-6) and VAE-SOFT's residual is smaller than
  VAE-FREE's in at least 4 of 5 seeds.
- Q2 supported ("physics is not free") only if VAE-HARD or VAE-SOFT has worse fidelity (a) than
  VAE-FREE with a seed-paired interval excluding zero; otherwise reported as no detectable cost.
- Q3 is descriptive. A null or negative result is reported as such. Anything added after the first
  run is labelled exploratory.

## Known limits to state
Simulated data; one topology; hourly resolution; small NumPy VAE rather than diffusion or flow
matching; linear surrogate used in the soft penalty; no differential privacy (memorisation share is
a crude proxy only); results do not transfer to real Alliander data without a new study.

## Deviations
- Simulator calibration (before any model was trained): the first impedance values made every day
  violate the 0.95 pu limit, so a single scale factor (0.43) was applied to all line impedances so that
  about 10% of days violate. Chosen using simulated data only; no model output was seen.
- Integration into `ghg-scenario-model` (after the first run): two EXPLORATORY emissions-accounting
  metrics were added (`emis_err_flat`, `emis_err_tou`). The locked metrics were re-run through the
  repository command and match the standalone run exactly (maximum absolute difference 0).
