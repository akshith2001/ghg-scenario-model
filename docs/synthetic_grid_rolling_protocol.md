> HISTORICAL RECORD: predates the validation/accounting correction. Numerical results and inferential claims below are not current validated findings. See [correction notes](synthetic_grid_corrections.md).

# Locked protocol, Study 3: rolling-origin evaluation on real building loads

Status: written before any Study 3 model was trained and before any Study 3 result was seen.
Changes after the first run are logged under "Deviations" and never edited above.

## Why a third study
Study 2 used one chronological split (train 2016, test 2017) and fixed data across seeds, so its
intervals reflect training randomness only and one split can be lucky or unlucky. Study 3 repeats the
real-data comparison over five rolling origins, so the DATA changes between replicates, and adds a
stronger classical baseline.

## Data and windows (fixed)
- Same real BDG2 extract as Study 2 (8 buildings, daily kWh, days with all 8 readings, 645 days sorted).
- Five origins o = 0, 60, 120, 180, 240. For origin o, with `days` = the sorted complete days:
  train = days[o : o+240], validation = days[o+240 : o+280], test = days[o+280 : o+400].
- Each origin recalibrates the scale factor on ITS train window only (10% of train days below 0.95 pu),
  maps buildings to buses 1-8 alphabetically, and uses one model seed (seed = origin index).
- Windows overlap, so replicates are not independent. Intervals treat the five origins as replicates
  and are therefore optimistic; this is stated in the write-up.

## Arms (fixed)
GAUSS, VAE-FREE, VAE-SOFT, VAE-HARD exactly as in Study 2 (VAE: hidden 32, latent 3, 300 epochs, batch
64, Adam, validation selection). New: GMM, a full-covariance Gaussian mixture on the 8 standardised
injections with 1 to 4 components chosen by BIC on the train window, voltages from the exact solver.

## Metrics (fixed)
As Study 2: power-flow residual, voltage-violation-rate gap, per-building Wasserstein distance,
correlation-matrix error, TSTR ratio (target Eagle_food_Kay), exploratory error in mean total daily
energy. For every origin the drift-floor reference (the real train window scored against that origin's
test window) is computed with the same metrics, and results are reported as the arm's value MINUS the
floor ("excess over floor") as well as raw, so drift is not mistaken for model error.
Memorisation share is reported but not interpreted (it was zero everywhere in Study 2).

## Predeclared outcomes (paired over the five origins)
- S3-1: the hard layer's residual is below 1e-6 at every origin, and its Wasserstein difference against
  VAE-FREE has a paired 95% interval that includes zero or is negative (no fidelity cost).
- S3-2: GAUSS has a lower Wasserstein distance than VAE-HARD at 4 or more of the 5 origins.
- S3-3: GMM has a lower Wasserstein distance than GAUSS at 4 or more of the 5 origins.
- Null or contrary results are reported as such. Anything added after the first run is exploratory.

## Known limits to state
Daily totals only; eight buildings at different sites (no real feeder); overlapping windows; five
origins; scale factor calibrated per window; low dimension (8 features); no hourly data.

## Deviations
- After the first run (discarded, kept in `output/synthetic_grid_rolling/discarded_first_run/`): at origin
  240 two meters (Lamb_food_Sylvia, Lamb_lodging_Harley) read a constant 0.01 kWh per day for the whole
  test window, probably a meter outage or closed building. The locked metrics standardise by the test-window
  sd, which is then zero, so the Wasserstein distance divided by about 1e-8 and gave values near 6.7e5.
  The metric definition was therefore changed for Study 3 only: features are standardised by the real
  TRAIN window's mean and sd, and the correlation-matrix error uses only features that vary in the test
  window. Studies 1 and 2 keep their locked definitions (their windows have no flat meters). The two flat
  meters stay in the data; no windows or buildings were removed. Nothing else was changed, and the arms,
  origins and predeclared outcomes are as written above.
