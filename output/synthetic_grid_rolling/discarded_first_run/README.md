# Discarded first run (kept for transparency)

This run used the metric definitions locked in `docs/synthetic_grid_rolling_protocol.md`, which
standardise by the standard deviation of the real TEST window. At origin 240 two meters
(Lamb_food_Sylvia, Lamb_lodging_Harley) report a constant 0.01 kWh per day for the whole test window, so
that standard deviation is zero and the Wasserstein distance divides by about 1e-8, producing values of
about 6.7e5 and wrecking the means. The metric definition was fixed (see the protocol's Deviations
section) and the study was re-run. The numbers in this folder should not be used.
