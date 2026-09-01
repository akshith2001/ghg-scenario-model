# Scientific-ML foundation experiment

## Question

Can a small neural surrogate trained on noisy synthetic observations become more
consistent with the known greenhouse-gas accounting equation when that equation is
included directly in the training objective?

## Design

The benchmark generates synthetic activity and emission-factor pairs. The noiseless
target follows the auditable relationship:

`emissions = activity × emission factor`

The same one-hidden-layer neural network is trained twice from identical initial
weights. The data-only model minimises prediction error against noisy observations.
The physics-informed model minimises the same error plus a penalty for disagreement
with the physical equation. Both are tested on held-out interpolation data and on an
extrapolation range outside the training distribution.

Run the locked default experiment with:

```bash
ghg-sciml
```

The command writes machine-readable metrics to `outputs/sciml_benchmark.json` and a
prediction-versus-physical-target figure to `figures/sciml_benchmark.svg`.

## Locked default result

With seed 2026, 600 synthetic training samples, 15% observation noise and a physics
weight of 1.5, the physics-informed model reduced interpolation RMSE from 9.69 to
6.22 kg CO2e (35.8%). On the deliberate extrapolation split, RMSE fell from 455.98
to 400.56 kg CO2e (12.2%). The remaining extrapolation error is substantial. The
result therefore supports the limited claim that the constraint improves consistency
in this setup; it does not show that a small neural network can extrapolate reliably.

## Interpretation boundary

This is a deliberately small educational experiment. It demonstrates the mechanics
of a physics-informed loss, deterministic evaluation and out-of-range testing. The
data are synthetic; the governing equation is simple; and the neural network is
implemented directly with NumPy for transparency. It is not evidence that the model
is suitable for real emissions inventories, inverse problems, industrial control or
scientific foundation models. A next study would use an established deep-learning
framework, multiple random seeds, stronger baselines, partial/noisy physics, genuine
spatiotemporal data and uncertainty-aware neural operators.
