# Corrected synthetic-energy study results

Full runs completed on 12 November 2026, with a 300-epoch training budget, five seeds for the simulated and chronological real-data studies, and five overlapping rolling origins. See the [run manifest](../output/synthetic_grid_reviewed/manifest.json) for versions and input fingerprints.

All 73 automated tests passed. These experiments validate a research workflow on a simulated feeder, not a real electricity network.

| Study | Arm | Mean emissions relative error | Mean voltage-limit day share | Maximum nodal imbalance (pu) |
|---|---|---:|---:|---:|
| simulated | GAUSS | 0.83% | 10.65% | 3.27e-11 |
| simulated | VAE-FREE | 0.71% | 9.28% | 2.43e-11 |
| simulated | VAE-SOFT | 0.67% | 9.53% | 1.92e-11 |
| simulated | VAE-HARD | 0.53% | 10.45% | 2.05e-11 |
| real | GAUSS | 6.60% | 12.48% | 9.36e-11 |
| real | VAE-FREE | 4.93% | 6.81% | 2.53e-11 |
| real | VAE-SOFT | 4.83% | 7.52% | 2.94e-11 |
| real | VAE-HARD | 5.83% | 8.51% | 5.05e-11 |
| rolling | GAUSS | 8.27% | 11.17% | 7.46e-11 |
| rolling | VAE-FREE | 6.90% | 6.58% | 2.49e-11 |
| rolling | VAE-SOFT | 7.00% | 7.08% | 3.15e-11 |
| rolling | VAE-HARD | 7.77% | 7.08% | 3.21e-11 |
| rolling | GMM | 8.44% | 10.17% | 6.30e-11 |

## Interpretation

Solver-derived voltages are consistent by construction. Small independent nodal-balance residuals check numerical solution accuracy; nonzero voltage-limit violations show that solvability does not imply safe operation. Emissions use slack-bus imports including line losses, with an illustrative factor. Daily-input losses are a mean-power approximation.

Rolling comparisons are descriptive because windows overlap. No equivalence or absence-of-cost conclusion follows from nonsignificance. Train/test distances are empirical references, not irreducible error floors. No single generator is established as universally best.

The real downstream task predicts one building from others on the same day, rather than future demand. Retain the simple baselines and report limitations alongside neural results.
