# Synthetic-grid correction notes

The voltage metric now validates input and solves the flow even when reported voltages
are absent. Zero in that case is solver consistency by construction, not an independent
power-balance test or evidence of learned physical constraints. Both voltage and slack
calculations reject nonconvergence. Feeder import uses slack power, including resistive
losses. Explicit t=1 supports daily mean-power data, with 24-hour energy integration;
hourly time-of-use factors cannot be applied to daily inputs. Real-study emissions
comparisons now use the same grid-side boundary (on the simulated scaled feeder).

Rolling summaries omit confidence intervals because windows overlap. Legacy floor/excess
CSV field names remain for plotting compatibility: they mean empirical train/test
reference and difference from that reference, not an irreducible floor. Historical
protocols and outputs are retained with warnings for auditability. Their no-cost,
equivalence and drift-decomposition claims are not justified by these experiments.

Full studies have now been rerun; see [corrected results](synthetic_grid_results.md)
and output/synthetic_grid_reviewed/manifest.json. The original outputs remain historical.
No real-feeder, operational deployment or privacy guarantee is established.

The integrated project additionally reconstructs nodal complex-power balance from bus voltages and line impedances, and reports voltage-limit violations separately. These are numerical checks of the simulated feeder, not real-network validation.
