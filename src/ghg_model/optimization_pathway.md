# Least-cost abatement pathway optimisation

## Research question

Given a portfolio of candidate emissions-reduction measures, each with an
illustrative cost and abatement potential, which combination meets a stated
emissions-reduction target at the lowest cost — and how sensitive is that
answer to plausible uncertainty in the input assumptions?

This extends the scenario model's existing "what changes if we intervene"
question with a "which combination of interventions is worth doing, and how
confident can we be in that combination" question.

## Method

Each candidate measure is treated as an indivisible (0/1) decision, since
most real interventions — replacing a boiler, retrofitting lighting — are
not naturally divisible. The least-cost combination meeting or exceeding a
target abatement is found by solving a 0/1 knapsack-style mixed-integer
linear program:

```
minimise   sum(cost_i * x_i)
subject to sum(abatement_i * x_i) >= target
           x_i in {0, 1}
```

A continuous relaxation (allowing fractional adoption) is also reported for
comparison. It is a lower bound on cost, not an achievable plan, and is
labelled as such in all output.

## Robustness: rank-reversal analysis

Costs and abatement potentials are estimated, not measured. The
`rank_reversal_analysis` function perturbs every measure's cost and
abatement within a stated relative uncertainty (independently, uniformly),
re-solves the optimisation, and records how often each measure appears in
the optimal set across repeated trials.

This directly answers a question the base-case optimum cannot: is a
selected measure robustly worth doing, or does it only look optimal under
one specific set of point estimates? A measure selected in the base case
but included in a small fraction of perturbed trials is a fragile
recommendation, not a robust one.

## Illustrative dataset

`data/abatement_options.json` contains eight illustrative measures spanning
electricity, heating, waste, transport, and a hospitality-specific water-use
measure directly connected to the tap-running case study documented
elsewhere in this repository. All costs, abatement figures and uncertainty
ranges are placeholders for demonstrating the method; they are not derived
from a real audit and must not be used to justify a real spending decision.

## Example result

Running the optimiser against the illustrative dataset with a 2,500 kg
CO2e/year target selects five measures at a total annualised cost of
approximately £20/year (two of the five are net savings). Under 1,000
Monte Carlo perturbations of costs and abatement within their stated
uncertainty ranges, the two net-saving measures (reduced tap-running time
and delivery-route consolidation) are selected in 100% of trials, while a
larger single-measure option (loft and pipe insulation) is selected in
under 2% of trials — despite having the single largest illustrative
abatement figure in the dataset. This is reported as a demonstration of the
method's ability to distinguish robust choices from point-estimate
artefacts, not as a real efficiency finding for any venue.

## Limitations

- All costs, abatement potentials and uncertainty ranges in the shipped
  dataset are illustrative placeholders.
- The 0/1 formulation ignores interaction effects between measures (e.g.
  a boiler upgrade changing the marginal value of insulation); a full
  treatment would need cross-terms or a more detailed engineering model.
- Uncertainty is sampled independently and uniformly per measure; real
  costs are likely correlated (e.g. driven by a shared energy-price
  assumption) and may follow a different distribution shape.
- The optimiser recommends a static one-off pathway; it does not sequence
  measures over time or account for capital constraints or financing.
- This is a demonstration of method, not a real-world capital-planning
  recommendation for any organisation.

## Next steps

- Replace illustrative costs and abatement figures with vendor quotes or
  audited measurements for a specific venue.
- Add correlated uncertainty sampling (e.g. via a shared energy-price
  factor) to the rank-reversal analysis.
- Extend to a multi-period formulation with a capital budget constraint per
  year.
- Compare the MILP solution against a reinforcement-learning formulation of
  the same sequential adoption-under-budget problem, to test whether a
  learned policy adds value over the exact optimum on this simple problem
  before applying either method to a larger, more realistic action space.
