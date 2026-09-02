import unittest

from ghg_model.optimization import (
    AbatementOption,
    solve_least_cost_pathway,
    rank_reversal_analysis,
)


def _sample_options():
    return [
        AbatementOption("A", "electricity", annual_cost_gbp=100.0, abatement_kgco2e_per_year=500.0),
        AbatementOption("B", "heating", annual_cost_gbp=50.0, abatement_kgco2e_per_year=300.0),
        AbatementOption("C", "waste", annual_cost_gbp=200.0, abatement_kgco2e_per_year=800.0),
        AbatementOption("D", "transport", annual_cost_gbp=-30.0, abatement_kgco2e_per_year=150.0),
    ]


class TestSolveLeastCostPathway(unittest.TestCase):
    def test_zero_target_selects_nothing(self):
        result = solve_least_cost_pathway(_sample_options(), 0.0)
        self.assertTrue(result.feasible)
        self.assertEqual(result.selected, [])
        self.assertEqual(result.total_cost_gbp, 0.0)

    def test_meets_or_exceeds_target(self):
        result = solve_least_cost_pathway(_sample_options(), 400.0)
        self.assertTrue(result.feasible)
        self.assertGreaterEqual(result.total_abatement_kgco2e, 400.0 - 1e-6)

    def test_negative_cost_option_always_worth_taking_when_useful(self):
        # Option D is a net saving (-30) with real abatement; a small target
        # reachable by D alone should select D and only D.
        result = solve_least_cost_pathway(_sample_options(), 150.0)
        self.assertTrue(result.feasible)
        self.assertIn("D", result.selected)
        self.assertLessEqual(result.total_cost_gbp, 0.0)

    def test_infeasible_target_reports_infeasible(self):
        result = solve_least_cost_pathway(_sample_options(), 10_000.0)
        self.assertFalse(result.feasible)

    def test_empty_options_raises(self):
        with self.assertRaises(ValueError):
            solve_least_cost_pathway([], 100.0)

    def test_negative_target_raises(self):
        with self.assertRaises(ValueError):
            solve_least_cost_pathway(_sample_options(), -1.0)

    def test_partial_relaxation_is_never_more_expensive_than_integer_solution(self):
        target = 700.0
        integer_result = solve_least_cost_pathway(_sample_options(), target)
        relaxed_result = solve_least_cost_pathway(
            _sample_options(), target, allow_partial=True
        )
        self.assertTrue(integer_result.feasible)
        self.assertTrue(relaxed_result.feasible)
        self.assertLessEqual(
            relaxed_result.total_cost_gbp, integer_result.total_cost_gbp + 1e-6
        )

    def test_least_cost_is_at_least_as_good_as_a_naive_alternative(self):
        # Sanity check against a naive "take everything" baseline: the
        # optimiser must never cost more than adopting all options, when all
        # options together are feasible.
        options = _sample_options()
        target = 900.0
        result = solve_least_cost_pathway(options, target)
        naive_cost = sum(o.annual_cost_gbp for o in options)
        self.assertTrue(result.feasible)
        self.assertLessEqual(result.total_cost_gbp, naive_cost + 1e-6)


class TestRankReversalAnalysis(unittest.TestCase):
    def test_output_shape(self):
        summary = rank_reversal_analysis(
            _sample_options(), 400.0, samples=50, seed=1
        )
        self.assertEqual(summary["samples"], 50)
        self.assertIn("selection_frequency", summary)
        self.assertEqual(
            set(summary["selection_frequency"].keys()), {"A", "B", "C", "D"}
        )
        for freq in summary["selection_frequency"].values():
            self.assertGreaterEqual(freq, 0.0)
            self.assertLessEqual(freq, 1.0)

    def test_reproducible_with_fixed_seed(self):
        s1 = rank_reversal_analysis(_sample_options(), 400.0, samples=30, seed=7)
        s2 = rank_reversal_analysis(_sample_options(), 400.0, samples=30, seed=7)
        self.assertEqual(s1, s2)


if __name__ == "__main__":
    unittest.main()
