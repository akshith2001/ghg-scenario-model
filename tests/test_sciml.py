import unittest

from ghg_model.sciml import run_experiment


class ScientificMachineLearningTests(unittest.TestCase):
    def test_experiment_is_reproducible(self) -> None:
        first, _ = run_experiment(seed=17, samples=120)
        second, _ = run_experiment(seed=17, samples=120)
        self.assertEqual(first, second)

    def test_physics_constraint_reduces_violation(self) -> None:
        result, _ = run_experiment(seed=2026, samples=180)
        self.assertLess(
            result.physics_informed_extrapolation.mean_physics_violation_kg_co2e,
            result.data_only_extrapolation.mean_physics_violation_kg_co2e,
        )

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_experiment(samples=99)
        with self.assertRaises(ValueError):
            run_experiment(noise_fraction=-0.1)


if __name__ == "__main__":
    unittest.main()
