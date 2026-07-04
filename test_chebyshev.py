import unittest
from unittest.mock import patch

import numpy as np

from algorithm import _bundle_update_adaptive, algorithm_adaptive
from baseline import uniform_discretisation
from bundle import Bundle, GN, T_map
from chebyshev import (
    solve_weighted_chebyshev_dual,
    weighted_chebyshev_inner,
)
from objectives import make_mlp_nonconvex


class WeightedChebyshevDualTests(unittest.TestCase):
    def test_dual_solution_is_feasible(self):
        normalized_grads = np.eye(2)
        lam = np.array([0.25, 0.75])

        beta, residual = solve_weighted_chebyshev_dual(
            normalized_grads, lam
        )

        np.testing.assert_allclose(lam @ beta, 1.0, atol=1e-10)
        np.testing.assert_allclose(beta, np.array([0.4, 1.2]), atol=1e-8)
        self.assertAlmostEqual(residual, np.sqrt(1.6), places=8)

    def test_opposing_gradients_have_zero_residual(self):
        normalized_grads = np.array([[1.0], [-1.0]])
        lam = np.array([0.5, 0.5])

        beta, residual = solve_weighted_chebyshev_dual(
            normalized_grads, lam
        )

        np.testing.assert_allclose(lam @ beta, 1.0, atol=1e-10)
        self.assertLessEqual(residual, 1e-10)


class WeightedChebyshevInnerTests(unittest.TestCase):
    @staticmethod
    def _constant_oracle(_):
        return np.zeros(2), np.array([[1.0], [-1.0]])

    def test_exact_policy_stops_at_chebyshev_stationarity(self):
        result = weighted_chebyshev_inner(
            lam=np.array([0.75, 0.25]),
            x0=np.zeros(1),
            initial_fvals=np.zeros(2),
            initial_grads=np.array([[1.0], [-1.0]]),
            L=np.ones(2),
            objectives=[],
            grad_objectives=[],
            joint_oracle=self._constant_oracle,
            max_steps=2,
            direction_policy="chebyshev_only",
        )

        self.assertEqual(result["status"], "chebyshev_stationary")
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["direction_history"], [])

    def test_hybrid_policy_labels_scalarized_fallback(self):
        result = weighted_chebyshev_inner(
            lam=np.array([0.75, 0.25]),
            x0=np.zeros(1),
            initial_fvals=np.zeros(2),
            initial_grads=np.array([[1.0], [-1.0]]),
            L=np.ones(2),
            objectives=[],
            grad_objectives=[],
            joint_oracle=self._constant_oracle,
            max_steps=1,
            direction_policy="hybrid",
        )

        self.assertEqual(result["steps"], 1)
        self.assertEqual(
            result["direction_history"], ["scalarized_fallback"]
        )
        np.testing.assert_allclose(result["x"], np.array([-0.5]))


class SmoothMlpObjectiveTests(unittest.TestCase):
    def test_softplus_gradient_matches_directional_difference(self):
        objectives, gradients, _, _ = make_mlp_nonconvex(
            K=3,
            p=3,
            n=18,
            h=4,
            seed=5,
            activation="softplus",
            smoothness_probes=5,
        )
        rng = np.random.default_rng(7)
        d = 4 * 3 + 4 + 3 * 4 + 3
        theta = rng.normal(scale=0.2, size=d)
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction)
        step = 1e-6

        finite_difference = (
            objectives[0](theta + step * direction)
            - objectives[0](theta - step * direction)
        ) / (2.0 * step)
        analytic = float(gradients[0](theta) @ direction)

        self.assertAlmostEqual(finite_difference, analytic, places=6)

    def test_every_moo_class_has_a_nonzero_smoothness_estimate(self):
        _, _, smoothness, _ = make_mlp_nonconvex(
            K=4,
            p=5,
            n=20,
            h=4,
            seed=14,
            activation="softplus",
            smoothness_probes=5,
        )

        self.assertTrue(np.all(np.isfinite(smoothness)))
        self.assertTrue(np.all(smoothness > 0.0))


class CheckpointIsolationTests(unittest.TestCase):
    def test_checkpoint_frequency_does_not_change_outer_preferences(self):
        objectives = [
            lambda x: 0.5 * float((x[0] - 1.0) ** 2),
            lambda x: 0.5 * float((x[0] + 1.0) ** 2),
        ]
        gradients = [
            lambda x: np.array([x[0] - 1.0]),
            lambda x: np.array([x[0] + 1.0]),
        ]

        def metric(*args, **kwargs):
            return 1.0, np.array([0.99, 0.01])

        def selector(bundle, prev_lam=None, **kwargs):
            if prev_lam is None:
                lam = np.array([0.6, 0.4])
            else:
                lam = np.asarray(prev_lam)[::-1].copy()
            return 1.0, lam

        def run(checkpoint_interval):
            return algorithm_adaptive(
                K=2,
                d=1,
                objectives=objectives,
                grad_objectives=gradients,
                L=np.ones(2),
                x0=np.array([0.2]),
                max_outer=3,
                max_inner=1,
                eval_every_n_grads=checkpoint_interval,
                lambda_selector="sample",
            )

        with (
            patch("algorithm._pc_star_metric", side_effect=metric),
            patch("algorithm._maximise_GN_sampled", side_effect=selector),
        ):
            frequent = run(None)
            sparse = run(10_000)

        np.testing.assert_allclose(
            frequent["lambda_history"],
            sparse["lambda_history"],
        )
        np.testing.assert_allclose(
            frequent["bundle"].points,
            sparse["bundle"].points,
        )


class BundleUpdateCacheTests(unittest.TestCase):
    def test_cached_t_map_chain_matches_reference_implementation(self):
        objectives = [
            lambda x: 0.5 * float((x[0] - 1.0) ** 2 + 2.0 * x[1] ** 2),
            lambda x: 0.5 * float(3.0 * x[0] ** 2 + (x[1] + 1.0) ** 2),
        ]
        gradients = [
            lambda x: np.array([x[0] - 1.0, 2.0 * x[1]]),
            lambda x: np.array([3.0 * x[0], x[1] + 1.0]),
        ]
        L = np.array([2.0, 3.0])
        lam = np.array([0.35, 0.65])
        initial_points = [
            np.array([0.4, -0.3]),
            np.array([-0.2, 0.5]),
        ]
        steps = 8

        def make_bundle():
            result = Bundle(K=2, d=2, L=L)
            for point in initial_points:
                result.add_point(point, objectives, gradients)
            return result

        reference = make_bundle()
        base_m = reference.m
        for _ in range(steps):
            point = T_map(reference, lam)
            reference.add_point(point, objectives, gradients)

        candidate_grads = np.asarray(reference.grads[base_m:])
        scalarized = np.einsum("skd,k->sd", candidate_grads, lam)
        winner = int(np.argmin(np.einsum("sd,sd->s", scalarized, scalarized)))
        expected_point = reference.points[base_m + winner].copy()
        expected_fvals = reference.fvals[base_m + winner].copy()
        expected_grads = reference.grads[base_m + winner].copy()

        cached = make_bundle()
        used = _bundle_update_adaptive(
            bundle=cached,
            lam=lam,
            pc_fn=GN,
            objectives=objectives,
            grad_objectives=gradients,
            max_steps=steps,
            prune=True,
        )

        self.assertEqual(used, steps)
        self.assertEqual(cached.m, base_m + 1)
        np.testing.assert_allclose(cached.points[-1], expected_point)
        np.testing.assert_allclose(cached.fvals[-1], expected_fvals)
        np.testing.assert_allclose(cached.grads[-1], expected_grads)


class BaselineCheckpointTests(unittest.TestCase):
    def test_in_pass_checkpoints_do_not_change_final_solutions(self):
        objectives = [
            lambda x: 0.5 * float((x[0] - 1.0) ** 2),
            lambda x: 0.5 * float((x[0] + 1.0) ** 2),
        ]
        gradients = [
            lambda x: np.array([x[0] - 1.0]),
            lambda x: np.array([x[0] + 1.0]),
        ]
        common = dict(
            K=2,
            objectives=objectives,
            grad_objectives=gradients,
            L=np.ones(2),
            x0=np.array([0.25]),
            resolution=4,
            n_passes=1,
            steps_per_point_per_pass=3,
            coverage_mode=None,
        )

        pass_only = uniform_discretisation(
            **common,
            eval_every_n_grads=None,
        )
        in_pass = uniform_discretisation(
            **common,
            eval_every_n_grads=4,
        )

        np.testing.assert_allclose(
            pass_only["final_solutions"],
            in_pass["final_solutions"],
        )
        self.assertEqual(
            pass_only["total_iters_history"][-1],
            in_pass["total_iters_history"][-1],
        )
        self.assertGreater(
            len(in_pass["total_iters_history"]),
            len(pass_only["total_iters_history"]),
        )

    def test_fixed_coverage_target_stops_during_a_pass(self):
        objectives = [
            lambda x: 0.5 * float((x[0] - 1.0) ** 2),
            lambda x: 0.5 * float((x[0] + 1.0) ** 2),
        ]
        gradients = [
            lambda x: np.array([x[0] - 1.0]),
            lambda x: np.array([x[0] + 1.0]),
        ]
        result = uniform_discretisation(
            K=2,
            objectives=objectives,
            grad_objectives=gradients,
            L=np.ones(2),
            x0=np.array([0.25]),
            resolution=4,
            n_passes=1,
            steps_per_point_per_pass=3,
            eval_every_n_grads=4,
            coverage_mode="gn",
            target_cov=1e9,
        )

        full_iterations = 5 * 3
        self.assertLess(result["total_iters_history"][-1], full_iterations)


if __name__ == "__main__":
    unittest.main()
