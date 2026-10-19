"""Deterministic numerical checks; does not reproduce published benchmarks."""
import itertools
import math
import unittest
import numpy as np
from numpy.testing import assert_allclose
from robust_homography import (fit_homography, transfer, errors_squared, estimate,
                               quality, all_inlier_probability, required_trials)
from demo import make_data, run, failure_case, TRUE_H, SEED
from application import warp_image, run as run_application


class HomographyChecks(unittest.TestCase):
    def test_minimal_exact_DLT_and_scale(self):
        x = np.array([[0., 0], [300, 0], [300, 200], [0, 200]])
        y = transfer(TRUE_H, x)
        H = fit_homography(x, y)
        assert_allclose(H, TRUE_H, atol=1e-9, rtol=0)
        assert_allclose(transfer(H*1e-14, x), y, atol=1e-10, rtol=0)

    def test_nonminimal_exact_DLT(self):
        x, _, _, grid = make_data()
        H = fit_homography(x, transfer(TRUE_H, x))
        assert_allclose(transfer(H, grid), transfer(TRUE_H, grid), atol=1e-9, rtol=0)

    def test_any_three_collinear_rejected(self):
        x = np.array([[0., 0], [1, 0], [2, 0], [0, 1]])
        with self.assertRaises(ValueError):
            fit_homography(x, transfer(TRUE_H, x))

    def test_globally_collinear_rejected(self):
        x = np.column_stack((np.arange(12.), np.arange(12.)*2))
        with self.assertRaises(ValueError):
            estimate(x, x+1, max_trials=16)

    def test_finite_population_probability_by_enumeration(self):
        subsets = list(itertools.combinations(range(10), 4))
        genuine = sum(all(i < 6 for i in sample) for sample in subsets)
        self.assertEqual(genuine, math.comb(6, 4))
        self.assertAlmostEqual(all_inlier_probability(6, 10), genuine/len(subsets))
        self.assertLess(all_inlier_probability(6, 10), all_inlier_probability(6, 10, exact=False))
        self.assertEqual(all_inlier_probability(3, 10), 0)
        self.assertEqual(all_inlier_probability(10, 10), 1)

    def test_trial_budget_edges_and_attainment(self):
        self.assertEqual(required_trials(1), 1)
        self.assertEqual(required_trials(0), math.inf)
        q = all_inlier_probability(6, 20)
        K = required_trials(q, 0.99)
        self.assertGreaterEqual(-np.expm1(K*np.log1p(-q)), 0.99)
        self.assertLess(-np.expm1((K-1)*np.log1p(-q)), 0.99)

    def test_forward_threshold_uses_destination_pixels(self):
        x = np.array([[1., 2], [3, 4]])
        y = x + np.array([[3., 4], [0, 0]])
        assert_allclose(errors_squared(np.eye(3), x, y), [25, 0], atol=0, rtol=0)
        self.assertEqual(quality(np.array([25., 0]), 4)[0], -1)
        self.assertEqual(quality(np.array([25., 0]), 5)[0], -2)

    def test_horizon_points_are_not_inliers(self):
        H = np.array([[1., 0, 0], [0, 1, 0], [1, 0, -1]])
        errors = errors_squared(H, np.array([[1., 2]]), np.array([[0., 0]]))
        self.assertTrue(np.isinf(errors[0]))

    def test_consensus_and_MSAC_are_different_objectives(self):
        a, b = np.array([0., 0, 100]), np.array([8., 8, 8])
        self.assertLess(quality(b, 3, "consensus"), quality(a, 3, "consensus"))
        self.assertLess(quality(a, 3, "msac"), quality(b, 3, "msac"))

    def test_deterministic_and_model_consistent_mask(self):
        x, y, _, _ = make_data()
        a = estimate(x, y, seed=SEED)
        b = estimate(x, y, seed=SEED)
        assert_allclose(a.H, b.H, atol=0, rtol=0)
        np.testing.assert_array_equal(a.inliers, errors_squared(a.H, x, y) <= 9)
        self.assertLessEqual(a.draws, 3000)

    def test_same_outer_budget_local_optimization_nonworsening(self):
        x, y, _, _ = make_data()
        plain = estimate(x, y, seed=SEED, max_trials=300, adaptive=False,
                         local_mode="none", final_polish=False)
        local = estimate(x, y, seed=SEED, max_trials=300, adaptive=False,
                         local_mode="inner", final_polish=False)
        self.assertLessEqual(quality(local.errors2, 3), quality(plain.errors2, 3))
        self.assertEqual(plain.draws, local.draws)
        self.assertEqual(plain.rejected_samples, local.rejected_samples)

    def test_noisy_outlier_recovery(self):
        report, _ = run()
        self.assertLess(report["robust"]["grid_vector_rms_px"], 1.0)
        self.assertGreater(report["robust"]["precision"], 0.95)
        self.assertGreater(report["robust"]["recall"], 0.90)

    def test_resize_coordinate_conjugation(self):
        _, _, _, x = make_data()
        A = np.array([[0.5, 0, -10], [0, 0.5, -5], [0, 0, 1.]])
        B = np.array([[0.75, 0, 3], [0, 0.75, -8], [0, 0, 1.]])
        new_H = B @ TRUE_H @ np.linalg.inv(A)
        assert_allclose(transfer(new_H, transfer(A, x)), transfer(B, transfer(TRUE_H, x)),
                        atol=1e-10, rtol=0)

    def test_inverse_warp_convention(self):
        image = np.arange(30, dtype=np.float32).reshape(5, 6)
        assert_allclose(warp_image(image, np.eye(3), image.shape), image, atol=0, rtol=0)
        translation = np.array([[1., 0, 1], [0, 1, 0], [0, 0, 1]])
        moved = warp_image(image, translation, image.shape)
        assert_allclose(moved[:, 1:], image[:, :-1], atol=0, rtol=0)
        rgb = np.stack((image, image+1, image+2), axis=-1)
        assert_allclose(warp_image(rgb, np.eye(3), image.shape), rgb, atol=0, rtol=0)

    def test_integrated_registration(self):
        report, _ = run_application()
        self.assertGreater(report["source_hull_fraction"], 0.5)
        self.assertLess(report["valid_region_intensity_MAE_0_to_1"], 0.04)

    def test_dominant_wrong_plane_is_not_semantic_success(self):
        report = failure_case()
        self.assertGreater(report["selected_support"], 120)
        self.assertLess(report["intended_matches_in_selected_support"], 10)
        self.assertGreater(report["grid_error_against_intended_plane_px"], 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
