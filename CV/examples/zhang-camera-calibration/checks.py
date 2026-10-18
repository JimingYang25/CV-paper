"""Numerical properties, not paper benchmark reproduction. python checks.py"""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from calibration import (board_points, project, homography, initialize,
                         intrinsics_from_homographies, transform_intrinsics, rms)
from demo import make_views, parallel_homographies, run, TRUE_K
from application import run as run_application


class CalibrationChecks(unittest.TestCase):
    def test_projection_hand_example(self):
        K = np.array([[800., 0, 320], [0, 820, 240], [0, 0, 1]])
        uv = project(np.array([[0.1, 0.05, 0.]]), K, np.eye(3), np.array([0., 0., 1.]))
        assert_allclose(uv, [[400, 281]], atol=1e-12)

    def test_four_point_homography(self):
        xy = np.array([[0., 0], [2, 0], [2, 1], [0, 1]])
        H = np.array([[150., 20, 300], [5, 180, 200], [0.1, -0.05, 1]])
        p = np.column_stack((xy, np.ones(4))) @ H.T
        recovered = homography(xy, p[:, :2]/p[:, 2:])
        assert_allclose(recovered, H, atol=1e-9)

    def test_collinear_points_rejected(self):
        xy = np.column_stack((np.arange(6.), np.zeros(6)))
        with self.assertRaises(ValueError):
            homography(xy, xy+4)

    def test_general_intrinsics_and_pose_exact(self):
        K = TRUE_K.copy()
        K[0, 1] = 7.0  # all five intrinsic parameters in the linear stage
        xyz, true_poses, uv = make_views(K=K, distortion=(0, 0), noise=0)
        recovered, poses, _ = initialize(xyz, uv)
        assert_allclose(recovered, K, atol=1e-6)
        for (R, t), (Rt, tt), observation in zip(poses, true_poses, uv):
            assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
            assert_allclose(np.linalg.det(R), 1, atol=1e-12)
            assert_allclose(R, Rt, atol=1e-8)
            assert_allclose(t, tt, atol=1e-8)
            self.assertLess(rms(observation, project(xyz, recovered, R, t)), 1e-7)

    def test_parallel_plane_poses_rejected(self):
        with self.assertRaises(ValueError):
            intrinsics_from_homographies(parallel_homographies())

    def test_board_unit_scale(self):
        xyz, _, uv = make_views(distortion=(0, 0), noise=0)
        K1, poses1, _ = initialize(xyz, uv)
        K2, poses2, _ = initialize(xyz*1000, uv)  # metres -> millimetres
        assert_allclose(K1, K2, atol=1e-6)
        for (_, t1), (_, t2) in zip(poses1, poses2):
            assert_allclose(t2, 1000*t1, atol=1e-6)

    def test_resize_crop_coordinate_contract(self):
        xyz, poses, _ = make_views(noise=0)
        A = np.array([[0.5, 0, -20], [0, 0.5, -10], [0, 0, 1.]])
        K2 = transform_intrinsics(TRUE_K, A)
        R, t = poses[0]
        uv = project(xyz, TRUE_K, R, t, (-0.12, 0.03))
        transformed = (np.column_stack((uv, np.ones(len(uv)))) @ A.T)[:, :2]
        assert_allclose(project(xyz, K2, R, t, (-0.12, 0.03)), transformed, atol=1e-12)

    def test_radial_distortion_hand_example(self):
        xyz = np.array([[0., 0, 0], [0.5, 0, 0]])
        uv = project(xyz, np.eye(3), np.eye(3), np.array([0., 0., 1.]), (-0.2, 0.04))
        assert_allclose(uv, [[0, 0], [0.47625, 0]], atol=1e-12)

    def test_joint_noisy_refinement(self):
        report, _ = run()
        self.assertLess(report["refined_vector_rms_px"], report["initial_zero_skew_vector_rms_px"])
        # Thresholds are synthetic acceptance bounds, not hardware/model benchmarks.
        self.assertLess(report["refined_vector_rms_px"], 0.30)
        self.assertLess(max(report["relative_focal_error_percent"]), 1.0)
        self.assertLess(report["known_pose_synthetic_heldout_vector_rms_px"], 2.0)
        self.assertEqual(report["clean_diverse_constraint_rank"], 5)
        self.assertEqual(report["clean_parallel_constraint_rank"], 2)

    def test_integrated_ray_plane_application(self):
        report = run_application()
        self.assertTrue(report["true_parameters_roundtrip_passed"])
        self.assertLess(report["intersection_error_mm"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
