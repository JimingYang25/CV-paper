"""Deterministic numerical checks for Schur-complement bundle adjustment."""

from __future__ import annotations

import numpy as np

from bundle_adjustment import (
    BAProblem,
    apply_increment,
    camera_centers,
    full_normal_step,
    huber_cost_and_weight,
    linearize,
    observation_jacobians,
    project_observations,
    schur_step,
    skew,
    so3_exp,
    solve_bundle_adjustment,
)
from demo import K, reconstruction_metrics, synthetic_problem


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_skew_cross_product() -> None:
    a = np.array([0.3, -0.5, 0.7])
    b = np.array([-0.2, 0.9, 0.4])
    require(np.allclose(skew(a) @ b, np.cross(a, b)), "skew must encode cross product")


def check_so3_exponential() -> None:
    R = so3_exp(np.array([0.2, -0.1, 0.05]))
    require(np.max(np.abs(R.T @ R - np.eye(3))) < 1e-12, "Exp must be orthonormal")
    require(abs(np.linalg.det(R) - 1.0) < 1e-12, "Exp must have determinant +1")
    small = np.array([1e-10, -2e-10, 3e-10])
    require(np.allclose(so3_exp(small), np.eye(3) + skew(small), atol=1e-15),
            "small-angle branch")


def check_pose_jacobian_finite_difference() -> None:
    R = so3_exp(np.array([0.08, -0.13, 0.04]))
    t = np.array([0.03, -0.02, 0.15])
    point = np.array([0.4, -0.25, 4.8])
    pixel, J_pose, _ = observation_jacobians(K, R, t, point)
    numerical = np.empty((2, 6))
    epsilon = 1e-7
    for axis in range(6):
        step = np.zeros(6)
        step[axis] = epsilon
        dR = so3_exp(step[:3])
        perturbed_R = dR @ R
        perturbed_t = dR @ t + step[3:]
        changed, _, _ = observation_jacobians(K, perturbed_R, perturbed_t, point)
        numerical[:, axis] = (changed - pixel) / epsilon
    require(np.max(np.abs(numerical - J_pose)) < 2e-4, "analytic pose Jacobian")


def check_point_jacobian_finite_difference() -> None:
    R = so3_exp(np.array([0.08, -0.13, 0.04]))
    t = np.array([0.03, -0.02, 0.15])
    point = np.array([0.4, -0.25, 4.8])
    pixel, _, J_point = observation_jacobians(K, R, t, point)
    numerical = np.empty((2, 3))
    epsilon = 1e-7
    for axis in range(3):
        changed_point = point.copy()
        changed_point[axis] += epsilon
        changed, _, _ = observation_jacobians(K, R, t, changed_point)
        numerical[:, axis] = (changed - pixel) / epsilon
    require(np.max(np.abs(numerical - J_point)) < 2e-4, "analytic point Jacobian")


def check_huber_definition() -> None:
    cost, weight = huber_cost_and_weight(np.array([1.0, 0.0]), 2.0)
    require(abs(cost - 0.5) < 1e-12 and weight == 1.0, "Huber quadratic branch")
    cost, weight = huber_cost_and_weight(np.array([4.0, 0.0]), 2.0)
    require(abs(cost - 6.0) < 1e-12 and abs(weight - 0.5) < 1e-12,
            "Huber linear branch and IRLS weight")


def small_system(seed: int = 4):
    return synthetic_problem(
        seed, cameras=4, points=28, noise_px=0.3,
        outlier_fraction=0.0, dropout_fraction=0.08,
    )


def check_block_dimensions_and_sparsity() -> None:
    data = small_system()
    problem = data["problem"]
    system = linearize(problem, problem.rotations, problem.translations, problem.points)
    require(system.camera_blocks.shape == (3, 6, 6), "one camera must be fixed")
    require(system.point_blocks.shape == (27, 3, 3), "one point must fix scale")
    require(len(system.cross_blocks) < 3 * 27, "visibility/dropout should preserve sparsity")
    require(len(system.cross_blocks) > 0, "cross blocks must exist")


def check_schur_matches_full_normal_step() -> None:
    data = small_system()
    problem = data["problem"]
    system = linearize(problem, problem.rotations, problem.translations, problem.points)
    schur_camera, schur_point, reduced = schur_step(system, damping=1e-3)
    full_camera, full_point, full = full_normal_step(system, damping=1e-3)
    require(np.max(np.abs(schur_camera - full_camera)) < 2e-9, "camera Schur step")
    require(np.max(np.abs(schur_point - full_point)) < 2e-9, "point back-substitution")
    require(reduced.shape[0] == 18 and full.shape[0] == 18 + 81,
            "reduced dimension must eliminate point variables")


def check_one_step_decreases_clean_cost() -> None:
    data = small_system()
    problem = data["problem"]
    system = linearize(problem, problem.rotations, problem.translations, problem.points)
    dc, dp, _ = schur_step(system, damping=1e-3)
    candidate = apply_increment(
        problem.rotations, problem.translations, problem.points, system, dc, dp
    )
    next_system = linearize(problem, *candidate)
    require(next_system.robust_cost < system.robust_cost, "accepted GN/LM candidate should decrease cost")


def check_solver_decreases_clean_cost() -> None:
    data = small_system(5)
    result = solve_bundle_adjustment(data["problem"], max_iterations=25)
    require(result.final_cost < 0.05 * result.initial_cost, "clean optimization should reduce cost")
    require(result.final_rmse_px < 0.7, "clean final reprojection RMSE")


def check_anchors_remain_fixed() -> None:
    data = small_system(6)
    problem = data["problem"]
    result = solve_bundle_adjustment(problem, max_iterations=20)
    require(np.array_equal(result.rotations[0], problem.rotations[0]), "camera rotation anchor")
    require(np.array_equal(result.translations[0], problem.translations[0]), "camera translation anchor")
    require(np.array_equal(result.points[0], problem.points[0]), "scale landmark anchor")


def check_gauge_guard() -> None:
    data = small_system()
    try:
        solve_bundle_adjustment(data["problem"], fixed_point=None, max_iterations=2)
    except ValueError as error:
        require("scale" in str(error), "gauge error should name the missing scale anchor")
    else:
        raise AssertionError("joint BA without a scale anchor must be rejected")


def check_similarity_scale_invariance() -> None:
    data = small_system()
    problem = data["problem"]
    R = data["rotations_true"]
    t = data["translations_true"]
    points = data["points_true"]
    centers = camera_centers(R, t)
    origin = centers[0]
    scale = 3.1
    scaled_centers = origin + scale * (centers - origin)
    scaled_points = origin + scale * (points - origin)
    scaled_t = -np.einsum("cij,cj->ci", R, scaled_centers)
    original, _ = project_observations(
        K, R, t, points, problem.camera_indices, problem.point_indices
    )
    scaled, _ = project_observations(
        K, R, scaled_t, scaled_points, problem.camera_indices, problem.point_indices
    )
    require(np.max(np.abs(original - scaled)) < 1e-10, "global monocular scale gauge")


def check_crop_resize_intrinsics() -> None:
    data = small_system()
    problem = data["problem"]
    pixels, _ = project_observations(
        K, data["rotations_true"], data["translations_true"], data["points_true"],
        problem.camera_indices, problem.point_indices,
    )
    A = np.array([[0.55, 0.0, -41.0], [0.0, 0.72, 18.0], [0.0, 0.0, 1.0]])
    transformed = (A @ np.column_stack((pixels, np.ones(len(pixels)))).T).T
    transformed = transformed[:, :2] / transformed[:, 2:3]
    direct, _ = project_observations(
        A @ K, data["rotations_true"], data["translations_true"], data["points_true"],
        problem.camera_indices, problem.point_indices,
    )
    require(np.max(np.abs(transformed - direct)) < 1e-10, "preprocessing must transform K")


def check_pose_only_keeps_points() -> None:
    data = small_system(7)
    problem = data["problem"]
    result = solve_bundle_adjustment(
        problem, optimize_cameras=True, optimize_points=False, max_iterations=12
    )
    require(np.array_equal(result.points, problem.points), "pose-only must not move points")


def check_structure_only_keeps_cameras() -> None:
    data = small_system(8)
    problem = data["problem"]
    result = solve_bundle_adjustment(
        problem, optimize_cameras=False, optimize_points=True, max_iterations=12
    )
    require(np.array_equal(result.rotations, problem.rotations), "structure-only rotations")
    require(np.array_equal(result.translations, problem.translations), "structure-only translations")


def check_deterministic_solver() -> None:
    data = small_system(9)
    a = solve_bundle_adjustment(data["problem"], max_iterations=12)
    b = solve_bundle_adjustment(data["problem"], max_iterations=12)
    require(np.array_equal(a.points, b.points), "solver should be deterministic")
    require(a.cost_history == b.cost_history, "iteration trace should be deterministic")


def check_robust_loss_beats_linear_on_structured_state() -> None:
    data = synthetic_problem(
        20261023, cameras=6, points=80, outlier_fraction=0.08
    )
    linear = solve_bundle_adjustment(data["problem"], loss="linear", max_iterations=30)
    robust = solve_bundle_adjustment(data["problem"], loss="huber", max_iterations=30)
    linear_metrics = reconstruction_metrics(data, linear)
    robust_metrics = reconstruction_metrics(data, robust)
    require(robust_metrics["true_inlier_rmse_final_px"] < linear_metrics["true_inlier_rmse_final_px"],
            "Huber should improve the true-inlier fit on this fixed synthetic case")
    require(robust_metrics["camera_center_error_mm_median"] < 0.3 * linear_metrics["camera_center_error_mm_median"],
            "Huber should reduce camera-center error on this fixed synthetic case")


def check_camera_center_identity() -> None:
    data = small_system()
    centers = camera_centers(data["rotations_true"], data["translations_true"])
    origins = np.einsum("cij,cj->ci", data["rotations_true"], centers)
    origins += data["translations_true"]
    require(np.max(np.abs(origins)) < 1e-12, "camera centers must map to camera origins")


CHECKS = [
    check_skew_cross_product,
    check_so3_exponential,
    check_pose_jacobian_finite_difference,
    check_point_jacobian_finite_difference,
    check_huber_definition,
    check_block_dimensions_and_sparsity,
    check_schur_matches_full_normal_step,
    check_one_step_decreases_clean_cost,
    check_solver_decreases_clean_cost,
    check_anchors_remain_fixed,
    check_gauge_guard,
    check_similarity_scale_invariance,
    check_crop_resize_intrinsics,
    check_pose_only_keeps_points,
    check_structure_only_keeps_cameras,
    check_deterministic_solver,
    check_robust_loss_beats_linear_on_structured_state,
    check_camera_center_identity,
]


if __name__ == "__main__":
    for check in CHECKS:
        check()
        print(f"PASS {check.__name__}")
    print(f"PASS all {len(CHECKS)} checks")
