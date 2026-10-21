"""Deterministic numerical checks for the educational EPnP reconstruction."""

from __future__ import annotations

import numpy as np

from demo import K, coherent_wrong_pose_failure, synthetic_correspondences
from epnp_pose import (
    _beta_products,
    absolute_orientation,
    barycentric_coordinates,
    build_projection_matrix,
    camera_center_world,
    choose_control_points,
    distance_constraint_system,
    pixels_to_normalized,
    project_points,
    ransac_epnp,
    refine_pose,
    rotation_error_degrees,
    rotation_xyz,
    solve_epnp,
    translation_error,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def exact_scene(seed: int = 8, count: int = 50):
    rng = np.random.default_rng(seed)
    world = rng.uniform([-0.20, -0.14, -0.06], [0.20, 0.14, 0.06], (count, 3))
    R = rotation_xyz((9.0, -14.0, 5.0))
    t = np.array([0.03, -0.018, 0.82])
    pixels, valid = project_points(world, R, t, K)
    require(bool(np.all(valid)), "exact scene must be in front of the camera")
    return world, pixels, R, t


def check_projection_and_normalization() -> None:
    world, pixels, R, t = exact_scene()
    normalized = pixels_to_normalized(pixels, K)
    camera = (R @ world.T).T + t
    require(np.max(np.abs(normalized - camera[:, :2] / camera[:, 2:3])) < 1e-12,
            "pixel normalization must recover camera rays")


def check_barycentric_reconstruction() -> None:
    world, _, _, _ = exact_scene()
    control, values, ratio = choose_control_points(world)
    alpha = barycentric_coordinates(world, control)
    require(np.max(np.abs(alpha @ control - world)) < 1e-12,
            "barycentric coordinates must reconstruct the points")
    require(np.max(np.abs(np.sum(alpha, axis=1) - 1.0)) < 1e-12,
            "barycentric coordinates must sum to one")
    require(np.all(values[:-1] >= values[1:]) and ratio > 0.0,
            "PCA diagnostics must be ordered and positive")


def check_projection_nullspace() -> None:
    world, pixels, R, t = exact_scene()
    control, _, _ = choose_control_points(world)
    alpha = barycentric_coordinates(world, control)
    M = build_projection_matrix(alpha, pixels_to_normalized(pixels, K))
    control_camera = (R @ control.T).T + t
    relative = np.linalg.norm(M @ control_camera.reshape(-1)) / np.linalg.norm(control_camera)
    require(relative < 1e-11, "true camera control points must lie in M's nullspace")


def check_distance_constraint_algebra() -> None:
    world, _, R, t = exact_scene()
    control, _, _ = choose_control_points(world)
    camera_control = (R @ control.T).T + t
    beta = np.array([1.7, -0.3, 0.8, 0.2])
    basis = np.zeros((4, 4, 3))
    basis[0] = camera_control / beta[0]
    L, rho = distance_constraint_system(basis, control)
    require(np.max(np.abs(L @ _beta_products(beta) - rho)) < 1e-12,
            "beta-product system must reproduce the six control distances")


def check_absolute_orientation() -> None:
    world, _, R, t = exact_scene()
    camera = (R @ world.T).T + t
    Re, te = absolute_orientation(world, camera)
    require(rotation_error_degrees(Re, R) < 1e-5, "absolute orientation rotation")
    require(translation_error(te, t) < 1e-12, "absolute orientation translation")
    require(abs(np.linalg.det(Re) - 1.0) < 1e-12, "rotation must be proper")


def check_exact_nonminimal_epnp() -> None:
    world, pixels, R, t = exact_scene()
    pose = solve_epnp(world, pixels, K)
    require(rotation_error_degrees(pose.R, R) < 1e-5, "exact EPnP rotation")
    require(translation_error(pose.t, t) < 1e-10, "exact EPnP translation")
    require(pose.reprojection_rmse_px < 1e-9, "exact EPnP reprojection")


def check_refinement_nonworsening() -> None:
    data = synthetic_correspondences(77, inliers=80, outliers=0, noise_px=0.8)
    closed = solve_epnp(data["points_world"], data["points_px"], K)
    refined = refine_pose(data["points_world"], data["points_px"], K, closed)
    require(refined.reprojection_rmse_px <= closed.reprojection_rmse_px + 1e-9,
            "least-squares refinement must not worsen its optimized residual")
    require(refined.optimizer_nfev is not None and refined.jacobian_condition is not None,
            "refinement must return optimizer diagnostics")


def check_crop_resize_conjugation() -> None:
    _, pixels, _, _ = exact_scene()
    A = np.array([[0.5, 0.0, -37.0], [0.0, 0.75, 21.0], [0.0, 0.0, 1.0]])
    processed_h = (A @ np.column_stack((pixels, np.ones(len(pixels)))).T).T
    processed = processed_h[:, :2] / processed_h[:, 2:3]
    before = pixels_to_normalized(pixels, K)
    after = pixels_to_normalized(processed, A @ K)
    require(np.max(np.abs(before - after)) < 1e-12,
            "pixel transform and intrinsics must be conjugated")


def check_world_scale_equivariance() -> None:
    world, pixels, R, t = exact_scene()
    scale = 3.7
    pose = solve_epnp(scale * world, pixels, K)
    require(rotation_error_degrees(pose.R, R) < 1e-5, "world scaling preserves rotation")
    require(translation_error(pose.t, scale * t) < 1e-9, "translation must scale with world units")


def check_camera_center_convention() -> None:
    _, _, R, t = exact_scene()
    center = camera_center_world(R, t)
    require(np.linalg.norm(R @ center + t) < 1e-12,
            "camera center must map to the camera origin")


def check_exact_planar_rejected() -> None:
    world, _, R, t = exact_scene()
    world[:, 2] = 0.0
    pixels, _ = project_points(world, R, t, K)
    try:
        solve_epnp(world, pixels, K)
    except ValueError as error:
        require("planar" in str(error), "planar rejection should explain the cause")
    else:
        raise AssertionError("exact planar input must be rejected by this reconstruction")


def check_minimal_inputs_rejected() -> None:
    world, pixels, _, _ = exact_scene(count=6)
    for count in (4, 5):
        try:
            solve_epnp(world[:count], pixels[:count], K)
        except ValueError as error:
            require("six" in str(error), "minimal-case guard should state the scope")
        else:
            raise AssertionError("this non-minimal implementation must reject four/five points")


def check_ransac_determinism() -> None:
    data = synthetic_correspondences(20261022)
    kwargs = dict(threshold_px=3.0, min_draws=96, max_draws=3500, seed=20261022)
    a = ransac_epnp(data["points_world"], data["points_px"], K, **kwargs)
    b = ransac_epnp(data["points_world"], data["points_px"], K, **kwargs)
    require(np.array_equal(a.inliers, b.inliers) and np.allclose(a.pose.t, b.pose.t),
            "fixed seed must reproduce the robust result")


def check_random_outlier_recovery() -> None:
    data = synthetic_correspondences(20261022)
    fit = ransac_epnp(
        data["points_world"], data["points_px"], K,
        threshold_px=3.0, min_draws=96, max_draws=3500, seed=20261022,
    )
    truth = data["true_inliers"]
    tp = np.count_nonzero(fit.inliers & truth)
    precision = tp / np.count_nonzero(fit.inliers)
    recall = tp / np.count_nonzero(truth)
    require(precision > 0.98 and recall > 0.98, "random-outlier recovery")
    require(rotation_error_degrees(fit.pose.R, data["R"]) < 1.0, "robust rotation")
    require(translation_error(fit.pose.t, data["t"]) < 0.005, "robust translation")


def check_coherent_distractor_failure() -> None:
    report = coherent_wrong_pose_failure()
    require(report["selected_distractor_matches"] > report["selected_intended_matches"],
            "geometric consensus should expose the dominant-distractor failure")
    require(report["error_to_distractor"]["rotation_deg"] < 1.0,
            "the selected pose should be the coherent distractor")


def check_invalid_ransac_controls() -> None:
    world, pixels, _, _ = exact_scene()
    try:
        ransac_epnp(world, pixels, K, sample_size=5)
    except ValueError:
        pass
    else:
        raise AssertionError("RANSAC must reject an unsupported sample size")


CHECKS = [
    check_projection_and_normalization,
    check_barycentric_reconstruction,
    check_projection_nullspace,
    check_distance_constraint_algebra,
    check_absolute_orientation,
    check_exact_nonminimal_epnp,
    check_refinement_nonworsening,
    check_crop_resize_conjugation,
    check_world_scale_equivariance,
    check_camera_center_convention,
    check_exact_planar_rejected,
    check_minimal_inputs_rejected,
    check_ransac_determinism,
    check_random_outlier_recovery,
    check_coherent_distractor_failure,
    check_invalid_ransac_controls,
]


if __name__ == "__main__":
    for check in CHECKS:
        check()
        print(f"PASS {check.__name__}")
    print(f"PASS all {len(CHECKS)} checks")
