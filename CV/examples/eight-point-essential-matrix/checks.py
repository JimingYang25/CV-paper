"""Numerical checks for the educational essential-matrix implementation."""

from __future__ import annotations

import numpy as np

from demo import K1, K2, synthetic_matches
from essential_geometry import (
    design_matrix,
    direction_error_degrees,
    essential_from_pose,
    estimate_essential_eight_point,
    fundamental_from_essential,
    hartley_similarity,
    homogeneous,
    pixels_to_normalized,
    project_scene,
    ransac_essential_eight_point,
    recover_relative_pose,
    rotation_error_degrees,
    rotation_y,
    sampson_squared,
    select_pose_by_cheirality,
    triangulate_linear,
)


def close_projective(A: np.ndarray, B: np.ndarray, tolerance: float) -> bool:
    a = A.reshape(-1) / np.linalg.norm(A)
    b = B.reshape(-1) / np.linalg.norm(B)
    return min(np.linalg.norm(a - b), np.linalg.norm(a + b)) < tolerance


def exact_scene(seed: int = 7, count: int = 40):
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        (rng.uniform(-1.5, 1.5, count), rng.uniform(-1.0, 1.0, count), rng.uniform(4.0, 9.0, count))
    )
    R = rotation_y(6.0)
    t = np.array([0.3, -0.02, 0.05])
    p1, p2 = project_scene(X, R, t, K1, K2)
    return X, R, t, p1, p2


def check_epipolar_constraint() -> None:
    _, R, t, p1, p2 = exact_scene()
    x1 = pixels_to_normalized(p1, K1)
    x2 = pixels_to_normalized(p2, K2)
    E = essential_from_pose(R, t)
    residual = np.sum(homogeneous(x2) * (E @ homogeneous(x1).T).T, axis=1)
    assert np.max(np.abs(residual)) < 1e-12


def check_exact_eight_point() -> None:
    _, R, t, p1, p2 = exact_scene()
    x1 = pixels_to_normalized(p1, K1)
    x2 = pixels_to_normalized(p2, K2)
    E = estimate_essential_eight_point(x1, x2)
    assert close_projective(E, essential_from_pose(R, t), 1e-9)


def check_essential_singular_values() -> None:
    _, _, _, p1, p2 = exact_scene()
    E = estimate_essential_eight_point(pixels_to_normalized(p1, K1), pixels_to_normalized(p2, K2))
    s = np.linalg.svd(E, compute_uv=False)
    assert abs(s[0] - s[1]) < 1e-12
    assert s[2] < 1e-12


def check_fundamental_pixel_relation() -> None:
    _, R, t, p1, p2 = exact_scene()
    F = fundamental_from_essential(essential_from_pose(R, t), K1, K2)
    residual = np.sum(homogeneous(p2) * (F @ homogeneous(p1).T).T, axis=1)
    assert np.max(np.abs(residual)) < 1e-10


def check_sampson_scale_invariance() -> None:
    _, R, t, p1, p2 = exact_scene()
    F = fundamental_from_essential(essential_from_pose(R, t), K1, K2)
    perturbed = p2.copy()
    perturbed[:, 0] += np.linspace(-1.0, 1.0, len(p2))
    a = sampson_squared(F, p1, perturbed)
    b = sampson_squared(-13.7 * F, p1, perturbed)
    assert np.allclose(a, b, rtol=1e-12, atol=1e-12)


def check_pose_and_cheirality() -> None:
    _, R, t, p1, p2 = exact_scene()
    pose = recover_relative_pose(p1, p2, K1, K2)
    assert rotation_error_degrees(pose.R, R) < 1e-5
    assert direction_error_degrees(pose.t_direction, t) < 1e-5
    assert np.count_nonzero(pose.positive_depth) == len(p1)
    assert sorted(pose.cheirality_counts)[-1] == len(p1)
    assert sorted(pose.cheirality_counts)[-2] == 0


def check_triangulation_scale() -> None:
    X, R, t, p1, p2 = exact_scene()
    x1 = pixels_to_normalized(p1, K1)
    x2 = pixels_to_normalized(p2, K2)
    t_direction = t / np.linalg.norm(t)
    reconstructed, finite = triangulate_linear(x1, x2, R, t_direction)
    expected = X / np.linalg.norm(t)
    assert np.all(finite)
    assert np.max(np.linalg.norm(reconstructed - expected, axis=1)) < 1e-8


def check_translation_scale_ambiguity() -> None:
    _, R, t, _, _ = exact_scene()
    assert close_projective(essential_from_pose(R, t), essential_from_pose(R, 17.0 * t), 1e-12)


def check_hartley_conditioning() -> None:
    rng = np.random.default_rng(11)
    p1 = np.column_stack((rng.uniform(2000, 6000, 80), rng.uniform(1000, 3500, 80)))
    p2 = p1 + rng.normal(0.0, 80.0, p1.shape)
    raw = np.linalg.cond(design_matrix(p1, p2))
    q1, _ = hartley_similarity(p1)
    q2, _ = hartley_similarity(p2)
    conditioned = np.linalg.cond(design_matrix(q1, q2))
    assert conditioned < raw / 1000.0


def check_preprocessing_conjugation() -> None:
    _, R, t, p1, p2 = exact_scene()
    A1 = np.array([[0.5, 0.0, -20.0], [0.0, 0.5, -10.0], [0.0, 0.0, 1.0]])
    A2 = np.array([[0.75, 0.0, -30.0], [0.0, 0.75, -15.0], [0.0, 0.0, 1.0]])
    p1h = (A1 @ homogeneous(p1).T).T
    p2h = (A2 @ homogeneous(p2).T).T
    q1, q2 = p1h[:, :2] / p1h[:, 2:3], p2h[:, :2] / p2h[:, 2:3]
    E = estimate_essential_eight_point(pixels_to_normalized(q1, A1 @ K1), pixels_to_normalized(q2, A2 @ K2))
    assert close_projective(E, essential_from_pose(R, t), 1e-9)


def check_deterministic_ransac() -> None:
    data = synthetic_matches(20261021)
    args = (data["points1_px"], data["points2_px"], K1, K2)
    a = ransac_essential_eight_point(*args, threshold_px=3.0, seed=99, max_draws=3000)
    b = ransac_essential_eight_point(*args, threshold_px=3.0, seed=99, max_draws=3000)
    assert np.array_equal(a.inliers, b.inliers)
    assert np.allclose(a.E, b.E)
    assert np.array_equal(a.inliers, a.sampson_squared_px <= 9.0)


def check_robust_outlier_recovery() -> None:
    data = synthetic_matches(20261021)
    p1, p2, truth = data["points1_px"], data["points2_px"], data["true_inliers"]
    fit = ransac_essential_eight_point(p1, p2, K1, K2, threshold_px=3.0, seed=20261021, max_draws=4000)
    x1, x2 = pixels_to_normalized(p1, K1), pixels_to_normalized(p2, K2)
    pose = select_pose_by_cheirality(fit.E, x1[fit.inliers], x2[fit.inliers])
    tp = np.count_nonzero(fit.inliers & truth)
    assert tp / np.count_nonzero(fit.inliers) > 0.98
    assert tp / np.count_nonzero(truth) > 0.95
    assert rotation_error_degrees(pose.R, data["R"]) < 1.0
    assert direction_error_degrees(pose.t_direction, data["t"]) < 6.0


def check_noise_recovery() -> None:
    data = synthetic_matches(1234, outliers=0, noise_px=0.4)
    pose = recover_relative_pose(data["points1_px"], data["points2_px"], K1, K2)
    assert rotation_error_degrees(pose.R, data["R"]) < 1.0
    assert direction_error_degrees(pose.t_direction, data["t"]) < 5.0


def check_zero_baseline_is_undefined() -> None:
    try:
        essential_from_pose(np.eye(3), np.zeros(3))
    except ValueError as error:
        assert "zero translation" in str(error)
    else:
        raise AssertionError("zero baseline should have no essential direction")


def check_minimal_right_nullspace() -> None:
    _, R, t, p1, p2 = exact_scene(count=8)
    x1 = pixels_to_normalized(p1, K1)
    x2 = pixels_to_normalized(p2, K2)
    E = estimate_essential_eight_point(x1, x2)
    assert close_projective(E, essential_from_pose(R, t), 1e-8)


def check_low_support_stopping_edge() -> None:
    rng = np.random.default_rng(404)
    p1 = np.column_stack((rng.uniform(0, 1280, 24), rng.uniform(0, 720, 24)))
    p2 = np.column_stack((rng.uniform(0, 1280, 24), rng.uniform(0, 720, 24)))
    result = ransac_essential_eight_point(
        p1, p2, K1, K2, threshold_px=1e-9,
        min_draws=16, max_draws=32, seed=404, inner_trials=0,
    )
    assert result.draws == 32
    assert np.count_nonzero(result.inliers) == 0


CHECKS = [
    check_epipolar_constraint,
    check_exact_eight_point,
    check_essential_singular_values,
    check_fundamental_pixel_relation,
    check_sampson_scale_invariance,
    check_pose_and_cheirality,
    check_triangulation_scale,
    check_translation_scale_ambiguity,
    check_hartley_conditioning,
    check_preprocessing_conjugation,
    check_deterministic_ransac,
    check_robust_outlier_recovery,
    check_noise_recovery,
    check_zero_baseline_is_undefined,
    check_minimal_right_nullspace,
    check_low_support_stopping_edge,
]


if __name__ == "__main__":
    for check in CHECKS:
        check()
        print(f"PASS {check.__name__}")
    print(f"{len(CHECKS)} checks passed")

