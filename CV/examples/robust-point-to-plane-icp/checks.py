"""Numerical checks for the educational robust ICP implementation."""

from __future__ import annotations

import numpy as np

from application import demonstration
from robust_icp import (
    ICPConfig,
    exp_se3,
    huber_cost,
    huber_weights,
    invert_transform,
    linearize_point_to_plane,
    nearest_neighbors,
    point_to_plane_icp,
    pose_error,
    skew,
    transform_normals,
    transform_points,
)
from synthetic_scene import make_corner_model, make_planar_model, make_registration_case


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS {name}{': ' + detail if detail else ''}")


def main() -> None:
    rng = np.random.default_rng(4)

    a, b = rng.normal(size=3), rng.normal(size=3)
    check("skew-cross identity", np.allclose(skew(a) @ b, np.cross(a, b), atol=1e-12))

    check("zero twist is identity", np.allclose(exp_se3(np.zeros(6)), np.eye(4), atol=1e-15))

    xi = np.array([0.1, -0.06, 0.04, 0.2, -0.1, 0.05])
    T = exp_se3(xi)
    check("SE(3) rotation is proper", np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(T[:3, :3]), 1.0, atol=1e-12))

    points = rng.normal(size=(12, 3))
    roundtrip = transform_points(invert_transform(T), transform_points(T, points))
    check("rigid inverse round-trip", np.allclose(roundtrip, points, atol=1e-12))

    normals = rng.normal(size=(12, 3)); normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    T_shifted = T.copy(); T_shifted[:3, 3] += 100.0
    check("normal transform ignores translation", np.allclose(transform_normals(T, normals), transform_normals(T_shifted, normals), atol=1e-12))
    check("normal transform preserves unit length", np.allclose(np.linalg.norm(transform_normals(T, normals), axis=1), 1.0, atol=1e-12))

    query = np.array([[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]])
    target = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    idx, dist = nearest_neighbors(query, target, chunk_size=1)
    check("nearest-neighbor indices", np.array_equal(idx, np.array([0, 1])))
    check("nearest-neighbor distances", np.allclose(dist, 0.1, atol=1e-12))

    r = np.array([0.0, 0.5, 2.0])
    check("Huber IRLS weights", np.allclose(huber_weights(r, 1.0), [1.0, 1.0, 0.5]))
    check("Huber cost", np.isclose(huber_cost(r, 1.0), (0.0 + 0.125 + 1.5) / 3.0))

    x = rng.normal(size=(20, 3))
    q = x + 0.01 * rng.normal(size=(20, 3))
    n = rng.normal(size=(20, 3)); n /= np.linalg.norm(n, axis=1, keepdims=True)
    residual, J = linearize_point_to_plane(x, q, n)
    eps = 1e-7
    J_fd = np.empty_like(J)
    for k in range(6):
        step = np.zeros(6); step[k] = eps
        xp = transform_points(exp_se3(step), x)
        rp = np.einsum("ni,ni->n", n, xp - q)
        J_fd[:, k] = (rp - residual) / eps
    check("analytic left-SE(3) Jacobian", np.max(np.abs(J - J_fd)) < 2e-7, f"max error {np.max(np.abs(J-J_fd)):.3e}")

    plane, plane_n = make_planar_model(19)
    _, J_plane = linearize_point_to_plane(plane, plane, plane_n)
    eig_plane = np.linalg.eigvalsh(J_plane.T @ J_plane)
    check("single plane has three unobservable directions", np.count_nonzero(eig_plane < 1e-10 * eig_plane[-1]) == 3, f"eigenvalues {eig_plane}")

    corner, corner_n, _ = make_corner_model()
    _, J_corner = linearize_point_to_plane(corner, corner, corner_n)
    eig_corner = np.linalg.eigvalsh(J_corner.T @ J_corner)
    check("multi-plane scene is full rank", eig_corner[0] > 1e-4 * eig_corner[-1], f"ratio {eig_corner[0]/eig_corner[-1]:.3e}")

    case = make_registration_case(seed=8)
    robust = point_to_plane_icp(
        case["source_points"], case["target_points"], case["target_normals"],
        source_normals=case["source_normals"], initial_T_target_from_source=case["T_initial"],
        config=ICPConfig(distance_threshold=0.14, huber_delta=0.012, trim_fraction=0.82, normal_angle_deg=50.0),
    )
    rot, trans = pose_error(robust.T_target_from_source, case["T_true"])
    check("robust ICP rotation accuracy", rot < 0.12, f"{rot:.5f} deg")
    check("robust ICP translation accuracy", trans < 0.0010, f"{1000*trans:.5f} mm")
    check("robust ICP has useful coverage", robust.coverage > 0.65, f"{robust.coverage:.4f}")
    check("robust ICP final system is well conditioned", np.isfinite(robust.condition_number) and robust.condition_number < 100.0, f"{robust.condition_number:.3f}")

    plain = point_to_plane_icp(
        case["source_points"], case["target_points"], case["target_normals"],
        source_normals=case["source_normals"], initial_T_target_from_source=case["T_initial"],
        config=ICPConfig(distance_threshold=0.14, huber_delta=1e6, trim_fraction=1.0, normal_angle_deg=180.0),
    )
    _, plain_trans = pose_error(plain.T_target_from_source, case["T_true"])
    check("robust policy improves outlier case", trans < 0.5 * plain_trans, f"robust {1000*trans:.3f} mm vs plain {1000*plain_trans:.3f} mm")

    app = demonstration(seed=11)
    check("tracking gate accepts good prediction", app["good_prediction"]["integrate_depth"])
    check("tracking gate rejects bad prediction", not app["bad_prediction"]["integrate_depth"], app["bad_prediction"]["reason"])

    repeated = point_to_plane_icp(
        case["source_points"], case["target_points"], case["target_normals"],
        source_normals=case["source_normals"], initial_T_target_from_source=case["T_initial"],
        config=ICPConfig(distance_threshold=0.14, huber_delta=0.012, trim_fraction=0.82, normal_angle_deg=50.0),
    )
    check("deterministic repeat", np.array_equal(robust.T_target_from_source, repeated.T_target_from_source))

    print("\n21/21 checks passed")


if __name__ == "__main__":
    main()
