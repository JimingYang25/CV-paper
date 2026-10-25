"""Educational robust point-to-plane ICP with explicit SE(3) conventions.

Convention
----------
T_target_from_source maps column-vector points from the source frame into the
target frame: x_target = R @ x_source + t.  Twists are ordered
[omega_x, omega_y, omega_z, v_x, v_y, v_z] and are applied on the left:
T <- Exp(delta_xi^) @ T.

This NumPy implementation favors inspectability over production speed.  It uses
chunked brute-force nearest neighbors; production systems should use a KD-tree,
projective association, or a GPU spatial index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


Array = np.ndarray


def skew(a: Array) -> Array:
    """Return [a]_x such that [a]_x b == a cross b."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]],
        dtype=np.float64,
    )


def exp_se3(xi: Array) -> Array:
    """SE(3) exponential for xi=[omega, v], returned as a 4x4 matrix."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    omega, v = xi[:3], xi[3:]
    theta = float(np.linalg.norm(omega))
    Omega = skew(omega)
    Omega2 = Omega @ Omega
    if theta < 1e-8:
        # Series through the first useful curvature terms.
        R = np.eye(3) + Omega + 0.5 * Omega2
        V = np.eye(3) + 0.5 * Omega + (1.0 / 6.0) * Omega2
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta * theta)
        c = (theta - np.sin(theta)) / (theta * theta * theta)
        R = np.eye(3) + a * Omega + b * Omega2
        V = np.eye(3) + b * Omega + c * Omega2
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def transform_points(T_target_from_source: Array, points_source: Array) -> Array:
    """Transform an [N,3] point array without constructing homogeneous points."""
    T = np.asarray(T_target_from_source, dtype=np.float64)
    p = np.asarray(points_source, dtype=np.float64)
    return p @ T[:3, :3].T + T[:3, 3]


def transform_normals(T_target_from_source: Array, normals_source: Array) -> Array:
    """Rotate [N,3] normals; translation must not affect a direction."""
    T = np.asarray(T_target_from_source, dtype=np.float64)
    n = np.asarray(normals_source, dtype=np.float64) @ T[:3, :3].T
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)


def invert_transform(T_a_from_b: Array) -> Array:
    """Invert a rigid 4x4 transform."""
    T = np.asarray(T_a_from_b, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def nearest_neighbors(source: Array, target: Array, chunk_size: int = 1024) -> tuple[Array, Array]:
    """Exact nearest neighbors with bounded [chunk,N_target] temporary memory."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("source and target must have shapes [N,3] and [M,3]")
    if len(target) == 0:
        raise ValueError("target must contain at least one point")
    indices = np.empty(len(source), dtype=np.int64)
    distances = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        diff = source[start:stop, None, :] - target[None, :, :]
        distance2 = np.einsum("cmi,cmi->cm", diff, diff)
        idx = np.argmin(distance2, axis=1)
        indices[start:stop] = idx
        distances[start:stop] = np.sqrt(distance2[np.arange(stop - start), idx])
    return indices, distances


def huber_weights(residuals: Array, delta: float) -> Array:
    """IRLS weights psi(r)/r for Huber loss (the square-root is applied later)."""
    r = np.abs(np.asarray(residuals, dtype=np.float64))
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    return np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-15))


def huber_cost(residuals: Array, delta: float) -> float:
    """Mean Huber cost using rho=0.5*r^2 in the quadratic region."""
    a = np.abs(np.asarray(residuals, dtype=np.float64))
    rho = np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))
    return float(np.mean(rho)) if len(rho) else float("inf")


@dataclass(frozen=True)
class ICPConfig:
    max_iterations: int = 40
    distance_threshold: float = 0.12
    normal_angle_deg: float = 45.0
    huber_delta: float = 0.012
    trim_fraction: float = 0.90
    damping: float = 1e-8
    step_tolerance: float = 1e-8
    relative_cost_tolerance: float = 1e-7
    min_correspondences: int = 30
    chunk_size: int = 1024


@dataclass
class ICPResult:
    T_target_from_source: Array
    converged: bool
    reason: str
    iterations: list[dict]
    correspondence_count: int
    inlier_rmse: float
    coverage: float
    normal_matrix_eigenvalues: Array
    condition_number: float


def _select_correspondences(
    x_target: Array,
    target_points: Array,
    target_normals: Array,
    distances: Array,
    indices: Array,
    source_normals_target: Optional[Array],
    config: ICPConfig,
) -> tuple[Array, Array, Array, Array]:
    q = target_points[indices]
    n = target_normals[indices]
    residual = np.einsum("ni,ni->n", n, x_target - q)
    valid = np.isfinite(distances) & (distances <= config.distance_threshold)
    if source_normals_target is not None:
        cosine = np.abs(np.einsum("ni,ni->n", source_normals_target, n))
        valid &= cosine >= np.cos(np.deg2rad(config.normal_angle_deg))
    keep = np.flatnonzero(valid)
    if len(keep) and config.trim_fraction < 1.0:
        count = max(config.min_correspondences, int(np.floor(config.trim_fraction * len(keep))))
        count = min(count, len(keep))
        order = np.argsort(np.abs(residual[keep]), kind="stable")[:count]
        keep = keep[order]
    return x_target[keep], q[keep], n[keep], residual[keep]


def linearize_point_to_plane(x_target: Array, q_target: Array, n_target: Array) -> tuple[Array, Array]:
    """Return residual r [K] and left-perturbation Jacobian J [K,6]."""
    x = np.asarray(x_target, dtype=np.float64)
    q = np.asarray(q_target, dtype=np.float64)
    n = np.asarray(n_target, dtype=np.float64)
    r = np.einsum("ni,ni->n", n, x - q)
    # n^T(omega cross x + v) = (x cross n)^T omega + n^T v.
    J = np.concatenate([np.cross(x, n), n], axis=1)
    return r, J


def _spectrum(H: Array) -> tuple[Array, float]:
    eigenvalues = np.linalg.eigvalsh(0.5 * (H + H.T))
    largest = max(float(eigenvalues[-1]), 0.0)
    floor = max(1e-12 * largest, 1e-15)
    positive = eigenvalues[eigenvalues > floor]
    condition = float("inf") if len(positive) < 6 else float(positive[-1] / positive[0])
    return eigenvalues, condition


def point_to_plane_icp(
    source_points: Array,
    target_points: Array,
    target_normals: Array,
    *,
    initial_T_target_from_source: Optional[Array] = None,
    source_normals: Optional[Array] = None,
    config: ICPConfig = ICPConfig(),
) -> ICPResult:
    """Robust local ICP using nearest neighbors and Huber IRLS."""
    p = np.asarray(source_points, dtype=np.float64)
    q_all = np.asarray(target_points, dtype=np.float64)
    n_all = np.asarray(target_normals, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or q_all.shape != n_all.shape or q_all.shape[1] != 3:
        raise ValueError("point and normal arrays must be [N,3]")
    if source_normals is not None and np.asarray(source_normals).shape != p.shape:
        raise ValueError("source_normals must have the same shape as source_points")
    if not 0.0 < config.trim_fraction <= 1.0:
        raise ValueError("trim_fraction must be in (0,1]")

    T = np.eye(4) if initial_T_target_from_source is None else np.array(initial_T_target_from_source, dtype=np.float64, copy=True)
    history: list[dict] = []
    previous_cost = float("inf")
    converged, reason = False, "maximum iterations reached"
    last_H = np.zeros((6, 6), dtype=np.float64)
    last_r = np.array([], dtype=np.float64)
    last_count = 0

    for iteration in range(config.max_iterations):
        x = transform_points(T, p)
        source_n = None if source_normals is None else transform_normals(T, source_normals)
        indices, distances = nearest_neighbors(x, q_all, config.chunk_size)
        xk, qk, nk, rk = _select_correspondences(
            x, q_all, n_all, distances, indices, source_n, config
        )
        last_count = len(rk)
        if last_count < config.min_correspondences:
            reason = f"too few correspondences ({last_count})"
            break

        r, J = linearize_point_to_plane(xk, qk, nk)
        weights = huber_weights(r, config.huber_delta)
        sw = np.sqrt(weights)
        Jw, rw = J * sw[:, None], r * sw
        H = Jw.T @ Jw
        g = Jw.T @ rw
        last_H, last_r = H, r
        eigenvalues, condition = _spectrum(H)
        diagonal_scale = max(float(np.trace(H) / 6.0), 1.0)
        try:
            delta = np.linalg.solve(H + config.damping * diagonal_scale * np.eye(6), -g)
        except np.linalg.LinAlgError:
            reason = "normal equations are singular"
            break

        cost = huber_cost(r, config.huber_delta)
        step_norm = float(np.linalg.norm(delta))
        history.append(
            {
                "iteration": iteration,
                "cost": cost,
                "rmse": float(np.sqrt(np.mean(r * r))),
                "correspondences": int(last_count),
                "step_norm": step_norm,
                "condition_number": condition,
                "min_eigenvalue": float(eigenvalues[0]),
            }
        )
        T = exp_se3(delta) @ T

        relative_drop = (previous_cost - cost) / max(previous_cost, 1e-15)
        if step_norm < config.step_tolerance:
            converged, reason = True, "step tolerance reached"
            break
        if np.isfinite(previous_cost) and 0.0 <= relative_drop < config.relative_cost_tolerance:
            converged, reason = True, "relative cost tolerance reached"
            break
        previous_cost = cost

    # Recompute final correspondences so result metrics describe returned T.
    x = transform_points(T, p)
    source_n = None if source_normals is None else transform_normals(T, source_normals)
    indices, distances = nearest_neighbors(x, q_all, config.chunk_size)
    xk, qk, nk, rk = _select_correspondences(x, q_all, n_all, distances, indices, source_n, config)
    if len(rk) >= config.min_correspondences:
        r, J = linearize_point_to_plane(xk, qk, nk)
        weights = huber_weights(r, config.huber_delta)
        Jw = J * np.sqrt(weights)[:, None]
        last_H, last_r, last_count = Jw.T @ Jw, r, len(r)
    eigenvalues, condition = _spectrum(last_H)
    rmse = float(np.sqrt(np.mean(last_r * last_r))) if len(last_r) else float("inf")
    return ICPResult(
        T_target_from_source=T,
        converged=converged,
        reason=reason,
        iterations=history,
        correspondence_count=int(last_count),
        inlier_rmse=rmse,
        coverage=float(last_count / max(len(p), 1)),
        normal_matrix_eigenvalues=eigenvalues,
        condition_number=condition,
    )


def point_to_point_icp(
    source_points: Array,
    target_points: Array,
    *,
    initial_T_target_from_source: Optional[Array] = None,
    max_iterations: int = 40,
    distance_threshold: float = 0.12,
    trim_fraction: float = 0.90,
    step_tolerance: float = 1e-8,
) -> tuple[Array, list[dict]]:
    """Small point-to-point SVD baseline for convergence comparisons."""
    p = np.asarray(source_points, dtype=np.float64)
    q_all = np.asarray(target_points, dtype=np.float64)
    T = np.eye(4) if initial_T_target_from_source is None else np.array(initial_T_target_from_source, dtype=np.float64, copy=True)
    history: list[dict] = []
    for iteration in range(max_iterations):
        x = transform_points(T, p)
        indices, distances = nearest_neighbors(x, q_all)
        keep = np.flatnonzero(distances <= distance_threshold)
        if len(keep) < 3:
            break
        count = max(3, min(len(keep), int(np.floor(trim_fraction * len(keep)))))
        keep = keep[np.argsort(distances[keep], kind="stable")[:count]]
        a, b = x[keep], q_all[indices[keep]]
        ac, bc = a.mean(axis=0), b.mean(axis=0)
        U, _, Vt = np.linalg.svd((a - ac).T @ (b - bc))
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0.0:
            Vt[-1] *= -1.0
            R = Vt.T @ U.T
        t = bc - R @ ac
        update = np.eye(4)
        update[:3, :3], update[:3, 3] = R, t
        T = update @ T
        step = np.linalg.norm(np.r_[rotation_vector(R), t])
        history.append({"iteration": iteration, "rmse": float(np.sqrt(np.mean(distances[keep] ** 2))), "step_norm": float(step)})
        if step < step_tolerance:
            break
    return T, history


def rotation_vector(R: Array) -> Array:
    """Stable SO(3) logarithm for evaluation and tiny update norms."""
    R = np.asarray(R, dtype=np.float64)
    cosine = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cosine))
    if theta < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta / (2.0 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def pose_error(T_est: Array, T_true: Array) -> tuple[float, float]:
    """Return rotation error in degrees and translation error in metres."""
    delta = np.asarray(T_est) @ invert_transform(np.asarray(T_true))
    rotation_deg = float(np.rad2deg(np.linalg.norm(rotation_vector(delta[:3, :3]))))
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    return rotation_deg, translation_m
