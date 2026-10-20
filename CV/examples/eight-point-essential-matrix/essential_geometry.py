"""Inspectable two-view geometry for a normalized eight-point lesson.

Convention
----------
``x2 ~ R @ x1 + t`` maps a 3-D point expressed in camera-1 coordinates
into camera-2 coordinates.  Pixel points are stored as ``(u, v)``.

This is educational NumPy, not a replacement for a production five-point
USAC implementation or bundle adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class PoseResult:
    E: Array
    R: Array
    t_direction: Array
    points3d: Array
    positive_depth: Array
    cheirality_counts: tuple[int, int, int, int]


@dataclass(frozen=True)
class RansacResult:
    E: Array
    inliers: Array
    sampson_squared_px: Array
    draws: int
    valid_hypotheses: int


def _points2(points: Array, name: str = "points") -> Array:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2]")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    return points


def homogeneous(points: Array) -> Array:
    """Convert ``[N,2]`` Euclidean points to homogeneous ``[N,3]``."""
    points = _points2(points)
    return np.column_stack((points, np.ones(len(points), dtype=np.float64)))


def skew(vector: Array) -> Array:
    """Return the 3x3 cross-product matrix ``[vector]_x``."""
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def essential_from_pose(R: Array, t: Array) -> Array:
    """Build ``E=[t]_x R`` for ``X2 = R X1 + t`` and normalize its scale."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    E = skew(t) @ R
    norm = np.linalg.norm(E)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("zero translation has no observable essential matrix")
    return E / norm


def pixels_to_normalized(points_px: Array, K: Array) -> Array:
    """Map distorted-free pixels ``[N,2]`` through ``K^-1`` to ``[N,2]`` rays."""
    points_px = _points2(points_px, "points_px")
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    rays = np.linalg.solve(K, homogeneous(points_px).T).T
    finite = np.abs(rays[:, 2]) > 1e-14
    if not np.all(finite):
        raise ValueError("K^-1 mapped a point to infinity")
    return rays[:, :2] / rays[:, 2:3]


def normalized_to_pixels(points: Array, K: Array) -> Array:
    """Map normalized camera coordinates ``[N,2]`` to pixels ``[N,2]``."""
    points = _points2(points)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    pixels_h = (K @ homogeneous(points).T).T
    return pixels_h[:, :2] / pixels_h[:, 2:3]


def hartley_similarity(points: Array) -> tuple[Array, Array]:
    """Center 2-D points and set their RMS radius to sqrt(2).

    Returns normalized points and a 3x3 similarity ``T`` satisfying
    ``x_normalized ~ T @ x``.
    """
    points = _points2(points)
    center = np.mean(points, axis=0)
    centered = points - center
    rms = np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
    if rms <= 1e-12:
        raise ValueError("points have no spatial spread")
    scale = np.sqrt(2.0) / rms
    T = np.array(
        [[scale, 0.0, -scale * center[0]],
         [0.0, scale, -scale * center[1]],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transformed = (T @ homogeneous(points).T).T
    return transformed[:, :2], T


def design_matrix(points1: Array, points2: Array) -> Array:
    """Build ``A [e11,...,e33]^T = 0`` using row-major vectorization."""
    points1 = _points2(points1, "points1")
    points2 = _points2(points2, "points2")
    if points1.shape != points2.shape:
        raise ValueError("point arrays must have the same shape")
    x1, y1 = points1.T
    x2, y2 = points2.T
    return np.column_stack(
        (x2 * x1, x2 * y1, x2,
         y2 * x1, y2 * y1, y2,
         x1, y1, np.ones(len(points1)))
    )


def project_to_essential(E: Array) -> Array:
    """Project a 3x3 matrix to singular values ``(s,s,0)`` in Frobenius norm."""
    U, singular, Vt = np.linalg.svd(np.asarray(E, dtype=np.float64).reshape(3, 3))
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1.0
    s = 0.5 * (singular[0] + singular[1])
    E_projected = U @ np.diag([s, s, 0.0]) @ Vt
    norm = np.linalg.norm(E_projected)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("degenerate essential estimate")
    return E_projected / norm


def estimate_essential_eight_point(
    points1_normalized: Array,
    points2_normalized: Array,
    *,
    condition: bool = True,
) -> Array:
    """Estimate an essential matrix from at least eight calibrated matches.

    Inputs are dimensionless normalized camera coordinates, not pixels.
    ``condition=True`` applies an additional Hartley similarity in each view.
    """
    points1 = _points2(points1_normalized, "points1_normalized")
    points2 = _points2(points2_normalized, "points2_normalized")
    if points1.shape != points2.shape or len(points1) < 8:
        raise ValueError("need matching [N,2] arrays with N >= 8")

    if condition:
        p1, T1 = hartley_similarity(points1)
        p2, T2 = hartley_similarity(points2)
    else:
        p1, p2 = points1, points2
        T1 = T2 = np.eye(3)

    A = design_matrix(p1, p2)
    if np.linalg.matrix_rank(A, tol=1e-11) < 8:
        raise ValueError("rank-deficient correspondence configuration")
    # full_matrices=True matters for the minimal 8x9 case: Vt must retain the
    # one-dimensional right nullspace as its ninth row.
    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    E_conditioned = Vt[-1].reshape(3, 3)
    E = T2.T @ E_conditioned @ T1
    return project_to_essential(E)


def fundamental_from_essential(E: Array, K1: Array, K2: Array) -> Array:
    """Convert normalized-coordinate ``E`` to pixel-coordinate ``F``."""
    K1 = np.asarray(K1, dtype=np.float64).reshape(3, 3)
    K2 = np.asarray(K2, dtype=np.float64).reshape(3, 3)
    E = np.asarray(E, dtype=np.float64).reshape(3, 3)
    F = np.linalg.solve(K2.T, E) @ np.linalg.inv(K1)
    norm = np.linalg.norm(F)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("degenerate fundamental matrix")
    return F / norm


def sampson_squared(matrix: Array, points1: Array, points2: Array) -> Array:
    """First-order squared geometric error for one epipolar matrix.

    The returned unit is squared pixels when ``matrix`` is F and inputs are
    pixels; it is squared normalized-coordinate units for E and calibrated rays.
    """
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    x1 = homogeneous(points1)
    x2 = homogeneous(points2)
    Mx1 = (matrix @ x1.T).T
    Mtx2 = (matrix.T @ x2.T).T
    numerator = np.sum(x2 * Mx1, axis=1) ** 2
    denominator = (
        Mx1[:, 0] ** 2 + Mx1[:, 1] ** 2
        + Mtx2[:, 0] ** 2 + Mtx2[:, 1] ** 2
    )
    return numerator / np.maximum(denominator, np.finfo(np.float64).tiny)


def decompose_essential(E: Array) -> tuple[tuple[Array, Array], ...]:
    """Return the four ``(R,t)`` candidates encoded by an essential matrix."""
    U, _, Vt = np.linalg.svd(project_to_essential(E))
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1.0
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    if np.linalg.det(R1) < 0:
        R1 *= -1.0
    if np.linalg.det(R2) < 0:
        R2 *= -1.0
    t = U[:, 2]
    t /= np.linalg.norm(t)
    return ((R1, t), (R1, -t), (R2, t), (R2, -t))


def triangulate_linear(
    points1_normalized: Array,
    points2_normalized: Array,
    R: Array,
    t: Array,
) -> tuple[Array, Array]:
    """DLT triangulation for ``P1=[I|0]`` and ``P2=[R|t]``.

    Returns Euclidean camera-1 points ``[N,3]`` and a finite mask ``[N]``.
    """
    p1 = _points2(points1_normalized, "points1_normalized")
    p2 = _points2(points2_normalized, "points2_normalized")
    if p1.shape != p2.shape:
        raise ValueError("point arrays must have the same shape")
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    P1 = np.column_stack((np.eye(3), np.zeros(3)))
    P2 = np.column_stack((R, t))
    output = np.full((len(p1), 3), np.nan, dtype=np.float64)
    finite = np.zeros(len(p1), dtype=bool)
    for index, ((x1, y1), (x2, y2)) in enumerate(zip(p1, p2)):
        A = np.stack(
            (x1 * P1[2] - P1[0], y1 * P1[2] - P1[1],
             x2 * P2[2] - P2[0], y2 * P2[2] - P2[1])
        )
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        if abs(Xh[3]) > 1e-12:
            output[index] = Xh[:3] / Xh[3]
            finite[index] = np.all(np.isfinite(output[index]))
    return output, finite


def select_pose_by_cheirality(
    E: Array,
    points1_normalized: Array,
    points2_normalized: Array,
    *,
    max_depth: float = np.inf,
) -> PoseResult:
    """Choose the decomposition with most finite points in front of both cameras."""
    p1 = _points2(points1_normalized, "points1_normalized")
    p2 = _points2(points2_normalized, "points2_normalized")
    candidates = decompose_essential(E)
    records = []
    for R, t in candidates:
        X1, finite = triangulate_linear(p1, p2, R, t)
        X2 = (R @ X1.T).T + t
        positive = (
            finite & (X1[:, 2] > 1e-9) & (X2[:, 2] > 1e-9)
            & (X1[:, 2] < max_depth) & (X2[:, 2] < max_depth)
        )
        records.append((int(np.count_nonzero(positive)), R, t, X1, positive))
    best = max(records, key=lambda item: item[0])
    counts = tuple(record[0] for record in records)
    return PoseResult(
        E=project_to_essential(E),
        R=best[1],
        t_direction=best[2],
        points3d=best[3],
        positive_depth=best[4],
        cheirality_counts=counts,
    )


def _required_draws(confidence: float, inlier_ratio: float, sample_size: int) -> int:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if inlier_ratio <= 0.0:
        return np.iinfo(np.int32).max
    if inlier_ratio >= 1.0:
        return 1
    success = inlier_ratio ** sample_size
    if success <= np.finfo(np.float64).eps:
        return np.iinfo(np.int32).max
    return max(1, ceil(log(1.0 - confidence) / log(1.0 - success)))


def ransac_essential_eight_point(
    points1_px: Array,
    points2_px: Array,
    K1: Array,
    K2: Array,
    *,
    threshold_px: float = 1.5,
    confidence: float = 0.999,
    min_draws: int = 64,
    max_draws: int = 3000,
    seed: int = 0,
    local_refits: int = 2,
    inner_trials: int = 64,
) -> RansacResult:
    """Small didactic 8-point RANSAC scored by pixel-domain Sampson error.

    This deliberately omits five-point hypotheses, PROSAC, the complete
    LO-RANSAC schedule, degeneracy recovery, and nonlinear bundle adjustment.
    """
    p1_px = _points2(points1_px, "points1_px")
    p2_px = _points2(points2_px, "points2_px")
    if p1_px.shape != p2_px.shape or len(p1_px) < 8:
        raise ValueError("need matching [N,2] arrays with N >= 8")
    if threshold_px <= 0 or min_draws < 1 or max_draws < min_draws:
        raise ValueError("invalid RANSAC controls")
    p1 = pixels_to_normalized(p1_px, K1)
    p2 = pixels_to_normalized(p2_px, K2)
    threshold2 = threshold_px * threshold_px
    rng = np.random.default_rng(seed)
    best_E = None
    best_mask = np.zeros(len(p1), dtype=bool)
    # MSAC-style capped loss is the primary objective.  Consensus size breaks
    # ties; this avoids preferring a loose, high-count model by construction.
    best_score = (-np.inf, 0)
    valid = 0
    adaptive_limit = max_draws
    draws = 0
    for draws in range(1, max_draws + 1):
        sample = rng.choice(len(p1), size=8, replace=False)
        try:
            E = estimate_essential_eight_point(p1[sample], p2[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        valid += 1
        F = fundamental_from_essential(E, K1, K2)
        residual2 = sampson_squared(F, p1_px, p2_px)
        mask = residual2 <= threshold2
        count = int(np.count_nonzero(mask))
        capped = float(np.sum(np.minimum(residual2, threshold2)))
        score = (-capped, count)
        if score > best_score:
            best_E, best_mask, best_score = E, mask, score
            ratio = count / len(p1)
            adaptive_limit = min(
                adaptive_limit,
                max(min_draws, _required_draws(confidence, ratio, 8)),
            )
        if draws >= adaptive_limit and draws >= min_draws:
            break
    if best_E is None:
        raise RuntimeError("RANSAC found no valid essential hypothesis")

    # Try small non-minimal subsets from the best basin.  This is a compact
    # LO-RANSAC-style stabilizer, not the complete published LO procedure.
    for _ in range(inner_trials):
        support = np.flatnonzero(best_mask)
        if len(support) < 8:
            break
        sample = rng.choice(support, size=min(16, len(support)), replace=False)
        try:
            candidate = estimate_essential_eight_point(p1[sample], p2[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        F = fundamental_from_essential(candidate, K1, K2)
        residual2 = sampson_squared(F, p1_px, p2_px)
        mask = residual2 <= threshold2
        score = (-float(np.sum(np.minimum(residual2, threshold2))), int(np.count_nonzero(mask)))
        if score > best_score:
            best_E, best_mask, best_score = candidate, mask, score

    # Consensus refits are evaluated on all data. Retain the current model when
    # the selected capped-loss objective does not improve.
    for _ in range(local_refits):
        if np.count_nonzero(best_mask) < 8:
            break
        try:
            candidate = estimate_essential_eight_point(p1[best_mask], p2[best_mask])
        except (ValueError, np.linalg.LinAlgError):
            break
        F = fundamental_from_essential(candidate, K1, K2)
        residual2 = sampson_squared(F, p1_px, p2_px)
        mask = residual2 <= threshold2
        score = (-float(np.sum(np.minimum(residual2, threshold2))), int(np.count_nonzero(mask)))
        if score <= best_score:
            break
        best_E, best_mask, best_score = candidate, mask, score

    final_F = fundamental_from_essential(best_E, K1, K2)
    final_residual2 = sampson_squared(final_F, p1_px, p2_px)
    final_mask = final_residual2 <= threshold2
    return RansacResult(best_E, final_mask, final_residual2, draws, valid)


def recover_relative_pose(
    points1_px: Array,
    points2_px: Array,
    K1: Array,
    K2: Array,
    *,
    inliers: Array | None = None,
) -> PoseResult:
    """Estimate E from selected matches and choose its physical pose hypothesis."""
    p1_px = _points2(points1_px, "points1_px")
    p2_px = _points2(points2_px, "points2_px")
    if inliers is None:
        inliers = np.ones(len(p1_px), dtype=bool)
    inliers = np.asarray(inliers, dtype=bool).reshape(-1)
    if len(inliers) != len(p1_px) or np.count_nonzero(inliers) < 8:
        raise ValueError("inlier mask must select at least eight matches")
    p1 = pixels_to_normalized(p1_px, K1)
    p2 = pixels_to_normalized(p2_px, K2)
    E = estimate_essential_eight_point(p1[inliers], p2[inliers])
    return select_pose_by_cheirality(E, p1[inliers], p2[inliers])


def rotation_error_degrees(R_estimate: Array, R_true: Array) -> float:
    delta = np.asarray(R_estimate).reshape(3, 3) @ np.asarray(R_true).reshape(3, 3).T
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def direction_error_degrees(estimate: Array, truth: Array) -> float:
    estimate = np.asarray(estimate, dtype=np.float64).reshape(3)
    truth = np.asarray(truth, dtype=np.float64).reshape(3)
    estimate /= np.linalg.norm(estimate)
    truth /= np.linalg.norm(truth)
    cosine = np.clip(np.dot(estimate, truth), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rotation_y(degrees: float) -> Array:
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def project_scene(points3d_cam1: Array, R: Array, t: Array, K1: Array, K2: Array) -> tuple[Array, Array]:
    """Project camera-1 3-D points into two views under ``X2=R X1+t``."""
    X1 = np.asarray(points3d_cam1, dtype=np.float64)
    if X1.ndim != 2 or X1.shape[1] != 3:
        raise ValueError("points3d_cam1 must have shape [N,3]")
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    X2 = (R @ X1.T).T + t
    if np.any(X1[:, 2] <= 0) or np.any(X2[:, 2] <= 0):
        raise ValueError("scene must be visible in front of both cameras")
    p1 = X1[:, :2] / X1[:, 2:3]
    p2 = X2[:, :2] / X2[:, 2:3]
    return normalized_to_pixels(p1, K1), normalized_to_pixels(p2, K2)

