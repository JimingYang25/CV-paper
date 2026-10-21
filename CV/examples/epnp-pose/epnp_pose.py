"""Inspectable non-planar EPnP reconstruction with robust fitting.

Pose convention
---------------
``X_camera = R @ X_world + t``.  Pixel points use ``(u, v)`` ordering.

Scope
-----
This educational implementation reconstructs the four-control-point,
non-planar EPnP path and the three beta initializations used by the public
reference/OpenCV implementations.  It conservatively requires six or more
correspondences because it does not reproduce every minimal-case recovery
detail of the paper.  It also deliberately rejects planar and nearly planar
control-point configurations; use a solver with an explicit planar path (for
example IPPE for markers) in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


Array = np.ndarray
CONTROL_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass(frozen=True)
class EPnPDiagnostics:
    control_points_world: Array
    barycentric_coordinates: Array
    betas: Array
    initializer_index: int
    design_singular_values: Array
    scatter_eigenvalues: Array
    planarity_ratio: float


@dataclass(frozen=True)
class PoseEstimate:
    R: Array
    t: Array
    reprojection_rmse_px: float
    positive_depth_fraction: float
    method: str
    diagnostics: EPnPDiagnostics | None = None
    optimizer_nfev: int | None = None
    jacobian_condition: float | None = None


@dataclass(frozen=True)
class RansacPose:
    pose: PoseEstimate
    inliers: Array
    reprojection_errors_px: Array
    draws: int
    valid_hypotheses: int


def _points3(points: Array, name: str) -> Array:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3]")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    return points


def _points2(points: Array, name: str) -> Array:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2]")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    return points


def homogeneous2(points: Array) -> Array:
    points = _points2(points, "points")
    return np.column_stack((points, np.ones(len(points))))


def pixels_to_normalized(points_px: Array, K: Array) -> Array:
    points_px = _points2(points_px, "points_px")
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    rays = np.linalg.solve(K, homogeneous2(points_px).T).T
    if np.any(np.abs(rays[:, 2]) <= 1e-14):
        raise ValueError("K^-1 mapped a pixel to infinity")
    return rays[:, :2] / rays[:, 2:3]


def project_points(
    points_world: Array,
    R: Array,
    t: Array,
    K: Array,
) -> tuple[Array, Array]:
    """Project world points; returns pixels and a positive finite depth mask."""
    points_world = _points3(points_world, "points_world")
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    camera = (R @ points_world.T).T + t
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-12)
    pixels = np.full((len(points_world), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        projected = (K @ camera[valid].T).T
        pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def reprojection_errors(
    points_world: Array,
    points_px: Array,
    R: Array,
    t: Array,
    K: Array,
) -> Array:
    points_px = _points2(points_px, "points_px")
    projected, valid = project_points(points_world, R, t, K)
    if len(projected) != len(points_px):
        raise ValueError("3-D and 2-D arrays must have the same length")
    errors = np.full(len(points_px), np.inf, dtype=np.float64)
    errors[valid] = np.linalg.norm(projected[valid] - points_px[valid], axis=1)
    return errors


def choose_control_points(
    points_world: Array,
    *,
    min_planarity_ratio: float = 1e-8,
) -> tuple[Array, Array, float]:
    """Choose centroid plus three PCA-axis virtual control points.

    Returns control points ``[4,3]``, descending scatter eigenvalues ``[3]``,
    and ``lambda_min/lambda_max``.  The non-planar reconstruction requires a
    non-singular tetrahedron.
    """
    points_world = _points3(points_world, "points_world")
    if len(points_world) < 4:
        raise ValueError("EPnP requires at least four correspondences")
    centroid = np.mean(points_world, axis=0)
    centered = points_world - centroid
    scatter = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    if eigenvalues[0] <= np.finfo(np.float64).eps:
        raise ValueError("3-D points have no spatial extent")
    ratio = float(eigenvalues[-1] / eigenvalues[0])
    if ratio < min_planarity_ratio:
        raise ValueError(
            f"planar/near-planar configuration is outside this reconstruction (ratio={ratio:.3e})"
        )
    control = np.empty((4, 3), dtype=np.float64)
    control[0] = centroid
    for axis in range(3):
        scale = np.sqrt(eigenvalues[axis] / len(points_world))
        control[axis + 1] = centroid + scale * eigenvectors[:, axis]
    return control, eigenvalues, ratio


def barycentric_coordinates(points_world: Array, control_world: Array) -> Array:
    """Express every 3-D point as affine weights of four control points."""
    points_world = _points3(points_world, "points_world")
    control = np.asarray(control_world, dtype=np.float64).reshape(4, 3)
    basis = (control[1:] - control[0]).T
    if np.linalg.cond(basis) > 1e12:
        raise ValueError("control-point tetrahedron is numerically singular")
    tail = np.linalg.solve(basis, (points_world - control[0]).T).T
    alpha = np.column_stack((1.0 - np.sum(tail, axis=1), tail))
    return alpha


def build_projection_matrix(alpha: Array, normalized_points: Array) -> Array:
    """Build EPnP's ``M [C_0^c,...,C_3^c]^T = 0`` matrix."""
    alpha = np.asarray(alpha, dtype=np.float64)
    normalized = _points2(normalized_points, "normalized_points")
    if alpha.shape != (len(normalized), 4):
        raise ValueError("alpha must have shape [N,4]")
    M = np.zeros((2 * len(alpha), 12), dtype=np.float64)
    x, y = normalized.T
    for j in range(4):
        a = alpha[:, j]
        M[0::2, 3 * j] = a
        M[0::2, 3 * j + 2] = -a * x
        M[1::2, 3 * j + 1] = a
        M[1::2, 3 * j + 2] = -a * y
    return M


def distance_constraint_system(
    nullspace_basis: Array,
    control_world: Array,
) -> tuple[Array, Array]:
    """Build the constant-size ``L b = rho`` beta-product system."""
    basis = np.asarray(nullspace_basis, dtype=np.float64).reshape(4, 4, 3)
    control = np.asarray(control_world, dtype=np.float64).reshape(4, 3)
    L = np.empty((6, 10), dtype=np.float64)
    rho = np.empty(6, dtype=np.float64)
    for row, (a, b) in enumerate(CONTROL_PAIRS):
        differences = basis[:, a, :] - basis[:, b, :]
        d0, d1, d2, d3 = differences
        L[row] = (
            np.dot(d0, d0),
            2.0 * np.dot(d0, d1),
            np.dot(d1, d1),
            2.0 * np.dot(d0, d2),
            2.0 * np.dot(d1, d2),
            np.dot(d2, d2),
            2.0 * np.dot(d0, d3),
            2.0 * np.dot(d1, d3),
            2.0 * np.dot(d2, d3),
            np.dot(d3, d3),
        )
        rho[row] = np.sum((control[a] - control[b]) ** 2)
    return L, rho


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-14:
        return 0.0
    return numerator / denominator


def beta_initializations(L: Array, rho: Array) -> tuple[Array, Array, Array]:
    """Three standard EPnP beta approximations before Gauss-Newton."""
    L = np.asarray(L, dtype=np.float64).reshape(6, 10)
    rho = np.asarray(rho, dtype=np.float64).reshape(6)

    b4 = np.linalg.lstsq(L[:, [0, 1, 3, 6]], rho, rcond=None)[0]
    beta1 = np.zeros(4)
    if b4[0] < 0:
        beta1[0] = np.sqrt(-b4[0])
        beta1[1:] = [-_safe_divide(b4[k], beta1[0]) for k in range(1, 4)]
    else:
        beta1[0] = np.sqrt(b4[0])
        beta1[1:] = [_safe_divide(b4[k], beta1[0]) for k in range(1, 4)]

    b3 = np.linalg.lstsq(L[:, :3], rho, rcond=None)[0]
    beta2 = np.zeros(4)
    if b3[0] < 0:
        beta2[0] = np.sqrt(-b3[0])
        beta2[1] = np.sqrt(-b3[2]) if b3[2] < 0 else 0.0
    else:
        beta2[0] = np.sqrt(b3[0])
        beta2[1] = np.sqrt(b3[2]) if b3[2] > 0 else 0.0
    if b3[1] < 0:
        beta2[0] *= -1.0

    b5 = np.linalg.lstsq(L[:, :5], rho, rcond=None)[0]
    beta3 = np.zeros(4)
    if b5[0] < 0:
        beta3[0] = np.sqrt(-b5[0])
        beta3[1] = np.sqrt(-b5[2]) if b5[2] < 0 else 0.0
    else:
        beta3[0] = np.sqrt(b5[0])
        beta3[1] = np.sqrt(b5[2]) if b5[2] > 0 else 0.0
    if b5[1] < 0:
        beta3[0] *= -1.0
    beta3[2] = _safe_divide(b5[3], beta3[0])
    return beta1, beta2, beta3


def _beta_products(beta: Array) -> Array:
    b0, b1, b2, b3 = np.asarray(beta, dtype=np.float64).reshape(4)
    return np.array((
        b0 * b0, b0 * b1, b1 * b1, b0 * b2, b1 * b2,
        b2 * b2, b0 * b3, b1 * b3, b2 * b3, b3 * b3,
    ))


def refine_betas_gauss_newton(
    L: Array,
    rho: Array,
    beta: Array,
    *,
    iterations: int = 5,
) -> Array:
    """Run the constant-size beta Gauss-Newton step used by EPnP."""
    L = np.asarray(L, dtype=np.float64).reshape(6, 10)
    rho = np.asarray(rho, dtype=np.float64).reshape(6)
    beta = np.asarray(beta, dtype=np.float64).reshape(4).copy()
    for _ in range(iterations):
        b0, b1, b2, b3 = beta
        J = np.column_stack((
            2 * L[:, 0] * b0 + L[:, 1] * b1 + L[:, 3] * b2 + L[:, 6] * b3,
            L[:, 1] * b0 + 2 * L[:, 2] * b1 + L[:, 4] * b2 + L[:, 7] * b3,
            L[:, 3] * b0 + L[:, 4] * b1 + 2 * L[:, 5] * b2 + L[:, 8] * b3,
            L[:, 6] * b0 + L[:, 7] * b1 + L[:, 8] * b2 + 2 * L[:, 9] * b3,
        ))
        residual = rho - L @ _beta_products(beta)
        step = np.linalg.lstsq(J, residual, rcond=None)[0]
        beta += step
        if np.linalg.norm(step) < 1e-12:
            break
    return beta


def absolute_orientation(points_world: Array, points_camera: Array) -> tuple[Array, Array]:
    """Rigidly align world points to reconstructed camera-frame points."""
    world = _points3(points_world, "points_world")
    camera = _points3(points_camera, "points_camera")
    if world.shape != camera.shape:
        raise ValueError("point arrays must have the same shape")
    world_center = np.mean(world, axis=0)
    camera_center = np.mean(camera, axis=0)
    H = (camera - camera_center).T @ (world - world_center)
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    t = camera_center - R @ world_center
    return R, t


def _pose_from_beta(
    beta: Array,
    basis: Array,
    alpha: Array,
    points_world: Array,
    points_px: Array,
    K: Array,
    initializer_index: int,
    singular_values: Array,
    control_world: Array,
    scatter_eigenvalues: Array,
    planarity_ratio: float,
) -> PoseEstimate:
    control_camera = np.tensordot(beta, basis, axes=(0, 0))
    points_camera = alpha @ control_camera
    if np.median(points_camera[:, 2]) < 0:
        control_camera *= -1.0
        points_camera *= -1.0
        beta = -beta
    R, t = absolute_orientation(points_world, points_camera)
    errors = reprojection_errors(points_world, points_px, R, t, K)
    finite = np.isfinite(errors)
    rmse = float(np.sqrt(np.mean(errors[finite] ** 2))) if np.any(finite) else np.inf
    _, depth = project_points(points_world, R, t, K)
    diagnostics = EPnPDiagnostics(
        control_points_world=control_world.copy(),
        barycentric_coordinates=alpha.copy(),
        betas=beta.copy(),
        initializer_index=initializer_index,
        design_singular_values=singular_values.copy(),
        scatter_eigenvalues=scatter_eigenvalues.copy(),
        planarity_ratio=planarity_ratio,
    )
    return PoseEstimate(R, t, rmse, float(np.mean(depth)), "EPnP closed-form", diagnostics)


def solve_epnp(
    points_world: Array,
    points_px: Array,
    K: Array,
    *,
    min_planarity_ratio: float = 1e-8,
) -> PoseEstimate:
    """Solve the educational non-planar EPnP path from six or more matches."""
    world = _points3(points_world, "points_world")
    pixels = _points2(points_px, "points_px")
    if len(world) != len(pixels) or len(world) < 6:
        raise ValueError(
            "this educational non-minimal EPnP path needs at least six "
            "correspondences"
        )
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    normalized = pixels_to_normalized(pixels, K)
    control, scatter_eigenvalues, planarity_ratio = choose_control_points(
        world, min_planarity_ratio=min_planarity_ratio
    )
    alpha = barycentric_coordinates(world, control)
    M = build_projection_matrix(alpha, normalized)
    _, singular_values, Vt = np.linalg.svd(M, full_matrices=True)
    # Smallest first, matching beta_0 ... beta_3 in the distance system.
    basis = np.stack((Vt[-1], Vt[-2], Vt[-3], Vt[-4])).reshape(4, 4, 3)
    L, rho = distance_constraint_system(basis, control)
    initializations = beta_initializations(L, rho)
    candidates = []
    for index, initial in enumerate(initializations, start=1):
        beta = refine_betas_gauss_newton(L, rho, initial)
        candidate = _pose_from_beta(
            beta, basis, alpha, world, pixels, K, index, singular_values,
            control, scatter_eigenvalues, planarity_ratio,
        )
        candidates.append(candidate)
    return min(candidates, key=lambda pose: (-pose.positive_depth_fraction, pose.reprojection_rmse_px))


def refine_pose(
    points_world: Array,
    points_px: Array,
    K: Array,
    initial_pose: PoseEstimate,
    *,
    robust_loss: str = "linear",
    loss_scale_px: float = 2.0,
    max_nfev: int = 100,
) -> PoseEstimate:
    """Refine one pose by pixel reprojection residuals using SciPy."""
    world = _points3(points_world, "points_world")
    pixels = _points2(points_px, "points_px")
    if world.shape[0] != pixels.shape[0]:
        raise ValueError("point arrays must have the same length")
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    x0 = np.r_[Rotation.from_matrix(initial_pose.R).as_rotvec(), initial_pose.t]

    def residual(parameters: Array) -> Array:
        R = Rotation.from_rotvec(parameters[:3]).as_matrix()
        projected, valid = project_points(world, R, parameters[3:], K)
        value = np.full((len(world), 2), 1e3, dtype=np.float64)
        value[valid] = projected[valid] - pixels[valid]
        return value.ravel()

    result = least_squares(
        residual, x0, method="trf", loss=robust_loss, f_scale=loss_scale_px,
        max_nfev=max_nfev, x_scale="jac",
    )
    R = Rotation.from_rotvec(result.x[:3]).as_matrix()
    t = result.x[3:]
    errors = reprojection_errors(world, pixels, R, t, K)
    finite = np.isfinite(errors)
    rmse = float(np.sqrt(np.mean(errors[finite] ** 2))) if np.any(finite) else np.inf
    _, depth = project_points(world, R, t, K)
    singular = np.linalg.svd(result.jac, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
    return PoseEstimate(
        R, t, rmse, float(np.mean(depth)),
        f"reprojection refinement ({robust_loss})",
        initial_pose.diagnostics, int(result.nfev), condition,
    )


def _required_draws(confidence: float, inlier_ratio: float, sample_size: int) -> int:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between zero and one")
    if inlier_ratio <= 0:
        return np.iinfo(np.int32).max
    if inlier_ratio >= 1:
        return 1
    success = inlier_ratio ** sample_size
    if success <= np.finfo(np.float64).eps:
        return np.iinfo(np.int32).max
    return max(1, ceil(log(1.0 - confidence) / log(1.0 - success)))


def ransac_epnp(
    points_world: Array,
    points_px: Array,
    K: Array,
    *,
    threshold_px: float = 3.0,
    confidence: float = 0.999,
    sample_size: int = 6,
    min_draws: int = 64,
    max_draws: int = 3000,
    seed: int = 0,
    local_rounds: int = 3,
) -> RansacPose:
    """Educational EPnP RANSAC with MSAC scoring and robust local refinement."""
    world = _points3(points_world, "points_world")
    pixels = _points2(points_px, "points_px")
    if len(world) != len(pixels) or len(world) < sample_size or sample_size < 6:
        raise ValueError("invalid correspondence count or sample size")
    if threshold_px <= 0 or min_draws < 1 or max_draws < min_draws:
        raise ValueError("invalid robust-estimation controls")
    rng = np.random.default_rng(seed)
    threshold2 = threshold_px * threshold_px
    best_pose = None
    best_mask = np.zeros(len(world), dtype=bool)
    best_score = (-np.inf, 0)
    adaptive_limit = max_draws
    valid_hypotheses = 0
    draws = 0

    def evaluate(pose: PoseEstimate):
        errors = reprojection_errors(world, pixels, pose.R, pose.t, K)
        squared = errors * errors
        mask = squared <= threshold2
        capped = float(np.sum(np.minimum(squared, threshold2)))
        score = (-capped, int(np.count_nonzero(mask)))
        return score, mask, errors

    for draws in range(1, max_draws + 1):
        sample = rng.choice(len(world), size=sample_size, replace=False)
        try:
            pose = solve_epnp(world[sample], pixels[sample], K)
        except (ValueError, np.linalg.LinAlgError):
            continue
        valid_hypotheses += 1
        score, mask, _ = evaluate(pose)
        if score > best_score:
            best_pose, best_mask, best_score = pose, mask, score
            ratio = np.count_nonzero(mask) / len(mask)
            adaptive_limit = min(
                adaptive_limit,
                max(min_draws, _required_draws(confidence, ratio, sample_size)),
            )
        if draws >= adaptive_limit and draws >= min_draws:
            break
    if best_pose is None:
        raise RuntimeError("RANSAC produced no valid EPnP hypothesis")

    for _ in range(local_rounds):
        if np.count_nonzero(best_mask) < 6:
            break
        try:
            closed = solve_epnp(world[best_mask], pixels[best_mask], K)
            candidate = refine_pose(
                world[best_mask], pixels[best_mask], K, closed,
                robust_loss="huber", loss_scale_px=threshold_px,
            )
        except (ValueError, np.linalg.LinAlgError):
            break
        score, mask, _ = evaluate(candidate)
        if score <= best_score:
            break
        best_pose, best_mask, best_score = candidate, mask, score

    errors = reprojection_errors(world, pixels, best_pose.R, best_pose.t, K)
    final_mask = errors <= threshold_px
    return RansacPose(best_pose, final_mask, errors, draws, valid_hypotheses)


def rotation_error_degrees(R_estimate: Array, R_true: Array) -> float:
    delta = np.asarray(R_estimate).reshape(3, 3) @ np.asarray(R_true).reshape(3, 3).T
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def translation_error(t_estimate: Array, t_true: Array) -> float:
    return float(np.linalg.norm(np.asarray(t_estimate).reshape(3) - np.asarray(t_true).reshape(3)))


def camera_center_world(R: Array, t: Array) -> Array:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return -R.T @ t


def rotation_xyz(degrees_xyz: tuple[float, float, float]) -> Array:
    return Rotation.from_euler("xyz", degrees_xyz, degrees=True).as_matrix()

