"""Inspectable bundle adjustment with analytic Jacobians and Schur elimination.

Conventions
-----------
Camera poses map world points to camera coordinates:
``X_camera = R @ X_world + t``. Pixels use ``(u, v)`` ordering. Pose updates
are left-multiplicative SE(3) increments:
``R_new = Exp(domega) @ R`` and ``t_new = Exp(domega) @ t + dv``.

Scope
-----
This is an educational small/medium-problem solver. It uses dense storage for
the reduced camera system while preserving the landmark-block Schur algebra.
Production systems should use Ceres, g2o, GTSAM, COLMAP, RootBA, or another
tested sparse solver with explicit manifold, ordering, and gauge handling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class BAProblem:
    K: Array
    rotations: Array
    translations: Array
    points: Array
    camera_indices: Array
    point_indices: Array
    pixels: Array


@dataclass(frozen=True)
class BlockLinearization:
    camera_blocks: Array
    point_blocks: Array
    cross_blocks: dict[tuple[int, int], Array]
    camera_gradient: Array
    point_gradient: Array
    variable_cameras: Array
    variable_points: Array
    robust_cost: float
    pixel_rmse: float
    residuals: Array
    observation_weights: Array


@dataclass(frozen=True)
class BAResult:
    rotations: Array
    translations: Array
    points: Array
    initial_cost: float
    final_cost: float
    initial_rmse_px: float
    final_rmse_px: float
    cost_history: tuple[float, ...]
    damping_history: tuple[float, ...]
    accepted_steps: int
    rejected_steps: int
    iterations: int
    converged: bool
    loss: str
    fixed_camera: int | None
    fixed_point: int | None


def skew(vector: Array) -> Array:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(rotation_vector: Array) -> Array:
    """Rodrigues exponential map with a stable small-angle branch."""
    omega = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    Omega = skew(omega)
    if theta < 1e-8:
        return np.eye(3) + Omega + 0.5 * (Omega @ Omega)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * Omega + b * (Omega @ Omega)


def _validate_problem(problem: BAProblem) -> BAProblem:
    K = np.asarray(problem.K, dtype=np.float64).reshape(3, 3)
    rotations = np.asarray(problem.rotations, dtype=np.float64)
    translations = np.asarray(problem.translations, dtype=np.float64)
    points = np.asarray(problem.points, dtype=np.float64)
    camera_indices = np.asarray(problem.camera_indices, dtype=np.int64).reshape(-1)
    point_indices = np.asarray(problem.point_indices, dtype=np.int64).reshape(-1)
    pixels = np.asarray(problem.pixels, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape [C,3,3]")
    if translations.shape != (len(rotations), 3):
        raise ValueError("translations must have shape [C,3]")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [P,3]")
    if pixels.shape != (len(camera_indices), 2) or len(point_indices) != len(pixels):
        raise ValueError("observation arrays must have matching [M] / [M,2] shapes")
    if np.any(camera_indices < 0) or np.any(camera_indices >= len(rotations)):
        raise ValueError("camera index out of range")
    if np.any(point_indices < 0) or np.any(point_indices >= len(points)):
        raise ValueError("point index out of range")
    arrays = (K, rotations, translations, points, pixels)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("problem arrays must be finite")
    return BAProblem(K, rotations, translations, points, camera_indices, point_indices, pixels)


def project_observations(
    K: Array,
    rotations: Array,
    translations: Array,
    points: Array,
    camera_indices: Array,
    point_indices: Array,
) -> tuple[Array, Array]:
    """Project sparse observations; returns pixels [M,2] and depths [M]."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    camera_indices = np.asarray(camera_indices, dtype=np.int64)
    point_indices = np.asarray(point_indices, dtype=np.int64)
    world = points[point_indices]
    camera = np.einsum("mij,mj->mi", rotations[camera_indices], world)
    camera += translations[camera_indices]
    depths = camera[:, 2].copy()
    homogeneous = (K @ camera.T).T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, depths


def projection_jacobian(K: Array, point_camera: Array) -> Array:
    """Derivative of pixel coordinates with respect to camera XYZ."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    x, y, z = np.asarray(point_camera, dtype=np.float64).reshape(3)
    if z <= 1e-9:
        raise ValueError("projection Jacobian requires positive depth")
    fx, fy = K[0, 0], K[1, 1]
    return np.array([
        [fx / z, 0.0, -fx * x / (z * z)],
        [0.0, fy / z, -fy * y / (z * z)],
    ])


def observation_jacobians(K: Array, R: Array, t: Array, point: Array) -> tuple[Array, Array, Array]:
    """Return predicted pixel, pose Jacobian [2,6], and point Jacobian [2,3]."""
    point_camera = np.asarray(R) @ np.asarray(point) + np.asarray(t)
    J_projection = projection_jacobian(K, point_camera)
    J_pose = J_projection @ np.column_stack((-skew(point_camera), np.eye(3)))
    J_point = J_projection @ np.asarray(R)
    homogeneous = np.asarray(K) @ point_camera
    pixel = homogeneous[:2] / homogeneous[2]
    return pixel, J_pose, J_point


def huber_cost_and_weight(residual: Array, delta_px: float) -> tuple[float, float]:
    """Isotropic 2-D Huber cost and IRLS weight for one observation."""
    norm = float(np.linalg.norm(residual))
    if norm <= delta_px:
        return 0.5 * norm * norm, 1.0
    return delta_px * (norm - 0.5 * delta_px), delta_px / max(norm, 1e-15)


def evaluate(
    problem: BAProblem,
    rotations: Array,
    translations: Array,
    points: Array,
    *,
    loss: str = "huber",
    huber_delta_px: float = 2.5,
) -> tuple[float, float, Array, Array]:
    predicted, depths = project_observations(
        problem.K, rotations, translations, points,
        problem.camera_indices, problem.point_indices,
    )
    if np.any(depths <= 1e-9) or not np.all(np.isfinite(predicted)):
        return np.inf, np.inf, predicted - problem.pixels, np.zeros(len(problem.pixels))
    residuals = predicted - problem.pixels
    squared = np.sum(residuals * residuals, axis=1)
    rmse = float(np.sqrt(np.mean(squared)))
    if loss == "linear":
        return 0.5 * float(np.sum(squared)), rmse, residuals, np.ones(len(residuals))
    if loss != "huber" or huber_delta_px <= 0:
        raise ValueError("loss must be 'linear' or 'huber' with positive delta")
    costs = np.empty(len(residuals))
    weights = np.empty(len(residuals))
    for index, residual in enumerate(residuals):
        costs[index], weights[index] = huber_cost_and_weight(residual, huber_delta_px)
    return float(np.sum(costs)), rmse, residuals, weights


def linearize(
    problem: BAProblem,
    rotations: Array,
    translations: Array,
    points: Array,
    *,
    fixed_camera: int | None = 0,
    fixed_point: int | None = 0,
    optimize_cameras: bool = True,
    optimize_points: bool = True,
    loss: str = "huber",
    huber_delta_px: float = 2.5,
) -> BlockLinearization:
    """Build block normal equations at the current state."""
    problem = _validate_problem(problem)
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if optimize_cameras and optimize_points and (fixed_camera is None or fixed_point is None):
        raise ValueError("joint monocular BA needs explicit camera and scale anchors")
    variable_cameras = np.array([
        index for index in range(len(rotations))
        if optimize_cameras and index != fixed_camera
    ], dtype=np.int64)
    variable_points = np.array([
        index for index in range(len(points))
        if optimize_points and index != fixed_point
    ], dtype=np.int64)
    camera_lookup = {int(value): index for index, value in enumerate(variable_cameras)}
    point_lookup = {int(value): index for index, value in enumerate(variable_points)}
    U = np.zeros((len(variable_cameras), 6, 6))
    V = np.zeros((len(variable_points), 3, 3))
    gc = np.zeros((len(variable_cameras), 6))
    gp = np.zeros((len(variable_points), 3))
    W: dict[tuple[int, int], Array] = {}

    cost, rmse, residuals, weights = evaluate(
        problem, rotations, translations, points,
        loss=loss, huber_delta_px=huber_delta_px,
    )
    if not np.isfinite(cost):
        raise ValueError("current state projects one or more observations behind the camera")

    for observation, (camera_id, point_id) in enumerate(zip(
        problem.camera_indices, problem.point_indices
    )):
        _, J_camera, J_point = observation_jacobians(
            problem.K, rotations[camera_id], translations[camera_id], points[point_id]
        )
        sqrt_weight = np.sqrt(weights[observation])
        residual = sqrt_weight * residuals[observation]
        J_camera = sqrt_weight * J_camera
        J_point = sqrt_weight * J_point
        camera_variable = camera_lookup.get(int(camera_id))
        point_variable = point_lookup.get(int(point_id))
        if camera_variable is not None:
            U[camera_variable] += J_camera.T @ J_camera
            gc[camera_variable] += J_camera.T @ residual
        if point_variable is not None:
            V[point_variable] += J_point.T @ J_point
            gp[point_variable] += J_point.T @ residual
        if camera_variable is not None and point_variable is not None:
            key = (camera_variable, point_variable)
            W[key] = W.get(key, np.zeros((6, 3))) + J_camera.T @ J_point
    return BlockLinearization(
        U, V, W, gc, gp, variable_cameras, variable_points,
        cost, rmse, residuals, weights,
    )


def _damped_blocks(blocks: Array, damping: float) -> Array:
    result = blocks.copy()
    for index in range(len(result)):
        diagonal = np.maximum(np.diag(result[index]), 1e-12)
        result[index] += damping * np.diag(diagonal)
    return result


def schur_step(system: BlockLinearization, damping: float) -> tuple[Array, Array, Array]:
    """Solve a damped block system by eliminating 3-D point increments."""
    if damping < 0:
        raise ValueError("damping must be nonnegative")
    U = _damped_blocks(system.camera_blocks, damping)
    V = _damped_blocks(system.point_blocks, damping)
    camera_count, point_count = len(U), len(V)
    dc = np.zeros((camera_count, 6))
    dp = np.zeros((point_count, 3))

    if camera_count == 0:
        for point in range(point_count):
            dp[point] = np.linalg.solve(V[point], -system.point_gradient[point])
        return dc, dp, np.empty((0, 0))
    if point_count == 0:
        for camera in range(camera_count):
            dc[camera] = np.linalg.solve(U[camera], -system.camera_gradient[camera])
        return dc, dp, np.zeros((6 * camera_count, 6 * camera_count))

    size = 6 * camera_count
    reduced = np.zeros((size, size))
    rhs = -system.camera_gradient.reshape(-1).copy()
    for camera in range(camera_count):
        block = slice(6 * camera, 6 * camera + 6)
        reduced[block, block] += U[camera]

    point_connections: list[list[tuple[int, Array]]] = [[] for _ in range(point_count)]
    for (camera, point), cross in system.cross_blocks.items():
        point_connections[point].append((camera, cross))
    inverse_V = [np.linalg.inv(V[point]) for point in range(point_count)]
    for point, connections in enumerate(point_connections):
        if not connections:
            continue
        inv = inverse_V[point]
        for camera_a, cross_a in connections:
            block_a = slice(6 * camera_a, 6 * camera_a + 6)
            rhs[block_a] += cross_a @ inv @ system.point_gradient[point]
            for camera_b, cross_b in connections:
                block_b = slice(6 * camera_b, 6 * camera_b + 6)
                reduced[block_a, block_b] -= cross_a @ inv @ cross_b.T
    dc[:] = np.linalg.solve(reduced, rhs).reshape(camera_count, 6)

    for point, connections in enumerate(point_connections):
        right = system.point_gradient[point].copy()
        for camera, cross in connections:
            right += cross.T @ dc[camera]
        dp[point] = np.linalg.solve(V[point], -right)
    return dc, dp, reduced


def full_normal_step(system: BlockLinearization, damping: float) -> tuple[Array, Array, Array]:
    """Dense full-system solve used only to validate Schur equivalence."""
    U = _damped_blocks(system.camera_blocks, damping)
    V = _damped_blocks(system.point_blocks, damping)
    camera_count, point_count = len(U), len(V)
    camera_size = 6 * camera_count
    point_size = 3 * point_count
    H = np.zeros((camera_size + point_size, camera_size + point_size))
    g = np.r_[system.camera_gradient.reshape(-1), system.point_gradient.reshape(-1)]
    for camera in range(camera_count):
        block = slice(6 * camera, 6 * camera + 6)
        H[block, block] = U[camera]
    for point in range(point_count):
        block = slice(camera_size + 3 * point, camera_size + 3 * point + 3)
        H[block, block] = V[point]
    for (camera, point), cross in system.cross_blocks.items():
        cblock = slice(6 * camera, 6 * camera + 6)
        pblock = slice(camera_size + 3 * point, camera_size + 3 * point + 3)
        H[cblock, pblock] += cross
        H[pblock, cblock] += cross.T
    step = np.linalg.solve(H, -g)
    return step[:camera_size].reshape(camera_count, 6), step[camera_size:].reshape(point_count, 3), H


def apply_increment(
    rotations: Array,
    translations: Array,
    points: Array,
    system: BlockLinearization,
    camera_step: Array,
    point_step: Array,
) -> tuple[Array, Array, Array]:
    new_rotations = np.asarray(rotations, dtype=np.float64).copy()
    new_translations = np.asarray(translations, dtype=np.float64).copy()
    new_points = np.asarray(points, dtype=np.float64).copy()
    for variable, camera in enumerate(system.variable_cameras):
        delta_rotation = so3_exp(camera_step[variable, :3])
        new_rotations[camera] = delta_rotation @ new_rotations[camera]
        new_translations[camera] = (
            delta_rotation @ new_translations[camera] + camera_step[variable, 3:]
        )
    for variable, point in enumerate(system.variable_points):
        new_points[point] += point_step[variable]
    return new_rotations, new_translations, new_points


def solve_bundle_adjustment(
    problem: BAProblem,
    *,
    fixed_camera: int | None = 0,
    fixed_point: int | None = 0,
    optimize_cameras: bool = True,
    optimize_points: bool = True,
    loss: str = "huber",
    huber_delta_px: float = 2.5,
    max_iterations: int = 30,
    initial_damping: float = 1e-3,
    step_tolerance: float = 1e-8,
) -> BAResult:
    """Levenberg–Marquardt loop using a landmark Schur complement."""
    problem = _validate_problem(problem)
    if max_iterations < 1 or initial_damping <= 0 or step_tolerance <= 0:
        raise ValueError("invalid solver controls")
    rotations = problem.rotations.copy()
    translations = problem.translations.copy()
    points = problem.points.copy()
    initial_cost, initial_rmse, _, _ = evaluate(
        problem, rotations, translations, points,
        loss=loss, huber_delta_px=huber_delta_px,
    )
    current_cost = initial_cost
    damping = initial_damping
    costs = [current_cost]
    dampings = [damping]
    accepted = 0
    rejected = 0
    converged = False

    for iteration in range(1, max_iterations + 1):
        system = linearize(
            problem, rotations, translations, points,
            fixed_camera=fixed_camera, fixed_point=fixed_point,
            optimize_cameras=optimize_cameras, optimize_points=optimize_points,
            loss=loss, huber_delta_px=huber_delta_px,
        )
        camera_step, point_step, _ = schur_step(system, damping)
        step_norm = float(np.sqrt(np.sum(camera_step ** 2) + np.sum(point_step ** 2)))
        if step_norm < step_tolerance:
            converged = True
            break
        candidate = apply_increment(
            rotations, translations, points, system, camera_step, point_step
        )
        candidate_cost, _, _, _ = evaluate(
            problem, *candidate, loss=loss, huber_delta_px=huber_delta_px
        )
        if candidate_cost < current_cost:
            rotations, translations, points = candidate
            relative = (current_cost - candidate_cost) / max(1.0, current_cost)
            current_cost = candidate_cost
            damping = max(1e-12, damping / 3.0)
            accepted += 1
            costs.append(current_cost)
            dampings.append(damping)
            if relative < 1e-12:
                converged = True
                break
        else:
            damping = min(1e12, damping * 10.0)
            rejected += 1
            costs.append(current_cost)
            dampings.append(damping)
    else:
        iteration = max_iterations

    final_cost, final_rmse, _, _ = evaluate(
        problem, rotations, translations, points,
        loss=loss, huber_delta_px=huber_delta_px,
    )
    return BAResult(
        rotations, translations, points, initial_cost, final_cost,
        initial_rmse, final_rmse, tuple(costs), tuple(dampings),
        accepted, rejected, iteration, converged, loss, fixed_camera, fixed_point,
    )


def camera_centers(rotations: Array, translations: Array) -> Array:
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    return -np.einsum("cij,cj->ci", np.swapaxes(rotations, 1, 2), translations)


def rotation_error_degrees(estimate: Array, truth: Array) -> Array:
    relative = np.einsum("cij,ckj->cik", np.asarray(estimate), np.asarray(truth))
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))
