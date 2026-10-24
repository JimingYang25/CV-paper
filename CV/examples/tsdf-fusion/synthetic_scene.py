"""Deterministic analytic depth scenes used only for validation and teaching."""

from __future__ import annotations

import numpy as np

from tsdf_fusion import PinholeCamera


def camera_pose(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [tx, ty, tz]
    return pose


def pixel_rays(camera: PinholeCamera) -> tuple[np.ndarray, np.ndarray]:
    """Return camera ray x/z and y/z ratios as ``[H,W]`` arrays."""
    v, u = np.indices((camera.height, camera.width), dtype=np.float64)
    return (u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy


def render_sloped_plane(
    camera: PinholeCamera,
    world_from_camera: np.ndarray,
    *,
    z0: float = 1.2,
    slope_x: float = 0.15,
    slope_y: float = -0.08,
) -> np.ndarray:
    """Render ``z_world = z0 + slope_x*x + slope_y*y`` as optical-axis depth."""
    pose = np.asarray(world_from_camera, dtype=np.float64)
    if not np.allclose(pose[:3, :3], np.eye(3), atol=1e-12):
        raise ValueError("this analytic renderer supports translated, axis-aligned cameras")
    rx, ry = pixel_rays(camera)
    tx, ty, tz = pose[:3, 3]
    denominator = 1.0 - slope_x * rx - slope_y * ry
    depth = (z0 + slope_x * tx + slope_y * ty - tz) / denominator
    return np.where(depth > 0, depth, np.nan)


def corrupt_depth(
    clean_depth: np.ndarray,
    *,
    seed: int,
    noise_std: float = 0.012,
    hole_fraction: float = 0.04,
    outlier_fraction: float = 0.025,
    outlier_magnitude: float = 0.22,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add noise, missing values, and tagged outliers.

    Returns ``(depth, confidence, outlier_mask)``.  Confidence is an explicit
    synthetic sensor-quality channel: holes get 0 and tagged outliers get 0.05.
    """
    rng = np.random.default_rng(seed)
    depth = np.asarray(clean_depth, dtype=np.float64).copy()
    finite = np.isfinite(depth)
    depth[finite] += rng.normal(0.0, noise_std, size=int(np.count_nonzero(finite)))
    random_field = rng.random(depth.shape)
    holes = finite & (random_field < hole_fraction)
    outliers = finite & ~holes & (
        random_field >= hole_fraction
    ) & (random_field < hole_fraction + outlier_fraction)
    signs = rng.choice(np.array([-1.0, 1.0]), size=int(np.count_nonzero(outliers)))
    depth[outliers] += outlier_magnitude * signs
    depth[holes] = np.nan
    confidence = np.ones_like(depth)
    confidence[holes] = 0.0
    confidence[outliers] = 0.05
    return depth, confidence, outliers


def plane_truth_on_grid(
    x_world: np.ndarray,
    y_world: np.ndarray,
    *,
    z0: float = 1.2,
    slope_x: float = 0.15,
    slope_y: float = -0.08,
) -> np.ndarray:
    return z0 + slope_x * x_world + slope_y * y_world


def render_table_with_box(
    camera: PinholeCamera,
    world_from_camera: np.ndarray,
    *,
    table_z: float = 1.30,
    box_front_z: float = 0.88,
    box_half_x: float = 0.13,
    box_half_y: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a fronto-parallel table and a nearer rectangular obstacle."""
    pose = np.asarray(world_from_camera, dtype=np.float64)
    if not np.allclose(pose[:3, :3], np.eye(3), atol=1e-12):
        raise ValueError("this analytic renderer supports translated, axis-aligned cameras")
    rx, ry = pixel_rays(camera)
    tx, ty, tz = pose[:3, 3]
    table_depth = table_z - tz
    box_depth = box_front_z - tz
    box_x = tx + box_depth * rx
    box_y = ty + box_depth * ry
    sees_box = (np.abs(box_x) <= box_half_x) & (np.abs(box_y) <= box_half_y)
    depth = np.full((camera.height, camera.width), table_depth, dtype=np.float64)
    depth[sees_box] = box_depth
    return depth, sees_box
