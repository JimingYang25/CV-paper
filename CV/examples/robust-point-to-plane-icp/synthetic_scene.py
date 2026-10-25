"""Deterministic synthetic geometry for robust ICP experiments."""

from __future__ import annotations

import numpy as np

from robust_icp import exp_se3, invert_transform, transform_normals, transform_points


def _grid(a0: float, a1: float, na: int, b0: float, b1: float, nb: int) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.meshgrid(np.linspace(a0, a1, na), np.linspace(b0, b1, nb), indexing="xy")
    return a.ravel(), b.ravel()


def make_corner_model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three structured surfaces; returns points, normals, integer surface IDs."""
    # Floor z = gently varying height. Analytic normals break repeated-grid symmetry.
    x, y = _grid(-0.65, 0.65, 29, -0.50, 0.52, 23)
    z = 0.018 * np.sin(5.0 * x) * np.cos(4.0 * y)
    dzdx = 0.09 * np.cos(5.0 * x) * np.cos(4.0 * y)
    dzdy = -0.072 * np.sin(5.0 * x) * np.sin(4.0 * y)
    p0 = np.c_[x, y, z]
    n0 = np.c_[-dzdx, -dzdy, np.ones_like(x)]

    # Back wall y ~= 0.52, facing into the workspace.
    x, z = _grid(-0.65, 0.65, 27, 0.02, 0.72, 17)
    y = 0.52 + 0.012 * np.sin(4.0 * x + 2.0 * z)
    dydx = 0.048 * np.cos(4.0 * x + 2.0 * z)
    dydz = 0.024 * np.cos(4.0 * x + 2.0 * z)
    p1 = np.c_[x, y, z]
    n1 = np.c_[dydx, -np.ones_like(x), dydz]

    # Side wall x ~= 0.65, with different curvature.
    y, z = _grid(-0.50, 0.52, 23, 0.03, 0.72, 17)
    x = 0.65 + 0.010 * np.cos(3.0 * y - 5.0 * z)
    dxdy = -0.030 * np.sin(3.0 * y - 5.0 * z)
    dxdz = 0.050 * np.sin(3.0 * y - 5.0 * z)
    p2 = np.c_[x, y, z]
    n2 = np.c_[-np.ones_like(y), dxdy, dxdz]

    points = np.vstack([p0, p1, p2]).astype(np.float64)
    normals = np.vstack([n0, n1, n2]).astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    ids = np.concatenate([
        np.zeros(len(p0), dtype=np.int64),
        np.ones(len(p1), dtype=np.int64),
        np.full(len(p2), 2, dtype=np.int64),
    ])
    return points, normals, ids


def make_planar_model(n: int = 31) -> tuple[np.ndarray, np.ndarray]:
    """A single z=0 plane, intentionally rank-deficient for point-to-plane ICP."""
    x, y = _grid(-0.7, 0.7, n, -0.6, 0.6, n)
    points = np.c_[x, y, np.zeros_like(x)]
    normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(points), 1))
    return points, normals


def make_registration_case(
    *, seed: int = 7, outlier_fraction: float = 0.15, noise_sigma: float = 0.0015
) -> dict:
    """Create a target model and a noisy source scan with known target<-source pose."""
    rng = np.random.default_rng(seed)
    target, target_normals, surface_ids = make_corner_model()
    # Keep a partial, nonuniform view to resemble a depth scan.
    visible = (target[:, 0] > -0.52) & (target[:, 1] > -0.40)
    indices = np.flatnonzero(visible)[::2]
    target_sample = target[indices]
    normal_sample = target_normals[indices]

    T_true = exp_se3(np.array([0.065, -0.045, 0.055, 0.045, -0.028, 0.038]))
    T_source_from_target = invert_transform(T_true)
    source = transform_points(T_source_from_target, target_sample)
    source_normals = transform_normals(T_source_from_target, normal_sample)
    source += rng.normal(scale=noise_sigma, size=source.shape)

    outlier_count = int(round(outlier_fraction * len(source)))
    if outlier_count:
        outliers = rng.uniform([-0.75, -0.58, -0.12], [0.78, 0.67, 0.82], size=(outlier_count, 3))
        outlier_normals = rng.normal(size=(outlier_count, 3))
        outlier_normals /= np.linalg.norm(outlier_normals, axis=1, keepdims=True)
        source = np.vstack([source, outliers])
        source_normals = np.vstack([source_normals, outlier_normals])

    initial = exp_se3(np.array([0.020, -0.012, 0.015, 0.012, -0.008, 0.010]))
    return {
        "source_points": source,
        "source_normals": source_normals,
        "target_points": target,
        "target_normals": target_normals,
        "target_surface_ids": surface_ids,
        "T_true": T_true,
        "T_initial": initial,
        "inlier_count": len(target_sample),
    }
