"""Deterministic synthetic dense features resembling tactile/scene patch structure."""

from __future__ import annotations

import numpy as np

from dino_gram import l2_normalize


def make_dense_case(
    seed: int = 0,
    *,
    batch: int = 6,
    grid_size: int = 8,
    feature_dim: int = 12,
    outlier_fraction: float = 0.12,
) -> dict:
    """Return early-anchor and degraded patch features with known spatial structure."""
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, grid_size),
        np.linspace(-1.0, 1.0, grid_size),
        indexing="ij",
    )
    x, y = x.ravel(), y.ravel()
    P = grid_size * grid_size
    base = []
    marker_masks = []
    for b in range(batch):
        angle = 0.15 * np.sin(0.7 * b)
        ca, sa = np.cos(angle), np.sin(angle)
        xr, yr = ca * x - sa * y, sa * x + ca * y
        cx, cy = 0.28 * np.sin(0.8 * b), 0.25 * np.cos(0.6 * b)
        marker = np.exp(-((xr - cx) ** 2 + (yr - cy) ** 2) / 0.10)
        ridge = np.exp(-((yr + 0.28 * np.sin(2.2 * xr + 0.3 * b)) ** 2) / 0.06)
        basis = np.stack(
            [
                np.ones(P), xr, yr, xr * yr, xr * xr, yr * yr,
                marker, ridge, np.sin(np.pi * xr), np.cos(np.pi * yr),
            ],
            axis=-1,
        )
        base.append(basis)
        marker_masks.append(marker > 0.45)
    basis = np.stack(base, axis=0)  # [B,P,10]
    marker_masks = np.stack(marker_masks, axis=0)

    projection = rng.normal(size=(basis.shape[-1], feature_dim))
    anchor = l2_normalize(basis @ projection + 0.025 * rng.normal(size=(batch, P, feature_dim)))

    # A common direction and patch outliers mimic degraded local consistency.
    q, _ = np.linalg.qr(rng.normal(size=(feature_dim, feature_dim)))
    common = l2_normalize(rng.normal(size=(1, 1, feature_dim)))
    degraded = 0.62 * (anchor @ q) + 0.45 * common + 0.22 * rng.normal(size=anchor.shape)
    outlier_count = int(round(outlier_fraction * batch * P))
    flat_indices = rng.choice(batch * P, size=outlier_count, replace=False)
    degraded.reshape(batch * P, feature_dim)[flat_indices] = rng.normal(size=(outlier_count, feature_dim))
    degraded = l2_normalize(degraded)
    return {
        "anchor_features": anchor,
        "degraded_features": degraded,
        "marker_masks": marker_masks,
        "grid_size": grid_size,
        "query_index": (grid_size // 2) * grid_size + grid_size // 2,
    }


def shuffled_anchor(anchor_features: np.ndarray, seed: int = 123) -> np.ndarray:
    """Create a plausible but spatially wrong anchor by permuting patch positions."""
    rng = np.random.default_rng(seed)
    shuffled = np.array(anchor_features, copy=True)
    permutation = rng.permutation(shuffled.shape[1])
    return shuffled[:, permutation]
