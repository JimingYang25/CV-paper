"""Small, inspectable NumPy reference for projective TSDF fusion.

Conventions
-----------
* Images are ``[H, W]`` arrays.
* Depth is camera optical-axis ``z`` in metres, not Euclidean ray length.
* ``world_from_camera`` maps homogeneous camera points into the world frame.
* A positive TSDF value is in observed free space, in front of a surface.
* Volume arrays use ``[z, y, x]`` storage; metric points use ``[x, y, z]``.

This is educational dense-grid code.  It intentionally exposes every transform
and update and is not a replacement for Open3D, voxblox, or NVIDIA nvblox.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    """Pinhole intrinsics for an image of fixed size."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def scaled(self, scale_x: float, scale_y: float) -> "PinholeCamera":
        """Return intrinsics for a resize using pixel-centre coordinates."""
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError("resize scales must be positive")
        return PinholeCamera(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=(self.cx + 0.5) * scale_x - 0.5,
            cy=(self.cy + 0.5) * scale_y - 0.5,
            width=int(round(self.width * scale_x)),
            height=int(round(self.height * scale_y)),
        )


@dataclass(frozen=True)
class VolumeConfig:
    """Metric bounds and fusion settings for a dense volume."""

    origin_world: tuple[float, float, float]
    dimensions_xyz: tuple[int, int, int]
    voxel_size: float
    truncation: float
    max_weight: float = 64.0

    def __post_init__(self) -> None:
        if len(self.origin_world) != 3 or len(self.dimensions_xyz) != 3:
            raise ValueError("origin and dimensions must have three elements")
        if any(n <= 1 for n in self.dimensions_xyz):
            raise ValueError("every volume dimension must exceed one")
        if self.voxel_size <= 0 or self.truncation <= 0:
            raise ValueError("voxel size and truncation must be positive")
        if self.truncation < self.voxel_size:
            raise ValueError("truncation must be at least one voxel")
        if self.max_weight <= 0:
            raise ValueError("max_weight must be positive")


@dataclass(frozen=True)
class IntegrationStats:
    projected_voxels: int
    updated_voxels: int
    invalid_depth_voxels: int
    behind_band_voxels: int


@dataclass(frozen=True)
class SurfaceMap:
    """First positive-to-nonpositive zero crossing along world +z."""

    x_world: np.ndarray  # [Ny, Nx]
    y_world: np.ndarray  # [Ny, Nx]
    z_world: np.ndarray  # [Ny, Nx], NaN where not extracted
    support_weight: np.ndarray  # [Ny, Nx]

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.z_world)


def _validate_pose(world_from_camera: np.ndarray) -> np.ndarray:
    pose = np.asarray(world_from_camera, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("world_from_camera must have shape [4, 4]")
    if not np.all(np.isfinite(pose)):
        raise ValueError("pose must be finite")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
        raise ValueError("pose must have homogeneous last row [0,0,0,1]")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("pose rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("pose rotation determinant must be +1")
    return pose


class TSDFVolume:
    """Dense projective TSDF with a weighted running-average update."""

    def __init__(self, config: VolumeConfig):
        self.config = config
        nx, ny, nz = config.dimensions_xyz
        self.tsdf = np.ones((nz, ny, nx), dtype=np.float32)
        self.weight = np.zeros((nz, ny, nx), dtype=np.float32)

        origin = np.asarray(config.origin_world, dtype=np.float64)
        xs = origin[0] + (np.arange(nx, dtype=np.float64) + 0.5) * config.voxel_size
        ys = origin[1] + (np.arange(ny, dtype=np.float64) + 0.5) * config.voxel_size
        zs = origin[2] + (np.arange(nz, dtype=np.float64) + 0.5) * config.voxel_size
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        self._centres_world = np.stack((xx, yy, zz), axis=-1)

    @property
    def centres_world(self) -> np.ndarray:
        """Voxel centres as ``[Nz, Ny, Nx, 3]`` metric ``(x,y,z)`` points."""
        return self._centres_world

    @property
    def memory_bytes(self) -> int:
        """Bytes used by persistent TSDF and weight arrays only."""
        return int(self.tsdf.nbytes + self.weight.nbytes)

    def integrate(
        self,
        depth_m: np.ndarray,
        camera: PinholeCamera,
        world_from_camera: np.ndarray,
        *,
        confidence: np.ndarray | None = None,
        observation_weight: float = 1.0,
        depth_min: float = 0.1,
        depth_max: float = 10.0,
    ) -> IntegrationStats:
        """Project every voxel into one depth image and fuse valid evidence.

        ``confidence`` is a non-negative ``[H,W]`` relative precision map.  A
        value of zero rejects the pixel.  It is not calibrated uncertainty.
        """
        depth = np.asarray(depth_m, dtype=np.float64)
        expected = (camera.height, camera.width)
        if depth.shape != expected:
            raise ValueError(f"depth shape {depth.shape} does not match {expected}")
        if observation_weight <= 0:
            raise ValueError("observation_weight must be positive")
        if not (0 <= depth_min < depth_max):
            raise ValueError("require 0 <= depth_min < depth_max")

        if confidence is None:
            confidence_image = np.ones_like(depth)
        else:
            confidence_image = np.asarray(confidence, dtype=np.float64)
            if confidence_image.shape != expected:
                raise ValueError("confidence must have the same shape as depth")
            if np.any(~np.isfinite(confidence_image)) or np.any(confidence_image < 0):
                raise ValueError("confidence must be finite and non-negative")

        pose_wc = _validate_pose(world_from_camera)
        pose_cw = np.linalg.inv(pose_wc)
        points_world = self._centres_world.reshape(-1, 3)
        points_camera = points_world @ pose_cw[:3, :3].T + pose_cw[:3, 3]
        z_camera = points_camera[:, 2]

        positive_z = z_camera > 0
        safe_z = np.where(positive_z, z_camera, 1.0)
        u_float = camera.fx * points_camera[:, 0] / safe_z + camera.cx
        v_float = camera.fy * points_camera[:, 1] / safe_z + camera.cy
        u = np.rint(u_float).astype(np.int64)
        v = np.rint(v_float).astype(np.int64)
        in_image = (
            positive_z
            & (u >= 0)
            & (u < camera.width)
            & (v >= 0)
            & (v < camera.height)
        )

        flat_indices = np.flatnonzero(in_image)
        if flat_indices.size == 0:
            return IntegrationStats(0, 0, 0, 0)
        sampled_depth = depth[v[flat_indices], u[flat_indices]]
        sampled_confidence = confidence_image[v[flat_indices], u[flat_indices]]
        valid_depth = (
            np.isfinite(sampled_depth)
            & (sampled_depth >= depth_min)
            & (sampled_depth <= depth_max)
            & (sampled_confidence > 0)
        )

        valid_indices = flat_indices[valid_depth]
        signed_distance = sampled_depth[valid_depth] - z_camera[valid_indices]
        within_back_band = signed_distance >= -self.config.truncation
        update_indices = valid_indices[within_back_band]
        signed_distance = signed_distance[within_back_band]
        if update_indices.size == 0:
            return IntegrationStats(
                int(flat_indices.size),
                0,
                int(np.count_nonzero(~valid_depth)),
                int(np.count_nonzero(~within_back_band)),
            )

        new_tsdf = np.minimum(1.0, signed_distance / self.config.truncation)
        valid_confidence = sampled_confidence[valid_depth][within_back_band]
        new_weight = observation_weight * valid_confidence

        tsdf_flat = self.tsdf.reshape(-1)
        weight_flat = self.weight.reshape(-1)
        old_weight = weight_flat[update_indices].astype(np.float64)
        denominator = old_weight + new_weight
        fused = (
            old_weight * tsdf_flat[update_indices].astype(np.float64)
            + new_weight * new_tsdf
        ) / denominator
        tsdf_flat[update_indices] = fused.astype(np.float32)
        weight_flat[update_indices] = np.minimum(
            denominator, self.config.max_weight
        ).astype(np.float32)

        return IntegrationStats(
            projected_voxels=int(flat_indices.size),
            updated_voxels=int(update_indices.size),
            invalid_depth_voxels=int(np.count_nonzero(~valid_depth)),
            behind_band_voxels=int(np.count_nonzero(~within_back_band)),
        )

    def extract_zero_crossings_z(self, *, min_weight: float = 1.0) -> SurfaceMap:
        """Extract the first observed front surface in each world-z column.

        Linear interpolation between adjacent voxel-centre TSDF samples gives a
        sub-voxel zero crossing.  This extractor is intentionally specialized;
        a production mapper normally uses Marching Cubes or TSDF raycasting.
        """
        if min_weight <= 0:
            raise ValueError("min_weight must be positive")
        nz, ny, nx = self.tsdf.shape
        left = self.tsdf[:-1]
        right = self.tsdf[1:]
        supported = (self.weight[:-1] >= min_weight) & (self.weight[1:] >= min_weight)
        crossings = supported & (left > 0.0) & (right <= 0.0)
        has_crossing = np.any(crossings, axis=0)
        first = np.argmax(crossings, axis=0)

        yy, xx = np.indices((ny, nx))
        f0 = left[first, yy, xx].astype(np.float64)
        f1 = right[first, yy, xx].astype(np.float64)
        denominator = f0 - f1
        alpha = np.divide(f0, denominator, out=np.zeros_like(f0), where=denominator != 0)

        origin = np.asarray(self.config.origin_world, dtype=np.float64)
        x_world = origin[0] + (xx + 0.5) * self.config.voxel_size
        y_world = origin[1] + (yy + 0.5) * self.config.voxel_size
        z_world = origin[2] + (first + 0.5 + alpha) * self.config.voxel_size
        z_world = np.where(has_crossing, z_world, np.nan)
        support = np.minimum(
            self.weight[first, yy, xx], self.weight[np.minimum(first + 1, nz - 1), yy, xx]
        )
        support = np.where(has_crossing, support, 0.0)
        return SurfaceMap(x_world=x_world, y_world=y_world, z_world=z_world, support_weight=support)


def surface_metrics(
    surface: SurfaceMap,
    truth_z: np.ndarray,
    evaluation_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Coverage and absolute z error for an extracted height field."""
    truth = np.asarray(truth_z, dtype=np.float64)
    if truth.shape != surface.z_world.shape:
        raise ValueError("truth_z must match the surface map shape")
    if evaluation_mask is None:
        requested = np.isfinite(truth)
    else:
        requested = np.asarray(evaluation_mask, dtype=bool) & np.isfinite(truth)
        if requested.shape != truth.shape:
            raise ValueError("evaluation_mask must match truth_z")
    requested_count = int(np.count_nonzero(requested))
    if requested_count == 0:
        raise ValueError("evaluation mask is empty")
    accepted = requested & surface.valid
    accepted_count = int(np.count_nonzero(accepted))
    if accepted_count == 0:
        return {"coverage": 0.0, "mae_m": float("nan"), "p95_abs_m": float("nan")}
    error = np.abs(surface.z_world[accepted] - truth[accepted])
    return {
        "coverage": accepted_count / requested_count,
        "mae_m": float(np.mean(error)),
        "p95_abs_m": float(np.percentile(error, 95)),
    }
