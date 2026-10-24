"""Generate the lesson's original GitHub-renderable TSDF illustration."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from demo import default_camera, run_variant
from synthetic_scene import camera_pose, corrupt_depth, render_sloped_plane


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    plt.rcParams["svg.fonttype"] = "none"
    camera = default_camera()
    pose = camera_pose(-0.08, -0.04, 0.0)
    clean = render_sloped_plane(camera, pose)
    noisy, _, _ = corrupt_depth(clean, seed=20261025)
    volume, _ = run_variant(seed=20261025, frames=8, confidence_weighted=True)
    surface = volume.extract_zero_crossings_z(min_weight=0.5)

    ny, nx = surface.z_world.shape
    centre_y = ny // 2
    x = surface.x_world[centre_y]
    tsdf_slice = volume.tsdf[:, centre_y, :]
    weight_slice = volume.weight[:, centre_y, :]
    origin_z = volume.config.origin_world[2]
    nz = volume.tsdf.shape[0]
    z = origin_z + (np.arange(nz) + 0.5) * volume.config.voxel_size

    fig = plt.figure(figsize=(14, 4.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.05, 1.25, 1.15])

    ax0 = fig.add_subplot(grid[0, 0])
    im0 = ax0.imshow(noisy, cmap="viridis", vmin=1.0, vmax=1.4)
    ax0.set_title("1  Noisy depth frame")
    ax0.set_xlabel("u [pixel]")
    ax0.set_ylabel("v [pixel]")
    fig.colorbar(im0, ax=ax0, shrink=0.72, label="optical z [m]")

    ax1 = fig.add_subplot(grid[0, 1])
    ray_z = np.linspace(0.85, 1.50, 180)
    measured_z = 1.22
    mu = volume.config.truncation
    signed = np.minimum(1.0, (measured_z - ray_z) / mu)
    signed[ray_z > measured_z + mu] = np.nan
    ax1.plot(ray_z, signed, color="#2357a7", linewidth=2.3)
    ax1.axhline(0, color="black", linewidth=1)
    ax1.axvline(measured_z, color="#c43c35", linestyle="--", label="surface")
    ax1.axvspan(measured_z - mu, measured_z + mu, color="#f0c36d", alpha=0.25)
    ax1.set_ylim(-1.08, 1.08)
    ax1.set_title("2  One projective TSDF")
    ax1.set_xlabel("voxel optical z [m]")
    ax1.set_ylabel("normalized signed distance")
    ax1.legend(frameon=False, loc="lower left")

    ax2 = fig.add_subplot(grid[0, 2])
    masked = np.ma.masked_where(weight_slice <= 0, tsdf_slice)
    im2 = ax2.imshow(
        masked,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        origin="lower",
        extent=[x.min(), x.max(), z.min(), z.max()],
        aspect="auto",
    )
    ax2.contour(x, z, tsdf_slice, levels=[0.0], colors="black", linewidths=1.6)
    ax2.set_title("3  Fused TSDF slice")
    ax2.set_xlabel("world x [m]")
    ax2.set_ylabel("world z [m]")
    fig.colorbar(im2, ax=ax2, shrink=0.72, label="TSDF")

    ax3 = fig.add_subplot(grid[0, 3])
    surface_image = ax3.imshow(
        surface.z_world,
        origin="lower",
        extent=[
            surface.x_world.min(),
            surface.x_world.max(),
            surface.y_world.min(),
            surface.y_world.max(),
        ],
        interpolation="nearest",
        cmap="viridis",
        vmin=1.0,
        vmax=1.4,
    )
    ax3.set_aspect("equal")
    ax3.set_title("4  Sub-voxel zero set")
    ax3.set_xlabel("world x [m]")
    ax3.set_ylabel("world y [m]")
    fig.colorbar(surface_image, ax=ax3, shrink=0.72, label="world z [m]")

    fig.suptitle("Projective TSDF fusion: depth evidence → signed field → surface", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_format = args.output.suffix.lstrip(".").lower() or "svg"
    fig.savefig(args.output, format=output_format, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
