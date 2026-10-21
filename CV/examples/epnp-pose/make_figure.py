"""Generate the original EPnP control-point and observability SVG."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from demo import K, synthetic_correspondences, thickness_sweep
from epnp_pose import (
    choose_control_points,
    project_points,
    ransac_epnp,
)


def main(output: str) -> None:
    data = synthetic_correspondences(20261022)
    world = data["points_world"]
    pixels = data["points_px"]
    truth = data["true_inliers"]
    controls, _, _ = choose_control_points(world[truth])
    fit = ransac_epnp(
        world, pixels, K, threshold_px=3.0,
        min_draws=96, max_draws=3500, seed=20261022,
    )
    predicted, valid = project_points(world, fit.pose.R, fit.pose.t, K)
    study = thickness_sweep(trials=20)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(14.2, 4.35), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.25, 1.0])

    ax0 = fig.add_subplot(grid[0], projection="3d")
    ax0.scatter(world[truth, 0], world[truth, 1], world[truth, 2],
                s=9, alpha=0.45, color="#1677b8", label="3-D correspondences")
    ax0.scatter(controls[:, 0], controls[:, 1], controls[:, 2],
                s=62, marker="D", color="#db4b3f", edgecolor="white",
                linewidth=0.8, label="virtual controls")
    for index, control in enumerate(controls):
        ax0.text(*control, f"  C{index}", color="#8f241d", weight="bold")
    for i in range(1, 4):
        ax0.plot([controls[0, 0], controls[i, 0]],
                 [controls[0, 1], controls[i, 1]],
                 [controls[0, 2], controls[i, 2]], color="#db4b3f", lw=1.2)
    ax0.set_title("A  Four virtual controls (object frame, m)")
    ax0.set_xlabel("object x [m]")
    ax0.set_ylabel("object y [m]")
    ax0.set_zlabel("")
    ax0.legend(loc="upper left", frameon=False)
    ax0.view_init(elev=22, azim=-55)

    ax1 = fig.add_subplot(grid[1])
    bad = ~truth
    ax1.scatter(pixels[bad, 0], pixels[bad, 1], s=13, color="#aeb4bb",
                alpha=0.7, label="random mismatches")
    ax1.scatter(pixels[truth, 0], pixels[truth, 1], s=12, color="#1677b8",
                alpha=0.75, label="genuine measurements")
    selected = fit.inliers & valid
    stride = np.flatnonzero(selected)[::7]
    ax1.quiver(predicted[stride, 0], predicted[stride, 1],
               pixels[stride, 0] - predicted[stride, 0],
               pixels[stride, 1] - predicted[stride, 1],
               angles="xy", scale_units="xy", scale=1, color="#db4b3f",
               width=0.004, label="reprojection residual")
    ax1.set_title("B  Robust pose and pixel residuals")
    ax1.set_xlabel("u [px]")
    ax1.set_ylabel("v [px]")
    ax1.set_xlim(40, 1240)
    ax1.set_ylim(700, 20)
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(alpha=0.15)
    ax1.legend(loc="lower right", frameon=False)

    ax2 = fig.add_subplot(grid[2])
    rows = study["rows"]
    ratios = np.array([row["median_scatter_planarity_ratio"] for row in rows])
    conditions = np.array([row["median_refinement_jacobian_condition"] for row in rows])
    ax2.semilogx(ratios, conditions, marker="o", color="#7747b7", lw=2)
    for row, x, y in zip(rows, ratios, conditions):
        ax2.annotate(f'{1000 * row["thickness_m"]:g} mm', (x, y),
                     xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax2.invert_xaxis()
    ax2.set_title("C  Near-planar observability")
    ax2.set_xlabel(r"scatter ratio  $\lambda_3/\lambda_1$  (smaller → thinner)")
    ax2.set_ylabel("median refinement Jacobian condition")
    ax2.grid(True, which="both", alpha=0.22)
    ax2.text(0.45, 0.04,
             "20 synthetic trials / thickness\nexact planar inputs are rejected",
             transform=ax2.transAxes, color="#4d545c", fontsize=8)

    fig.suptitle("EPnP: compress object geometry, recover metric pose, audit observability",
                 fontsize=14, weight="bold")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_format = "svg" if path.suffix.lower() == ".svg" else "png"
    metadata = {"Date": "2026-10-22"} if image_format == "svg" else None
    fig.savefig(path, format=image_format, metadata=metadata, dpi=160)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python make_figure.py OUTPUT.svg")
    main(sys.argv[1])
