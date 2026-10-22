"""Generate the original bundle-adjustment factorization SVG."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bundle_adjustment import full_normal_step, linearize, solve_bundle_adjustment
from demo import reconstruction_metrics, synthetic_problem


def main(output: str) -> None:
    data = synthetic_problem(
        20261023, cameras=6, points=80, outlier_fraction=0.08
    )
    problem = data["problem"]
    system = linearize(problem, problem.rotations, problem.translations, problem.points)
    _, _, full_normal = full_normal_step(system, damping=1e-3)
    linear = solve_bundle_adjustment(problem, loss="linear", max_iterations=30)
    robust = solve_bundle_adjustment(problem, loss="huber", max_iterations=30)
    linear_metrics = reconstruction_metrics(data, linear)
    robust_metrics = reconstruction_metrics(data, robust)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.25), constrained_layout=True)

    # Panel A: a small visible subgraph from the executed problem.
    ax = axes[0]
    camera_ids = np.arange(4)
    point_ids = np.arange(12)
    camera_xy = np.column_stack((np.zeros(len(camera_ids)), np.linspace(0.1, 0.9, len(camera_ids))))
    point_xy = np.column_stack((np.ones(len(point_ids)), np.linspace(0.04, 0.96, len(point_ids))))
    edge_count = 0
    for camera, point in zip(problem.camera_indices, problem.point_indices):
        if camera in camera_ids and point in point_ids:
            ax.plot([camera_xy[camera, 0], point_xy[point, 0]],
                    [camera_xy[camera, 1], point_xy[point, 1]],
                    color="#b8bec7", lw=0.7, alpha=0.65, zorder=1)
            edge_count += 1
    ax.scatter(camera_xy[:, 0], camera_xy[:, 1], s=120, marker="s",
               color="#e05245", edgecolor="white", label="camera pose", zorder=3)
    ax.scatter(point_xy[:, 0], point_xy[:, 1], s=45, marker="o",
               color="#1f78b4", edgecolor="white", label="3-D landmark", zorder=3)
    for index, xy in enumerate(camera_xy):
        ax.text(xy[0] - 0.07, xy[1], f"C{index}", ha="right", va="center")
    ax.set_xlim(-0.28, 1.12)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    ax.set_title(f"A  Sparse observation graph ({edge_count} shown edges)")
    ax.legend(loc="lower center", frameon=False, ncol=2)

    # Panel B: actual damped normal-equation sparsity.
    ax = axes[1]
    pattern = np.abs(full_normal) > 1e-12
    ax.imshow(pattern, cmap="Greys", interpolation="nearest", aspect="equal")
    camera_size = 6 * len(system.variable_cameras)
    ax.axvline(camera_size - 0.5, color="#e05245", lw=1.2)
    ax.axhline(camera_size - 0.5, color="#e05245", lw=1.2)
    ax.text(0.18, 0.04, "U: camera blocks", transform=ax.transAxes,
            color="#a72f27", ha="center")
    ax.text(0.72, 0.96, "V: landmark blocks", transform=ax.transAxes,
            color="#245b86", ha="center", va="top")
    ax.set_title(f"B  Full normal matrix: {pattern.shape[0]}×{pattern.shape[1]}")
    ax.set_xlabel("camera increments  |  point increments")
    ax.set_ylabel("nonzero pattern")

    # Panel C: each method's own normalized objective and truth-side diagnostics.
    ax = axes[2]
    for result, color, label in (
        (linear, "#d28e26", "linear loss"),
        (robust, "#6741b8", "Huber loss"),
    ):
        values = np.asarray(result.cost_history)
        ax.semilogy(np.arange(len(values)), values / values[0],
                    color=color, lw=2.1, label=label)
    ax.set_xlabel("LM attempt")
    ax.set_ylabel("own objective / initial objective")
    ax.grid(True, which="both", alpha=0.22)
    ax.set_title("C  Executed optimization trace")
    ax.legend(frameon=False)
    ax.text(
        0.03, 0.04,
        "final true-inlier RMSE\n"
        f"linear: {linear_metrics['true_inlier_rmse_final_px']:.2f} px\n"
        f"Huber: {robust_metrics['true_inlier_rmse_final_px']:.2f} px",
        transform=ax.transAxes, color="#4a5058",
    )

    fig.suptitle("Bundle adjustment: sparse factors → block system → robust refinement",
                 fontsize=14, weight="bold")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_format = "svg" if path.suffix.lower() == ".svg" else "png"
    metadata = {"Date": "2026-10-23"} if image_format == "svg" else None
    fig.savefig(path, format=image_format, metadata=metadata, dpi=160)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python make_figure.py OUTPUT.svg")
    main(sys.argv[1])
