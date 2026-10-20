"""Create the original SVG used by the lesson."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from demo import K1, K2, parallax_sweep
from essential_geometry import essential_from_pose, fundamental_from_essential, homogeneous, project_scene, rotation_y


def epipolar_segment(line: np.ndarray, width: float, height: float):
    a, b, c = line
    candidates = []
    if abs(b) > 1e-12:
        for x in (0.0, width):
            y = -(a * x + c) / b
            if 0.0 <= y <= height:
                candidates.append((x, y))
    if abs(a) > 1e-12:
        for y in (0.0, height):
            x = -(b * y + c) / a
            if 0.0 <= x <= width:
                candidates.append((x, y))
    if len(candidates) < 2:
        return None
    return np.asarray(candidates[:2])


def main(output: str) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)

    # A: coordinate convention and scale ambiguity.
    ax = axes[0]
    R = rotation_y(6.0)
    t = np.array([0.30, -0.02, 0.05])
    C2 = -R.T @ t
    ax.scatter([0, C2[0]], [0, C2[2]], s=90, c=["#155e75", "#c2410c"], zorder=4)
    ax.annotate("camera 1\nP₁=[I|0]", (0, 0), xytext=(8, 10), textcoords="offset points")
    ax.annotate("camera 2 center\nC₂=−Rᵀt", (C2[0], C2[2]), xytext=(-78, 20), textcoords="offset points")
    ax.arrow(0, 0, C2[0], C2[2], width=0.002, length_includes_head=True, color="#64748b", alpha=0.8)
    rng = np.random.default_rng(8)
    X = np.column_stack((rng.uniform(-1.6, 1.6, 20), rng.uniform(4.0, 8.0, 20)))
    ax.scatter(X[:, 0], X[:, 1], marker=".", color="#334155", alpha=0.75, label="scene points")
    for x, z in X[::5]:
        ax.plot([0, x], [0, z], color="#0e7490", alpha=0.15)
        ax.plot([C2[0], x], [C2[2], z], color="#c2410c", alpha=0.15)
    ax.text(0.02, 0.97, "X₂ = R X₁ + t\nE = [t]×R\n‖t‖ is not observable", transform=ax.transAxes, va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1"})
    ax.set(xlabel="camera-1 x (arbitrary scene units)", ylabel="camera-1 z", title="A. Relative-pose convention")
    ax.set_ylim(-0.5, 8.4)
    ax.grid(alpha=0.2)

    # B: an epipolar line is a locus, not a recovered point.
    ax = axes[1]
    X3 = np.column_stack((rng.uniform(-1.4, 1.4, 9), rng.uniform(-0.8, 0.8, 9), rng.uniform(4.0, 8.0, 9)))
    p1, p2 = project_scene(X3, R, t, K1, K2)
    F = fundamental_from_essential(essential_from_pose(R, t), K1, K2)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(p1)))
    lines = (F @ homogeneous(p1).T).T
    for point, line, color in zip(p2, lines, colors):
        segment = epipolar_segment(line, 1280, 720)
        if segment is not None:
            ax.plot(segment[:, 0], segment[:, 1], color=color, alpha=0.55, linewidth=1.2)
        ax.scatter(point[0], point[1], color=color, edgecolor="white", linewidth=0.5, s=38, zorder=3)
    ax.set(xlim=(0, 1280), ylim=(720, 0), xlabel="u₂ / pixel column", ylabel="v₂ / pixel row",
           title="B. x₂ᵀ F x₁ = 0 in image 2")
    ax.text(0.03, 0.96, "colored dot: true match\nline: possible locations from x₁", transform=ax.transAxes,
            va="top", bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1"})
    ax.grid(alpha=0.15)

    # C: executed low-parallax sweep.
    ax = axes[2]
    rows = parallax_sweep()["rows"]
    baseline = np.array([row["baseline"] for row in rows])
    t_median = np.array([row["median_translation_direction_error_deg"] for row in rows])
    t_p90 = np.array([row["p90_translation_direction_error_deg"] for row in rows])
    r_median = np.array([row["median_rotation_error_deg"] for row in rows])
    order = np.argsort(baseline)
    ax.plot(baseline[order], t_median[order], "o-", color="#c2410c", label="translation median")
    ax.plot(baseline[order], t_p90[order], "s--", color="#ea580c", label="translation P90")
    ax.plot(baseline[order], r_median[order], "^-", color="#155e75", label="rotation median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="baseline / scene unit (log)", ylabel="angular error / degree (log)",
           title="C. Low parallax breaks translation")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best", fontsize=8)
    ax.text(0.03, 0.04, "20 noisy, outlier-free trials per baseline\nfixed 4° rotation; seed 20261021",
            transform=ax.transAxes, va="bottom", fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1"})

    fig.suptitle("Normalized eight-point geometry: constraint, pose convention, and observability", fontsize=15)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_format = destination.suffix.lstrip(".").lower() or "svg"
    metadata = {"Date": "2026-10-21", "Creator": "Matplotlib + original CV-paper code"} if output_format == "svg" else None
    fig.savefig(destination, format=output_format, dpi=150, metadata=metadata)
    plt.close(fig)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "../../assets/eight-point-essential-matrix/relative-pose-and-parallax.svg"
    main(target)

