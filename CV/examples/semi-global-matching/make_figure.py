"""Generate the original Census + SGM explanatory SVG."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from demo import synthetic_stereo
from sgm_stereo import compute_left_right, compute_sgm, disparity_metrics


def main(output: str) -> None:
    data = synthetic_stereo()
    raw = compute_sgm(data["left"], data["right"], 16, path_count=0)
    left, _, consistent = compute_left_right(
        data["left"], data["right"], 16,
        p1=1.5, p2=8.0, path_count=8,
    )
    filtered = left.disparity.copy()
    filtered[~consistent] = np.nan
    raw_metrics = disparity_metrics(
        raw.disparity, data["disparity_true"], data["evaluation_mask"]
    )
    filtered_metrics = disparity_metrics(
        filtered, data["disparity_true"], data["evaluation_mask"]
    )

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 4, figsize=(15.3, 3.9), constrained_layout=True)

    row = int(0.36 * data["left"].shape[0])
    col = int(0.52 * data["left"].shape[1])
    true_disparity = int(data["disparity_true"][row, col])
    right_col = col - true_disparity

    ax = axes[0]
    ax.imshow(data["left"], cmap="gray", vmin=0, vmax=255)
    ax.axhline(row, color="#f7d154", lw=1.1)
    ax.scatter([col], [row], color="#e53935", s=38, edgecolor="white", zorder=3)
    ax.annotate(
        f"pL=({col},{row})",
        (col, row), xytext=(col - 41, row - 13),
        arrowprops={"arrowstyle": "->", "color": "#e53935"},
        color="white", fontsize=8,
    )
    ax.set_title("A  Left reference and epipolar row")
    ax.set_axis_off()

    ax = axes[1]
    ax.imshow(data["right"], cmap="gray", vmin=0, vmax=255)
    ax.axhline(row, color="#f7d154", lw=1.1)
    ax.scatter([right_col], [row], color="#1e88e5", s=38, edgecolor="white", zorder=3)
    ax.annotate(
        f"pR=({right_col},{row})\nd={true_disparity}px",
        (right_col, row), xytext=(right_col + 8, row - 14),
        arrowprops={"arrowstyle": "->", "color": "#1e88e5"},
        color="white", fontsize=8,
    )
    ax.set_title("B  Right match after gain, bias, noise")
    ax.set_axis_off()

    ax = axes[2]
    image = ax.imshow(filtered, cmap="turbo", vmin=0, vmax=15)
    center_x, center_y = data["left"].shape[1] * 0.80, data["left"].shape[0] * 0.78
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)):
        ax.annotate(
            "", xy=(center_x, center_y),
            xytext=(center_x - 12 * dx, center_y - 12 * dy),
            arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.0},
        )
    ax.text(
        0.03, 0.96,
        f"LR coverage {100*filtered_metrics['coverage']:.1f}%\n"
        f"bad-1px {filtered_metrics['bad_1px_percent']:.2f}%",
        transform=ax.transAxes, va="top", color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3, "edgecolor": "none"},
    )
    ax.set_title("C  8 paths + left/right filtering")
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label="disparity (px)")

    ax = axes[3]
    disparities = np.arange(16)
    raw_curve = raw.matching_cost[row, col]
    sgm_curve = left.aggregated_cost[row, col]
    raw_normalized = raw_curve - np.min(raw_curve)
    sgm_normalized = sgm_curve - np.min(sgm_curve)
    ax.plot(disparities, raw_normalized, "o-", color="#d28e26", label="Census cost")
    ax.plot(disparities, sgm_normalized, "o-", color="#6741b8", label="8-path sum")
    ax.axvline(true_disparity, color="#2e7d32", ls="--", lw=1.4, label="true d")
    ax.set_xlabel("disparity hypothesis d (px)")
    ax.set_ylabel("cost above minimum")
    ax.set_title("D  One pixel: evidence + smoothness")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.04, 0.03,
        f"raw scene MAE {raw_metrics['mae_px']:.3f}px\n"
        f"filtered MAE {filtered_metrics['mae_px']:.3f}px",
        transform=ax.transAxes, color="#4a5058",
        bbox={"facecolor": "white", "alpha": 0.82, "pad": 2, "edgecolor": "none"},
    )

    fig.suptitle(
        "Census + Semi-Global Matching: local evidence, multi-path regularity, explicit rejection",
        fontsize=13.5, weight="bold",
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_format = "svg" if path.suffix.lower() == ".svg" else "png"
    metadata = {"Date": "2026-10-24"} if image_format == "svg" else None
    fig.savefig(path, format=image_format, metadata=metadata, dpi=170)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python make_figure.py OUTPUT.svg")
    main(sys.argv[1])
