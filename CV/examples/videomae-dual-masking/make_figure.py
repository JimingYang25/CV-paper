"""Generate an original SVG for VideoMAE tube and dual masking."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from application import phase_predictions
from demo import leakage_trial
from videomae_math import random_mask, tube_mask


plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 10.5,
    "svg.fonttype": "none",
    "svg.hashsalt": "videomae-dual-masking-2026-10-30",
})


def box(ax, xy, width, height, text, color):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.018",
                           fc=color, ec="#334155", lw=1.0)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=7.7)


def arrow(ax, start, end, color="#475569"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                color=color, lw=1.2))


def main() -> None:
    fig = plt.figure(figsize=(12.2, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(grid[0, 0]); ax0.set_xlim(0, 1); ax0.set_ylim(0, 1); ax0.axis("off")
    box(ax0, (0.03, 0.67), 0.18, 0.18, "video clip\n[B,C,T,H,W]", "#dbeafe")
    box(ax0, (0.28, 0.67), 0.18, 0.18, "2×16×16\ncube tokens", "#e0e7ff")
    box(ax0, (0.54, 0.67), 0.18, 0.18, "tube mask\n≈90% hidden", "#fee2e2")
    box(ax0, (0.80, 0.67), 0.17, 0.18, "ViT encoder\nvisible only", "#ede9fe")
    box(ax0, (0.55, 0.22), 0.20, 0.18, "dual-masked\nlight decoder", "#fef3c7")
    box(ax0, (0.22, 0.22), 0.22, 0.18, "normalized pixels\nmasked MSE", "#dcfce7")
    arrow(ax0, (0.21, 0.76), (0.28, 0.76)); arrow(ax0, (0.46, 0.76), (0.54, 0.76))
    arrow(ax0, (0.72, 0.76), (0.80, 0.76)); arrow(ax0, (0.885, 0.67), (0.72, 0.40), "#7c3aed")
    arrow(ax0, (0.55, 0.31), (0.44, 0.31), "#7c3aed")
    ax0.text(0.50, 0.08, "discard decoder after pre-training; adapt encoder downstream",
             ha="center", color="#5b21b6")
    ax0.set_title("A. Asymmetric masked video autoencoding")

    rng = np.random.default_rng(4)
    random = random_mask((8, 4, 4), 0.75, rng).reshape(8, 16)
    tube = tube_mask((8, 4, 4), 0.75, rng).reshape(8, 16)
    ax1 = fig.add_subplot(grid[0, 1])
    combined = np.concatenate([random, np.ones((8, 1)), tube], axis=1)
    ax1.imshow(combined, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax1.axvline(15.5, color="white", lw=5)
    ax1.text(7.5, -1.0, "random mask", ha="center", color="#334155")
    ax1.text(24.5, -1.0, "tube mask", ha="center", color="#334155")
    ax1.set(yticks=np.arange(8), ylabel="tubelet time", xlabel="spatial token index",
            title="B. Tube masking blocks same-site temporal copying")
    ax1.set_xticks([0, 5, 10, 15, 17, 22, 27, 32], [0, 5, 10, 15, 0, 5, 10, 15])

    trials = [leakage_trial(seed) for seed in range(20)]
    random_loss = [item["random_copy_mse"] for item in trials]
    tube_loss = [item["tube_copy_mse"] for item in trials]
    ax2 = fig.add_subplot(grid[1, 0])
    for before, after in zip(random_loss, tube_loss):
        ax2.plot([0, 1], [before, after], color="#94a3b8", lw=0.9, alpha=0.7)
    ax2.scatter(np.zeros(20), random_loss, color="#0284c7", s=22, zorder=3)
    ax2.scatter(np.ones(20), tube_loss, color="#dc2626", s=22, zorder=3)
    ax2.set_xticks([0, 1], ["random", "tube"])
    ax2.set(xlim=(-0.3, 1.3), ylabel="temporal-copy masked MSE",
            title="C. Controlled leakage probe: tube is harder (20/20)")
    ax2.grid(axis="y", alpha=0.23)

    predictions = phase_predictions()
    ax3 = fig.add_subplot(grid[1, 1])
    t = np.arange(len(predictions["target"]))
    ax3.step(t, predictions["target"], where="post", lw=2.0, color="#0f172a", label="target")
    ax3.step(t, predictions["frame"] + 0.04, where="post", lw=0.8, alpha=0.65,
             color="#ef4444", label="frame appearance")
    ax3.step(t, predictions["ordered"] - 0.04, where="post", lw=1.6,
             color="#7c3aed", label="clip + ordered decoder")
    ax3.set(xlabel="frame", ylabel="surgical phase", yticks=range(5), ylim=(-0.3, 4.3),
            title="D. Synthetic downstream phase stream")
    ax3.grid(alpha=0.2); ax3.legend(fontsize=7.4, loc="upper left")

    fig.suptitle("VideoMAE: high-ratio tube masking and efficient dual-masked decoding",
                 fontsize=14, weight="bold")
    output = Path(__file__).resolve().parents[2] / "assets" / "videomae-dual-masking" / "tube-and-dual-masking.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Title": "VideoMAE tube and dual masking", "Date": "2026-10-30"})
    print(output)


if __name__ == "__main__":
    main()
