"""Generate an original SVG explaining streaming spatial memory."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

from streaming_memory import binary_iou, run_stream
from synthetic_video import make_video


plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 10.5,
    "svg.fonttype": "none",
    "svg.hashsalt": "sam2-streaming-memory-2026-10-29",
})


def box(ax, xy, width, height, text, color):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.018",
                           fc=color, ec="#334155", lw=1.0)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=7.5)


def arrow(ax, start, end, color="#475569", style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
                                mutation_scale=10, color=color, lw=1.2))


def main() -> None:
    video = make_video(seed=7)
    common = dict(features=video["features"], object_scores=video["object_scores"],
                  initial_mask=video["masks"][0], use_recent_memory=True, max_recent=6)
    no_object, _, _ = run_stream(**common, write_no_object_memory=True)
    corrected, _, _ = run_stream(**common, write_no_object_memory=True,
                                 correction_masks={16: video["masks"][16]})
    visible = np.array([mask.any() for mask in video["masks"]])
    no_object_iou = np.array([binary_iou(no_object[t], video["masks"][t])
                              if visible[t] else np.nan for t in range(len(visible))])
    corrected_iou = np.array([binary_iou(corrected[t], video["masks"][t])
                              if visible[t] else np.nan for t in range(len(visible))])

    fig = plt.figure(figsize=(12.2, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(grid[0, 0]); ax0.set_xlim(0, 1); ax0.set_ylim(0, 1); ax0.axis("off")
    box(ax0, (0.03, 0.68), 0.18, 0.18, "current frame\nHiera features", "#dbeafe")
    box(ax0, (0.30, 0.68), 0.20, 0.18, "memory attention\nself + cross + MLP", "#ede9fe")
    box(ax0, (0.59, 0.68), 0.18, 0.18, "prompt-aware\nmask decoder", "#fef3c7")
    box(ax0, (0.82, 0.68), 0.15, 0.18, "mask, pointer,\npresence", "#dcfce7")
    box(ax0, (0.57, 0.23), 0.21, 0.18, "memory encoder\nfeature + mask", "#fee2e2")
    box(ax0, (0.25, 0.23), 0.22, 0.18, "memory bank\nprompted + FIFO", "#ffedd5")
    arrow(ax0, (0.21, 0.77), (0.30, 0.77)); arrow(ax0, (0.50, 0.77), (0.59, 0.77))
    arrow(ax0, (0.77, 0.77), (0.82, 0.77)); arrow(ax0, (0.895, 0.68), (0.73, 0.41), "#dc2626")
    arrow(ax0, (0.57, 0.32), (0.47, 0.32), "#dc2626"); arrow(ax0, (0.36, 0.41), (0.40, 0.68), "#dc2626")
    ax0.text(0.50, 0.08, "streaming state loop: predictions become bounded future context",
             ha="center", color="#9f1239")
    ax0.set_title("A. SAM 2's stateful perception loop (conceptual)")

    ax1 = fig.add_subplot(grid[0, 1])
    logits = np.array([1.0 / np.sqrt(2.0), 0.0])
    weights = np.exp(logits) / np.exp(logits).sum()
    image = np.array([[logits[0], logits[1]], [weights[0], weights[1]]])
    heat = ax1.imshow(image, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks([0, 1], ["matching token", "other token"])
    ax1.set_yticks([0, 1], [r"logit $qk^T/\sqrt{d}$", "softmax weight"])
    for row in range(2):
        for col in range(2):
            ax1.text(col, row, f"{image[row, col]:.3f}", ha="center", va="center",
                     color="white" if image[row, col] > 0.55 else "#0f172a")
    ax1.set_title("B. Numerical attention microcase: q = [1, 0]")
    fig.colorbar(heat, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(grid[1, 0]); ax2.set_xlim(-0.5, 23.5); ax2.set_ylim(0, 3.2)
    for t in range(24):
        if t == 0:
            color, label = "#2563eb", "initial prompt"
        elif t in (9, 10, 11):
            color, label = "#64748b", "no-object memory"
        elif t == 16:
            color, label = "#7c3aed", "correction"
        else:
            color, label = "#93c5fd", "recent memory"
        ax2.add_patch(Rectangle((t - 0.4, 1.20), 0.8, 0.75, fc=color, ec="white"))
        if t in (0, 9, 16):
            ax2.annotate(label, (t, 1.98), xytext=(t, 2.65 if t != 9 else 2.35), ha="center",
                         fontsize=7.3, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1})
    ax2.annotate("FIFO retains at most 6 recent frames", (20, 1.15), (13.2, 0.35),
                 arrowprops={"arrowstyle": "-[", "color": "#dc2626", "lw": 1.3}, color="#9f1239")
    ax2.set(yticks=[], xlabel="frame index", title="C. Pinned prompts + bounded recent state")
    ax2.spines[["left", "right", "top"]].set_visible(False)

    ax3 = fig.add_subplot(grid[1, 1])
    frames = np.arange(len(visible))
    ax3.plot(frames, no_object_iou, "o-", ms=3.5, lw=1.5, color="#0f766e", label="no-object memory")
    ax3.plot(frames, corrected_iou, "o-", ms=3.5, lw=1.5, color="#7c3aed", label="+ correction at t=16")
    ax3.axvspan(8.5, 11.5, color="#cbd5e1", alpha=0.55, label="occluded")
    ax3.axvline(16, color="#7c3aed", ls="--", lw=1)
    ax3.set(xlabel="frame index", ylabel="visible-frame region IoU", ylim=(-0.03, 1.05),
            title="D. Executed synthetic mechanism study (seed 7)")
    ax3.grid(alpha=0.22); ax3.legend(fontsize=7.4, loc="lower left")

    fig.suptitle("SAM 2 streaming memory: prompted state, spatial retrieval, and correction",
                 fontsize=14, weight="bold")
    output = Path(__file__).resolve().parents[2] / "assets" / "sam2-streaming-memory" / "streaming-memory.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Title": "SAM 2 streaming memory", "Date": "2026-10-29"})
    print(output)


if __name__ == "__main__":
    main()
