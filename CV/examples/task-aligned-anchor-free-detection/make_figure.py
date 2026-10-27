"""Generate an original SVG for task-aligned anchor-free detection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from detector_math import dfl_target, task_aligned_assign
from synthetic_scene import candidate_pool


plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 10.5,
    "svg.fonttype": "none",
    "svg.hashsalt": "task-aligned-anchor-free-detection-2026-10-28",
})


def box(ax, xy, width, height, text, color):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.02", fc=color, ec="#334155", lw=1.0)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=8)


def arrow(ax, start, end, color="#475569"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=color, lw=1.2))


def main() -> None:
    fig = plt.figure(figsize=(12.2, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(grid[0, 0]); ax0.set_xlim(0, 1); ax0.set_ylim(0, 1); ax0.axis("off")
    box(ax0, (0.03, 0.65), 0.18, 0.19, "FPN maps\n[B,C,H,W]", "#dbeafe")
    box(ax0, (0.30, 0.65), 0.20, 0.19, "decoupled head\nclass + 4K logits", "#fee2e2")
    box(ax0, (0.61, 0.65), 0.16, 0.19, "DFL decode\nE[bin] × stride", "#fef3c7")
    box(ax0, (0.82, 0.65), 0.15, 0.19, "score filter\n+ NMS", "#dcfce7")
    box(ax0, (0.30, 0.20), 0.20, 0.19, "TAL assignment\ns^α · IoU^β", "#ede9fe")
    box(ax0, (0.61, 0.20), 0.16, 0.19, "QFL + DFL\n+ IoU loss", "#ffedd5")
    arrow(ax0, (0.21, 0.745), (0.30, 0.745)); arrow(ax0, (0.50, 0.745), (0.61, 0.745))
    arrow(ax0, (0.77, 0.745), (0.82, 0.745)); arrow(ax0, (0.40, 0.65), (0.40, 0.39), "#7c3aed")
    arrow(ax0, (0.50, 0.295), (0.61, 0.295), "#7c3aed")
    ax0.text(0.5, 0.05, "training path (purple) is removed at inference", ha="center", color="#6d28d9")
    ax0.set_title("A. Tensor and control flow")

    ax1 = fig.add_subplot(grid[0, 1])
    target = 4.35
    distribution = dfl_target(np.array(target), 16)
    colors = ["#94a3b8"] * 16; colors[4] = "#2563eb"; colors[5] = "#60a5fa"
    ax1.bar(np.arange(16), distribution, color=colors, edgecolor="white")
    ax1.axvline(target, color="#dc2626", ls="--", lw=1.7, label="expectation = 4.35")
    ax1.annotate("0.65", (4, 0.65), xytext=(4, 0.75), ha="center", arrowprops={"arrowstyle": "-", "color": "#475569"})
    ax1.annotate("0.35", (5, 0.35), xytext=(5.6, 0.52), ha="center", arrowprops={"arrowstyle": "-", "color": "#475569"})
    ax1.set(xlabel="distance bin k", ylabel="target probability", ylim=(0, 0.9), title="B. DFL keeps sub-bin location")
    ax1.legend(loc="upper right", frameon=False)

    case = candidate_pool(seed=7)
    score_only = task_aligned_assign(case["scores"], case["boxes"], case["points"],
                                     case["gt_labels"], case["gt_boxes"], topk=5, beta=0.0)
    aligned = task_aligned_assign(case["scores"], case["boxes"], case["points"],
                                  case["gt_labels"], case["gt_boxes"], topk=5, beta=6.0)
    own_class = np.repeat(np.arange(3), 32)
    own_score = case["scores"][np.arange(len(own_class)), own_class]
    own_iou = score_only.overlaps[np.arange(len(own_class)), own_class]
    ax2 = fig.add_subplot(grid[1, 0])
    ax2.scatter(own_score, own_iou, s=17, c="#cbd5e1", alpha=0.75, label="candidate")
    score_idx = np.flatnonzero(score_only.assigned_gt >= 0)
    tal_idx = np.flatnonzero(aligned.assigned_gt >= 0)
    ax2.scatter(own_score[score_idx], own_iou[score_idx], s=55, facecolors="none", edgecolors="#dc2626",
                linewidths=1.4, label="score-only top-k")
    ax2.scatter(own_score[tal_idx], own_iou[tal_idx], s=28, c="#2563eb", marker="x", label="TAL top-k")
    ax2.set(xlabel="class probability s", ylabel="IoU u", xlim=(0, 1), ylim=(0, 1),
            title=r"C. Assignment uses $t=s^{\alpha}u^{\beta}$")
    ax2.grid(alpha=0.2); ax2.legend(fontsize=7.5, loc="lower right")

    score_means, aligned_means = [], []
    for seed in range(30):
        trial = candidate_pool(seed)
        kwargs = dict(class_probabilities=trial["scores"], predicted_boxes=trial["boxes"],
                      anchor_points=trial["points"], gt_labels=trial["gt_labels"], gt_boxes=trial["gt_boxes"], topk=5)
        for result, output in [(task_aligned_assign(**kwargs, beta=0.0), score_means),
                               (task_aligned_assign(**kwargs, beta=6.0), aligned_means)]:
            pos = np.flatnonzero(result.assigned_gt >= 0)
            output.append(np.mean(result.overlaps[pos, result.assigned_gt[pos]]))
    ax3 = fig.add_subplot(grid[1, 1])
    for before, after in zip(score_means, aligned_means):
        ax3.plot([0, 1], [before, after], color="#94a3b8", lw=0.8, alpha=0.6)
    ax3.scatter(np.zeros(30), score_means, color="#dc2626", s=18, zorder=3)
    ax3.scatter(np.ones(30), aligned_means, color="#2563eb", s=18, zorder=3)
    ax3.set_xticks([0, 1], ["score only\nβ=0", "task aligned\nβ=6"])
    ax3.set(ylabel="mean IoU of selected positives", ylim=(0.45, 0.82), title="D. Controlled 30-seed assignment ablation")
    ax3.grid(axis="y", alpha=0.25)
    ax3.text(0.5, 0.465, "every paired trial improves", ha="center", color="#334155")

    fig.suptitle("Task-aligned anchor-free detection: align confidence, localization, and deployment semantics",
                 fontsize=14, weight="bold")
    output = Path(__file__).resolve().parents[2] / "assets" / "task-aligned-anchor-free-detection" / "tal-dfl-pipeline.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Title": "Task-aligned anchor-free detection", "Date": "2026-10-28"})
    print(output)


if __name__ == "__main__":
    main()
