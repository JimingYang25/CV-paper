"""Generate the lesson's original SVG with pipeline and measured toy results."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from dino_gram import gram_matrix, refine_patch_features, similarity_map
from synthetic_dense import make_dense_case, shuffled_anchor


plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 10.5,
    "svg.fonttype": "none",
    "svg.hashsalt": "dinov3-gram-anchoring-2026-10-27",
})


def box(ax, xy, width, height, text, color):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.02", fc=color, ec="#334155", lw=1.0)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=8)


def arrow(ax, start, end, color="#475569"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=color, lw=1.2))


def main() -> None:
    case = make_dense_case(seed=3)
    anchor, degraded = case["anchor_features"], case["degraded_features"]
    refined = refine_patch_features(degraded, anchor, steps=250)
    wrong_trace = refine_patch_features(degraded, shuffled_anchor(anchor, 994), steps=250)
    grid = case["grid_size"]
    query = case["query_index"]

    fig = plt.figure(figsize=(12.2, 8.0), constrained_layout=True)
    outer = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(outer[0, 0]); ax0.set_xlim(0, 1); ax0.set_ylim(0, 1); ax0.axis("off")
    box(ax0, (0.03, 0.68), 0.20, 0.17, "global + local\nviews", "#dbeafe")
    box(ax0, (0.34, 0.67), 0.20, 0.19, "student\n(backprop)", "#fee2e2")
    box(ax0, (0.34, 0.24), 0.20, 0.19, "EMA teacher\n(stop-grad)", "#dcfce7")
    box(ax0, (0.68, 0.67), 0.27, 0.19, "DINO CLS loss\n+ iBOT patch loss", "#fef3c7")
    box(ax0, (0.68, 0.24), 0.27, 0.19, "early Gram teacher\npatch relation target", "#ede9fe")
    arrow(ax0, (0.23, 0.76), (0.34, 0.76)); arrow(ax0, (0.23, 0.72), (0.34, 0.34))
    arrow(ax0, (0.54, 0.76), (0.68, 0.76)); arrow(ax0, (0.54, 0.34), (0.68, 0.34))
    arrow(ax0, (0.44, 0.67), (0.44, 0.43), "#16a34a")
    ax0.text(0.46, 0.54, "EMA", color="#15803d")
    ax0.text(0.5, 0.05, r"refinement: $L_{DINO}+L_{iBOT}+L_{DKoLeo}+w_G L_{Gram}$", ha="center", fontsize=8.5)
    ax0.set_title("A. Training signals and parameter flow")

    sub1 = outer[0, 1].subgridspec(1, 3)
    mats = [gram_matrix(anchor[:1])[0], gram_matrix(degraded[:1])[0], gram_matrix(refined.features[:1])[0]]
    names = ["early anchor", "degraded", "Gram-refined"]
    for j, (mat, name) in enumerate(zip(mats, names)):
        ax = fig.add_subplot(sub1[0, j]); ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.75, 0.985, "B. Patch × patch cosine Gram matrices", ha="center", va="top", fontsize=10.5)

    sub2 = outer[1, 0].subgridspec(1, 3)
    sim = [similarity_map(anchor[:1], query)[0], similarity_map(degraded[:1], query)[0], similarity_map(refined.features[:1], query)[0]]
    for j, (values, name) in enumerate(zip(sim, names)):
        ax = fig.add_subplot(sub2[0, j]); ax.imshow(values.reshape(grid, grid), vmin=-1, vmax=1, cmap="viridis")
        qy, qx = divmod(query, grid); ax.plot(qx, qy, "r+", ms=9, mew=2)
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.25, 0.49, "C. One query patch → dense similarity map", ha="center", va="top", fontsize=10.5)

    ax3 = fig.add_subplot(outer[1, 1])
    ax3.semilogy(refined.losses, color="#2563eb", lw=2, label="correct anchor objective")
    ax3.semilogy(wrong_trace.losses, color="#dc2626", lw=1.6, ls="--", label="wrong anchor objective")
    ax3.axhline(refined.losses[0], color="#64748b", lw=1, ls=":", label="no refinement")
    ax3.set(xlabel="optimization step", ylabel="Gram MSE", title="D. Both teachers are fit—only one is correct")
    ax3.grid(alpha=0.25, which="both"); ax3.legend(fontsize=8)

    fig.suptitle("DINOv3 Gram anchoring preserves dense patch relationships, not an absolute feature basis", fontsize=14, weight="bold")
    output = Path(__file__).resolve().parents[2] / "assets" / "dinov3-gram-anchoring" / "gram-anchoring.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Title": "DINOv3 Gram anchoring", "Date": "2026-10-27"})
    print(output)


if __name__ == "__main__":
    main()
