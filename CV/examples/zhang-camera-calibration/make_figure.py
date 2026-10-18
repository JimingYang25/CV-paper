"""Original synthetic illustration; output files are explicitly requested by CLI."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from demo import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="Destination .svg or .png")
    args = parser.parse_args()
    report, (uv, before, after, diverse, parallel) = run()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for i, points in enumerate(uv):
        axes[0].plot(points[:, 0], points[:, 1], ".", ms=2, alpha=0.55)
    axes[0].set(xlim=(0, 1280), ylim=(960, 0), xlabel="u [px]", ylabel="v [px]",
                title="A  Diverse synthetic views")
    axes[0].set_aspect("equal", adjustable="box")
    for V, name, color in [(diverse, "Diverse normals: rank 5", "#136f8a"),
                           (parallel, "Parallel normals: rank 2", "#b54838")]:
        values = np.linalg.svd(V, compute_uv=False)
        axes[1].semilogy(np.arange(1, 7), np.maximum(values/values[0], 1e-18), "o-", label=name, color=color)
    axes[1].set(title="B  Intrinsic constraint spectrum", xlabel="Singular-value index",
                ylabel="Singular value / largest", xticks=np.arange(1, 7), ylim=(1e-18, 3))
    axes[1].legend(fontsize=9, loc="lower left")
    axes[1].grid(alpha=0.2)
    view = np.arange(1, len(before)+1)
    axes[2].bar(view-0.18, before, width=0.36, color="#b54838", label="Linear, zero distortion")
    axes[2].bar(view+0.18, after, width=0.36, color="#136f8a", label="Joint refinement")
    axes[2].set(title="C  Same observations, lower residual", xlabel="View index",
                ylabel="Pixel-vector RMS [px]", xticks=[1, 3, 6, 9, 12])
    axes[2].legend(fontsize=9)
    fig.suptitle("Zhang calibration: coverage, observability, and refinement", fontsize=16)
    fig.supxlabel("Synthetic only | 12 views × 54 corners | 0.15 px noise per coordinate | seed 20261019", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    print(f"Wrote {args.output}; refined RMS {report['refined_vector_rms_px']:.6f} px")


if __name__ == "__main__":
    main()
