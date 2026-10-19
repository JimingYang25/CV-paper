"""Original synthetic spatial illustration; no published paper figure is reused."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from demo import run
from application import run as application
from robust_homography import transfer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report, (source, destination, truth, grid, result, all_fit) = run()
    app_report, (reference, current, rectified, valid) = application()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    # Sparse spatial illustration to keep SVG inspectable and compact.
    ax = axes[0, 0]
    for mask, color, name in [(truth, "#127c8b", "Genuine matches"),
                              (~truth, "#b64d3e", "Incorrect matches")]:
        ax.scatter(destination[mask, 0], destination[mask, 1], s=8, alpha=0.55, color=color, label=name)
    ax.set(xlim=(0, 720), ylim=(540, 0), xlabel="current u [px]", ylabel="current v [px]",
           title="A  Input is 60% incorrect correspondences")
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    predicted = transfer(result.H, source[truth])
    delta = destination[truth]-predicted
    ax.quiver(predicted[:, 0], predicted[:, 1], 15*delta[:, 0], 15*delta[:, 1],
              angles="xy", scale_units="xy", scale=1, width=0.004, color="#127c8b")
    ax.set(xlim=(0, 720), ylim=(540, 0), xlabel="current u [px]", ylabel="current v [px]",
           title="B  Genuine-match residuals (arrows magnified 15x)")
    axes[1, 0].imshow(current, cmap="gray", vmin=0, vmax=1)
    polygon = np.array([[80, 60], [560, 60], [560, 420], [80, 420], [80, 60.]])
    overlay = transfer(result.H, polygon)
    axes[1, 0].plot(overlay[:, 0], overlay[:, 1], color="#d6a629", linewidth=2)
    axes[1, 0].set(title="C  Estimated reference ROI in the current image", xlabel="current u [px]", ylabel="current v [px]")
    axes[1, 1].imshow(rectified, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].plot(polygon[:, 0], polygon[:, 1], color="#d6a629", linewidth=2)
    axes[1, 1].set(title="D  Inverse-resampled reference coordinates", xlabel="reference u [px]", ylabel="reference v [px]")
    for ax in axes.flat:
        ax.set_aspect("equal")
    fig.suptitle("LO-RANSAC-style robust homography and planar registration", fontsize=16)
    fig.supxlabel(f"Synthetic only | 250 matches, 100 genuine | held-out grid RMS {report['robust']['grid_vector_rms_px']:.3f} px | intensity MAE {app_report['valid_region_intensity_MAE_0_to_1']:.4f}", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
