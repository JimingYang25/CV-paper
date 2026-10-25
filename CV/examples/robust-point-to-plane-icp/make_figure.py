"""Generate the original GitHub-renderable SVG used by the lesson."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robust_icp import ICPConfig, linearize_point_to_plane, point_to_plane_icp, transform_points
from synthetic_scene import make_planar_model, make_registration_case


plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "svg.fonttype": "none",
    "svg.hashsalt": "robust-point-to-plane-icp-2026-10-26",
})


def main() -> None:
    case = make_registration_case(seed=7)
    config = ICPConfig(distance_threshold=0.14, huber_delta=0.012, trim_fraction=0.82, normal_angle_deg=50.0)
    result = point_to_plane_icp(
        case["source_points"], case["target_points"], case["target_normals"],
        source_normals=case["source_normals"], initial_T_target_from_source=case["T_initial"], config=config,
    )

    fig = plt.figure(figsize=(12.0, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(grid[0, 0], projection="3d")
    target = case["target_points"][::8]
    before = transform_points(case["T_initial"], case["source_points"])[::5]
    after = transform_points(result.T_target_from_source, case["source_points"])[::5]
    ax0.scatter(target[:, 0], target[:, 1], target[:, 2], s=4, c="#3b82f6", alpha=0.45, label="target model")
    ax0.scatter(before[:, 0], before[:, 1], before[:, 2], s=5, c="#ef4444", alpha=0.45, label="initial scan")
    ax0.scatter(after[:, 0], after[:, 1], after[:, 2], s=5, c="#10b981", alpha=0.45, label="aligned scan")
    ax0.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="A. Local scan-to-model alignment")
    ax0.legend(loc="upper left", fontsize=7)
    ax0.view_init(elev=23, azim=-56)

    ax1 = fig.add_subplot(grid[0, 1])
    ax1.axhline(0.0, color="#334155", lw=2, label="target tangent plane")
    q = np.array([0.18, 0.0]); x = np.array([-0.05, 0.24]); n = np.array([0.0, 1.0])
    ax1.plot(*q, "o", color="#3b82f6", label=r"target $q_i$")
    ax1.plot(*x, "o", color="#ef4444", label=r"transformed source $x_i$")
    ax1.annotate("", xy=q, xytext=x, arrowprops={"arrowstyle": "->", "lw": 2, "color": "#f59e0b"})
    ax1.arrow(q[0], q[1], 0.0, 0.18, width=0.004, head_width=0.035, color="#7c3aed", length_includes_head=True)
    ax1.text(q[0] + 0.025, 0.14, r"normal $n_i$", color="#7c3aed")
    ax1.text(0.0, 0.11, r"$r_i=n_i^T(x_i-q_i)$", color="#b45309")
    ax1.set_xlim(-0.25, 0.45); ax1.set_ylim(-0.08, 0.38); ax1.set_aspect("equal")
    ax1.set_title("B. Point-to-plane residual")
    ax1.set_xlabel("tangent direction"); ax1.set_ylabel("normal direction")
    ax1.legend(loc="upper left", fontsize=7)

    ax2 = fig.add_subplot(grid[1, 0])
    iterations = [h["iteration"] for h in result.iterations]
    rmse = [1000.0 * h["rmse"] for h in result.iterations]
    step = [h["step_norm"] for h in result.iterations]
    ax2.plot(iterations, rmse, "o-", color="#2563eb", label="point-to-plane RMSE [mm]")
    ax2.set(xlabel="iteration", ylabel="RMSE [mm]", title="C. Reassociate, linearize, update")
    ax2.grid(alpha=0.25)
    twin = ax2.twinx(); twin.semilogy(iterations, step, "s--", color="#dc2626", label=r"$\|\Delta\xi\|_2$")
    twin.set_ylabel("twist-step norm")
    lines = ax2.lines + twin.lines
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=7)

    ax3 = fig.add_subplot(grid[1, 1])
    plane, plane_n = make_planar_model(21)
    _, Jp = linearize_point_to_plane(plane, plane, plane_n)
    ep = np.linalg.eigvalsh(Jp.T @ Jp)
    ec = result.normal_matrix_eigenvalues
    floor = 1e-12
    k = np.arange(1, 7)
    ax3.semilogy(k, np.maximum(ep, floor), "o-", color="#ef4444", label="one plane (rank deficient)")
    ax3.semilogy(k, np.maximum(ec, floor), "s-", color="#10b981", label="corner scene (6-DoF)")
    ax3.set(xlabel="ordered eigenvalue", ylabel=r"eigenvalue of $J^T WJ$", title="D. Geometry controls observability")
    ax3.set_xticks(k); ax3.grid(alpha=0.25, which="both"); ax3.legend(fontsize=7)

    fig.suptitle("Robust point-to-plane ICP: geometry, optimization, and a safety diagnostic", fontsize=14, weight="bold")
    output = Path(__file__).resolve().parents[2] / "assets" / "robust-point-to-plane-icp" / "robust-icp.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Title": "Robust point-to-plane ICP", "Date": "2026-10-26"})
    print(output)


if __name__ == "__main__":
    main()
