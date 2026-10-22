"""Deterministic bundle-adjustment experiments and ablations."""

from __future__ import annotations

import argparse
import json

import numpy as np

from bundle_adjustment import (
    BAProblem,
    camera_centers,
    evaluate,
    project_observations,
    rotation_error_degrees,
    so3_exp,
    solve_bundle_adjustment,
)


K = np.array([[820.0, 0.0, 640.0], [0.0, 810.0, 360.0], [0.0, 0.0, 1.0]])


def synthetic_problem(
    seed: int = 20261023,
    *,
    cameras: int = 7,
    points: int = 140,
    noise_px: float = 0.8,
    outlier_fraction: float = 0.10,
    dropout_fraction: float = 0.18,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    points_true = np.column_stack((
        rng.uniform(-1.10, 1.10, points),
        rng.uniform(-0.68, 0.68, points),
        rng.uniform(3.2, 6.0, points),
    ))
    rotations_true = np.empty((cameras, 3, 3))
    centers_true = np.empty((cameras, 3))
    for camera in range(cameras):
        centers_true[camera] = np.array([
            -0.75 + 0.25 * camera,
            0.035 * np.sin(0.8 * camera),
            0.018 * np.cos(0.6 * camera),
        ])
        rotations_true[camera] = so3_exp(np.radians([
            0.35 * np.sin(camera),
            -2.0 + 0.65 * camera,
            0.25 * np.cos(0.7 * camera),
        ]))
    translations_true = -np.einsum("cij,cj->ci", rotations_true, centers_true)

    camera_indices, point_indices, pixels = [], [], []
    for camera in range(cameras):
        camera_ids = np.full(points, camera, dtype=np.int64)
        point_ids = np.arange(points, dtype=np.int64)
        projected, depths = project_observations(
            K, rotations_true, translations_true, points_true,
            camera_ids, point_ids,
        )
        visible = (
            (depths > 0.1)
            & (projected[:, 0] > 30.0) & (projected[:, 0] < 1250.0)
            & (projected[:, 1] > 25.0) & (projected[:, 1] < 695.0)
        )
        keep = visible & (rng.random(points) >= dropout_fraction)
        if camera in (0, 1):
            keep[0] = bool(visible[0])
        ids = point_ids[keep]
        camera_indices.extend([camera] * len(ids))
        point_indices.extend(ids.tolist())
        pixels.extend(projected[keep].tolist())
    camera_indices = np.asarray(camera_indices, dtype=np.int64)
    point_indices = np.asarray(point_indices, dtype=np.int64)
    pixels = np.asarray(pixels, dtype=np.float64)
    pixels += rng.normal(0.0, noise_px, pixels.shape)

    is_outlier = np.zeros(len(pixels), dtype=bool)
    outlier_count = int(round(outlier_fraction * len(pixels)))
    if outlier_count:
        chosen = rng.choice(len(pixels), size=outlier_count, replace=False)
        is_outlier[chosen] = True
        directions = rng.normal(size=(outlier_count, 2))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        magnitudes = rng.uniform(35.0, 110.0, (outlier_count, 1))
        pixels[chosen] += directions * magnitudes

    rotations_initial = rotations_true.copy()
    translations_initial = translations_true.copy()
    points_initial = points_true + rng.normal(0.0, 0.025, points_true.shape)
    for camera in range(1, cameras):
        dR = so3_exp(np.radians(rng.normal(0.0, 0.35, 3)))
        rotations_initial[camera] = dR @ rotations_initial[camera]
        translations_initial[camera] = dR @ translations_initial[camera]
        translations_initial[camera] += rng.normal(0.0, 0.012, 3)
    # Explicit metric gauge: camera 0 defines the world frame, point 0 fixes scale.
    rotations_initial[0] = rotations_true[0]
    translations_initial[0] = translations_true[0]
    points_initial[0] = points_true[0]

    problem = BAProblem(
        K, rotations_initial, translations_initial, points_initial,
        camera_indices, point_indices, pixels,
    )
    return {
        "problem": problem,
        "rotations_true": rotations_true,
        "translations_true": translations_true,
        "centers_true": centers_true,
        "points_true": points_true,
        "inliers": ~is_outlier,
        "noise_px": noise_px,
        "outlier_fraction": outlier_fraction,
    }


def rmse_on_mask(problem: BAProblem, rotations, translations, points, mask: np.ndarray) -> float:
    projected, depths = project_observations(
        problem.K, rotations, translations, points,
        problem.camera_indices, problem.point_indices,
    )
    if np.any(depths[mask] <= 0):
        return float("inf")
    errors = projected[mask] - problem.pixels[mask]
    return float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))


def reconstruction_metrics(data: dict[str, object], result) -> dict[str, object]:
    problem = data["problem"]
    rotation_errors = rotation_error_degrees(result.rotations, data["rotations_true"])
    center_errors = 1000.0 * np.linalg.norm(
        camera_centers(result.rotations, result.translations) - data["centers_true"], axis=1
    )
    point_errors = 1000.0 * np.linalg.norm(result.points - data["points_true"], axis=1)
    return {
        "robust_cost_initial": result.initial_cost,
        "robust_cost_final": result.final_cost,
        "all_observation_rmse_initial_px": result.initial_rmse_px,
        "all_observation_rmse_final_px": result.final_rmse_px,
        "true_inlier_rmse_final_px": rmse_on_mask(
            problem, result.rotations, result.translations, result.points, data["inliers"]
        ),
        "camera_rotation_error_deg_median": float(np.median(rotation_errors)),
        "camera_rotation_error_deg_max": float(np.max(rotation_errors)),
        "camera_center_error_mm_median": float(np.median(center_errors)),
        "camera_center_error_mm_max": float(np.max(center_errors)),
        "landmark_error_mm_median": float(np.median(point_errors)),
        "landmark_error_mm_p90": float(np.percentile(point_errors, 90)),
        "iterations": result.iterations,
        "accepted_steps": result.accepted_steps,
        "rejected_steps": result.rejected_steps,
        "converged": result.converged,
    }


def run_once(seed: int = 20261023) -> dict[str, object]:
    data = synthetic_problem(seed)
    problem = data["problem"]
    linear = solve_bundle_adjustment(
        problem, loss="linear", max_iterations=35, initial_damping=1e-3,
    )
    robust = solve_bundle_adjustment(
        problem, loss="huber", huber_delta_px=2.5,
        max_iterations=35, initial_damping=1e-3,
    )
    return {
        "dataset": {
            "cameras": len(problem.rotations),
            "landmarks": len(problem.points),
            "observations": len(problem.pixels),
            "noise_std_px_per_axis": data["noise_px"],
            "outlier_observations": int(np.count_nonzero(~data["inliers"])),
            "fixed_camera": 0,
            "fixed_scale_landmark": 0,
        },
        "linear_loss": reconstruction_metrics(data, linear),
        "huber_loss": reconstruction_metrics(data, robust),
    }


def ablations(seed: int = 20261023, trials: int = 10) -> dict[str, object]:
    variants = {
        "pose_only_huber": dict(optimize_cameras=True, optimize_points=False, loss="huber"),
        "structure_only_huber": dict(optimize_cameras=False, optimize_points=True, loss="huber"),
        "joint_linear": dict(optimize_cameras=True, optimize_points=True, loss="linear"),
        "joint_huber": dict(optimize_cameras=True, optimize_points=True, loss="huber"),
    }
    collected: dict[str, dict[str, list[float]]] = {
        name: {"inlier_rmse": [], "rotation": [], "center": [], "point": []}
        for name in variants
    }
    failures = {name: 0 for name in variants}
    for offset in range(trials):
        data = synthetic_problem(seed + offset, cameras=6, points=90)
        for name, options in variants.items():
            try:
                result = solve_bundle_adjustment(
                    data["problem"], max_iterations=30,
                    huber_delta_px=2.5, **options,
                )
                metrics = reconstruction_metrics(data, result)
            except (ValueError, np.linalg.LinAlgError):
                failures[name] += 1
                continue
            collected[name]["inlier_rmse"].append(metrics["true_inlier_rmse_final_px"])
            collected[name]["rotation"].append(metrics["camera_rotation_error_deg_median"])
            collected[name]["center"].append(metrics["camera_center_error_mm_median"])
            collected[name]["point"].append(metrics["landmark_error_mm_median"])

    def summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"median": None, "p90": None, "worst": None}
        array = np.asarray(values)
        return {
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)),
            "worst": float(np.max(array)),
        }

    return {
        "trials": trials,
        "variants": {
            name: {
                "failures": failures[name],
                "true_inlier_rmse_px": summary(values["inlier_rmse"]),
                "median_camera_rotation_error_deg": summary(values["rotation"]),
                "median_camera_center_error_mm": summary(values["center"]),
                "median_landmark_error_mm": summary(values["point"]),
            }
            for name, values in collected.items()
        },
        "warning": "Synthetic paired trials; equal trial counts are not runtime comparisons or real-data evidence.",
    }


def gauge_demo(seed: int = 20261023) -> dict[str, object]:
    data = synthetic_problem(seed, outlier_fraction=0.0, noise_px=0.0)
    problem = data["problem"]
    scale = 2.75
    centers = camera_centers(data["rotations_true"], data["translations_true"])
    origin = centers[0]
    scaled_centers = origin + scale * (centers - origin)
    scaled_points = origin + scale * (data["points_true"] - origin)
    scaled_translations = -np.einsum(
        "cij,cj->ci", data["rotations_true"], scaled_centers
    )
    original_px, _ = project_observations(
        problem.K, data["rotations_true"], data["translations_true"], data["points_true"],
        problem.camera_indices, problem.point_indices,
    )
    scaled_px, _ = project_observations(
        problem.K, data["rotations_true"], scaled_translations, scaled_points,
        problem.camera_indices, problem.point_indices,
    )
    try:
        solve_bundle_adjustment(problem, fixed_camera=0, fixed_point=None, max_iterations=2)
    except ValueError as error:
        guard = str(error)
    else:
        guard = "ERROR: missing-scale anchor was not rejected"
    return {
        "scale_factor": scale,
        "maximum_pixel_difference_after_global_scaling": float(np.max(np.abs(original_px - scaled_px))),
        "solver_guard_without_scale_anchor": guard,
        "meaning": "Fixing one camera removes global SE(3), but monocular scale still needs an anchor or prior.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--gauge", action="store_true")
    args = parser.parse_args()
    if args.ablations:
        output = ablations(args.seed)
    elif args.gauge:
        output = gauge_demo(args.seed)
    else:
        output = run_once(args.seed)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
