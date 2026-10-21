"""Deterministic EPnP experiments and ablations."""

from __future__ import annotations

import argparse
import json

import numpy as np

from epnp_pose import (
    camera_center_world,
    project_points,
    ransac_epnp,
    refine_pose,
    rotation_error_degrees,
    rotation_xyz,
    solve_epnp,
    translation_error,
)


K = np.array([[910.0, 0.0, 640.0], [0.0, 895.0, 360.0], [0.0, 0.0, 1.0]])


def synthetic_correspondences(
    seed: int,
    *,
    inliers: int = 140,
    outliers: int = 60,
    noise_px: float = 0.65,
    thickness_m: float = 0.08,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    world = np.column_stack((
        rng.uniform(-0.16, 0.16, inliers),
        rng.uniform(-0.11, 0.11, inliers),
        rng.uniform(-0.5 * thickness_m, 0.5 * thickness_m, inliers),
    ))
    R = rotation_xyz((11.0, -17.0, 6.0))
    t = np.array([0.035, -0.022, 0.78])
    pixels, valid = project_points(world, R, t, K)
    if not np.all(valid):
        raise RuntimeError("synthetic points should remain visible")
    pixels += rng.normal(0.0, noise_px, pixels.shape)
    if outliers:
        outlier_world = np.column_stack((
            rng.uniform(-0.16, 0.16, outliers),
            rng.uniform(-0.11, 0.11, outliers),
            rng.uniform(-0.5 * thickness_m, 0.5 * thickness_m, outliers),
        ))
        outlier_pixels = np.column_stack((
            rng.uniform(80.0, 1200.0, outliers),
            rng.uniform(50.0, 670.0, outliers),
        ))
        world = np.vstack((world, outlier_world))
        pixels = np.vstack((pixels, outlier_pixels))
    truth = np.zeros(inliers + outliers, dtype=bool)
    truth[:inliers] = True
    order = rng.permutation(len(truth))
    return {
        "points_world": world[order],
        "points_px": pixels[order],
        "true_inliers": truth[order],
        "R": R,
        "t": t,
    }


def pose_metrics(pose, R_true: np.ndarray, t_true: np.ndarray) -> dict[str, float | int | None]:
    return {
        "rotation_error_deg": rotation_error_degrees(pose.R, R_true),
        "translation_error_mm": 1000.0 * translation_error(pose.t, t_true),
        "reprojection_rmse_px_on_fit_set": pose.reprojection_rmse_px,
        "positive_depth_fraction": pose.positive_depth_fraction,
        "beta_initializer": pose.diagnostics.initializer_index if pose.diagnostics else None,
        "optimizer_nfev": pose.optimizer_nfev,
        "jacobian_condition": pose.jacobian_condition,
    }


def run_once(seed: int = 20261022) -> dict[str, object]:
    data = synthetic_correspondences(seed)
    world, pixels, truth = data["points_world"], data["points_px"], data["true_inliers"]
    all_match = solve_epnp(world, pixels, K)
    robust = ransac_epnp(
        world, pixels, K, threshold_px=3.0, confidence=0.999,
        sample_size=6, min_draws=96, max_draws=3500, seed=seed,
    )
    selected = robust.inliers
    tp = int(np.count_nonzero(selected & truth))
    precision = tp / max(1, int(np.count_nonzero(selected)))
    recall = tp / int(np.count_nonzero(truth))
    robust_metrics = pose_metrics(robust.pose, data["R"], data["t"])
    robust_metrics.update({
        "support": int(np.count_nonzero(selected)),
        "precision": precision,
        "recall": recall,
        "draws": robust.draws,
        "valid_hypotheses": robust.valid_hypotheses,
        "median_selected_reprojection_px": float(np.median(robust.reprojection_errors_px[selected])),
    })
    return {
        "dataset": {
            "matches": len(world),
            "genuine": int(np.count_nonzero(truth)),
            "random_outliers": int(np.count_nonzero(~truth)),
            "noise_std_px_per_axis": 0.65,
            "object_extent_m": [0.32, 0.22, 0.08],
        },
        "all_match_epnp": pose_metrics(all_match, data["R"], data["t"]),
        "robust_epnp": robust_metrics,
        "true_camera_center_world_m": camera_center_world(data["R"], data["t"]).tolist(),
    }


def ablations(seed: int = 20261022, trials: int = 20) -> dict[str, object]:
    metrics: dict[str, list[float]] = {
        "all_rotation": [], "all_translation_mm": [],
        "oracle_closed_rotation": [], "oracle_closed_translation_mm": [],
        "oracle_refined_rotation": [], "oracle_refined_translation_mm": [],
        "robust_rotation": [], "robust_translation_mm": [],
        "robust_precision": [], "robust_recall": [],
    }
    initializer_counts = {"1": 0, "2": 0, "3": 0}
    robust_failures = 0
    for offset in range(trials):
        data = synthetic_correspondences(seed + offset)
        world, pixels, truth = data["points_world"], data["points_px"], data["true_inliers"]

        all_pose = solve_epnp(world, pixels, K)
        metrics["all_rotation"].append(rotation_error_degrees(all_pose.R, data["R"]))
        metrics["all_translation_mm"].append(1000 * translation_error(all_pose.t, data["t"]))

        closed = solve_epnp(world[truth], pixels[truth], K)
        refined = refine_pose(world[truth], pixels[truth], K, closed)
        initializer_counts[str(closed.diagnostics.initializer_index)] += 1
        metrics["oracle_closed_rotation"].append(rotation_error_degrees(closed.R, data["R"]))
        metrics["oracle_closed_translation_mm"].append(1000 * translation_error(closed.t, data["t"]))
        metrics["oracle_refined_rotation"].append(rotation_error_degrees(refined.R, data["R"]))
        metrics["oracle_refined_translation_mm"].append(1000 * translation_error(refined.t, data["t"]))

        robust = ransac_epnp(
            world, pixels, K, threshold_px=3.0, confidence=0.999,
            min_draws=96, max_draws=3500, seed=seed + offset,
        )
        tp = int(np.count_nonzero(robust.inliers & truth))
        precision = tp / max(1, int(np.count_nonzero(robust.inliers)))
        recall = tp / int(np.count_nonzero(truth))
        rerr = rotation_error_degrees(robust.pose.R, data["R"])
        terr = 1000 * translation_error(robust.pose.t, data["t"])
        metrics["robust_rotation"].append(rerr)
        metrics["robust_translation_mm"].append(terr)
        metrics["robust_precision"].append(precision)
        metrics["robust_recall"].append(recall)
        robust_failures += int(rerr > 5.0 or terr > 50.0)

    def error_summary(values: list[float]) -> dict[str, float]:
        a = np.asarray(values)
        return {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "worst": float(np.max(a))}

    def quality_summary(values: list[float]) -> dict[str, float]:
        a = np.asarray(values)
        return {"median": float(np.median(a)), "p10": float(np.percentile(a, 10)), "minimum": float(np.min(a))}

    return {
        "trials": trials,
        "all_match_rotation_deg": error_summary(metrics["all_rotation"]),
        "all_match_translation_mm": error_summary(metrics["all_translation_mm"]),
        "oracle_closed_rotation_deg": error_summary(metrics["oracle_closed_rotation"]),
        "oracle_closed_translation_mm": error_summary(metrics["oracle_closed_translation_mm"]),
        "oracle_refined_rotation_deg": error_summary(metrics["oracle_refined_rotation"]),
        "oracle_refined_translation_mm": error_summary(metrics["oracle_refined_translation_mm"]),
        "robust_rotation_deg": error_summary(metrics["robust_rotation"]),
        "robust_translation_mm": error_summary(metrics["robust_translation_mm"]),
        "robust_precision": quality_summary(metrics["robust_precision"]),
        "robust_recall": quality_summary(metrics["robust_recall"]),
        "robust_failures_rotation_over_5deg_or_translation_over_50mm": robust_failures,
        "oracle_closed_form_beta_initializer_counts": initializer_counts,
        "warning": "One synthetic distribution; equal trials do not imply equal runtime or a real-data benchmark.",
    }


def thickness_sweep(seed: int = 20261022, trials: int = 20) -> dict[str, object]:
    rows = []
    for thickness in (0.12, 0.04, 0.012, 0.004, 0.0012, 0.0004):
        rotations, translations, ratios, conditions = [], [], [], []
        failures = 0
        for offset in range(trials):
            data = synthetic_correspondences(
                seed + 1000 + offset, outliers=0, noise_px=0.65, thickness_m=thickness
            )
            try:
                closed = solve_epnp(data["points_world"], data["points_px"], K)
                refined = refine_pose(data["points_world"], data["points_px"], K, closed)
            except (ValueError, np.linalg.LinAlgError):
                failures += 1
                continue
            rotations.append(rotation_error_degrees(refined.R, data["R"]))
            translations.append(1000 * translation_error(refined.t, data["t"]))
            ratios.append(closed.diagnostics.planarity_ratio)
            conditions.append(refined.jacobian_condition)
        rows.append({
            "thickness_m": thickness,
            "successful_trials": len(rotations),
            "rejected_trials": failures,
            "median_scatter_planarity_ratio": float(np.median(ratios)) if ratios else None,
            "median_rotation_error_deg": float(np.median(rotations)) if rotations else None,
            "p90_rotation_error_deg": float(np.percentile(rotations, 90)) if rotations else None,
            "median_translation_error_mm": float(np.median(translations)) if translations else None,
            "p90_translation_error_mm": float(np.percentile(translations, 90)) if translations else None,
            "median_refinement_jacobian_condition": float(np.median(conditions)) if conditions else None,
        })
    return {
        "trials_per_thickness": trials,
        "rows": rows,
        "important": "Thickness is object-coordinate depth spread, not camera distance; exact planar handling is omitted.",
    }


def coherent_wrong_pose_failure(seed: int = 20261022) -> dict[str, object]:
    """Show that geometric consensus does not know which object pose was intended."""
    rng = np.random.default_rng(seed + 5000)
    intended_count, distractor_count = 100, 150
    intended_world = np.column_stack((
        rng.uniform(-0.16, 0.16, intended_count),
        rng.uniform(-0.11, 0.11, intended_count),
        rng.uniform(-0.04, 0.04, intended_count),
    ))
    distractor_world = np.column_stack((
        rng.uniform(-0.16, 0.16, distractor_count),
        rng.uniform(-0.11, 0.11, distractor_count),
        rng.uniform(-0.04, 0.04, distractor_count),
    ))
    R_intended, t_intended = rotation_xyz((11.0, -17.0, 6.0)), np.array([0.035, -0.022, 0.78])
    R_distractor, t_distractor = rotation_xyz((-7.0, 13.0, -9.0)), np.array([-0.055, 0.038, 0.91])
    intended_px, _ = project_points(intended_world, R_intended, t_intended, K)
    distractor_px, _ = project_points(distractor_world, R_distractor, t_distractor, K)
    intended_px += rng.normal(0.0, 0.55, intended_px.shape)
    distractor_px += rng.normal(0.0, 0.55, distractor_px.shape)
    world = np.vstack((intended_world, distractor_world))
    pixels = np.vstack((intended_px, distractor_px))
    intended_mask = np.zeros(len(world), dtype=bool)
    intended_mask[:intended_count] = True
    order = rng.permutation(len(world))
    world, pixels, intended_mask = world[order], pixels[order], intended_mask[order]
    result = ransac_epnp(
        world, pixels, K, threshold_px=2.5, min_draws=128,
        max_draws=4000, seed=seed + 5000,
    )
    return {
        "intended_consistent_matches": intended_count,
        "distractor_consistent_matches": distractor_count,
        "selected_support": int(np.count_nonzero(result.inliers)),
        "selected_intended_matches": int(np.count_nonzero(result.inliers & intended_mask)),
        "selected_distractor_matches": int(np.count_nonzero(result.inliers & ~intended_mask)),
        "error_to_intended": {
            "rotation_deg": rotation_error_degrees(result.pose.R, R_intended),
            "translation_mm": 1000 * translation_error(result.pose.t, t_intended),
        },
        "error_to_distractor": {
            "rotation_deg": rotation_error_degrees(result.pose.R, R_distractor),
            "translation_mm": 1000 * translation_error(result.pose.t, t_distractor),
        },
        "meaning": "RANSAC selected the larger coherent pose; geometry alone has no intended-object identity.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--thickness", action="store_true")
    parser.add_argument("--failure-case", action="store_true")
    args = parser.parse_args()
    if args.failure_case:
        output = coherent_wrong_pose_failure(args.seed)
    elif args.ablations:
        output = ablations(args.seed)
    elif args.thickness:
        output = thickness_sweep(args.seed)
    else:
        output = run_once(args.seed)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

