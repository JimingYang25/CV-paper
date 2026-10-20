"""Deterministic experiments for normalized eight-point relative pose."""

from __future__ import annotations

import argparse
import json

import numpy as np

from essential_geometry import (
    direction_error_degrees,
    estimate_essential_eight_point,
    pixels_to_normalized,
    ransac_essential_eight_point,
    recover_relative_pose,
    rotation_error_degrees,
    rotation_y,
    select_pose_by_cheirality,
    project_scene,
)


K1 = np.array([[820.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
K2 = np.array([[790.0, 0.0, 620.0], [0.0, 810.0, 350.0], [0.0, 0.0, 1.0]])


def synthetic_matches(
    seed: int,
    *,
    inliers: int = 160,
    outliers: int = 80,
    noise_px: float = 0.55,
    baseline: float = 0.28,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    R = rotation_y(4.0)
    direction = np.array([1.0, 0.10, 0.16])
    direction /= np.linalg.norm(direction)
    t = baseline * direction
    X = np.column_stack(
        (rng.uniform(-2.2, 2.2, inliers),
         rng.uniform(-1.25, 1.25, inliers),
         rng.uniform(4.0, 10.0, inliers))
    )
    p1, p2 = project_scene(X, R, t, K1, K2)
    p1 += rng.normal(0.0, noise_px, p1.shape)
    p2 += rng.normal(0.0, noise_px, p2.shape)
    if outliers:
        q1 = np.column_stack((rng.uniform(40, 1240, outliers), rng.uniform(30, 690, outliers)))
        q2 = np.column_stack((rng.uniform(40, 1240, outliers), rng.uniform(30, 690, outliers)))
        p1 = np.vstack((p1, q1))
        p2 = np.vstack((p2, q2))
    truth = np.zeros(inliers + outliers, dtype=bool)
    truth[:inliers] = True
    order = rng.permutation(len(truth))
    return {
        "points1_px": p1[order],
        "points2_px": p2[order],
        "true_inliers": truth[order],
        "R": R,
        "t": t,
        "points3d": X,
    }


def pose_metrics(pose, R_true: np.ndarray, t_true: np.ndarray) -> dict[str, float | list[int]]:
    return {
        "rotation_error_deg": rotation_error_degrees(pose.R, R_true),
        "translation_direction_error_deg": direction_error_degrees(pose.t_direction, t_true),
        "positive_depth_points": int(np.count_nonzero(pose.positive_depth)),
        "cheirality_counts": list(pose.cheirality_counts),
    }


def run_once(seed: int = 20261021) -> dict[str, object]:
    data = synthetic_matches(seed)
    p1, p2, truth = data["points1_px"], data["points2_px"], data["true_inliers"]
    robust = ransac_essential_eight_point(
        p1, p2, K1, K2, threshold_px=3.0, confidence=0.999,
        min_draws=96, max_draws=4000, seed=seed,
    )
    n1 = pixels_to_normalized(p1, K1)
    n2 = pixels_to_normalized(p2, K2)
    pose = select_pose_by_cheirality(robust.E, n1[robust.inliers], n2[robust.inliers])
    predicted = robust.inliers
    tp = int(np.count_nonzero(predicted & truth))
    precision = tp / max(1, int(np.count_nonzero(predicted)))
    recall = tp / int(np.count_nonzero(truth))

    all_pose = recover_relative_pose(p1, p2, K1, K2)
    result: dict[str, object] = {
        "dataset": {
            "matches": len(p1),
            "genuine": int(np.count_nonzero(truth)),
            "random_outliers": int(np.count_nonzero(~truth)),
            "coordinate_noise_std_px_per_axis": 0.55,
            "baseline_scene_units": float(np.linalg.norm(data["t"])),
        },
        "all_matches_nonrobust": pose_metrics(all_pose, data["R"], data["t"]),
        "eight_point_ransac": {
            **pose_metrics(pose, data["R"], data["t"]),
            "support": int(np.count_nonzero(predicted)),
            "precision": precision,
            "recall": recall,
            "draws": robust.draws,
            "valid_hypotheses": robust.valid_hypotheses,
            "median_inlier_sampson_px": float(np.sqrt(np.median(robust.sampson_squared_px[predicted]))),
        },
        "scale_warning": "t_direction is unit length; monocular image pairs do not recover baseline magnitude",
    }
    return result


def ablation(seed: int = 20261021, trials: int = 20) -> dict[str, object]:
    all_rotation, all_translation = [], []
    robust_rotation, robust_translation = [], []
    conditioned_rotation, raw_rotation = [], []
    conditioned_translation, raw_translation = [], []
    precisions, recalls, supports = [], [], []
    for offset in range(trials):
        data = synthetic_matches(seed + offset)
        p1, p2 = data["points1_px"], data["points2_px"]
        all_pose = recover_relative_pose(p1, p2, K1, K2)
        all_rotation.append(rotation_error_degrees(all_pose.R, data["R"]))
        all_translation.append(direction_error_degrees(all_pose.t_direction, data["t"]))

        robust = ransac_essential_eight_point(
            p1, p2, K1, K2, threshold_px=3.0, confidence=0.999,
            min_draws=96, max_draws=4000, seed=seed + offset,
        )
        n1 = pixels_to_normalized(p1, K1)
        n2 = pixels_to_normalized(p2, K2)
        pose = select_pose_by_cheirality(robust.E, n1[robust.inliers], n2[robust.inliers])
        robust_rotation.append(rotation_error_degrees(pose.R, data["R"]))
        robust_translation.append(direction_error_degrees(pose.t_direction, data["t"]))
        true_mask = data["true_inliers"]
        true_positive = int(np.count_nonzero(robust.inliers & true_mask))
        precisions.append(true_positive / max(1, int(np.count_nonzero(robust.inliers))))
        recalls.append(true_positive / int(np.count_nonzero(true_mask)))
        supports.append(int(np.count_nonzero(robust.inliers)))

        clean = synthetic_matches(seed + offset, outliers=0)
        c1 = pixels_to_normalized(clean["points1_px"], K1)
        c2 = pixels_to_normalized(clean["points2_px"], K2)
        for condition, rotation_target, translation_target in (
            (True, conditioned_rotation, conditioned_translation),
            (False, raw_rotation, raw_translation),
        ):
            E = estimate_essential_eight_point(c1, c2, condition=condition)
            clean_pose = select_pose_by_cheirality(E, c1, c2)
            rotation_target.append(rotation_error_degrees(clean_pose.R, clean["R"]))
            translation_target.append(direction_error_degrees(clean_pose.t_direction, clean["t"]))

    def summary(values: list[float]) -> dict[str, float]:
        array = np.asarray(values)
        return {
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)),
            "worst": float(np.max(array)),
        }

    def quality_summary(values: list[float]) -> dict[str, float]:
        array = np.asarray(values)
        return {
            "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)),
            "minimum": float(np.min(array)),
        }

    return {
        "trials": trials,
        "same_scene_distribution_independent_noise_and_matches": True,
        "all_matches_rotation_deg": summary(all_rotation),
        "all_matches_translation_direction_deg": summary(all_translation),
        "robust_rotation_deg": summary(robust_rotation),
        "robust_translation_direction_deg": summary(robust_translation),
        "robust_translation_failures_over_45_deg": int(np.count_nonzero(np.asarray(robust_translation) > 45.0)),
        "robust_precision": quality_summary(precisions),
        "robust_recall": quality_summary(recalls),
        "robust_support": summary(supports),
        "clean_conditioned_rotation_deg": summary(conditioned_rotation),
        "clean_conditioned_translation_direction_deg": summary(conditioned_translation),
        "clean_unconditioned_rotation_deg": summary(raw_rotation),
        "clean_unconditioned_translation_direction_deg": summary(raw_translation),
        "warning": "Equal trial counts do not imply equal runtime; this is a synthetic sensitivity study, not a benchmark.",
    }


def parallax_sweep(seed: int = 20261021, trials: int = 20) -> dict[str, object]:
    rows = []
    for baseline in (0.40, 0.16, 0.06, 0.02, 0.008):
        rot, trans, disparity = [], [], []
        for offset in range(trials):
            data = synthetic_matches(
                seed + 1000 + offset, outliers=0, baseline=baseline, noise_px=0.55
            )
            p1, p2 = data["points1_px"], data["points2_px"]
            pose = recover_relative_pose(p1, p2, K1, K2)
            rot.append(rotation_error_degrees(pose.R, data["R"]))
            trans.append(direction_error_degrees(pose.t_direction, data["t"]))
            disparity.append(float(np.median(np.linalg.norm(p2 - p1, axis=1))))
        rows.append({
            "baseline": baseline,
            "median_total_pixel_motion": float(np.median(disparity)),
            "median_rotation_error_deg": float(np.median(rot)),
            "median_translation_direction_error_deg": float(np.median(trans)),
            "p90_translation_direction_error_deg": float(np.percentile(trans, 90)),
        })
    return {
        "trials_per_baseline": trials,
        "rows": rows,
        "important": "Total pixel motion includes the fixed 4-degree rotation; it is not a parallax measurement.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--parallax", action="store_true")
    args = parser.parse_args()
    if args.ablations:
        output = ablation(args.seed)
    elif args.parallax:
        output = parallax_sweep(args.seed)
    else:
        output = run_once(args.seed)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

