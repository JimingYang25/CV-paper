"""Integrated example: refine a wrist-camera map before transforming a task target."""

from __future__ import annotations

import json

import numpy as np

from bundle_adjustment import (
    camera_centers,
    project_observations,
    rotation_error_degrees,
    solve_bundle_adjustment,
)
from demo import reconstruction_metrics, synthetic_problem


def main() -> None:
    data = synthetic_problem(
        20261223, cameras=8, points=160, noise_px=0.6,
        outlier_fraction=0.04, dropout_fraction=0.22,
    )
    result = solve_bundle_adjustment(
        data["problem"], loss="huber", huber_delta_px=2.0,
        max_iterations=40, initial_damping=1e-3,
    )
    metrics = reconstruction_metrics(data, result)

    # A higher-level detector/VLM has already bound landmarks to the intended
    # fixture. We additionally select four well-observed IDs using only track
    # length and post-fit residuals (synthetic truth is not used for selection).
    projected, _ = project_observations(
        data["problem"].K, result.rotations, result.translations, result.points,
        data["problem"].camera_indices, data["problem"].point_indices,
    )
    observation_errors = np.linalg.norm(projected - data["problem"].pixels, axis=1)
    candidates = []
    for point_id in range(1, len(result.points)):
        mask = data["problem"].point_indices == point_id
        if np.count_nonzero(mask) >= 5 and np.max(observation_errors[mask]) < 3.0:
            candidates.append((
                point_id, int(np.count_nonzero(mask)), float(np.median(observation_errors[mask]))
            ))
    candidates.sort(key=lambda item: (-item[1], item[2], item[0]))
    if len(candidates) < 4:
        raise RuntimeError("not enough stable fixture landmarks for the task target")
    selected_ids = np.array([item[0] for item in candidates[:4]])
    target_world = np.mean(result.points[selected_ids], axis=0)
    true_target_world = np.mean(data["points_true"][selected_ids], axis=0)
    final_camera = len(result.rotations) - 1
    target_final_camera = (
        result.rotations[final_camera] @ target_world
        + result.translations[final_camera]
    )
    true_target_final_camera = (
        data["rotations_true"][final_camera] @ true_target_world
        + data["translations_true"][final_camera]
    )
    centers = camera_centers(result.rotations, result.translations)
    center_errors_mm = 1000.0 * np.linalg.norm(centers - data["centers_true"], axis=1)
    rotation_errors = rotation_error_degrees(result.rotations, data["rotations_true"])

    report = {
        "application": "metric wrist-camera map for fixture inspection",
        "pose_convention": "X_camera = R_world_to_camera @ X_world + t_world_to_camera",
        "selected_landmark_ids": selected_ids.tolist(),
        "refined_target_world_m": target_world.tolist(),
        "refined_target_in_final_camera_m": target_final_camera.tolist(),
        "synthetic_truth_errors": {
            "target_world_mm": float(1000.0 * np.linalg.norm(target_world - true_target_world)),
            "target_in_final_camera_mm": float(
                1000.0 * np.linalg.norm(target_final_camera - true_target_final_camera)
            ),
            "final_camera_rotation_deg": float(rotation_errors[final_camera]),
            "final_camera_center_mm": float(center_errors_mm[final_camera]),
            "median_camera_center_mm": float(np.median(center_errors_mm)),
        },
        "optimization": {
            "observations": len(data["problem"].pixels),
            "accepted_steps": result.accepted_steps,
            "rejected_steps": result.rejected_steps,
            "true_inlier_rmse_px": metrics["true_inlier_rmse_final_px"],
        },
        "identity_contract": "Landmark identity is supplied upstream; BA only enforces multi-view geometry.",
        "safety_gate": "Do not command a robot from this pose alone; validate calibration, timestamps, observability, covariance, and hand-eye/world transforms.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
