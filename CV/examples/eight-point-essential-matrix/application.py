"""Toy robot-camera update: vision supplies direction, odometry supplies scale."""

from __future__ import annotations

import json

import numpy as np

from demo import K1, K2, synthetic_matches
from essential_geometry import (
    direction_error_degrees,
    pixels_to_normalized,
    ransac_essential_eight_point,
    rotation_error_degrees,
    select_pose_by_cheirality,
)


def main() -> None:
    data = synthetic_matches(20261121, inliers=180, outliers=70, baseline=0.28)
    p1, p2 = data["points1_px"], data["points2_px"]
    estimate = ransac_essential_eight_point(
        p1, p2, K1, K2, threshold_px=3.0, seed=20261121,
        min_draws=96, max_draws=4000,
    )
    x1 = pixels_to_normalized(p1, K1)
    x2 = pixels_to_normalized(p2, K2)
    pose = select_pose_by_cheirality(
        estimate.E, x1[estimate.inliers], x2[estimate.inliers]
    )

    # A wheel encoder, depth sensor, or known kinematic increment—not the image
    # pair—provides this magnitude. Here it is a deterministic synthetic input.
    external_baseline_m = 0.28
    t_metric_camera2_from_camera1_m = external_baseline_m * pose.t_direction

    # Camera-2 center expressed in camera-1 coordinates.  This is not ``t``.
    camera2_center_in_camera1_m = -pose.R.T @ t_metric_camera2_from_camera1_m
    report = {
        "vision_output": {
            "rotation_camera1_to_camera2": pose.R.tolist(),
            "unit_translation_camera1_to_camera2": pose.t_direction.tolist(),
            "support": int(np.count_nonzero(estimate.inliers)),
        },
        "external_scale_input_m": external_baseline_m,
        "metric_translation_camera1_to_camera2_m": t_metric_camera2_from_camera1_m.tolist(),
        "camera2_center_in_camera1_m": camera2_center_in_camera1_m.tolist(),
        "synthetic_truth_errors": {
            "rotation_deg": rotation_error_degrees(pose.R, data["R"]),
            "translation_direction_deg": direction_error_degrees(pose.t_direction, data["t"]),
        },
        "deployment_gate": "Do not update a map when inlier coverage or parallax is inadequate; support count alone is insufficient.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

