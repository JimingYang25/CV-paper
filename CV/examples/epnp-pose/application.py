"""Synthetic robot fixture: transform a planned grasp from object to camera."""

from __future__ import annotations

import json

import numpy as np

from demo import K, synthetic_correspondences
from epnp_pose import (
    camera_center_world,
    ransac_epnp,
    rotation_error_degrees,
    translation_error,
)


def main() -> None:
    data = synthetic_correspondences(20261222, inliers=160, outliers=55, noise_px=0.55)
    result = ransac_epnp(
        data["points_world"], data["points_px"], K,
        threshold_px=2.5, min_draws=96, max_draws=3500, seed=20261222,
    )
    pose = result.pose

    # A grasp target and approach axis authored in the metric fixture frame.
    grasp_object_m = np.array([0.045, -0.018, 0.025])
    approach_object = np.array([0.0, 0.0, -1.0])
    grasp_camera_m = pose.R @ grasp_object_m + pose.t
    approach_camera = pose.R @ approach_object
    approach_camera /= np.linalg.norm(approach_camera)

    true_grasp_camera_m = data["R"] @ grasp_object_m + data["t"]
    report = {
        "pose_convention": "X_camera = R_object_to_camera @ X_object + t_object_to_camera",
        "support": int(np.count_nonzero(result.inliers)),
        "estimated_object_origin_in_camera_m": pose.t.tolist(),
        "estimated_camera_center_in_object_m": camera_center_world(pose.R, pose.t).tolist(),
        "grasp_point_object_m": grasp_object_m.tolist(),
        "grasp_point_camera_m": grasp_camera_m.tolist(),
        "approach_axis_camera_unit": approach_camera.tolist(),
        "synthetic_truth_errors": {
            "rotation_deg": rotation_error_degrees(pose.R, data["R"]),
            "translation_mm": 1000.0 * translation_error(pose.t, data["t"]),
            "grasp_point_mm": float(1000.0 * np.linalg.norm(grasp_camera_m - true_grasp_camera_m)),
        },
        "safety_gate": "A pose estimate is not collision-safe; verify calibration, residual coverage, uncertainty, and robot/world transforms.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

