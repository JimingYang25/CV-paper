"""Integrated example: conservative robot sweep-volume obstacle gate."""

from __future__ import annotations

import json

import numpy as np

from demo import default_camera
from synthetic_scene import camera_pose, corrupt_depth, render_table_with_box
from tsdf_fusion import TSDFVolume, VolumeConfig


def main() -> None:
    camera = default_camera()
    config = VolumeConfig(
        origin_world=(-0.60, -0.50, 0.40),
        dimensions_xyz=(48, 40, 52),
        voxel_size=0.025,
        truncation=0.10,
        max_weight=32.0,
    )
    volume = TSDFVolume(config)
    poses = [
        camera_pose(-0.08, -0.03, 0.0),
        camera_pose(-0.04, 0.03, 0.0),
        camera_pose(0.00, -0.02, 0.0),
        camera_pose(0.04, 0.03, 0.0),
        camera_pose(0.08, -0.03, 0.0),
        camera_pose(0.00, 0.00, 0.0),
    ]
    for index, pose in enumerate(poses):
        clean, _ = render_table_with_box(camera, pose)
        depth, confidence, _ = corrupt_depth(
            clean,
            seed=51025 + index,
            noise_std=0.006,
            hole_fraction=0.025,
            outlier_fraction=0.01,
            outlier_magnitude=0.18,
        )
        volume.integrate(
            depth,
            camera,
            pose,
            confidence=confidence,
            depth_min=0.4,
            depth_max=1.8,
        )

    surface = volume.extract_zero_crossings_z(min_weight=1.0)
    sweep = (np.abs(surface.x_world) <= 0.24) & (np.abs(surface.y_world) <= 0.12)
    accepted = sweep & surface.valid & (surface.support_weight >= 2.0)
    sweep_pixels = int(np.count_nonzero(sweep))
    accepted_pixels = int(np.count_nonzero(accepted))
    coverage = accepted_pixels / sweep_pixels
    z_values = surface.z_world[accepted]
    near_percentile = float(np.percentile(z_values, 10)) if z_values.size else float("nan")

    required_coverage = 0.75
    expected_table_z = 1.30
    intrusion_margin_m = 0.15
    intrusion_threshold = expected_table_z - intrusion_margin_m
    if coverage < required_coverage:
        decision = "STOP"
        reason = "insufficient observed support in the commanded sweep volume"
    elif near_percentile < intrusion_threshold:
        decision = "STOP"
        reason = "a fused surface intrudes into the commanded sweep volume"
    else:
        decision = "CLEAR_FOR_PLANNER_REVIEW"
        reason = "coverage and geometric margin pass this illustrative gate"

    result = {
        "application": "TSDF obstacle gate for a robot end-effector sweep",
        "coordinate_contract": {
            "depth": "camera optical-axis z in metres",
            "pose": "world_from_camera",
            "volume_storage": "[z,y,x] with metric points [x,y,z]",
        },
        "upstream_contract": (
            "Calibration and poses are assumed known; a VLA or motion planner supplies "
            "the metric sweep region. TSDF fusion does not infer task intent."
        ),
        "measurement": {
            "sweep_columns": sweep_pixels,
            "accepted_columns": accepted_pixels,
            "coverage": coverage,
            "surface_z_10th_percentile_m": near_percentile,
            "expected_table_z_m": expected_table_z,
            "intrusion_threshold_z_m": intrusion_threshold,
        },
        "decision": decision,
        "reason": reason,
        "safety_gate": (
            "Do not command hardware from this example alone; validate calibration, pose "
            "drift, timing, occlusion, dynamic objects, map freshness, and emergency stopping."
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
