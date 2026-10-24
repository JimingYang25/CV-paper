"""Executed synthetic ablations for the projective TSDF reference."""

from __future__ import annotations

import argparse
import json

import numpy as np

from synthetic_scene import camera_pose, corrupt_depth, plane_truth_on_grid, render_sloped_plane
from tsdf_fusion import PinholeCamera, TSDFVolume, VolumeConfig, surface_metrics


def default_camera() -> PinholeCamera:
    return PinholeCamera(fx=95.0, fy=95.0, cx=47.5, cy=39.5, width=96, height=80)


def default_config(truncation_voxels: float = 4.0) -> VolumeConfig:
    voxel_size = 0.025
    return VolumeConfig(
        origin_world=(-0.60, -0.50, 0.45),
        dimensions_xyz=(48, 40, 48),
        voxel_size=voxel_size,
        truncation=truncation_voxels * voxel_size,
        max_weight=32.0,
    )


def frame_poses() -> list[np.ndarray]:
    translations = [
        (-0.08, -0.04),
        (-0.05, 0.04),
        (-0.02, -0.02),
        (0.00, 0.05),
        (0.03, -0.05),
        (0.05, 0.02),
        (0.08, -0.01),
        (0.02, 0.00),
    ]
    return [camera_pose(x, y, 0.0) for x, y in translations]


def run_variant(
    *,
    seed: int,
    frames: int,
    confidence_weighted: bool,
    truncation_voxels: float = 4.0,
    pose_z_bias_m: float = 0.0,
) -> tuple[TSDFVolume, dict[str, float]]:
    camera = default_camera()
    volume = TSDFVolume(default_config(truncation_voxels))
    poses = frame_poses()[:frames]
    for index, true_pose in enumerate(poses):
        clean = render_sloped_plane(camera, true_pose)
        depth, confidence, _ = corrupt_depth(clean, seed=seed + 97 * index)
        estimated_pose = true_pose.copy()
        estimated_pose[2, 3] += pose_z_bias_m
        volume.integrate(
            depth,
            camera,
            estimated_pose,
            confidence=confidence if confidence_weighted else None,
            depth_min=0.4,
            depth_max=2.0,
        )
    surface = volume.extract_zero_crossings_z(min_weight=0.5)
    truth = plane_truth_on_grid(surface.x_world, surface.y_world)
    evaluation = (np.abs(surface.x_world) <= 0.42) & (np.abs(surface.y_world) <= 0.32)
    metrics = surface_metrics(surface, truth, evaluation)
    metrics["observed_voxel_percent"] = float(100.0 * np.mean(volume.weight > 0))
    metrics["volume_memory_mib"] = volume.memory_bytes / (1024.0**2)
    return volume, metrics


def fixed_experiment(seed: int = 20261025) -> dict[str, object]:
    variants: dict[str, dict[str, float]] = {}
    settings = {
        "single_frame_unit_weight": dict(frames=1, confidence_weighted=False),
        "eight_frames_unit_weight": dict(frames=8, confidence_weighted=False),
        "eight_frames_confidence_weighted": dict(frames=8, confidence_weighted=True),
        "eight_frames_pose_bias_4cm": dict(
            frames=8, confidence_weighted=True, pose_z_bias_m=0.04
        ),
        "eight_frames_truncation_2_voxels": dict(
            frames=8, confidence_weighted=True, truncation_voxels=2.0
        ),
        "eight_frames_truncation_8_voxels": dict(
            frames=8, confidence_weighted=True, truncation_voxels=8.0
        ),
    }
    for name, kwargs in settings.items():
        _, variants[name] = run_variant(seed=seed, **kwargs)
    return {
        "dataset": {
            "seed": seed,
            "image_shape": [80, 96],
            "frames": 8,
            "depth_noise_std_m": 0.012,
            "hole_fraction": 0.04,
            "outlier_fraction": 0.025,
            "outlier_magnitude_m": 0.22,
            "voxel_size_m": 0.025,
        },
        "variants": variants,
        "warning": "Analytic synthetic height field; not a real reconstruction benchmark.",
    }


def paired_ablation(trials: int = 8) -> dict[str, object]:
    names = [
        "single_frame_unit_weight",
        "eight_frames_unit_weight",
        "eight_frames_confidence_weighted",
        "eight_frames_pose_bias_4cm",
    ]
    values = {name: {"mae_m": [], "p95_abs_m": [], "coverage": []} for name in names}
    failures = {name: 0 for name in names}
    for trial in range(trials):
        result = fixed_experiment(seed=20261025 + 1009 * trial)["variants"]
        for name in names:
            metrics = result[name]
            if not np.isfinite(metrics["mae_m"]):
                failures[name] += 1
                continue
            for metric in values[name]:
                values[name][metric].append(metrics[metric])

    summary: dict[str, object] = {}
    for name in names:
        summary[name] = {"failures": failures[name]}
        for metric, samples in values[name].items():
            data = np.asarray(samples, dtype=np.float64)
            summary[name][metric] = {
                "median": float(np.median(data)),
                "p90": float(np.percentile(data, 90)),
                "worst": float(np.max(data) if metric != "coverage" else np.min(data)),
            }
    return {
        "trials": trials,
        "variants": summary,
        "warning": "Paired synthetic seeds; not confidence intervals or real-domain evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()
    result = paired_ablation(args.trials) if args.ablations else fixed_experiment()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
