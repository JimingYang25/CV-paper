"""Integrated example: conservative stereo clearance gate for a mobile robot."""

from __future__ import annotations

import json

import numpy as np

from demo import synthetic_stereo
from sgm_stereo import compute_left_right, disparity_to_depth


def main() -> None:
    data = synthetic_stereo(20261024)
    left, _, consistent = compute_left_right(
        data["left"],
        data["right"],
        16,
        method="census",
        census_window=(5, 5),
        p1=1.5,
        p2=8.0,
        path_count=8,
        consistency_tolerance_px=1.0,
    )

    focal_length_px = 320.0
    baseline_m = 0.075
    depth = disparity_to_depth(
        left.disparity, focal_length_px, baseline_m, valid=consistent
    )

    # An upstream detector, VLM, or navigation stack supplies this image-space
    # corridor-of-interest. SGM estimates geometry but does not infer intent.
    rows, cols = data["left"].shape
    corridor = np.zeros((rows, cols), dtype=bool)
    corridor[int(0.20 * rows):int(0.55 * rows), int(0.42 * cols):int(0.64 * cols)] = True
    confident = consistent & (left.confidence >= 1.0)
    accepted = corridor & confident & np.isfinite(depth)
    corridor_pixels = int(np.count_nonzero(corridor))
    coverage = float(np.count_nonzero(accepted) / corridor_pixels)

    if np.any(accepted):
        accepted_depth = depth[accepted]
        median_depth = float(np.median(accepted_depth))
        near_percentile = float(np.percentile(accepted_depth, 10))
        median_disparity = float(np.median(left.disparity[accepted]))
    else:
        median_depth = float("nan")
        near_percentile = float("nan")
        median_disparity = float("nan")

    stop_distance_m = 2.30
    if coverage < 0.60:
        decision = "UNKNOWN_STOP"
        reason = "insufficient consistent confident depth coverage"
    elif near_percentile < stop_distance_m:
        decision = "STOP"
        reason = "near-depth percentile violates the clearance threshold"
    else:
        decision = "PROCEED_WITH_SYSTEM_LEVEL_CHECKS"
        reason = "stereo gate passed, but control still needs calibrated system uncertainty"

    truth_depth = disparity_to_depth(
        data["disparity_true"], focal_length_px, baseline_m,
        valid=data["evaluation_mask"],
    )
    truth_evaluation = accepted & np.isfinite(truth_depth)
    depth_mae = (
        float(np.mean(np.abs(depth[truth_evaluation] - truth_depth[truth_evaluation])))
        if np.any(truth_evaluation) else float("nan")
    )

    report = {
        "application": "rectified stereo clearance gate for a mobile robot",
        "geometry_contract": {
            "disparity": "d = x_left - x_right in pixels",
            "depth": "Z = focal_length_px * baseline_m / disparity",
            "focal_length_px": focal_length_px,
            "baseline_m": baseline_m,
        },
        "upstream_contract": "The image-space corridor is supplied by navigation or a VLM; SGM does not infer semantic intent.",
        "measurement": {
            "corridor_pixels": corridor_pixels,
            "accepted_pixels": int(np.count_nonzero(accepted)),
            "coverage": coverage,
            "median_disparity_px": median_disparity,
            "median_depth_m": median_depth,
            "depth_10th_percentile_m": near_percentile,
        },
        "synthetic_truth_only": {
            "accepted_pixel_depth_mae_m": depth_mae,
        },
        "decision": decision,
        "reason": reason,
        "safety_gate": "Do not command a real robot from this demo alone; validate rectification, calibration drift, synchronization, exposure, uncertainty, and emergency stopping.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
