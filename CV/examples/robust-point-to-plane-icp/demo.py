"""Run robust point-to-plane ICP, baselines, and deterministic ablations."""

from __future__ import annotations

import argparse
import json

import numpy as np

from robust_icp import ICPConfig, point_to_plane_icp, point_to_point_icp, pose_error
from synthetic_scene import make_registration_case


def run_once(seed: int = 7) -> dict:
    case = make_registration_case(seed=seed)
    config = ICPConfig(
        max_iterations=45,
        distance_threshold=0.14,
        huber_delta=0.012,
        trim_fraction=0.82,
        normal_angle_deg=50.0,
    )
    robust = point_to_plane_icp(
        case["source_points"],
        case["target_points"],
        case["target_normals"],
        source_normals=case["source_normals"],
        initial_T_target_from_source=case["T_initial"],
        config=config,
    )
    plain = point_to_plane_icp(
        case["source_points"],
        case["target_points"],
        case["target_normals"],
        source_normals=case["source_normals"],
        initial_T_target_from_source=case["T_initial"],
        config=ICPConfig(
            max_iterations=45,
            distance_threshold=0.14,
            huber_delta=1e6,
            trim_fraction=1.0,
            normal_angle_deg=180.0,
        ),
    )
    p2p_T, p2p_history = point_to_point_icp(
        case["source_points"], case["target_points"],
        initial_T_target_from_source=case["T_initial"],
        max_iterations=45, distance_threshold=0.14, trim_fraction=0.82,
    )
    r_rot, r_trans = pose_error(robust.T_target_from_source, case["T_true"])
    u_rot, u_trans = pose_error(plain.T_target_from_source, case["T_true"])
    p_rot, p_trans = pose_error(p2p_T, case["T_true"])
    return {
        "seed": seed,
        "robust": {
            "rotation_error_deg": r_rot,
            "translation_error_mm": 1000.0 * r_trans,
            "iterations": len(robust.iterations),
            "inlier_rmse_mm": 1000.0 * robust.inlier_rmse,
            "coverage": robust.coverage,
            "condition_number": robust.condition_number,
            "reason": robust.reason,
        },
        "plain_point_to_plane": {
            "rotation_error_deg": u_rot,
            "translation_error_mm": 1000.0 * u_trans,
            "iterations": len(plain.iterations),
        },
        "trimmed_point_to_point": {
            "rotation_error_deg": p_rot,
            "translation_error_mm": 1000.0 * p_trans,
            "iterations": len(p2p_history),
        },
    }


def run_trials(trials: int = 12) -> dict:
    rows = [run_once(seed) for seed in range(trials)]
    summary = {}
    for method in ("robust", "plain_point_to_plane", "trimmed_point_to_point"):
        summary[method] = {
            "median_rotation_error_deg": float(np.median([r[method]["rotation_error_deg"] for r in rows])),
            "median_translation_error_mm": float(np.median([r[method]["translation_error_mm"] for r in rows])),
            "median_iterations": float(np.median([r[method]["iterations"] for r in rows])),
        }
    return {"trials": trials, "summary": summary, "per_trial": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run_trials(args.trials), indent=2))
