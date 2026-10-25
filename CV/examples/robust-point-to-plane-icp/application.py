"""Conservative depth-to-model tracking gate before TSDF integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robust_icp import ICPConfig, ICPResult, point_to_plane_icp, pose_error
from synthetic_scene import make_registration_case


@dataclass(frozen=True)
class TrackingDecision:
    integrate_depth: bool
    reason: str
    result: ICPResult


def track_then_gate(
    source_points: np.ndarray,
    source_normals: np.ndarray,
    model_points: np.ndarray,
    model_normals: np.ndarray,
    T_prediction: np.ndarray,
    *,
    minimum_coverage: float = 0.50,
    maximum_rmse_m: float = 0.007,
    maximum_condition_number: float = 2.0e4,
) -> TrackingDecision:
    """Estimate pose and reject unsafe TSDF integration without modifying a map."""
    result = point_to_plane_icp(
        source_points,
        model_points,
        model_normals,
        source_normals=source_normals,
        initial_T_target_from_source=T_prediction,
        config=ICPConfig(
            max_iterations=35,
            distance_threshold=0.14,
            normal_angle_deg=50.0,
            huber_delta=0.012,
            trim_fraction=0.82,
        ),
    )
    checks = [
        (result.correspondence_count >= 30, "too few geometrically valid correspondences"),
        (result.coverage >= minimum_coverage, "insufficient source-to-model coverage"),
        (result.inlier_rmse <= maximum_rmse_m, "point-to-plane residual is too large"),
        (np.isfinite(result.condition_number) and result.condition_number <= maximum_condition_number,
         "pose is weakly observable or ill-conditioned"),
    ]
    for passed, reason in checks:
        if not passed:
            return TrackingDecision(False, reason, result)
    return TrackingDecision(True, "tracking quality passed all gates", result)


def demonstration(seed: int = 11) -> dict:
    case = make_registration_case(seed=seed, outlier_fraction=0.15)
    accepted = track_then_gate(
        case["source_points"], case["source_normals"],
        case["target_points"], case["target_normals"], case["T_initial"],
    )
    rot, trans = pose_error(accepted.result.T_target_from_source, case["T_true"])

    # A bad odometric prediction is outside ICP's local basin; the gate must not
    # let that transform corrupt the persistent TSDF map.
    bad_prediction = np.array(case["T_initial"], copy=True)
    bad_prediction[:3, 3] += np.array([0.30, -0.24, 0.18])
    rejected = track_then_gate(
        case["source_points"], case["source_normals"],
        case["target_points"], case["target_normals"], bad_prediction,
    )
    return {
        "good_prediction": {
            "integrate_depth": accepted.integrate_depth,
            "reason": accepted.reason,
            "rotation_error_deg": rot,
            "translation_error_mm": 1000.0 * trans,
            "coverage": accepted.result.coverage,
            "rmse_mm": 1000.0 * accepted.result.inlier_rmse,
        },
        "bad_prediction": {
            "integrate_depth": rejected.integrate_depth,
            "reason": rejected.reason,
            "coverage": rejected.result.coverage,
            "rmse_mm": 1000.0 * rejected.result.inlier_rmse,
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(demonstration(), indent=2))
