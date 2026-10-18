"""Integrated synthetic robot-fixture example; NOT a real robot calibration.

Raw distorted pixel -> resize/crop coordinates -> inverse radial model ->
camera ray -> intersection with a known planar board in its metric frame.
Board-to-camera extrinsics are assumed known independently here.
"""
import json
import numpy as np
from scipy.optimize import root
from calibration import initialize, project, refine, transform_intrinsics
from demo import make_views, TRUE_K, TRUE_DISTORTION


def normalized_ray(uv, K, distortion):
    distorted = np.linalg.solve(K, np.r_[uv, 1.])
    distorted = distorted[:2] / distorted[2]
    k1, k2 = distortion

    def equation(xy):
        r2 = np.dot(xy, xy)
        return xy * (1+k1*r2+k2*r2*r2) - distorted

    result = root(equation, distorted, tol=1e-11)
    if not result.success or np.linalg.norm(equation(result.x)) > 1e-9:
        raise RuntimeError("Radial inverse failed; inspect field of view and camera model")
    # This is a direction with z=1, not a unit-length direction.
    return np.r_[result.x, 1.]


def intersect_board(uv, K, distortion, R_board_to_camera, t_board_to_camera):
    ray = normalized_ray(uv, K, distortion)
    R, t = R_board_to_camera, t_board_to_camera
    ray_board = R.T @ ray
    if abs(ray_board[2]) < 1e-10:
        raise ValueError("Ray is parallel to board")
    depth = (R.T @ t)[2] / ray_board[2]
    if depth <= 0:
        raise ValueError("Intersection is behind camera")
    return R.T @ (depth * ray - t)


def run():
    xyz, poses, observations = make_views()
    K0, poses0, _ = initialize(xyz, observations)
    K0[0, 1] = 0
    fitted = refine(xyz, observations, K0, poses0)
    R, t = poses[0]  # externally known fixture pose in this synthetic application
    target = np.array([0.04, -0.025, 0.])  # metres in board coordinates
    raw_pixel = project(target[None], TRUE_K, R, t, TRUE_DISTORTION)[0]
    A = np.array([[0.5, 0, -20], [0, 0.5, -10], [0, 0, 1.]])
    processed_pixel = (A @ np.r_[raw_pixel, 1.])[:2]
    reference = intersect_board(processed_pixel, transform_intrinsics(TRUE_K, A), TRUE_DISTORTION, R, t)
    np.testing.assert_allclose(reference, target, atol=1e-10)
    estimate = intersect_board(processed_pixel, transform_intrinsics(fitted.K, A), fitted.distortion, R, t)
    return {"synthetic_only": True, "raw_pixel": raw_pixel.tolist(),
            "processed_pixel": processed_pixel.tolist(), "target_board_m": target.tolist(),
            "estimated_board_m": estimate.tolist(),
            "intersection_error_mm": float(1000*np.linalg.norm(estimate-target)),
            "true_parameters_roundtrip_passed": True}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
