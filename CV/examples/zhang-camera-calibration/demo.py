"""Deterministic synthetic calibration; run `python demo.py`. No image downloads."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from calibration import board_points, initialize, project, refine, rms, constraint_matrix

SEED = 20261019
TRUE_K = np.array([[920., 0, 640], [0, 900, 480], [0, 0, 1]])
TRUE_DISTORTION = np.array([-0.12, 0.03])


def make_views(K=TRUE_K, distortion=TRUE_DISTORTION, count=12, noise=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    xyz = board_points()
    poses, observations = [], []
    for _ in range(2000):
        R = Rotation.from_euler("xyz", rng.uniform([-0.55, -0.55, -0.3], [0.55, 0.55, 0.3])).as_matrix()
        t = rng.uniform([-0.14, -0.10, 0.38], [0.14, 0.10, 0.62])
        uv = project(xyz, K, R, t, distortion)
        if np.all((uv > [20, 20]) & (uv < [1260, 940])):
            poses.append((R, t))
            observations.append(uv + rng.normal(0, noise, uv.shape))
        if len(poses) == count:
            return xyz, poses, np.stack(observations)
    raise RuntimeError("Could not sample enough visible boards")


def parallel_homographies(K=TRUE_K):
    result = []
    for angle in np.linspace(-0.5, 0.5, 12):
        R = Rotation.from_euler("z", angle).as_matrix()
        t = np.array([0.10*np.sin(angle), 0.04*np.cos(angle), 0.5+0.1*angle])
        result.append(K @ np.column_stack((R[:, :2], t)))
    return result


def run():
    xyz, true_poses, observations = make_views()
    K0, poses0, H = initialize(xyz, observations)
    # Linear solution may include skew; refinement deliberately constrains it to 0.
    K0_zero_skew = K0.copy()
    K0_zero_skew[0, 1] = 0
    initial_errors = np.array([rms(uv, project(xyz, K0_zero_skew, R, t))
                               for uv, (R, t) in zip(observations, poses0)])
    fitted = refine(xyz, observations, K0_zero_skew, poses0)
    # Known-pose synthetic check, not a real held-out-board pose estimation test.
    _, heldout_poses, _ = make_views(count=3, noise=0, seed=SEED+1)
    heldout_error = rms(
        np.stack([project(xyz, TRUE_K, R, t, TRUE_DISTORTION) for R, t in heldout_poses]),
        np.stack([project(xyz, fitted.K, R, t, fitted.distortion) for R, t in heldout_poses]))
    clean_H = [TRUE_K @ np.column_stack((R[:, :2], t)) for R, t in true_poses]
    diverse_V, parallel_V = constraint_matrix(clean_H), constraint_matrix(parallel_homographies())
    report = {
        "seed": SEED, "views": len(observations), "corners_per_view": len(xyz),
        "noise_std_per_coordinate_px": 0.15,
        "K_true": TRUE_K.tolist(), "K_linear": K0.tolist(), "K_refined": fitted.K.tolist(),
        "distortion_true": TRUE_DISTORTION.tolist(), "distortion_refined": fitted.distortion.tolist(),
        "initial_zero_skew_vector_rms_px": float(np.sqrt(np.mean(initial_errors**2))),
        "refined_vector_rms_px": float(np.sqrt(np.mean(fitted.per_view_rms**2))),
        "known_pose_synthetic_heldout_vector_rms_px": heldout_error,
        "relative_focal_error_percent": (100*np.abs(np.diag(fitted.K)[:2]/np.diag(TRUE_K)[:2]-1)).tolist(),
        "clean_diverse_constraint_rank": int(np.linalg.matrix_rank(diverse_V)),
        "clean_parallel_constraint_rank": int(np.linalg.matrix_rank(parallel_V)),
        "optimizer_nfev": fitted.optimizer.nfev,
        "optimizer_success": bool(fitted.optimizer.success),
    }
    return report, (observations, initial_errors, fitted.per_view_rms, diverse_V, parallel_V)


if __name__ == "__main__":
    report, _ = run()
    print(json.dumps(report, indent=2))
