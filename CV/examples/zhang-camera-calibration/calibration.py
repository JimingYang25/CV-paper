"""Educational Zhang initialization + joint radial-distortion refinement.

NumPy float64 throughout. Board-to-camera: Xc = R @ Xb + t.
Linear initialization supports skew; nonlinear refinement assumes zero skew.
Not an OpenCV replacement: no corner detector, outlier handling or fisheye model.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def board_points(cols=9, rows=6, spacing=0.025):
    """[rows*cols,3] board coordinates in metres; cols/rows are inner corners."""
    xy = np.stack(np.meshgrid(np.arange(cols), np.arange(rows)), axis=-1).reshape(-1, 2)
    xy = (xy - xy.mean(axis=0)) * spacing
    return np.column_stack((xy, np.zeros(len(xy))))


def project(xyz, K, R, t, distortion=(0.0, 0.0)):
    """[N,3] -> [N,2] continuous pixels; reject points behind/at the camera."""
    camera = np.asarray(xyz, dtype=np.float64) @ R.T + t
    if np.any(camera[:, 2] <= 1e-9):
        raise ValueError("All calibration points must have positive camera depth")
    xy = camera[:, :2] / camera[:, 2:3]
    radius2 = np.sum(xy * xy, axis=1)
    k1, k2 = distortion
    factor = 1 + k1 * radius2 + k2 * radius2**2
    homogeneous = np.column_stack((xy * factor[:, None], np.ones(len(xy)))) @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def normalize_points(xy):
    xy = np.asarray(xy, dtype=np.float64)
    center = xy.mean(axis=0)
    distance = np.linalg.norm(xy - center, axis=1).mean()
    if distance <= np.finfo(float).eps:
        raise ValueError("Coincident points cannot determine a homography")
    scale = np.sqrt(2) / distance
    T = np.array([[scale, 0, -scale * center[0]],
                  [0, scale, -scale * center[1]], [0, 0, 1]])
    normalized = np.column_stack((xy, np.ones(len(xy)))) @ T.T
    return normalized[:, :2], T


def homography(plane_xy, image_uv):
    """Normalized DLT, [N,2] + [N,2] -> [3,3]; N>=4, not collinear."""
    if len(plane_xy) != len(image_uv) or len(plane_xy) < 4:
        raise ValueError("Need >=4 paired points")
    plane, Tp = normalize_points(plane_xy)
    image, Ti = normalize_points(image_uv)
    X, Y = plane.T
    u, v = image.T
    ones, zeros = np.ones(len(X)), np.zeros(len(X))
    A = np.empty((2 * len(X), 9))
    A[0::2] = np.column_stack((-X, -Y, -ones, zeros, zeros, zeros, u*X, u*Y, u))
    A[1::2] = np.column_stack((zeros, zeros, zeros, -X, -Y, -ones, v*X, v*Y, v))
    if np.linalg.matrix_rank(A) < 8:
        raise ValueError("Degenerate point layout for homography")
    # Full V is essential when the minimal design matrix is 8 x 9.
    _, _, Vt = np.linalg.svd(A, full_matrices=(A.shape[0] < 9))
    H = np.linalg.solve(Ti, Vt[-1].reshape(3, 3) @ Tp)
    return H / (H[2, 2] if abs(H[2, 2]) > 1e-12 else np.linalg.norm(H))


def vij(H, i, j):
    a, b = H[:, i], H[:, j]
    return np.array([a[0]*b[0], a[0]*b[1]+a[1]*b[0], a[1]*b[1],
                     a[2]*b[0]+a[0]*b[2], a[2]*b[1]+a[1]*b[2], a[2]*b[2]])


def constraint_matrix(homographies):
    """[M,3,3] -> row-normalized [2M,6] Zhang constraint matrix."""
    V = np.stack([row for H in homographies for row in
                  (vij(H, 0, 1), vij(H, 0, 0)-vij(H, 1, 1))])
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def intrinsics_from_homographies(homographies):
    V = constraint_matrix(homographies)
    if len(homographies) < 3 or np.linalg.matrix_rank(V) < 5:
        raise ValueError("Unobservable general intrinsics: use >=3 diverse plane orientations")
    _, _, Vt = np.linalg.svd(V, full_matrices=True)
    b = Vt[-1]
    B = np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])
    if B[2, 2] < 0:
        B = -B
    # B = K^-T K^-1 = U.T @ U. Scale of B cancels on normalizing K.
    try:
        U = np.linalg.cholesky(B).T
    except np.linalg.LinAlgError as error:
        raise ValueError("Non-positive B: poor views, noise or model mismatch; do not eigen-clip") from error
    K = np.linalg.inv(U)
    return K / K[2, 2]


def pose_from_homography(H, K):
    A = np.linalg.solve(K, H)
    scale = 2 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    if scale * A[2, 2] < 0:
        scale = -scale
    r1, r2 = scale*A[:, 0], scale*A[:, 1]
    M = np.column_stack((r1, r2, np.cross(r1, r2)))
    U, _, Vt = np.linalg.svd(M)
    correction = np.diag([1, 1, np.linalg.det(U @ Vt)])
    return U @ correction @ Vt, scale * A[:, 2]


def rms(observed, predicted):
    """Pixel-vector RMS: sqrt(mean over points of du^2+dv^2)."""
    return float(np.sqrt(np.mean(np.sum((np.asarray(predicted)-observed)**2, axis=-1))))


@dataclass
class Calibration:
    K: np.ndarray
    distortion: np.ndarray
    poses: list
    per_view_rms: np.ndarray
    optimizer: object


def refine(xyz, observations, K_initial, poses_initial, distortion_initial=(0.0, 0.0),
           radial_terms=2):
    """Joint fit: M arrays [N,2], K[3,3], M*(R[3,3],t[3]) -> Calibration.

    Zero skew, two radial terms, independent equal-variance pixel noise assumed.
    Pose translations use the same units as xyz. Finite-difference Jacobian.
    """
    if radial_terms not in (0, 1, 2):
        raise ValueError("radial_terms must be 0, 1 or 2")
    observations = np.asarray(observations, dtype=np.float64)
    pose_offset = 4 + radial_terms
    initial = [np.log(K_initial[0, 0]), np.log(K_initial[1, 1]),
               K_initial[0, 2], K_initial[1, 2], *distortion_initial[:radial_terms]]
    for R, t in poses_initial:
        initial.extend(Rotation.from_matrix(R).as_rotvec())
        initial.extend(t)

    def unpack(parameters):
        fx, fy = np.exp(parameters[:2])
        cx, cy = parameters[2:4]
        distortion = np.zeros(2)
        distortion[:radial_terms] = parameters[4:pose_offset]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        poses = [(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:])
                 for p in parameters[pose_offset:].reshape(-1, 6)]
        return K, distortion, poses

    def residual(parameters):
        K, distortion, poses = unpack(parameters)
        predicted = np.stack([project(xyz, K, R, t, distortion) for R, t in poses])
        return (predicted - observations).ravel()

    # A conservative board-origin depth lower bound; all point depths are also
    # checked by project(). This does not make arbitrary initialization safe.
    lower, upper = np.full(len(initial), -np.inf), np.full(len(initial), np.inf)
    lower[pose_offset+5::6] = 1e-4
    solution = least_squares(residual, initial, bounds=(lower, upper),
                             method="trf", loss="linear", x_scale="jac",
                             ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=300)
    if not solution.success:
        raise RuntimeError(solution.message)
    K, distortion, poses = unpack(solution.x)
    errors = np.array([rms(uv, project(xyz, K, R, t, distortion))
                       for uv, (R, t) in zip(observations, poses)])
    return Calibration(K, distortion, poses, errors, solution)


def initialize(xyz, observations):
    if not np.allclose(xyz[:, 2], 0):
        raise ValueError("This initialization requires a Z=0 planar board")
    homographies = [homography(xyz[:, :2], uv) for uv in observations]
    K = intrinsics_from_homographies(homographies)
    poses = [pose_from_homography(H, K) for H in homographies]
    return K, poses, homographies


def transform_intrinsics(K, pixel_affine):
    """K' = A K for a declared continuous-pixel coordinate transform A[3,3]."""
    return np.asarray(pixel_affine) @ K
