"""Inspectable LO-RANSAC-style homography estimator, NumPy float64.

This is an educational reconstruction, not the authors' implementation,
LO+-RANSAC, OpenCV USAC, MAGSAC++, or a global-optimality certificate.
Forward transfer errors and thresholds are in DESTINATION pixel units.
"""
from dataclasses import dataclass
import itertools
import math
import numpy as np
from scipy.optimize import least_squares


def _points(xy):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.all(np.isfinite(xy)):
        raise ValueError("Expected finite [N,2] point array")
    return xy


def transfer(H, xy):
    """H[3,3], xy[N,2] -> [N,2]; points near infinity return NaNs."""
    xy = _points(xy)
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)) or not np.any(H):
        raise ValueError("Expected finite nonzero [3,3] homography")
    H = H / np.max(np.abs(H))  # make the infinity guard invariant to H's global scale
    homogeneous = np.column_stack((xy, np.ones(len(xy)))) @ H.T
    denominator = homogeneous[:, 2]
    result = np.full((len(xy), 2), np.nan)
    good = np.abs(denominator) > 1e-10
    result[good] = homogeneous[good, :2] / denominator[good, None]
    return result


def normalize_points(xy):
    center = xy.mean(axis=0)
    mean_distance = np.linalg.norm(xy-center, axis=1).mean()
    if mean_distance < 1e-12:
        raise ValueError("Coincident points")
    scale = np.sqrt(2) / mean_distance
    T = np.array([[scale, 0, -scale*center[0]],
                  [0, scale, -scale*center[1]], [0, 0, 1.]])
    return (xy-center)*scale, T


def _degenerate_minimal(xy):
    # Coordinates have already been normalized; reject any almost-collinear triple.
    for a, b, c in itertools.combinations(xy, 3):
        ab, ac = b-a, c-a
        twice_area = ab[0]*ac[1]-ab[1]*ac[0]
        if abs(twice_area) < 1e-3:
            return True
    return False


def fit_homography(source, destination):
    """Normalized DLT: [N,2] + [N,2] -> source-to-destination H[3,3]."""
    source, destination = _points(source), _points(destination)
    if source.shape != destination.shape or len(source) < 4:
        raise ValueError("Need >=4 paired correspondences")
    x, Tx = normalize_points(source)
    y, Ty = normalize_points(destination)
    if len(x) == 4 and (_degenerate_minimal(x) or _degenerate_minimal(y)):
        raise ValueError("Degenerate minimal sample")
    X, Y = x.T
    u, v = y.T
    zero, one = np.zeros(len(x)), np.ones(len(x))
    D = np.empty((2*len(x), 9))
    D[0::2] = np.column_stack((-X, -Y, -one, zero, zero, zero, u*X, u*Y, u))
    D[1::2] = np.column_stack((zero, zero, zero, -X, -Y, -one, v*X, v*Y, v))
    if np.linalg.matrix_rank(D) < 8:
        raise ValueError("Rank-deficient homography design")
    # An 8x9 matrix needs the full right basis; larger systems need only thin SVD.
    _, _, Vt = np.linalg.svd(D, full_matrices=(len(D) < 9))
    H = np.linalg.solve(Ty, Vt[-1].reshape(3, 3) @ Tx)
    scale = H[2, 2] if abs(H[2, 2]) > 1e-10 else np.linalg.norm(H)
    H /= scale
    if not np.all(np.isfinite(H)) or np.linalg.matrix_rank(H) < 3:
        raise ValueError("Singular or nonfinite homography")
    return H


def errors_squared(H, source, destination):
    predicted = transfer(H, source)
    errors = np.sum((predicted-destination)**2, axis=1)
    return np.where(np.isfinite(errors), errors, np.inf)


def quality(errors2, threshold, score="consensus"):
    """Lexicographic minimization. MSAC is optional; default is count then cost."""
    count = int(np.count_nonzero(errors2 <= threshold**2))
    cost = float(np.minimum(errors2, threshold**2).sum())
    if score == "consensus":
        return (-count, cost)
    if score == "msac":
        return (cost, -count)
    raise ValueError("score must be consensus or msac")


def all_inlier_probability(inliers, total, sample_size=4, exact=True):
    """Uniform distinct points within a trial; trials may repeat the same subset."""
    if not 0 <= inliers <= total or sample_size < 1 or total < sample_size:
        raise ValueError("Invalid counts")
    if inliers < sample_size:
        return 0.0
    if not exact:
        return (inliers/total)**sample_size
    return math.prod((inliers-j)/(total-j) for j in range(sample_size))


def required_trials(probability, confidence=0.999):
    """Nominal draws to sample an all-inlier subset, NOT model correctness."""
    if not 0 < confidence < 1 or not 0 <= probability <= 1:
        raise ValueError("Invalid probability/confidence")
    if probability == 0:
        return math.inf
    if probability == 1:
        return 1
    return math.ceil(math.log1p(-confidence)/math.log1p(-probability))


def polish_homography(H, source, destination):
    """Fixed-consensus geometric least squares; h33=1 chart only."""
    if abs(H[2, 2]) < 1e-8:
        return H
    initial = (H/H[2, 2]).ravel()[:8]

    def residual(parameters):
        candidate = np.r_[parameters, 1.].reshape(3, 3)
        error = transfer(candidate, source)-destination
        return np.nan_to_num(error, nan=1e6, posinf=1e6, neginf=-1e6).ravel()

    result = least_squares(residual, initial, method="trf", x_scale="jac",
                           loss="linear", max_nfev=100,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    if not result.success:
        return H
    return np.r_[result.x, 1.].reshape(3, 3)


def local_optimize(H, source, destination, threshold, rng, mode, score,
                   inner_trials=10, max_local_points=64):
    """Nonminimal restarts + narrowing thresholds; return best at ORIGINAL threshold.

    Differences from full published variants: unweighted DLT, width multiplier 2,
    at most 64 points per fit, four narrowing steps, no epipolar/fisheye solver.
    """
    best_H = H.copy()
    best_errors = errors_squared(H, source, destination)
    best_quality = quality(best_errors, threshold, score)
    fit_count = 0

    def fit_indices(indices):
        nonlocal fit_count
        if len(indices) < 4:
            return None
        if len(indices) > max_local_points:
            indices = rng.choice(indices, max_local_points, replace=False)
        fit_count += 1
        try:
            return fit_homography(source[indices], destination[indices])
        except (ValueError, np.linalg.LinAlgError):
            return None

    def consider(candidate):
        nonlocal best_H, best_errors, best_quality
        if candidate is None:
            return
        errors = errors_squared(candidate, source, destination)
        candidate_quality = quality(errors, threshold, score)
        if candidate_quality < best_quality:
            best_H, best_errors, best_quality = candidate, errors, candidate_quality

    initial_indices = np.flatnonzero(best_errors <= threshold**2)
    simple = fit_indices(initial_indices)
    consider(simple)
    if mode == "simple":
        return best_H, best_errors, fit_count
    if mode != "inner":
        raise ValueError("Unknown local optimization mode")
    # Wider support can recover genuine observations missed by a noisy minimal fit.
    wide = fit_indices(np.flatnonzero(best_errors <= (2*threshold)**2))
    consider(wide)
    base = np.flatnonzero(best_errors <= threshold**2)
    if len(base) < 4:
        return best_H, best_errors, fit_count
    nonminimal_size = min(12, max(4, len(base)//2))
    for _ in range(inner_trials):
        current = fit_indices(rng.choice(base, nonminimal_size, replace=False))
        consider(current)
        for width in np.linspace(2, 1, 4):
            if current is None:
                break
            errors = errors_squared(current, source, destination)
            current = fit_indices(np.flatnonzero(errors <= (width*threshold)**2))
            consider(current)
    return best_H, best_errors, fit_count


@dataclass
class RobustResult:
    H: np.ndarray
    inliers: np.ndarray
    errors2: np.ndarray
    draws: int
    valid_hypotheses: int
    rejected_samples: int
    local_calls: int
    local_fits: int
    nominal_target_draws: int
    stopped_at_cap: bool


def estimate(source, destination, threshold=3.0, seed=0, max_trials=3000,
             confidence=0.999, adaptive=True, local_mode="inner",
             final_polish=True, score="consensus", min_trials=32):
    """Estimate source-to-destination H. Returns a model-consistent inlier mask.

    Separate outer/inner RNGs preserve identical outer-sample prefixes in ablations.
    Degenerate draws count toward the budget; hence the nominal confidence is not
    a guarantee even with exact finite-population sampling probabilities.
    """
    source, destination = _points(source), _points(destination)
    if source.shape != destination.shape or len(source) < 4:
        raise ValueError("Need >=4 paired correspondences")
    if not np.isfinite(threshold) or threshold <= 0 or max_trials < 1 or min_trials < 1:
        raise ValueError("Invalid threshold/budget")
    if local_mode not in ("none", "simple", "inner"):
        raise ValueError("Unknown local mode")
    # Validate confidence/score even if all hypotheses will be rejected.
    required_trials(0.5, confidence)
    quality(np.zeros(1), threshold, score)
    outer_rng = np.random.default_rng(seed)
    inner_rng = np.random.default_rng(seed ^ 0x5A17)
    best_H, best_errors, best_quality = None, None, (math.inf, math.inf)
    best_raw_support, budget_support = 0, 0
    target = max_trials
    draws = valid = rejected = local_calls = local_fits = 0
    while draws < target:
        draws += 1
        indices = outer_rng.choice(len(source), 4, replace=False)
        try:
            H = fit_homography(source[indices], destination[indices])
        except (ValueError, np.linalg.LinAlgError):
            rejected += 1
            continue
        valid += 1
        errors = errors_squared(H, source, destination)
        count = int(np.count_nonzero(errors <= threshold**2))
        # Track the best RAW minimal support separately from the optimized support.
        if count > best_raw_support:
            best_raw_support = count
            if local_mode != "none":
                local_calls += 1
                H, errors, fits = local_optimize(H, source, destination, threshold,
                                                 inner_rng, local_mode, score)
                local_fits += fits
        candidate_quality = quality(errors, threshold, score)
        if candidate_quality < best_quality:
            best_H, best_errors, best_quality = H, errors, candidate_quality
            budget_support = max(budget_support, int(np.count_nonzero(errors <= threshold**2)))
            if adaptive:
                probability = all_inlier_probability(budget_support, len(source))
                nominal = required_trials(probability, confidence)
                target = min(max_trials, max(min_trials, nominal))
    if best_H is None:
        raise ValueError("No nondegenerate hypothesis found")
    if final_polish:
        # Fixed support during fitting, but every proposed model is rescored on ALL data.
        mask = best_errors <= threshold**2
        try:
            candidates = [fit_homography(source[mask], destination[mask])]
            candidates.append(polish_homography(candidates[0], source[mask], destination[mask]))
        except (ValueError, np.linalg.LinAlgError):
            candidates = []
        for H in candidates:
            errors = errors_squared(H, source, destination)
            candidate_quality = quality(errors, threshold, score)
            if candidate_quality < best_quality:
                best_H, best_errors, best_quality = H, errors, candidate_quality
    return RobustResult(best_H, best_errors <= threshold**2, best_errors, draws,
                        valid, rejected, local_calls, local_fits, int(target),
                        draws >= max_trials)
