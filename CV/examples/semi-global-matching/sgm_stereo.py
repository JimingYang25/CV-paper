"""Educational dense stereo with Census costs and Semi-Global Matching.

Coordinate convention
---------------------
Inputs are rectified grayscale images with shape [H, W]. A pixel in the left
image at (y, x_left) corresponds to (y, x_right) in the right image. Positive
left disparity is d = x_left - x_right. Metric depth is Z = f_x * B / d.

Scope
-----
This implementation favors inspectability over speed. It stores float32 cost
volumes densely and loops over image pixels in Python while vectorizing over
disparity. It is not a reproduction of OpenCV StereoSGBM or NVIDIA VPI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


DIRECTIONS = {
    1: ((0, 1),),
    2: ((0, 1), (0, -1)),
    4: ((0, 1), (0, -1), (1, 0), (-1, 0)),
    8: (
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ),
}

POPCOUNT_TABLE = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).sum(axis=1)


@dataclass(frozen=True)
class CostVolume:
    costs: Array
    valid_hypotheses: Array
    reference_valid: Array
    method: str
    disparity_count: int


@dataclass(frozen=True)
class SGMResult:
    disparity: Array
    confidence: Array
    valid: Array
    matching_cost: Array
    aggregated_cost: Array
    path_count: int
    reference: str


def _validate_image_pair(reference_image: Array, match_image: Array) -> tuple[Array, Array]:
    reference = np.asarray(reference_image)
    match = np.asarray(match_image)
    if reference.ndim != 2 or match.ndim != 2:
        raise ValueError("stereo inputs must be grayscale arrays with shape [H,W]")
    if reference.shape != match.shape:
        raise ValueError("rectified stereo inputs must have identical shapes")
    if reference.size == 0:
        raise ValueError("stereo inputs must be non-empty")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(match)):
        raise ValueError("stereo inputs must be finite")
    reference = np.clip(reference, 0, 255).astype(np.uint8, copy=False)
    match = np.clip(match, 0, 255).astype(np.uint8, copy=False)
    return reference, match


def census_transform(image: Array, window: tuple[int, int] = (5, 5)) -> tuple[Array, Array]:
    """Return uint64 Census descriptors and a valid-center mask.

    Each bit records whether a neighbor intensity is less than the center.
    Windows may contain at most 65 pixels, including the center.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("Census input must have shape [H,W]")
    height, width = (int(window[0]), int(window[1]))
    if (
        height < 1 or width < 1 or height % 2 == 0 or width % 2 == 0
        or height * width <= 1
    ):
        raise ValueError("Census window dimensions must be odd and contain neighbors")
    bit_count = height * width - 1
    if bit_count > 64:
        raise ValueError("educational uint64 Census supports at most 64 comparisons")
    image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    rows, cols = image.shape
    radius_y, radius_x = height // 2, width // 2
    descriptor = np.zeros((rows, cols), dtype=np.uint64)
    valid = np.zeros((rows, cols), dtype=bool)
    if rows < height or cols < width:
        return descriptor, valid
    valid[radius_y:rows - radius_y, radius_x:cols - radius_x] = True
    center = image[radius_y:rows - radius_y, radius_x:cols - radius_x]
    inner = descriptor[radius_y:rows - radius_y, radius_x:cols - radius_x]
    for offset_y in range(-radius_y, radius_y + 1):
        for offset_x in range(-radius_x, radius_x + 1):
            if offset_y == 0 and offset_x == 0:
                continue
            neighbor = image[
                radius_y + offset_y:rows - radius_y + offset_y,
                radius_x + offset_x:cols - radius_x + offset_x,
            ]
            inner <<= np.uint64(1)
            inner |= (neighbor < center).astype(np.uint64)
    return descriptor, valid


def _hamming_distance(first: Array, second: Array) -> Array:
    xor = np.ascontiguousarray(np.bitwise_xor(first, second))
    bytes_view = xor.view(np.uint8).reshape(xor.shape + (8,))
    return POPCOUNT_TABLE[bytes_view].sum(axis=-1).astype(np.float32)


def build_cost_volume(
    reference_image: Array,
    match_image: Array,
    max_disparity: int,
    *,
    reference: str = "left",
    method: str = "census",
    census_window: tuple[int, int] = (5, 5),
    absolute_truncation: float = 31.0,
) -> CostVolume:
    """Build [H,W,D] costs for a left or right reference image.

    For a left reference, match x is x-d. For a right reference, match x is
    x+d while the stored disparity remains the positive left-minus-right value.
    """
    reference_image, match_image = _validate_image_pair(reference_image, match_image)
    if reference not in {"left", "right"}:
        raise ValueError("reference must be 'left' or 'right'")
    if max_disparity < 1 or max_disparity > reference_image.shape[1]:
        raise ValueError("max_disparity must be in [1,W]")
    rows, cols = reference_image.shape
    valid_hypotheses = np.zeros((rows, cols, max_disparity), dtype=bool)

    if method == "census":
        reference_descriptor, reference_valid = census_transform(reference_image, census_window)
        match_descriptor, match_valid = census_transform(match_image, census_window)
        bit_count = census_window[0] * census_window[1] - 1
        invalid_cost = float(bit_count + 64)
        costs = np.full((rows, cols, max_disparity), invalid_cost, dtype=np.float32)
        for disparity in range(max_disparity):
            if reference == "left":
                ref_slice = (slice(None), slice(disparity, cols))
                match_slice = (slice(None), slice(0, cols - disparity))
            else:
                ref_slice = (slice(None), slice(0, cols - disparity))
                match_slice = (slice(None), slice(disparity, cols))
            valid = reference_valid[ref_slice] & match_valid[match_slice]
            distances = _hamming_distance(
                reference_descriptor[ref_slice], match_descriptor[match_slice]
            )
            target = costs[:, :, disparity][ref_slice]
            target[valid] = distances[valid]
            valid_hypotheses[:, :, disparity][ref_slice] = valid
    elif method == "absolute":
        if absolute_truncation <= 0:
            raise ValueError("absolute_truncation must be positive")
        reference_valid = np.ones((rows, cols), dtype=bool)
        invalid_cost = float(absolute_truncation + 64.0)
        costs = np.full((rows, cols, max_disparity), invalid_cost, dtype=np.float32)
        for disparity in range(max_disparity):
            if reference == "left":
                ref_slice = (slice(None), slice(disparity, cols))
                match_slice = (slice(None), slice(0, cols - disparity))
            else:
                ref_slice = (slice(None), slice(0, cols - disparity))
                match_slice = (slice(None), slice(disparity, cols))
            difference = np.abs(
                reference_image[ref_slice].astype(np.float32)
                - match_image[match_slice].astype(np.float32)
            )
            costs[:, :, disparity][ref_slice] = np.minimum(difference, absolute_truncation)
            valid_hypotheses[:, :, disparity][ref_slice] = True
    else:
        raise ValueError("method must be 'census' or 'absolute'")

    return CostVolume(
        costs=costs,
        valid_hypotheses=valid_hypotheses,
        reference_valid=reference_valid,
        method=method,
        disparity_count=max_disparity,
    )


def aggregate_direction(costs: Array, direction: tuple[int, int], p1: float, p2: float) -> Array:
    """Apply the normalized SGM recurrence along one image direction."""
    costs = np.asarray(costs, dtype=np.float32)
    if costs.ndim != 3:
        raise ValueError("costs must have shape [H,W,D]")
    if p1 < 0 or p2 < p1:
        raise ValueError("penalties must satisfy 0 <= P1 <= P2")
    dy, dx = int(direction[0]), int(direction[1])
    if (dy, dx) == (0, 0) or abs(dy) > 1 or abs(dx) > 1:
        raise ValueError("direction must be a nonzero 8-neighborhood step")

    rows, cols, disparities = costs.shape
    aggregated = np.empty_like(costs)
    y_order = range(rows) if dy >= 0 else range(rows - 1, -1, -1)
    x_order = range(cols) if dx >= 0 else range(cols - 1, -1, -1)
    infinity = np.float32(1e12)

    for y in y_order:
        for x in x_order:
            previous_y, previous_x = y - dy, x - dx
            if not (0 <= previous_y < rows and 0 <= previous_x < cols):
                aggregated[y, x] = costs[y, x]
                continue
            previous = aggregated[previous_y, previous_x]
            previous_min = float(np.min(previous))
            same = previous
            lower = np.full(disparities, infinity, dtype=np.float32)
            higher = np.full(disparities, infinity, dtype=np.float32)
            lower[1:] = previous[:-1] + p1
            higher[:-1] = previous[1:] + p1
            jump = previous_min + p2
            aggregated[y, x] = (
                costs[y, x]
                + np.minimum.reduce((same, lower, higher, np.full(disparities, jump)))
                - previous_min
            )
    return aggregated


def aggregate_costs(costs: Array, p1: float, p2: float, path_count: int = 8) -> Array:
    """Sum normalized path costs. path_count=0 returns raw matching costs."""
    costs = np.asarray(costs, dtype=np.float32)
    if path_count == 0:
        return costs.copy()
    if path_count not in DIRECTIONS:
        raise ValueError("path_count must be one of 0, 1, 2, 4, or 8")
    total = np.zeros_like(costs)
    for direction in DIRECTIONS[path_count]:
        total += aggregate_direction(costs, direction, p1, p2)
    return total


def select_disparity(
    aggregated_cost: Array,
    valid_hypotheses: Array,
    *,
    subpixel: bool = True,
) -> tuple[Array, Array, Array]:
    """Winner-take-all disparity, second-best margin, and validity."""
    aggregated_cost = np.asarray(aggregated_cost, dtype=np.float32)
    valid_hypotheses = np.asarray(valid_hypotheses, dtype=bool)
    if aggregated_cost.shape != valid_hypotheses.shape or aggregated_cost.ndim != 3:
        raise ValueError("aggregated costs and validity must share shape [H,W,D]")
    masked = np.where(valid_hypotheses, aggregated_cost, np.float32(np.inf))
    best_index = np.argmin(masked, axis=2)
    best = np.take_along_axis(masked, best_index[..., None], axis=2)[..., 0]
    if masked.shape[2] > 1:
        two = np.partition(masked, 1, axis=2)[..., :2]
        with np.errstate(invalid="ignore"):
            confidence = two[..., 1] - two[..., 0]
    else:
        confidence = np.full(masked.shape[:2], np.inf, dtype=np.float32)
    valid = np.isfinite(best)
    disparity = best_index.astype(np.float32)

    if subpixel and masked.shape[2] >= 3:
        rows, cols = np.indices(best_index.shape)
        interior = valid & (best_index > 0) & (best_index < masked.shape[2] - 1)
        minus = masked[rows, cols, np.maximum(best_index - 1, 0)]
        center = masked[rows, cols, best_index]
        plus = masked[rows, cols, np.minimum(best_index + 1, masked.shape[2] - 1)]
        with np.errstate(invalid="ignore"):
            denominator = minus - 2.0 * center + plus
        fit = interior & np.isfinite(minus) & np.isfinite(plus) & (denominator > 1e-6)
        offset = np.zeros_like(disparity)
        offset[fit] = 0.5 * (minus[fit] - plus[fit]) / denominator[fit]
        disparity += np.clip(offset, -1.0, 1.0)
    disparity[~valid] = np.nan
    confidence[~valid] = 0.0
    return disparity, confidence.astype(np.float32), valid


def compute_sgm(
    reference_image: Array,
    match_image: Array,
    max_disparity: int,
    *,
    reference: str = "left",
    method: str = "census",
    census_window: tuple[int, int] = (5, 5),
    p1: float = 1.5,
    p2: float = 8.0,
    path_count: int = 8,
    subpixel: bool = True,
    minimum_confidence: float = 0.0,
) -> SGMResult:
    volume = build_cost_volume(
        reference_image,
        match_image,
        max_disparity,
        reference=reference,
        method=method,
        census_window=census_window,
    )
    aggregated = aggregate_costs(volume.costs, p1, p2, path_count)
    disparity, confidence, valid = select_disparity(
        aggregated, volume.valid_hypotheses, subpixel=subpixel
    )
    valid &= volume.reference_valid & (confidence >= minimum_confidence)
    disparity = disparity.copy()
    disparity[~valid] = np.nan
    return SGMResult(
        disparity=disparity,
        confidence=confidence,
        valid=valid,
        matching_cost=volume.costs,
        aggregated_cost=aggregated,
        path_count=path_count,
        reference=reference,
    )


def left_right_consistency(
    left_disparity: Array,
    right_disparity: Array,
    left_valid: Array,
    right_valid: Array,
    *,
    tolerance_px: float = 1.0,
) -> Array:
    """Return left pixels whose corresponding right disparity agrees."""
    left_disparity = np.asarray(left_disparity, dtype=np.float32)
    right_disparity = np.asarray(right_disparity, dtype=np.float32)
    left_valid = np.asarray(left_valid, dtype=bool)
    right_valid = np.asarray(right_valid, dtype=bool)
    if not (
        left_disparity.shape == right_disparity.shape
        == left_valid.shape == right_valid.shape
    ):
        raise ValueError("left/right disparity and validity shapes must match")
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be nonnegative")
    rows, cols = left_disparity.shape
    y, x = np.indices((rows, cols))
    safe_left = np.where(np.isfinite(left_disparity), left_disparity, 0.0)
    right_x = np.rint(x - safe_left).astype(np.int64)
    inside = (right_x >= 0) & (right_x < cols)
    clipped_x = np.clip(right_x, 0, cols - 1)
    sampled = right_disparity[y, clipped_x]
    sampled_valid = right_valid[y, clipped_x]
    return (
        left_valid
        & inside
        & sampled_valid
        & np.isfinite(sampled)
        & (np.abs(left_disparity - sampled) <= tolerance_px)
    )


def compute_left_right(
    left_image: Array,
    right_image: Array,
    max_disparity: int,
    *,
    consistency_tolerance_px: float = 1.0,
    **sgm_options,
) -> tuple[SGMResult, SGMResult, Array]:
    left = compute_sgm(
        left_image, right_image, max_disparity,
        reference="left", **sgm_options,
    )
    right = compute_sgm(
        right_image, left_image, max_disparity,
        reference="right", **sgm_options,
    )
    consistent = left_right_consistency(
        left.disparity, right.disparity, left.valid, right.valid,
        tolerance_px=consistency_tolerance_px,
    )
    return left, right, consistent


def disparity_to_depth(
    disparity: Array,
    focal_length_px: float,
    baseline_m: float,
    valid: Array | None = None,
) -> Array:
    """Convert positive disparity to depth in meters; invalid values are NaN."""
    disparity = np.asarray(disparity, dtype=np.float64)
    if focal_length_px <= 0 or baseline_m <= 0:
        raise ValueError("focal length and baseline must be positive")
    mask = np.isfinite(disparity) & (disparity > 0)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != disparity.shape:
            raise ValueError("valid mask must match disparity shape")
        mask &= valid
    depth = np.full(disparity.shape, np.nan, dtype=np.float64)
    depth[mask] = focal_length_px * baseline_m / disparity[mask]
    return depth


def disparity_metrics(estimate: Array, truth: Array, mask: Array) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    requested = np.asarray(mask, dtype=bool)
    evaluated = requested & np.isfinite(estimate) & np.isfinite(truth)
    if not np.any(evaluated):
        return {"coverage": 0.0, "mae_px": float("nan"), "bad_1px_percent": float("nan")}
    error = np.abs(estimate[evaluated] - truth[evaluated])
    return {
        "coverage": float(np.count_nonzero(evaluated) / max(1, np.count_nonzero(requested))),
        "mae_px": float(np.mean(error)),
        "bad_1px_percent": float(100.0 * np.mean(error > 1.0)),
    }
