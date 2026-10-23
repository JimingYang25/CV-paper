"""Deterministic numerical checks for Census + Semi-Global Matching."""

from __future__ import annotations

import numpy as np

from demo import synthetic_stereo
from sgm_stereo import (
    aggregate_costs,
    aggregate_direction,
    build_cost_volume,
    census_transform,
    compute_left_right,
    compute_sgm,
    disparity_metrics,
    disparity_to_depth,
    left_right_consistency,
    select_disparity,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_census_affine_intensity_invariance() -> None:
    image = (20 + np.arange(15 * 19).reshape(15, 19) % 90).astype(np.uint8)
    transformed = (2 * image.astype(np.int16) + 5).astype(np.uint8)
    first, valid_first = census_transform(image)
    second, valid_second = census_transform(transformed)
    require(np.array_equal(valid_first, valid_second), "Census valid mask")
    require(np.array_equal(first[valid_first], second[valid_second]),
            "strictly monotonic intensity mapping must preserve Census bits")


def clean_shift_pair(seed: int = 9, disparity: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, (24, 40), dtype=np.uint8)
    right = rng.integers(0, 256, left.shape, dtype=np.uint8)
    right[:, :left.shape[1] - disparity] = left[:, disparity:]
    return left, right


def check_true_census_cost_is_zero() -> None:
    left, right = clean_shift_pair()
    volume = build_cost_volume(left, right, 9, census_window=(5, 5))
    region = volume.costs[3:-3, 7:-3, 4]
    require(np.max(region) == 0.0, "clean shifted Census descriptors should match exactly")


def check_cost_volume_shapes_and_invalid_hypotheses() -> None:
    left, right = clean_shift_pair()
    volume = build_cost_volume(left, right, 9)
    require(volume.costs.shape == (24, 40, 9), "cost volume shape")
    require(volume.valid_hypotheses.shape == volume.costs.shape, "validity shape")
    require(not np.any(volume.valid_hypotheses[:, 0, 1:]),
            "left pixels cannot use disparities larger than x")


def check_hand_derived_path_recurrence() -> None:
    costs = np.array([[[0.0, 2.0, 4.0], [5.0, 1.0, 3.0]]], dtype=np.float32)
    result = aggregate_direction(costs, (0, 1), p1=1.0, p2=4.0)
    require(np.allclose(result[0, 0], costs[0, 0]), "path boundary initialization")
    require(np.allclose(result[0, 1], [5.0, 2.0, 6.0]),
            "normalized SGM recurrence")


def check_path_sum_for_constant_cost() -> None:
    costs = np.full((4, 5, 3), 2.0, dtype=np.float32)
    for paths in (1, 2, 4, 8):
        result = aggregate_costs(costs, p1=1.0, p2=4.0, path_count=paths)
        require(np.allclose(result, 2.0 * paths), f"{paths}-path constant aggregation")


def check_raw_winner_take_all_clean_shift() -> None:
    left, right = clean_shift_pair()
    result = compute_sgm(left, right, 9, path_count=0, subpixel=False)
    region = result.disparity[3:-3, 8:-3]
    require(np.mean(region == 4.0) > 0.98, "WTA should recover a clean random shift")


def check_quadratic_subpixel_minimum() -> None:
    disparities = np.arange(5, dtype=np.float32)
    costs = ((disparities - 2.25) ** 2)[None, None, :]
    estimate, _, valid = select_disparity(costs, np.ones_like(costs, dtype=bool))
    require(valid[0, 0], "parabolic example must be valid")
    require(abs(float(estimate[0, 0]) - 2.25) < 1e-6, "quadratic subpixel fit")


def check_left_right_consistency_geometry() -> None:
    left = np.full((2, 8), 2.0, dtype=np.float32)
    right = np.full((2, 8), 2.0, dtype=np.float32)
    valid = np.ones((2, 8), dtype=bool)
    mask = left_right_consistency(left, right, valid, valid, tolerance_px=0.1)
    require(not np.any(mask[:, :2]) and np.all(mask[:, 2:]), "x_right = x_left - d")
    right[0, 3] = 5.0
    mask = left_right_consistency(left, right, valid, valid, tolerance_px=0.1)
    require(not mask[0, 5], "inconsistent correspondence must be rejected")


def check_depth_formula_and_invalidity() -> None:
    disparity = np.array([[8.0, 0.0, np.nan]])
    depth = disparity_to_depth(disparity, focal_length_px=400.0, baseline_m=0.08)
    require(abs(float(depth[0, 0]) - 4.0) < 1e-12, "Z=fB/d")
    require(np.isnan(depth[0, 1]) and np.isnan(depth[0, 2]), "nonpositive disparity invalid")


def check_resize_depth_invariance() -> None:
    focal, disparity, baseline = 420.0, 7.5, 0.09
    original = focal * baseline / disparity
    scale_x = 0.625
    resized = (scale_x * focal) * baseline / (scale_x * disparity)
    require(abs(original - resized) < 1e-12, "horizontal resize must scale fx and disparity")


def check_input_guards() -> None:
    image = np.zeros((8, 9), dtype=np.uint8)
    try:
        compute_sgm(image, image[:, :-1], 4)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched stereo shapes must be rejected")
    try:
        aggregate_costs(np.zeros((2, 2, 3)), 2.0, 1.0, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("P2 smaller than P1 must be rejected")


def check_deterministic_solver() -> None:
    data = synthetic_stereo(rows=44, cols=68)
    first = compute_sgm(data["left"], data["right"], 16, path_count=4)
    second = compute_sgm(data["left"], data["right"], 16, path_count=4)
    require(np.array_equal(first.valid, second.valid), "deterministic validity")
    require(np.allclose(first.disparity, second.disparity, equal_nan=True),
            "deterministic disparity")


def check_sgm_improves_raw_matching() -> None:
    data = synthetic_stereo()
    raw = compute_sgm(data["left"], data["right"], 16, path_count=0)
    smooth = compute_sgm(data["left"], data["right"], 16, path_count=4)
    raw_metrics = disparity_metrics(raw.disparity, data["disparity_true"], data["evaluation_mask"])
    smooth_metrics = disparity_metrics(
        smooth.disparity, data["disparity_true"], data["evaluation_mask"]
    )
    require(smooth_metrics["mae_px"] < raw_metrics["mae_px"], "SGM should improve fixed WTA case")
    require(smooth_metrics["bad_1px_percent"] < raw_metrics["bad_1px_percent"],
            "SGM should reduce fixed bad-pixel rate")


def check_census_beats_absolute_under_radiometric_shift() -> None:
    data = synthetic_stereo()
    census = compute_sgm(data["left"], data["right"], 16, path_count=8)
    absolute = compute_sgm(
        data["left"], data["right"], 16,
        method="absolute", p1=5.0, p2=28.0, path_count=8,
    )
    census_metrics = disparity_metrics(
        census.disparity, data["disparity_true"], data["evaluation_mask"]
    )
    absolute_metrics = disparity_metrics(
        absolute.disparity, data["disparity_true"], data["evaluation_mask"]
    )
    require(census_metrics["mae_px"] < 0.2 * absolute_metrics["mae_px"],
            "Census should tolerate the fixed gain/bias shift better")


def check_left_right_filter_reduces_bad_pixels() -> None:
    data = synthetic_stereo()
    left, _, consistent = compute_left_right(
        data["left"], data["right"], 16, path_count=8,
    )
    unfiltered = disparity_metrics(
        left.disparity, data["disparity_true"], data["evaluation_mask"]
    )
    filtered_disparity = left.disparity.copy()
    filtered_disparity[~consistent] = np.nan
    filtered = disparity_metrics(
        filtered_disparity, data["disparity_true"], data["evaluation_mask"]
    )
    require(filtered["bad_1px_percent"] < unfiltered["bad_1px_percent"],
            "consistency filter should reduce fixed bad-pixel rate")
    require(0.90 < filtered["coverage"] < 1.0, "filter must expose its coverage tradeoff")


CHECKS = [
    check_census_affine_intensity_invariance,
    check_true_census_cost_is_zero,
    check_cost_volume_shapes_and_invalid_hypotheses,
    check_hand_derived_path_recurrence,
    check_path_sum_for_constant_cost,
    check_raw_winner_take_all_clean_shift,
    check_quadratic_subpixel_minimum,
    check_left_right_consistency_geometry,
    check_depth_formula_and_invalidity,
    check_resize_depth_invariance,
    check_input_guards,
    check_deterministic_solver,
    check_sgm_improves_raw_matching,
    check_census_beats_absolute_under_radiometric_shift,
    check_left_right_filter_reduces_bad_pixels,
]


if __name__ == "__main__":
    for check in CHECKS:
        check()
        print(f"PASS {check.__name__}")
    print(f"PASS all {len(CHECKS)} checks")
