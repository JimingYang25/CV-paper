"""Deterministic Census + SGM experiments on rendered rectified stereo pairs."""

from __future__ import annotations

import argparse
import json

import numpy as np

from sgm_stereo import (
    compute_left_right,
    compute_sgm,
    disparity_metrics,
)


def _smooth_noise(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    values = rng.normal(size=(rows, cols))
    for _ in range(2):
        padded = np.pad(values, 1, mode="reflect")
        values = sum(
            padded[dy:dy + rows, dx:dx + cols]
            for dy in range(3) for dx in range(3)
        ) / 9.0
    values /= max(float(np.std(values)), 1e-9)
    return values


def synthetic_stereo(
    seed: int = 20261024,
    *,
    rows: int = 80,
    cols: int = 128,
    gain: float = 1.10,
    bias: float = 9.0,
    noise_std: float = 2.0,
) -> dict[str, np.ndarray | float]:
    """Render a z-buffered, integer-disparity scene with radiometric shift."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((rows, cols))
    texture = (
        118.0
        + 42.0 * _smooth_noise(rng, rows, cols)
        + 17.0 * np.sin(0.17 * x + 0.09 * y)
        + 11.0 * np.cos(0.11 * x - 0.13 * y)
    )
    texture += rng.normal(0.0, 4.0, (rows, cols))
    left = np.clip(texture, 0, 255).astype(np.uint8)

    disparity = np.full((rows, cols), 3, dtype=np.int32)
    rectangle = (
        (y >= int(0.20 * rows)) & (y < int(0.72 * rows))
        & (x >= int(0.25 * cols)) & (x < int(0.62 * cols))
    )
    disparity[rectangle] = 9
    ellipse = (
        ((x - 0.78 * cols) / (0.12 * cols)) ** 2
        + ((y - 0.52 * rows) / (0.20 * rows)) ** 2
        <= 1.0
    )
    disparity[ellipse] = 6
    small_near = (
        ((x - 0.52 * cols) / (0.08 * cols)) ** 2
        + ((y - 0.36 * rows) / (0.11 * rows)) ** 2
        <= 1.0
    )
    disparity[small_near] = 12

    right_raw = rng.uniform(70.0, 180.0, (rows, cols))
    z_buffer = np.full((rows, cols), -1, dtype=np.int32)
    owner = np.full((rows, cols), -1, dtype=np.int64)
    visible_left = np.zeros((rows, cols), dtype=bool)
    for row in range(rows):
        for col in range(cols):
            current_disparity = int(disparity[row, col])
            right_col = col - current_disparity
            if right_col < 0:
                continue
            if current_disparity >= z_buffer[row, right_col]:
                previous_owner = int(owner[row, right_col])
                if previous_owner >= 0:
                    visible_left.flat[previous_owner] = False
                z_buffer[row, right_col] = current_disparity
                owner[row, right_col] = row * cols + col
                visible_left[row, col] = True
                right_raw[row, right_col] = float(left[row, col])

    right = np.clip(
        gain * right_raw + bias + rng.normal(0.0, noise_std, (rows, cols)),
        0,
        255,
    ).astype(np.uint8)
    interior = np.zeros((rows, cols), dtype=bool)
    interior[3:rows - 3, 3:cols - 3] = True
    evaluation_mask = (
        visible_left
        & interior
        & (x >= disparity)
        & (disparity > 0)
    )
    return {
        "left": left,
        "right": right,
        "disparity_true": disparity.astype(np.float32),
        "visible_left": visible_left,
        "evaluation_mask": evaluation_mask,
        "gain": gain,
        "bias": bias,
        "noise_std": noise_std,
    }


def _metrics(result, truth: np.ndarray, evaluation_mask: np.ndarray) -> dict[str, float]:
    return disparity_metrics(result.disparity, truth, evaluation_mask)


def run_once(seed: int = 20261024) -> dict[str, object]:
    data = synthetic_stereo(seed)
    left = data["left"]
    right = data["right"]
    truth = data["disparity_true"]
    mask = data["evaluation_mask"]
    max_disparity = 16

    variants = {}
    for name, paths in (
        ("raw_winner_take_all", 0),
        ("sgm_2_paths", 2),
        ("sgm_4_paths", 4),
        ("sgm_8_paths", 8),
    ):
        result = compute_sgm(
            left, right, max_disparity,
            method="census", census_window=(5, 5),
            p1=1.5, p2=8.0, path_count=paths,
        )
        variants[name] = _metrics(result, truth, mask)

    absolute = compute_sgm(
        left, right, max_disparity,
        method="absolute", p1=5.0, p2=28.0, path_count=8,
    )
    variants["absolute_cost_8_paths"] = _metrics(absolute, truth, mask)

    left_result, _, consistent = compute_left_right(
        left, right, max_disparity,
        method="census", census_window=(5, 5),
        p1=1.5, p2=8.0, path_count=8,
        consistency_tolerance_px=1.0,
    )
    filtered = left_result.disparity.copy()
    filtered[~consistent] = np.nan
    variants["census_8_paths_left_right"] = disparity_metrics(filtered, truth, mask)

    return {
        "dataset": {
            "seed": seed,
            "shape": list(left.shape),
            "maximum_disparity_exclusive": max_disparity,
            "evaluation_pixels": int(np.count_nonzero(mask)),
            "right_gain": data["gain"],
            "right_bias": data["bias"],
            "right_noise_std": data["noise_std"],
            "disparities_present": sorted(np.unique(truth).astype(int).tolist()),
        },
        "variants": variants,
        "warning": "Synthetic rendered scene; these values are validation evidence, not a real-data benchmark.",
    }


def ablations(seed: int = 20261024, trials: int = 8) -> dict[str, object]:
    names = (
        "raw_winner_take_all",
        "sgm_2_paths",
        "sgm_4_paths",
        "sgm_8_paths",
        "absolute_cost_8_paths",
        "census_8_paths_left_right",
    )
    collected = {
        name: {"mae_px": [], "bad_1px_percent": [], "coverage": []}
        for name in names
    }
    failures = {name: 0 for name in names}
    for offset in range(trials):
        try:
            output = run_once(seed + offset)
        except (ValueError, np.linalg.LinAlgError):
            for name in names:
                failures[name] += 1
            continue
        for name in names:
            metrics = output["variants"][name]
            for metric in collected[name]:
                value = float(metrics[metric])
                if np.isfinite(value):
                    collected[name][metric].append(value)
                else:
                    failures[name] += 1

    def summarize(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"median": None, "p90": None, "worst": None}
        array = np.asarray(values)
        return {
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)),
            "worst": float(np.max(array)),
        }

    return {
        "trials": trials,
        "variants": {
            name: {
                "failures": failures[name],
                **{metric: summarize(values) for metric, values in collected[name].items()},
            }
            for name in names
        },
        "warning": "Paired synthetic seeds; not confidence intervals, runtime data, or real-domain evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()
    output = ablations(args.seed, args.trials) if args.ablations else run_once(args.seed)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
