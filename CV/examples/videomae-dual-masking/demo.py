"""Controlled leakage and dual-decoder coverage experiments."""

from __future__ import annotations

import numpy as np

from synthetic_video import make_motion_video, make_redundant_video
from videomae_math import (
    attention_pair_counts,
    cubeify,
    dual_mask_selection,
    masked_mse,
    normalize_cubes,
    random_mask,
    running_cell_keep,
    temporal_copy_probe,
    tube_mask,
)


def leakage_trial(seed: int, ratio: float = 0.90) -> dict[str, float]:
    video = make_redundant_video(seed)
    cubes, grid = cubeify(video)
    # Raw-pixel copy error isolates temporal leakage. The actual pre-training
    # loss below uses normalized cube pixels, matching the official objective.
    target = cubes[0]
    rng = np.random.default_rng(10_000 + seed)
    masks = {
        "random": random_mask(grid, ratio, rng),
        "tube": tube_mask(grid, ratio, rng),
    }
    result = {}
    for name, mask in masks.items():
        prediction = temporal_copy_probe(target, grid, mask)
        result[f"{name}_copy_mse"] = masked_mse(target, prediction, mask.reshape(-1))
        result[f"{name}_mask_ratio"] = float(mask.mean())
    result["tube_minus_random_mse"] = result["tube_copy_mse"] - result["random_copy_mse"]
    return result


def leakage_study(seeds: int = 20) -> dict[str, float]:
    trials = [leakage_trial(seed) for seed in range(seeds)]
    gaps = np.array([item["tube_minus_random_mse"] for item in trials])
    return {
        "seeds": float(seeds),
        "mean_random_copy_mse": float(np.mean([item["random_copy_mse"] for item in trials])),
        "mean_tube_copy_mse": float(np.mean([item["tube_copy_mse"] for item in trials])),
        "median_tube_minus_random_mse": float(np.median(gaps)),
        "tube_is_harder_count": float(np.sum(gaps > 0.0)),
    }


def dual_mask_study(seed: int = 7, period: int = 2) -> dict[str, float | int]:
    video = make_motion_video(seed)
    cubes, grid = cubeify(video)
    target, _, _ = normalize_cubes(cubes)
    target = target[0]
    encoder_mask = tube_mask(grid, 0.90, np.random.default_rng(seed))
    prediction = temporal_copy_probe(target, grid, encoder_mask)
    full_loss = masked_mse(target, prediction, encoder_mask.reshape(-1))

    squared = np.mean((target - prediction) ** 2, axis=-1).reshape(grid)
    subset_sum, subset_count = 0.0, 0
    all_supervised = np.zeros(grid, dtype=int)
    first_decoder = None
    for offset in range(period):
        decoder_keep = running_cell_keep(grid, period=period, offset=offset)
        decoder_input, supervised = dual_mask_selection(encoder_mask, decoder_keep)
        if first_decoder is None:
            first_decoder = decoder_input
        subset_sum += float(squared[supervised].sum())
        subset_count += int(supervised.sum())
        all_supervised += supervised
    cycled_loss = subset_sum / subset_count
    counts = attention_pair_counts(encoder_mask, first_decoder)
    return {
        "full_masked_loss": full_loss,
        "cycled_dual_loss": cycled_loss,
        "absolute_loss_difference": abs(full_loss - cycled_loss),
        "masked_tokens_covered_once": int(np.all(all_supervised[encoder_mask] == 1)),
        **counts,
    }


def main() -> None:
    print("20-seed temporal-copy leakage probe")
    for name, value in leakage_study().items():
        print(f"  {name}: {value:.6f}")
    print("\ndual-decoder cyclic coverage")
    for name, value in dual_mask_study().items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
