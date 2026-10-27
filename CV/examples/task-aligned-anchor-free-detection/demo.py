"""Executed ablations for DFL representation and task-aligned assignment."""

from __future__ import annotations

import numpy as np

from detector_math import dfl_target, task_aligned_assign
from synthetic_scene import alignment_microcase, candidate_pool


def dfl_ablation(samples: int = 5000, seed: int = 17) -> dict[str, float]:
    """Compare representational quantization error, not trained-detector accuracy."""
    rng = np.random.default_rng(seed)
    targets = rng.uniform(0.0, 15.0, samples)
    hard_decode = np.rint(targets)
    distributions = dfl_target(targets, num_bins=16)
    expectation_decode = distributions @ np.arange(16, dtype=np.float64)
    return {
        "hard_bin_mae": float(np.mean(np.abs(hard_decode - targets))),
        "two_bin_expectation_mae": float(np.mean(np.abs(expectation_decode - targets))),
        "hard_bin_max_error": float(np.max(np.abs(hard_decode - targets))),
        "two_bin_max_error": float(np.max(np.abs(expectation_decode - targets))),
    }


def assignment_microcase() -> dict[str, float | int]:
    case = alignment_microcase()
    score_only = task_aligned_assign(**{
        "class_probabilities": case["scores"], "predicted_boxes": case["boxes"],
        "anchor_points": case["points"], "gt_labels": case["gt_labels"], "gt_boxes": case["gt_boxes"],
    }, topk=1, alpha=1.0, beta=0.0)
    aligned = task_aligned_assign(**{
        "class_probabilities": case["scores"], "predicted_boxes": case["boxes"],
        "anchor_points": case["points"], "gt_labels": case["gt_labels"], "gt_boxes": case["gt_boxes"],
    }, topk=1, alpha=1.0, beta=6.0)
    score_index = int(np.flatnonzero(score_only.assigned_gt >= 0)[0])
    aligned_index = int(np.flatnonzero(aligned.assigned_gt >= 0)[0])
    return {
        "score_only_index": score_index,
        "score_only_iou": float(score_only.overlaps[score_index, 0]),
        "task_aligned_index": aligned_index,
        "task_aligned_iou": float(aligned.overlaps[aligned_index, 0]),
    }


def assignment_ablation(seeds: int = 30, topk: int = 5) -> dict[str, float]:
    score_only_means, aligned_means = [], []
    for seed in range(seeds):
        case = candidate_pool(seed)
        kwargs = dict(
            class_probabilities=case["scores"], predicted_boxes=case["boxes"],
            anchor_points=case["points"], gt_labels=case["gt_labels"], gt_boxes=case["gt_boxes"],
            topk=topk, alpha=1.0,
        )
        score_only = task_aligned_assign(**kwargs, beta=0.0)
        aligned = task_aligned_assign(**kwargs, beta=6.0)
        for result, destination in [(score_only, score_only_means), (aligned, aligned_means)]:
            positive = np.flatnonzero(result.assigned_gt >= 0)
            destination.append(float(np.mean(result.overlaps[positive, result.assigned_gt[positive]])))
    score_only_array, aligned_array = np.asarray(score_only_means), np.asarray(aligned_means)
    return {
        "trials": float(seeds),
        "score_only_mean_iou": float(np.mean(score_only_array)),
        "task_aligned_mean_iou": float(np.mean(aligned_array)),
        "median_paired_gain": float(np.median(aligned_array - score_only_array)),
        "wins": float(np.sum(aligned_array > score_only_array)),
    }


def main() -> None:
    print("DFL representation ablation")
    for name, value in dfl_ablation().items():
        print(f"  {name}: {value:.12f}")
    print("\nScore/localization microcase")
    for name, value in assignment_microcase().items():
        print(f"  {name}: {value}")
    print("\n30-seed assignment ablation")
    for name, value in assignment_ablation().items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
