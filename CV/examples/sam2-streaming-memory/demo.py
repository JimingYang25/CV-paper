"""Controlled streaming-memory ablations on synthetic video features."""

from __future__ import annotations

import numpy as np

from streaming_memory import binary_iou, boundary_f_score, run_stream
from synthetic_video import make_video


def evaluate(predictions: np.ndarray, targets: np.ndarray, visible_only: bool = True) -> dict[str, float]:
    visible = np.array([mask.any() for mask in targets])
    indices = np.flatnonzero(visible) if visible_only else np.arange(len(targets))
    j = np.array([binary_iou(predictions[i], targets[i]) for i in indices])
    f = np.array([boundary_f_score(predictions[i], targets[i]) for i in indices])
    return {"J": float(np.mean(j)), "F": float(np.mean(f)), "J&F": float(np.mean((j + f) / 2.0))}


def run_ablation(seed: int = 7) -> dict[str, dict[str, float]]:
    video = make_video(seed=seed)
    common = dict(features=video["features"], object_scores=video["object_scores"],
                  initial_mask=video["masks"][0])
    pinned_only, _, _ = run_stream(**common, use_recent_memory=False)
    recent_skip_absent, _, _ = run_stream(**common, use_recent_memory=True, max_recent=6,
                                          write_no_object_memory=False)
    recent_no_object, _, _ = run_stream(**common, use_recent_memory=True, max_recent=6,
                                        write_no_object_memory=True)
    corrected, _, _ = run_stream(**common, use_recent_memory=True, max_recent=6,
                                 write_no_object_memory=True,
                                 correction_masks={16: video["masks"][16]})
    return {
        "pinned_prompt_only": evaluate(pinned_only, video["masks"]),
        "streaming_fifo_skip_absent": evaluate(recent_skip_absent, video["masks"]),
        "streaming_fifo_with_no_object_memory": evaluate(recent_no_object, video["masks"]),
        "streaming_no_object_plus_correction_frame_16": evaluate(corrected, video["masks"]),
    }


def multi_seed_ablation(seeds: int = 12) -> dict[str, float]:
    streaming_gains, no_object_gains, correction_gains = [], [], []
    for seed in range(seeds):
        result = run_ablation(seed)
        base = result["pinned_prompt_only"]["J&F"]
        skipped = result["streaming_fifo_skip_absent"]["J&F"]
        no_object = result["streaming_fifo_with_no_object_memory"]["J&F"]
        corrected = result["streaming_no_object_plus_correction_frame_16"]["J&F"]
        streaming_gains.append(skipped - base)
        no_object_gains.append(no_object - skipped)
        correction_gains.append(corrected - no_object)
    return {
        "seeds": float(seeds),
        "median_recent_memory_gain_over_pinned": float(np.median(streaming_gains)),
        "median_no_object_memory_gain": float(np.median(no_object_gains)),
        "median_correction_gain": float(np.median(correction_gains)),
        "recent_memory_wins": float(np.sum(np.asarray(streaming_gains) > 0.0)),
        "no_object_memory_wins": float(np.sum(np.asarray(no_object_gains) > 0.0)),
        "correction_wins": float(np.sum(np.asarray(correction_gains) > 0.0)),
    }


def main() -> None:
    print("single-seed visible-frame segmentation metrics")
    for method, metrics in run_ablation().items():
        print(f"  {method}")
        for name, value in metrics.items():
            print(f"    {name}: {value:.6f}")
    print("\n12-seed paired mechanism study")
    for name, value in multi_seed_ablation().items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
