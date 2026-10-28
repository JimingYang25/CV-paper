"""Visual-tactile marker-region propagation with drift and correction gates."""

from __future__ import annotations

import numpy as np

from streaming_memory import binary_iou, run_stream
from synthetic_video import make_video


def tactile_tracking_gate(seed: int = 21, correction_frame: int | None = 16) -> dict[str, float | int | bool]:
    """Propagate one prompted tactile contact region and gate downstream use."""
    video = make_video(seed=seed)
    corrections = {} if correction_frame is None else {correction_frame: video["masks"][correction_frame]}
    predictions, diagnostics, bank = run_stream(
        video["features"], video["object_scores"], video["masks"][0],
        use_recent_memory=True, max_recent=6, write_no_object_memory=True,
        correction_masks=corrections,
    )
    visible = np.array([mask.any() for mask in video["masks"]])
    ious = np.array([binary_iou(predictions[i], video["masks"][i]) for i in np.flatnonzero(visible)])
    occluded = np.array(video["occluded_frames"], dtype=int)
    false_positive_pixels = int(np.count_nonzero(predictions[occluded]))
    recent_quality = float(np.mean(ious[-5:]))
    minimum_quality = float(np.min(ious))
    deploy = bool(np.mean(ious) >= 0.75 and minimum_quality >= 0.45 and false_positive_pixels == 0)
    return {
        "visible_frames": int(visible.sum()),
        "mean_visible_iou": float(np.mean(ious)),
        "minimum_visible_iou": minimum_quality,
        "last_five_mean_iou": recent_quality,
        "occluded_false_positive_pixels": false_positive_pixels,
        "conditioning_memories": len(bank.conditioning),
        "recent_memories": len(bank.recent),
        "memory_writes": int(sum(item.get("memory_written", 0.0) for item in diagnostics)),
        "deploy_contact_region": deploy,
    }


def main() -> None:
    for correction in (None, 16):
        print("with correction at frame 16" if correction is not None else "without corrective prompt")
        for name, value in tactile_tracking_gate(correction_frame=correction).items():
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
