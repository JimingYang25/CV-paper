"""Integrated visual-tactile marker displacement gate after dense detection."""

from __future__ import annotations

import numpy as np

from detector_math import multiclass_nms
from synthetic_scene import marker_predictions


def marker_displacement_gate(seed: int = 3, corrupted: bool = False) -> dict[str, float | int | bool]:
    """Detect, de-duplicate, estimate motion, and gate a tactile control signal."""
    case = marker_predictions(seed=seed, corrupted=corrupted)
    boxes, scores, labels, _ = multiclass_nms(
        case["boxes"], case["scores"], score_threshold=0.25, iou_threshold=0.45
    )
    # Retain the highest-score surviving detection for every marker identity.
    current_by_label: dict[int, np.ndarray] = {}
    for box, label in zip(boxes, labels):
        current_by_label.setdefault(int(label), box)
    matched_labels = sorted(current_by_label)
    reference = case["reference_boxes"][matched_labels]
    current = np.stack([current_by_label[label] for label in matched_labels])
    reference_centers = (reference[:, :2] + reference[:, 2:]) / 2.0
    current_centers = (current[:, :2] + current[:, 2:]) / 2.0
    motions = current_centers - reference_centers
    translation = np.median(motions, axis=0)
    residuals = np.linalg.norm(motions - translation, axis=1)
    p90 = float(np.quantile(residuals, 0.90))
    support = len(matched_labels)
    deploy = bool(support >= 8 and p90 < 1.0)
    return {
        "support": support,
        "surviving_detections": len(boxes),
        "estimated_dx_px": float(translation[0]),
        "estimated_dy_px": float(translation[1]),
        "translation_error_px": float(np.linalg.norm(translation - case["true_translation"])),
        "motion_residual_p90_px": p90,
        "deploy_control_signal": deploy,
    }


def main() -> None:
    for corrupted in (False, True):
        title = "corrupted marker field" if corrupted else "healthy marker field"
        print(title)
        for name, value in marker_displacement_gate(corrupted=corrupted).items():
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
