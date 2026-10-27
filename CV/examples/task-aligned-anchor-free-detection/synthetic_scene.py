"""Deterministic synthetic cases for assignment, DFL, and detector post-processing."""

from __future__ import annotations

import numpy as np

from detector_math import box_iou


def alignment_microcase() -> dict[str, np.ndarray]:
    """A score/localization conflict whose correct choice is easy to inspect."""
    gt_boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    gt_labels = np.array([0])
    points = np.array([[2.5, 2.5], [5.0, 5.0], [7.0, 7.0]])
    boxes = np.array([
        [0.0, 0.0, 5.0, 5.0],       # score 0.99, IoU 0.25
        [0.5, 0.5, 9.5, 9.5],       # score 0.72, IoU 0.81
        [-1.0, -1.0, 9.0, 9.0],     # score 0.82, IoU about 0.68
    ])
    scores = np.array([[0.99], [0.72], [0.82]])
    return {"gt_boxes": gt_boxes, "gt_labels": gt_labels, "points": points,
            "boxes": boxes, "scores": scores}


def candidate_pool(seed: int, candidates_per_gt: int = 32) -> dict[str, np.ndarray]:
    """Make separated targets with deliberately imperfect score/IoU correlation."""
    rng = np.random.default_rng(seed)
    gt_boxes = np.array([
        [0.0, 0.0, 16.0, 14.0],
        [24.0, 0.0, 40.0, 16.0],
        [0.0, 24.0, 18.0, 40.0],
    ])
    gt_labels = np.arange(len(gt_boxes), dtype=np.int64)
    all_points, all_boxes, all_scores = [], [], []
    for class_index, gt in enumerate(gt_boxes):
        width, height = gt[2] - gt[0], gt[3] - gt[1]
        margin = 0.05 * np.array([width, height])
        points = rng.uniform(gt[:2] + margin, gt[2:] - margin,
                             size=(candidates_per_gt, 2))
        shifts = rng.normal(0.0, [3.2, 2.8], size=(candidates_per_gt, 2))
        scales = np.exp(rng.normal(0.0, 0.24, size=(candidates_per_gt, 2)))
        centers = (gt[:2] + gt[2:]) / 2.0 + shifts
        sizes = np.array([width, height]) * scales
        boxes = np.concatenate([centers - sizes / 2.0, centers + sizes / 2.0], axis=1)

        # Semantic confidence is only weakly coupled to localization: a realistic
        # source of task misalignment, with a small positive correlation plus noise.
        ious = box_iou(boxes, gt[None])[:, 0]
        own_score = np.clip(0.12 + 0.28 * ious + 0.60 * rng.random(candidates_per_gt), 0.01, 0.99)
        scores = rng.uniform(0.002, 0.06, size=(candidates_per_gt, len(gt_boxes)))
        scores[:, class_index] = own_score
        all_points.append(points); all_boxes.append(boxes); all_scores.append(scores)
    return {
        "gt_boxes": gt_boxes,
        "gt_labels": gt_labels,
        "points": np.concatenate(all_points),
        "boxes": np.concatenate(all_boxes),
        "scores": np.concatenate(all_scores),
    }


def marker_predictions(seed: int = 0, corrupted: bool = False) -> dict[str, np.ndarray]:
    """Synthetic visual-tactile marker detections with duplicate dense predictions."""
    rng = np.random.default_rng(seed)
    columns, rows = 4, 3
    centers = np.array([[12.0 + 14.0 * x, 10.0 + 14.0 * y]
                        for y in range(rows) for x in range(columns)])
    half = np.array([2.2, 2.2])
    reference_boxes = np.concatenate([centers - half, centers + half], axis=1)
    true_translation = np.array([2.4, -1.3])
    classes = len(reference_boxes)

    boxes, scores = [], []
    corrupt_ids = {1, 5, 9, 10} if corrupted else set()
    for marker_id, reference in enumerate(reference_boxes):
        displacement = true_translation + rng.normal(0.0, 0.12, size=2)
        if marker_id in corrupt_ids:
            displacement += np.array([4.5, -3.8])
        current = reference + np.tile(displacement, 2)
        for duplicate in range(3):
            jitter = rng.normal(0.0, 0.18 + 0.08 * duplicate, size=4)
            boxes.append(current + jitter)
            row = np.full(classes, 0.01)
            row[marker_id] = 0.94 - 0.11 * duplicate
            scores.append(row)
    # One confident-looking but spatially inconsistent false proposal for marker 0.
    false_box = reference_boxes[0] + np.array([9.0, 5.0, 9.0, 5.0])
    boxes.append(false_box)
    row = np.full(classes, 0.01); row[0] = 0.72; scores.append(row)
    return {
        "reference_boxes": reference_boxes,
        "true_translation": true_translation,
        "boxes": np.asarray(boxes),
        "scores": np.asarray(scores),
    }
