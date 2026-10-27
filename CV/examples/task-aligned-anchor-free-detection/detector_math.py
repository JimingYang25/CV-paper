"""Transparent NumPy reference for a modern task-aligned anchor-free detector.

This is teaching code: tensors are explicit and loops are preferred where they
make assignment or NMS semantics easier to audit. Production detectors should
use vectorized framework kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax along ``axis``."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def make_anchor_points(
    feature_shapes: tuple[tuple[int, int], ...],
    strides: tuple[int, ...],
    offset: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create image-coordinate feature points.

    Returns:
        points: ``[A, 2]`` in ``(x, y)`` image pixels.
        stride_per_point: ``[A]`` in pixels.
        level_per_point: ``[A]`` integer feature-level IDs.
    """
    if len(feature_shapes) != len(strides):
        raise ValueError("feature_shapes and strides must have the same length")
    all_points, all_strides, all_levels = [], [], []
    for level, ((height, width), stride) in enumerate(zip(feature_shapes, strides)):
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        points = np.stack([(xx + offset) * stride, (yy + offset) * stride], axis=-1)
        all_points.append(points.reshape(-1, 2))
        all_strides.append(np.full(height * width, float(stride)))
        all_levels.append(np.full(height * width, level, dtype=np.int64))
    return np.concatenate(all_points), np.concatenate(all_strides), np.concatenate(all_levels)


def distance_to_box(points: np.ndarray, distances: np.ndarray, strides: np.ndarray | float = 1.0) -> np.ndarray:
    """Decode ``(left, top, right, bottom)`` feature-unit distances to ``xyxy`` pixels."""
    points = np.asarray(points, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    stride = np.asarray(strides, dtype=np.float64)
    if stride.ndim == 1:
        stride = stride[..., None]
    d = distances * stride
    return np.stack(
        [points[..., 0] - d[..., 0], points[..., 1] - d[..., 1],
         points[..., 0] + d[..., 2], points[..., 1] + d[..., 3]],
        axis=-1,
    )


def box_to_distance(
    points: np.ndarray,
    boxes: np.ndarray,
    strides: np.ndarray | float = 1.0,
    reg_max: float | None = None,
) -> np.ndarray:
    """Encode ``xyxy`` pixel boxes as point-to-side distances in feature units."""
    points = np.asarray(points, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    stride = np.asarray(strides, dtype=np.float64)
    if stride.ndim == 1:
        stride = stride[..., None]
    distances = np.stack(
        [points[..., 0] - boxes[..., 0], points[..., 1] - boxes[..., 1],
         boxes[..., 2] - points[..., 0], boxes[..., 3] - points[..., 1]],
        axis=-1,
    ) / stride
    if reg_max is not None:
        distances = np.clip(distances, 0.0, float(reg_max) - 1e-3)
    return distances


def dfl_target(target: np.ndarray, num_bins: int) -> np.ndarray:
    """Encode a continuous target as the two-bin DFL optimum distribution."""
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    target = np.clip(np.asarray(target, dtype=np.float64), 0.0, num_bins - 1.0)
    left = np.floor(target).astype(np.int64)
    right = np.minimum(left + 1, num_bins - 1)
    weight_right = target - left
    weight_left = 1.0 - weight_right
    distribution = np.zeros(target.shape + (num_bins,), dtype=np.float64)
    np.put_along_axis(distribution, left[..., None], weight_left[..., None], axis=-1)
    # At the last bin left == right, so accumulation rather than replacement is required.
    current = np.take_along_axis(distribution, right[..., None], axis=-1)
    np.put_along_axis(distribution, right[..., None], current + weight_right[..., None], axis=-1)
    return distribution


def dfl_decode(logits: np.ndarray) -> np.ndarray:
    """Decode DFL logits by the expectation of integer bin coordinates."""
    probabilities = softmax(logits, axis=-1)
    bins = np.arange(logits.shape[-1], dtype=np.float64)
    return np.sum(probabilities * bins, axis=-1)


def dfl_loss(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-target Distribution Focal Loss (two neighboring cross-entropies)."""
    distribution = dfl_target(target, logits.shape[-1])
    probabilities = softmax(logits, axis=-1)
    return -np.sum(distribution * np.log(np.maximum(probabilities, EPS)), axis=-1)


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Pairwise IoU for ``boxes1[N,4]`` and ``boxes2[M,4]`` in ``xyxy`` format."""
    boxes1 = np.asarray(boxes1, dtype=np.float64)
    boxes2 = np.asarray(boxes2, dtype=np.float64)
    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.maximum(bottom_right - top_left, 0.0)
    intersection = wh[..., 0] * wh[..., 1]
    area1 = np.prod(np.maximum(boxes1[:, 2:] - boxes1[:, :2], 0.0), axis=-1)
    area2 = np.prod(np.maximum(boxes2[:, 2:] - boxes2[:, :2], 0.0), axis=-1)
    union = area1[:, None] + area2[None, :] - intersection
    return np.where(union > 0.0, intersection / np.maximum(union, EPS), 0.0)


def ciou_aligned(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Complete IoU for aligned ``[N,4]`` box pairs."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    iou = np.diag(box_iou(predicted, target))
    pred_center = (predicted[:, :2] + predicted[:, 2:]) / 2.0
    target_center = (target[:, :2] + target[:, 2:]) / 2.0
    rho2 = np.sum((pred_center - target_center) ** 2, axis=-1)
    enclosing_tl = np.minimum(predicted[:, :2], target[:, :2])
    enclosing_br = np.maximum(predicted[:, 2:], target[:, 2:])
    c2 = np.sum((enclosing_br - enclosing_tl) ** 2, axis=-1) + EPS
    pred_wh = np.maximum(predicted[:, 2:] - predicted[:, :2], EPS)
    target_wh = np.maximum(target[:, 2:] - target[:, :2], EPS)
    v = 4.0 / np.pi**2 * (
        np.arctan(target_wh[:, 0] / target_wh[:, 1])
        - np.arctan(pred_wh[:, 0] / pred_wh[:, 1])
    ) ** 2
    alpha = v / np.maximum(1.0 - iou + v, EPS)
    return iou - rho2 / c2 - alpha * v


def quality_focal_loss(logits: np.ndarray, targets: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Quality Focal Loss for continuous targets in ``[0,1]``."""
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = sigmoid(logits)
    bce = -(targets * np.log(np.maximum(probabilities, EPS))
            + (1.0 - targets) * np.log(np.maximum(1.0 - probabilities, EPS)))
    return np.abs(targets - probabilities) ** beta * bce


@dataclass(frozen=True)
class Assignment:
    assigned_gt: np.ndarray       # [A], -1 means background
    target_scores: np.ndarray     # [A, C], IoU-valued positive class targets
    alignment: np.ndarray         # [A, G], s^alpha * IoU^beta before top-k
    overlaps: np.ndarray          # [A, G]
    selected: np.ndarray          # [A, G] after inside/top-k filtering


def task_aligned_assign(
    class_probabilities: np.ndarray,
    predicted_boxes: np.ndarray,
    anchor_points: np.ndarray,
    gt_labels: np.ndarray,
    gt_boxes: np.ndarray,
    topk: int = 10,
    alpha: float = 1.0,
    beta: float = 6.0,
) -> Assignment:
    """TOOD-style assignment with an explicit FCOS inside-box constraint.

    For each ground truth, select the top-k inside points by
    ``score_for_gt_class**alpha * IoU**beta``. If one point is selected by
    multiple targets, retain the target with the largest IoU.
    """
    scores = np.asarray(class_probabilities, dtype=np.float64)
    boxes = np.asarray(predicted_boxes, dtype=np.float64)
    points = np.asarray(anchor_points, dtype=np.float64)
    labels = np.asarray(gt_labels, dtype=np.int64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    anchors, classes = scores.shape
    ground_truths = gt_boxes.shape[0]
    if boxes.shape != (anchors, 4) or points.shape != (anchors, 2):
        raise ValueError("incompatible prediction shapes")
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("ground-truth label outside class range")

    overlaps = box_iou(boxes, gt_boxes) if ground_truths else np.empty((anchors, 0))
    selected = np.zeros((anchors, ground_truths), dtype=bool)
    alignment = np.zeros((anchors, ground_truths), dtype=np.float64)
    for gt_index in range(ground_truths):
        gt = gt_boxes[gt_index]
        inside = (
            (points[:, 0] > gt[0]) & (points[:, 0] < gt[2])
            & (points[:, 1] > gt[1]) & (points[:, 1] < gt[3])
        )
        metric = scores[:, labels[gt_index]] ** alpha * overlaps[:, gt_index] ** beta
        alignment[:, gt_index] = np.where(inside, metric, 0.0)
        candidates = np.flatnonzero(inside & (metric > 0.0))
        if candidates.size:
            count = min(int(topk), candidates.size)
            order = np.argsort(metric[candidates], kind="stable")[-count:]
            selected[candidates[order], gt_index] = True

    assigned_gt = np.full(anchors, -1, dtype=np.int64)
    for anchor_index in np.flatnonzero(np.any(selected, axis=1)):
        candidate_gts = np.flatnonzero(selected[anchor_index])
        assigned_gt[anchor_index] = candidate_gts[np.argmax(overlaps[anchor_index, candidate_gts])]

    target_scores = np.zeros_like(scores)
    positive = assigned_gt >= 0
    positive_indices = np.flatnonzero(positive)
    if positive_indices.size:
        assigned = assigned_gt[positive_indices]
        target_scores[positive_indices, labels[assigned]] = overlaps[positive_indices, assigned]
    return Assignment(assigned_gt, target_scores, alignment, overlaps, selected)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """Greedy single-class non-maximum suppression; returns kept indices."""
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        overlaps = box_iou(boxes[current:current + 1], boxes[remaining])[0]
        order = remaining[overlaps <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def multiclass_nms(
    boxes: np.ndarray,
    class_probabilities: np.ndarray,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Class-aware NMS over ``boxes[N,4]`` and ``scores[N,C]``."""
    selected: list[tuple[float, int, int]] = []
    for class_index in range(class_probabilities.shape[1]):
        candidates = np.flatnonzero(class_probabilities[:, class_index] >= score_threshold)
        if not candidates.size:
            continue
        kept_local = nms(boxes[candidates], class_probabilities[candidates, class_index], iou_threshold)
        selected.extend(
            (float(class_probabilities[candidates[i], class_index]), class_index, int(candidates[i]))
            for i in kept_local
        )
    selected.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not selected:
        return (np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    score, label, index = map(np.asarray, zip(*selected))
    index = index.astype(np.int64)
    return boxes[index], score.astype(np.float64), label.astype(np.int64), index
