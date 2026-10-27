"""Numerical, shape, assignment, suppression, and application checks."""

from __future__ import annotations

import numpy as np

from application import marker_displacement_gate
from demo import assignment_ablation, assignment_microcase, dfl_ablation
from detector_math import (
    box_iou,
    box_to_distance,
    ciou_aligned,
    dfl_decode,
    dfl_loss,
    dfl_target,
    distance_to_box,
    make_anchor_points,
    multiclass_nms,
    nms,
    quality_focal_loss,
    sigmoid,
    softmax,
    task_aligned_assign,
)
from synthetic_scene import alignment_microcase


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS {name}{': ' + detail if detail else ''}")


def main() -> None:
    extreme = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    check("sigmoid finite", np.all(np.isfinite(extreme)))
    check("sigmoid midpoint", extreme[1] == 0.5)

    rng = np.random.default_rng(4)
    logits = rng.normal(size=(7, 16))
    probabilities = softmax(logits)
    check("softmax sums to one", np.allclose(probabilities.sum(-1), 1.0))
    check("softmax shift invariant", np.allclose(probabilities, softmax(logits + 91.0)))

    targets = np.array([0.0, 4.35, 7.0, 15.0])
    encoded = dfl_target(targets, 16)
    check("DFL targets sum to one", np.allclose(encoded.sum(-1), 1.0))
    check("DFL two-bin weights", np.allclose(encoded[1, [4, 5]], [0.65, 0.35]))
    check("DFL endpoint retained", encoded[-1, -1] == 1.0)
    decoded_targets = encoded @ np.arange(16)
    check("DFL target expectation exact", np.allclose(decoded_targets, targets))
    decode_logits = np.log(np.maximum(encoded, 1e-12))
    check("DFL logit decode", np.allclose(dfl_decode(decode_logits), targets, atol=2e-10))
    check("DFL optimum beats uniform", np.all(dfl_loss(decode_logits, targets) < dfl_loss(np.zeros_like(encoded), targets)))

    points, strides, levels = make_anchor_points(((2, 3), (1, 2)), (8, 16))
    check("feature-grid point count", points.shape == (8, 2) and strides.shape == (8,) and levels.shape == (8,))
    check("half-cell image coordinate", np.allclose(points[0], [4.0, 4.0]) and np.allclose(points[-1], [24.0, 8.0]))
    distances = rng.uniform(0.2, 4.0, size=(8, 4))
    boxes = distance_to_box(points, distances, strides)
    recovered = box_to_distance(points, boxes, strides)
    check("distance-box round trip", np.allclose(recovered, distances))

    base = np.array([[0.0, 0.0, 10.0, 10.0]])
    disjoint = np.array([[20.0, 20.0, 30.0, 30.0]])
    check("identical IoU", box_iou(base, base)[0, 0] == 1.0)
    check("disjoint IoU", box_iou(base, disjoint)[0, 0] == 0.0)
    random_boxes = np.array([[0, 0, 5, 6], [2, 1, 8, 9]], dtype=float)
    check("IoU transpose symmetry", np.allclose(box_iou(base, random_boxes), box_iou(random_boxes, base).T))
    check("identical CIoU", np.allclose(ciou_aligned(base, base), 1.0))
    shifted = np.array([[1.0, 0.0, 11.0, 10.0]])
    check("CIoU penalizes center distance", ciou_aligned(shifted, base)[0] < box_iou(shifted, base)[0, 0])

    q = np.array([0.2, 0.7])
    q_logits = np.log(q / (1.0 - q))
    check("QFL vanishes at matched quality", np.all(quality_focal_loss(q_logits, q) < 1e-28))
    check("QFL penalizes wrong confident prediction", quality_focal_loss(np.array([6.0]), np.array([0.0]))[0]
          > quality_focal_loss(np.array([-6.0]), np.array([0.0]))[0])

    micro = assignment_microcase()
    check("score-only chooses mislocalized point", micro["score_only_index"] == 0 and micro["score_only_iou"] == 0.25)
    check("TAL chooses localized point", micro["task_aligned_index"] == 1 and np.isclose(micro["task_aligned_iou"], 0.81))

    # One point selected by two overlapping GTs is resolved toward higher IoU.
    assignment = task_aligned_assign(
        np.array([[0.9, 0.9]]), np.array([[0.0, 0.0, 8.0, 8.0]]), np.array([[4.0, 4.0]]),
        np.array([0, 1]), np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 8.0, 8.0]]), topk=1,
    )
    check("assignment conflict uses higher IoU", assignment.assigned_gt[0] == 1)

    duplicate_boxes = np.array([[0, 0, 10, 10], [0.5, 0.5, 10.5, 10.5], [20, 20, 30, 30]], dtype=float)
    duplicate_scores = np.array([0.9, 0.8, 0.7])
    check("NMS suppresses duplicate", np.array_equal(nms(duplicate_boxes, duplicate_scores, 0.5), [0, 2]))
    class_scores = np.array([[0.9, 0.01], [0.01, 0.8], [0.7, 0.01]])
    kept = multiclass_nms(duplicate_boxes, class_scores, 0.25, 0.5)
    check("class-aware NMS retains cross-class overlap", set(kept[2].tolist()) == {0, 1})

    dfl = dfl_ablation()
    check("DFL removes target quantization in representation test", dfl["two_bin_expectation_mae"] < 1e-12
          and dfl["hard_bin_mae"] > 0.20, f"{dfl}")
    ablation = assignment_ablation()
    check("TAL wins all controlled assignment trials", ablation["wins"] == ablation["trials"], f"{ablation}")
    check("TAL raises positive localization quality", ablation["median_paired_gain"] > 0.10, f"{ablation}")

    healthy = marker_displacement_gate(corrupted=False)
    corrupted = marker_displacement_gate(corrupted=True)
    check("healthy tactile field accepted", bool(healthy["deploy_control_signal"]), f"{healthy}")
    check("healthy translation recovered", healthy["translation_error_px"] < 0.15, f"{healthy}")
    check("inconsistent tactile field rejected", not bool(corrupted["deploy_control_signal"]), f"{corrupted}")

    print("\n28/28 checks passed")


if __name__ == "__main__":
    main()
