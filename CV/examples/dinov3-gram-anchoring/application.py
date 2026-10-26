"""Unlabeled dense-feature checkpoint gate for visual-tactile deployment."""

from __future__ import annotations

from dataclasses import dataclass

from demo import evaluate
from dino_gram import refine_patch_features
from synthetic_dense import make_dense_case, shuffled_anchor


@dataclass(frozen=True)
class CheckpointDecision:
    deploy: bool
    reason: str
    metrics: dict


def gate_checkpoint(candidate_features, trusted_anchor_features, query_index) -> CheckpointDecision:
    """Gate a candidate encoder checkpoint on unlabeled calibration frames."""
    metrics = evaluate(candidate_features, trusted_anchor_features, query_index)
    checks = [
        (metrics["gram_mse_to_correct_anchor"] <= 0.010, "patch-relation Gram drift is too large"),
        (metrics["query_map_correlation"] >= 0.93, "query similarity maps are inconsistent"),
        (metrics["query_top8_iou"] >= 0.70, "top-k local matches are unstable"),
    ]
    for passed, reason in checks:
        if not passed:
            return CheckpointDecision(False, reason, metrics)
    return CheckpointDecision(True, "dense-feature consistency passed every gate", metrics)


def demonstration(seed: int = 21) -> dict:
    case = make_dense_case(seed)
    anchor, degraded = case["anchor_features"], case["degraded_features"]
    refined = refine_patch_features(degraded, anchor, steps=250).features
    wrong = refine_patch_features(degraded, shuffled_anchor(anchor, seed + 99), steps=250).features
    decisions = {
        "degraded_checkpoint": gate_checkpoint(degraded, anchor, case["query_index"]),
        "correctly_anchored_checkpoint": gate_checkpoint(refined, anchor, case["query_index"]),
        "wrong_teacher_checkpoint": gate_checkpoint(wrong, anchor, case["query_index"]),
    }
    return {
        name: {"deploy": d.deploy, "reason": d.reason, **d.metrics}
        for name, d in decisions.items()
    }


if __name__ == "__main__":
    import json
    print(json.dumps(demonstration(), indent=2))
