"""Run paired loss-level Gram-anchoring experiments."""

from __future__ import annotations

import argparse
import json

import numpy as np

from dino_gram import gram_anchor_loss, refine_patch_features, similarity_map
from synthetic_dense import make_dense_case, shuffled_anchor


def pearson_per_image(a: np.ndarray, b: np.ndarray) -> float:
    values = []
    for ai, bi in zip(a, b):
        ai, bi = ai - ai.mean(), bi - bi.mean()
        values.append(float(np.dot(ai, bi) / max(np.linalg.norm(ai) * np.linalg.norm(bi), 1e-12)))
    return float(np.mean(values))


def topk_iou(a: np.ndarray, b: np.ndarray, k: int = 8) -> float:
    values = []
    for ai, bi in zip(a, b):
        ia = set(np.argpartition(ai, -k)[-k:].tolist())
        ib = set(np.argpartition(bi, -k)[-k:].tolist())
        values.append(len(ia & ib) / len(ia | ib))
    return float(np.mean(values))


def evaluate(features: np.ndarray, correct_anchor: np.ndarray, query: int) -> dict:
    target_map = similarity_map(correct_anchor, query)
    candidate_map = similarity_map(features, query)
    return {
        "gram_mse_to_correct_anchor": gram_anchor_loss(features, correct_anchor),
        "query_map_correlation": pearson_per_image(candidate_map, target_map),
        "query_top8_iou": topk_iou(candidate_map, target_map, 8),
    }


def run_once(seed: int = 0, steps: int = 250) -> dict:
    case = make_dense_case(seed)
    anchor = case["anchor_features"]
    degraded = case["degraded_features"]
    wrong = shuffled_anchor(anchor, seed + 991)
    refined = refine_patch_features(degraded, anchor, steps=steps)
    wrong_refined = refine_patch_features(degraded, wrong, steps=steps)
    return {
        "seed": seed,
        "baseline": evaluate(degraded, anchor, case["query_index"]),
        "correct_gram_anchor": {
            **evaluate(refined.features, anchor, case["query_index"]),
            "first_optimization_loss": refined.losses[0],
            "last_optimization_loss": refined.losses[-1],
        },
        "wrong_shuffled_anchor": {
            **evaluate(wrong_refined.features, anchor, case["query_index"]),
            "loss_to_wrong_anchor": gram_anchor_loss(wrong_refined.features, wrong),
        },
    }


def run_trials(trials: int = 12, steps: int = 250) -> dict:
    rows = [run_once(seed, steps) for seed in range(trials)]
    metrics = ("gram_mse_to_correct_anchor", "query_map_correlation", "query_top8_iou")
    summary = {}
    for variant in ("baseline", "correct_gram_anchor", "wrong_shuffled_anchor"):
        summary[variant] = {
            f"median_{metric}": float(np.median([row[variant][metric] for row in rows]))
            for metric in metrics
        }
    return {"trials": trials, "steps": steps, "summary": summary, "per_trial": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    print(json.dumps(run_trials(args.trials, args.steps), indent=2))
