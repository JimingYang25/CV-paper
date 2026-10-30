"""Numerical and behavioral checks for the ASFormer teaching reference."""

from __future__ import annotations

import numpy as np

from application import phase_assistance_gate
from asformer_math import (
    attention_pair_count,
    collapse_segments,
    cross_entropy,
    dilated_temporal_mix,
    edit_score,
    evaluate,
    hierarchical_attention_refine,
    iterative_decoder_refine,
    levenshtein,
    masked_scaled_dot_attention,
    paper_smoothing_loss,
    released_code_smoothing_loss,
    segment_f1,
    sliding_window_mask,
    softmax,
)
from synthetic_sequence import make_phase_sequence


def main() -> None:
    passed = 0

    def check(condition: bool, message: str) -> None:
        nonlocal passed
        if not condition:
            raise AssertionError(message)
        passed += 1
        print(f"[ok {passed:02d}] {message}")

    x = np.array([[1000.0, 1001.0, 999.0], [-2.0, -2.0, -2.0]])
    p = softmax(x)
    check(np.all(np.isfinite(p)), "softmax remains finite for large logits")
    check(np.allclose(p.sum(axis=1), 1.0), "softmax rows sum to one")
    check(np.allclose(p[1], 1.0 / 3.0), "equal logits produce a uniform distribution")

    mask = sliding_window_mask(7, 3)
    check(mask.shape == (7, 7), "sliding mask has [T,T] shape")
    check(np.all(mask.sum(axis=1) == 3), "interior and edge queries retain the requested width")
    check(np.array_equal(np.flatnonzero(mask[3]), np.array([2, 3, 4])), "window is centered for an odd width")
    check(np.array_equal(np.flatnonzero(mask[0]), np.array([0, 1, 2])), "boundary window shifts instead of shrinking")

    q = np.eye(4)
    out, weights = masked_scaled_dot_attention(q, q, q, sliding_window_mask(4, 2))
    check(out.shape == (4, 4), "attention preserves query length and value dimension")
    check(np.allclose(weights.sum(axis=1), 1.0), "attention weights normalize per query")
    check(np.all(weights[~sliding_window_mask(4, 2)] == 0.0), "masked pairs receive exactly zero weight")
    check(np.all(out >= 0.0), "identity-value attention yields nonnegative mixtures")

    signal = np.arange(6.0)[:, None]
    mixed = dilated_temporal_mix(signal, dilation=2)
    check(np.isclose(mixed[3, 0], 3.0), "symmetric dilated mixing preserves a linear interior signal")
    check(np.isclose(mixed[0, 0], 0.5), "edge replication is deterministic")

    constant = np.tile(np.array([[0.2, 0.8]]), (5, 1))
    check(np.isclose(paper_smoothing_loss(constant), 0.0), "paper smoothness is zero for constant probabilities")
    jump = np.array([[0.99, 0.01], [0.01, 0.99]])
    check(paper_smoothing_loss(jump) > 0.9, "paper smoothness penalizes a sharp probability jump")
    extreme_logits = np.array([[50.0, -50.0], [-50.0, 50.0]])
    check(released_code_smoothing_loss(extreme_logits) <= 16.0, "released-code log penalty is truncated at tau squared")
    check(np.isclose(cross_entropy(np.array([[0.8, 0.2], [0.1, 0.9]]), np.array([0, 1])), -np.log(0.72) / 2), "cross-entropy indexes the target class")

    check(collapse_segments([0, 0, 1, 1, 0]) == [0, 1, 0], "segment collapse removes only adjacent duplicates")
    check(levenshtein([0, 1, 2], [0, 2]) == 1, "Levenshtein distance handles deletion")
    check(np.isclose(edit_score([0, 0, 1, 1], [0, 0, 1, 1]), 100.0), "perfect segment sequence has edit score 100")
    check(np.isclose(segment_f1([0, 0, 1, 1], [0, 0, 1, 1], 0.5), 100.0), "perfect segments have F1 100")
    oversegmented = evaluate([0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 1])
    check(oversegmented.accuracy > 80.0 and oversegmented.edit < 100.0, "segment metrics expose errors hidden by high frame accuracy")

    features, logits, labels = make_phase_sequence(seed=23)
    encoder, attentions = hierarchical_attention_refine(logits, features)
    check(encoder.shape == logits.shape, "hierarchical encoder preserves [T,C]")
    check(np.allclose(encoder.sum(axis=1), 1.0), "encoder probabilities remain normalized")
    check(len(attentions) == 6 and all(a.shape == (labels.size, labels.size) for a in attentions), "one auditable attention map is returned per hierarchy level")
    stages = iterative_decoder_refine(encoder, stages=3)
    check(len(stages) == 3, "requested decoder-stage count is honored")
    check(all(np.allclose(stage.sum(axis=1), 1.0) for stage in stages), "all decoder probabilities remain normalized")
    baseline = evaluate(logits.argmax(axis=1), labels)
    refined = evaluate(stages[-1].argmax(axis=1), labels)
    check(refined.edit > baseline.edit, "refinement reduces synthetic over-segmentation by edit score")
    check(refined.f1_at_50 > baseline.f1_at_50, "refinement improves synthetic segmental F1@50")

    windows = (1, 2, 4, 8, 16, 32)
    local_pairs = attention_pair_count(labels.size, windows)
    check(local_pairs == labels.size * sum(windows), "pair counter matches exact shifted-window widths")
    check(local_pairs < len(windows) * labels.size**2, "hierarchical local attention uses fewer pairs than full attention")

    decision = phase_assistance_gate(stages[-1], 131, requested_phase=2)
    check(decision.enabled, "stable high-confidence treatment enables the illustrative gate")
    wrong_phase = phase_assistance_gate(stages[-1], 180, requested_phase=2)
    check(not wrong_phase.enabled, "wrong phase disables the illustrative gate")
    check(passed == 33, "all 34 checks completed")


if __name__ == "__main__":
    main()
