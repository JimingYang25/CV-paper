"""Numerical, state, ablation, and application checks for the memory toy."""

from __future__ import annotations

import numpy as np

from application import tactile_tracking_gate
from demo import evaluate, run_ablation
from streaming_memory import (
    StreamingMemoryBank,
    binary_iou,
    boundary_f_score,
    l2_normalize,
    mask_weighted_pointer,
    run_stream,
    scaled_dot_product_attention,
    sigmoid,
    softmax,
    temporal_sine_encoding,
)
from synthetic_video import make_video


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS {name}{': ' + detail if detail else ''}")


def main() -> None:
    extreme = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    check("sigmoid finite", np.all(np.isfinite(extreme)))
    check("sigmoid midpoint", extreme[1] == 0.5)

    logits = np.array([[1.0, -2.0, 3.0], [1001.0, 998.0, 1003.0]])
    probabilities = softmax(logits)
    check("softmax sums to one", np.allclose(probabilities.sum(-1), 1.0))
    check("softmax shift invariant", np.allclose(probabilities[0], probabilities[1]))

    normalized = l2_normalize(np.array([[3.0, 4.0], [5.0, 12.0]]))
    check("L2 normalization", np.allclose(np.linalg.norm(normalized, axis=-1), 1.0))
    encoding = temporal_sine_encoding(np.array([0, 1, 2]), 8)
    check("temporal encoding shape", encoding.shape == (3, 8))
    check("age-zero sine/cosine", np.allclose(encoding[0, 0::2], 0.0)
          and np.allclose(encoding[0, 1::2], 1.0))
    try:
        temporal_sine_encoding(0, 7)
        odd_dimension_rejected = False
    except ValueError:
        odd_dimension_rejected = True
    check("odd temporal dimension rejected", odd_dimension_rejected)

    query = np.array([[1.0, 0.0]])
    key = np.array([[1.0, 0.0], [0.0, 1.0]])
    value = np.array([1.0, 0.0])
    output, peak = scaled_dot_product_attention(query, key, value)
    expected = np.exp(1 / np.sqrt(2)) / (np.exp(1 / np.sqrt(2)) + 1.0)
    check("attention numerical microcase", np.allclose(output[0], expected))
    check("attention peak diagnostic", np.allclose(peak[0], expected))
    many_query = np.repeat(query, 5, axis=0)
    chunked, _ = scaled_dot_product_attention(many_query, key, value, chunk_size=2)
    unchunked, _ = scaled_dot_product_attention(many_query, key, value, chunk_size=99)
    check("query chunking invariant", np.allclose(chunked, unchunked))
    biased, _ = scaled_dot_product_attention(query, key, value, key_bias=np.array([-3.0, 0.0]))
    check("key bias redirects attention", biased[0] < output[0])

    mask = np.array([[1, 1], [0, 0]], dtype=bool)
    disjoint = ~mask
    empty = np.zeros_like(mask)
    check("identical region IoU", binary_iou(mask, mask) == 1.0)
    check("disjoint region IoU", binary_iou(mask, disjoint) == 0.0)
    check("empty-empty IoU convention", binary_iou(empty, empty) == 1.0)
    check("identical boundary F", boundary_f_score(mask, mask) == 1.0)
    check("missing boundary F", boundary_f_score(mask, empty) == 0.0)

    features = np.zeros((2, 2, 2), dtype=float)
    features[0, 0] = [3.0, 4.0]
    pointer = mask_weighted_pointer(features, np.array([[1.0, 0.0], [0.0, 0.0]]))
    check("mask-weighted pointer", np.allclose(pointer, [0.6, 0.8]))
    check("empty-mask pointer", np.allclose(mask_weighted_pointer(features, empty), 0.0))

    rng = np.random.default_rng(3)
    bank = StreamingMemoryBank(max_recent=2, max_conditioning=2)
    small_features = l2_normalize(rng.normal(size=(2, 3, 8)))
    small_mask = np.ones((2, 3), dtype=float)
    bank.add(0, small_features, small_mask, prompted=True)
    for frame in (1, 2, 3):
        bank.add(frame, small_features, small_mask, prompted=False)
    check("conditioning memory stays pinned", [entry.frame_index for entry in bank.conditioning] == [0])
    check("recent memory obeys FIFO", [entry.frame_index for entry in bank.recent] == [2, 3])
    keys, values, pointers, biases, ages = bank.token_tensors(current_frame=4)
    check("flattened spatial token count", keys.shape == (18, 8) and values.shape == (18,))
    check("pointer token count", pointers.shape == (3, 8) and ages.shape == (3,))
    check("recency bias shape", biases.shape == (18,))
    read, diagnostics = bank.read(small_features, current_frame=4)
    check("memory read shape", read.shape == (2, 3))
    check("memory read bounded", np.all((read >= 0.0) & (read <= 1.0)))
    check("diagnostic token accounting", diagnostics["memory_tokens"] == 18.0)

    video = make_video(seed=7)
    predictions, diagnostics, result_bank = run_stream(
        video["features"], video["object_scores"], video["masks"][0],
        correction_masks={16: video["masks"][16]},
    )
    check("stream output shape", predictions.shape == video["masks"].shape)
    check("occlusion head suppresses masks", not predictions[list(video["occluded_frames"])].any())
    check("corrective prompt is pinned", [entry.frame_index for entry in result_bank.conditioning] == [0, 16])
    check("recent stream stays bounded", len(result_bank.recent) == 6)
    check("every frame has diagnostics", len(diagnostics) == len(video["features"]))
    check("visible-frame metric finite", np.isfinite(evaluate(predictions, video["masks"])["J&F"]))

    ablation = run_ablation(seed=7)
    base = ablation["pinned_prompt_only"]["J&F"]
    negative = ablation["streaming_fifo_with_no_object_memory"]["J&F"]
    corrected = ablation["streaming_no_object_plus_correction_frame_16"]["J&F"]
    check("negative memory improves controlled trial", negative > base, f"{base:.4f} -> {negative:.4f}")
    check("correction improves controlled trial", corrected > negative, f"{negative:.4f} -> {corrected:.4f}")

    without = tactile_tracking_gate(correction_frame=None)
    with_correction = tactile_tracking_gate(correction_frame=16)
    check("uncorrected tactile track is gated", not bool(without["deploy_contact_region"]), f"{without}")
    check("corrected tactile track is accepted", bool(with_correction["deploy_contact_region"]),
          f"{with_correction}")
    check("no hallucination during occlusion", with_correction["occluded_false_positive_pixels"] == 0)

    print("\n38/38 checks passed")


if __name__ == "__main__":
    main()
