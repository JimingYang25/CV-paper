"""Auditable NumPy mechanisms for ASFormer-style temporal segmentation.

This is a teaching reference, not a weight-compatible reimplementation of the
authors' PyTorch repository.  It isolates the paper's important operations:
hierarchical local attention, dilated temporal mixing, iterative probability
refinement, smoothing losses, and segment-level metrics.

Tensor convention in this file is time-major: [T, D] features and [T, C]
logits/probabilities.  The official repository uses [B, D, T].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


EPS = 1e-12


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    return x - (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)))


def sliding_window_mask(length: int, window_size: int) -> np.ndarray:
    """Return [T,T] mask with exactly a local window around each query.

    For even windows, the query sees one more key on its right. Near the ends,
    the interval shifts to retain ``window_size`` valid keys where possible.
    This is an explicit, unambiguous analogue of ASFormer's blockwise sliding
    mask; it also avoids the known row-indexing bug documented by the authors.
    """
    if length <= 0 or window_size <= 0:
        raise ValueError("length and window_size must be positive")
    width = min(length, window_size)
    mask = np.zeros((length, length), dtype=bool)
    left = (width - 1) // 2
    for t in range(length):
        start = min(max(0, t - left), length - width)
        mask[t, start : start + width] = True
    return mask


def masked_scaled_dot_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute masked single-head attention.

    Args:
        query: [Tq,Dq]
        key: [Tk,Dq]
        value: [Tk,Dv]
        mask: [Tq,Tk], True for legal query-key pairs

    Returns:
        output [Tq,Dv], attention weights [Tq,Tk]
    """
    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    legal = np.asarray(mask, dtype=bool)
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("query, key, and value must be rank-2")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("incompatible Q/K/V dimensions")
    if legal.shape != (q.shape[0], k.shape[0]):
        raise ValueError("mask shape must be [Tq,Tk]")
    if np.any(~np.any(legal, axis=1)):
        raise ValueError("every query needs at least one legal key")

    scores = (q @ k.T) / np.sqrt(q.shape[1])
    scores = np.where(legal, scores, -np.inf)
    row_max = np.max(scores, axis=1, keepdims=True)
    weights = np.where(legal, np.exp(scores - row_max), 0.0)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights @ v, weights


def dilated_temporal_mix(
    x: np.ndarray,
    dilation: int,
    kernel: Sequence[float] = (0.25, 0.50, 0.25),
) -> np.ndarray:
    """Depthwise length-3 dilated temporal convolution with edge replication."""
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    if x.ndim != 2 or k.shape != (3,) or dilation < 1:
        raise ValueError("expected x [T,D], three kernel values, dilation >= 1")
    if not np.isclose(np.sum(k), 1.0):
        raise ValueError("teaching kernel must sum to one")
    t = np.arange(x.shape[0])
    left = x[np.maximum(t - dilation, 0)]
    right = x[np.minimum(t + dilation, x.shape[0] - 1)]
    return k[0] * left + k[1] * x + k[2] * right


def hierarchical_attention_refine(
    logits: np.ndarray,
    features: np.ndarray,
    windows: Iterable[int] = (1, 2, 4, 8, 16, 32),
    attention_blend: float = 0.45,
    convolution_blend: float = 0.25,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """ASFormer-inspired local-to-global encoder mechanism.

    Attention similarities come from normalized visual features while values
    are class probabilities. This deliberately does not pretend to replace
    learned Q/K/V projections; it makes the locality mechanism inspectable.
    """
    logits = np.asarray(logits, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    if logits.ndim != 2 or features.ndim != 2 or logits.shape[0] != features.shape[0]:
        raise ValueError("logits [T,C] and features [T,D] must share T")
    if not (0.0 <= attention_blend <= 1.0 and 0.0 <= convolution_blend <= 1.0):
        raise ValueError("blend weights must be in [0,1]")

    probs = softmax(logits, axis=1)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, EPS)
    attentions: list[np.ndarray] = []
    for layer, width in enumerate(windows):
        width = min(logits.shape[0], int(width))
        local = sliding_window_mask(logits.shape[0], width)
        attended, weights = masked_scaled_dot_attention(normalized, normalized, probs, local)
        mixed = dilated_temporal_mix(probs, max(1, 2**layer))
        probs = (
            (1.0 - attention_blend - convolution_blend) * probs
            + attention_blend * attended
            + convolution_blend * mixed
        )
        probs = np.maximum(probs, EPS)
        probs /= np.sum(probs, axis=1, keepdims=True)
        attentions.append(weights)
    return probs, attentions


def iterative_decoder_refine(
    encoder_probs: np.ndarray,
    stages: int = 3,
    base_blend: float = 0.42,
    decay: float = 3.0,
) -> list[np.ndarray]:
    """Probability-space refinement illustrating ASFormer's decoder cascade.

    Stage ``s`` uses alpha_s = exp(-decay*s), matching the schedule in the
    official code. The implementation keeps the value space entirely in the
    previous prediction, matching the paper's prose rather than the released
    code's observed call path; the discrepancy is discussed in the lesson.
    """
    p0 = np.asarray(encoder_probs, dtype=np.float64)
    if p0.ndim != 2 or stages < 0:
        raise ValueError("encoder_probs must be [T,C] and stages nonnegative")
    current = p0 / np.sum(p0, axis=1, keepdims=True)
    outputs: list[np.ndarray] = []
    for stage in range(stages):
        alpha = float(np.exp(-decay * stage))
        width = min(current.shape[0], 2 ** (stage + 2))
        descriptor = log_softmax(np.log(np.maximum(current, EPS)), axis=1)
        mask = sliding_window_mask(current.shape[0], width)
        attended, _ = masked_scaled_dot_attention(descriptor, descriptor, current, mask)
        local = dilated_temporal_mix(current, 2**stage)
        proposal = 0.5 * attended + 0.5 * local
        blend = min(0.95, base_blend * (0.55 + 0.45 * alpha))
        current = (1.0 - blend) * current + blend * proposal
        current = np.maximum(current, EPS)
        current /= np.sum(current, axis=1, keepdims=True)
        outputs.append(current.copy())
    return outputs


def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or labels.shape != (probs.shape[0],):
        raise ValueError("expected probabilities [T,C] and labels [T]")
    return float(-np.mean(np.log(np.maximum(probs[np.arange(probs.shape[0]), labels], EPS))))


def paper_smoothing_loss(probs: np.ndarray) -> float:
    """Equation printed in the paper: mean squared adjacent probability change."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] < 2:
        raise ValueError("expected probabilities [T,C] with T >= 2")
    return float(np.mean((probs[1:] - probs[:-1]) ** 2))


def released_code_smoothing_loss(logits: np.ndarray, tau: float = 4.0) -> float:
    """Released-code variant: truncated squared adjacent log-probability change."""
    lp = log_softmax(np.asarray(logits, dtype=np.float64), axis=1)
    if lp.ndim != 2 or lp.shape[0] < 2 or tau <= 0:
        raise ValueError("expected logits [T,C], T >= 2, tau > 0")
    return float(np.mean(np.minimum((lp[1:] - lp[:-1]) ** 2, tau**2)))


def collapse_segments(labels: Sequence[int]) -> list[int]:
    arr = np.asarray(labels, dtype=np.int64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("labels must be a nonempty vector")
    return arr[np.r_[True, arr[1:] != arr[:-1]]].tolist()


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    previous = np.arange(len(b) + 1, dtype=np.int64)
    for i, ai in enumerate(a, start=1):
        current = np.empty(len(b) + 1, dtype=np.int64)
        current[0] = i
        for j, bj in enumerate(b, start=1):
            current[j] = min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (ai != bj))
        previous = current
    return int(previous[-1])


def edit_score(prediction: Sequence[int], target: Sequence[int]) -> float:
    p, y = collapse_segments(prediction), collapse_segments(target)
    return 100.0 * (1.0 - levenshtein(p, y) / max(len(p), len(y)))


def _segments(labels: Sequence[int]) -> list[tuple[int, int, int]]:
    arr = np.asarray(labels, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(arr[1:] != arr[:-1]) + 1]
    ends = np.r_[starts[1:], arr.size]
    return [(int(arr[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def segment_f1(prediction: Sequence[int], target: Sequence[int], overlap: float = 0.5) -> float:
    pred, gt = _segments(prediction), _segments(target)
    used: set[int] = set()
    true_positive = 0
    for plabel, ps, pe in pred:
        best_iou, best_index = 0.0, -1
        for index, (glabel, gs, ge) in enumerate(gt):
            if glabel != plabel or index in used:
                continue
            intersection = max(0, min(pe, ge) - max(ps, gs))
            union = max(pe, ge) - min(ps, gs)
            iou = intersection / union
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_iou >= overlap:
            true_positive += 1
            used.add(best_index)
    false_positive = len(pred) - true_positive
    false_negative = len(gt) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    return 100.0 * (2 * true_positive / denominator if denominator else 1.0)


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    edit: float
    f1_at_50: float
    segments: int


def evaluate(prediction: Sequence[int], target: Sequence[int]) -> Metrics:
    p = np.asarray(prediction, dtype=np.int64)
    y = np.asarray(target, dtype=np.int64)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("prediction and target must have the same vector shape")
    return Metrics(
        accuracy=100.0 * float(np.mean(p == y)),
        edit=edit_score(p, y),
        f1_at_50=segment_f1(p, y, 0.50),
        segments=len(collapse_segments(p)),
    )


def attention_pair_count(length: int, windows: Iterable[int]) -> int:
    """Exact legal Q-K pairs for this reference's shifted local masks."""
    return int(sum(np.sum(sliding_window_mask(length, int(w))) for w in windows))
