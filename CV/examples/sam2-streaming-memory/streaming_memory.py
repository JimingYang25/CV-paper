"""Auditable NumPy reference for streaming spatial memory attention.

This is not SAM 2 and contains no Meta weights.  It isolates the mechanisms an
engineer must understand: spatial memory, object pointers, conditioning-frame
retention, recent-frame FIFO memory, temporal encodings, and occlusion-gated
memory writes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), EPS)


def temporal_sine_encoding(age: int | np.ndarray, dim: int) -> np.ndarray:
    """Standard sine/cosine encoding for non-negative relative frame age."""
    if dim % 2:
        raise ValueError("temporal encoding dimension must be even")
    age = np.asarray(age, dtype=np.float64)[..., None]
    frequency = np.exp(-np.log(10000.0) * np.arange(0, dim, 2) / dim)
    angle = age * frequency
    encoding = np.empty(age.shape[:-1] + (dim,), dtype=np.float64)
    encoding[..., 0::2] = np.sin(angle)
    encoding[..., 1::2] = np.cos(angle)
    return encoding


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    logit_scale: float = 1.0,
    chunk_size: int = 256,
    key_bias: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``softmax(scale * QK^T/sqrt(D))V`` in query chunks.

    Args:
        query: ``[Nq, D]``.
        key: ``[Nk, D]``.
        value: ``[Nk, Dv]`` or ``[Nk]``.
        logit_scale: extra multiplicative sharpness applied to standard logits.
        chunk_size: number of query rows processed at a time.

    Returns:
        output: ``[Nq, Dv]``.
        peak_weight: ``[Nq]`` maximum attention probability, useful only as a
            diagnostic; it is not a calibrated confidence.
    """
    query = np.asarray(query, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    if query.ndim != 2 or key.ndim != 2 or query.shape[1] != key.shape[1]:
        raise ValueError("query and key must be [tokens, common_dim]")
    squeeze = value.ndim == 1
    if squeeze:
        value = value[:, None]
    if value.shape[0] != key.shape[0]:
        raise ValueError("key/value token counts differ")
    outputs, peaks = [], []
    scale = float(logit_scale) / np.sqrt(query.shape[1])
    if key_bias is not None:
        key_bias = np.asarray(key_bias, dtype=np.float64).reshape(1, -1)
        if key_bias.shape[1] != len(key):
            raise ValueError("key_bias must have one value per key token")
    for start in range(0, len(query), int(chunk_size)):
        logits = query[start:start + chunk_size] @ key.T * scale
        if key_bias is not None:
            logits = logits + key_bias
        weights = softmax(logits, axis=-1)
        outputs.append(weights @ value)
        peaks.append(np.max(weights, axis=-1))
    output = np.concatenate(outputs, axis=0)
    if squeeze:
        output = output[:, 0]
    return output, np.concatenate(peaks)


def binary_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = np.count_nonzero(predicted & target)
    union = np.count_nonzero(predicted | target)
    return 1.0 if union == 0 else float(intersection / union)


def boundary_map(mask: np.ndarray) -> np.ndarray:
    """Four-neighbour binary boundary without SciPy."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior


def boundary_f_score(predicted: np.ndarray, target: np.ndarray, tolerance: int = 1) -> float:
    """Boundary F score with square dilation tolerance."""
    pred_b = boundary_map(predicted)
    target_b = boundary_map(target)
    if not pred_b.any() and not target_b.any():
        return 1.0
    if not pred_b.any() or not target_b.any():
        return 0.0

    def dilate(mask: np.ndarray) -> np.ndarray:
        result = np.zeros_like(mask)
        padded = np.pad(mask, tolerance)
        for dy in range(2 * tolerance + 1):
            for dx in range(2 * tolerance + 1):
                result |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
        return result

    precision = np.count_nonzero(pred_b & dilate(target_b)) / np.count_nonzero(pred_b)
    recall = np.count_nonzero(target_b & dilate(pred_b)) / np.count_nonzero(target_b)
    return float(2 * precision * recall / max(precision + recall, EPS))


def mask_weighted_pointer(features: np.ndarray, mask_probability: np.ndarray) -> np.ndarray:
    """Pool ``[H,W,D]`` features into one normalized object vector ``[D]``."""
    features = np.asarray(features, dtype=np.float64)
    weights = np.asarray(mask_probability, dtype=np.float64).reshape(-1)
    flat = features.reshape(-1, features.shape[-1])
    total = weights.sum()
    if total <= EPS:
        return np.zeros(features.shape[-1], dtype=np.float64)
    return l2_normalize(np.sum(flat * weights[:, None], axis=0))


@dataclass(frozen=True)
class MemoryEntry:
    frame_index: int
    features: np.ndarray       # [H, W, D]
    mask_probability: np.ndarray  # [H, W]
    object_pointer: np.ndarray # [D]
    prompted: bool


class StreamingMemoryBank:
    """Separate pinned conditioning memories from a recent-frame FIFO queue."""

    def __init__(self, max_recent: int = 6, max_conditioning: int = 4):
        self.max_recent = int(max_recent)
        self.max_conditioning = int(max_conditioning)
        self.conditioning: list[MemoryEntry] = []
        self.recent: list[MemoryEntry] = []

    def __len__(self) -> int:
        return len(self.conditioning) + len(self.recent)

    def add(
        self,
        frame_index: int,
        features: np.ndarray,
        mask_probability: np.ndarray,
        prompted: bool = False,
    ) -> None:
        pointer = mask_weighted_pointer(features, mask_probability)
        entry = MemoryEntry(int(frame_index), np.asarray(features).copy(),
                            np.asarray(mask_probability).copy(), pointer, bool(prompted))
        if prompted:
            self.conditioning = [item for item in self.conditioning if item.frame_index != frame_index]
            self.conditioning.append(entry)
            self.conditioning = self.conditioning[-self.max_conditioning:]
        else:
            self.recent = [item for item in self.recent if item.frame_index != frame_index]
            self.recent.append(entry)
            self.recent = self.recent[-self.max_recent:]

    def entries(self) -> list[MemoryEntry]:
        return self.conditioning + self.recent

    def token_tensors(
        self,
        current_frame: int,
        temporal_strength: float = 0.03,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flatten memory into key, mask-value, and pointer tensors.

        Temporal encoding is added only to recent unprompted spatial memories;
        conditioning memories are deliberately time-agnostic.
        """
        if not len(self):
            raise ValueError("cannot read an empty memory bank")
        keys, values, key_biases, pointers, pointer_ages = [], [], [], [], []
        for entry in self.entries():
            feature = np.asarray(entry.features, dtype=np.float64)
            age = max(0, int(current_frame) - entry.frame_index)
            if not entry.prompted and temporal_strength:
                feature = feature + temporal_strength * temporal_sine_encoding(age, feature.shape[-1])
            keys.append(feature.reshape(-1, feature.shape[-1]))
            values.append(entry.mask_probability.reshape(-1))
            # An explicit recency prior makes this untrained toy behave like a
            # learned temporal attention policy. SAM 2 uses learned temporal
            # embeddings rather than this hand-set scalar bias.
            decay = 0.10 if entry.prompted else 0.04
            key_biases.append(np.full(entry.mask_probability.size, -decay * age))
            if np.linalg.norm(entry.object_pointer) > EPS:
                pointers.append(entry.object_pointer)
                pointer_ages.append(age)
        pointer_tensor = np.stack(pointers) if pointers else np.empty((0, keys[0].shape[-1]))
        return (l2_normalize(np.concatenate(keys)), np.concatenate(values), pointer_tensor,
                np.concatenate(key_biases), np.asarray(pointer_ages, dtype=np.float64))

    def read(
        self,
        current_features: np.ndarray,
        current_frame: int,
        logit_scale: float = 22.0,
        pointer_weight: float = 0.10,
        temporal_strength: float = 0.03,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Propagate a soft mask to the current frame by spatial cross-attention."""
        height, width, dim = current_features.shape
        query = l2_normalize(current_features.reshape(-1, dim))
        key, value, pointers, key_bias, pointer_ages = self.token_tensors(current_frame, temporal_strength)
        memory_probability, peak = scaled_dot_product_attention(
            query, key, value, logit_scale, key_bias=key_bias
        )

        if len(pointers):
            pointer_weights = np.exp(-0.25 * pointer_ages)
            pointer = l2_normalize(np.sum(l2_normalize(pointers) * pointer_weights[:, None], axis=0))
            pointer_similarity = query @ pointer
            # Map cosine similarity to a soft object-likeness score. This is a
            # didactic pointer read, not SAM 2's learned mask decoder.
            pointer_probability = sigmoid(12.0 * (pointer_similarity - 0.56))
            probability = (1.0 - pointer_weight) * memory_probability + pointer_weight * pointer_probability
        else:
            probability = memory_probability
        diagnostics = {
            "memory_tokens": float(len(key)),
            "pointer_tokens": float(len(pointers)),
            "mean_peak_attention": float(np.mean(peak)),
            "max_probability": float(np.max(probability)),
        }
        return probability.reshape(height, width), diagnostics


def run_stream(
    features: np.ndarray,
    object_scores: np.ndarray,
    initial_mask: np.ndarray,
    threshold: float = 0.50,
    max_recent: int = 6,
    use_recent_memory: bool = True,
    write_no_object_memory: bool = True,
    correction_masks: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict[str, float]], StreamingMemoryBank]:
    """Run a toy one-object streaming segmenter over ``[T,H,W,D]`` features."""
    features = np.asarray(features, dtype=np.float64)
    object_scores = np.asarray(object_scores, dtype=np.float64)
    correction_masks = correction_masks or {}
    bank = StreamingMemoryBank(max_recent=max_recent if use_recent_memory else 0, max_conditioning=4)
    bank.add(0, features[0], initial_mask.astype(float), prompted=True)
    predictions = [np.asarray(initial_mask, dtype=bool)]
    diagnostics: list[dict[str, float]] = [{"object_score": float(object_scores[0]), "memory_written": 1.0}]

    for frame_index in range(1, len(features)):
        if frame_index in correction_masks:
            probability = np.asarray(correction_masks[frame_index], dtype=float)
            prediction = probability >= 0.5
            bank.add(frame_index, features[frame_index], probability, prompted=True)
            diagnostics.append({"object_score": float(object_scores[frame_index]), "memory_written": 1.0,
                                "corrective_prompt": 1.0})
            predictions.append(prediction)
            continue

        probability, diag = bank.read(features[frame_index], frame_index)
        visible = object_scores[frame_index] >= 0.5
        prediction = probability >= threshold if visible else np.zeros_like(probability, dtype=bool)
        should_write = use_recent_memory and (visible or write_no_object_memory)
        if should_write:
            # When the object is predicted absent, the toy stores an empty mask
            # paired with the current features. This acts as negative/no-object
            # memory. SAM 2 instead learns explicit occlusion/no-object embeddings.
            stored_probability = probability if visible else np.zeros_like(probability)
            bank.add(frame_index, features[frame_index], stored_probability, prompted=False)
        diag.update({"object_score": float(object_scores[frame_index]),
                     "memory_written": float(should_write), "visible_decision": float(visible)})
        diagnostics.append(diag)
        predictions.append(prediction)
    return np.asarray(predictions), diagnostics, bank
