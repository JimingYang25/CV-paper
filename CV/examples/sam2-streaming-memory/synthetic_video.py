"""Deterministic moving-object video with drift, distractors, and occlusion."""

from __future__ import annotations

import numpy as np

from streaming_memory import l2_normalize


def ellipse_mask(height: int, width: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def make_video(
    seed: int = 7,
    frames: int = 24,
    height: int = 24,
    width: int = 32,
    dim: int = 8,
) -> dict[str, np.ndarray | tuple[int, ...]]:
    """Create per-pixel features rather than RGB to isolate memory behavior."""
    if dim != 8:
        raise ValueError("this deterministic construction uses dim=8")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    x_norm = 2.0 * xx / max(width - 1, 1) - 1.0
    y_norm = 2.0 * yy / max(height - 1, 1) - 1.0

    proto_start = l2_normalize(np.array([2.3, 0.1, 1.5, 0.2, 0.4, 0.0, 0.3, 0.1]))
    proto_end = l2_normalize(np.array([0.1, 2.2, 1.2, 0.5, 0.1, 0.3, 0.2, 0.4]))
    proto_bg = l2_normalize(np.array([-1.4, -1.1, 0.0, 0.7, 0.2, -0.3, 0.1, 0.2]))
    proto_distractor = l2_normalize(np.array([1.7, 0.2, 1.0, 0.0, 0.5, 0.1, 0.2, 0.0]))
    occluded_frames = (9, 10, 11)

    feature_video, masks, distractors, object_scores = [], [], [], []
    for time in range(frames):
        fraction = time / max(frames - 1, 1)
        target_proto = l2_normalize((1.0 - fraction) * proto_start + fraction * proto_end)
        cx = 5.2 + 0.82 * time
        cy = 11.5 + 2.1 * np.sin(0.38 * time)
        rx = 3.1 + 0.45 * np.sin(0.27 * time)
        ry = 2.7 + 0.35 * np.cos(0.31 * time)
        visible = time not in occluded_frames
        target = ellipse_mask(height, width, cx, cy, rx, ry) if visible else np.zeros((height, width), bool)

        distractor = np.zeros((height, width), bool)
        if time >= 6:
            distractor = ellipse_mask(height, width, 25.0 - 0.28 * time, 6.0 + 0.18 * time, 2.7, 2.4)

        features = np.broadcast_to(proto_bg, (height, width, dim)).copy()
        features += np.stack([
            0.12 * x_norm, 0.10 * y_norm, 0.05 * x_norm * y_norm,
            0.04 * np.sin(xx / 3), 0.04 * np.cos(yy / 3),
            np.zeros_like(x_norm), np.zeros_like(x_norm), np.zeros_like(x_norm),
        ], axis=-1)
        features[distractor] = proto_distractor + rng.normal(0.0, 0.07, size=(distractor.sum(), dim))
        if visible:
            local_texture = np.stack([
                0.08 * np.sin(xx[target] * 0.7), 0.08 * np.cos(yy[target] * 0.6),
                0.04 * x_norm[target], 0.04 * y_norm[target],
                np.zeros(target.sum()), np.zeros(target.sum()), np.zeros(target.sum()), np.zeros(target.sum()),
            ], axis=-1)
            features[target] = target_proto + local_texture + rng.normal(0.0, 0.055, size=(target.sum(), dim))
        features += rng.normal(0.0, 0.035, size=features.shape)
        feature_video.append(l2_normalize(features))
        masks.append(target)
        distractors.append(distractor)
        # Simulated output of an imperfect learned presence/occlusion head. It is
        # separate from mask ground truth so the memory-write policy can be tested.
        object_scores.append((0.92 if visible else 0.08) + rng.normal(0.0, 0.025))

    return {
        "features": np.asarray(feature_video),
        "masks": np.asarray(masks),
        "distractor_masks": np.asarray(distractors),
        "object_scores": np.clip(object_scores, 0.0, 1.0),
        "occluded_frames": occluded_frames,
    }
