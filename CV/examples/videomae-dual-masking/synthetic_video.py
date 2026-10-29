"""Deterministic videos for masking leakage and surgical-phase integration."""

from __future__ import annotations

import numpy as np


def make_redundant_video(seed: int = 0, frames: int = 16, height: int = 64, width: int = 64) -> np.ndarray:
    """Return a slowly changing textured clip that exposes temporal-copy leakage."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    fixed = np.stack([
        0.48 + 0.28 * np.sin(xx / 5.0) * np.cos(yy / 8.0),
        0.42 + 0.24 * np.cos(xx / 7.0 + yy / 9.0),
        0.38 + 0.22 * np.sin((xx - yy) / 6.0),
    ])
    clip = []
    for t in range(frames):
        frame = fixed + 0.025 * np.sin(0.25 * t)
        frame += rng.normal(0.0, 0.012, size=frame.shape)
        clip.append(np.clip(frame, 0.0, 1.0))
    return np.asarray(clip).transpose(1, 0, 2, 3)[None]


def make_motion_video(seed: int = 0, frames: int = 16, height: int = 64, width: int = 64) -> np.ndarray:
    """Return one RGB clip ``[1,3,T,H,W]`` with camera drift and moving objects."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    clip = []
    texture = (0.15 * np.sin(xx / 4.3) + 0.12 * np.cos(yy / 5.1))[None]
    for t in range(frames):
        base = np.concatenate([
            0.35 + texture,
            0.28 + 0.7 * texture,
            0.22 - 0.5 * texture,
        ], axis=0)
        base = np.roll(base, shift=(t // 3, t // 4), axis=(1, 2))
        cx, cy = 10 + 2.4 * t, 31 + 9 * np.sin(0.42 * t)
        object_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 7.0 ** 2
        base[:, object_mask] = np.array([0.86, 0.27, 0.18])[:, None]
        tool_x = int(56 - 1.8 * t)
        tool = (np.abs(xx - tool_x) <= 2) & (yy > 12) & (yy < 54)
        base[:, tool] = np.array([0.72, 0.76, 0.82])[:, None]
        base += rng.normal(0.0, 0.018, size=base.shape)
        clip.append(np.clip(base, 0.0, 1.0))
    return np.asarray(clip).transpose(1, 0, 2, 3)[None]


def make_phase_sequence(
    seed: int,
    frames_per_phase: int = 24,
    noisy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Create frame-level proxy observations for five ordered surgical phases.

    Appearance changes gradually, while two oscillatory channels encode motion
    frequency that can only be estimated reliably over a temporal window.
    """
    rng = np.random.default_rng(seed)
    phases = 5
    total = phases * frames_per_phase
    labels = np.repeat(np.arange(phases), frames_per_phase)
    appearance = np.array([
        [0.12, 0.18, 0.23],
        [0.26, 0.24, 0.27],
        [0.38, 0.31, 0.31],
        [0.49, 0.39, 0.36],
        [0.59, 0.48, 0.43],
    ])
    frequency = np.array([0.16, 0.31, 0.52, 0.78, 1.05])
    observations = np.zeros((total, 5), dtype=np.float64)
    illumination = rng.normal(0.0, 0.035, size=3) if noisy else np.zeros(3)
    for t, phase in enumerate(labels):
        local = t % frames_per_phase
        slow_drift = 0.018 * np.sin(0.07 * t + np.arange(3))
        observations[t, :3] = appearance[phase] + illumination + slow_drift
        observations[t, 3] = np.sin(frequency[phase] * local + 0.4)
        observations[t, 4] = np.cos(frequency[phase] * local + 0.4)
    noise = 0.025 if not noisy else 0.085
    observations += rng.normal(0.0, noise, size=observations.shape)
    if noisy:
        # Specular/smoke bursts create locally plausible but wrong appearance.
        for start, wrong_phase in ((19, 2), (47, 0), (73, 4), (99, 1)):
            stop = min(start + 5, total)
            observations[start:stop, :3] = appearance[wrong_phase] + rng.normal(
                0.0, 0.04, size=(stop - start, 3)
            )
    return observations, labels
