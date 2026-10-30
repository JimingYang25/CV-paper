"""Deterministic synthetic long-video features for mechanism validation."""

from __future__ import annotations

import numpy as np


PHASE_NAMES = ("preparation", "exposure", "treatment", "closure")


def make_phase_sequence(
    seed: int = 0,
    lengths: tuple[int, ...] = (42, 58, 63, 37),
    feature_dim: int = 12,
    noise: float = 0.72,
    corruption_rate: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return features [T,D], classifier logits [T,C], and labels [T]."""
    if len(lengths) != len(PHASE_NAMES):
        raise ValueError("one duration is required per phase")
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    classes = len(lengths)

    prototypes = rng.normal(size=(classes, feature_dim))
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    features = 2.3 * prototypes[labels] + noise * rng.normal(size=(labels.size, feature_dim))

    # The frame classifier is intentionally competent but locally unstable.
    logits = 1.55 * features @ prototypes.T
    logits += 0.55 * rng.normal(size=logits.shape)

    count = max(1, int(corruption_rate * labels.size))
    corrupt = rng.choice(labels.size, size=count, replace=False)
    wrong = (labels[corrupt] + rng.integers(1, classes, size=count)) % classes
    logits[corrupt, labels[corrupt]] -= 3.6
    logits[corrupt, wrong] += 3.6

    # Boundary frames are genuinely ambiguous, as in clip encoders whose
    # receptive fields straddle two phases.
    starts = np.cumsum(lengths)[:-1]
    for boundary in starts:
        lo, hi = max(0, boundary - 3), min(labels.size, boundary + 3)
        logits[lo:hi] *= 0.52
    return features, logits, labels


def make_repeated_phase_sequence(seed: int = 101) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A failure case: the first phase legitimately reappears at the end."""
    rng = np.random.default_rng(seed)
    lengths = (34, 44, 47, 36, 29)
    labels = np.repeat(np.array([0, 1, 2, 3, 0], dtype=np.int64), lengths)
    classes, feature_dim = 4, 12
    prototypes = rng.normal(size=(classes, feature_dim))
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    features = 2.2 * prototypes[labels] + 0.82 * rng.normal(size=(labels.size, feature_dim))
    logits = 1.45 * features @ prototypes.T + 0.65 * rng.normal(size=(labels.size, classes))
    return features, logits, labels
