"""Surgical-phase assistance gate built around temporal segmentation output."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asformer_math import hierarchical_attention_refine, iterative_decoder_refine
from synthetic_sequence import PHASE_NAMES, make_phase_sequence


@dataclass(frozen=True)
class GateDecision:
    enabled: bool
    reason: str
    phase: str
    confidence: float
    normalized_entropy: float


def phase_assistance_gate(
    probabilities: np.ndarray,
    frame: int,
    requested_phase: int,
    persistence: int = 8,
    min_confidence: float = 0.72,
    max_entropy: float = 0.58,
) -> GateDecision:
    """Conservative example gate; never use as a clinical safety system."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or not 0 <= frame < probabilities.shape[0]:
        raise ValueError("probabilities must be [T,C] and frame must be valid")
    start = max(0, frame - persistence + 1)
    history = probabilities[start : frame + 1].argmax(axis=1)
    current = probabilities[frame]
    predicted = int(np.argmax(current))
    confidence = float(current[predicted])
    entropy = float(-np.sum(current * np.log(np.maximum(current, 1e-12))) / np.log(current.size))
    name = PHASE_NAMES[predicted]

    if predicted != requested_phase:
        return GateDecision(False, "wrong phase", name, confidence, entropy)
    if history.size < persistence or np.any(history != requested_phase):
        return GateDecision(False, "phase not persistent", name, confidence, entropy)
    if confidence < min_confidence:
        return GateDecision(False, "low confidence", name, confidence, entropy)
    if entropy > max_entropy:
        return GateDecision(False, "high entropy", name, confidence, entropy)
    return GateDecision(True, "all temporal checks passed", name, confidence, entropy)


def main() -> None:
    features, logits, labels = make_phase_sequence(seed=23)
    encoder, _ = hierarchical_attention_refine(logits, features)
    probabilities = iterative_decoder_refine(encoder, stages=3)[-1]

    treatment_frames = np.flatnonzero(labels == 2)
    candidates = {
        "at boundary": int(treatment_frames[1]),
        "stable middle": int(treatment_frames[len(treatment_frames) // 2]),
        "after treatment": int(np.flatnonzero(labels == 3)[8]),
    }
    print("Illustrative assistive-robot gate for the requested phase 'treatment'")
    for case, frame in candidates.items():
        decision = phase_assistance_gate(probabilities, frame, requested_phase=2)
        print(
            f"  {case:16s} t={frame:3d} enabled={str(decision.enabled):5s} "
            f"phase={decision.phase:11s} conf={decision.confidence:.3f} "
            f"entropy={decision.normalized_entropy:.3f} reason={decision.reason}"
        )

    # A model can be confidently wrong. Force a coherent wrong distribution to
    # prove the gate cannot replace independent sensing or human supervision.
    wrong = probabilities.copy()
    wrong[110:135] = np.array([0.02, 0.02, 0.03, 0.93])
    decision = phase_assistance_gate(wrong, 125, requested_phase=3)
    print("\ncoherent-error stress test:")
    print(
        f"  true={PHASE_NAMES[labels[125]]}, predicted={decision.phase}, "
        f"enabled={decision.enabled}, reason={decision.reason}"
    )
    print("  lesson: confidence + persistence do not detect a coherent semantic error")


if __name__ == "__main__":
    main()
