"""Frozen-video-feature integration for ordered surgical phase recognition.

The clip descriptor is a transparent proxy for a frozen VideoMAE encoder.  It
demonstrates feature caching, temporal windows, a linear-distance head, and an
ordered Viterbi decoder; it is not a claim of VideoMAE benchmark performance.
"""

from __future__ import annotations

import numpy as np

from synthetic_video import make_phase_sequence


def clip_descriptors(observations: np.ndarray, radius: int = 6) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float64)
    padded = np.pad(observations, ((radius, radius), (0, 0)), mode="edge")
    descriptors = []
    for t in range(len(observations)):
        window = padded[t:t + 2 * radius + 1]
        delta = np.diff(window, axis=0)
        motion = window[:, 3:5]
        lag1_dot = np.mean(np.sum(motion[1:] * motion[:-1], axis=1))
        lag1_cross = np.mean(motion[:-1, 0] * motion[1:, 1] - motion[:-1, 1] * motion[1:, 0])
        lag2_dot = np.mean(np.sum(motion[2:] * motion[:-2], axis=1))
        lag2_cross = np.mean(motion[:-2, 0] * motion[2:, 1] - motion[:-2, 1] * motion[2:, 0])
        descriptors.append(np.concatenate([
            window[:, :3].mean(axis=0),
            window[:, :3].std(axis=0),
            np.mean(np.abs(delta), axis=0),
            np.array([lag1_dot, lag1_cross, lag2_dot, lag2_cross]),
        ]))
    return np.asarray(descriptors)


def fit_centroids(features: np.ndarray, labels: np.ndarray, classes: int = 5) -> tuple[np.ndarray, np.ndarray]:
    # A variance floor prevents almost-constant nuisance statistics in the
    # clean reference sequence from exploding under deployment noise.
    scale = np.maximum(features.std(axis=0), 0.05)
    normalized = features / scale
    centroids = np.stack([normalized[labels == label].mean(axis=0) for label in range(classes)])
    return centroids, scale


def emission_cost(features: np.ndarray, centroids: np.ndarray, scale: np.ndarray) -> np.ndarray:
    normalized = features / scale
    return np.mean((normalized[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)


def ordered_viterbi(cost: np.ndarray, transition_penalty: float = 0.12) -> np.ndarray:
    """Decode phases with only stay or advance-by-one transitions."""
    frames, classes = cost.shape
    score = np.full((frames, classes), np.inf)
    parent = np.full((frames, classes), -1, dtype=int)
    score[0, 0] = cost[0, 0]
    for t in range(1, frames):
        for phase in range(classes):
            candidates = [(score[t - 1, phase], phase)]
            if phase > 0:
                candidates.append((score[t - 1, phase - 1] + transition_penalty, phase - 1))
            best, previous = min(candidates)
            score[t, phase] = best + cost[t, phase]
            parent[t, phase] = previous
    state = int(np.argmin(score[-1] + 0.4 * (classes - 1 - np.arange(classes))))
    path = np.empty(frames, dtype=int)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        if t:
            state = parent[t, state]
    return path


def boundary_f1(predicted: np.ndarray, target: np.ndarray, tolerance: int = 6) -> float:
    pred = np.flatnonzero(np.diff(predicted) != 0) + 1
    true = np.flatnonzero(np.diff(target) != 0) + 1
    if not len(pred) and not len(true):
        return 1.0
    matched_true: set[int] = set()
    matches = 0
    for boundary in pred:
        candidates = [i for i, other in enumerate(true)
                      if i not in matched_true and abs(int(other) - int(boundary)) <= tolerance]
        if candidates:
            chosen = min(candidates, key=lambda i: abs(int(true[i]) - int(boundary)))
            matched_true.add(chosen); matches += 1
    precision = matches / max(len(pred), 1)
    recall = matches / max(len(true), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def phase_predictions() -> dict[str, np.ndarray]:
    train_x, train_y = make_phase_sequence(seed=2, noisy=False)
    test_x, test_y = make_phase_sequence(seed=91, noisy=True)

    frame_centroids, frame_scale = fit_centroids(train_x[:, :3], train_y)
    frame_cost = emission_cost(test_x[:, :3], frame_centroids, frame_scale)
    frame_prediction = np.argmin(frame_cost, axis=1)

    train_clip = clip_descriptors(train_x)
    test_clip = clip_descriptors(test_x)
    clip_centroids, clip_scale = fit_centroids(train_clip, train_y)
    clip_cost = emission_cost(test_clip, clip_centroids, clip_scale)
    clip_prediction = np.argmin(clip_cost, axis=1)
    ordered_prediction = ordered_viterbi(clip_cost)
    return {
        "target": test_y,
        "frame": frame_prediction,
        "clip": clip_prediction,
        "ordered": ordered_prediction,
        "train_clip": train_clip,
    }


def run_application() -> dict[str, float | int | bool]:
    predictions = phase_predictions()
    test_y = predictions["target"]
    frame_prediction = predictions["frame"]
    clip_prediction = predictions["clip"]
    ordered_prediction = predictions["ordered"]
    train_clip = predictions["train_clip"]

    accuracy = lambda prediction: float(np.mean(prediction == test_y))
    accepted = bool(accuracy(ordered_prediction) >= 0.80 and
                    boundary_f1(ordered_prediction, test_y) >= 0.75 and
                    np.all(np.diff(ordered_prediction) >= 0))
    return {
        "frames": len(test_y),
        "frame_feature_dim": 3,
        "clip_feature_dim": train_clip.shape[1],
        "frame_only_accuracy": accuracy(frame_prediction),
        "clip_accuracy": accuracy(clip_prediction),
        "ordered_clip_accuracy": accuracy(ordered_prediction),
        "frame_only_boundary_f1": boundary_f1(frame_prediction, test_y),
        "clip_boundary_f1": boundary_f1(clip_prediction, test_y),
        "ordered_clip_boundary_f1": boundary_f1(ordered_prediction, test_y),
        "ordered_monotonic": bool(np.all(np.diff(ordered_prediction) >= 0)),
        "deploy_phase_stream": accepted,
    }


def main() -> None:
    for name, value in run_application().items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
