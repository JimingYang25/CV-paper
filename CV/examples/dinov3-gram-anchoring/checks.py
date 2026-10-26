"""Numerical and behavioral checks for DINO and Gram-anchoring primitives."""

from __future__ import annotations

import numpy as np

from application import demonstration
from demo import evaluate
from dino_gram import (
    cross_entropy,
    dino_cross_view_loss,
    ema_teacher_update,
    gram_anchor_loss,
    gram_anchor_loss_and_grad,
    gram_matrix,
    l2_normalize,
    refine_patch_features,
    softmax,
    teacher_distribution,
    update_center,
)
from synthetic_dense import make_dense_case, shuffled_anchor


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS {name}{': ' + detail if detail else ''}")


def entropy(p: np.ndarray) -> float:
    return float(np.mean(-np.sum(p * np.log(np.maximum(p, 1e-15)), axis=-1)))


def main() -> None:
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(7, 11))
    p = softmax(logits)
    check("softmax sums to one", np.allclose(p.sum(axis=-1), 1.0, atol=1e-12))
    check("softmax shift invariance", np.allclose(p, softmax(logits + 137.0), atol=1e-12))
    check("lower temperature sharpens", entropy(softmax(logits, 0.2)) < entropy(softmax(logits, 1.0)))

    center = np.linspace(-0.2, 0.2, 11)
    expected = softmax(logits - center, 0.4)
    check("teacher centering and sharpening", np.allclose(teacher_distribution(logits, center, 0.4), expected))

    uniform_target = np.full((3, 4), 0.25)
    uniform_logits = np.zeros((3, 4))
    check("uniform cross entropy", np.allclose(cross_entropy(uniform_target, uniform_logits, 1.0), np.log(4.0)))

    teacher = np.tile(np.array([[[2.0, 0.0, -1.0]]]), (2, 3, 1))
    student = np.tile(np.array([[[5.0, 0.0, -2.5]]]), (4, 3, 1))
    result = dino_cross_view_loss(teacher, student, np.zeros(3), teacher_temperature=0.4, student_temperature=1.0)
    check("DINO cross-view pair count", result.pair_count == 6, f"{result.pair_count}")
    check("matched DINO distributions", result.loss < 0.10, f"loss {result.loss:.6f}")

    old_center = np.array([1.0, 2.0, 3.0])
    batch_logits = np.array([[[3.0, 4.0, 5.0]], [[5.0, 6.0, 7.0]]])
    check("center EMA update", np.allclose(update_center(old_center, batch_logits, 0.75), [1.75, 2.75, 3.75]))
    check("teacher parameter EMA", np.allclose(ema_teacher_update(np.zeros(3), np.ones(3), 0.9), 0.1))

    features = rng.normal(size=(2, 9, 6))
    normalized = l2_normalize(features)
    check("feature normalization", np.allclose(np.linalg.norm(normalized, axis=-1), 1.0, atol=1e-12))
    gram = gram_matrix(features)
    check("Gram symmetry", np.allclose(gram, np.swapaxes(gram, -1, -2), atol=1e-12))
    check("Gram unit diagonal", np.allclose(np.diagonal(gram, axis1=-2, axis2=-1), 1.0, atol=1e-12))

    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    rotated = features @ q
    check("orthogonal-basis invariance", gram_anchor_loss(features, rotated) < 1e-28)
    scales = rng.uniform(0.2, 3.0, size=(2, 9, 1))
    check("positive per-token scale invariance", gram_anchor_loss(features, features * scales) < 1e-28)
    check("identical Gram loss is zero", gram_anchor_loss(features, features) == 0.0)

    # Finite-difference test includes the derivative through row normalization.
    z = rng.normal(size=(1, 4, 3))
    anchor = rng.normal(size=(1, 4, 3))
    _, analytic = gram_anchor_loss_and_grad(z, anchor)
    eps = 1e-6
    numeric = np.zeros_like(z)
    for index in np.ndindex(z.shape):
        plus, minus = z.copy(), z.copy()
        plus[index] += eps; minus[index] -= eps
        numeric[index] = (gram_anchor_loss(plus, anchor) - gram_anchor_loss(minus, anchor)) / (2.0 * eps)
    error = float(np.max(np.abs(analytic - numeric)))
    check("Gram analytic gradient", error < 2e-9, f"max error {error:.3e}")

    case = make_dense_case(seed=8)
    baseline_loss = gram_anchor_loss(case["degraded_features"], case["anchor_features"])
    refined = refine_patch_features(case["degraded_features"], case["anchor_features"], steps=250)
    final_loss = gram_anchor_loss(refined.features, case["anchor_features"])
    check("correct anchoring reduces Gram error", final_loss < 1e-4 * baseline_loss, f"{baseline_loss:.4e} -> {final_loss:.4e}")
    metrics = evaluate(refined.features, case["anchor_features"], case["query_index"])
    check("query similarity map recovered", metrics["query_map_correlation"] > 0.999, f"corr {metrics['query_map_correlation']:.6f}")

    wrong = shuffled_anchor(case["anchor_features"], 99)
    wrong_refined = refine_patch_features(case["degraded_features"], wrong, steps=250).features
    wrong_to_wrong = gram_anchor_loss(wrong_refined, wrong)
    wrong_to_correct = gram_anchor_loss(wrong_refined, case["anchor_features"])
    check("wrong anchor fits itself but not truth", wrong_to_wrong < 1e-4 and wrong_to_correct > 0.05, f"wrong {wrong_to_wrong:.3e}, correct {wrong_to_correct:.3e}")

    app = demonstration(seed=21)
    check("deployment gate rejects degraded checkpoint", not app["degraded_checkpoint"]["deploy"])
    check("deployment gate accepts correctly anchored checkpoint", app["correctly_anchored_checkpoint"]["deploy"])
    check("deployment gate rejects wrong teacher", not app["wrong_teacher_checkpoint"]["deploy"])

    print("\n22/22 checks passed")


if __name__ == "__main__":
    main()
