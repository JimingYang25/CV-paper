"""NumPy reference for DINO cross-view distillation and DINOv3 Gram anchoring.

This is an independent educational implementation of the loss mathematics. It is
not Meta's training code and does not reproduce DINOv3's model or scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


def softmax(logits: Array, temperature: float = 1.0, axis: int = -1) -> Array:
    """Numerically stable temperature softmax."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def teacher_distribution(logits: Array, center: Array, temperature: float) -> Array:
    """Centered, sharpened teacher probabilities [..., K]."""
    logits = np.asarray(logits, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (logits.shape[-1],):
        raise ValueError("center must have shape [K]")
    return softmax(logits - center, temperature=temperature, axis=-1)


def cross_entropy(target_prob: Array, student_logits: Array, student_temperature: float) -> Array:
    """Cross entropy over K prototypes, preserving all leading dimensions."""
    q = softmax(student_logits, temperature=student_temperature, axis=-1)
    p = np.asarray(target_prob, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("target probabilities and student logits must have matching shapes")
    return -np.sum(p * np.log(np.maximum(q, 1e-15)), axis=-1)


@dataclass(frozen=True)
class DINOLossResult:
    loss: float
    pair_count: int
    teacher_entropy: float
    student_entropy: float


def dino_cross_view_loss(
    teacher_global_logits: Array,
    student_all_logits: Array,
    center: Array,
    *,
    teacher_temperature: float = 0.04,
    student_temperature: float = 0.1,
) -> DINOLossResult:
    """DINO loss for G teacher global views and V student global/local views.

    Inputs have shapes teacher [G,B,K], student [V,B,K]. For student global view
    s<G, the same-index teacher view is skipped; all other cross-view pairs are used.
    """
    teacher = np.asarray(teacher_global_logits, dtype=np.float64)
    student = np.asarray(student_all_logits, dtype=np.float64)
    if teacher.ndim != 3 or student.ndim != 3:
        raise ValueError("teacher and student logits must be [views,batch,prototypes]")
    G, B, K = teacher.shape
    V, Bs, Ks = student.shape
    if (B, K) != (Bs, Ks) or G > V:
        raise ValueError("batch/prototype dimensions must match and G must not exceed V")

    p = teacher_distribution(teacher, center, teacher_temperature)
    losses = []
    for t in range(G):
        for s in range(V):
            if s == t:  # exclude identical global crop pairing
                continue
            losses.append(cross_entropy(p[t], student[s], student_temperature).mean())
    q = softmax(student, student_temperature, axis=-1)
    teacher_entropy = float(np.mean(-np.sum(p * np.log(np.maximum(p, 1e-15)), axis=-1)))
    student_entropy = float(np.mean(-np.sum(q * np.log(np.maximum(q, 1e-15)), axis=-1)))
    return DINOLossResult(float(np.mean(losses)), len(losses), teacher_entropy, student_entropy)


def update_center(center: Array, teacher_logits: Array, momentum: float = 0.9) -> Array:
    """EMA update of the original DINO teacher-logit center [K]."""
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0,1)")
    center = np.asarray(center, dtype=np.float64)
    logits = np.asarray(teacher_logits, dtype=np.float64)
    batch_center = logits.reshape(-1, logits.shape[-1]).mean(axis=0)
    return momentum * center + (1.0 - momentum) * batch_center


def ema_teacher_update(teacher: Array, student: Array, momentum: float) -> Array:
    """One parameter-array EMA update: teacher <- m*teacher + (1-m)*student."""
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0,1)")
    teacher = np.asarray(teacher, dtype=np.float64)
    student = np.asarray(student, dtype=np.float64)
    if teacher.shape != student.shape:
        raise ValueError("teacher and student parameter arrays must match")
    return momentum * teacher + (1.0 - momentum) * student


def l2_normalize(features: Array, eps: float = 1e-12) -> Array:
    """Normalize the final feature axis."""
    x = np.asarray(features, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def gram_matrix(patch_features: Array, *, remove_negative: bool = False) -> Array:
    """Per-image patch cosine matrix [B,P,P] from features [B,P,D]."""
    x = l2_normalize(patch_features)
    gram = x @ np.swapaxes(x, -1, -2)
    return np.maximum(gram, 0.0) if remove_negative else gram


def gram_anchor_loss(
    student_features: Array,
    anchor_features: Array,
    *,
    remove_negative: bool = False,
) -> float:
    """Mean-squared difference between per-image student and anchor Gram matrices."""
    gs = gram_matrix(student_features, remove_negative=remove_negative)
    ga = gram_matrix(anchor_features, remove_negative=remove_negative)
    if gs.shape != ga.shape:
        raise ValueError("student and anchor must share [B,P] dimensions")
    return float(np.mean((gs - ga) ** 2))


def gram_anchor_loss_and_grad(student_raw: Array, anchor_features: Array) -> tuple[float, Array]:
    """Exact all-correlation Gram MSE and gradient with respect to raw student features.

    This differentiable reference implements the paper equation without negative-
    similarity clipping. The official repository optionally clips negatives.
    """
    z = np.asarray(student_raw, dtype=np.float64)
    anchor = np.asarray(anchor_features, dtype=np.float64)
    if z.shape != anchor.shape or z.ndim != 3:
        raise ValueError("student_raw and anchor_features must both be [B,P,D]")
    norm = np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
    x = z / norm
    a = l2_normalize(anchor)
    error = x @ np.swapaxes(x, -1, -2) - a @ np.swapaxes(a, -1, -2)
    loss = float(np.mean(error * error))
    B, P, _ = z.shape
    grad_x = 4.0 * (error @ x) / (B * P * P)
    # Jacobian of row normalization x=z/||z||: (I - xx^T)/||z||.
    radial = np.sum(grad_x * x, axis=-1, keepdims=True)
    grad_z = (grad_x - x * radial) / norm
    return loss, grad_z


@dataclass
class RefinementTrace:
    features: Array
    losses: list[float]


def refine_patch_features(
    initial_features: Array,
    anchor_features: Array,
    *,
    steps: int = 250,
    learning_rate: float = 0.35,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> RefinementTrace:
    """Adam optimization of patch features for a transparent loss-level experiment."""
    z = np.array(initial_features, dtype=np.float64, copy=True)
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    losses: list[float] = []
    for step in range(1, steps + 1):
        loss, grad = gram_anchor_loss_and_grad(z, anchor_features)
        losses.append(loss)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        z -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
    return RefinementTrace(l2_normalize(z), losses)


def similarity_map(features: Array, query_index: int) -> Array:
    """Cosine similarity from one patch to every patch, shape [B,P]."""
    x = l2_normalize(features)
    if not 0 <= query_index < x.shape[1]:
        raise IndexError("query_index outside patch range")
    return np.einsum("bd,bpd->bp", x[:, query_index], x)
