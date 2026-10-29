"""Auditable NumPy primitives for VideoMAE-style masking.

This is a mechanism reference, not the official VideoMAE model.  It contains no
pretrained weights and replaces learned reconstruction with transparent probes.
"""

from __future__ import annotations

import numpy as np


EPS = 1e-6


def cubeify(video: np.ndarray, tubelet: tuple[int, int, int] = (2, 8, 8)) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Convert ``[B,C,T,H,W]`` video into ``[B,N,C*pt*ph*pw]`` cubes."""
    video = np.asarray(video, dtype=np.float64)
    if video.ndim != 5:
        raise ValueError("video must have shape [B,C,T,H,W]")
    batch, channels, frames, height, width = video.shape
    pt, ph, pw = map(int, tubelet)
    if frames % pt or height % ph or width % pw:
        raise ValueError("T, H, and W must be divisible by the tubelet size")
    gt, gh, gw = frames // pt, height // ph, width // pw
    cubes = video.reshape(batch, channels, gt, pt, gh, ph, gw, pw)
    cubes = cubes.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return cubes.reshape(batch, gt * gh * gw, channels * pt * ph * pw), (gt, gh, gw)


def uncubeify(
    cubes: np.ndarray,
    video_shape: tuple[int, int, int, int, int],
    tubelet: tuple[int, int, int] = (2, 8, 8),
) -> np.ndarray:
    """Invert :func:`cubeify` for cubes ordered time-major, row-major, column-major."""
    cubes = np.asarray(cubes, dtype=np.float64)
    batch, channels, frames, height, width = map(int, video_shape)
    pt, ph, pw = map(int, tubelet)
    gt, gh, gw = frames // pt, height // ph, width // pw
    expected = (batch, gt * gh * gw, channels * pt * ph * pw)
    if cubes.shape != expected:
        raise ValueError(f"expected cubes {expected}, got {cubes.shape}")
    video = cubes.reshape(batch, gt, gh, gw, channels, pt, ph, pw)
    video = video.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return video.reshape(video_shape)


def normalize_cubes(cubes: np.ndarray, channels: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize each cube per channel over its spatiotemporal pixels."""
    cubes = np.asarray(cubes, dtype=np.float64)
    if cubes.shape[-1] % channels:
        raise ValueError("flattened cube dimension must be divisible by channels")
    shaped = cubes.reshape(*cubes.shape[:-1], channels, -1)
    mean = shaped.mean(axis=-1, keepdims=True)
    variance = shaped.var(axis=-1, keepdims=True, ddof=1)
    normalized = (shaped - mean) / (np.sqrt(variance) + EPS)
    return normalized.reshape(cubes.shape), mean, variance


def _count(total: int, ratio: float) -> int:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0,1]")
    return int(round(total * ratio))


def tube_mask(grid: tuple[int, int, int], ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Mask identical spatial sites at every tubelet time: ``[Gt,Gh,Gw]``."""
    gt, gh, gw = map(int, grid)
    spatial = gh * gw
    flat = np.zeros(spatial, dtype=bool)
    chosen = rng.choice(spatial, size=_count(spatial, ratio), replace=False)
    flat[chosen] = True
    return np.broadcast_to(flat.reshape(1, gh, gw), (gt, gh, gw)).copy()


def random_mask(grid: tuple[int, int, int], ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Independently choose masked cubes across the full space-time grid."""
    total = int(np.prod(grid))
    flat = np.zeros(total, dtype=bool)
    flat[rng.choice(total, size=_count(total, ratio), replace=False)] = True
    return flat.reshape(grid)


def frame_mask(grid: tuple[int, int, int], ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Mask complete tubelet-time slices; ratio is quantized by ``Gt``."""
    gt, gh, gw = map(int, grid)
    hidden = np.zeros(gt, dtype=bool)
    hidden[rng.choice(gt, size=_count(gt, ratio), replace=False)] = True
    return np.broadcast_to(hidden[:, None, None], (gt, gh, gw)).copy()


def running_cell_keep(grid: tuple[int, int, int], period: int = 2, offset: int = 0) -> np.ndarray:
    """Didactic cyclic decoder sampler with distributed space-time coverage.

    This is not a source-identical reproduction of the official running-cell
    generator.  Across offsets ``0..period-1``, every token is selected once.
    """
    if period < 1:
        raise ValueError("period must be positive")
    gt, gh, gw = map(int, grid)
    tt, yy, xx = np.mgrid[:gt, :gh, :gw]
    return ((xx + 2 * yy + 3 * tt - int(offset)) % period) == 0


def dual_mask_selection(encoder_mask: np.ndarray, decoder_keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return decoder inputs and supervised targets for a dual-masked step.

    Encoder-visible tokens always enter the decoder.  Decoder-kept tokens that
    were hidden from the encoder are represented by mask tokens and supervised.
    """
    encoder_mask = np.asarray(encoder_mask, dtype=bool)
    decoder_keep = np.asarray(decoder_keep, dtype=bool)
    if encoder_mask.shape != decoder_keep.shape:
        raise ValueError("encoder and decoder masks must share a grid")
    supervised = encoder_mask & decoder_keep
    decoder_input = (~encoder_mask) | supervised
    return decoder_input, supervised


def masked_mse(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    """MSE over selected token vectors; arrays are ``[N,K]`` and mask is ``[N]``."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    if target.shape != prediction.shape or target.ndim != 2 or len(selected) != len(target):
        raise ValueError("target/prediction must be [N,K] with an N-element mask")
    if not selected.any():
        raise ValueError("loss mask selects no tokens")
    return float(np.mean((target[selected] - prediction[selected]) ** 2))


def temporal_copy_probe(cubes: np.ndarray, grid: tuple[int, int, int], mask: np.ndarray) -> np.ndarray:
    """Reconstruct hidden cubes by copying the nearest visible same-site cube.

    When an entire spatial tube is masked, no same-site temporal shortcut exists;
    the probe falls back to the mean of all visible tokens.  This intentionally
    measures leakage available to a trivial predictor, not model quality.
    """
    cubes = np.asarray(cubes, dtype=np.float64)
    if cubes.ndim != 2:
        raise ValueError("cubes must be [N,K]")
    gt, gh, gw = map(int, grid)
    mask = np.asarray(mask, dtype=bool).reshape(grid)
    if cubes.shape[0] != gt * gh * gw:
        raise ValueError("cube count and grid disagree")
    tokens = cubes.reshape(gt, gh, gw, -1)
    prediction = tokens.copy()
    visible_global = tokens[~mask]
    fallback = visible_global.mean(axis=0) if len(visible_global) else np.zeros(tokens.shape[-1])
    for t, y, x in np.argwhere(mask):
        candidates = np.flatnonzero(~mask[:, y, x])
        if len(candidates):
            nearest = candidates[np.argmin(np.abs(candidates - t))]
            prediction[t, y, x] = tokens[nearest, y, x]
        else:
            prediction[t, y, x] = fallback
    return prediction.reshape(cubes.shape)


def attention_pair_counts(encoder_mask: np.ndarray, decoder_input: np.ndarray) -> dict[str, int]:
    """Quadratic self-attention logit counts for one head/layer."""
    encoder_mask = np.asarray(encoder_mask, dtype=bool)
    decoder_input = np.asarray(decoder_input, dtype=bool)
    total = encoder_mask.size
    visible = int((~encoder_mask).sum())
    decoder = int(decoder_input.sum())
    return {
        "tokens_total": total,
        "tokens_encoder_visible": visible,
        "tokens_decoder": decoder,
        "encoder_full_pairs": total * total,
        "encoder_masked_pairs": visible * visible,
        "decoder_full_pairs": total * total,
        "decoder_dual_pairs": decoder * decoder,
    }
