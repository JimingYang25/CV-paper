# ASFormer temporal segmentation — NumPy mechanism reference

This directory accompanies the [2026-10-31 ASFormer lesson](../../lessons/video-and-motion/2026-10-31-asformer-temporal-segmentation.md). It is a small, auditable mechanism study, **not** a trained reproduction of the official network and not weight-compatible with the authors' PyTorch repository.

## What is implemented

- numerically stable masked single-head attention;
- shifted local-window masks and exact attention-pair counting;
- length-3 dilated temporal mixing;
- local-to-global probability refinement;
- a three-stage decoder analogue with the official `exp(-3s)` residual-weight schedule;
- both the paper's printed probability-smoothing loss and the released code's truncated log-probability loss;
- frame accuracy, normalized segment edit score, and segmental F1;
- a synthetic surgical-phase integration with a deliberately failing coherent-error test.

The decoder analogue follows the **paper's stated value-space interpretation**. The released code's observed tensor path differs; read the lesson's paper-versus-code audit before adapting this reference.

## Run

```bash
python -m pip install -r requirements.txt
python checks.py
python demo.py
python ablation.py
python application.py
```

All scripts are deterministic. NumPy is the only dependency.

## Files

- `asformer_math.py` — shapes, attention, temporal mixing, refinement, losses, and metrics.
- `synthetic_sequence.py` — long-video feature and label generator.
- `demo.py` — one end-to-end tensor and metric trace.
- `ablation.py` — 20 paired trials over window schedules and decoder depth.
- `application.py` — conservative surgical-phase assistance gate and coherent-error failure.
- `checks.py` — 34 numerical and behavioral checks.
- `VALIDATION.md` — recorded output from the executed scripts.

For a full training reproduction, use the [official ASFormer repository](https://github.com/ChinaYi/ASFormer) and its dataset instructions. Apply the window-mask correction documented in that repository's README when training new checkpoints; retain the original line only when checkpoint compatibility requires it.
