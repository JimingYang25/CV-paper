# Executed validation — DINOv3 Gram anchoring

Date: 2026-10-27 (Asia/Shanghai)

This record covers an independent NumPy reconstruction of the loss mathematics and
a controlled dense-feature experiment. It is **not** a DINOv3 pretraining run, model
benchmark, or reproduction of Meta's reported results.

## Environment used

- Python 3.12
- NumPy 2.2.6
- Matplotlib 3.10.6 (figure generation only)
- CPU, `float64`

## Commands executed

```bash
python -m py_compile *.py
python checks.py
python demo.py --trials 12 --steps 250
python application.py
python make_figure.py
```

## Numerical checks

`checks.py` completed with **22/22 passing checks**. The checks cover:

- stable softmax, shift invariance, temperature sharpening, and centering;
- cross-entropy and exact DINO cross-view pair enumeration;
- center and teacher-parameter exponential moving averages;
- feature normalization and Gram symmetry/unit diagonal;
- invariance of the Gram target to an orthogonal feature-basis rotation and positive
  per-token rescaling;
- an analytic Gram-loss gradient checked against central finite differences, with
  maximum absolute error `6.636e-11` at step `1e-6`;
- correct-teacher refinement, wrong-teacher failure, and checkpoint gating.

For seeded case `8`, 250 Adam steps reduced correct-anchor Gram MSE from
`1.9334e-01` to `8.5284e-06`; the query similarity-map correlation became
`0.999819`. A shuffled teacher was fit to `9.842e-06` loss against itself but
remained `1.979e-01` away from the correct anchor. A low training loss therefore
does not certify a good teacher.

## Twelve-seed paired experiment

Every trial used a structured `8×8` patch grid, 12-dimensional features, a common
degrading direction, Gaussian perturbations, and 12% patch outliers. The experiment
optimizes patch features directly; it isolates the Gram objective and does not train
an image encoder.

| Variant | Median Gram MSE to correct anchor | Median query-map correlation | Median top-8 IoU |
|---|---:|---:|---:|
| Degraded baseline, no refinement | 0.179011 | 0.389123 | 0.186414 |
| Correct Gram anchor, 250 steps | **0.00000619** | **0.999909** | **0.962963** |
| Shuffled wrong anchor, 250 steps | 0.254710 | 0.019680 | 0.098413 |

Across the 12 correctly anchored runs, final Gram MSE ranged from
`4.77846e-06` to `1.16062e-05`, and query-map correlation ranged from `0.999819`
to `0.999980`. Wrong-anchor runs nevertheless fit their own wrong objectives to
between `6.66519e-06` and `1.36571e-05`.

## Visual-tactile checkpoint gate

On seeded calibration case `21`:

- the degraded checkpoint was rejected: Gram MSE `0.155496`;
- the correctly anchored candidate was accepted: Gram MSE `5.01941e-06`, query-map
  correlation `0.999792`, top-8 IoU `0.962963`;
- the wrong-teacher candidate was rejected against the trusted anchor: Gram MSE
  `0.248517`.

This gate is an educational example for **checkpoint validation on unlabeled
calibration frames**. The feature tensors are synthetic. Thresholds are not validated
for a real tactile sensor and must not be copied into a safety-critical system.

## Scope and non-claims

The following were not executed: ViT or ConvNeXt pretraining; DINO, iBOT, KoLeo, or
Sinkhorn training at scale; LVD-1689M or ImageNet; public dense benchmarks; official
DINOv3 checkpoints; PyTorch; CUDA; ONNX; TensorRT; Jetson; real visual-tactile,
robotic, satellite, or surgical data. No paper benchmark is attributed to this code.
