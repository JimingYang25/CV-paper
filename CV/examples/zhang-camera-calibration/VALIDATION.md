# Validation — Zhang Camera Calibration

Executed 2026-10-19. These are deterministic synthetic numerical experiments, not a reproduction of Zhang's published benchmarks or evidence of real-camera/robot accuracy.

## Environment

- Python 3.12.14, Linux CPU, float64 estimation.
- NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8.
- OPENBLAS_NUM_THREADS=1 for the commands below.
- No PyTorch, OpenCV, CUDA, ONNX, TensorRT, or Jetson execution.
- No images, pretrained weights, external datasets, or actual hardware were used.

## Commands

Run from CV/examples/zhang-camera-calibration:

```bash
OPENBLAS_NUM_THREADS=1 python checks.py
OPENBLAS_NUM_THREADS=1 python demo.py
OPENBLAS_NUM_THREADS=1 python ablations.py
OPENBLAS_NUM_THREADS=1 python application.py
OPENBLAS_NUM_THREADS=1 python make_figure.py ../../assets/zhang-camera-calibration/observability-and-refinement.svg
```

The generated SVG was also rendered to PNG and visually inspected for legibility, axes, clipping, and consistency with the numerical output. The preview PNG is not committed.

## Numerical checks

All **10 tests passed**.

| Test | Acceptance condition |
|---|---|
| Pinhole hand calculation | (0.1,0.05,1) camera point → (400,281) px at absolute tolerance 1e-12 |
| Four-point DLT | Recover specified homography from minimal four correspondences, absolute tolerance 1e-9 |
| Collinear layout | Raises ValueError, no fabricated camera |
| Exact general intrinsics and poses | Focal lengths, principal point, skew recovered to absolute tolerance 1e-6; rotations/translations to 1e-8; per-view residual below 1e-7 px |
| Parallel plane normals | General intrinsic initialization raises ValueError |
| Board units | Metres → millimetres leaves intrinsics unchanged to 1e-6 and scales translation by 1000 to 1e-6 |
| Resize/crop coordinate contract | Transformed projections match new intrinsics to absolute tolerance 1e-12 |
| Radial hand calculation | Center fixed; normalized (0.5,0) → (0.47625,0), tolerance 1e-12 |
| Noisy joint refinement | Lowers residual; RMS below 0.30 px; both relative focal errors below 1%; known-pose synthetic held-out error below 2 px; diverse rank 5, parallel rank 2 |
| Integrated plane application | True-camera round trip within 1e-10 metres; fitted-camera synthetic point error below 1 mm |

Acceptance bounds describe this controlled fixture only. They are not proposed universal calibration-quality criteria.

## Synthetic experiment specification

- Seed 20261019; NumPy default_rng.
- Twelve views, 54 inner corners arranged 9 by 6; 0.025-metre spacing; centered board coordinates; board Z=0.
- Camera focal lengths (920,900) px, principal point (640,480) px, zero skew.
- True normalized radial coefficients (-0.12,0.03).
- Nominal image 1280 by 960; accept a view only if all corners remain inside a 20-pixel border.
- XYZ Euler rotations sampled in radians from [-0.55,0.55], [-0.55,0.55], [-0.3,0.3].
- Board-origin translation ranges: X [-0.14,0.14], Y [-0.10,0.10], Z [0.38,0.62] metres.
- Independent Gaussian coordinate noise, zero mean, 0.15 px standard deviation.
- General five-parameter intrinsic initialization; zero-skew, two-radial-term nonlinear fit.
- Residual is predicted minus observed distorted pixels. Reported RMS averages squared two-dimensional vector lengths over points, not components.

## Recorded outputs

| Output | Value |
|---|---:|
| Initial zero-skew, zero-distortion RMS | 2.021746743 px |
| Two-radial-term refined RMS | 0.205512233 px |
| Refined focal lengths | 920.770792151, 900.656264215 px |
| Refined principal point | 640.553644580, 480.223925748 px |
| Refined radial coefficients | -0.120948450, 0.034464648 |
| Absolute relative focal errors | 0.083781756%, 0.072918246% |
| Three extra clean views with known true poses, pixel disagreement | 0.556774490 px RMS |
| True-camera application round trip | Passed, absolute tolerance 1e-10 m |
| Fitted-camera application intersection error | 0.438793091 mm |

The optimizer reported success and nfev=6 (function evaluations). This counter is not an iteration count or a wall-clock benchmark and does not count every finite-difference residual call.

### Radial-model ablation

All fits use the same noisy observations and initialization; only the fitted radial term count changes.

| Terms | Vector RMS (px) | Focal error X / Y (%) | Principal-point error (px) |
|---|---:|---:|---:|
| 0 | 0.869283880 | 4.208941525 / 3.556398143 | 28.904662814 |
| 1 | 0.214172832 | 0.264906484 / 0.275611277 | 0.844030828 |
| 2 | 0.205512233 | 0.083781756 / 0.072918246 | 0.597214419 |

## Limitations of this validation

The generator and estimator share a pinhole-plus-two-radial-term model. The held-out check uses true poses, and the robot-fixture application uses a true independently supplied fixture pose. Neither exercises real pose estimation, corner detection, refraction, rolling shutter, blur, board manufacturing errors, outliers, autofocus, or deployment precision.

No timing, model export, calibration confidence intervals, real optical-tactile sensing, surgical use, or hardware performance was validated. The code rejects some invalid inputs, but is an educational implementation rather than a complete robust calibration library.
