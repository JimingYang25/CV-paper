# Executed validation — normalized eight-point essential matrix

Date: 2026-10-21 (Asia/Shanghai)

## Environment

- Linux 6.18.55 x86_64, glibc 2.39
- Python 3.12.14
- NumPy 2.3.5
- Matplotlib 3.10.8
- `OPENBLAS_NUM_THREADS=1`
- Geometry and search: CPU `float64`; plotting only: Matplotlib

No OpenCV, GPU, ONNX, TensorRT, Jetson, real camera, or learned matcher was used. The implementation is an educational normalized eight-point reconstruction with a compact RANSAC/inner-refit wrapper, not OpenCV's five-point USAC solver.

## Commands actually executed

```bash
OPENBLAS_NUM_THREADS=1 python checks.py
OPENBLAS_NUM_THREADS=1 python demo.py
OPENBLAS_NUM_THREADS=1 python demo.py --ablations
OPENBLAS_NUM_THREADS=1 python demo.py --parallax
OPENBLAS_NUM_THREADS=1 python application.py
OPENBLAS_NUM_THREADS=1 python make_figure.py ../../assets/eight-point-essential-matrix/relative-pose-and-parallax.svg
python -m py_compile essential_geometry.py demo.py application.py checks.py make_figure.py
```

## Numerical checks

All 16 checks passed:

1. exact calibrated epipolar constraint;
2. exact normalized eight-point recovery;
3. essential singular values `(s, s, 0)`;
4. pixel-domain fundamental-matrix relation;
5. Sampson error invariance to matrix scale;
6. exact four-way pose decomposition and cheirality selection;
7. triangulation scale follows the chosen unit translation;
8. translation-magnitude ambiguity;
9. Hartley conditioning of a large-coordinate design matrix;
10. resize/crop conjugation of pixels and intrinsics;
11. deterministic RANSAC and final-mask consistency;
12. robust recovery with random outliers for the fixed main seed;
13. noisy outlier-free recovery;
14. explicit rejection of a zero-baseline essential matrix;
15. correct right nullspace in the minimal `8 x 9` SVD case;
16. zero-support adaptive-stopping edge case without division by zero.

The conditioning check used a constructed 80-match coordinate set. Its NumPy `cond(A)` changed from `388380838.1661373` before centering/scaling to `48.44270378005741` afterward. This is a numerical check, not a speed or accuracy benchmark.

## Main deterministic experiment

Seed `20261021`: 240 matches, of which 160 are noisy genuine projections and 80 are independent random outliers. Gaussian coordinate noise has standard deviation `0.55 px` on each axis in both images. The true baseline is `0.28` arbitrary scene units; depths are sampled from `[4, 10]` units. The RANSAC threshold is `3 px`.

| Estimator | Rotation error | Translation-direction error | Support | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| All-match eight-point | 91.2161° | 129.0887° | not thresholded | — | — |
| Eight-point RANSAC + compact inner refits | 0.6084° | 5.6111° | 160 | 0.99375 | 0.99375 |

The robust run used 1,675 outer draws, all of which produced valid hypotheses. Its median Sampson residual among selected support was `0.5397 px`. The selected cheirality counts were `(160, 0, 0, 0)` in the code's deterministic candidate order.

## Twenty-seed study

The study changes random 3-D points, image noise, outliers, and RANSAC sampling while retaining one synthetic scene distribution. It comprises 20 trials per listed variant. Equal trial counts do not imply equal runtime.

| Quantity | Median | P90 | Worst |
|---|---:|---:|---:|
| All-match rotation error | 79.4689° | 146.3624° | 163.6153° |
| All-match translation-direction error | 67.9245° | 143.5970° | 154.7821° |
| Robust rotation error | 0.4198° | 0.8694° | 3.2050° |
| Robust translation-direction error | 3.6070° | 9.6911° | 129.3011° |
| Clean conditioned rotation error | 0.2832° | 0.3987° | 0.5708° |
| Clean conditioned translation-direction error | 1.5917° | 4.5934° | 9.1917° |
| Clean unconditioned rotation error | 0.2483° | 0.6370° | 0.6937° |
| Clean unconditioned translation-direction error | 10.3990° | 18.1745° | 27.8557° |

Robust support had median 161 matches. Precision had median `0.99377`, 10th percentile `0.98638`, and minimum `0.97561`; recall had median `1.0`, 10th percentile `0.99375`, and minimum `0.94375`. One of 20 robust trials exceeded 45° translation-direction error despite those apparently strong masks. This failure is retained intentionally: epipolar support and rotation quality do not certify translation direction.

## Low-parallax study

Twenty noisy, outlier-free trials were run at each baseline. The rotation remained fixed at 4°. “Total pixel motion” is omitted below because it mixes rotation and translation and is not a parallax measure.

| Baseline / scene unit | Median rotation error | Median translation-direction error | Translation P90 |
|---:|---:|---:|---:|
| 0.400 | 0.2227° | 1.1832° | 3.4501° |
| 0.160 | 0.2130° | 2.9374° | 9.6622° |
| 0.060 | 0.2238° | 15.9485° | 26.4759° |
| 0.020 | 0.1940° | 41.2158° | 132.7580° |
| 0.008 | 0.1577° | 65.9348° | 139.7409° |

This controlled result supports only the local claim that translation direction becomes unstable as translational parallax shrinks under this noise/model setup. It is not a universal threshold.

## Integrated robot-camera example

The synthetic application had 180 genuine and 70 random matches. Vision estimated 181 support matches, a `1.0428°` rotation error, and a `3.5800°` translation-direction error. An explicit external input supplied the `0.28 m` baseline magnitude. The resulting camera-1-to-camera-2 translation was:

```text
[0.27347149, 0.01676977, 0.05772450] m
```

The camera-2 center in camera-1 coordinates, which is `-R^T t` rather than `t`, was:

```text
[-0.26745047, -0.01552904, -0.08141926] m
```

These are errors against synthetic truth, not real robot localization results.

## Claims not tested

- real-image matching, lens-distortion removal, rolling shutter, dynamic scenes, repeated texture, or planar degeneracy recovery;
- calibrated confidence intervals, loop closure, bundle adjustment, or multi-frame scale recovery;
- OpenCV five-point, USAC, MAGSAC++, PROSAC, or the complete LO-RANSAC algorithm;
- latency, memory, power, FP32/FP16 sensitivity, ONNX export, TensorRT execution, or Jetson performance;
- real VLA, surgical, endoscopic, or visual-tactile accuracy.

