# Validation — LO-RANSAC-style Robust Homography

Executed 2026-10-20. Independent educational reconstruction; not an author-code or OpenCV USAC benchmark reproduction.

## Environment and boundaries

- Python 3.12.14, Linux CPU.
- NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8.
- Float64 geometry and fitting; float32 synthetic image intensities/resampling.
- OPENBLAS_NUM_THREADS=1 for recorded numerical commands.
- No PyTorch, OpenCV, CUDA, ONNX, TensorRT, Jetson or real camera execution.
- No external images, detectors, descriptors, pretrained weights, tactile data, or surgical data.
- The optional OpenCV integration snippet in the lesson was **not executed**.

## Commands

Run from CV/examples/lo-ransac-homography:

```bash
OPENBLAS_NUM_THREADS=1 python checks.py
OPENBLAS_NUM_THREADS=1 python demo.py
OPENBLAS_NUM_THREADS=1 python demo.py --ablations
OPENBLAS_NUM_THREADS=1 python demo.py --failure-case
OPENBLAS_NUM_THREADS=1 python application.py
OPENBLAS_NUM_THREADS=1 python make_figure.py ../../assets/lo-ransac-homography/robust-planar-registration.svg
python -m py_compile robust_homography.py demo.py application.py checks.py make_figure.py
```

All five Python files passed compilation. The generated SVG was also rendered to PNG and visually inspected for readable labels, units, geometry, masks, and clipping. The preview PNG is not committed.

## Sixteen numerical checks passed

| Check | Acceptance condition |
|---|---|
| Exact four-point DLT and homography scale | Matrix agreement within 1e-9 absolute; projections unchanged under multiplication by 1e-14 within 1e-10 |
| Exact nonminimal DLT | Clean grid projections agree within 1e-9 absolute |
| Three collinear points in minimal sample | Rejected with ValueError |
| Globally collinear data | No fabricated robust model; raises ValueError |
| Exact finite-population probability | Equals exhaustive four-subset enumeration for N=10, I=6; zero/all-inlier cases checked |
| Trial-budget calculation | Reaches 0.99 sampling probability for N=20, I=6, m=4; one fewer draw does not; zero/one probability edges checked |
| Residual units and threshold | Squared distance of (3,4) is 25; fails threshold 4 and passes threshold 5 |
| Points at projective infinity | Receive infinite error, not inlier status |
| Consensus versus MSAC | Explicit residual fixture ranks models differently under the two objectives |
| Determinism and mask consistency | Same seed produces identical matrix in this environment; returned mask equals final-model residual thresholding |
| Paired fixed-budget local optimization | Same outer draw/rejection counts; selected score is not worse with local optimization |
| Noisy outlier recovery | Clean-grid RMS below 1 px; selected-support precision above 0.95; genuine-match recall above 0.90 |
| Coordinate-system changes | H_new = A_destination H inverse(A_source) preserves projections within 1e-10 absolute |
| Inverse image-warp direction | Exact identity for grayscale/RGB and correct one-pixel translation |
| Integrated synthetic registration | Source hull fraction above 0.5; valid-region intensity MAE below 0.04 on [0,1] |
| Coherent wrong-plane failure | Support above 120, fewer than 10 intended matches, intended-plane grid error above 10 px |

Acceptance thresholds are fixture-specific numerical tests, not universal quality gates or correctness probabilities.

## Single-run setup and results

Data seed and estimator seed: 20261020.

- 250 pairs, 100 genuine, 150 wrong destination identities.
- Clean source points uniform in [30,610] by [30,450] pixels.
- Both genuine source and destination coordinates get independent 0.7-pixel per-coordinate Gaussian noise.
- Wrong destination positions uniform in [0,720] by [0,540].
- True H has rows [1.02,0.08,35], [-0.04,0.98,22], [0.00008,-0.00006,1].
- A separate clean 15-by-11 grid (165 points) evaluates destination-pixel vector RMS.
- Threshold 3 px; consensus-first scoring with truncated-square tie-break.
- Exact finite-population adaptive stopping; nominal confidence 0.999; draw limits 32 to 3000.
- Inner local fits: ten restarts, at most twelve points in each initial local sample, four narrowing thresholds, at most 64 points per local fit.
- Separate random streams preserve identical outer-sample prefixes for paired comparisons.

| Output | Recorded value |
|---|---:|
| All-match DLT held-out grid RMS | 634.140671985 px |
| Robust held-out grid RMS | 0.346216620 px |
| Selected support | 97 |
| Selected-support precision / genuine recall | 1.0 / 0.97 |
| Draws / valid hypotheses / rejected samples | 314 / 313 / 1 |
| Local calls / local fitting attempts | 3 / 144 |
| Stopped at maximum cap | False |

Small floating-point differences are expected across library/BLAS versions.

## Paired local-optimization ablation

One fixed observation set, 20 estimator seeds 20261020–20261039, 300 outer draws per run, adaptive stopping disabled. There are 80 estimator runs total. Equal outer sampling is **not equal compute or equal runtime**. No estimator timing comparison was recorded.

| Variant | Median grid RMS (px) | P90 (px) | Worst (px) | Runs >2 px | Mean support | Mean local calls / fitting attempts |
|---|---:|---:|---:|---:|---:|---:|
| Minimal only | 2.603519297 | 5.804880805 | 13.543358961 | 14/20 | 69.75 | 0 / 0 |
| Final refit only | 0.689795005 | 1.495318731 | 6.826463717 | 2/20 | 91.30 | 0 / 0 |
| Simple local + final | 0.402054068 | 0.736536426 | 1.901610025 | 0/20 | 96.55 | 4.4 / 4.4 |
| Inner local + final | 0.346216620 | 0.810131236 | 1.031990071 | 0/20 | 97.35 | 4.4 / 227.7 |

The inner variant has a better median and worst case here, but its P90 is worse than the simple variant's P90. Percentiles use NumPy's default interpolation. The >2 px threshold is an explicitly chosen toy error cutoff.

These are search-seed variations on one dataset, not independent-scene trials or confidence intervals.

## Finite-population sampling calculation

For N=20, I=6, sample size 4:

- Exact all-inlier subset probability: 0.0030959752321981426.
- Common approximation: 0.0081.
- Exact draw budget for target sampling probability 0.99: 1486.
- Approximate budget: 567.
- Actual all-inlier sampling success at the approximate budget: 0.8276349975803952.

This calculation assumes known genuine-inlier count and independent outer trials with distinct points inside each trial. It is not a model-correctness guarantee. Degeneracy, noisy hypotheses, estimated/false support, wrong-plane selection and finite caps invalidate that interpretation.

## Structured failure and application

The structured failure uses 100 intended-plane and 150 coherent distractor-plane matches. Noise remains 0.7 px per coordinate in both images. The estimator selects support 148 containing **zero intended-plane matches**. Its error against the intended plane on the common clean held-out grid is 38.488518373 px. This test succeeds by demonstrating an intended **semantic failure** of maximum consensus.

The planar-background registration application uses a [480,640] synthetic reference and [540,720] current image. The matches are generated, not detected from these images.

- Source support convex-hull fraction: 0.723914481.
- Valid rectified output fraction: 0.971041667.
- Valid-region intensity MAE on [0,1]: 0.006639454.
- This intensity error includes two bilinear resamplings and is not a metric pose/tactile accuracy measurement.
- Support/coverage gates are heuristics and can accept a coherent wrong plane.

## Not established

No real-image generalization, detector quality, learned matching, object identity reliability, contact deformation, force/depth inference, endoscopic reliability, calibrated uncertainty, FP32/INT8 geometry parity, latency/power, or hardware deployment was validated.

This implementation uses unweighted DLT and a chosen local schedule. It is not a complete reproduction of the weighted/refined LO+ procedure, USAC, MAGSAC++, or the 2025 paper's experimental suite.
