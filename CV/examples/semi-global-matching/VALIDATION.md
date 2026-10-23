# Executed Validation: Census + Semi-Global Matching

Date: 2026-10-24

This record separates behavior executed for the educational NumPy implementation from claims in the cited papers. The synthetic scene has known disparity, but it is not a real-data benchmark.

## Environment

| Component | Version |
|---|---:|
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| Matplotlib | 3.10.8 |

## Commands

    python checks.py
    python demo.py
    python demo.py --ablations
    python application.py
    python make_figure.py ../../assets/semi-global-matching/census-sgm-stereo.svg
    python -m py_compile sgm_stereo.py demo.py checks.py application.py make_figure.py

Every command exited with status 0.

## Numerical checks

All 15 checks passed:

1. Census descriptors are invariant to a positive affine intensity transform away from clipping;
2. the true integer shift has zero Census-Hamming cost in the valid interior;
3. left- and right-reference cost volumes have the expected shapes and invalid states;
4. a three-disparity hand calculation matches the normalized SGM recurrence;
5. path aggregation is unchanged by adding a disparity-independent constant to one predecessor vector;
6. raw winner-takes-all recovers a clean constant shift;
7. quadratic interpolation recovers the vertex of a 2.25-disparity parabola;
8. left/right consistency uses the coordinate relation `x_R = x_L - d_L`;
9. disparity-to-depth conversion preserves invalid states and matches `Z = fB/d`;
10. resize-aware intrinsic scaling preserves the metric depth implied by disparity;
11. invalid hyperparameters and unsupported Census windows raise explicit errors;
12. the seeded synthetic generator and estimator are deterministic;
13. eight-path SGM improves bad-pixel rate over raw winner-takes-all on the fixed scene;
14. Census cost beats truncated absolute intensity difference under the fixed gain/bias shift;
15. left/right filtering lowers the bad-pixel rate on accepted pixels.

## Fixed synthetic scene

The fixed scene is an 80×128 rectified pair with foreground layers at 3, 6, 9, and 12 px disparity, z-buffered occlusions, right-image gain 1.10, bias 9, and Gaussian noise with standard deviation 2 intensity levels.

| Variant | Coverage | MAE (px) | Bad-1 rate |
|---|---:|---:|---:|
| raw Census WTA | 1.000000 | 0.391107 | 4.910249% |
| two-path SGM | 1.000000 | 0.216305 | 2.304574% |
| four-path SGM | 1.000000 | 0.215718 | 2.304574% |
| eight-path SGM | 1.000000 | 0.219117 | 2.316155% |
| eight-path absolute-difference cost | 1.000000 | 2.438151 | 60.926462% |
| eight-path Census + left/right filter | 0.976144 | 0.165595 | 0.747420% |

The four-path result is slightly better than the eight-path result before filtering on this one synthetic scene. This is useful counterevidence to a simplistic “more paths must be better” claim: path count changes an anisotropic approximation, and the interaction with boundaries, occlusions, and penalties is data dependent.

## Paired eight-seed ablation

All variants used the same eight scene seeds. There were zero failed trials.

| Variant | Median MAE | 90th-percentile MAE | Worst MAE | Median bad-1 | 90th-percentile bad-1 | Worst bad-1 |
|---|---:|---:|---:|---:|---:|---:|
| raw Census WTA | 0.389919 px | 0.409930 px | 0.413222 px | 4.910249% | 5.026057% | 5.188188% |
| two-path SGM | 0.212797 px | 0.223864 px | 0.229939 px | 2.281413% | 2.503764% | 2.536190% |
| four-path SGM | 0.202389 px | 0.213677 px | 0.215718 px | 2.038217% | 2.199189% | 2.304574% |
| eight-path SGM | 0.208782 px | 0.218707 px | 0.219117 px | 2.130863% | 2.251303% | 2.316155% |
| eight-path absolute-difference cost | 2.380254 px | 2.450695 px | 2.479964 px | 59.467284% | 60.829184% | 60.926462% |
| eight-path Census + left/right filter | 0.159120 px | 0.165916 px | 0.166665 px | 0.628842% | 0.761515% | 0.794404% |

For the filtered variant, median, 90th-percentile, and worst accepted-pixel coverage were 0.976433, 0.977163, and 0.977649 respectively. “Worst” means the largest error but the largest coverage; coverage is not an error metric.

## Integrated robot-clearance application

The application applies confidence and left/right-consistency gates, converts disparity to depth with a 320 px focal length and 0.075 m baseline, then evaluates a predefined image corridor. Synthetic truth is used only for reporting the accepted-pixel depth error.

| Quantity | Executed value |
|---|---:|
| corridor pixels | 784 |
| accepted pixels | 711 |
| accepted coverage | 0.906888 |
| median disparity | 9.218750 px |
| median depth | 2.603390 m |
| 10th-percentile depth | 1.986207 m |
| accepted-pixel depth MAE | 0.379277 m |
| decision | `STOP` |

The `STOP` result follows from a conservative 2.30 m near-depth threshold. It is an illustrative perception gate, not a certified safety system.

## What was not validated

No Middlebury, KITTI, ETH3D, or other real stereo dataset was evaluated. OpenCV StereoSGBM, NVIDIA VPI, CUDA, Jetson, ONNX, TensorRT, FP16, INT8, latency, memory, bandwidth, power, and production safety were not tested. The code deliberately uses dense NumPy arrays and Python loops for inspectability, so no production-speed claim is made.
