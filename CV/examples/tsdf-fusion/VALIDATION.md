# Executed Validation: Projective TSDF Fusion

Date: 2026-10-25

This record distinguishes executed behavior of the educational dense NumPy implementation from results claimed by cited papers. Every reported accuracy value comes from an analytic synthetic scene with known geometry; it is not a real-data benchmark.

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
    python make_figure.py ../../assets/tsdf-fusion/projective-tsdf-fusion.svg
    python -m py_compile tsdf_fusion.py synthetic_scene.py demo.py checks.py application.py make_figure.py

Every command exited with status 0.

## Numerical checks

All 16 checks passed:

1. invalid camera and volume configurations are rejected;
2. voxel centres follow `[z,y,x]` storage and metric `(x,y,z)` coordinates;
3. projective distance has the declared sign and rejects voxels beyond the back truncation band;
4. two weighted observations match a hand-derived running average;
5. accumulated weights respect `max_weight`;
6. NaN, zero, and out-of-range depths do not update the volume;
7. confidence scales update precision and zero confidence rejects a sample;
8. `world_from_camera` translation is inverted in the correct direction;
9. non-rigid pose matrices are rejected;
10. linear interpolation recovers a hand-constructed sub-voxel zero crossing;
11. pixel-centre-aware intrinsic resizing preserves a camera ray;
12. reported persistent memory equals the TSDF and weight array storage;
13. repeated seeded fusion is bitwise deterministic;
14. eight-frame fusion improves surface MAE over one frame;
15. tagged-outlier confidence weighting improves over unit weighting;
16. a 4 cm pose bias degrades the surface by more than 2 cm relative to the correct-pose result.

## Fixed synthetic scene

The analytic sloped plane is rendered into eight 80×96 optical-depth frames. Each frame has 12 mm Gaussian noise, 4% missing pixels, 2.5% signed 22 cm outliers, and a synthetic confidence channel that assigns tagged outliers weight 0.05. The dense volume has 48×40×48 voxels at 25 mm resolution and uses 0.703125 MiB for persistent FP32 TSDF and weight arrays.

| Variant | Coverage | Surface z MAE | P95 absolute error | Observed voxels |
|---|---:|---:|---:|---:|
| one frame, unit weight | 0.928733 | 0.013965 m | 0.027498 m | 38.018663% |
| eight frames, unit weight | 1.000000 | 0.004519 m | 0.014861 m | 52.534722% |
| eight frames, confidence weighted | 1.000000 | 0.003190 m | 0.007587 m | 52.534722% |
| confidence weighted, +4 cm pose-z bias | 1.000000 | 0.039988 m | 0.046654 m | 53.157552% |
| confidence weighted, truncation 2 voxels | 1.000000 | 0.003186 m | 0.007494 m | 48.408203% |
| confidence weighted, truncation 8 voxels | 1.000000 | 0.003219 m | 0.007744 m | 60.822483% |

The synthetic confidence channel knows which samples were deliberately corrupted. Its improvement tests the weighting mechanism; it does not demonstrate that a learned real-sensor confidence is calibrated.

## Paired eight-seed ablation

There were zero failed trials.

| Variant | Median MAE | P90 MAE | Worst MAE | Median P95 | Worst P95 | Minimum coverage |
|---|---:|---:|---:|---:|---:|---:|
| one frame, unit | 0.016192 m | 0.017116 m | 0.018287 m | 0.034300 m | 0.125348 m | 0.899321 |
| eight frames, unit | 0.004283 m | 0.004452 m | 0.004519 m | 0.013646 m | 0.014861 m | 1.000000 |
| eight frames, confidence | 0.003203 m | 0.003239 m | 0.003266 m | 0.007778 m | 0.007972 m | 1.000000 |
| confidence + 4 cm pose bias | 0.040085 m | 0.040141 m | 0.040149 m | 0.046537 m | 0.047062 m | 1.000000 |

These paired seeds are regression evidence, not confidence intervals, a real-domain generalization result, or a runtime study.

## Integrated robot application

Six frames observe a table at 1.30 m and a nearer box front at 0.88 m. A fixed metric sweep region contains 200 height-field columns.

| Quantity | Executed value |
|---|---:|
| accepted columns | 156 |
| coverage | 0.780000 |
| surface-z 10th percentile | 0.877837 m |
| expected table z | 1.300000 m |
| intrusion threshold | 1.150000 m |
| decision | `STOP` |

The decision is caused by the fused surface entering the sweep volume. This is an illustrative conservative gate, not a certified robot safety function.

## What was not validated

No real RGB-D, stereo, Kinect, RealSense, ZED, Middlebury, KITTI, TUM RGB-D, Replica, or surgical dataset was evaluated. ICP tracking, loop closure, dynamic-object handling, de-integration, Marching Cubes, ESDF construction, Open3D, voxblox, nvblox, CUDA, Jetson, ONNX, TensorRT, FP16, INT8, latency, bandwidth, power, and production memory were not tested. No claim is made about real-scene accuracy, real-time performance, or deployment safety.
