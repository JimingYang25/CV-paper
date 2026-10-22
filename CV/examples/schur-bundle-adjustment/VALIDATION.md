# Executed Validation: Schur-Complement Bundle Adjustment

Date: 2026-10-23

This record distinguishes executed behavior of the educational NumPy implementation from claims made by cited papers. All commands below were run from this directory with one OpenBLAS thread. Floating-point libraries can change last digits and nonlinear iteration counts.

## Environment

| Component | Version |
|---|---:|
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| Matplotlib | 3.10.8 |

## Commands

    OPENBLAS_NUM_THREADS=1 python checks.py
    OPENBLAS_NUM_THREADS=1 python demo.py
    OPENBLAS_NUM_THREADS=1 python demo.py --ablations
    OPENBLAS_NUM_THREADS=1 python demo.py --gauge
    OPENBLAS_NUM_THREADS=1 python application.py
    OPENBLAS_NUM_THREADS=1 python make_figure.py ../../assets/schur-bundle-adjustment/schur-factorization.svg
    python -m py_compile bundle_adjustment.py demo.py checks.py application.py make_figure.py

Every command exited with status 0.

## Numerical checks

All 18 checks passed:

1. skew matrix equals the cross product;
2. exponential-map rotation is orthonormal with determinant one;
3. analytic pose Jacobian agrees with finite differences;
4. analytic point Jacobian agrees with finite differences;
5. Huber cost and IRLS weight match their definitions;
6. block dimensions and camera–landmark sparsity are correct;
7. Schur and full damped normal-equation increments agree within tolerance;
8. one clean Schur step decreases cost;
9. the clean iterative solve decreases cost;
10. fixed camera and point anchors do not move;
11. joint monocular BA without explicit frame and scale anchors is rejected;
12. global similarity scale leaves pixels invariant;
13. crop/resize intrinsic transformation preserves projected pixel coordinates;
14. pose-only mode keeps landmarks fixed;
15. structure-only mode keeps cameras fixed;
16. repeated seeded solves are deterministic;
17. robust loss beats linear loss on a fixed contaminated reconstruction;
18. the world-to-camera camera-center identity is correct.

## Main contaminated reconstruction

The fixed seed creates 7 cameras, 140 landmarks, 789 observations, 0.8 px Gaussian noise per image axis, and 79 deliberately displaced observations. Camera 0 and landmark 0 are exact anchors.

| Metric | Linear loss | Huber loss |
|---|---:|---:|
| initial objective | 251829.100879 | 25049.465765 |
| final objective | 152274.249216 | 13966.145774 |
| initial all-observation RMSE | 25.265591 px | 25.265591 px |
| final all-observation RMSE | 19.646705 px | 22.572786 px |
| final true-inlier RMSE | 10.193323 px | 4.105506 px |
| median rotation error | 2.288729° | 0.042947° |
| median camera-center error | 194.854602 mm | 9.245501 mm |
| median landmark error | 168.780537 mm | 16.635513 mm |
| accepted / rejected steps | 25 / 0 | 35 / 0 |

The Huber run reached its 35-iteration limit, so its converged flag is false even though every accepted step reduced its robust objective. The linear run satisfies its stopping rule after 25 iterations. The lower all-observation RMSE of linear loss is not a better reconstruction: it reflects fitting the intentionally displaced observations.

## Paired ten-trial ablation

Each trial uses 6 cameras and 90 landmarks with the same synthetic noise and contamination process.

| Variant | Failures | Median inlier RMSE | Median rotation error | Median center error | Median landmark error |
|---|---:|---:|---:|---:|---:|
| pose-only Huber | 0 | 6.390505 px | 0.368030° | 29.964394 mm | 37.998544 mm |
| structure-only Huber | 0 | 6.328365 px | 0.465299° | 18.151060 mm | 60.801194 mm |
| joint linear | 0 | 11.127029 px | 4.422630° | 352.258350 mm | 609.406766 mm |
| joint Huber | 0 | 2.137173 px | 0.271969° | 24.070597 mm | 44.999901 mm |

These are synthetic paired trials, not confidence intervals, runtime comparisons, or real-data evidence. In structure-only mode the camera errors are unchanged initial errors because the cameras are fixed.

## Gauge demonstration

Scaling all camera centers and landmarks by 2.75 around the anchored first camera changes projected pixels by at most 2.2737367544323206e-13 px. Removing the explicit scale anchor causes the solver to stop with:

    joint monocular BA needs explicit camera and scale anchors

## Integrated wrist-camera application

The application uses 8 cameras, 160 landmarks, and 992 observations. Four fixture tracks are chosen using only track length and post-fit residuals; synthetic truth is used only for evaluation.

| Quantity | Executed value |
|---|---:|
| true-inlier RMSE | 0.793609 px |
| median camera-center error | 11.039252 mm |
| final camera rotation error | 0.330241° |
| final camera-center error | 22.490567 mm |
| world task-point error | 23.995155 mm |
| final-camera task-point error | 27.208916 mm |
| accepted / rejected steps | 40 / 0 |

The task-space error remains much larger than the image residual. The application therefore emits a safety gate rather than treating reprojection error as a robot-control guarantee.

## What was not validated

No real BAL dataset, Ceres, g2o, COLMAP, GTSAM, RootBA, sparse Cholesky, PCG, CUDA, Jetson, ONNX, TensorRT, FP32, FP16, INT8, latency, power, or memory benchmark was run. No claim is made about production speed, real-scene accuracy, or deployment safety.
