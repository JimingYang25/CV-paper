# EPnP reconstruction — executed validation

Executed on 2026-10-22. These records describe deterministic synthetic tests in one software environment. They are not benchmark, real-camera, safety, latency, or hardware claims.

## Environment

```text
Python 3.12.14
NumPy 2.3.5
SciPy 1.17.0
Matplotlib 3.10.8
```

`OPENBLAS_NUM_THREADS=1` was set for the numerical runs recorded below.

## Commands

```bash
OPENBLAS_NUM_THREADS=1 python checks.py
OPENBLAS_NUM_THREADS=1 python demo.py
OPENBLAS_NUM_THREADS=1 python demo.py --ablations
OPENBLAS_NUM_THREADS=1 python demo.py --thickness
OPENBLAS_NUM_THREADS=1 python demo.py --failure-case
OPENBLAS_NUM_THREADS=1 python application.py
OPENBLAS_NUM_THREADS=1 python make_figure.py ../../assets/epnp-pose/control-points-and-observability.svg
python -m py_compile epnp_pose.py demo.py application.py checks.py make_figure.py
```

All commands exited with status zero. The SVG was also rendered to a temporary 2,272×696 PNG and inspected for legibility; that review PNG is not part of the repository.

## Numerical checks

All 16 checks passed:

```text
PASS check_projection_and_normalization
PASS check_barycentric_reconstruction
PASS check_projection_nullspace
PASS check_distance_constraint_algebra
PASS check_absolute_orientation
PASS check_exact_nonminimal_epnp
PASS check_refinement_nonworsening
PASS check_crop_resize_conjugation
PASS check_world_scale_equivariance
PASS check_camera_center_convention
PASS check_exact_planar_rejected
PASS check_minimal_inputs_rejected
PASS check_ransac_determinism
PASS check_random_outlier_recovery
PASS check_coherent_distractor_failure
PASS check_invalid_ransac_controls
PASS all 16 checks
```

The exact, noise-free non-minimal test used 50 non-planar points. Rotation error was below `1e-5` degrees, translation error below `1e-10` world units, and pixel RMSE below `1e-9`.

The conservative minimal-input check is intentional. The EPnP paper supports (N\ge4), but this educational reconstruction requires (N\ge6) and does not claim to reproduce every minimal-case recovery branch. Exact planar inputs are also intentionally rejected.

## Main synthetic experiment

Seed: `20261022`.

- 200 total correspondences
- 140 genuine projections
- 60 independent random mismatches
- 0.65 px Gaussian noise standard deviation per coordinate
- 0.32×0.22×0.08 m object extent
- calibrated intrinsics `fx=910`, `fy=895`, `cx=640`, `cy=360`

| Result | All-match EPnP | Robust EPnP + Huber refinement |
|---|---:|---:|
| rotation error | 177.111011° | 0.160003° |
| translation error | 160.163328 mm | 0.301286 mm |
| reprojection RMSE on fit set | 272.664840 px | 0.882692 px |
| selected support | — | 140 |
| precision / recall | — | 1.0 / 1.0 |
| draws / valid hypotheses | — | 96 / 96 |
| median selected residual | — | 0.746208 px |
| beta initializer | 1 | 1 |
| refinement evaluations | — | 4 |
| refinement Jacobian condition | — | 34.784833 |

The true camera center in the object frame was `[-0.2591380969, -0.1154454711, -0.7277559962]` m.

## Twenty-seed ablation

| Variant / metric | Median | P90 | Worst |
|---|---:|---:|---:|
| all-match rotation error | 76.554693° | 114.425994° | 177.111011° |
| all-match translation error | 289.877102 mm | 452.749340 mm | 578.465376 mm |
| oracle closed-form rotation | 0.130730° | 0.236843° | 0.268036° |
| oracle closed-form translation | 0.427676 mm | 1.149310 mm | 1.256602 mm |
| oracle refined rotation | 0.108437° | 0.155468° | 0.213197° |
| oracle refined translation | 0.322599 mm | 0.544653 mm | 0.780166 mm |
| robust refined rotation | 0.108437° | 0.155468° | 0.213197° |
| robust refined translation | 0.322599 mm | 0.544653 mm | 0.780166 mm |

All 20 robust masks exactly matched the random-outlier truth, so robust-refined and oracle-refined values coincide. There were zero runs above either 5° rotation error or 50 mm translation error. This describes only the generated independent-random-outlier distribution.

The clean closed-form winner counts were:

```text
initializer 1: 6
initializer 2: 5
initializer 3: 9
```

## Thickness study

Twenty clean trials were run per object thickness with 0.65 px noise per coordinate.

| Thickness | Median scatter ratio | Rotation median / P90 | Translation median / P90 | Median Jacobian condition |
|---:|---:|---:|---:|---:|
| 0.1200 m | 1.376043e-1 | 0.088733° / 0.148788° | 0.309115 / 0.576162 mm | 25.882505 |
| 0.0400 m | 1.541572e-2 | 0.130005° / 0.252645° | 0.299333 / 0.808572 mm | 44.765198 |
| 0.0120 m | 1.388824e-3 | 0.151564° / 0.277721° | 0.316867 / 0.896705 mm | 50.640458 |
| 0.0040 m | 1.543273e-4 | 0.149717° / 0.264881° | 0.331636 / 0.915449 mm | 51.402017 |
| 0.0012 m | 1.388959e-5 | 0.146440° / 0.261365° | 0.334302 / 0.904423 mm | 51.467125 |
| 0.0004 m | 1.543289e-6 | 0.145100° / 0.260659° | 0.333419 / 0.901090 mm | 51.487453 |

No near-planar trial was rejected at the implementation threshold of `1e-8`. Exact planar input is rejected. The observed accuracy stability must not be generalized beyond this favorable view, noise level, and point distribution.

## Coherent wrong-object failure

The failure scene had 100 matches consistent with the intended pose and 150 consistent with a distractor pose.

```text
selected support: 150
selected intended matches: 0
selected distractor matches: 150
error to intended: 38.144475 deg, 169.064111 mm
error to distractor: 0.136734 deg, 0.134683 mm
```

This deliberately demonstrates that geometric consensus has no intended-object identity.

## Integrated fixture application

```text
support: 160
estimated object origin in camera [m]:
  [0.0350538683, -0.0220521409, 0.7806895116]
estimated camera center in object [m]:
  [-0.2579949782, -0.1150962823, -0.7289599586]
grasp point in camera [m]:
  [0.0741182946, -0.0404915271, 0.8139755393]
approach axis in camera:
  [0.2638148974, 0.2189268328, -0.9394002032]
truth errors:
  rotation 0.114628 deg
  translation 0.693575 mm
  transformed grasp point 0.645311 mm
```

The output explicitly warns that a pose estimate is not collision-safe.

## What was not tested

- real cameras, lens distortion, rolling shutter, motion blur, or temporal synchronization;
- OpenCV EPnP, P3P/AP3P, IPPE, SQPnP, USAC, or bundle adjustment comparisons;
- correlated learned-matcher errors, repeated texture, object symmetries, or occlusion;
- covariance calibration or safety thresholds;
- FP32, FP16, INT8, ONNX, TensorRT, Jetson, latency, memory, or power.

The numerical records support the implementation and lesson examples only within their stated scope.
