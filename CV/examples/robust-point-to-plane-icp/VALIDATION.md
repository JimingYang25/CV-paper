# Executed validation — robust point-to-plane ICP

Date: 2026-10-26 (Asia/Shanghai)

This file records commands actually executed in the lesson directory. These are
deterministic synthetic checks, not claims about a public benchmark or a production
sensor.

## Environment used

- Python 3.12
- NumPy 2.2.6
- Matplotlib 3.10.6 (figure generation only)
- CPU, `float64`

## Commands

```bash
python checks.py
python demo.py --trials 12
python application.py
python make_figure.py
```

## Numerical checks

`checks.py` completed with **21/21 passing checks**. Coverage includes:

- the skew/cross-product identity and an exact zero-twist SE(3) exponential;
- proper rotations, rigid inverse round-trips, and translation-free normal transforms;
- exact nearest-neighbor indices and distances;
- Huber costs and IRLS weights;
- the analytic left-perturbation Jacobian versus finite differences, with maximum
  absolute error `1.210e-07` at finite-difference step `1e-7`;
- the expected three-dimensional nullspace for a single plane;
- full-rank geometry for a three-surface corner;
- robust registration accuracy, coverage, conditioning, and deterministic repeat;
- acceptance of a good depth-to-model prediction and rejection of a poor one.

One seeded check (`seed=8`) recovered the known pose with:

- rotation error: **0.02519 degrees**;
- translation error: **0.08500 mm**;
- retained-source coverage: **0.7380**;
- final normal-matrix condition number: **10.916**.

The same outlier-corrupted input produced `3.120 mm` translation error with plain,
untrimmed least-squares point-to-plane ICP; the robust policy produced `0.085 mm`.
This is a controlled example, not a general superiority guarantee.

## Twelve-seed paired ablation

Every method received the same geometry, pose, noise draw, and 15% synthetic source
outliers for each seed. Medians over seeds `0..11` were:

| Variant | Rotation error | Translation error | Iterations |
|---|---:|---:|---:|
| Huber + trimming + normal gate, point-to-plane | **0.03576 deg** | **0.29851 mm** | 9.0 |
| Plain untrimmed point-to-plane | 0.39611 deg | 3.12955 mm | 5.0 |
| Trimmed point-to-point SVD | 1.88688 deg | 13.65561 mm | 15.5 |

Across those 12 runs, robust point-to-plane errors ranged from `0.00714` to
`0.08785` degrees and `0.08500` to `0.61696 mm`. Plain point-to-plane translation
errors ranged from `1.59977` to `4.00573 mm`; trimmed point-to-point errors ranged
from `13.12155` to `14.57665 mm`.

Interpretation boundary: the point-to-point baseline deliberately shares the local
nearest-neighbor policy but not a robust kernel or normal gate. The table isolates
the complete policy used by the tutorial; it does not attribute all improvement to
the residual metric alone.

## Tracking-gate application

The good motion prediction was accepted with `0.03281 deg` rotation error,
`0.29974 mm` translation error, `0.73114` coverage, and `0.97211 mm` residual RMSE.
A prediction translated by an additional `[0.30, -0.24, 0.18] m` was rejected for
insufficient coverage (`0.08093`). Its deceptively small retained residual
(`0.93381 mm`) demonstrates why residual-only gating is unsafe: a tiny selected
subset can fit well while most of the scan is unmatched.

## Deliberate failure retained

For a single `z=0` plane, the eigenvalues of the unweighted normal matrix were:

```text
[0, 0, 0, 48.13333333, 65.51481481, 361]
```

The three zero modes correspond to translation within the plane and rotation about
its normal. Damping can produce a numeric step, but it cannot manufacture missing
geometric information.

## Scope

No public dataset, GPU, ONNX export, TensorRT engine, CUDA implementation, real
depth camera, tactile sensor, or surgical dataset was evaluated. The brute-force
nearest-neighbor reference is intentionally educational and is not a latency
benchmark.
