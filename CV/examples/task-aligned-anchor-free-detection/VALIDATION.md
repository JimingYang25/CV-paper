# Executed validation — task-aligned anchor-free detection

**Date:** 2026-10-28 (Asia/Shanghai)  
**Environment:** Python 3.12.14, NumPy 2.3.5, Matplotlib 3.10.8  
**Scope:** mathematical and behavioral reference tests on deterministic synthetic data. No COCO training or external checkpoint evaluation was run.

## Commands executed

From the repository root:

```bash
python CV/examples/task-aligned-anchor-free-detection/checks.py
python CV/examples/task-aligned-anchor-free-detection/demo.py
python CV/examples/task-aligned-anchor-free-detection/application.py
MPLBACKEND=Agg python CV/examples/task-aligned-anchor-free-detection/make_figure.py
```

## Check result

```text
28/28 checks passed
```

The checks cover stable activations, DFL target construction and decoding, feature-grid coordinates, distance/box round trips, IoU/CIoU, QFL, task-aligned assignment and conflict resolution, class-aware NMS, controlled ablations, and the tactile-motion safety gate.

## DFL representation ablation

This compares representational quantization only; it is not a trained-detector accuracy experiment.

```text
hard_bin_mae:                0.250938660098
two_bin_expectation_mae:     0.000000000000
hard_bin_max_error:          0.499755792910
two_bin_max_error:           0.000000000000
```

The ideal DFL two-bin target exactly reconstructs in-range continuous distances by expectation, up to floating-point arithmetic.

## Score/localization microcase

```text
score_only_index:      0
score_only_iou:        0.25
task_aligned_index:    1
task_aligned_iou:      0.81
```

Score-only top-1 chooses the 0.99-confidence but poorly localized candidate. With `alpha=1`, `beta=6`, task alignment chooses the lower-confidence, much better localized candidate.

## Thirty-seed assignment ablation

```text
trials:                         30
score_only_mean_iou:             0.585848
task_aligned_mean_iou:           0.719097
median_paired_gain:              0.125664
task_aligned_wins:              30 / 30
```

This controlled experiment deliberately makes class confidence only weakly correlated with IoU. It validates the selection mechanism; it does not estimate real-dataset AP.

## Integrated visual-tactile application

Healthy marker field:

```text
support:                         12
surviving_detections:            13
estimated_dx_px:                  2.450193049455
estimated_dy_px:                 -1.368226217768
translation_error_px:             0.084700407346
motion_residual_p90_px:            0.360033716571
deploy_control_signal:            True
```

Corrupted marker field:

```text
support:                         12
surviving_detections:            13
estimated_dx_px:                  2.484337482470
estimated_dy_px:                 -1.564842126791
translation_error_px:             0.277946331281
motion_residual_p90_px:            5.938326495539
deploy_control_signal:           False
```

The negative control retains high support but injects four inconsistent marker motions. The residual gate rejects it, showing why detector confidence and NMS alone are insufficient for control.

## Figure generation

The SVG was generated successfully at:

```text
CV/assets/task-aligned-anchor-free-detection/tal-dfl-pipeline.svg
```

It is derived entirely from the included deterministic code and contains no third-party image asset.
