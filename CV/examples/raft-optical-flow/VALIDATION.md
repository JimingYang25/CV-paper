# Validation record — 2026-10-18

## Executed environment and commands

- Linux x86-64; Python 3.12.14.
- NumPy 2.3.5; Matplotlib 3.10.8.
- CPU, float64 reference calculations. No GPU or pretrained weights used.

```bash
cd CV/examples/raft-optical-flow
python checks.py
python demo.py
python make_figure.py
```

Eight numerical tests passed. The example and figure generator completed with
exit code 0. The figure's PNG preview was inspected before publication; only
the SVG publication figure is committed.

## Mathematical checks

1. Matrix multiplication correlation against an independently indexed scalar
   dot-product oracle, using two batches and non-square maps.
2. Target-only pooling preserves source axes and floors odd target sizes.
3. A hand-calculated bilinear value of 4.5 and zero-padding boundary behavior.
4. Bilinear coordinate derivatives checked by finite differences: `(2,4)`.
5. Known lookup values test `(x,y)` ordering and pyramid coordinate scaling.
6. Materialized and on-demand lookup agree on random batched, odd-sized maps,
   fractional coordinates and boundary/outside-map samples at `atol=rtol=1e-12`.
7. Convex upsampling checks output layout, displacement-unit scaling and
   zero-padding boundary behavior.
8. The translation experiment checks forward-flow sign and the visibility mask.

## Executed synthetic result

```json
{
  "seed": 18,
  "feature_shape": [1,32,12,18],
  "dtype": "float64",
  "true_flow_feature_pixels": [2.0,-1.0],
  "valid_pixels": 176,
  "valid_mean_epe_feature_pixels": 0.0,
  "valid_max_epe_feature_pixels": 0.0,
  "local_peak_displacement_xy": [2,-1],
  "lookup_shape": [1,75,12,18],
  "stored_vs_on_demand_max_abs_error": 8.326672684688674e-17,
  "correlation_pyramid_bytes": 487296
}
```

The synthetic descriptors are deliberately unique and copied to known target
locations. Their matching problem is therefore much easier than optical flow
on real images. Border points without valid correspondences are excluded from
EPE using a known synthetic visibility mask. The fixture neither evaluates nor
trains the RAFT feature encoder, context network, recurrent updater or mask head.

## Scope and exclusions

The executable code covers reference mathematical components. No automatic
differentiation, full RAFT training, upstream custom-kernel parity, pretrained
checkpoint inference, Sintel/KITTI reproduction, or real sensor evaluation was
performed. PyTorch architecture code in the lesson is labeled pseudocode and
was not executed. Memory tables count correlation elements analytically;
they are not measured process/GPU memory or latency. ONNX/TensorRT/FP16/INT8
deployment behavior and Jetson performance remain untested.
