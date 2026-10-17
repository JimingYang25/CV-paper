# Validation record �� 2026-10-17

## Environment

- Windows, x86-64.
- Python 3.14.6.
- PyTorch 2.11.0+cu128; all recorded checks and the optimization demo ran on CPU.
- Numerical interpolation/gradient checks used float64; the demo used float32.
- No full detector training or exported-runtime inference was performed.

## Executed commands

```bash
python checks.py
python demo.py
python make_figure.py
```

All commands completed with exit code 0. The figure was rendered to SVG and a
local PNG preview, which was visually inspected for labels and layout.

## Numerical checks

Six tests passed:

| Check | Independent evidence |
| --- | --- |
| Bilinear sample | Hand-calculated value 4.5 from a known 2��2 map |
| Multi-scale aggregation | Scalar oracle with explicit neighbor indexing, including B=2, M=2, non-square maps, multiple levels and out-of-map samples |
| Zero-padding boundary | Response 0.25 at the outer corner of a 2��2 ones map; response 0 far outside |
| Offset units and normalization | Known `(dx,dy)=(2,1)` on rectangular maps and unit total attention mass per head |
| Numerical gradients | `torch.autograd.gradcheck` for values, sample coordinates and logits, away from interpolation knots |
| Full module backward | Correct output shape, finite gradients, nonzero offset/weight-projection gradients |

The scalar forward oracle was compared at `rtol=atol=1e-11`. Gradient checking
used finite-difference epsilon `1e-6`, absolute tolerance `1e-5`, relative
tolerance `1e-3`. These checks did not compare against the upstream custom CUDA
kernel and do not establish deployment performance.

## Synthetic optimization result

Seed 7, 160 Adam steps, learning rate 0.08, one CPU thread. Value and output
projections were fixed to identity; only the offset bias was optimized.

```json
{
  "initial_response": 0.757465124130249,
  "final_response": 0.999541163444519,
  "target_xy": [17.0, 10.0],
  "learned_sample_xy": [16.997819900512695, 10.006311416625977],
  "learned_offset_xy": [1.9978199005126953, -0.9936885237693787],
  "error_pixels": 0.0066773355938494205
}
```

Both assertions passed: the response improved and localization error was below
0.2 feature pixels. Floating-point results may differ slightly across versions.
This is a smooth synthetic spatial retrieval task, not a learned detector or
evidence for real-world sensor performance.

## Not measured

COCO AP, real sensor generalization, CUDA kernel parity or speed, GPU memory
peaks, ONNX/TensorRT export and numerical parity, FP16/INT8 behavior, or Jetson
latency and power. The lesson proposes how to investigate those questions and
does not report hypothetical measurements as results.
