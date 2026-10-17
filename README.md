# CV-paper

Advanced computer vision and vision-language algorithm lessons for engineers.
Each lesson connects mathematical understanding to implementation, evidence,
and deployment. Explanations are in English.

## Lesson index

| Date (Asia/Shanghai) | Category | Lesson | Runnable example |
| --- | --- | --- | --- |
| 2026-10-17 | Attention and detection | [Multi-scale deformable attention: learn where to look](lessons/attention-and-detection/2026-10-17-deformable-attention.md) | [PyTorch, numerical checks, and offset-learning demo](examples/deformable-attention/) |

## Categories

| Category | Scope | Published lessons |
| --- | --- | --- |
| Attention and detection | Spatial attention, object queries, matching, detection | 1 |
| Geometry and 3D | Calibration, correspondence, depth, reconstruction, pose | 0 |
| Representation learning | Self-supervision, contrastive learning, visual encoders | 0 |
| Segmentation and dense prediction | Masks, spatial decoding, fine detail | 0 |
| Video and temporal reasoning | Tracking, optical flow, temporal models | 0 |
| Vision-language learning | Alignment, grounding, multimodal fusion, VLMs | 0 |
| Generative vision | Diffusion, restoration, conditional generation | 0 |
| Efficient inference | Quantization, ONNX, TensorRT, edge profiling | 0 |

## Reading and running

Begin with a lesson's problem and conceptual map, then derive its mathematics,
run the small example, and inspect the evidence and failure cases. Illustrations
are original unless explicitly attributed. Published benchmark results are
distinguished from locally executed checks.

For today's example:

```bash
git clone https://github.com/JimingYang25/CV-paper.git
cd CV-paper/examples/deformable-attention
python -m pip install -r requirements.txt
python checks.py
python demo.py
python make_figure.py
```

See [validation details](examples/deformable-attention/VALIDATION.md) for the
tested environment and the limits of those checks. The example is a readable
attention operator and synthetic optimization experiment; it does not include
a pretrained object detector.

Scheduled lessons are configured to start at 06:00 Asia/Shanghai; publication
follows preparation and verification. This repository index records actual
published lessons.
