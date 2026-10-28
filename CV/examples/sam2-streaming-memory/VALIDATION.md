# Executed validation — SAM 2 streaming-memory mechanism study

Validated on 2026-10-29 (Asia/Shanghai lesson date) in this environment:

- Python `3.12.14`
- NumPy `2.3.5`
- Matplotlib `3.10.8`
- Linux `6.18.55`, x86-64, glibc `2.39`

## Commands executed

```bash
python -m py_compile CV/examples/sam2-streaming-memory/*.py
python CV/examples/sam2-streaming-memory/checks.py
python CV/examples/sam2-streaming-memory/demo.py
python CV/examples/sam2-streaming-memory/application.py
MPLBACKEND=Agg python CV/examples/sam2-streaming-memory/make_figure.py
```

## Results

`checks.py`: **38/38 passed**. The assertions cover stable sigmoid/softmax, normalized features, temporal encoding, a hand-computed attention microcase, query chunk equivalence, region/boundary metrics, pointer pooling, conditioning retention, recent-memory FIFO eviction, token shapes, bounded memory reads, occlusion suppression, corrective-prompt state, controlled ablations, and the tactile deployment gate.

Single-seed visible-frame metrics (`seed=7`):

| Variant | J | F | J&F |
|---|---:|---:|---:|
| pinned prompt only | 0.706369 | 0.839603 | 0.772986 |
| recent FIFO, skip absent frames | 0.706369 | 0.839603 | 0.772986 |
| recent FIFO + no-object memory | 0.730664 | 0.851347 | 0.791006 |
| no-object memory + correction at frame 16 | 0.764376 | 0.867237 | 0.815806 |

Twelve-seed paired study:

```text
median_recent_memory_gain_over_pinned: 0.000000  (wins 2/12)
median_no_object_memory_gain:          0.018529  (wins 12/12)
median_correction_gain:                0.023202  (wins 12/12)
```

Visual-tactile gate (`seed=21`):

| State | Mean visible IoU | Minimum visible IoU | Occluded false-positive pixels | Deploy |
|---|---:|---:|---:|---:|
| no correction | 0.736507 | 0.500000 | 0 | false |
| correction at frame 16 | 0.763839 | 0.500000 | 0 | true |

The original SVG regenerated successfully at `CV/assets/sam2-streaming-memory/streaming-memory.svg`.

## Scope

These are deterministic synthetic mechanism tests. They do **not** load Meta weights, reproduce a SAM 2 benchmark, establish real-video generalization, or validate a safety-critical tactile/robotic/surgical system. The simulated object-presence score is an imperfect fixture output used to test state policy; it is not copied from the ground-truth mask during inference.
