# Executed validation — VideoMAE tube and dual masking

Validated on 2026-10-30 (Asia/Shanghai lesson date) with:

- Python `3.12.14`
- NumPy `2.3.5`
- Matplotlib `3.10.8`
- Linux `6.18.55`, x86-64, glibc `2.39`

## Commands executed

```bash
python -m py_compile CV/examples/videomae-dual-masking/*.py
python CV/examples/videomae-dual-masking/checks.py
python CV/examples/videomae-dual-masking/demo.py
python CV/examples/videomae-dual-masking/application.py
MPLBACKEND=Agg python CV/examples/videomae-dual-masking/make_figure.py
```

## Results

`checks.py`: **35/35 passed**. Assertions cover cube tokenization and exact inversion, per-cube/per-channel normalization, mask counts and layouts, cyclic running-cell coverage, dual-mask set logic, masked MSE, temporal-copy behavior, attention-pair accounting, and the surgical phase integration gate.

### Temporal-copy leakage probe

Twenty deterministic textured videos were evaluated with a deliberately weak same-spatial-site copy predictor:

```text
mean_random_copy_mse:             0.012121419233
mean_tube_copy_mse:               0.027562473871
median_tube_minus_random_mse:     0.015383466087
tube_is_harder_count:             20/20
```

This verifies that random masking exposes an easy temporal-copy shortcut in this fixture and that tube masking removes it. It does not measure learned representation quality.

### Dual-decoder cyclic coverage

```text
full_masked_loss:                 1.111612543861
cycled_dual_loss:                 1.111612543861
absolute_loss_difference:         2.220446049250313e-16
masked_tokens_covered_once:       true
tokens_total:                     512
tokens_encoder_visible:           48
tokens_decoder:                   280
encoder full / masked pairs:      262144 / 2304
decoder full / dual pairs:        262144 / 78400
```

The loss equality is a fixed-prediction, complete-cycle arithmetic property of the included didactic sampler. It is not a claim that sparse and full decoder training have identical gradients or convergence.

### Synthetic surgical phase integration

| Method | Frame accuracy | Boundary F1, ±6 frames |
|---|---:|---:|
| frame appearance only | 0.516667 | 0.126984 |
| temporal clip descriptor | 0.800000 | 0.727273 |
| clip descriptor + ordered decoding | 0.841667 | 1.000000 |

The ordered path was monotonic and passed the fixture deployment gate. The temporal descriptor is an inspectable stand-in for cached encoder features; it is not a trained VideoMAE model and is not clinical evidence.

The SVG regenerated successfully at `CV/assets/videomae-dual-masking/tube-and-dual-masking.svg`.

## Scope

No official VideoMAE weights or datasets were downloaded, no transformer was trained, and no paper benchmark was reproduced. Paper numbers in the lesson are explicitly attributed to the papers; the results above apply only to deterministic local mechanism tests.
