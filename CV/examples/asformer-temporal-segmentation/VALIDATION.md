# Executed validation — 2026-10-31

Environment: Python 3.12.14, NumPy 2.3.5, CPU. Commands were executed from this directory. These are synthetic mechanism results, not BMVC benchmark reproduction.

## Numerical and behavioral checks

```text
[ok 01] softmax remains finite for large logits
[ok 02] softmax rows sum to one
[ok 03] equal logits produce a uniform distribution
[ok 04] sliding mask has [T,T] shape
[ok 05] interior and edge queries retain the requested width
[ok 06] window is centered for an odd width
[ok 07] boundary window shifts instead of shrinking
[ok 08] attention preserves query length and value dimension
[ok 09] attention weights normalize per query
[ok 10] masked pairs receive exactly zero weight
[ok 11] identity-value attention yields nonnegative mixtures
[ok 12] symmetric dilated mixing preserves a linear interior signal
[ok 13] edge replication is deterministic
[ok 14] paper smoothness is zero for constant probabilities
[ok 15] paper smoothness penalizes a sharp probability jump
[ok 16] released-code log penalty is truncated at tau squared
[ok 17] cross-entropy indexes the target class
[ok 18] segment collapse removes only adjacent duplicates
[ok 19] Levenshtein distance handles deletion
[ok 20] perfect segment sequence has edit score 100
[ok 21] perfect segments have F1 100
[ok 22] segment metrics expose errors hidden by high frame accuracy
[ok 23] hierarchical encoder preserves [T,C]
[ok 24] encoder probabilities remain normalized
[ok 25] one auditable attention map is returned per hierarchy level
[ok 26] requested decoder-stage count is honored
[ok 27] all decoder probabilities remain normalized
[ok 28] refinement reduces synthetic over-segmentation by edit score
[ok 29] refinement improves synthetic segmental F1@50
[ok 30] pair counter matches exact shifted-window widths
[ok 31] hierarchical local attention uses fewer pairs than full attention
[ok 32] stable high-confidence treatment enables the illustrative gate
[ok 33] wrong phase disables the illustrative gate
[ok 34] all 34 checks completed
```

## One-sequence trace

```text
features=(200, 12), logits=(200, 4), labels=(200,)
frame classifier     acc= 78.00 edit=  5.00 F1@50=  0.00 segments=80
hierarchical encoder acc= 99.50 edit=100.00 F1@50=100.00 segments= 4
decoder 1            acc= 98.00 edit=100.00 F1@50=100.00 segments= 4
decoder 2            acc= 98.00 edit=100.00 F1@50=100.00 segments= 4
decoder 3            acc= 98.00 edit=100.00 F1@50=100.00 segments= 4
attention pairs: hierarchical=12,600, full=240,000, ratio=0.0525
```

The slight accuracy reduction after decoding illustrates that a refinement stage may trade a few boundary frames for a much cleaner segment sequence. Segment metrics remain perfect in this controlled sample.

## Paired ablation, 20 seeds

Mean ± population standard deviation:

| Configuration | Accuracy | Edit | F1@50 | Predicted segments |
|---|---:|---:|---:|---:|
| Frame classifier | 81.800 ± 3.047 | 6.182 ± 0.779 | 0.638 ± 1.280 | 65.850 ± 9.248 |
| Fixed local, width 8 | 97.725 ± 1.167 | 99.000 ± 4.359 | 99.444 ± 2.422 | 4.050 ± 0.218 |
| Hierarchical, widths 1…32 | 98.400 ± 1.190 | 97.333 ± 8.273 | 98.444 ± 4.874 | 4.150 ± 0.477 |
| Width 32 from layer 1 | 97.650 ± 1.388 | 100.000 ± 0.000 | 100.000 ± 0.000 | 4.000 ± 0.000 |

Decoder depth after the hierarchical encoder:

| Output | Edit | F1@50 |
|---|---:|---:|
| Encoder | 97.333 ± 8.273 | 98.444 ± 4.874 |
| Decoder 1 | 99.000 ± 4.359 | 99.444 ± 2.422 |
| Decoder 2 | 99.000 ± 4.359 | 99.444 ± 2.422 |
| Decoder 3 | 99.000 ± 4.359 | 99.444 ± 2.422 |

This toy study does **not** reproduce the paper's learned hierarchy ablation: the fixed-wide mechanism happens to score best on this simple four-phase generator. That negative result is retained. It shows why architectural claims require the official learned benchmark rather than a hand-designed smoother.

## Integrated application and failure

```text
at boundary      t=101 enabled=False phase=treatment   conf=0.532 entropy=0.847 reason=phase not persistent
stable middle    t=131 enabled=True  phase=treatment   conf=0.803 entropy=0.510 reason=all temporal checks passed
after treatment  t=171 enabled=False phase=closure     conf=0.526 entropy=0.813 reason=wrong phase

coherent-error stress test:
true=treatment, predicted=closure, enabled=True, reason=all temporal checks passed
```

The coherent-error case is intentional. A persistent, confident phase prediction can still be semantically wrong; the example gate is not a medical device or a sufficient safety layer.

The accompanying SVG was also parsed successfully with Python's XML parser.
