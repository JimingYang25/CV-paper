"""Run one transparent ASFormer mechanism study on a synthetic video."""

from __future__ import annotations

from asformer_math import (
    attention_pair_count,
    evaluate,
    hierarchical_attention_refine,
    iterative_decoder_refine,
)
from synthetic_sequence import make_phase_sequence


def main() -> None:
    features, logits, labels = make_phase_sequence(seed=7)
    windows = (1, 2, 4, 8, 16, 32)
    encoder, attentions = hierarchical_attention_refine(logits, features, windows)
    decoder_stages = iterative_decoder_refine(encoder, stages=3)

    predictions = [logits.argmax(axis=1), encoder.argmax(axis=1)]
    predictions.extend(stage.argmax(axis=1) for stage in decoder_stages)
    names = ["frame classifier", "hierarchical encoder", "decoder 1", "decoder 2", "decoder 3"]
    print("shape flow:")
    print(f"  features={features.shape}, logits={logits.shape}, labels={labels.shape}")
    print(f"  attention maps={[a.shape for a in attentions]}")
    print("\nmetrics (synthetic mechanism study, not paper reproduction):")
    for name, prediction in zip(names, predictions):
        m = evaluate(prediction, labels)
        print(
            f"  {name:20s} acc={m.accuracy:6.2f} edit={m.edit:6.2f} "
            f"F1@50={m.f1_at_50:6.2f} segments={m.segments:2d}"
        )

    local_pairs = attention_pair_count(labels.size, windows)
    full_pairs = len(windows) * labels.size * labels.size
    print(f"\nattention pairs: hierarchical={local_pairs:,}, full={full_pairs:,}, ratio={local_pairs/full_pairs:.4f}")


if __name__ == "__main__":
    main()
