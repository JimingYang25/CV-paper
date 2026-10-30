"""Paired synthetic trials for locality, hierarchy, and refinement depth."""

from __future__ import annotations

import numpy as np

from asformer_math import evaluate, hierarchical_attention_refine, iterative_decoder_refine
from synthetic_sequence import make_phase_sequence


def summarize(values: list[float]) -> str:
    arr = np.asarray(values)
    return f"{np.mean(arr):.3f} ± {np.std(arr):.3f}"


def main() -> None:
    configurations = {
        "frame classifier": None,
        "fixed local w=8": (8, 8, 8, 8, 8, 8),
        "hierarchical": (1, 2, 4, 8, 16, 32),
        "wide from layer 1": (32, 32, 32, 32, 32, 32),
    }
    records = {name: {"acc": [], "edit": [], "f1": [], "segments": []} for name in configurations}
    decoder_records = {stage: {"edit": [], "f1": []} for stage in range(4)}

    for seed in range(20):
        features, logits, labels = make_phase_sequence(seed=seed)
        for name, windows in configurations.items():
            if windows is None:
                prediction = logits.argmax(axis=1)
            else:
                probs, _ = hierarchical_attention_refine(logits, features, windows)
                prediction = probs.argmax(axis=1)
            metrics = evaluate(prediction, labels)
            records[name]["acc"].append(metrics.accuracy)
            records[name]["edit"].append(metrics.edit)
            records[name]["f1"].append(metrics.f1_at_50)
            records[name]["segments"].append(metrics.segments)

        encoded, _ = hierarchical_attention_refine(logits, features)
        stages = [encoded] + iterative_decoder_refine(encoded, stages=3)
        for stage, probabilities in enumerate(stages):
            metrics = evaluate(probabilities.argmax(axis=1), labels)
            decoder_records[stage]["edit"].append(metrics.edit)
            decoder_records[stage]["f1"].append(metrics.f1_at_50)

    print("20 paired synthetic trials; mean ± population std")
    print("configuration          accuracy       edit       F1@50     segments")
    for name, row in records.items():
        print(
            f"{name:20s} {summarize(row['acc']):>13s} {summarize(row['edit']):>13s} "
            f"{summarize(row['f1']):>13s} {summarize(row['segments']):>13s}"
        )
    print("\ndecoder depth after hierarchical encoder")
    for stage, row in decoder_records.items():
        print(f"  stage {stage}: edit={summarize(row['edit'])}, F1@50={summarize(row['f1'])}")


if __name__ == "__main__":
    main()
