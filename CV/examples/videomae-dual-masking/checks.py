"""Numerical, masking, cost, leakage, and application checks."""

from __future__ import annotations

import numpy as np

from application import run_application
from demo import dual_mask_study, leakage_study, leakage_trial
from synthetic_video import make_motion_video
from videomae_math import (
    attention_pair_counts,
    cubeify,
    dual_mask_selection,
    frame_mask,
    masked_mse,
    normalize_cubes,
    random_mask,
    running_cell_keep,
    temporal_copy_probe,
    tube_mask,
    uncubeify,
)


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS {name}{': ' + detail if detail else ''}")


def main() -> None:
    video = make_motion_video(seed=3)
    cubes, grid = cubeify(video)
    check("cube grid", grid == (8, 8, 8))
    check("cube tensor shape", cubes.shape == (1, 512, 384))
    check("cube round trip", np.allclose(uncubeify(cubes, video.shape), video))
    try:
        cubeify(video[:, :, :, :-1])
        rejected = False
    except ValueError:
        rejected = True
    check("non-divisible shape rejected", rejected)

    normalized, mean, variance = normalize_cubes(cubes)
    check("normalization statistics shape", mean.shape == variance.shape == (1, 512, 3, 1))
    normalized_channels = normalized.reshape(1, 512, 3, -1)
    check("normalized cube mean", np.max(np.abs(normalized_channels.mean(-1))) < 1e-10)
    nonconstant = variance[..., 0] > 1e-6
    check("normalized cube variance", np.allclose(
        normalized_channels.var(-1, ddof=1)[nonconstant], 1.0, atol=3e-4))

    tube = tube_mask(grid, 0.90, np.random.default_rng(0))
    random = random_mask(grid, 0.90, np.random.default_rng(0))
    frames = frame_mask(grid, 0.50, np.random.default_rng(0))
    check("tube mask shape", tube.shape == grid)
    check("tube mask quantized ratio", np.isclose(tube.mean(), 58 / 64))
    check("tube shares spatial map over time", np.all(tube == tube[0]))
    check("random mask exact count", random.sum() == round(0.90 * np.prod(grid)))
    check("random mask changes over time", not np.all(random == random[0]))
    check("frame mask hides whole slices", all(np.all(item) or not np.any(item) for item in frames))

    keep0 = running_cell_keep(grid, period=2, offset=0)
    keep1 = running_cell_keep(grid, period=2, offset=1)
    check("running cell keeps half", keep0.sum() == np.prod(grid) // 2)
    check("running cell offsets partition grid", np.all(keep0 ^ keep1) and not np.any(keep0 & keep1))
    decoder_input, supervised = dual_mask_selection(tube, keep0)
    check("dual supervision was encoder-hidden", np.all(tube[supervised]))
    check("decoder retains encoder-visible tokens", np.all(decoder_input[~tube]))
    check("decoder adds selected mask tokens", np.array_equal(supervised, tube & keep0))

    target = np.array([[0.0], [2.0], [4.0]])
    prediction = np.array([[0.0], [1.0], [2.0]])
    check("zero masked MSE", masked_mse(target, target, [False, True, True]) == 0.0)
    check("known masked MSE", masked_mse(target, prediction, [False, True, True]) == 2.5)
    probe = temporal_copy_probe(normalized[0], grid, random)
    check("copy probe shape", probe.shape == normalized[0].shape)
    check("copy probe preserves visible tokens", np.allclose(probe[~random.reshape(-1)],
                                                              normalized[0][~random.reshape(-1)]))

    trial = leakage_trial(seed=7)
    check("tube removes more temporal shortcuts", trial["tube_minus_random_mse"] > 0.0, f"{trial}")
    study = leakage_study()
    check("tube harder in every controlled seed", study["tube_is_harder_count"] == study["seeds"], f"{study}")
    check("positive median leakage gap", study["median_tube_minus_random_mse"] > 0.01, f"{study}")

    dual = dual_mask_study()
    check("cyclic decoder covers each masked token once", dual["masked_tokens_covered_once"] == 1, f"{dual}")
    check("cycled subset loss equals full loss", dual["absolute_loss_difference"] < 1e-12, f"{dual}")
    check("encoder masking reduces attention pairs", dual["encoder_masked_pairs"] < dual["encoder_full_pairs"])
    check("dual decoder reduces attention pairs", dual["decoder_dual_pairs"] < dual["decoder_full_pairs"])
    counts = attention_pair_counts(tube, decoder_input)
    check("attention count token accounting", counts["tokens_total"] == 512
          and counts["tokens_encoder_visible"] == 48)

    app = run_application()
    check("surgical application frame count", app["frames"] == 120)
    check("clip descriptor is temporal", app["clip_feature_dim"] > app["frame_feature_dim"])
    check("clip features beat frame appearance", app["clip_accuracy"] > app["frame_only_accuracy"], f"{app}")
    check("ordered decoding improves clip accuracy", app["ordered_clip_accuracy"] > app["clip_accuracy"], f"{app}")
    check("ordered decoding is monotonic", bool(app["ordered_monotonic"]))
    check("ordered boundaries beat frame baseline",
          app["ordered_clip_boundary_f1"] > app["frame_only_boundary_f1"], f"{app}")
    check("quality gate accepts controlled ordered stream", bool(app["deploy_phase_stream"]), f"{app}")

    print("\n35/35 checks passed")


if __name__ == "__main__":
    main()
