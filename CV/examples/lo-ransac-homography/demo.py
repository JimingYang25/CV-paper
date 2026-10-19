"""Deterministic synthetic matching, robust fit and paired ablations."""
import json
import numpy as np
from robust_homography import (transfer, fit_homography, estimate,
                               all_inlier_probability, required_trials)

SEED = 20261020
TRUE_H = np.array([[1.02, 0.08, 35.], [-0.04, 0.98, 22.], [8e-5, -6e-5, 1.]])


def make_data(seed=SEED, total=250, genuine=100, noise=0.7):
    rng = np.random.default_rng(seed)
    source_clean = rng.uniform([30, 30], [610, 450], (total, 2))
    destination_clean = transfer(TRUE_H, source_clean)
    source = source_clean + rng.normal(0, noise, (total, 2))
    destination = destination_clean + rng.normal(0, noise, (total, 2))
    truth = np.arange(total) < genuine
    destination[~truth] = rng.uniform([0, 0], [720, 540], (total-genuine, 2))
    order = rng.permutation(total)
    grid = np.stack(np.meshgrid(np.linspace(40, 600, 15), np.linspace(40, 440, 11)), axis=-1).reshape(-1, 2)
    return source[order], destination[order], truth[order], grid


def grid_rms(H, grid):
    errors = transfer(H, grid)-transfer(TRUE_H, grid)
    return float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))


def evaluate(result, truth, grid):
    true_positive = np.count_nonzero(result.inliers & truth)
    count = int(result.inliers.sum())
    return {"grid_vector_rms_px": grid_rms(result.H, grid), "support": count,
            "precision": float(true_positive/max(1, count)),
            "recall": float(true_positive/truth.sum()), "draws": result.draws,
            "valid_hypotheses": result.valid_hypotheses, "rejected_samples": result.rejected_samples,
            "local_calls": result.local_calls, "local_fits": result.local_fits,
            "stopped_at_cap": result.stopped_at_cap}


def run():
    source, destination, truth, grid = make_data()
    result = estimate(source, destination, seed=SEED)
    all_fit = fit_homography(source, destination)
    q_exact = all_inlier_probability(6, 20)
    q_approx = all_inlier_probability(6, 20, exact=False)
    approx_draws = required_trials(q_approx, 0.99)
    report = {"seed": SEED, "source_and_destination_noise_std_px": 0.7,
              "total_matches": len(source), "genuine_matches": int(truth.sum()),
              "H_true": TRUE_H.tolist(), "H_estimated": result.H.tolist(),
              "all_match_DLT_grid_rms_px": grid_rms(all_fit, grid),
              "robust": evaluate(result, truth, grid),
              "finite_population_example": {"N": 20, "I": 6, "sample_size": 4,
                  "q_exact": q_exact, "q_approx": q_approx,
                  "draws_exact_for_0_99": required_trials(q_exact, 0.99),
                  "draws_approx_for_0_99": approx_draws,
                  "actual_sampling_success_at_approx_budget": -np.expm1(approx_draws*np.log1p(-q_exact))}}
    return report, (source, destination, truth, grid, result, all_fit)


def ablate(repeats=20, draws=300):
    # Identical observations, 20 paired outer RNG seeds, fixed outer draw budget.
    source, destination, truth, grid = make_data()
    configurations = [
        ("minimal only", "none", False),
        ("final refit only", "none", True),
        ("simple local + final", "simple", True),
        ("inner local + final", "inner", True),
    ]
    output = []
    for name, mode, final in configurations:
        records = [evaluate(estimate(source, destination, seed=SEED+i, max_trials=draws,
                                     adaptive=False, local_mode=mode, final_polish=final), truth, grid)
                   for i in range(repeats)]
        errors = np.array([r["grid_vector_rms_px"] for r in records])
        output.append({"variant": name, "runs": repeats, "outer_draws_per_run": draws,
                       "median_grid_rms_px": float(np.median(errors)),
                       "p90_grid_rms_px": float(np.percentile(errors, 90)),
                       "worst_grid_rms_px": float(errors.max()),
                       "failures_over_2px": int(np.count_nonzero(errors > 2)),
                       "mean_support": float(np.mean([r["support"] for r in records])),
                       "mean_local_calls": float(np.mean([r["local_calls"] for r in records])),
                       "mean_local_fits": float(np.mean([r["local_fits"] for r in records]))})
    return output


def failure_case():
    """Two coherent planes: maximum consensus is not semantic target selection."""
    rng = np.random.default_rng(SEED+100)
    clean = rng.uniform([30, 30], [610, 450], (250, 2))
    intended = np.arange(250) < 100
    other_H = np.array([[0.94, -0.07, 70.], [0.03, 1.03, -5.],
                        [-4e-5, 8e-5, 1.]])
    source = clean+rng.normal(0, 0.7, clean.shape)
    destination = np.where(intended[:, None], transfer(TRUE_H, clean), transfer(other_H, clean))
    destination += rng.normal(0, 0.7, clean.shape)
    result = estimate(source, destination, seed=SEED)
    grid = make_data()[3]  # common clean held-out spatial grid, not the noisy fit points
    return {"synthetic_only": True, "intended_plane_matches": 100,
            "distractor_plane_matches": 150, "selected_support": int(result.inliers.sum()),
            "intended_matches_in_selected_support": int(np.count_nonzero(result.inliers & intended)),
            "grid_error_against_intended_plane_px": grid_rms(result.H, grid),
            "interpretation": "The dominant geometric consensus need not be the intended object."}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--failure-case", action="store_true")
    args = parser.parse_args()
    print(json.dumps(failure_case() if args.failure_case else ablate() if args.ablations else run()[0], indent=2))
