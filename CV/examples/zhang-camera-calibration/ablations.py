"""Small synthetic model ablation; not the paper's experimental protocol."""
import json
import numpy as np
from calibration import initialize, refine
from demo import make_views, TRUE_K, SEED


def run():
    xyz, _, uv = make_views()
    K0, poses0, _ = initialize(xyz, uv)
    K0[0, 1] = 0
    # Same corners, pose initialization, optimizer and noise; only radial model changes.
    results = []
    for order in (0, 1, 2):
        fitted = refine(xyz, uv, K0, poses0, radial_terms=order)
        results.append({
            "radial_terms": order,
            "vector_rms_px": float(np.sqrt(np.mean(fitted.per_view_rms**2))),
            "focal_error_percent": (100*np.abs(np.diag(fitted.K)[:2]/np.diag(TRUE_K)[:2]-1)).tolist(),
            "principal_point_error_px": float(np.linalg.norm(fitted.K[:2, 2]-TRUE_K[:2, 2])),
            "distortion": fitted.distortion.tolist(),
        })
    return {"seed": SEED, "synthetic_only": True, "results": results}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
