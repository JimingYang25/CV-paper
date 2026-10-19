"""Synthetic planar sensor-background registration, not contact/force inference.

Toy reference image -> warped current image -> correspondence estimation ->
robust H -> inverse resampling into reference coordinates.
The tentative feature matches are synthetic; no image detector was executed.
"""
import json
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import ConvexHull
from robust_homography import transfer, estimate
from demo import make_data, TRUE_H, SEED


def warp_image(image, H_source_to_target, output_shape):
    """[Hs,Ws] or [Hs,Ws,C] -> [Ht,Wt] or [Ht,Wt,C], float32 bilinear.

    Integer array indices are continuous pixel centers: (u,v)=(column,row).
    Output pixels sample the input using the INVERSE source-to-target transform.
    """
    if image.ndim not in (2, 3):
        raise ValueError("Expected grayscale [H,W] or multichannel [H,W,C] image")
    height, width = output_shape
    v, u = np.indices((height, width), dtype=np.float64)
    target = np.column_stack((u.ravel(), v.ravel()))
    source = transfer(np.linalg.inv(H_source_to_target), target)
    source = np.nan_to_num(source, nan=-1e6, posinf=-1e6, neginf=-1e6)
    coordinates = np.stack((source[:, 1], source[:, 0]))
    if image.ndim == 2:
        result = map_coordinates(image.astype(np.float32), coordinates, order=1,
                                 mode="constant", cval=0, prefilter=False)
        return result.reshape(height, width)
    return np.stack([map_coordinates(image[:, :, c].astype(np.float32), coordinates,
                                    order=1, mode="constant", cval=0, prefilter=False)
                     .reshape(height, width) for c in range(image.shape[2])], axis=-1)


def reference_image(height=480, width=640):
    v, u = np.indices((height, width), dtype=np.float32)
    checker = ((u//40 + v//40) % 2).astype(np.float32)
    circle = ((u-330)**2+(v-250)**2 < 75**2).astype(np.float32)
    image = 0.15+0.35*checker+0.35*circle+0.1*u/width
    return np.clip(image, 0, 1).astype(np.float32)


def run():
    source, destination, truth, grid = make_data()
    robust = estimate(source, destination, seed=SEED)
    # These are heuristic application gates, NOT calibrated failure probabilities.
    coverage = float(ConvexHull(source[robust.inliers]).volume/(640*480))
    if robust.inliers.sum() < 20 or coverage < 0.25:
        raise RuntimeError("Insufficient model support/coverage: keep previous state or reject frame")
    reference = reference_image()
    current = warp_image(reference, TRUE_H, (540, 720))
    rectified = warp_image(current, np.linalg.inv(robust.H), reference.shape)
    # A support mask excludes undefined pixels; not a photometric confidence score.
    current_mask = warp_image(np.ones_like(reference), TRUE_H, current.shape)
    rectified_mask = warp_image(current_mask, np.linalg.inv(robust.H), reference.shape)
    valid = rectified_mask > 0.999
    # Exclude the reference boundary to avoid partial interpolation neighborhoods.
    valid[:4] = valid[-4:] = False
    valid[:, :4] = valid[:, -4:] = False
    report = {"synthetic_only": True, "source_to_target_direction": "reference -> current",
              "support": int(robust.inliers.sum()), "source_hull_fraction": coverage,
              "rectified_valid_fraction": float(valid.mean()),
              "valid_region_intensity_MAE_0_to_1": float(np.abs(rectified-reference)[valid].mean()),
              "note": "Two bilinear resamplings; not feature-detection, tactile or hardware validation"}
    return report, (reference, current, rectified, valid)


if __name__ == "__main__":
    print(json.dumps(run()[0], indent=2))
