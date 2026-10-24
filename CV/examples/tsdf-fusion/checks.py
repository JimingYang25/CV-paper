"""Deterministic numerical checks for the educational TSDF implementation."""

from __future__ import annotations

import numpy as np

from demo import run_variant
from synthetic_scene import camera_pose
from tsdf_fusion import PinholeCamera, TSDFVolume, VolumeConfig


def tiny_camera() -> PinholeCamera:
    return PinholeCamera(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=1, height=1)


def tiny_config(*, max_weight: float = 8.0) -> VolumeConfig:
    return VolumeConfig(
        origin_world=(-0.05, -0.05, 0.50),
        dimensions_xyz=(2, 2, 4),
        voxel_size=0.10,
        truncation=0.10,
        max_weight=max_weight,
    )


def check_camera_and_config_guards() -> None:
    for constructor in (
        lambda: PinholeCamera(0, 1, 0, 0, 1, 1),
        lambda: VolumeConfig((0, 0, 0), (1, 2, 2), 0.1, 0.1),
        lambda: VolumeConfig((0, 0, 0), (2, 2, 2), 0.1, 0.05),
    ):
        try:
            constructor()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid configuration was accepted")


def check_voxel_centres_and_storage_order() -> None:
    volume = TSDFVolume(tiny_config())
    assert volume.centres_world.shape == (4, 2, 2, 3)
    np.testing.assert_allclose(volume.centres_world[0, 0, 0], [0.0, 0.0, 0.55])
    np.testing.assert_allclose(volume.centres_world[-1, -1, -1], [0.1, 0.1, 0.85])


def check_projective_signed_distance_and_truncation() -> None:
    volume = TSDFVolume(tiny_config())
    stats = volume.integrate(np.array([[0.60]]), tiny_camera(), np.eye(4))
    assert stats.updated_voxels > 0
    np.testing.assert_allclose(volume.tsdf[0, 0, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(volume.tsdf[1, 0, 0], -0.5, atol=1e-6)
    assert volume.weight[3, 0, 0] == 0.0  # 0.25 m behind, outside mu=0.10 m


def check_weighted_running_average() -> None:
    volume = TSDFVolume(tiny_config())
    volume.integrate(np.array([[0.60]]), tiny_camera(), np.eye(4), observation_weight=2.0)
    volume.integrate(np.array([[0.50]]), tiny_camera(), np.eye(4), observation_weight=1.0)
    expected = (2.0 * 0.5 + 1.0 * -0.5) / 3.0
    np.testing.assert_allclose(volume.tsdf[0, 0, 0], expected, atol=1e-6)
    np.testing.assert_allclose(volume.weight[0, 0, 0], 3.0)


def check_weight_saturation() -> None:
    volume = TSDFVolume(tiny_config(max_weight=2.0))
    for _ in range(5):
        volume.integrate(np.array([[0.60]]), tiny_camera(), np.eye(4))
    assert float(np.max(volume.weight)) <= 2.0


def check_invalid_depth_is_ignored() -> None:
    for value in (np.nan, 0.0, 20.0):
        volume = TSDFVolume(tiny_config())
        stats = volume.integrate(
            np.array([[value]]), tiny_camera(), np.eye(4), depth_min=0.1, depth_max=2.0
        )
        assert stats.updated_voxels == 0
        assert not np.any(volume.weight)


def check_confidence_controls_precision() -> None:
    volume = TSDFVolume(tiny_config())
    volume.integrate(
        np.array([[0.60]]), tiny_camera(), np.eye(4), confidence=np.array([[0.25]])
    )
    np.testing.assert_allclose(volume.weight[0, 0, 0], 0.25)
    rejected = TSDFVolume(tiny_config())
    rejected.integrate(
        np.array([[0.60]]), tiny_camera(), np.eye(4), confidence=np.array([[0.0]])
    )
    assert not np.any(rejected.weight)


def check_world_from_camera_pose_direction() -> None:
    volume = TSDFVolume(tiny_config())
    pose = camera_pose(tz=0.10)
    volume.integrate(np.array([[0.50]]), tiny_camera(), pose)
    # World z=0.55 becomes camera z=0.45; depth 0.50 leaves +0.05 m.
    np.testing.assert_allclose(volume.tsdf[0, 0, 0], 0.5, atol=1e-6)


def check_pose_validation() -> None:
    bad = np.eye(4)
    bad[0, 0] = 2.0
    try:
        TSDFVolume(tiny_config()).integrate(np.array([[0.6]]), tiny_camera(), bad)
    except ValueError:
        return
    raise AssertionError("non-rigid pose was accepted")


def check_subvoxel_zero_crossing() -> None:
    volume = TSDFVolume(tiny_config())
    volume.tsdf[:, 0, 0] = [1.0, 0.5, -0.5, -1.0]
    volume.weight[:, 0, 0] = 1.0
    surface = volume.extract_zero_crossings_z()
    np.testing.assert_allclose(surface.z_world[0, 0], 0.70, atol=1e-7)


def check_intrinsic_resize_preserves_ray() -> None:
    camera = PinholeCamera(100.0, 120.0, 7.5, 5.5, 16, 12)
    scaled = camera.scaled(0.5, 0.5)
    u, v = 10.0, 8.0
    u2, v2 = (u + 0.5) * 0.5 - 0.5, (v + 0.5) * 0.5 - 0.5
    ray = [(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy]
    ray2 = [(u2 - scaled.cx) / scaled.fx, (v2 - scaled.cy) / scaled.fy]
    np.testing.assert_allclose(ray, ray2, atol=1e-12)


def check_memory_accounting() -> None:
    volume = TSDFVolume(tiny_config())
    assert volume.memory_bytes == volume.tsdf.nbytes + volume.weight.nbytes
    assert volume.memory_bytes == 2 * 2 * 4 * 2 * 4


def check_deterministic_fusion() -> None:
    a, metrics_a = run_variant(seed=77, frames=8, confidence_weighted=True)
    b, metrics_b = run_variant(seed=77, frames=8, confidence_weighted=True)
    np.testing.assert_array_equal(a.tsdf, b.tsdf)
    np.testing.assert_array_equal(a.weight, b.weight)
    assert metrics_a == metrics_b


def check_multiframe_fusion_improves_single_frame() -> None:
    _, single = run_variant(seed=20261025, frames=1, confidence_weighted=False)
    _, fused = run_variant(seed=20261025, frames=8, confidence_weighted=False)
    assert fused["mae_m"] < single["mae_m"]


def check_confidence_weighting_rejects_tagged_outliers() -> None:
    _, unit = run_variant(seed=20261025, frames=8, confidence_weighted=False)
    _, weighted = run_variant(seed=20261025, frames=8, confidence_weighted=True)
    assert weighted["mae_m"] < unit["mae_m"]


def check_pose_bias_degrades_surface() -> None:
    _, correct = run_variant(seed=20261025, frames=8, confidence_weighted=True)
    _, biased = run_variant(
        seed=20261025, frames=8, confidence_weighted=True, pose_z_bias_m=0.04
    )
    assert biased["mae_m"] > correct["mae_m"] + 0.02


CHECKS = [
    check_camera_and_config_guards,
    check_voxel_centres_and_storage_order,
    check_projective_signed_distance_and_truncation,
    check_weighted_running_average,
    check_weight_saturation,
    check_invalid_depth_is_ignored,
    check_confidence_controls_precision,
    check_world_from_camera_pose_direction,
    check_pose_validation,
    check_subvoxel_zero_crossing,
    check_intrinsic_resize_preserves_ray,
    check_memory_accounting,
    check_deterministic_fusion,
    check_multiframe_fusion_improves_single_frame,
    check_confidence_weighting_rejects_tagged_outliers,
    check_pose_bias_degrades_surface,
]


def main() -> None:
    for check in CHECKS:
        check()
        print(f"PASS {check.__name__}")
    print(f"PASS all {len(CHECKS)} checks")


if __name__ == "__main__":
    main()
