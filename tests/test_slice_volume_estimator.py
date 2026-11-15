import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from FeederReference import FeederReference
from SliceVolumeEstimator import (
    Circular,
    DensityWeightModel,
    Elliptical,
    ProfileScaled,
    SliceVolumeEstimator,
)


def ellipse_mask(axes, angle=0, center=(260, 210)):
    mask = np.zeros((420, 520), np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    return mask


class SliceVolumeEstimatorTest(unittest.TestCase):
    def test_circular_slices_match_ellipsoid_volume(self):
        a_px, b_px, px_per_cm = 100, 40, 10.0
        expected = 4.0 / 3.0 * math.pi * (a_px / px_per_cm) * (b_px / px_per_cm) ** 2
        result = SliceVolumeEstimator(20, Circular()).estimate(
            ellipse_mask((a_px, b_px)), px_per_cm
        )
        self.assertAlmostEqual(result.volume_cm3, expected, delta=expected * 0.02)
        self.assertEqual(len(result.slice_widths_cm), 20)
        self.assertAlmostEqual(result.length_cm, 2 * a_px / px_per_cm, delta=0.2)

    def test_slice_count_converges_and_weight_is_separate(self):
        mask = ellipse_mask((100, 40))
        volumes = [
            SliceVolumeEstimator(n, Circular()).estimate(mask, 10.0).volume_cm3
            for n in (5, 20, 40, 100)
        ]
        self.assertLess(abs(volumes[-1] - volumes[-2]), abs(volumes[0] - volumes[-1]))
        density = DensityWeightModel(0.90)
        estimator = SliceVolumeEstimator(40, Elliptical(0.8), density)
        result = estimator.estimate(mask, 10.0)
        self.assertAlmostEqual(result.weight_kg_est, result.volume_cm3 * 0.90 / 1000)
        profile_result = SliceVolumeEstimator(20, ProfileScaled([0.5, 1, 0.5])).estimate(
            mask, 10.0
        )
        self.assertGreater(profile_result.volume_cm3, 0)

    def test_alignment_is_invariant_to_initial_rotation(self):
        results = [
            SliceVolumeEstimator(40, Circular()).estimate(ellipse_mask((100, 40), angle), 10.0)
            for angle in (0, 30, 90, 135)
        ]
        volumes = [result.volume_cm3 for result in results]
        self.assertLess((max(volumes) - min(volumes)) / np.mean(volumes), 0.01)
        for result in results:
            ys, xs = np.where(result.aligned_mask > 0)
            self.assertGreater(xs.max() - xs.min(), ys.max() - ys.min())

    def test_camera_tilt_is_removed_before_volume_estimation(self):
        frame = np.zeros((420, 520, 3), np.uint8)
        cv2.ellipse(frame, (260, 210), (150, 140), 20, 0, 360, (0, 128, 255), -1)
        cv2.ellipse(frame, (260, 210), (142, 132), 20, 0, 360, (0, 0, 0), -1)
        reference = FeederReference.from_frames([frame] * 5)
        projected = ellipse_mask((100, 47), 20)
        rectified = reference.rectify(projected, is_mask=True)
        expected = ellipse_mask((100, 50), 0)
        estimator = SliceVolumeEstimator(40, Circular())
        actual_volume = estimator.estimate(rectified, reference.px_per_cm).volume_cm3
        expected_volume = estimator.estimate(expected, reference.px_per_cm).volume_cm3
        self.assertLess(abs(actual_volume / expected_volume - 1), 0.02)

    def test_none_uses_per_column_slices_and_empty_masks_are_supported(self):
        mask = ellipse_mask((20, 10))
        result = SliceVolumeEstimator(None).estimate(mask, 2.0)
        self.assertGreaterEqual(result.n_slices, 30)
        empty = SliceVolumeEstimator().estimate(np.zeros_like(mask), 2.0)
        self.assertEqual(empty.volume_cm3, 0)


if __name__ == "__main__":
    unittest.main()
