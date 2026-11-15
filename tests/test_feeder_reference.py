import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from FeederReference import FeederReference
from defaults import FEEDER_DIAMETER_CM


def feeder_frame(axes=(150, 150), angle=0, occluded=False, stray=False):
    frame = np.zeros((420, 500, 3), dtype=np.uint8)
    cv2.ellipse(frame, (250, 210), axes, angle, 0, 360, (0, 128, 255), -1)
    inner_axes = (axes[0] - 8, axes[1] - 8)
    cv2.ellipse(frame, (250, 210), inner_axes, angle, 0, 360, (0, 0, 0), -1)
    if occluded:
        cv2.rectangle(frame, (0, 280), (500, 325), (100, 100, 100), -1)
    if stray:
        cv2.circle(frame, (35, 40), 13, (0, 128, 255), -1)
        cv2.circle(frame, (463, 370), 9, (0, 128, 255), -1)
    return frame


class FeederReferenceTest(unittest.TestCase):
    def test_ring_diameter_with_occlusion_and_stray_orange(self):
        frame = feeder_frame(occluded=True, stray=True)
        reference = FeederReference.from_frames([frame] * 5)
        self.assertLess(abs(reference.major_px / 300 - 1), 0.01)
        self.assertAlmostEqual(reference.px_per_cm, 300 / FEEDER_DIAMETER_CM, delta=0.1)
        self.assertGreaterEqual(reference.calib_uncertainty_pct, 0)
        debug = reference.debug_image()
        self.assertEqual(debug.shape, frame.shape)
        self.assertEqual(reference.rectify(np.zeros(frame.shape[:2], np.uint8), is_mask=True).dtype,
                         np.uint8)

    def test_ellipse_view_angle_is_rectified_to_a_circle(self):
        frame = feeder_frame(axes=(150, 140), angle=20)
        reference = FeederReference.from_frames([frame] * 5)
        self.assertLess(abs(reference.major_px / 300 - 1), 0.01)
        self.assertGreater(reference.minor_major_ratio, 0.9)
        rectified = reference.rectify(frame)
        self.assertGreaterEqual(rectified.shape[0], frame.shape[0])
        hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)
        orange = cv2.inRange(hsv, (5, 60, 40), (25, 255, 255))
        contours, _ = cv2.findContours(orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        fitted = cv2.fitEllipse(max(contours, key=cv2.contourArea))
        self.assertLess(abs(fitted[1][0] / fitted[1][1] - 1), 0.01)

    def test_requires_five_good_frame_fits_by_default(self):
        with self.assertRaisesRegex(RuntimeError, "at least 5"):
            FeederReference.from_frames([np.zeros((100, 100, 3), np.uint8)] * 4)


if __name__ == "__main__":
    unittest.main()
