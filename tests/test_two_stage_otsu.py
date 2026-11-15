import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from Factories import SegmentationMethod, SegmenterFactory
from SegmentationStrategy import ClassicalSegmenter, TwoStageOtsuSegmenter


class TwoStageOtsuTest(unittest.TestCase):
    def setUp(self):
        self.reference = SimpleNamespace(px_per_cm=2.0, major_px=400.0)
        self.segmenter = TwoStageOtsuSegmenter(self.reference)

    def test_emits_bird_candidates_and_rejects_long_structures(self):
        frame = np.full((300, 640, 3), 90, dtype=np.uint8)
        cv2.ellipse(frame, (320, 190), (65, 45), 15, 0, 360, (245, 245, 245), -1)
        cv2.rectangle(frame, (45, 70), (595, 78), (245, 245, 245), -1)
        result = self.segmenter.segment(frame)
        self.assertGreater(result.threshold_stage2, result.threshold_stage1)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(result.rejected_elongated), 1)
        self.assertGreater(result.candidates[0].area_cm2, 30)
        self.assertGreater(result.rejected_elongated[0].elongation_ratio, 4)
        self.assertEqual(result.candidates[0].mask.shape, frame.shape[:2])

    def test_factory_exposes_otsu2_without_changing_classical_baseline(self):
        factory = SegmenterFactory()
        self.assertIsInstance(factory.create(SegmentationMethod.CLASSICAL), ClassicalSegmenter)
        self.assertIsInstance(
            factory.create(SegmentationMethod.OTSU2, feeder_reference=self.reference),
            TwoStageOtsuSegmenter,
        )
        with self.assertRaisesRegex(ValueError, "feeder_reference"):
            factory.create(SegmentationMethod.OTSU2)

    def test_blank_frame_returns_empty_candidates(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        result = self.segmenter.segment(frame)
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.rejected_elongated, [])


if __name__ == "__main__":
    unittest.main()
