import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from BirdSelection import FeederRelativeBirdSelector
from SegmentationStrategy import BirdCandidate, ElongatedComponent


class IdentityReference:
    px_per_cm = 2.0
    major_px = 200.0
    center_px = (200.0, 200.0)
    rectified_center_px = center_px

    @staticmethod
    def rectify(image, is_mask=None):
        return image.copy()


def candidate(center_x, radius_x=20, radius_y=15):
    mask = np.zeros((400, 400), np.uint8)
    cv2.ellipse(mask, (center_x, 200), (radius_x, radius_y), 0, 0, 360, 255, -1)
    x, y, width, height = cv2.boundingRect(mask)
    area = cv2.countNonZero(mask) / IdentityReference.px_per_cm ** 2
    return BirdCandidate(mask, (x, y, width, height), area, 1.0, (center_x, 200))


class BirdSelectionTest(unittest.TestCase):
    def test_selects_nearest_nonborder_candidate_and_tracks_centimeters(self):
        selector = FeederRelativeBirdSelector(IdentityReference(), max_feeder_contact_ratio=1.0,
                                               max_track_jump_cm=20)
        frame = np.full((400, 400, 3), 150, np.uint8)
        first = selector.select([candidate(245), candidate(330)], frame).selected
        self.assertEqual(first.centroid_px, (245.0, 200.0))
        self.assertTrue(first.measurable)
        first_id = first.track_id
        next_frame = selector.select([candidate(255)], frame).selected
        self.assertEqual(next_frame.track_id, first_id)
        new_visit = selector.select([candidate(315)], frame).selected
        self.assertNotEqual(new_visit.track_id, first_id)

    def test_quality_rejects_feeder_contact_border_and_occluder_overlap(self):
        selector = FeederRelativeBirdSelector(IdentityReference())
        frame = np.full((400, 400, 3), 150, np.uint8)
        touching = candidate(240)
        result = selector.select([touching], frame)
        self.assertFalse(result.selected.measurable)
        self.assertGreater(touching.quality["feeder_contact"], 0.1)
        self.assertIn("feeder_contact", touching.quality["rejection_reasons"])

        edge = candidate(390)
        result = selector.select([edge], frame)
        self.assertIsNone(result.selected)
        self.assertFalse(edge.measurable)
        self.assertIn("frame_border", edge.quality["rejection_reasons"])

        occluded = candidate(300)
        structure_mask = np.zeros_like(frame[:, :, 0])
        cv2.line(structure_mask, (300, 180), (300, 220), 255, 3)
        structure = ElongatedComponent(structure_mask, (300, 180, 1, 41), 10, 20, 100)
        result = selector.select([occluded], frame, [structure])
        self.assertFalse(result.selected.measurable)
        self.assertGreater(result.selected.quality["occluder_contact"], 0)

    def test_temporal_area_drop_is_recorded_and_rejected(self):
        selector = FeederRelativeBirdSelector(IdentityReference(), max_feeder_contact_ratio=1.0)
        frame = np.full((400, 400, 3), 150, np.uint8)
        selector.select([candidate(300, 25, 20)], frame)
        small = candidate(302, 15, 12)
        selected = selector.select([small], frame).selected
        self.assertFalse(selected.measurable)
        self.assertGreater(selected.quality["temporal_area_deviation"], 0.15)
        self.assertIn("temporal_area_change", selected.quality["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
