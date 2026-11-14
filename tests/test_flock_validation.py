import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from compare_video import write_comparison


def write_csv(path, fields, rows):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FlockValidationTest(unittest.TestCase):
    def test_auto_discovery_offset_and_median_event_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = root / "gt.csv"
            write_csv(gt, ["video_name", "frame_idx", "x", "y", "w", "h",
                           "weight_kg", "event_id"], [
                dict(video_name="bird.mp4", frame_idx=0, x=10, y=10, w=20, h=20,
                     weight_kg=1, event_id="e1"),
                dict(video_name="bird.mp4", frame_idx=1, x=10, y=10, w=20, h=20,
                     weight_kg=3, event_id="e1"),
                dict(video_name="missing.mp4", frame_idx=0, x=0, y=0, w=5, h=5,
                     weight_kg=1, event_id="e2"),
            ])
            write_csv(root / "bird_results.csv", ["frame_idx", "crop_region", "volume", "score"], [
                dict(frame_idx=1, crop_region="(10, 10, 20, 20)", volume=1, score=1),
                dict(frame_idx=2, crop_region="(10, 10, 20, 20)", volume=3, score=1),
            ])
            out = root / "evaluation"
            result = subprocess.run(
                [sys.executable, str(ROOT / "src" / "evaluate_against_gt.py"), str(gt),
                 "--results-dir", str(root), "--gt-frame-offset", "1", "--out", str(out)],
                check=True, capture_output=True, text=True,
            )
            self.assertIn("Matched: 2", result.stdout)
            self.assertIn("[warn] no results file for missing.mp4", result.stdout)
            with (out / "per_event.csv").open(newline="") as fh:
                event = next(csv.DictReader(fh))
            self.assertEqual(float(event["actual_weight_kg"]), 2)
            self.assertEqual(float(event["median_estimate_kg"]), 2)

    def test_zero_weight_and_zero_estimate_do_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = root / "gt.csv"
            write_csv(gt, ["video_name", "frame_idx", "x", "y", "w", "h",
                           "weight_kg", "event_id"], [
                dict(video_name="bird.mp4", frame_idx=1, x=1, y=1, w=10, h=10,
                     weight_kg=0, event_id="e"),
            ])
            write_csv(root / "bird_results.csv", ["frame_idx", "crop_region", "volume", "score"], [
                dict(frame_idx=1, crop_region="(1, 1, 10, 10)", volume=0, score=1),
            ])
            result = subprocess.run(
                [sys.executable, str(ROOT / "src" / "evaluate_against_gt.py"), str(gt),
                 "--results-dir", str(root), "--out", str(root / "out")],
                check=True, capture_output=True, text=True,
            )
            self.assertIn("Density fit unavailable", result.stdout)

    def test_comparison_draws_boxes_on_offset_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "bird.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (160, 120))
            self.assertTrue(writer.isOpened())
            for _ in range(2):
                writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
            writer.release()
            gt = root / "gt.csv"
            write_csv(gt, ["video_name", "frame_idx", "x", "y", "w", "h",
                           "weight_kg", "event_id"], [
                dict(video_name="bird.mp4", frame_idx=0, x=20, y=60, w=20, h=20,
                     weight_kg=2, event_id="e"),
            ])
            estimates = root / "results.csv"
            write_csv(estimates, ["frame_idx", "crop_region", "volume", "score"], [
                dict(frame_idx=1, crop_region="(90, 60, 20, 20)", volume=2, score=1),
            ])
            output = root / "comparison.mp4"
            write_comparison(video, gt, estimates, output, frame_offset=1)
            cap = cv2.VideoCapture(str(output))
            ok, first = cap.read()
            ok2, second = cap.read()
            cap.release()
            self.assertTrue(ok and ok2)
            green = first[58:83, 18:43]
            red = first[58:83, 88:113]
            self.assertGreater(int(green[:, :, 1].max()), 150)
            self.assertGreater(int(red[:, :, 2].max()), 150)
            self.assertLess(int(second[58:83, 18:43].max()), 50)


if __name__ == "__main__":
    unittest.main()
