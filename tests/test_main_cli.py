import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from OutputWriters import FEATURE_CSV_FIELDS, FeatureCSVWriter
from main import main, process_video, recording_date_from_name, video_paths


class MainCliTest(unittest.TestCase):
    def test_feature_csv_schema_serializes_slice_widths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.csv"
            with FeatureCSVWriter(str(path)) as writer:
                writer.write({"video_name": "sample.mp4", "slice_widths_cm": [1.0, 2.5]})
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(tuple(rows[0]), FEATURE_CSV_FIELDS)
            self.assertEqual(json.loads(rows[0]["slice_widths_cm"]), [1.0, 2.5])

    def test_filename_date_and_video_directory_discovery(self):
        self.assertEqual(recording_date_from_name("house_2026-11-15_clip.mp4"), "2026-11-15")
        self.assertEqual(recording_date_from_name("bird_20261115.mp4"), "2026-11-15")
        self.assertEqual(recording_date_from_name("undated.mp4"), "")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "one.mp4").touch()
            (root / "notes.txt").touch()
            (root / "nested").mkdir()
            (root / "nested" / "two.avi").touch()
            self.assertEqual([path.name for path in video_paths(root)], ["two.avi", "one.mp4"])

    def test_pipeline_writes_rows_for_synthetic_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "flock_2026-11-15.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (320, 240))
            self.assertTrue(writer.isOpened())
            frame = np.full((240, 320, 3), 85, np.uint8)
            cv2.circle(frame, (160, 120), 45, (0, 128, 255), -1)
            cv2.ellipse(frame, (225, 120), (30, 20), 0, 0, 360, (245, 245, 245), -1)
            for _ in range(5):
                writer.write(frame)
            writer.release()

            output = process_video(video, root / "results", debug_every=0)
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertEqual(rows[0]["recording_date"], "2026-11-15")
            self.assertIn(rows[0]["measurable"], ("True", "False"))
            self.assertGreater(float(rows[0]["volume_cm3"]), 0)
            self.assertEqual(rows[0]["segmenter"], "otsu2")

    def test_cli_accepts_required_options(self):
        with self.assertRaises(SystemExit) as exit_info:
            main(["--help"])
        self.assertEqual(exit_info.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
