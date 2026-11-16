import csv
import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from AreaCalculatorStrategy import CalibratedAreaCalculator, mask_slice_profile
from FeederCalibration import calibrate_pixel_to_cm
from ObjectFilterStrategy import ObjectInfo, SimpleObjectFilter
from scale_region import scale_mask_for_frame
from tune_validation import collect_matches, evaluate_grid


def obj(mask, box=(0, 0, 1, 1), score=1):
    return ObjectInfo(mask, score, score, box)


class ScaleValidationTest(unittest.TestCase):
    def test_scale_region_selection_precedes_top_k_and_does_not_clip_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "template.png"
            cv2.imwrite(str(template), np.full((30, 30), 255, np.uint8))
            filt = SimpleObjectFilter(["Broiler"], str(template), 20, top_k=1,
                                      scale_polygon=[[0, 0], [60, 0], [60, 99], [0, 99]])
            mask = np.zeros((100, 160), np.uint8)
            cv2.rectangle(mask, (20, 20), (49, 49), 255, -1)
            cv2.rectangle(mask, (100, 20), (129, 49), 255, -1)

            def score(blob, contour, dist, area):
                box = cv2.boundingRect(contour)
                return obj(blob.copy(), box, 1 if box[0] < 60 else 100)

            with patch.object(filt, "score_blob", side_effect=score):
                selected, _ = filt.should_process(mask)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].crop_region[0], 20)
            self.assertEqual(cv2.countNonZero(selected[0].mask), 900)
            # 20% of this object lies outside the platform; it remains intact.
            filt.scale_polygon = [[0, 0], [43, 0], [43, 99], [0, 99]]
            with patch.object(filt, "score_blob", side_effect=score):
                selected, _ = filt.should_process(mask)
            self.assertEqual(len(selected), 1)
            self.assertEqual(cv2.countNonZero(selected[0].mask), 900)
            filt.scale_polygon[1][0] = filt.scale_polygon[2][0] = 42
            with patch.object(filt, "score_blob", side_effect=score):
                self.assertEqual(filt.should_process(mask)[0], [])

    def test_invalid_scale_region_is_rejected(self):
        for polygon in ([[[0, 0], [1, 1], [2, 2]]],
                        [[0, 0], [10, 0], [0, 3]],
                        [[0.5, 0], [1, 1], [0, 3]]):
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                scale_mask_for_frame(polygon, (10, 10))

    def test_alignment_preserves_complete_birds_near_frame_edges(self):
        filt = SimpleObjectFilter(["Broiler"], str(ROOT / "src" / "broiler_template.png"), 500)
        areas, weights = [], []
        for center_x in (110, 320, 530):
            mask = np.zeros((240, 640), np.uint8)
            cv2.ellipse(mask, (center_x, 120), (95, 50), 45, 0, 360, 255, -1)
            selected, _ = filt.should_process(mask)
            self.assertEqual(len(selected), 1)
            bird = selected[0]
            source_area = cv2.countNonZero(bird.source_mask)
            aligned_area = cv2.countNonZero(bird.mask)
            self.assertAlmostEqual(aligned_area / source_area, 1, delta=0.03)
            areas.append(aligned_area)
            weights.append(CalibratedAreaCalculator(0.1).volume_from_mask(bird).volume)
        self.assertEqual(areas, [areas[0]] * len(areas))
        self.assertEqual(weights, [weights[0]] * len(weights))

    def test_pca_alignment_preserves_area_for_rotated_ellipses(self):
        filt = SimpleObjectFilter(["Broiler"], str(ROOT / "src" / "broiler_template.png"), 500)
        for angle in (0, 30, 90, 135):
            with self.subTest(angle=angle):
                mask = np.zeros((500, 500), np.uint8)
                cv2.ellipse(mask, (42, 250), (100, 45), angle, 0, 360, 255, -1)
                source_area = cv2.countNonZero(mask)
                aligned = filt.align_mask_by_affine(mask)
                aligned_area = cv2.countNonZero(aligned)
                self.assertLessEqual(abs(aligned_area / source_area - 1), 0.01)
                self.assertEqual(set(np.unique(aligned)), {0, 255})
                aligned_angle = filt.object_angle(aligned)
                self.assertLess(min(abs(aligned_angle), abs(abs(aligned_angle) - 180)), 1.0)

    def test_object_info_volume_alias_and_explicit_units(self):
        bird = obj(np.ones((4, 4), np.uint8))
        CalibratedAreaCalculator(0.1).volume_from_mask(bird)
        self.assertGreater(bird.volume_cm3, 0)
        self.assertGreater(bird.weight_kg, 0)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(bird.volume, bird.weight_kg)
        with self.assertWarns(DeprecationWarning):
            bird.volume = 2.5
        self.assertEqual(bird.weight_kg, 2.5)

    def test_edge_clipped_candidate_is_not_weighed(self):
        filt = SimpleObjectFilter(["Broiler"], str(ROOT / "src" / "broiler_template.png"), 500)
        mask = np.zeros((240, 640), np.uint8)
        cv2.ellipse(mask, (600, 120), (95, 50), 0, 0, 360, 255, -1)
        self.assertEqual(filt.should_process(mask)[0], [])
        self.assertEqual(filt.last_rejections["frame_edge"], 1)
        filt.should_process(np.zeros_like(mask))
        self.assertEqual(filt.last_rejections["frame_edge"], 0)

    def test_frame_edge_can_be_allowed_without_disabling_size_filter(self):
        filt = SimpleObjectFilter(["Broiler"], str(ROOT / "src" / "broiler_template.png"),
                                  500, reject_frame_edge=False)
        mask = np.zeros((240, 640), np.uint8)
        cv2.ellipse(mask, (600, 120), (95, 50), 0, 0, 360, 255, -1)
        cv2.rectangle(mask, (20, 20), (30, 30), 255, -1)
        selected, _ = filt.should_process(mask)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].crop_region[0] + selected[0].crop_region[2], 640)
        self.assertEqual(filt.last_rejections["frame_edge"], 0)
        self.assertEqual(filt.last_rejections["size"], 1)

    def test_slices_preserve_baseline_and_density_scaling(self):
        mask = np.zeros((4, 4), np.uint8)
        mask[:2, :2] = 255
        mask[:, 2:] = 255
        bird = obj(mask)
        base = CalibratedAreaCalculator(1, 1000).volume_from_mask(bird).volume
        self.assertAlmostEqual(base, math.pi * 10 / 1000)
        self.assertAlmostEqual(CalibratedAreaCalculator(1, 2000, 4).volume_from_mask(bird).volume, base * 2)
        self.assertAlmostEqual(CalibratedAreaCalculator(1, 1000, 2).volume_from_mask(bird).volume, base)
        self.assertAlmostEqual(CalibratedAreaCalculator(1, 1000, 1).volume_from_mask(bird).volume,
                               math.pi * 4 / 1000)
        for slices in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                CalibratedAreaCalculator(1, slice_count=slices)

    def test_grid_ranks_only_calibration_and_rejects_missing_or_overlapping_events(self):
        bird = obj(np.ones((10, 10), np.uint8))
        matches = [("train", 1, bird, 1), ("test", 5, bird, 1)]
        grid = evaluate_grid(matches, ["train"], ["test"], [None, 2], [1000, 2000])
        self.assertEqual(grid[0]["density_kg_m3"], 1000)
        self.assertEqual(grid[0]["held_out_events"], 1)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            evaluate_grid(matches, ["train"], ["train"], [None], [900])
        with self.assertRaisesRegex(ValueError, "No matched frames"):
            evaluate_grid(matches, ["missing"], [], [None], [900])

    def test_collect_matches_reads_only_labeled_frames_and_matches_by_iou(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test_videos").mkdir()
            (root / "ground_truth").mkdir()
            video = root / "test_videos" / "bird.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (80, 80))
            self.assertTrue(writer.isOpened())
            writer.write(np.zeros((80, 80, 3), np.uint8))
            writer.release()
            with (root / "ground_truth" / "gt_boxes.csv").open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["video_name", "frame_idx", "x", "y", "w", "h", "weight_kg", "event_id"])
                w.writerow(["bird.mp4", 0, 10, 10, 20, 20, 2, "e"])
            candidate = obj(np.ones((20, 20), np.uint8), (10, 10, 20, 20))
            other = obj(np.ones((20, 20), np.uint8), (50, 50, 20, 20))
            with patch("tune_validation.ClassicalSegmenter") as segmenter, \
                 patch("tune_validation.SimpleObjectFilter") as filt:
                segmenter.return_value.segment.return_value = np.zeros((80, 80), np.uint8)
                filt.return_value.should_process.return_value = ([candidate, other], None)
                matches = collect_matches(root, {"bird.mp4": [[0, 0], [79, 0], [79, 79], [0, 79]]},
                                          [{"video_name": "bird.mp4", "pixel_to_cm": "0.1"}], 1, 0.3, 0.8)
            self.assertEqual(len(matches), 1)
            self.assertEqual((matches[0][0], matches[0][1], matches[0][2], matches[0][3]),
                             ("e", 2, candidate, 0.1))

    def test_notebook_cells_compile_and_tuning_defaults_off(self):
        notebook = json.loads((ROOT / "src" / "validation_walkthrough.ipynb").read_text())
        codes = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        for code in codes:
            compile(code, "<notebook>", "exec")
        self.assertTrue(any("RUN_TUNING = False" in code for code in codes))

    def test_notebook_displays_selected_masks_and_actual_slice_geometry(self):
        notebook = json.loads((ROOT / "src" / "validation_walkthrough.ipynb").read_text())
        codes = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        filter_cell = next(code for code in codes if code.startswith("obj_filter ="))
        estimate_cell = next(code for code in codes if code.startswith("calculator ="))
        slices_cell = next(code for code in codes if code.startswith("from AreaCalculatorStrategy import mask_slice_profile"))
        mask = np.zeros((300, 640), np.uint8)
        cv2.ellipse(mask, (150, 150), (90, 60), 0, 0, 360, 255, -1)
        frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        images, displays = [], []
        from Factories import AreaCalculatorFactory, AreaMethod
        namespace = dict(ROOT=ROOT, cv2=cv2, np=np, math=math, mask=mask, frame=frame,
                         video_name="bird.mp4", SimpleObjectFilter=SimpleObjectFilter,
                         show_image=lambda image: images.append(image.copy()),
                         display=lambda data: displays.append(data),
                         AreaCalculatorFactory=AreaCalculatorFactory, AreaMethod=AreaMethod,
                         pixel_to_cm=0.1, DENSITY=900.0, SLICE_COUNT=3, ALLOW_FRAME_EDGE=True)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(filter_cell, namespace)
            exec(estimate_cell, namespace)
            exec(slices_cell, namespace)
        self.assertEqual(len(namespace["selected"]), 1)
        selected = namespace["selected"][0]
        x, y, w, h = selected.crop_region
        np.testing.assert_array_equal(selected.source_mask, mask[y:y+h, x:x+w])
        self.assertEqual(len(images), 7)
        np.testing.assert_array_equal(images[2], selected.source_mask)
        self.assertTrue(np.any(images[3] != frame))
        np.testing.assert_array_equal(images[4], selected.mask)
        rows = displays[-1]
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["sample_column_px"] for r in rows],
                         list(mask_slice_profile(selected.mask, 3)["sample_x"]))
        self.assertAlmostEqual(sum(r["volume_cm3"] for r in rows),
                               selected.volume * 1_000_000 / namespace["DENSITY"], delta=0.001)
        for sample, top, bottom in zip(*(mask_slice_profile(selected.mask, 3)[key]
                                         for key in ("sample_x", "y_min", "y_max"))):
            np.testing.assert_array_equal(images[6][top:bottom+1, sample],
                                          np.tile([0, 255, 255], (bottom-top+1, 1)))

    def test_slice_profile_uses_occupied_columns_and_sampled_spans(self):
        mask = np.zeros((4, 7), np.uint8)
        for col, height in ((0, 1), (2, 3), (3, 2), (6, 4)):
            mask[:height, col] = 255
        profile = mask_slice_profile(mask, 2)
        self.assertEqual(profile["x_start"].tolist(), [0, 3])
        self.assertEqual(profile["x_end"].tolist(), [2, 6])
        self.assertEqual(profile["sample_x"].tolist(), [0, 3])
        self.assertEqual(profile["y_max"].tolist(), [0, 1])
        self.assertEqual(profile["thickness_px"].tolist(), [2, 2])
        self.assertEqual(len(mask_slice_profile(np.zeros_like(mask))["sample_x"]), 0)

    def test_runner_and_tuning_use_same_scale_mask_and_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test_videos").mkdir()
            (root / "ground_truth").mkdir()
            video = root / "test_videos" / "bird.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 300))
            self.assertTrue(writer.isOpened())
            frame = np.zeros((300, 640, 3), np.uint8)
            cv2.ellipse(frame, (150, 150), (90, 60), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(frame, (470, 150), (90, 60), 0, 0, 360, (255, 255, 255), -1)
            writer.write(frame)
            writer.release()
            with (root / "test_videos" / "metadata.csv").open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["video_name"])
                w.writerow(["bird.mp4"])
            events = [{"video_name": "bird.mp4", "pixel_to_cm": "0.1"}]
            with (root / "ground_truth" / "events.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["video_name", "pixel_to_cm"])
                w.writeheader()
                w.writerows(events)
            with (root / "ground_truth" / "gt_boxes.csv").open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["video_name", "frame_idx", "x", "y", "w", "h", "weight_kg", "event_id"])
                w.writerow(["bird.mp4", 1, 60, 90, 181, 121, 2, "e"])
            polygons = {"bird.mp4": [[0, 0], [300, 0], [300, 299], [0, 299]]}
            config = root / "polygons.json"
            config.write_text(json.dumps(polygons))
            subprocess.run([sys.executable, str(ROOT / "src" / "run_eval.py"), str(root),
                            "--scale-polygons", str(config), "--density", "1000", "--slices", "5"],
                           check=True, capture_output=True, text=True)
            with (root / "results" / "bird_results.csv").open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            matches = collect_matches(root, polygons, events, 0, 0.3, 0.8)
            self.assertEqual(len(matches), 1)
            predicted = CalibratedAreaCalculator(matches[0][3], 1000, 5).volume_from_mask(matches[0][2]).volume
            self.assertAlmostEqual(float(rows[0]["volume"]), predicted, places=5)

    def test_runner_and_tuning_allow_frame_edge_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test_videos").mkdir()
            (root / "ground_truth").mkdir()
            video = root / "test_videos" / "bird.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 300))
            self.assertTrue(writer.isOpened())
            frame = np.zeros((300, 640, 3), np.uint8)
            cv2.ellipse(frame, (600, 150), (95, 60), 0, 0, 360, (255, 255, 255), -1)
            writer.write(frame)
            writer.release()
            (root / "test_videos" / "metadata.csv").write_text("video_name\nbird.mp4\n")
            (root / "ground_truth" / "events.csv").write_text("video_name,pixel_to_cm\nbird.mp4,0.1\n")
            (root / "ground_truth" / "gt_boxes.csv").write_text(
                "video_name,frame_idx,x,y,w,h,weight_kg,event_id\nbird.mp4,1,505,90,135,121,2,e\n")
            events = [{"video_name": "bird.mp4", "pixel_to_cm": "0.1"}]
            for allow in (False, True):
                command = [sys.executable, str(ROOT / "src" / "run_eval.py"), str(root)]
                if allow:
                    command.append("--allow-frame-edge")
                subprocess.run(command, check=True, capture_output=True, text=True)
                with (root / "results" / "bird_results.csv").open(newline="") as fh:
                    rows = list(csv.DictReader(fh))
                matches = collect_matches(root, None, events, 0, 0.3,
                                          reject_frame_edge=not allow)
                self.assertEqual(len(rows), int(allow))
                self.assertEqual(len(matches), int(allow))

    def test_cropped_runner_and_tuning_calibrate_from_feeder_without_polygon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test_videos").mkdir()
            (root / "ground_truth").mkdir()
            video = root / "test_videos" / "bird.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 300))
            self.assertTrue(writer.isOpened())
            frame = np.zeros((300, 640, 3), np.uint8)
            cv2.ellipse(frame, (150, 150), (90, 60), 0, 0, 360, (255, 255, 255), -1)
            cv2.circle(frame, (500, 150), 50, (0, 128, 255), -1)
            feeder_only = np.zeros_like(frame)
            cv2.circle(feeder_only, (500, 150), 50, (0, 128, 255), -1)
            writer.write(frame)
            for _ in range(4):
                writer.write(feeder_only)
            writer.release()
            (root / "test_videos" / "metadata.csv").write_text("video_name\nbird.mp4\n")
            (root / "ground_truth" / "events.csv").write_text("video_name,pixel_to_cm\nbird.mp4,\n")
            (root / "ground_truth" / "gt_boxes.csv").write_text(
                "video_name,frame_idx,x,y,w,h,weight_kg,event_id\nbird.mp4,1,60,90,181,121,2,e\n")

            px = calibrate_pixel_to_cm(str(video))
            self.assertAlmostEqual(px, 35.56 / 100, delta=0.015)
            subprocess.run([sys.executable, str(ROOT / "src" / "run_eval.py"), str(root),
                            "--density", "1000", "--slices", "5"],
                           check=True, capture_output=True, text=True)
            with (root / "results" / "bird_results.csv").open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            matches = collect_matches(root, None, [{"video_name": "bird.mp4", "pixel_to_cm": ""}],
                                      0, 0.3, 0.8)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(matches), 1)
            self.assertAlmostEqual(matches[0][3], px)
            predicted = CalibratedAreaCalculator(px, 1000, 5).volume_from_mask(matches[0][2]).volume
            self.assertAlmostEqual(float(rows[0]["volume"]), predicted, places=5)

    def test_feeder_calibration_fails_when_feeder_is_not_visible(self):
        with patch("FeederCalibration.FeederReference.from_video",
                   side_effect=RuntimeError("Feeder calibration found only 0 good fits")):
            with self.assertRaisesRegex(RuntimeError, "No orange feeder detected"):
                calibrate_pixel_to_cm("cropped.mp4")


if __name__ == "__main__":
    unittest.main()
