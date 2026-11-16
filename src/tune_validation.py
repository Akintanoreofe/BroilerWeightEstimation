"""Explore slice counts and densities on annotated weighing events."""

import math
import statistics
from collections import defaultdict
from pathlib import Path

import cv2

from AreaCalculatorStrategy import CalibratedAreaCalculator
from FeederCalibration import calibrate_pixel_to_cm
from ObjectFilterStrategy import SimpleObjectFilter
from SegmentationStrategy import ClassicalSegmenter
from evaluate_against_gt import iou, load_gt


def collect_matches(export_dir, polygons, events, frame_offset, iou_threshold, overlap=0.8,
                    reject_frame_edge=True):
    """Re-segment labeled frames and retain full masks matched to scale readings.

    Pass polygons=None for already-cropped clips without a platform filter.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    export_dir = Path(export_dir)
    gt = load_gt(export_dir / "ground_truth" / "gt_boxes.csv", frame_offset)
    by_video = defaultdict(list)
    for (video, frame), rows in gt.items():
        if frame < 1:
            raise ValueError(f"Invalid one-based frame number for {video}: {frame}")
        by_video[video].append((frame, rows))
    event_by_video = {event["video_name"]: event for event in events}
    segmenter = ClassicalSegmenter()
    matches = []
    for video, frames in sorted(by_video.items()):
        if polygons is not None and video not in polygons:
            raise ValueError(f"No scale polygon for {video}")
        path = export_dir / "test_videos" / video
        fixed = event_by_video.get(video, {}).get("pixel_to_cm", "")
        px = float(fixed) if fixed else calibrate_pixel_to_cm(str(path))
        if not math.isfinite(px) or px <= 0:
            raise ValueError(f"Invalid pixel_to_cm for {video}: {px}")
        obj_filter = SimpleObjectFilter(
            target_labels=["Broiler"], template_mask=str(Path(__file__).parent / "broiler_template.png"),
            min_area=500.0, scale_polygon=polygons[video] if polygons is not None else None,
            min_scale_overlap=overlap, reject_frame_edge=reject_frame_edge)
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            for frame_idx, labels in sorted(frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot read frame {frame_idx} from {path}")
                objects, _ = obj_filter.should_process(segmenter.segment(frame))
                candidates = sorted(((iou(obj.crop_region, label["box"]), i, j)
                                     for i, obj in enumerate(objects)
                                     for j, label in enumerate(labels)), reverse=True)
                used_objects, used_labels = set(), set()
                for score, i, j in candidates:
                    if score < iou_threshold or i in used_objects or j in used_labels:
                        continue
                    used_objects.add(i)
                    used_labels.add(j)
                    matches.append((labels[j]["event_id"], labels[j]["weight_kg"], objects[i], px))
        finally:
            cap.release()
    return matches


def evaluate_grid(matches, train_ids, test_ids, slice_counts, densities):
    """Rank by calibration-event RMSE; report held-out event metrics separately."""
    train_ids, test_ids = set(map(str, train_ids)), set(map(str, test_ids))
    if not train_ids or train_ids & test_ids:
        raise ValueError("Choose nonempty calibration event IDs disjoint from held-out event IDs")
    if not slice_counts or not densities:
        raise ValueError("Provide at least one slice count and density")
    available = {event for event, _, _, _ in matches}
    missing = (train_ids | test_ids) - available
    if missing:
        raise ValueError(f"No matched frames for event IDs: {sorted(missing)}")
    if any(not math.isfinite(actual) or actual <= 0
           for event, actual, _, _ in matches if event in train_ids or event in test_ids):
        raise ValueError("Matched scale readings must be positive finite weights in kg")
    rows = []
    for slices in slice_counts:
        for density in densities:
            groups = defaultdict(list)
            for event, actual, obj, px in matches:
                if event in train_ids or event in test_ids:
                    estimate = CalibratedAreaCalculator(px, density, slices).volume_from_mask(obj).weight_kg
                    groups[event].append((actual, estimate))

            def metrics(ids):
                errors = [statistics.median(v[1] for v in groups[event])
                          - statistics.median(v[0] for v in groups[event]) for event in ids]
                return (math.sqrt(statistics.fmean(e * e for e in errors)),
                        statistics.fmean(abs(e) for e in errors)) if errors else (None, None)

            train_rmse, train_mae = metrics(train_ids)
            test_rmse, test_mae = metrics(test_ids)
            rows.append(dict(slices=slices, density_kg_m3=density, calibration_events=len(train_ids),
                             calibration_rmse_kg=train_rmse, calibration_mae_kg=train_mae,
                             held_out_events=len(test_ids), held_out_rmse_kg=test_rmse,
                             held_out_mae_kg=test_mae))
    return sorted(rows, key=lambda row: row["calibration_rmse_kg"])
