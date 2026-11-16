"""Run BroilerWeightEstimation on every clip in a FlockLabel evaluation export.

Usage:
    python src/run_eval.py /path/to/export --out /path/to/export/results
"""
import argparse
import csv
import math
import json
from pathlib import Path
import cv2

from FeederCalibration import calibrate_pixel_to_cm
from Factories import AreaCalculatorFactory, AreaMethod, SegmentationMethod, SegmenterFactory
from ObjectFilterStrategy import SimpleObjectFilter
from OutputWriters import CSVOutputWriter
from CameraLoader import CameraConfigLoader
from VideoProcessorPipeline import VideoProcessor
from scale_region import scale_mask_for_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("export_dir", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--scale-polygons", type=Path, help="JSON mapping exact video names to platform polygon points")
    p.add_argument("--min-scale-overlap", type=float, default=0.8)
    p.add_argument("--density", type=float, default=900.0, help="kg/m3")
    p.add_argument("--slices", type=int, help="number of slices; default is one per occupied column")
    p.add_argument("--allow-frame-edge", action="store_true",
                   help="allow contours touching the frame edge")
    a = p.parse_args()
    out = a.out or a.export_dir / "results"
    out.mkdir(parents=True, exist_ok=True)

    with open(a.export_dir / "ground_truth" / "events.csv", newline="") as fh:
        events = {r["video_name"]: r for r in csv.DictReader(fh)}
    with open(a.export_dir / "test_videos" / "metadata.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    stems = [Path(row["video_name"]).stem for row in rows]
    if len(stems) != len(set(stems)):
        raise ValueError("Video names must have unique stems to avoid overwriting results")
    polygons = None
    if a.scale_polygons:
        with a.scale_polygons.open() as fh:
            polygons = json.load(fh)
        if not isinstance(polygons, dict) or set(polygons) != {row["video_name"] for row in rows}:
            raise ValueError("scale polygons must contain exactly one entry per metadata video name")
        for row in rows:
            name = row["video_name"]
            cap = cv2.VideoCapture(str(a.export_dir / "test_videos" / name))
            try:
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok:
                raise RuntimeError(f"Cannot read first frame from {name}")
            try:
                scale_mask_for_frame(polygons[name], frame.shape)
            except ValueError as err:
                raise ValueError(f"Invalid scale polygon for {name}: {err}") from err
    if not math.isfinite(a.min_scale_overlap) or not 0 < a.min_scale_overlap <= 1:
        p.error("--min-scale-overlap must be in (0, 1]")
    if not math.isfinite(a.density) or a.density <= 0 or (a.slices is not None and a.slices < 1):
        p.error("--density and --slices must be positive")
    skipped = []
    for row in rows:
        name = row["video_name"]
        video = a.export_dir / "test_videos" / name
        stem = Path(name).stem
        fixed = events.get(name, {}).get("pixel_to_cm", "")
        try:
            px = float(fixed) if fixed else calibrate_pixel_to_cm(str(video))
        except RuntimeError as err:
            print(f"[skip] {name}: {err}")
            skipped.append(name)
            continue
        if not math.isfinite(px) or px <= 0:
            raise ValueError(f"Invalid pixel_to_cm for {name}: {px}")
        print(f"[calibration] {name}: {px:g} cm/pixel")
        cam = CameraConfigLoader().get_camera_info(stem, str(video), str(out / f"{stem}_annotated.mp4"))
        VideoProcessor(
            camera_config=cam,
            segmenter=SegmenterFactory().create(SegmentationMethod.CLASSICAL),
            area_calculator=AreaCalculatorFactory().create(
                AreaMethod.CALIBRATED, pixel_to_cm=px, broiler_density=a.density,
                slice_count=a.slices),
            obj_filter=SimpleObjectFilter(target_labels=["Broiler"],
                                          template_mask=str(Path(__file__).resolve().parent / "broiler_template.png"),
                                          min_area=500.0, scale_polygon=polygons[name] if polygons is not None else None,
                                          min_scale_overlap=a.min_scale_overlap,
                                          reject_frame_edge=not a.allow_frame_edge),
            output_writer=CSVOutputWriter(str(out / f"{stem}_results.csv")),
            show_img=False,
        ).run()
    if skipped:
        raise RuntimeError(f"Calibration failed for {len(skipped)} video(s): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
