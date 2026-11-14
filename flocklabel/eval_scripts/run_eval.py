"""Run BroilerWeightEstimation on every clip in a FlockLabel evaluation export.

Usage (from the repo's src/ folder):
    python /path/run_eval.py /path/to/export --out /path/to/export/results
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))          # repo src/ must be the working dir
from FeederCalibration import calibrate_pixel_to_cm
from Factories import AreaCalculatorFactory, AreaMethod, SegmentationMethod, SegmenterFactory
from ObjectFilterStrategy import SimpleObjectFilter
from OutputWriters import CSVOutputWriter
from CameraLoader import CameraConfigLoader
from VideoProcessorPipeline import VideoProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("export_dir", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    out = a.out or a.export_dir / "results"
    out.mkdir(parents=True, exist_ok=True)

    events = {r["video_name"]: r for r in
              csv.DictReader(open(a.export_dir / "ground_truth" / "events.csv"))}
    for row in csv.DictReader(open(a.export_dir / "test_videos" / "metadata.csv")):
        name = row["video_name"]
        video = a.export_dir / "test_videos" / name
        stem = Path(name).stem
        fixed = events.get(name, {}).get("pixel_to_cm", "")
        try:
            px = float(fixed) if fixed else calibrate_pixel_to_cm(str(video))
        except RuntimeError as err:
            print(f"[skip] {name}: {err}")
            continue
        cam = CameraConfigLoader().get_camera_info(stem, str(video), str(out / f"{stem}_annotated.mp4"))
        VideoProcessor(
            camera_config=cam,
            segmenter=SegmenterFactory().create(SegmentationMethod.CLASSICAL),
            area_calculator=AreaCalculatorFactory().create(
                AreaMethod.CALIBRATED, pixel_to_cm=px),  # density fixed at 900 kg/m3
            obj_filter=SimpleObjectFilter(target_labels=["Broiler"],
                                          template_mask="broiler_template.png", min_area=500.0),
            output_writer=CSVOutputWriter(str(out / f"{stem}_results.csv")),
            show_img=False,
        ).run()


if __name__ == "__main__":
    main()
