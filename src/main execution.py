"""Configure and run the broiler-weight estimation pipeline.

When executed as a script, :func:`main` processes the configured sample video,
calibrates its pixel scale, and writes annotated video and CSV results.
"""

import os
from pathlib import Path

from CameraLoader import CameraConfigLoader
from Factories import AreaCalculatorFactory, AreaMethod, SegmentationMethod, SegmenterFactory
from FeederCalibration import calibrate_pixel_to_cm
from ObjectFilterStrategy import SimpleObjectFilter
from OutputWriters import CSVOutputWriter
from VideoProcessorPipeline import VideoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_VIDEOS_DIR = SCRIPT_DIR.parent / "test_videos"
RESULTS_DIR = SCRIPT_DIR.parent / "results"


def main():
    """Run the pipeline on the configured camera video.

    Creates the result directory, constructs the camera configuration and
    processing strategies, calibrates from the source video, and runs the
    processor. The configured source is ``test_videos/cam1.mp4`` relative to
    the repository root.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    loader = CameraConfigLoader()

    video_path = str(TEST_VIDEOS_DIR / "cam1.mp4")

    cam = loader.get_camera_info(
        camera_id="cam1",
        video_path=video_path,
        video_save_path=str(RESULTS_DIR / "cam1_output.mp4"),
    )

    pixel_to_cm = calibrate_pixel_to_cm(video_path)

    seg_factory = SegmenterFactory()
    area_factory = AreaCalculatorFactory()

    segmenter = seg_factory.create(
        SegmentationMethod.CLASSICAL
    )

    obj_filter = SimpleObjectFilter(
        target_labels=["Broiler"],
        template_mask="broiler_template.png",
        min_area=500.0
    )

    area_calc = area_factory.create(
        AreaMethod.CALIBRATED,
        pixel_to_cm=pixel_to_cm
    )

    output_writer = CSVOutputWriter(csv_path=str(RESULTS_DIR / f"{cam.camera_id}_results.csv"))

    processor = VideoProcessor(
        camera_config=cam,
        segmenter=segmenter,
        area_calculator=area_calc,
        obj_filter=obj_filter,
        output_writer=output_writer,
        max_frames=None
    )
    processor.run()


if __name__ == "__main__":
    main()
