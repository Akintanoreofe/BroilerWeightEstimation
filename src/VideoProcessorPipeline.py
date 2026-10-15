"""Coordinate segmentation, object filtering, estimation, and video output."""

from typing import Optional
import cv2
from AreaCalculatorStrategy import AreaCalculator
from ObjectFilterStrategy import ObjectFilter
from OutputWriters import OutputWriter
from SegmentationStrategy import Segmenter
from CameraLoader import CameraConfig
from VideoIOClass import VideoReader, VideoWriter


class VideoProcessor:
    """Run the processing pipeline for a configured camera video."""

    def __init__(
        self,
        camera_config: CameraConfig,
        segmenter: Segmenter,
        area_calculator: AreaCalculator,
        obj_filter: ObjectFilter,
        output_writer: OutputWriter,
        show_img: bool = True,
        max_frames: Optional[int] = None
    ):
        """Initialize pipeline strategies and open its input/output streams.

        Parameters
        ----------
        camera_config : CameraConfig
            Camera identifier and input/output video paths.
        segmenter : Segmenter
            Strategy that creates a mask for each frame.
        area_calculator : AreaCalculator
            Strategy that estimates object volume or mass.
        obj_filter : ObjectFilter
            Strategy that selects candidate objects from masks.
        output_writer : OutputWriter
            Writer for per-object result records.
        show_img : bool, default=True
            Whether to display a live processing window.
        max_frames : int or None, default=None
            Maximum number of frames to process; ``None`` processes until the
            source ends.

        Raises
        ------
        RuntimeError
            If the configured input video cannot be opened.
        """
        self.config = camera_config
        self.segmenter = segmenter
        self.area_calculator = area_calculator
        self.obj_filter = obj_filter
        self.output_writer = output_writer
        self.max_frames = max_frames
        self.video_reader = VideoReader(str(camera_config.video_path))
        self.show_window = show_img

        fps, width, height = self.video_reader.get_info()
        self.video_writer = VideoWriter(
            output_path=str(camera_config.video_save_path),
            fps=fps,
            frame_size=(width, height)
        )

    def run(self):
        """Process frames until the video ends, the limit is reached, or quit.

        Selected objects are annotated in the output video and written as
        records. Resources are released by :meth:`clean_up` at normal exit.
        """

        frame_idx = 0

        window_name = f"Camera: {self.config.camera_id}"
        if self.show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)

        print(f"[VideoProcessor] Start processing camera: {self.config.camera_id}")

        while True:
            ret, frame = self.video_reader.read()
            if not ret:
                print("[VideoProcessor] End of video or cannot read frame.")
                break

            frame_idx += 1
            if self.max_frames is not None and frame_idx > self.max_frames:
                print(f"[VideoProcessor] Reached max_frames={self.max_frames}. Stopping.")
                break

            # 1) Segmentation
            objects = self.segmenter.segment(frame)

            # 2) Filtering
            selected_objs, clean_mask = self.obj_filter.should_process(objects)

            # 3) Weight
            for obj in selected_objs:

                obj = self.area_calculator.volume_from_mask(obj)
                x, y, w, h = obj.crop_region
                volume = obj.volume

                cx = x + w // 2
                cy = y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{volume:.2f} kg",
                            (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

                record = {
                    "camera_id": self.config.camera_id,
                    "frame_idx": frame_idx,
                    "label": obj.label,
                    "score": obj.confidence_dt + obj.confidence_tpl,
                    "volume": volume,
                    "crop_region": (x, y, w, h),
                }
                self.output_writer.write(record)

            if self.show_window:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    print("[VideoProcessor] Quitting due to user input.")
                    break

            self.video_writer.write(frame)

        self.clean_up()

    def clean_up(self):
        """Release video and result-writer resources and close display windows."""
        self.video_reader.release()
        self.video_writer.release()
        self.output_writer.close()

        if self.show_window:
            cv2.destroyAllWindows()

        print(f"[VideoProcessor] Finished camera: {self.config.camera_id}")