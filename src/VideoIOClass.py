"""OpenCV-backed readers and writers for video streams."""

from typing import Tuple
import cv2
import numpy as np


class VideoReader:
    """Read frames and metadata from an OpenCV video source."""

    def __init__(self, source: str):
        """Open a video source.

        Parameters
        ----------
        source : str
            Video file path or source accepted by ``cv2.VideoCapture``.

        Raises
        ------
        RuntimeError
            If the source cannot be opened.
        """

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

    def read(self) -> Tuple[bool, np.ndarray]:
        """Read the next frame.

        Returns
        -------
        tuple of bool and numpy.ndarray
            OpenCV's ``(success, frame)`` result. The frame may be empty when
            reading fails.
        """
        return self.cap.read()

    def release(self):
        """Release the underlying video capture resource."""
        self.cap.release()

    def get_info(self) -> Tuple[float, int, int]:
        """Return the stream frame rate and dimensions.

        Returns
        -------
        tuple of float, int, int
            Frames per second, frame width, and frame height.
        """
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return fps, width, height


class VideoWriter:
    """Write frames to an MP4 video using the ``mp4v`` codec."""

    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int]):
        """Create a writer for a video destination.

        Parameters
        ----------
        output_path : str
            Destination video path.
        fps : float
            Output frame rate.
        frame_size : tuple of int
            Frame dimensions as ``(width, height)``.
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    def write(self, frame: np.ndarray):
        """Write one frame to the destination video.

        Parameters
        ----------
        frame : numpy.ndarray
            Frame to write.
        """
        self.writer.write(frame)

    def release(self):
        """Release the underlying video writer resource."""
        self.writer.release()