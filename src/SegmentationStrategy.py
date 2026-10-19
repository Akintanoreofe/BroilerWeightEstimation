"""Interfaces and implementations for frame segmentation."""

from abc import ABC, abstractmethod
import cv2
import numpy as np


class Segmenter(ABC):
    """Abstract interface for producing a binary mask from a video frame."""

    @abstractmethod
    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Segment the objects of interest in a frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Input color image in OpenCV BGR format.

        Returns
        -------
        numpy.ndarray
            Binary segmentation mask.
        """
        pass


class DeepLearningSegmenter(Segmenter):
    """Placeholder for a deep-learning based segmentation model."""

    def __init__(self, model_path: str):
        """Initialize with the path to the segmentation model.

        Parameters
        ----------
        model_path : str
            Model file path.
        """
        self.model_path = model_path
        print(f"[DeepLearningSegmenter] Initialized with model: {model_path}")

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Segment a frame using the deep-learning model.

        Parameters
        ----------
        frame : numpy.ndarray
            Input color image in OpenCV BGR format.

        Raises
        ------
        NotImplementedError
            Always, because deep-learning inference is not implemented.
        """
        raise NotImplementedError("DeepLearningSegmenter is not yet implemented.")


class ClassicalSegmenter(Segmenter):
    """Create a foreground mask using thresholding and morphology."""

    def __init__(self):
        """Initialize the classical segmenter."""
        print("[ClassicalSegmenter] Initialized")

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Produce a binary mask while suppressing the orange feeder.

        Parameters
        ----------
        frame : numpy.ndarray
            Input color image in OpenCV BGR format.

        Returns
        -------
        numpy.ndarray
            Morphologically filtered binary mask.
        """

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        blur = cv2.GaussianBlur(v, (5, 5), 0)
        # Otsu chooses a brightness threshold from this frame rather than a fixed cutoff.
        _, mask = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove orange color (feeder) from mask
        feeder_mask = ((h > 8) & (h < 20) & (s > 150) & (v > 100)).astype(np.uint8) * 255
        mask[feeder_mask == 255] = 0

        kernel = np.ones((5, 5), np.uint8)
        # Opening removes small specks; closing fills gaps in the remaining foreground.
        mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel, iterations=2)

        mask_morph = cv2.erode(mask_morph, kernel, iterations = 1)
        mask_morph = cv2.dilate(mask_morph, kernel, iterations = 1)

        return mask_morph
