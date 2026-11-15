"""Interfaces and implementations for frame segmentation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import List, Tuple
import cv2
import numpy as np
from defaults import (
    CHROMA_WEIGHT,
    CLOSE_KERNEL_CM,
    DISTANCE_PEAK_RATIO,
    MAX_BIRD_ELONGATION_RATIO,
    MAX_COMPONENT_LONG_SIDE_FEEDER_RADII,
    MIN_BIRD_AREA_CM2,
    OTSU_LIGHTING_JUMP_RATIO,
    OPEN_KERNEL_CM,
)


class Segmenter(ABC):
    """Abstract interface for producing segmentation output from a frame."""

    @abstractmethod
    def segment(self, frame: np.ndarray):
        """Segment objects of interest from an OpenCV BGR frame."""
        raise NotImplementedError


@dataclass
class BirdCandidate:
    """A bird-shaped connected component and its calibrated image geometry."""

    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    area_cm2: float
    score: float
    centroid_px: Tuple[float, float]
    quality: dict = field(default_factory=dict)
    measurable: bool = True
    track_id: int = -1


@dataclass
class ElongatedComponent:
    """Automatically rejected elongated structure, available as an occluder."""

    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    area_cm2: float
    elongation_ratio: float
    long_side_cm: float


@dataclass
class SegmentationResult:
    """Intermediate masks, thresholds, bird candidates, and rejected structures."""

    whiteness: np.ndarray
    threshold_stage1: float
    threshold_stage2: float
    stage1_mask: np.ndarray
    stage2_mask: np.ndarray
    opened_mask: np.ndarray
    candidates: List[BirdCandidate]
    rejected_elongated: List[ElongatedComponent]
    threshold_jump: bool = False


class TwoStageOtsuSegmenter(Segmenter):
    """Segment bright unsaturated birds using two adaptive Otsu thresholds."""

    def __init__(
        self,
        feeder_reference,
        min_bird_area_cm2: float = MIN_BIRD_AREA_CM2,
        max_bird_elongation_ratio: float = MAX_BIRD_ELONGATION_RATIO,
        chroma_weight: float = CHROMA_WEIGHT,
    ):
        if feeder_reference.px_per_cm <= 0:
            raise ValueError("Feeder reference must provide positive px_per_cm.")
        if min_bird_area_cm2 <= 0 or max_bird_elongation_ratio <= 1:
            raise ValueError("Bird area and elongation thresholds must be positive.")
        self.reference = feeder_reference
        self.px_per_cm = float(feeder_reference.px_per_cm)
        self.min_bird_area_cm2 = float(min_bird_area_cm2)
        self.max_bird_elongation_ratio = float(max_bird_elongation_ratio)
        self.chroma_weight = float(chroma_weight)
        self.threshold_history = []

    def whiteness_channel(self, frame: np.ndarray) -> np.ndarray:
        """Return normalized Lab lightness discounted by chroma."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
        whiteness = lab[:, :, 0] - self.chroma_weight * chroma
        minimum, maximum = float(whiteness.min()), float(whiteness.max())
        if maximum <= minimum:
            return np.zeros(frame.shape[:2], np.uint8)
        normalized = cv2.normalize(whiteness, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.GaussianBlur(normalized.astype(np.uint8), (5, 5), 0)

    def segment(self, frame: np.ndarray) -> SegmentationResult:
        """Return stage diagnostics, candidates, and elongated occluders."""
        whiteness = self.whiteness_channel(frame)
        threshold_stage1, stage1 = cv2.threshold(
            whiteness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        bright_values = whiteness[stage1 > 0]
        if bright_values.size == 0:
            threshold_stage2 = threshold_stage1
            stage2 = np.zeros_like(stage1)
        else:
            threshold_stage2, _ = cv2.threshold(
                bright_values.reshape(-1, 1), 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            stage2 = (whiteness > threshold_stage2).astype(np.uint8) * 255

        opening_kernel = self._kernel(OPEN_KERNEL_CM)
        closing_kernel = self._kernel(CLOSE_KERNEL_CM)
        opened = cv2.morphologyEx(stage2, cv2.MORPH_OPEN, opening_kernel)
        opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, closing_kernel)
        opened = self._fill_holes(opened)
        components, rejected = self._components(opened, frame)
        prior_median = float(np.median(self.threshold_history)) if self.threshold_history else None
        threshold_jump = bool(
            prior_median and prior_median > 0
            and abs(float(threshold_stage2) - prior_median) / prior_median > OTSU_LIGHTING_JUMP_RATIO
        )
        if threshold_jump:
            print(f"[TwoStageOtsuSegmenter] Lighting threshold jump: t2={threshold_stage2:.1f}")
        self.threshold_history.append(float(threshold_stage2))
        return SegmentationResult(
            whiteness=whiteness,
            threshold_stage1=float(threshold_stage1),
            threshold_stage2=float(threshold_stage2),
            stage1_mask=stage1,
            stage2_mask=stage2,
            opened_mask=opened,
            candidates=components,
            rejected_elongated=rejected,
            threshold_jump=threshold_jump,
        )

    def _kernel(self, diameter_cm: float) -> np.ndarray:
        diameter_px = max(1, int(round(diameter_cm * self.px_per_cm)))
        diameter_px += 1 - diameter_px % 2
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter_px, diameter_px))

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flood = padded.copy()
        cv2.floodFill(flood, None, (0, 0), 255)
        holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
        return cv2.bitwise_or(mask, holes)

    def _components(self, mask: np.ndarray, frame: np.ndarray):
        labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates, rejected = [], []
        min_area_px = self.min_bird_area_cm2 * self.px_per_cm ** 2
        max_long_side_px = MAX_COMPONENT_LONG_SIDE_FEEDER_RADII * self.reference.major_px / 2.0
        for label in range(1, labels_count):
            x, y, width, height, area_px = map(int, stats[label])
            if area_px < min_area_px:
                continue
            component = np.where(labels == label, 255, 0).astype(np.uint8)
            for split_mask in self._split_multiple_peaks(component):
                ys, xs = np.where(split_mask > 0)
                if len(xs) == 0:
                    continue
                x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
                local = split_mask[y0:y1 + 1, x0:x1 + 1]
                rect = cv2.minAreaRect(cv2.findNonZero(local))
                side_a, side_b = rect[1]
                long_side_px = max(side_a, side_b)
                short_side_px = max(min(side_a, side_b), 1.0)
                ratio = long_side_px / short_side_px
                area_cm2 = float(cv2.countNonZero(local) / self.px_per_cm ** 2)
                if ratio > self.max_bird_elongation_ratio or long_side_px > max_long_side_px:
                    rejected.append(ElongatedComponent(
                        mask=split_mask,
                        bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                        area_cm2=area_cm2,
                        elongation_ratio=float(ratio),
                        long_side_cm=float(long_side_px / self.px_per_cm),
                    ))
                    continue
                candidates.append(BirdCandidate(
                    mask=split_mask,
                    bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                    area_cm2=area_cm2,
                    score=float(area_cm2 / max(self.min_bird_area_cm2, 1.0)),
                    centroid_px=(float(xs.mean()), float(ys.mean())),
                ))
        return candidates, rejected

    @staticmethod
    def _split_multiple_peaks(component: np.ndarray) -> List[np.ndarray]:
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        if distance.max() <= 0:
            return []
        peaks = (distance >= DISTANCE_PEAK_RATIO * distance.max()).astype(np.uint8)
        count, markers = cv2.connectedComponents(peaks)
        if count <= 2:
            return [component]
        sure_background = cv2.dilate(component, np.ones((3, 3), np.uint8), iterations=1)
        markers = markers.astype(np.int32) + 1
        markers[sure_background == 0] = 0
        watershed_input = cv2.cvtColor(component, cv2.COLOR_GRAY2BGR)
        cv2.watershed(watershed_input, markers)
        split = []
        for marker in range(2, count + 1):
            part = (markers == marker).astype(np.uint8) * 255
            if cv2.countNonZero(part):
                split.append(part)
        return split or [component]


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
