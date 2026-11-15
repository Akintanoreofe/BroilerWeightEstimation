"""PCA-aligned silhouette slicing and pluggable 3D volume models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


class CrossSection(ABC):
    """Convert a 2D silhouette width into a cross-sectional area."""

    @abstractmethod
    def area_cm2(self, width_cm: float, max_width_cm: float,
                 index: int, n_slices: int) -> float:
        """Return the modeled cross-sectional area in square centimeters."""


class Circular(CrossSection):
    """Circular cross-section with silhouette width as its diameter."""

    def area_cm2(self, width_cm: float, max_width_cm: float,
                 index: int, n_slices: int) -> float:
        return math.pi * (width_cm / 2.0) ** 2

    def __repr__(self):
        return "Circular()"


class Elliptical(CrossSection):
    """Elliptical cross-section whose depth is ``k`` times the width."""

    def __init__(self, k: float):
        if not math.isfinite(k) or k <= 0:
            raise ValueError("k must be positive and finite")
        self.k = float(k)

    def area_cm2(self, width_cm: float, max_width_cm: float,
                 index: int, n_slices: int) -> float:
        return math.pi * width_cm * (self.k * width_cm) / 4.0

    def __repr__(self):
        return f"Elliptical(k={self.k:g})"


class ProfileScaled(CrossSection):
    """Elliptical sections with depth from a normalized longitudinal profile."""

    def __init__(self, profile_ratios: Optional[Sequence[float]] = None):
        if profile_ratios is not None:
            values = np.asarray(profile_ratios, dtype=np.float64)
            if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("profile_ratios must be a non-empty sequence of finite nonnegative values")
            self.profile_ratios = values
        else:
            self.profile_ratios = None

    def area_cm2(self, width_cm: float, max_width_cm: float,
                 index: int, n_slices: int) -> float:
        if self.profile_ratios is None:
            depth_cm = width_cm
        else:
            position = 0.0 if n_slices <= 1 else index * (len(self.profile_ratios) - 1) / (n_slices - 1)
            left = int(math.floor(position))
            right = min(left + 1, len(self.profile_ratios) - 1)
            ratio = self.profile_ratios[left] * (right - position) + self.profile_ratios[right] * (position - left)
            depth_cm = max_width_cm * float(ratio)
        return math.pi * width_cm * depth_cm / 4.0

    def __repr__(self):
        return "ProfileScaled()"


class WeightModel(ABC):
    """Map physical volume to an estimated mass."""

    @abstractmethod
    def estimate_weight_kg(self, volume_cm3: float) -> float:
        """Return estimated mass in kilograms."""


class DensityWeightModel(WeightModel):
    """Estimate mass from a constant density in grams per cubic centimeter."""

    def __init__(self, density_g_cm3: float = 0.90):
        if not math.isfinite(density_g_cm3) or density_g_cm3 <= 0:
            raise ValueError("density_g_cm3 must be positive and finite")
        self.density_g_cm3 = float(density_g_cm3)

    def estimate_weight_kg(self, volume_cm3: float) -> float:
        if not math.isfinite(volume_cm3) or volume_cm3 < 0:
            raise ValueError("volume_cm3 must be finite and nonnegative")
        return volume_cm3 * self.density_g_cm3 / 1000.0


@dataclass
class SliceVolumeResult:
    """Volume estimate and the slice geometry used to produce it."""

    volume_cm3: float
    length_cm: float
    max_width_cm: float
    area_cm2: float
    perimeter_cm: float
    slice_widths_cm: list
    slice_volumes_cm3: list
    delta_x_cm: float
    n_slices: int
    aligned_mask: np.ndarray
    slice_positions_px: np.ndarray
    alignment_matrix: np.ndarray
    cross_section: str
    k: Optional[float]
    weight_kg_est: Optional[float] = None


class SliceVolumeEstimator:
    """Estimate body volume from an aligned binary bird outline."""

    def __init__(self, n_slices: Optional[int] = 20,
                 cross_section: Optional[CrossSection] = None,
                 weight_model: Optional[WeightModel] = None,
                 smoothing_strength: float = 0.0):
        if n_slices is not None and (isinstance(n_slices, bool)
                                     or not isinstance(n_slices, int) or n_slices < 1):
            raise ValueError("n_slices must be a positive integer or None")
        if not 0 <= smoothing_strength < 0.5:
            raise ValueError("smoothing_strength must be in [0, 0.5)")
        self.n_slices = n_slices
        self.cross_section = cross_section or Circular()
        self.weight_model = weight_model
        self.smoothing_strength = float(smoothing_strength)

    def estimate(self, bird_mask: np.ndarray, px_per_cm: float) -> SliceVolumeResult:
        """Align, slice, and integrate a binary mask in physical units."""
        if not math.isfinite(px_per_cm) or px_per_cm <= 0:
            raise ValueError("px_per_cm must be positive and finite")
        aligned, alignment_matrix = self._align_mask(bird_mask)
        contours, _ = cv2.findContours(aligned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return self._empty_result(aligned, alignment_matrix)
        contour = max(contours, key=cv2.contourArea)
        polygon = contour[:, 0, :].astype(np.float64)
        if self.smoothing_strength:
            polygon = self._smooth_polygon(polygon, self.smoothing_strength)

        x_min = float(polygon[:, 0].min())
        x_max = float(polygon[:, 0].max())
        length_px = x_max - x_min
        if length_px <= 0:
            return self._empty_result(aligned, alignment_matrix)
        n_slices = self.n_slices or max(1, int(np.count_nonzero(aligned.any(axis=0))))
        n_slices = min(n_slices, max(1, int(math.ceil(length_px))))
        delta_x_px = length_px / n_slices
        positions = x_min + (np.arange(n_slices, dtype=np.float64) + 0.5) * delta_x_px
        widths_px = np.asarray([self._polygon_width(polygon, x) for x in positions])
        widths_cm = widths_px / px_per_cm
        delta_x_cm = delta_x_px / px_per_cm
        max_width_cm = float(widths_cm.max(initial=0.0))
        slice_areas = [
            self.cross_section.area_cm2(float(width), max_width_cm, index, n_slices)
            for index, width in enumerate(widths_cm)
        ]
        slice_volumes = [float(area * delta_x_cm) for area in slice_areas]
        volume_cm3 = float(sum(slice_volumes))
        contour_area_cm2 = float(cv2.contourArea(contour) / px_per_cm ** 2)
        perimeter_cm = float(cv2.arcLength(contour, True) / px_per_cm)
        weight = (self.weight_model.estimate_weight_kg(volume_cm3)
                  if self.weight_model is not None else None)
        return SliceVolumeResult(
            volume_cm3=volume_cm3,
            length_cm=float(length_px / px_per_cm),
            max_width_cm=max_width_cm,
            area_cm2=contour_area_cm2,
            perimeter_cm=perimeter_cm,
            slice_widths_cm=widths_cm.tolist(),
            slice_volumes_cm3=slice_volumes,
            delta_x_cm=float(delta_x_cm),
            n_slices=n_slices,
            aligned_mask=aligned,
            slice_positions_px=positions,
            alignment_matrix=alignment_matrix,
            cross_section=type(self.cross_section).__name__,
            k=getattr(self.cross_section, "k", None),
            weight_kg_est=weight,
        )

    @staticmethod
    def _align_mask(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        binary = (mask > 0).astype(np.uint8) * 255
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return binary, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        cropped = binary[y0:y1 + 1, x0:x1 + 1]
        ys, xs = np.where(cropped > 0)
        coords = np.column_stack((xs, ys)).astype(np.float64)
        centered = coords - coords.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
        height, width = cropped.shape
        center = (float(xs.mean()), float(ys.mean()))
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        rotated_corners = cv2.transform(corners[None, :, :], matrix)[0]
        minimum, maximum = rotated_corners.min(axis=0), rotated_corners.max(axis=0)
        matrix[:, 2] -= minimum
        size = (max(1, int(math.ceil(maximum[0] - minimum[0]))),
                max(1, int(math.ceil(maximum[1] - minimum[1]))))
        rotated = cv2.warpAffine(cropped, matrix, size, flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated = (rotated > 0).astype(np.uint8) * 255
        ys, xs = np.where(rotated > 0)
        x_min, y_min = int(xs.min()), int(ys.min())
        aligned = rotated[y_min:ys.max() + 1, x_min:xs.max() + 1]
        matrix[:, 2] -= np.asarray([x_min, y_min])
        matrix[:, 2] -= matrix[:, :2] @ np.asarray([x0, y0], dtype=np.float64)
        return aligned, matrix

    @staticmethod
    def _polygon_width(polygon: np.ndarray, x_position: float) -> float:
        intersections = []
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            x1, y1 = start
            x2, y2 = end
            if x1 == x2 or not (min(x1, x2) <= x_position < max(x1, x2)):
                continue
            fraction = (x_position - x1) / (x2 - x1)
            intersections.append(y1 + fraction * (y2 - y1))
        return float(max(intersections) - min(intersections)) if len(intersections) >= 2 else 0.0

    @staticmethod
    def _smooth_polygon(polygon: np.ndarray, strength: float) -> np.ndarray:
        result = polygon.copy()
        keep = max(1, int(len(polygon) * (1.0 - strength) / 2.0))
        for axis in (0, 1):
            spectrum = np.fft.rfft(polygon[:, axis])
            spectrum[keep:-keep] = 0
            result[:, axis] = np.fft.irfft(spectrum, n=len(polygon))
        return result

    def _empty_result(self, aligned: np.ndarray, alignment_matrix: np.ndarray) -> SliceVolumeResult:
        return SliceVolumeResult(
            volume_cm3=0.0, length_cm=0.0, max_width_cm=0.0, area_cm2=0.0,
            perimeter_cm=0.0, slice_widths_cm=[], slice_volumes_cm3=[],
            delta_x_cm=0.0, n_slices=0, aligned_mask=aligned,
            slice_positions_px=np.array([]), alignment_matrix=alignment_matrix,
            cross_section=type(self.cross_section).__name__,
            k=getattr(self.cross_section, "k", None),
            weight_kg_est=(self.weight_model.estimate_weight_kg(0.0)
                           if self.weight_model is not None else None),
        )
