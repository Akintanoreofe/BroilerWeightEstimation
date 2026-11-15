"""Strategies for estimating broiler volume from segmented masks.

The calculators implement a common interface for deriving volume estimates
from :class:`ObjectInfo` instances.
"""


from abc import ABC, abstractmethod
import math
import numpy as np
from ObjectFilterStrategy import ObjectInfo


def mask_slice_profile(mask: np.ndarray, slice_count=None):
    """Return column groups and sampled diameters used by the volume model."""
    if slice_count is not None and (isinstance(slice_count, bool) or not isinstance(slice_count, int) or slice_count < 1):
        raise ValueError("slice_count must be a positive integer or None")
    mask_bool = mask > 0
    columns = np.flatnonzero(mask_bool.any(axis=0))
    if not len(columns):
        return {key: np.array([], dtype=int) for key in
                ("x_start", "x_end", "sample_x", "y_min", "y_max", "thickness_px")}
    valid = mask_bool[:, columns]
    top = valid.argmax(axis=0)
    bottom = mask_bool.shape[0] - 1 - np.flipud(valid).argmax(axis=0)
    count = len(columns) if slice_count is None else min(slice_count, len(columns))
    edges = np.linspace(0, len(columns), count + 1, dtype=int)
    samples = (edges[:-1] + edges[1:] - 1) // 2
    return {
        "x_start": columns[edges[:-1]],
        "x_end": columns[edges[1:] - 1],
        "sample_x": columns[samples],
        "y_min": top[samples],
        "y_max": bottom[samples],
        "thickness_px": np.diff(edges),
    }


class AreaCalculator(ABC):
    """Abstract interface for mask-based volume calculators."""

    @abstractmethod
    def volume_from_mask(self, obj: ObjectInfo) -> ObjectInfo:
        """Estimate volume from an object's mask.

        Parameters
        ----------
        obj : ObjectInfo
            Object whose mask is used for the estimate.

        Returns
        -------
        ObjectInfo
            The object with its volume estimate updated.
        """
        pass


class PixelAreaCalculator(AreaCalculator):
    """Placeholder calculator for pixel-based volume estimation."""

    def volume_from_mask(self, obj: ObjectInfo) -> ObjectInfo:
        """Estimate volume from a mask.

        Parameters
        ----------
        obj : ObjectInfo
            Object whose mask would be used for the estimate.

        Raises
        ------
        NotImplementedError
            Always, because pixel-based calculation is not implemented.
        """
        raise NotImplementedError("PixelAreaCalculator is not yet implemented.")


class CalibratedAreaCalculator(AreaCalculator):
    """Estimate volume using a pixel-to-centimeter scale and density."""

    def __init__(self, pixel_to_cm: float, broiler_density: float = 900.0, slice_count=None):
        """Initialize the calibrated volume calculator.

        Parameters
        ----------
        pixel_to_cm : float
            Physical length in centimeters represented by one pixel.
        broiler_density : float, default=900.0
            Assumed broiler density in kilograms per cubic meter.
        slice_count : int or None, default=None
            Number of equally spaced groups of occupied columns, sampled at
            each group's middle column. None uses every occupied column.
        """
        if not math.isfinite(pixel_to_cm) or pixel_to_cm <= 0:
            raise ValueError("pixel_to_cm must be positive and finite")
        if not math.isfinite(broiler_density) or broiler_density <= 0:
            raise ValueError("broiler_density must be positive and finite")
        if slice_count is not None and (isinstance(slice_count, bool) or not isinstance(slice_count, int) or slice_count < 1):
            raise ValueError("slice_count must be a positive integer or None")
        self.pixel_to_cm = pixel_to_cm
        self.density = broiler_density
        self.slice_count = slice_count

    def volume_from_mask(self, obj: ObjectInfo) -> ObjectInfo:
        """Estimate volume by summing circular slices along the mask width.

        Parameters
        ----------
        obj : ObjectInfo
            Object whose binary mask is used for the estimate.

        Returns
        -------
        ObjectInfo
            The same object, with ``volume_cm3`` and ``weight_kg`` updated.
            An empty mask produces a volume of ``0.0``.
        """

        mask = obj.mask
        mask_bool = mask > 0

        if not mask_bool.any():
            obj.volume_cm3 = 0.0
            obj.weight_kg = 0.0
            return obj

        profile = mask_slice_profile(mask, self.slice_count)
        slice_widths = profile["thickness_px"]
        diameters = (profile["y_max"] - profile["y_min"] + 1).astype(np.float32)
        radius = diameters * self.pixel_to_cm / 2.0
        # Group widths are in pixels; the calibrated sum is in cm³.
        V = float(np.pi * np.sum(radius ** 2 * slice_widths) * self.pixel_to_cm)

        # Convert cm³ to m³ before multiplying by density (kg/m³) to obtain mass in kg.
        obj.volume_cm3 = V
        obj.weight_kg = (V / 1_000_000) * self.density

        return obj
