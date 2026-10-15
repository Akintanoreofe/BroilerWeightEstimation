"""Strategies for estimating broiler volume from segmented masks.

The calculators implement a common interface for deriving volume estimates
from :class:`ObjectInfo` instances.
"""


from abc import ABC, abstractmethod
import numpy as np
from ObjectFilterStrategy import ObjectInfo


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

    def __init__(self, pixel_to_cm: float, broiler_density: float = 900.0):  # density kg/m3
        """Initialize the calibrated volume calculator.

        Parameters
        ----------
        pixel_to_cm : float
            Physical length in centimeters represented by one pixel.
        broiler_density : float, default=900.0
            Assumed broiler density in kilograms per cubic meter.
        """
        self.pixel_to_cm = pixel_to_cm
        self.density = broiler_density

    def volume_from_mask(self, obj: ObjectInfo) -> ObjectInfo:
        """Estimate volume by summing circular slices along the mask width.

        Parameters
        ----------
        obj : ObjectInfo
            Object whose binary mask is used for the estimate.

        Returns
        -------
        ObjectInfo
            The same object, with ``volume`` set to the estimated mass in kg.
            An empty mask produces a volume of ``0.0``.
        """

        mask = obj.mask
        mask_bool = mask > 0

        if not mask_bool.any():
            obj.volume = 0.0
            return obj

        H, W = mask_bool.shape

        col_has_obj = mask_bool.any(axis=0)
        if not col_has_obj.any():
            obj.volume = 0.0
            return obj

        mask_valid_cols = mask_bool[:, col_has_obj]

        y_min = mask_valid_cols.argmax(axis=0)

        flipped = np.flipud(mask_valid_cols)
        y_max_from_bottom = flipped.argmax(axis=0)
        y_max = (H - 1) - y_max_from_bottom
        width = (y_max - y_min + 1).astype(np.float32)
        radius = width * self.pixel_to_cm / 2.0
        V = float(np.pi * np.sum(radius ** 2) * self.pixel_to_cm)

        obj.volume = (V / 1_000_000) * self.density  # convert to cubic meters

        return obj
