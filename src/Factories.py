"""Factories and enumerations for pipeline strategy selection."""

from enum import Enum
from SegmentationStrategy import Segmenter, ClassicalSegmenter, DeepLearningSegmenter
from AreaCalculatorStrategy import AreaCalculator, PixelAreaCalculator, CalibratedAreaCalculator


class SegmentationMethod(Enum):
    """Available segmentation implementations."""

    DEEP = "deep"
    CLASSICAL = "classical"


class AreaMethod(Enum):
    """Available area and volume calculation implementations."""

    PIXEL = "pixel"
    CALIBRATED = "calibrated"


class SegmenterFactory:
    """Construct segmenters selected by :class:`SegmentationMethod`."""

    def create(self, method: SegmentationMethod, **kwargs) -> Segmenter:
        """Create a segmenter for the requested method.

        Parameters
        ----------
        method : SegmentationMethod
            Segmentation implementation to construct.
        **kwargs
            Optional constructor values. ``model_path`` is used by the deep
            segmenter and defaults to ``"model.onnx"``.

        Returns
        -------
        Segmenter
            A deep-learning or classical segmenter.

        Raises
        ------
        ValueError
            If ``method`` is not a supported segmentation method.
        """
        if method == SegmentationMethod.DEEP:
            return DeepLearningSegmenter(model_path=kwargs.get("model_path", "model.onnx"))
        elif method == SegmentationMethod.CLASSICAL:
            return ClassicalSegmenter()
        else:
            raise ValueError(f"Unknown segmentation method: {method}")


class AreaCalculatorFactory:
    """Construct volume calculators selected by :class:`AreaMethod`."""

    def create(self, method: AreaMethod, **kwargs) -> AreaCalculator:
        """Create a calculator for the requested method.

        Parameters
        ----------
        method : AreaMethod
            Calculator implementation to construct.
        **kwargs
            Optional constructor values. ``pixel_to_cm`` is required for the
            calibrated calculator.

        Returns
        -------
        AreaCalculator
            A pixel-based or calibrated calculator.

        Raises
        ------
        ValueError
            If the method is unsupported or calibrated calculation is
            requested without a nonzero ``pixel_to_cm`` value.
        """
        if method == AreaMethod.PIXEL:
            return PixelAreaCalculator()
        elif method == AreaMethod.CALIBRATED:
            pixel_to_cm = kwargs.get("pixel_to_cm")
            if pixel_to_cm == 0 or pixel_to_cm is None:
                raise ValueError("pixel_to_cm is required for CALIBRATED area method")
            return CalibratedAreaCalculator(pixel_to_cm=pixel_to_cm)
        else:
            raise ValueError(f"Unknown area method: {method}")