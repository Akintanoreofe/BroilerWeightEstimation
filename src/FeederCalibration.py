"""Compatibility wrapper for feeder-based video scale calibration."""

from FeederReference import FeederReference
from defaults import FEEDER_DIAMETER_CM


def calibrate_pixel_to_cm(video_path: str) -> float:
    """Return centimeters per pixel from a robust multi-frame feeder fit.

    Parameters
    ----------
    video_path : str
        Path or source accepted by :class:`VideoReader`.

    Returns
    -------
    float
        Physical centimeters represented by one pixel.

    """
    try:
        reference = FeederReference.from_video(video_path)
    except RuntimeError as error:
        if "good fits" in str(error):
            raise RuntimeError(
                f"No orange feeder detected with five reliable fits in: {video_path}"
            ) from error
        raise
    return 1.0 / reference.px_per_cm
