"""Pixel-to-centimetre calibration from the feeder (feeding pan) of known diameter.

The feeder is a 14-inch (35.56 cm) circle. Seen from an overhead camera it is a
circle or, if the camera is tilted, an ellipse whose major axis is still the
true diameter. A bird may hide part of it, so every method works on the convex
hull of the feeder mask, as the repository's ``FeederCalibration.py`` does.

Methods
-------
``"enclosing_circle"``
    Diameter of the minimum enclosing circle of the hull (repository method).
``"ellipse_major"``
    Major axis of an ellipse fitted to the hull (robust to camera tilt).
``"equivalent_area"``
    Diameter of the circle with the same area as the hull (biased low when
    the feeder is occluded; shown for comparison).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from broilerseg.config import FEEDER_DIAMETER_CM

METHODS = ("enclosing_circle", "ellipse_major", "equivalent_area")


@dataclass
class Calibration:
    """Result of one feeder measurement.

    Attributes
    ----------
    pixel_to_cm : float
        Centimetres represented by one pixel (``nan`` when it failed).
    diameter_px : float
        Feeder diameter in pixels from the chosen ``method``.
    method : str
        Which measurement produced ``pixel_to_cm``.
    center : tuple of float
        Feeder centre ``(x, y)`` in pixels.
    diameters_px : dict
        Diameter from every method, for comparison plots.
    ellipse : tuple or None
        OpenCV ellipse ``((cx, cy), (axis1, axis2), angle)``.
    hull : numpy.ndarray or None
        Convex hull points of the feeder mask.
    """

    pixel_to_cm: float
    diameter_px: float
    method: str
    center: tuple[float, float] = (np.nan, np.nan)
    diameters_px: dict[str, float] = field(default_factory=dict)
    ellipse: tuple | None = None
    hull: np.ndarray | None = None
    diameter_cm: float = FEEDER_DIAMETER_CM

    @property
    def ok(self) -> bool:
        """True when a finite, positive scale was measured."""
        return bool(np.isfinite(self.pixel_to_cm) and self.pixel_to_cm > 0)


def _failed(method: str) -> Calibration:
    return Calibration(np.nan, np.nan, method)


def calibrate_from_mask(feeder_mask: np.ndarray, diameter_cm: float = FEEDER_DIAMETER_CM,
                        method: str = "enclosing_circle",
                        min_diameter_px: float = 10.0) -> Calibration:
    """Measure the feeder in a boolean mask and return the pixel scale.

    Parameters
    ----------
    feeder_mask : numpy.ndarray
        Boolean mask of the feeder (e.g. from SAM 3 or YOLO).
    diameter_cm : float
        Real feeder diameter; defaults to 35.56 cm (14 in).
    method : str
        One of :data:`METHODS`.
    min_diameter_px : float
        Reject measurements smaller than this as spurious.

    Returns
    -------
    Calibration
        ``pixel_to_cm`` is ``nan`` if no usable feeder was found.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if feeder_mask is None or not np.asarray(feeder_mask).any():
        return _failed(method)
    pts = cv2.findNonZero(np.asarray(feeder_mask).astype(np.uint8))
    hull = cv2.convexHull(pts)
    (cx, cy), radius = cv2.minEnclosingCircle(hull)
    diameters = {"enclosing_circle": 2.0 * radius,
                 "equivalent_area": 2.0 * np.sqrt(cv2.contourArea(hull) / np.pi)}
    ellipse = None
    if len(hull) >= 5:
        ellipse = cv2.fitEllipse(hull)
        diameters["ellipse_major"] = float(max(ellipse[1]))
    else:
        diameters["ellipse_major"] = np.nan
    d = float(diameters[method])
    if not np.isfinite(d) or d < min_diameter_px:
        out = _failed(method)
        out.diameters_px, out.hull = diameters, hull
        return out
    return Calibration(diameter_cm / d, d, method, (float(cx), float(cy)), diameters,
                       ellipse, hull, diameter_cm)


def calibrate_from_hsv(frame_bgr: np.ndarray, diameter_cm: float = FEEDER_DIAMETER_CM,
                       method: str = "enclosing_circle") -> Calibration:
    """Repository fallback: find the orange feeder by colour, then measure it.

    Uses the same HSV range and morphology as ``src/FeederCalibration.py``.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((h > 8) & (h < 20) & (s > 150) & (v > 100)).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return calibrate_from_mask(mask.astype(bool), diameter_cm, method)


def summarize_calibrations(records: pd.DataFrame, group: str = "video_name",
                           value: str = "pixel_to_cm") -> pd.DataFrame:
    """Per-video robust scale: median, spread and coefficient of variation.

    The camera does not move during a clip, so the median over frames is a
    steadier scale than any single frame (the repository uses frame one only).
    """
    ok = records[np.isfinite(records[value])]
    g = ok.groupby(group)[value]
    out = pd.DataFrame({
        "pixel_to_cm_median": g.median(),
        "pixel_to_cm_mean": g.mean(),
        "pixel_to_cm_std": g.std(ddof=0),
        "n_frames": g.size(),
    })
    out["cv_percent"] = 100 * out["pixel_to_cm_std"] / out["pixel_to_cm_mean"]
    return out.reset_index()
