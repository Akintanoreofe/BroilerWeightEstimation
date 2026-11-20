"""Slice-based (stacked cylinder) volume and mass estimation.

This reproduces the method of the BroilerWeightEstimation repository
(``AreaCalculatorStrategy.CalibratedAreaCalculator`` and
``ObjectFilterStrategy.SimpleObjectFilter.align_mask_by_coords``):

1. Rotate the bird mask so its body axis is horizontal.
2. Split the occupied columns into ``slice_count`` groups (``None`` = one
   slice per column). Each group is a cylinder whose diameter is the mask's
   vertical span at the group's middle column and whose length is the group
   width.
3. ``V = pi * sum(r_cm^2 * width_px) * pixel_to_cm`` (cm^3), and
   ``mass_kg = V / 1e6 * density`` with density in kg / m^3.

The cross-sections are assumed circular; depth is not measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from broilerseg.config import DEFAULT_DENSITY_KG_M3


# --------------------------------------------------------------------- alignment
def object_angle(mask: np.ndarray) -> float:
    """Dominant contour orientation in degrees (repository ``object_angle``)."""
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    threshed = cv2.threshold(m, 0, 255, cv2.THRESH_OTSU)[1]
    threshed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, np.ones((1, 10), np.uint8))
    contours = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    angles = []
    for c in contours:
        if len(c) < 2:
            continue
        vx, vy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)[:2, 0]
        angles.append(math.degrees(math.atan2(float(vy), float(vx))))
    if not angles:
        return 0.0
    angles = np.array(angles)
    lo, hi = np.percentile(angles, (40, 60))
    middle = angles[(angles >= lo) & (angles <= hi)]
    return float(np.mean(middle if len(middle) else angles))


def align_mask(mask: np.ndarray, padding: int = 2) -> tuple[np.ndarray, float]:
    """Rotate mask coordinates so the body axis is horizontal.

    Returns the filled, tightly cropped aligned mask (``uint8`` 0/255) and the
    rotation angle in degrees.
    """
    mask_bool = np.asarray(mask) > 0
    if not mask_bool.any():
        return np.zeros((1, 1), np.uint8), 0.0
    angle = object_angle(mask_bool)
    ys, xs = np.nonzero(mask_bool)
    theta = np.deg2rad(-angle)
    c, s = np.cos(theta), np.sin(theta)
    x0, y0 = xs - xs.mean(), ys - ys.mean()
    xr, yr = c * x0 - s * y0, s * x0 + c * y0
    xn = np.rint(xr - xr.min()).astype(int) + padding
    yn = np.rint(yr - yr.min()).astype(int) + padding
    rotated = np.zeros((yn.max() + 1 + padding, xn.max() + 1 + padding), np.uint8)
    rotated[yn, xn] = 255
    contours, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(rotated)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled, angle


# --------------------------------------------------------------------- slicing
def mask_slice_profile(mask: np.ndarray, slice_count: int | None = None) -> dict:
    """Column groups and sampled diameters used by the volume model.

    Identical to the repository's ``mask_slice_profile``. Returns arrays
    ``x_start``, ``x_end``, ``sample_x``, ``y_min``, ``y_max`` and
    ``thickness_px`` (one entry per slice).
    """
    if slice_count is not None and (isinstance(slice_count, bool)
                                    or not isinstance(slice_count, int | np.integer)
                                    or slice_count < 1):
        raise ValueError("slice_count must be a positive integer or None")
    mask_bool = np.asarray(mask) > 0
    columns = np.flatnonzero(mask_bool.any(axis=0))
    keys = ("x_start", "x_end", "sample_x", "y_min", "y_max", "thickness_px")
    if not len(columns):
        return {k: np.array([], dtype=int) for k in keys}
    valid = mask_bool[:, columns]
    top = valid.argmax(axis=0)
    bottom = mask_bool.shape[0] - 1 - np.flipud(valid).argmax(axis=0)
    count = len(columns) if slice_count is None else min(int(slice_count), len(columns))
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


@dataclass
class VolumeEstimate:
    """Everything needed to inspect one volume / mass estimate.

    Attributes
    ----------
    volume_cm3 : float
        Sum of cylinder volumes.
    mass_kg : float
        ``volume_cm3 / 1e6 * density_kg_m3``.
    slice_volumes_cm3 : numpy.ndarray
        Volume of each cylinder.
    diameters_cm, thickness_cm : numpy.ndarray
        Per-slice geometry in centimetres.
    """

    volume_cm3: float
    mass_kg: float
    pixel_to_cm: float
    density_kg_m3: float
    slice_count: int | None
    aligned_mask: np.ndarray
    angle_deg: float
    profile: dict = field(repr=False)
    slice_volumes_cm3: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    diameters_cm: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    thickness_cm: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    @property
    def n_slices(self) -> int:
        """Number of cylinders actually used."""
        return len(self.slice_volumes_cm3)

    @property
    def length_cm(self) -> float:
        """Body length along the aligned axis."""
        return float(self.thickness_cm.sum())

    @property
    def max_width_cm(self) -> float:
        """Largest slice diameter."""
        return float(self.diameters_cm.max()) if len(self.diameters_cm) else 0.0

    @property
    def area_cm2(self) -> float:
        """Silhouette area of the aligned mask."""
        return float((self.aligned_mask > 0).sum() * self.pixel_to_cm ** 2)

    def summary(self) -> dict:
        """Flat dictionary for tables."""
        return {"volume_cm3": self.volume_cm3, "mass_kg": self.mass_kg,
                "n_slices": self.n_slices, "length_cm": self.length_cm,
                "max_width_cm": self.max_width_cm, "area_cm2": self.area_cm2,
                "angle_deg": self.angle_deg}


def estimate_volume(mask: np.ndarray, pixel_to_cm: float,
                    density_kg_m3: float = DEFAULT_DENSITY_KG_M3,
                    slice_count: int | None = None, align: bool = True) -> VolumeEstimate:
    """Estimate volume and mass of one bird mask.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean or 0/255 bird mask (full frame or crop).
    pixel_to_cm : float
        Centimetres per pixel from :mod:`broilerseg.calibration`.
    density_kg_m3 : float
        Broiler density.
    slice_count : int or None
        Number of cylinders; ``None`` = one per occupied column.
    align : bool
        Rotate to the body axis first (repository behaviour).
    """
    if not math.isfinite(pixel_to_cm) or pixel_to_cm <= 0:
        raise ValueError("pixel_to_cm must be positive and finite")
    if not math.isfinite(density_kg_m3) or density_kg_m3 <= 0:
        raise ValueError("density_kg_m3 must be positive and finite")
    if align:
        aligned, angle = align_mask(mask)
    else:
        aligned, angle = (np.asarray(mask) > 0).astype(np.uint8) * 255, 0.0
    profile = mask_slice_profile(aligned, slice_count)
    diam_px = (profile["y_max"] - profile["y_min"] + 1).astype(np.float64)
    if len(profile["thickness_px"]) == 0:
        diam_px = np.array([])
    diam_cm = diam_px * pixel_to_cm
    thick_cm = profile["thickness_px"].astype(np.float64) * pixel_to_cm
    slice_vol = np.pi * (diam_cm / 2.0) ** 2 * thick_cm
    volume = float(slice_vol.sum())
    return VolumeEstimate(volume, mass_from_volume(volume, density_kg_m3), pixel_to_cm,
                          density_kg_m3, slice_count, aligned, angle, profile, slice_vol,
                          diam_cm, thick_cm)


def mass_from_volume(volume_cm3, density_kg_m3=DEFAULT_DENSITY_KG_M3):
    """Convert cm^3 to kg with density in kg / m^3 (works on arrays)."""
    mass = np.asarray(volume_cm3, dtype=float) / 1_000_000 * density_kg_m3
    return float(mass) if mass.ndim == 0 else mass


def implied_density(actual_kg, volume_cm3):
    """Density (kg / m^3) that would make the volume match the scale reading."""
    return np.asarray(actual_kg, float) / (np.asarray(volume_cm3, float) / 1_000_000)


def volume_by_slice_count(mask: np.ndarray, pixel_to_cm: float,
                          slice_counts=(None, 5, 10, 15, 20, 30, 40, 60),
                          align: bool = True) -> pd.DataFrame:
    """Volume of one mask for several slice counts (``None`` = per column).

    Mass is linear in density, so a density sweep only needs these volumes.
    """
    rows = []
    for k in slice_counts:
        est = estimate_volume(mask, pixel_to_cm, slice_count=k, align=align)
        rows.append({"slice_count": "per-column" if k is None else int(k),
                     "n_slices": est.n_slices, "volume_cm3": est.volume_cm3})
    return pd.DataFrame(rows)
