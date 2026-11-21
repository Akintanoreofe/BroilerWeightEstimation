"""Shared plotting style, class colours and small helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from broilerseg.config import CLASS_COLORS, CLASS_NAMES

#: Colours for non-class series (teacher / student / scale reading ...).
SERIES = {"teacher": "#6a3d9a", "student": "#1b9e77", "actual": "#333333",
          "pred": "#d95f02", "good": "#1a9850", "bad": "#d73027", "neutral": "#7f7f7f"}


def set_style(dpi: int = 110) -> None:
    """Apply the package's matplotlib style (call once per notebook)."""
    mpl.rcParams.update({
        "figure.dpi": dpi, "savefig.dpi": 150, "figure.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 9, "legend.frameon": False,
        "image.interpolation": "nearest",
    })


def class_color(name: str) -> tuple[float, float, float]:
    """RGB colour (0-1) of a class."""
    return CLASS_COLORS.get(name, (0.5, 0.5, 0.5))


def class_legend(ax, classes=CLASS_NAMES, loc: str = "lower right") -> None:
    """Add a colour legend for the segmentation classes."""
    ax.legend(handles=[Patch(color=class_color(c), label=c) for c in classes], loc=loc,
              frameon=True, framealpha=0.85)


def new_ax(ax=None, figsize=(6, 4)):
    """Return ``(fig, ax)``, creating a figure if ``ax`` is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax


def image_ax(ax, image, title: str | None = None) -> None:
    """Show an image without axes decoration."""
    if np.ndim(image) == 2:
        ax.imshow(image, cmap="gray", vmin=0, vmax=max(1, float(np.max(image))))
    else:
        ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title)


def save(fig, path: str | Path | None) -> None:
    """Save a figure if ``path`` is given (parent folders are created)."""
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")


def thumbnail(image: np.ndarray, max_side: int = 640) -> tuple[np.ndarray, float]:
    """Downscale an image for display; returns the image and the scale used."""
    h, w = image.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s == 1.0:
        return image, 1.0
    return cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), s


def crop_box(mask_or_box, shape, pad: float = 0.25) -> tuple[int, int, int, int]:
    """Return a padded ``(x1, y1, x2, y2)`` crop around a mask or box."""
    if isinstance(mask_or_box, np.ndarray) and mask_or_box.ndim == 2:
        ys, xs = np.nonzero(mask_or_box)
        if len(xs) == 0:
            return 0, 0, shape[1], shape[0]
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    else:
        x1, y1, x2, y2 = mask_or_box
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    return (int(max(0, x1 - pw)), int(max(0, y1 - ph)), int(min(shape[1], x2 + pw)),
            int(min(shape[0], y2 + ph)))


# --------------------------------------------------------------------- weight unit
#: Weight columns that hold kilograms in package tables.
WEIGHT_COLUMNS = ("actual_kg", "pred_kg", "event_weight_kg", "error_kg", "pred_std_kg",
                  "weight_kg", "min_weight_kg", "max_weight_kg")
_WEIGHT = {"unit": "kg", "factor": 1.0, "decimals": 3}


def set_weight_unit(unit: str = "kg") -> None:
    """Show weights in plots and tables as ``"kg"`` (default) or ``"g"``.

    Only the display changes: calculations, saved CSVs and density
    (kg / m^3) stay in kilograms.
    """
    if unit not in ("kg", "g"):
        raise ValueError("unit must be 'kg' or 'g'")
    _WEIGHT.update(unit=unit, factor=1.0 if unit == "kg" else 1000.0,
                   decimals=3 if unit == "kg" else 0)


def weight_unit() -> str:
    """Return the current display unit (``"kg"`` or ``"g"``)."""
    return _WEIGHT["unit"]


def weight_factor() -> float:
    """Return the multiplier from kilograms to the display unit."""
    return _WEIGHT["factor"]


def fmt_weight(kg: float, signed: bool = False) -> str:
    """Format a weight given in kg, e.g. ``"2.345 kg"`` or ``"2345 g"``."""
    sign = "+" if signed else ""
    return f"{kg * _WEIGHT['factor']:{sign}.{_WEIGHT['decimals']}f} {_WEIGHT['unit']}"


def to_display(df, columns=WEIGHT_COLUMNS):
    """Copy of ``df`` with weight columns converted to the display unit (same names)."""
    out = df.copy()
    for c in columns:
        if c in out:
            out[c] = out[c] * _WEIGHT["factor"]
    return out


def weight_table(df):
    """Copy of ``df`` for display: weight columns converted and renamed ``*_kg`` -> ``*_g``."""
    out = to_display(df)
    if _WEIGHT["unit"] == "kg":
        return out
    ren = {c: c[:-3] + "_g" for c in out.columns
           if c in WEIGHT_COLUMNS or c.endswith(("_mae_kg", "_rmse_kg", "_bias_kg"))}
    for c in out.columns:
        if c.endswith(("_mae_kg", "_rmse_kg", "_bias_kg")) and c not in WEIGHT_COLUMNS:
            out[c] = out[c] * _WEIGHT["factor"]
    return out.rename(columns=ren)
