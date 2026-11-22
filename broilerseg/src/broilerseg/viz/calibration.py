"""Plots that show how the feeder gives the pixel-to-centimetre scale."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse

from broilerseg.calibration import Calibration
from broilerseg.viz.style import class_color, crop_box, image_ax, new_ax


def _draw_geometry(ax, cal: Calibration, offset=(0, 0)) -> None:
    ox, oy = offset
    if cal.hull is not None:
        h = cal.hull.reshape(-1, 2)
        ax.plot(np.r_[h[:, 0], h[0, 0]] - ox, np.r_[h[:, 1], h[0, 1]] - oy, color="white",
                lw=1.2, ls="--", label="convex hull")
    d = cal.diameters_px
    cx, cy = cal.center
    if np.isfinite(d.get("enclosing_circle", np.nan)):
        ax.add_patch(Circle((cx - ox, cy - oy), d["enclosing_circle"] / 2, fill=False,
                            ec="cyan", lw=2, label="min enclosing circle"))
    if cal.ellipse is not None:
        (ex, ey), (a1, a2), ang = cal.ellipse
        ax.add_patch(Ellipse((ex - ox, ey - oy), a1, a2, angle=ang, fill=False, ec="magenta",
                             lw=1.5, ls="-.", label="fitted ellipse"))
    if np.isfinite(cal.diameter_px):
        r = cal.diameter_px / 2
        ax.annotate("", (cx - ox + r, cy - oy), (cx - ox - r, cy - oy),
                    arrowprops={"arrowstyle": "<->", "color": "yellow", "lw": 2})
        ax.text(cx - ox, cy - oy - 6, f"{cal.diameter_px:.1f} px = {cal.diameter_cm:.2f} cm",
                color="yellow", ha="center", va="bottom", fontsize=9, fontweight="bold",
                bbox={"fc": "black", "alpha": 0.5, "ec": "none"})


def plot_feeder_calibration(image: np.ndarray, cal: Calibration, feeder_mask: np.ndarray,
                            title: str = "Pixel-to-cm calibration from the feeder"):
    """Full frame + zoom showing the feeder mask, hull, circle, ellipse and diameter."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.4, 1, 0.8]})
    over = image.copy()
    if feeder_mask is not None and feeder_mask.any():
        col = np.array(class_color("feeder")) * 255
        over[feeder_mask] = (0.5 * over[feeder_mask] + 0.5 * col).astype(np.uint8)
    image_ax(axes[0], over, "Frame with feeder mask")
    _draw_geometry(axes[0], cal)
    if feeder_mask is not None and feeder_mask.any():
        x1, y1, x2, y2 = crop_box(feeder_mask, image.shape, pad=0.2)
        image_ax(axes[1], over[y1:y2, x1:x2], "Zoom on the feeder")
        _draw_geometry(axes[1], cal, (x1, y1))
        axes[1].legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.8)
    else:
        axes[1].text(0.5, 0.5, "no feeder found", ha="center", transform=axes[1].transAxes)
        axes[1].axis("off")
    ax = axes[2]
    names = list(cal.diameters_px)
    vals = [cal.diameters_px[n] for n in names]
    bars = ax.barh(names, vals, color=["cyan" if n == cal.method else "#bbbbbb" for n in names])
    for b, v in zip(bars, vals, strict=False):
        if np.isfinite(v):
            ax.text(v, b.get_y() + b.get_height() / 2,
                    f" {v:.1f}px -> {cal.diameter_cm / v:.4f} cm/px", va="center", fontsize=8)
    ax.set_xlabel("feeder diameter (px)")
    ax.set_title(f"Methods (used: {cal.method})")
    ax.set_xlim(0, np.nanmax(vals) * 1.8 if np.isfinite(np.nanmax(vals)) else 1)
    fig.suptitle(f"{title}   |   pixel_to_cm = {cal.pixel_to_cm:.5f} cm/px "
                 f"({1 / cal.pixel_to_cm if cal.ok else float('nan'):.2f} px/cm)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_calibration_stability(records: pd.DataFrame, value: str = "pixel_to_cm_frame",
                               ax=None, title: str = "Is the scale stable within each video?",
                               max_videos: int = 60):
    """Per-frame scale per video (dots), the median used (line) and CV % (<= ``max_videos``)."""
    all_vids = sorted(records.video_name.unique())
    vids = all_vids
    if len(all_vids) > max_videos:
        keep = np.unique(np.linspace(0, len(all_vids) - 1, max_videos).astype(int))
        vids = [all_vids[k] for k in keep]
        title += f" (showing {len(vids)} of {len(all_vids)} videos)"
    fig, ax = new_ax(ax, (max(6, 0.6 * len(vids) + 3), 4.5))
    rng = np.random.default_rng(0)
    for k, v in enumerate(vids):
        y = records.loc[records.video_name == v, value].to_numpy(float)
        y = y[np.isfinite(y)]
        if not len(y):
            ax.text(k, 0, "no feeder", rotation=90, fontsize=7, ha="center")
            continue
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(y)), y, s=14, alpha=0.6,
                   color=class_color("feeder"))
        med = np.median(y)
        ax.hlines(med, k - 0.3, k + 0.3, color="black", lw=2)
        ax.text(k, y.max(), f"CV {100 * y.std() / y.mean():.1f}%", ha="center", va="bottom",
                fontsize=7)
    ax.set_xticks(range(len(vids)))
    ax.set_xticklabels(vids, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("cm per pixel")
    ax.set_title(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_calibration_methods(calibrations, labels=None,
                             title: str = "Feeder diameter by method, frame by frame"):
    """Compare the three diameter methods across many calibrations."""
    df = pd.DataFrame([c.diameters_px for c in calibrations])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for col in df:
        axes[0].plot(df.index, df[col], marker="o", ms=3, lw=1, label=col)
    axes[0].set_xlabel("frame #")
    axes[0].set_ylabel("diameter (px)")
    axes[0].legend()
    axes[0].set_title("Per frame")
    if "enclosing_circle" in df:
        rel = df.div(df["enclosing_circle"], axis=0)
        axes[1].boxplot([rel[c].dropna() for c in rel], tick_labels=list(rel.columns))
        axes[1].axhline(1, color="gray", ls="--")
        axes[1].set_ylabel("ratio to enclosing circle")
        axes[1].set_title("Agreement between methods")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
