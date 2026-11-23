"""Plots for stage 1 of two-stage labelling: the pen region and its crop."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle

from broilerseg.segmentation.base import SegmentationResult
from broilerseg.viz.overlays import blend_masks, plot_overlay
from broilerseg.viz.style import image_ax, thumbnail

ROI_COLOR = (0.10, 0.85, 0.95)
METHOD_COLORS = {"auto": "tab:green", "auto_folder": "tab:olive", "manual": "tab:blue",
                 "folder_median": "tab:orange", "full_frame": "tab:red", "failed": "tab:red",
                 "typical": "tab:purple"}


def _draw_roi(ax, roi, shape, scale: float = 1.0, label: str | None = None) -> None:
    x0, y0, x1, y1 = (v * scale for v in roi.pixels(shape))
    col = METHOD_COLORS.get(roi.method, ROI_COLOR)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=2.2))
    if len(roi.polygon) >= 3:
        ax.add_patch(Polygon(roi.polygon_pixels(shape) * scale, closed=True, fill=False,
                             ec=ROI_COLOR, lw=1.3, ls="--"))
    if label:
        ax.text(x0 + 4, y0 + 4, label, va="top", fontsize=8, color="black",
                bbox={"fc": col, "ec": "none", "alpha": 0.85, "pad": 1.5})


def plot_pen_rois(images, rois, titles=None, ncols: int = 5, size: float = 3.4,
                  title: str = "Stage 1: pen region found in each video", max_items: int = 30):
    """Frames with the pen box (solid, coloured by method) and outline (dashed)."""
    items = list(zip(images, rois, titles or [""] * len(images), strict=False))[:max_items]
    n = max(1, len(items))
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols * 16 / 10, size * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (img, roi, t) in zip(axes.ravel(), items, strict=False):
        small, s = thumbnail(img, 720)
        image_ax(ax, small, t)
        ax.title.set_fontsize(8)
        _draw_roi(ax, roi, img.shape, s,
                  f"{roi.method} {roi.score:.2f} | {100 * roi.area_fraction:.0f}% of frame")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_pen_crops(images, rois, titles=None, mask_outside: bool = False, ncols: int = 5,
                   size: float = 3.4, title: str = "What the teacher sees in stage 2 (pen crops)",
                   max_items: int = 30):
    """Show the crops themselves, with their size in pixels."""
    items = list(zip(images, rois, titles or [""] * len(images), strict=False))[:max_items]
    n = max(1, len(items))
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (img, roi, t) in zip(axes.ravel(), items, strict=False):
        crop = roi.crop(img, mask_outside)
        image_ax(ax, thumbnail(crop, 640)[0], f"{t}\n{crop.shape[1]} x {crop.shape[0]} px")
        ax.title.set_fontsize(8)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_pen_candidates(image: np.ndarray, result: SegmentationResult, chosen=None,
                        title: str = "Stage 1 candidates for the pen"):
    """Every pen candidate the prompts returned, with prompt and score."""
    cands = result.of_class("pen")
    n = max(1, min(len(cands), 6))
    fig, axes = plt.subplots(1, n + 1, figsize=(4.6 * (n + 1), 3.6), squeeze=False)
    small, s = thumbnail(image, 720)
    image_ax(axes[0, 0], small, "frame")
    for ax, inst in zip(axes[0, 1:], cands[:6], strict=False):
        m = cv2.resize(inst.mask.astype(np.uint8), (small.shape[1], small.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
        show = small.astype(np.float32).copy()
        show[m] = 0.5 * show[m] + 0.5 * np.array(ROI_COLOR) * 255
        tag = " <- chosen" if chosen is not None and inst is chosen else ""
        frac = inst.area / (image.shape[0] * image.shape[1])
        image_ax(ax, show.astype(np.uint8),
                 f'"{inst.prompt}" {inst.score:.2f}, {100 * frac:.0f}%{tag}')
        ax.title.set_fontsize(9)
    if not cands:
        axes[0, 1].text(0.5, 0.5, "no pen found\n(try other PEN_PROMPTS\nor a manual box)",
                        ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 1].axis("off")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_roi_summary(rois: pd.DataFrame, title: str = "Stage 1 summary: pen regions",
                     max_boxes: int = 400):
    """How each region was found, how big the crops are, and where they sit."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    counts = rois.method.value_counts()
    axes[0].bar(counts.index, counts.values,
                color=[METHOD_COLORS.get(m, "grey") for m in counts.index])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom")
    axes[0].set(title="How the region was found", ylabel="videos")
    axes[0].tick_params(axis="x", rotation=20)

    ax = axes[1]
    for m, g in rois.groupby("method"):
        ax.scatter(100 * (g.fx1 - g.fx0), 100 * (g.fy1 - g.fy0), s=18, alpha=0.7,
                   color=METHOD_COLORS.get(m, "grey"), label=m)
    ax.set(xlim=(0, 105), ylim=(0, 105), xlabel="crop width (% of frame)",
           ylabel="crop height (% of frame)", title="Crop size")
    ax.legend(fontsize=8)
    if "crop_width" in rois and rois.crop_width.max() > 0:
        ax.text(0.02, 0.02, f"median crop {int(rois.crop_width.median())} x "
                f"{int(rois.crop_height.median())} px", transform=ax.transAxes, fontsize=9)

    ax = axes[2]
    for r in rois.head(max_boxes).itertuples():
        ax.add_patch(Rectangle((r.fx0, r.fy0), r.fx1 - r.fx0, r.fy1 - r.fy0, fill=False,
                               ec=METHOD_COLORS.get(r.method, "grey"), alpha=0.35, lw=1))
    ax.set(xlim=(0, 1), ylim=(1, 0), aspect=16 / 9 / (16 / 9), title="Where the crops sit",
           xlabel="x (fraction of width)", ylabel="y (fraction of height)")
    ax.set_aspect(9 / 16)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_two_stage(image: np.ndarray, full_result: SegmentationResult | None, roi,
                   crop: np.ndarray, crop_result: SegmentationResult,
                   title: str = "One stage (whole frame) vs two stages (pen crop)"):
    """Whole-frame teacher output, the pen region, and the teacher on the crop."""
    n = 3 if full_result is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(7.2 * n, 5.4))
    k = 0
    if full_result is not None:
        small, s = thumbnail(blend_masks(image, full_result), 1280)
        image_ax(axes[0], small, f"1 stage: whole frame  {full_result.counts()}")
        k = 1
    small, s = thumbnail(image, 1280)
    image_ax(axes[k], small, "stage 1: pen region")
    _draw_roi(axes[k], roi, image.shape, s, f"{roi.method} {roi.score:.2f}")
    plot_overlay(crop, crop_result, ax=axes[k + 1], show_boxes=False,
                 title=f"stage 2: pen crop  {crop_result.counts()}")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig
