"""Mask overlays, per-class panels, galleries and bird-selection diagnostics."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from broilerseg.config import CLASS_NAMES
from broilerseg.segmentation.base import Instance, SegmentationResult
from broilerseg.segmentation.postprocess import platform_region
from broilerseg.viz.style import class_color, class_legend, crop_box, image_ax, new_ax

_ORDER = ("scale", "feeder", "broiler")


def blend_masks(image: np.ndarray, result: SegmentationResult, alpha: float = 0.45,
                classes=_ORDER, contours: bool = True) -> np.ndarray:
    """Return an RGB image with class-coloured masks and outlines blended in."""
    out = image.astype(np.float32).copy()
    for name in classes:
        col = np.array(class_color(name)) * 255
        for inst in result.of_class(name):
            out[inst.mask] = (1 - alpha) * out[inst.mask] + alpha * col
    out = np.clip(out, 0, 255).astype(np.uint8)
    if contours:
        thick = max(2, round(max(image.shape[:2]) / 400))
        for name in classes:
            col = tuple(int(255 * c) for c in class_color(name))
            for inst in result.of_class(name):
                cnts, _ = cv2.findContours(inst.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, cnts, -1, col, thick)
    return out


def plot_overlay(image: np.ndarray, result: SegmentationResult, ax=None, title: str | None = None,
                 show_boxes: bool = True, show_scores: bool = True, legend: bool = True,
                 alpha: float = 0.45):
    """Image with every instance coloured by class, boxes and confidence labels."""
    fig, ax = new_ax(ax, (8, 6))
    image_ax(ax, blend_masks(image, result, alpha), title)
    for inst in result.instances:
        x, y, w, h = inst.box_xywh
        col = class_color(inst.class_name)
        if show_boxes:
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=col, lw=1.2, ls="--"))
        if show_scores:
            ax.text(x, y - 4, f"{inst.class_name} {inst.score:.2f}", color="black", fontsize=8,
                    bbox={"fc": col, "ec": "none", "alpha": 0.85, "pad": 1.5})
    if legend:
        class_legend(ax)
    return fig


def plot_segmentation_panels(image: np.ndarray, result: SegmentationResult,
                             title: str | None = None):
    """Six panels: original, one per class, combined overlay and label map."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    image_ax(axes[0], image, "Original frame")
    for k, name in enumerate(CLASS_NAMES, start=1):
        found = result.of_class(name)
        union = result.union(name)
        panel = (image * 0.35).astype(np.uint8)
        panel[union] = (np.array(class_color(name)) * 255).astype(np.uint8)
        scores = ", ".join(f"{i.score:.2f}" for i in found[:4]) or "-"
        image_ax(axes[k], panel, f"{name}: {len(found)} instance(s)\nscores {scores}")
        if union.any():
            axes[k].text(0.01, 0.01, f"area {union.sum():,} px", transform=axes[k].transAxes,
                         color="white", fontsize=9, va="bottom")
    plot_overlay(image, result, ax=axes[4], title="Combined overlay", show_scores=False)
    lm = result.label_map()
    cmap = plt.matplotlib.colors.ListedColormap([(0.1, 0.1, 0.1)]
                                                + [class_color(c) for c in CLASS_NAMES])
    axes[5].imshow(lm, cmap=cmap, vmin=-0.5, vmax=len(CLASS_NAMES) + 0.5)
    axes[5].set_xticks([])
    axes[5].set_yticks([])
    axes[5].grid(False)
    axes[5].set_title("Label map (one class per pixel)")
    if result.seconds:
        fig.text(0.99, 0.01, f"{result.source}: {result.seconds:.2f} s",
                 ha="right", fontsize=9, color="gray")
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_instances_by_prompt(image: np.ndarray, result: SegmentationResult, ax=None,
                             title: str = "Which text prompt found each instance"):
    """Colour each instance by the SAM 3 prompt that produced it."""
    fig, ax = new_ax(ax, (8, 6))
    prompts = sorted({i.prompt for i in result.instances})
    cmap = plt.get_cmap("tab10")
    out = image.astype(np.float32).copy()
    for k, p in enumerate(prompts):
        col = np.array(cmap(k % 10)[:3]) * 255
        for inst in result.instances:
            if inst.prompt == p:
                out[inst.mask] = 0.5 * out[inst.mask] + 0.5 * col
    image_ax(ax, out.astype(np.uint8), title)
    from matplotlib.patches import Patch

    handles = [Patch(color=cmap(k % 10),
                     label=f"'{p}' ({sum(i.prompt == p for i in result.instances)})")
               for k, p in enumerate(prompts)]
    if handles:
        ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.85)
    return fig


def plot_gallery(items, ncols: int = 4, size: float = 4.0, title: str | None = None,
                 show_scores: bool = False):
    """Grid of overlays. ``items`` is a list of ``(image, result, caption)``."""
    n = len(items)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "nothing to show", ha="center")
        ax.axis("off")
        return fig
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * 0.75 * nrows),
                             squeeze=False)
    for ax, (img, res, cap) in zip(axes.ravel(), items, strict=False):
        plot_overlay(img, res, ax=ax, title=None, show_boxes=False, show_scores=show_scores,
                     legend=False)
        c = res.counts()
        ax.set_title(f"{cap}\nB{c['broiler']} S{c['scale']} F{c['feeder']}", fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    class_legend(axes.ravel()[0], loc="upper left")
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_bird_selection(image: np.ndarray, result: SegmentationResult, chosen: Instance | None,
                        gt_box_xyxy=None, info: dict | None = None, ax=None,
                        title: str = "Which bird is on the scale?"):
    """Show all broiler candidates, the platform region, the GT box and the chosen bird."""
    fig, ax = new_ax(ax, (8, 6))
    plat = platform_region(result)
    out = (image * 0.6).astype(np.uint8)
    out[plat] = (0.6 * out[plat] + 0.4 * np.array([90, 150, 255])).astype(np.uint8)
    for inst in result.of_class("broiler"):
        out[inst.mask] = (0.5 * out[inst.mask] + 0.5 * np.array([200, 200, 200])).astype(np.uint8)
    if chosen is not None:
        out[chosen.mask] = (0.3 * out[chosen.mask]
                            + 0.7 * np.array(class_color("broiler")) * 255).astype(np.uint8)
    image_ax(ax, out, title)
    for k, inst in enumerate(result.of_class("broiler")):
        x, y, w, h = inst.box_xywh
        is_c = chosen is not None and inst is chosen
        ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=2.5 if is_c else 1,
                               ec="yellow" if is_c else "white", ls="-" if is_c else ":"))
        ax.text(x + w, y, f"#{k} {inst.score:.2f}", color="white", fontsize=8, va="bottom")
    if gt_box_xyxy is not None:
        x1, y1, x2, y2 = gt_box_xyxy
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=2))
        ax.text(x1, y2 + 4, "FlockLabel box", color="lime", fontsize=8, va="top")
    if info:
        txt = "\n".join(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in info.items())
        ax.text(0.01, 0.99, txt, transform=ax.transAxes, va="top", fontsize=8,
                bbox={"fc": "white", "alpha": 0.85, "ec": "none"})
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=(90 / 255, 150 / 255, 1), label="scale+feeder platform"),
                       Patch(color=(0.78, 0.78, 0.78), label="other broilers"),
                       Patch(color=class_color("broiler"), label="chosen bird"),
                       Patch(fc="none", ec="lime", label="FlockLabel box")],
              loc="lower right", frameon=True, framealpha=0.85)
    return fig


def plot_mask_cleanup(image: np.ndarray, raw: np.ndarray, cleaned: np.ndarray,
                      title: str = "Bird mask before / after clean-up"):
    """Zoomed comparison of the raw and cleaned bird mask, with the changed pixels."""
    x1, y1, x2, y2 = crop_box(raw | cleaned, image.shape)
    crop = image[y1:y2, x1:x2]
    r, c = raw[y1:y2, x1:x2], cleaned[y1:y2, x1:x2]
    diff = np.zeros(crop.shape, np.uint8)
    diff[c & ~r] = (40, 200, 40)
    diff[r & ~c] = (220, 40, 40)
    diff[c & r] = (90, 90, 90)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    image_ax(axes[0], crop, "Crop")
    image_ax(axes[1], r, f"Raw ({r.sum():,} px)")
    image_ax(axes[2], c, f"Cleaned ({c.sum():,} px)")
    image_ax(axes[3], diff, "green = added (holes), red = removed")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_score_distributions(results, title: str = "Confidence of every instance"):
    """Histogram of instance scores per class over many results."""
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(5 * len(CLASS_NAMES), 3.4),
                             sharey=True)
    for ax, c in zip(axes, CLASS_NAMES, strict=False):
        s = [i.score for r in results for i in r.of_class(c)]
        ax.hist(s, bins=np.linspace(0, 1, 21), color=class_color(c), edgecolor="white")
        ax.set_title(f"{c} (n={len(s)})")
        ax.set_xlabel("score")
        if s:
            ax.axvline(np.median(s), color="black", ls="--", lw=1)
    axes[0].set_ylabel("instances")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
