"""Plots of the slice-based volume model: alignment, slices, cylinders, convergence."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from broilerseg.segmentation.base import Instance, SegmentationResult
from broilerseg.viz.overlays import blend_masks
from broilerseg.viz.style import SERIES, class_color, crop_box, fmt_weight, image_ax, new_ax
from broilerseg.volume import VolumeEstimate, object_angle


def plot_alignment(mask: np.ndarray, est: VolumeEstimate,
                   title: str = "Body-axis alignment before slicing"):
    """Original bird mask with its fitted axis, and the rotated mask."""
    x1, y1, x2, y2 = crop_box(mask, mask.shape, pad=0.15)
    crop = mask[y1:y2, x1:x2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    image_ax(axes[0], crop, f"Mask in the frame (axis {object_angle(crop):.1f} deg)")
    ys, xs = np.nonzero(crop)
    if len(xs):
        cx, cy = xs.mean(), ys.mean()
        a = np.deg2rad(est.angle_deg)
        L = 0.6 * max(crop.shape)
        axes[0].plot([cx - L * np.cos(a), cx + L * np.cos(a)],
                     [cy - L * np.sin(a), cy + L * np.sin(a)], color="red", lw=2,
                     label="body axis")
        axes[0].legend(loc="lower right")
    image_ax(axes[1], est.aligned_mask, "Rotated: body axis horizontal")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_slices(est: VolumeEstimate, ax=None, max_lines: int = 60,
                title: str | None = None):
    """Aligned mask with alternating slice colours and the sampled diameters."""
    fig, ax = new_ax(ax, (8, 4.5))
    m = est.aligned_mask > 0
    canvas = np.zeros((*m.shape, 3), np.float32)
    canvas[m] = 0.25
    p = est.profile
    cmap = [np.array([0.35, 0.65, 0.95]), np.array([0.95, 0.55, 0.25])]
    for k, (xs, xe) in enumerate(zip(p["x_start"], p["x_end"], strict=False)):
        sub = m[:, xs:xe + 1]
        canvas[:, xs:xe + 1][sub] = cmap[k % 2]
    ax.imshow(canvas)
    ax.grid(False)
    step = max(1, int(np.ceil(len(p["sample_x"]) / max_lines)))
    for k in range(0, len(p["sample_x"]), step):
        x = p["sample_x"][k]
        ax.plot([x, x], [p["y_min"][k], p["y_max"][k]], color="yellow", lw=1)
    label = "per-column" if est.slice_count is None else est.slice_count
    ax.set_title(title or f"{est.n_slices} slices (slice_count={label}); yellow = diameter")
    ax.set_xlabel("pixels along body axis")
    ax.set_ylabel("pixels")
    return fig


def plot_slice_profile(est: VolumeEstimate, ax=None, title: str = "Per-slice geometry"):
    """Bar of each cylinder's volume with the diameter profile on a twin axis."""
    fig, ax = new_ax(ax, (8, 4))
    x = np.cumsum(est.thickness_cm) - est.thickness_cm / 2
    ax.bar(x, est.slice_volumes_cm3, width=est.thickness_cm * 0.95,
           color=class_color("broiler"), edgecolor="none", label="slice volume")
    ax.set_xlabel("position along body (cm)")
    ax.set_ylabel("slice volume (cm$^3$)")
    ax2 = ax.twinx()
    ax2.plot(x, est.diameters_cm, color=SERIES["teacher"], lw=2, label="diameter")
    ax2.set_ylabel("diameter (cm)", color=SERIES["teacher"])
    ax2.grid(False)
    ax.set_title(f"{title}: total {est.volume_cm3:,.0f} cm$^3$")
    return fig


def plot_cylinders_3d(est: VolumeEstimate, ax=None, max_cylinders: int = 40,
                      title: str = "Stacked-cylinder body model"):
    """3D rendering of the cylinders whose volumes are summed."""
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection="3d")
    fig = ax.figure
    n = len(est.diameters_cm)
    if n == 0:
        return fig
    # Merge columns into at most ``max_cylinders`` for a readable picture.
    groups = np.array_split(np.arange(n), min(n, max_cylinders))
    theta = np.linspace(0, 2 * np.pi, 28)
    x0 = 0.0
    cmap = plt.get_cmap("YlOrBr")
    rmax = est.diameters_cm.max() / 2
    for g in groups:
        length = est.thickness_cm[g].sum()
        r = np.sqrt((est.diameters_cm[g] ** 2 * est.thickness_cm[g]).sum() / length) / 2
        xx = np.array([x0, x0 + length])
        T, X = np.meshgrid(theta, xx)
        ax.plot_surface(X, r * np.cos(T), r * np.sin(T), color=cmap(0.3 + 0.6 * r / rmax),
                        alpha=0.9, linewidth=0, shade=True)
        x0 += length
    ax.set_xlabel("length (cm)")
    ax.set_ylabel("cm")
    ax.set_zlabel("cm")
    ax.set_box_aspect((x0, 2 * rmax, 2 * rmax))
    ax.set_title(title)
    return fig


def plot_slice_convergence(vol_df: pd.DataFrame, ax=None,
                           title: str = "Does the volume depend on the slice count?"):
    """Volume vs number of slices (from ``volume_by_slice_count``)."""
    fig, ax = new_ax(ax, (7, 4))
    d = vol_df.sort_values("n_slices")
    ax.plot(d.n_slices, d.volume_cm3, marker="o", color=SERIES["pred"])
    for _, r in d.iterrows():
        ax.annotate(str(r.slice_count), (r.n_slices, r.volume_cm3), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("number of slices (log)")
    ax.set_ylabel("volume (cm$^3$)")
    ax.set_title(title)
    return fig


def plot_volume_storyboard(image: np.ndarray, result: SegmentationResult, bird: Instance,
                           est: VolumeEstimate, actual_kg: float | None = None,
                           title: str | None = None):
    """One figure that walks from the raw frame to the weight estimate."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(gs[0, 0])
    image_ax(ax0, blend_masks(image, result), "1. SAM / YOLO masks")
    x1, y1, x2, y2 = crop_box(bird.mask, image.shape, pad=0.3)
    ax1 = fig.add_subplot(gs[0, 1])
    crop = image[y1:y2, x1:x2].copy()
    bm = bird.mask[y1:y2, x1:x2]
    crop[bm] = (0.45 * crop[bm] + 0.55 * np.array(class_color("broiler")) * 255).astype(np.uint8)
    image_ax(ax1, crop, f"2. Chosen bird ({bird.area:,} px, score {bird.score:.2f})")
    ax2 = fig.add_subplot(gs[0, 2])
    image_ax(ax2, est.aligned_mask, f"3. Aligned (rotated {est.angle_deg:.1f} deg)")
    plot_slices(est, ax=fig.add_subplot(gs[1, 0]), title="4. Slices and diameters")
    plot_cylinders_3d(est, ax=fig.add_subplot(gs[1, 1], projection="3d"),
                      title="5. Cylinders")
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    lines = [
        ("pixel_to_cm", f"{est.pixel_to_cm:.5f} cm/px"),
        ("slices", f"{est.n_slices}"),
        ("length", f"{est.length_cm:.1f} cm"),
        ("max width", f"{est.max_width_cm:.1f} cm"),
        ("silhouette area", f"{est.area_cm2:.0f} cm$^2$"),
        ("volume", f"{est.volume_cm3:,.0f} cm$^3$"),
        ("density", f"{est.density_kg_m3:.0f} kg/m$^3$"),
        ("ESTIMATE", fmt_weight(est.mass_kg)),
    ]
    if actual_kg is not None and np.isfinite(actual_kg):
        err = est.mass_kg - actual_kg
        lines += [("scale reading", fmt_weight(actual_kg)),
                  ("error", f"{fmt_weight(err, signed=True)} ({100 * err / actual_kg:+.1f}%)")]
    for k, (a, b) in enumerate(lines):
        bold = a in ("ESTIMATE", "error")
        ax5.text(0.02, 0.95 - k * 0.09, a, fontsize=12, transform=ax5.transAxes,
                 fontweight="bold" if bold else "normal")
        ax5.text(0.55, 0.95 - k * 0.09, b, fontsize=12, transform=ax5.transAxes,
                 fontweight="bold" if bold else "normal")
    ax5.set_title("6. Numbers", loc="left")
    if title:
        fig.suptitle(title, fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
