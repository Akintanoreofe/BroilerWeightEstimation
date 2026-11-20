"""Plots of the inputs: video inventory, sampling plan, labels, YOLO dataset, weights."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from broilerseg.config import CLASS_NAMES
from broilerseg.viz.style import SERIES, class_color, image_ax


def plot_video_inventory(videos: pd.DataFrame, title: str = "Video folder at a glance"):
    """Duration, frame count, fps and resolution of every video."""
    v = videos.reset_index(drop=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    x = np.arange(len(v))
    labels = [s[:22] for s in v.stem]
    axes[0, 0].bar(x, v.duration_s, color=SERIES["teacher"])
    axes[0, 0].set_ylabel("seconds")
    axes[0, 0].set_title(f"Duration (total {v.duration_s.sum() / 60:.1f} min)")
    axes[0, 1].bar(x, v.n_frames, color=SERIES["student"])
    axes[0, 1].set_ylabel("frames")
    axes[0, 1].set_title(f"Frame count (total {int(v.n_frames.sum()):,})")
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    axes[1, 0].hist(v.fps, bins=15, color=SERIES["neutral"])
    axes[1, 0].set_xlabel("frames per second")
    axes[1, 0].set_title("Frame rate")
    sc = axes[1, 1].scatter(v.width, v.height, s=40 + 4 * v.size_mb, c=v.duration_s,
                            cmap="viridis", alpha=0.75, edgecolor="k")
    axes[1, 1].set_xlabel("width (px)")
    axes[1, 1].set_ylabel("height (px)")
    axes[1, 1].set_title("Resolution (size = file MB, colour = duration)")
    fig.colorbar(sc, ax=axes[1, 1], label="seconds")
    unread = int((~v.readable).sum())
    fig.suptitle(f"{title}: {len(v)} videos, {v.size_mb.sum():.0f} MB"
                 + (f", {unread} unreadable" if unread else ""), fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_sampling_plan(plan: pd.DataFrame, title: str = "Which frames will be labelled"):
    """One line per video with a tick at each sampled frame."""
    vids = list(dict.fromkeys(plan.video_stem))
    fig, ax = plt.subplots(figsize=(12, 0.32 * len(vids) + 1.5))
    for k, v in enumerate(vids):
        p = plan[plan.video_stem == v]
        ax.hlines(k, 0, 1, color="#dddddd", lw=6)
        ax.scatter(p.position, np.full(len(p), k), marker="|", s=160, color=SERIES["pred"])
    ax.set_yticks(range(len(vids)))
    ax.set_yticklabels([v[:30] for v in vids], fontsize=7)
    ax.set_xlabel("position in video (0 = start, 1 = end)")
    ax.set_xlim(-0.02, 1.02)
    ax.invert_yaxis()
    ax.set_title(f"{title}: {len(plan)} frames from {len(vids)} videos")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_contact_sheet(images, titles=None, ncols: int = 5, size: float = 3.2,
                       title: str | None = None):
    """Grid of raw images."""
    n = len(images)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * 0.75 * nrows),
                             squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        if k < n:
            image_ax(ax, images[k], titles[k] if titles else None)
            ax.title.set_fontsize(8)
        else:
            ax.axis("off")
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_label_statistics(manifest: pd.DataFrame,
                          title: str = "What did the teacher label?"):
    """Plot counts, scores, areas and missing classes across the labelled frames."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    cols = [class_color(c) for c in CLASS_NAMES]
    mean_n = [manifest[f"n_{c}"].mean() for c in CLASS_NAMES]
    axes[0, 0].bar(CLASS_NAMES, mean_n, color=cols)
    axes[0, 0].set_title("Mean instances per frame")
    missing = [(manifest[f"n_{c}"] == 0).mean() * 100 for c in CLASS_NAMES]
    axes[0, 1].bar(CLASS_NAMES, missing, color=cols)
    axes[0, 1].set_ylabel("% of frames")
    axes[0, 1].set_title("Frames where a class was NOT found")
    mx = int(max(1, manifest[[f"n_{c}" for c in CLASS_NAMES]].to_numpy().max()))
    bins = np.arange(-0.5, mx + 1.5)
    for c, col in zip(CLASS_NAMES, cols, strict=False):
        axes[0, 2].hist(manifest[f"n_{c}"], bins=bins, alpha=0.55, color=col, label=c)
    axes[0, 2].set_xlabel("instances in a frame")
    axes[0, 2].legend()
    axes[0, 2].set_title("Instances per frame")
    for c, col in zip(CLASS_NAMES, cols, strict=False):
        s = manifest[f"score_{c}"].dropna()
        axes[1, 0].hist(s, bins=np.linspace(0, 1, 21), alpha=0.55, color=col, label=c)
        a = manifest.loc[manifest[f"area_{c}"] > 0, f"area_{c}"]
        if len(a):
            axes[1, 1].hist(np.log10(a), bins=25, alpha=0.55, color=col, label=c)
    axes[1, 0].set_xlabel("best score in frame")
    axes[1, 0].set_title("Teacher confidence")
    axes[1, 0].legend()
    axes[1, 1].set_xlabel("log10(mask area px)")
    axes[1, 1].set_title("Mask sizes")
    axes[1, 1].legend()
    axes[1, 2].hist(manifest.seconds, bins=20, color=SERIES["teacher"])
    axes[1, 2].set_xlabel("seconds per frame")
    axes[1, 2].set_title(f"Labelling time (total {manifest.seconds.sum() / 60:.1f} min)")
    fig.suptitle(f"{title} ({len(manifest)} frames)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_split_summary(summary: pd.DataFrame, title: str = "YOLO dataset built from SAM 3"):
    """Images and polygons per split, and polygon fidelity per class."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    order = [s for s in ("train", "val", "test") if s in set(summary.split)]
    vids = summary.groupby("split").video_stem.nunique().reindex(order)
    imgs = summary.groupby("split").size().reindex(order)
    axes[0].bar(order, imgs, color=SERIES["neutral"])
    for k, s in enumerate(order):
        axes[0].text(k, imgs[s], f"{imgs[s]} img\n{vids[s]} videos", ha="center", va="bottom",
                     fontsize=8)
    axes[0].set_title("Images per split (split by video)")
    w = 0.25
    for j, c in enumerate(CLASS_NAMES):
        vals = summary.groupby("split")[f"n_{c}"].sum().reindex(order)
        axes[1].bar(np.arange(len(order)) + (j - 1) * w, vals, w, color=class_color(c), label=c)
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order)
    axes[1].legend()
    axes[1].set_title("Polygons per class")
    data = [summary[f"fidelity_{c}"].dropna() for c in CLASS_NAMES]
    bp = axes[2].boxplot(data, tick_labels=list(CLASS_NAMES), patch_artist=True)
    for patch, c in zip(bp["boxes"], CLASS_NAMES, strict=False):
        patch.set_facecolor(class_color(c))
    axes[2].set_ylabel("IoU(teacher mask, polygon)")
    axes[2].set_title("Polygon fidelity (1 = lossless)")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_yolo_label_check(image_path: str | Path, label_path: str | Path, ax=None,
                          title: str | None = None):
    """Draw the polygons stored in a YOLO label file on its image."""
    from broilerseg.yolo_dataset import read_yolo_label

    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    fig = ax.figure
    image_ax(ax, img, title or f"Label file check: {Path(label_path).name}")
    for cid, poly in read_yolo_label(label_path, w, h):
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
        ax.fill(poly[:, 0], poly[:, 1], color=class_color(name), alpha=0.35)
        ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]],
                color=class_color(name), lw=1.5)
        ax.plot(poly[:, 0], poly[:, 1], ".", color="white", ms=2)
    return fig


def plot_weight_dataset_overview(samples: pd.DataFrame,
                                 title: str = "FlockLabel validation data"):
    """Scale readings per event, frames per event and the weight distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8),
                             gridspec_kw={"width_ratios": [2, 1.2, 1]})
    events = sorted(samples.event_id.unique(), key=lambda e: samples.loc[
        samples.event_id == e, "weight_kg"].median())
    rng = np.random.default_rng(0)
    for k, e in enumerate(events):
        y = samples.loc[samples.event_id == e, "weight_kg"].to_numpy()
        axes[0].scatter(k + rng.uniform(-0.2, 0.2, len(y)), y, s=10, alpha=0.6,
                        color=SERIES["actual"])
        axes[0].hlines(np.median(y), k - 0.35, k + 0.35, color=SERIES["pred"], lw=2)
    axes[0].set_xticks(range(len(events)))
    axes[0].set_xticklabels(events, rotation=70, fontsize=7)
    axes[0].set_xlabel("event (sorted by weight)")
    axes[0].set_ylabel("scale reading (kg)")
    axes[0].set_title("Per-frame scale readings (orange = event median)")
    cnt = samples.groupby("event_id").size().reindex(events)
    axes[1].bar(range(len(events)), cnt, color=SERIES["neutral"])
    axes[1].set_xticks(range(len(events)))
    axes[1].set_xticklabels(events, rotation=70, fontsize=7)
    axes[1].set_title("Labelled frames per event")
    med = samples.groupby("event_id").weight_kg.median()
    axes[2].hist(med, bins=min(20, max(5, len(med))), color=SERIES["teacher"])
    axes[2].set_xlabel("event median weight (kg)")
    axes[2].set_title("Weight distribution")
    fig.suptitle(f"{title}: {samples.event_id.nunique()} events, {len(samples)} frames, "
                 f"{samples.video_name.nunique()} videos", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_motion_profile(motion: pd.DataFrame, chosen=None, ax=None,
                        title: str = "Activity along the video"):
    """Frame-difference curve with the frames chosen for labelling marked."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    fig = ax.figure
    ax.fill_between(motion.frame_idx, motion.motion.fillna(0), color=SERIES["teacher"],
                    alpha=0.35)
    ax.plot(motion.frame_idx, motion.motion, color=SERIES["teacher"], lw=1)
    for f in chosen or []:
        ax.axvline(f, color=SERIES["pred"], lw=1, alpha=0.8)
    ax.set_xlabel("frame")
    ax.set_ylabel("mean |difference|")
    ax.set_title(title + (" (orange = sampled frames)" if chosen else ""))
    return fig
