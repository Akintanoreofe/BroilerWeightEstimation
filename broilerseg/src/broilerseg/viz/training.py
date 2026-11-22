"""Plots of student training and teacher-student agreement."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from broilerseg.config import CLASS_NAMES
from broilerseg.segmentation.base import SegmentationResult
from broilerseg.viz.overlays import blend_masks
from broilerseg.viz.style import SERIES, class_color, image_ax, new_ax


def plot_training_curves(df: pd.DataFrame, title: str = "YOLOv8-seg student training"):
    """Losses (train vs val) and box / mask metrics per epoch from ``results.csv``."""
    ep = df["epoch"] if "epoch" in df else np.arange(len(df))
    losses = sorted({c.split("/", 1)[1] for c in df if c.startswith(("train/", "val/"))
                     and "loss" in c})
    metrics = [c for c in df if c.startswith("metrics/")]
    n = len(losses) + 2
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    axs = axes.ravel()
    for ax, name in zip(axs, losses, strict=False):
        for split, col in (("train", SERIES["teacher"]), ("val", SERIES["pred"])):
            c = f"{split}/{name}"
            if c in df:
                ax.plot(ep, df[c], color=col, label=split)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    for ax, kind in zip(axs[len(losses):len(losses) + 2], ("(M)", "(B)"), strict=False):
        for c in metrics:
            if c.endswith(kind):
                ax.plot(ep, df[c], label=c.replace("metrics/", ""))
        ax.set_ylim(0, 1.02)
        ax.set_title("Mask metrics" if kind == "(M)" else "Box metrics")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    for ax in axs[n:]:
        ax.axis("off")
    best = ""
    if "metrics/mAP50-95(M)" in df:
        i = int(df["metrics/mAP50-95(M)"].idxmax())
        best_val = df["metrics/mAP50-95(M)"].iloc[i]
        best = f" | best mask mAP50-95 {best_val:.3f} at epoch {ep.iloc[i]}"
    fig.suptitle(title + best, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_teacher_student(image: np.ndarray, teacher: SegmentationResult,
                         student: SegmentationResult, title: str | None = None):
    """Teacher overlay, student overlay, pixel disagreement map and per-class IoU."""
    from broilerseg.distill import compare_results

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.6]})
    image_ax(axes[0], blend_masks(image, teacher), f"Teacher: {teacher.source}")
    image_ax(axes[1], blend_masks(image, student),
             f"Student: {student.source} ({1000 * student.seconds:.0f} ms)")
    dis = (image * 0.25).astype(np.uint8)
    for c in CLASS_NAMES:
        t, s = teacher.union(c), student.union(c)
        dis[t & s] = (np.array(class_color(c)) * 200).astype(np.uint8)
        dis[t & ~s] = (230, 40, 40)
        dis[s & ~t] = (40, 120, 255)
    image_ax(axes[2], dis, "Agreement: class colour = both, red = teacher only, "
                           "blue = student only")
    axes[2].title.set_fontsize(8)
    rows = compare_results(teacher, student)
    ious = [r["iou"] if np.isfinite(r["iou"]) else 0 for r in rows]
    axes[3].barh(list(CLASS_NAMES), ious, color=[class_color(c) for c in CLASS_NAMES])
    for k, v in enumerate(ious):
        axes[3].text(v, k, f" {v:.2f}", va="center")
    axes[3].set_xlim(0, 1.15)
    axes[3].set_title("IoU per class")
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_iou_distribution(table: pd.DataFrame, ax=None,
                          title: str = "Student vs teacher mask IoU on held-out frames"):
    """Box + jittered points of per-image IoU for each class."""
    fig, ax = new_ax(ax, (8, 4.5))
    rng = np.random.default_rng(0)
    for k, c in enumerate(CLASS_NAMES):
        v = table.loc[table.class_name == c, "iou"].dropna().to_numpy()
        if not len(v):
            continue
        bp = ax.boxplot([v], positions=[k], widths=0.5, patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor((*class_color(c), 0.45))
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=10, color="black", alpha=0.5)
        ax.text(k, 1.03, f"median {np.median(v):.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("IoU")
    ax.set_title(title)
    return fig


def plot_count_agreement(table: pd.DataFrame,
                         title: str = "Instance counts: teacher vs student"):
    """Scatter of instance counts per image, one panel per class."""
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(5 * len(CLASS_NAMES), 4))
    for ax, c in zip(axes, CLASS_NAMES, strict=False):
        d = table[table.class_name == c]
        if d.empty:
            continue
        m = int(max(d.teacher_n.max(), d.student_n.max(), 1))
        cnt = d.groupby(["teacher_n", "student_n"]).size().reset_index(name="k")
        ax.scatter(cnt.teacher_n, cnt.student_n, s=40 + 30 * cnt.k, color=class_color(c),
                   alpha=0.8, edgecolor="k")
        ax.plot([0, m], [0, m], ls="--", color="gray")
        ax.set_xlabel("teacher instances")
        ax.set_ylabel("student instances")
        ax.set_title(f"{c}: exact match {100 * (d.teacher_n == d.student_n).mean():.0f}%")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_speed_accuracy(rows: list[dict], title: str = "Speed vs agreement with SAM 3"):
    """Latency bars and a speed / IoU scatter. ``rows`` hold name, ms_mean, iou, size_mb."""
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].barh(d.name, d.ms_mean, xerr=d.get("ms_std"), color=SERIES["student"])
    for k, r in d.iterrows():
        axes[0].text(r.ms_mean, k, f" {r.ms_mean:.0f} ms ({1000 / r.ms_mean:.1f} fps)",
                     va="center", fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("ms per frame (log) on THIS machine")
    axes[0].set_title("Latency")
    if "iou" in d:
        axes[1].scatter(d.ms_mean, d.iou, s=60 + 4 * d.get("size_mb", pd.Series(0, d.index)),
                        color=SERIES["teacher"], edgecolor="k")
        for _, r in d.iterrows():
            axes[1].annotate(r["name"], (r.ms_mean, r.iou), xytext=(5, 5),
                             textcoords="offset points", fontsize=8)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("ms per frame (log)")
        axes[1].set_ylabel("mean IoU vs teacher")
        axes[1].set_title("Trade-off (marker size = model MB)")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_student_metrics(overall: dict | list, per_class: pd.DataFrame | None = None,
                         title: str = "Student accuracy on held-out videos (vs teacher labels)"):
    """Overall mask / box metrics per split, and mAP50-95 per class."""
    rows = overall if isinstance(overall, list) else [overall]
    df = pd.DataFrame(rows).set_index("split")
    cols = ["mask_mAP50", "mask_mAP50-95", "mask_precision", "mask_recall",
            "box_mAP50", "box_mAP50-95"]
    fig, axes = plt.subplots(1, 2 if per_class is not None else 1, figsize=(16, 4.5),
                             squeeze=False, gridspec_kw={"width_ratios": [2, 1]}
                             if per_class is not None else None)
    ax = axes[0, 0]
    x = np.arange(len(cols))
    w = 0.8 / max(1, len(df))
    for k, (split, r) in enumerate(df.iterrows()):
        vals = [r.get(c, np.nan) for c in cols]
        bars = ax.bar(x + k * w, vals, w, label=split,
                      color=[SERIES["student"], SERIES["teacher"], SERIES["pred"]][k % 3])
        for b, v in zip(bars, vals, strict=False):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7)
    ax.set_xticks(x + w * (len(df) - 1) / 2)
    ax.set_xticklabels(cols, rotation=20, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend(title="split")
    ax.set_title("Overall")
    if per_class is not None:
        ax2 = axes[0, 1]
        y = np.arange(len(per_class))
        ax2.barh(y - 0.2, per_class["mask_mAP50-95"], 0.4,
                 color=[class_color(c) for c in per_class.class_name], label="mask")
        ax2.barh(y + 0.2, per_class["box_mAP50-95"], 0.4, color="#bbbbbb", label="box")
        ax2.set_yticks(y)
        ax2.set_yticklabels(per_class.class_name)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("mAP50-95")
        ax2.legend()
        ax2.set_title("Per class")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig
