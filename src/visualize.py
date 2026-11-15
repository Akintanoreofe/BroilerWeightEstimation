"""Matplotlib diagnostics for feeder calibration, segmentation, and slicing."""

import csv
import math
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from defaults import (
    MAX_BIRD_ELLIPSE_ASPECT_RATIO,
    MAX_FEEDER_CONTACT_RATIO,
    MIN_BIRD_SOLIDITY_RATIO,
)


def _matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm, patches
    except ImportError as error:
        raise ImportError(
            "Visualization requires matplotlib; install requirements-visualization.txt."
        ) from error
    return plt, cm, patches


def plot_feeder_calibration(frame, reference):
    """Display RANSAC inliers, the fitted ellipse, and the old enclosing circle."""
    plt, _, _ = _matplotlib()
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.imshow(cv2.cvtColor(reference.debug_image(), cv2.COLOR_BGR2RGB))
    axis.set_title(
        f"px_per_cm={reference.px_per_cm:.4f}; "
        f"minor/major={reference.minor_major_ratio:.3f}; "
        f"uncertainty={reference.calib_uncertainty_pct:.2f}%"
    )
    axis.axis("off")
    figure.tight_layout()
    return figure


def plot_segmentation_steps(frame, result, selected=None):
    """Show whiteness, both Otsu stages, morphology, and selected/rejected masks."""
    plt, _, _ = _matplotlib()
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    axes[0].imshow(result.whiteness, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Whiteness channel W")
    histogram = axes[1]
    histogram.hist(result.whiteness.ravel(), bins=64, color="gray")
    histogram.axvline(result.threshold_stage1, color="blue", label=f"t1={result.threshold_stage1:.1f}")
    histogram.axvline(result.threshold_stage2, color="red", label=f"t2={result.threshold_stage2:.1f}")
    histogram.set_title("Otsu thresholds")
    histogram.legend()
    for axis, image, title in (
        (axes[2], result.stage1_mask, "After Otsu stage 1"),
        (axes[3], result.stage2_mask, "After Otsu stage 2"),
        (axes[4], result.opened_mask, "After opening / closing / fill"),
    ):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)

    overlay = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for item in result.rejected_elongated:
        contour, _ = cv2.findContours(item.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contour, -1, (255, 0, 0), 2)
    if selected is not None:
        contours, _ = cv2.findContours(selected.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    axes[5].imshow(overlay)
    axes[5].set_title("Selected bird (green); rejected structures (red)")
    for axis in axes:
        if axis is not histogram:
            axis.axis("off")
    figure.tight_layout()
    return figure


def plot_quality_metrics(candidate):
    """Display the selected bird's quality metrics and gate status."""
    plt, _, _ = _matplotlib()
    figure, axis = plt.subplots(figsize=(9, 3))
    quality = candidate.quality
    values = [
        ("Feeder contact", quality.get("feeder_contact")),
        ("Occluder contact", quality.get("occluder_contact")),
        ("Solidity", quality.get("solidity")),
        ("Ellipse aspect", quality.get("ellipse_aspect")),
        ("Sharpness", quality.get("sharpness")),
        ("Border contact", quality.get("border_contact")),
        ("Measurable", candidate.measurable),
    ]
    rows = []
    for name, value in values:
        if name == "Feeder contact":
            passed = value <= MAX_FEEDER_CONTACT_RATIO
        elif name == "Occluder contact":
            passed = value == 0
        elif name == "Solidity":
            passed = value >= MIN_BIRD_SOLIDITY_RATIO
        elif name == "Ellipse aspect":
            passed = value <= MAX_BIRD_ELLIPSE_ASPECT_RATIO
        elif name == "Border contact":
            passed = not value
        elif name == "Measurable":
            passed = value
        else:
            passed = None
        status = "INFO" if passed is None else "PASS" if passed else "FAIL"
        formatted = f"{value:.4g}" if isinstance(value, (int, float)) else str(value)
        rows.append([name, formatted, status])
    axis.axis("off")
    axis.table(cellText=rows, colLabels=("Metric", "Value", "Gate"), loc="center")
    axis.set_title(f"Track {candidate.track_id}: quality gates")
    figure.tight_layout()
    return figure


def _slice_colors(count):
    plt, cm, _ = _matplotlib()
    return cm.turbo(np.linspace(0.02, 0.98, max(count, 1)))


def plot_slices(frame, bird_mask, result, px_per_cm):
    """Show a six-panel, color-linked view of each volume-estimation slice."""
    plt, _, patches = _matplotlib()
    count = result.n_slices
    colors = _slice_colors(count)
    figure = plt.figure(figsize=(18, 11))
    axes = [
        figure.add_subplot(2, 3, 1),
        figure.add_subplot(2, 3, 2),
        figure.add_subplot(2, 3, 3),
        figure.add_subplot(2, 3, 4),
        figure.add_subplot(2, 3, 5, projection="3d"),
        figure.add_subplot(2, 3, 6),
    ]

    original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    axes[0].imshow(original)
    contours, _ = cv2.findContours((bird_mask > 0).astype(np.uint8) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        axes[0].plot(contour[:, 0, 0], contour[:, 0, 1], color="white", linewidth=1.5)
        ys, xs = np.where(bird_mask > 0)
        coords = np.column_stack((xs, ys)).astype(np.float64)
        center = coords.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(coords.T))
        axis_vector = eigenvectors[:, np.argmax(eigenvalues)]
        projection = (coords - center) @ axis_vector
        endpoints = center + np.outer([projection.min(), projection.max()], axis_vector)
        axes[0].plot(endpoints[:, 0], endpoints[:, 1], "--", color="yellow", linewidth=2)
        inverse = np.linalg.inv(np.vstack((result.alignment_matrix, [0, 0, 1])))
        height = result.aligned_mask.shape[0]
        for index, (position, color) in enumerate(zip(result.slice_positions_px, colors)):
            endpoints_aligned = np.array([[position, 0, 1], [position, height - 1, 1]])
            endpoints_source = endpoints_aligned @ inverse.T
            axes[0].plot(endpoints_source[:, 0], endpoints_source[:, 1],
                         color=color, linewidth=1)
    axes[0].set_title("(a) Outline, PCA axis, and slice cuts")
    axes[0].axis("off")

    axes[1].imshow(bird_mask, cmap="gray")
    axes[1].set_title("(b) Binary bird mask")
    axes[1].axis("off")

    aligned_height, aligned_width = result.aligned_mask.shape
    extent = (0, aligned_width / px_per_cm, 0, aligned_height / px_per_cm)
    axes[2].imshow(result.aligned_mask, cmap="gray", origin="lower", extent=extent)
    center_y_cm = aligned_height / (2 * px_per_cm)
    for index, (position, width, color) in enumerate(zip(
            result.slice_positions_px, result.slice_widths_cm, colors)):
        axes[2].add_patch(patches.Rectangle(
            ((position - result.delta_x_cm * px_per_cm / 2) / px_per_cm,
             center_y_cm - width / 2),
            result.delta_x_cm, width, fill=False, edgecolor=color, linewidth=1.2,
        ))
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("Body length (cm)")
    axes[2].set_ylabel("Body width (cm)")
    axes[2].set_title(f"(c) {count} slices, Δx={result.delta_x_cm:.2f} cm")

    centers_cm = (result.slice_positions_px / px_per_cm).tolist()
    axes[3].bar(centers_cm, result.slice_widths_cm, width=result.delta_x_cm,
                color=colors, edgecolor="black", linewidth=0.25)
    axes[3].set_xlabel("Body length (cm)")
    axes[3].set_ylabel("Slice width wᵢ (cm)")
    axes[3].set_title("(d) Silhouette width by slice")

    solid = axes[4]
    for index, (x_center, width_cm, volume, color) in enumerate(zip(
            centers_cm, result.slice_widths_cm, result.slice_volumes_cm3, colors)):
        if width_cm <= 0 or result.delta_x_cm <= 0:
            continue
        area_cm2 = volume / result.delta_x_cm
        depth_cm = (width_cm if result.cross_section == "Circular" else
                    result.k * width_cm if result.k is not None else
                    4.0 * area_cm2 / (math.pi * width_cm))
        theta = np.linspace(0, 2 * np.pi, 25)
        x_values = np.array([x_center - result.delta_x_cm / 2,
                             x_center + result.delta_x_cm / 2])
        xx, tt = np.meshgrid(x_values, theta)
        yy = width_cm / 2 * np.cos(tt)
        zz = depth_cm / 2 * np.sin(tt)
        solid.plot_surface(xx, yy, zz, color=color, linewidth=0, antialiased=True,
                           shade=True, alpha=0.92)
    if result.slice_widths_cm:
        last_width = result.slice_widths_cm[-1]
        last_volume = result.slice_volumes_cm3[-1]
        last_depth = (last_width if result.cross_section == "Circular" else
                      result.k * last_width if result.k is not None else
                      4.0 * (last_volume / result.delta_x_cm) / (math.pi * max(last_width, 1e-9)))
        theta = np.linspace(0, 2 * np.pi, 40)
        radius = np.linspace(0, 1, 8)
        rr, tt = np.meshgrid(radius, theta)
        x_end = centers_cm[-1] + result.delta_x_cm / 2
        solid.plot_surface(np.full_like(rr, x_end),
                           rr * last_width / 2 * np.cos(tt),
                           rr * last_depth / 2 * np.sin(tt),
                           color=colors[-1], alpha=0.8)
    solid.set_xlabel("Length (cm)")
    solid.set_ylabel("Width (cm)")
    solid.set_zlabel("Depth (cm)")
    solid.view_init(elev=15, azim=-65)
    solid.set_title("(e) Sliced solid")

    axes[5].bar(np.arange(count), result.slice_volumes_cm3, color=colors,
                edgecolor="black", linewidth=0.25)
    axes[5].set_xlabel("Slice i")
    axes[5].set_ylabel("Volume (cm³)")
    formula = ("π(wᵢ/2)²·Δx" if result.cross_section == "Circular" else
               f"π·wᵢ·({result.k:g}wᵢ)/4·Δx" if result.k is not None else
               "π·wᵢ·dᵢ/4·Δx")
    weight = result.weight_kg_est
    weight_text = f"; weight={weight:.3f} kg" if weight is not None else ""
    axes[5].set_title(f"(f) {formula}\nV={result.volume_cm3:.1f} cm³{weight_text}")
    figure.tight_layout()
    return figure


def plot_slice_count_sensitivity(mask, px_per_cm, cross_sections: Optional[Sequence] = None):
    """Compare integrated volume as slice count increases."""
    from SliceVolumeEstimator import Circular, Elliptical, ProfileScaled, SliceVolumeEstimator

    plt, _, _ = _matplotlib()
    models = cross_sections or (Circular(), Elliptical(0.8), ProfileScaled())
    counts = list(range(5, 101, 5))
    figure, axis = plt.subplots(figsize=(9, 5))
    for model in models:
        volumes = [SliceVolumeEstimator(count, model).estimate(mask, px_per_cm).volume_cm3
                   for count in counts]
        axis.plot(counts, volumes, marker="o", markersize=3, label=type(model).__name__)
    axis.set_xlabel("Number of slices")
    axis.set_ylabel("Volume (cm³)")
    axis.set_title("Slice-count sensitivity")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def plot_video_features(csv_path, track_id=None):
    """Plot selected-track volume over time and per-visit medians."""
    plt, _, _ = _matplotlib()
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if track_id is not None:
        rows = [row for row in rows if row.get("track_id") == str(track_id)]
    figure, axis = plt.subplots(figsize=(12, 5))
    if not rows:
        axis.set_title("No selected bird rows")
        return figure
    x = np.asarray([float(row["timestamp_s"] or 0) for row in rows])
    y = np.asarray([float(row["volume_cm3"] or 0) for row in rows])
    measurable = np.asarray([row.get("measurable", "").lower() == "true" for row in rows])
    axis.scatter(x[measurable], y[measurable], color="tab:blue", label="Measurable")
    axis.scatter(x[~measurable], y[~measurable], color="gray", label="Rejected")
    tracks = sorted({row.get("track_id", "") for row in rows if row.get("track_id", "")})
    for track in tracks:
        values = [float(row["volume_cm3"]) for row in rows
                  if row.get("track_id") == track and row.get("volume_cm3")]
        if values:
            axis.axhline(float(np.median(values)), linestyle="--", alpha=0.3)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Volume (cm³)")
    axis.set_title("Volume over time; dashed lines are per-visit medians")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def plot_weight_validation(features_csv, labels_csv):
    """Plot visit-median volume against scale weight and fit density through zero."""
    plt, _, _ = _matplotlib()
    with Path(features_csv).open(newline="", encoding="utf-8") as stream:
        features = list(csv.DictReader(stream))
    with Path(labels_csv).open(newline="", encoding="utf-8") as stream:
        labels = list(csv.DictReader(stream))
    volumes = {}
    for row in features:
        if row.get("track_id") and row.get("volume_cm3"):
            key = (row["video_name"], row["track_id"])
            volumes.setdefault(key, []).append(float(row["volume_cm3"]))
    actual = {}
    for row in labels:
        if "weight_kg" in row:
            actual[(row.get("video_name"), row.get("track_id", ""))] = float(row["weight_kg"])
    matched = [(float(np.median(values)), actual[key]) for key, values in volumes.items()
               if key in actual]
    if not matched:
        raise ValueError("No matching visit labels with weight_kg were found.")
    x = np.asarray([item[0] for item in matched])
    y = np.asarray([item[1] for item in matched])
    slope = float(np.dot(x, y) / np.dot(x, x))
    prediction = slope * x
    error_pct = 100.0 * (prediction - y) / y
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(x, y)
    axes[0].plot(x, slope * x, color="tab:red", label=f"density={slope * 1000:.3f} g/cm³")
    axes[0].set_xlabel("Median volume per visit (cm³)")
    axes[0].set_ylabel("Scale weight (kg)")
    axes[0].legend()
    axes[1].scatter(x, error_pct)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Median volume per visit (cm³)")
    axes[1].set_ylabel("Percentage error (%)")
    figure.tight_layout()
    return figure
