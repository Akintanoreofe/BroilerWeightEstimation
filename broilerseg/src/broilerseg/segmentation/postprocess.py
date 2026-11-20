"""Mask clean-up, de-duplication and selection of the bird standing on the scale."""

from __future__ import annotations

import cv2
import numpy as np

from broilerseg.segmentation.base import Instance, SegmentationResult


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def box_iou(a, b) -> float:
    """IoU of two ``(x1, y1, x2, y2)`` boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mask_nms(instances: list[Instance], iou_threshold: float = 0.6) -> list[Instance]:
    """Drop lower-scoring duplicates of the same class (mask IoU above threshold).

    SAM 3 is queried with several phrases per class, so the same bird is often
    returned more than once.
    """
    kept: list[Instance] = []
    for inst in sorted(instances, key=lambda i: i.score, reverse=True):
        if all(
            k.class_name != inst.class_name or mask_iou(k.mask, inst.mask) < iou_threshold
            for k in kept
        ):
            kept.append(inst)
    return kept


def clean_mask(mask: np.ndarray, keep_largest: bool = True, fill_holes: bool = True,
               open_px: int = 3) -> np.ndarray:
    """Morphologically clean a boolean mask.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean mask.
    keep_largest : bool
        Keep only the largest connected component.
    fill_holes : bool
        Fill interior holes (feathers, specular highlights).
    open_px : int
        Kernel size of a morphological opening that removes speckles; 0 skips it.
    """
    m = mask.astype(np.uint8)
    if open_px > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
    if keep_largest:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 1:
            m = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    if fill_holes:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        m = np.zeros_like(m)
        cv2.drawContours(m, contours, -1, 1, thickness=cv2.FILLED)
    return m.astype(bool)


def touches_border(mask: np.ndarray, margin: int = 10) -> bool:
    """Return True when the mask comes within ``margin`` pixels of the image edge.

    The repository pipeline drops such birds because they are probably clipped.
    """
    if not mask.any():
        return False
    return bool(
        mask[:margin].any() or mask[-margin:].any() or mask[:, :margin].any()
        or mask[:, -margin:].any()
    )


def platform_region(result: SegmentationResult) -> np.ndarray:
    """Return the filled convex hull of the scale and feeder masks.

    A bird standing on the scale hides part of the platform, so the raw scale
    mask has a bird-shaped hole. The convex hull restores the platform outline.
    """
    union = result.union("scale") | result.union("feeder")
    out = np.zeros(result.image_shape, dtype=np.uint8)
    if union.any():
        pts = cv2.findNonZero(union.astype(np.uint8))
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out.astype(bool)


def on_scale_fraction(mask: np.ndarray, platform: np.ndarray) -> float:
    """Fraction of a bird mask that lies over the platform region."""
    area = mask.sum()
    return float(np.logical_and(mask, platform).sum() / area) if area else 0.0


def select_bird(result: SegmentationResult, gt_box_xyxy=None, min_on_scale: float = 0.3,
                min_box_iou: float = 0.1) -> tuple[Instance | None, dict]:
    """Choose the broiler whose weight the scale is reading.

    With a FlockLabel ground-truth box, the broiler instance with the highest
    box IoU is chosen. Without one, the broiler with the largest fraction of
    its mask over the scale / feeder platform is chosen.

    Returns
    -------
    instance : Instance or None
        The chosen bird (``None`` if nothing qualifies).
    info : dict
        Diagnostic numbers for plots and tables: ``n_candidates``,
        ``box_iou``, ``on_scale``, ``reason``.
    """
    birds = result.of_class("broiler")
    platform = platform_region(result)
    info = {"n_candidates": len(birds), "box_iou": np.nan, "on_scale": np.nan,
            "reason": "ok"}
    if not birds:
        info["reason"] = "no_broiler"
        return None, info
    if gt_box_xyxy is not None:
        ious = [box_iou(b.box_xyxy, gt_box_xyxy) for b in birds]
        k = int(np.argmax(ious))
        info["box_iou"] = ious[k]
        if ious[k] < min_box_iou:
            info["reason"] = "no_match_to_gt_box"
            return None, info
        chosen = birds[k]
    else:
        fractions = [on_scale_fraction(b.mask, platform) for b in birds]
        k = int(np.argmax(fractions))
        if fractions[k] < min_on_scale:
            info["on_scale"] = fractions[k]
            info["reason"] = "no_bird_on_scale"
            return None, info
        chosen = birds[k]
    info["on_scale"] = on_scale_fraction(chosen.mask, platform) if platform.any() else np.nan
    return chosen, info
