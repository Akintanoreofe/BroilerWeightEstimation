"""Mask clean-up, de-duplication and selection of the bird standing on the scale."""

from __future__ import annotations

import cv2
import numpy as np

from broilerseg.segmentation.base import Instance, SegmentationResult, box_from_mask


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


def merge_single(result: SegmentationResult, classes=("feeder",),
                 pad: float = 0.05) -> SegmentationResult:
    """Collapse classes that have one object in the scene into a single instance.

    The most confident instance is kept, together with every other instance of
    the same class whose box overlaps its box (widened by ``pad`` of its size):
    those are pieces of the same object (e.g. the feeder's outer pan and inner
    ring) and their masks are merged. Instances elsewhere are dropped.
    """
    classes = tuple(c for c in classes if len(result.of_class(c)) > 1)
    if not classes:
        return result
    keep = [i for i in result.instances if i.class_name not in classes]
    for c in classes:
        insts = result.of_class(c)
        best = max(insts, key=lambda i: (i.score, i.area))
        x0, y0, x1, y1 = best.box_xyxy
        dx, dy = pad * (x1 - x0), pad * (y1 - y0)
        grow = (x0 - dx, y0 - dy, x1 + dx, y1 + dy)
        group = [i for i in insts if i is best or box_iou(i.box_xyxy, grow) > 0]
        mask = np.logical_or.reduce([i.mask for i in group])
        keep.append(Instance(c, mask, best.score, box_from_mask(mask), best.prompt))
    return SegmentationResult(result.image_shape, keep, result.seconds, result.source)


def select_bird(result: SegmentationResult, gt_box_xyxy=None, min_box_iou: float = 0.1,
                avoid_border: bool = True) -> tuple[Instance | None, dict]:
    """Choose the broiler to estimate.

    With a FlockLabel ground-truth box (evaluation against the scale reading),
    the broiler whose box overlaps it most is chosen. Without one, the most
    confident broiler that is fully in view (not cut by the image border) is
    chosen; use :func:`select_birds` to estimate every visible bird.

    Returns
    -------
    instance : Instance or None
        The chosen bird (``None`` if nothing qualifies).
    info : dict
        ``n_candidates``, ``box_iou`` and ``reason`` for plots and tables.
    """
    birds = result.of_class("broiler")
    info = {"n_candidates": len(birds), "box_iou": np.nan, "reason": "ok"}
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
        return birds[k], info
    cands = select_birds(result) if avoid_border else birds
    if not cands:
        info["reason"] = "all_birds_cut_by_border"
        return None, info
    return max(cands, key=lambda b: b.score), info


def select_birds(result: SegmentationResult, margin: int = 2) -> list[Instance]:
    """Return every broiler fully in view (not touching the image border)."""
    return [b for b in result.of_class("broiler") if not touches_border(b.mask, margin)]
