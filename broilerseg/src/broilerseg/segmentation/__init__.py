"""Segmenters (SAM 3 teacher, YOLO student) and mask post-processing.

``Sam3Segmenter`` and ``YoloSegmenter`` import heavy libraries (torch,
transformers, ultralytics) only when they are instantiated, so the rest of the
package works without them.
"""

from broilerseg.segmentation.base import Instance, SegmentationResult, Segmenter, box_from_mask
from broilerseg.segmentation.postprocess import (
    box_iou,
    clean_mask,
    mask_iou,
    mask_nms,
    on_scale_fraction,
    platform_region,
    select_bird,
    touches_border,
)


def __getattr__(name):  # lazy heavy imports
    if name == "Sam3Segmenter":
        from broilerseg.segmentation.sam3 import Sam3Segmenter

        return Sam3Segmenter
    if name == "YoloSegmenter":
        from broilerseg.segmentation.yolo import YoloSegmenter

        return YoloSegmenter
    raise AttributeError(name)


__all__ = [
    "Instance", "SegmentationResult", "Segmenter", "box_from_mask", "box_iou", "clean_mask",
    "mask_iou", "mask_nms", "on_scale_fraction", "platform_region", "select_bird",
    "touches_border", "Sam3Segmenter", "YoloSegmenter",
]
