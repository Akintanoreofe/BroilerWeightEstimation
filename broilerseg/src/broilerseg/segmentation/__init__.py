"""Segmenters (SAM 3 / Grounded SAM 2 teachers, YOLO student) and mask post-processing.

The model classes import heavy libraries (torch, transformers, ultralytics)
only when they are instantiated, so the rest of the package works without them.
"""

from broilerseg.segmentation.base import Instance, SegmentationResult, Segmenter, box_from_mask
from broilerseg.segmentation.postprocess import (
    box_iou,
    clean_mask,
    mask_iou,
    mask_nms,
    merge_single,
    select_bird,
    select_birds,
    touches_border,
)

#: Teacher names accepted by :func:`load_teacher`.
TEACHERS = ("grounded_sam2", "sam3")


def load_mixed_teacher(mapping: dict[str, str], prompts=None, default: str | None = None,
                       per_teacher: dict | None = None, **kwargs):
    """Load a :class:`~broilerseg.segmentation.mixed.MixedTeacher`.

    Parameters
    ----------
    mapping : dict
        ``class -> teacher name``, e.g.
        ``{"broiler": "sam3", "feeder": "grounded_sam2"}``.
        Each named teacher is loaded once.
    prompts : PromptConfig, optional
        Phrases for every class.
    default : str, optional
        Class whose teacher handles other classes (stage-1 ``"pen"``).
    per_teacher : dict, optional
        Extra arguments for one teacher only, e.g.
        ``{"grounded_sam2": {"box_threshold": 0.25}}``.
    **kwargs
        Passed to every teacher (``max_side``, ``device`` ...).
    """
    from broilerseg.segmentation.mixed import MixedTeacher

    loaded = {}
    for name in dict.fromkeys(mapping.values()):
        loaded[name] = load_teacher(name, prompts=prompts, **kwargs,
                                    **(per_teacher or {}).get(name, {}))
    return MixedTeacher({c: loaded[n] for c, n in mapping.items()}, prompts, default)


def load_teacher(name: str = "sam3", **kwargs):
    """Create a teacher segmenter by name.

    Parameters
    ----------
    name : {"sam3", "grounded_sam2"}
        ``sam3`` (default): Meta's SAM 3, gated (needs approved access).
        ``grounded_sam2``: Grounding DINO + SAM 2.1, public models, no access
        request. Both work with the two-stage pen crop (:mod:`broilerseg.roi`).
    **kwargs
        Passed to the segmenter (e.g. ``prompts``, ``device``, ``max_side``,
        and for Grounded SAM 2 ``dino="tiny"``, ``sam2="base_plus"``,
        ``box_threshold``).
    """
    if name == "grounded_sam2":
        from broilerseg.segmentation.grounded_sam2 import GroundedSam2Segmenter

        return GroundedSam2Segmenter(**kwargs)
    if name == "sam3":
        from broilerseg.segmentation.sam3 import Sam3Segmenter

        return Sam3Segmenter(**kwargs)
    raise ValueError(f"teacher must be one of {TEACHERS}, got {name!r}")


def __getattr__(name):  # lazy heavy imports
    if name == "Sam3Segmenter":
        from broilerseg.segmentation.sam3 import Sam3Segmenter

        return Sam3Segmenter
    if name == "GroundedSam2Segmenter":
        from broilerseg.segmentation.grounded_sam2 import GroundedSam2Segmenter

        return GroundedSam2Segmenter
    if name == "MixedTeacher":
        from broilerseg.segmentation.mixed import MixedTeacher

        return MixedTeacher
    if name == "YoloSegmenter":
        from broilerseg.segmentation.yolo import YoloSegmenter

        return YoloSegmenter
    raise AttributeError(name)


__all__ = [
    "Instance", "SegmentationResult", "Segmenter", "box_from_mask", "box_iou", "clean_mask",
    "mask_iou", "mask_nms", "merge_single", "select_bird", "select_birds",
    "touches_border", "load_teacher", "load_mixed_teacher", "TEACHERS", "MixedTeacher",
    "Sam3Segmenter", "GroundedSam2Segmenter",
    "YoloSegmenter",
]
