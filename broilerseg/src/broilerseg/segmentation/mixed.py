"""Use a different teacher per class (e.g. SAM 3 for birds, Grounded SAM 2 for the scale)."""

from __future__ import annotations

import time

import numpy as np

from broilerseg.config import PromptConfig
from broilerseg.segmentation.base import SegmentationResult, Segmenter


class MixedTeacher:
    """Route each class to the teacher that finds it best and merge the results.

    Parameters
    ----------
    teachers : dict
        ``class name -> segmenter``, e.g. ``{"broiler": sam3, "feeder": gsam2}``.
        One segmenter may serve several classes;
        it then runs once per image with just those classes' prompts.
    prompts : PromptConfig, optional
        The phrases for every class (each teacher's own prompts if omitted).
    default : str, optional
        Class whose teacher also handles classes not listed in ``teachers``
        (e.g. ``"pen"`` for stage 1). Defaults to the first class.

    The object behaves like a single teacher: ``segment``, ``name``,
    ``device`` and a swappable ``prompts`` (so :class:`broilerseg.roi.PenDetector`
    and prompt tests work unchanged).
    """

    def __init__(self, teachers: dict[str, Segmenter], prompts: PromptConfig | None = None,
                 default: str | None = None) -> None:
        if not teachers:
            raise ValueError("teachers is empty")
        self.teachers = dict(teachers)
        self.default = default or next(iter(self.teachers))
        if prompts is None:
            merged: dict[str, list[str]] = {}
            for cls, seg in self.teachers.items():
                merged[cls] = list(seg.prompts.prompts.get(cls, [cls]))
            first = self.teachers[self.default].prompts
            prompts = PromptConfig(prompts=merged, score_threshold=first.score_threshold,
                                   mask_threshold=first.mask_threshold)
        self.prompts = prompts
        parts = {}
        for cls, seg in self.teachers.items():
            parts.setdefault(seg.name, []).append(cls)
        self.name = "mixed: " + "; ".join(f"{n} -> {', '.join(c)}" for n, c in parts.items())
        self.device = getattr(self.teachers[self.default], "device", "cpu")

    def teacher_for(self, cls: str) -> Segmenter:
        """Return the segmenter used for class ``cls``."""
        return self.teachers.get(cls, self.teachers[self.default])

    def segment(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Run each teacher once on its own classes and merge the instances."""
        t0 = time.perf_counter()
        groups: dict[int, tuple[Segmenter, list[str]]] = {}
        for cls in self.prompts.prompts:
            seg = self.teacher_for(cls)
            groups.setdefault(id(seg), (seg, []))[1].append(cls)
        insts = []
        for seg, classes in groups.values():
            old = seg.prompts
            seg.prompts = PromptConfig(prompts={c: self.prompts.prompts[c] for c in classes},
                                       single_instance=self.prompts.single_instance,
                                       score_threshold=self.prompts.score_threshold,
                                       mask_threshold=self.prompts.mask_threshold)
            try:
                insts += seg.segment(image_rgb).instances
            finally:
                seg.prompts = old
        return SegmentationResult(image_rgb.shape[:2], insts, time.perf_counter() - t0, self.name)
