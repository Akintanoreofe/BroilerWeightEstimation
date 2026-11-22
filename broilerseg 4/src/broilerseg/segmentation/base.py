"""Common data structures shared by every segmenter (SAM 3 teacher, YOLO student)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from broilerseg.config import CLASS_NAMES


@dataclass
class Instance:
    """One segmented object.

    Attributes
    ----------
    class_name : str
        One of :data:`broilerseg.config.CLASS_NAMES`.
    mask : numpy.ndarray
        Boolean ``(H, W)`` mask in full-image coordinates.
    score : float
        Model confidence in ``[0, 1]``.
    box_xyxy : tuple of float
        Bounding box ``(x1, y1, x2, y2)`` in pixels.
    prompt : str
        Text prompt (SAM 3) or ``"yolo"`` that produced the instance.
    """

    class_name: str
    mask: np.ndarray
    score: float
    box_xyxy: tuple[float, float, float, float]
    prompt: str = ""

    @property
    def area(self) -> int:
        """Mask area in pixels."""
        return int(self.mask.sum())

    @property
    def box_xywh(self) -> tuple[float, float, float, float]:
        """Bounding box as ``(x, y, w, h)``."""
        x1, y1, x2, y2 = self.box_xyxy
        return x1, y1, x2 - x1, y2 - y1


@dataclass
class SegmentationResult:
    """All instances found in one image, plus timing information."""

    image_shape: tuple[int, int]
    instances: list[Instance] = field(default_factory=list)
    seconds: float = 0.0
    source: str = ""

    def of_class(self, class_name: str) -> list[Instance]:
        """Return instances of ``class_name`` sorted by descending score."""
        found = [i for i in self.instances if i.class_name == class_name]
        return sorted(found, key=lambda i: i.score, reverse=True)

    def best(self, class_name: str) -> Instance | None:
        """Return the highest-scoring instance of a class, or ``None``."""
        found = self.of_class(class_name)
        return found[0] if found else None

    def union(self, class_name: str) -> np.ndarray:
        """Return the union of all masks of a class (boolean ``(H, W)``)."""
        out = np.zeros(self.image_shape, dtype=bool)
        for inst in self.of_class(class_name):
            out |= inst.mask
        return out

    def counts(self) -> dict[str, int]:
        """Return the number of instances per class."""
        return {c: len(self.of_class(c)) for c in CLASS_NAMES}

    def label_map(self, order: tuple[str, ...] = ("scale", "feeder", "broiler")) -> np.ndarray:
        """Return an integer map: 0 background, ``k`` = ``CLASS_NAMES.index(c) + 1``.

        Classes are painted in ``order`` so later classes sit on top (a bird on
        the scale stays visible).
        """
        out = np.zeros(self.image_shape, dtype=np.uint8)
        for name in order:
            out[self.union(name)] = CLASS_NAMES.index(name) + 1
        return out


@runtime_checkable
class Segmenter(Protocol):
    """Anything with ``segment(image_rgb) -> SegmentationResult``."""

    name: str

    def segment(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Segment one RGB ``uint8`` image."""
        ...


def box_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    """Tight ``(x1, y1, x2, y2)`` box around a boolean mask (zeros if empty)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
