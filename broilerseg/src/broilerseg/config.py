"""Project-wide constants and small configuration dataclasses.

Everything that a notebook may want to change lives here so experiments stay
reproducible: class names, SAM 3 text prompts, the feeder diameter used for
pixel-to-centimetre calibration, and the default density / slice settings of
the slice-based volume model taken from the BroilerWeightEstimation repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Feeder (feeding pan) diameter: 14 inches, as in ``src/FeederCalibration.py``.
FEEDER_DIAMETER_CM: float = 14 * 2.54  # 35.56 cm

#: Stage-1 text prompts for the pen region (two-stage labelling, see ``roi.py``).
#: SAM 3 and Grounding DINO respond best to short noun phrases; edit in a notebook.
PEN_PROMPTS: tuple[str, ...] = ("animal pen", "wood shavings bedding", "fenced enclosure")

#: Segmentation classes, in YOLO class-id order.
CLASS_NAMES: tuple[str, ...] = ("broiler", "scale", "feeder")

#: Default broiler density of the repository's calculator (kg / m^3).
DEFAULT_DENSITY_KG_M3: float = 900.0

#: Fixed RGB colours (0-1 floats) used by every plot, one per class.
CLASS_COLORS: dict[str, tuple[float, float, float]] = {
    "broiler": (0.98, 0.75, 0.18),  # warm yellow
    "scale": (0.20, 0.55, 0.90),  # blue
    "feeder": (0.90, 0.30, 0.25),  # orange-red
}


@dataclass
class PromptConfig:
    """SAM 3 text prompts for each class.

    SAM 3 segments *concepts* from short noun phrases. Several phrases per class
    are tried and their detections pooled, which helps with unusual viewpoints.
    Edit these lists in a notebook if a class is missed.
    """

    prompts: dict[str, list[str]] = field(
        default_factory=lambda: {
            "broiler": ["chicken", "broiler chicken"],
            "scale": ["weighing scale platform", "square platform"],
            "feeder": ["orange feeder pan", "round feeding bowl"],
        }
    )
    score_threshold: float = 0.4  # SAM 3 only; Grounded SAM 2 uses box_threshold
    mask_threshold: float = 0.5


@dataclass
class VolumeConfig:
    """Settings of the slice-based (stacked cylinder) volume model.

    Attributes
    ----------
    density_kg_m3 : float
        Broiler density used to turn volume into mass.
    slice_count : int or None
        Number of slices; ``None`` uses one slice per occupied pixel column,
        which is the repository default.
    align : bool
        Rotate the mask to its body axis before slicing (repository behaviour).
    """

    density_kg_m3: float = DEFAULT_DENSITY_KG_M3
    slice_count: int | None = None
    align: bool = True
