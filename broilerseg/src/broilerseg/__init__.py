"""broilerseg: SAM 3 labelling, YOLOv8-seg distillation and broiler weight estimation.

Modules
-------
config          constants (classes, prompts, feeder diameter, density)
io              video folders and FlockLabel exports
segmentation    teachers (Grounded SAM 2, SAM 3), YOLO student, mask post-processing
calibration     pixel-to-cm from the feeder
volume          slice-based (stacked cylinder) volume and mass
experiments     end-to-end weight experiment on FlockLabel data
evaluation      metrics, event tables, density x slice sweeps
massmodel       one fitted mass equation m = a V^b with shape / calibration settings
labeling        auto-label video frames with SAM 3
yolo_dataset    SAM 3 masks -> YOLO-seg dataset
distill         train / compare / export the student
viz             every plot
synthetic       demo data and a colour segmenter for dry runs
"""

from broilerseg.calibration import Calibration, calibrate_from_hsv, calibrate_from_mask
from broilerseg.config import (
    CLASS_NAMES,
    DEFAULT_DENSITY_KG_M3,
    FEEDER_DIAMETER_CM,
    PromptConfig,
    VolumeConfig,
)
from broilerseg.volume import VolumeEstimate, estimate_volume, mask_slice_profile

__version__ = "0.5.0"

__all__ = [
    "CLASS_NAMES", "DEFAULT_DENSITY_KG_M3", "FEEDER_DIAMETER_CM", "PromptConfig", "VolumeConfig",
    "Calibration", "calibrate_from_hsv", "calibrate_from_mask", "VolumeEstimate",
    "estimate_volume", "mask_slice_profile", "__version__",
]
