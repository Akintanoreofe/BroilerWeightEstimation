"""YOLOv8-seg student wrapped in the same interface as the SAM 3 teacher."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from broilerseg.config import CLASS_NAMES
from broilerseg.segmentation.base import Instance, SegmentationResult, box_from_mask


class YoloSegmenter:
    """Run an Ultralytics YOLO segmentation model (``.pt``, ONNX or NCNN folder).

    Parameters
    ----------
    weights : str or Path
        Trained weights, e.g. ``runs/segment/student/weights/best.pt`` or the
        ``best_ncnn_model`` folder exported for the Raspberry Pi.
    imgsz : int
        Inference size; use the size the model was trained / exported at.
    conf : float
        Confidence threshold.
    device : str, optional
        Torch device; Ultralytics picks one if omitted.
    """

    def __init__(self, weights: str | Path, imgsz: int = 640, conf: float = 0.25,
                 device: str | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="segment")
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.name = f"YOLO ({Path(weights).name})"

    def segment(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Segment one RGB ``uint8`` image."""
        t0 = time.perf_counter()
        h, w = image_rgb.shape[:2]
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        kwargs = {"imgsz": self.imgsz, "conf": self.conf, "verbose": False,
                  "retina_masks": True}
        if self.device:
            kwargs["device"] = self.device
        res = self.model.predict(bgr, **kwargs)[0]
        instances: list[Instance] = []
        if res.masks is not None:
            names = res.names
            masks = res.masks.data.cpu().numpy()
            for m, c, s in zip(masks, res.boxes.cls.cpu().numpy(),
                               res.boxes.conf.cpu().numpy(), strict=False):
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                mask = m > 0.5
                cname = names[int(c)] if names else CLASS_NAMES[int(c)]
                instances.append(Instance(cname, mask, float(s), box_from_mask(mask), "yolo"))
        return SegmentationResult((h, w), instances, time.perf_counter() - t0, self.name)
