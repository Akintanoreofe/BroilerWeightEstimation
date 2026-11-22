"""Grounded SAM 2 teacher: Grounding DINO finds boxes from text, SAM 2.1 draws the masks.

Both models are public on Hugging Face (no access request) and Apache-2.0
licensed. The pipeline is the published "Grounded SAM" recipe:

1. Grounding DINO (open-set detector) turns each class's text phrases into
   boxes with confidence scores.
2. Duplicate boxes of a class are removed with box NMS.
3. SAM 2.1 receives every box at once (one image encoding) and returns one
   mask per box.
4. Duplicate masks are removed with mask NMS, exactly as for SAM 3.

The class has the same interface as :class:`Sam3Segmenter`, so every notebook
cell, the labeller and the weight experiment work unchanged.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from broilerseg.config import PromptConfig
from broilerseg.segmentation.base import Instance, SegmentationResult, box_from_mask
from broilerseg.segmentation.postprocess import box_iou, mask_nms
from broilerseg.segmentation.sam3 import pick_device

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

#: Grounding DINO checkpoints (``tiny`` is faster, ``base`` is more accurate).
DINO_MODELS = {"tiny": "IDEA-Research/grounding-dino-tiny",
               "base": "IDEA-Research/grounding-dino-base"}
#: SAM 2.1 checkpoints, smallest to largest.
SAM2_MODELS = {"tiny": "facebook/sam2.1-hiera-tiny", "small": "facebook/sam2.1-hiera-small",
               "base_plus": "facebook/sam2.1-hiera-base-plus",
               "large": "facebook/sam2.1-hiera-large"}


def box_nms(dets: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Keep the highest-scoring box among overlapping boxes of the same class."""
    kept: list[dict] = []
    for d in sorted(dets, key=lambda d: d["score"], reverse=True):
        if all(k["class_name"] != d["class_name"] or box_iou(k["box"], d["box"]) < iou_threshold
               for k in kept):
            kept.append(d)
    return kept


class GroundedSam2Segmenter:
    """Text-prompted broiler / scale / feeder segmentation without gated models.

    Parameters
    ----------
    prompts : PromptConfig, optional
        Phrases per class (shared with SAM 3). ``prompts.mask_threshold`` is
        not used; detection thresholds are set below.
    dino : str
        ``"tiny"``, ``"base"`` or a Hugging Face model id.
    sam2 : str
        ``"tiny"``, ``"small"``, ``"base_plus"``, ``"large"`` or a model id.
    box_threshold : float
        Minimum Grounding DINO box confidence.
    text_threshold : float
        Minimum phrase-matching confidence inside Grounding DINO.
    device : str, optional
        ``"cuda"``, ``"mps"`` or ``"cpu"``; picked automatically if omitted.
    max_side : int or None
        Downscale frames whose longest side is larger before inference.
    box_nms_iou, mask_nms_iou : float
        Duplicate-removal thresholds.
    """

    def __init__(self, prompts: PromptConfig | None = None, dino: str = "base",
                 sam2: str = "large", box_threshold: float = 0.3,
                 text_threshold: float = 0.25, device: str | None = None,
                 max_side: int | None = 1280, box_nms_iou: float = 0.5,
                 mask_nms_iou: float = 0.6) -> None:
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2Model,
            Sam2Processor,
        )

        self.torch = torch
        self.device = pick_device(device)
        self.prompts = prompts or PromptConfig()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_side = max_side
        self.box_nms_iou = box_nms_iou
        self.mask_nms_iou = mask_nms_iou
        dino_id = DINO_MODELS.get(dino, dino)
        sam2_id = SAM2_MODELS.get(sam2, sam2)
        self.dino_processor = AutoProcessor.from_pretrained(dino_id)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id)
        self.dino = self.dino.to(self.device).eval()
        self.sam_processor = Sam2Processor.from_pretrained(sam2_id)
        self.sam = Sam2Model.from_pretrained(sam2_id).to(self.device).eval()
        self.name = f"Grounded SAM 2 ({dino_id.split('/')[-1]} + {sam2_id.split('/')[-1]})"

    # ------------------------------------------------------------------ steps
    def _detect(self, pil) -> list[dict]:
        """Grounding DINO boxes for every class (one text query per class)."""
        torch = self.torch
        dets = []
        for class_name, phrases in self.prompts.prompts.items():
            text = ". ".join(p.lower().strip(". ") for p in phrases) + "."
            inputs = self.dino_processor(images=pil, text=text,
                                         return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.dino(**inputs)
            kwargs = {"text_threshold": self.text_threshold,
                      "target_sizes": [pil.size[::-1]]}
            try:  # newer transformers
                res = self.dino_processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=self.box_threshold, **kwargs)[0]
            except TypeError:  # older transformers
                res = self.dino_processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=self.box_threshold, **kwargs)[0]
            labels = res.get("text_labels", res.get("labels", [""] * len(res["scores"])))
            for box, score, label in zip(res["boxes"], res["scores"], labels, strict=False):
                dets.append({"class_name": class_name,
                             "box": [float(v) for v in box.tolist()],
                             "score": float(score), "phrase": str(label) or text})
        return box_nms(dets, self.box_nms_iou)

    def _segment_boxes(self, pil, boxes: list[list[float]]) -> np.ndarray:
        """SAM 2.1 masks for all boxes at once; returns ``(N, H, W)`` bool."""
        torch = self.torch
        inputs = self.sam_processor(images=pil, input_boxes=[boxes],
                                    return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam(**inputs, multimask_output=False)
        masks = self.sam_processor.post_process_masks(outputs.pred_masks.cpu(),
                                                      inputs["original_sizes"])[0]
        masks = masks.numpy() if hasattr(masks, "numpy") else np.asarray(masks)
        return masks.reshape(len(boxes), *masks.shape[-2:]) > 0

    # ------------------------------------------------------------------ public
    def segment(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Segment one RGB ``uint8`` frame into broiler / scale / feeder instances."""
        from PIL import Image

        t0 = time.perf_counter()
        full_h, full_w = image_rgb.shape[:2]
        scale = 1.0
        small = image_rgb
        if self.max_side and max(full_h, full_w) > self.max_side:
            scale = self.max_side / max(full_h, full_w)
            small = cv2.resize(image_rgb, (round(full_w * scale), round(full_h * scale)),
                               interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(small)
        dets = self._detect(pil)
        instances: list[Instance] = []
        if dets:
            masks = self._segment_boxes(pil, [d["box"] for d in dets])
            for d, m in zip(dets, masks, strict=False):
                m = m.astype(np.uint8)
                if m.shape != (full_h, full_w):
                    m = cv2.resize(m, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
                m = m.astype(bool)
                if m.any():
                    instances.append(Instance(d["class_name"], m, d["score"], box_from_mask(m),
                                              d["phrase"]))
        instances = mask_nms(instances, self.mask_nms_iou)
        return SegmentationResult((full_h, full_w), instances, time.perf_counter() - t0,
                                  self.name)
