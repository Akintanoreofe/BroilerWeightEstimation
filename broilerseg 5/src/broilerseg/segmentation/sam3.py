"""SAM 3 teacher: text-prompted segmentation of broiler, scale and feeder.

Uses the Hugging Face ``transformers`` implementation of SAM 3 because Meta's
reference repository needs CUDA/Triton, while the ``transformers`` port runs on
CUDA, Apple Silicon (MPS) and CPU. The checkpoint ``facebook/sam3`` is gated:
request access on its Hugging Face page, then run ``hf auth login`` once.
"""

from __future__ import annotations

import os
import time

import numpy as np

from broilerseg.config import PromptConfig
from broilerseg.segmentation.base import Instance, SegmentationResult, box_from_mask
from broilerseg.segmentation.postprocess import mask_nms

# Let PyTorch fall back to CPU for the few operators MPS lacks.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device(prefer: str | None = None) -> str:
    """Return ``prefer`` or the best available torch device: cuda > mps > cpu."""
    if prefer:
        return prefer
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Sam3Segmenter:
    """Segment broilers, the scale and the feeder with SAM 3 text prompts.

    Parameters
    ----------
    model_id : str
        Hugging Face model id or a local folder holding the checkpoint.
    prompts : PromptConfig, optional
        Phrases per class and thresholds.
    device : str, optional
        ``"cuda"``, ``"mps"`` or ``"cpu"``; picked automatically if omitted.
    max_side : int or None
        Downscale frames whose longest side exceeds this before inference
        (masks are returned at full resolution). Speeds up 4K CCTV frames.
    nms_iou : float
        Mask IoU above which duplicates of the same class are merged.

    Examples
    --------
    >>> seg = Sam3Segmenter()                       # doctest: +SKIP
    >>> result = seg.segment(frame_rgb)             # doctest: +SKIP
    >>> result.counts()                             # doctest: +SKIP
    {'broiler': 1, 'scale': 1, 'feeder': 1}
    """

    name = "SAM 3"

    def __init__(self, model_id: str = "facebook/sam3", prompts: PromptConfig | None = None,
                 device: str | None = None, max_side: int | None = 1280,
                 nms_iou: float = 0.6) -> None:
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = pick_device(device)
        self.prompts = prompts or PromptConfig()
        self.max_side = max_side
        self.nms_iou = nms_iou
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()

    # ------------------------------------------------------------------ helpers
    def _resize(self, image_rgb: np.ndarray) -> tuple[np.ndarray, float]:
        import cv2

        h, w = image_rgb.shape[:2]
        if not self.max_side or max(h, w) <= self.max_side:
            return image_rgb, 1.0
        s = self.max_side / max(h, w)
        return cv2.resize(image_rgb, (round(w * s), round(h * s)),
                          interpolation=cv2.INTER_AREA), s

    def _run_prompt(self, img_inputs, vision_embeds, phrase: str):
        torch = self.torch
        with torch.no_grad():
            if vision_embeds is not None:
                text_inputs = self.processor(text=phrase, return_tensors="pt").to(self.device)
                outputs = self.model(vision_embeds=vision_embeds, **text_inputs)
            else:  # older transformers without get_vision_features
                inputs = self.processor(images=img_inputs["_pil"], text=phrase,
                                        return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
        return self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.prompts.score_threshold,
            mask_threshold=self.prompts.mask_threshold,
            target_sizes=img_inputs["original_sizes"].tolist(),
        )[0]

    # ------------------------------------------------------------------ public
    def segment(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Segment one RGB ``uint8`` frame into broiler / scale / feeder instances."""
        import cv2
        from PIL import Image

        t0 = time.perf_counter()
        full_h, full_w = image_rgb.shape[:2]
        small, _ = self._resize(image_rgb)
        pil = Image.fromarray(small)
        img_inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        img_inputs["_pil"] = pil
        vision_embeds = None
        if hasattr(self.model, "get_vision_features"):
            with self.torch.no_grad():
                vision_embeds = self.model.get_vision_features(
                    pixel_values=img_inputs["pixel_values"])

        instances: list[Instance] = []
        for class_name, phrases in self.prompts.prompts.items():
            for phrase in phrases:
                res = self._run_prompt(img_inputs, vision_embeds, phrase)
                for m, s in zip(res["masks"], res["scores"], strict=False):
                    mask = m.detach().cpu().numpy().astype(np.uint8)
                    if mask.shape != (full_h, full_w):
                        mask = cv2.resize(mask, (full_w, full_h),
                                          interpolation=cv2.INTER_NEAREST)
                    mask = mask.astype(bool)
                    if mask.any():
                        instances.append(Instance(class_name, mask, float(s),
                                                  box_from_mask(mask), phrase))
        instances = mask_nms(instances, self.nms_iou)
        return SegmentationResult((full_h, full_w), instances,
                                  time.perf_counter() - t0, self.name)
