"""Grounded SAM 2 logic with the two models replaced by stand-ins (no downloads)."""

import numpy as np
import pytest

from broilerseg.config import PromptConfig
from broilerseg.segmentation import TEACHERS, load_teacher
from broilerseg.segmentation.grounded_sam2 import GroundedSam2Segmenter, box_nms


class FakeGroundedSam2(GroundedSam2Segmenter):
    """Skips model loading; detector and mask model are simulated."""

    def __init__(self, dets, max_side=None):
        self.prompts = PromptConfig()
        self.max_side = max_side
        self.box_nms_iou = 0.5
        self.mask_nms_iou = 0.6
        self.name = "fake grounded sam2"
        self._dets = dets
        self.seen_boxes = None

    def _detect(self, pil):
        return box_nms([dict(d) for d in self._dets], self.box_nms_iou)

    def _segment_boxes(self, pil, boxes):
        self.seen_boxes = boxes
        h, w = pil.size[1], pil.size[0]
        out = np.zeros((len(boxes), h, w), bool)
        for k, (x1, y1, x2, y2) in enumerate(boxes):
            out[k, int(y1):int(y2), int(x1):int(x2)] = True
        return out


def det(cls, box, score):
    return {"class_name": cls, "box": box, "score": score, "phrase": cls}


def test_box_nms_is_per_class():
    d = [det("broiler", [0, 0, 10, 10], 0.9), det("broiler", [1, 1, 10, 10], 0.5),
         det("scale", [0, 0, 10, 10], 0.8)]
    kept = box_nms(d)
    assert [k["score"] for k in kept] == [0.9, 0.8]


def test_segment_builds_instances_and_rescales_masks():
    dets = [det("broiler", [100, 100, 200, 160], 0.8), det("broiler", [102, 101, 200, 160], 0.6),
            det("scale", [50, 50, 300, 250], 0.9), det("feeder", [60, 60, 120, 120], 0.7)]
    seg = FakeGroundedSam2(dets, max_side=200)       # 400x300 frame is halved before inference
    res = seg.segment(np.zeros((300, 400, 3), np.uint8))
    assert res.image_shape == (300, 400)
    assert res.counts() == {"broiler": 1, "scale": 1, "feeder": 1}
    bird = res.best("broiler")
    assert bird.score == pytest.approx(0.8)
    assert bird.mask.shape == (300, 400)
    assert bird.box_xyxy == pytest.approx((200, 200, 400, 300), abs=2)  # mapped back to full size


def test_no_detections_gives_empty_result():
    res = FakeGroundedSam2([]).segment(np.zeros((50, 60, 3), np.uint8))
    assert res.instances == [] and res.image_shape == (50, 60)


def test_load_teacher_rejects_unknown_name():
    assert TEACHERS == ("grounded_sam2", "sam3")
    with pytest.raises(ValueError):
        load_teacher("sam4")
