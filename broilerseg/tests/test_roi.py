"""Stage 1 (pen region) and two-stage labelling."""

import json

import numpy as np
import pandas as pd
import pytest

from broilerseg.io import read_frame, scan_videos
from broilerseg.labeling import auto_label_videos, load_result, plan_sampling
from broilerseg.roi import (
    CroppedSegmenter,
    PenDetector,
    PenROI,
    apply_manual_rois,
    detect_pen_roi,
    find_pen_rois,
    rois_lookup,
    typical_roi,
)
from broilerseg.segmentation.base import Instance, SegmentationResult
from broilerseg.synthetic import ColorPenDetector, ColorSegmenter, make_demo_data
from broilerseg.yolo_dataset import build_yolo_dataset


def test_roi_box_fractions_and_pixels():
    roi = PenROI.from_box((960, 0, 2880, 2160), (3840, 2160))
    assert (roi.fx0, roi.fx1) == (0.25, 0.75)
    assert roi.pixels((2160, 3840)) == (960, 0, 2880, 2160)
    assert roi.pixels((480, 704)) == (176, 0, 528, 480)          # same camera, sub-stream
    with pytest.raises(ValueError):
        PenROI.from_box((960, 0, 2880, 2160))


def test_crop_paste_roundtrip_and_mask_outside():
    img = np.zeros((100, 200, 3), np.uint8)
    mask = np.zeros((100, 200), bool)
    mask[20:80, 60:160] = True
    roi = PenROI.from_mask(mask, 0.9, margin=0.0)
    assert roi.pixels(img.shape) == (60, 20, 160, 80)
    crop = roi.crop(img)
    assert crop.shape == (60, 100, 3)
    m = np.zeros(crop.shape[:2], bool)
    m[10:20, 30:40] = True
    full = roi.paste(SegmentationResult(crop.shape[:2], [Instance("broiler", m, 0.8,
                                                                  (30, 10, 40, 20))]),
                     img.shape)
    inst = full.instances[0]
    assert inst.mask.shape == (100, 200) and inst.mask[30:40, 90:100].all()
    assert inst.box_xyxy == (90, 30, 100, 40) and inst.area == 100
    tri = PenROI(0.3, 0.2, 0.8, 0.8, [(0.3, 0.2), (0.8, 0.2), (0.3, 0.8)])
    grey = tri.crop(np.full((100, 200, 3), 255, np.uint8), mask_outside=True)
    assert (grey[-1, -1] == 114).all() and (grey[1, 1] == 255).all()
    d = json.loads(json.dumps(tri.as_dict()))
    assert PenROI.from_dict(d).pixels((100, 200)) == tri.pixels((100, 200))


def test_pen_detector_swaps_and_restores_prompts():
    from broilerseg.config import PromptConfig

    class Fake:
        name = "fake"

        def __init__(self):
            self.prompts = PromptConfig()
            self.seen = None

        def segment(self, img):
            self.seen = dict(self.prompts.prompts)
            m = np.zeros(img.shape[:2], bool)
            m[10:90, 50:150] = True
            return SegmentationResult(img.shape[:2], [Instance("pen", m, 0.7, (50, 10, 150, 90),
                                                               "animal pen")])

    t = Fake()
    roi, diag = detect_pen_roi([np.zeros((100, 200, 3), np.uint8)] * 2, PenDetector(t),
                               margin=0.0)
    assert list(t.seen) == ["pen"] and "broiler" in t.prompts.prompts
    assert roi.method == "auto" and roi.pixels((100, 200)) == (50, 10, 150, 90)
    assert len(diag) == 2


def test_two_stage_labelling_on_nested_demo(tmp_path):
    p = make_demo_data(tmp_path / "d", n_events=4, frames_per_event=4, nested=True)
    vids = scan_videos(p["videos"])
    assert (vids.width == 800).all()                             # pen + 160 px of surroundings
    rois = find_pen_rois(vids, ColorPenDetector(), frames_per_video=2,
                         cache=tmp_path / "rois.csv", progress=False)
    assert len(rois) == 4 and (rois.method == "auto").all()
    assert (rois.x0 >= 130).all() and (rois.x1 >= 780).all()     # strip cut (3% margin)
    again = find_pen_rois(vids, None, cache=tmp_path / "rois.csv", progress=False)  # cached
    assert np.allclose(again.fx0, rois.fx0)
    folder = find_pen_rois(vids, ColorPenDetector(), 3, scope="folder", progress=False)
    assert (folder.method == "auto_folder").all()
    man = apply_manual_rois(rois, {"day1": (100, 0, 800, 480)})
    assert set(man[man.folder.str.startswith("day1")].x0) == {100}
    assert typical_roi(rois).method == "typical"

    plan = plan_sampling(vids, 2)
    man = auto_label_videos(p["videos"], tmp_path / "lab", ColorSegmenter(), plan=plan,
                            rois=rois, progress=False)
    assert man.cropped.all() and (man.width < 800).all() and (man.full_width == 800).all()
    res = load_result(tmp_path / "lab" / man.masks.iloc[0])
    assert res.counts()["feeder"] == 1 and res.counts()["scale"] == 1
    with pytest.raises(ValueError):                              # no mixing crop settings
        auto_label_videos(p["videos"], tmp_path / "lab", ColorSegmenter(), plan=plan,
                          progress=False)
    ds = build_yolo_dataset(tmp_path / "lab", tmp_path / "yolo")
    assert np.nanmin(ds.fidelity_broiler) > 0.9

    # the wrapper gives the same objects, in full-frame coordinates
    frame = read_frame(vids.path.iloc[0], 1)
    roi = rois_lookup(rois)[vids.video_id.iloc[0]]
    wrapped = CroppedSegmenter(ColorSegmenter(), roi)
    full = wrapped.segment(frame)
    assert full.image_shape == frame.shape[:2]
    fx = full.best("feeder").box_xyxy[0]
    auto = CroppedSegmenter(ColorSegmenter(), detector=ColorPenDetector()).segment(frame)
    assert abs(auto.best("feeder").box_xyxy[0] - fx) < 2
    assert isinstance(pd.DataFrame(rois).polygon.iloc[0], str)


def test_load_student_wraps_pen_crop(tmp_path, monkeypatch):
    from broilerseg import distill
    from broilerseg.segmentation import yolo

    class FakeYolo:
        def __init__(self, weights, imgsz=640, conf=0.25, device=None):
            self.name, self.imgsz = "fake", imgsz

    monkeypatch.setattr(yolo, "YoloSegmenter", FakeYolo)
    (tmp_path / "best.pt").write_bytes(b"x")
    card = {"imgsz": 480, "pen_crop": True,
            "pen_roi": PenROI(0.2, 0.0, 0.9, 1.0).as_dict()}
    (tmp_path / "model_card.json").write_text(json.dumps(card))
    seg, _ = distill.load_student(tmp_path)
    assert isinstance(seg, CroppedSegmenter) and seg.imgsz == 480   # attribute pass-through
    seg2, _ = distill.load_student(tmp_path, pen_crop=False)
    assert isinstance(seg2, FakeYolo)
