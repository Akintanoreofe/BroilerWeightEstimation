import math

import cv2
import numpy as np
import pytest

from broilerseg.calibration import calibrate_from_hsv, calibrate_from_mask
from broilerseg.config import FEEDER_DIAMETER_CM
from broilerseg.evaluation import best_density, regression_metrics, split_events, sweep_grid
from broilerseg.segmentation import Instance, SegmentationResult, box_from_mask, select_bird
from broilerseg.segmentation.postprocess import clean_mask, mask_nms
from broilerseg.volume import (
    align_mask,
    estimate_volume,
    implied_density,
    mask_slice_profile,
    mass_from_volume,
)
from broilerseg.yolo_dataset import mask_to_polygons, polygons_to_mask, split_by_group


def disk(h, w, c, r):
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, c, r, 1, -1)
    return m.astype(bool)


# ------------------------------------------------------------------ volume
def test_rectangle_is_one_cylinder():
    # 40 px long, 10 px tall rectangle at 0.5 cm/px -> cylinder r=2.5 cm, L=20 cm
    m = np.zeros((30, 60), bool)
    m[10:20, 10:50] = True
    est = estimate_volume(m, 0.5, density_kg_m3=1000, align=False)
    expected = math.pi * 2.5 ** 2 * 20
    assert est.volume_cm3 == pytest.approx(expected)
    assert est.mass_kg == pytest.approx(expected / 1000)
    assert est.n_slices == 40


def test_slice_count_groups_columns_like_repo():
    m = np.zeros((20, 50), bool)
    m[5:15, 5:45] = True
    p = mask_slice_profile(m, 4)
    assert list(p["thickness_px"]) == [10, 10, 10, 10]
    assert p["x_start"][0] == 5 and p["x_end"][-1] == 44
    with pytest.raises(ValueError):
        mask_slice_profile(m, 0)


def test_alignment_is_rotation_invariant():
    base = np.zeros((300, 300), np.uint8)
    cv2.ellipse(base, (150, 150), (90, 35), 0, 0, 360, 1, -1)
    rot = np.zeros_like(base)
    cv2.ellipse(rot, (150, 150), (90, 35), 40, 0, 360, 1, -1)
    v0 = estimate_volume(base.astype(bool), 0.2).volume_cm3
    v1 = estimate_volume(rot.astype(bool), 0.2).volume_cm3
    assert v1 == pytest.approx(v0, rel=0.03)
    aligned, _ = align_mask(rot)
    assert aligned.shape[1] > aligned.shape[0]  # long axis horizontal


def test_mass_and_implied_density_are_inverse():
    assert mass_from_volume(2000, 900) == pytest.approx(1.8)
    assert implied_density(1.8, 2000) == pytest.approx(900)


# ------------------------------------------------------------------ calibration
def test_calibration_from_full_and_occluded_feeder():
    m = disk(400, 400, (200, 200), 70)
    cal = calibrate_from_mask(m)
    assert cal.diameter_px == pytest.approx(141, abs=2)
    assert cal.pixel_to_cm == pytest.approx(FEEDER_DIAMETER_CM / cal.diameter_px)
    occluded = m.copy()
    occluded[150:250, 230:] = False  # a bird covers part of the rim
    assert calibrate_from_mask(occluded).diameter_px == pytest.approx(cal.diameter_px, rel=0.03)
    assert not calibrate_from_mask(np.zeros((10, 10), bool)).ok


def test_hsv_fallback_finds_orange_feeder():
    img = np.full((300, 300, 3), (60, 80, 90), np.uint8)
    cv2.circle(img, (150, 150), 60, (0, 110, 240), -1)  # BGR orange
    assert calibrate_from_hsv(img).diameter_px == pytest.approx(121, abs=3)


# ------------------------------------------------------------------ segmentation helpers
def _inst(name, mask, score=0.9):
    return Instance(name, mask, score, box_from_mask(mask))


def test_nms_and_bird_selection():
    h, w = 200, 300
    scale = np.zeros((h, w), bool)
    scale[50:150, 100:250] = True
    on = disk(h, w, (180, 100), 20)
    off = disk(h, w, (40, 40), 20)
    res = SegmentationResult((h, w), [_inst("scale", scale), _inst("broiler", on),
                                      _inst("broiler", on, 0.5), _inst("broiler", off)])
    res.instances = mask_nms(res.instances)
    assert len(res.of_class("broiler")) == 2
    bird, info = select_bird(res)
    assert bird.mask[100, 180] and info["on_scale"] > 0.9
    bird_gt, _ = select_bird(res, (20, 20, 60, 60))
    assert bird_gt.mask[40, 40]


def test_clean_mask_fills_holes_and_drops_specks():
    m = disk(100, 100, (50, 50), 30)
    m[45:55, 45:55] = False
    m[2:4, 2:4] = True
    c = clean_mask(m)
    assert c[50, 50] and not c[3, 3]


# ------------------------------------------------------------------ yolo dataset
def test_polygon_roundtrip_and_video_split():
    m = disk(200, 200, (100, 100), 50)
    polys = mask_to_polygons(m)
    back = polygons_to_mask(polys, m.shape)
    assert (back & m).sum() / (back | m).sum() > 0.95
    split = split_by_group([f"v{i}" for i in range(10)], (0.8, 0.2, 0.0))
    assert sum(v == "val" for v in split.values()) == 2
    assert split_by_group(["only"])["only"] == "train"


# ------------------------------------------------------------------ evaluation
def test_metrics_and_density_fit():
    m = regression_metrics([1, 2, 3], [1.1, 2.1, 3.1])
    assert m["bias"] == pytest.approx(0.1) and m["mae"] == pytest.approx(0.1)
    import pandas as pd

    vt = pd.DataFrame({"event_id": ["a", "b", "c", "d"], "slice_count": 10,
                       "volume_cm3": [2000, 3000, 4000, 5000],
                       "actual_kg": [1.6, 2.4, 3.2, 4.0]})
    cal, held = split_events(vt.event_id, 0.5, 0)
    assert set(cal).isdisjoint(held)
    bd = best_density(vt, cal, held)
    assert bd.best_density_kg_m3.iloc[0] == pytest.approx(800)
    grid = sweep_grid(vt, [700, 800, 900], cal, held)
    assert grid.loc[grid.held_rmse_kg.idxmin(), "density_kg_m3"] == 800


def test_weight_unit_display_only():
    import pandas as pd

    from broilerseg import viz

    df = pd.DataFrame({"actual_kg": [2.5], "pred_kg": [2.4], "cal_rmse_kg": [0.1],
                       "density_kg_m3": [900.0]})
    try:
        viz.set_weight_unit("g")
        t = viz.weight_table(df)
        assert t.actual_g.iloc[0] == 2500 and t.cal_rmse_g.iloc[0] == pytest.approx(100)
        assert t.density_kg_m3.iloc[0] == 900          # density is not a weight
        assert df.actual_kg.iloc[0] == 2.5              # original untouched
        assert viz.fmt_weight(-0.0425, signed=True) == "-42 g"
        with pytest.raises(ValueError):
            viz.set_weight_unit("lb")
    finally:
        viz.set_weight_unit("kg")
    assert viz.fmt_weight(2.5) == "2.500 kg"
