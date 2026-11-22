"""End-to-end checks on synthetic videos (no SAM 3 / YOLO needed)."""

import json

import numpy as np
import pytest

from broilerseg.experiments import WeightExperiment
from broilerseg.io import load_flocklabel_export, scan_videos, subsample_per_event
from broilerseg.labeling import auto_label_videos, load_result, plan_sampling
from broilerseg.synthetic import ColorSegmenter, make_demo_data
from broilerseg.yolo_dataset import build_yolo_dataset, read_yolo_label


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    return make_demo_data(tmp_path_factory.mktemp("demo"), n_events=4, frames_per_event=6)


def test_flocklabel_export_and_weight_experiment(demo):
    exp = load_flocklabel_export(demo["export"].parent)  # parent folder also works
    assert exp.kind == "evaluation" and len(exp.samples) == 24
    assert len(subsample_per_event(exp.samples, 3)) == 12
    rec = WeightExperiment(ColorSegmenter()).run(exp.samples)
    assert (rec.status == "ok").all()
    true_px = json.loads((demo["export"] / "export_info.json").read_text())["true_pixel_to_cm"]
    assert rec.pixel_to_cm.median() == pytest.approx(true_px, rel=0.01)
    # synthetic birds are ellipsoids with depth 0.8*width at 1000 kg/m^3 -> ~800 implied
    assert 700 < rec.implied_density.median() < 900


def test_label_build_and_read_back(demo, tmp_path):
    vids = scan_videos(demo["videos"])
    plan = plan_sampling(vids, 3)
    man = auto_label_videos(demo["videos"], tmp_path / "labels", ColorSegmenter(), plan=plan,
                            progress=False)
    assert len(man) == 12
    res = load_result(tmp_path / "labels" / man.masks.iloc[0])
    assert res.counts() == {"broiler": 2, "scale": 1, "feeder": 1}
    again = auto_label_videos(demo["videos"], tmp_path / "labels", ColorSegmenter(), plan=plan,
                              progress=False)
    assert len(again) == 12  # resumed, nothing duplicated
    ds = build_yolo_dataset(tmp_path / "labels", tmp_path / "yolo")
    assert set(ds.split) == {"train", "val"}
    assert ds.groupby("video_stem").split.nunique().max() == 1  # split by video
    assert np.nanmin(ds.fidelity_broiler) > 0.9
    r = ds.iloc[0]
    lines = read_yolo_label(tmp_path / "yolo" / "labels" / r.split / f"{r.stem}.txt", 640, 480)
    assert sorted({c for c, _ in lines}) == [0, 1, 2]


def test_nested_folders_unique_ids_and_folder_split(tmp_path):
    p = make_demo_data(tmp_path / "nest", n_events=6, frames_per_event=4, nested=True)
    vids = scan_videos(p["videos"])
    assert len(vids) == 6 and vids.name.nunique() == 1          # every file is called ch01.mp4
    assert vids.video_id.nunique() == 6 and vids.folder.nunique() == 6
    assert "day1__cam1__ch01" in set(vids.video_id)
    plan = plan_sampling(vids, 2)
    man = auto_label_videos(p["videos"], tmp_path / "lab", ColorSegmenter(), plan=plan,
                            progress=False)
    assert man.stem.nunique() == 12                              # no frame overwritten another
    ds = build_yolo_dataset(tmp_path / "lab", tmp_path / "yolo", split_by="folder")
    assert ds.groupby("split_group").split.nunique().max() == 1


def test_save_student_card_without_weights(tmp_path):
    import json

    from broilerseg.distill import save_student

    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    (run / "results.csv").write_text("epoch,metrics/mAP50-95(M)\n1,0.5\n")
    (run / "results.png").write_bytes(b"png")
    out = save_student(run, tmp_path / "models", "demo", card={"imgsz": 320, "teacher": "x"})
    card = json.loads((out / "model_card.json").read_text())
    assert card["imgsz"] == 320 and card["weights"] is None
    assert (out / "results.csv").is_file() and (out / "training" / "results.png").is_file()
