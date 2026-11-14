"""Tests for the Qt-free core: file-name dates, import, labels and both exporters.

Run from the flocklabel folder:  python -m pytest -q
Demo footage is generated once into a temporary folder. Set FLOCKLABEL_REAL_VIDEO
to one of your own videos to also run the real-file test.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flocklabel.core.db import Project  # noqa: E402
from flocklabel.core.export import export_dataset, export_evaluation, split_by_video  # noqa: E402
from flocklabel.core.parser import parse_name  # noqa: E402
from flocklabel.core.video import iter_frames, probe, scan_folder  # noqa: E402

REAL = os.environ.get("FLOCKLABEL_REAL_VIDEO")


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo_videos")
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_demo_data.py"), str(out)],
                   check=True, capture_output=True)
    return out


@pytest.fixture()
def project(tmp_path, demo_dir):
    p = Project(tmp_path / "proj")
    rep = scan_folder(p, demo_dir)
    assert rep.added == 13 and rep.no_date_in_name == ["misc_export_0001.mp4"]
    yield p
    p.close()


def bird_box(size):
    cx, cy = 610, 495
    return {"x1": cx - 95 * size, "y1": cy - 62 * size, "x2": cx + 0.95 * 95 * size + 26 * size, "y2": cy + 62 * size}


def label_demo(project, demo_dir, n=6):
    """Label n frames per demo video with the box and the reading shown on screen."""
    truth = json.loads((demo_dir / "demo_ground_truth.json").read_text())["clips"]
    for t in truth:
        cid = project.conn.execute("SELECT id FROM clips WHERE rel_path=?", (t["file"],)).fetchone()["id"]
        on, _ = t["bird_on_scale_frames"]
        size = 0.85 + (t["weight_g"] - 1800) / 1400 * 0.35
        for k, f in enumerate(range(on, on + n)):
            # readings wobble by a few grams frame to frame, as a real scale does
            project.set_label(cid, f, dict(bird_box(size), weight_g=t["weight_g"] + (k % 3) - 1, time_ms=f * 100.0))
    return truth


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("name, start, end", [
    ("2025-10-01_16-59-45_to_16-59-47__ch00__SWF_1000381605376.mp4", "2025-10-01 16:59:45", "2025-10-01 16:59:47"),
    ("00d5b8aa-2025-10-01_16-59-45_to_16-59-47__ch00__SWF_1000381605376.mp4", "2025-10-01 16:59:45", "2025-10-01 16:59:47"),
    ("2025-10-01_23-59-58_to_00-00-03__ch00.mp4", "2025-10-01 23:59:58", "2025-10-02 00:00:03"),
    ("ch01_20251014073000_20251014080000.mp4", "2025-10-14 07:30:00", "2025-10-14 08:00:00"),
    ("NVR 2025-10-14 07.30.00.mp4", "2025-10-14 07:30:00", None),
])
def test_dates_from_names(name, start, end):
    p = parse_name(name)
    assert str(p.start) == start
    assert (str(p.end) if p.end else None) == end


def test_names_without_dates():
    assert parse_name("video.mp4") is None
    assert parse_name("2025-13-45_99-99-99.mp4") is None      # impossible date is not mis-read
    assert parse_name("cam-20251014073000.mp4", r"cam-(?P<start>\d{14})").start.hour == 7


# ------------------------------------------------------------- import/labels
def test_import_and_calendar(project):
    days = project.days_with_footage()
    assert {"2025-10-14", "2025-10-15", "2025-10-16"} <= set(days)
    clips = project.clips(day="2025-10-14")
    assert len(clips) == 4
    assert clips[0]["start_local"] == "2025-10-14T07:40:45" and clips[0]["end_local"] == "2025-10-14T07:40:51"
    assert clips[0]["n_frames"] == 60 and clips[0]["width"] == 1280


def test_labels_one_per_frame_and_status(project, demo_dir):
    label_demo(project, demo_dir)
    cid = project.clips(day="2025-10-14")[0]["id"]
    labs = project.labels(cid)
    assert len(labs) == 6
    f = min(labs)
    project.set_label(cid, f, dict(labs[f], x1=10, x2=5))          # replace, and coords get ordered
    assert project.label(cid, f)["x1"] == 5 and len(project.labels(cid)) == 6
    project.set_label(cid, f, None)
    assert len(project.labels(cid)) == 5
    assert project.days_with_footage()["2025-10-14"] == "some"
    for c in project.clips(day="2025-10-14"):
        project.set_clip_status(c["id"], "done")
    assert project.days_with_footage()["2025-10-14"] == "done"


def test_unweighed_and_skipped_are_not_exported(project, demo_dir, tmp_path):
    label_demo(project, demo_dir)
    cid = project.clips(day="2025-10-14")[0]["id"]
    f = min(project.labels(cid))
    project.set_label(cid, f, dict(project.label(cid, f), weight_g=None))
    rows = project.export_rows()
    assert len(rows) == 12 * 6 - 1
    project.set_clip_status(cid, "skip")
    assert len(project.export_rows()) == 11 * 6
    assert len(project.export_rows(only_done=True)) == 0
    assert {r["start_local"][:10] for r in project.export_rows(date_from="2025-10-16")} == {"2025-10-16"}


def test_split_is_by_video():
    s = split_by_video(range(20), seed=3)
    assert sorted(s) == list(range(20)) and set(s.values()) == {"train", "val", "test"}


# ----------------------------------------------------------------- exports
def test_export_dataset(project, demo_dir, tmp_path):
    truth = label_demo(project, demo_dir)
    counts = export_dataset(project, tmp_path / "ds", max_frames_per_video=4)
    assert counts == {"videos": 12, "images": 48, "splits": {"train": 8, "val": 2, "test": 2}}
    with open(tmp_path / "ds" / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    by_video = {}
    for r in rows:
        by_video.setdefault(r["video"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_video.values())
    first = [r for r in rows if r["video"] == truth[0]["file"]]
    assert float(first[0]["video_median_weight_g"]) == pytest.approx(truth[0]["weight_g"], abs=1)
    coco = json.loads((tmp_path / "ds" / "annotations_coco.json").read_text())
    assert len(coco["annotations"]) == 48 and "weight_g" in coco["annotations"][0]["attributes"]
    lab = next((tmp_path / "ds" / "labels").rglob("*.txt")).read_text().split()
    assert lab[0] == "0" and all(0 < float(v) <= 1 for v in lab[1:])
    seq = json.loads(next((tmp_path / "ds" / "sequences").glob("*.json")).read_text())
    assert len(seq["frames"]) == 4


def test_export_evaluation(project, demo_dir, tmp_path):
    truth = label_demo(project, demo_dir)
    project.crop = (300, 300, 940, 380)
    counts = export_evaluation(project, tmp_path / "ev", date_from="2025-10-14", date_to="2025-10-14")
    assert counts["videos"] == 4 and counts["boxes"] == 24
    with open(tmp_path / "ev" / "test_videos" / "metadata.csv") as fh:
        meta = list(csv.DictReader(fh))
    assert float(meta[0]["weight_kg"]) == pytest.approx(truth[0]["weight_g"] / 1000, abs=0.002)
    with open(tmp_path / "ev" / "ground_truth" / "gt_boxes.csv") as fh:
        gt = [g for g in csv.DictReader(fh) if g["video_name"] == meta[0]["video_name"]]
    assert [int(g["frame_idx"]) for g in gt] == list(range(1, 7))          # 1-based, like the pipeline
    size = 0.85 + (truth[0]["weight_g"] - 1800) / 1400 * 0.35
    assert int(gt[0]["x"]) == round(bird_box(size)["x1"] - 300)           # shifted into the crop
    info = probe(tmp_path / "ev" / "test_videos" / meta[0]["video_name"])
    assert (info["width"], info["height"], info["n_frames"]) == (940, 380, 6)


def test_nothing_to_export(project, tmp_path):
    with pytest.raises(ValueError):
        export_dataset(project, tmp_path / "x")


# ------------------------------------------------------------ real footage
@pytest.mark.skipif(not REAL, reason="set FLOCKLABEL_REAL_VIDEO to run")
def test_real_video_import_label_export(tmp_path):
    vids = tmp_path / "vids"
    vids.mkdir()
    shutil.copy(REAL, vids / Path(REAL).name)
    p = Project(tmp_path / "proj")
    rep = scan_folder(p, vids)
    assert rep.added == 1 and not rep.no_date_in_name and not rep.unreadable
    c = p.clips()[0]
    frames = list(iter_frames(p.abs_path(c["rel_path"])))
    assert len(frames) == c["n_frames"]
    w, h = c["width"], c["height"]
    for f in (5, 6, 7):
        p.set_label(c["id"], f, {"x1": w * 0.3, "y1": h * 0.4, "x2": w * 0.45, "y2": h * 0.6,
                                 "weight_g": 2400.0, "time_ms": frames[f][2]})
    counts = export_dataset(p, tmp_path / "ds")
    assert counts["images"] == 3
    ev = export_evaluation(p, tmp_path / "ev")
    assert ev["videos"] == 1 and ev["boxes"] == 3
    out = probe(next((tmp_path / "ev" / "test_videos").glob("*.mp4")))
    assert out["n_frames"] == 3 and (out["width"], out["height"]) == (w, h)
    p.close()
