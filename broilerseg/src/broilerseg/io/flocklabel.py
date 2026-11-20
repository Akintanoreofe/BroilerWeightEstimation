"""Read the exports written by the FlockLabel GUI.

FlockLabel writes two layouts (see its ``export.py``):

*Evaluation export* (``export_evaluation``)::

    test_videos/<video>.mp4, test_videos/metadata.csv     (video_name, weight_kg)
    ground_truth/gt_boxes.csv   (video_name, frame_idx [1-based in the clip],
                                 x, y, w, h, weight_kg, event_id, bird_id, flags)
    ground_truth/events.csv     (event_id, video_name, weight_kg, ..., pixel_to_cm, ...)
    export_info.json            ("kind": "evaluation")

*Dataset export* (``export_dataset``)::

    images/{train,val,test}/*.jpg, manifest.csv (image_path, split, video,
    frame_idx, x1, y1, x2, y2, img_w, img_h, weight_g, video_median_weight_g)

Both are turned into one tidy ``samples`` table: one row per labelled frame
with its box, scale reading (kg) and event id.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from broilerseg.io.video import iter_frames

SAMPLE_COLUMNS = ["sample_id", "kind", "file", "frame_index0", "video_name", "frame_label",
                  "event_id", "x1", "y1", "x2", "y2", "weight_kg", "event_weight_kg",
                  "pixel_to_cm_hint", "split"]


@dataclass
class FlockLabelExport:
    """A loaded FlockLabel export.

    Attributes
    ----------
    kind : str
        ``"evaluation"`` or ``"dataset"``.
    root : Path
        Export folder.
    samples : pandas.DataFrame
        One row per labelled frame (see :data:`SAMPLE_COLUMNS`).
    events : pandas.DataFrame
        One row per weighing event / video.
    info : dict
        Contents of ``export_info.json`` (may be empty).
    """

    kind: str
    root: Path
    samples: pd.DataFrame
    events: pd.DataFrame
    info: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"FlockLabelExport(kind={self.kind!r}, root='{self.root}', "
                f"events={len(self.events)}, frames={len(self.samples)})")


def _find_root(path: Path) -> tuple[str, Path]:
    candidates = [path] + sorted(p for p in path.glob("*") if p.is_dir())
    candidates += sorted(p for p in path.glob("*/*") if p.is_dir())
    for c in candidates:
        if (c / "ground_truth" / "gt_boxes.csv").is_file():
            return "evaluation", c
        if (c / "manifest.csv").is_file():
            return "dataset", c
    raise FileNotFoundError(
        f"No FlockLabel export found in {path}. Expected ground_truth/gt_boxes.csv "
        "(evaluation export) or manifest.csv (dataset export).")


def _load_evaluation(root: Path, frame_offset: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = pd.read_csv(root / "ground_truth" / "gt_boxes.csv")
    ev_path = root / "ground_truth" / "events.csv"
    events = pd.read_csv(ev_path) if ev_path.is_file() else pd.DataFrame()
    meta = pd.read_csv(root / "test_videos" / "metadata.csv")
    missing = [v for v in meta.video_name if not (root / "test_videos" / v).is_file()]
    if missing:
        raise FileNotFoundError(f"Videos listed in metadata.csv but missing: {missing[:5]}")
    if events.empty:
        events = meta.assign(event_id=np.arange(len(meta)))
    if "pixel_to_cm" not in events:
        events["pixel_to_cm"] = np.nan
    px_hint = dict(zip(events.video_name, pd.to_numeric(events.pixel_to_cm, errors="coerce"),
                       strict=False))
    ev_w = dict(zip(meta.video_name, meta.weight_kg, strict=False))
    s = pd.DataFrame({
        "kind": "video",
        "file": [str(root / "test_videos" / v) for v in gt.video_name],
        "frame_index0": gt.frame_idx.astype(int) + frame_offset - 1,
        "video_name": gt.video_name,
        "frame_label": gt.frame_idx.astype(int),
        "event_id": gt.event_id.astype(str),
        "x1": gt.x.astype(float), "y1": gt.y.astype(float),
        "x2": gt.x.astype(float) + gt.w.astype(float),
        "y2": gt.y.astype(float) + gt.h.astype(float),
        "weight_kg": gt.weight_kg.astype(float),
        "event_weight_kg": gt.video_name.map(ev_w),
        "pixel_to_cm_hint": gt.video_name.map(px_hint),
        "split": "",
    })
    events = events.assign(event_id=events.event_id.astype(str))
    return s, events


def _load_dataset(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(root / "manifest.csv")
    s = pd.DataFrame({
        "kind": "image",
        "file": [str(root / p) for p in m.image_path],
        "frame_index0": -1,
        "video_name": m.video.map(lambda v: Path(str(v)).name),
        "frame_label": m.frame_idx.astype(int),
        "event_id": m.video.astype(str),
        "x1": m.x1.astype(float), "y1": m.y1.astype(float),
        "x2": m.x2.astype(float), "y2": m.y2.astype(float),
        "weight_kg": m.weight_g.astype(float) / 1000,
        "event_weight_kg": m.video_median_weight_g.astype(float) / 1000,
        "pixel_to_cm_hint": np.nan,
        "split": m.split.astype(str),
    })
    events = (s.groupby(["event_id", "video_name"], as_index=False)
              .agg(weight_kg=("event_weight_kg", "first"), n_frames=("weight_kg", "size"),
                   min_weight_kg=("weight_kg", "min"), max_weight_kg=("weight_kg", "max")))
    return s, events


def load_flocklabel_export(path: str | Path, frame_offset: int = 0) -> FlockLabelExport:
    """Load a FlockLabel evaluation or dataset export.

    Parameters
    ----------
    path : path-like
        The export folder, or a folder up to two levels above it.
    frame_offset : int
        Added to ground-truth frame numbers before reading (use ``1`` if your
        GT frames are zero-based); same meaning as ``--gt-frame-offset`` in the
        repository's ``evaluate_against_gt.py``.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    kind, root = _find_root(path)
    samples, events = (_load_evaluation(root, frame_offset) if kind == "evaluation"
                       else _load_dataset(root))
    samples.insert(0, "sample_id", np.arange(len(samples)))
    info_path = root / "export_info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    return FlockLabelExport(kind, root, samples[SAMPLE_COLUMNS].reset_index(drop=True),
                            events, info)


def subsample_per_event(samples: pd.DataFrame, k: int | None, seed: int = 0) -> pd.DataFrame:
    """Keep at most ``k`` evenly spaced frames per event (``None`` keeps all)."""
    if not k:
        return samples
    parts = []
    for _, g in samples.sort_values(["event_id", "frame_label"]).groupby("event_id"):
        if len(g) <= k:
            parts.append(g)
        else:
            idx = np.unique(np.round(np.linspace(0, len(g) - 1, k)).astype(int))
            parts.append(g.iloc[idx])
    return pd.concat(parts).reset_index(drop=True) if parts else samples.iloc[:0]


def iter_sample_images(samples: pd.DataFrame) -> Iterator[tuple[pd.Series, np.ndarray]]:
    """Yield ``(row, image_rgb)`` for every sample, decoding each video once."""
    for file, g in samples.groupby("file", sort=False):
        if (g.kind == "image").all():
            for _, row in g.iterrows():
                bgr = cv2.imread(str(row.file))
                if bgr is not None:
                    yield row, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            continue
        by_index = {int(r.frame_index0): r for _, r in g.iterrows()}
        for idx, frame in iter_frames(file, indices=list(by_index)):
            yield by_index[idx], frame
