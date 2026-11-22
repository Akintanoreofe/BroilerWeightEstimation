"""Turn SAM 3 masks into an Ultralytics YOLO segmentation dataset.

Each instance mask becomes one polygon line in a ``labels/<split>/<stem>.txt``
file: ``class x1 y1 x2 y2 ...`` with coordinates normalised to ``[0, 1]``.
Splits are made **by video** so near-identical frames from one clip never sit
in both train and validation (the same rule FlockLabel uses).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from broilerseg.config import CLASS_NAMES
from broilerseg.labeling import load_result
from broilerseg.segmentation.postprocess import mask_iou


def mask_to_polygons(mask: np.ndarray, epsilon_px: float = 1.0, min_area_px: float = 50.0
                     ) -> list[np.ndarray]:
    """Outer contours of a mask as simplified ``(N, 2)`` pixel polygons.

    Separate blobs become separate polygons; blobs below ``min_area_px`` are
    dropped. ``epsilon_px`` is the Douglas-Peucker tolerance.
    """
    cnts, _ = cv2.findContours(np.asarray(mask).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area_px:
            continue
        p = cv2.approxPolyDP(c, epsilon_px, True).reshape(-1, 2).astype(np.float32)
        if len(p) >= 3:
            polys.append(p)
    return polys


def polygons_to_mask(polygons, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise pixel polygons back into a boolean mask."""
    out = np.zeros(shape, np.uint8)
    for p in polygons:
        cv2.fillPoly(out, [np.round(p).astype(np.int32)], 1)
    return out.astype(bool)


def format_label_line(class_id: int, polygon: np.ndarray, width: int, height: int) -> str:
    """One YOLO-seg label line."""
    norm = polygon / np.array([width, height], np.float32)
    norm = np.clip(norm, 0, 1)
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in norm.reshape(-1))


def read_yolo_label(path: str | Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    """Read a YOLO-seg label file into ``(class_id, pixel polygon)`` pairs."""
    out = []
    for line in Path(path).read_text().splitlines():
        vals = line.split()
        if len(vals) < 7:
            continue
        pts = np.array(vals[1:], np.float32).reshape(-1, 2) * np.array([width, height])
        out.append((int(vals[0]), pts))
    return out


def split_by_group(groups, ratios=(0.8, 0.2, 0.0), seed: int = 42) -> dict[str, str]:
    """Assign each group (video) to train / val / test."""
    ids = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_val = max(1, round(n * ratios[1])) if n >= 2 and ratios[1] > 0 else 0
    n_test = round(n * ratios[2]) if n >= 3 else 0
    n_test = min(n_test, n - n_val - 1)
    n_train = n - n_val - n_test
    return {g: "train" if i < n_train else "val" if i < n_train + n_val else "test"
            for i, g in enumerate(ids)}


def build_yolo_dataset(label_dir: str | Path, out_dir: str | Path,
                       ratios=(0.8, 0.2, 0.0), seed: int = 42, min_score: float = 0.0,
                       epsilon_px: float = 1.0, min_area_px: float = 50.0,
                       require_broiler: bool = False, split_by: str = "video") -> pd.DataFrame:
    """Convert an :func:`broilerseg.labeling.auto_label_videos` folder to YOLO-seg.

    Parameters
    ----------
    label_dir : path-like
        Folder holding ``manifest.csv``, ``frames/`` and ``masks/``.
    out_dir : path-like
        New dataset folder (``images/``, ``labels/``, ``data.yaml``).
    ratios : tuple of float
        Train / val / test fractions of *videos*.
    seed : int
        Seed of the video split.
    min_score : float
        Drop teacher instances below this confidence (pseudo-label filtering).
    epsilon_px, min_area_px : float
        Polygon simplification and minimum blob size.
    require_broiler : bool
        Skip frames where the teacher found no broiler.
    split_by : {"video", "folder"}
        Keep whole videos together (default), or whole sub-folders (e.g. days
        or cameras) for a harder test of generalisation.

    Returns
    -------
    pandas.DataFrame
        One row per image with split, polygon counts per class and the IoU
        between each class's teacher mask and its rasterised polygons
        ("polygon fidelity", a check that conversion loses little).
    """
    src, out = Path(label_dir).expanduser(), Path(out_dir).expanduser()
    manifest = pd.read_csv(src / "manifest.csv")
    if require_broiler:
        manifest = manifest[manifest.n_broiler > 0]
    if split_by == "folder":
        groups = (manifest.folder if "folder" in manifest
                  else manifest.video.map(lambda v: str(Path(v).parent)))
    elif split_by == "video":
        groups = manifest.video_stem
    else:
        raise ValueError("split_by must be 'video' or 'folder'")
    manifest = manifest.assign(split_group=groups.astype(str))
    group_split = split_by_group(manifest.split_group, ratios, seed)
    split = {stem: group_split[g]
             for stem, g in zip(manifest.video_stem, manifest.split_group, strict=False)}
    for s in ("train", "val", "test"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    rows = []
    for r in manifest.itertuples():
        res = load_result(src / r.masks)
        h, w = res.image_shape
        sp = split[r.video_stem]
        lines, row = [], {"stem": r.stem, "video_stem": r.video_stem, "split_group": r.split_group,
                          "split": sp}
        for cid, cname in enumerate(CLASS_NAMES):
            polys_all, teacher = [], np.zeros((h, w), bool)
            for inst in res.of_class(cname):
                if inst.score < min_score:
                    continue
                teacher |= inst.mask
                polys = mask_to_polygons(inst.mask, epsilon_px, min_area_px)
                polys_all += polys
                lines += [format_label_line(cid, p, w, h) for p in polys]
            row[f"n_{cname}"] = len(polys_all)
            row[f"fidelity_{cname}"] = (mask_iou(teacher, polygons_to_mask(polys_all, (h, w)))
                                        if teacher.any() else np.nan)
        shutil.copy2(src / r.image, out / "images" / sp / f"{r.stem}.jpg")
        (out / "labels" / sp / f"{r.stem}.txt").write_text("\n".join(lines) + ("\n" if lines
                                                                               else ""))
        rows.append(row)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    test_line = "test: images/test\n" if any(v == "test" for v in split.values()) else ""
    # With a single video there is nothing to hold out; validate on train and say so.
    val_dir = "images/val" if any(v == "val" for v in split.values()) else "images/train"
    if val_dir == "images/train":
        print("[broilerseg] Only one video: validation uses the training images. "
              "Add more videos for an honest validation score.")
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: {val_dir}\n"
        f"{test_line}names:\n{names}\n")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "dataset_summary.csv", index=False)
    return summary
