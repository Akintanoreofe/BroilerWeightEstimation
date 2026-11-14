"""Export labelled frames.

* :func:`export_dataset`: images, YOLO labels, COCO JSON (``weight_g`` per box),
  ``manifest.csv`` and one sequence file per video. Splits are made by video so
  frames of one weighing never land in both train and test.
* :func:`export_evaluation`: the layout BroilerWeightEstimation reads, one
  trimmed (and optionally cropped) clip per video in ``test_videos/`` with
  ``metadata.csv`` (median reading, kg), plus per-frame ground truth for
  ``evaluate_against_gt.py``.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2

from flocklabel import __version__
from .db import Project
from .video import iter_frames

MIN_BOX_SIDE_PX = 277 // 3      # BroilerWeightEstimation drops smaller contours (template 277 px / 3)
EDGE_MARGIN_PX = 10             # ...and contours touching the frame border
Progress = Optional[Callable[[int, int, str], bool]]


def split_by_video(clip_ids, ratios=(0.7, 0.15, 0.15), seed: int = 42) -> dict[int, str]:
    ids = sorted(set(clip_ids))
    random.Random(seed).shuffle(ids)
    n_train, n_val = round(len(ids) * ratios[0]), round(len(ids) * ratios[1])
    return {c: "train" if i < n_train else "val" if i < n_train + n_val else "test" for i, c in enumerate(ids)}


def _even_sample(frames: list[int], k: Optional[int]) -> list[int]:
    if not k or len(frames) <= k:
        return frames
    if k == 1:
        return [frames[len(frames) // 2]]
    step = (len(frames) - 1) / (k - 1)
    return sorted({frames[round(i * step)] for i in range(k)})


def _group(rows: list[dict]) -> dict[int, list[dict]]:
    by_clip: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_clip[r["clip_id"]].append(r)
    return dict(by_clip)


def _nothing() -> ValueError:
    return ValueError("Nothing to export: no frames with both a box and a scale reading "
                      "for the chosen dates (videos marked 'skip' are left out).")


def export_dataset(project: Project, out_dir: Path | str, *, only_done: bool = False,
                   date_from: Optional[str] = None, date_to: Optional[str] = None,
                   ratios=(0.7, 0.15, 0.15), seed: int = 42, max_frames_per_video: Optional[int] = None,
                   jpeg_quality: int = 95, write_crops: bool = False, progress: Progress = None) -> dict:
    out = Path(out_dir)
    by_clip = _group(project.export_rows(only_done=only_done, date_from=date_from, date_to=date_to))
    if not by_clip:
        raise _nothing()
    for cid, rows in by_clip.items():
        keep = set(_even_sample([r["frame_idx"] for r in rows], max_frames_per_video))
        by_clip[cid] = [r for r in rows if r["frame_idx"] in keep]
    split = split_by_video(by_clip, ratios, seed)
    for sub in ("train", "val", "test"):
        for kind in ("images", "labels") + (("crops",) if write_crops else ()):
            (out / kind / sub).mkdir(parents=True, exist_ok=True)
    (out / "sequences").mkdir(parents=True, exist_ok=True)

    coco = {"info": {"description": "FlockLabel export", "version": __version__,
                     "date_created": datetime.now().isoformat(timespec="seconds")},
            "categories": [{"id": 1, "name": "chicken"}], "images": [], "annotations": []}
    cols = ["image_path", "split", "video", "timestamp_local", "frame_idx", "x1", "y1", "x2", "y2",
            "img_w", "img_h", "weight_g", "video_median_weight_g"]
    n_done = 0
    total = sum(len(v) for v in by_clip.values())
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        mw = csv.writer(fh)
        mw.writerow(cols)
        for cid, rows in by_clip.items():
            wanted = {r["frame_idx"]: r for r in rows}
            median_w = statistics.median(r["weight_g"] for r in rows)
            sp, seq = split[cid], []
            path = project.abs_path(rows[0]["rel_path"])
            for f, bgr, _ in iter_frames(path, last=max(wanted)):
                if f not in wanted:
                    continue
                r = wanted[f]
                n_done += 1
                if progress and progress(n_done, total, Path(r["rel_path"]).name) is False:
                    raise InterruptedError("Export cancelled")
                H, W = bgr.shape[:2]
                ts = r["timestamp"]
                stem = f"{ts:%Y%m%d_%H%M%S}_v{cid}_f{f:05d}"
                img_rel = Path("images", sp, stem + ".jpg")
                cv2.imwrite(str(out / img_rel), bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                x1, y1 = max(0.0, r["x1"]), max(0.0, r["y1"])
                x2, y2 = min(float(W), r["x2"]), min(float(H), r["y2"])
                (out / "labels" / sp / (stem + ".txt")).write_text(
                    f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
                image_id = len(coco["images"]) + 1
                coco["images"].append({"id": image_id, "file_name": img_rel.as_posix(), "width": W, "height": H,
                                       "video": r["rel_path"], "frame_idx": f,
                                       "timestamp_local": ts.isoformat(timespec="milliseconds")})
                coco["annotations"].append({
                    "id": image_id, "image_id": image_id, "category_id": 1,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    "area": round((x2 - x1) * (y2 - y1), 1), "iscrowd": 0,
                    "attributes": {"weight_g": r["weight_g"], "video_id": cid, "video_median_weight_g": median_w}})
                mw.writerow([img_rel.as_posix(), sp, r["rel_path"], ts.isoformat(timespec="milliseconds"), f,
                             round(x1), round(y1), round(x2), round(y2), W, H, r["weight_g"], median_w])
                seq.append({"frame_idx": f, "image_path": img_rel.as_posix(),
                            "timestamp_local": ts.isoformat(timespec="milliseconds"),
                            "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                            "weight_g": r["weight_g"]})
                if write_crops:
                    cv2.imwrite(str(out / "crops" / sp / (stem + ".jpg")), bgr[int(y1):int(y2), int(x1):int(x2)],
                                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            (out / "sequences" / f"{cid}.json").write_text(json.dumps(
                {"video_id": cid, "video": rows[0]["rel_path"], "split": sp, "median_weight_g": median_w,
                 "frames": seq}, indent=2))
    (out / "annotations_coco.json").write_text(json.dumps(coco, indent=1))
    (out / "data.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
                                   "test: images/test\nnames:\n  0: chicken\n")
    counts = {"videos": len(by_clip), "images": len(coco["images"]),
              "splits": {s: sum(v == s for v in split.values()) for s in ("train", "val", "test")}}
    settings = {"only_done": only_done, "date_from": date_from, "date_to": date_to, "ratios": list(ratios),
                "seed": seed, "max_frames_per_video": max_frames_per_video}
    (out / "export_info.json").write_text(json.dumps({"app": "FlockLabel", "version": __version__,
                                                      "created": datetime.now().isoformat(timespec="seconds"),
                                                      "settings": settings, "counts": counts}, indent=2))
    project.log_export("dataset", out, json.dumps(settings), counts["images"], counts["images"])
    return counts


def export_evaluation(project: Project, out_dir: Path | str, *, only_done: bool = False,
                      date_from: Optional[str] = None, date_to: Optional[str] = None,
                      progress: Progress = None) -> dict:
    out = Path(out_dir)
    by_clip = _group(project.export_rows(only_done=only_done, date_from=date_from, date_to=date_to))
    if not by_clip:
        raise _nothing()
    (out / "test_videos").mkdir(parents=True, exist_ok=True)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    crop = project.crop
    warnings: list[str] = []
    meta, events, gt = [], [], []
    for i, (cid, rows) in enumerate(by_clip.items()):
        first = rows[0]
        if progress and progress(i, len(by_clip), Path(first["rel_path"]).name) is False:
            raise InterruptedError("Export cancelled")
        frames = {r["frame_idx"]: r for r in rows}
        f0, f1 = min(frames), max(frames)
        W, H = first["width"], first["height"]
        cx, cy, cw, ch = crop if crop else (0, 0, W, H)
        cx, cy = max(0, cx), max(0, cy)
        cw, ch = min(cw, W - cx), min(ch, H - cy)
        ts = frames[f0]["timestamp"]
        name = f"v{cid:05d}_{ts:%Y%m%d_%H%M%S}.mp4"
        writer = None
        try:
            for f, bgr, _ in iter_frames(project.abs_path(first["rel_path"]), last=f1):
                if f < f0:
                    continue
                if writer is None:
                    H, W = bgr.shape[:2]
                    cw, ch = min(cw, W - cx), min(ch, H - cy)
                    writer = cv2.VideoWriter(str(out / "test_videos" / name), cv2.VideoWriter_fourcc(*"mp4v"),
                                             first["fps"], (cw, ch))
                writer.write(bgr[cy:cy + ch, cx:cx + cw])
        finally:
            if writer is not None:
                writer.release()
        weights = [r["weight_g"] for r in rows]
        median_kg = statistics.median(weights) / 1000
        meta.append([name, round(median_kg, 4)])
        for f, r in sorted(frames.items()):
            x1, y1 = max(r["x1"] - cx, 0), max(r["y1"] - cy, 0)
            x2, y2 = min(r["x2"] - cx, cw), min(r["y2"] - cy, ch)
            tag = f"{Path(first['rel_path']).name} frame {f}"
            if x2 <= x1 or y2 <= y1:
                warnings.append(f"{tag}: box is outside the evaluation crop")
                continue
            if min(x1, y1) <= EDGE_MARGIN_PX or x2 >= cw - EDGE_MARGIN_PX or y2 >= ch - EDGE_MARGIN_PX:
                warnings.append(f"{tag}: box touches the crop edge; the pipeline drops such birds")
            if x2 - x1 < MIN_BOX_SIDE_PX or y2 - y1 < MIN_BOX_SIDE_PX:
                warnings.append(f"{tag}: box {x2 - x1:.0f}x{y2 - y1:.0f}px is under the pipeline's "
                                f"{MIN_BOX_SIDE_PX}px minimum")
            gt.append([name, f - f0 + 1, round(x1), round(y1), round(x2 - x1), round(y2 - y1),
                       round(r["weight_g"] / 1000, 4), cid, "", ""])
        events.append([cid, name, round(median_kg, 4), round(min(weights) / 1000, 4), round(max(weights) / 1000, 4),
                       f1 - f0 + 1, ts.isoformat(timespec="seconds"), project.pixel_to_cm or "",
                       first["rel_path"], f0])

    def write(path: Path, header: list[str], data: list[list]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(data)

    write(out / "test_videos" / "metadata.csv", ["video_name", "weight_kg"], meta)
    write(out / "ground_truth" / "gt_boxes.csv",
          ["video_name", "frame_idx", "x", "y", "w", "h", "weight_kg", "event_id", "bird_id", "flags"], gt)
    write(out / "ground_truth" / "events.csv",
          ["event_id", "video_name", "weight_kg", "min_weight_kg", "max_weight_kg", "n_frames", "start_local",
           "pixel_to_cm", "source_video", "source_first_frame"], events)
    counts = {"videos": len(events), "boxes": len(gt), "warnings": warnings}
    (out / "export_info.json").write_text(json.dumps(
        {"app": "FlockLabel", "version": __version__, "kind": "evaluation",
         "created": datetime.now().isoformat(timespec="seconds"), "crop": crop,
         "filters": {"only_done": only_done, "date_from": date_from, "date_to": date_to},
         "counts": {"videos": len(events), "boxes": len(gt)}, "warnings": warnings}, indent=2))
    project.log_export("evaluation", out, json.dumps({"crop": crop}), len(events), len(gt))
    return counts
