"""Auto-label frames from a folder of videos with the SAM 3 teacher.

Output layout (``out_dir``)::

    frames/<video-stem>_f000123.jpg     sampled frames
    masks/<video-stem>_f000123.npz      masks (N, H, W) bool, class_ids, scores, boxes
    overlays/<video-stem>_f000123.jpg   quick-look overlay (optional)
    manifest.csv                        one row per frame with counts, scores, areas

With ``rois`` (stage 1, :mod:`broilerseg.roi`) the saved frames and masks are
the **pen crops**, and ``roi_x0 ... roi_y1`` in the manifest give where each
crop sits in the full frame.

Re-running skips frames whose ``.npz`` already exists, so a long labelling run
can be interrupted and resumed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from broilerseg.config import CLASS_COLORS, CLASS_NAMES
from broilerseg.io.video import iter_frames, sample_frame_indices, scan_videos
from broilerseg.segmentation.base import Instance, SegmentationResult, Segmenter


def frame_stem(video_stem: str, index: int) -> str:
    """File stem used for a sampled frame."""
    return f"{video_stem}_f{int(index):06d}"


def save_result(path: str | Path, result: SegmentationResult) -> None:
    """Save a segmentation result as a compressed ``.npz``."""
    inst = result.instances
    h, w = result.image_shape
    np.savez_compressed(
        path,
        masks=np.stack([i.mask for i in inst]) if inst else np.zeros((0, h, w), bool),
        class_ids=np.array([CLASS_NAMES.index(i.class_name) for i in inst], np.int16),
        scores=np.array([i.score for i in inst], np.float32),
        boxes=np.array([i.box_xyxy for i in inst], np.float32).reshape(-1, 4),
        prompts=np.array([i.prompt for i in inst], dtype=object),
        image_shape=np.array([h, w]),
        seconds=np.array(result.seconds),
    )


def load_result(path: str | Path) -> SegmentationResult:
    """Load a result saved by :func:`save_result`."""
    d = np.load(path, allow_pickle=True)
    shape = tuple(int(v) for v in d["image_shape"])
    inst = [Instance(CLASS_NAMES[int(c)], m.astype(bool), float(s), tuple(map(float, b)), str(p))
            for m, c, s, b, p in zip(d["masks"], d["class_ids"], d["scores"], d["boxes"],
                                     d["prompts"], strict=False)]
    return SegmentationResult(shape, inst, float(d["seconds"]), "saved")


def render_overlay(image_rgb: np.ndarray, result: SegmentationResult, alpha: float = 0.45
                   ) -> np.ndarray:
    """Fast OpenCV overlay (used for the saved quick-look JPEGs)."""
    out = image_rgb.astype(np.float32).copy()
    for name in ("scale", "feeder", "broiler"):
        color = np.array(CLASS_COLORS[name]) * 255
        for inst in result.of_class(name):
            out[inst.mask] = (1 - alpha) * out[inst.mask] + alpha * color
    out = out.astype(np.uint8)
    for inst in result.instances:
        color = tuple(int(255 * c) for c in CLASS_COLORS[inst.class_name])
        cnts, _ = cv2.findContours(inst.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
        x1, y1 = int(inst.box_xyxy[0]), int(inst.box_xyxy[1])
        cv2.putText(out, f"{inst.class_name} {inst.score:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def plan_sampling(videos: pd.DataFrame, frames_per_video: int = 20, strategy: str = "uniform",
                  seed: int = 0, skip_start: int = 0) -> pd.DataFrame:
    """Decide which frames to label: one row per (video, frame index)."""
    rows = []
    for _, v in videos[videos.readable].iterrows():
        idx = sample_frame_indices(int(v.n_frames), frames_per_video, strategy, seed,
                                   v.path, skip_start)
        vid = v.video_id if "video_id" in v and isinstance(v.video_id, str) else v.stem
        folder = v.folder if "folder" in v else "."
        rows += [{"video": v.path, "video_stem": vid, "folder": folder, "frame_idx": i,
                  "n_frames": int(v.n_frames), "position": i / max(1, v.n_frames - 1)}
                 for i in idx]
    plan = pd.DataFrame(rows, columns=["video", "video_stem", "folder", "frame_idx", "n_frames",
                                       "position"])
    if plan.drop_duplicates("video").video_stem.duplicated().any():
        raise ValueError("Two videos map to the same id; pass the table from scan_videos()")
    return plan


def _row(stem: str, plan_row, result: SegmentationResult, roi_px=None,
         full_shape=None) -> dict:
    row = {"stem": stem, "image": f"frames/{stem}.jpg", "masks": f"masks/{stem}.npz",
           "video": plan_row.video, "video_stem": plan_row.video_stem,
           "folder": getattr(plan_row, "folder", "."),
           "frame_idx": int(plan_row.frame_idx), "height": result.image_shape[0],
           "width": result.image_shape[1], "seconds": result.seconds}
    for c in CLASS_NAMES:
        found = result.of_class(c)
        row[f"n_{c}"] = len(found)
        row[f"score_{c}"] = max((i.score for i in found), default=np.nan)
        row[f"area_{c}"] = int(sum(i.area for i in found))
    row["cropped"] = roi_px is not None
    if roi_px is not None:
        row.update(roi_x0=roi_px[0], roi_y0=roi_px[1], roi_x1=roi_px[2], roi_y1=roi_px[3],
                   full_height=full_shape[0], full_width=full_shape[1])
    return row


def auto_label_videos(video_dir: str | Path, out_dir: str | Path, segmenter: Segmenter,
                      frames_per_video: int = 20, strategy: str = "uniform", seed: int = 0,
                      skip_start: int = 0, save_overlays: bool = True,
                      plan: pd.DataFrame | None = None, jpeg_quality: int = 95,
                      progress: bool = True, rois=None,
                      mask_outside: bool = False) -> pd.DataFrame:
    """Sample frames from every video and label them with ``segmenter``.

    Parameters
    ----------
    video_dir : path-like
        Folder of raw videos (scanned recursively).
    out_dir : path-like
        Where frames, masks, overlays and ``manifest.csv`` are written.
    segmenter : Segmenter
        Normally :class:`broilerseg.segmentation.Sam3Segmenter`.
    frames_per_video : int
        Frames sampled per video.
    strategy : {"uniform", "random", "motion"}
        Frame sampling strategy (see :func:`broilerseg.io.sample_frame_indices`).
    seed, skip_start : int
        Random seed and number of initial frames to ignore.
    save_overlays : bool
        Also write quick-look overlay JPEGs.
    jpeg_quality : int
        JPEG quality of saved frames.
    progress : bool
        Show a progress bar (needs ``tqdm``).
    plan : pandas.DataFrame, optional
        A sampling plan from :func:`plan_sampling`; overrides the arguments above.
    rois : pandas.DataFrame or dict, optional
        Stage-1 pen regions (:func:`broilerseg.roi.find_pen_rois`). Each frame
        is cropped to its video's pen before the teacher runs, and the crop is
        what gets saved. Videos without a region are labelled uncropped.
    mask_outside : bool
        With ``rois``: grey out what lies inside the crop but outside the pen.

    Returns
    -------
    pandas.DataFrame
        The manifest (also written to ``out_dir/manifest.csv``).
    """
    out = Path(out_dir).expanduser()
    for sub in ("frames", "masks") + (("overlays",) if save_overlays else ()):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if plan is None:
        plan = plan_sampling(scan_videos(video_dir), frames_per_video, strategy, seed,
                             skip_start)
    manifest_path = out / "manifest.csv"
    old = pd.read_csv(manifest_path) if manifest_path.is_file() else pd.DataFrame()
    done = set(old.stem) if "stem" in old else set()
    from broilerseg.roi import rois_lookup

    lookup = rois_lookup(rois)
    if len(old) and bool(lookup) != bool(old.get("cropped", pd.Series([False])).any()):
        raise ValueError(
            f"{out} already holds {'cropped' if not lookup else 'uncropped'} labels. Use a "
            "different out_dir for labels made with a different pen-crop setting.")
    rows = []
    groups = list(plan.groupby("video", sort=False))
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(plan), desc=f"Labelling with {segmenter.name}")
        except ImportError:  # pragma: no cover
            bar = None
    for video, g in groups:
        by_idx = {int(r.frame_idx): r for r in g.itertuples()}
        for idx, frame in iter_frames(video, indices=list(by_idx)):
            r = by_idx[idx]
            stem = frame_stem(r.video_stem, idx)
            if stem in done and (out / "masks" / f"{stem}.npz").is_file():
                if bar:
                    bar.update(1)
                continue
            full_shape, roi_px = frame.shape[:2], None
            roi = lookup.get(r.video_stem)
            if roi is not None:
                roi_px = roi.pixels(full_shape)
                frame = roi.crop(frame, mask_outside)
            result = segmenter.segment(frame)
            cv2.imwrite(str(out / "frames" / f"{stem}.jpg"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            save_result(out / "masks" / f"{stem}.npz", result)
            if save_overlays:
                cv2.imwrite(str(out / "overlays" / f"{stem}.jpg"),
                            cv2.cvtColor(render_overlay(frame, result), cv2.COLOR_RGB2BGR))
            rows.append(_row(stem, r, result, roi_px, full_shape))
            if bar:
                bar.update(1)
    if bar:
        bar.close()
    manifest = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if len(old) \
        else pd.DataFrame(rows)
    if len(manifest):
        manifest = manifest.drop_duplicates("stem", keep="last").reset_index(drop=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest
