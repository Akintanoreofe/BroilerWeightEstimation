"""Synthetic scenes for dry runs and tests (no SAM 3 or GPU needed).

``make_demo_data`` writes a small folder of videos *and* a FlockLabel-style
evaluation export with known bird weights. ``ColorSegmenter`` segments those
synthetic frames by colour, so both notebooks can be run end to end before the
real models are installed. Set ``DRY_RUN = True`` at the top of a notebook.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from broilerseg.config import FEEDER_DIAMETER_CM
from broilerseg.segmentation.base import Instance, SegmentationResult, box_from_mask

SCALE_RGB = (170, 172, 175)
FEEDER_RGB = (235, 110, 20)
BIRD_RGB = (245, 240, 225)


def draw_scene(h: int, w: int, bird, others=(), feeder=(None, None), scale=None,
               seed: int = 0) -> np.ndarray:
    """Draw litter, a scale platform, an orange feeder and white broilers (RGB)."""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.float32)
    img[:] = (95, 75, 55)
    img += rng.normal(0, 8, img.shape)
    if scale is not None:
        x1, y1, x2, y2 = scale
        img[y1:y2, x1:x2] = SCALE_RGB
    fc, fr = feeder
    if fc is not None:
        cv2.circle(img, fc, fr, FEEDER_RGB, -1)
        cv2.circle(img, fc, int(fr * 0.6), (200, 90, 15), -1)  # inner tray, still orange
    for (cx, cy), (a, b), ang in (*others, *([bird] if bird else [])):
        cv2.ellipse(img, (int(cx), int(cy)), (int(a), int(b)), ang, 0, 360, BIRD_RGB, -1)
        hx = int(cx + 0.95 * a * np.cos(np.deg2rad(ang)))
        hy = int(cy + 0.95 * a * np.sin(np.deg2rad(ang)))
        cv2.circle(img, (hx, hy), int(b * 0.45), BIRD_RGB, -1)  # head
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class ColorSegmenter:
    """Colour-threshold segmenter for :func:`draw_scene` images only."""

    name = "ColorSegmenter (synthetic)"

    def segment(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Segment a synthetic frame."""
        t0 = time.perf_counter()
        img = image_rgb.astype(np.int16)
        h, w = img.shape[:2]

        def near(rgb, tol):
            return (np.abs(img - np.array(rgb)).max(axis=2) < tol).astype(np.uint8)

        inst = []
        feeder = cv2.morphologyEx(((img[..., 0] > 180) & (img[..., 1] < 140)
                                   & (img[..., 2] < 60)).astype(np.uint8),
                                  cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        groups = (("scale", near(SCALE_RGB, 18)), ("feeder", feeder),
                  ("broiler", near(BIRD_RGB, 25)))
        for name, m in groups:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
            for k in range(1, n):
                if stats[k, cv2.CC_STAT_AREA] < 200:
                    continue
                mk = lab == k
                if name == "scale":  # include what sits on the platform, as SAM 3 tends to
                    pts = cv2.findNonZero(mk.astype(np.uint8))
                    full = np.zeros((h, w), np.uint8)
                    cv2.fillConvexPoly(full, cv2.convexHull(pts), 1)
                    mk = full.astype(bool)
                inst.append(Instance(name, mk, 0.9, box_from_mask(mk), "colour"))
        return SegmentationResult((h, w), inst, time.perf_counter() - t0, self.name)


def surround_scene(img: np.ndarray, width: int = 160, seed: int = 0) -> np.ndarray:
    """Add a strip of concrete floor and a desk left of the pen (stage-1 demo)."""
    rng = np.random.default_rng(seed)
    h = img.shape[0]
    side = np.empty((h, width, 3), np.float32)
    side[:] = (150, 150, 152)
    side += rng.normal(0, 6, side.shape)
    side[h // 5: h // 2, 15: width - 30] = (60, 62, 70)        # desk / computer
    side[:, width - 8:] = (235, 235, 235)                       # white fence rail
    return np.hstack([np.clip(side, 0, 255).astype(np.uint8), img])


class ColorPenDetector:
    """Stage-1 stand-in: the pen is the litter-coloured region of a synthetic frame."""

    name = "ColorPenDetector (synthetic)"

    def segment(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Return one ``"pen"`` instance (the litter area)."""
        t0 = time.perf_counter()
        img = cv2.medianBlur(image_rgb, 5).astype(np.int16)
        h, w = img.shape[:2]
        litter = (np.abs(img - np.array((95, 75, 55))).max(axis=2) < 35).astype(np.uint8)
        litter = cv2.morphologyEx(litter, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(litter)
        inst = []
        if n > 1:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mk = lab == k
            inst.append(Instance("pen", mk, 0.95, box_from_mask(mk), "litter colour"))
        return SegmentationResult((h, w), inst, time.perf_counter() - t0, self.name)


def _bird_weight_kg(a_px, b_px, px_to_cm, density=1000.0, depth_ratio=0.8):
    """Weight of an ellipsoid bird (semi-axes a, b, c = depth_ratio * b)."""
    a, b = a_px * px_to_cm, b_px * px_to_cm
    vol_cm3 = 4 / 3 * np.pi * a * b * (depth_ratio * b)
    return vol_cm3 / 1e6 * density


def make_demo_data(out_dir: str | Path, n_events: int = 6, frames_per_event: int = 12,
                   size=(480, 640), seed: int = 0, nested: bool = False) -> dict[str, Path]:
    """Write synthetic videos and a FlockLabel evaluation export.

    With ``nested=True`` the raw videos go into ``day<k>/cam<j>/`` sub-folders
    and several share the same file name, like real CCTV dumps, and each raw
    frame gets a strip of floor and a desk left of the pen (so the stage-1 pen
    crop has something to remove).

    Returns
    -------
    dict
        ``{"videos": <folder of raw videos>, "export": <FlockLabel export>}``.
    """
    out = Path(out_dir)
    vids, exp = out / "raw_videos", out / "flocklabel_export"
    (exp / "test_videos").mkdir(parents=True, exist_ok=True)
    (exp / "ground_truth").mkdir(parents=True, exist_ok=True)
    vids.mkdir(parents=True, exist_ok=True)
    h, w = size
    rng = np.random.default_rng(seed)
    fr = 70
    px_to_cm = FEEDER_DIAMETER_CM / (2 * fr)
    meta, gt, events = [], [], []
    for e in range(n_events):
        name = f"v{e:05d}_20251001_1200{e:02d}.mp4"
        scale = (200, 120, 520, 400)
        feeder = ((280, 200), fr)
        a = rng.uniform(55, 72)
        b = a * rng.uniform(0.45, 0.55)
        weight = _bird_weight_kg(a, b, px_to_cm)
        writer = cv2.VideoWriter(str(exp / "test_videos" / name),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        raw_path = (vids / f"day{e // 2 + 1}" / f"cam{e % 2 + 1}" / "ch01.mp4") if nested \
            else vids / name
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        side = 160 if nested else 0
        raw = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), 10,
                              (w + side, h))
        other = (((80, 380), (50, 25), 20.0),)
        for f in range(frames_per_event):
            cx = 400 + 15 * np.sin(f / 3)
            cy = 290 + 10 * np.cos(f / 4)
            ang = 15 + 20 * np.sin(f / 5) + 25 * e
            img = draw_scene(h, w, ((cx, cy), (a, b), ang), other, feeder, scale, seed + f)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            raw.write(cv2.cvtColor(surround_scene(img, side, seed + f), cv2.COLOR_RGB2BGR)
                      if side else bgr)
            mk = np.zeros((h, w), np.uint8)
            cv2.ellipse(mk, (int(cx), int(cy)), (int(a), int(b)), ang, 0, 360, 1, -1)
            x1, y1, x2, y2 = box_from_mask(mk.astype(bool))
            reading = weight * (1 + rng.normal(0, 0.01))
            gt.append([name, f + 1, round(x1), round(y1), round(x2 - x1), round(y2 - y1),
                       round(reading, 4), e, "", ""])
        writer.release()
        raw.release()
        meta.append([name, round(weight, 4)])
        events.append([e, name, round(weight, 4), round(weight * 0.99, 4),
                       round(weight * 1.01, 4), frames_per_event, "2025-10-01T12:00:00", "",
                       name, 0])
    pd.DataFrame(meta, columns=["video_name", "weight_kg"]).to_csv(
        exp / "test_videos" / "metadata.csv", index=False)
    pd.DataFrame(gt, columns=["video_name", "frame_idx", "x", "y", "w", "h", "weight_kg",
                              "event_id", "bird_id", "flags"]).to_csv(
        exp / "ground_truth" / "gt_boxes.csv", index=False)
    pd.DataFrame(events, columns=["event_id", "video_name", "weight_kg", "min_weight_kg",
                                  "max_weight_kg", "n_frames", "start_local", "pixel_to_cm",
                                  "source_video", "source_first_frame"]).to_csv(
        exp / "ground_truth" / "events.csv", index=False)
    (exp / "export_info.json").write_text(json.dumps(
        {"app": "FlockLabel", "kind": "evaluation", "note": "synthetic demo",
         "true_pixel_to_cm": px_to_cm}, indent=2))
    return {"videos": vids, "export": exp}
