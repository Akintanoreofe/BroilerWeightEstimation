"""Stage 1 of two-stage labelling: find the pen, crop it, then segment inside it.

The CCTV view shows much more than the pen (the floor around it, a desk, a
computer, cables). Segmenting the whole frame makes the teacher confuse the
scale, the feeder and the birds with that clutter, and a 16:9 4K frame squeezed
into the teacher's input leaves each bird only a few pixels wide.

The two-stage pipeline fixes both:

1. **Stage 1 - pen region.** A text prompt ("animal pen", ...) is run through
   the same teacher model. The best pen mask over a few frames of each video
   gives a :class:`PenROI`: a box (plus an optional outline polygon), stored as
   *fractions* of the frame so it also fits other resolutions of the same camera.
2. **Stage 2 - objects.** Frames are cropped to the pen (at full resolution, so
   no detail is lost and pixel-to-cm calibration is unchanged) and only the crop
   is segmented into broiler / scale / feeder.

:class:`CroppedSegmenter` wraps any segmenter (SAM 3, Grounded SAM 2 or the YOLO
student) so it runs on the pen crop but returns masks in full-frame
coordinates. Everything downstream (bird selection, calibration, volume) works
unchanged.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from broilerseg.config import PEN_PROMPTS, PromptConfig
from broilerseg.io.video import read_frame
from broilerseg.segmentation.base import Instance, SegmentationResult, Segmenter

#: Grey used to blank out everything outside the pen outline (YOLO's padding grey).
FILL_GREY = 114


# =========================================================================== PenROI
@dataclass
class PenROI:
    """The pen region of a camera view, as fractions of the frame.

    Attributes
    ----------
    fx0, fy0, fx1, fy1 : float
        Box corners as fractions of the frame width / height (0-1).
    polygon : list of (float, float)
        Optional pen outline (fractions), used to grey out what lies inside
        the box but outside the pen when ``mask_outside=True``.
    score : float
        Detector confidence (1.0 for manual boxes).
    method : str
        ``auto``, ``auto_folder``, ``folder_median``, ``manual`` or ``full_frame``.
    """

    fx0: float = 0.0
    fy0: float = 0.0
    fx1: float = 1.0
    fy1: float = 1.0
    polygon: list = field(default_factory=list)
    score: float = 1.0
    method: str = "full_frame"

    # ------------------------------------------------------------ constructors
    @classmethod
    def full_frame(cls) -> PenROI:
        """Return a region covering the whole frame (no crop)."""
        return cls()

    @classmethod
    def from_box(cls, box, image_size: tuple[int, int] | None = None,
                 method: str = "manual") -> PenROI:
        """Build from ``(x0, y0, x1, y1)``.

        Values all <= 1 are read as fractions. Otherwise they are pixels and
        ``image_size = (width, height)`` of the frame they refer to is needed.
        """
        x0, y0, x1, y1 = (float(v) for v in box)
        if max(x0, y0, x1, y1) > 1.0:
            if image_size is None:
                raise ValueError("pixel boxes need image_size=(width, height)")
            w, h = image_size
            x0, x1, y0, y1 = x0 / w, x1 / w, y0 / h, y1 / h
        roi = cls(x0, y0, x1, y1, [], 1.0, method)
        return roi.clipped()

    @classmethod
    def from_mask(cls, mask: np.ndarray, score: float = 1.0, margin: float = 0.03,
                  method: str = "auto", min_part: float = 0.05) -> PenROI:
        """Box and convex outline of a pen mask, widened by ``margin`` (fraction).

        Mask pieces smaller than ``min_part`` of the biggest piece are ignored,
        so a stray speck does not stretch the box.
        """
        h, w = mask.shape
        m = _drop_small_parts(mask, min_part)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            return cls.full_frame()
        pts = np.column_stack([xs, ys]).astype(np.int32)
        hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
        hull = _expand_polygon(hull, margin * max(w, h))
        poly = [(float(x / w), float(y / h)) for x, y in hull]
        roi = cls(xs.min() / w - margin, ys.min() / h - margin,
                  (xs.max() + 1) / w + margin, (ys.max() + 1) / h + margin,
                  poly, float(score), method)
        return roi.clipped()

    @classmethod
    def from_dict(cls, d: dict) -> PenROI:
        """Inverse of :meth:`as_dict` (also accepts a ``polygon`` JSON string)."""
        poly = d.get("polygon", [])
        if isinstance(poly, str):
            poly = json.loads(poly) if poly else []
        if isinstance(poly, float):  # NaN from a CSV
            poly = []
        return cls(float(d["fx0"]), float(d["fy0"]), float(d["fx1"]), float(d["fy1"]),
                   [tuple(p) for p in poly], float(d.get("score", 1.0)),
                   str(d.get("method", "manual")))

    # ------------------------------------------------------------ geometry
    def clipped(self) -> PenROI:
        """Clip the region to the frame (in place) and return it."""
        self.fx0, self.fy0 = max(0.0, self.fx0), max(0.0, self.fy0)
        self.fx1, self.fy1 = min(1.0, self.fx1), min(1.0, self.fy1)
        self.polygon = [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))
                        for x, y in self.polygon]
        return self

    @property
    def is_full_frame(self) -> bool:
        """True if the region is (almost) the whole frame."""
        return (self.fx0 <= 0.001 and self.fy0 <= 0.001 and self.fx1 >= 0.999
                and self.fy1 >= 0.999 and not self.polygon)

    @property
    def area_fraction(self) -> float:
        """Share of the frame inside the box."""
        return max(0.0, self.fx1 - self.fx0) * max(0.0, self.fy1 - self.fy0)

    def pixels(self, shape: tuple[int, int]) -> tuple[int, int, int, int]:
        """Integer box ``(x0, y0, x1, y1)`` for an image of ``shape = (H, W)``."""
        h, w = shape[:2]
        x0, y0 = int(np.floor(self.fx0 * w)), int(np.floor(self.fy0 * h))
        x1, y1 = int(np.ceil(self.fx1 * w)), int(np.ceil(self.fy1 * h))
        x0, y0 = min(max(0, x0), w - 1), min(max(0, y0), h - 1)
        return x0, y0, max(x0 + 1, min(w, x1)), max(y0 + 1, min(h, y1))

    def polygon_pixels(self, shape: tuple[int, int]) -> np.ndarray:
        """Outline polygon in pixels of an image of ``shape`` (``(N, 2)``)."""
        h, w = shape[:2]
        return np.array([(x * w, y * h) for x, y in self.polygon], np.float64).reshape(-1, 2)

    def inside_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Boolean full-frame mask of the pen outline (the box if no outline)."""
        h, w = shape[:2]
        out = np.zeros((h, w), np.uint8)
        if len(self.polygon) >= 3:
            cv2.fillPoly(out, [np.round(self.polygon_pixels(shape)).astype(np.int32)], 1)
        else:
            x0, y0, x1, y1 = self.pixels(shape)
            out[y0:y1, x0:x1] = 1
        return out.astype(bool)

    def crop(self, image: np.ndarray, mask_outside: bool = False,
             fill: int = FILL_GREY) -> np.ndarray:
        """Cut the pen out of a frame at full resolution.

        With ``mask_outside=True`` the pixels inside the box but outside the pen
        outline are painted grey, so the teacher cannot latch onto them.
        """
        x0, y0, x1, y1 = self.pixels(image.shape)
        out = image[y0:y1, x0:x1].copy()
        if mask_outside and len(self.polygon) >= 3:
            keep = self.inside_mask(image.shape)[y0:y1, x0:x1]
            out[~keep] = fill
        return out

    def paste(self, result: SegmentationResult, shape: tuple[int, int]) -> SegmentationResult:
        """Move a result computed on the crop back into full-frame coordinates."""
        h, w = shape[:2]
        x0, y0, x1, y1 = self.pixels(shape)
        insts = []
        for i in result.instances:
            m = np.zeros((h, w), bool)
            m[y0:y1, x0:x1] = i.mask[: y1 - y0, : x1 - x0]
            bx0, by0, bx1, by1 = i.box_xyxy
            insts.append(Instance(i.class_name, m, i.score,
                                  (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0), i.prompt))
        return SegmentationResult((h, w), insts, result.seconds, result.source)

    def cut(self, result: SegmentationResult, mask_outside: bool = False) -> SegmentationResult:
        """Inverse of :meth:`paste`: express a full-frame result in crop coordinates."""
        x0, y0, x1, y1 = self.pixels(result.image_shape)
        keep = self.inside_mask(result.image_shape)[y0:y1, x0:x1] if mask_outside else None
        insts = []
        for i in result.instances:
            m = i.mask[y0:y1, x0:x1] & (keep if keep is not None else True)
            if m.any():
                bx0, by0, bx1, by1 = i.box_xyxy
                insts.append(Instance(i.class_name, m, i.score,
                                      (bx0 - x0, by0 - y0, bx1 - x0, by1 - y0), i.prompt))
        return SegmentationResult((y1 - y0, x1 - x0), insts, result.seconds, result.source)

    def as_dict(self) -> dict:
        """Plain dict (JSON-friendly), e.g. for a model card."""
        return {"fx0": round(self.fx0, 5), "fy0": round(self.fy0, 5),
                "fx1": round(self.fx1, 5), "fy1": round(self.fy1, 5),
                "polygon": [(round(x, 5), round(y, 5)) for x, y in self.polygon],
                "score": round(self.score, 4), "method": self.method}


def _drop_small_parts(mask: np.ndarray, min_part: float) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n <= 2:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= min_part * areas.max()) + 1
    return np.isin(lab, keep)


def _expand_polygon(poly: np.ndarray, px: float) -> np.ndarray:
    """Push every vertex ``px`` pixels away from the centroid."""
    if px <= 0 or len(poly) < 3:
        return poly
    c = poly.mean(axis=0)
    d = poly - c
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return poly + d / np.maximum(n, 1e-9) * px


def box_iou_frac(a: PenROI, b: PenROI) -> float:
    """IoU of two regions' boxes (fractions)."""
    ix = max(0.0, min(a.fx1, b.fx1) - max(a.fx0, b.fx0))
    iy = max(0.0, min(a.fy1, b.fy1) - max(a.fy0, b.fy0))
    inter = ix * iy
    union = a.area_fraction + b.area_fraction - inter
    return inter / union if union > 0 else 0.0


# =========================================================================== detector
class PenDetector:
    """Find the pen with a text-prompted teacher (SAM 3 or Grounded SAM 2).

    The teacher's own prompts are swapped for the pen phrases while it runs, so
    the same loaded model does both stages (no second model in memory).

    Parameters
    ----------
    teacher : Segmenter
        A loaded teacher (``load_teacher(...)``) or any object whose
        ``segment`` returns instances of class ``"pen"``.
    phrases : sequence of str
        Text prompts for the pen.
    score_threshold : float, optional
        SAM 3 confidence threshold for the pen (the class threshold if None).
    """

    def __init__(self, teacher: Segmenter, phrases=PEN_PROMPTS,
                 score_threshold: float | None = 0.3) -> None:
        self.teacher = teacher
        self.phrases = list(phrases)
        self.score_threshold = score_threshold
        self.name = f"pen detector ({getattr(teacher, 'name', 'teacher')})"

    def segment(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Return candidate ``"pen"`` instances for one frame."""
        old = getattr(self.teacher, "prompts", None)
        if not isinstance(old, PromptConfig):
            return self.teacher.segment(image_rgb)  # e.g. a synthetic stand-in
        pen = PromptConfig(prompts={"pen": self.phrases},
                           score_threshold=self.score_threshold or old.score_threshold,
                           mask_threshold=old.mask_threshold)
        self.teacher.prompts = pen
        try:
            return self.teacher.segment(image_rgb)
        finally:
            self.teacher.prompts = old


def choose_pen(result: SegmentationResult, min_area: float = 0.05, max_area: float = 0.95,
               prefer: str = "score") -> Instance | None:
    """Pick the pen among candidate instances.

    Candidates covering less than ``min_area`` or more than ``max_area`` of the
    frame are ignored. ``prefer="score"`` takes the most confident one,
    ``"largest"`` the biggest.
    """
    h, w = result.image_shape
    cands = [i for i in result.of_class("pen") if min_area <= i.area / (h * w) <= max_area]
    if not cands:
        return None
    key = (lambda i: (i.score, i.area)) if prefer == "score" else (lambda i: (i.area, i.score))
    return max(cands, key=key)


def detect_pen_roi(frames, detector, margin: float = 0.03, min_area: float = 0.05,
                   max_area: float = 0.95, prefer: str = "score",
                   min_agreement: float = 0.5) -> tuple[PenROI, list[dict]]:
    """Pen region from one or more frames of the same camera view.

    The pen is found in every frame; frames whose box disagrees with the median
    box (IoU below ``min_agreement``) are dropped, and the rest are merged so a
    bird or person hiding part of the pen in one frame does not shrink the crop.

    Returns
    -------
    (PenROI, list of dict)
        The region (``method="full_frame"`` if no frame found a pen) and one
        diagnostic dict per frame.
    """
    found, diag = [], []
    shape = None
    for k, frame in enumerate(frames):
        shape = frame.shape[:2]
        t0 = time.perf_counter()
        res = detector.segment(frame)
        inst = choose_pen(res, min_area, max_area, prefer)
        roi = PenROI.from_mask(inst.mask, inst.score, 0.0) if inst is not None else None
        diag.append({"frame": k, "candidates": len(res.of_class("pen")),
                     "score": inst.score if inst else np.nan,
                     "area": inst.area / (shape[0] * shape[1]) if inst else np.nan,
                     "prompt": inst.prompt if inst else "", "seconds": time.perf_counter() - t0})
        if inst is not None:
            found.append((roi, inst))
    if not found:
        return PenROI.full_frame(), diag
    med = PenROI(*(float(np.median([r.__dict__[k] for r, _ in found]))
                   for k in ("fx0", "fy0", "fx1", "fy1")))
    keep = [(r, i) for r, i in found if box_iou_frac(r, med) >= min_agreement] or found
    union = np.zeros(shape, bool)
    for _, inst in keep:
        union |= inst.mask
    roi = PenROI.from_mask(union, float(np.mean([i.score for _, i in keep])), margin, "auto")
    return roi, diag


# =========================================================================== per video
def _sample_positions(n_frames: int, k: int) -> list[int]:
    """``k`` frame indices spread over the middle of a video."""
    if n_frames <= 1:
        return [0]
    pos = np.linspace(0.2, 0.8, max(1, k))
    return sorted({int(round(p * (n_frames - 1))) for p in pos})


def find_pen_rois(videos: pd.DataFrame, detector, frames_per_video: int = 2,
                  scope: str = "video", margin: float = 0.03, min_area: float = 0.05,
                  max_area: float = 0.95, prefer: str = "score",
                  cache: str | Path | None = None, progress: bool = True) -> pd.DataFrame:
    """Stage 1 for a table of videos (from :func:`broilerseg.io.scan_videos`).

    Parameters
    ----------
    videos : pandas.DataFrame
        Needs ``path``, ``video_id``, ``n_frames``; ``folder``, ``width`` and
        ``height`` are used if present.
    detector : PenDetector
        Usually ``PenDetector(teacher)``.
    frames_per_video : int
        Frames looked at per video (``scope="video"``) or per folder.
    scope : {"video", "folder"}
        ``"video"``: one region per video (safest if a camera was moved).
        ``"folder"``: one region per sub-folder, from frames spread over its
        videos (faster; fine when a folder is one fixed camera).
    margin : float
        Extra border around the pen, as a fraction of the frame.
    min_area, max_area : float
        Pen candidates outside this share of the frame are ignored.
    prefer : {"score", "largest"}
        Which candidate wins when several prompts return a pen.
    cache : path-like, optional
        CSV to resume from and to save to after every video.
    progress : bool
        Show a progress bar.

    Returns
    -------
    pandas.DataFrame
        One row per video: the region as fractions (``fx0``...), in pixels
        (``x0``...), ``score``, ``method`` and the outline as JSON. Videos where
        no pen was found get their folder's median region, else the full frame.
    """
    if scope not in ("video", "folder"):
        raise ValueError("scope must be 'video' or 'folder'")
    vids = videos.reset_index(drop=True)
    if "folder" not in vids:
        vids = vids.assign(folder=".")
    cache = Path(cache).expanduser() if cache else None
    old = pd.read_csv(cache) if cache and cache.is_file() else pd.DataFrame()
    if len(old) and "scope" in old and (old.scope != scope).any():
        old = pd.DataFrame()  # a different scope: start again
    done = set(old.video_id) if "video_id" in old else set()
    rows = [] if old.empty else old.to_dict("records")

    units = ([(r.video_id, vids.iloc[[i]]) for i, r in vids.iterrows()] if scope == "video"
             else list(vids.groupby("folder", sort=True)))
    units = [(k, g) for k, g in units if not set(g.video_id) <= done]
    bar = None
    if progress and units:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(units), desc=f"Stage 1: finding the pen ({scope})")
        except ImportError:  # pragma: no cover
            bar = None
    for _, g in units:
        frames = []
        if scope == "video":
            r = g.iloc[0]
            frames = [read_frame(r.path, i) for i in _sample_positions(int(r.n_frames),
                                                                       frames_per_video)]
        else:  # spread over the folder's videos
            pick = g.iloc[np.linspace(0, len(g) - 1, min(len(g), frames_per_video)).astype(int)]
            reps = int(np.ceil(frames_per_video / len(pick)))
            for r in pick.itertuples():
                frames += [read_frame(r.path, i) for i in _sample_positions(int(r.n_frames),
                                                                            reps)]
            frames = frames[:max(1, frames_per_video)]
        frames = [f for f in frames if f is not None]
        roi, diag = (detect_pen_roi(frames, detector, margin, min_area, max_area, prefer)
                     if frames else (PenROI.full_frame(), []))
        method = ("auto" if scope == "video" else "auto_folder") if roi.method == "auto" \
            else "failed"
        for r in g.itertuples():
            rows.append(_roi_row(r, roi, method, scope,
                                 float(np.nansum([d["seconds"] for d in diag]))))
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(cache, index=False)
        if bar:
            bar.update(1)
    if bar:
        bar.close()
    out = fill_failed_rois(pd.DataFrame(rows))
    keep = set(vids.video_id)
    out = out[out.video_id.isin(keep)].reset_index(drop=True)
    if cache:
        out.to_csv(cache, index=False)
    return out


def _roi_row(r, roi: PenROI, method: str, scope: str, seconds: float) -> dict:
    w = int(getattr(r, "width", 0) or 0)
    h = int(getattr(r, "height", 0) or 0)
    x0, y0, x1, y1 = roi.pixels((h, w)) if w and h else (0, 0, 0, 0)
    return {"video_id": r.video_id, "folder": getattr(r, "folder", "."), "path": str(r.path),
            "width": w, "height": h, "fx0": roi.fx0, "fy0": roi.fy0, "fx1": roi.fx1,
            "fy1": roi.fy1, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "crop_width": x1 - x0, "crop_height": y1 - y0,
            "area_fraction": roi.area_fraction, "score": roi.score, "method": method,
            "scope": scope, "seconds": seconds, "polygon": json.dumps(roi.polygon)}


def fill_failed_rois(rois: pd.DataFrame) -> pd.DataFrame:
    """Give videos where no pen was found their folder's median region (else full frame)."""
    if rois.empty:
        return rois
    rois = rois.copy()
    ok = ~rois.method.isin(["failed", "full_frame", "folder_median"])
    for i in np.flatnonzero(~ok.to_numpy()):
        r = rois.iloc[i]
        same = rois[ok & (rois.folder == r.folder)]
        if len(same):
            roi = PenROI(*(float(same[k].median()) for k in ("fx0", "fy0", "fx1", "fy1")),
                         [], float(same.score.median()), "folder_median")
        else:
            roi = PenROI.full_frame()
        _set_row(rois, i, roi)
    return rois


def _set_row(rois: pd.DataFrame, i: int, roi: PenROI) -> None:
    r = rois.iloc[i]
    w, h = int(r.width or 0), int(r.height or 0)
    x0, y0, x1, y1 = roi.pixels((h, w)) if w and h else (0, 0, 0, 0)
    vals = {"fx0": roi.fx0, "fy0": roi.fy0, "fx1": roi.fx1, "fy1": roi.fy1, "x0": x0, "y0": y0,
            "x1": x1, "y1": y1, "crop_width": x1 - x0, "crop_height": y1 - y0,
            "area_fraction": roi.area_fraction, "score": roi.score, "method": roi.method,
            "polygon": json.dumps(roi.polygon)}
    for k, v in vals.items():
        rois.iloc[i, rois.columns.get_loc(k)] = v


def apply_manual_rois(rois: pd.DataFrame, manual: dict | None) -> pd.DataFrame:
    """Override regions by hand.

    ``manual`` maps a ``video_id``, a folder (e.g. ``"2025-09-27/ch00_3840x2160"``)
    or ``"*"`` (every video) to a box ``(x0, y0, x1, y1)``: pixels of that
    video's frames, or fractions if every value is <= 1. Video ids win over
    folders, folders over ``"*"``.
    """
    if not manual:
        return rois
    rois = rois.copy()
    for i, r in enumerate(rois.itertuples()):
        folder = str(r.folder)
        key = next((k for k in (r.video_id, folder) if k in manual), None)
        if key is None:
            key = next((k for k in manual if k != "*" and folder.startswith(str(k))), None)
        if key is None and "*" in manual:
            key = "*"
        if key is None:
            continue
        roi = PenROI.from_box(manual[key], (int(r.width), int(r.height)))
        _set_row(rois, i, roi)
    return rois


def rois_lookup(rois: pd.DataFrame | dict | None) -> dict[str, PenROI]:
    """``video_id -> PenROI`` from a table made by :func:`find_pen_rois`."""
    if rois is None:
        return {}
    if isinstance(rois, dict):
        return {k: v if isinstance(v, PenROI) else PenROI.from_dict(v) for k, v in rois.items()}
    return {r["video_id"]: PenROI.from_dict(r) for r in rois.to_dict("records")}


def typical_roi(rois: pd.DataFrame) -> PenROI:
    """One region for deployment: the median box, with the outline of the closest video."""
    good = rois[rois.method != "full_frame"]
    if good.empty:
        return PenROI.full_frame()
    med = PenROI(*(float(good[k].median()) for k in ("fx0", "fy0", "fx1", "fy1")),
                 [], float(good.score.median()), "typical")
    best = max(good.to_dict("records"), key=lambda r: box_iou_frac(PenROI.from_dict(r), med))
    med.polygon = PenROI.from_dict(best).polygon
    return med


def save_rois(rois: pd.DataFrame, path: str | Path) -> Path:
    """Write the region table as CSV."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    rois.to_csv(path, index=False)
    return path


def load_rois(path: str | Path) -> pd.DataFrame:
    """Read a region table written by :func:`save_rois` / :func:`find_pen_rois`."""
    return pd.read_csv(Path(path).expanduser())


# =========================================================================== wrapper
class CroppedSegmenter:
    """Run any segmenter on the pen crop; return masks in full-frame coordinates.

    Parameters
    ----------
    inner : Segmenter
        The stage-2 model (SAM 3, Grounded SAM 2 or the YOLO student).
    roi : PenROI, optional
        A fixed pen region (fixed camera). If omitted, ``detector`` finds it.
    detector : PenDetector, optional
        Finds the region when ``roi`` is not given.
    mask_outside : bool
        Grey out pixels inside the box but outside the pen outline.
    margin : float
        Border added around a detected pen.
    redetect : {"once", "every"}
        With a detector: find the pen once per frame size (fixed camera,
        fast) or in every frame.

    Other attributes (``device``, ``imgsz``, ...) are passed through to ``inner``.
    """

    def __init__(self, inner: Segmenter, roi: PenROI | None = None, detector=None,
                 mask_outside: bool = False, margin: float = 0.03,
                 redetect: str = "once") -> None:
        self.inner = inner
        self.roi = roi
        self.detector = detector
        self.mask_outside = mask_outside
        self.margin = margin
        self.redetect = redetect
        self._cache: dict[tuple[int, int], PenROI] = {}
        self.last_roi: PenROI | None = None
        self.name = f"{getattr(inner, 'name', 'segmenter')} on pen crop"

    def __getattr__(self, item):  # pass-through for device, imgsz, prompts, ...
        if item == "inner":
            raise AttributeError(item)
        return getattr(self.inner, item)

    def roi_for(self, image_rgb: np.ndarray) -> PenROI:
        """Return the region used for this image."""
        if self.roi is not None:
            return self.roi
        if self.detector is None:
            return PenROI.full_frame()
        key = image_rgb.shape[:2]
        if self.redetect == "every" or key not in self._cache:
            self._cache[key] = detect_pen_roi([image_rgb], self.detector, self.margin)[0]
        return self._cache[key]

    def segment_crop(self, image_rgb: np.ndarray
                     ) -> tuple[np.ndarray, SegmentationResult, PenROI]:
        """``(crop, result in crop coordinates, region)``."""
        roi = self.roi_for(image_rgb)
        crop = roi.crop(image_rgb, self.mask_outside)
        return crop, self.inner.segment(crop), roi

    def segment(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Segment the pen crop and return full-frame masks."""
        t0 = time.perf_counter()
        _, res, roi = self.segment_crop(image_rgb)
        self.last_roi = roi
        out = roi.paste(res, image_rgb.shape[:2])
        out.seconds = time.perf_counter() - t0
        out.source = self.name
        return out
