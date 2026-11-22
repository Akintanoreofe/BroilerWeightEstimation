"""End-to-end weight experiment: segment -> calibrate -> select bird -> volume -> mass.

The same class runs with either segmenter (SAM 3 teacher or YOLO student), so
the two can be compared frame by frame on the FlockLabel validation data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from broilerseg.calibration import Calibration, calibrate_from_mask
from broilerseg.config import FEEDER_DIAMETER_CM, VolumeConfig
from broilerseg.io.flocklabel import iter_sample_images
from broilerseg.segmentation.base import Instance, SegmentationResult, Segmenter
from broilerseg.segmentation.postprocess import clean_mask, select_bird, touches_border
from broilerseg.volume import VolumeEstimate, estimate_volume, implied_density


def _progress(iterable, total=None, desc=""):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except ImportError:  # pragma: no cover - tqdm is optional
        return iterable


@dataclass
class WeightExperiment:
    """Run the full weight-estimation pipeline on FlockLabel samples.

    Parameters
    ----------
    segmenter : Segmenter
        ``Sam3Segmenter``, ``YoloSegmenter`` or anything with ``segment()``.
    volume : VolumeConfig
        Density, slice count and alignment.
    calibration_method : str
        Feeder measurement method (see :mod:`broilerseg.calibration`).
    calibration_mode : {"video_median", "frame", "export_hint"}
        ``video_median`` uses the median scale of all frames of a video;
        ``frame`` uses each frame's own feeder; ``export_hint`` uses
        ``pixel_to_cm`` from FlockLabel's ``events.csv`` when present.
    feeder_diameter_cm : float
        Known feeder diameter.
    use_gt_box : bool
        Pick the bird that best matches the FlockLabel box (recommended for
        validation); otherwise pick the bird standing on the scale.
    border_margin : int
        Birds within this many pixels of the frame edge are flagged.
    """

    segmenter: Segmenter
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    calibration_method: str = "enclosing_circle"
    calibration_mode: str = "video_median"
    feeder_diameter_cm: float = FEEDER_DIAMETER_CM
    use_gt_box: bool = True
    border_margin: int = 10
    results: dict[int, SegmentationResult] = field(default_factory=dict, repr=False)
    calibrations: dict[int, Calibration] = field(default_factory=dict, repr=False)
    birds: dict[int, Instance] = field(default_factory=dict, repr=False)
    estimates: dict[int, VolumeEstimate] = field(default_factory=dict, repr=False)
    records: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    samples: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    # ----------------------------------------------------------------- running
    def segment(self, samples: pd.DataFrame) -> None:
        """Segment every sample (results are cached by ``sample_id``)."""
        self.samples = samples.reset_index(drop=True)
        todo = self.samples[~self.samples.sample_id.isin(self.results)]
        for row, image in _progress(iter_sample_images(todo), total=len(todo),
                                    desc=f"Segmenting with {self.segmenter.name}"):
            self.results[int(row.sample_id)] = self.segmenter.segment(image)

    def run(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Segment (if needed), calibrate, estimate and return the records table."""
        self.segment(samples)
        rows = []
        for _, s in self.samples.iterrows():
            sid = int(s.sample_id)
            res = self.results.get(sid)
            feeder = res.best("feeder") if res else None
            cal = calibrate_from_mask(feeder.mask if feeder else None, self.feeder_diameter_cm,
                                      self.calibration_method)
            self.calibrations[sid] = cal
            gt = (s.x1, s.y1, s.x2, s.y2) if self.use_gt_box else None
            bird, info = select_bird(res, gt) if res else (None, {"reason": "unreadable"})
            rows.append({
                "sample_id": sid, "video_name": s.video_name, "event_id": str(s.event_id),
                "frame_label": s.frame_label, "actual_kg": s.weight_kg,
                "event_weight_kg": s.event_weight_kg, "status": info.get("reason", "ok"),
                "n_broiler": len(res.of_class("broiler")) if res else 0,
                "n_scale": len(res.of_class("scale")) if res else 0,
                "n_feeder": len(res.of_class("feeder")) if res else 0,
                "bird_score": bird.score if bird else np.nan,
                "box_iou": info.get("box_iou", np.nan), "on_scale": info.get("on_scale", np.nan),
                "feeder_score": feeder.score if feeder else np.nan,
                "feeder_diameter_px": cal.diameter_px, "pixel_to_cm_frame": cal.pixel_to_cm,
                "pixel_to_cm_hint": s.pixel_to_cm_hint,
                "seg_seconds": res.seconds if res else np.nan,
            })
            if bird is not None:
                self.birds[sid] = Instance(bird.class_name, clean_mask(bird.mask), bird.score,
                                           bird.box_xyxy, bird.prompt)
        rec = pd.DataFrame(rows)
        rec["pixel_to_cm"] = self._choose_scale(rec)
        rec = self._estimate(rec)
        self.records = rec
        return rec

    def _choose_scale(self, rec: pd.DataFrame) -> pd.Series:
        frame = rec.pixel_to_cm_frame
        median = rec.groupby("video_name").pixel_to_cm_frame.transform("median")
        if self.calibration_mode == "frame":
            return frame
        if self.calibration_mode == "export_hint":
            return rec.pixel_to_cm_hint.where(np.isfinite(rec.pixel_to_cm_hint), median)
        if self.calibration_mode == "video_median":
            return median
        raise ValueError("calibration_mode must be 'video_median', 'frame' or 'export_hint'")

    def _estimate(self, rec: pd.DataFrame) -> pd.DataFrame:
        extra = []
        for _, r in rec.iterrows():
            sid = int(r.sample_id)
            out = {"touches_border": np.nan, "bird_area_px": np.nan, "volume_cm3": np.nan,
                   "pred_kg": np.nan, "length_cm": np.nan, "max_width_cm": np.nan,
                   "area_cm2": np.nan, "angle_deg": np.nan, "status": r.status}
            bird = self.birds.get(sid)
            if bird is not None:
                out["touches_border"] = touches_border(bird.mask, self.border_margin)
                out["bird_area_px"] = bird.area
                if not np.isfinite(r.pixel_to_cm):
                    out["status"] = "no_feeder"
                else:
                    est = estimate_volume(bird.mask, float(r.pixel_to_cm),
                                          self.volume.density_kg_m3, self.volume.slice_count,
                                          self.volume.align)
                    self.estimates[sid] = est
                    s = est.summary()
                    out.update({k: s[k] for k in ("volume_cm3", "length_cm", "max_width_cm",
                                                  "area_cm2", "angle_deg")})
                    out["pred_kg"] = s["mass_kg"]
            extra.append(out)
        rec = rec.drop(columns="status").join(pd.DataFrame(extra, index=rec.index))
        rec["error_kg"] = rec.pred_kg - rec.actual_kg
        rec["pct_error"] = 100 * rec.error_kg / rec.actual_kg
        rec["implied_density"] = implied_density(rec.actual_kg, rec.volume_cm3)
        return rec

    # ----------------------------------------------------------------- analysis
    def volume_table(self, slice_counts=(None, 5, 10, 15, 20, 30, 40)) -> pd.DataFrame:
        """Volume of every estimated bird for several slice counts (long table)."""
        rows = []
        valid = self.records.dropna(subset=["volume_cm3"])
        for _, r in valid.iterrows():
            mask = self.birds[int(r.sample_id)].mask
            for k in slice_counts:
                est = estimate_volume(mask, float(r.pixel_to_cm), slice_count=k,
                                      align=self.volume.align)
                rows.append({"sample_id": int(r.sample_id), "event_id": r.event_id,
                             "slice_count": "per-column" if k is None else int(k),
                             "volume_cm3": est.volume_cm3, "actual_kg": r.actual_kg})
        return pd.DataFrame(rows)

    def image(self, sample_id: int) -> np.ndarray:
        """Re-read the RGB image of a sample (images are not kept in memory)."""
        row = self.samples[self.samples.sample_id == sample_id]
        for _, img in iter_sample_images(row):
            return img
        raise RuntimeError(f"Cannot read sample {sample_id}")
