"""Feeder-relative candidate selection, tracking, and measurement quality."""

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from defaults import (
    MAX_DIST_FEEDER_RADII,
    MAX_FEEDER_CONTACT_RATIO,
    MAX_PLAUSIBLE_BIRD_AREA_CM2,
    MAX_TEMPORAL_AREA_DEVIATION_RATIO,
    MAX_TRACK_JUMP_CM,
    MAX_BIRD_ELLIPSE_ASPECT_RATIO,
    MIN_BIRD_SOLIDITY_RATIO,
    MIN_BIRD_AREA_CM2,
    MIN_PARALLEL_OCCLUSION_EDGE_CM,
)
from SegmentationStrategy import BirdCandidate, ElongatedComponent


@dataclass
class SelectionResult:
    """All candidate diagnostics and the closest feeder-relative candidate."""

    candidates: List[BirdCandidate]
    selected: Optional[BirdCandidate]


class FeederRelativeBirdSelector:
    """Select and quality-gate the bird nearest the feeder without a scene ROI."""

    def __init__(
        self,
        feeder_reference,
        max_dist_feeder_radii: float = MAX_DIST_FEEDER_RADII,
        max_track_jump_cm: float = MAX_TRACK_JUMP_CM,
        max_feeder_contact_ratio: float = MAX_FEEDER_CONTACT_RATIO,
    ):
        if max_dist_feeder_radii <= 0 or max_track_jump_cm <= 0:
            raise ValueError("Distance and track-jump limits must be positive.")
        if not 0 <= max_feeder_contact_ratio <= 1:
            raise ValueError("max_feeder_contact_ratio must be in [0, 1].")
        self.reference = feeder_reference
        self.max_dist_feeder_radii = max_dist_feeder_radii
        self.max_track_jump_cm = max_track_jump_cm
        self.max_feeder_contact_ratio = max_feeder_contact_ratio
        self._next_track_id = 1
        self._active_track_id = None
        self._last_center_cm = None
        self._area_history: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=5))

    def select(
        self,
        candidates: List[BirdCandidate],
        frame: np.ndarray,
        occluders: Optional[List[ElongatedComponent]] = None,
    ) -> SelectionResult:
        """Attach quality metrics to every candidate and choose one bird."""
        occluders = occluders or []
        rectified_frame = self.reference.rectify(frame)
        rectified_occluders = [self.reference.rectify(item.mask, is_mask=True) for item in occluders]
        occluder_mask = np.zeros(rectified_frame.shape[:2], np.uint8)
        for mask in rectified_occluders:
            if mask.shape == occluder_mask.shape:
                occluder_mask = cv2.bitwise_or(occluder_mask, mask)
        dilation_size = max(1, int(round(self.reference.px_per_cm)))
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_size | 1, dilation_size | 1)
        )
        dilated_occluders = cv2.dilate(occluder_mask, dilation_kernel)

        eligible = []
        for candidate in candidates:
            rectified_mask = self.reference.rectify(candidate.mask, is_mask=True)
            ys, xs = np.where(rectified_mask > 0)
            if len(xs) == 0:
                candidate.quality = {"rejection_reason": "empty_mask"}
                candidate.measurable = False
                continue
            center_x, center_y = float(xs.mean()), float(ys.mean())
            feeder_center = self.reference.rectified_center_px
            distance_px = math.hypot(center_x - feeder_center[0], center_y - feeder_center[1])
            distance_radii = distance_px / (self.reference.major_px / 2.0)
            candidate.quality = self._quality(
                candidate, rectified_mask, rectified_frame, dilated_occluders,
                distance_radii,
            )
            candidate.quality["dist_feeder_radii"] = distance_radii
            candidate.quality["dist_feeder_cm"] = distance_px / self.reference.px_per_cm
            candidate.quality["border_contact"] = self._touches_border(candidate, frame.shape)
            candidate.measurable = True
            if candidate.quality["border_contact"]:
                self._reject(candidate, "frame_border")
            elif distance_radii > self.max_dist_feeder_radii:
                self._reject(candidate, "outside_feeder_distance")
            else:
                eligible.append((distance_radii, candidate, rectified_mask))

        eligible.sort(key=lambda row: row[0])
        selected = eligible[0][1] if eligible else None
        if selected is None:
            self._active_track_id = None
            self._last_center_cm = None
            return SelectionResult(candidates, None)

        _, selected, selected_mask = eligible[0]
        center = selected.quality["centroid_rectified_px"]
        center_cm = (
            (center[0] - self.reference.rectified_center_px[0]) / self.reference.px_per_cm,
            (center[1] - self.reference.rectified_center_px[1]) / self.reference.px_per_cm,
        )
        if (self._active_track_id is None or self._last_center_cm is None
                or math.dist(center_cm, self._last_center_cm) > self.max_track_jump_cm):
            self._active_track_id = self._next_track_id
            self._next_track_id += 1
            self._area_history[self._active_track_id].clear()
        selected.track_id = int(self._active_track_id)
        self._last_center_cm = center_cm

        history = self._area_history[selected.track_id]
        if history:
            median_area = float(np.median(history))
            area_deviation = abs(selected.area_cm2 - median_area) / max(median_area, 1e-6)
            selected.quality["temporal_area_deviation"] = area_deviation
            if area_deviation > MAX_TEMPORAL_AREA_DEVIATION_RATIO:
                self._reject(selected, "temporal_area_change")
        history.append(selected.area_cm2)
        if selected.quality["feeder_contact"] > self.max_feeder_contact_ratio:
            self._reject(selected, "feeder_contact")
        if selected.quality["occluder_contact"] > 0:
            self._reject(selected, "occluder_contact")
        if selected.quality["parallel_edge_occlusion"]:
            self._reject(selected, "parallel_edge_occlusion")
        if not selected.quality["plausible_area"]:
            self._reject(selected, "implausible_area")
        if not selected.quality["shape_complete"]:
            self._reject(selected, "incomplete_shape")
        return SelectionResult(candidates, selected)

    def _quality(self, candidate, mask, frame, occluders, distance_radii):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return {"rejection_reason": "empty_mask"}
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area else 0.0
        if len(contour) >= 5:
            fitted = cv2.fitEllipse(contour)
            ellipse_aspect = max(fitted[1]) / max(min(fitted[1]), 1e-6)
        else:
            ellipse_aspect = 1.0
        feeder_center = self.reference.rectified_center_px
        feeder_radius = self.reference.major_px / 2.0
        points = contour[:, 0, :].astype(np.float32)
        radial_distance = np.linalg.norm(points - np.asarray(feeder_center), axis=1)
        near_feeder = (radial_distance <= feeder_radius + self.reference.px_per_cm) | (
            radial_distance <= feeder_radius
        )
        feeder_contact = float(np.mean(near_feeder))
        contour_mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, 1)
        contour_pixels = contour_mask > 0
        occluder_contact = (
            float(np.count_nonzero(contour_pixels & (occluders > 0)) / np.count_nonzero(contour_pixels))
            if np.count_nonzero(contour_pixels) else 0.0
        )
        x, y, width, height = cv2.boundingRect(contour)
        crop = frame[y:y + height, x:x + width]
        sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                                        cv2.CV_64F).var()) if crop.size else 0.0
        parallel_edge = self._parallel_edge_occlusion(contour_mask, occluders)
        area_cm2 = float(cv2.countNonZero(mask) / self.reference.px_per_cm ** 2)
        ys, xs = np.where(mask > 0)
        return {
            "feeder_contact": feeder_contact,
            "occluder_contact": occluder_contact,
            "solidity": float(solidity),
            "ellipse_aspect": float(ellipse_aspect),
            "sharpness": sharpness,
            "plausible_area": MIN_BIRD_AREA_CM2 <= area_cm2 <= MAX_PLAUSIBLE_BIRD_AREA_CM2,
            "parallel_edge_occlusion": parallel_edge,
            "centroid_rectified_px": (float(xs.mean()), float(ys.mean())),
            "dist_feeder_radii": distance_radii,
            "area_cm2": area_cm2,
            "shape_complete": solidity >= MIN_BIRD_SOLIDITY_RATIO
                              and ellipse_aspect <= MAX_BIRD_ELLIPSE_ASPECT_RATIO,
        }

    def _parallel_edge_occlusion(self, contour_mask, occluders):
        if not np.any(occluders):
            return False
        min_line_px = max(3, int(round(MIN_PARALLEL_OCCLUSION_EDGE_CM * self.reference.px_per_cm)))
        max_gap_px = max(1, int(round(self.reference.px_per_cm)))
        bird_lines = cv2.HoughLinesP(cv2.Canny(contour_mask, 50, 150), 1, np.pi / 180,
                                     threshold=min_line_px, minLineLength=min_line_px,
                                     maxLineGap=max_gap_px)
        occluder_lines = cv2.HoughLinesP(cv2.Canny(occluders, 50, 150), 1, np.pi / 180,
                                         threshold=min_line_px, minLineLength=min_line_px,
                                         maxLineGap=max_gap_px)
        if bird_lines is None or occluder_lines is None:
            return False
        proximity_px = 2.0 * self.reference.px_per_cm
        for line in np.asarray(bird_lines).reshape(-1, 4):
            x1, y1, x2, y2 = line
            angle = math.atan2(y2 - y1, x2 - x1)
            midpoint = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            for other in np.asarray(occluder_lines).reshape(-1, 4):
                ox1, oy1, ox2, oy2 = other
                other_angle = math.atan2(oy2 - oy1, ox2 - ox1)
                delta = abs((angle - other_angle + math.pi / 2) % math.pi - math.pi / 2)
                other_midpoint = np.array([(ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0])
                if delta < math.radians(10) and np.linalg.norm(midpoint - other_midpoint) <= proximity_px:
                    return True
        return False

    @staticmethod
    def _touches_border(candidate, frame_shape):
        x, y, width, height = candidate.bbox
        return x <= 0 or y <= 0 or x + width >= frame_shape[1] or y + height >= frame_shape[0]

    @staticmethod
    def _reject(candidate, reason):
        candidate.measurable = False
        reasons = candidate.quality.setdefault("rejection_reasons", [])
        if reason not in reasons:
            reasons.append(reason)
