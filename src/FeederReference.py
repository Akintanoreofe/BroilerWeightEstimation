"""Robust, per-video feeder calibration and planar-view rectification."""

from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from defaults import (
    FEEDER_DIAMETER_CM,
    MIN_FEEDER_ARC_COVERAGE_RATIO,
    PLANE_CORRECTION,
    RANSAC_INLIER_TOLERANCE_RATIO,
    RANSAC_ITERATIONS,
)


@dataclass(frozen=True)
class _FrameFit:
    ellipse: Tuple[float, float, float, float, float]
    points: np.ndarray
    inliers: np.ndarray
    coverage_ratio: float
    enclosing_circle: Tuple[Tuple[float, float], float]
    frame: np.ndarray


class FeederReference:
    """Video-specific feeder size, centre, viewing angle, and rectification.

    All frame measurements are newly estimated for each invocation. The
    feeder's major ellipse axis supplies the un-foreshortened diameter.
    """

    def __init__(
        self,
        center_px: Tuple[float, float],
        major_px: float,
        minor_px: float,
        angle_deg: float,
        calib_uncertainty_pct: float,
        frame_shape: Tuple[int, ...],
        debug_fit: _FrameFit,
        feeder_diameter_cm: float = FEEDER_DIAMETER_CM,
        plane_correction: float = PLANE_CORRECTION,
    ):
        self.center_px = (float(center_px[0]), float(center_px[1]))
        self.major_px = float(major_px)
        self.minor_px = float(minor_px)
        self.angle_deg = float(angle_deg)
        self.calib_uncertainty_pct = float(calib_uncertainty_pct)
        self.feeder_diameter_cm = float(feeder_diameter_cm)
        self.plane_correction = float(plane_correction)
        self.px_per_cm = self.major_px / self.feeder_diameter_cm * self.plane_correction
        self.frame_shape = tuple(frame_shape)
        self._debug_fit = debug_fit
        self.minor_major_ratio = self.minor_px / self.major_px
        self.rectified_center_px = self._transform_point(self.center_px)
        print(
            f"[FeederReference] px_per_cm={self.px_per_cm:.6f}, "
            f"minor/major={self.minor_major_ratio:.4f}, "
            f"calib_uncertainty_pct={self.calib_uncertainty_pct:.3f}"
        )
        old_diameter_px = 2.0 * debug_fit.enclosing_circle[1]
        print(
            f"[FeederReference] old enclosing-circle pixel_to_cm="
            f"{self.feeder_diameter_cm / old_diameter_px:.6f}; "
            f"robust ellipse pixel_to_cm={1.0 / self.px_per_cm:.6f}"
        )
        if self.minor_major_ratio < 0.85:
            print("[FeederReference] Warning: camera is too oblique for reliable planar correction.")

    @classmethod
    def from_video(
        cls,
        video_path: str,
        sample_count: int = 30,
        feeder_diameter_cm: float = FEEDER_DIAMETER_CM,
        plane_correction: float = PLANE_CORRECTION,
        min_good_fits: int = 5,
    ) -> "FeederReference":
        """Calibrate from frames sampled uniformly across a video."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video for feeder calibration: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            capture.release()
            raise RuntimeError(f"Video has no readable frame count: {video_path}")
        positions = np.unique(np.linspace(0, frame_count - 1, min(sample_count, frame_count), dtype=int))
        frames = []
        try:
            for position in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                ok, frame = capture.read()
                if ok:
                    frames.append(frame)
        finally:
            capture.release()
        return cls.from_frames(
            frames,
            feeder_diameter_cm=feeder_diameter_cm,
            plane_correction=plane_correction,
            min_good_fits=min_good_fits,
        )

    @classmethod
    def from_frames(
        cls,
        frames: Iterable[np.ndarray],
        feeder_diameter_cm: float = FEEDER_DIAMETER_CM,
        plane_correction: float = PLANE_CORRECTION,
        min_good_fits: int = 5,
    ) -> "FeederReference":
        """Build one robust reference from a sequence of sampled BGR frames."""
        if feeder_diameter_cm <= 0 or not math.isfinite(feeder_diameter_cm):
            raise ValueError("feeder_diameter_cm must be positive and finite")
        if plane_correction <= 0 or not math.isfinite(plane_correction):
            raise ValueError("plane_correction must be positive and finite")
        fits = []
        for frame in frames:
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                continue
            fit = cls._fit_frame(frame)
            if fit is not None and fit.coverage_ratio >= MIN_FEEDER_ARC_COVERAGE_RATIO:
                fits.append(fit)
        if len(fits) < min_good_fits:
            raise RuntimeError(
                f"Feeder calibration found only {len(fits)} good fits; "
                f"at least {min_good_fits} are required."
            )

        ellipses = np.asarray([fit.ellipse for fit in fits], dtype=np.float64)
        center = np.median(ellipses[:, :2], axis=0)
        major = float(np.median(ellipses[:, 2]))
        minor = float(np.median(ellipses[:, 3]))
        angle_radians = np.deg2rad(ellipses[:, 4] * 2.0)
        angle = math.degrees(
            math.atan2(float(np.median(np.sin(angle_radians))),
                       float(np.median(np.cos(angle_radians)))) / 2.0
        ) % 180.0
        mad_pct = 100.0 * float(np.median(np.abs(ellipses[:, 2] - major))) / major
        chosen = min(fits, key=lambda fit: abs(fit.ellipse[2] - major))
        return cls(
            center_px=(center[0], center[1]),
            major_px=major,
            minor_px=minor,
            angle_deg=angle,
            calib_uncertainty_pct=mad_pct,
            frame_shape=chosen.frame.shape,
            debug_fit=chosen,
            feeder_diameter_cm=feeder_diameter_cm,
            plane_correction=plane_correction,
        )

    @staticmethod
    def _adaptive_orange_mask(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        saturated = (saturation > 60) & (value > 40)
        if np.count_nonzero(saturated) >= 5:
            histogram = np.bincount(hue[saturated], minlength=180).astype(np.float32)
            histogram = cv2.GaussianBlur(histogram.reshape(1, -1), (9, 1), 0).ravel()
            dominant_hue = int(np.argmax(histogram))
            hue_delta = np.abs(hue.astype(np.int16) - dominant_hue)
            hue_delta = np.minimum(hue_delta, 180 - hue_delta)
            hue_mask = hue_delta <= 8
        else:
            hue_mask = (hue >= 8) & (hue <= 20)
        mask = (hue_mask & (saturation > 60) & (value > 40)).astype(np.uint8) * 255
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    @classmethod
    def _fit_frame(cls, frame: np.ndarray) -> Optional[_FrameFit]:
        orange = cls._adaptive_orange_mask(frame)
        contours, _ = cv2.findContours(orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [contour for contour in contours if len(contour) >= 5]
        if not contours:
            return None
        points = np.vstack(contours).reshape(-1, 2).astype(np.float32)
        if len(points) < 5:
            return None

        circle_center, circle_radius = cv2.minEnclosingCircle(points.reshape(-1, 1, 2))
        rng = np.random.default_rng(0)
        best = None
        tolerance_px = None
        for _ in range(RANSAC_ITERATIONS):
            sample = points[rng.choice(len(points), 5, replace=False)]
            try:
                candidate = cv2.fitEllipse(sample.reshape(-1, 1, 2))
            except cv2.error:
                continue
            major = max(candidate[1])
            minor = min(candidate[1])
            if major <= 0 or minor / major < 0.4:
                continue
            tolerance_px = max(1.0, major * RANSAC_INLIER_TOLERANCE_RATIO)
            errors, angles = cls._ellipse_errors(points, candidate)
            inliers = errors <= tolerance_px
            coverage = cls._angular_coverage(angles[inliers])
            if coverage < MIN_FEEDER_ARC_COVERAGE_RATIO:
                continue
            score = (int(np.count_nonzero(inliers)), coverage, -float(np.median(errors[inliers])))
            if best is None or score > best[0]:
                best = (score, candidate, inliers)
        if best is None:
            return None

        try:
            refined = cv2.fitEllipse(points[best[2]].reshape(-1, 1, 2))
        except cv2.error:
            return None
        errors, angles = cls._ellipse_errors(points, refined)
        tolerance_px = max(1.0, max(refined[1]) * RANSAC_INLIER_TOLERANCE_RATIO)
        inliers = errors <= tolerance_px
        coverage = cls._angular_coverage(angles[inliers])
        if coverage < MIN_FEEDER_ARC_COVERAGE_RATIO or np.count_nonzero(inliers) < 5:
            return None

        (center_x, center_y), (axis_a, axis_b), cv_angle = refined
        if axis_a >= axis_b:
            major, minor, angle = axis_a, axis_b, cv_angle
        else:
            major, minor, angle = axis_b, axis_a, cv_angle + 90.0
        ellipse = (float(center_x), float(center_y), float(major), float(minor), float(angle % 180.0))
        return _FrameFit(
            ellipse=ellipse,
            points=points,
            inliers=points[inliers],
            coverage_ratio=coverage,
            enclosing_circle=(circle_center, float(circle_radius)),
            frame=frame.copy(),
        )

    @staticmethod
    def _ellipse_errors(points: np.ndarray, ellipse) -> Tuple[np.ndarray, np.ndarray]:
        (center_x, center_y), (axis_a, axis_b), angle_deg = ellipse
        if axis_a <= 0 or axis_b <= 0:
            return np.full(len(points), np.inf), np.zeros(len(points))
        theta = math.radians(angle_deg)
        dx = points[:, 0] - center_x
        dy = points[:, 1] - center_y
        x_rot = math.cos(theta) * dx + math.sin(theta) * dy
        y_rot = -math.sin(theta) * dx + math.cos(theta) * dy
        a, b = axis_a / 2.0, axis_b / 2.0
        radial = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)
        errors = np.abs(radial - 1.0) * min(a, b)
        angles = np.arctan2(y_rot / b, x_rot / a)
        return errors, angles

    @staticmethod
    def _angular_coverage(angles: np.ndarray) -> float:
        if len(angles) < 2:
            return 0.0
        values = np.sort(np.mod(angles, 2.0 * np.pi))
        gaps = np.diff(np.concatenate((values, values[:1] + 2.0 * np.pi)))
        return float((2.0 * np.pi - np.max(gaps)) / (2.0 * np.pi))

    def _affine(self) -> np.ndarray:
        center = np.asarray(self.center_px, dtype=np.float64)
        rotate = cv2.getRotationMatrix2D((0.0, 0.0), self.angle_deg, 1.0)[:, :2]
        stretch = np.diag([1.0, self.major_px / self.minor_px])
        linear = rotate.T @ stretch @ rotate
        translation = center - linear @ center
        return np.column_stack((linear, translation))

    def _transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        matrix = self._affine()
        height, width = self.frame_shape[:2]
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
        transformed = corners @ matrix[:, :2].T + matrix[:, 2]
        minimum = transformed.min(axis=0)
        result = matrix[:, :2] @ np.asarray(point) + matrix[:, 2] - minimum
        return tuple(result.tolist())

    def rectify(self, image_or_mask: np.ndarray, is_mask: Optional[bool] = None) -> np.ndarray:
        """Remove the feeder's ellipticity without clipping the transformed frame."""
        if image_or_mask.shape[:2] != self.frame_shape[:2]:
            raise ValueError("Image dimensions must match the calibrated video frame.")
        if self.minor_px <= 0:
            raise RuntimeError("Feeder minor axis is invalid; cannot rectify frame.")
        if is_mask is None:
            is_mask = image_or_mask.ndim == 2 and set(np.unique(image_or_mask)).issubset({0, 1, 255})
        matrix = self._affine()
        height, width = image_or_mask.shape[:2]
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
        transformed = corners @ matrix[:, :2].T + matrix[:, 2]
        minimum = transformed.min(axis=0)
        maximum = transformed.max(axis=0)
        matrix[:, 2] -= minimum
        out_size = (int(math.ceil(maximum[0] - minimum[0])),
                    int(math.ceil(maximum[1] - minimum[1])))
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        result = cv2.warpAffine(image_or_mask, matrix, out_size, flags=interpolation,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return ((result > 0).astype(np.uint8) * 255) if is_mask else result

    def debug_image(self) -> np.ndarray:
        """Return the sampled frame annotated with RANSAC and enclosing circle."""
        fit = self._debug_fit
        image = fit.frame.copy()
        for x, y in fit.inliers.astype(int):
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
        center = (int(round(self.center_px[0])), int(round(self.center_px[1])))
        axes = (int(round(self.major_px / 2)), int(round(self.minor_px / 2)))
        cv2.ellipse(image, center, axes, self.angle_deg, 0, 360, (255, 0, 0), 2)
        circle_center, circle_radius = fit.enclosing_circle
        cv2.circle(image, tuple(np.rint(circle_center).astype(int)),
                   int(round(circle_radius)), (0, 0, 255), 2)
        return image
