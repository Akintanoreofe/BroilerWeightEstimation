"""Estimate video pixel scale from the feeder visible in its first frame."""

import cv2
import numpy as np
from VideoIOClass import VideoReader

FEEDER_DIAMETER_CM = 14 * 2.54  # 14 inches = 35.56 cm


def calibrate_pixel_to_cm(video_path: str) -> float:
    """Estimate centimeters per pixel using the orange feeder in frame one.

    The feeder is detected by its HSV color range; a circle enclosing its
    convex hull supplies the pixel diameter used for calibration.

    Parameters
    ----------
    video_path : str
        Path or source accepted by :class:`VideoReader`.

    Returns
    -------
    float
        Physical centimeters represented by one pixel.

    Raises
    ------
    RuntimeError
        If the first frame cannot be read, no feeder is detected, or the
        detected feeder diameter is less than 10 pixels.
    """
    reader = VideoReader(video_path)
    ret, frame = reader.read()
    reader.release()

    if not ret:
        raise RuntimeError(f"Cannot read first frame from: {video_path}")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Tight orange HSV range for the feeder plastic
    orange_mask = ((h > 8) & (h < 20) & (s > 150) & (v > 100)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No orange feeder detected in the first frame.")

    # Convex hull of all orange fragments to handle partial occlusion by chickens
    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)
    _, radius = cv2.minEnclosingCircle(hull)
    diameter_pixels = radius * 2

    if diameter_pixels < 10:
        raise RuntimeError(f"Detected feeder too small ({diameter_pixels:.1f}px). Check video.")

    pixel_to_cm = FEEDER_DIAMETER_CM / diameter_pixels
    print(f"[FeederCalibration] Feeder diameter: {diameter_pixels:.1f}px -> pixel_to_cm={pixel_to_cm:.4f}")
    return pixel_to_cm
