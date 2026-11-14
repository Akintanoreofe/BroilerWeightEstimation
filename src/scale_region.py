"""Validate a manually annotated weighing-platform polygon."""

import cv2
import numpy as np


def scale_mask_for_frame(polygon, shape):
    points = np.asarray(polygon)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.issubdtype(points.dtype, np.number)
            or not np.isfinite(points).all()
            or not np.equal(points, np.floor(points)).all()):
        raise ValueError("scale polygon must contain at least three integer [x, y] points")
    if ((points[:, 0] < 0).any() or (points[:, 0] >= shape[1]).any()
            or (points[:, 1] < 0).any() or (points[:, 1] >= shape[0]).any()):
        raise ValueError("scale polygon points must lie inside the frame")
    points = points.astype(np.int32)
    if cv2.contourArea(points) <= 0:
        raise ValueError("scale polygon must have nonzero area")
    result = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(result, [points], 255)
    return result
