"""Filter segmented masks and score candidate broiler objects."""

from dataclasses import dataclass
import math
from typing import List, Tuple
from abc import ABC, abstractmethod
import cv2
import numpy as np


@dataclass
class ObjectInfo:
    """Mask, confidence values, crop coordinates, and estimates for an object.

    Attributes
    ----------
    mask : numpy.ndarray
        Segmented mask for the object.
    confidence_dt : float
        Confidence derived from the distance transform.
    confidence_tpl : float
        Confidence from comparison with the template mask.
    crop_region : tuple of int
        Object bounding rectangle as ``(x, y, width, height)``.
    volume : float, default=0.0
        Estimated object mass or volume, as assigned by a calculator.
    label : str, default="broiler"
        Object category.
    area : float, default=0.0
        Contour area in pixels.
    """

    mask: np.ndarray
    confidence_dt: float
    confidence_tpl: float
    crop_region: Tuple[int, int, int, int]  # (x, y, w, h)
    volume: float = 0.0
    label: str = "broiler"  # default label
    area: float = 0.0


class ObjectFilter(ABC):
    """Abstract interface for selecting objects from a segmentation mask."""

    @abstractmethod
    def should_process(self, mask: np.ndarray) -> Tuple[List['ObjectInfo'], np.ndarray]:
        """Filter a segmentation mask and select candidate objects.

        Parameters
        ----------
        mask : numpy.ndarray
            Input binary segmentation mask.

        Returns
        -------
        list of ObjectInfo
            Objects selected for further processing.
        numpy.ndarray
            Mask containing the contours retained by the filter.
        """
        pass


class SimpleObjectFilter(ObjectFilter):
    """Filter contours and score single-object masks against a template."""

    def __init__(self, target_labels: List[str], template_mask: str,
                 min_area: float, min_score: float = 0.0, top_k: int = 3, padding: int = 5,
                 peak_thresh: float = 0.6, alpha: float = 0.4, beta: float = 0.6,
                 too_small_thresh: int = 3, img_border: int = 1):
        """Initialize the contour filter and load a grayscale template.

        Parameters
        ----------
        target_labels : list of str
            Labels retained as filter configuration; contour selection does
            not currently inspect labels.
        template_mask : str
            Path to the grayscale template mask.
        min_area : float
            Minimum contour area in pixels.
        min_score : float, default=0.0
            Minimum combined score for a selected object.
        top_k : int, default=3
            Maximum number of candidates retained.
        padding : int, default=5
            Padding added when constructing square crop regions.
        peak_thresh : float, default=0.6
            Distance-transform threshold used to count peaks.
        alpha : float, default=0.4
            Weight applied to the normalized distance-transform score.
        beta : float, default=0.6
            Weight applied to the template score.
        too_small_thresh : int, default=3
            Divisor used to reject contours smaller than the template
            dimensions divided by this value.
        img_border : int, default=1
            Border margin at which contours are rejected.
        """

        self.target_labels = set(target_labels)
        self.min_area = min_area
        self.min_score = min_score
        self.peak_thresh = peak_thresh
        self.alpha = alpha
        self.beta = beta
        self.top_k = top_k
        self.padding = padding
        self.too_small_thresh = too_small_thresh
        self.img_border = img_border

        self.template_mask = cv2.imread(template_mask, cv2.IMREAD_GRAYSCALE)
        self.template_mask = self.template_mask.astype("uint8")
        self.template_mask = (self.template_mask > 0).astype("uint8") * 255
        self.th, self.tw = self.template_mask.shape[:2]
        self.template_vector = self.template_mask.astype("float32").reshape(-1)

    def should_process(self, mask: np.ndarray) -> Tuple[List[ObjectInfo], np.ndarray]:
        """Clean contours and return the highest-scoring single-object masks.

        Parameters
        ----------
        mask : numpy.ndarray
            Input segmentation mask.

        Returns
        -------
        list of ObjectInfo
            Up to ``top_k`` candidates above ``min_score``.
        numpy.ndarray
            Mask containing contours that pass the size and border filters.
        """

        clean_mask = np.zeros_like(mask)
        single_chicken_masks = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if self.should_filter_contour(c, area, mask.shape):
                continue

            cv2.drawContours(clean_mask, [c], -1, 255, -1)

            blob_mask = np.zeros_like(mask)
            cv2.drawContours(blob_mask, [c], -1, 255, -1)
            dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
            dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
            _, peaks = cv2.threshold(dist_norm, self.peak_thresh, 1.0, cv2.THRESH_BINARY)
            peaks = (peaks * 255).astype("uint8")
            peak_cnts, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_peaks = len(peak_cnts)

            # Multiple distance-transform peaks usually indicate touching birds, so reject the blob.
            if num_peaks == 1:
                obj_info = self.score_blob(blob_mask, c, dist, area)
                if obj_info:
                    single_chicken_masks.append(obj_info)

        if not single_chicken_masks:
            return [], clean_mask

        final_conf = self.combine_scores(single_chicken_masks)
        selected_objs = self.select_topk(final_conf, single_chicken_masks)

        return selected_objs, clean_mask
    
    def should_filter_contour(self, cnt: np.ndarray, area: float, imgShape: np.ndarray) -> bool:
        """Determine whether a contour is too small or touches the image edge.

        Parameters
        ----------
        cnt : numpy.ndarray
            Contour to inspect.
        area : float
            Contour area in pixels.
        imgShape : numpy.ndarray
            Image shape, whose first two values are height and width.

        Returns
        -------
        bool
            ``True`` if the contour should be discarded.
        """
        x, y, w, h = cv2.boundingRect(cnt)
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

        too_small_h = h < (self.th // self.too_small_thresh)
        too_small_w = w < (self.tw // self.too_small_thresh)

        touches_border = (leftmost[0] <= self.img_border or
                          rightmost[0] >= imgShape[1] - self.img_border or
                          topmost[1] <= self.img_border or
                          bottommost[1] >= imgShape[0] - self.img_border)

        if area < self.min_area or too_small_h or too_small_w or touches_border:
            return True

        return False    
    
    def score_blob(self, blob_mask: np.ndarray, contour: np.array, dist: np.ndarray,
                   area: float) -> ObjectInfo:
        """Build object metadata and confidence scores for a contour.

        Parameters
        ----------
        blob_mask : numpy.ndarray
            Filled mask for the contour.
        contour : numpy.ndarray
            Contour whose bounds and area are recorded.
        dist : numpy.ndarray
            Distance transform used for the shape confidence.
        area : float
            Contour area in pixels.

        Returns
        -------
        ObjectInfo or None
            Scored object, or ``None`` if mask alignment finds no contour.
        """

        confidence_dt = float(dist.max()) / np.sqrt(area + 1e-6)  # avoid division by zero

        x, y, w, h = cv2.boundingRect(contour)
        crop_mask = self.align_mask_by_affine(blob_mask)
        if crop_mask is None:
            return crop_mask

        crop_flip = cv2.flip(crop_mask, 1)
        conf_norm = self.cosine_scores(crop_mask, self.template_vector)
        conf_flip = self.cosine_scores(crop_flip, self.template_vector)
        confidence_tpl  = max(conf_norm, conf_flip)

        return ObjectInfo(
            mask=crop_mask,
            confidence_dt=confidence_dt,
            confidence_tpl=confidence_tpl,
            crop_region=(x, y, w, h),
            area=cv2.contourArea(contour)
        )

    def combine_scores(self, objs_info: List[ObjectInfo]) -> np.ndarray:
        """Combine normalized distance-transform and template scores.

        Parameters
        ----------
        objs_info : list of ObjectInfo
            Objects to score.

        Returns
        -------
        numpy.ndarray
            Weighted combined score for each object.
        """

        dt_arr = [obj_info.confidence_dt for obj_info in objs_info]
        tpl_arr = [obj_info.confidence_tpl for obj_info in objs_info]
        dt_arr = np.array(dt_arr)
        tpl_arr = np.array(tpl_arr)

        # Normalize distance-transform scores across candidates; template similarity is already in [0, 1].
        dt_min, dt_max = dt_arr.min(), dt_arr.max()
        dt_norm = (dt_arr - dt_min) / (dt_max - dt_min + 1e-6)
        tpl_norm = np.clip(tpl_arr, 0.0, 1.0)
        final_conf = self.alpha * dt_norm + self.beta * tpl_norm

        return final_conf

    def get_square_bbox(self, x: int, y: int, w: int, h: int,
                        H: int, W: int) -> Tuple[int, int, int, int]:
        """Expand a bounding box to a padded square within image bounds.

        Parameters
        ----------
        x, y : int
            Top-left corner of the original box.
        w, h : int
            Width and height of the original box.
        H, W : int
            Height and width of the image.

        Returns
        -------
        tuple of int
            Square box as ``(x, y, side, side)``, with its origin clamped
            against the image dimensions.
        """

        center_x = x + w // 2
        center_y = y + h // 2
        side_length = max(w, h) + 2 * self.padding

        x_new = center_x - side_length // 2
        y_new = center_y - side_length // 2

        x_new = max(0, min(x_new, W - side_length))
        y_new = max(0, min(y_new, H - side_length))

        return int(x_new), int(y_new), int(side_length), int(side_length)

    def align_mask_by_affine(self, crop: np.array) -> np.ndarray:
        """Rotate a crop to its estimated object angle and extract its box.

        Parameters
        ----------
        crop : numpy.ndarray
            Binary object mask to align.

        Returns
        -------
        numpy.ndarray or None
            Aligned square crop, or ``None`` when rotation yields no contour.
        """

        H, W = crop.shape[:2]
        mean_ang = self.object_angle(crop)

        M = cv2.getRotationMatrix2D((crop.shape[1]//2, crop.shape[0]//2), mean_ang, 1)
        crop = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]), cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        x, y, w, h = self.get_square_bbox(x, y, w, h, H, W)
        cropped = crop[y:y+h, x:x+w]

        return cropped

    def align_mask_by_coords(self, mask: np.ndarray) -> np.ndarray:
        """Rotate foreground coordinates and return a cropped binary mask.

        Parameters
        ----------
        mask : numpy.ndarray
            Binary mask whose foreground coordinates are aligned.

        Returns
        -------
        numpy.ndarray
            Rotated, filled mask cropped to a square bounding box. An empty
            input produces an empty mask with the input shape.
        """

        H, W = mask.shape[:2]
        mean_ang = self.object_angle(mask)

        mask_bool = mask > 0

        if not mask_bool.any():
            return np.zeros_like(mask, dtype=mask.dtype)

        ys, xs = np.where(mask_bool)

        cx = xs.mean()
        cy = ys.mean()

        theta = np.deg2rad(-mean_ang)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        x_shift = xs - cx
        y_shift = ys - cy

        x_rot = cos_t * x_shift - sin_t * y_shift
        y_rot = sin_t * x_shift + cos_t * y_shift

        x_rot_min = x_rot.min()
        y_rot_min = y_rot.min()

        x_rot_shifted = x_rot - x_rot_min
        y_rot_shifted = y_rot - y_rot_min

        x_new = np.rint(x_rot_shifted).astype(int)
        y_new = np.rint(y_rot_shifted).astype(int)

        new_w = x_new.max() + 1
        new_h = y_new.max() + 1

        rotated_mask = np.zeros((new_h, new_w), dtype=mask.dtype)
        fg_value = mask.max()
        rotated_mask[y_new, x_new] = fg_value

        contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(rotated_mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

        rotated_bool = filled > 0
        if not rotated_bool.any():
            return filled

        ys2, xs2 = np.where(rotated_bool)
        y_min, y_max = ys2.min(), ys2.max()
        x_min, x_max = xs2.min(), xs2.max()

        x_min, y_min, w, h = self.get_square_bbox(x_min, y_min, x_max-x_min, y_max-y_min, H, W)
        rotated_cropped = filled[y_min:y_min+h, x_min:x_min+w]

        return rotated_cropped

    def object_angle(self, mask: np.ndarray) -> float:
        """Estimate the dominant contour orientation in degrees.

        Parameters
        ----------
        mask : numpy.ndarray
            Binary mask to analyze.

        Returns
        -------
        float
            Mean of the central contour-angle range, or ``0.0`` when no
            usable contours are found.
        """

        threshed = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)[1]
        threshed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, np.ones((1, 10)))
        contours = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

        if not contours:
            return 0.0

        angles = []

        for c in contours:
            if len(c) < 2:
                continue
            line = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = line[0].item(), line[1].item()
            ang = (180/np.pi)*math.atan2(vy, vx)
            angles.append(ang)

        if not angles:
            return 0.0

        angles = np.array(angles)
        lo_val, up_val = np.percentile(angles, (40, 60))
        filtered = angles[(angles >= lo_val) & (angles <= up_val)]

        if len(filtered) == 0:
            return float(np.mean(angles))

        return float(np.mean(filtered))

    def cosine_scores(self, crop_mask: np.ndarray, template_vector: np.ndarray) -> float:
        """Compute cosine similarity between a resized mask and template.

        Parameters
        ----------
        crop_mask : numpy.ndarray
            Candidate mask, resized to the template dimensions.
        template_vector : numpy.ndarray
            Flattened template mask.

        Returns
        -------
        float
            Cosine similarity of the flattened masks.
        """

        crop_mask = cv2.resize(crop_mask, (self.tw, self.th), interpolation=cv2.INTER_NEAREST)

        crop_vector = crop_mask.astype("float32").reshape(-1)
        dot = np.dot(crop_vector, template_vector)
        norms = (np.linalg.norm(crop_vector) * np.linalg.norm(template_vector) + 1e-6)
        score = float(dot / norms)

        return score

    def select_topk(self, scores: np.ndarray, chickens_info: List[ObjectInfo]) -> List[ObjectInfo]:
        """Select up to ``top_k`` objects whose scores meet the threshold.

        Parameters
        ----------
        scores : numpy.ndarray
            Combined score for each candidate.
        chickens_info : list of ObjectInfo
            Candidate objects in the same order as ``scores``.

        Returns
        -------
        list of ObjectInfo
            Candidates ranked highest-first and meeting ``min_score``.
        """

        idxs = np.argsort(-scores)[:self.top_k]
        selected_chickens = [chickens_info[i] for i in idxs if scores[i] >= self.min_score]

        return selected_chickens