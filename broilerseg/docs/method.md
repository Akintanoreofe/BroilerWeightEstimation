# Method

## 1. Segmentation: SAM 3 teacher

SAM 3 ("Segment Anything with Concepts") returns every instance of a concept named by a short noun phrase. `Sam3Segmenter` computes the image embedding **once per frame** and queries several phrases per class (`config.PromptConfig`):

| class   | default phrases                                  |
|---------|--------------------------------------------------|
| broiler | "chicken", "broiler chicken"                     |
| scale   | "weighing scale platform", "square platform"     |
| feeder  | "orange feeder pan", "round feeding bowl"        |

Detections from all phrases are pooled. Duplicates of the same class with a mask IoU of 0.6 or more are merged by **mask NMS**, which keeps the higher score. Masks below `score_threshold` (default 0.4) are dropped.

## 2. Which bird is being weighed?

`segmentation.select_bird`:

* **With a FlockLabel box** (validation): the broiler with the highest box IoU to the labelled box is chosen.
* **Without one** (deployment): the bird with the largest fraction of its mask over the **platform region** is chosen. The platform region is the filled convex hull of the scale and feeder masks, which restores the scale area that the bird itself hides.

The chosen mask is cleaned: a small opening, then the largest connected component is kept and holes are filled. Birds within 10 px of the frame edge are flagged (`touches_border`), because the repository pipeline drops them as probably clipped.

## 3. Pixel-to-cm calibration from the feeder

The feeder is a 14-inch (35.56 cm) circle. Every measurement uses the **convex hull** of the feeder mask, which tolerates a bird covering part of the rim.

| method              | diameter in pixels                        | note |
|---------------------|-------------------------------------------|------|
| `enclosing_circle`  | 2 x radius of the minimum enclosing circle | repository method (default) |
| `ellipse_major`     | major axis of a fitted ellipse            | robust to camera tilt |
| `equivalent_area`   | 2 sqrt(A / pi)                            | biased low under occlusion; shown for comparison |

`pixel_to_cm = 35.56 / diameter_px`. The camera is fixed, so the default `calibration_mode="video_median"` uses the **median over all frames of a video**. The repository uses frame one only. The colour-based detector from the repository (`calibrate_from_hsv`) is available as a cross-check.

## 4. Slice-based volume (repository method)

1. **Align:** fit lines to the mask contour, take the mean of the 40th-60th percentile of their angles, and rotate the mask coordinates so the body axis is horizontal (`volume.object_angle`, `volume.align_mask`).
2. **Slice:** split the *occupied columns* into `slice_count` equal groups. `None` means one per column, the repository default. Each group is one cylinder:
   * diameter `d_i` = vertical span of the mask at the group's **middle column**, in px;
   * length `w_i` = number of columns in the group, in px.
3. **Volume:** V = pi * sum_i (d_i * p / 2)^2 * w_i * p (cm^3), where p = `pixel_to_cm`.
4. **Mass:** m = V / 10^6 * rho (kg), with rho in kg/m^3 (default 900).

The cross-sections are assumed circular: the bird's depth is not measured, so its height is taken to equal its width.

## 5. Choosing density and slice count

Volume does not depend on density, so `WeightExperiment.volume_table` computes the volumes once per slice count, and every density is scored from them:

* **Implied density** per frame: `rho* = actual_kg / V_m3`. Its spread shows whether one density can fit every bird, and its trend against size shows shape bias.
* **Grid sweep** (`evaluation.sweep_grid`): event-level RMSE / MAE / bias of the median estimate vs the median scale reading, as in `tune_validation.py`, on separate **calibration** and **held-out** events.
* **Closed form** (`evaluation.best_density`): because median(rho V) = rho median(V), the least-squares density on calibration events is rho = sum(V a) / sum(V^2).

Always report held-out error. Calibration error is optimistic.

## 6. Distillation into YOLOv8-seg

Response-based distillation: the student learns to reproduce the teacher's masks.

1. Sample frames from each video (`uniform`, `random` or `motion`).
2. Label them with SAM 3 (`labeling.auto_label_videos`; resumable).
3. Convert each instance mask to polygons (Douglas-Peucker, 1 px tolerance). **Polygon fidelity** is the IoU of a mask with its rasterised polygons.
4. Split **by video** into train / val, so near-duplicate frames don't leak between them.
5. Fine-tune COCO-pretrained `yolov8n-seg.pt` (about 3.4 M parameters).
6. Evaluate the student against the teacher on validation videos: per-class mask IoU, instance-count agreement, latency. Then compare **weight accuracy** with student masks against teacher masks on FlockLabel data, because a high mask IoU doesn't guarantee good weights.
7. Export to **NCNN** for the Raspberry Pi (ONNX as a fallback).
