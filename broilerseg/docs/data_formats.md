# Data formats

## Inputs

### Folder of raw videos (Notebook 2, mask creation)

Any folder. It is scanned **recursively** for `.mp4 .avi .mov .mkv .m4v .wmv .mpg .mpeg`. Files whose names start with `.` are ignored.

### FlockLabel exports (weight validation)

`io.load_flocklabel_export(path)` accepts the export folder or a folder up to two levels above it, and detects which kind it is.

**Evaluation export** (`export_evaluation` in FlockLabel):

```
test_videos/<v00008_20251001_165951>.mp4     trimmed (optionally cropped) clip per weighing
test_videos/metadata.csv                     video_name, weight_kg (median reading)
ground_truth/gt_boxes.csv                    video_name, frame_idx (1-based in the clip), x, y, w, h,
                                             weight_kg, event_id, bird_id, flags
ground_truth/events.csv                      event_id, video_name, weight_kg, min/max_weight_kg,
                                             n_frames, start_local, pixel_to_cm, source_video, ...
export_info.json                             "kind": "evaluation", crop, warnings
```

**Dataset export** (`export_dataset` in FlockLabel):

```
images/{train,val,test}/*.jpg, labels/... (YOLO boxes), annotations_coco.json
manifest.csv    image_path, split, video, timestamp_local, frame_idx, x1, y1, x2, y2,
                img_w, img_h, weight_g, video_median_weight_g
```

Both become one `samples` table (one row per labelled frame):

| column | meaning |
|---|---|
| `sample_id` | row id |
| `kind` | `video` (read a frame from a clip) or `image` |
| `file` | clip or image path |
| `frame_index0` | zero-based frame to read (`frame_idx - 1 + frame_offset`) |
| `video_name`, `frame_label`, `event_id` | identifiers |
| `x1, y1, x2, y2` | FlockLabel bird box (pixels) |
| `weight_kg` | scale reading on that frame |
| `event_weight_kg` | median reading of the event |
| `pixel_to_cm_hint` | from `events.csv`, if FlockLabel stored one |

`frame_offset` has the same meaning as `--gt-frame-offset` in the repository's `evaluate_against_gt.py`.

## Outputs

### Label folder (`labeling.auto_label_videos`)

```
frames/<video-stem>_f000123.jpg
masks/<video-stem>_f000123.npz      masks (N,H,W) bool, class_ids, scores, boxes (xyxy),
                                    prompts, image_shape, seconds
overlays/<video-stem>_f000123.jpg   quick-look
manifest.csv                        stem, image, masks, video, video_stem, frame_idx, height,
                                    width, seconds, n_<class>, score_<class>, area_<class>
```

Load a result back with `labeling.load_result(path)`.

### YOLO-seg dataset (`yolo_dataset.build_yolo_dataset`)

```
images/{train,val[,test]}/<stem>.jpg
labels/{train,val[,test]}/<stem>.txt    "<class> x1 y1 x2 y2 ..." normalised polygons
data.yaml                               names: 0 broiler, 1 scale, 2 feeder
dataset_summary.csv                     split, polygons per class, polygon fidelity per class
```

### Weight experiment records (`experiments.WeightExperiment.run`)

One row per sample: `status` (`ok`, `no_broiler`, `no_match_to_gt_box`, `no_bird_on_scale`, `no_feeder`), counts per class, `bird_score`, `box_iou`, `on_scale`, `feeder_diameter_px`, `pixel_to_cm_frame`, `pixel_to_cm` (used), `volume_cm3`, `pred_kg`, `actual_kg`, `error_kg`, `pct_error`, `implied_density`, `length_cm`, `max_width_cm`, `area_cm2`, `angle_deg`, `touches_border`, `seg_seconds`.
