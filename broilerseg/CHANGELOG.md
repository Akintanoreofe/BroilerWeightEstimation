# Changelog

## 0.4.0 (2026-11-22)

* Three notebooks: 01 teacher + fitted weight, 02 distillation (label, train, evaluate, SAVE),
  03 saved YOLO + fitted weight (+ comparison with 01, videos, Raspberry Pi files).
* Nested video folders: `scan_videos` records `folder`, `rel_path` and a unique `video_id`
  (`day1__cam2__ch01`), so equal file names in different folders never collide.
* `build_yolo_dataset(split_by="folder")` holds out whole sub-folders.
* `distill.evaluate_student`, `save_student`, `load_student` (model folder + `model_card.json`).
* New plots: `plot_folder_summary`, `plot_student_metrics`. Fit plot zooms to the data and uses
  linear axes when weights are similar.

## 0.3.1 (2026-11-21)

* Notebook 1 rebuilt around the fitted equation: measure every bird -> fit -> every weight,
  plot, storyboard and video comes from the fitted equation (out-of-fold for accuracy).
  Optional install cell (branch + folder) at the top.
* `MassModel.estimate()` (VolumeEstimate with fitted settings, for plots) and
  `massmodel.fitted_records()` (records recomputed with the fitted equation).

## 0.3.0 (2026-11-21)

* New `massmodel` module: one mass equation `m = a * V(offset, trim, slices, calibration)^b`.
  `a, b` in closed form (log-space weighted least squares, ridge prior b -> 1, b fixed when
  event weights barely vary); offset / trim / slice count / calibration method and scope by
  grid search; leave-one-event-out CV plus nested CV for an unbiased error.
* `estimate_density`: most accurate single density with leave-one-event-out range and
  bootstrap 95% interval.
* New plots in `viz.mass_model`; Notebook 1 sections 8-9.
* `docs/methodology.pdf` (LaTeX source in `docs/methodology/`).

## 0.2.1 (2026-11-20)

* `viz.set_weight_unit("g" | "kg")` shows weights in grams or kilograms in every plot and table
  (`WEIGHT_UNIT` in the notebooks). Calculations, saved CSVs and density stay in kg.
* `viz.weight_table`, `viz.fmt_weight`, `viz.to_display` helpers.

## 0.2.0 (2026-11-20)

* New teacher `GroundedSam2Segmenter`: Grounding DINO (text -> boxes) + SAM 2.1 (boxes -> masks).
  Public Apache-2.0 models, no access request. Default teacher in both notebooks.
* `segmentation.load_teacher("grounded_sam2" | "sam3")` switches teacher with one setting
  (`TEACHER` in the notebooks).
* Notebooks write outputs to `~/broilerseg_outputs/` by default (works when Jupyter starts in `/`).
* `viz.plot_pred_vs_actual` accepts `xlabel` / `ylabel`.

## 0.1.0 (2026-11-20)

First experimental release.

* SAM 3 teacher via Hugging Face `transformers` (CUDA / Apple MPS / CPU), with multi-prompt pooling and mask NMS for broiler, scale and feeder.
* Feeder-based pixel-to-cm calibration (enclosing circle, ellipse major axis, equivalent area) plus the repository's HSV fallback, and a per-video median.
* Slice-based volume model reproducing BroilerWeightEstimation (`object_angle`, coordinate alignment, `mask_slice_profile`, cylinder sum).
* FlockLabel evaluation and dataset export loaders.
* `WeightExperiment` pipeline, metrics, event tables, density x slice sweeps, closed-form density.
* Resumable SAM 3 auto-labelling of video folders; YOLO-seg dataset builder split by video.
* YOLOv8-seg student training, teacher-student comparison, benchmarking, NCNN/ONNX export.
* `viz` module with 40+ plots; two notebooks; synthetic demo data; CLI.
