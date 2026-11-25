# Changelog

## 0.6.2 (2026-11-25)

* `patch_nms_time_limit(cpu_on_mps=True)`: when training on the Apple GPU, Ultralytics'
  validation NMS runs on the CPU (results go back to the GPU). On MPS it slowed down as the
  model improved, hit even the raised 18 s limit, dropped detections (recall fell from 0.85
  to 0.22) and made validation take up to 40 minutes. Applied by `train_student` and
  `evaluate_student`.

## 0.6.1 (2026-11-25)

* Memory: saved labels are no longer all loaded at once. Full-resolution pen-crop masks are
  about 4 MB each (~85 MB per frame), so holding every frame needed tens of GB and got the
  kernel killed (and caused swapping / NMS time-outs). New `LazyResults`,
  `load_frames_lazily` and `count_labels` read one frame at a time;
  `teacher_student_table(..., keep=())` no longer keeps every student result.
* Notebook 2 uses them in the labelling cell and in section 7.

## 0.6.0 (2026-11-24)

* **The weighing scale is no longer segmented.** Classes are now `broiler` and `feeder`
  (`CLASS_NAMES`); SAM 3 labelled the scale unreliably. YOLO learns two classes.
* Bird choice (`select_bird`): with a FlockLabel box (evaluation), the broiler whose box
  overlaps it most; without one, the most confident bird fully in view. New `select_birds`
  returns every bird fully in view (flock estimate on the Pi). `platform_region` and
  `on_scale_fraction` are removed.
* The feeder (pixel-to-cm calibration) is one object: `merge_single` merges overlapping
  feeder pieces (outer pan, inner ring ...) into one mask and drops stray detections. It is
  applied by SAM 3, Grounded SAM 2, the YOLO student and `load_result`
  (`PromptConfig.single_instance`).
* Label files from earlier versions still load: their scale masks are ignored and feeder
  pieces merged, so existing labels can be reused without running the teacher again.
* Notebook 2 (NMS / memory): the teacher is freed before training and re-loaded after,
  `BATCH = 8`, `CACHE = "ram"`, `EPOCHS = 60`, `PATIENCE = 15`; the NMS time-limit fix
  (0.5.3) stays in `train_student`. New dataset folder `yolo_dataset_2class`.

## 0.5.4 (2026-11-24)

* `MixedTeacher` / `load_mixed_teacher`: a different teacher per class, e.g. SAM 3 for
  broiler and feeder, Grounded SAM 2 for the scale. Each model runs once per image on its
  own classes; stage 1 (the pen) uses the `default` class's teacher.
* Notebooks 1 and 2: `TEACHER = "mixed"` with `MIXED_TEACHER = {...}`. Notebook 2:
  `COMPARE_TEACHERS = True` shows SAM 3 vs Grounded SAM 2 on the same pen crops for the
  scale and the feeder, with a count / score table.

## 0.5.3 (2026-11-23)

* `patch_nms_time_limit` (applied by `train_student` and `evaluate_student`): Ultralytics'
  NMS gets 1 s per image instead of 0.05 s, so validation on crowded pen crops is no longer
  cut short ("NMS time limit exceeded"), which made recall, mAP, early stopping and the
  choice of `best.pt` unreliable.

## 0.5.2 (2026-11-23)

* `train_student` finds the starting weights locally (`~/broilerseg_models/base/`) and, if
  missing, downloads them there once with resumable retries (`resolve_base_model`), so an
  unstable network no longer stops training ("Curl return value 56").
* `train_student` applies the Apple-GPU workaround `patch_mps_assigner` automatically when
  training on `mps` (the "shape mismatch ... cannot be broadcast" crash).

## 0.5.1 (2026-11-23)

* `read_frame` decodes from the start when jumping to a frame fails (damaged or recovered
  H.265 files, wrong frame counts); new `safe_read_frame` returns None instead of failing.
* Stage 1 (`find_pen_rois`) skips unreadable frames; a video with none gets its folder's
  median pen region instead of stopping the run.

## 0.5.0 (2026-11-23)

* **Two-stage segmentation** (new `broilerseg.roi`). Stage 1 finds the pen with a text prompt
  (`PEN_PROMPTS`, same teacher model) and crops it at full resolution. Stage 2 segments
  broiler / scale / feeder inside the crop only. `PenROI` (box + outline, stored as frame
  fractions), `PenDetector`, `find_pen_rois` (per video or per folder, resumable, manual
  overrides with `apply_manual_rois`), `CroppedSegmenter` (any segmenter on the crop,
  full-frame masks out).
* `auto_label_videos(rois=..., mask_outside=...)` saves pen crops and records their position
  in the manifest. It refuses to mix cropped and uncropped labels in one folder.
* `load_student` applies the pen region saved in the model card, so Notebook 3 and the
  Raspberry Pi crop exactly as in training (`pen_crop=False` to switch off).
* SAM 3 is now the default teacher (`load_teacher()`); Grounded SAM 2 is still available.
* Notebook 2: `ONLY_RESOLUTION` (e.g. 3840 x 2160 only), new stage-1 section (candidates,
  pixel grid, regions, crops, one- vs two-stage comparison), labels in `labels_pen/`, and the
  pen region is saved with the model. Notebooks 1 and 3: optional / automatic pen crop.
* New plots: `plot_pen_candidates`, `plot_pen_rois`, `plot_pen_crops`, `plot_roi_summary`,
  `plot_two_stage`.
* `evaluate_student` writes its plots next to the training run (not the working folder).
* FFmpeg HEVC header warnings are silenced.

## 0.4.1 (2026-11-22)

* Plots that grew with the data are capped (sampling plan, folder summary, video inventory,
  calibration stability, per-event, nested CV), so thousands of videos no longer exceed
  matplotlib's 65,536-pixel image limit.
* Notebook 2: `MAX_VIDEOS` (spread evenly over all folders) and a labelling-time estimate.

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
