# Changelog

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
