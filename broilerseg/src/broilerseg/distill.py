"""Knowledge distillation: train a YOLOv8-seg student on SAM 3 pseudo-labels.

This is response-based distillation: the teacher's masks are the student's
training targets. The student (``yolov8n-seg``, ~3.4 M parameters) is small
enough for a Raspberry Pi once exported to NCNN, which is the format
Ultralytics recommends for ARM CPUs.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from broilerseg.config import CLASS_NAMES
from broilerseg.segmentation.base import SegmentationResult, Segmenter
from broilerseg.segmentation.postprocess import mask_iou


def pick_train_device() -> str:
    """Ultralytics device string: ``"0"`` (CUDA), ``"mps"`` or ``"cpu"``."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_student(data_yaml: str | Path, base_model: str = "yolov8n-seg.pt", epochs: int = 100,
                  imgsz: int = 640, batch: int = 16, device: str | None = None,
                  project: str | Path = "runs/broilerseg", name: str = "student",
                  patience: int = 30, **overrides) -> Path:
    """Fine-tune a YOLOv8 segmentation model on the SAM 3 dataset.

    Parameters
    ----------
    data_yaml : path-like
        ``data.yaml`` from :func:`broilerseg.yolo_dataset.build_yolo_dataset`.
    base_model : str
        COCO-pretrained start point: ``yolov8n-seg.pt`` (Pi), ``yolov8s-seg.pt``.
    epochs, imgsz, batch, patience : int
        Usual Ultralytics settings; ``patience`` stops early when validation
        stops improving.
    device : str, optional
        Defaults to :func:`pick_train_device`.
    project, name : str
        Output folder is ``project/name``.
    **overrides
        Any other ``model.train`` argument (e.g. ``workers=0``, ``mosaic=0.5``).

    Returns
    -------
    Path
        The run folder (``weights/best.pt``, ``results.csv``, plots).
    """
    from ultralytics import YOLO

    model = YOLO(base_model)
    args = {"data": str(data_yaml), "epochs": epochs, "imgsz": imgsz, "batch": batch,
            "device": device or pick_train_device(), "project": str(project), "name": name,
            "patience": patience, "exist_ok": True, "plots": True}
    args.update(overrides)
    model.train(**args)
    return Path(model.trainer.save_dir)


def load_training_curves(run_dir: str | Path) -> pd.DataFrame:
    """Read ``results.csv`` of a training run (column names stripped)."""
    df = pd.read_csv(Path(run_dir) / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    return df


def compare_results(teacher: SegmentationResult, student: SegmentationResult) -> list[dict]:
    """Per-class IoU between teacher and student union masks for one image."""
    rows = []
    for c in CLASS_NAMES:
        t, s = teacher.union(c), student.union(c)
        rows.append({"class_name": c, "iou": mask_iou(t, s) if (t.any() or s.any()) else np.nan,
                     "teacher_n": len(teacher.of_class(c)), "student_n": len(student.of_class(c)),
                     "teacher_area": int(t.sum()), "student_area": int(s.sum())})
    return rows


def teacher_student_table(
    images: dict[str, np.ndarray], teacher_results: dict[str, SegmentationResult],
    student: Segmenter,
) -> tuple[pd.DataFrame, dict[str, SegmentationResult]]:
    """Run the student on images the teacher labelled and measure agreement.

    Returns a long table (one row per image and class) and the student results.
    """
    rows, student_results = [], {}
    for stem, img in images.items():
        res = student.segment(img)
        student_results[stem] = res
        for r in compare_results(teacher_results[stem], res):
            rows.append({"stem": stem, "student_seconds": res.seconds, **r})
    return pd.DataFrame(rows), student_results


def benchmark(segmenter: Segmenter, images, warmup: int = 2, repeats: int = 1) -> dict:
    """Mean / std latency (ms) and throughput of a segmenter on some images."""
    images = list(images)
    for img in images[:warmup]:
        segmenter.segment(img)
    times = []
    for _ in range(repeats):
        for img in images:
            t0 = time.perf_counter()
            segmenter.segment(img)
            times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"name": segmenter.name, "ms_mean": float(t.mean()), "ms_std": float(t.std()),
            "fps": float(1000 / t.mean()), "n": len(t)}


def export_for_raspberry_pi(weights: str | Path, imgsz: int = 640,
                            formats=("ncnn", "onnx"), half: bool = False) -> dict[str, Path]:
    """Export trained weights for edge deployment.

    NCNN is the fastest Ultralytics format on Raspberry Pi CPUs; ONNX is kept
    as a portable fallback. Copy the exported folder / file to the Pi and load
    it with ``YOLO("best_ncnn_model", task="segment")``.
    """
    from ultralytics import YOLO

    out = {}
    for fmt in formats:
        model = YOLO(str(weights))
        out[fmt] = Path(model.export(format=fmt, imgsz=imgsz, half=half))
    return out


def path_size_mb(path: str | Path) -> float:
    """Size of a file or folder in MB."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
