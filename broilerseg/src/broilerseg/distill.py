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


# ----------------------------------------------------------------- evaluate / save / load
def evaluate_student(weights: str | Path, data_yaml: str | Path, split: str = "val",
                     imgsz: int = 640, device: str | None = None,
                     project: str | Path | None = None) -> tuple[dict, pd.DataFrame]:
    """Ultralytics validation of the student on a dataset split.

    Returns overall box / mask metrics and a per-class table (mask and box
    mAP50-95). The validation plots go to ``project/<split>``; by default
    ``project`` is an ``evaluation`` folder next to the training run
    (never the current working directory, which may be read-only).
    """
    from ultralytics import YOLO

    if project is None:
        project = Path(weights).expanduser().resolve().parent.parent / "evaluation"
    model = YOLO(str(weights))
    r = model.val(data=str(data_yaml), split=split, imgsz=imgsz, plots=True,
                  device=device or pick_train_device(), verbose=False,
                  project=str(project), name=split, exist_ok=True)
    overall = {"split": split,
               "mask_mAP50": float(r.seg.map50), "mask_mAP50-95": float(r.seg.map),
               "mask_precision": float(r.seg.mp), "mask_recall": float(r.seg.mr),
               "box_mAP50": float(r.box.map50), "box_mAP50-95": float(r.box.map),
               "box_precision": float(r.box.mp), "box_recall": float(r.box.mr)}
    names = r.names if isinstance(r.names, dict) else dict(enumerate(r.names))
    per_class = pd.DataFrame({"class_name": [names[i] for i in range(len(r.seg.maps))],
                              "mask_mAP50-95": [float(v) for v in r.seg.maps],
                              "box_mAP50-95": [float(v) for v in r.box.maps]})
    return overall, per_class


def save_student(run_dir: str | Path, out_dir: str | Path, name: str = "broiler_student",
                 exports: dict | None = None, card: dict | None = None,
                 weights: str | Path | None = None) -> Path:
    """Copy the trained student and everything needed to use it into one folder.

    Layout::

        <out_dir>/<name>/
            best.pt                 trained weights (if a training run exists)
            best_ncnn_model/        Raspberry Pi export (if exported)
            best.onnx               portable export (if exported)
            data.yaml, results.csv  dataset definition and training curves
            training/*.png          Ultralytics training plots
            model_card.json         classes, image size, teacher, metrics, dates

    Notebook 3 loads this folder with :func:`load_student`.
    """
    import json
    import shutil
    from datetime import datetime

    dest = Path(out_dir).expanduser() / name
    dest.mkdir(parents=True, exist_ok=True)
    run = Path(run_dir) if run_dir else None
    src_w = Path(weights) if weights else (run / "weights" / "best.pt" if run else None)
    if src_w is not None and src_w.is_file():
        shutil.copy2(src_w, dest / "best.pt")
    if run is not None and run.is_dir():
        for f in ("results.csv", "args.yaml"):
            if (run / f).is_file():
                shutil.copy2(run / f, dest / f)
        (dest / "training").mkdir(exist_ok=True)
        for f in list(run.glob("*.png")) + list(run.glob("*.jpg")):
            shutil.copy2(f, dest / "training" / f.name)
    for path in (exports or {}).values():
        path = Path(path)
        if path.is_dir():
            shutil.copytree(path, dest / path.name, dirs_exist_ok=True)
        elif path.is_file():
            shutil.copy2(path, dest / path.name)
    info = {"name": name, "saved": datetime.now().isoformat(timespec="seconds"),
            "classes": list(CLASS_NAMES), "weights": "best.pt" if (dest / "best.pt").is_file()
            else None, "exports": {k: Path(v).name for k, v in (exports or {}).items()}}
    info.update(card or {})
    (dest / "model_card.json").write_text(json.dumps(info, indent=2, default=str))
    return dest


def load_student(model_dir: str | Path, fmt: str = "pt", conf: float = 0.25,
                 device: str | None = None, pen_crop: bool | None = None):
    """Load a model saved by :func:`save_student`.

    Parameters
    ----------
    model_dir : path-like
        The saved folder, or a ``.pt`` file directly.
    fmt : {"pt", "ncnn", "onnx"}
        Which copy to run (``ncnn`` / ``onnx`` behave as on the Raspberry Pi).
    conf : float
        Detection confidence threshold.
    device : str, optional
        Torch device; Ultralytics picks one if omitted.
    pen_crop : bool, optional
        If the model was trained on pen crops (two-stage labelling), wrap it in
        :class:`broilerseg.roi.CroppedSegmenter` with the pen region saved in
        the model card, so it gets the same view it was trained on and returns
        full-frame masks. ``None`` follows the card; ``False`` uses whole frames.

    Returns
    -------
    (YoloSegmenter or CroppedSegmenter, dict)
        The segmenter and the model card.
    """
    import json

    from broilerseg.segmentation.yolo import YoloSegmenter

    p = Path(model_dir).expanduser()
    if p.is_file():
        return YoloSegmenter(p, conf=conf, device=device), {"weights": p.name}
    card_path = p / "model_card.json"
    card = json.loads(card_path.read_text()) if card_path.is_file() else {}
    imgsz = int(card.get("imgsz", 640))
    exported = card.get("exports", {})
    target = {"pt": p / "best.pt", "ncnn": p / exported.get("ncnn", "best_ncnn_model"),
              "onnx": p / exported.get("onnx", "best.onnx")}[fmt]
    if not target.exists():
        raise FileNotFoundError(f"{target} not found in {p}; available: "
                                f"{sorted(x.name for x in p.iterdir())}")
    seg = YoloSegmenter(target, imgsz=imgsz, conf=conf, device=device)
    use_crop = card.get("pen_crop", False) if pen_crop is None else pen_crop
    if use_crop and card.get("pen_roi"):
        from broilerseg.roi import CroppedSegmenter, PenROI

        seg = CroppedSegmenter(seg, PenROI.from_dict(card["pen_roi"]),
                               mask_outside=bool(card.get("pen_mask_outside", False)))
    return seg, card
