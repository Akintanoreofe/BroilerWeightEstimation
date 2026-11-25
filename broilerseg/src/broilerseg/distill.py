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


#: Where :func:`resolve_base_model` keeps downloaded starting weights.
BASE_WEIGHTS_DIR = Path("~/broilerseg_models/base")
_ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"


def resolve_base_model(base_model: str | Path, weights_dir: str | Path = BASE_WEIGHTS_DIR,
                       retries: int = 10) -> str:
    """Local path of the starting weights, downloading them once if needed.

    A full path (``~`` allowed) is used as is. A bare name such as
    ``yolov8n-seg.pt`` is looked up in the working folder, then in
    ``weights_dir`` (``~/broilerseg_models/base``); if missing it is downloaded
    there with retries that resume where a dropped connection stopped
    (Ultralytics' own download gives up on unstable networks, "Curl return
    value 56"). Model definitions (``.yaml``) are returned unchanged.
    """
    p = Path(str(base_model)).expanduser()
    if p.suffix != ".pt" or p.is_file():
        return str(p)
    if p.parent != Path("."):
        raise FileNotFoundError(f"Starting weights not found: {p}")
    target = Path(weights_dir).expanduser() / p.name
    if target.is_file() and target.stat().st_size > 1e6:
        return str(target)
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    url = _ASSETS_URL + p.name
    for attempt in range(1, retries + 1):
        have = target.stat().st_size if target.exists() else 0
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r, \
                    open(target, "ab" if have and r.status == 206 else "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            if target.stat().st_size > 1e6:
                print(f"Downloaded {p.name} -> {target}")
                return str(target)
        except Exception as e:  # noqa: BLE001 - network errors of every kind: retry
            print(f"download attempt {attempt}/{retries} failed ({e}); retrying ...")
            time.sleep(min(30, 3 * attempt))
    raise ConnectionError(
        f"Could not download {url}. Download it in a browser and save it as {target}, "
        f"or set BASE_MODEL to the file's full path.")


def patch_mps_assigner() -> bool:
    """Work around an Apple-GPU (MPS) bug in Ultralytics' label matching.

    On MPS the task-aligned assigner can crash with "shape mismatch: value
    tensor of shape [...] cannot be broadcast to indexing result". The patch
    runs only that step on the CPU; the network keeps training on the GPU.
    Safe to call more than once. Returns True if the patch is active.
    """
    import torch
    from ultralytics.utils import tal

    cls = tal.TaskAlignedAssigner
    if getattr(cls, "_broilerseg_cpu_on_mps", False):
        return True
    orig = cls.forward

    def forward(self, *args, **kwargs):
        dev = next((a.device for a in args if torch.is_tensor(a)), None)
        if dev is None or dev.type != "mps":
            return orig(self, *args, **kwargs)

        def to_cpu(a):
            return a.cpu() if torch.is_tensor(a) else a

        out = orig(self, *map(to_cpu, args), **{k: to_cpu(v) for k, v in kwargs.items()})
        return tuple(t.to(dev) if torch.is_tensor(t) else t for t in out)

    cls.forward = forward
    cls._broilerseg_cpu_on_mps = True
    return True


def patch_nms_time_limit(max_time_img: float = 1.0) -> bool:
    """Give Ultralytics' NMS more time per image during validation.

    Ultralytics stops NMS after ``2.0 + max_time_img * batch`` seconds
    (``max_time_img = 0.05`` by default) and silently drops the detections it
    has not processed ("NMS time limit exceeded"). With ~30 birds per crop
    and an untrained model on a Mac, every validation batch hits that limit,
    so recall, mAP, early stopping and the choice of ``best.pt`` are all
    distorted. This raises ``max_time_img`` (default 1 s per image). Safe to
    call more than once; returns False if this Ultralytics version has no
    such setting.
    """
    import importlib
    import inspect
    import sys

    patched = False
    for mod_name in ("ultralytics.utils.nms", "ultralytics.utils.ops"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        orig = getattr(mod, "non_max_suppression", None)
        if orig is None:
            continue
        orig = getattr(orig, "_broilerseg_orig", orig)
        if "max_time_img" not in inspect.signature(orig).parameters:
            continue

        def nms(*args, _orig=orig, **kwargs):
            kwargs["max_time_img"] = max(float(kwargs.get("max_time_img", 0.05)), max_time_img)
            return _orig(*args, **kwargs)

        nms._broilerseg_orig = orig
        # replace it wherever it was imported by name, too
        for m in list(sys.modules.values()):
            if getattr(m, "__name__", "").startswith("ultralytics") and \
                    getattr(getattr(m, "non_max_suppression", None), "_broilerseg_orig",
                            getattr(m, "non_max_suppression", None)) is orig:
                m.non_max_suppression = nms
        patched = True
    return patched


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
        COCO-pretrained start point: ``yolov8n-seg.pt`` (Pi), ``yolov8s-seg.pt``,
        or a full path. Bare names are fetched once into ``~/broilerseg_models/base``
        (see :func:`resolve_base_model`).
    epochs, imgsz, batch, patience : int
        Usual Ultralytics settings; ``patience`` stops early when validation
        stops improving.
    device : str, optional
        Defaults to :func:`pick_train_device`. On ``"mps"`` the Apple-GPU
        workaround :func:`patch_mps_assigner` is applied automatically.
        :func:`patch_nms_time_limit` is always applied, so validation is not
        cut short on crowded images.
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

    device = device or pick_train_device()
    if str(device) == "mps":
        patch_mps_assigner()
    patch_nms_time_limit()
    model = YOLO(resolve_base_model(base_model))
    args = {"data": str(data_yaml), "epochs": epochs, "imgsz": imgsz, "batch": batch,
            "device": device, "project": str(project), "name": name,
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
    images, teacher_results, student: Segmenter, keep=None,
) -> tuple[pd.DataFrame, dict[str, SegmentationResult]]:
    """Run the student on images the teacher labelled and measure agreement.

    ``images`` and ``teacher_results`` may be lazy mappings
    (:class:`broilerseg.labeling.LazyResults`): each frame is handled and then
    released. ``keep`` limits which student results are returned (all if
    None; ``()`` for none). Full-resolution masks of many frames do not fit
    in memory, so re-run the student on the few frames you want to plot.

    Returns a long table (one row per image and class) and the student results.
    """
    rows, student_results = [], {}
    keep = None if keep is None else set(keep)
    for stem in images:
        img = images[stem]
        res = student.segment(img)
        if keep is None or stem in keep:
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
    patch_nms_time_limit()
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
