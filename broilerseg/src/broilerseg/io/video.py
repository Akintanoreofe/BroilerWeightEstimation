"""Scan a folder of videos and read / sample frames from them."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

# Silence harmless FFmpeg HEVC header warnings from CCTV/NVR files
# (e.g. "vps_reserved_ffff_16bits is not 0xffff"). Must be set before cv2 loads.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg")


def find_videos(folder: str | Path, recursive: bool = True) -> list[Path]:
    """Return every video file under ``folder`` (sorted)."""
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern)
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
                  and not p.name.startswith("."))


def video_info(path: str | Path) -> dict:
    """Frame count, fps, size and duration of one video."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    try:
        ok = cap.isOpened()
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if ok else 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if ok else 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if ok else 0
    finally:
        cap.release()
    return {"path": str(path), "name": path.name, "stem": path.stem, "readable": ok,
            "n_frames": n, "fps": fps, "width": w, "height": h,
            "duration_s": n / fps if fps > 0 else np.nan,
            "size_mb": path.stat().st_size / 1e6}


def video_id(path: str | Path, root: str | Path) -> str:
    """Return a unique, file-name-safe id: the path below ``root`` joined with ``__``.

    ``root/day1/cam2/ch01.mp4`` -> ``day1__cam2__ch01``. Videos with the same
    file name in different folders therefore never collide.
    """
    rel = Path(path).resolve().relative_to(Path(root).resolve()).with_suffix("")
    return "__".join(part.replace(" ", "_") for part in rel.parts)


def scan_videos(folder: str | Path, recursive: bool = True) -> pd.DataFrame:
    """Inventory table of every video under ``folder`` (searched recursively).

    Besides the video properties it records ``folder`` (the sub-folder below
    ``folder``, ``"."`` for the top level), ``rel_path`` and ``video_id``
    (unique across sub-folders).
    """
    root = Path(folder).expanduser()
    rows = []
    for p in find_videos(root, recursive):
        info = video_info(p)
        rel = p.resolve().relative_to(root.resolve())
        info.update(rel_path=str(rel), folder=str(rel.parent), video_id=video_id(p, root))
        rows.append(info)
    return pd.DataFrame(rows, columns=["path", "name", "stem", "video_id", "folder", "rel_path",
                                       "readable", "n_frames", "fps", "width", "height",
                                       "duration_s", "size_mb"])


def read_frame(path: str | Path, index: int, rgb: bool = True) -> np.ndarray:
    """Read one frame by zero-based index."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index} of {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if rgb else frame


def iter_frames(path: str | Path, indices=None, rgb: bool = True
                ) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, frame)`` sequentially; only ``indices`` if given.

    Sequential decoding is faster and more reliable than seeking for
    CCTV-converted MP4 files.
    """
    wanted = None if indices is None else set(int(i) for i in indices)
    last = None if wanted is None else (max(wanted) if wanted else -1)
    cap = cv2.VideoCapture(str(path))
    try:
        i = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if last is not None and i > last:
                break
            if wanted is None or i in wanted:
                yield i, (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if rgb else frame)
    finally:
        cap.release()


def motion_scores(path: str | Path, stride: int = 5, size: int = 160) -> pd.DataFrame:
    """Mean absolute difference between consecutive sampled frames.

    High scores mark activity (a bird stepping on or off the scale).
    """
    rows, prev = [], None
    for i, frame in iter_frames(path, rgb=False):
        if i % stride:
            continue
        g = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        rows.append({"frame_idx": i, "motion": np.nan if prev is None else
                     float(np.abs(g - prev).mean())})
        prev = g
    return pd.DataFrame(rows, columns=["frame_idx", "motion"])


def sample_frame_indices(n_frames: int, k: int, strategy: str = "uniform", seed: int = 0,
                         path: str | Path | None = None, skip_start: int = 0) -> list[int]:
    """Choose ``k`` frame indices from a video.

    Parameters
    ----------
    n_frames : int
        Frames in the video.
    k : int
        Frames wanted.
    strategy : {"uniform", "random", "motion"}
        ``uniform`` spaces frames evenly; ``random`` draws without replacement;
        ``motion`` keeps the most active frames (needs ``path``), spread out so
        they are not all neighbours.
    seed : int
        Random seed.
    path : path-like, optional
        Video path, required for ``"motion"``.
    skip_start : int
        Ignore the first frames (CCTV overlays, black frames).
    """
    lo = min(max(0, skip_start), max(0, n_frames - 1))
    pool = np.arange(lo, n_frames)
    if k <= 0 or len(pool) == 0:
        return []
    if len(pool) <= k:
        return pool.tolist()
    if strategy == "uniform":
        return sorted({int(pool[round(j)]) for j in np.linspace(0, len(pool) - 1, k)})
    if strategy == "random":
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(pool, size=k, replace=False).tolist())
    if strategy == "motion":
        if path is None:
            raise ValueError("strategy='motion' needs the video path")
        scores = motion_scores(path).dropna()
        scores = scores[scores.frame_idx >= lo]
        gap = max(1, len(pool) // (3 * k))
        chosen: list[int] = []
        for f in scores.sort_values("motion", ascending=False).frame_idx:
            if all(abs(int(f) - c) >= gap for c in chosen):
                chosen.append(int(f))
            if len(chosen) == k:
                break
        return sorted(chosen)
    raise ValueError("strategy must be 'uniform', 'random' or 'motion'")
