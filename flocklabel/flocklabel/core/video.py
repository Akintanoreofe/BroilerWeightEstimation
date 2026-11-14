"""Video access and folder import.

Frames are always counted in *decode order* from the start of the file. The app
and the exporters both decode sequentially, so frame numbers agree even for
variable-frame-rate NVR files whose headers report the wrong rate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from .db import Project
from .parser import parse_name

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".asf", ".dav", ".h264", ".264", ".ts"}


def probe(path: Path | str) -> dict:
    """Width, height, frame count and average fps from the container header."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {"readable": False, "width": 0, "height": 0, "n_frames": 0, "fps": 0.0}
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
        return {"readable": bool(ok and w and h), "width": w, "height": h, "n_frames": n,
                "fps": fps if 0 < fps < 500 else 25.0}
    finally:
        cap.release()


def iter_frames(path: Path | str, last: Optional[int] = None) -> Iterator[tuple[int, np.ndarray, float]]:
    """Yield ``(frame_idx, bgr, time_ms)`` in decode order, up to frame ``last`` inclusive."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        idx = 0
        while last is None or idx <= last:
            ok, bgr = cap.read()
            if not ok:
                break
            yield idx, bgr, float(cap.get(cv2.CAP_PROP_POS_MSEC))
            idx += 1
    finally:
        cap.release()


@dataclass
class ImportReport:
    added: int = 0
    skipped_existing: int = 0
    no_date_in_name: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)


def list_videos(folder: Path, limit: Optional[int] = None) -> list[str]:
    """Relative paths of videos under ``folder``; hidden folders are skipped."""
    found: list[str] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if Path(name).suffix.lower() in VIDEO_EXTS and not name.startswith("."):
                found.append((Path(root) / name).relative_to(folder).as_posix())
                if limit and len(found) >= limit:
                    return found
    return found


def scan_folder(project: Project, folder: Path | str, pattern: str = "auto",
                progress: Optional[Callable[[int, int, str], bool]] = None) -> ImportReport:
    """Add every video under ``folder`` to the project.

    The recording time comes from the file name; when the name has none, the
    file's modification date is used and the clip is marked ``file_date``.
    ``progress(i, n, name)`` may return ``False`` to stop.
    """
    folder = Path(folder)
    project.set_meta("video_root", str(folder.resolve()))
    project.set_meta("pattern", pattern)
    files = list_videos(folder)
    existing = {r[0] for r in project.conn.execute("SELECT rel_path FROM clips")}
    report = ImportReport()
    for i, rel in enumerate(files):
        if progress and progress(i, len(files), rel) is False:
            break
        if rel in existing:
            report.skipped_existing += 1
            continue
        info = probe(folder / rel)
        if not info["readable"]:
            report.unreadable.append(rel)
        parsed = parse_name(Path(rel).name, pattern) or parse_name(rel, pattern)
        if parsed:
            start, source = parsed.start, "filename"
        else:
            start = datetime.fromtimestamp(os.path.getmtime(folder / rel)).replace(microsecond=0)
            source = "file_date"
            report.no_date_in_name.append(rel)
        dur = info["n_frames"] / info["fps"] if info["fps"] else 0.0
        end = parsed.end if parsed and parsed.end else start + timedelta(seconds=dur)
        project.add_clip(rel_path=rel, start_local=start.isoformat(timespec="seconds"),
                         end_local=end.isoformat(timespec="seconds"), fps=info["fps"],
                         n_frames=info["n_frames"], width=info["width"], height=info["height"],
                         time_source=source, readable=int(info["readable"]))
        existing.add(rel)
        report.added += 1
    if progress:
        progress(len(files), len(files), "")
    return report
