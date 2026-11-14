"""Background video decoding.

A clip is decoded once, front to back, on a worker thread. Each frame is
shrunk to display size and kept in memory (or as JPEG once memory gets
tight), together with its real timestamp. Playing, stepping and jumping then
only read the cache, so the window stays responsive with 4K H.265 files and
variable frame rates.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

DISPLAY_MAX_WIDTH = 1600
RAW_BUDGET_BYTES = 1_200_000_000     # beyond this, frames are kept as JPEG


class FrameCache:
    """Display-size RGB frames plus timestamps, filled by :class:`ClipLoader`."""

    def __init__(self):
        self._frames: list = []          # np.ndarray (RGB) or bytes (JPEG of BGR)
        self.times_ms: list[float] = []
        self._raw_bytes = 0
        self._lock = threading.Lock()
        self.width = self.height = 0     # original video size
        self.disp_width = self.disp_height = 0

    def __len__(self) -> int:
        return len(self._frames)

    def add(self, rgb: np.ndarray, t_ms: float) -> None:
        with self._lock:
            if self._raw_bytes + rgb.nbytes <= RAW_BUDGET_BYTES:
                self._frames.append(rgb)
                self._raw_bytes += rgb.nbytes
            else:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
                self._frames.append(buf.tobytes() if ok else rgb)
            self.times_ms.append(t_ms)

    def get(self, i: int) -> Optional[np.ndarray]:
        with self._lock:
            if not 0 <= i < len(self._frames):
                return None
            f = self._frames[i]
        if isinstance(f, bytes):
            bgr = cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return f

    def time_ms(self, i: int) -> Optional[float]:
        with self._lock:
            return self.times_ms[i] if 0 <= i < len(self.times_ms) else None


class ClipLoader(QThread):
    """Decode a video into a :class:`FrameCache`."""

    frame_ready = Signal(int)            # index of the newest decoded frame
    loaded = Signal(int)                 # total frames
    failed = Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.cache = FrameCache()
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.failed.emit(f"Cannot open {self.path}")
            return
        try:
            n = 0
            while not self._stop:
                ok, bgr = cap.read()
                if not ok:
                    break
                t = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                h, w = bgr.shape[:2]
                if n == 0:
                    self.cache.width, self.cache.height = w, h
                    dw = min(DISPLAY_MAX_WIDTH, w)
                    self.cache.disp_width, self.cache.disp_height = dw, max(1, round(h * dw / w))
                if w != self.cache.disp_width:
                    bgr = cv2.resize(bgr, (self.cache.disp_width, self.cache.disp_height),
                                     interpolation=cv2.INTER_AREA)
                self.cache.add(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t)
                self.frame_ready.emit(n)
                n += 1
            if self._stop:
                return
            if n == 0:
                self.failed.emit("The video opened but no frame could be decoded. It may be damaged, "
                                 "or in a codec this OpenCV build lacks.")
                return
            self.loaded.emit(n)
        finally:
            cap.release()
