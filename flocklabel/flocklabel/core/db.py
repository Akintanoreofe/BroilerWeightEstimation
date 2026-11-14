"""SQLite storage for a FlockLabel project.

A project is a folder holding ``flocklabel.sqlite``. Each video is a *clip*;
each labelled frame holds one box (the bird on the scale) and the scale
reading seen on that frame, in grams. Video paths are stored relative to the
video folder, POSIX style, so a project moves between Mac and Windows.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

DB_NAME = "flocklabel.sqlite"
SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS clips (
    id            INTEGER PRIMARY KEY,
    rel_path      TEXT NOT NULL UNIQUE,
    start_local   TEXT NOT NULL,          -- ISO date-time from the file name (NVR local time)
    end_local     TEXT NOT NULL,
    fps           REAL NOT NULL,          -- average: frames / duration
    n_frames      INTEGER NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    time_source   TEXT NOT NULL,          -- filename | file_date
    status        TEXT NOT NULL DEFAULT 'todo',   -- todo | done | skip
    readable      INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_clips_time ON clips(start_local);

CREATE TABLE IF NOT EXISTS labels (
    id            INTEGER PRIMARY KEY,
    clip_id       INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    frame_idx     INTEGER NOT NULL,       -- 0-based, in decode order
    time_ms       REAL,                   -- frame timestamp inside the video
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,  -- video pixels
    weight_g      REAL,                   -- scale reading on this frame
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (clip_id, frame_idx)
);

CREATE TABLE IF NOT EXISTS exports (
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, out_dir TEXT NOT NULL, settings TEXT NOT NULL,
    n_images INTEGER, n_boxes INTEGER, created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

LABEL_COLS = ("frame_idx", "time_ms", "x1", "y1", "x2", "y2", "weight_g")


class Project:
    """Open (or create) a project folder."""

    def __init__(self, folder: Path | str):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / DB_NAME
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))
            self.set_meta("weight_unit", "g")

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: Optional[str]) -> None:
        with self.tx():
            if value is None:
                self.conn.execute("DELETE FROM meta WHERE key=?", (key,))
            else:
                self.conn.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    @property
    def video_root(self) -> Optional[Path]:
        root = self.get_meta("video_root")
        return Path(root) if root else None

    def abs_path(self, rel_path: str) -> Path:
        return (self.video_root or self.folder) / Path(rel_path)

    @property
    def crop(self) -> Optional[tuple[int, int, int, int]]:
        """Evaluation crop (x, y, w, h) in video pixels, or ``None`` for the full frame."""
        raw = self.get_meta("eval_crop")
        return tuple(int(v) for v in raw.split(",")) if raw else None  # type: ignore[return-value]

    @crop.setter
    def crop(self, value: Optional[tuple[int, int, int, int]]) -> None:
        self.set_meta("eval_crop", ",".join(str(int(v)) for v in value) if value else None)

    @property
    def pixel_to_cm(self) -> Optional[float]:
        raw = self.get_meta("pixel_to_cm")
        return float(raw) if raw else None

    @pixel_to_cm.setter
    def pixel_to_cm(self, value: Optional[float]) -> None:
        self.set_meta("pixel_to_cm", f"{value:.6f}" if value else None)

    @contextmanager
    def tx(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def close(self) -> None:
        self.conn.close()

    # ----------------------------------------------------------------- clips
    def add_clip(self, **c: Any) -> Optional[int]:
        cols = ("rel_path", "start_local", "end_local", "fps", "n_frames", "width", "height",
                "time_source", "readable")
        with self.tx():
            cur = self.conn.execute(
                f"INSERT INTO clips({','.join(cols)}) VALUES({','.join('?' * len(cols))}) "
                "ON CONFLICT(rel_path) DO NOTHING", tuple(c.get(k) for k in cols))
        return cur.lastrowid if cur.rowcount else None

    def clip(self, clip_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT c.*, (SELECT COUNT(*) FROM labels l WHERE l.clip_id=c.id) AS n_labels, "
            "(SELECT COUNT(*) FROM labels l WHERE l.clip_id=c.id AND l.weight_g IS NOT NULL) AS n_weighed "
            "FROM clips c WHERE c.id=?", (clip_id,)).fetchone()

    def clips(self, day: Optional[str] = None) -> list[sqlite3.Row]:
        """Readable clips, optionally only those overlapping ``day`` (``YYYY-MM-DD``)."""
        sql = ("SELECT c.*, (SELECT COUNT(*) FROM labels l WHERE l.clip_id=c.id) AS n_labels, "
               "(SELECT COUNT(*) FROM labels l WHERE l.clip_id=c.id AND l.weight_g IS NOT NULL) AS n_weighed "
               "FROM clips c WHERE c.readable=1")
        args: list = []
        if day:
            sql += " AND substr(c.start_local,1,10) <= ? AND substr(c.end_local,1,10) >= ?"
            args += [day, day]
        return self.conn.execute(sql + " ORDER BY c.start_local, c.rel_path", args).fetchall()

    def days_with_footage(self) -> dict[str, str]:
        """``YYYY-MM-DD`` → ``none`` (nothing labelled), ``some`` or ``done`` (every clip done/skipped)."""
        rows = self.conn.execute(
            "SELECT substr(start_local,1,10) AS day, COUNT(*) AS n, "
            "SUM(status IN ('done','skip')) AS n_done, "
            "SUM((SELECT COUNT(*) FROM labels l WHERE l.clip_id=clips.id) > 0) AS n_lab "
            "FROM clips WHERE readable=1 GROUP BY day").fetchall()
        return {r["day"]: "done" if r["n_done"] == r["n"] else "some" if r["n_lab"] else "none" for r in rows}

    def set_clip_status(self, clip_id: int, status: str) -> None:
        if status not in ("todo", "done", "skip"):
            raise ValueError(status)
        with self.tx():
            self.conn.execute("UPDATE clips SET status=? WHERE id=?", (status, clip_id))

    def update_clip_timing(self, clip_id: int, n_frames: int, fps: float, end_local: Optional[str]) -> None:
        """Store the exact frame count found by a full decode (headers can be wrong)."""
        with self.tx():
            if end_local:
                self.conn.execute("UPDATE clips SET n_frames=?, fps=?, end_local=? WHERE id=?",
                                  (n_frames, fps, end_local, clip_id))
            else:
                self.conn.execute("UPDATE clips SET n_frames=?, fps=? WHERE id=?", (n_frames, fps, clip_id))

    # ---------------------------------------------------------------- labels
    def label(self, clip_id: int, frame_idx: int) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM labels WHERE clip_id=? AND frame_idx=?",
                              (clip_id, frame_idx)).fetchone()
        return dict(r) if r else None

    def labels(self, clip_id: int) -> dict[int, dict]:
        """Labels of one clip keyed by frame."""
        return {r["frame_idx"]: dict(r) for r in self.conn.execute(
            "SELECT * FROM labels WHERE clip_id=? ORDER BY frame_idx", (clip_id,))}

    def set_label(self, clip_id: int, frame_idx: int, data: Optional[dict]) -> None:
        """Replace the label of one frame (``None`` deletes it).

        ``data`` holds ``x1, y1, x2, y2`` and optionally ``weight_g``, ``time_ms``.
        """
        with self.tx():
            if data is None:
                self.conn.execute("DELETE FROM labels WHERE clip_id=? AND frame_idx=?", (clip_id, frame_idx))
                return
            x1, x2 = sorted((float(data["x1"]), float(data["x2"])))
            y1, y2 = sorted((float(data["y1"]), float(data["y2"])))
            self.conn.execute(
                "INSERT INTO labels(clip_id,frame_idx,time_ms,x1,y1,x2,y2,weight_g) VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(clip_id,frame_idx) DO UPDATE SET time_ms=excluded.time_ms, x1=excluded.x1, "
                "y1=excluded.y1, x2=excluded.x2, y2=excluded.y2, weight_g=excluded.weight_g, "
                "updated_at=datetime('now')",
                (clip_id, frame_idx, data.get("time_ms"), x1, y1, x2, y2, data.get("weight_g")))

    def export_rows(self, *, only_done: bool = False, date_from: Optional[str] = None,
                    date_to: Optional[str] = None) -> list[dict]:
        """Weighed labels joined with their clip, filtered by status and recording date (inclusive)."""
        sql = ("SELECT l.*, c.rel_path, c.start_local, c.fps, c.width, c.height, c.status "
               "FROM labels l JOIN clips c ON c.id=l.clip_id "
               "WHERE l.weight_g IS NOT NULL AND c.status != 'skip'")
        if only_done:
            sql += " AND c.status='done'"
        if date_from:
            sql += " AND substr(c.start_local,1,10) >= ?"
        if date_to:
            sql += " AND substr(c.start_local,1,10) <= ?"
        args = [d for d in (date_from, date_to) if d]
        out = []
        for r in self.conn.execute(sql + " ORDER BY c.start_local, l.clip_id, l.frame_idx", args):
            d = dict(r)
            sec = d["time_ms"] / 1000 if d["time_ms"] is not None else d["frame_idx"] / d["fps"]
            d["timestamp"] = datetime.fromisoformat(d["start_local"]) + timedelta(seconds=sec)
            out.append(d)
        return out

    def log_export(self, kind: str, out_dir: Path, settings: str, n_images: int, n_boxes: int) -> None:
        with self.tx():
            self.conn.execute("INSERT INTO exports(kind,out_dir,settings,n_images,n_boxes) VALUES(?,?,?,?,?)",
                              (kind, str(out_dir), settings, n_images, n_boxes))
