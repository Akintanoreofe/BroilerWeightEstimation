"""FlockLabel main window.

Layout: a narrow browser on the left (calendar + videos of the selected day)
and the video on the right with one bar of controls under it. Each frame can
hold one box (the bird on the scale) and the scale reading seen on that frame.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QDate, QRectF, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QDoubleValidator, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateEdit, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressDialog, QPushButton, QSlider, QSpinBox, QSplitter, QToolButton,
                               QVBoxLayout, QWidget)

from flocklabel import __version__
from flocklabel.core.db import Project
from flocklabel.core.export import export_dataset, export_evaluation
from flocklabel.core.parser import parse_name
from flocklabel.core.video import list_videos, scan_folder

from .loader import ClipLoader
from .widgets import BOX_COLOR, BOX_NO_WEIGHT, STATUS_COLORS, AnnotationCanvas, BoxItem, DayStrip, FootageCalendar

GUIDE = (
    "<b>1. Make a project.</b> File → New / open project, then choose or create an <i>empty</i> folder "
    "(e.g. Documents/flock_project). Everything you label is saved there as you go.<br><br>"
    "<b>2. Add videos.</b> File → Import videos → Browse to the folder that holds the videos → Import. "
    "The preview shows the date and time read from each file name.<br><br>"
    "<b>3. Open a video.</b> Click a bold day on the calendar, then a video in the list below it "
    "(or a block on the strip under the video).<br><br>"
    "<b>4. Find a frame with the bird on the scale.</b> Space plays / pauses, ← → step one frame.<br><br>"
    "<b>5. Label it.</b> Drag a box around the bird. Type the reading on the scale display into "
    "<i>Scale reading</i> and press Enter. Done for that frame.<br><br>"
    "<b>6. More frames.</b> Press → to go to the next frame, then D to copy the box and reading from the last "
    "labelled frame. Drag the box if the bird moved; type a new reading if it changed.<br><br>"
    "<b>7. Next video.</b> Click <i>Done</i> when a video is finished (or <i>Skip</i>), then Page Down.<br><br>"
    "<b>8. Export.</b> File → Export dataset (images + boxes + weights), or Export evaluation set "
    "(for BroilerWeightEstimation).<br><br>"
    "Quit: ⌘Q (Ctrl+Q on Windows). This guide: F1.")

SHORTCUTS = (
    "Space  play / pause\n←  →  previous / next frame\nShift + ←  →  back / forward 10 frames\n"
    "Home / End  first / last frame\n,  .  previous / next labelled frame\n"
    "D  copy box + reading from the last labelled frame\nEnter  type the scale reading\n"
    "Esc  leave the reading field\nDelete / Backspace  remove this frame's label\n"
    "Page Up / Page Down  previous / next video\nCtrl+Z / Ctrl+Shift+Z  undo / redo\n"
    "F  fit video to window, mouse wheel  zoom\nCtrl+E  export dataset\n\nOn a Mac, Ctrl is ⌘.")


def progress_callback(dialog: QProgressDialog) -> Callable[[int, int, str], bool]:
    def cb(i: int, n: int, text: str) -> bool:
        dialog.setMaximum(max(n, 1))
        dialog.setValue(min(i, max(n, 1)))
        if text:
            dialog.setLabelText(text)
        QApplication.processEvents()
        return not dialog.wasCanceled()
    return cb


class WeightEdit(QLineEdit):
    """Reading field: Enter saves, Esc gives the keys back to the video."""

    def __init__(self, on_enter: Callable[[], None], on_escape: Callable[[], None]):
        super().__init__()
        self._enter, self._escape = on_enter, on_escape
        v = QDoubleValidator(0.0, 1e6, 3)
        v.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.setValidator(v)
        self.setPlaceholderText("e.g. 2412")
        self.setFixedWidth(110)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._enter()
            return
        if e.key() == Qt.Key.Key_Escape:
            self._escape()
            return
        super().keyPressEvent(e)


# --------------------------------------------------------------------------- dialogs
class ImportDialog(QDialog):
    def __init__(self, parent, start_dir: str, pattern: str):
        super().__init__(parent)
        self.setWindowTitle("Import videos")
        self.resize(760, 480)
        self.folder = QLineEdit(start_dir)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.custom = QLineEdit("" if pattern == "auto" else pattern)
        self.custom.setPlaceholderText("leave empty: the date and time are found automatically")
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        row = QHBoxLayout()
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        form = QFormLayout()
        form.addRow("Video folder", row)
        form.addRow("Custom regex (optional)", self.custom)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(QLabel("What will be imported:"))
        lay.addWidget(self.preview, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._refresh)
        self.folder.textChanged.connect(lambda: self._timer.start(300))
        self.custom.textChanged.connect(lambda: self._timer.start(300))
        self._refresh()

    def pattern(self) -> str:
        return self.custom.text().strip() or "auto"

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with your videos", self.folder.text())
        if d:
            self.folder.setText(d)

    def _refresh(self):
        folder = Path(self.folder.text().strip())
        if not folder.is_dir():
            self.preview.setPlainText("Click Browse… and choose the folder that holds your videos.")
            return
        if folder == Path.home() or folder == Path(folder.anchor):
            self.preview.setPlainText("That is your whole home folder or disk. Choose the folder with the videos.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            rels = list_videos(folder, limit=5001)
        except OSError as err:
            self.preview.setPlainText(str(err))
            return
        finally:
            QApplication.restoreOverrideCursor()
        if not rels:
            self.preview.setPlainText("No videos here (.mp4 .avi .mov .mkv .m4v .asf .dav .h264 .ts).")
            return
        try:
            parsed = [(r, parse_name(Path(r).name, self.pattern())) for r in rels[:300]]
        except Exception as err:
            self.preview.setPlainText(f"Custom regex error: {err}")
            return
        ok = sum(p is not None for _, p in parsed)
        lines = [f"✓ {p.start:%Y-%m-%d  %H:%M:%S}{'  →  ' + format(p.end, '%H:%M:%S') if p.end else ''}     {r}"
                 if p else f"✗ no date in name (file date used)     {r}" for r, p in parsed[:25]]
        more = "5000+" if len(rels) > 5000 else str(len(rels))
        self.preview.setPlainText(f"{more} videos. {ok} of the first {len(parsed)} names have a date and time.\n\n"
                                  + "\n".join(lines))


class ExportDialog(QDialog):
    def __init__(self, parent, kind: str, start_dir: str, days: Optional[tuple[str, str]]):
        super().__init__(parent)
        self.kind = kind
        self.setWindowTitle("Export dataset" if kind == "dataset" else "Export evaluation set")
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        self.out = QLineEdit(str(Path(start_dir) / f"{'dataset' if kind == 'dataset' else 'eval'}_{stamp}"))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.out, 1)
        row.addWidget(browse)
        form = QFormLayout()
        form.addRow("Save to", row)
        self.only_done = QCheckBox("Only videos marked Done")
        form.addRow("", self.only_done)
        self.use_dates = QCheckBox("Only these days")
        self.date_from, self.date_to = QDateEdit(), QDateEdit()
        lo, hi = days if days else (QDate.currentDate().toString("yyyy-MM-dd"),) * 2
        for de, val in ((self.date_from, lo), (self.date_to, hi)):
            de.setCalendarPopup(True)
            de.setDisplayFormat("yyyy-MM-dd")
            de.setDate(QDate.fromString(val, "yyyy-MM-dd"))
            de.setEnabled(False)
            self.use_dates.toggled.connect(de.setEnabled)
        dr = QHBoxLayout()
        for w in (self.use_dates, QLabel("from"), self.date_from, QLabel("to"), self.date_to):
            dr.addWidget(w)
        dr.addStretch(1)
        form.addRow("Dates", dr)
        if kind == "dataset":
            self.train, self.val, self.test = QSpinBox(), QSpinBox(), QSpinBox()
            sp = QHBoxLayout()
            for name, sb, v in (("train", self.train, 70), ("val", self.val, 15), ("test", self.test, 15)):
                sb.setRange(0, 100)
                sb.setSuffix(" %")
                sb.setValue(v)
                sp.addWidget(QLabel(name))
                sp.addWidget(sb)
            sp.addStretch(1)
            form.addRow("Split by video", sp)
            self.seed = QSpinBox()
            self.seed.setRange(0, 999999)
            self.seed.setValue(42)
            form.addRow("Random seed", self.seed)
            self.max_frames = QSpinBox()
            self.max_frames.setRange(0, 100000)
            self.max_frames.setSpecialValueText("all")
            form.addRow("Frames per video (max)", self.max_frames)
            self.crops = QCheckBox("Also save a cropped image of each bird")
            form.addRow("", self.crops)
        else:
            form.addRow(QLabel("One trimmed clip per video (cropped to the evaluation crop if set), metadata.csv\n"
                               "with the median reading in kg, and per-frame ground truth for scoring."))
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Export into", str(Path(self.out.text()).parent))
        if d:
            self.out.setText(str(Path(d) / Path(self.out.text()).name))

    def options(self) -> dict:
        opts: dict = {"only_done": self.only_done.isChecked()}
        if self.use_dates.isChecked():
            a, b = (d.date().toString("yyyy-MM-dd") for d in (self.date_from, self.date_to))
            opts.update(date_from=min(a, b), date_to=max(a, b))
        if self.kind == "dataset":
            tot = max(1, self.train.value() + self.val.value() + self.test.value())
            opts.update(ratios=(self.train.value() / tot, self.val.value() / tot, self.test.value() / tot),
                        seed=self.seed.value(), max_frames_per_video=self.max_frames.value() or None,
                        write_crops=self.crops.isChecked())
        return opts


# ----------------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self, show_guide: bool = True):
        super().__init__()
        self.setWindowTitle(f"FlockLabel {__version__}")
        self.resize(1440, 900)
        self.settings = QSettings("FlockLabel", "FlockLabel")
        self.project: Optional[Project] = None
        self.clip = None
        self.loader: Optional[ClipLoader] = None
        self._old_loaders: list[ClipLoader] = []
        self.frame = 0
        self.n_frames = 0
        self._want_frame: Optional[int] = None
        self.labels: dict[int, dict] = {}
        self.day: Optional[date] = None
        self._undo: list[tuple] = []
        self._redo: list[tuple] = []
        self.unit = self.settings.value("unit", "g")
        self.play_timer = QTimer(self)
        self.play_timer.setSingleShot(True)
        self.play_timer.timeout.connect(self._play_step)
        self.playing = False

        self._build_ui()
        self._build_menus()
        self._build_shortcuts()
        self.statusBar().showMessage("File → New / open project to start.  Help: F1")
        if show_guide:
            QTimer.singleShot(0, self.show_guide)

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        self.calendar = FootageCalendar()
        self.calendar.setMaximumHeight(230)
        self.calendar.selectionChanged.connect(self.on_day_selected)
        self.day_title = QLabel("Videos")
        self.video_list = QListWidget()
        self.video_list.itemClicked.connect(lambda it: self.open_clip(it.data(Qt.ItemDataRole.UserRole)))
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 0, 6)
        ll.addWidget(self.calendar)
        ll.addWidget(self.day_title)
        ll.addWidget(self.video_list, 1)
        left.setMinimumWidth(220)
        left.setMaximumWidth(340)

        self.canvas = AnnotationCanvas()
        self.canvas.box_drawn.connect(self.on_box_drawn)
        self.canvas.crop_drawn.connect(self.on_crop_drawn)

        self.info = QLabel("No video open")
        self.info.setStyleSheet("color: palette(mid);")

        def tbtn(text, tip, slot, w=38):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setFixedSize(QSize(w, 30))
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)        # keys stay with the video
            b.clicked.connect(slot)
            return b

        self.btn_play = tbtn("▶", "Play / pause (Space)", self.toggle_play, 46)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.setTracking(True)
        self.slider.valueChanged.connect(lambda v: self.show_frame(v))
        self.speed = QComboBox()
        self.speed.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for s in ("0.25×", "0.5×", "1×", "2×", "4×"):
            self.speed.addItem(s, float(s[:-1]))
        self.speed.setCurrentIndex(2)
        transport = QHBoxLayout()
        transport.setSpacing(4)
        for w in (tbtn("⏮", "First frame (Home)", lambda: self.show_frame(0)),
                  tbtn("◀", "Previous frame (←)", lambda: self.step(-1)), self.btn_play,
                  tbtn("▶|", "Next frame (→)", lambda: self.step(1)),
                  tbtn("⏭", "Last frame (End)", lambda: self.show_frame(self.n_frames - 1))):
            transport.addWidget(w)
        transport.addWidget(self.slider, 1)
        transport.addWidget(self.speed)

        self.weight = WeightEdit(self.save_weight, lambda: self.canvas.setFocus())
        self.unit_box = QComboBox()
        self.unit_box.addItems(["g", "kg"])
        self.unit_box.setCurrentText(self.unit)
        self.unit_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.unit_box.currentTextChanged.connect(self._unit_changed)
        save = QPushButton("Save ⏎")
        save.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        save.clicked.connect(self.save_weight)
        self.count_lbl = QLabel("0 frames labelled")
        self.btn_done = QPushButton("Done ✓")
        self.btn_done.setCheckable(True)
        self.btn_done.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_done.setToolTip("Mark this video finished")
        self.btn_done.clicked.connect(lambda on: self.set_status("done" if on else "todo"))
        self.btn_skip = QPushButton("Skip")
        self.btn_skip.setCheckable(True)
        self.btn_skip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_skip.setToolTip("No usable weighing in this video; left out of exports")
        self.btn_skip.clicked.connect(lambda on: self.set_status("skip" if on else "todo"))
        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(QLabel("<b>Scale reading</b>"))
        bar.addWidget(self.weight)
        bar.addWidget(self.unit_box)
        bar.addWidget(save)
        for text, tip, slot in (("Copy last (D)", "Copy box and reading from the last labelled frame", self.copy_last),
                                ("Remove (Del)", "Remove this frame's box and reading", self.delete_label),
                                ("◀ labelled (,)", "Previous labelled frame", lambda: self.jump_labelled(-1)),
                                ("labelled ▶ (.)", "Next labelled frame", lambda: self.jump_labelled(1))):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch(1)
        bar.addWidget(self.count_lbl)
        bar.addWidget(self.btn_skip)
        bar.addWidget(self.btn_done)

        self.strip = DayStrip()
        self.strip.clip_clicked.connect(lambda cid: self.open_clip(cid))

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 6, 6, 4)
        rl.setSpacing(4)
        rl.addWidget(self.canvas, 1)
        rl.addWidget(self.info)
        rl.addLayout(transport)
        rl.addLayout(bar)
        rl.addWidget(self.strip)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 1180])
        split.setCollapsible(1, False)
        self.setCentralWidget(split)

    def _build_menus(self):
        mb = self.menuBar()

        def act(menu, text, slot, key=None):
            a = QAction(text, self)
            if key:
                a.setShortcut(QKeySequence(key))
            a.triggered.connect(lambda checked=False: slot())
            menu.addAction(a)
            return a

        m = mb.addMenu("&File")
        act(m, "New / open project…", self.choose_project, "Ctrl+O")
        act(m, "Open last project", self.open_last_project, "Ctrl+Shift+O")
        m.addSeparator()
        act(m, "Import videos…", self.import_videos, "Ctrl+I")
        m.addSeparator()
        act(m, "Export dataset…", lambda: self.export("dataset"), "Ctrl+E")
        act(m, "Export evaluation set…", lambda: self.export("evaluation"), "Ctrl+Shift+E")
        m.addSeparator()
        act(m, "Quit", self.close, QKeySequence.StandardKey.Quit)
        m = mb.addMenu("&Edit")
        act(m, "Undo", self.undo, QKeySequence.StandardKey.Undo)
        act(m, "Redo", self.redo, "Ctrl+Shift+Z")
        m.addSeparator()
        act(m, "Draw evaluation crop…", self.start_crop)
        act(m, "Clear evaluation crop", self.clear_crop)
        act(m, "Set pixel_to_cm…", self.set_pixel_to_cm)
        m = mb.addMenu("&Video")
        act(m, "Previous video", lambda: self.next_clip(-1), "PgUp")
        act(m, "Next video", lambda: self.next_clip(1), "PgDown")
        act(m, "Fit to window", self.canvas.fit, "F")
        m = mb.addMenu("&Help")
        act(m, "Getting started", self.show_guide, "F1")
        act(m, "Keyboard shortcuts", lambda: QMessageBox.information(self, "Keyboard shortcuts", SHORTCUTS))

    def _build_shortcuts(self):
        for key, fn in {"Space": self.toggle_play, "Left": lambda: self.step(-1), "Right": lambda: self.step(1),
                        "Shift+Left": lambda: self.step(-10), "Shift+Right": lambda: self.step(10),
                        "Home": lambda: self.show_frame(0), "End": lambda: self.show_frame(self.n_frames - 1),
                        ",": lambda: self.jump_labelled(-1), ".": lambda: self.jump_labelled(1),
                        "D": self.copy_last, "Delete": self.delete_label, "Backspace": self.delete_label,
                        "Return": self.focus_weight, "Enter": self.focus_weight,
                        "Escape": self.cancel_mode}.items():
            QShortcut(QKeySequence(key), self).activated.connect(fn)

    def show_guide(self):
        box = QMessageBox(self)
        box.setWindowTitle("Getting started")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(GUIDE)
        last = self.settings.value("last_project", "")
        reopen = None
        if self.project is None and last and Path(last).is_dir():
            reopen = box.addButton(f"Open {Path(last).name}", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if reopen is not None and box.clickedButton() is reopen:
            self.open_project(last)

    # ----------------------------------------------------------------- project
    def choose_project(self):
        d = QFileDialog.getExistingDirectory(self, "Project folder (choose an empty folder to start a new project)",
                                             self.settings.value("last_project", str(Path.home())))
        if d:
            self.open_project(d)

    def open_last_project(self):
        last = self.settings.value("last_project", "")
        self.open_project(last) if last and Path(last).is_dir() else self.choose_project()

    def open_project(self, folder: str):
        self.close_clip()
        if self.project:
            self.project.close()
        try:
            self.project = Project(folder)
        except Exception as err:
            QMessageBox.critical(self, "Cannot open project", str(err))
            self.project = None
            return
        self.settings.setValue("last_project", folder)
        self.setWindowTitle(f"FlockLabel {__version__} — {Path(folder).name}")
        self._undo.clear()
        self._redo.clear()
        days = sorted(self.project.days_with_footage())
        self.calendar.set_days(self.project.days_with_footage())
        if days:
            self.calendar.setSelectedDate(QDate.fromString(days[0], "yyyy-MM-dd"))
            self.on_day_selected()
            self.statusBar().showMessage(f"{len(days)} days with video. Click a bold day, then a video.", 8000)
        else:
            self.video_list.clear()
            self.strip.set_day(None, [])
            self.statusBar().showMessage("New project. File → Import videos to add your footage.")
            QTimer.singleShot(0, self.import_videos)

    def _need_project(self) -> bool:
        if self.project is None:
            QMessageBox.information(self, "FlockLabel", "First make or open a project: File → New / open project.")
            return False
        return True

    def import_videos(self):
        if not self._need_project():
            return
        dlg = ImportDialog(self, self.project.get_meta("video_root") or str(Path.home()),
                           self.project.get_meta("pattern", "auto"))
        if not dlg.exec():
            return
        prog = QProgressDialog("Reading videos…", "Stop", 0, 1, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(300)
        prog.setMinimumWidth(560)
        try:
            rep = scan_folder(self.project, dlg.folder.text().strip(), dlg.pattern(), progress_callback(prog))
        except Exception as err:
            prog.close()
            QMessageBox.critical(self, "Import failed", str(err))
            return
        prog.close()
        msg = f"Added {rep.added} videos ({rep.skipped_existing} were already in the project)."
        if rep.no_date_in_name:
            msg += (f"\n\n{len(rep.no_date_in_name)} names had no date and time; the file's own date was used:\n"
                    + "\n".join(rep.no_date_in_name[:10]))
        if rep.unreadable:
            msg += f"\n\n{len(rep.unreadable)} could not be read:\n" + "\n".join(rep.unreadable[:10])
        QMessageBox.information(self, "Import finished", msg)
        self.open_project(str(self.project.folder))

    def export(self, kind: str):
        if not self._need_project():
            return
        days = sorted(self.project.days_with_footage())
        dlg = ExportDialog(self, kind, str(self.project.folder), (days[0], days[-1]) if days else None)
        if not dlg.exec():
            return
        out = Path(dlg.out.text().strip())
        if out.exists() and any(out.iterdir()) and QMessageBox.question(
                self, "Export", f"{out} is not empty. Write into it anyway?") != QMessageBox.StandardButton.Yes:
            return
        self.pause()
        prog = QProgressDialog("Exporting…", "Stop", 0, 1, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(300)
        prog.setMinimumWidth(560)
        try:
            fn = export_dataset if kind == "dataset" else export_evaluation
            counts = fn(self.project, out, progress=progress_callback(prog), **dlg.options())
        except (ValueError, InterruptedError, OSError) as err:
            prog.close()
            QMessageBox.warning(self, "Export", str(err))
            return
        prog.close()
        if kind == "dataset":
            msg = (f"Saved {counts['images']} labelled frames from {counts['videos']} videos.\n"
                   f"Videos per split: {counts['splits']}\n\n{out}")
        else:
            msg = (f"Saved {counts['videos']} clips and {counts['boxes']} ground-truth boxes to\n{out}\n\n"
                   f"Then, in BroilerWeightEstimation/src:\n  python run_eval.py \"{out}\"\n"
                   f"  python evaluate_against_gt.py \"{out / 'ground_truth' / 'gt_boxes.csv'}\" "
                   f"--results-dir \"{out / 'results'}\"")
            if counts["warnings"]:
                msg += f"\n\n{len(counts['warnings'])} warnings (all in export_info.json):\n" + \
                       "\n".join(counts["warnings"][:8])
        QMessageBox.information(self, "Export finished", msg)

    # --------------------------------------------------------------- browsing
    def on_day_selected(self):
        if not self.project:
            return
        qd = self.calendar.selectedDate()
        self.day = date(qd.year(), qd.month(), qd.day())
        clips = self.project.clips(day=self.day.isoformat())
        self.day_title.setText(f"<b>{self.day:%a %d %b %Y}</b> — {len(clips)} videos")
        self.video_list.clear()
        for c in clips:
            s, e = datetime.fromisoformat(c["start_local"]), datetime.fromisoformat(c["end_local"])
            need = c["n_labels"] - c["n_weighed"]
            mark = {"done": "✓ ", "skip": "– "}.get(c["status"], "")
            text = f"{mark}{s:%H:%M:%S} → {e:%H:%M:%S}"
            if c["n_labels"]:
                text += f"   {c['n_weighed']} labelled" + (f", {need} need reading" if need else "")
            if c["time_source"] != "filename":
                text += "   (date from file)"
            it = QListWidgetItem(text)
            it.setData(Qt.ItemDataRole.UserRole, c["id"])
            it.setToolTip(c["rel_path"])
            state = c["status"] if c["status"] in ("done", "skip") else ("some" if c["n_labels"] else "todo")
            if state != "todo":
                it.setForeground(STATUS_COLORS[state])
            self.video_list.addItem(it)
            if self.clip is not None and c["id"] == self.clip["id"]:
                it.setSelected(True)
        self.strip.set_day(self.day, clips)
        self.strip.set_current(self.clip["id"] if self.clip else None, self._frame_time())

    def _refresh_lists(self):
        if self.project:
            self.calendar.set_days(self.project.days_with_footage())
            if self.day:
                self.on_day_selected()

    def next_clip(self, direction: int):
        if not self.project:
            return
        clips = self.project.clips()
        if not clips:
            return
        ids = [c["id"] for c in clips]
        if self.clip is None or self.clip["id"] not in ids:
            target = clips[0]
        else:
            i = ids.index(self.clip["id"]) + direction
            if not 0 <= i < len(ids):
                self.statusBar().showMessage("No more videos that way.", 2500)
                return
            target = clips[i]
        d = datetime.fromisoformat(target["start_local"]).date()
        if d != self.day:
            self.calendar.setSelectedDate(QDate(d.year, d.month, d.day))
        self.open_clip(target["id"])

    def set_status(self, status: str):
        if not (self.project and self.clip):
            self._sync_status_buttons()
            return
        self.project.set_clip_status(self.clip["id"], status)
        self.clip = self.project.clip(self.clip["id"])
        self._sync_status_buttons()
        self._refresh_lists()
        if status == "done":
            self.statusBar().showMessage("Marked done. Page Down opens the next video.", 4000)

    def _sync_status_buttons(self):
        st = self.clip["status"] if self.clip else "todo"
        for b, on in ((self.btn_done, st == "done"), (self.btn_skip, st == "skip")):
            b.blockSignals(True)
            b.setChecked(on)
            b.blockSignals(False)

    # ----------------------------------------------------------------- video
    def close_clip(self):
        self.pause()
        if self.loader is not None:
            self.loader.stop()
            for sig in (self.loader.frame_ready, self.loader.loaded, self.loader.failed):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            old = self.loader
            self._old_loaders.append(old)
            old.finished.connect(lambda o=old: self._old_loaders.remove(o) if o in self._old_loaders else None)
        self.loader, self.clip, self.labels = None, None, {}
        self.frame, self.n_frames, self._want_frame = 0, 0, None
        self.canvas.clear()
        self.weight.clear()
        self.info.setText("No video open")

    def open_clip(self, clip_id: int):
        if not self.project:
            return
        if self.clip is not None and self.clip["id"] == clip_id:
            return
        clip = self.project.clip(clip_id)
        path = self.project.abs_path(clip["rel_path"])
        if not path.exists():
            QMessageBox.warning(self, "Video not found", f"{path}\n\nIf the folder or drive moved, import it again "
                                "from the new place (File → Import videos).")
            return
        self.close_clip()
        self.clip = clip
        self.labels = self.project.labels(clip_id)
        self.n_frames = max(1, clip["n_frames"])
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.n_frames - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.canvas.set_video_size(clip["width"], clip["height"])
        crop = self.project.crop
        self.canvas.set_crop(QRectF(*crop) if crop else None)
        self.canvas.show_message("Loading video…")
        self._want_frame = min(self.labels) if self.labels else 0
        self.loader = ClipLoader(str(path), self)
        self.loader.frame_ready.connect(self._on_frame_ready)
        self.loader.loaded.connect(self._on_loaded)
        self.loader.failed.connect(self._on_load_failed)
        self.loader.start()
        self._sync_status_buttons()
        self._update_count()
        self.canvas.setFocus()
        for i in range(self.video_list.count()):
            it = self.video_list.item(i)
            it.setSelected(it.data(Qt.ItemDataRole.UserRole) == clip_id)
        self.strip.set_current(clip_id, datetime.fromisoformat(clip["start_local"]))
        self.info.setText(f"Loading {Path(clip['rel_path']).name} …")

    def _on_frame_ready(self, i: int):
        if self._want_frame is not None and i >= self._want_frame:
            target, self._want_frame = self._want_frame, None
            self.show_frame(target)
        elif i % 10 == 0:
            self._update_info()

    def _on_loaded(self, n: int):
        cache = self.loader.cache
        self.n_frames = n
        self.slider.blockSignals(True)
        self.slider.setRange(0, n - 1)
        self.slider.blockSignals(False)
        dur = (cache.times_ms[-1] - cache.times_ms[0]) / 1000 if n > 1 else 0
        fps = (n - 1) / dur if dur > 0 else self.clip["fps"]
        end = None
        if self.clip["time_source"] != "filename" or self.clip["end_local"] == self.clip["start_local"]:
            end = (datetime.fromisoformat(self.clip["start_local"]) + timedelta(seconds=dur)).isoformat(
                timespec="seconds")
        if n != self.clip["n_frames"] or abs(fps - self.clip["fps"]) > 0.01:
            self.project.update_clip_timing(self.clip["id"], n, fps, end)
            self.clip = self.project.clip(self.clip["id"])
        if self._want_frame is not None:
            target, self._want_frame = min(self._want_frame, n - 1), None
            self.show_frame(target)
        self._update_info()

    def _on_load_failed(self, msg: str):
        self.canvas.show_message("This video could not be played")
        QMessageBox.warning(self, "Cannot play video", f"{msg}\n\n{self.clip['rel_path'] if self.clip else ''}")

    def _cached(self) -> int:
        return len(self.loader.cache) if self.loader else 0

    def show_frame(self, i: int):
        if not self.clip or not self.loader:
            return
        i = max(0, min(int(i), self.n_frames - 1))
        rgb = self.loader.cache.get(i)
        if rgb is None:                       # not decoded yet: show it as soon as it is
            self._want_frame = i
            self.canvas.show_message(f"Loading… {self._cached()} / {self.n_frames} frames")
            return
        self.frame = i
        self.canvas.show_frame(rgb, self.loader.cache.width or self.clip["width"])
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self._draw_box()
        self._update_info()
        self.strip.set_current(self.clip["id"], self._frame_time())

    def step(self, d: int):
        if self.clip:
            self.pause()
            self.show_frame(self.frame + d)

    def jump_labelled(self, direction: int):
        frames = sorted(self.labels)
        cands = [f for f in frames if (f > self.frame if direction > 0 else f < self.frame)]
        if cands:
            self.pause()
            self.show_frame(cands[0] if direction > 0 else cands[-1])
        else:
            self.statusBar().showMessage("No labelled frame that way.", 2000)

    def toggle_play(self):
        if self.playing:
            self.pause()
        elif self.clip and self.loader:
            if self.frame >= self.n_frames - 1:
                self.show_frame(0)
            self.playing = True
            self.btn_play.setText("⏸")
            self.play_timer.start(0)

    def pause(self):
        self.playing = False
        self.play_timer.stop()
        self.btn_play.setText("▶")

    def _play_step(self):
        if not self.playing or not self.loader:
            return
        nxt = self.frame + 1
        if nxt >= self.n_frames:
            self.pause()
            return
        if nxt >= self._cached():             # decoder behind: wait a moment
            self.play_timer.start(30)
            return
        cache = self.loader.cache
        t0, t1 = cache.time_ms(self.frame), cache.time_ms(nxt)
        gap = (t1 - t0) if (t0 is not None and t1 is not None and t1 > t0) else 1000 / max(1.0, self.clip["fps"])
        self.show_frame(nxt)
        if self.playing:   # real timing, so irregular NVR frame gaps play as recorded
            self.play_timer.start(int(min(1500, max(10, gap / self.speed.currentData()))))

    def _frame_time(self) -> Optional[datetime]:
        if not self.clip:
            return None
        t = self.loader.cache.time_ms(self.frame) if self.loader else None
        sec = t / 1000 if t is not None else self.frame / max(1e-6, self.clip["fps"])
        return datetime.fromisoformat(self.clip["start_local"]) + timedelta(seconds=sec)

    def _update_info(self):
        if not self.clip:
            self.info.setText("No video open")
            return
        t = self._frame_time()
        loading = "" if self._cached() >= self.n_frames else f"   loading {self._cached()}/{self.n_frames}"
        self.info.setText(f"{t:%Y-%m-%d  %H:%M:%S}.{t.microsecond // 1000:03d}     frame {self.frame + 1} of "
                          f"{self.n_frames}{loading}     {self.clip['width']}×{self.clip['height']}     "
                          f"{Path(self.clip['rel_path']).name}")

    # ----------------------------------------------------------------- labels
    def _fmt(self, grams: Optional[float]) -> str:
        if grams is None:
            return ""
        return f"{grams / 1000:.3f}".rstrip("0").rstrip(".") if self.unit == "kg" else f"{grams:.1f}".rstrip("0").rstrip(".")

    def _unit_changed(self, unit: str):
        self.unit = unit
        self.settings.setValue("unit", unit)
        self._draw_box()

    def _draw_box(self):
        lab = self.labels.get(self.frame)
        if lab is None:
            self.canvas.set_box(None)
            self.weight.clear()
            return
        w = lab["weight_g"]
        caption = f"{self._fmt(w)} {self.unit}" if w is not None else "type reading ⏎"
        rect = QRectF(lab["x1"], lab["y1"], lab["x2"] - lab["x1"], lab["y2"] - lab["y1"])
        self.canvas.set_box(BoxItem(rect, caption, BOX_COLOR if w is not None else BOX_NO_WEIGHT, self.on_box_moved))
        if not self.weight.hasFocus():
            self.weight.setText(self._fmt(w))

    def _apply(self, frame: int, before: Optional[dict], after: Optional[dict], text: str):
        """Change one frame's label, with undo."""
        cid = self.clip["id"]
        self.project.set_label(cid, frame, after)
        self._undo.append((cid, frame, before, after, text))
        self._redo.clear()
        self._reload_labels()

    def _reload_labels(self):
        self.labels = self.project.labels(self.clip["id"])
        self._draw_box()
        self._update_count()

    def _update_count(self):
        n = sum(1 for lab in self.labels.values() if lab["weight_g"] is not None)
        need = len(self.labels) - n
        self.count_lbl.setText(f"{n} frames labelled" + (f",  {need} need a reading" if need else ""))
        self.count_lbl.setStyleSheet("color:#c98a12" if need else "")

    def _time_ms(self) -> Optional[float]:
        return self.loader.cache.time_ms(self.frame) if self.loader else None

    def on_box_drawn(self, rect: QRectF):
        if not self.clip or self.loader is None or self.loader.cache.get(self.frame) is None:
            return
        self.pause()
        before = self.labels.get(self.frame)
        after = {"x1": rect.left(), "y1": rect.top(), "x2": rect.right(), "y2": rect.bottom(),
                 "time_ms": self._time_ms(), "weight_g": before["weight_g"] if before else None}
        self._apply(self.frame, before, after, "draw box")
        self.focus_weight()

    def on_box_moved(self, old: QRectF, new: QRectF):
        before = self.labels.get(self.frame)
        if before is None:
            return
        after = dict(before, x1=new.left(), y1=new.top(), x2=new.right(), y2=new.bottom())
        self._apply(self.frame, before, after, "move box")

    def focus_weight(self):
        if self.weight.hasFocus():          # Enter pressed inside the field
            self.save_weight()
            return
        if self.clip and self.frame in self.labels:
            self.pause()
            self.weight.setFocus()
            self.weight.selectAll()
            if not self.weight.text():
                last = [lab["weight_g"] for f, lab in sorted(self.labels.items())
                        if lab["weight_g"] is not None and f < self.frame]
                if last:
                    self.weight.setText(self._fmt(last[-1]))
                    self.weight.selectAll()
            self.statusBar().showMessage("Type the scale reading and press Enter (Esc to go back to the video).", 4000)
        elif self.clip:
            self.statusBar().showMessage("Drag a box around the bird first.", 3000)

    def save_weight(self):
        if not self.clip:
            return
        lab = self.labels.get(self.frame)
        if lab is None:
            self.statusBar().showMessage("Drag a box around the bird first, then type the reading.", 4000)
            return
        text = self.weight.text().strip().replace(",", ".")
        if not text:
            self.canvas.setFocus()
            return
        try:
            value = float(text)
        except ValueError:
            self.statusBar().showMessage("Type a number, e.g. 2412 (g) or 2.412 (kg).", 4000)
            return
        grams = value * 1000 if self.unit == "kg" else value
        if grams <= 0:
            self.statusBar().showMessage("The reading must be above 0: is the bird on the scale in this frame?", 5000)
            return
        if self.unit == "g" and grams < 20 and QMessageBox.question(
                self, "Check unit", f"{value} g is very light for a bird. Did you mean {value} kg?") \
                == QMessageBox.StandardButton.Yes:
            grams = value * 1000
        self._apply(self.frame, lab, dict(lab, weight_g=grams), "set reading")
        self.canvas.setFocus()
        self._refresh_lists()
        self.statusBar().showMessage(f"Saved frame {self.frame + 1}: {self._fmt(grams)} {self.unit}. "
                                     "→ next frame, D to copy this box.", 4000)

    def copy_last(self):
        if not self.clip:
            return
        frames = sorted(self.labels)
        src = next((f for f in reversed(frames) if f < self.frame), None)
        if src is None:
            src = next((f for f in frames if f > self.frame), None)
        if src is None:
            self.statusBar().showMessage("Nothing to copy yet: label one frame first.", 3000)
            return
        if self.loader is None or self.loader.cache.get(self.frame) is None:
            return
        s = self.labels[src]
        after = {k: s[k] for k in ("x1", "y1", "x2", "y2", "weight_g")}
        after["time_ms"] = self._time_ms()
        self._apply(self.frame, self.labels.get(self.frame), after, "copy box")
        self._refresh_lists()
        self.statusBar().showMessage(f"Copied from frame {src + 1}. Drag the box if the bird moved; "
                                     "press Enter to change the reading.", 4000)

    def delete_label(self):
        if self.clip and self.frame in self.labels:
            self._apply(self.frame, self.labels[self.frame], None, "remove label")
            self._refresh_lists()

    def undo(self):
        if self._undo:
            cid, frame, before, after, text = self._undo.pop()
            self.project.set_label(cid, frame, before)
            self._redo.append((cid, frame, before, after, text))
            self._after_history(cid, frame, f"Undid {text}")

    def redo(self):
        if self._redo:
            cid, frame, before, after, text = self._redo.pop()
            self.project.set_label(cid, frame, after)
            self._undo.append((cid, frame, before, after, text))
            self._after_history(cid, frame, f"Redid {text}")

    def _after_history(self, cid: int, frame: int, msg: str):
        if self.clip is None or self.clip["id"] != cid:
            self.open_clip(cid)
        if self.clip and self.clip["id"] == cid:
            self._reload_labels()
            self.show_frame(frame)          # waits for the frame if the video is still loading
        self._refresh_lists()
        self.statusBar().showMessage(msg, 2500)

    # ------------------------------------------------------------------- crop
    def start_crop(self):
        if not self.clip:
            QMessageBox.information(self, "Evaluation crop", "Open a video first.")
            return
        self.canvas.set_mode("crop")
        self.statusBar().showMessage("Drag a rectangle around the scale AND the orange feeder (the pipeline "
                                     "calibrates from it). Esc cancels.", 10000)

    def cancel_mode(self):
        if self.weight.hasFocus():
            self.canvas.setFocus()
            return
        if self.canvas.mode == "crop":
            self.canvas.set_mode("label")
            self.statusBar().showMessage("Crop cancelled.", 2000)

    def on_crop_drawn(self, rect: QRectF):
        r = rect.toRect()
        self.project.crop = (r.x(), r.y(), r.width(), r.height())
        self.canvas.set_crop(QRectF(r))
        self.canvas.set_mode("label")
        self.statusBar().showMessage(f"Evaluation crop saved: {r.width()}×{r.height()} at ({r.x()}, {r.y()}).", 6000)

    def clear_crop(self):
        if self.project:
            self.project.crop = None
            self.canvas.set_crop(None)

    def set_pixel_to_cm(self):
        if not self._need_project():
            return
        val, ok = QInputDialog.getDouble(self, "pixel_to_cm", "Centimetres per pixel in the evaluation clips "
                                         "(0 = let the pipeline measure the feeder):",
                                         self.project.pixel_to_cm or 0.0, 0, 10, 5)
        if ok:
            self.project.pixel_to_cm = val or None

    # ------------------------------------------------------------------ close
    def closeEvent(self, e):
        self.pause()
        for ld in [self.loader, *self._old_loaders]:
            if ld is not None:
                ld.stop()
                ld.wait(2000)
        if self.project:
            self.project.close()
        super().closeEvent(e)


def run(argv: list[str]) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("FlockLabel")
    app.setOrganizationName("FlockLabel")
    win = MainWindow(show_guide=not argv)
    win.show()
    if argv:
        QTimer.singleShot(0, lambda: win.open_project(argv[0]))
    return app.exec()

