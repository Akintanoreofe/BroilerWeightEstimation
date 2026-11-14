"""Custom Qt widgets: footage calendar, day strip and annotation canvas."""

from __future__ import annotations

from datetime import date, datetime
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QDate, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QCursor, QFont, QImage, QPainter, QPalette, QPen, QPixmap,
                           QTextCharFormat)
from PySide6.QtWidgets import (QCalendarWidget, QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QSizePolicy, QWidget)

STATUS_COLORS = {"todo": QColor("#5b7fa6"), "some": QColor("#c98a12"), "done": QColor("#2e8b57"),
                 "skip": QColor("#8a8a8a"), "none": QColor("#5b7fa6")}
BOX_COLOR = QColor("#3ddc84")
BOX_NO_WEIGHT = QColor("#ffb020")


class FootageCalendar(QCalendarWidget):
    """Month calendar; days with footage are bold and tinted by progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setGridVisible(False)
        self.setVerticalHeaderFormat(QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader)
        self._marked: list[QDate] = []

    def set_days(self, days: dict[str, str]) -> None:
        for d in self._marked:
            self.setDateTextFormat(d, QTextCharFormat())
        self._marked = []
        for day, state in days.items():
            qd = QDate.fromString(day, "yyyy-MM-dd")
            if not qd.isValid():
                continue
            color = STATUS_COLORS.get(state, STATUS_COLORS["none"])
            fmt = QTextCharFormat()
            fmt.setFontWeight(700)
            fmt.setForeground(QBrush(color))
            tint = QColor(color)
            tint.setAlpha(50)
            fmt.setBackground(QBrush(tint))
            self.setDateTextFormat(qd, fmt)
            self._marked.append(qd)


class DayStrip(QWidget):
    """The selected day from 00:00 to 24:00; each video is a block. Click a block to open it.

    Wheel zooms around the cursor (down to one minute), drag pans, double-click shows the whole day.
    """

    clip_clicked = Signal(int)

    RULER_H, BAR_H = 18, 26

    def __init__(self, parent=None):
        super().__init__(parent)
        self.day: Optional[date] = None
        self.clips: list[dict] = []
        self.current: Optional[int] = None
        self.cursor_time: Optional[datetime] = None
        self.t0, self.t1 = 0.0, 86400.0
        self._press: Optional[QPointF] = None
        self._press_view = (0.0, 0.0)
        self._dragged = False
        self.setMouseTracking(True)
        self.setFixedHeight(self.RULER_H + self.BAR_H + 8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_day(self, day: Optional[date], clips: list) -> None:
        self.day = day
        self.clips = []
        for c in clips:
            state = c["status"] if c["status"] in ("done", "skip") else ("some" if c["n_labels"] else "todo")
            self.clips.append({"id": c["id"], "start": datetime.fromisoformat(c["start_local"]),
                               "end": datetime.fromisoformat(c["end_local"]), "state": state})
        if self.clips:   # zoom to the recorded part of the day
            lo = min(self._sod(c["start"]) for c in self.clips)
            hi = max(self._sod(c["end"]) for c in self.clips)
            pad = max(300.0, (hi - lo) * 0.05)
            self.t0, self.t1 = max(0.0, lo - pad), min(86400.0, hi + pad)
        else:
            self.t0, self.t1 = 0.0, 86400.0
        self.update()

    def set_current(self, clip_id: Optional[int], when: Optional[datetime] = None) -> None:
        self.current, self.cursor_time = clip_id, when
        self.update()

    def _sod(self, dt: datetime) -> float:
        if self.day is None:
            return 0.0
        base = datetime.combine(self.day, datetime.min.time())
        return min(max((dt - base).total_seconds(), 0.0), 86400.0)

    def _x(self, t: float) -> float:
        return 8 + (t - self.t0) / (self.t1 - self.t0) * max(1, self.width() - 16)

    def _t(self, x: float) -> float:
        return self.t0 + (x - 8) / max(1, self.width() - 16) * (self.t1 - self.t0)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        text = self.palette().color(QPalette.ColorRole.WindowText)
        grid = QColor(text)
        grid.setAlpha(45)
        p.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.Base))
        if self.day is None:
            p.setPen(text)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Pick a day on the calendar")
            return
        span = self.t1 - self.t0
        step = next(s for s in (10, 30, 60, 300, 600, 1800, 3600, 7200, 10800, 21600)
                    if span / s <= max(4, self.width() / 80))
        f = p.font()
        f.setPointSizeF(max(7.5, f.pointSizeF() - 1.5))
        p.setFont(f)
        t = (int(self.t0) // step) * step
        while t <= self.t1:
            if t >= self.t0:
                x = int(self._x(t))
                p.setPen(grid)
                p.drawLine(x, self.RULER_H - 3, x, self.height())
                p.setPen(text)
                hh, rem = divmod(int(t), 3600)
                mm, ss = divmod(rem, 60)
                p.drawText(x + 3, 13, f"{hh:02d}:{mm:02d}" + (f":{ss:02d}" if step < 60 else ""))
            t += step
        y = self.RULER_H + 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grid)
        p.drawRect(QRectF(8, y, self.width() - 16, self.BAR_H))
        for c in self.clips:
            x0, x1 = self._x(self._sod(c["start"])), self._x(self._sod(c["end"]))
            if x1 < 0 or x0 > self.width():
                continue
            p.setBrush(STATUS_COLORS[c["state"]])
            p.setPen(QPen(text, 2) if c["id"] == self.current else Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(x0, y + 3, max(4.0, x1 - x0), self.BAR_H - 6), 2, 2)
        if self.cursor_time is not None and self.cursor_time.date() == self.day:
            x = int(self._x(self._sod(self.cursor_time)))
            p.setPen(QPen(QColor("#d64545"), 2))
            p.drawLine(x, y, x, y + self.BAR_H)

    def _hit(self, pos: QPointF) -> Optional[int]:
        if self.day is None or not self.RULER_H <= pos.y() <= self.RULER_H + self.BAR_H + 4:
            return None
        t = self._t(pos.x())
        tol = 4 / max(1, self.width()) * (self.t1 - self.t0)      # 4 px, so short clips are clickable
        for c in self.clips:
            if self._sod(c["start"]) - tol <= t <= self._sod(c["end"]) + tol:
                return c["id"]
        return None

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._press, self._press_view, self._dragged = e.position(), (self.t0, self.t1), False

    def mouseMoveEvent(self, e):
        if self._press is not None and e.buttons() & Qt.MouseButton.LeftButton:
            dx = e.position().x() - self._press.x()
            if abs(dx) > 4:
                self._dragged = True
                t0, t1 = self._press_view
                shift = -dx / max(1, self.width() - 16) * (t1 - t0)
                shift = max(-t0, min(shift, 86400 - t1))
                self.t0, self.t1 = t0 + shift, t1 + shift
                self.update()
        else:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor if self._hit(e.position())
                                   else Qt.CursorShape.ArrowCursor))

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._press is not None and not self._dragged:
            hit = self._hit(e.position())
            if hit is not None:
                self.clip_clicked.emit(hit)
        self._press = None

    def mouseDoubleClickEvent(self, _):
        self.t0, self.t1 = 0.0, 86400.0
        self.update()

    def wheelEvent(self, e):
        if self.day is None:
            return
        anchor = self._t(e.position().x())
        span = min(86400.0, max(60.0, (self.t1 - self.t0) * (0.8 if e.angleDelta().y() > 0 else 1.25)))
        frac = (anchor - self.t0) / (self.t1 - self.t0)
        t0 = max(0.0, min(anchor - frac * span, 86400.0 - span))
        self.t0, self.t1 = t0, t0 + span
        self.update()


class BoxItem(QGraphicsRectItem):
    """The bird's box, in video-pixel coordinates. Drag inside to move, drag an edge or corner to resize."""

    HANDLE = 12

    def __init__(self, rect: QRectF, caption: str, color: QColor, on_change: Callable[[QRectF, QRectF], None]):
        super().__init__(rect)
        self.on_change = on_change
        self.color = color
        self._mode: Optional[str] = None
        self._start_pos = QPointF()
        self._start_rect = QRectF()
        self.setAcceptHoverEvents(True)
        self.setZValue(10)
        self._caption = QGraphicsSimpleTextItem(caption, self)
        self._caption.setBrush(QBrush(color))
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        self._caption.setFont(f)
        self._caption.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        pen = QPen(color, 2.5)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(28)
        self.setBrush(QBrush(fill))
        self._place_caption()

    def _place_caption(self) -> None:
        r = self.rect()
        self._caption.setPos(r.left(), r.top())

    def _view_scale(self) -> float:
        views = self.scene().views() if self.scene() else []
        return max(views[0].transform().m11(), 1e-3) if views else 1.0

    def edge_at(self, pos: QPointF) -> Optional[str]:
        r, h = self.rect(), self.HANDLE / self._view_scale()
        vert = "t" if abs(pos.y() - r.top()) <= h else "b" if abs(pos.y() - r.bottom()) <= h else ""
        horiz = "l" if abs(pos.x() - r.left()) <= h else "r" if abs(pos.x() - r.right()) <= h else ""
        return (vert + horiz) or None

    def hoverMoveEvent(self, e):
        shapes = {"tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
                  "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor,
                  "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
                  "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor}
        self.setCursor(QCursor(shapes.get(self.edge_at(e.pos()), Qt.CursorShape.SizeAllCursor)))

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            e.ignore()
            return
        self._mode = self.edge_at(e.pos()) or "move"
        self._start_pos, self._start_rect = e.scenePos(), QRectF(self.rect())
        e.accept()

    def mouseMoveEvent(self, e):
        if not self._mode:
            return
        d = e.scenePos() - self._start_pos
        r = QRectF(self._start_rect)
        if self._mode == "move":
            r.translate(d)
        else:
            if "l" in self._mode:
                r.setLeft(r.left() + d.x())
            if "r" in self._mode:
                r.setRight(r.right() + d.x())
            if "t" in self._mode:
                r.setTop(r.top() + d.y())
            if "b" in self._mode:
                r.setBottom(r.bottom() + d.y())
        r = r.normalized().intersected(self.scene().sceneRect())
        if r.width() >= 4 and r.height() >= 4:
            self.setRect(r)
            self._place_caption()

    def mouseReleaseEvent(self, e):
        if self._mode and self.rect() != self._start_rect:
            self.on_change(QRectF(self._start_rect), QRectF(self.rect()))
        self._mode = None


class AnnotationCanvas(QGraphicsView):
    """Shows the frame at its true shape. Drag on the picture draws the bird's box.

    The picture may be shown smaller than the video (fast 4K playback), but the
    scene is in original video pixels, so boxes are stored at full resolution.
    """

    box_drawn = Signal(QRectF)
    crop_drawn = Signal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QBrush(QColor("#151617")))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._pix = QGraphicsPixmapItem()
        self._pix.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self._pix)
        self._box: Optional[BoxItem] = None
        self._crop_item: Optional[QGraphicsRectItem] = None
        self._rubber: Optional[QGraphicsRectItem] = None
        self._origin = QPointF()
        self._message: Optional[QGraphicsSimpleTextItem] = None
        self.mode = "label"
        self._fit = True

    # -------------------------------------------------------------- content
    def set_video_size(self, w: int, h: int) -> None:
        if self.sceneRect() != QRectF(0, 0, w, h):
            self.setSceneRect(QRectF(0, 0, w, h))
            self.fit()

    def show_frame(self, rgb: np.ndarray, video_w: int) -> None:
        h, w = rgb.shape[:2]
        rgb = np.ascontiguousarray(rgb)
        img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._pix.setPixmap(QPixmap.fromImage(img))
        self._pix.setScale(video_w / w)
        self.show_message(None)

    def clear(self) -> None:
        self._pix.setPixmap(QPixmap())
        self.set_box(None)
        self.set_crop(None)

    def show_message(self, text: Optional[str]) -> None:
        if self._message is not None:
            self.scene().removeItem(self._message)
            self._message = None
        if text:
            self._message = self.scene().addSimpleText(text)
            f = QFont()
            f.setPointSize(16)
            f.setBold(True)
            self._message.setFont(f)
            self._message.setBrush(QBrush(QColor("#f0f0f0")))
            self._message.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            self._message.setPos(self.sceneRect().topLeft() + QPointF(20, 20))
            self._message.setZValue(30)

    def set_box(self, box: Optional[BoxItem]) -> None:
        if self._box is not None:
            self.scene().removeItem(self._box)
        self._box = box
        if box is not None:
            self.scene().addItem(box)

    def set_crop(self, rect: Optional[QRectF]) -> None:
        if self._crop_item is not None:
            self.scene().removeItem(self._crop_item)
            self._crop_item = None
        if rect is not None:
            pen = QPen(QColor("#e0b030"), 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._crop_item = self.scene().addRect(rect, pen)
            self._crop_item.setZValue(5)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.viewport().setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def fit(self) -> None:
        self._fit = True
        if not self.sceneRect().isEmpty():
            self.resetTransform()
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # --------------------------------------------------------------- events
    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit:
            self.fit()

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        if f < 1 and self.transform().m11() * f < self._fit_scale():
            self.fit()
            return
        self._fit = False
        self.scale(f, f)

    def _fit_scale(self) -> float:
        r = self.sceneRect()
        if r.isEmpty():
            return 1.0
        return min(self.viewport().width() / r.width(), self.viewport().height() / r.height())

    def _over_box(self, pos) -> bool:
        item = self.itemAt(pos)
        while item is not None and not isinstance(item, BoxItem):
            item = item.parentItem()
        return isinstance(item, BoxItem) and self.mode == "label"

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and not self.sceneRect().isEmpty() \
                and not self._over_box(e.position().toPoint()):
            self._origin = self.mapToScene(e.position().toPoint())
            pen = QPen(QColor("#e0b030") if self.mode == "crop" else QColor("#ffe24a"), 1.5, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._rubber = self.scene().addRect(QRectF(self._origin, self._origin), pen)
            self._rubber.setZValue(20)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._rubber is not None:
            p = self.mapToScene(e.position().toPoint())
            self._rubber.setRect(QRectF(self._origin, p).normalized().intersected(self.sceneRect()))
            return
        if not self._over_box(e.position().toPoint()):
            self.viewport().setCursor(QCursor(Qt.CursorShape.CrossCursor))
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._rubber is not None:
            rect = self._rubber.rect()
            self.scene().removeItem(self._rubber)
            self._rubber = None
            if rect.width() > 6 and rect.height() > 6:
                (self.crop_drawn if self.mode == "crop" else self.box_drawn).emit(rect)
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if not self._over_box(e.position().toPoint()):
            self.fit()
        else:
            super().mouseDoubleClickEvent(e)
