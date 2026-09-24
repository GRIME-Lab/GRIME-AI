#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# RoiEditorLabel.py
#
# Image label with a single editable rectangular region of interest.
#   - Drag outside the ROI (or on an image with no ROI) to draw a new one;
#     the previous ROI is replaced.
#   - Drag inside the ROI to move it.
#   - Drag an edge or corner handle to resize it.
#   - The cursor shows the action available under the mouse.
#   - Esc cancels the draw, move or resize in progress.
# The ROI is kept inside the image and reported in normalised image
# coordinates [x, y, w, h] (0.0-1.0), independent of the display size.

from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QRect, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap

HANDLE_SIZE    = 8      # px, side of the square resize handles
GRAB_MARGIN    = 6      # px, how close to an edge counts as grabbing it
MIN_ROI_PIXELS = 10     # px on screen, smallest ROI that is kept
ROI_COLOR      = QColor(255, 0, 0)

_CURSORS = {
    "move": Qt.SizeAllCursor,
    "l": Qt.SizeHorCursor,  "r": Qt.SizeHorCursor,
    "t": Qt.SizeVerCursor,  "b": Qt.SizeVerCursor,
    "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
    "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
    "draw": Qt.CrossCursor,
}


class RoiEditorLabel(QLabel):

    roiChanged = pyqtSignal(object)     # normalised [x, y, w, h], or None when cleared

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)     # needed for Esc
        self.setAlignment(Qt.AlignCenter)
        self._source = None         # full-resolution QPixmap
        self._roi = None            # QRectF in normalised image coordinates
        self._action = None         # "draw", "move" or a handle name while dragging
        self._press_pos = None      # QPointF, normalised
        self._roi_at_press = None   # QRectF, restored on Esc

    # ── Public API ────────────────────────────────────────────────────────────
    def setImage(self, pixmap: QPixmap):
        self._source = pixmap if pixmap is not None and not pixmap.isNull() else None
        self._rescale()

    def setNormalizedRoi(self, roi):
        self._roi = QRectF(*roi) if roi else None
        self._action = None
        self.update()

    def normalizedRoi(self):
        if self._roi is None:
            return None
        r = self._roi
        return [r.x(), r.y(), r.width(), r.height()]

    def clearRoi(self):
        self.setNormalizedRoi(None)
        self.roiChanged.emit(None)

    # ── Geometry ──────────────────────────────────────────────────────────────
    def _rescale(self):
        if self._source is None:
            super().setPixmap(QPixmap())
            return
        super().setPixmap(self._source.scaled(self.width(), self.height(),
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _image_rect(self) -> QRect:
        """Where the scaled pixmap is drawn inside the label."""
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return QRect()
        return QRect((self.width() - pm.width()) // 2, (self.height() - pm.height()) // 2,
                     pm.width(), pm.height())

    def _to_norm(self, pos) -> QPointF:
        ir = self._image_rect()
        x = min(max((pos.x() - ir.x()) / ir.width(), 0.0), 1.0)
        y = min(max((pos.y() - ir.y()) / ir.height(), 0.0), 1.0)
        return QPointF(x, y)

    def _to_screen(self, r: QRectF) -> QRectF:
        ir = self._image_rect()
        return QRectF(ir.x() + r.x() * ir.width(), ir.y() + r.y() * ir.height(),
                      r.width() * ir.width(), r.height() * ir.height())

    def _hit_test(self, pos):
        """Action available at pos: a handle name, "move", or "draw"."""
        if self._roi is None:
            return "draw"
        s = self._to_screen(self._roi)
        x, y, m = pos.x(), pos.y(), GRAB_MARGIN
        near_l, near_r = abs(x - s.left()) <= m, abs(x - s.right()) <= m
        near_t, near_b = abs(y - s.top()) <= m, abs(y - s.bottom()) <= m
        in_x = s.left() - m <= x <= s.right() + m
        in_y = s.top() - m <= y <= s.bottom() + m
        if near_t and near_l: return "tl"
        if near_t and near_r: return "tr"
        if near_b and near_l: return "bl"
        if near_b and near_r: return "br"
        if near_l and in_y:   return "l"
        if near_r and in_y:   return "r"
        if near_t and in_x:   return "t"
        if near_b and in_x:   return "b"
        if s.contains(QPointF(x, y)): return "move"
        return "draw"

    def _min_size(self):
        ir = self._image_rect()
        return MIN_ROI_PIXELS / max(ir.width(), 1), MIN_ROI_PIXELS / max(ir.height(), 1)

    # ── Mouse and keyboard ────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self._source is None \
                or not self._image_rect().contains(event.pos()):
            return super().mousePressEvent(event)
        self.setFocus()
        self._action = self._hit_test(event.pos())
        self._press_pos = self._to_norm(event.pos())
        self._roi_at_press = QRectF(self._roi) if self._roi is not None else None
        if self._action == "draw":
            self._roi = QRectF(self._press_pos, self._press_pos)   # replaces any previous ROI
        self.update()

    def mouseMoveEvent(self, event):
        if self._action is None:
            if self._source is not None and self._image_rect().contains(event.pos()):
                self.setCursor(_CURSORS[self._hit_test(event.pos())])
            else:
                self.unsetCursor()
            return
        p = self._to_norm(event.pos())
        if self._action == "draw":
            self._roi = QRectF(self._press_pos, p).normalized()
        elif self._action == "move":
            r = QRectF(self._roi_at_press)
            dx = min(max(p.x() - self._press_pos.x(), -r.left()), 1.0 - r.right())
            dy = min(max(p.y() - self._press_pos.y(), -r.top()), 1.0 - r.bottom())
            r.translate(dx, dy)
            self._roi = r
        else:
            self._roi = self._resized(self._roi_at_press, self._action, p)
        self.update()

    def _resized(self, r0: QRectF, handle: str, p: QPointF) -> QRectF:
        min_w, min_h = self._min_size()
        left, top, right, bottom = r0.left(), r0.top(), r0.right(), r0.bottom()
        if "l" in handle: left   = min(p.x(), right - min_w)
        if "r" in handle: right  = max(p.x(), left + min_w)
        if "t" in handle: top    = min(p.y(), bottom - min_h)
        if "b" in handle: bottom = max(p.y(), top + min_h)
        return QRectF(QPointF(max(left, 0.0), max(top, 0.0)),
                      QPointF(min(right, 1.0), min(bottom, 1.0)))

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self._action is None:
            return super().mouseReleaseEvent(event)
        action, self._action = self._action, None
        min_w, min_h = self._min_size()
        if action == "draw" and (self._roi.width() < min_w or self._roi.height() < min_h):
            self._roi = self._roi_at_press      # a click, not a drag: keep the previous ROI
            self.update()
            return
        self.update()
        self.roiChanged.emit(self.normalizedRoi())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self._action is not None:
            self._action = None
            self._roi = self._roi_at_press
            self.update()
            return      # consume Esc so it does not also close the dialog
        super().keyPressEvent(event)

    # ── Painting ──────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        super().paintEvent(event)
        if self._roi is None or self._source is None:
            return
        s = self._to_screen(self._roi)
        painter = QPainter(self)
        painter.setPen(QPen(ROI_COLOR, 2))
        painter.drawRect(s)
        if self._action is None:
            painter.setBrush(ROI_COLOR)
            half = HANDLE_SIZE / 2
            xs = (s.left(), s.center().x(), s.right())
            ys = (s.top(), s.center().y(), s.bottom())
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    if i == 1 and j == 1:
                        continue
                    painter.drawRect(QRectF(x - half, y - half, HANDLE_SIZE, HANDLE_SIZE))
        painter.end()
