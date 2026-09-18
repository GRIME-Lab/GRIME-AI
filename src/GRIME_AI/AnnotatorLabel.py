#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

def simplify_path(path, stride=5):
    if len(path) < 3:
        return []
    return path[::stride]


class AnnotatorLabel(QtWidgets.QLabel):
    """
    Draws onto a fixed-resolution base pixmap and
    always scales that pixmap to fit the current label size.
    """

    def __init__(self, pixmap, mode='drag', shape_label=None, parent=None):
        super().__init__(parent)

        # 1) keep the full-res base pixmap
        self._base = pixmap.copy()

        # 2) drawing state
        self.mode = mode
        self.shapes = []         # finalized shapes + their labels
        self.annotations = []    # raw point lists
        self.drawing = False

        # points for click-mode
        self.active_pts = []

        # path for drag-mode
        self.active_path = []

        # bbox temp state
        self.active_bbox_start = None
        self.active_bbox_end   = None

        # the label (id+name) to assign to every new shape
        self._shape_label = shape_label or {"id": -1, "name": "unknown"}

        # allow the label to stretch/shrink
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored
        )
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)

        # initial render
        self._render_scaled()

    def resizeEvent(self, ev):
        # rerender whenever the widget changes size
        self._render_scaled()
        super().resizeEvent(ev)

    def _render_scaled(self, include_temp=False):
        canvas = self._base.copy()
        painter = QtGui.QPainter(canvas)

        # draw finalized shapes (green)
        pen = QtGui.QPen(Qt.green, 2)
        pen.setCosmetic(True)
        painter.setPen(pen)

        for s in self.shapes:
            pts = s['points']
            if s['type'] == 'bbox':
                xs = [p.x() for p in pts]
                ys = [p.y() for p in pts]
                rect = QtCore.QRect(min(xs), min(ys),
                                    max(xs) - min(xs),
                                    max(ys) - min(ys))
                painter.drawRect(rect)
            elif s['type'] == 'click':
                for i, p in enumerate(pts):
                    painter.drawEllipse(p, 2, 2)
                    if i > 0:
                        painter.drawLine(pts[i-1], p)
                if len(pts) > 2:
                    painter.drawLine(pts[-1], pts[0])
            elif s['type'] == 'drag':
                if len(pts) > 2:
                    path = QtGui.QPainterPath()
                    path.moveTo(pts[0])
                    for pt in pts[1:]:
                        path.lineTo(pt)
                    path.closeSubpath()
                    painter.setBrush(QtGui.QColor(0, 255, 0, 80))  # semi-transparent fill
                    painter.drawPath(path)
                else:
                    for i in range(len(pts)):
                        painter.drawLine(pts[i], pts[(i + 1) % len(pts)])


        # in-progress click points (red)
        if self.active_pts:
            pen.setColor(Qt.red)
            painter.setPen(pen)
            for i, p in enumerate(self.active_pts):
                painter.drawEllipse(p, 2, 2)
                if i > 0:
                    painter.drawLine(self.active_pts[i-1], p)

        # in-progress drag path (green dots)
        if include_temp and self.active_path:
            pen.setColor(Qt.green)
            painter.setPen(pen)
            for p in self.active_path:
                painter.drawPoint(p)

        # in-progress bbox preview (red dashed)
        if include_temp and self.mode == 'bbox' \
           and self.active_bbox_start and self.active_bbox_end:
            dash_pen = QtGui.QPen(Qt.red, 1, Qt.DashLine)
            dash_pen.setCosmetic(True)
            painter.setPen(dash_pen)
            x0, y0 = self.active_bbox_start.x(), self.active_bbox_start.y()
            x1, y1 = self.active_bbox_end.x(),   self.active_bbox_end.y()
            rect = QtCore.QRect(QtCore.QPoint(x0, y0),
                                QtCore.QPoint(x1, y1))
            painter.drawRect(rect)

        painter.end()

        # scale to fit widget
        scaled = canvas.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def _map_to_base(self, pos):
        """
        Map a mouse pos in widget coords back
        to a QPoint in the base pixmap’s coords.
        """
        pm = self.pixmap()
        if pm is None:
            return None

        bw, bh = self._base.width(), self._base.height()
        pw, ph = pm.width(), pm.height()

        sx = pw / bw
        sy = ph / bh

        dx = (self.width()  - pw) // 2
        dy = (self.height() - ph) // 2

        x = (pos.x() - dx) / sx
        y = (pos.y() - dy) / sy

        x = max(0, min(x, bw - 1))
        y = max(0, min(y, bh - 1))
        return QtCore.QPoint(int(x), int(y))

    def mousePressEvent(self, ev):
        bp = self._map_to_base(ev.pos())
        if not bp:
            return super().mousePressEvent(ev)

        if ev.button() == Qt.LeftButton:
            if self.mode == 'bbox':
                self.drawing = True
                self.active_bbox_start = bp
                self.active_bbox_end   = bp
            elif self.mode == 'click':
                self.active_pts.append(bp)
            else:  # drag
                self.drawing = True
                self.active_path = [bp]

        elif ev.button() == Qt.RightButton and self.mode == 'click':
            pts = list(self.active_pts)
            self.active_pts.clear()
            self._finalize('click', pts)

        self._render_scaled(include_temp=True)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.drawing:
            bp = self._map_to_base(ev.pos())
            if not bp:
                return
            if self.mode == 'bbox':
                self.active_bbox_end = bp
            elif self.mode == 'drag':
                self.active_path.append(bp)
            self._render_scaled(include_temp=True)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.drawing:
            if self.mode == 'bbox':
                x0, y0 = self.active_bbox_start.x(), self.active_bbox_start.y()
                x1, y1 = self.active_bbox_end.x(),   self.active_bbox_end.y()
                corners = [
                    QtCore.QPoint(x0, y0),
                    QtCore.QPoint(x1, y0),
                    QtCore.QPoint(x1, y1),
                    QtCore.QPoint(x0, y1),
                ]
                self.drawing = False
                self.active_bbox_start = None
                self.active_bbox_end   = None
                self._finalize('bbox', corners)

            elif self.mode == 'drag':
                self.drawing = False
                raw = [(p.x(), p.y()) for p in self.active_path]
                simp = simplify_path(raw, stride=5)
                pts = [QtCore.QPoint(x, y) for x, y in simp]
                self.active_path.clear()
                self._finalize('drag', pts)

        super().mouseReleaseEvent(ev)

    def _finalize(self, kind, pts):
        if len(pts) < 3:
            return

        # record shape + label
        entry = {
            "type": kind,
            "points": pts,
            "label": self._shape_label
        }
        self.shapes.append(entry)
        self.annotations.append([(p.x(), p.y()) for p in pts])

        # persist on base canvas
        painter = QtGui.QPainter(self._base)
        pen = QtGui.QPen(Qt.green, 1)
        pen.setCosmetic(True)
        painter.setPen(pen)

        if kind == "click":
            for i, p in enumerate(pts):
                painter.drawEllipse(p, 1, 1)
                if i > 0:
                    painter.drawLine(pts[i-1], p)
            painter.drawLine(pts[-1], pts[0])
        else:
            for i in range(len(pts)):
                painter.drawLine(pts[i], pts[(i+1) % len(pts)])

        painter.end()
        self._render_scaled()
