#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Nov 9, 2026
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

"""
PointerSlider: a QSlider drawn as a track with a pointer, used for the stacked
Too bright / Too dark brightness limits in the Image Triage Options dialog.

The two sliders sit one above the other on the same 0-255 scale:

    Too bright   pointer above its track, pointing down; the track is colored
                 from the pointer to the maximum (the poor-quality range).
    Too dark     pointer below its track, pointing up; the track is colored
                 from the minimum to the pointer (the poor-quality range).

The rest of each track is white, so the white stretch between the two pointers
is the range of brightness that is kept.

Set the dynamic property "limitKind" to "too_bright" or "too_dark" (in the .ui
file or with setProperty). Everything else is a normal QSlider: setValue,
setRange, valueChanged and keyboard control work as usual. Clicking or dragging
anywhere on the slider moves the pointer to that position.
"""

from PyQt5.QtWidgets import QSlider, QSizePolicy
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize
from PyQt5.QtGui import QPainter, QColor, QPolygonF, QPen, QBrush

# Pointer and track sizes, in pixels
POINTER_WIDTH  = 14
POINTER_HEIGHT = 20
POINTER_BODY   = 13     # height of the rectangular part; the rest is the tip
TRACK_HEIGHT   = 8
TRACK_GAP      = 2      # space between the pointer tip and the track
EDGE_MARGIN    = 1

# Colors per limit: pointer fill, outline, poor-quality track fill.
# Orange and dark blue differ in lightness as well as hue, so they can be told
# apart with common forms of color blindness.
LIMIT_STYLES = {
    "too_bright": {"pointer": "#E07B00", "outline": "#7A4300", "fill": "#F2A65A",
                   "direction": "down", "fill_side": "above"},
    "too_dark":   {"pointer": "#2B5D8A", "outline": "#1D3F5E", "fill": "#6E98C2",
                   "direction": "up",   "fill_side": "below"},
}
DISABLED_STYLE = {"pointer": "#B4B4B4", "outline": "#8A8A8A", "fill": "#D2D2D2"}
TRACK_BACKGROUND = "#FFFFFF"
FOCUS_COLOR      = "#000000"


class PointerSlider(QSlider):

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    # ------------------------------------------------------------------------------------------------------------------
    def _style(self):
        kind = self.property("limitKind") or "too_bright"
        style = dict(LIMIT_STYLES.get(kind, LIMIT_STYLES["too_bright"]))
        if not self.isEnabled():
            style.update(DISABLED_STYLE)
        return style

    def _widget_height(self):
        return EDGE_MARGIN * 2 + POINTER_HEIGHT + TRACK_GAP + TRACK_HEIGHT

    def sizeHint(self):
        return QSize(200, self._widget_height())

    def minimumSizeHint(self):
        return QSize(POINTER_WIDTH * 4, self._widget_height())

    # ------------------------------------------------------------------------------------------------------------------
    # Value <-> position. The pointer's center runs from half a pointer width in
    # from the left edge to half a pointer width in from the right edge.
    # ------------------------------------------------------------------------------------------------------------------
    def _usable_width(self):
        return max(1.0, self.width() - POINTER_WIDTH)

    def _x_for_value(self, value):
        span = self.maximum() - self.minimum()
        fraction = 0.0 if span == 0 else (value - self.minimum()) / span
        return POINTER_WIDTH / 2.0 + fraction * self._usable_width()

    def _value_for_x(self, x):
        fraction = (x - POINTER_WIDTH / 2.0) / self._usable_width()
        fraction = max(0.0, min(1.0, fraction))
        return self.minimum() + int(round(fraction * (self.maximum() - self.minimum())))

    # ------------------------------------------------------------------------------------------------------------------
    # Mouse: click or drag anywhere to move the pointer there
    # ------------------------------------------------------------------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or not self.isEnabled():
            return super().mousePressEvent(event)
        self.setFocus(Qt.MouseFocusReason)
        self.setSliderDown(True)
        self.setValue(self._value_for_x(event.x()))
        event.accept()

    def mouseMoveEvent(self, event):
        if self.isSliderDown():
            self.setValue(self._value_for_x(event.x()))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.isSliderDown():
            self.setSliderDown(False)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------------------------------------------------------
    def paintEvent(self, event):
        style   = self._style()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pointer_down = style["direction"] == "down"
        if pointer_down:
            pointer_top = EDGE_MARGIN
            track_top   = EDGE_MARGIN + POINTER_HEIGHT + TRACK_GAP
        else:
            track_top   = EDGE_MARGIN
            pointer_top = EDGE_MARGIN + TRACK_HEIGHT + TRACK_GAP

        left    = POINTER_WIDTH / 2.0
        right   = self.width() - POINTER_WIDTH / 2.0
        x_value = self._x_for_value(self.value())
        radius  = TRACK_HEIGHT / 2.0
        outline = QPen(QColor(style["outline"]), 1)

        # Track: white, then the poor-quality part in the limit's color
        track = QRectF(left, track_top, right - left, TRACK_HEIGHT)
        painter.setPen(outline)
        painter.setBrush(QBrush(QColor(TRACK_BACKGROUND)))
        painter.drawRoundedRect(track, radius, radius)

        if style["fill_side"] == "above":
            fill = QRectF(x_value, track_top, right - x_value, TRACK_HEIGHT)
        else:
            fill = QRectF(left, track_top, x_value - left, TRACK_HEIGHT)
        if fill.width() > 0:
            painter.setBrush(QBrush(QColor(style["fill"])))
            painter.drawRoundedRect(fill, radius, radius)

        # Pointer
        half = POINTER_WIDTH / 2.0
        if pointer_down:
            top = pointer_top
            points = [QPointF(x_value - half, top), QPointF(x_value + half, top),
                      QPointF(x_value + half, top + POINTER_BODY),
                      QPointF(x_value, top + POINTER_HEIGHT),
                      QPointF(x_value - half, top + POINTER_BODY)]
        else:
            bottom = pointer_top + POINTER_HEIGHT
            points = [QPointF(x_value, pointer_top),
                      QPointF(x_value + half, bottom - POINTER_BODY),
                      QPointF(x_value + half, bottom), QPointF(x_value - half, bottom),
                      QPointF(x_value - half, bottom - POINTER_BODY)]
        painter.setPen(QPen(QColor(style["outline"]), 1.2))
        painter.setBrush(QBrush(QColor(style["pointer"])))
        painter.drawPolygon(QPolygonF(points))

        # Keyboard focus: dotted box around the pointer
        if self.hasFocus():
            painter.setPen(QPen(QColor(FOCUS_COLOR), 1, Qt.DotLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(x_value - half - 2, pointer_top - 1,
                                    POINTER_WIDTH + 4, POINTER_HEIGHT + 2))
        painter.end()
