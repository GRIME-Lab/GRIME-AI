#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# downloader_viewers.py
#
# Viewers for the three kinds of data the Aerial and Elevation Downloader
# fetches. Each is a tab: NAIP imagery, the 1 m elevation model, and lidar
# point clouds.
#
# The views are chosen for braided sand-bed rivers, where the questions are
# where the water is, where the bars are, and how high they stand:
#
#   NAIP   natural color, color infrared, NDVI and NDWI. Water is dark in the
#          near infrared and bars are bright, so NDWI separates them far more
#          cleanly than anything in the visible bands.
#   DEM    hillshade, color relief and slope, with an elevation histogram and a
#          profile along any row or column. A profile across a bar is directly
#          comparable with the inundation thresholds estimated from camera
#          imagery, which is how that estimate gets checked against survey.
#   Lidar  point count, extents, return and classification breakdown, and a
#          gridded elevation image built from the points.
#
# Readers are imported where they are used: a machine without laspy still gets
# the other two tabs.

import os
import math

import numpy as np
import cv2

from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
                             QLineEdit, QPushButton, QComboBox, QSlider, QSpinBox,
                             QDoubleSpinBox, QFileDialog, QMessageBox, QPlainTextEdit,
                             QCheckBox, QSplitter, QGraphicsView, QGraphicsScene,
                             QGraphicsPixmapItem, QSizePolicy, QDialog)

# The host's settings helper, with a local stand-in so these viewers also run
# outside the application.
try:
    from appcore.plugins import plugin_settings
except ImportError:
    plugin_settings = None

# Settings belong to the plugin, not to this module, so a viewer opened on its
# own still writes to the plugin's file rather than one of its own.
SETTINGS_OWNER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "aerial_elevation_downloader.py")

RASTER_FILTER = "Raster (*.tif *.tiff *.jp2 *.png *.jpg);;All files (*.*)"
LIDAR_FILTER = "Lidar (*.las *.laz);;All files (*.*)"
PREVIEW_LONG_EDGE = 1600        # tiles are far larger than any panel
VOID_BELOW = -9000.0            # 3DEP stores voids as a large negative value
HILLSHADE_AZIMUTH = 315.0
HILLSHADE_ALTITUDE = 45.0
LIDAR_MAX_POINTS = 4_000_000    # read a sample beyond this, to stay responsive
# The last file is reopened when a viewer is shown again, unless it is large
# enough that the wait would be a surprise; then the path is filled in and the
# user presses Reload.
AUTOLOAD_MAX_BYTES = 250 * 1024 * 1024
LIDAR_GRID_CELLS = 1200         # long edge of the gridded elevation image

# Point classes worth naming; the rest are reported by number.
LAS_CLASSES = {1: "Unclassified", 2: "Ground", 3: "Low vegetation",
               4: "Medium vegetation", 5: "High vegetation", 6: "Building",
               7: "Low point", 9: "Water", 11: "Road surface"}


# ======================================================================================================================
# Shared helpers
# ======================================================================================================================
def to_pixmap(display) -> QPixmap:
    """BGR or grayscale uint8 array to a QPixmap."""
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    height, width = display.shape[:2]
    rgb = np.ascontiguousarray(display[:, :, ::-1])
    return QPixmap.fromImage(QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888))


def stretch(values, low_percentile=2.0, high_percentile=98.0) -> np.ndarray:
    """Scale to 0-255 on percentiles, so one bright roof does not flatten the rest."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros(values.shape, np.uint8)
    low, high = np.percentile(finite, [low_percentile, high_percentile])
    if high - low < 1e-9:
        high = low + 1.0
    return np.clip((values - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def downscale(image, long_edge=PREVIEW_LONG_EDGE):
    scale = long_edge / max(image.shape[0], image.shape[1])
    if scale < 1.0:
        interpolation = cv2.INTER_AREA
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
    return image


def plot_profile(values, title, x_label, y_label, width=900, height=320):
    """A line plot as a BGR image, drawn without a plotting library."""
    image = np.full((height, width, 3), 255, np.uint8)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        cv2.putText(image, "No data along this line", (20, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
        return image

    margin_left, margin_right, margin_top, margin_bottom = 70, 20, 30, 40
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    low, high = float(finite.min()), float(finite.max())
    span = max(high - low, 1e-6)

    cv2.rectangle(image, (margin_left, margin_top),
                  (margin_left + plot_w, margin_top + plot_h), (200, 200, 200), 1)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(margin_top + plot_h * fraction)
        cv2.line(image, (margin_left, y), (margin_left + plot_w, y), (235, 235, 235), 1)
        label = f"{high - span * fraction:,.2f}"
        cv2.putText(image, label, (5, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (80, 80, 80), 1, cv2.LINE_AA)

    xs = np.linspace(margin_left, margin_left + plot_w, len(values)).astype(np.int32)
    ys = margin_top + plot_h - ((values - low) / span * plot_h)
    points = [(int(x), int(y)) for x, y in zip(xs, ys) if np.isfinite(y)]
    for first, second in zip(points, points[1:]):
        cv2.line(image, first, second, (180, 90, 30), 2, cv2.LINE_AA)

    cv2.putText(image, title, (margin_left, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(image, x_label, (width // 2 - 40, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(image, y_label, (5, margin_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (80, 80, 80), 1, cv2.LINE_AA)
    return image


def plot_histogram(values, title, bins=60, width=900, height=260):
    """A histogram as a BGR image."""
    image = np.full((height, width, 3), 255, np.uint8)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return image

    counts, edges = np.histogram(finite, bins=bins)
    margin_left, margin_bottom, margin_top = 60, 34, 26
    plot_w = width - margin_left - 20
    plot_h = height - margin_top - margin_bottom
    peak = max(counts.max(), 1)

    for index, count in enumerate(counts):
        x0 = int(margin_left + plot_w * index / bins)
        x1 = int(margin_left + plot_w * (index + 1) / bins) - 1
        bar = int(plot_h * count / peak)
        cv2.rectangle(image, (x0, margin_top + plot_h - bar), (max(x1, x0 + 1), margin_top + plot_h),
                      (180, 120, 60), -1)

    cv2.rectangle(image, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h),
                  (200, 200, 200), 1)
    cv2.putText(image, title, (margin_left, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (30, 30, 30), 1, cv2.LINE_AA)
    for fraction in (0.0, 0.5, 1.0):
        x = int(margin_left + plot_w * fraction)
        value = edges[0] + (edges[-1] - edges[0]) * fraction
        cv2.putText(image, f"{value:,.2f}", (x - 25, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(image, f"{peak:,}", (5, margin_top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (80, 80, 80), 1, cv2.LINE_AA)
    return image


MODE_PAN = "pan"
MODE_SELECT = "select"
MODE_DRAW = "draw"


def _mode_icon(kind, size=22):
    """
    The pan, select and draw icons, drawn rather than loaded, so the plugin
    carries no image files.
    """
    from PyQt5.QtGui import QIcon, QPixmap, QPen, QBrush, QPolygonF, QColor
    from PyQt5.QtCore import QPointF

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    ink = QColor(70, 70, 70)
    painter.setPen(QPen(ink, 1.4))
    painter.setBrush(QBrush(ink))
    unit = size / 22.0

    if kind == MODE_SELECT:
        arrow = QPolygonF([QPointF(6 * unit, 3 * unit), QPointF(6 * unit, 17 * unit),
                           QPointF(10 * unit, 13 * unit), QPointF(12.5 * unit, 18.5 * unit),
                           QPointF(15 * unit, 17.5 * unit), QPointF(12.5 * unit, 12 * unit),
                           QPointF(17 * unit, 11.5 * unit)])
        painter.drawPolygon(arrow)
    elif kind == MODE_DRAW:
        painter.drawLine(QPointF(4 * unit, 18 * unit), QPointF(6 * unit, 14 * unit))
        pencil = QPolygonF([QPointF(6 * unit, 14 * unit), QPointF(14 * unit, 5 * unit),
                            QPointF(17 * unit, 8 * unit), QPointF(9 * unit, 16 * unit)])
        painter.drawPolygon(pencil)
        painter.drawLine(QPointF(4 * unit, 18 * unit), QPointF(9 * unit, 16 * unit))
    else:
        painter.setBrush(Qt.NoBrush)
        # A hand: palm with four fingers and a thumb.
        painter.drawRoundedRect(QRectF(7 * unit, 9 * unit, 9 * unit, 9 * unit),
                                2 * unit, 2 * unit)
        for offset in range(4):
            x = (8 + offset * 2.2) * unit
            painter.drawLine(QPointF(x, 9 * unit), QPointF(x, (4.5 + offset % 2) * unit))
        painter.drawLine(QPointF(7 * unit, 12 * unit), QPointF(4 * unit, 9 * unit))
    painter.end()
    return QIcon(pixmap)


class ModeBar(QWidget):
    """
    Draw, Select and Pan, the same three tools SAGE uses, so the gesture means
    the same thing everywhere in the application.
    """

    def __init__(self, view, modes=(MODE_DRAW, MODE_SELECT, MODE_PAN), parent=None):
        super().__init__(parent)
        from PyQt5.QtWidgets import QButtonGroup, QToolButton
        self._view = view
        self._buttons = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        group = QButtonGroup(self)
        group.setExclusive(True)
        for mode in modes:
            button = QToolButton()
            button.setCheckable(True)
            button.setIcon(_mode_icon(mode))
            button.setText(mode.title())
            button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            button.setToolTip({MODE_DRAW: "Drag to mark a region.",
                               MODE_SELECT: "Click a unit to select it, double click to "
                                            "look inside it.",
                               MODE_PAN: "Drag to move the view."}[mode])
            button.clicked.connect(lambda _checked=False, chosen=mode: self.set_mode(chosen))
            group.addButton(button)
            layout.addWidget(button)
            self._buttons[mode] = button

        first = modes[-1] if MODE_PAN in modes else modes[0]
        self.set_mode(MODE_PAN if MODE_PAN in modes else first)

    def set_mode(self, mode):
        self._view.set_mode(mode)
        if mode in self._buttons:
            self._buttons[mode].setChecked(True)

    def mode(self):
        return self._view.mode()


class ZoomableView(QGraphicsView):
    """
    Pans and zooms one image. The wheel zooms about the cursor, dragging pans,
    and Fit returns to the whole tile, so a river can be filled to the panel
    without cropping the file or reloading it.
    """

    view_changed = pyqtSignal()      # zoomed, panned or resized
    roi_changed = pyqtSignal(object)  # (left, top, right, bottom) in image pixels, or None
    cursor_moved = pyqtSignal(object)   # (row, column) under the pointer, or None
    clicked = pyqtSignal(object)        # (row, column) clicked in Select mode
    double_clicked = pyqtSignal(object)  # (row, column) double clicked in Select mode

    ZOOM_STEP = 1.25
    MIN_SCALE = 0.02
    MAX_SCALE = 60.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        # Panning is done here rather than with ScrollHandDrag, because that mode
        # forces an open-hand cursor and a hand is useless for pointing at a cell.
        self.setDragMode(QGraphicsView.NoDrag)
        self.setCursor(Qt.CrossCursor)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.black)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(420, 320)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._fitted = False
        self._mode = MODE_PAN
        self._roi_item = None
        self._roi_origin = None
        self._pan_from = None
        for bar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            bar.valueChanged.connect(self.view_changed)

    # ------------------------------------------------------------------------------------------------------------------
    def set_pixmap(self, pixmap: QPixmap, keep_view=False):
        """Show a pixmap. keep_view holds the current zoom and centre, which is
        what a change of display mode wants: the same patch, drawn differently."""
        had_image = not self._item.pixmap().isNull()
        same_size = had_image and self._item.pixmap().size() == pixmap.size()
        centre = self.mapToScene(self.viewport().rect().center())
        transform = self.transform()

        self._item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

        if keep_view and same_size:
            self.setTransform(transform)
            self.centerOn(centre)
        else:
            self.fit()
        self.view_changed.emit()

    def set_message(self, text):
        self._item.setPixmap(QPixmap())
        self._scene.setSceneRect(QRectF(0, 0, 1, 1))
        self.setToolTip(text)

    def fit(self):
        if self._item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self._item, Qt.KeepAspectRatio)
        self._fitted = True
        self.view_changed.emit()

    def zoom(self, factor):
        scale = self.transform().m11() * factor
        if self.MIN_SCALE <= scale <= self.MAX_SCALE:
            self.scale(factor, factor)
            self._fitted = False
            self.view_changed.emit()

    def zoom_percent(self) -> float:
        return self.transform().m11() * 100.0

    def visible_image_rect(self):
        """
        The part of the image on screen, as integer pixel bounds
        (left, top, right, bottom), clipped to the image. Plots use this so
        they describe what is actually in view rather than the whole tile.
        """
        pixmap = self._item.pixmap()
        if pixmap.isNull():
            return None
        scene_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        left = max(int(math.floor(scene_rect.left())), 0)
        top = max(int(math.floor(scene_rect.top())), 0)
        right = min(int(math.ceil(scene_rect.right())), pixmap.width())
        bottom = min(int(math.ceil(scene_rect.bottom())), pixmap.height())
        if right - left < 2 or bottom - top < 2:
            return None
        return left, top, right, bottom

    # ------------------------------------------------------------------------------------------------------------------
    # ── Region of interest ────────────────────────────────────────────────
    def set_mode(self, mode):
        """Pan, Select or Draw. The cursor follows the tool."""
        self._mode = mode
        self.setCursor({MODE_PAN: Qt.OpenHandCursor,
                        MODE_SELECT: Qt.ArrowCursor,
                        MODE_DRAW: Qt.CrossCursor}.get(mode, Qt.ArrowCursor))

    def mode(self):
        return self._mode

    def roi(self):
        """The region as (left, top, right, bottom) in image pixels, or None."""
        if self._roi_item is None:
            return None
        rect = self._roi_item.rect()
        pixmap = self._item.pixmap()
        left = max(int(rect.left()), 0)
        top = max(int(rect.top()), 0)
        right = min(int(rect.right()), pixmap.width())
        bottom = min(int(rect.bottom()), pixmap.height())
        if right - left < 2 or bottom - top < 2:
            return None
        return left, top, right, bottom

    def set_roi(self, bounds):
        """Show a region, or clear it with None."""
        from PyQt5.QtWidgets import QGraphicsRectItem
        from PyQt5.QtGui import QPen, QColor
        if self._roi_item is not None:
            self._scene.removeItem(self._roi_item)
            self._roi_item = None
        if bounds:
            left, top, right, bottom = bounds
            self._roi_item = QGraphicsRectItem(QRectF(left, top, right - left, bottom - top))
            pen = QPen(QColor(255, 60, 60), 0)      # 0 keeps it one pixel at any zoom
            pen.setCosmetic(True)
            self._roi_item.setPen(pen)
            self._scene.addItem(self._roi_item)
        self.roi_changed.emit(self.roi())

    def mousePressEvent(self, event):
        if self._mode == MODE_SELECT and event.button() == Qt.LeftButton:
            self.clicked.emit(self._cell_at(event.pos()))
            return
        if self._mode == MODE_DRAW and event.button() == Qt.LeftButton:
            point = self.mapToScene(event.pos())
            self._roi_origin = point
            self.set_roi((point.x(), point.y(), point.x() + 1, point.y() + 1))
            return
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._pan_from = event.pos()
            self.setCursor(Qt.ClosedHandCursor)      # only while actually dragging
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._mode == MODE_SELECT and event.button() == Qt.LeftButton:
            self.double_clicked.emit(self._cell_at(event.pos()))
            return
        # Fit is on the toolbar; double click belongs to Select.
        super().mouseDoubleClickEvent(event)

    def _cell_at(self, position):
        """The image cell under a viewport point, or None."""
        point = self.mapToScene(position)
        pixmap = self._item.pixmap()
        if pixmap.isNull() or not (0 <= point.x() < pixmap.width()
                                   and 0 <= point.y() < pixmap.height()):
            return None
        return int(point.y()), int(point.x())

    def mouseMoveEvent(self, event):
        point = self.mapToScene(event.pos())
        pixmap = self._item.pixmap()
        if (not pixmap.isNull() and 0 <= point.x() < pixmap.width()
                and 0 <= point.y() < pixmap.height()):
            self.cursor_moved.emit((int(point.y()), int(point.x())))
        else:
            self.cursor_moved.emit(None)

        if self._mode == MODE_DRAW and self._roi_origin is not None:
            scene_point = self.mapToScene(event.pos())
            self.set_roi((min(self._roi_origin.x(), scene_point.x()),
                          min(self._roi_origin.y(), scene_point.y()),
                          max(self._roi_origin.x(), scene_point.x()),
                          max(self._roi_origin.y(), scene_point.y())))
            return
        if self._pan_from is not None:
            delta = event.pos() - self._pan_from
            self._pan_from = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._mode == MODE_DRAW and self._roi_origin is not None:
            self._roi_origin = None
            self.roi_changed.emit(self.roi())
            return
        if self._pan_from is not None:
            self._pan_from = None
            self.setCursor(Qt.OpenHandCursor)
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        self.zoom(self.ZOOM_STEP if event.angleDelta().y() > 0 else 1.0 / self.ZOOM_STEP)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fitted:
            self.fit()          # only while untouched, so a zoomed view is not reset
        else:
            self.view_changed.emit()


class _ViewerBase(QWidget):
    """A file row, an image panel, a plot panel and a log, shared by the viewers."""

    def __init__(self, file_filter, parent=None, settings=None):
        super().__init__(parent)
        self._file_filter = file_filter
        self._path = ""
        # One settings file for the whole plugin. The owner passes its own, so
        # the viewers do not write a second file named after this module.
        self._settings = settings if settings is not None else (
            plugin_settings(SETTINGS_OWNER) if plugin_settings else None)
        self._restored = False

        self._edit_path = QLineEdit()
        self._edit_path.setPlaceholderText("Open a downloaded file")
        button_browse = QPushButton("Open")
        button_browse.clicked.connect(self._browse)
        self._button_reload = QPushButton("Reload")
        self._button_reload.setToolTip("Read the file in the box again.")
        self._button_reload.clicked.connect(self._reload)

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        file_row.addWidget(self._edit_path, 1)
        file_row.addWidget(button_browse)
        file_row.addWidget(self._button_reload)

        self._image = ZoomableView()
        self._image.set_message("Nothing loaded.")
        self._plot = QLabel("")
        self._plot.setAlignment(Qt.AlignCenter)
        self._plot.setMinimumHeight(220)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(110)

        splitter = QSplitter(Qt.Vertical)
        image_box = QGroupBox("View")
        image_layout = QVBoxLayout(image_box)

        zoom_row = QHBoxLayout()
        self._label_zoom = QLabel("Fit")
        for label, handler in (("\u2212", lambda: self._zoom(1 / ZoomableView.ZOOM_STEP)),
                               ("+", lambda: self._zoom(ZoomableView.ZOOM_STEP)),
                               ("Fit", self._fit),
                               ("1:1", self._actual_size)):
            button = QPushButton(label)
            button.setFixedWidth(46)
            button.clicked.connect(handler)
            zoom_row.addWidget(button)
        zoom_row.addWidget(self._label_zoom)
        zoom_row.addStretch(1)
        zoom_row.addWidget(QLabel("Wheel zooms, drag pans, double click fits"))
        image_layout.addLayout(zoom_row)
        image_layout.addWidget(self._image, 1)
        plot_box = QGroupBox("Plot")
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.addWidget(self._plot)
        splitter.addWidget(image_box)
        splitter.addWidget(plot_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self._layout = QVBoxLayout(self)
        self._layout.addLayout(file_row)
        self._controls = QHBoxLayout()
        self._layout.addLayout(self._controls)
        self._layout.addWidget(splitter, 1)
        self._layout.addWidget(self._log)

    # ------------------------------------------------------------------------------------------------------------------
    def _bind(self, widget, key, default=None):
        """Restore a control from the settings and save it when it changes, under
        this viewer's own name so the three tabs keep separate choices."""
        if self._settings is not None and hasattr(self._settings, "bind"):
            self._settings.bind(widget, f"{type(self).__name__}.{key}", default)
        return widget

    def _remember(self, key, value):
        if self._settings is not None:
            self._settings[f"{type(self).__name__}.{key}"] = value
            self._settings.save()

    def _recall(self, key, default=None):
        if self._settings is None:
            return default
        return self._settings.get(f"{type(self).__name__}.{key}", default)

    def _browse(self):
        start = os.path.dirname(self._path) if self._path else self._recall("folder", "")
        path, _ = QFileDialog.getOpenFileName(self, "Open File", start, self._file_filter)
        if path:
            self.load(path)

    def showEvent(self, event):
        """Reopen the file from last time, the first time this tab is shown."""
        super().showEvent(event)
        if self._restored:
            return
        self._restored = True
        path = self._recall("file", "")
        if not path or not os.path.exists(path):
            return
        self._edit_path.setText(path)
        if os.path.getsize(path) <= AUTOLOAD_MAX_BYTES:
            self.load(path)
        else:
            self._log.appendPlainText(
                f"Last file was {os.path.getsize(path) / 1e6:,.0f} MB; "
                "press Reload to open it.")

    def _reload(self):
        path = self._edit_path.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.information(self, "Reload", "No file to reload.")
            return
        self.load(path)

    def load(self, path):
        self._path = path
        self._edit_path.setText(path)
        self._remember("folder", os.path.dirname(path))
        self._remember("file", path)
        self._log.clear()
        self._image.set_message("Loading\u2026")
        try:
            self._load(path)
        except Exception as err:
            import traceback
            traceback.print_exc()
            self._log.appendPlainText(f"Could not read: {type(err).__name__}: {err}")
            QMessageBox.warning(self, "Open File", f"Could not read this file:\n{err}")

    def _load(self, path):
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    def _zoom(self, factor):
        self._image.zoom(factor)
        self._update_zoom_label()

    def _fit(self):
        self._image.fit()
        self._label_zoom.setText("Fit")

    def _actual_size(self):
        self._image.resetTransform()
        self._image._fitted = False
        self._update_zoom_label()

    def _update_zoom_label(self):
        self._label_zoom.setText(f"{self._image.zoom_percent():,.0f}%")

    def _show(self, display, plot=None, keep_view=True):
        """keep_view holds the zoom and centre when only the display mode changed."""
        self._image.set_pixmap(to_pixmap(display), keep_view=keep_view)
        self._update_zoom_label()
        if plot is not None:
            self._plot.setPixmap(to_pixmap(plot).scaled(
                self._plot.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ======================================================================================================================
# NAIP
# ======================================================================================================================
class NaipViewerTab(_ViewerBase):
    """Natural color, color infrared, NDVI and NDWI from a NAIP tile."""

    MODES = ["Natural color", "Color infrared", "NDVI (vegetation)", "NDWI (water)"]

    def __init__(self, parent=None, settings=None):
        super().__init__(RASTER_FILTER, parent, settings)
        self._bands = None

        self._combo_mode = QComboBox()
        self._combo_mode.addItems(self.MODES)
        self._combo_mode.currentIndexChanged.connect(self._render)
        self._check_stretch = QCheckBox("Percentile stretch")
        self._check_stretch.setChecked(True)
        self._check_stretch.toggled.connect(self._render)
        self._check_visible_only = QCheckBox("Plot the visible area only")
        self._check_visible_only.setChecked(True)
        self._check_visible_only.setToolTip(
            "The histogram and the percentages describe the part of the tile on screen.")
        self._check_visible_only.toggled.connect(self._update_plot)
        self._controls.addWidget(QLabel("View:"))
        self._controls.addWidget(self._combo_mode)
        self._controls.addWidget(self._check_stretch)
        self._controls.addWidget(self._check_visible_only)
        self._controls.addStretch(1)
        self._image.view_changed.connect(self._update_plot)
        self._index = None          # NDVI or NDWI of the whole tile, when shown
        self._index_name = ""

        self._bind(self._combo_mode, "mode")
        self._bind(self._check_stretch, "stretch")
        self._bind(self._check_visible_only, "visible_only")

    def _load(self, path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("The file is not a readable raster. "
                             "JPEG2000 needs an OpenCV build with JP2 support.")
        # Resizing drops a trailing singleton dimension, so the band axis is
        # restored afterwards rather than before.
        image = downscale(image)
        if image.ndim == 2:
            image = image[:, :, None]
        self._bands = image.astype(np.float32)

        count = self._bands.shape[2]
        if count == 1:
            self._log.appendPlainText(
                "This file has a single band, so it is not aerial imagery. "
                "Elevation tiles belong in the Elevation tab.")
        self._log.appendPlainText(
            f"{self._bands.shape[1]} x {self._bands.shape[0]} pixels, {count} band(s)"
            + ("" if count >= 4 else "; no near infrared, so NDVI and NDWI are unavailable"))
        self._render()

    def _render(self):
        if self._bands is None:
            return
        bands = self._bands
        if bands.ndim == 2:                      # defensive: always keep a band axis
            bands = bands[:, :, None]
        count = bands.shape[2]
        mode = self._combo_mode.currentIndex()

        if mode in (2, 3) and count < 4:
            self._log.appendPlainText("This tile has no near infrared band.")
            return

        if mode in (1,) and count < 4:
            self._log.appendPlainText("Color infrared needs four bands.")
            return

        if mode == 0:
            display = bands[:, :, :3] if count >= 3 else np.repeat(bands[:, :, :1], 3, axis=2)
            display = (np.dstack([stretch(display[:, :, i]) for i in range(3)])
                       if self._check_stretch.isChecked()
                       else np.clip(display, 0, 255).astype(np.uint8))
            self._index, self._index_name = None, ""
            plot = None
        elif mode == 1:
            # NIR, red, green as blue, green, red: vegetation reads bright red.
            blue, green, red = bands[:, :, 2], bands[:, :, 0], bands[:, :, 3]
            display = np.dstack([stretch(green), stretch(red), stretch(blue)])
            self._index, self._index_name = None, ""
            plot = None
        else:
            red = bands[:, :, 2]
            green = bands[:, :, 1]
            nir = bands[:, :, 3]
            if mode == 2:
                index = (nir - red) / np.maximum(nir + red, 1e-6)
                title = "NDVI"
            else:
                index = (green - nir) / np.maximum(green + nir, 1e-6)
                title = "NDWI"
            index = np.clip(index, -1, 1)
            scaled = ((index + 1) / 2 * 255).astype(np.uint8)
            colormap = cv2.COLORMAP_SUMMER if mode == 2 else cv2.COLORMAP_OCEAN
            display = cv2.applyColorMap(scaled, colormap)
            self._index, self._index_name = index, title
            plot = None

        self._show(display)
        self._update_plot()

    def _update_plot(self):
        """Histogram of what is on screen: the index when one is shown, the
        visible bands otherwise."""
        if self._bands is None:
            return
        bands = self._bands if self._bands.ndim == 3 else self._bands[:, :, None]
        left, top = 0, 0
        bottom, right = bands.shape[0], bands.shape[1]
        visible = self._image.visible_image_rect() if self._check_visible_only.isChecked() else None
        if visible:
            left, top, right, bottom = visible
        extent = "visible area" if visible else "whole tile"

        if self._index is not None:
            patch = self._index[top:bottom, left:right]
            above = float((patch > 0).mean() * 100)
            plot = plot_histogram(patch.ravel(),
                                  f"{self._index_name} over the {extent}: above 0 is "
                                  + ("vegetated" if self._index_name == "NDVI" else "water"))
            self._log.appendPlainText(
                f"{self._index_name} over the {extent}: {above:,.1f}% above zero, "
                f"median {float(np.median(patch)):.3f}")
        else:
            patch = bands[top:bottom, left:right, :min(3, bands.shape[2])]
            plot = plot_histogram(patch.ravel(), f"Brightness over the {extent}")

        self._plot.setPixmap(to_pixmap(plot).scaled(
            self._plot.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ======================================================================================================================
# DEM
# ======================================================================================================================
class DemViewerTab(_ViewerBase):
    """Hillshade, color relief and slope, with a histogram and a profile."""

    MODES = ["Hillshade", "Color relief", "Slope"]

    def __init__(self, parent=None, settings=None):
        super().__init__(RASTER_FILTER, parent, settings)
        self._elevation = None

        self._combo_mode = QComboBox()
        self._combo_mode.addItems(self.MODES)
        self._combo_mode.currentIndexChanged.connect(self._render)
        self._combo_axis = QComboBox()
        self._combo_axis.addItems(["Profile across a row", "Profile down a column"])
        self._combo_axis.currentIndexChanged.connect(self._render)
        self._slider_line = QSlider(Qt.Horizontal)
        self._slider_line.setRange(0, 100)
        self._slider_line.setValue(50)
        self._slider_line.valueChanged.connect(self._render)

        self._controls.addWidget(QLabel("View:"))
        self._controls.addWidget(self._combo_mode)
        self._controls.addWidget(self._combo_axis)
        self._check_visible_only = QCheckBox("Plot the visible area only")
        self._check_visible_only.setChecked(True)
        self._check_visible_only.setToolTip(
            "Profile and histogram describe the part of the tile on screen, so zooming "
            "in on a reach plots that reach rather than the whole tile.")
        self._check_visible_only.toggled.connect(self._update_plot)

        self._modes = ModeBar(self._image)
        self._button_clear_roi = QPushButton("Clear Region")
        self._button_clear_roi.clicked.connect(lambda: self._image.set_roi(None))
        self._button_bars = QPushButton("Compound Bars")
        self._button_bars.setToolTip("Find the bar units in the region, or in the whole view "
                                     "when no region is drawn.")
        self._button_bars.clicked.connect(self._open_compound_bars)
        self._bars_dialog = None

        self._button_3d = QPushButton("3D Surface")
        self._button_3d.setToolTip("Open the elevation as a rotatable 3-D surface.")
        self._button_3d.clicked.connect(self._open_surface)
        self._surface_dialog = None

        self._controls.addWidget(QLabel("Position:"))
        self._controls.addWidget(self._slider_line, 1)
        self._controls.addWidget(self._check_visible_only)
        self._controls.addWidget(self._modes)
        self._controls.addWidget(self._button_clear_roi)
        self._controls.addWidget(self._button_bars)
        self._controls.addWidget(self._button_3d)

        # Under the cursor: elevation as the DEM holds it, plus whatever else is
        # worth reporting later.
        self._status = QLabel("Move the pointer over the image for values.")
        self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._layout.addWidget(self._status)
        self._image.cursor_moved.connect(self._update_status)

        self._bind(self._combo_mode, "mode")
        self._bind(self._combo_axis, "profile_axis")
        self._bind(self._slider_line, "profile_position")
        self._bind(self._check_visible_only, "visible_only")
        self._image.view_changed.connect(self._update_plot)

    def _load(self, path):
        raster = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raster is None:
            raise ValueError("The file is not a readable raster.")
        if raster.ndim == 3:
            raster = raster[:, :, 0]
        elevation = downscale(raster).astype(np.float32)
        elevation[elevation < VOID_BELOW] = np.nan
        self._elevation = elevation

        finite = elevation[np.isfinite(elevation)]
        if finite.size:
            percentiles = np.percentile(finite, [5, 50, 95])
            self._log.appendPlainText(
                f"{elevation.shape[1]} x {elevation.shape[0]} cells; "
                f"elevation {finite.min():,.2f} to {finite.max():,.2f}, "
                f"median {percentiles[1]:,.2f}, relief {finite.max() - finite.min():,.2f}; "
                f"{100 * (1 - finite.size / elevation.size):,.1f}% void")
        self._render()

    def _render(self):
        if self._elevation is None:
            return
        elevation = self._elevation
        filled = np.nan_to_num(elevation, nan=float(np.nanmedian(elevation)))
        mode = self._combo_mode.currentIndex()

        dy, dx = np.gradient(filled)
        if mode == 0:
            display = _hillshade_image(filled)
        elif mode == 1:
            display = cv2.applyColorMap(stretch(elevation, 1, 99), cv2.COLORMAP_TURBO)
            display[~np.isfinite(elevation)] = 0
        else:
            slope_degrees = np.degrees(np.arctan(np.hypot(dx, dy)))
            display = cv2.applyColorMap(stretch(slope_degrees, 1, 99), cv2.COLORMAP_INFERNO)

        rows, columns = elevation.shape
        fraction = self._slider_line.value() / 100.0
        along_row = self._combo_axis.currentIndex() == 0
        if along_row:
            index = min(int(fraction * (rows - 1)), rows - 1)
            values = elevation[index, :]
            marker = ((0, index), (columns - 1, index))
            title = f"Elevation across row {index} of {rows}"
        else:
            index = min(int(fraction * (columns - 1)), columns - 1)
            values = elevation[:, index]
            marker = ((index, 0), (index, rows - 1))
            title = f"Elevation down column {index} of {columns}"

        display = display.copy()
        cv2.line(display, marker[0], marker[1], (0, 0, 255), max(1, rows // 400))
        self._show(display)
        self._update_plot()

    def _region(self):
        """The drawn region, else the visible area, else the whole tile."""
        bounds = self._image.roi()
        if bounds:
            return bounds, "region"
        if self._check_visible_only.isChecked():
            visible = self._image.visible_image_rect()
            if visible:
                return visible, "visible area"
        rows, columns = self._elevation.shape
        return (0, 0, columns, rows), "whole tile"

    def _update_status(self, cell):
        """Values for the cell under the pointer, in the DEM's own units."""
        if cell is None or self._elevation is None:
            self._status.setText("Move the pointer over the image for values.")
            return
        row, column = cell
        rows, columns = self._elevation.shape
        if not (0 <= row < rows and 0 <= column < columns):
            return

        elevation = float(self._elevation[row, column])
        parts = [f"row {row}, column {column}"]
        parts.append("elevation void" if not np.isfinite(elevation)
                     else f"elevation {elevation:,.3f}")

        # Slope from the immediate neighbours, which is what the eye is reading
        # off the slope view.
        if 0 < row < rows - 1 and 0 < column < columns - 1:
            patch = self._elevation[row - 1:row + 2, column - 1:column + 2]
            if np.isfinite(patch).all():
                dy = (float(patch[2, 1]) - float(patch[0, 1])) / 2.0
                dx = (float(patch[1, 2]) - float(patch[1, 0])) / 2.0
                parts.append(f"slope {math.degrees(math.atan(math.hypot(dx, dy))):,.2f}\u00b0")

        roi = self._image.roi()
        if roi:
            left, top, right, bottom = roi
            inside = left <= column < right and top <= row < bottom
            parts.append("in region" if inside else "outside region")

        self._status.setText("   |   ".join(parts))

    def _open_compound_bars(self):
        """Analyse the drawn region for amalgamated bar units."""
        if self._elevation is None:
            QMessageBox.information(self, "Compound Bars", "Open an elevation file first.")
            return
        (left, top, right, bottom), source = self._region()
        patch = self._elevation[top:bottom, left:right]
        title = (f"{os.path.basename(self._path)} ({source}: rows {top}-{bottom}, "
                 f"columns {left}-{right})")
        if self._bars_dialog is None:
            self._bars_dialog = CompoundBarsDialog(patch, title, self, self._settings)
        else:
            self._bars_dialog.set_elevation(patch, title)
        self._bars_dialog.show()
        self._bars_dialog.raise_()

    def _open_surface(self):
        """
        The 3-D surface of the drawn region, or of what is on screen, so zooming
        to a bar and opening this shows the bar rather than the whole tile.
        """
        if self._elevation is None:
            QMessageBox.information(self, "3D Surface", "Open an elevation file first.")
            return

        (left, top, right, bottom), source = self._region()
        elevation = self._elevation[top:bottom, left:right]
        title = (f"{os.path.basename(self._path)} ({source}: rows {top}-{bottom}, "
                 f"columns {left}-{right})")

        if self._surface_dialog is None:
            self._surface_dialog = make_surface_dialog(elevation, title, self)
        else:
            self._surface_dialog.set_elevation(elevation, title)
        self._surface_dialog.show()
        self._surface_dialog.raise_()

    def _update_plot(self):
        """
        The profile along the chosen line, over the visible part of the tile when
        that box is ticked. Called whenever the view is zoomed, panned or resized,
        so the plot always matches what is on screen.
        """
        if self._elevation is None:
            return
        elevation = self._elevation
        rows, columns = elevation.shape
        fraction = self._slider_line.value() / 100.0
        along_row = self._combo_axis.currentIndex() == 0

        left, top, right, bottom = 0, 0, columns, rows
        visible = self._image.visible_image_rect() if self._check_visible_only.isChecked() else None
        if visible:
            left, top, right, bottom = visible

        if along_row:
            index = min(int(fraction * (rows - 1)), rows - 1)
            values = elevation[index, left:right]
            extent = f"columns {left} to {right}" if visible else f"all {columns} columns"
            title = f"Elevation across row {index}, {extent}"
        else:
            index = min(int(fraction * (columns - 1)), columns - 1)
            values = elevation[top:bottom, index]
            extent = f"rows {top} to {bottom}" if visible else f"all {rows} rows"
            title = f"Elevation down column {index}, {extent}"

        self._plot.setPixmap(to_pixmap(
            plot_profile(np.asarray(values, np.float32), title,
                         "pixels along the line", "elevation")).scaled(
            self._plot.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


def _hillshade_image(filled, azimuth=HILLSHADE_AZIMUTH, altitude=HILLSHADE_ALTITUDE):
    import math
    dy, dx = np.gradient(filled)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    azimuth_radians = math.radians(360.0 - azimuth + 90.0)
    altitude_radians = math.radians(altitude)
    shaded = (math.sin(altitude_radians) * np.cos(slope)
              + math.cos(altitude_radians) * np.sin(slope)
              * np.cos(azimuth_radians - aspect))
    gray = (np.clip(shaded, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ======================================================================================================================
# 3D surface
# ======================================================================================================================
# Matplotlib redraws every quad on each rotation, so the cell count decides how
# usable the view is. 120 cells is about 14,000 quads, which turns smoothly;
# 250 is four times that and drags on most machines.
SURFACE_MAX_CELLS = 120
SURFACE_SLOW_CELLS = 200       # warn beyond this
SURFACE_COLORMAPS = ["terrain", "gist_earth", "viridis", "cividis", "gray"]
# The OpenGL views push geometry to the GPU once, so they carry far more than
# the CPU-drawn ones.
GL_SURFACE_CELLS = 600
GL_MAX_POINTS = 2_000_000


def opengl_available() -> bool:
    """Whether pyqtgraph's OpenGL viewer can be used on this machine."""
    try:
        import pyqtgraph.opengl      # noqa: F401  (presence check only)
        import OpenGL                # noqa: F401
        return True
    except Exception:
        return False


def make_surface_dialog(elevation, title, parent=None):
    """
    The fastest surface view this machine can give: OpenGL when pyqtgraph and a
    usable context are present, otherwise the matplotlib one. A failure to
    create the OpenGL widget is not fatal; it costs speed, not the feature.
    """
    if opengl_available():
        try:
            return GLSurfaceDialog(elevation, title, parent)
        except Exception as err:
            print(f"[downloader_viewers] OpenGL surface unavailable, "
                  f"using the CPU view: {type(err).__name__}: {err}")
    return SurfaceDialog(elevation, title, parent)


def make_point_cloud_dialog(points, title, parent=None):
    """As above, for the lidar point cloud."""
    if opengl_available():
        try:
            return GLPointCloudDialog(points, title, parent)
        except Exception as err:
            print(f"[downloader_viewers] OpenGL point cloud unavailable, "
                  f"using the CPU view: {type(err).__name__}: {err}")
    return PointCloudDialog(points, title, parent)


class GLSurfaceDialog(QDialog):
    """
    The elevation grid as a rotatable surface, drawn by the GPU through
    pyqtgraph's OpenGL widget. The geometry is uploaded once and redrawn there,
    so rotation stays smooth at cell counts that make a matplotlib surface
    unusable. Machines without a usable OpenGL context fall back to
    SurfaceDialog, which draws the same thing on the CPU.

    Drag rotates, the wheel zooms, and the middle button pans.
    """

    def __init__(self, elevation, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"3D Surface: {title}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self._elevation = elevation

        import pyqtgraph.opengl as gl
        self._gl = gl
        self._view = gl.GLViewWidget()
        self._surface = None
        self._grid_item = None

        self._spin_cells = QSpinBox()
        self._spin_cells.setRange(40, 4000)
        self._spin_cells.setValue(GL_SURFACE_CELLS)
        self._spin_cells.setKeyboardTracking(False)
        self._spin_cells.setToolTip("Cells along the longer side. The GPU handles far more "
                                    "than the drawn-on-the-CPU view can.")
        self._spin_exaggeration = QDoubleSpinBox()
        self._spin_exaggeration.setRange(1.0, 100.0)
        self._spin_exaggeration.setValue(8.0)
        self._spin_exaggeration.setKeyboardTracking(False)
        self._spin_exaggeration.setToolTip("Vertical exaggeration. 1 is true scale.")
        self._combo_colormap = QComboBox()
        self._combo_colormap.addItems(SURFACE_COLORMAPS)
        self._check_wireframe = QCheckBox("Wireframe")
        self._label_status = QLabel("")
        button_redraw = QPushButton("Redraw")

        for widget in (self._spin_cells, self._spin_exaggeration):
            widget.valueChanged.connect(self._draw)
        self._combo_colormap.currentIndexChanged.connect(self._draw)
        self._check_wireframe.toggled.connect(self._draw)
        button_redraw.clicked.connect(self._draw)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Cells:"));        controls.addWidget(self._spin_cells)
        controls.addWidget(QLabel("Exaggeration:")); controls.addWidget(self._spin_exaggeration)
        controls.addWidget(QLabel("Colors:"));       controls.addWidget(self._combo_colormap)
        controls.addWidget(self._check_wireframe)
        controls.addWidget(button_redraw)
        controls.addWidget(self._label_status)
        controls.addStretch(1)
        controls.addWidget(QLabel("Drag rotates, wheel zooms, middle button pans"))

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self._view, 1)
        self.resize(950, 720)
        self._draw()

    # ------------------------------------------------------------------------------------------------------------------
    def set_elevation(self, elevation, title):
        self._elevation = elevation
        self.setWindowTitle(f"3D Surface: {title}")
        self._draw()

    def _draw(self):
        import time
        elevation = self._elevation
        if elevation is None or elevation.size == 0:
            return
        started = time.time()

        cells = self._spin_cells.value()
        step = max(1, int(max(elevation.shape) / cells))
        grid = elevation[::step, ::step].astype(np.float32)
        finite = np.isfinite(grid)
        if not finite.any():
            return
        grid = np.where(finite, grid, float(np.nanmedian(grid[finite])))

        rows, columns = grid.shape
        span = max(float(grid.max() - grid.min()), 1e-6)
        # The grid is drawn in cell units, with height scaled so the exaggeration
        # means the same whatever the relief.
        height_scale = (max(rows, columns) * 0.02 * self._spin_exaggeration.value()) / span
        z = (grid - grid.min()) * height_scale

        colors = _colormap_rgba(grid, self._combo_colormap.currentText())

        if self._surface is not None:
            self._view.removeItem(self._surface)
        self._surface = self._gl.GLSurfacePlotItem(
            x=np.arange(rows, dtype=np.float32),
            y=np.arange(columns, dtype=np.float32),
            z=z, colors=colors, shader=None, smooth=False,
            drawEdges=self._check_wireframe.isChecked(),
            drawFaces=not self._check_wireframe.isChecked())
        self._surface.translate(-rows / 2.0, -columns / 2.0, -float(z.mean()))
        self._view.addItem(self._surface)
        self._view.setCameraPosition(distance=max(rows, columns) * 1.6)

        self._label_status.setText(
            f"{rows} x {columns} cells, {(rows - 1) * (columns - 1):,} quads, "
            f"{time.time() - started:,.2f}s")


class GLPointCloudDialog(QDialog):
    """
    The lidar returns as a rotatable point cloud on the GPU. Millions of points
    are practical here, where a matplotlib scatter is not.
    """

    COLOR_BY = ["Elevation", "Classification", "Flight line"]

    def __init__(self, points, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"3D Points: {title}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self._points = points

        import pyqtgraph.opengl as gl
        self._gl = gl
        self._view = gl.GLViewWidget()
        self._scatter = None

        self._spin_points = QSpinBox()
        self._spin_points.setRange(10_000, 20_000_000)
        self._spin_points.setSingleStep(100_000)
        self._spin_points.setValue(GL_MAX_POINTS)
        self._spin_points.setKeyboardTracking(False)
        self._spin_size = QDoubleSpinBox()
        self._spin_size.setRange(0.5, 20.0)
        self._spin_size.setValue(2.0)
        self._spin_size.setKeyboardTracking(False)
        self._spin_exaggeration = QDoubleSpinBox()
        self._spin_exaggeration.setRange(1.0, 100.0)
        self._spin_exaggeration.setValue(8.0)
        self._spin_exaggeration.setKeyboardTracking(False)
        self._combo_color = QComboBox()
        self._combo_color.addItems(self.COLOR_BY)
        self._label_status = QLabel("")
        button_redraw = QPushButton("Redraw")

        for widget in (self._spin_points, self._spin_size, self._spin_exaggeration):
            widget.valueChanged.connect(self._draw)
        self._combo_color.currentIndexChanged.connect(self._draw)
        button_redraw.clicked.connect(self._draw)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Points:"));       controls.addWidget(self._spin_points)
        controls.addWidget(QLabel("Size:"));         controls.addWidget(self._spin_size)
        controls.addWidget(QLabel("Exaggeration:")); controls.addWidget(self._spin_exaggeration)
        controls.addWidget(QLabel("Color by:"));     controls.addWidget(self._combo_color)
        controls.addWidget(button_redraw)
        controls.addWidget(self._label_status)
        controls.addStretch(1)
        controls.addWidget(QLabel("Drag rotates, wheel zooms, middle button pans"))

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self._view, 1)
        self.resize(950, 720)
        self._draw()

    # ------------------------------------------------------------------------------------------------------------------
    def set_points(self, points, title):
        self._points = points
        self.setWindowTitle(f"3D Points: {title}")
        self._draw()

    def _draw(self):
        import time
        x, y, z, classification, source = self._points
        if len(z) == 0:
            return
        started = time.time()

        step = max(1, len(z) // self._spin_points.value())
        x, y, z = x[::step], y[::step], z[::step]
        classification, source = classification[::step], source[::step]

        by = self._combo_color.currentIndex()
        values = {1: classification, 2: source}.get(by, z).astype(np.float64)
        colormap = {1: "tab20", 2: "tab10"}.get(by, "terrain")
        colors = _colormap_rgba(values, colormap)

        # Centred on the data, with height scaled for the exaggeration.
        span_x = max(x.max() - x.min(), 1e-6)
        span_y = max(y.max() - y.min(), 1e-6)
        span_z = max(z.max() - z.min(), 1e-6)
        height_scale = (max(span_x, span_y) * 0.02 * self._spin_exaggeration.value()) / span_z
        data = np.column_stack([x - x.mean(), y - y.mean(), (z - z.mean()) * height_scale])

        if self._scatter is not None:
            self._view.removeItem(self._scatter)
        self._scatter = self._gl.GLScatterPlotItem(
            pos=data.astype(np.float32), color=colors,
            size=self._spin_size.value(), pxMode=True)
        self._view.addItem(self._scatter)
        self._view.setCameraPosition(distance=max(span_x, span_y) * 1.4)

        self._label_status.setText(
            f"{len(z):,} point(s) drawn, {time.time() - started:,.2f}s")


def _colormap_rgba(values, colormap_name):
    """Values to RGBA, using matplotlib's colormaps without drawing anything."""
    from matplotlib import cm
    finite = values[np.isfinite(values)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 1.0
    scaled = (values - low) / max(high - low, 1e-9)
    return cm.get_cmap(colormap_name)(np.clip(scaled, 0, 1)).astype(np.float32)


class SurfaceDialog(QDialog):
    """
    The elevation grid as a rotatable 3-D surface. Dragging turns it, the wheel
    zooms, and the vertical exaggeration makes a river bar legible: a metre of
    relief across a kilometre of bar is invisible at true scale.

    The grid is subsampled, because a surface plot of a full tile is unusably
    slow and adds nothing at this scale.
    """

    def __init__(self, elevation, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"3D Surface: {title}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self._elevation = elevation

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

        self._figure = Figure(figsize=(8, 6), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._axes = self._figure.add_subplot(111, projection="3d")
        toolbar = NavigationToolbar2QT(self._canvas, self)

        self._spin_cells = QSpinBox()
        self._spin_cells.setRange(40, 600)
        self._spin_cells.setValue(SURFACE_MAX_CELLS)
        self._spin_cells.setKeyboardTracking(False)      # redraw once, not per keystroke
        self._spin_cells.setToolTip(
            "Cells along the longer side. The cost grows with the square of this, "
            "so 240 is four times the work of 120.")
        self._spin_exaggeration = QDoubleSpinBox()
        self._spin_exaggeration.setRange(1.0, 100.0)
        self._spin_exaggeration.setValue(8.0)
        self._spin_exaggeration.setSingleStep(1.0)
        self._spin_exaggeration.setKeyboardTracking(False)
        self._spin_exaggeration.setToolTip("Vertical exaggeration. 1 is true scale.")
        self._combo_colormap = QComboBox()
        self._combo_colormap.addItems(SURFACE_COLORMAPS)
        self._check_wireframe = QCheckBox("Wireframe")
        self._check_wireframe.setToolTip("Lines rather than shaded quads. Faster to rotate.")
        self._label_status = QLabel("")
        button_redraw = QPushButton("Redraw")

        for widget in (self._spin_cells, self._spin_exaggeration):
            widget.valueChanged.connect(self._draw)
        self._combo_colormap.currentIndexChanged.connect(self._draw)
        self._check_wireframe.toggled.connect(self._draw)
        button_redraw.clicked.connect(self._draw)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Cells:"));        controls.addWidget(self._spin_cells)
        controls.addWidget(QLabel("Exaggeration:")); controls.addWidget(self._spin_exaggeration)
        controls.addWidget(QLabel("Colors:"));       controls.addWidget(self._combo_colormap)
        controls.addWidget(self._check_wireframe)
        controls.addWidget(button_redraw)
        controls.addWidget(self._label_status)
        controls.addStretch(1)
        controls.addWidget(QLabel("Drag to rotate, wheel to zoom"))

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(toolbar)
        layout.addWidget(self._canvas, 1)
        self.resize(900, 700)
        self._draw()

    # ------------------------------------------------------------------------------------------------------------------
    def set_elevation(self, elevation, title):
        self._elevation = elevation
        self.setWindowTitle(f"3D Surface: {title}")
        self._draw()

    def _draw(self):
        elevation = self._elevation
        if elevation is None or elevation.size == 0:
            return

        cells = self._spin_cells.value()
        step = max(1, int(max(elevation.shape) / cells))
        grid = elevation[::step, ::step].astype(np.float64)
        grid = np.where(np.isfinite(grid), grid, np.nan)
        if np.all(np.isnan(grid)):
            return
        grid = np.where(np.isnan(grid), np.nanmedian(grid), grid)

        rows, columns = grid.shape
        x, y = np.meshgrid(np.arange(columns), np.arange(rows))

        # Keep the view while redrawing, so a rotation survives a settings change.
        elevation_angle, azimuth = self._axes.elev, self._axes.azim
        self._axes.clear()

        import time
        from PyQt5.QtWidgets import QApplication
        started = time.time()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            colormap = self._combo_colormap.currentText()
            if self._check_wireframe.isChecked():
                self._axes.plot_wireframe(x, y, grid, rstride=1, cstride=1, linewidth=0.4,
                                          color="0.3")
            else:
                # shade=False skips per-quad lighting, which is most of the cost
                # and adds nothing over a colormap keyed to elevation.
                self._axes.plot_surface(x, y, grid, cmap=colormap, linewidth=0,
                                        antialiased=False, shade=False,
                                        rstride=1, cstride=1)
        finally:
            QApplication.restoreOverrideCursor()

        # Box aspect sets the exaggeration. The height axis is drawn as a
        # fraction of the longer horizontal axis, scaled by the factor, so the
        # number means the same whatever the grid size or the relief.
        exaggeration = self._spin_exaggeration.value()
        height_fraction = 0.02 * exaggeration
        self._axes.set_box_aspect((columns, rows, max(columns, rows) * height_fraction))
        self._axes.set_xlabel("column")
        self._axes.set_ylabel("row")
        self._axes.set_zlabel("elevation")
        self._axes.view_init(elev=elevation_angle, azim=azimuth)
        self._canvas.draw_idle()

        quads = (rows - 1) * (columns - 1)
        message = f"{rows} x {columns} cells, {quads:,} quads, {time.time() - started:,.1f}s"
        if cells > SURFACE_SLOW_CELLS:
            message += "  (lower Cells for smoother rotation)"
        self._label_status.setText(message)


class PointCloudDialog(QDialog):
    """
    The returns themselves as a rotatable 3-D scatter. A surface hides what a
    point cloud is doing at the edges of water and under vegetation; this shows
    the returns, colored by elevation, class or flight line.

    Points are subsampled, because drawing millions of markers is slower than
    it is informative.
    """

    COLOR_BY = ["Elevation", "Classification", "Flight line"]

    def __init__(self, points, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"3D Points: {title}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self._points = points          # x, y, z, classification, source

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

        self._figure = Figure(figsize=(8, 6), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._axes = self._figure.add_subplot(111, projection="3d")
        toolbar = NavigationToolbar2QT(self._canvas, self)

        self._spin_points = QSpinBox()
        self._spin_points.setRange(5_000, 500_000)
        self._spin_points.setSingleStep(5_000)
        self._spin_points.setValue(80_000)
        self._spin_points.setToolTip("Points drawn. Higher is slower.")
        self._spin_size = QDoubleSpinBox()
        self._spin_size.setRange(0.1, 20.0)
        self._spin_size.setValue(1.0)
        self._spin_size.setToolTip("Marker size.")
        self._spin_exaggeration = QDoubleSpinBox()
        self._spin_exaggeration.setRange(1.0, 100.0)
        self._spin_exaggeration.setValue(8.0)
        self._spin_exaggeration.setToolTip("Vertical exaggeration. 1 is true scale.")
        self._combo_color = QComboBox()
        self._combo_color.addItems(self.COLOR_BY)
        self._label_status = QLabel("")
        button_redraw = QPushButton("Redraw")

        for widget in (self._spin_points, self._spin_size, self._spin_exaggeration):
            widget.setKeyboardTracking(False)        # redraw once, not per keystroke
            widget.valueChanged.connect(self._draw)
        self._combo_color.currentIndexChanged.connect(self._draw)
        button_redraw.clicked.connect(self._draw)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Points:"));       controls.addWidget(self._spin_points)
        controls.addWidget(QLabel("Size:"));         controls.addWidget(self._spin_size)
        controls.addWidget(QLabel("Exaggeration:")); controls.addWidget(self._spin_exaggeration)
        controls.addWidget(QLabel("Color by:"));     controls.addWidget(self._combo_color)
        controls.addWidget(button_redraw)
        controls.addWidget(self._label_status)
        controls.addStretch(1)
        controls.addWidget(QLabel("Drag to rotate, wheel to zoom"))

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(toolbar)
        layout.addWidget(self._canvas, 1)
        self.resize(950, 720)
        self._draw()

    # ------------------------------------------------------------------------------------------------------------------
    def set_points(self, points, title):
        self._points = points
        self.setWindowTitle(f"3D Points: {title}")
        self._draw()

    def _draw(self):
        x, y, z, classification, source = self._points
        if len(z) == 0:
            return

        wanted = self._spin_points.value()
        step = max(1, len(z) // wanted)
        x, y, z = x[::step], y[::step], z[::step]
        classification, source = classification[::step], source[::step]

        by = self._combo_color.currentIndex()
        if by == 1:
            colors, colormap = classification, "tab20"
        elif by == 2:
            colors, colormap = source, "tab10"
        else:
            colors, colormap = z, "terrain"

        elevation_angle, azimuth = self._axes.elev, self._axes.azim
        self._axes.clear()
        self._axes.scatter(x, y, z, c=colors, cmap=colormap,
                           s=self._spin_size.value(), marker=".", linewidths=0)

        span_x = max(x.max() - x.min(), 1e-6)
        span_y = max(y.max() - y.min(), 1e-6)
        height_fraction = 0.02 * self._spin_exaggeration.value()
        self._axes.set_box_aspect((span_x, span_y, max(span_x, span_y) * height_fraction))
        self._axes.set_xlabel("east")
        self._axes.set_ylabel("north")
        self._axes.set_zlabel("elevation")
        self._axes.view_init(elev=elevation_angle, azim=azimuth)
        self._canvas.draw_idle()
        self._label_status.setText(f"{len(z):,} point(s) drawn")


class CompoundBarsDialog(QDialog):
    """
    Whether the bar in the region is one unit or several amalgamated ones.
    Every setting is editable, because the answer depends on them: a lower
    prominence threshold splits a bar into more units, and a coarser slice
    interval merges them.
    """

    # The base is the elevation model as it came; the overlay is a rendering of
    # it that shows form. Opacity blends between them, so the findings can be
    # checked against the data with the slope fading in and out over it.
    BASES = ["Elevation (grey)", "Hillshade", "None (black)"]
    OVERLAYS = ["Slope", "Color relief", "Hillshade", "None"]

    def __init__(self, elevation, title, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle(f"Compound Bars: {title}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self._elevation = elevation
        self._result = None
        self._settings = settings

        import compound_bars
        self._bars = compound_bars

        self._spin_levels = QSpinBox()
        self._spin_levels.setRange(5, 500)
        self._spin_levels.setValue(compound_bars.DEFAULT_LEVELS)
        self._spin_levels.setToolTip("Slices between the lowest and highest point in the region.")
        self._spin_prominence = QDoubleSpinBox()
        self._spin_prominence.setRange(0.0, 100.0)
        self._spin_prominence.setDecimals(3)
        self._spin_prominence.setSingleStep(0.05)
        self._spin_prominence.setSpecialValueText("Automatic")
        self._spin_prominence.setToolTip(
            "How far a summit must stand above the level where it merges with its "
            "neighbour to count as a unit, in the elevation model's own units. "
            "Automatic uses the larger of three times the estimated noise and a "
            "small fraction of the relief.")
        self._spin_min_area = QSpinBox()
        self._spin_min_area.setRange(0, 1000000)
        self._spin_min_area.setValue(compound_bars.DEFAULT_MIN_AREA_CELLS)
        self._spin_min_area.setToolTip("Smallest crest area counted as a unit, in cells.")
        self._combo_detrend = QComboBox()
        self._combo_detrend.addItems(["No detrend", "Remove slope", "Remove slope and curvature"])
        self._combo_detrend.setCurrentIndex(compound_bars.DEFAULT_DETREND_ORDER)
        self._combo_detrend.setToolTip(
            "A reach falls downstream, so slicing raw elevation cuts bands across the "
            "region instead of closed lobes. Removing that trend leaves the relief "
            "relative to the local surface, which is what a bar platform is.")
        self._combo_detrend.currentIndexChanged.connect(self._run)
        self._spin_max_units = QSpinBox()
        self._spin_max_units.setRange(1, 200)
        self._spin_max_units.setValue(compound_bars.DEFAULT_MAX_UNITS)
        self._spin_max_units.setToolTip("Report at most this many units, the most prominent "
                                        "first.")
        self._spin_smoothing = QSpinBox()
        self._spin_smoothing.setRange(0, 51)
        self._spin_smoothing.setValue(compound_bars.DEFAULT_SMOOTHING_CELLS)
        self._spin_smoothing.setToolTip("Median filter width before slicing. 0 leaves the "
                                        "surface as it is.")
        self._combo_method = QComboBox()
        self._combo_method.addItems(["Margins (slope)", "Prominence (elevation)", "Both"])
        self._combo_method.setToolTip(
            "Margins finds units by the slope breaks around them, which separates "
            "platforms of the same height. Prominence finds them by how far each "
            "summit stands above its neighbours. Both reports each and compares them.")
        self._combo_method.currentIndexChanged.connect(self._run)
        self._spin_flat = QDoubleSpinBox()
        self._spin_flat.setRange(1.0, 95.0)
        self._spin_flat.setValue(compound_bars.DEFAULT_FLAT_PERCENTILE)
        self._spin_flat.setKeyboardTracking(False)
        self._spin_flat.setToolTip("Slope percentile below which ground is flat enough to "
                                   "seed a platform.")
        self._spin_margin = QDoubleSpinBox()
        self._spin_margin.setRange(5.0, 99.9)
        self._spin_margin.setValue(compound_bars.DEFAULT_MARGIN_PERCENTILE)
        self._spin_margin.setKeyboardTracking(False)
        self._spin_margin.setToolTip("Slope percentile a boundary must reach to count as a "
                                     "margin; weaker ones are dissolved and their regions "
                                     "merged.")
        self._spin_detail = QSpinBox()
        self._spin_detail.setRange(1, 6)
        self._spin_detail.setValue(compound_bars.DEFAULT_DETAIL_STEPS)
        self._spin_detail.setKeyboardTracking(False)
        self._spin_detail.setToolTip(
            "How many margin thresholds to run. One splits the region into bars; "
            "more find what stands on them, because a weaker threshold subdivides "
            "a bar into the platforms it is built from.")
        self._spin_detail.valueChanged.connect(self._run)
        for widget in (self._spin_flat, self._spin_margin):
            widget.valueChanged.connect(self._run)

        self._combo_background = QComboBox()
        self._combo_background.addItems(self.BASES)
        self._combo_background.setToolTip("The layer underneath: the elevation model itself.")
        self._combo_overlay = QComboBox()
        self._combo_overlay.addItems(self.OVERLAYS)
        self._combo_overlay.setToolTip("The layer blended over the elevation model. Slope "
                                       "shows the breaks that bound bar platforms.")
        self._combo_overlay.currentIndexChanged.connect(self._draw)
        self._check_analysis_surface = QCheckBox("Analysed surface")
        self._check_analysis_surface.setToolTip(
            "Draw over the smoothed, detrended surface the units were found on, "
            "rather than the elevation model as it came. Unticked is the data "
            "itself, which is what the findings should be judged against.")
        self._check_analysis_surface.toggled.connect(self._draw)
        self._slider_opacity = QSlider(Qt.Horizontal)
        self._slider_opacity.setRange(0, 100)
        self._slider_opacity.setValue(100)
        self._slider_opacity.setFixedWidth(90)
        self._slider_opacity.setToolTip("How strongly the unit outlines are drawn over the "
                                        "data underneath.")
        self._slider_opacity.valueChanged.connect(self._draw)
        self._spin_line = QSpinBox()
        self._spin_line.setRange(1, 10)
        self._spin_line.setValue(1)
        self._spin_line.setKeyboardTracking(False)
        self._spin_line.setToolTip("Outline width in image pixels.")
        self._spin_line.valueChanged.connect(self._draw)
        self._check_ellipses = QCheckBox("Ellipses")
        self._check_ellipses.setChecked(True)
        self._check_ellipses.toggled.connect(self._draw)
        self._combo_stack = QComboBox()
        self._combo_stack.addItem("All stacks")
        self._combo_stack.setToolTip(
            "Draw one stack level on its own. A stacked unit shares most of its "
            "outline with the unit it sits on, so the two are hard to separate "
            "when everything is drawn together.")
        self._combo_stack.currentIndexChanged.connect(self._draw)
        self._check_labels = QCheckBox("Labels")
        self._check_labels.setChecked(True)
        self._check_labels.toggled.connect(self._draw)
        for widget in (self._spin_levels, self._spin_prominence, self._spin_max_units,
                       self._spin_min_area, self._spin_smoothing):
            widget.setKeyboardTracking(False)
            widget.valueChanged.connect(self._run)
        self._combo_background.currentIndexChanged.connect(self._draw)

        button_run = QPushButton("Recompute")
        button_run.clicked.connect(self._run)
        button_save = QPushButton("Save Report")
        button_save.clicked.connect(self._save)

        # The settings live in a panel beside the view rather than in a row above
        # it: a row this long cannot shrink, which fixes the window's minimum
        # width and drags when a slider inside it is used.
        from PyQt5.QtWidgets import QFormLayout, QScrollArea

        def group(title, rows):
            box = QGroupBox(title)
            form = QFormLayout(box)
            form.setLabelAlignment(Qt.AlignRight)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            for label, widget in rows:
                if label is None:
                    form.addRow(widget)
                else:
                    form.addRow(label, widget)
            return box

        detection = group("Detection", [
            ("Method:", self._combo_method),
            ("Flat %:", self._spin_flat),
            ("Margin %:", self._spin_margin),
            ("Levels:", self._spin_detail),
            ("Slices:", self._spin_levels),
            ("Prominence:", self._spin_prominence),
            ("Min area:", self._spin_min_area),
            ("Max units:", self._spin_max_units),
            ("Smoothing:", self._spin_smoothing),
            ("Trend:", self._combo_detrend),
        ])
        display = group("Display", [
            ("Base:", self._combo_background),
            ("Overlay:", self._combo_overlay),
            ("Opacity:", self._slider_opacity),
            ("Line:", self._spin_line),
            ("Show:", self._combo_stack),
            (None, self._check_analysis_surface),
            (None, self._check_ellipses),
            (None, self._check_labels),
        ])

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(detection)
        panel_layout.addWidget(display)
        panel_layout.addWidget(button_run)
        panel_layout.addWidget(button_save)
        panel_layout.addStretch(1)

        self._panel = QScrollArea()
        self._panel.setWidget(panel)
        self._panel.setWidgetResizable(True)
        self._panel.setFixedWidth(260)
        self._slider_opacity.setFixedWidth(120)

        self._view = ZoomableView()
        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        font = self._report.font()
        font.setFamily("Courier New")
        self._report.setFont(font)
        self._report.setMinimumHeight(180)

        splitter = QSplitter(Qt.Vertical)
        view_box = QGroupBox("Units")
        view_layout = QVBoxLayout(view_box)
        view_layout.addWidget(self._view)
        report_box = QGroupBox("Findings")
        report_layout = QVBoxLayout(report_box)
        report_layout.addWidget(self._report)
        splitter.addWidget(view_box)
        splitter.addWidget(report_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self._status = QLabel("Move the pointer over the image for values.")
        self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._view.cursor_moved.connect(self._update_status)
        self._view.double_clicked.connect(self._drill_into)
        self._view.clicked.connect(self._select_unit)
        self._highlight = None
        self._modes = ModeBar(self._view, modes=(MODE_SELECT, MODE_PAN))
        self._detail_dialog = None

        body = QHBoxLayout()
        body.addWidget(self._panel)
        body.addWidget(splitter, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._modes)
        layout.addLayout(body, 1)
        layout.addWidget(self._status)
        self.resize(1140, 820)

        # Restore the settings before the first run, so it computes with them
        # rather than with the defaults and then again.
        self._bind_settings()
        self._run()

    # ------------------------------------------------------------------------------------------------------------------
    def _bind_settings(self):
        """Every control persists: the analysis is a matter of tuning, and losing
        the tuning on every restart makes it unusable."""
        if self._settings is None or not hasattr(self._settings, "bind"):
            return
        for widget, key in ((self._combo_method, "method"),
                            (self._spin_flat, "flat_percentile"),
                            (self._spin_margin, "margin_percentile"),
                            (self._spin_detail, "detail_steps"),
                            (self._spin_levels, "levels"),
                            (self._spin_prominence, "prominence"),
                            (self._spin_min_area, "min_area"),
                            (self._spin_max_units, "max_units"),
                            (self._spin_smoothing, "smoothing"),
                            (self._combo_detrend, "detrend"),
                            (self._combo_background, "base_layer"),
                            (self._combo_overlay, "overlay_layer"),
                            (self._check_analysis_surface, "analysis_surface"),
                            (self._slider_opacity, "opacity"),
                            (self._spin_line, "line_width"),
                            (self._check_ellipses, "ellipses"),
                            (self._check_labels, "labels")):
            self._settings.bind(widget, f"CompoundBars.{key}")

    # ------------------------------------------------------------------------------------------------------------------
    def set_elevation(self, elevation, title):
        self._elevation = elevation
        self.setWindowTitle(f"Compound Bars: {title}")
        self._run()

    def _run(self):
        if self._elevation is None or self._elevation.size == 0:
            return
        method = self._combo_method.currentIndex()      # 0 margins, 1 prominence, 2 both
        tree = margins = None

        if method in (1, 2):
            tree = self._bars.analyze(
                self._elevation,
                levels=self._spin_levels.value(),
                prominence=self._spin_prominence.value() or None,    # 0 means automatic
                min_area_cells=self._spin_min_area.value(),
                smoothing_cells=self._spin_smoothing.value(),
                max_units=self._spin_max_units.value(),
                detrend_order=self._combo_detrend.currentIndex())
        if method in (0, 2):
            margins = self._bars.analyze_margins(
                self._elevation,
                smoothing_cells=self._spin_smoothing.value(),
                detrend_order=self._combo_detrend.currentIndex(),
                flat_percentile=self._spin_flat.value(),
                margin_percentile=self._spin_margin.value(),
                min_area_cells=self._spin_min_area.value(),
                max_units=self._spin_max_units.value(),
                detail_steps=self._spin_detail.value())

        text = []
        if margins is not None:
            text.append(self._bars.describe_margins(margins))
        if tree is not None:
            text.append(self._bars.describe(tree))
        if tree is not None and margins is not None:
            text.append(self._bars.combine(tree, margins))
        self._report.setPlainText("\n\n".join(filter(None, text)))
        self._highlight = None          # the units are renumbered by each run

        # Offer the stack levels this result actually has.
        result = margins if margins is not None else tree
        levels = self._bars.stack_levels(result) if result else []
        previous = self._combo_stack.currentText()
        self._combo_stack.blockSignals(True)
        self._combo_stack.clear()
        self._combo_stack.addItem("All stacks")
        for level in levels:
            self._combo_stack.addItem(f"Stack {level} only")
        index = self._combo_stack.findText(previous)
        self._combo_stack.setCurrentIndex(index if index >= 0 else 0)
        self._combo_stack.blockSignals(False)

        # The margins result is drawn when it is there: its outlines are the
        # boundaries the slope view shows.
        self._result = margins if margins is not None else tree
        self._draw()

    def _draw(self):
        if not self._result or self._result.get("error"):
            return
        # Over the elevation model as it came by default, so the findings can be
        # judged against the data rather than against the surface they were
        # computed on.
        if self._check_analysis_surface.isChecked():
            surface = self._result["surface"]
        else:
            surface = np.asarray(self._elevation, np.float32).copy()
            surface[surface < self._bars.VOID_BELOW] = np.nan
            if np.isnan(surface).any():
                surface = np.where(np.isnan(surface), np.nanmedian(surface), surface)

        base = self._layer(self._combo_background.currentText(), surface)
        over = self._layer(self._combo_overlay.currentText(), surface)

        opacity = self._slider_opacity.value() / 100.0
        if over is None or opacity <= 0.0:
            background = base
        elif opacity >= 1.0:
            background = over
        else:
            background = cv2.addWeighted(over, opacity, base, 1.0 - opacity, 0)

        # The outlines go on last and at full strength: they are the findings,
        # and fading them makes them harder to judge rather than easier.
        drawn = self._bars.draw(self._result, background,
                                line_width=self._spin_line.value(),
                                show_ellipses=self._check_ellipses.isChecked(),
                                label=self._check_labels.isChecked(),
                                only_stack=self._selected_stack(),
                                highlight=self._highlight)
        self._view.set_pixmap(to_pixmap(drawn), keep_view=True)

    def _layer(self, name, surface):
        """One rendering of the surface, or None."""
        if name.startswith("None"):
            return np.zeros(surface.shape + (3,), np.uint8) if name == "None (black)" else None
        if name.startswith("Elevation"):
            return cv2.cvtColor(stretch(surface, 1, 99), cv2.COLOR_GRAY2BGR)
        if name == "Hillshade":
            return _hillshade_image(surface)
        if name == "Slope":
            dy, dx = np.gradient(surface)
            return cv2.applyColorMap(
                stretch(np.degrees(np.arctan(np.hypot(dx, dy))), 1, 99), cv2.COLORMAP_INFERNO)
        return cv2.applyColorMap(stretch(surface, 1, 99), cv2.COLORMAP_TURBO)

    def _update_status(self, cell):
        """
        Values under the pointer: the elevation as the DEM holds it, the value
        the analysis worked from, and which unit the cell falls in.
        """
        if cell is None or self._elevation is None:
            self._status.setText("Move the pointer over the image for values.")
            return
        row, column = cell
        rows, columns = self._elevation.shape
        if not (0 <= row < rows and 0 <= column < columns):
            return

        elevation = float(self._elevation[row, column])
        parts = [f"row {row}, column {column}"]
        parts.append("elevation void" if not np.isfinite(elevation)
                     else f"DEM elevation {elevation:,.3f}")

        if self._result and not self._result.get("error"):
            surface = self._result.get("surface")
            if surface is not None and surface.shape == self._elevation.shape:
                parts.append(f"analysed {float(surface[row, column]):,.3f}")
            slope = self._result.get("slope")
            if slope is not None and slope.shape == self._elevation.shape:
                parts.append(f"slope {float(slope[row, column]):,.2f}\u00b0")
            parts.append(self._unit_at(row, column))

        self._status.setText("   |   ".join(part for part in parts if part))

    def _select_unit(self, cell):
        """Click marks a unit, so the row in the table and the shape on screen
        can be matched without counting outlines."""
        unit = self._innermost_unit(*cell) if cell else None
        self._highlight = unit["index"] if unit else None
        if unit:
            parent = f", on unit {unit['parent']}" if unit.get("parent") else ""
            self._status.setText(
                f"unit {unit['index']}, stack {unit.get('stack', 0)}{parent}   |   "
                f"summit {unit['summit']:,.2f}   |   area {unit['area_cells']:,} cells"
                f"   |   double click to look inside it")
        self._draw()

    def _drill_into(self, cell):
        """
        Re-run the analysis inside one unit. Thresholds are percentiles of the
        slope, so a region holding big bars ranks an inner margin low and
        dissolves it. Inside one unit, that same margin is among the strongest
        there is, which is why looking inside finds what the whole region hides.
        """
        if cell is None or not self._result or self._result.get("error"):
            return
        row, column = cell
        unit = self._innermost_unit(row, column)
        if unit is None:
            self._status.setText("Double click inside a unit to look into it.")
            return

        outline = np.asarray(unit["outline"], np.int32)
        if outline.size < 3:
            return
        mask = np.zeros(self._elevation.shape, np.uint8)
        cv2.fillPoly(mask, [outline], 1)
        rows, columns = np.nonzero(mask)
        if rows.size == 0:
            return
        top, bottom = int(rows.min()), int(rows.max()) + 1
        left, right = int(columns.min()), int(columns.max()) + 1

        # Outside the unit is dropped, so its own slope distribution sets the
        # thresholds rather than the neighbours'.
        patch = np.asarray(self._elevation[top:bottom, left:right], np.float32).copy()
        patch[mask[top:bottom, left:right] == 0] = np.nan
        if np.isnan(patch).all():
            return

        title = f"unit {unit['index']} of {self.windowTitle().split(':', 1)[-1].strip()}"
        if self._detail_dialog is None:
            self._detail_dialog = CompoundBarsDialog(patch, title, self, self._settings)
        else:
            self._detail_dialog.set_elevation(patch, title)
        self._detail_dialog.show()
        self._detail_dialog.raise_()

    def _innermost_unit(self, row, column):
        containing = [unit for unit in self._result.get("units", [])
                      if np.asarray(unit["outline"], np.int32).size
                      and cv2.pointPolygonTest(np.asarray(unit["outline"], np.int32),
                                               (float(column), float(row)), False) >= 0]
        if not containing:
            return None
        # The smallest one wins: it is the most specific unit under the pointer.
        # Stack level breaks ties, since two units of equal area can be at
        # different levels.
        return min(containing, key=lambda unit: (unit["area_cells"],
                                                 -unit.get("stack", 0)))

    def _unit_at(self, row, column) -> str:
        """The innermost unit containing the cell, and what it sits on."""
        if not self._result:
            return ""
        containing = []
        for unit in self._result.get("units", []):
            outline = np.asarray(unit["outline"], np.int32)
            if outline.size and cv2.pointPolygonTest(outline, (float(column), float(row)),
                                                     False) >= 0:
                containing.append(unit)
        if not containing:
            return "no unit"
        innermost = min(containing, key=lambda unit: (unit["area_cells"],
                                                      -unit.get("stack", 0)))
        text = f"unit {innermost['index']}, stack {innermost.get('stack', 0)}"
        if innermost.get("parent"):
            text += f", on unit {innermost['parent']}"
        if len(containing) > 1:
            text += f" ({len(containing)} unit(s) enclose this point)"
        return text

    def _selected_stack(self):
        """The stack level to draw, or None for all of them."""
        text = self._combo_stack.currentText()
        if text.startswith("Stack "):
            try:
                return int(text.split()[1])
            except (IndexError, ValueError):
                return None
        return None

    def _save(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Report", "compound_bars.txt",
                                              "Text (*.txt);;All files (*.*)")
        if not path:
            return
        try:
            with open(path, "w") as handle:
                handle.write(self._report.toPlainText() + "\n")
        except Exception as err:
            QMessageBox.warning(self, "Save Report", f"Could not write the report:\n{err}")


# ======================================================================================================================
# Lidar
# ======================================================================================================================
class LidarViewerTab(_ViewerBase):
    """Point cloud summary and a gridded elevation image built from the returns."""

    MODES = ["Highest return", "Lowest return", "Point density", "Passes per cell"]

    def __init__(self, parent=None, settings=None):
        super().__init__(LIDAR_FILTER, parent, settings)
        self._points = None            # x, y, z, classification, flight line
        self._cell_size = None         # ground units per grid cell, for density
        self._combo_mode = QComboBox()
        self._combo_mode.addItems(self.MODES)
        self._combo_mode.currentIndexChanged.connect(self._render)
        self._check_ground = QCheckBox("Ground returns only")
        self._check_ground.setToolTip("Class 2, which is the bare earth surface.")
        self._check_ground.toggled.connect(self._render)
        self._check_thin = QCheckBox("Thin to one point per cell")
        self._check_thin.setToolTip(
            "Keep a single point per grid cell, so overlapping flight lines do not "
            "weight one strip of ground more than another. Statistics computed from "
            "the raw returns are biased along the overlap bands; this shows what the "
            "data looks like without that bias.")
        self._check_thin.toggled.connect(self._render)

        self._button_surface = QPushButton("3D Surface")
        self._button_surface.setToolTip("The gridded elevation as a rotatable surface.")
        self._button_surface.clicked.connect(self._open_surface)
        self._button_points = QPushButton("3D Points")
        self._button_points.setToolTip("The returns themselves as a rotatable point cloud.")
        self._button_points.clicked.connect(self._open_points)
        self._surface_dialog = None
        self._points_dialog = None

        self._controls.addWidget(QLabel("Grid:"))
        self._controls.addWidget(self._combo_mode)
        self._controls.addWidget(self._check_ground)
        self._controls.addWidget(self._check_thin)
        self._controls.addWidget(self._button_surface)
        self._controls.addWidget(self._button_points)
        self._controls.addStretch(1)

        self._bind(self._combo_mode, "mode")
        self._bind(self._check_ground, "ground_only")
        self._bind(self._check_thin, "thin")

    def _load(self, path):
        try:
            import laspy
        except ImportError as err:
            raise ImportError("Reading lidar needs the laspy package: pip install laspy[lazrs]. "
                              f"Original error: {err}") from err

        with laspy.open(path) as reader:
            header = reader.header
            count = header.point_count
            cloud = reader.read()

        x = np.asarray(cloud.x, np.float64)
        y = np.asarray(cloud.y, np.float64)
        z = np.asarray(cloud.z, np.float64)
        classification = np.asarray(getattr(cloud, "classification", np.zeros(len(z))), np.int16)
        # Every point records the flight line it came from, which is how passes
        # per cell is counted rather than inferred from brightness.
        source = np.asarray(getattr(cloud, "point_source_id", np.zeros(len(z))), np.int32)

        if len(z) > LIDAR_MAX_POINTS:
            step = len(z) // LIDAR_MAX_POINTS + 1
            x, y, z = x[::step], y[::step], z[::step]
            classification, source = classification[::step], source[::step]
            self._log.appendPlainText(f"Sampled every {step} point(s) for display.")

        self._points = (x, y, z, classification, source)
        lines = np.unique(source)
        self._log.appendPlainText(
            f"Flight lines in this tile: {len(lines)}"
            + (f" (ids {lines.min()} to {lines.max()})" if len(lines) > 1 else ""))

        self._log.appendPlainText(
            f"{count:,} point(s); extent {x.min():,.1f} to {x.max():,.1f} east, "
            f"{y.min():,.1f} to {y.max():,.1f} north; "
            f"elevation {z.min():,.2f} to {z.max():,.2f}")
        density = len(z) / max((x.max() - x.min()) * (y.max() - y.min()), 1e-6)
        self._log.appendPlainText(f"About {density:,.1f} point(s) per square unit of the file's CRS.")

        counts = np.bincount(np.clip(classification, 0, 31), minlength=32)
        named = [f"{LAS_CLASSES.get(code, f'class {code}')}: {value:,}"
                 for code, value in enumerate(counts) if value]
        self._log.appendPlainText("Classes: " + ", ".join(named))
        self._render()

    def _open_surface(self):
        """The gridded elevation of the current view as a 3-D surface."""
        if self._points is None:
            QMessageBox.information(self, "3D Surface", "Open a lidar file first.")
            return
        grid = self._elevation_grid()
        if grid is None:
            QMessageBox.information(self, "3D Surface", "No returns to grid.")
            return
        title = os.path.basename(self._path)
        if self._surface_dialog is None:
            self._surface_dialog = make_surface_dialog(grid, title, self)
        else:
            self._surface_dialog.set_elevation(grid, title)
        self._surface_dialog.show()
        self._surface_dialog.raise_()

    def _open_points(self):
        """The returns as a 3-D scatter, with the current filters applied."""
        if self._points is None:
            QMessageBox.information(self, "3D Points", "Open a lidar file first.")
            return
        points = self._filtered_points()
        if len(points[2]) == 0:
            QMessageBox.information(self, "3D Points", "No returns after filtering.")
            return
        title = os.path.basename(self._path)
        if self._points_dialog is None:
            self._points_dialog = make_point_cloud_dialog(points, title, self)
        else:
            self._points_dialog.set_points(points, title)
        self._points_dialog.show()
        self._points_dialog.raise_()

    def _filtered_points(self):
        """The points as the checkboxes leave them: ground only and thinned."""
        x, y, z, classification, source = self._points
        if self._check_ground.isChecked():
            keep = classification == 2
            x, y, z = x[keep], y[keep], z[keep]
            classification, source = classification[keep], source[keep]
        if self._check_thin.isChecked() and len(z):
            _, _, flat = self._cell_index(x, y)
            _, keep = np.unique(flat, return_index=True)
            x, y, z = x[keep], y[keep], z[keep]
            classification, source = classification[keep], source[keep]
        return x, y, z, classification, source

    def _cell_index(self, x, y):
        """Grid shape and the cell each point falls in, north up."""
        span_x = max(x.max() - x.min(), 1e-6)
        span_y = max(y.max() - y.min(), 1e-6)
        if span_x >= span_y:
            columns = LIDAR_GRID_CELLS
            rows = max(int(LIDAR_GRID_CELLS * span_y / span_x), 1)
        else:
            rows = LIDAR_GRID_CELLS
            columns = max(int(LIDAR_GRID_CELLS * span_x / span_y), 1)
        column_index = np.clip(((x - x.min()) / span_x * (columns - 1)).astype(np.int32),
                               0, columns - 1)
        row_index = np.clip(((y.max() - y) / span_y * (rows - 1)).astype(np.int32), 0, rows - 1)
        return rows, columns, row_index * columns + column_index

    def _elevation_grid(self):
        """Highest return per cell, as a grid, with the current filters applied."""
        x, y, z, _, _ = self._filtered_points()
        if len(z) == 0:
            return None
        rows, columns, flat = self._cell_index(x, y)
        grid = np.full(rows * columns, np.nan, np.float32)
        order = np.argsort(z)
        grid[flat[order]] = z[order]          # ascending, so the highest wins
        return grid.reshape(rows, columns)

    def _render(self):
        if self._points is None:
            return
        x, y, z, classification, source = self._points
        if self._check_ground.isChecked():
            keep = classification == 2
            if not keep.any():
                self._log.appendPlainText("No ground-classified points in this file.")
                return
            x, y, z, source = x[keep], y[keep], z[keep], source[keep]

        span_x = max(x.max() - x.min(), 1e-6)
        span_y = max(y.max() - y.min(), 1e-6)
        if span_x >= span_y:
            columns = LIDAR_GRID_CELLS
            rows = max(int(LIDAR_GRID_CELLS * span_y / span_x), 1)
        else:
            rows = LIDAR_GRID_CELLS
            columns = max(int(LIDAR_GRID_CELLS * span_x / span_y), 1)

        column_index = np.clip(((x - x.min()) / span_x * (columns - 1)).astype(np.int32),
                               0, columns - 1)
        # North up: the first row is the top of the image.
        row_index = np.clip(((y.max() - y) / span_y * (rows - 1)).astype(np.int32), 0, rows - 1)
        flat = row_index * columns + column_index
        cell_area = (span_x / columns) * (span_y / rows)      # square ground units

        if self._check_thin.isChecked():
            # One point per cell: the first of each, which removes the weighting
            # that overlapping flight lines give to the strips they share.
            _, keep = np.unique(flat, return_index=True)
            before = len(flat)
            x, y, z, source, flat = x[keep], y[keep], z[keep], source[keep], flat[keep]
            self._log.appendPlainText(
                f"Thinned {before:,} to {len(flat):,} point(s), one per cell.")

        mode = self._combo_mode.currentIndex()
        if mode == 3:
            # Passes per cell: distinct flight lines contributing to each cell.
            order = np.lexsort((source, flat))
            flat_sorted, source_sorted = flat[order], source[order]
            new_cell = np.empty(len(flat_sorted), bool)
            new_cell[0] = True
            new_cell[1:] = (flat_sorted[1:] != flat_sorted[:-1]) | \
                           (source_sorted[1:] != source_sorted[:-1])
            passes = np.bincount(flat_sorted[new_cell], minlength=rows * columns)
            grid = passes.astype(np.float32).reshape(rows, columns)
            display = np.clip(grid, 0, max(grid.max(), 1))
            image = cv2.applyColorMap(
                (display / max(display.max(), 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            covered = passes[passes > 0]
            plot = plot_histogram(covered.astype(np.float32), "Flight lines per cell")
            if covered.size:
                self._log.appendPlainText(
                    f"Passes per cell: median {int(np.median(covered))}, "
                    f"maximum {int(covered.max())}; "
                    f"{100 * float((covered > 1).mean()):,.1f}% of covered cells "
                    f"were flown more than once.")
        elif mode == 2:
            counts = np.bincount(flat, minlength=rows * columns).astype(np.float32)
            grid = counts / max(cell_area, 1e-9)              # per square ground unit
            image = cv2.applyColorMap(stretch(grid.reshape(rows, columns), 1, 99),
                                      cv2.COLORMAP_MAGMA)
            occupied = grid[counts > 0]
            plot = plot_histogram(occupied, "Points per square unit of the file's CRS")
            if occupied.size:
                self._log.appendPlainText(
                    f"Density: median {np.median(occupied):,.1f}, "
                    f"95th percentile {np.percentile(occupied, 95):,.1f} "
                    f"point(s) per square unit; cell is "
                    f"{span_x / columns:,.2f} by {span_y / rows:,.2f} units.")
        else:
            grid = np.full(rows * columns, np.nan, np.float32)
            order = np.argsort(z)                    # last write wins
            if mode == 0:
                grid[flat[order]] = z[order]         # ascending: highest last
            else:
                grid[flat[order[::-1]]] = z[order[::-1]]
            grid = grid.reshape(rows, columns)
            filled = np.nan_to_num(grid, nan=float(np.nanmedian(grid)))
            image = _hillshade_image(filled)
            plot = plot_histogram(z, "Return elevation")

        self._show(image, plot)
