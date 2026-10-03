#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# aerial_elevation_downloader.py
#
# Plugin: Aerial and Elevation Downloader.
#
# NAIP aerial imagery and 3DEP elevation data for a reach, from The National
# Map. No account and no API key: TNMAccess is open.
#
# The reach is given as an NWIS gage number with a buffer, or as a bounding
# box. Using the gage number means the same number that pulls stage for a site
# also pulls the imagery over it.
#
# Search first, then choose: products are listed with their year and size, and
# only the ticked rows are downloaded, so a wide box does not commit to every
# tile it touches.
#
# Downloaded rasters can be previewed in the tab: aerial tiles as they are, and
# elevation tiles as a hillshade with a histogram, which is the same quantity
# the sandbar inundation thresholds estimate from imagery.
#
# The search and download logic lives in tnm_download.py beside this file, so
# the command-line tool and this tab do the same thing.

import os
import math
import traceback

import numpy as np
import cv2

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
                             QLabel, QLineEdit, QPushButton, QDoubleSpinBox, QComboBox,
                             QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QPlainTextEdit, QProgressBar, QFileDialog, QMessageBox,
                             QSplitter, QAbstractItemView)

import tnm_download as tnm      # beside this file; shared with the CLI

# The host's settings helper, with a local stand-in so this plugin still runs
# on its own. Same behavior either way: one JSON file named after the plugin.
try:
    from appcore.plugins import plugin_settings
except ImportError:
    import json

    class _Settings(dict):
        def __init__(self, path):
            super().__init__()
            self.path = path
            try:
                with open(path, "r") as handle:
                    self.update(json.load(handle))
            except Exception:
                pass

        def save(self):
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "w") as handle:
                    json.dump(dict(self), handle, indent=4)
            except Exception as err:
                print(f"[aerial_elevation_downloader] Could not save settings: {err}")

        def bind(self, widget, key, default=None):
            from PyQt5.QtWidgets import (QLineEdit, QCheckBox, QComboBox,
                                         QSpinBox, QDoubleSpinBox, QSlider)
            pairs = {QLineEdit: ("setText", "text", "editingFinished"),
                     QCheckBox: ("setChecked", "isChecked", "toggled"),
                     QComboBox: ("setCurrentIndex", "currentIndex", "currentIndexChanged"),
                     QSpinBox: ("setValue", "value", "valueChanged"),
                     QDoubleSpinBox: ("setValue", "value", "valueChanged"),
                     QSlider: ("setValue", "value", "valueChanged")}
            for widget_type, (setter, getter, signal) in pairs.items():
                if isinstance(widget, widget_type):
                    value = self.get(key, default)
                    if value is not None:
                        getattr(widget, setter)(value)
                    getattr(widget, signal).connect(
                        lambda *_, w=widget, k=key, g=getter: (self.__setitem__(k, getattr(w, g)()),
                                                               self.save()))
                    return self
            return self

    def plugin_settings(plugin_file, defaults=None):
        folder = os.path.dirname(os.path.abspath(plugin_file))
        return _Settings(os.path.join(
            folder, os.path.splitext(os.path.basename(plugin_file))[0] + ".json"))

PLUGIN = {
    "title":       "Aerial and Elevation Downloader",
    "class":       "AerialElevationPlugin",
    "description": "NAIP imagery and 3DEP elevation data for a reach, from The National Map",
    "surface":     "tools",
    "size":        [1300, 850],
    "api_version": 2,
}

KINDS = [("NAIP imagery", "naip"),
         ("Elevation, 1 m DEM", "dem"),
         ("Lidar point cloud", "lidar")]
RASTER_EXTENSIONS = (".tif", ".tiff", ".jp2", ".png", ".jpg")
# Preview is downscaled to this long edge: a NAIP quarter quad is far larger
# than any panel, and a full-size decode of every tile is wasted work.
PREVIEW_LONG_EDGE = 1400


# ======================================================================================================================
# Workers
# ======================================================================================================================
class SearchWorker(QThread):
    """Asks The National Map what covers the box."""

    status   = pyqtSignal(str)
    finished = pyqtSignal(list, str)        # items, error

    def __init__(self, bbox, kind):
        super().__init__()
        self._bbox = bbox
        self._kind = kind

    def run(self):
        try:
            self.status.emit("Reading the dataset list\u2026")
            names = tnm.dataset_names(self._kind)
            if not names:
                self.finished.emit([], "The National Map has no dataset matching "
                                        f"{tnm.DATASET_PATTERNS[self._kind]}.")
                return

            items = []
            for name in names:
                self.status.emit(f"Searching {name}\u2026")
                items += tnm.search(self._bbox, name, tnm.PRODUCT_FORMATS[self._kind])

            seen, unique = set(), []
            for item in items:
                url = item.get("downloadURL")
                if url and url not in seen:
                    seen.add(url)
                    unique.append(item)
            unique.sort(key=lambda i: (tnm.item_year(i), str(i.get("title") or "")))
            self.finished.emit(unique, "")
        except Exception as err:
            traceback.print_exc()
            self.finished.emit([], f"{type(err).__name__}: {err}")


class DownloadWorker(QThread):
    """Fetches the chosen products, one at a time, and can be stopped."""

    progress = pyqtSignal(int, int)          # done, total
    status   = pyqtSignal(str)
    file_done = pyqtSignal(str)
    finished = pyqtSignal(int, int, int)     # downloaded, skipped, failed

    def __init__(self, items, out_dir, kind):
        super().__init__()
        self._items = items
        self._out_dir = out_dir
        self._kind = kind
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        downloaded = skipped = failed = 0
        for index, item in enumerate(self._items, start=1):
            if self._cancelled:
                self.status.emit(f"Cancelled after {index - 1} of {len(self._items)}.")
                break
            url = item.get("downloadURL")
            name = os.path.basename(url.split("?")[0])
            destination = os.path.join(self._out_dir, self._kind, tnm.item_year(item), name)
            if os.path.exists(destination):
                self.status.emit(f"Already present: {name}")
                self.file_done.emit(destination)
                skipped += 1
            else:
                self.status.emit(f"Downloading {name}\u2026")
                try:
                    tnm.download(url, destination)
                    self.file_done.emit(destination)
                    downloaded += 1
                except Exception as err:
                    self.status.emit(f"Failed: {name}: {type(err).__name__}: {err}")
                    failed += 1
            self.progress.emit(index, len(self._items))
        self.finished.emit(downloaded, skipped, failed)


# ======================================================================================================================
# Tab
# ======================================================================================================================
class AerialElevationDownloaderTab(QWidget):

    view_requested = pyqtSignal(str, str)       # kind, path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self._search_worker = None
        self._download_worker = None
        self._downloaded = []
        self._settings = plugin_settings(__file__)
        self._build_ui()
        self._bind_settings()

    # ------------------------------------------------------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        area = QGroupBox("Area of interest")
        grid = QGridLayout(area)
        self._edit_gage = QLineEdit()
        self._edit_gage.setPlaceholderText("NWIS gage number, e.g. 06768000")
        self._spin_buffer = QDoubleSpinBox()
        self._spin_buffer.setRange(0.1, 50.0)
        self._spin_buffer.setValue(2.0)
        self._spin_buffer.setSuffix(" km")
        self._spin_buffer.setToolTip("Half-width of the box around the gage.")
        self._button_locate = QPushButton("Look Up Gage")
        self._button_locate.clicked.connect(self._locate_gage)

        self._edits_bbox = []
        bbox_row = QHBoxLayout()
        for label in ("Min lon", "Min lat", "Max lon", "Max lat"):
            edit = QLineEdit()
            edit.setPlaceholderText(label)
            self._edits_bbox.append(edit)
            bbox_row.addWidget(QLabel(label + ":"))
            bbox_row.addWidget(edit)

        grid.addWidget(QLabel("Gage:"), 0, 0)
        grid.addWidget(self._edit_gage, 0, 1)
        grid.addWidget(QLabel("Buffer:"), 0, 2)
        grid.addWidget(self._spin_buffer, 0, 3)
        grid.addWidget(self._button_locate, 0, 4)
        grid.addLayout(bbox_row, 1, 0, 1, 5)
        grid.setColumnStretch(1, 1)

        controls = QHBoxLayout()
        self._combo_kind = QComboBox()
        for label, _ in KINDS:
            self._combo_kind.addItem(label)
        self._edit_out = QLineEdit()
        self._edit_out.setPlaceholderText("Download folder")
        button_browse = QPushButton("Browse")
        button_browse.clicked.connect(self._browse_out)
        self._button_search = QPushButton("Search")
        self._button_search.clicked.connect(self._search)
        self._button_download = QPushButton("Download Selected")
        self._button_download.clicked.connect(self._download)
        self._button_download.setEnabled(False)
        self._button_cancel = QPushButton("Cancel")
        self._button_cancel.clicked.connect(self._cancel)
        self._button_cancel.setEnabled(False)
        controls.addWidget(QLabel("Data:"))
        controls.addWidget(self._combo_kind)
        controls.addWidget(QLabel("Save to:"))
        controls.addWidget(self._edit_out, 1)
        controls.addWidget(button_browse)
        controls.addWidget(self._button_search)
        controls.addWidget(self._button_download)
        controls.addWidget(self._button_cancel)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["", "Product", "Year", "Size (MB)", "Format"])
        # Interactive throughout, so every divider can be dragged. Stretch on the
        # Product column would take the width of the rest but freeze its own edge.
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        self._table.setColumnWidth(0, 28)       # the tick box
        self._table.setColumnWidth(1, 320)      # Product
        self._table.setColumnWidth(2, 60)       # Year
        self._table.setColumnWidth(3, 90)       # Size
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setMinimumHeight(220)
        self._table.itemSelectionChanged.connect(self._preview_selected_row)

        table_buttons = QHBoxLayout()
        for label, checked in (("Select All", True), ("Select None", False)):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, state=checked: self._set_all_checked(state))
            table_buttons.addWidget(button)
        self._label_summary = QLabel("No search yet.")
        table_buttons.addWidget(self._label_summary)
        table_buttons.addStretch(1)

        self._button_view = QPushButton("Open in Viewer")
        self._button_view.setToolTip("Open the selected downloaded file in the NAIP, "
                                     "Elevation or Lidar tab.")
        self._button_view.clicked.connect(self._open_in_viewer)
        from downloader_viewers import ZoomableView
        self._preview = ZoomableView()
        self._preview.set_message("Select a downloaded product to preview it.")
        self._check_hillshade = QCheckBox("Hillshade elevation")
        self._check_hillshade.setChecked(True)
        self._check_hillshade.setToolTip("Shade elevation tiles by slope instead of showing "
                                         "raw values, which are unreadable as grey levels.")
        self._check_hillshade.toggled.connect(self._preview_selected_row)
        self._label_stats = QLabel("")
        self._label_stats.setWordWrap(True)

        preview_box = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.addWidget(self._button_view)
        preview_layout.addWidget(self._check_hillshade)
        preview_layout.addWidget(self._preview, 1)
        preview_layout.addWidget(self._label_stats)

        table_box = QGroupBox("Products")
        table_layout = QVBoxLayout(table_box)
        table_layout.addLayout(table_buttons)
        table_layout.addWidget(self._table)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(table_box)
        splitter.addWidget(preview_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)

        layout.addWidget(area)
        layout.addLayout(controls)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._progress)
        layout.addWidget(self._log)

    def _bind_settings(self):
        """Folders and choices come back the next time the plugin is opened.
        The search results themselves are not stored: they are a live query."""
        self._settings.bind(self._edit_gage, "gage")
        self._settings.bind(self._spin_buffer, "buffer_km")
        self._settings.bind(self._edit_out, "output_folder")
        self._settings.bind(self._combo_kind, "data_kind")
        self._settings.bind(self._check_hillshade, "preview_hillshade")
        for edit, key in zip(self._edits_bbox,
                             ("bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat")):
            self._settings.bind(edit, key)

    def _remember_bbox(self, bbox):
        for key, value in zip(("bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"),
                              bbox):
            self._settings[key] = f"{value:.6f}"
        self._settings.save()

    # ------------------------------------------------------------------------------------------------------------------
    # Area
    # ------------------------------------------------------------------------------------------------------------------
    def _browse_out(self):
        folder = QFileDialog.getExistingDirectory(self, "Download Folder", self._edit_out.text())
        if folder:
            self._edit_out.setText(folder)
            self._settings["output_folder"] = folder
            self._settings.save()

    def _locate_gage(self):
        gage = self._edit_gage.text().strip()
        if not gage:
            QMessageBox.warning(self, "Gage", "Enter an NWIS gage number.")
            return
        try:
            longitude, latitude, name = tnm.gage_location(gage)
        except Exception as err:
            QMessageBox.warning(self, "Gage", f"Could not look up gage {gage}:\n{err}")
            return
        bbox = tnm.box_around(longitude, latitude, self._spin_buffer.value())
        for edit, value in zip(self._edits_bbox, bbox):
            edit.setText(f"{value:.6f}")     # set in code, so store it explicitly
        self._remember_bbox(bbox)
        self._log.appendPlainText(f"{gage}: {name} at {latitude:.5f}, {longitude:.5f}")

    def _bbox(self):
        """The box from the four fields, or None with a message shown."""
        try:
            values = [float(edit.text()) for edit in self._edits_bbox]
        except ValueError:
            QMessageBox.warning(self, "Area", "Enter a bounding box, or look up a gage.")
            return None
        if values[0] >= values[2] or values[1] >= values[3]:
            QMessageBox.warning(self, "Area", "The minimum values must be below the maximum ones.")
            return None
        return values

    def _kind(self):
        return KINDS[self._combo_kind.currentIndex()][1]

    # ------------------------------------------------------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------------------------------------------------------
    def _search(self):
        bbox = self._bbox()
        if bbox is None:
            return
        self._button_search.setEnabled(False)
        self._button_download.setEnabled(False)
        self._table.setRowCount(0)
        self._log.appendPlainText("Searching The National Map\u2026")

        self._search_worker = SearchWorker(bbox, self._kind())
        self._search_worker.status.connect(self._log.appendPlainText)
        self._search_worker.finished.connect(self._on_search_finished)
        self._search_worker.start()

    def _on_search_finished(self, items, error):
        self._button_search.setEnabled(True)
        if error:
            self._log.appendPlainText(f"Search failed: {error}")
            QMessageBox.warning(self, "National Map", error)
            return

        self._items = items
        self._table.setRowCount(len(items))
        for row, item in enumerate(items):
            check = QTableWidgetItem()
            check.setFlags(check.flags() | Qt.ItemIsUserCheckable)
            check.setCheckState(Qt.Checked)
            self._table.setItem(row, 0, check)
            self._table.setItem(row, 1, QTableWidgetItem(str(item.get("title") or "")))
            self._table.setItem(row, 2, QTableWidgetItem(tnm.item_year(item)))
            size = (item.get("sizeInBytes") or 0) / 1e6
            size_item = QTableWidgetItem(f"{size:,.1f}")
            size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 3, size_item)
            self._table.setItem(row, 4, QTableWidgetItem(str(item.get("format") or "")))

        total = sum(item.get("sizeInBytes") or 0 for item in items) / 1e9
        years = sorted({tnm.item_year(item) for item in items})
        self._label_summary.setText(
            f"{len(items)} product(s), {total:,.2f} GB"
            + (f", years {years[0]} to {years[-1]}" if years else ""))
        self._button_download.setEnabled(bool(items))
        self._log.appendPlainText(self._label_summary.text())

    def _set_all_checked(self, checked):
        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self._table.rowCount()):
            self._table.item(row, 0).setCheckState(state)

    def _checked_items(self):
        return [item for row, item in enumerate(self._items)
                if self._table.item(row, 0).checkState() == Qt.Checked]

    # ------------------------------------------------------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------------------------------------------------------
    def _download(self):
        items = self._checked_items()
        if not items:
            QMessageBox.information(self, "National Map", "No products are ticked.")
            return
        out_dir = self._edit_out.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Download Folder", "Choose where to save the files.")
            return

        size = sum(item.get("sizeInBytes") or 0 for item in items) / 1e9
        if size > 1.0:
            answer = QMessageBox.question(
                self, "National Map",
                f"{len(items)} product(s), about {size:,.2f} GB. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if answer != QMessageBox.Yes:
                return

        self._button_download.setEnabled(False)
        self._button_search.setEnabled(False)
        self._button_cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        self._download_worker = DownloadWorker(items, out_dir, self._kind())
        self._download_worker.status.connect(self._log.appendPlainText)
        self._download_worker.progress.connect(self._on_progress)
        self._download_worker.file_done.connect(self._on_file_done)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.start()

    def _cancel(self):
        if self._download_worker is not None and self._download_worker.isRunning():
            self._button_cancel.setEnabled(False)
            self._log.appendPlainText("Cancelling\u2026")
            self._download_worker.cancel()

    def _on_progress(self, done, total):
        self._progress.setMaximum(total)
        self._progress.setValue(done)

    def _on_file_done(self, path):
        self._downloaded.append(path)
        row = len(self._downloaded) - 1
        # Mark the matching row so it is clear which products are on disk.
        name = os.path.basename(path)
        for row in range(self._table.rowCount()):
            title = self._table.item(row, 1).text()
            if title and title in name:
                self._table.item(row, 1).setToolTip(path)
                self._table.item(row, 1).setText(title + "  \u2713")
                break

    def _on_download_finished(self, downloaded, skipped, failed):
        self._progress.setVisible(False)
        self._button_download.setEnabled(True)
        self._button_search.setEnabled(True)
        self._button_cancel.setEnabled(False)
        self._log.appendPlainText(
            f"Done. {downloaded} downloaded, {skipped} already present, {failed} failed.")
        if self._downloaded:
            self._show_preview(self._downloaded[-1])

    # ------------------------------------------------------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------------------------------------------------------
    def _preview_selected_row(self):
        rows = {index.row() for index in self._table.selectedIndexes()}
        if not rows:
            return
        tooltip = self._table.item(min(rows), 1).toolTip()
        if tooltip and os.path.exists(tooltip):
            self._show_preview(tooltip)

    def _selected_path(self):
        """The downloaded file for the selected row, or "" when it is not on disk."""
        rows = {index.row() for index in self._table.selectedIndexes()}
        if not rows:
            return ""
        return self._table.item(min(rows), 1).toolTip()

    def _open_in_viewer(self):
        path = self._selected_path()
        if not path or not os.path.exists(path):
            QMessageBox.information(self, "Open in Viewer",
                                    "Select a row that has been downloaded.")
            return
        self.view_requested.emit(self._kind(), path)

    def _show_preview(self, path):
        """Aerial tiles as they are; elevation as a hillshade with its statistics."""
        if not path.lower().endswith(RASTER_EXTENSIONS):
            self._preview.set_message(f"No preview for {os.path.basename(path)}")
            self._label_stats.setText("")
            return

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            self._preview.set_message(f"Could not read {os.path.basename(path)}")
            self._label_stats.setText("")
            return

        scale = PREVIEW_LONG_EDGE / max(image.shape[0], image.shape[1])
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        single_band = image.ndim == 2
        if single_band and self._check_hillshade.isChecked():
            display, stats = _hillshade(image)
        elif single_band:
            display = cv2.normalize(image.astype(np.float32), None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            stats = _elevation_stats(image)
        else:
            display = image[:, :, :3]
            if display.dtype != np.uint8:
                display = cv2.normalize(display.astype(np.float32), None, 0, 255,
                                        cv2.NORM_MINMAX).astype(np.uint8)
            stats = f"{image.shape[1]} x {image.shape[0]} pixels, {image.shape[2]} band(s)"

        height, width = display.shape[:2]
        qimage = QImage(np.ascontiguousarray(display[:, :, ::-1]).data,
                        width, height, 3 * width, QImage.Format_RGB888)
        self._preview.set_pixmap(QPixmap.fromImage(qimage))
        self._label_stats.setText(f"{os.path.basename(path)}\n{stats}")


def _elevation_stats(band) -> str:
    values = band.astype(np.float64)
    values = values[np.isfinite(values)]
    # Voids are stored as a large negative sentinel in 3DEP rasters.
    values = values[values > -9000]
    if not values.size:
        return "No valid elevation values."
    percentiles = np.percentile(values, [5, 50, 95])
    return (f"Elevation: min {values.min():,.2f}, max {values.max():,.2f}, "
            f"median {percentiles[1]:,.2f}; 5th to 95th percentile "
            f"{percentiles[0]:,.2f} to {percentiles[2]:,.2f}; "
            f"relief {values.max() - values.min():,.2f}")


def _hillshade(band, azimuth_degrees=315.0, altitude_degrees=45.0):
    """Shaded relief from a single-band elevation raster, plus its statistics."""
    elevation = band.astype(np.float32)
    elevation[elevation < -9000] = np.nan
    filled = np.nan_to_num(elevation, nan=float(np.nanmedian(elevation)))

    dy, dx = np.gradient(filled)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    azimuth = math.radians(360.0 - azimuth_degrees + 90.0)
    altitude = math.radians(altitude_degrees)
    shaded = (math.sin(altitude) * np.cos(slope)
              + math.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect))
    shaded = np.clip(shaded, 0, 1)

    gray = (shaded * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), _elevation_stats(band)


# ======================================================================================================================
# Plugin container: download and the three viewers
# ======================================================================================================================
class AerialElevationPlugin(QWidget):
    """
    Downloading and viewing, in one window: the download tab fetches products,
    and the three viewers read what it saved. Opening a downloaded file in the
    matching viewer is one click from the download table.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        from PyQt5.QtWidgets import QTabWidget

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget(self)
        layout.addWidget(self._tabs)

        self.download_tab = AerialElevationDownloaderTab(self)
        self._tabs.addTab(self.download_tab, "Download")

        # The viewers are optional: a reader problem must not cost the download tab.
        try:
            from downloader_viewers import NaipViewerTab, DemViewerTab, LidarViewerTab
            # One settings object for the whole plugin, so the tabs do not each
            # write the file and overwrite one another's keys.
            settings = self.download_tab._settings
            self.naip_tab = NaipViewerTab(self, settings)
            self.dem_tab = DemViewerTab(self, settings)
            self.lidar_tab = LidarViewerTab(self, settings)
            self._tabs.addTab(self.naip_tab, "NAIP")
            self._tabs.addTab(self.dem_tab, "Elevation")
            self._tabs.addTab(self.lidar_tab, "Lidar")
            self.download_tab.view_requested.connect(self._open_in_viewer)
        except Exception as err:
            self.naip_tab = self.dem_tab = self.lidar_tab = None
            print(f"[aerial_elevation_downloader] Viewers unavailable: "
                  f"{type(err).__name__}: {err}")

        # Morphology works on the DEM open in the Elevation tab, over its region.
        try:
            if self.dem_tab is None:
                raise RuntimeError("the Elevation tab is unavailable")
            from morphology import MorphologyTab
            self.morphology_tab = MorphologyTab(self.dem_tab, self.download_tab,
                                                self.download_tab._settings, self)
            self._tabs.insertTab(self._tabs.indexOf(self.dem_tab) + 1,
                                 self.morphology_tab, "Morphology")
        except Exception as err:
            self.morphology_tab = None
            print(f"[aerial_elevation_downloader] Morphology unavailable: "
                  f"{type(err).__name__}: {err}")

        # Sentinel-2 shares the Download tab's area, so the gage lookup is set once.
        try:
            from sentinel2 import Sentinel2Tab
            self.sentinel2_tab = Sentinel2Tab(self.download_tab, self.download_tab._settings, self)
            self._tabs.insertTab(1, self.sentinel2_tab, "Sentinel-2")
        except Exception as err:
            self.sentinel2_tab = None
            print(f"[aerial_elevation_downloader] Sentinel-2 unavailable: "
                  f"{type(err).__name__}: {err}")

    def _open_in_viewer(self, kind, path):
        """Send a downloaded file to the tab that knows how to read it."""
        tab = {"naip": self.naip_tab, "dem": self.dem_tab, "lidar": self.lidar_tab}.get(kind)
        if tab is None:
            return
        tab.load(path)
        self._tabs.setCurrentWidget(tab)


# ======================================================================================================================
# Standalone use: python aerial_elevation_downloader.py
# ======================================================================================================================
def main(argv=None):
    import sys
    from PyQt5.QtWidgets import QApplication
    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(argv)
    window = AerialElevationPlugin()
    window.setWindowTitle(PLUGIN["title"])
    window.resize(*PLUGIN["size"])
    window.show()
    return app.exec_()


if __name__ == "__main__":
    import sys
    sys.exit(main())
