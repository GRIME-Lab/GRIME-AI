#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# sentinel2.py
#
# Sentinel-2 Level-2A (surface reflectance) scenes for a reach: search by area,
# dates and cloud cover, preview, and download the chosen bands.
#
# The catalogue is Element 84's Earth Search, a public STAC API over the
# Sentinel-2 archive on AWS. No account or key is needed, and the bands are
# cloud-optimised GeoTIFFs, so with rasterio installed only the part covering
# the reach is read rather than the whole 110 km tile.

import os
import json
import datetime

import numpy as np
import cv2
import requests

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
                             QLabel, QLineEdit, QPushButton, QSpinBox, QDateEdit,
                             QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView, QPlainTextEdit, QProgressBar, QFileDialog,
                             QMessageBox, QSplitter)

STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
USER_AGENT = "GRIME-AI sentinel2"
REQUEST_TIMEOUT = 60
PAGE_SIZE = 100

# Earth Search asset names, with what each band is and its resolution.
BANDS = [
    ("visual", "True color (RGB)", 10),
    ("blue",   "Blue, B02", 10),
    ("green",  "Green, B03", 10),
    ("red",    "Red, B04", 10),
    ("nir",    "NIR, B08", 10),
    ("swir16", "SWIR 1, B11", 20),
    ("swir22", "SWIR 2, B12", 20),
    ("scl",    "Scene classification", 20),
]
DEFAULT_BANDS = ("visual", "green", "nir", "swir16")   # enough for NDWI and MNDWI


# ======================================================================================================================
# Catalogue
# ======================================================================================================================
def search(bbox, start, end, max_cloud, limit=1000):
    """
    Scenes intersecting bbox [minlon, minlat, maxlon, maxlat] between start and
    end (datetime.date), with cloud cover below max_cloud percent. Newest first.
    """
    body = {
        "collections": [COLLECTION],
        "bbox": [float(v) for v in bbox],
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": float(max_cloud)}},
        "limit": PAGE_SIZE,
    }
    items, url, payload = [], STAC_SEARCH, body
    while url and len(items) < limit:
        response = requests.post(url, json=payload, headers={"User-Agent": USER_AGENT},
                                 timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        page = response.json()
        items += page.get("features", [])
        # STAC pages by a "next" link that carries its own body.
        following = next((link for link in page.get("links", []) if link.get("rel") == "next"),
                         None)
        if not following or not page.get("features"):
            break
        url = following.get("href")
        payload = following.get("body", payload)

    items.sort(key=lambda item: item["properties"].get("datetime", ""), reverse=True)
    return items[:limit]


def scene_date(item) -> str:
    return item["properties"].get("datetime", "")[:10]


def scene_tile(item) -> str:
    properties = item["properties"]
    return (properties.get("grid:code") or properties.get("s2:mgrs_tile")
            or properties.get("mgrs:utm_zone", "")) or ""


def thumbnail(item):
    """The scene's preview image as a BGR array, or None."""
    asset = item.get("assets", {}).get("thumbnail")
    if not asset:
        return None
    response = requests.get(asset["href"], headers={"User-Agent": USER_AGENT},
                            timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ======================================================================================================================
# Download
# ======================================================================================================================
def rasterio_available() -> bool:
    try:
        import rasterio        # noqa: F401  (presence check only)
        return True
    except ImportError:
        return False


def download_band(item, band, bbox, out_dir, progress=None, clip=True):
    """
    One band of one scene into out_dir/<date>_<tile>_<band>.tif. With rasterio
    and clip on, only the window covering bbox is read, which is a few hundred
    kilobytes rather than the whole tile. Returns the path written.
    """
    asset = item.get("assets", {}).get(band)
    if not asset:
        raise KeyError(f"The scene has no '{band}' band.")
    href = asset["href"]
    name = f"{scene_date(item)}_{scene_tile(item)}_{band}.tif"
    destination = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(destination):
        return destination

    if clip and rasterio_available():
        return _clip_to_reach(href, bbox, destination)

    # Without rasterio the whole tile comes down, which is what makes rasterio
    # worth installing for this.
    with requests.get(href, headers={"User-Agent": USER_AGENT}, stream=True,
                      timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        written = 0
        partial = destination + ".part"
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
                written += len(chunk)
                if progress and total:
                    progress(written, total)
        os.replace(partial, destination)
    return destination


def _clip_to_reach(href, bbox, destination):
    """Read only the window covering the reach from a cloud-optimised GeoTIFF."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as source:
            bounds = transform_bounds("EPSG:4326", source.crs, *bbox, densify_pts=21)
            window = from_bounds(*bounds, transform=source.transform)
            window = window.round_offsets().round_lengths()
            data = source.read(window=window, boundless=False)
            profile = source.profile.copy()
            profile.update(height=data.shape[1], width=data.shape[2],
                           transform=source.window_transform(window),
                           driver="GTiff", compress="deflate", tiled=False)
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
    with rasterio.open(destination, "w", **profile) as target:
        target.write(data)
    return destination


# ======================================================================================================================
# Workers
# ======================================================================================================================
class SearchWorker(QThread):
    finished = pyqtSignal(list, str)

    def __init__(self, bbox, start, end, max_cloud):
        super().__init__()
        self._args = (bbox, start, end, max_cloud)

    def run(self):
        try:
            self.finished.emit(search(*self._args), "")
        except Exception as err:
            self.finished.emit([], f"{type(err).__name__}: {err}")


class DownloadWorker(QThread):
    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    finished = pyqtSignal(int, int)

    def __init__(self, items, bands, bbox, out_dir, clip):
        super().__init__()
        self._items, self._bands = items, bands
        self._bbox, self._out_dir, self._clip = bbox, out_dir, clip
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        done = failed = 0
        jobs = [(item, band) for item in self._items for band in self._bands]
        for position, (item, band) in enumerate(jobs, start=1):
            if self._cancelled:
                self.status.emit(f"Cancelled after {position - 1} of {len(jobs)}.")
                break
            folder = os.path.join(self._out_dir, "sentinel2", scene_date(item))
            try:
                path = download_band(item, band, self._bbox, folder, clip=self._clip)
                self.status.emit(f"Saved {os.path.basename(path)}")
                done += 1
            except Exception as err:
                self.status.emit(f"Failed {scene_date(item)} {band}: {type(err).__name__}: {err}")
                failed += 1
            self.progress.emit(position, len(jobs))
        self.finished.emit(done, failed)


class ThumbnailWorker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, item):
        super().__init__()
        self._item = item

    def run(self):
        try:
            self.finished.emit(thumbnail(self._item), "")
        except Exception as err:
            self.finished.emit(None, f"{type(err).__name__}: {err}")


# ======================================================================================================================
# Tab
# ======================================================================================================================
class Sentinel2Tab(QWidget):
    """
    Search, preview and download Sentinel-2 scenes for the area set on the
    Download tab, so the gage lookup and bounding box are shared.
    """

    def __init__(self, area_source, settings=None, parent=None):
        super().__init__(parent)
        self._area_source = area_source           # the Download tab, for its bbox
        self._settings = settings
        self._items = []
        self._search_worker = None
        self._download_worker = None
        self._thumbnail_worker = None
        self._build_ui()
        self._bind_settings()

    # ------------------------------------------------------------------------------------------------------------------
    def _build_ui(self):
        from PyQt5.QtWidgets import QFormLayout, QScrollArea

        def form(title):
            box = QGroupBox(title)
            layout = QFormLayout(box)
            layout.setLabelAlignment(Qt.AlignLeft)
            layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
            layout.setHorizontalSpacing(6)
            layout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
            return box, layout

        # ── Search ──────────────────────────────────────────────────────────
        today = QDate.currentDate()
        self._date_start = QDateEdit(today.addMonths(-3))
        self._date_end = QDateEdit(today)
        for edit in (self._date_start, self._date_end):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd")
        self._spin_cloud = QSpinBox()
        self._spin_cloud.setRange(0, 100)
        self._spin_cloud.setValue(20)
        self._spin_cloud.setSuffix(" %")
        self._spin_cloud.setToolTip("Scenes cloudier than this are left out.")
        self._button_search = QPushButton("Search")
        self._button_search.clicked.connect(self._search)
        self._label_area = QLabel("Area: set on the Download tab.")
        self._label_area.setWordWrap(True)

        search_box, search_form = form("Search")
        search_form.addRow("From:", self._date_start)
        search_form.addRow("To:", self._date_end)
        search_form.addRow("Max cloud:", self._spin_cloud)
        search_form.addRow(self._button_search)
        search_form.addRow(self._label_area)

        # ── Bands ───────────────────────────────────────────────────────────
        bands_box = QGroupBox("Bands to download")
        bands_layout = QVBoxLayout(bands_box)
        bands_layout.setSpacing(2)
        self._band_checks = {}
        for key, label, resolution in BANDS:
            check = QCheckBox(f"{label} ({resolution} m)")
            check.setChecked(key in DEFAULT_BANDS)
            bands_layout.addWidget(check)
            self._band_checks[key] = check

        # ── Output ──────────────────────────────────────────────────────────
        self._edit_out = QLineEdit()
        self._edit_out.setPlaceholderText("Download folder")
        button_browse = QPushButton("Browse")
        button_browse.clicked.connect(self._browse)
        self._check_clip = QCheckBox("Clip to the area")
        self._check_clip.setChecked(True)
        self._check_clip.setToolTip(
            "Read only the part of each band covering the area. Needs rasterio; "
            "without it the whole 110 km tile is downloaded.")
        self._button_download = QPushButton("Download Selected")
        self._button_download.clicked.connect(self._download)
        self._button_download.setEnabled(False)
        self._button_cancel = QPushButton("Cancel")
        self._button_cancel.clicked.connect(self._cancel)
        self._button_cancel.setEnabled(False)

        output_box = QGroupBox("Save to")
        output_layout = QVBoxLayout(output_box)
        output_layout.setSpacing(4)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self._edit_out, 1)
        folder_row.addWidget(button_browse)
        output_layout.addLayout(folder_row)
        output_layout.addWidget(self._check_clip)
        buttons = QHBoxLayout()
        buttons.addWidget(self._button_download)
        buttons.addWidget(self._button_cancel)
        output_layout.addLayout(buttons)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(search_box)
        panel_layout.addWidget(bands_box)
        panel_layout.addWidget(output_box)
        panel_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(panel.sizeHint().width() + 20)

        # ── Scenes and preview ─────────────────────────────────────────────
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["", "Date", "Cloud %", "Tile"])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        self._table.setColumnWidth(0, 28)
        self._table.setColumnWidth(1, 100)
        self._table.setColumnWidth(2, 64)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.itemSelectionChanged.connect(self._preview_selected)

        table_buttons = QHBoxLayout()
        for label, state in (("Select All", True), ("Select None", False)):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, checked=state: self._set_all(checked))
            table_buttons.addWidget(button)
        self._label_summary = QLabel("No search yet.")
        table_buttons.addWidget(self._label_summary)
        table_buttons.addStretch(1)

        table_box = QGroupBox("Scenes")
        table_layout = QVBoxLayout(table_box)
        table_layout.addLayout(table_buttons)
        table_layout.addWidget(self._table)

        self._preview = QLabel("Select a scene to preview it.")
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumSize(220, 200)
        self._preview.setWordWrap(True)
        preview_box = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.addWidget(self._preview)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(table_box)
        splitter.addWidget(preview_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(110)
        if not rasterio_available():
            self._log.appendPlainText(
                "rasterio is not installed, so each band downloads as a whole tile "
                "(100 to 200 MB). pip install rasterio to fetch only the area.")

        right = QVBoxLayout()
        right.addWidget(splitter, 1)
        right.addWidget(self._progress)
        right.addWidget(self._log)

        layout = QHBoxLayout(self)
        layout.addWidget(scroll)
        layout.addLayout(right, 1)

    def _bind_settings(self):
        if self._settings is None or not hasattr(self._settings, "bind"):
            return
        self._settings.bind(self._spin_cloud, "Sentinel2.max_cloud")
        self._settings.bind(self._edit_out, "Sentinel2.output_folder")
        self._settings.bind(self._check_clip, "Sentinel2.clip")
        for key, check in self._band_checks.items():
            self._settings.bind(check, f"Sentinel2.band.{key}")

    # ------------------------------------------------------------------------------------------------------------------
    def _bbox(self):
        """The area from the Download tab, where the gage lookup lives."""
        try:
            values = [float(edit.text()) for edit in self._area_source._edits_bbox]
        except (ValueError, AttributeError):
            return None
        if values[0] >= values[2] or values[1] >= values[3]:
            return None
        return values

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Download Folder", self._edit_out.text())
        if folder:
            self._edit_out.setText(folder)
            self._edit_out.editingFinished.emit()

    # ------------------------------------------------------------------------------------------------------------------
    def _search(self):
        bbox = self._bbox()
        if bbox is None:
            QMessageBox.warning(self, "Sentinel-2", "Set the area on the Download tab first: "
                                                    "look up a gage or enter a bounding box.")
            return
        self._label_area.setText("Area: " + ", ".join(f"{v:.4f}" for v in bbox))
        start = self._date_start.date().toPyDate()
        end = self._date_end.date().toPyDate()
        if start > end:
            QMessageBox.warning(self, "Sentinel-2", "The start date is after the end date.")
            return

        self._button_search.setEnabled(False)
        self._table.setRowCount(0)
        self._log.appendPlainText(f"Searching {start} to {end}, cloud under "
                                  f"{self._spin_cloud.value()}%\u2026")
        self._search_worker = SearchWorker(bbox, start, end, self._spin_cloud.value())
        self._search_worker.finished.connect(self._on_search_finished)
        self._search_worker.start()

    def _on_search_finished(self, items, error):
        self._button_search.setEnabled(True)
        if error:
            self._log.appendPlainText(f"Search failed: {error}")
            QMessageBox.warning(self, "Sentinel-2", error)
            return
        self._items = items
        self._table.setRowCount(len(items))
        for row, item in enumerate(items):
            check = QTableWidgetItem()
            check.setFlags(check.flags() | Qt.ItemIsUserCheckable)
            check.setCheckState(Qt.Checked)
            self._table.setItem(row, 0, check)
            self._table.setItem(row, 1, QTableWidgetItem(scene_date(item)))
            cloud = item["properties"].get("eo:cloud_cover")
            cloud_item = QTableWidgetItem("" if cloud is None else f"{cloud:,.1f}")
            cloud_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 2, cloud_item)
            self._table.setItem(row, 3, QTableWidgetItem(scene_tile(item)))
        self._label_summary.setText(f"{len(items)} scene(s)")
        self._button_download.setEnabled(bool(items))
        self._log.appendPlainText(f"{len(items)} scene(s) found.")

    def _set_all(self, checked):
        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self._table.rowCount()):
            self._table.item(row, 0).setCheckState(state)

    # ------------------------------------------------------------------------------------------------------------------
    def _preview_selected(self):
        rows = {index.row() for index in self._table.selectedIndexes()}
        if not rows or not self._items:
            return
        item = self._items[min(rows)]
        self._preview.setText("Loading preview\u2026")
        self._thumbnail_worker = ThumbnailWorker(item)
        self._thumbnail_worker.finished.connect(self._show_thumbnail)
        self._thumbnail_worker.start()

    def _show_thumbnail(self, image, error):
        if error or image is None:
            self._preview.setText(f"No preview: {error}" if error else "No preview.")
            return
        height, width = image.shape[:2]
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        pixmap = QPixmap.fromImage(QImage(rgb.data, width, height, 3 * width,
                                          QImage.Format_RGB888))
        self._preview.setPixmap(pixmap.scaled(self._preview.size(), Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation))

    # ------------------------------------------------------------------------------------------------------------------
    def _download(self):
        items = [item for row, item in enumerate(self._items)
                 if self._table.item(row, 0).checkState() == Qt.Checked]
        bands = [key for key, check in self._band_checks.items() if check.isChecked()]
        out_dir = self._edit_out.text().strip()
        bbox = self._bbox()
        if not items or not bands:
            QMessageBox.information(self, "Sentinel-2", "Tick at least one scene and one band.")
            return
        if not out_dir:
            QMessageBox.warning(self, "Sentinel-2", "Choose where to save the files.")
            return
        clip = self._check_clip.isChecked()
        if clip and not rasterio_available():
            answer = QMessageBox.question(
                self, "Sentinel-2",
                "rasterio is not installed, so clipping is not possible and each band "
                f"downloads as a whole tile: {len(items) * len(bands)} file(s) of 100 to "
                "200 MB. Continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return

        self._button_download.setEnabled(False)
        self._button_cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._download_worker = DownloadWorker(items, bands, bbox, out_dir, clip)
        self._download_worker.status.connect(self._log.appendPlainText)
        self._download_worker.progress.connect(self._on_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.start()

    def _cancel(self):
        if self._download_worker is not None and self._download_worker.isRunning():
            self._button_cancel.setEnabled(False)
            self._download_worker.cancel()

    def _on_progress(self, done, total):
        self._progress.setMaximum(total)
        self._progress.setValue(done)

    def _on_download_finished(self, done, failed):
        self._progress.setVisible(False)
        self._button_download.setEnabled(True)
        self._button_cancel.setEnabled(False)
        self._log.appendPlainText(f"Done: {done} band file(s) saved, {failed} failed.")
