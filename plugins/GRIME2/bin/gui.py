from __future__ import annotations

import os
import sys
import csv
import json
from pathlib import Path

import cv2
import numpy as np
# Qt binding is chosen by qtcompat so this same file runs standalone under
# PySide6 and as a GRIME AI plugin tab under PyQt5.
from grime2py.qtcompat import QtCore, QtGui, QtWidgets, exec_app

from grime2py import waterline
from grime2py.core import (
    CalibExecutive,
    CalibrationConfig,
    content_bottom,
    detect_water_level,
    form_octagon_calib_json_string,
)
from grime2py.cli import (
    IMAGE_EXTENSIONS,
    _detect_image,
    _make_gif,
    _read_metadata,
    _search_poly_for_image,
)

THUMB_W = 132
THUMB_H = 88

# Persistent settings. Kept as plain JSON next to the user's documents so
# it is easy to find, read and delete by hand. Every read and write is
# best-effort: a locked or unwritable folder degrades to no persistence
# rather than stopping the application from starting.
SETTINGS_DIR = os.path.join(os.path.expanduser("~"), "Documents", "GRIME2")
SETTINGS_PATH = os.path.join(SETTINGS_DIR, "settings.json")


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(data: dict) -> bool:
    try:
        os.makedirs(SETTINGS_DIR, exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        return True
    except Exception:
        return False


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GRIME2 Python")
        self.resize(1200, 800)
        self.current_image = None
        self.image_path = None
        self.calibration_config = None
        self.calibration = CalibExecutive()
        self.last_waterline = None
        self.image_paths: list[Path] = []
        self.folder_path: str = ""
        self.calibration_path: str = ""
        self.truth_row: int | None = None       # shift-clicked reference row
        self.settings = load_settings()
        self._thumb_token = 0
        # None = fit to window (the original behaviour), float = scale factor
        # where 1.0 is one screen pixel per image pixel.
        self._zoom: float | None = None
        self._pixmap: QtGui.QPixmap | None = None
        self._last_overlay = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)

        # ---- File menu ----
        file_menu = self.menuBar().addMenu("&File")

        open_action = QtGui.QAction("&Open Image...", self)
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        open_folder_action = QtGui.QAction("Open &Folder...", self)
        open_folder_action.setShortcut("Ctrl+Shift+O")
        open_folder_action.triggered.connect(self.browse_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()
        exit_action = QtGui.QAction("E&xit", self)
        exit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ---- View menu ----
        view_menu = self.menuBar().addMenu("&View")

        zoom_in_action = QtGui.QAction("Zoom &In", self)
        zoom_in_action.setShortcut(QtGui.QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QtGui.QAction("Zoom &Out", self)
        zoom_out_action.setShortcut(QtGui.QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)

        view_menu.addSeparator()
        zoom_reset_action = QtGui.QAction("Actual Size (&100%)", self)
        zoom_reset_action.setShortcut("Ctrl+0")
        zoom_reset_action.triggered.connect(self.zoom_reset)
        view_menu.addAction(zoom_reset_action)

        zoom_fit_action = QtGui.QAction("&Fit to Window", self)
        zoom_fit_action.setShortcut("Ctrl+9")
        zoom_fit_action.triggered.connect(self.zoom_fit)
        view_menu.addAction(zoom_fit_action)

        view_menu.addSeparator()
        self.chk_auto_find = QtGui.QAction("&Auto Find Line", self)
        self.chk_auto_find.setCheckable(True)
        self.chk_auto_find.setChecked(True)
        self.chk_auto_find.setStatusTip(
            "Run Find Line automatically when a filmstrip image is selected.")
        view_menu.addAction(self.chk_auto_find)

        self.chk_mirror = QtGui.QAction("&Mirror Re-rank", self)
        self.chk_mirror.setCheckable(True)
        self.chk_mirror.setChecked(waterline.V2_MIRROR_RERANK)
        self.chk_mirror.setStatusTip(
            "Prefer the most reflection-symmetric edge. Helps where heavy "
            "biofouling and its reflection straddle the waterline; can move "
            "an already-correct answer on clean frames.")
        self.chk_mirror.toggled.connect(self._set_mirror)
        view_menu.addAction(self.chk_mirror)

        view_menu.addSeparator()
        clear_truth_action = QtGui.QAction("Clear &Truth Row", self)
        clear_truth_action.setStatusTip(
            "Shift+click the image to mark the true waterline; this clears it.")
        clear_truth_action.triggered.connect(self.clear_truth_row)
        view_menu.addAction(clear_truth_action)

        # ---- Calibrate menu ----
        calib_menu = self.menuBar().addMenu("&Calibrate")

        calibrate_action = QtGui.QAction("&Calibrate", self)
        calibrate_action.triggered.connect(self.calibrate)
        calib_menu.addAction(calibrate_action)

        calib_menu.addSeparator()
        load_calibration_action = QtGui.QAction("&Load Calibration...", self)
        load_calibration_action.triggered.connect(self.load_calibration)
        calib_menu.addAction(load_calibration_action)

        save_calibration_action = QtGui.QAction("&Save Calibration...", self)
        save_calibration_action.triggered.connect(self.save_calibration)
        calib_menu.addAction(save_calibration_action)

        # ---- top-level actions on the menu bar ----
        find_action = QtGui.QAction("Find &Line", self)
        find_action.triggered.connect(self.find_line)
        self.menuBar().addAction(find_action)

        process_folder_action = QtGui.QAction("&Process Folder", self)
        process_folder_action.triggered.connect(self.process_folder)
        self.menuBar().addAction(process_folder_action)

        metadata_action = QtGui.QAction("Show &Metadata", self)
        metadata_action.triggered.connect(self.show_metadata)
        self.menuBar().addAction(metadata_action)

        gif_action = QtGui.QAction("Make &GIF", self)
        gif_action.triggered.connect(self.make_gif)
        self.menuBar().addAction(gif_action)

        # Zoom readout lives in the status bar. It used to sit on a toolbar
        # of its own directly under the menu, where an unloaded image showed
        # as two stray hyphens across an otherwise empty bar.
        self.lbl_cursor = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.lbl_cursor)
        self.lbl_count = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.lbl_count)
        self.lbl_zoom = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.lbl_zoom)

        content_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(content_layout)

        # Mouse tracking so the status bar can report the image-space row
        # under the cursor, and shift-click can mark a known true row.
        self.image_label = QtWidgets.QLabel("No image loaded")
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background: #111; color: #fff;")
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                       QtWidgets.QSizePolicy.Policy.Ignored)

        self.image_scroll = QtWidgets.QScrollArea()
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_scroll.setMinimumSize(800, 600)
        self.image_scroll.setStyleSheet("border: 1px solid #999; background: #111;")
        # Ctrl+wheel zooms; plain wheel scrolls as usual.
        self.image_scroll.viewport().installEventFilter(self)
        content_layout.addWidget(self.image_scroll, 1)

        controls = QtWidgets.QGroupBox("Calibration")
        controls_layout = QtWidgets.QFormLayout(controls)
        self.facet_length_edit = QtWidgets.QLineEdit("0.599")
        self.zero_offset_edit = QtWidgets.QLineEdit("3.5")
        controls_layout.addRow("Facet length", self.facet_length_edit)
        controls_layout.addRow("Zero offset", self.zero_offset_edit)

        self.target_edits = self._add_coordinate_group(controls_layout, "Target ROI", (0, 0, 0, 0))
        self.search_edits = self._add_coordinate_group(
            controls_layout,
            "Search ROI",
            ((0, 0), (0, 0), (0, 0), (0, 0)),
            point_count=4,
        )
        # A list box rather than a label so the result can be selected and
        # copied. A QLabel renders the text but offers no way to get it out.
        self.result_summary = QtWidgets.QListWidget()
        self.result_summary.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.result_summary.setMinimumHeight(120)
        self.result_summary.setWordWrap(True)
        self.result_summary.setAlternatingRowColors(True)
        self.result_summary.setToolTip(
            "Select rows and press Ctrl+C, or right-click to copy.")
        self.result_summary.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.ActionsContextMenu)

        copy_selected = QtGui.QAction("Copy", self.result_summary)
        copy_selected.setShortcut(QtGui.QKeySequence.StandardKey.Copy)
        copy_selected.setShortcutContext(
            QtCore.Qt.ShortcutContext.WidgetShortcut)
        copy_selected.triggered.connect(self._copy_result)
        self.result_summary.addAction(copy_selected)

        copy_all = QtGui.QAction("Copy All", self.result_summary)
        copy_all.triggered.connect(lambda: self._copy_result(all_rows=True))
        self.result_summary.addAction(copy_all)

        self._set_result(["No result"])
        controls_layout.addRow("Result", self.result_summary)
        content_layout.addWidget(controls)

        # ---- filmstrip ----
        self.strip = QtWidgets.QListWidget()
        self.strip.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.strip.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        self.strip.setWrapping(False)
        self.strip.setMovement(QtWidgets.QListView.Movement.Static)
        self.strip.setIconSize(QtCore.QSize(THUMB_W, THUMB_H))
        self.strip.setFixedHeight(THUMB_H + 46)
        self.strip.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.strip.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.strip.currentItemChanged.connect(self._on_strip_changed)
        main_layout.addWidget(self.strip)

        # Restore the previous session once the window exists, so any
        # message from loading has somewhere to be logged.
        QtCore.QTimer.singleShot(0, self._restore_session)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)
        main_layout.addWidget(self.log)

    # ------------------------------------------------------------------
    # Folder browsing and filmstrip
    # ------------------------------------------------------------------
    def _restore_session(self) -> None:
        """Re-open the folder and calibration from the previous session.

        The calibration is applied FIRST so that if a folder is restored
        too, auto-find on the first image already has a real calibration
        rather than the fabricated image-fraction default.
        Anything that no longer exists is skipped quietly; a stale path is
        not an error worth interrupting startup for.
        """
        calibration = self.settings.get("last_calibration", "")
        if calibration and os.path.isfile(calibration):
            self._apply_calibration_file(calibration)
        elif calibration:
            self.log_message(f"Previous calibration no longer exists: {calibration}")

        folder = self.settings.get("last_folder", "")
        if folder and os.path.isdir(folder):
            self.load_folder(folder)
        elif folder:
            self.log_message(f"Previous folder no longer exists: {folder}")

    def _apply_calibration_file(self, path: str) -> bool:
        """Load a calibration JSON into the model. Shared by the menu action
        and by session restore, so both behave identically."""
        try:
            self.calibration_config = CalibrationConfig.from_json(
                Path(path).read_text(encoding="utf-8"))
            self._sync_controls(self.calibration_config)
            points = self.calibration_config.pixel_to_world_points
            if len(points) >= 4:
                self.calibration.calibrate_from_points(
                    [point[:2] for point in points],
                    [point[2:] for point in points],
                    self.calibration_config.image_size,
                )
            self.calibration_path = str(path)
            self.settings["last_calibration"] = self.calibration_path
            save_settings(self.settings)
            self.log_message(f"Loaded calibration: {path}")
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            self.log_message(f"Could not load calibration: {error}")
            return False

    def browse_folder(self) -> None:
        start = self.folder_path or str(Path.home())
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select image folder", start)
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder: str) -> None:
        """Populate the filmstrip from a folder of images."""
        path = Path(folder)
        if not path.is_dir():
            self.log_message(f"Not a folder: {folder}")
            return
        self.folder_path = str(path)
        self.settings["last_folder"] = self.folder_path
        save_settings(self.settings)
        self.image_paths = sorted(
            entry for entry in path.iterdir()
            if entry.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.strip.clear()
        self._thumb_token += 1
        self.lbl_count.setText(f"{len(self.image_paths)} images")
        if not self.image_paths:
            self.log_message(f"No images found in {path}")
            return
        for index, item_path in enumerate(self.image_paths):
            item = QtWidgets.QListWidgetItem(str(index + 1))
            item.setToolTip(item_path.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(item_path))
            item.setSizeHint(QtCore.QSize(THUMB_W + 12, THUMB_H + 34))
            self.strip.addItem(item)
        self.log_message(f"Loaded folder: {path} ({len(self.image_paths)} images)")
        # Thumbnails load in the background so a large folder does not
        # freeze the window; the first image is shown immediately.
        QtCore.QTimer.singleShot(0, lambda: self._load_thumbnails(self._thumb_token, 0))
        self.strip.setCurrentRow(0)

    def _load_thumbnails(self, token: int, start: int, batch: int = 8) -> None:
        if token != self._thumb_token:
            return                      # a newer folder was loaded
        end = min(start + batch, self.strip.count())
        for index in range(start, end):
            item = self.strip.item(index)
            if item is None:
                continue
            source = item.data(QtCore.Qt.ItemDataRole.UserRole)
            pixmap = QtGui.QPixmap(source)
            if not pixmap.isNull():
                item.setIcon(QtGui.QIcon(pixmap.scaled(
                    THUMB_W, THUMB_H,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation)))
        if end < self.strip.count():
            QtCore.QTimer.singleShot(
                0, lambda: self._load_thumbnails(token, end, batch))

    def _set_truth_row(self, row: int) -> None:
        """Mark a known-correct row. Drawn in green, and the error against
        the detected row is reported, so a frame can be assessed without
        leaving the application."""
        self.truth_row = int(row)
        message = "Truth row y=%d" % self.truth_row
        if self.last_waterline is not None:
            message += " | detected y=%.0f | error %+.0f px" % (
                self.last_waterline, self.last_waterline - self.truth_row)
        self.log_message(message)
        rows = [self.result_summary.item(i).text()
                for i in range(self.result_summary.count())
                if not self.result_summary.item(i).text().startswith("Truth row")]
        self._set_result(rows + [message])
        self._redraw()

    def clear_truth_row(self) -> None:
        self.truth_row = None
        self._redraw()

    def _redraw(self) -> None:
        if self.current_image is not None:
            self._display_image(self.current_image, overlay=self._last_overlay)

    def _set_result(self, lines) -> None:
        """Replace the result list. Long strings are split on '; ' so each
        fact lands on its own selectable row."""
        self.result_summary.clear()
        for line in lines:
            for part in str(line).split("; "):
                part = part.strip()
                if part:
                    self.result_summary.addItem(part)

    def _copy_result(self, all_rows: bool = False) -> None:
        if all_rows:
            items = [self.result_summary.item(i)
                     for i in range(self.result_summary.count())]
        else:
            items = self.result_summary.selectedItems()
            if not items:
                items = [self.result_summary.item(i)
                         for i in range(self.result_summary.count())]
        text = "\n".join(item.text() for item in items)
        QtWidgets.QApplication.clipboard().setText(text)
        self.statusBar().showMessage("Copied %d line(s)" % len(items), 2000)

    def _set_mirror(self, enabled: bool) -> None:
        """Toggle the mirror re-rank. It is a module-level switch in
        waterline.py so the CLI and the GUI share one implementation; this
        flips it and re-runs the current frame."""
        waterline.V2_MIRROR_RERANK = bool(enabled)
        self.log_message("Mirror re-rank %s" % ("enabled" if enabled else "disabled"))
        if self.current_image is not None:
            self.find_line()

    def _on_strip_changed(self, current, _previous) -> None:
        if current is None:
            return
        self.open_image_path(current.data(QtCore.Qt.ItemDataRole.UserRole))
        if self.chk_auto_find.isChecked() and self.current_image is not None:
            self.find_line()

    def open_image_path(self, path: str) -> None:
        """Load one image and redraw. Shared by the filmstrip and Open Image."""
        image = cv2.imread(str(path))
        if image is None:
            self.log_message(f"Could not load image: {path}")
            return
        self.current_image = image
        self.image_path = str(path)
        self.truth_row = None           # a marker belongs to one frame only
        self.log_message(f"Loaded: {path}")
        self._display_image(image, overlay=None)

    def log_message(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _add_coordinate_group(self, layout: QtWidgets.QFormLayout, title: str, values, point_count: int = 1):
        group = QtWidgets.QGroupBox(title)
        group_layout = QtWidgets.QGridLayout(group)
        edits = []
        if point_count == 1:
            values = [values]
        for index, point in enumerate(values):
            row = index
            labels = ("x", "y", "w", "h") if len(point) == 4 else ("x", "y")
            point_edits = []
            for column, (label, value) in enumerate(zip(labels, point)):
                edit = QtWidgets.QLineEdit(str(value))
                edit.setMaximumWidth(70)
                group_layout.addWidget(QtWidgets.QLabel(f"{index + 1}{label}"), row, column * 2)
                group_layout.addWidget(edit, row, column * 2 + 1)
                point_edits.append(edit)
            edits.append(point_edits)
        layout.addRow(group)
        return edits[0] if point_count == 1 else edits

    @staticmethod
    def _read_numbers(edits):
        return tuple(int(edit.text()) for edit in edits)

    def _make_calibration_config(self, image: np.ndarray) -> CalibrationConfig:
        h, w = image.shape[:2]
        try:
            facet_length = float(self.facet_length_edit.text())
            zero_offset = float(self.zero_offset_edit.text())
            target = self._read_numbers(self.target_edits)
            search = tuple(self._read_numbers(edits) for edits in self.search_edits)
            if target[2] <= 0 or target[3] <= 0:
                raise ValueError
            if any(point[0] == 0 and point[1] == 0 for point in search):
                raise ValueError
            return CalibrationConfig(
                calib_type="Octagon",
                facet_length=facet_length,
                zero_offset=zero_offset,
                target_roi=target,
                search_poly=search,
            )
        except (TypeError, ValueError):
            # Previously this substituted a fabricated ROI silently, so the
            # user believed their entered numbers had been used.
            self.log_message("Calibration fields are incomplete or invalid;"
                             " using image-fraction defaults.")
            return CalibrationConfig(
                calib_type="Octagon",
                facet_length=0.599,
                zero_offset=3.5,
                target_roi=(int(w * 0.1), int(h * 0.1), int(w * 0.2), int(h * 0.2)),
                search_poly=((int(w * 0.15), int(h * 0.65)), (int(w * 0.6), int(h * 0.65)),
                             (int(w * 0.15), content_bottom(image)), (int(w * 0.6), content_bottom(image))),
            )

    def _sync_controls(self, config: CalibrationConfig) -> None:
        self.facet_length_edit.setText(str(config.facet_length))
        self.zero_offset_edit.setText(str(config.zero_offset))
        for edit, value in zip(self.target_edits, config.target_roi):
            edit.setText(str(value))
        for edits, point in zip(self.search_edits, config.search_poly):
            for edit, value in zip(edits, point):
                edit.setText(str(value))

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    ZOOM_STEP = 1.25
    ZOOM_MIN = 0.05
    ZOOM_MAX = 16.0
    # Hard memory ceiling on the scaled pixmap. A 2048x1152 frame at 16x is
    # 32768x18432, about 2.4 GB, which kills the process. The factor limit
    # alone is not enough because the cost depends on the source size.
    MAX_SCALED_PIXELS = 40_000_000

    def _current_scale(self) -> float:
        """The factor actually in use, resolving 'fit' against the viewport."""
        if self._zoom is not None:
            return self._zoom
        if self._pixmap is None or self._pixmap.isNull():
            return 1.0
        viewport = self.image_scroll.viewport().size()
        return min(viewport.width() / self._pixmap.width(),
                   viewport.height() / self._pixmap.height())

    def zoom_in(self) -> None:
        self._set_zoom(self._current_scale() * self.ZOOM_STEP)

    def zoom_out(self) -> None:
        self._set_zoom(self._current_scale() / self.ZOOM_STEP)

    def zoom_reset(self) -> None:
        """100% -- one screen pixel per image pixel."""
        self._set_zoom(1.0)

    def zoom_fit(self) -> None:
        """Scale to the window, which is what the view did before zoom existed."""
        self._zoom = None
        self._apply_zoom()

    def _max_scale(self) -> float:
        """Largest factor that keeps the scaled pixmap inside the budget."""
        if self._pixmap is None or self._pixmap.isNull():
            return self.ZOOM_MAX
        source_pixels = float(self._pixmap.width() * self._pixmap.height())
        if source_pixels <= 0:
            return self.ZOOM_MAX
        return min(self.ZOOM_MAX,
                   float(np.sqrt(self.MAX_SCALED_PIXELS / source_pixels)))

    def _set_zoom(self, factor: float) -> None:
        limit = self._max_scale()
        wanted = float(factor)
        self._zoom = max(self.ZOOM_MIN, min(limit, wanted))
        if wanted > limit + 1e-9:
            self.lbl_zoom.setToolTip(
                "Zoom limited to %.0f%% for this image size (memory cap)."
                % (limit * 100.0))
        else:
            self.lbl_zoom.setToolTip("")
        self._apply_zoom()

    def _apply_zoom(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            self.lbl_zoom.setText("")
            return
        scale = self._current_scale()
        width = max(1, int(round(self._pixmap.width() * scale)))
        height = max(1, int(round(self._pixmap.height() * scale)))
        # Smooth when shrinking, nearest when magnifying, so individual
        # pixels stay crisp at high zoom.
        mode = (QtCore.Qt.TransformationMode.SmoothTransformation if scale < 1.0
                else QtCore.Qt.TransformationMode.FastTransformation)
        self.image_label.setPixmap(self._pixmap.scaled(
            width, height, QtCore.Qt.AspectRatioMode.KeepAspectRatio, mode))
        self.image_label.resize(width, height)
        self.lbl_zoom.setText("Zoom: fit (%.0f%%)" % (scale * 100.0)
                              if self._zoom is None
                              else "Zoom: %.0f%%" % (scale * 100.0))

    def _to_image_point(self, position) -> tuple[int, int] | None:
        """Label coordinates -> image pixel coordinates, undoing the zoom."""
        if self._pixmap is None or self._pixmap.isNull():
            return None
        scale = self._current_scale()
        if scale <= 0:
            return None
        # The label is sized to the scaled pixmap, so there is no centring
        # offset to remove; the pixmap fills it exactly.
        x = int(position.x() / scale)
        y = int(position.y() / scale)
        if not (0 <= x < self._pixmap.width() and 0 <= y < self._pixmap.height()):
            return None
        return x, y

    def eventFilter(self, obj, event):
        if obj is self.image_label:
            if event.type() == QtCore.QEvent.Type.MouseMove:
                point = self._to_image_point(event.pos())
                self.lbl_cursor.setText(
                    "x=%d  y=%d" % point if point else "")
            elif (event.type() == QtCore.QEvent.Type.MouseButtonPress
                  and event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                point = self._to_image_point(event.pos())
                if point is not None:
                    self._set_truth_row(point[1])
                    return True
            return False
        if (obj is self.image_scroll.viewport()
                and event.type() == QtCore.QEvent.Type.Wheel
                and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
            delta = event.angleDelta().y()
            if delta:
                self.zoom_in() if delta > 0 else self.zoom_out()
            return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._zoom is None:
            self._apply_zoom()          # 'fit' tracks the window

    def _display_image(
        self,
        image: np.ndarray,
        overlay: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        # np.ascontiguousarray + copy(): QImage does not own the numpy
        # buffer, and `rgb` is local, so the previous code could be drawing
        # from freed memory. This is the standard PySide crash.
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        height, width, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, width, height, width * 3,
                            QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self._last_overlay = overlay
        if overlay is not None or self.truth_row is not None:
            painter = QtGui.QPainter(pixmap)
            if overlay is not None:
                pen = QtGui.QPen(QtGui.QColor("red"))
                pen.setWidth(3)
                painter.setPen(pen)
                start, end = overlay
                painter.drawLine(QtCore.QPointF(start[0], start[1]),
                                 QtCore.QPointF(end[0], end[1]))
            if self.truth_row is not None:
                pen = QtGui.QPen(QtGui.QColor(0, 230, 60))
                pen.setWidth(3)
                painter.setPen(pen)
                painter.drawLine(QtCore.QPointF(0, self.truth_row),
                                 QtCore.QPointF(pixmap.width() - 1, self.truth_row))
            painter.end()
        # Keep the full-resolution pixmap; zoom scales a copy of it, so
        # zooming never resamples an already-resampled image.
        self._pixmap = pixmap
        self._apply_zoom()

    def open_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open image", str(Path.home()), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        self.open_image_path(path)
        # If the file belongs to the loaded folder, sync the filmstrip.
        for index in range(self.strip.count()):
            item = self.strip.item(index)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == path:
                self.strip.blockSignals(True)
                self.strip.setCurrentRow(index)
                self.strip.blockSignals(False)
                break

    def calibrate(self) -> None:
        if self.current_image is None:
            self.log_message("Load an image before calibrating.")
            return

        cfg = self._make_calibration_config(self.current_image)
        h, w = self.current_image.shape[:2]
        ok = self.calibration.calibrate_octagon_image(
            self.current_image,
            facet_length=cfg.facet_length,
            zero_offset=cfg.zero_offset,
            target_roi=cfg.target_roi if cfg.target_roi[2] > 0 else None,
        )
        if not ok:
            self.log_message(
                "Calibration failed (octagon not found, or reprojection rms"
                " %.4f exceeded the %.4f limit)."
                % (self.calibration.model.reprojection_rms,
                   CalibExecutive.max_reprojection_rms))
            return

        # Derive the waterline search polygon from the octagon we just
        # found, the same way cli._calibrate_image does. Previously the GUI
        # kept whatever polygon was in the form, and the fallback default
        # starts at 65% of image height -- below the waterline on these
        # frames -- so the GUI and the CLI disagreed on the same image.
        pixel_points = self.calibration.model.pixel_points
        search_poly = cfg.search_poly
        if len(pixel_points) >= 8:
            xs = [float(point[0]) for point in pixel_points]
            ys = [float(point[1]) for point in pixel_points]
            min_x = max(0, int(min(xs) - 100))
            max_x = min(w - 1, int(max(xs) + 300))
            top_y = min(h - 2, int(max(ys) + 100))
            # Stop above the caption banner. h-20 reached into it and the
            # step into solid black won on every frame.
            bottom_y = content_bottom(self.current_image)
            if bottom_y - top_y > 40:
                search_poly = ((min_x, top_y), (max_x, top_y),
                               (min_x, bottom_y), (max_x, bottom_y))
                self.log_message(
                    "Search region set from the octagon: x %d-%d, y %d-%d"
                    % (min_x, max_x, top_y, bottom_y))

        self.calibration_config = CalibrationConfig(
            facet_length=cfg.facet_length,
            zero_offset=cfg.zero_offset,
            target_roi=cfg.target_roi,
            search_poly=search_poly,
            image_size=(w, h),
            pixel_to_world_points=tuple(
                (*pixel, *world)
                for pixel, world in zip(self.calibration.model.pixel_points, self.calibration.model.world_points)
            ),
        )
        self._sync_controls(self.calibration_config)
        world = self.calibration.pixel_to_world((w * 0.5, h * 0.5))
        self.log_message(
            "Calibration succeeded: reprojection rms=%.4f max=%.4f world units;"
            " sample world point: %s"
            % (self.calibration.model.reprojection_rms,
               self.calibration.model.reprojection_max, world))
        # The frame on screen was detected (if at all) with the old or
        # fabricated polygon; redo it with the real one.
        if self.chk_auto_find.isChecked():
            self.find_line()
        self._display_image(self.current_image)

    def find_line(self) -> None:
        if self.current_image is None:
            self.log_message("Load an image before finding a water line.")
            return

        # Only fabricate a config when there is no real one. Previously this
        # ran on every call and logged the "fields incomplete" warning even
        # when a loaded calibration was about to replace it -- which would
        # spam the log once per filmstrip click.
        config = (self.calibration_config if self.calibration_config is not None
                  else self._make_calibration_config(self.current_image))
        # Use the same search polygon the CLI would, target tracking
        # included. Previously the GUI skipped tracking, so the GUI and the
        # CLI gave different answers for the same image.
        search_poly, movement = _search_poly_for_image(self.current_image, config)
        result = detect_water_level(self.current_image, search_poly)
        if movement != (0.0, 0.0):
            self.log_message("Target moved by (%.1f, %.1f) px; search region adjusted"
                             % movement)
        if not result.found:
            self._set_result(["No waterline found in the search region."]
                             + list(result.messages))
            self.log_message("No waterline found in the search region.")
            return
        self.last_waterline = result.y
        self._set_result(
            [f"Image: {os.path.basename(self.image_path)}" if self.image_path else "",
             f"Pixel row: {result.y:.2f}",
             f"Angle: {result.angle:.2f} degrees",
             f"Confidence: {result.confidence:.2f}"]
            + list(result.messages)
        )
        self.log_message(f"Detected waterline center row: {result.y:.2f}; angle: {result.angle:.2f} degrees")
        self._display_image(self.current_image, overlay=result.endpoints)

    def load_calibration(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load calibration",
            self.calibration_path or str(Path.home()), "JSON (*.json)")
        if not path:
            return
        self._apply_calibration_file(path)

    def save_calibration(self) -> None:
        if self.calibration_config is None:
            self.log_message("Calibrate or load a calibration before saving.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save calibration",
            self.calibration_path or "calibration.json", "JSON (*.json)")
        if not path:
            return
        try:
            Path(path).write_text(form_octagon_calib_json_string(self.calibration_config), encoding="utf-8")
            self.calibration_path = str(path)
            self.settings["last_calibration"] = self.calibration_path
            save_settings(self.settings)
            self.log_message(f"Saved calibration: {path}")
        except OSError as error:
            self.log_message(f"Could not save calibration: {error}")

    def process_folder(self) -> None:
        if self.calibration_config is None:
            self.log_message("Load or create a calibration before processing a folder.")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Process image folder",
            self.folder_path or str(Path.home()))
        if not folder:
            return
        output, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save results", "water_levels.csv", "CSV (*.csv)")
        if not output:
            return
        paths = sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
        records = [_detect_image(path, self.calibration_config) for path in paths]
        if records:
            with Path(output).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        self.log_message(f"Processed {len(records)} images; results saved to {output}")

    def show_metadata(self) -> None:
        if self.image_path is None:
            self.log_message("Load an image before showing metadata.")
            return
        try:
            self.log_message(json.dumps(_read_metadata(self.image_path), default=str))
        except (OSError, ValueError) as error:
            self.log_message(f"Could not read metadata: {error}")

    def make_gif(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Create GIF from folder",
            self.folder_path or str(Path.home()))
        if not folder:
            return
        output, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save GIF", "images.gif", "GIF (*.gif)")
        if not output:
            return
        try:
            count = _make_gif(folder, output, 100, None)
            self.log_message(f"Created GIF with {count} frames: {output}")
        except (OSError, ValueError) as error:
            self.log_message(f"Could not create GIF: {error}")


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI. Reuses an existing QApplication when one is already
    running, so the window can also be opened from inside a host
    application (GRIME AI) rather than only as a standalone process."""
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv if argv is None else list(argv))
    window = MainWindow()
    window.show()
    if not owns_app:
        main.window = window        # keep a reference; host owns the loop
        return 0
    return exec_app(app)


if __name__ == "__main__":
    raise SystemExit(main())
