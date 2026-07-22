# sam2_gui/ui/main_window.py
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
    QToolBar,
    QSpinBox,
    QLabel,
    QCheckBox,
)
from PyQt5.QtWidgets import QAction
from PyQt5.QtCore import Qt, QTimer
import os
import csv
import copy
import json
import math
import random

import numpy as np

from SAGE.utils.image_io import load_image_rgb
from SAGE.core.controller import SegmentationController
from SAGE.core.renderer import Renderer
from SAGE.ui.canvas import Canvas
from SAGE.ui.filmstrip import Filmstrip
from SAGE.ui.sidebar import Sidebar, MASK_STATE_LOCKED, MASK_STATE_LOADED, MASK_STATE_PERSISTS
from SAGE.utils.mask_ops import compute_mask_stats
from SAGE.utils.coco_buffer import CocoBuffer, ann_to_mask
from SAGE.ui.mask_item import MaskItem
from SAGE.settings_manager import SettingsManager


class MainWindow(QMainWindow):
    def __init__(self, model_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Segmentation & Annotation for Geospatial Ecohydrology (SAGE)")

        # Initialize settings manager
        self.settings_manager = SettingsManager()

        # Store model manager for reloading images
        self.model_manager = model_manager

        # Mask store + current image tracking
        self.mask_store = {}
        self.current_image_path = None
        self._coco_buffer = None

        # Polygon sampling mode
        self.polygon_sampling_mode = "dense"  # "dense", "random", "poisson"

        # Controller and renderer will be created when first image is loaded
        self.controller = None
        self.renderer = None
        self.image_np = None

        # Seed mask (global / site-level)
        self.seed_mask_path = None  # path to .tif/.tiff mask
        self.seed_mask_bool = None  # cached boolean mask resized to current image
        self.auto_seed_enabled = True  # auto-apply when each image loads
        self._loaded_signature = None  # content hash of masks as last loaded/saved
        self.eraser_radius = 18
        self._opacity_percent = int(120 / 255 * 100)  # default; overwritten below from sage.json

        # Read sage.json early so opacity is correct before any image auto-loads
        self._settings_dir = os.path.join(
            os.path.expanduser("~"), "Documents", "GRIME-AI", "settings"
        )
        self._sage_settings_path = os.path.join(self._settings_dir, "sage.json")
        _early_settings = self._read_sage_settings()
        if "mask_opacity_percent" in _early_settings:
            self._opacity_percent = int(_early_settings["mask_opacity_percent"])

        central = QWidget()

        # ---------------------------------------------------------
        # TOP-LEVEL LAYOUT: VERTICAL (folder row above everything)
        # ---------------------------------------------------------
        main_layout = QVBoxLayout(central)

        # Folder row ABOVE the image
        ROW_HEIGHT = 28
        folder_row = QHBoxLayout()
        folder_row.setContentsMargins(4, 4, 4, 4)
        folder_row.setSpacing(6)
        self.folder_edit = QLineEdit()
        self.folder_edit.setFixedHeight(ROW_HEIGHT)
        self.folder_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.folder_browse_btn = QPushButton("Browse")
        self.folder_browse_btn.setFixedHeight(ROW_HEIGHT)
        self.folder_browse_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.folder_browse_btn.clicked.connect(self._browse_folder)
        # h = self.folder_browse_btn.sizeHint().height()
        # self.folder_edit.setFixedHeight(h)
        # self.folder_browse_btn.setFixedHeight(h)
        folder_row.addWidget(self.folder_edit, stretch=1)
        folder_row.addWidget(self.folder_browse_btn)
        main_layout.addLayout(folder_row)

        # ---------------------------------------------------------
        # BELOW THAT: Canvas + Sidebar horizontally
        # ---------------------------------------------------------
        layout = QHBoxLayout()

        self.canvas = Canvas(
            on_left_click=self._on_left_click,
            on_right_click=self._on_right_click_handler,
            parent=self
        )
        # Expand to fill the window, and allow it to shrink so the window can
        # collapse smaller (QGraphicsView otherwise floors the height).
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(0, 0)
        self.canvas.eraser_move.connect(self._erase_seeds_at)
        # Default "Click or Drag" button maps to paint-style placement.
        self.canvas.set_segmentation_mode("paint")

        # Canvas column: canvas above, thumbnail filmstrip below.
        canvas_col = QVBoxLayout()
        canvas_col.setContentsMargins(0, 0, 0, 0)
        canvas_col.setSpacing(2)
        canvas_col.addWidget(self.canvas, stretch=1)

        self.filmstrip = Filmstrip(parent=self)
        self.filmstrip.image_clicked.connect(self._load_new_image)
        canvas_col.addWidget(self.filmstrip)

        layout.addLayout(canvas_col, stretch=4)

        self.sidebar = Sidebar(
            controller=None,  # Will be set when image is loaded
            on_run_segmentation=self._run_segmentation,
            on_visibility_changed=self._update_canvas,
            on_clear_points=self._clear_points,
            parent=self,
        )

        # Sidebar requests COCO save → MainWindow handles it
        self.sidebar.save_all_coco_requested.connect(self.save_all_coco)

        # Seed mask signals
        self.sidebar.load_seed_mask_requested.connect(self._browse_seed_mask)
        self.sidebar.clear_seed_mask_requested.connect(self._clear_seed_mask)
        self.sidebar.seed_points_now_requested.connect(self._seed_points_from_mask_current_image)
        self.sidebar.auto_seed_toggled.connect(self._on_auto_seed_toggled)
        self.sidebar.eraser_toggled.connect(self._on_eraser_toggled)
        self.sidebar.mask_selected.connect(self._on_mask_selected)
        self.sidebar.mask_reclicked.connect(self._on_mask_reclicked)
        self.sidebar.mask_unlabeled_requested.connect(self._mask_unlabeled)
        self.sidebar.mask_renamed.connect(self._on_mask_renamed)
        self.sidebar.label_class_renamed.connect(self._on_label_class_renamed)

        self.selected_mask_id = -1

        # New: segmentation mode + sampling mode signals
        self.sidebar.segmentation_mode_changed.connect(self._on_segmentation_mode_changed)
        self.sidebar.polygon_sampling_changed.connect(self._on_polygon_sampling_changed)

        # Canvas polygon signal - handle both SAM2 and manual
        self.canvas.polygon_drawn.connect(self._on_polygon_drawn_dispatcher)

        # Edge Trace signal
        self.canvas.edge_trace_stroke.connect(self._on_edge_trace_stroke)
        # Canvas asks this before allowing any drawing; blocks (with one dialog)
        # when no label is active, instead of looping on every drag event.
        self.canvas._can_annotate = lambda: self.sidebar.get_active_label() is not None
        self.canvas.label_required.connect(self._warn_no_label)
        self.sidebar.tool_mode_changed.connect(self.canvas.set_tool_mode)
        self.canvas.mask_clicked.connect(self._on_canvas_mask_clicked)

        # Wrap the sidebar in a scroll area so the window can shrink below the
        # combined height of the sidebar's stacked widgets. The scroll area is
        # forced to a zero minimum height, so instead of flooring the window it
        # shows a vertical scrollbar when the viewport is shorter than the
        # sidebar needs. widgetResizable keeps the sidebar filling the column
        # width (no horizontal scrollbar).
        from PyQt5.QtWidgets import QScrollArea
        self.sidebar.setMinimumHeight(0)
        self._sidebar_scroll = QScrollArea()
        self._sidebar_scroll.setWidget(self.sidebar)
        self._sidebar_scroll.setWidgetResizable(True)
        self._sidebar_scroll.setFrameShape(QScrollArea.NoFrame)
        self._sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._sidebar_scroll.setMinimumHeight(0)
        layout.addWidget(self._sidebar_scroll, stretch=1)
        main_layout.addLayout(layout, stretch=1)

        self.setCentralWidget(central)

        # ---------------------------------------------------------
        # Toolbar: Mask Opacity spinbox
        # ---------------------------------------------------------
        toolbar = QToolBar("View")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel("  Mask Opacity: "))
        self.opacity_spinbox = QSpinBox()
        self.opacity_spinbox.setRange(0, 100)
        self.opacity_spinbox.setSuffix(" %")
        self.opacity_spinbox.setValue(int(120 / 255 * 100))  # match sidebar default
        self.opacity_spinbox.setFixedWidth(72)
        self.opacity_spinbox.setToolTip("Mask overlay opacity (0 = transparent, 100 = opaque)")
        self.opacity_spinbox.valueChanged.connect(self._on_opacity_spinbox_changed)
        toolbar.addWidget(self.opacity_spinbox)

        toolbar.addSeparator()
        self.border_checkbox = QCheckBox("Show Borders")
        self.border_checkbox.setChecked(False)
        self.border_checkbox.setToolTip("Show/hide the border outline around masked regions")
        self.border_checkbox.stateChanged.connect(self._on_border_checkbox_changed)
        toolbar.addWidget(self.border_checkbox)
        self._show_borders = False

        self.flash_checkbox = QCheckBox("Flash")
        self.flash_checkbox.setChecked(True)
        self.flash_checkbox.setToolTip("Briefly flash a mask when selected")
        self.flash_checkbox.stateChanged.connect(self._on_flash_checkbox_changed)
        toolbar.addWidget(self.flash_checkbox)
        self._flash_enabled = True

        self.other_checkbox = QCheckBox("Display Other")
        self.other_checkbox.setChecked(False)
        self.other_checkbox.setToolTip(
            "Preview the 'Other' region — every pixel not covered by a defined mask")
        self.other_checkbox.stateChanged.connect(self._on_other_checkbox_changed)
        toolbar.addWidget(self.other_checkbox)
        self._display_other = False

        # Flash animation state
        self._flash_timer = QTimer(self)
        self._flash_timer.setInterval(100)          # blink speed (lower = faster)
        self._flash_timer.timeout.connect(self._on_flash_tick)
        self._flash_mask_id = -1
        self._flash_on = False

        self._flash_ticks = 0
        self._FLASH_TOTAL_TICKS = 35                 # 100ms x 35 = ~3.5s
        self._flash_base_pixmap = None      # cached full canvas; ticks blit onto this
        self._flash_entry = None            # the mask entry being flashed

        # ---------------------------------------------------------
        # Menu bar: File
        # ---------------------------------------------------------
        self._labels_dir = os.path.join(
            os.path.expanduser("~"), "Documents", "GRIME-AI", "labels"
        )

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        export_action = QAction("Export Labels…", self)
        export_action.setStatusTip("Export label classes to a CSV file")
        export_action.triggered.connect(self._export_labels)
        file_menu.addAction(export_action)

        import_action = QAction("Import Labels…", self)
        import_action.setStatusTip("Import label classes from a CSV file")
        import_action.triggered.connect(self._import_labels)
        file_menu.addAction(import_action)

        help_menu = menu_bar.addMenu("Help")
        about_action = QAction("About SAGE…", self)
        about_action.setStatusTip("Version and attribution")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        self.showMaximized()

        # Load saved folder path and populate
        saved_folder = self.settings_manager.get_folder_path()
        if saved_folder and os.path.isdir(saved_folder):
            self.folder_edit.setText(saved_folder)
            self._populate_image_list(saved_folder)

        # Auto-load last used labels file from sage.json
        self._autoload_labels()

        # Restore saved opacity
        self._autoload_opacity()

        # Restore saved border setting
        self._autoload_borders()
        self._autoload_flash()

        # Restore saved edge trace settings
        self._autoload_edge_trace_settings()

    # ------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------
    def _on_auto_seed_toggled(self, on: bool):
        self.auto_seed_enabled = on

    # -------------------------------------------------------------------------
    # Sage settings (~/Documents/GRIME-AI/settings/sage.json)
    # -------------------------------------------------------------------------

    def _read_sage_settings(self) -> dict:
        """Read sage.json, returning an empty dict on any error."""
        try:
            with open(self._sage_settings_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_sage_settings(self, data: dict):
        """Merge data into sage.json, creating the file and directory if needed."""
        os.makedirs(self._settings_dir, exist_ok=True)
        current = self._read_sage_settings()
        current.update(data)
        with open(self._sage_settings_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)

    def _autoload_labels(self):
        """On startup, load the last used labels file if it still exists."""
        settings = self._read_sage_settings()
        last_labels = settings.get("last_labels_file")
        if last_labels and os.path.isfile(last_labels):
            self._load_labels_from_path(last_labels, silent=True)

    def _autoload_opacity(self):
        """Sync the spinbox to _opacity_percent (already loaded from sage.json in __init__)."""
        self.opacity_spinbox.blockSignals(True)
        self.opacity_spinbox.setValue(self._opacity_percent)
        self.opacity_spinbox.blockSignals(False)

    def _autoload_flash(self):
        """Restore flash setting from sage.json (default True on first use)."""
        settings = self._read_sage_settings()
        enabled = settings.get("flash_enabled", True)
        self._flash_enabled = enabled
        self.flash_checkbox.blockSignals(True)
        self.flash_checkbox.setChecked(enabled)
        self.flash_checkbox.blockSignals(False)

    def _autoload_borders(self):
        """Restore show_borders setting from sage.json."""
        settings = self._read_sage_settings()
        show = settings.get("show_borders", False)
        self._show_borders = show
        self.border_checkbox.blockSignals(True)
        self.border_checkbox.setChecked(show)
        self.border_checkbox.blockSignals(False)

    def _autoload_edge_trace_settings(self):
        """Restore edge trace interval and width from sage.json."""
        settings = self._read_sage_settings()
        interval = settings.get("edge_trace_interval", 40)
        width = settings.get("edge_trace_width", 80)
        self.canvas._edge_trace_interval = interval
        self.canvas._ET_MAX_RAY = width
        self.canvas._update_et_hud_text()

    # -------------------------------------------------------------------------
    # Label CSV Export / Import
    # -------------------------------------------------------------------------

    def _get_version(self):
        """SAGE versions within the GRIME AI ecosystem. Confirm/adjust the
        import below if the version module path or variable differs."""
        try:
            from GRIME_AI.version import __version__ as _v
            return str(_v)
        except Exception:
            try:
                from GRIME_AI import version as _vmod
                return str(getattr(_vmod, "__version__", None)
                           or getattr(_vmod, "VERSION", None) or "unknown")
            except Exception:
                return "unknown"

    def _show_about(self):
        from pathlib import Path
        from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QPushButton
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtCore import Qt

        dlg = QDialog(self)
        dlg.setWindowTitle("About SAGE")
        layout = QVBoxLayout(dlg)

        logo_path = Path(__file__).resolve().parent.parent / "resources" / "sage_logo.png"
        pix = QPixmap(str(logo_path))
        if not pix.isNull():
            logo = QLabel()
            logo.setAlignment(Qt.AlignCenter)
            logo.setPixmap(pix.scaled(pix.width() // 2, pix.height() // 2,
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation))
            layout.addWidget(logo)

        name = QLabel("SAGE — Segmentation & Annotation for Geospatial Ecohydrology")
        name.setAlignment(Qt.AlignCenter)
        name.setWordWrap(True)
        layout.addWidget(name)

        ver = QLabel(f"Version {self._get_version()}")
        ver.setAlignment(Qt.AlignCenter)
        layout.addWidget(ver)

        eco = QLabel("Part of the GRIME AI ecosystem.")
        eco.setAlignment(Qt.AlignCenter)
        layout.addWidget(eco)

        btn = QPushButton("OK")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)

        dlg.exec_()

    def _export_labels(self):
        pairs = self.sidebar.get_label_classes_with_ids()
        if not pairs:
            QMessageBox.warning(self, "Export Labels",
                "No label classes defined. Add labels before exporting.")
            return

        os.makedirs(self._labels_dir, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Labels", self._labels_dir,
            "CSV Files (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for name, cid in pairs:
                    writer.writerow([cid, name])
            self._write_sage_settings({"last_labels_file": path})
            QMessageBox.information(self, "Export Labels",
                f"Exported {len(pairs)} label(s) to:\n{path}")
        except OSError as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def _import_labels(self):
        os.makedirs(self._labels_dir, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Labels", self._labels_dir,
            "CSV Files (*.csv)"
        )
        if not path:
            return
        self._load_labels_from_path(path, silent=False)

    def _load_labels_from_path(self, path: str, silent: bool = False):
        """Parse a labels CSV and apply to the sidebar. Persists path to sage.json."""
        try:
            pairs = []
            seen_names = set()
            seen_ids = set()
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for lineno, row in enumerate(reader, start=1):
                    if not row or all(c.strip() == "" for c in row):
                        continue
                    if len(row) < 2:
                        if not silent:
                            QMessageBox.warning(self, "Import Labels",
                                f"Line {lineno} is malformed (expected: id,name):\n{','.join(row)}")
                        return
                    try:
                        lid = int(row[0].strip())
                    except ValueError:
                        if not silent:
                            QMessageBox.warning(self, "Import Labels",
                                f"Line {lineno} has a non-integer ID: \"{row[0]}\"")
                        return
                    name = row[1].strip()
                    if not name:
                        if not silent:
                            QMessageBox.warning(self, "Import Labels",
                                f"Line {lineno} has an empty label name.")
                        return
                    if name in seen_names:
                        if not silent:
                            QMessageBox.warning(self, "Import Labels",
                                f"Duplicate label name on line {lineno}: \"{name}\"\n"
                                "Each label must be unique.")
                        return
                    if lid in seen_ids:
                        if not silent:
                            QMessageBox.warning(self, "Import Labels",
                                f"Duplicate label ID on line {lineno}: {lid}\n"
                                "Each ID must be unique.")
                        return
                    seen_names.add(name)
                    seen_ids.add(lid)
                    pairs.append((name, lid))

            if not pairs:
                if not silent:
                    QMessageBox.warning(self, "Import Labels", "No labels found in file.")
                return

            self.sidebar.set_label_classes(pairs)
            self._write_sage_settings({"last_labels_file": path})
            if not silent:
                QMessageBox.information(self, "Import Labels",
                    f"Imported {len(pairs)} label(s) from:\n{path}")

        except OSError as e:
            if not silent:
                QMessageBox.critical(self, "Import Failed", str(e))

    # ------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------
    def _on_mask_selected(self, mask_id: int):
        self.selected_mask_id = mask_id
        self._start_flash(mask_id)
        self._update_canvas()

    def _start_flash(self, mask_id: int):
        """Restart the flash for a newly selected mask. Renders the full canvas
        ONCE, then each tick only blits the one brightened mask onto that cached
        pixmap — so the flash rate follows the timer interval, not the (heavy)
        full-canvas rebuild time."""
        self._flash_timer.stop()
        self._flash_base_pixmap = None
        self._flash_entry = None

        if (not self._flash_enabled or mask_id < 0
                or self.controller is None or self.renderer is None):
            self._flash_mask_id = -1
            self._flash_on = False
            self._update_canvas()
            return

        self._flash_mask_id = mask_id
        self._flash_entry = next(
            (m for m in self.controller.masks if m["id"] == mask_id), None)
        if self._flash_entry is None:
            self._update_canvas()
            return

        # Render the base canvas once (all masks, borders, hit-test items).
        self._flash_on = False          # base must be unflashed
        self._update_canvas()
        self._flash_base_pixmap = self.canvas._pixmap_item.pixmap()

        self._flash_on = True
        self._flash_ticks = 0
        self._paint_flash_frame(True)     # show first bright frame immediately
        self._flash_timer.start()

    def _on_mask_reclicked(self, mask_id: int):
        """Re-clicking the already-selected list row stops an active flash."""
        if self._flash_timer.isActive() and mask_id == self._flash_mask_id:
            self._stop_flash()

    def _stop_flash(self):
        """Stop an in-progress flash immediately and settle on the plain canvas."""
        if not self._flash_timer.isActive() and self._flash_base_pixmap is None:
            return
        self._flash_timer.stop()
        self._flash_on = False
        self._flash_mask_id = -1
        self._flash_entry = None
        if self._flash_base_pixmap is not None:
            self.canvas.set_pixmap(self._flash_base_pixmap)
            self._flash_base_pixmap = None
        else:
            self._update_canvas()

    def _paint_flash_frame(self, on: bool):
        if self._flash_base_pixmap is None:
            return
        if on and self._flash_entry is not None:
            pm = self.renderer.overlay_single_mask(
                self._flash_base_pixmap, self._flash_entry, self.controller.opacity)
        else:
            pm = self._flash_base_pixmap
        self.canvas.set_pixmap(pm)

    def _on_flash_tick(self):
        self._flash_ticks += 1
        if self._flash_ticks >= self._FLASH_TOTAL_TICKS:
            self._flash_timer.stop()
            self._flash_on = False
            self._flash_mask_id = -1
            self._flash_entry = None
            if self._flash_base_pixmap is not None:
                self.canvas.set_pixmap(self._flash_base_pixmap)  # settle on plain base
            self._flash_base_pixmap = None
            return
        self._flash_on = not self._flash_on
        self._paint_flash_frame(self._flash_on)

    def _on_flash_checkbox_changed(self, state):
        self._flash_enabled = bool(state)
        self._write_sage_settings({"flash_enabled": self._flash_enabled})
        if not self._flash_enabled:
            self._flash_timer.stop()
            self._flash_mask_id = -1
            self._update_canvas()

    def _on_label_class_renamed(self, old_name: str, new_name: str):
        """Propagate a label class rename to all existing mask entries."""
        if self.controller is None:
            return
        renamed = False
        for m in self.controller.masks:
            if m["label"] == old_name:
                m["label"] = new_name
                renamed = True
        if renamed:
            self.controller.dirty = True
        self.sidebar.refresh_masks()

    def _on_mask_renamed(self, mask_id: int, new_name: str):
        """Rename a single specific mask instance and update its color to match the label class."""
        if self.controller is None:
            return
        for m in self.controller.masks:
            if m["id"] == mask_id:
                m["label"] = new_name
                m["color"] = self.sidebar.get_color_for_label(new_name)
                self.controller.dirty = True
                break
        self.sidebar.refresh_masks()
        self._update_canvas()

    def _on_eraser_toggled(self, on: bool):
        self.canvas.set_eraser_enabled(on)
        # optional: change cursor
        self.canvas.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    # ------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------
    def _on_polygon_drawn_dispatcher(self, points):
        """Route to SAM2 or manual polygon handler based on mode"""
        if self.canvas._segmentation_mode in ("manual_polygon", "manual_freeform"):
            self._on_manual_polygon_drawn(points)
        else:
            self._on_polygon_drawn(points)  # SAM2 Polygon and SAM2 Freehand

    # ------------------------------------------------------------------------
    # Edge Trace handler
    # ------------------------------------------------------------------------
    def _on_edge_trace_stroke(self, fg_pts, bg_pts):
        """
        Receives pre-computed fg/bg points from Canvas (Canny already done there).
        Loads them into the controller and runs SAM2 segmentation.
        """
        if self.controller is None:
            return
        if not fg_pts:
            return

        label = self.sidebar.get_active_label()
        if label is None:
            QMessageBox.warning(self, "No Label Defined",
                "Please define at least one label class before annotating.")
            return

        for x, y in fg_pts:
            self.controller.add_point(x, y, is_fg=True)
        for x, y in bg_pts:
            self.controller.add_point(x, y, is_fg=False)

        color = self.sidebar.get_color_for_label(label)
        mask_entry = self.controller.run_segmentation(label=label, color=color)
        if mask_entry:
            self.sidebar.refresh_masks()
        self._update_canvas()

    # ------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------
    def _on_manual_polygon_drawn(self, points):
        """
        Create mask directly from polygon without SAM2.
        For ambiguous features like grass where SAM2 fails.
        """
        if self.controller is None:
            return

        if len(points) < 3:
            return

        import cv2

        # Get label from active label class
        label = self.sidebar.get_active_label()
        if label is None:
            QMessageBox.warning(self, "No Label Defined",
                "Please define at least one label class before annotating.")
            return

        # Create mask from polygon
        h, w = self.image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        polygon_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon_array], 1)
        mask = mask.astype(bool)

        # Route through the controller so the manual mask is clipped against
        # owned pixels at the same chokepoint as the SAM2 paths (first-wins).
        color = self.sidebar.get_color_for_label(label)
        mask_entry = self.controller.add_manual_mask(mask, label=label, color=color)
        if mask_entry is None:
            return  # stroke fell entirely on already-owned pixels

        self.sidebar.refresh_masks()
        self._update_canvas()

    # ---------------------------------------------------------
    # Folder browsing
    # ---------------------------------------------------------
    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_edit.setText(folder)

            # Save folder path to settings
            self.settings_manager.set_folder_path(folder)

            # Populate image list
            self._populate_image_list(folder)

    # ---------------------------------------------------------
    # Seed mask browsing
    # ---------------------------------------------------------
    def _browse_seed_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Seed Mask",
            "",
            "Mask Files (*.tif *.tiff *.png *.jpg *.jpeg *.npy)"
        )
        if not path:
            return

        self.seed_mask_path = path
        self.seed_mask_bool = None  # reset cache

        # Auto-populate label classes from the mask filename stem
        # e.g. "water_surface_mask.tif" → "water_surface"
        stem = os.path.splitext(os.path.basename(path))[0]
        for suffix in ("_mask", "_roi", "_label", "_labels", "_seg"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        self.sidebar.add_label_classes_from_mask([stem])

        if self.auto_seed_enabled and self.controller is not None:
            self._seed_points_from_mask_current_image()

        self.sidebar.set_mask_subpanel_state(MASK_STATE_LOADED, path)

    def _clear_seed_mask(self):
        self.seed_mask_path = None
        self.seed_mask_bool = None
        self.sidebar.set_mask_subpanel_state(MASK_STATE_LOCKED)

    def _load_seed_mask_bool(self, target_shape):
        """
        Returns boolean mask (H,W) resized to target_shape using nearest-neighbor.
        Caches result per-image-size to avoid re-reading on every image.
        """
        if self.seed_mask_path is None:
            return None

        # If cached and already correct size, reuse
        if self.seed_mask_bool is not None and self.seed_mask_bool.shape == target_shape:
            return self.seed_mask_bool

        path = self.seed_mask_path
        ext = os.path.splitext(path)[1].lower()

        try:
            if ext in [".tif", ".tiff"]:
                from PIL import Image
                m = np.array(Image.open(path))
            elif ext == ".npy":
                m = np.load(path)
            else:
                import cv2
                m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if m is None:
                    return None
                if m.ndim == 3:
                    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

            mask = self._normalize_seed_mask(m)
            print("Seed mask foreground fraction:", mask.mean())

            # Resize if needed
            if mask.shape != target_shape:
                import cv2
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            import cv2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_u8 = mask.astype(np.uint8) * 255
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = (mask_u8 > 0)

            self.seed_mask_bool = mask
            return mask

        except Exception as e:
            print("Seed mask load error:", e)
            return None

    def _normalize_seed_mask(self, mask):
        """
        Returns boolean mask where True = ROI (foreground).

        Handles common cases:
        - binary 0/255 masks
        - 0/1 masks
        - float masks
        - masks with huge white background (auto-invert)
        """
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]

        m = mask.astype(np.float32)
        m = np.nan_to_num(m, nan=0.0)

        # If it looks binary-ish, threshold at >0
        uniq = np.unique(m)
        if uniq.size <= 10:
            roi = m > 0
        else:
            # otherwise use Otsu (robust for grayscale)
            import cv2
            mm = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, thr = cv2.threshold(mm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            roi = thr > 0

        # Auto-invert if ROI covers most of the image (common when background is white)
        # If > 50% is True, it's probably inverted.
        if roi.mean() > 0.5:
            roi = ~roi

        return roi

    # ---------------------------------------------------------
    # Seed points from input mask
    # ---------------------------------------------------------
    def _seed_points_from_mask_current_image(self):
        if self.controller is None or self.image_np is None:
            return
        if not self.seed_mask_path:
            QMessageBox.warning(self, "No Seed Mask", "Please load a seed mask (.tif) first.")
            return

        # Load / resize seed mask to match current image
        mask_bool = self._load_seed_mask_bool(target_shape=self.image_np.shape[:2])
        if mask_bool is None or mask_bool.sum() == 0:
            QMessageBox.warning(self, "Invalid Seed Mask", "Seed mask is empty or could not be loaded.")
            return

        seed = abs(hash(self.current_image_path)) % (2 ** 32)
        fg_points, bg_points = self._sample_points_from_mask(mask_bool, n_fg=30, n_bg=0, seed=seed)

        # Put points into the existing controller path (supported)
        self.controller.clear_points()
        for x, y in fg_points:
            self.controller.add_point(x, y, is_fg=True)
        for x, y in bg_points:
            self.controller.add_point(x, y, is_fg=False)

        print("Seed mask stats:", mask_bool.shape, "fg_frac=", float(mask_bool.mean()))
        self._update_canvas()

    def _sample_points_from_mask(self, mask_bool, n_fg=30, n_bg=30, seed=0):
        rng = np.random.default_rng(seed)
        h, w = mask_bool.shape

        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return [], []

        # FG points inside ROI
        k_fg = min(n_fg, len(xs))
        idx_fg = rng.choice(len(xs), size=k_fg, replace=False)
        fg_points = [(int(xs[i]), int(ys[i])) for i in idx_fg]

        if n_bg <= 0:
            return fg_points, []

        # BG candidates: outside ROI
        outside = ~mask_bool

        # Build "safe BG zones"
        safe = np.zeros_like(mask_bool, dtype=bool)

        # sky strip
        safe[: int(0.15 * h), :] = True
        # bottom strip
        safe[int(0.90 * h):, :] = True
        # left/right margins
        safe[:, : int(0.05 * w)] = True
        safe[:, int(0.95 * w):] = True

        bg_candidate = outside & safe
        bys, bxs = np.where(bg_candidate)

        # fallback: if safe zones empty, use any outside
        if len(bxs) == 0:
            bys, bxs = np.where(outside)

        if len(bxs) == 0:
            return fg_points, []

        k_bg = min(n_bg, len(bxs))
        idx_bg = rng.choice(len(bxs), size=k_bg, replace=False)
        bg_points = [(int(bxs[i]), int(bys[i])) for i in idx_bg]

        return fg_points, bg_points

    # ---------------------------------------------------------
    # Populate mask name
    # ---------------------------------------------------------
    def _default_label_from_seed_mask(self):
        """
        Extract class name from seed mask filename.

        Expected pattern:
          site_CLASS_otherinfo.tif
          e.g. ninemileprairie_GR_2000_03.tif

        Returns:
          "GR"
        """
        if not self.seed_mask_path:
            return None

        name = os.path.basename(self.seed_mask_path)
        stem = os.path.splitext(name)[0]

        parts = stem.split("_")
        if len(parts) < 2:
            return None

        class_name = parts[1]  # <-- CLASS POSITION (fixed)
        return class_name

    # ---------------------------------------------------------
    # Erase seed points
    # ---------------------------------------------------------
    def _erase_seeds_at(self, x, y):
        if self.controller is None:
            return

        # 1) remove SAM points + manual points from controller
        self.controller.remove_points_in_circle(x, y, radius=self.eraser_radius)

        # 2) also remove paint stroke points (works for (x,y) or (x,y,is_fg))
        if hasattr(self.canvas, "_paint_points") and self.canvas._paint_points:
            r2 = float(self.eraser_radius) ** 2
            new_pts = []
            for p in self.canvas._paint_points:
                px, py = p[0], p[1]  # <-- only take first two always
                dx = px - x
                dy = py - y
                if (dx * dx + dy * dy) > r2:
                    new_pts.append(p)  # keep original tuple as-is (2 or 3 items)
            self.canvas._paint_points = new_pts

        self._update_canvas()

    def _toggle_seed_eraser(self):
        self.eraser_enabled = self.seed_eraser_btn.isChecked()
        self.seed_eraser_btn.setText("Eraser: ON" if self.eraser_enabled else "Eraser: OFF")
        self.canvas.set_eraser_enabled(self.eraser_enabled)

    # ---------------------------------------------------------
    # Populate image list
    # ---------------------------------------------------------
    def _populate_image_list(self, folder):
        """Populate the thumbnail filmstrip from a folder."""
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Folder Not Found",
                                f"This folder no longer exists or isn't accessible:\n{folder}")
            self.filmstrip.populate("", [])
            return

        try:
            names = sorted(os.listdir(folder))
        except OSError as e:
            QMessageBox.critical(self, "Cannot Open Folder",
                                 f"Could not read folder:\n{folder}\n\n{e}")
            self.filmstrip.populate("", [])
            return

        image_files = [
            name for name in names
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        ]

        # Always open a disk-backed edit buffer for the folder so masks persist
        # across image switches even before the first save. load() seeds from an
        # existing COCO file if present, otherwise starts from an empty document.
        self._coco_buffer = CocoBuffer(folder)
        self._coco_buffer.load()
        cats = self._coco_buffer.categories()
        if cats:
            self.sidebar.set_label_classes(cats)

        self.filmstrip.populate(folder, image_files)

        if image_files:
            self._load_new_image(image_files[0])
        else:
            QMessageBox.information(self, "No Images",
                                    "No images (.jpg/.jpeg/.png/.tif/.tiff) were found in this folder.")

    # ---------------------------------------------------------
    # Clear Points
    # ---------------------------------------------------------
    def _clear_points(self):
        if self.controller is None:
            return
        self.controller.clear_points()

        # Clear paint stroke visuals if they exist
        if hasattr(self.canvas, '_paint_points'):
            self.canvas._paint_points = []

        self._update_canvas()

    # ---------------------------------------------------------
    # Interaction handlers
    # ---------------------------------------------------------
    def _on_left_click(self, x, y):
        if self.controller is None:
            return
        if self.sidebar.get_active_label() is None:
            QMessageBox.warning(self, "No Label Defined",
                "Please define at least one label class before annotating.")
            return
        self.controller.add_point(x, y, is_fg=True)
        self._update_canvas()

    def _on_right_click_handler(self, x_or_mask_id, y=None):
        """
        Handle right-clicks: either on a mask (mask_id only)
        or on empty space (x, y coordinates)
        """
        if self.controller is None:
            return

        if y is None:
            # It's a mask_id - delete it
            reply = QMessageBox.question(
                self,
                'Delete Mask',
                'Delete this mask?',
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.controller.delete_mask(x_or_mask_id)
                self.sidebar.refresh_masks()
                self._update_canvas()
        else:
            # It's coordinates - add negative point
            if self.sidebar.get_active_label() is None:
                QMessageBox.warning(self, "No Label Defined",
                    "Please define at least one label class before annotating.")
                return
            self.controller.add_point(x_or_mask_id, y, is_fg=False)
            self._update_canvas()

    def _run_segmentation(self):
        if self.controller is None:
            return

        label = self.sidebar.get_active_label()
        if label is None:
            QMessageBox.warning(self, "No Label Defined",
                "Please define at least one label class before annotating.")
            return
        color = self.sidebar.get_color_for_label(label)

        # Smart Select path
        if getattr(self, "_smart_select_active", False):
            if not self.controller.fg_points:
                QMessageBox.information(self, "Smart Select",
                    "Left-click to mark what you want (positive), "
                    "right-click to mark what to exclude (negative).")
                return
            mask_entry, info = self.controller.run_smart_select_segmentation(
                label=label, color=color)
            if mask_entry is None:
                QMessageBox.warning(self, "Smart Select",
                    "No matching region found.\n\n"
                    f"Reason: {info.get('reason', 'unknown')}\n\n"
                    "Smart Select works best on large, uniform regions. "
                    "For small or isolated objects, try SAM2 Point instead "
                    "(one or two clicks on the object).")
                return
            self.sidebar.refresh_masks()
            self._update_canvas()
            if hasattr(self.canvas, "_paint_points"):
                self.canvas._paint_points = []
            return

        mask_entry = self.controller.run_segmentation(label=label, color=color)

        if mask_entry is not None:
            if hasattr(self.canvas, '_paint_points'):
                self.canvas._paint_points = []
            self.sidebar.refresh_masks()
            self._update_canvas()

    def _on_opacity_spinbox_changed(self, percent: int):
        self._opacity_percent = percent
        if self.controller is None:
            return
        self.controller.set_opacity(int(percent / 100 * 255))
        self._write_sage_settings({"mask_opacity_percent": percent})
        self._update_canvas()

    def _on_border_checkbox_changed(self, state):
        self._show_borders = bool(state)
        self._update_canvas()

    def _on_other_checkbox_changed(self, state):
        self._display_other = bool(state)
        self._update_canvas()

    def _mask_unlabeled(self):
        """Create/regenerate the 'Other' mask covering all unlabeled pixels."""
        if self.controller is None:
            return
        color = self.sidebar.get_color_for_label("Other")
        entry = self.controller.fill_other(label="Other", color=color)
        if entry is None:
            QMessageBox.information(self, "Mask Unlabeled",
                                    "Every pixel is already labeled — nothing to mask.")
            return
        self.sidebar.refresh_masks()
        self._update_canvas()

    def _on_canvas_mask_clicked(self, mask_id: int):
        """Select tool: a mask was clicked on the canvas — highlight its list
        row and scroll to it. Clicking the mask that's flashing stops the flash."""
        if self._flash_timer.isActive() and mask_id == self._flash_mask_id:
            self._stop_flash()
        self.selected_mask_id = mask_id
        self.sidebar.select_mask_in_list(mask_id)
        self._update_canvas()

    def _warn_no_label(self):
        """Shown once when the user attempts to draw with no active label."""
        QMessageBox.warning(self, "No Label Defined",
            "Please define or select a label class before annotating.")

    def _compute_other_overlay(self):
        """Transient 'Other' preview: complement of the union of all real masks,
        in the Other color. Not stored — display only."""
        if self.controller is None or self.image_np is None:
            return None
        h, w = self.image_np.shape[:2]
        union = np.zeros((h, w), dtype=bool)
        for m in self.controller.masks:
            if m.get("is_fill"):
                continue
            union |= m["mask"].astype(bool)
        complement = ~union
        if not complement.any():
            return None
        return {"id": -999, "label": "Other", "mask": complement,
                "color": (192, 38, 211), "visible": True}

    def closeEvent(self, event):
        """On exit, if there are unsaved annotation edits, ask whether to save.
        Save promotes them to instances_default.json; Don't Save discards the
        working file so nothing is written; Cancel aborts the exit."""
        # Fold the current image's edits into the in-memory buffer so the
        # unsaved-changes check reflects everything.
        if self.current_image_path and self.controller is not None:
            self._flush_current_to_buffer()

        if self._coco_buffer is not None and self._coco_buffer.has_unsaved_changes():
            choice = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved annotation changes.\n\nSave them before exiting?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if choice == QMessageBox.Cancel:
                event.ignore()
                return
            if choice == QMessageBox.Save:
                try:
                    self._coco_buffer.save()
                except Exception as e:
                    QMessageBox.critical(self, "Save Failed",
                                         f"Could not save annotations:\n{e}")
                    event.ignore()
                    return
            else:  # Discard — delete the working file, leave the original untouched
                self._coco_buffer.discard_temp()
        elif self._coco_buffer is not None:
            # No changes: remove the seeded working file so it can't shadow the
            # original on next launch.
            self._coco_buffer.discard_temp()

        self._write_sage_settings({
            "show_borders": self._show_borders,
            "edge_trace_interval": self.canvas._edge_trace_interval,
            "edge_trace_width": self.canvas._ET_MAX_RAY,
        })
        super().closeEvent(event)

    def _on_opacity_changed(self, value):
        """Legacy hook kept for sidebar signal compatibility — converts 0-255 to spinbox %."""
        self.opacity_spinbox.blockSignals(True)
        self.opacity_spinbox.setValue(int(value / 255 * 100))
        self.opacity_spinbox.blockSignals(False)
        if self.controller is None:
            return
        self.controller.set_opacity(value)
        self._update_canvas()

    # ---------------------------------------------------------
    # Segmentation mode + polygon sampling handlers
    # ---------------------------------------------------------
    def _on_segmentation_mode_changed(self, mode: str):
        # "points" (the merged SAM2 button) and Smart Select both use paint-style
        # placement so a single click or a click-drag both work.
        self._smart_select_active = (mode == "smart_select")
        canvas_mode = "paint" if mode in ("points", "smart_select") else mode
        self.canvas.set_segmentation_mode(canvas_mode)

    def _on_polygon_sampling_changed(self, mode: str):
        if mode in ("dense", "random", "poisson"):
            self.polygon_sampling_mode = mode

    # ---------------------------------------------------------
    # Polygon drawn handler
    # ---------------------------------------------------------
    def _on_polygon_drawn(self, points):
        """
        points: list of (x, y) tuples defining a closed polygon in image coords.
        Sample a small number of interior points using the selected polygon
        sampling strategy and let SAM2 find the object naturally.
        """
        if self.controller is None:
            return
        if len(points) < 3:
            return

        interior_points = self._sample_points_inside_polygon(points)
        if not interior_points:
            return

        for x, y in interior_points:
            self.controller.add_point(x, y, is_fg=True)

        label = self.sidebar.get_active_label()
        if label is None:
            QMessageBox.warning(self, "No Label Defined",
                "Please define at least one label class before annotating.")
            self.controller.clear_points()
            return
        color = self.sidebar.get_color_for_label(label)
        mask_entry = self.controller.run_segmentation(label=label, color=color)

        if mask_entry is not None:
            self.sidebar.refresh_masks()
            self._update_canvas()

    # ---------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------
    def _update_canvas(self):
        if self.controller is None or self.renderer is None:
            return

        base_pixmap = self.renderer.base_pixmap()
        masks = self.controller.get_visible_masks()
        if self._display_other:
            other = self._compute_other_overlay()
            if other is not None:
                masks = masks + [other]
        flash_id = self._flash_mask_id if self._flash_on else -1
        pixmap_with_masks = self.renderer.overlay_masks(
            base_pixmap, masks, opacity=self.controller.opacity,
            selected_mask_id=self.selected_mask_id,
            show_borders=self._show_borders,
            flash_mask_id=flash_id,
        )
        self.canvas.set_pixmap(pixmap_with_masks)
        self.canvas.update_seed_points(
            self.controller.fg_points,
            self.controller.bg_points,
        )

        # Add invisible mask items for right-click detection
        self._add_mask_items_to_canvas()

        # Update segment button state
        self.sidebar.update_segment_button_state()

    def _add_mask_items_to_canvas(self):
        """Add invisible MaskItem polygons to canvas for right-click detection"""
        import cv2

        # First, remove any existing mask items
        for item in self.canvas._scene.items():
            if isinstance(item, MaskItem):
                self.canvas._scene.removeItem(item)

        # Add new mask items for all masks (not just visible ones, so you can delete hidden ones too)
        if self.controller is None:
            return

        for m in self.controller.masks:
            mask = m["mask"].astype(np.uint8)

            # Extract contours to get polygon points
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Add a MaskItem for each contour
            for cnt in contours:
                if len(cnt) < 3:
                    continue

                # Convert contour to list of (x, y) tuples
                polygon_points = [(float(pt[0][0]), float(pt[0][1])) for pt in cnt]

                # Create and add the invisible mask item
                mask_item = MaskItem(polygon_points, m["id"])
                if m.get("is_fill"):
                    mask_item.setZValue(50)   # 'Other' full-frame target sits below real masks
                self.canvas._scene.addItem(mask_item)

    def keyPressEvent(self, event):
        key = event.key()

        # Esc - stop an in-progress flash
        if key == Qt.Key_Escape and self._flash_timer.isActive():
            self._stop_flash()
            return

        # Enter keys - run segmentation
        if key in (13, 16777220):
            self._run_segmentation()

        # Backspace - remove last point
        elif key == Qt.Key_Backspace:
            if self.controller is not None:
                self.controller.remove_last_point()
                self._update_canvas()

        super().keyPressEvent(event)

    # ---------------------------------------------------------
    # Polygon sampling strategies
    # ---------------------------------------------------------
    def _sample_points_inside_polygon(self, polygon_points):
        polygon = np.array(polygon_points, dtype=float)
        xs = polygon[:, 0]
        ys = polygon[:, 1]

        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        if max_x <= min_x or max_y <= min_y:
            return []

        if self.polygon_sampling_mode == "dense":
            return self._sample_dense_grid(polygon, min_x, max_x, min_y, max_y)
        elif self.polygon_sampling_mode == "random":
            return self._sample_random_uniform(polygon, min_x, max_x, min_y, max_y)
        else:  # "poisson"
            return self._sample_poisson_disk(polygon, min_x, max_x, min_y, max_y)

    def _point_in_polygon(self, x, y, polygon):
        """
        Standard ray-casting point-in-polygon test.
        polygon: Nx2 array
        """
        num = len(polygon)
        inside = False
        j = num - 1
        for i in range(num):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            # Check if point is between yi and yj in y, and to the left of the edge
            intersect = ((yi > y) != (yj > y)) and (
                    x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
            )
            if intersect:
                inside = not inside
            j = i
        return inside

    def _sample_dense_grid(self, polygon, min_x, max_x, min_y, max_y, step=20):
        points = []
        # Step as int pixels
        min_x_int = int(math.floor(min_x))
        max_x_int = int(math.ceil(max_x))
        min_y_int = int(math.floor(min_y))
        max_y_int = int(math.ceil(max_y))

        for x in range(min_x_int, max_x_int + 1, step):
            for y in range(min_y_int, max_y_int + 1, step):
                if self._point_in_polygon(x + 0.5, y + 0.5, polygon):
                    points.append((x + 0.5, y + 0.5))
        return points

    def _sample_random_uniform(
            self, polygon, min_x, max_x, min_y, max_y, num_points=9
    ):
        points = []
        attempts = 0
        max_attempts = num_points * 20

        while len(points) < num_points and attempts < max_attempts:
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            if self._point_in_polygon(x, y, polygon):
                points.append((x, y))
            attempts += 1

        return points

    def _sample_poisson_disk(
            self, polygon, min_x, max_x, min_y, max_y, radius=25.0, k=30
    ):
        """
        Simple Poisson disk sampling (Bridson) restricted to polygon.
        radius: minimum distance between points.
        k: attempts per active point.
        """
        cell_size = radius / math.sqrt(2)
        grid_width = int(math.ceil((max_x - min_x) / cell_size))
        grid_height = int(math.ceil((max_y - min_y) / cell_size))

        # Grid cells store indices into samples list or -1
        grid = [[-1 for _ in range(grid_height)] for _ in range(grid_width)]
        samples = []
        active = []

        def grid_coords(px, py):
            gx = int((px - min_x) / cell_size)
            gy = int((py - min_y) / cell_size)
            return gx, gy

        # Initialize with one random point inside polygon
        init_attempts = 0
        while True:
            if init_attempts > 1000:
                return []  # fallback
            init_x = random.uniform(min_x, max_x)
            init_y = random.uniform(min_y, max_y)
            if self._point_in_polygon(init_x, init_y, polygon):
                samples.append((init_x, init_y))
                gx, gy = grid_coords(init_x, init_y)
                if 0 <= gx < grid_width and 0 <= gy < grid_height:
                    grid[gx][gy] = 0
                active.append(0)
                break
            init_attempts += 1

        while active:
            idx = random.choice(active)
            base_x, base_y = samples[idx]
            found = False

            for _ in range(k):
                r = random.uniform(radius, 2 * radius)
                theta = random.uniform(0, 2 * math.pi)
                nx = base_x + r * math.cos(theta)
                ny = base_y + r * math.sin(theta)

                if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                    continue
                if not self._point_in_polygon(nx, ny, polygon):
                    continue

                gx, gy = grid_coords(nx, ny)
                if not (0 <= gx < grid_width and 0 <= gy < grid_height):
                    continue

                ok = True
                # Check neighbors in grid
                for ix in range(max(gx - 2, 0), min(gx + 3, grid_width)):
                    for iy in range(max(gy - 2, 0), min(gy + 3, grid_height)):
                        s_idx = grid[ix][iy]
                        if s_idx != -1:
                            sx, sy = samples[s_idx]
                            if (sx - nx) ** 2 + (sy - ny) ** 2 < radius ** 2:
                                ok = False
                                break
                    if not ok:
                        break

                if ok:
                    samples.append((nx, ny))
                    grid[gx][gy] = len(samples) - 1
                    active.append(len(samples) - 1)
                    found = True

            if not found:
                active.remove(idx)

        return samples

    # ---------------------------------------------------------
    # Save COCO for ALL images with labels
    # ---------------------------------------------------------
    def save_all_coco(self):
        folder = self.folder_edit.text().strip()
        if not folder:
            return

        # Reject save if any two label classes share the same ID.
        pairs = self.sidebar.get_label_classes_with_ids()
        ids = [cid for _, cid in pairs]
        dup_ids = {i for i in ids if ids.count(i) > 1}
        if dup_ids:
            dups = ", ".join(
                f"ID {i}: " + " / ".join(n for n, c in pairs if c == i)
                for i in sorted(dup_ids)
            )
            QMessageBox.warning(
                self, "Duplicate Label IDs",
                "Cannot save — these label classes share the same ID:\n\n"
                f"{dups}\n\nGive each label a unique ID and try again."
            )
            return

        # Flush the open image into the buffer, then ensure buffer categories
        # are current, then back up the original and promote the working copy.
        if self._coco_buffer is None:
            self._coco_buffer = CocoBuffer(folder)
            self._coco_buffer.load()

        # Make sure any images visited this session but still cached are flushed.
        if self.current_image_path and self.controller is not None:
            self._flush_current_to_buffer()
        for path, masks in list(self.mask_store.items()):
            fname = os.path.basename(path)
            try:
                img = load_image_rgb(path)
                h, w = img.shape[:2]
            except Exception:
                continue
            name_to_id = {n: i for n, i in self.sidebar.get_label_classes_with_ids()}
            self._coco_buffer.flush_image(fname, masks, h, w, name_to_id)

        self._coco_buffer.set_categories(self.sidebar.get_label_classes_with_ids())

        try:
            out_path = self._coco_buffer.save()
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))
            return
        if self.controller is not None:
            self.controller.dirty = False
            self._loaded_signature = self._masks_signature()

        n_imgs = len(self._coco_buffer.doc["images"])
        n_anns = len(self._coco_buffer.doc["annotations"])
        QMessageBox.information(
            self, "COCO Saved",
            f"Backed up previous file (if any) and wrote:\n{out_path}\n\n"
            f"Images: {n_imgs}\nAnnotations: {n_anns}"
        )

    # ---------------------------------------------------------
    # Load a new image when double-clicked in the sidebar
    # ---------------------------------------------------------
    def _masks_signature(self):
        """Content hash of the current masks (id, label, visibility, pixels).
        Stable across a lossless load->flush round-trip; changes only on a real
        edit. Drift-proof: needs nothing from the controller object."""
        import hashlib
        if self.controller is None:
            return None
        h = hashlib.md5()
        for m in sorted(self.controller.masks, key=lambda mm: mm.get("id", 0)):
            h.update(str(m.get("id", "")).encode())
            h.update(str(m.get("label", "")).encode())
            h.update(b"1" if m.get("visible", True) else b"0")
            h.update(np.ascontiguousarray(m["mask"]).tobytes())
        return h.hexdigest()

    def _flush_current_to_buffer(self):
        """Write the open image's masks to the disk buffer as polygons+RLE+bbox."""
        if self._coco_buffer is None or self.current_image_path is None:
            return
        # Only flush images whose mask CONTENT actually changed since load/save.
        # A content signature is drift-proof and can't be fooled by a re-encode
        # (polygon->RLE) that leaves the masks identical — which is what caused
        # the spurious 'unsaved changes' prompt on close.
        if self.controller is None:
            return
        if self._masks_signature() == self._loaded_signature:
            return
        self.controller.recompute_fill()   # keep 'Other' exact after edits
        filename = os.path.basename(self.current_image_path)
        h, w = self.image_np.shape[:2]
        name_to_id = {n: i for n, i in self.sidebar.get_label_classes_with_ids()}
        self._coco_buffer.set_categories(self.sidebar.get_label_classes_with_ids())
        self._coco_buffer.flush_image(
            filename, self.controller.masks, h, w, name_to_id
        )

    def _masks_from_buffer(self, filename):
        """Rasterize this image's buffered annotations into controller mask entries."""
        h, w = self.image_np.shape[:2]
        entries = []
        id_to_name = {int(cid): name
                      for name, cid in self.sidebar.get_label_classes_with_ids()}
        for ann in self._coco_buffer.annotations_for(filename):
            mask = ann_to_mask(ann, h, w)
            if not mask.any():
                continue
            label = ann.get("label") or id_to_name.get(
                int(ann.get("category_id", -1)), "Region")
            entries.append({
                "id": next(self.controller._mask_id_counter),
                "label": label,
                "mask": mask,
                "color": self.sidebar.get_color_for_label(label),
                "visible": True,
                "stats": compute_mask_stats(mask),
            })
        return entries

    def _load_new_image(self, filename):
        folder = self.folder_edit.text().strip()
        if not folder:
            return

        full_path = os.path.join(folder, filename)

        # Flush outgoing image's masks to the disk buffer (frees RAM), then
        # drop them from mask_store so only the open image stays rasterized.
        if self.current_image_path and self.controller is not None:
            self._flush_current_to_buffer()
            self.mask_store.pop(self.current_image_path, None)

        # Update current image path
        self.current_image_path = full_path

        # Keep filmstrip selection in sync (e.g. when loaded programmatically)
        self.filmstrip.select_name(filename)

        # Load new image
        self.image_np = load_image_rgb(full_path)

        # Reset controller + renderer
        self.controller = SegmentationController(self.model_manager, self.image_np)
        self.selected_mask_id = -1
        self.controller.set_opacity(int(self._opacity_percent / 100 * 255))
        self.renderer = Renderer(self.image_np)

        # Restore masks: prefer in-session store, else rasterize from buffer.
        if full_path in self.mask_store:
            self.controller.masks = copy.deepcopy(self.mask_store[full_path])
        elif self._coco_buffer is not None:
            self.controller.masks = self._masks_from_buffer(filename)
        # Restored state is not an edit; snapshot it as the clean baseline.
        self.controller.dirty = False
        self._loaded_signature = self._masks_signature()

        # Auto-seed points from seed mask if available
        if self.seed_mask_path and self.auto_seed_enabled:
            self._seed_points_from_mask_current_image()

        # Drive mask sub-panel state
        if self.seed_mask_path:
            self.sidebar.set_mask_subpanel_state(MASK_STATE_PERSISTS, self.seed_mask_path)
        elif self.canvas._segmentation_mode == "mask":
            self.sidebar.set_mask_subpanel_state(MASK_STATE_LOCKED)

        self.sidebar.set_seed_controls_state(auto_on=self.auto_seed_enabled)

        # Update sidebar controller reference
        self.sidebar.controller = self.controller
        self.sidebar.refresh_masks()

        # Update canvas and reset zoom for new image
        self._update_canvas()
        self.canvas.reset_zoom()