# segment_images_tab.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Ported: direct port of Segment Images tab from GRIME_AI_ML_ImageProcessingDlg.py
# License: Apache License, Version 2.0

import os
import re

import numpy as np

from pathlib import Path
from typing import Optional
from datetime import datetime

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox, QSizePolicy, QTableWidget, QToolButton, QSplitter, QSplitterHandle, QVBoxLayout
from PyQt5.uic import loadUi

from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor
from GRIME_AI.dialogs.ML_image_processing.model_config_manager import ModelConfigManager
from GRIME_AI.utils.resource_utils import ui_path
from GRIME_AI.GRIME_AI_CSS_Styles import BUTTON_CSS_STEEL_BLUE, BUTTON_CSS_RED_OUTLINE, BUTTON_CSS_YELLOW_OUTLINE
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

# import torch if using torch metadata extraction
try:
    import torch
except Exception:
    torch = None


# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===            HELPER FUNCTIONS            ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
def _normalize_labels(raw):
    """
    Normalize checkpoint label data into a list of dicts with at minimum a
    'name' key, and 'id' where available.  Preserving 'id' is critical so
    that the correct COCO category ID is used at inference rather than the
    listbox position.
    """
    if raw is None:
        return None
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        # Already a list of dicts — preserve id if present
        result = []
        for c in raw:
            name = c.get("name") or c.get("label") or c.get("class") or repr(c)
            entry = {"name": str(name)}
            if "id" in c:
                entry["id"] = c["id"]
            result.append(entry)
        return result
    if isinstance(raw, dict):
        # Dict keyed by id or name — convert to list of dicts
        try:
            items = sorted(raw.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])
            return [{"id": int(k) if str(k).isdigit() else None, "name": str(v)} for k, v in items]
        except Exception:
            return [{"name": str(v)} for v in raw.values()]
    if isinstance(raw, str):
        return [{"name": raw}]
    try:
        return [{"name": str(x)} for x in list(raw)]
    except Exception:
        return None

# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===         class SegmentImagesTab         ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class ToggleSplitterHandle(QSplitterHandle):
    """Splitter handle with a small toggle button to collapse/expand the right panel."""

    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self._collapsed_sizes = None

        self._btn = QToolButton(self)
        self._btn.setFixedSize(20, 48)
        self._btn.setText("◀")
        self._btn.setToolTip("Hide/show metadata panel")
        self._btn.setStyleSheet("""
QToolButton {
    background-color: palette(button);
    border: 1px solid palette(mid);
    border-radius: 4px;
    font-size: 10px;
    color: palette(buttontext);
}
QToolButton:hover { background-color: palette(highlight); color: white; }
""")
        self._btn.clicked.connect(self._toggle)
        self._btn.raise_()
        self._btn.show()

    def sizeHint(self):
        sh = super().sizeHint()
        return QtCore.QSize(max(sh.width(), 20), sh.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        bw, bh = self._btn.width(), self._btn.height()
        x = (self.width() - bw) // 2
        y = max(0, (self.height() - bh) // 2)
        self._btn.move(x, y)
        self._btn.raise_()

    def paintEvent(self, event):
        # Paint a visible divider line behind the button
        from PyQt5.QtGui import QPainter, QColor
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(self.palette().Button))
        painter.setPen(QColor("#999999"))
        cx = self.width() // 2
        painter.drawLine(cx, 0, cx, self.height())
        painter.end()

    def _toggle(self):
        splitter = self.splitter()
        sizes = splitter.sizes()
        if sizes[1] > 0:
            self._collapsed_sizes = sizes[1]
            splitter.setSizes([sum(sizes), 0])
            self._btn.setText("▶")
        else:
            restore = self._collapsed_sizes or 250
            total = sum(sizes)
            splitter.setSizes([total - restore, restore])
            self._btn.setText("◀")


class _ToggleSplitter(QSplitter):
    """QSplitter that uses ToggleSplitterHandle for the main left/right split."""

    def createHandle(self):
        return ToggleSplitterHandle(self.orientation(), self)


class SegmentImagesTab(QWidget):
    """
    Direct port of the Segment Images tab UI and logic.
    This preserves original comments, method names, and expected attributes.
    """

    # Signal to notify parent/dialog that segmentation should start.
    ml_segment_signal = pyqtSignal()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        # Load the dedicated Segment Images UI
        loadUi(ui_path("ML_image_processing/segment_images_tab.ui"), self)

        self.setup_ui_properties()

        # Preserve dialog-level state expected by methods copied from the original dialog
        self.transferred_items = set()
        self.selected_label_categories = []
        self.categories_available = False

        # Multi-folder state: list of dicts {"path": str, "recursive": bool}
        self.image_folders = []

        # Default selection
        self.selected_segment_model = "sam2"

        layout = self.horizontalLayoutSegmentImages
        layout.setStretch(0, 4)  # left content area
        layout.setStretch(1, 1)  # right splitter panel

        settings_folder = Path(GRIME_AI_Save_Utils().get_settings_folder()).resolve()
        config_file = (settings_folder / "site_config.json").resolve()

        # Backup existing config if present
        if not config_file.exists():
            # Start with template
            self.site_config = self.create_site_config_template()
        else:
            self.site_config = JsonEditor().load_json_file(config_file)

        # WIRE SIGNALS TO WIDGETS (BUTTONS, RADIOBUTTONS, ETC.)
        self.setup_connections()

        self.populate_segment_images_tab()

        self.setup_from_config_file()

        # UPDATE BUTTON STATE AFTER INITIALIZATION
        try:
            self.updateSegmentButtonState()
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_from_config_file(self):
        """
        Initialize dialog controls from a configuration dictionary.
        """
        load_model_conf = self.site_config.get("load_model", {})
        model_type = load_model_conf.get("MODEL", "").lower()

        # Select the correct model file key based on MODEL type
        if model_type == "sam2":
            model_file = load_model_conf.get("SAM2_MODEL", "")
        elif model_type == "segformer":
            model_file = load_model_conf.get("SEGFORMER_MODEL", "")
        elif model_type == "yolo":
            model_file = load_model_conf.get("YOLO_MODEL", "")
        else:
            model_file = ""

        if model_file:
            self.lineEdit_segmentation_model_file.setText(model_file)
            # Label loading is deferred to showEvent (see _labels_loaded_once) so any
            # warning dialog appears only after the ML dialog is fully constructed and
            # visible, never behind the "ML Loading" splash during construction.
        else:
            self.lineEdit_segmentation_model_file.clear()

        # Image folders list
        saved_folders = load_model_conf.get("segmentation_image_folders", [])
        self.image_folders = saved_folders if isinstance(saved_folders, list) else []
        self._refresh_folder_list_widget()

        # Output folder. Prefer the saved model config; otherwise fall back to a
        # value pushed by an activated recipe (apply_recipe writes this key).
        output_folder = load_model_conf.get("output_folder", "")
        if not output_folder:
            try:
                output_folder = JsonEditor().getValue("Model_Segmentation_Output_Folder") or ""
            except Exception:
                output_folder = ""
        if output_folder:
            self.lineEdit_output_folder.setText(output_folder)
        else:
            self.lineEdit_output_folder.clear()

        # Checkboxes (default to True if missing)
        self.checkBox_save_predicted_masks.setChecked(load_model_conf.get("save_model_masks", True))
        self.checkBox_copyOriginalModelImage.setChecked(load_model_conf.get("copy_original_model_image", True))
        self.checkBox_save_probability_maps.setChecked(load_model_conf.get("save_probability_maps", True))
        self.checkBox_save_diagnostic_panels.setChecked(load_model_conf.get("save_diagnostic_panels", False))

        # Season filter
        # Segment seasons — restore from config
        segment_seasons = load_model_conf.get("segment_seasons", [])
        self._set_segment_seasons(segment_seasons)

        # Set the appropriate radio button based on MODEL
        if model_type == "sam2":
            self.radioButton_segment_model_sam2.setChecked(True)
        elif model_type == "segformer":
            self.radioButton_segment_model_segformer.setChecked(True)
        elif model_type == "yolo":
            self.radioButton_segment_model_yolo.setChecked(True)
        else:
            # No model type stored — default to SAM2
            self.radioButton_segment_model_sam2.setChecked(True)

        # Blob filter mode (SAM2 post-processing)
        _blob_mode = str(load_model_conf.get("blob_filter_mode", "mahalanobis")).strip().lower()
        if _blob_mode == "circular":
            self.radioButton_blob_circular.setChecked(True)
        elif _blob_mode == "knn":
            self.radioButton_blob_knn.setChecked(True)
        elif _blob_mode == "none":
            self.radioButton_blob_none.setChecked(True)
        else:
            self.radioButton_blob_mahalanobis.setChecked(True)
        self._update_blob_filter_enabled()

        # Select items in the listbox based on SEGMENTATION_CATEGORIES
        stored_categories = load_model_conf.get("SEGMENTATION_CATEGORIES", [])
        if stored_categories:
            for idx in range(self.listWidget_labels.count()):
                item = self.listWidget_labels.item(idx)
                if item and item.text().strip() in stored_categories:
                    item.setSelected(True)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def showEvent(self, event):
        """Called when tab becomes visible - populate labels/metadata if model path exists.
        Deferred here (not __init__) so any warning dialog appears on top of the visible
        dialog rather than behind the 'ML Loading' splash during construction."""
        super().showEvent(event)

        # Check if model path is populated
        model_path = self.lineEdit_segmentation_model_file.text().strip()
        if model_path and os.path.exists(model_path):
            if not getattr(self, "_labels_loaded_once", False):
                self._labels_loaded_once = True
                try:
                    self.populate_model_labels(model_path)
                except Exception as e:
                    print(f"Failed to populate labels on tab show: {e}")
            try:
                self.populate_model_metadata(model_path)
            except Exception as e:
                print(f"Failed to populate metadata on tab show: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_values(self, site_config: dict) -> dict:
        """
        Collect values from dialog controls and update the given site_config dictionary.
        Returns the updated site_config.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # _____ GRIME AI ML parameters _______________________________________
        site_config["save_model_masks"] = self.checkBox_save_predicted_masks.isChecked()
        site_config["copy_original_model_image"] = self.checkBox_copyOriginalModelImage.isChecked()
        site_config["save_probability_maps"] = self.checkBox_save_probability_maps.isChecked()
        site_config["save_diagnostic_panels"] = self.checkBox_save_diagnostic_panels.isChecked()

        site_config.setdefault("load_model", {})

        # Save the full folder list
        site_config["load_model"]["segmentation_image_folders"] = self.image_folders

        # segmentation_images_path: always the first folder — inference engine expects a single directory
        first_folder = self.image_folders[0]["path"] if self.image_folders else ""
        site_config["load_model"]["segmentation_images_path"] = first_folder

        # Output folder — user-specified, with timestamped subfolder
        output_folder = self.lineEdit_output_folder.text().strip()
        site_config["load_model"]["output_folder"] = output_folder
        if output_folder:
            predictions_output_path = os.path.normpath(os.path.join(output_folder, f"{timestamp}_predictions"))
            site_config["load_model"]["predictions_output_path"] = predictions_output_path

        # Selected segmentation categories from listbox
        selected_labels = [
            self.listWidget_labels.item(idx).text().strip()
            for idx in range(self.listWidget_labels.count())
            if self.listWidget_labels.item(idx) and self.listWidget_labels.item(idx).isSelected()
        ]
        site_config["load_model"]["SEGMENTATION_CATEGORIES"] = selected_labels

        # Season filter
        # Segment seasons
        site_config["load_model"]["segment_seasons"] = self._get_segment_seasons()

        # Selected model type from radio buttons
        if self.radioButton_segment_model_sam2.isChecked():
            site_config["load_model"]["MODEL"] = "sam2"
            site_config["load_model"]["SAM2_MODEL"] = self.lineEdit_segmentation_model_file.text().strip()
            site_config["load_model"]["SEGFORMER_MODEL"] = ""
            site_config["load_model"]["YOLO_MODEL"] = ""
        elif self.radioButton_segment_model_segformer.isChecked():
            site_config["load_model"]["MODEL"] = "segformer"
            site_config["load_model"]["SAM2_MODEL"] = ""
            site_config["load_model"]["SEGFORMER_MODEL"] = self.lineEdit_segmentation_model_file.text().strip()
            site_config["load_model"]["YOLO_MODEL"] = ""
        elif self.radioButton_segment_model_yolo.isChecked():
            site_config["load_model"]["MODEL"] = "yolo"
            site_config["load_model"]["SAM2_MODEL"] = ""
            site_config["load_model"]["SEGFORMER_MODEL"] = ""
            site_config["load_model"]["YOLO_MODEL"] = self.lineEdit_segmentation_model_file.text().strip()
        else:
            site_config["load_model"]["MODEL"] = ""
            site_config["load_model"]["SAM2_MODEL"] = ""
            site_config["load_model"]["SEGFORMER_MODEL"] = ""
            site_config["load_model"]["YOLO_MODEL"] = ""

        # Blob filter mode (SAM2 post-processing)
        if self.radioButton_blob_circular.isChecked():
            site_config["load_model"]["blob_filter_mode"] = "circular"
        elif self.radioButton_blob_knn.isChecked():
            site_config["load_model"]["blob_filter_mode"] = "knn"
        elif self.radioButton_blob_none.isChecked():
            site_config["load_model"]["blob_filter_mode"] = "none"
        else:
            site_config["load_model"]["blob_filter_mode"] = "mahalanobis"

        return site_config

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def populate_segment_images_tab(self):
        """
        Initialize segmentation tab UI state that the original dialog expected.
        Mirrors original dialog setup: ensures attributes exist and sets sensible defaults.
        """
        # Ensure attributes expected by other methods exist
        if not hasattr(self, "transferred_items"):
            self.transferred_items = set()
        if not hasattr(self, "image_folders"):
            self.image_folders = []
        if not hasattr(self, "selected_label_categories"):
            self.selected_label_categories = []
        if not hasattr(self, "categories_available"):
            self.categories_available = False

        # Clear labels and model path by default (keeps behavior predictable)
        try:
            self.listWidget_labels.clear()
        except Exception:
            pass

        try:
            self.lineEdit_segmentation_model_file.setText("")
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_ui_properties(self):
        """Set size policies and layout stretch factors."""
        self.pushButton_Segment.setMinimumHeight(38)
        self.pushButton_Segment.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        try:
            self.listWidget_labels.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass

        try:
            self.splitter_right_panel.setSizes([1000])
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_connections(self):
        """Connect signals with their slot methods (mirrors original dialog)."""

        # Segment model radio buttons
        self.radioButton_segment_model_sam2.toggled.connect(lambda checked: self.set_segment_model("sam2", checked))
        self.radioButton_segment_model_sam2.toggled.connect(self.update_model_config)

        self.radioButton_segment_model_segformer.toggled.connect(lambda checked: self.set_segment_model("segformer", checked))
        self.radioButton_segment_model_segformer.toggled.connect(self.update_model_config)

        self.radioButton_segment_model_yolo.toggled.connect(lambda checked: self.set_segment_model("yolo", checked))
        self.radioButton_segment_model_yolo.toggled.connect(self.update_model_config)

        # Blob filter mode radios (SAM2 post-processing)
        self.radioButton_blob_mahalanobis.toggled.connect(self.update_model_config)
        self.radioButton_blob_circular.toggled.connect(self.update_model_config)
        self.radioButton_blob_none.toggled.connect(self.update_model_config)
        self.radioButton_blob_knn.toggled.connect(self.update_model_config)
        # Blob filter is SAM2-only — enable the group only when SAM2 is selected
        self.radioButton_segment_model_sam2.toggled.connect(lambda _c: self._update_blob_filter_enabled())
        self.radioButton_segment_model_segformer.toggled.connect(lambda _c: self._update_blob_filter_enabled())
        self.radioButton_segment_model_yolo.toggled.connect(lambda _c: self._update_blob_filter_enabled())

        # Buttons
        self.pushButton_Select_Model.clicked.connect(self.select_segmentation_model)
        self.pushButton_Select_Model.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_Select_Output_Folder.clicked.connect(self.select_output_folder)
        self.pushButton_Select_Output_Folder.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_Add_Folder.clicked.connect(self.add_folder_flat)
        self.pushButton_Add_Folder.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_Add_Folder_Recursive.clicked.connect(self.add_folder_recursive)
        self.pushButton_Add_Folder_Recursive.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_Remove_Folder.clicked.connect(self.remove_selected_folder)
        self.pushButton_Remove_Folder.setStyleSheet(BUTTON_CSS_YELLOW_OUTLINE)

        self.pushButton_Clear_Folders.clicked.connect(self.clear_all_folders)
        self.pushButton_Clear_Folders.setStyleSheet(BUTTON_CSS_RED_OUTLINE)

        self.pushButton_Segment.clicked.connect(self.segment_images)
        self.pushButton_Segment.setText("▶  Segment Images")
        self.pushButton_Segment.setStyleSheet("""
QPushButton {
    background-color: steelblue;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 12pt;
    font-weight: bold;
}
QPushButton:hover { background-color: #5a93c2; }
QPushButton:disabled {
    background-color: gray;
    color: #ccc;
}
""")

        # ── Rounded corners and styling — set per-widget for guaranteed rendering ──
        # Per-widget setStyleSheet with a self-referencing selector (e.g.
        # QGroupBox#name { ... }) has higher specificity than any parent cascade
        # or qdarkstyle rule, so corners and title positioning render correctly
        # in both light and dark mode.

        GROUPBOX_LIGHT = """
QGroupBox {{
    border: 1px solid #aaa;
    border-radius: 6px;
    margin-top: 18px;
    padding-top: 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    background-color: palette(window);
    color: palette(windowtext);
}}
"""
        LISTWIDGET_LIGHT = """
QListWidget {{
    border: 1px solid #aaa;
    border-radius: 6px;
}}
"""
        TABLEWIDGET_LIGHT = """
QTableWidget {{
    border: 1px solid #aaa;
    border-radius: 6px;
}}
"""
        LINEEDIT_LIGHT = """
QLineEdit {
    border: 1px solid #888;
    border-radius: 6px;
    padding: 2px 6px;
}
QLineEdit:focus {
    border: 1px solid #3b6a93;
}
"""
        for gb in self.findChildren(QtWidgets.QGroupBox):
            gb.setStyleSheet(GROUPBOX_LIGHT)
        for lw in self.findChildren(QtWidgets.QListWidget):
            lw.setStyleSheet(LISTWIDGET_LIGHT)
        for tw in self.findChildren(QtWidgets.QTableWidget):
            tw.setStyleSheet(TABLEWIDGET_LIGHT)
        self.lineEdit_segmentation_model_file.setStyleSheet(LINEEDIT_LIGHT)
        self.lineEdit_output_folder.setStyleSheet(LINEEDIT_LIGHT)

        # ── Right margin on left panel so it doesn't butt against metadata ────
        try:
            self.widget_left_panel.layout().setContentsMargins(4, 4, 8, 4)
        except Exception:
            pass

        # ── Dark mode support ─────────────────────────────────────────────────
        self._apply_dark_mode_if_active()

        # ── Splitter initial sizes (left panel gets most space) ───────────────
        try:
            # Re-parent splitter_main children into a _ToggleSplitter so the
            # custom toggle handle is used without requiring a .ui change.
            old = self.splitter_main
            toggle_splitter = _ToggleSplitter(QtCore.Qt.Horizontal, self)
            toggle_splitter.setObjectName("splitter_main")
            toggle_splitter.setChildrenCollapsible(False)

            # Move children across
            while old.count():
                w = old.widget(0)
                w.setParent(None)
                toggle_splitter.addWidget(w)

            # Swap into the parent layout
            layout = old.parent().layout()
            idx = layout.indexOf(old)
            layout.removeWidget(old)
            old.deleteLater()
            layout.insertWidget(idx, toggle_splitter)

            self.splitter_main = toggle_splitter
            self.splitter_main.setHandleWidth(20)
            self.splitter_main.setSizes([650, 250])
        except Exception as e:
            print(f"Toggle splitter setup failed: {e}")
            try:
                self.splitter_main.setSizes([650, 250])
            except Exception:
                pass

        # Line edits and checkboxes → flush config on change
        self.lineEdit_segmentation_model_file.textChanged.connect(self.onModelPathChanged)
        self.lineEdit_segmentation_model_file.textChanged.connect(self.update_model_config)

        self.checkBox_save_predicted_masks.toggled.connect(self.on_save_predicted_masks_toggled)
        self.checkBox_save_predicted_masks.toggled.connect(self.update_model_config)

        self.checkBox_save_probability_maps.toggled.connect(self.on_save_probability_maps_toggled)
        self.checkBox_save_probability_maps.toggled.connect(self.update_model_config)

        self.checkBox_copyOriginalModelImage.toggled.connect(self.on_copy_original_toggled)
        self.checkBox_copyOriginalModelImage.toggled.connect(self.update_model_config)

        self.checkBox_save_diagnostic_panels.toggled.connect(self.on_save_diagnostic_panels_toggled)
        self.checkBox_save_diagnostic_panels.toggled.connect(self.update_model_config)

        # Segment seasons dual listbox — double-click to move items between lists.
        self.listWidget_availableSegmentSeasons.itemDoubleClicked.connect(
            self._on_available_season_double_clicked
        )
        self.listWidget_segmentSeasons.itemDoubleClicked.connect(
            self._on_selected_season_double_clicked
        )
        self._init_segment_season_lists()

        # Labels list affects segment button state
        try:
            self.listWidget_labels.itemSelectionChanged.connect(self.updateSegmentButtonState)
            self.listWidget_labels.itemSelectionChanged.connect(self.update_model_config)
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    def _apply_dark_mode_if_active(self):
        """Apply a dark palette and per-widget dark stylesheet if in dark mode."""
        try:
            app_palette = self.palette()
            window_color = app_palette.color(QPalette.Window)
            is_dark = window_color.lightness() < 128
            if is_dark:
                dark = QPalette()
                dark.setColor(QPalette.Window,          QColor(45,  45,  45))
                dark.setColor(QPalette.WindowText,      QColor(220, 220, 220))
                dark.setColor(QPalette.Base,            QColor(30,  30,  30))
                dark.setColor(QPalette.AlternateBase,   QColor(50,  50,  50))
                dark.setColor(QPalette.Text,            QColor(220, 220, 220))
                dark.setColor(QPalette.Button,          QColor(55,  55,  55))
                dark.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
                dark.setColor(QPalette.Highlight,       QColor(42,  130, 218))
                dark.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
                dark.setColor(QPalette.ToolTipBase,     QColor(30,  30,  30))
                dark.setColor(QPalette.ToolTipText,     QColor(220, 220, 220))
                dark.setColor(QPalette.PlaceholderText, QColor(140, 140, 140))
                self.setPalette(dark)

                # Detect the actual window background from qdarkstyle's palette
                # so the title background matches exactly.
                bg = app_palette.color(QPalette.Window).name()

                GROUPBOX_DARK = f"""
QGroupBox {{
    border: 1px solid #555;
    border-radius: 6px;
    margin-top: 18px;
    padding-top: 6px;
    color: #dcdcdc;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    background-color: {bg};
    color: #dcdcdc;
}}
"""
                LISTWIDGET_DARK = """
QListWidget {
    border: 1px solid #555;
    border-radius: 6px;
    background-color: #1e1e1e;
    color: #dcdcdc;
}
"""
                TABLEWIDGET_DARK = """
QTableWidget {
    border: 1px solid #555;
    border-radius: 6px;
    background-color: #1e1e1e;
    color: #dcdcdc;
}
"""
                LINEEDIT_DARK = """
QLineEdit {
    border: 1px solid #555;
    border-radius: 6px;
    padding: 2px 6px;
    background-color: #1e1e1e;
    color: #dcdcdc;
}
QLineEdit:focus {
    border: 1px solid #5a93c2;
}
"""
                for gb in self.findChildren(QtWidgets.QGroupBox):
                    gb.setStyleSheet(GROUPBOX_DARK)
                for lw in self.findChildren(QtWidgets.QListWidget):
                    lw.setStyleSheet(LISTWIDGET_DARK)
                for tw in self.findChildren(QtWidgets.QTableWidget):
                    tw.setStyleSheet(TABLEWIDGET_DARK)
                self.lineEdit_segmentation_model_file.setStyleSheet(LINEEDIT_DARK)
                self.lineEdit_output_folder.setStyleSheet(LINEEDIT_DARK)
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    def set_segment_model(self, model_name: str, checked: bool):
        """Update selected_segment_model when a radio button is toggled on."""
        if checked:
            if model_name == "yolo":
                import importlib.util
                if importlib.util.find_spec("ultralytics") is None:
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self,
                        "ultralytics Not Installed",
                        "The ultralytics package is required for YOLOv11-seg inference "
                        "but is not installed in this environment.\n\n"
                        "Install it with:\n"
                        "  pip install ultralytics\n\n"
                        "Reverting to SAM2. All other GRIME AI features remain available."
                    )
                    self.radioButton_segment_model_sam2.setChecked(True)
                    return

            self.selected_segment_model = model_name
            print(f"Selected segment model: {self.selected_segment_model}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def select_segmentation_model(self):
        """
        Open a file dialog to select a segmentation model file (only .torch files).
        Clears the label list and populates it from the model metadata.
        """
        model_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Segmentation Model",
            "",
            "Model Files (*.torch *.pt)"
        )

        if model_file:
            self.lineEdit_segmentation_model_file.setText(model_file)

            print("Segmentation model selected:", model_file)

            # Clear the label listbox before repopulating
            self.listWidget_labels.clear()

            # Try to populate labels from model metadata if available
            try:
                self.populate_model_labels(model_file)
            except Exception as e:
                print(f"populate_model_labels failed: {e}")

            # Populate model metadata listbox
            try:
                self.populate_model_metadata(model_file)
            except Exception as e:
                print(f"populate_model_metadata failed: {e}")

            self.updateSegmentButtonState()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def select_output_folder(self):
        """Open directory dialog to choose predictions output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder for Predictions")
        if folder:
            self.lineEdit_output_folder.setText(folder.replace("\\", "/"))
            self.updateSegmentButtonState()
            self.update_model_config()

    def set_output_folder(self, folder):
        """Set the predictions output folder programmatically (e.g. from an
        activated recipe). Mirrors the visible effect of choosing one."""
        self.lineEdit_output_folder.setText((folder or "").replace("\\", "/"))
        try:
            self.updateSegmentButtonState()
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def add_folder_flat(self):
        """Open directory dialog and add as a flat (non-recursive) folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if folder:
            folder = folder.replace("\\", "/")
            if not any(f["path"] == folder for f in self.image_folders):
                self.image_folders.append({"path": folder, "recursive": False})
                self._refresh_folder_list_widget()
                self.updateSegmentButtonState()
                self.update_model_config()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def add_folder_recursive(self):
        """
        Open directory dialog, walk the entire tree, and add each subfolder
        that directly contains images as an individual flat entry.
        This allows the inference engine (which expects a single flat folder)
        to process each folder without modification.
        """
        root = QFileDialog.getExistingDirectory(self, "Select Root Images Folder (Recursive)")
        if not root:
            return
        root = root.replace("\\", "/")
        image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        import re as _re
        _predictions_pattern = _re.compile(r'\d{8}_\d{6}_predictions', _re.IGNORECASE)
        added = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirpath = dirpath.replace("\\", "/")
            # Skip any predictions output folder (e.g. 20260419_164239_predictions (sam2))
            folder_name = Path(dirpath).name
            if _predictions_pattern.search(folder_name):
                dirnames.clear()  # don't descend into subfolders of predictions either
                continue
            has_images = any(Path(f).suffix.lower() in image_extensions for f in filenames)
            if has_images:
                if not any(entry["path"] == dirpath for entry in self.image_folders):
                    self.image_folders.append({"path": dirpath, "recursive": False})
                    added += 1
        if added:
            self._refresh_folder_list_widget()
            self.updateSegmentButtonState()
            self.update_model_config()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def remove_selected_folder(self):
        """Remove the currently selected folder from the list."""
        row = self.listWidget_image_folders.currentRow()
        if row >= 0:
            del self.image_folders[row]
            self._refresh_folder_list_widget()
            self.updateSegmentButtonState()
            self.update_model_config()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def clear_all_folders(self):
        """Clear all folders from the list."""
        self.image_folders.clear()
        self._refresh_folder_list_widget()
        self.updateSegmentButtonState()
        self.update_model_config()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _refresh_folder_list_widget(self):
        """Rebuild the listWidget_image_folders display from self.image_folders."""
        self.listWidget_image_folders.clear()
        image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        total_images = 0
        for entry in self.image_folders:
            mode = "Recursive" if entry["recursive"] else "Flat"
            self.listWidget_image_folders.addItem(f"[{mode}]  {entry['path']}")
            total_images += self._count_images(entry["path"], entry["recursive"], image_extensions)
        if self.image_folders:
            count_text = f"{len(self.image_folders)} folder(s) — {total_images} image(s) found"
        else:
            count_text = "No folders selected"
        try:
            self.label_folder_image_count.setText(count_text)
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _count_images(self, folder: str, recursive: bool, extensions: set) -> int:
        """Return the number of image files in folder (optionally recursive)."""
        try:
            p = Path(folder)
            if not p.is_dir():
                return 0
            pattern = "**/*" if recursive else "*"
            return sum(1 for f in p.glob(pattern) if f.is_file() and f.suffix.lower() in extensions)
        except Exception:
            return 0

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _collect_image_paths(self) -> list:
        """Return a flat list of all image file paths across all registered folders."""
        image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        paths = []
        for entry in self.image_folders:
            p = Path(entry["path"])
            if not p.is_dir():
                continue
            pattern = "**/*" if entry["recursive"] else "*"
            for f in sorted(p.glob(pattern)):
                if f.is_file() and f.suffix.lower() in image_extensions:
                    paths.append(str(f).replace("\\", "/"))
        return paths

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def onModelPathChanged(self, path: str):
        """React to model path changes (basic validation)."""
        if path:
            p = Path(path)
            if not p.exists():
                print(f"[SegmentImagesTab] Warning: model path does not exist: {path}")
        self.updateSegmentButtonState()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def updateSegmentButtonState(self):
        """
        Enable the Segment button only when a model path and at least one
        image folder are provided, and at least one label is selected if labels exist.
        """
        model_path = self.lineEdit_segmentation_model_file.text().strip()
        has_folders = bool(self.image_folders)
        output_folder = self.lineEdit_output_folder.text().strip()

        labels_exist = (self.listWidget_labels.count() > 0)
        labels_selected = True
        if labels_exist:
            labels_selected = any(self.listWidget_labels.item(i).isSelected() for i in range(self.listWidget_labels.count()))

        enabled = bool(model_path and has_folders and output_folder and labels_selected)
        try:
            self.pushButton_Segment.setEnabled(enabled)
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def segment_images(self):
        """
        Called when the Segment button is clicked.
        Validates label selections, updates configuration, runs segmentation,
        then post-processes outputs according to the two checkboxes.
        """
        if self.categories_available:
            # Ensure that at least one label is selected
            selected_labels = []
            for idx in range(self.listWidget_labels.count()):
                item = self.listWidget_labels.item(idx)
                if item and item.isSelected():
                    selected_labels.append(item.text())

            if not selected_labels:
                QMessageBox.warning(self, "Segmentation Error",
                                    "Please select at least one label before segmenting images.")
                return

            # Populate selected_label_categories from selection
            # Use the real category ID stored via UserRole, not listbox position
            self.selected_label_categories = []
            if self.categories_available == True:
                for idx in range(self.listWidget_labels.count()):
                    item = self.listWidget_labels.item(idx)
                    item_text = item.text().strip()
                    if item_text and item_text in selected_labels:
                        cat_id = item.data(QtCore.Qt.UserRole)
                        if cat_id is None:
                            cat_id = idx + 1  # fallback for older checkpoints without stored IDs
                        self.selected_label_categories.append({
                            "id": cat_id,
                            "name": item_text
                        })

            print("Selected categories:", self.selected_label_categories)
        else:   # IF IT IS AN OLDER MODEL NOT CONTAINING LABELS, DEFAULT TO ID: 2, NAME: Water
            self.selected_label_categories = [{"id": 2, "name": "water"}]

        # update model config file (JSON) for downstream pipeline
        self.update_model_config()

        # Kick off the actual segmentation
        self.ml_segment_signal.emit()

        # Close dialog as “Accepted”
        #QtCore.QMetaObject.invokeMethod(
        #    self, 'done', Qt.QueuedConnection,
        #    QtCore.Q_ARG(int, QDialog.Accepted)
        #)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _update_blob_filter_enabled(self):
        """The blob filter is SAM2-only; enable its group only for SAM2."""
        try:
            self.groupBox_blobFilter_segment.setEnabled(
                self.radioButton_segment_model_sam2.isChecked())
        except Exception:
            pass

    def update_model_config(self):
        """
        Gather all dialog values and update the JSON configuration file.
        Preserves existing settings from other classes by merging instead of resetting.
        """
        settings_folder = Path(GRIME_AI_Save_Utils().get_settings_folder()).resolve()
        config_file = (settings_folder / "site_config.json").resolve()

        # Use ModelConfigManager to handle backup + load
        manager = ModelConfigManager(filepath=config_file)
        site_config = manager.backup_config()  # backs up existing file and loads config (or {} if none)

        # Merge values from controls directly into site_config (includes segmentation categories now)
        site_config = self.get_values(site_config)

        # Save updated config using ModelConfigManager
        manager.config = site_config
        try:
            manager.save_config(config_file)
            print("Custom JSON file 'site_config.json' updated successfully.")
        except Exception as e:
            print(f"Failed to save config: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def on_save_predicted_masks_toggled(self, checked: bool):
        print(f"Save Masks checkbox toggled: {checked}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def on_save_probability_maps_toggled(self, checked: bool):
        print(f"Save Probability Maps checkbox toggled: {checked}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def on_copy_original_toggled(self, checked: bool):
        print(f"Copy Original Image checkbox toggled: {checked}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def on_save_diagnostic_panels_toggled(self, checked: bool):
        print(f"Save Diagnostic Panels checkbox toggled: {checked}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def populate_model_labels(self, model_path):
        """
        Load label categories from a torch checkpoint that contains metadata.
        Expected checkpoint layout (example you provided):
            ckpt = {
                "model_state_dict": ...,
                "categories": self.categories,
                "creation_UTC": ...,
                ...
            }
        If categories are not found, the UI is updated exactly as before.
        """
        # Try safe allowlist first, then fall back to full load only if necessary
        ckpt = None
        labels = None

        try:
            # Minimal allowlist for the numpy scalar global reported by PyTorch
            allowlist = [np.core.multiarray.scalar]

            try:
                with torch.serialization.safe_globals(allowlist):
                    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            except Exception as e_safe:
                # If the error message names additional globals, try to extract numpy-like symbols and add them
                msg = str(e_safe)
                found = re.findall(r"([A-Za-z0-9_\.]+numpy[^\s\]\)]+)", msg)
                for sym in found:
                    try:
                        parts = sym.split(".")
                        mod = __import__(".".join(parts[:-1]), fromlist=[parts[-1]])
                        obj = getattr(mod, parts[-1])
                        if obj not in allowlist:
                            allowlist.append(obj)
                    except Exception:
                        pass
                # Retry with expanded allowlist
                with torch.serialization.safe_globals(allowlist):
                    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)

        except Exception as e_outer:
            # Safe allowlist failed; fallback to full unpickle only if you trust the file source
            print("Safe allowlist load failed:", e_outer)
            try:
                print("Falling back to full load with weights_only=False (trusted source only).")
                ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            except Exception as e_full:
                print("Full load failed:", e_full)
                ckpt = None

        # Extract categories if checkpoint loaded
        if isinstance(ckpt, dict):
            labels = ckpt.get("categories") or ckpt.get("classes") or ckpt.get("labels") \
                     or (ckpt.get("meta") or {}).get("classes") or ckpt.get("target_label")

        labels = _normalize_labels(labels)

        try:
            target_label = ckpt.get("target_label")
            if not target_label:
                print("No target label found in checkpoint")
        except Exception:
            target_label = None

        # Clear previous entries
        self.listWidget_labels.clear()

        # No labels found — same UI behavior as before
        if not labels:
            QMessageBox.warning(
                self,
                "Load Model Categories",
                "No category list found in:\n" + model_path
            )
            self.categories_available = False
            self.listWidget_labels.addItem("<Older model format>")
            self.listWidget_labels.addItem("<Labels unavailable>")
            self.listWidget_labels.addItem("<ID 2 : Water assumed>")
            self.listWidget_labels.setDisabled(True)
            return

        # Populate listbox with label name; store real category ID via UserRole
        self.categories_available = True
        self.listWidget_labels.setDisabled(False)
        for entry in labels:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("label") or entry.get("class") or entry.get("title")
                if name is None:
                    continue
                cat_id = entry.get("id")
            else:
                name = str(entry)
                cat_id = None
            item = QtWidgets.QListWidgetItem(str(name))
            if cat_id is not None:
                item.setData(QtCore.Qt.UserRole, int(cat_id))
            self.listWidget_labels.addItem(item)

        self.select_label(target_label)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def populate_model_metadata(self, model_path):
        """Load and display ALL metadata from torch checkpoint as a two-column table."""
        from PyQt5.QtWidgets import QTableWidgetItem
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QFont, QColor

        # Display-name map (display only — raw checkpoint keys are unchanged).
        _LABELS = {
            "categories": "Categories",
            "creation_UTC": "Creation (UTC)",
            "site_name": "Site name",
            "learning_rate": "Learning rate",
            "epochs": "Epochs",
            "num_classes": "Number of classes",
            "val_loss": "Validation loss",
            "val_accuracy": "Validation accuracy",
            "miou": "Mean IoU",
            "target_category_name": "Target category",
            "base_model": "Base model",
            "category_centroids": "Category centroids",
            "blob_filter_mahal_fraction": "Mahalanobis fallback fraction",
            "blob_centroids": "Centroids",
            "blob_knn_k": "kNN k",
            "blob_knn_threshold": "kNN threshold",
            "suggested_blob_filter_mode": "Suggested filter mode",
        }

        def _prettify(k):
            s = str(k).replace("_", " ").strip()
            return (s[:1].upper() + s[1:]) if s else s

        self.listWidget_modelMetadata.setRowCount(0)

        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            skip_keys = {'model_state_dict', 'state_dict', 'optimizer', 'optimizer_state_dict'}
            blob_keys = {'blob_centroid_cov', 'blob_centroid_mean', 'blob_filter_n_sigma',
                         'blob_filter_radius', 'blob_filter_mode'}

            # ── GroupBox title — just "Model Metadata" ────────────────────────
            self.groupBox_modelMetadata.setTitle("Model Metadata")

            # ── Blob filter analysis ──────────────────────────────────────────
            import numpy as np
            import math as _math
            raw_cov  = checkpoint.get("blob_centroid_cov")
            n_sigma  = checkpoint.get("blob_filter_n_sigma", 2.5)
            fallback = checkpoint.get("blob_filter_radius")
            diag_px  = _math.sqrt(1024**2 + 576**2)
            cov_ok   = False
            if raw_cov is not None:
                try:
                    cov = np.array(raw_cov, dtype=np.float64)
                    np.linalg.inv(cov)
                    cov_ok = True
                except Exception:
                    pass

            # ── Helper: add a row ─────────────────────────────────────────────
            def _add_row(label, value, label_align=Qt.AlignLeft, bold_label=False):
                row = self.listWidget_modelMetadata.rowCount()
                self.listWidget_modelMetadata.insertRow(row)

                key_item = QTableWidgetItem(str(label))
                key_item.setFlags(Qt.ItemIsEnabled)
                key_item.setTextAlignment(label_align | Qt.AlignVCenter)
                if bold_label:
                    f = QFont()
                    f.setBold(True)
                    key_item.setFont(f)

                val_item = QTableWidgetItem(str(value))
                val_item.setFlags(Qt.ItemIsEnabled)
                val_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

                self.listWidget_modelMetadata.setItem(row, 0, key_item)
                self.listWidget_modelMetadata.setItem(row, 1, val_item)

            def _add_sub_row(label, value):
                """Indented sub-property row, label right-aligned."""
                row = self.listWidget_modelMetadata.rowCount()
                self.listWidget_modelMetadata.insertRow(row)

                key_item = QTableWidgetItem(f"    {label}")
                key_item.setFlags(Qt.ItemIsEnabled)
                key_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                key_item.setForeground(QColor("#666666"))

                val_item = QTableWidgetItem(str(value))
                val_item.setFlags(Qt.ItemIsEnabled)
                val_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

                self.listWidget_modelMetadata.setItem(row, 0, key_item)
                self.listWidget_modelMetadata.setItem(row, 1, val_item)

            # ── Main metadata rows (skip blob keys — handled separately) ──────
            for key, value in checkpoint.items():
                if key in skip_keys or key in blob_keys:
                    continue
                if value is None:
                    display_value = "None"
                elif isinstance(value, (int, str)):
                    display_value = str(value)
                elif isinstance(value, float):
                    display_value = f"{value:.4f}"
                elif isinstance(value, (list, tuple)):
                    display_value = f"[{len(value)} items]"
                elif isinstance(value, dict):
                    display_value = f"{{dict with {len(value)} keys}}"
                else:
                    display_value = f"{type(value).__name__}"
                _add_row(_LABELS.get(key, _prettify(key)), display_value)

            # ── Mahalanobis blob filter section ───────────────────────────────
            if cov_ok:
                fallback_px = int(round(float(fallback) * diag_px)) if fallback else "?"
                _add_row("Mahalanobis spatial filter", "", bold_label=True)
                _add_sub_row("n-sigma",   f"{float(n_sigma):.2f}")
                _add_sub_row("Fallback fraction",  f"{float(fallback)*100:.1f}%  (~{fallback_px}px)")
                _add_sub_row("Covariance matrix", "[2×2 items]")
            elif fallback:
                fallback_px = int(round(float(fallback) * diag_px))
                _add_row("Circular spatial filter", "", bold_label=True)
                _add_sub_row("Radius", f"{float(fallback)*100:.1f}%  (~{fallback_px}px)")
            else:
                _add_row("Spatial filter", "No metadata in checkpoint")

            self.listWidget_modelMetadata.resizeColumnToContents(0)
            self.listWidget_modelMetadata.horizontalHeader().setStretchLastSection(True)

        except Exception as e:
            row = self.listWidget_modelMetadata.rowCount()
            self.listWidget_modelMetadata.insertRow(row)
            self.listWidget_modelMetadata.setItem(row, 0, QTableWidgetItem("Error"))
            self.listWidget_modelMetadata.setItem(row, 1, QTableWidgetItem(str(e)))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def select_label(self, target_label):
        target_text = ""

        # Case 1: dict
        if isinstance(target_label, dict):
            target_text = target_label.get("label_name", "")

        # Case 2: list of dicts
        elif isinstance(target_label, list) and target_label:
            if isinstance(target_label[0], dict):
                target_text = target_label[0].get("label_name", "")
            elif isinstance(target_label[0], str):
                target_text = target_label[0]

        # Case 3: plain string
        elif isinstance(target_label, str):
            target_text = target_label

        if target_text:
            items = self.listWidget_labels.findItems(target_text, QtCore.Qt.MatchExactly)
            if items:
                item = items[0]
                self.listWidget_labels.clearSelection()  # optional, ensures only one selected
                item.setSelected(True)  # mark as selected
                self.listWidget_labels.setCurrentItem(item)  # make it active
                self.listWidget_labels.scrollToItem(item)  # ensure visible

    # ── Season filter helpers (mirrors training_tab pattern) ──────────────────

    # ── Segment seasons dual listbox ──────────────────────────────────────────

    _SEASON_ORDER    = ["Winter", "Spring", "Summer", "Fall"]
    _SEASON_TYPE     = "Meteorological"
    _ALL_SEASONS_PLACEHOLDER = "All Seasons"
    _SEASON_DATES = {
        "Winter": "Dec 1 - Feb 28",
        "Spring": "Mar 1 - May 31",
        "Summer": "Jun 1 - Aug 31",
        "Fall":   "Sep 1 - Nov 30",
    }

    @classmethod
    def _season_label(cls, season: str) -> str:
        """Return display label with date range, e.g. 'Winter (Dec 1 - Feb 28)'."""
        dates = cls._SEASON_DATES.get(season, "")
        return f"{season} ({dates})" if dates else season

    @staticmethod
    def _season_from_label(label: str) -> str:
        """Strip the date suffix from a display label to recover the plain season name."""
        return label.split(" (")[0]

    def _init_segment_season_lists(self) -> None:
        """Ensure right listbox shows 'All Seasons' placeholder on startup."""
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _on_available_season_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """Move a season from Available → Selected. Remove 'All Seasons' from Selected."""
        label = item.text()
        # Remove from available
        row = self.listWidget_availableSegmentSeasons.row(item)
        self.listWidget_availableSegmentSeasons.takeItem(row)
        # Remove 'All Seasons' placeholder from Selected (do not return it to Available)
        lw_sel = self.listWidget_segmentSeasons
        for i in range(lw_sel.count() - 1, -1, -1):
            if lw_sel.item(i).text() == self._ALL_SEASONS_PLACEHOLDER:
                lw_sel.takeItem(i)
        # Add to Selected using display label
        lw_sel.addItem(label)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _on_selected_season_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """Move a season from Selected → Available. Restore 'All Seasons' if Selected becomes empty."""
        label = item.text()
        if label == self._ALL_SEASONS_PLACEHOLDER:
            return  # nothing to do
        season = self._season_from_label(label)
        # Remove from Selected
        row = self.listWidget_segmentSeasons.row(item)
        self.listWidget_segmentSeasons.takeItem(row)
        # Add back to Available in canonical order using display label
        self._insert_season_sorted(self.listWidget_availableSegmentSeasons, season)
        # Restore placeholder if Selected is now empty
        if self.listWidget_segmentSeasons.count() == 0:
            self.listWidget_segmentSeasons.addItem(self._ALL_SEASONS_PLACEHOLDER)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _insert_season_sorted(self, lw: QtWidgets.QListWidget, season: str) -> None:
        """Insert season display label into lw respecting _SEASON_ORDER."""
        label = self._season_label(season)
        existing_labels = [lw.item(i).text() for i in range(lw.count())]
        existing_seasons = [self._season_from_label(t) for t in existing_labels]
        order = self._SEASON_ORDER
        pos = len(existing_labels)
        if season in order:
            season_idx = order.index(season)
            for i, s in enumerate(existing_seasons):
                if s in order and order.index(s) > season_idx:
                    pos = i
                    break
        lw.insertItem(pos, label)

    def _get_segment_seasons(self) -> list:
        """Return list of plain season names to segment, or [] if all seasons.
        Also cleans up the placeholder if real seasons are present."""
        lw = self.listWidget_segmentSeasons
        items = [lw.item(i).text() for i in range(lw.count())]
        real_labels = [t for t in items if t != self._ALL_SEASONS_PLACEHOLDER]
        real = [self._season_from_label(t) for t in real_labels]

        # Lazy cleanup: remove placeholder if real seasons were dragged in
        if real and self._ALL_SEASONS_PLACEHOLDER in items:
            for i in range(lw.count() - 1, -1, -1):
                if lw.item(i).text() == self._ALL_SEASONS_PLACEHOLDER:
                    lw.takeItem(i)

        # Lazy restore: if empty, put placeholder back
        if not real and lw.count() == 0:
            lw.addItem(self._ALL_SEASONS_PLACEHOLDER)

        return real  # empty list means all seasons

    def _set_segment_seasons(self, segment_seasons: list) -> None:
        """Restore dual-listbox state from a list of plain season names to segment."""
        season_set = set(segment_seasons)
        self.listWidget_availableSegmentSeasons.clear()
        self.listWidget_segmentSeasons.clear()
        for season in self._SEASON_ORDER:
            label = self._season_label(season)
            if season in season_set:
                self.listWidget_segmentSeasons.addItem(label)
            else:
                self.listWidget_availableSegmentSeasons.addItem(label)
        if not season_set:
            self.listWidget_segmentSeasons.addItem(self._ALL_SEASONS_PLACEHOLDER)


# WILL NEED THIS IN ORDER TO UPDATE THE JSON SETTINGS FILE
'''
        # Normalize segmentation images folder
        seg_images_folder = self.lineEdit_segmentation_images_folder.text().strip()
        seg_images_folder = os.path.abspath(seg_images_folder).replace("\\", "/") if seg_images_folder else ""

        # Handle segmentation model path
        seg_model_path = self.lineEdit_segmentation_model_file.text().strip()
        if seg_model_path and seg_model_path.lower().endswith('.torch') and os.path.isfile(seg_model_path):
            abs_model_path = os.path.normpath(seg_model_path.strip())
            if self.selected_segment_model == "sam2":
                site_config["load_model"]["SAM2_MODEL"] = abs_model_path
            elif self.selected_segment_model == "segformer":
                site_config["load_model"]["SEGFORMER_MODEL"] = abs_model_path
            elif self.selected_segment_model == "maskrcnn":
                site_config["load_model"]["MASKRCNN_MODEL"] = abs_model_path
            print("Updated MODEL path to:", abs_model_path)
        else:
            print("No valid segmentation model file selected; using default MODEL path.")
'''