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
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox, QSizePolicy
from PyQt5.uic import loadUi

from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor
from GRIME_AI.dialogs.ML_image_processing.model_config_manager import ModelConfigManager
from GRIME_AI.utils.resource_utils import ui_path
from GRIME_AI.GRIME_AI_CSS_Styles import BUTTON_CSS_STEEL_BLUE

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
            self.populate_model_labels(model_file)
            # Populate metadata will be done on showEvent to avoid loading on init
        else:
            self.lineEdit_segmentation_model_file.clear()

        # Image folders list
        saved_folders = load_model_conf.get("segmentation_image_folders", [])
        self.image_folders = saved_folders if isinstance(saved_folders, list) else []
        self._refresh_folder_list_widget()

        # Output folder
        output_folder = load_model_conf.get("output_folder", "")
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
        """Called when tab becomes visible - populate metadata if model path exists."""
        super().showEvent(event)
        
        # Check if model path is populated
        model_path = self.lineEdit_segmentation_model_file.text().strip()
        if model_path and os.path.exists(model_path):
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
        self.pushButton_Segment.setStyleSheet(
            'QPushButton {background-color: steelblue; color: white; }'
            'QPushButton:disabled {background-color: gray; color: black; }'
        )
        self.pushButton_Segment.setMinimumSize(150, 40)
        self.pushButton_Segment.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Keep local widget policies similar to original dialog

        try:
            self.listWidget_labels.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass

        try:
            self.splitter_right_panel.setSizes([1000, 1000])  # equal initial split
        except Exception:
            pass

        try:
            self.pushButton_Segment.setMinimumSize(150, 40)
            self.pushButton_Segment.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
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
        self.pushButton_Remove_Folder.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_Clear_Folders.clicked.connect(self.clear_all_folders)
        self.pushButton_Clear_Folders.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_Segment.clicked.connect(self.segment_images)
        self.pushButton_Segment.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

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

        # Segment seasons dual listbox — no signal connections needed.
        # Placeholder cleanup is handled lazily in _get_segment_seasons().
        self._init_segment_season_lists()

        # Labels list affects segment button state
        try:
            self.listWidget_labels.itemSelectionChanged.connect(self.updateSegmentButtonState)
            self.listWidget_labels.itemSelectionChanged.connect(self.update_model_config)
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
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
        """Load and display ALL metadata from torch checkpoint."""
        self.listWidget_modelMetadata.clear()
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Skip model_state_dict (too large to display)
            skip_keys = {'model_state_dict', 'state_dict', 'optimizer', 'optimizer_state_dict'}
            
            # Iterate through all keys in checkpoint
            for key, value in checkpoint.items():
                if key in skip_keys:
                    continue
                
                # Format the value for display
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
                
                # Add to listbox
                self.listWidget_modelMetadata.addItem(f"{key}: {display_value}")
                
        except Exception as e:
            self.listWidget_modelMetadata.addItem(f"Error: {e}")

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

    def _init_segment_season_lists(self) -> None:
        """Ensure right listbox shows 'All Seasons' placeholder on startup."""
        pass

    def _get_segment_seasons(self) -> list:
        """Return list of seasons to segment, or [] if all seasons.
        Also cleans up the placeholder if real seasons are present."""
        lw = self.listWidget_segmentSeasons
        items = [lw.item(i).text() for i in range(lw.count())]
        real = [t for t in items if t != self._ALL_SEASONS_PLACEHOLDER]

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
        """Restore dual-listbox state from a list of seasons to segment."""
        season_set = set(segment_seasons)
        self.listWidget_availableSegmentSeasons.clear()
        self.listWidget_segmentSeasons.clear()
        for season in self._SEASON_ORDER:
            if season in season_set:
                self.listWidget_segmentSeasons.addItem(season)
            else:
                self.listWidget_availableSegmentSeasons.addItem(season)
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