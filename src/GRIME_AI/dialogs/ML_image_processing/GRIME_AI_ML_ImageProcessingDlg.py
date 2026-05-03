#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os
import getpass
import cv2
import json
from json import JSONDecodeError
from promptlib import Files
from typing import List, Tuple, Dict
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")      # <<< FORCE Qt5Agg backend for PyQt5

from GRIME_AI.utils.resource_utils import ui_path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QSizePolicy, QListWidget
from PyQt5.uic import loadUi

from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI.dialogs.ML_image_processing.training_tab import TrainingTab
from GRIME_AI.dialogs.ML_image_processing.segment_images_tab import SegmentImagesTab
from GRIME_AI.dialogs.ML_image_processing.roi_analyzer_tab import ROIAnalyzerTab
from GRIME_AI.dialogs.ML_image_processing.correlation_analyzer_tab import CorrelationAnalyzerTab
from GRIME_AI.dialogs.ML_image_processing.coco_generation import COCOGeneration
from GRIME_AI.dialogs.ML_image_processing.model_config_manager import ModelConfigManager
from GRIME_AI.dialogs.ML_image_processing.annotation_analyzer_tab import AnnotationAnalyzerTab
from GRIME_AI.dialogs.ML_image_processing.mask_viewer import CocoViewerTab
from GRIME_AI.dialogs.ML_image_processing.surface_state_classifier_tab import SurfaceStateClassifierTab

# ======================================================================================================================
# ======================================================================================================================
#  =====     =====     =====     =====     class GRIME_AI_ML_ImageProcessingDlg      =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class GRIME_AI_ML_ImageProcessingDlg(QDialog):

    ml_train_signal = pyqtSignal()
    ml_segment_signal = pyqtSignal()

    # ******************************************************************************************************************
    # * INITIALIZE DIALOGBOX AND  TABS     *     INITIALIZE DIALOGBOX AND TABS     *     INITIALIZE DIALOGBOX AND TABS *
    # ******************************************************************************************************************
    def __init__(self, parent=None):
        super().__init__(parent)

        loadUi(ui_path("ML_image_processing/QDialog_ML_ImageProcessing.ui"), self)

        # --------------------------------------------------------------------------------------------------------------
        # RESIZE DIALOG: 50% wider, 20% shorter
        # --------------------------------------------------------------------------------------------------------------
        default_size = self.size()
        default_width = default_size.width()
        default_height = default_size.height()

        new_width = int(default_width * 1.5)  # 50% wider
        new_height = int(default_height * 0.8)  # 20% shorter

        self.resize(new_width, new_height)
        self.setMinimumSize(int(new_width * 0.7), int(new_height * 0.7))

        print(f"Dialog resized: {default_width}x{default_height} → {new_width}x{new_height}")

        # --------------------------------------------------------------------------------------------------------------
        # TRAINING TAB    ---    TRAINING TAB    ---   TRAINING TAB    ---    TRAINING TAB    ---    TRAINING TAB
        # --------------------------------------------------------------------------------------------------------------
        # Instantiate the TrainingTab and add it to the tabWidget
        self.training_tab = TrainingTab(self)
        self.tabWidget.addTab(self.training_tab, "Train Model")

        self.training_tab.ml_train_signal.connect(self.ml_train_signal)

        # --------------------------------------------------------------------------------------------------------------
        # SEGMENT IMAGES TAB    ---    SEGMENT IMAGES TAB    ---    SEGMENT IMAGES TAB    ---   SEGMENT IMAGES TAB
        # --------------------------------------------------------------------------------------------------------------
        # Instantiate the SegmentTab and add it to the tabWidget
        self.segment_tab = SegmentImagesTab(self)
        self.tabWidget.addTab(self.segment_tab, "Segment Images")

        self.segment_tab.ml_segment_signal.connect(self.ml_segment_signal)

        # --------------------------------------------------------------------------------------------------------------
        # ROI ANALYZER TAB     ---     ROI ANALYZER TAB     ---     ROI ANALYZER TAB     ---     ROI ANALYZER TAB
        # --------------------------------------------------------------------------------------------------------------
        # Create the ROI Analyzer tab widget
        self.roi_tab = ROIAnalyzerTab(self)

        # Load the ROI Analyzer tab UI into the ROIAnalyzerTab instance
        loadUi(ui_path("ML_image_processing/roi_analyzer_tab.ui"), self.roi_tab)

        self.roi_tab.configure_filmstrip()
        self.roi_tab.wire_connections()

        # Add ROI Analyzer tab to the tabWidget
        self.tabWidget.addTab(self.roi_tab, "ROI Analyzer")

        # --------------------------------------------------------------------------------------------------------------
        # COCO GENERATION TAB    ---    COCO GENERATION TAB    ---    COCO GENERATION TAB    ---    COCO GENERATION TAB
        # --------------------------------------------------------------------------------------------------------------
        self.coco_generation_tab = COCOGeneration(self)

        self.tabWidget.addTab(self.coco_generation_tab, "COCO 1.0 Generator")

        # --------------------------------------------------------------------------------------------------------------
        #  ANNOTATION ANALYZER TAB        ---         ANNOTATION ANALYZER TAB         ---       ANNOTATION ANALYZER TAB
        # --------------------------------------------------------------------------------------------------------------
        # Create the Image Annotation Analyzer tab widget
        self.annotation_analyzer_tab = AnnotationAnalyzerTab(self)

        # Add Image Annotation tab to the tabWidget
        self.tabWidget.addTab(self.annotation_analyzer_tab, "Annotation Analyzer")

        # --------------------------------------------------------------------------------------------------------------
        #  COCO VIEWER TAB   ---   COCO VIEWER TAB   ---   COCO VIEWER TAB   ---   COCO VIEWER TAB   ---   COCO VIEWER TAB
        # --------------------------------------------------------------------------------------------------------------
        self.coco_viewer_tab = CocoViewerTab(self)

        self.tabWidget.addTab(self.coco_viewer_tab, "COCO Viewer")

        # --------------------------------------------------------------------------------------------------------------
        #  CORRELATION ANALYZER TAB   ---   CORRELATION ANALYZER TAB   ---   CORRELATION ANALYZER TAB
        # --------------------------------------------------------------------------------------------------------------
        self.correlation_tab = CorrelationAnalyzerTab(self)

        loadUi(ui_path("ML_image_processing/correlation_analyzer_tab.ui"), self.correlation_tab)

        self.correlation_tab.wire_connections()

        self.tabWidget.addTab(self.correlation_tab, "Correlation Analyzer (EXPERIMENTAL)")

        # --------------------------------------------------------------------------------------------------------------
        #  SURFACE STATE CLASSIFIER TAB   ---   SURFACE STATE CLASSIFIER TAB   ---   SURFACE STATE CLASSIFIER TAB
        # --------------------------------------------------------------------------------------------------------------
        self.surface_state_tab = SurfaceStateClassifierTab(self)

        loadUi(ui_path("ML_image_processing/surface_state_classifier_tab.ui"), self.surface_state_tab)

        self.surface_state_tab.configure_filmstrip()
        self.surface_state_tab.wire_connections()

        self.tabWidget.addTab(self.surface_state_tab, "Surface State Classifier")

        # --------------------------------------------------------------------------------------------------------------
        # LOAD CONFIGURATION SETTINGS THAT MAY BE REQUIRED FOR THE  TABS
        # --------------------------------------------------------------------------------------------------------------
        settings_folder = Path(GRIME_AI_Save_Utils().get_settings_folder()).resolve()
        config_file = (settings_folder / "site_config.json").resolve()

        mgr = ModelConfigManager(str(config_file))

        if not config_file.exists():
            # File doesn’t exist → start fresh from template
            self.site_config = mgr.create_template()
            self.update_model_config()
        else:
            try:
                loaded = mgr.load_config(return_type="dict")
                # If load_config returns None or empty dict, fall back
                if not loaded or not isinstance(loaded, dict):
                    self.site_config = mgr.create_template()
                    self.update_model_config()
                else:
                    self.site_config = loaded
            except (JSONDecodeError, ValueError):
                # File exists but is empty or invalid JSON → reset
                self.site_config = mgr.create_template()
                self.update_model_config()

        # Initialize tracking variables.
        self.selected_label_categories = []
        self.categories_available = False

        # Call helper methods.
        #self.setup_from_config_file()

        self.setup_ui_properties()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setup_ui_properties(self):
        """Set size policies and layout stretch factors."""
        self.tabWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        '''
        self.listWidget_availableFolders.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.listWidget_selectedFolders.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.listWidget_availableFolders.setMinimumHeight(200)
        self.listWidget_selectedFolders.setMinimumHeight(200)
        self.adjustSize()
        self.setMinimumSize(self.size())
        self.verticalTabParametersLayout.setStretch(0, 1)
        self.verticalTabParametersLayout.setStretch(1, 0)
        self.horizontalMainLayout.setStretch(0, 1)
        self.horizontalMainLayout.setStretch(1, 3)
        self.horizontalListLayout.setStretch(0, 1)
        self.horizontalListLayout.setStretch(1, 0)
        self.horizontalListLayout.setStretch(2, 1)
        '''

    ###JES - ARE THESE OBSOLETE FUNCTIONS?
    '''
    def load_labels_from_annotation(self, folder_path):
        annotation_file = os.path.join(folder_path, "instances_default.json")
        if not os.path.exists(annotation_file):
            return []

        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        labels = set()
        if "annotations" in data and "categories" in data:
            for cat in data["categories"]:
                labels.add(f"{cat['id']} - {cat['name']}")
        return sorted(labels)

    def update_annotation_listbox(self, listbox_widget, folder_path):
        labels = self.load_labels_from_annotation(folder_path)
        listbox_widget.clear()
        listbox_widget.addItems(labels)
    '''

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_coco_for_all_images(self, base_image_folder: str):
        """
        Walks through the annotation_store for every image,
        builds a single COCO 1.0 JSON, and writes it to
        'annotations/coco_all.json' under base_image_folder.
        """
        # Prepare output folder
        base_p = Path(base_image_folder)
        ann_folder = base_p / "annotations"
        ann_folder.mkdir(parents=True, exist_ok=True)

        coco = {
            "info": {
                "description": "Combined dataset",
                "version": "1.0",
                "year": 2025
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        image_id_map = {}  # map image path -> image_id
        category_map = {}  # map cat_id -> cat_name
        ann_id = 1

        # Helper to compute polygon area via shoelace
        def polygon_area(coords):
            area = 0.0
            n = len(coords)
            for i in range(n):
                x1, y1 = coords[i]
                x2, y2 = coords[(i + 1) % n]
                area += x1 * y2 - x2 * y1
            return abs(area) * 0.5

        # 1) Iterate through every image in your store
        for idx, (img_path, shapes) in enumerate(self.annotation_store.items(), start=1):
            if not shapes:
                continue

            img_p = Path(img_path)
            # assign a unique image_id
            image_id = idx
            image_id_map[img_path] = image_id

            # read size
            img = cv2.imread(str(img_p))
            if img is None:
                print(f"Warning: cannot load {img_p}, skipping.")
                continue
            h, w = img.shape[:2]

            # append image entry
            coco["images"].append({
                "id": image_id,
                "file_name": img_p.name,
                "width": w,
                "height": h
            })

            # collect categories seen in this image
            for shape in shapes:
                cid = shape["label"]["id"]
                cname = shape["label"]["name"]
                category_map[cid] = cname

            # build annotation entries
            for shape in shapes:
                pts = shape["points"]
                # segmentation: flatten [x1,y1,x2,y2,...]
                seg = []
                for p in pts:
                    seg.extend([p.x(), p.y()])

                xs = [p.x() for p in pts]
                ys = [p.y() for p in pts]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = polygon_area([(p.x(), p.y()) for p in pts])

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": shape["label"]["id"],
                    "segmentation": [seg],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0
                })
                ann_id += 1

        # 2) Finalize category list
        coco["categories"] = [
            {"id": cid, "name": cname, "supercategory": ""}
            for cid, cname in category_map.items()
        ]

        # 3) Write out the combined COCO JSON
        out_file = ann_folder / "coco_all.json"
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(coco, f, indent=2)
            print(f"Saved global COCO file to {out_file}")
        except Exception as e:
            print(f"Failed to write global COCO JSON: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def reject(self):
        # DO NOTHING. LET IT CLOSE. IF THE CALLING PROGRAM CREATED THE DIALOG USING EXEC, THE CALLING INSTANTIATING
        # PROGRAM CAN INSPECT THE RETURN RESULT
        super().reject()

    # ******************************************************************************************************************
    # *   CREATE COCO FILE      *      CREATE COCO FILE      *      CREATE COCO FILE      *      CREATE COCO FILE      *
    # ******************************************************************************************************************
    def updateCOCOButtonState(self):
        """Enable the Generate COCO button only if a folder is provided in the COCO tab."""
        folder_entered = bool(self.lineEdit_cocoFolder.text().strip())
        self.pushButton_generateCOCO.setEnabled(folder_entered)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def updateMaskFieldState(self, checked):
        """Enable/disable the mask file field and its Browse button based on the Single Mask checkbox state."""
        self.lineEdit_maskFile.setEnabled(checked)
        self.pushButton_maskBrowse.setEnabled(checked)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def selectCocoFolder(self):
        """Open a folder chooser for the COCO generation folder."""
        folder = Files().dir()  # You can replace this with QFileDialog.getExistingDirectory if needed.
        if folder:
            self.lineEdit_cocoFolder.setText(folder)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def selectMaskFile(self):
        """Open a file chooser to select a mask file."""
        mask_file = Files().file()  # Replace with QFileDialog.getOpenFileName if preferred.
        if mask_file:
            self.lineEdit_maskFile.setText(mask_file)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def find_coco_products(self, root_folder):
        """
        Recursively search for all occurrences of 'COCO Products' folders under the given root.
        """
        found = []
        for dirpath, dirnames, _ in os.walk(root_folder):
            print(f"Scanning directory: {dirpath}")
            if "COCO Products" in dirnames:
                coco_path = os.path.join(dirpath, "COCO Products")
                print(f"  Found 'COCO Products' folder: {coco_path}")
                found.append(coco_path)
        if not found:
            print("No 'COCO Products' folder found under", root_folder)
        return found

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_values(self, site_config: dict) -> dict:
        """
        Collect values from dialog controls and return them as a dictionary.
        """
        site_config["train_model"]["TRAINING_CATEGORIES"] = self.get_selected_training_labels()

        # Build Path section
        root_folder = os.path.normpath(os.path.abspath(self.training_tab.get_training_images_root_folder))
        selected_folders = site_config.get("selected_folders", [])
        if selected_folders:
            new_folders = []
            new_annotations = []
            for folder in selected_folders:
                folder_fwd = os.path.normpath(folder)
                folder_path = os.path.normpath(os.path.join(root_folder, folder_fwd))
                annotation_path = os.path.normpath(os.path.join(root_folder, folder_fwd, "instances_default.json"))
                new_folders.append(folder_path)
                new_annotations.append(annotation_path)
            site_config["Path"] = [{
                "siteName": "custom",
                "directoryPaths": {
                    "folders": new_folders,
                    "annotations": new_annotations
                }
            }]
            print("Updated Path section from selected folders.")
        else:
            if hasattr(self, "current_path") and self.current_path:
                site_config["Path"] = self.current_path
                print("Right listbox is empty; retaining existing Path section from config.")
            else:
                site_config["Path"] = []
                print("Right listbox is empty and no existing Path data available; setting Path to empty.")

        # _____ site name _____________________________________________________
        site_config["siteName"] = self.lineEdit_siteName.text()

        # _____ hyperparameters _______________________________________________
        lr_text = self.lineEdit_learningRates.text()
        try:
            site_config["learningRates"] = [float(x.strip()) for x in lr_text.split(",") if x.strip()]
        except Exception as e:
            print("Error parsing learning rates:", e)
            site_config["learningRates"] = lr_text
        site_config["optimizer"] = self.comboBox_optimizer.currentText()
        site_config["loss_function"] = self.comboBox_lossFunction.currentText()
        site_config["weight_decay"] = self.doubleSpinBox_weightDecay.value()
        site_config["number_of_epochs"] = self.spinBox_epochs.value()
        site_config["batch_size"] = self.spinBox_batchSize.value()

        # _____ LoRA hyperparameters __________________________________________
        site_config["lora_rank"] = self.spinBox_loraRank.value()
        site_config["lora_alpha"] = self.spinBox_loraAlpha.value()
        site_config["lora_dropout"] = self.doubleSpinBox_loraDropout.value()
        ui_text = self.comboBox_loraBias.currentText().strip().lower()

        if ui_text.startswith("none"):
            site_config["lora_bias"] = "none"
        elif ui_text.startswith("lora"):
            site_config["lora_bias"] = "lora_only"
        elif ui_text.startswith("all"):
            site_config["lora_bias"] = "all"
        else:
            site_config["lora_bias"] = "none"

        # Target modules checkboxes
        lora_target_modules = []
        if self.checkBox_loraQuery.isChecked():
            lora_target_modules.append("query")
        if self.checkBox_loraKey.isChecked():
            lora_target_modules.append("key")
        if self.checkBox_loraValue.isChecked():
            lora_target_modules.append("value")

        site_config["lora_target_modules"] = lora_target_modules

        # _____ GRIME AI ML parameters  _______________________________________
        site_config["save_model_frequency"] = self.spinBox_saveFrequency.value()
        site_config["validation_frequency"] = self.spinBox_validationFrequency.value()
        site_config["early_stopping"] = self.checkBox_earlyStopping.isChecked()
        site_config["patience"] = self.spinBox_patience.value()
        site_config["device"] = self.comboBox_device.currentText()
        site_config["folder_path"] = self.lineEdit_model_training_images_path.text()
        avail_root = self.training_tab.listWidget_availableFolders.invisibleRootItem()
        site_config["available_folders"] = [
            avail_root.child(i).text(0).lstrip('\u2605 ')
            for i in range(avail_root.childCount())
        ]
        sel_root = self.training_tab.listWidget_selectedFolders.invisibleRootItem()
        site_config["selected_folders"] = [
            sel_root.child(i).text(0).lstrip('\u2605 ')
            for i in range(sel_root.childCount())
        ]

        return site_config

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def load_config_from_json(self, filepath):
        """
        Load configuration values from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the JSON configuration file.

        Returns
        -------
        dict
            Configuration dictionary if file exists and is valid.
            Empty dict if file does not exist or cannot be loaded.
        """
        if not os.path.isfile(filepath):
            print(f"Config file not found: {filepath}")
            return {}

        try:
            with open(filepath, "r") as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {filepath}: {e}")
            return {}
        except Exception as e:
            print(f"Unexpected error loading config from {filepath}: {e}")
            return {}

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def update_model_config(self):

        """
        Gather all dialog values and create a JSON configuration file.
        The main structure is site_config, not values.
        """
        settings_folder = Path(GRIME_AI_Save_Utils().get_settings_folder()).resolve()
        config_file = (settings_folder / "site_config.json").resolve()

        # Use ModelConfigManager to handle backup + load
        manager = ModelConfigManager(filepath=config_file)
        site_config = manager.backup_config()  # backs up existing file and loads config (or {} if none)

        # Merge values from controls into site_config
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
    def update_training_categories(self, site_config: dict) -> dict:
        """
        Populate load_model.TRAINING_CATEGORIES with selected training labels.
        Assumes site_config["load_model"] already exists.
        """
        site_config["load_model"]["TRAINING_CATEGORIES"] = self.get_selected_training_labels()
        return site_config

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def update_segmentation_categories(self, site_config: dict) -> dict:
        """
        Populate load_model.SEGMENTATION_CATEGORIES with selected segmentation labels.
        Assumes site_config["load_model"] already exists.
        """
        site_config["load_model"]["SEGMENTATION_CATEGORIES"] = self.get_selected_segmentation_labels()
        return site_config

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_selected_training_labels(self):
        """
        Fetch the selected training label from comboBox_train_label_selection
        and return it as a list of dictionaries.

        Each dictionary contains:
        - "label_id": the parsed ID from the comboBox text
        - "label_name": the parsed label name

        Returns
        -------
        list[dict]
            A list with one dictionary if selection is valid, else an empty list.
        """
        selected_text = self.comboBox_train_label_selection.currentText().strip()

        if "-" not in selected_text:
            return []  # malformed or empty selection

        label_id, label_name = map(str.strip, selected_text.split("-", 1))

        return [{
            "label_id": label_id,
            "label_name": label_name
        }]

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_selected_segmentation_labels(self):
        """
        Return all selected items from listWidget_labels as a list of dictionaries.

        Each dictionary contains:
        - "label_id": 1-based ordinal position of the item in the listbox
        - "label_name": text of the item

        Returns
        -------
        list[dict]
            List of dictionaries with selected segmentation labels and their IDs.
        """
        selected_items = []
        for index in range(self.listWidget_labels.count()):
            item = self.listWidget_labels.item(index)
            if item.isSelected():
                selected_items.append({
                    "label_id": index + 1,
                    "label_name": item.text()
                })
        return selected_items

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    '''
    def initialize_dialog_from_config(self, config):
        self.site_config = config
        self.setup_from_config_file()
    '''

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def accept(self):
        self.update_model_config()
        super().accept()

     # ------------------------------------------------------------------------------------------------------------------
    # GET/SET FUNCTION PASS-THRUS FROM TABS
    # ------------------------------------------------------------------------------------------------------------------
    def get_selected_training_model(self):
        return self.training_tab.get_selected_training_model()

    def get_copy_original_image(self):
        return self.segment_tab.checkBox_copyOriginalModelImage.isChecked()

    def get_saved_masks(self):
        return self.segment_tab.checkBox_save_predicted_masks.isChecked()

    def get_save_probability_maps(self):
        return self.segment_tab.checkBox_save_probability_maps.isChecked()

    def get_save_diagnostic_panels(self):
        return self.segment_tab.checkBox_save_diagnostic_panels.isChecked()

    def get_selected_training_labels(self):
        return self.training_tab.get_selected_training_labels()

    def get_selected_label_categories(self):
        return self.segment_tab.selected_label_categories

    def get_selected_segment_model(self):
        return self.segment_tab.selected_segment_model

