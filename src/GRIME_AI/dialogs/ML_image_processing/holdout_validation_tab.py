#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation: University of Nebraska-Lincoln / Blade Vision Systems, LLC
# License: Apache License, Version 2.0

"""
Holdout Validation Tab
=======================
UI tab for running post-training holdout validation against annotated
images from held-out seasons. Mirrors the training tab folder discovery
pattern (recursive instances_default.json scan) and the dual-listbox
season selector pattern.

Settings (model path, image root, seasons, category, overlays toggle)
are persisted to the GRIME-AI.json settings file and restored on init.
"""

import os
import json
from pathlib import Path
from typing import List

import torch
from PyQt5.QtWidgets import (
    QWidget, QFileDialog, QMessageBox, QTreeWidgetItem
)
from PyQt5.QtCore import Qt
from PyQt5 import uic

from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor
from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils

# Settings key namespace
_SETTINGS_KEY = "HoldoutValidationTab"


# ── Module-level helpers ───────────────────────────────────────────────────────

def _iter_dirs(root: Path):
    """Recursively yield every subdirectory under root."""
    bad = ["anaconda3", "miniconda3", "ProgramData", "Windows"]
    if any(b in str(root).lower() for b in bad):
        return
    if not root.exists():
        return
    for entry in os.scandir(root):
        if entry.is_dir():
            sub = Path(entry.path)
            yield sub
            yield from _iter_dirs(sub)


def _has_annotation(folder: Path) -> bool:
    """Return True if folder contains instances_default.json and at least one image."""
    ann = folder / "instances_default.json"
    if not ann.exists():
        return False
    imgs = [e for e in os.scandir(folder)
            if e.is_file() and e.name.lower().endswith((".jpg", ".jpeg", ".png"))]
    return bool(imgs)


# ── Tab class ─────────────────────────────────────────────────────────────────

class HoldoutValidationTab(QWidget):

    _SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
    _SEASON_TYPE  = "Meteorological"

    def __init__(self, parent=None):
        super().__init__(parent)

        ui_path = os.path.join(os.path.dirname(__file__), "holdout_validation_tab.ui")
        uic.loadUi(ui_path, self)

        self._torch_path = ""
        self._image_dirs = []
        self._ann_paths  = []

        self._setup_connections()
        self._load_settings()
        self._update_run_button_state()

    # ── Connections ───────────────────────────────────────────────────────────

    def _setup_connections(self):
        self.pushButton_browseModel.clicked.connect(self._browse_model)
        self.pushButton_browseImageRoot.clicked.connect(self._browse_image_root)
        self.pushButton_runEvaluation.clicked.connect(self._run_evaluation)
        self.treeWidget_folders.itemSelectionChanged.connect(self._update_run_button_state)
        self.listWidget_evalSeasons.model().rowsInserted.connect(self._update_run_button_state)
        self.listWidget_evalSeasons.model().rowsRemoved.connect(self._update_run_button_state)

        # Enter in image root line edit triggers scan
        self.lineEdit_imageRootPath.returnPressed.connect(self._scan_folders)

        # Double-click to move seasons between listboxes
        self.listWidget_availableSeasons.itemDoubleClicked.connect(self._move_season_to_eval)
        self.listWidget_evalSeasons.itemDoubleClicked.connect(self._move_season_to_available)

    # ── Settings ──────────────────────────────────────────────────────────────

    def _settings_path(self) -> str:
        folder = GRIME_AI_Save_Utils().get_settings_folder()
        return os.path.join(folder, "GRIME-AI.json")

    def _load_settings(self):
        path = self._settings_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                all_settings = json.load(f)
        except Exception:
            return

        s = all_settings.get(_SETTINGS_KEY, {})
        if not s:
            return

        # Model path
        model_path = s.get("model_path", "")
        if model_path and os.path.exists(model_path):
            self.lineEdit_modelPath.setText(model_path)
            self._torch_path = model_path
            self._load_model_metadata(model_path)

        # Image root — scan immediately so tree is populated
        image_root = s.get("image_root", "")
        if image_root and os.path.isdir(image_root):
            self.lineEdit_imageRootPath.setText(image_root)
            self._scan_folders(select_all=False)

            # Restore previously selected folders by stored paths
            selected_paths = set(s.get("selected_folders", []))
            if selected_paths:
                root = self.treeWidget_folders
                for i in range(root.topLevelItemCount()):
                    item = root.topLevelItem(i)
                    item_path = item.data(0, Qt.UserRole)
                    if item_path in selected_paths:
                        item.setSelected(True)
                    else:
                        item.setSelected(False)

        # Holdout seasons — remove from available, add to eval
        saved_seasons = s.get("eval_seasons", [])
        if saved_seasons:
            self.listWidget_evalSeasons.clear()
            for season in saved_seasons:
                self.listWidget_evalSeasons.addItem(season)
                # Remove from available
                for i in range(self.listWidget_availableSeasons.count()):
                    if self.listWidget_availableSeasons.item(i).text() == season:
                        self.listWidget_availableSeasons.takeItem(i)
                        break

        # Category
        saved_category = s.get("category", "")
        if saved_category:
            for i in range(self.listWidget_categories.count()):
                if self.listWidget_categories.item(i).text().endswith(saved_category):
                    self.listWidget_categories.setCurrentRow(i)
                    break

        # Save overlays toggle
        save_overlays = s.get("save_overlays", True)
        self.checkBox_saveOverlays.setChecked(save_overlays)

    def _save_settings(self):
        path = self._settings_path()
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    all_settings = json.load(f)
            else:
                all_settings = {}
        except Exception:
            all_settings = {}

        # Collect selected folder paths
        selected_paths = []
        root = self.treeWidget_folders
        for i in range(root.topLevelItemCount()):
            item = root.topLevelItem(i)
            if item.isSelected():
                p = item.data(0, Qt.UserRole)
                if p:
                    selected_paths.append(p)

        # Collect current category name
        cat_item = self.listWidget_categories.currentItem()
        cat_name = cat_item.text().split(" - ", 1)[-1].strip() if cat_item else ""

        all_settings[_SETTINGS_KEY] = {
            "model_path":       self._torch_path,
            "image_root":       self.lineEdit_imageRootPath.text().strip(),
            "selected_folders": selected_paths,
            "eval_seasons":     self._get_eval_seasons(),
            "category":         cat_name,
            "save_overlays":    self.checkBox_saveOverlays.isChecked(),
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_settings, f, indent=2)

    # ── Model browsing ────────────────────────────────────────────────────────

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Trained Model", "", "GRIME AI Checkpoint (*.torch)"
        )
        if not path:
            return
        self.lineEdit_modelPath.setText(path)
        self._torch_path = path
        self._load_model_metadata(path)
        self._update_run_button_state()
        self._save_settings()

    def _load_model_metadata(self, path: str):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            ckpt.pop("model_state_dict", None)
        except Exception as e:
            self.label_modelInfo.setText(f"Could not load checkpoint: {e}")
            return

        site   = ckpt.get("site_name", "N/A")
        epochs = ckpt.get("epochs", "N/A")
        lr     = ckpt.get("learning_rate", "N/A")
        miou   = ckpt.get("miou", "N/A")
        cats   = ckpt.get("categories", [])

        info = (f"Site: {site}  |  Epoch: {epochs}  |  LR: {lr}  |  "
                f"mIoU: {miou}  |  Categories: {[c['name'] for c in cats]}")
        self.label_modelInfo.setText(info)

        self.listWidget_categories.clear()
        for c in cats:
            self.listWidget_categories.addItem(f"{c['id']} - {c['name']}")
        if self.listWidget_categories.count() > 0:
            self.listWidget_categories.setCurrentRow(0)

    # ── Folder discovery ──────────────────────────────────────────────────────

    def _browse_image_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select Image Root Folder")
        if not path:
            return
        self.lineEdit_imageRootPath.setText(path)
        self._scan_folders()
        self._save_settings()

    def _scan_folders(self, select_all: bool = True):
        raw = self.lineEdit_imageRootPath.text().strip()
        if not raw:
            return
        root = Path(raw).resolve()
        if not root.is_dir():
            QMessageBox.warning(self, "Invalid Folder", f"Not a directory:\n{root}")
            return

        self.treeWidget_folders.clear()
        valid = []

        if _has_annotation(root):
            valid.append(root)
        for folder in _iter_dirs(root):
            if _has_annotation(folder):
                valid.append(folder)

        if not valid:
            QMessageBox.information(self, "No Annotation Folders",
                                    "No folders with instances_default.json were found.")
            return

        for vf in sorted(set(valid)):
            try:
                rel = vf.relative_to(root)
                display = str(rel)
            except ValueError:
                display = str(vf)
            item = QTreeWidgetItem(self.treeWidget_folders, [display])
            item.setData(0, Qt.UserRole, str(vf))
            self.treeWidget_folders.addTopLevelItem(item)

        if select_all:
            self.treeWidget_folders.selectAll()

        self._update_run_button_state()

    # ── Season helpers ────────────────────────────────────────────────────────

    def _move_season_to_eval(self, item):
        """Double-click in Available moves season to Evaluate."""
        self.listWidget_availableSeasons.takeItem(
            self.listWidget_availableSeasons.row(item)
        )
        self.listWidget_evalSeasons.addItem(item.text())

    def _move_season_to_available(self, item):
        """Double-click in Evaluate moves season back to Available."""
        self.listWidget_evalSeasons.takeItem(
            self.listWidget_evalSeasons.row(item)
        )
        self.listWidget_availableSeasons.addItem(item.text())

    def _get_eval_seasons(self) -> List[str]:
        lw = self.listWidget_evalSeasons
        return [lw.item(i).text() for i in range(lw.count())]

    # ── Run button state ──────────────────────────────────────────────────────

    def _update_run_button_state(self):
        model_ok   = bool(self._torch_path and os.path.exists(self._torch_path))
        seasons_ok = bool(self._get_eval_seasons())
        folders_ok = bool(self.treeWidget_folders.selectedItems())
        cat_ok     = bool(self.listWidget_categories.currentItem())

        self.pushButton_runEvaluation.setEnabled(
            model_ok and seasons_ok and folders_ok and cat_ok
        )

    # ── Run evaluation ────────────────────────────────────────────────────────

    def _run_evaluation(self):
        eval_seasons = self._get_eval_seasons()
        if not eval_seasons:
            QMessageBox.warning(self, "No Seasons Selected",
                                "Drag at least one season into the Evaluate list.")
            return

        selected_items = self.treeWidget_folders.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Folders Selected",
                                "Select at least one annotation folder.")
            return

        root = self.lineEdit_imageRootPath.text().strip()
        image_dirs, ann_paths = [], []
        for item in selected_items:
            folder_path = item.data(0, Qt.UserRole) or os.path.join(root, item.text(0))
            ann_file    = os.path.join(folder_path, "instances_default.json")
            if os.path.exists(ann_file):
                image_dirs.append(folder_path)
                ann_paths.append(ann_file)

        if not image_dirs:
            QMessageBox.warning(self, "No Valid Folders",
                                "None of the selected folders contain instances_default.json.")
            return

        cat_item = self.listWidget_categories.currentItem()
        if not cat_item:
            QMessageBox.warning(self, "No Category", "Select a category to evaluate.")
            return
        cat_name = cat_item.text().split(" - ", 1)[-1].strip()

        model_dir = str(Path(self._torch_path).parent)

        total_estimate = sum(
            len([f for f in os.listdir(d)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            for d in image_dirs
        )

        progress_closed = [False]
        progressBar = QProgressWheel(
            title=f"Holdout Validation — {', '.join(eval_seasons)}",
            total=max(total_estimate, 1),
            on_close=lambda: progress_closed.__setitem__(0, True)
        )
        progressBar.show()

        # Save settings before running
        self._save_settings()

        from GRIME_AI.ml_core.holdout_evaluator import HoldoutEvaluator

        evaluator = HoldoutEvaluator(
            torch_path       = self._torch_path,
            image_dirs       = image_dirs,
            annotation_paths = ann_paths,
            eval_seasons     = eval_seasons,
            category_name    = cat_name,
            output_root      = model_dir,
            save_overlays    = self.checkBox_saveOverlays.isChecked(),
            progressBar      = progressBar,
        )
        evaluator.progress_bar_closed = progress_closed[0]

        try:
            aggregate = evaluator.run()
            progressBar.close()

            if aggregate is None:
                return

            msg = (
                f"Holdout Validation Complete\n\n"
                f"Seasons evaluated: {', '.join(eval_seasons)}\n"
                f"Images:    {aggregate['n_images']}\n"
                f"Mean IoU:  {aggregate['iou']:.4f}\n"
                f"Mean Dice: {aggregate['dice']:.4f}\n"
                f"Mean Acc:  {aggregate['accuracy']:.4f}\n\n"
                f"Results saved to:\n{model_dir}/Holdout_Validation/"
            )
            QMessageBox.information(self, "Holdout Validation Complete", msg)

        except Exception as e:
            progressBar.close()
            import traceback
            QMessageBox.critical(self, "Evaluation Error",
                                 f"Error during evaluation:\n{e}\n\n{traceback.format_exc()}")
