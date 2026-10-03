#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# sam3_concept_tab.py
#
# SAM3 Concept Segmentation plugin for the ML Image Processing dialog.
#
# Point at a folder of images, type a concept prompt ("water", "sky", ...), and
# SAM3 (via Ultralytics) segments every instance of that concept in each image.
# Outputs mirror the SAM2/SegFormer engines: an overlay PNG, a binary _mask.png,
# and a COCO predictions.json — written to a predictions folder this plugin
# controls (Ultralytics' own save is disabled).
#
# Self-registering plugin: drop this file in the application's plugins folder. The
# PLUGIN manifest at the bottom lists it in Tools > Plugins and the ML dialog tab bar.
#
# VERIFIED API (Ultralytics 8.4.x, confirmed on-machine):
#   from ultralytics.models.sam import SAM3SemanticPredictor
#   predictor = SAM3SemanticPredictor(overrides={"model": <path>, "task": "segment",
#                                                 "mode": "predict", "conf": 0.25,
#                                                 "save": False})
#   predictor.set_image(image_path)
#   results = predictor(text=["water"])          # NOTE: keyword is `text`, a list
#   results[0].masks.xy    -> list[np.ndarray]   polygon vertices per instance
#   results[0].masks.data  -> torch.Tensor (N,H,W) binary masks
#   results[0].boxes.xyxy  -> torch.Tensor (N,4)  bboxes
#
# Author: John Edward Stranzl, Jr.
# License: Apache License, Version 2.0

import os
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtWidgets import (QWidget, QFileDialog, QListWidgetItem, QMessageBox,
                             QLabel, QLineEdit, QPushButton, QListWidget,
                             QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QProgressBar)
from PyQt5.QtGui import QPixmap, QIcon


# ----------------------------------------------------------------------
# Namespace-agnostic host resolution.
#
# This plugin needs the application's shared COCO helpers and (optionally)
# its PROJECT_ROOT / JsonEditor. Those live in the installed app package —
# but the plugin must not hardcode that package's name, nor a list of names.
#
# Instead we derive the host package at runtime from the object that loads
# the plugin: the ML dialog passed as `parent`. type(parent).__module__ is
# e.g. "<pkg>.dialogs.ML_image_processing..."; its first segment is the
# host package. We import "<pkg>.ml_core.ml_helpers" / "<pkg>" from that.
# No literal namespace, no list. If resolution fails, the plugin degrades:
# COCO output is skipped (with a clear message), settings persistence is
# disabled, and the browse start-dir falls back to the home directory.
# ----------------------------------------------------------------------
def _resolve_host(parent):
    """Return (helpers_module_or_None, project_root_str, json_editor_cls_or_None)
    by deriving the host package from the loading dialog. No hardcoded namespace."""
    import importlib

    pkg = None
    if parent is not None:
        mod = type(parent).__module__ or ""
        pkg = mod.split(".")[0] or None
    if not pkg:
        # Opened from Tools > Plugins, which builds the widget with no parent.
        # The shared "appcore" alias names no product, so it is used here.
        pkg = "appcore"

    helpers = None
    project_root = str(Path.home())
    json_editor = None

    if pkg:
        try:
            helpers = importlib.import_module(f"{pkg}.ml_core.ml_helpers")
        except Exception:
            helpers = None
        try:
            host_pkg = importlib.import_module(pkg)
            project_root = str(getattr(host_pkg, "PROJECT_ROOT", project_root))
        except Exception:
            pass
        try:
            je_mod = importlib.import_module(f"{pkg}.JSON_Editor")
            json_editor = getattr(je_mod, "JsonEditor", None)
        except Exception:
            json_editor = None

    return helpers, project_root, json_editor

_SETTINGS_KEY_FOLDER = "SAM3_Concept_Images_Folder"
_SETTINGS_KEY_PROMPT = "SAM3_Concept_Prompt"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class SAM3ConceptTab(QWidget):
    """Folder-level SAM3 concept segmentation with overlay/mask/COCO output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Derive host-provided helpers at runtime (no hardcoded namespace).
        self._helpers, self._project_root, self._json_editor = _resolve_host(parent)
        self._image_paths = []
        self._pendingThumbnails = []
        self._loadToken = 0
        self._predictor = None            # created once, reused across the folder
        self._model_path = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI (built in code so the plugin is a single self-contained file)
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)

        # Model row
        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("SAM3 model:"))
        self.lineEdit_model = QLineEdit()
        self.lineEdit_model.setPlaceholderText("Path to sam3.pt…")
        default_model = os.path.join(str(self._project_root), "sam3.pt")
        if os.path.exists(default_model):
            self.lineEdit_model.setText(default_model)
        self.pushButton_browse_model = QPushButton("Browse")
        row_model.addWidget(self.lineEdit_model)
        row_model.addWidget(self.pushButton_browse_model)
        root.addLayout(row_model)

        # Images folder row
        row_folder = QHBoxLayout()
        row_folder.addWidget(QLabel("Images folder:"))
        self.lineEdit_folder = QLineEdit()
        self.lineEdit_folder.setPlaceholderText("Folder of images to segment…")
        self.pushButton_browse_folder = QPushButton("Browse")
        row_folder.addWidget(self.lineEdit_folder)
        row_folder.addWidget(self.pushButton_browse_folder)
        root.addLayout(row_folder)

        # Prompt + confidence row
        row_prompt = QHBoxLayout()
        row_prompt.addWidget(QLabel("Concept prompt(s):"))
        self.lineEdit_prompt = QLineEdit()
        self.lineEdit_prompt.setPlaceholderText('e.g. water, sky, vegetation')
        row_prompt.addWidget(self.lineEdit_prompt)
        row_prompt.addWidget(QLabel("Confidence:"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.05, 0.95)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)
        row_prompt.addWidget(self.spin_conf)
        root.addLayout(row_prompt)

        # Filmstrip
        self.listWidget_filmstrip = QListWidget()
        self.listWidget_filmstrip.setViewMode(QListWidget.IconMode)
        self.listWidget_filmstrip.setIconSize(QSize(100, 100))
        self.listWidget_filmstrip.setFlow(QListWidget.LeftToRight)
        self.listWidget_filmstrip.setWrapping(False)
        self.listWidget_filmstrip.setFixedHeight(132)
        self.listWidget_filmstrip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.listWidget_filmstrip.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        root.addWidget(self.listWidget_filmstrip)

        # Progress + run
        self.progress = QProgressBar()
        self.progress.setValue(0)
        root.addWidget(self.progress)

        row_run = QHBoxLayout()
        row_run.addStretch(1)
        self.pushButton_run = QPushButton("Run SAM3 Segmentation")
        self.pushButton_run.setMinimumHeight(36)
        row_run.addWidget(self.pushButton_run)
        row_run.addStretch(1)
        root.addLayout(row_run)

        self.label_status = QLabel("")
        root.addWidget(self.label_status)

    # ------------------------------------------------------------------
    # Plugin lifecycle hooks (called by _add_tab_safe post-list)
    # ------------------------------------------------------------------
    def configure_filmstrip(self):
        lw = self.listWidget_filmstrip
        lw.setWrapping(False)
        lw.setSpacing(0)
        lw.setFixedHeight(lw.iconSize().height() + 16)

    def wire_connections(self):
        self.pushButton_browse_model.clicked.connect(self._browse_model)
        self.pushButton_browse_folder.clicked.connect(self._browse_folder)
        self.lineEdit_folder.editingFinished.connect(self._on_folder_changed)
        self.pushButton_run.clicked.connect(self._run)

        if self._json_editor is not None:
            saved = self._json_editor().getValue(_SETTINGS_KEY_FOLDER)
            if saved and os.path.isdir(saved):
                self.lineEdit_folder.setText(saved)
                self._on_folder_changed()
            saved_prompt = self._json_editor().getValue(_SETTINGS_KEY_PROMPT)
            if saved_prompt:
                self.lineEdit_prompt.setText(saved_prompt)

    # ------------------------------------------------------------------
    # Folder / filmstrip
    # ------------------------------------------------------------------
    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SAM3 model", str(self._project_root), "SAM3 weights (*.pt)")
        if path:
            self.lineEdit_model.setText(path)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Images Folder", str(self._project_root))
        if folder:
            self.lineEdit_folder.setText(folder)
            self._on_folder_changed()

    def _on_folder_changed(self):
        folder = self.lineEdit_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            return
        if self._json_editor is not None:
            self._json_editor().update_json_entry(_SETTINGS_KEY_FOLDER, folder)
        self._image_paths = [
            os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMAGE_EXTS)
        ]
        self._populate_filmstrip(self._image_paths)

    def _populate_filmstrip(self, paths):
        lw = self.listWidget_filmstrip
        lw.clear()
        self._loadToken += 1
        token = self._loadToken
        self._pendingThumbnails = []
        for idx, p in enumerate(paths):
            item = QListWidgetItem(QIcon(), str(idx + 1))
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
            item.setToolTip(os.path.basename(p))
            item.setSizeHint(QSize(lw.iconSize().width(), lw.iconSize().height() + 16))
            lw.addItem(item)
            self._pendingThumbnails.append((item, p, token))
        QTimer.singleShot(50, lambda: self._load_batch(token))

    def _load_batch(self, token):
        if token != self._loadToken:
            return
        lw = self.listWidget_filmstrip
        for _ in range(min(10, len(self._pendingThumbnails))):
            item, p, _tok = self._pendingThumbnails.pop(0)
            pix = QPixmap(p)
            if not pix.isNull():
                item.setIcon(QIcon(pix.scaled(lw.iconSize(), Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation)))
        if self._pendingThumbnails:
            QTimer.singleShot(50, lambda: self._load_batch(token))

    # ------------------------------------------------------------------
    # Predictor (created once, reused across the folder)
    # ------------------------------------------------------------------
    def _ensure_predictor(self, model_path, conf):
        """Build SAM3SemanticPredictor once; rebuild only if the model path changes."""
        if self._predictor is not None and self._model_path == model_path:
            return self._predictor
        from ultralytics.models.sam import SAM3SemanticPredictor
        self._predictor = SAM3SemanticPredictor(overrides={
            "model": model_path,
            "task": "segment",
            "mode": "predict",
            "conf": float(conf),
            "save": False,        # we write our own outputs; don't touch runs_dir
            "verbose": False,
        })
        self._model_path = model_path
        return self._predictor

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _run(self):
        model_path = self.lineEdit_model.text().strip()
        folder = self.lineEdit_folder.text().strip()
        prompt_raw = self.lineEdit_prompt.text().strip()

        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "SAM3", "Select a valid sam3.pt model file.")
            return
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "SAM3", "Select a valid images folder.")
            return
        if not prompt_raw:
            QMessageBox.warning(self, "SAM3",
                                "Enter one or more concepts (e.g. 'water, sky, vegetation').")
            return
        if not self._image_paths:
            QMessageBox.warning(self, "SAM3", "No images found in the folder.")
            return

        # Comma-separated concepts -> SAM3 text list. This ordering is also the
        # class-index order SAM3 returns (result.boxes.cls indexes into it).
        concepts = [c.strip() for c in prompt_raw.split(",") if c.strip()]
        if not concepts:
            QMessageBox.warning(self, "SAM3", "No valid concepts parsed.")
            return

        if self._json_editor is not None:
            self._json_editor().update_json_entry(_SETTINGS_KEY_PROMPT, prompt_raw)

        label = "_".join(concepts)
        out_dir = os.path.join(folder, f"{label}_predictions (sam3)")
        os.makedirs(out_dir, exist_ok=True)

        try:
            predictor = self._ensure_predictor(model_path, self.spin_conf.value())
        except Exception as e:
            QMessageBox.critical(self, "SAM3", f"Failed to load SAM3 model:\n{e}")
            return

        # One COCO category per concept: id = index+1, name = concept.
        cat_id_for = {c: i + 1 for i, c in enumerate(concepts)}
        coco = None
        if self._helpers is not None:
            coco = self._helpers.init_coco_structure(
                [{"id": cat_id_for[c], "name": c} for c in concepts])
        else:
            self.label_status.setText(
                "Note: host COCO helpers not found — writing overlays/masks only, "
                "no predictions.json.")

        # A distinct overlay colour per concept (BGR).
        palette = [(0, 150, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255),
                   (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 0, 255)]
        color_for = {c: palette[i % len(palette)] for i, c in enumerate(concepts)}

        self.progress.setRange(0, len(self._image_paths))
        image_id = 1
        annotation_id = 1
        total_instances = 0
        images_with_hits = 0
        per_concept_counts = {c: 0 for c in concepts}

        for i, image_path in enumerate(self._image_paths):
            self.progress.setValue(i)
            self.label_status.setText(f"Segmenting {concepts} — "
                                      f"{i + 1}/{len(self._image_paths)}: "
                                      f"{os.path.basename(image_path)}")
            try:
                predictor.set_image(image_path)
                results = predictor(text=concepts)
            except Exception as e:
                self.label_status.setText(f"Error on {os.path.basename(image_path)}: {e}")
                continue

            n, counts = self._write_outputs(results[0], image_path, out_dir,
                                            cat_id_for, color_for, coco,
                                            image_id, annotation_id)
            annotation_id += n
            image_id += 1
            total_instances += n
            for c, k in counts.items():
                per_concept_counts[c] += k
            if n > 0:
                images_with_hits += 1

        if self._helpers is not None and coco is not None:
            self._helpers.save_coco_json(coco, out_dir)
        self.progress.setValue(len(self._image_paths))

        breakdown = ", ".join(f"{c}: {per_concept_counts[c]}" for c in concepts)
        self.label_status.setText(
            f"Done. {images_with_hits}/{len(self._image_paths)} images had a hit. "
            f"Instances — {breakdown}. Output: {out_dir}")
        QMessageBox.information(self, "SAM3",
                                f"Segmentation complete.\n\n"
                                f"Concepts: {', '.join(concepts)}\n"
                                f"Images with a hit: {images_with_hits}/{len(self._image_paths)}\n"
                                f"Instances per concept:\n  {breakdown}\n\n"
                                f"Output:\n{out_dir}")

    # ------------------------------------------------------------------
    def _write_outputs(self, result, image_path, out_dir, cat_id_for, color_for,
                       coco, image_id, annotation_id):
        """Write one color overlay + a per-concept mask (each in its own
        subfolder) + per-instance COCO tagged by matched concept.
        Returns (total_instances_written, {concept: count})."""
        counts = {c: 0 for c in cat_id_for}
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return 0, counts
        h, w = img_bgr.shape[:2]
        base = os.path.splitext(os.path.basename(image_path))[0]

        # One accumulator mask per concept -> each written to <out_dir>/<concept>/.
        per_concept_mask = {c: np.zeros((h, w), np.uint8) for c in cat_id_for}

        def _flush_masks():
            for c, m in per_concept_mask.items():
                cdir = os.path.join(out_dir, c)
                os.makedirs(cdir, exist_ok=True)
                cv2.imwrite(os.path.join(cdir, f"{base}_mask.png"), m)

        masks = getattr(result, "masks", None)
        if masks is None or masks.data is None or len(masks.data) == 0:
            _flush_masks()   # empty masks per concept, so folders stay complete
            cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"), img_bgr)
            return 0, counts

        data = masks.data.cpu().numpy()                 # (N, mh, mw)
        names = getattr(result, "names", None) or {}
        cls = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None \
            else np.zeros(len(data), int)

        def concept_of(idx):
            ci = int(cls[idx]) if idx < len(cls) else 0
            if isinstance(names, dict):
                return names.get(ci, str(ci))
            if isinstance(names, (list, tuple)) and ci < len(names):
                return names[ci]
            return str(ci)

        overlay = img_bgr.copy()
        rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        n_written = 0
        for idx in range(len(data)):
            m_full = cv2.resize(data[idx].astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST)
            if int(m_full.sum()) == 0:
                continue
            concept = concept_of(idx)
            mbool = m_full.astype(bool)
            # accumulate into THIS concept's mask only
            if concept in per_concept_mask:
                per_concept_mask[concept][mbool] = 255
            color = color_for.get(concept, (0, 150, 255))
            overlay[mbool] = (0.4 * np.array(color) + 0.6 * overlay[mbool]).astype(np.uint8)

            if self._helpers is not None and coco is not None:
                cat_id = cat_id_for.get(concept, 1)
                before = len(coco.get("annotations", []))
                self._helpers.add_coco_entries(
                    coco, image_path, m_full, rgb_image,
                    image_id, annotation_id + n_written)
                for ann in coco["annotations"][before:]:
                    ann["category_id"] = cat_id
            if concept in counts:
                counts[concept] += 1
            n_written += 1

        _flush_masks()   # one mask per concept, each in <out_dir>/<concept>/
        cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"), overlay)
        return n_written, counts


# ======================================================================
# Plugin manifest — the ML dialog's _load_plugins reads this to register
# the tab. Placed in the application's plugins folder (app_identity.PLUGINS_DIR).
# ======================================================================
PLUGIN = {
    "title": "SAM3 Concept Segmentation",
    "class": "SAM3ConceptTab",
    "ui": None,        # UI is built in code (no .ui file)
    "post": ["configure_filmstrip", "wire_connections"],
    "surface": ["tools", "ml"],   # Tools > Plugins menu and the ML dialog tab bar
    "size": [1200, 400],          # opening window size (width, height), 3:1 for the path rows
    "api_version": 1,
}
