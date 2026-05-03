#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# surface_state_classifier_tab.py
# Surface State Classifier tab for GRIME AI ML Image Processing Dialog.
# Classifies the surface condition (e.g. ice, snow) of a segmented ROI
# using SVM or Random Forest trained on user-labeled positive examples.
#
# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# License: Apache License, Version 2.0

import os
import re
import cv2
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Tuple, Optional

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QFileDialog, QListWidgetItem, QMessageBox
)
from PyQt5.QtGui import QPixmap, QIcon, QImage, QColor, QFont

from GRIME_AI import PROJECT_ROOT
from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_CSS_Styles import BUTTON_CSS_STEEL_BLUE
from GRIME_AI.utils.resource_utils import ui_path

# ---------------------------------------------------------------------------
# Qt UserRole constants
# ---------------------------------------------------------------------------
ROLE_INDEX    = Qt.UserRole          # int  — index into self._pairs
ROLE_POSITIVE = Qt.UserRole + 1      # bool — user-labeled positive example
ROLE_RESULT   = Qt.UserRole + 2      # Optional[bool] — classifier output (None = unclassified)

# Badge colors
COLOR_POSITIVE     = QColor(0,   180,  0)    # green  — labeled positive
COLOR_NEG_RESULT   = QColor(180,   0,  0)    # red    — classified negative
COLOR_POS_RESULT   = QColor(0,   120, 220)   # blue   — classified positive
COLOR_UNCLASSIFIED = QColor(80,   80, 80)    # grey   — no result yet

BADGE_SIZE = 14   # pixels — small corner badge on thumbnail


# ===========================================================================
# Feature extraction worker
# ===========================================================================
class FeatureExtractionWorker(QThread):
    """
    Extracts HSV statistics, GLCM, Gabor filter bank, and Shannon entropy
    from every (image, mask) pair in self._pairs for the selected class mask.
    Emits progress and the completed feature matrix.
    """

    progress   = pyqtSignal(int)           # current iteration index
    finished   = pyqtSignal(object)        # pd.DataFrame of features, one row per frame
    error      = pyqtSignal(str)

    # Gabor filter bank parameters
    GABOR_ORIENTATIONS = [0, 30, 60, 90, 120, 150]   # degrees
    GABOR_SCALES       = [4, 8, 16]                   # wavelength in pixels

    def __init__(self, pairs: List[Tuple[str, str]], parent=None):
        super().__init__(parent)
        self._pairs = pairs

    # ------------------------------------------------------------------
    def run(self):
        try:
            rows = []
            for i, (orig_path, mask_path) in enumerate(self._pairs):
                self.progress.emit(i)
                row = self._extract_one(orig_path, mask_path)
                if row is not None:
                    rows.append(row)

            df = pd.DataFrame(rows)
            self.finished.emit(df)

        except Exception as exc:
            self.error.emit(str(exc))

    # ------------------------------------------------------------------
    def _extract_one(self, orig_path: str, mask_path: str) -> Optional[dict]:
        """Return a feature dict for one image/mask pair, or None on failure."""
        try:
            img_bgr = cv2.imread(orig_path)
            mask    = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or mask is None:
                return None

            # Binarise mask — any non-zero pixel belongs to the ROI
            _, bin_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

            row = {"image_path": orig_path, "mask_path": mask_path}

            # ── HSV statistics ─────────────────────────────────────────
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            roi_pixels = img_hsv[bin_mask > 0]   # shape (N, 3)
            if roi_pixels.shape[0] == 0:
                return None

            h_vals = roi_pixels[:, 0] * 2.0      # scale 0–180 → 0–360
            s_vals = roi_pixels[:, 1]
            v_vals = roi_pixels[:, 2]

            row["hsv_h_mean"]   = float(np.mean(h_vals))
            row["hsv_h_std"]    = float(np.std(h_vals))
            row["hsv_s_mean"]   = float(np.mean(s_vals))
            row["hsv_s_std"]    = float(np.std(s_vals))
            row["hsv_v_mean"]   = float(np.mean(v_vals))
            row["hsv_v_std"]    = float(np.std(v_vals))

            # ── Shannon entropy ────────────────────────────────────────
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[bin_mask > 0].astype(np.uint8)
            hist, _ = np.histogram(roi_gray, bins=256, range=(0, 256))
            hist = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            row["entropy"] = float(entropy)

            # ── GLCM features ──────────────────────────────────────────
            try:
                from GRIME_AI.GRIME_AI_Texture import GLCMTexture
                glcm_feats = GLCMTexture().compute_features(img_bgr, mask=bin_mask)
                row["glcm_contrast"]    = float(glcm_feats.get("contrast",    0.0))
                row["glcm_homogeneity"] = float(glcm_feats.get("homogeneity", 0.0))
                row["glcm_correlation"] = float(glcm_feats.get("correlation", 0.0))
                row["glcm_energy"]      = float(glcm_feats.get("energy",      0.0))
            except Exception as e:
                print(f"[SSC] GLCM failed for {orig_path}: {e}")
                row["glcm_contrast"] = row["glcm_homogeneity"] = 0.0
                row["glcm_correlation"] = row["glcm_energy"]   = 0.0

            # ── Gabor filter bank ──────────────────────────────────────
            gray_float = gray.astype(np.float32) / 255.0
            for theta_deg in self.GABOR_ORIENTATIONS:
                theta = np.deg2rad(theta_deg)
                for lam in self.GABOR_SCALES:
                    kernel = cv2.getGaborKernel(
                        ksize=(31, 31),
                        sigma=lam / 2.0,
                        theta=theta,
                        lambd=float(lam),
                        gamma=0.5,
                        psi=0,
                        ktype=cv2.CV_32F,
                    )
                    response = cv2.filter2D(gray_float, cv2.CV_32F, kernel)
                    roi_resp = response[bin_mask > 0]
                    col_prefix = f"gabor_t{theta_deg:03d}_l{lam:02d}"
                    row[f"{col_prefix}_mean"] = float(np.mean(roi_resp))
                    row[f"{col_prefix}_var"]  = float(np.var(roi_resp))

            return row

        except Exception as e:
            print(f"[SSC] Feature extraction failed for {orig_path}: {e}")
            return None


# ===========================================================================
# Classifier training / inference worker
# ===========================================================================
class ClassifierWorker(QThread):
    """
    Trains SVM or Random Forest on labeled examples, then classifies all frames.
    """

    finished = pyqtSignal(object, object)   # (trained_model, predictions: pd.Series)
    error    = pyqtSignal(str)

    def __init__(self, feature_df: pd.DataFrame, positive_indices: List[int],
                 use_rf: bool, rf_trees: int, rf_max_depth: int,
                 svm_c: float, svm_kernel: str, parent=None):
        super().__init__(parent)
        self._df              = feature_df
        self._pos_indices     = set(positive_indices)
        self._use_rf          = use_rf
        self._rf_trees        = rf_trees
        self._rf_max_depth    = rf_max_depth if rf_max_depth > 0 else None
        self._svm_c           = svm_c
        self._svm_kernel      = svm_kernel

    # ------------------------------------------------------------------
    def run(self):
        try:
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler

            feature_cols = [c for c in self._df.columns
                            if c not in ("image_path", "mask_path")]

            X = self._df[feature_cols].values.astype(np.float32)
            y_labels = np.array([
                1 if i in self._pos_indices else 0
                for i in range(len(self._df))
            ])

            # Only train on labeled rows (positive + a balanced negative sample)
            pos_idx  = np.where(y_labels == 1)[0]
            neg_pool = np.where(y_labels == 0)[0]

            if len(pos_idx) == 0:
                self.error.emit("No positive examples selected.")
                return

            # Balance negative sample to 3× positives (or all if fewer available)
            n_neg = min(len(neg_pool), len(pos_idx) * 3)
            rng   = np.random.default_rng(42)
            neg_idx = rng.choice(neg_pool, size=n_neg, replace=False)

            train_idx = np.concatenate([pos_idx, neg_idx])
            X_train   = X[train_idx]
            y_train   = y_labels[train_idx]

            # Scale features
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_all   = scaler.transform(X)

            # Build classifier
            if self._use_rf:
                clf = RandomForestClassifier(
                    n_estimators=self._rf_trees,
                    max_depth=self._rf_max_depth,
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                clf = SVC(
                    C=self._svm_c,
                    kernel=self._svm_kernel,
                    probability=True,
                    random_state=42,
                )

            clf.fit(X_train, y_train)
            predictions = clf.predict(X_all)

            # Bundle scaler with classifier for export
            pipeline = {"scaler": scaler, "classifier": clf,
                        "feature_cols": feature_cols}

            self.finished.emit(pipeline, pd.Series(predictions,
                                                    index=self._df.index))

        except Exception as exc:
            self.error.emit(str(exc))


# ===========================================================================
# SurfaceStateClassifierTab
# ===========================================================================
class SurfaceStateClassifierTab(QWidget):
    """
    Tab for the GRIME AI ML Image Processing dialog.
    Loads segmented image/mask pairs, lets the user mark positive examples
    of a surface condition, trains SVM or RF, classifies all frames, and
    exports the trained model and per-frame results CSV.
    """

    # ── Virtual filmstrip parameters (match ROI Analyzer) ──────────────────
    _BATCH_SIZE  = 10
    _BATCH_DELAY = 50   # ms

    def __init__(self, parent=None):
        super().__init__(parent)

        # File pairs: list of (orig_path, mask_path)
        self._pairs: List[Tuple[str, str]] = []

        # Virtual filmstrip state
        self._pending_thumbnails = []
        self._load_token         = 0

        # Raw pixmaps for the two display labels (rescaled on resize)
        self._raw_pixmap_orig    = None
        self._raw_pixmap_overlay = None

        # Feature matrix (populated by Extract Features)
        self._feature_df: Optional[pd.DataFrame] = None

        # Trained pipeline dict: {"scaler", "classifier", "feature_cols"}
        self._trained_pipeline = None

        # Per-frame classification results: pd.Series of 0/1
        self._predictions: Optional[pd.Series] = None

        # Worker threads
        self._extract_worker: Optional[FeatureExtractionWorker] = None
        self._classify_worker: Optional[ClassifierWorker] = None
        self._progress_wheel: Optional[QProgressWheel] = None

    # -----------------------------------------------------------------------
    # Public setup — called from GRIME_AI_ML_ImageProcessingDlg after loadUi
    # -----------------------------------------------------------------------
    def configure_filmstrip(self):
        """Mirror ROI Analyzer: single-row, no frame, no wrapping."""
        from PyQt5.QtWidgets import QFrame
        lw = self.listWidget_filmstrip
        lw.setWrapping(False)
        lw.setSpacing(0)
        lw.setContentsMargins(0, 0, 0, 0)
        lw.setViewportMargins(0, 0, 0, 0)
        lw.setFrameShape(QFrame.NoFrame)
        lw.setFixedHeight(lw.iconSize().height())

    # -----------------------------------------------------------------------
    def wire_connections(self):
        # Browse buttons
        self.pushButton_browse_images_folder.clicked.connect(
            self._browse_images_folder)
        self.pushButton_browse_images_folder.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_browse_model_file.clicked.connect(
            self._browse_model_file)
        self.pushButton_browse_model_file.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        # Folder / model path edits
        self.lineEdit_segmented_images_folder.editingFinished.connect(
            self._on_folder_changed)
        self.lineEdit_model_file.editingFinished.connect(
            self._on_model_changed)

        # Filmstrip
        self.listWidget_filmstrip.itemClicked.connect(
            self._on_filmstrip_item_clicked)
        self.listWidget_filmstrip.itemSelectionChanged.connect(
            self._on_selection_changed)

        # Filmstrip filter
        self.comboBox_filmstrip_filter.currentIndexChanged.connect(
            self._apply_filmstrip_filter)

        # Classifier radio buttons — toggle param widgets
        self.radioButton_svm.toggled.connect(self._on_classifier_toggled)
        self.radioButton_rf.toggled.connect(self._on_classifier_toggled)

        # Select / Unselect All
        self.pushButton_select_all.clicked.connect(self._select_all)
        self.pushButton_select_all.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_unselect_all.clicked.connect(self._unselect_all)
        self.pushButton_unselect_all.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        # Workflow buttons
        self.pushButton_extract_features.clicked.connect(
            self._on_extract_features)
        self.pushButton_extract_features.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_load_features.clicked.connect(self._on_load_features)
        self.pushButton_load_features.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_train_classifier.clicked.connect(
            self._on_train_classifier)
        self.pushButton_train_classifier.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_classify_frames.clicked.connect(
            self._on_classify_frames)
        self.pushButton_classify_frames.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_export_model.clicked.connect(self._on_export_model)
        self.pushButton_export_model.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_export_results.clicked.connect(self._on_export_results)
        self.pushButton_export_results.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        # Splitter stretch
        self.splitter_display.setStretchFactor(0, 4)
        self.splitter_display.setStretchFactor(1, 1)

        # Initial button states
        self._set_button_states()

    # =======================================================================
    # Resize / show events
    # =======================================================================
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_images()

    def showEvent(self, event):
        super().showEvent(event)
        if self._pairs and self.listWidget_filmstrip.count() > 0:
            self._display_frame(0)

    # =======================================================================
    # Browse helpers
    # =======================================================================
    def _browse_images_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Segmented Images Folder", str(PROJECT_ROOT))
        if folder:
            self.lineEdit_segmented_images_folder.setText(folder)
            self._on_folder_changed()

    def _browse_model_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Segmentation Model", str(PROJECT_ROOT),
            "Torch Model (*.torch);;All Files (*)")
        if path:
            self.lineEdit_model_file.setText(path)
            self._on_model_changed()

    # =======================================================================
    # Folder / model change handlers
    # =======================================================================
    def _on_folder_changed(self):
        folder = self.lineEdit_segmented_images_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            return
        self._load_pairs_from_folder(folder)

    def _on_model_changed(self):
        model_path = self.lineEdit_model_file.text().strip()
        if not model_path or not os.path.isfile(model_path):
            return
        self._load_classes_from_model(model_path)

    # -----------------------------------------------------------------------
    def _feature_csv_path(self) -> Optional[str]:
        """Return the expected feature CSV path for the current segmented images folder."""
        folder = self.lineEdit_segmented_images_folder.text().strip()
        if not folder:
            return None
        folder_name = os.path.basename(os.path.normpath(folder))
        return os.path.join(folder, f"{folder_name}_surface_state_features.csv")

    # -----------------------------------------------------------------------
    def _load_pairs_from_folder(self, folder: str):
        """Scan folder for (original_image, mask) pairs and populate filmstrip."""
        try:
            from GRIME_AI.GRIME_AI_ROI_Analyzer import GRIME_AI_ROI_Analyzer
            pairs = GRIME_AI_ROI_Analyzer("", "").generate_file_pairs(folder)
        except Exception as e:
            QMessageBox.warning(self, "Surface State Classifier",
                                f"Could not load image pairs:\n{e}")
            return

        if not pairs:
            QMessageBox.warning(self, "Surface State Classifier",
                                "No image/mask pairs found in the selected folder.")
            return

        self._pairs = pairs
        self._feature_df    = None
        self._predictions   = None
        self._trained_pipeline = None
        self._set_button_states()

        self.populate_filmstrip([orig for orig, _ in pairs])

        # Auto-detect existing feature CSV
        csv_path = self._feature_csv_path()
        if csv_path and os.path.isfile(csv_path):
            reply = QMessageBox.question(
                self, "Feature CSV Found",
                f"A feature CSV was found for this folder:\n{os.path.basename(csv_path)}\n\nLoad it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._load_features_from_csv(csv_path)

    # -----------------------------------------------------------------------
    def _load_classes_from_model(self, model_path: str):
        """Read class labels from .torch metadata and populate comboBox_class_selection."""
        try:
            import torch
            metadata = torch.load(model_path, map_location="cpu",
                                  weights_only=False)
            # Expect metadata dict with "categories" or "label_names" key
            categories = (metadata.get("categories")
                          or metadata.get("label_names")
                          or metadata.get("classes")
                          or [])

            self.comboBox_class_selection.clear()
            if not categories:
                QMessageBox.warning(
                    self, "Surface State Classifier",
                    "No class labels found in model metadata.")
                return

            for cat in categories:
                # Handle both str and dict ({"id": 1, "name": "water"})
                label = cat if isinstance(cat, str) else cat.get("name", str(cat))
                self.comboBox_class_selection.addItem(label)

        except Exception as e:
            QMessageBox.warning(self, "Surface State Classifier",
                                f"Could not read model metadata:\n{e}")

    # =======================================================================
    # Virtual filmstrip
    # =======================================================================
    def populate_filmstrip(self, image_paths: List[str]):
        """
        Insert placeholder items immediately, then load thumbnails in batches
        via QTimer — identical strategy to ROI Analyzer.
        """
        lw = self.listWidget_filmstrip
        lw.clear()

        self._load_token += 1
        token = self._load_token
        self._pending_thumbnails.clear()

        icon_size = lw.iconSize()
        for idx, path in enumerate(image_paths):
            item = QListWidgetItem(QIcon(), "")
            item.setData(ROLE_INDEX,    idx)
            item.setData(ROLE_POSITIVE, False)
            item.setData(ROLE_RESULT,   None)
            item.setSizeHint(icon_size)
            lw.addItem(item)
            self._pending_thumbnails.append((item, path, token))

        if lw.count():
            self._display_frame(0)

        QTimer.singleShot(self._BATCH_DELAY,
                          lambda: self._load_next_batch(token))

    # -----------------------------------------------------------------------
    def _load_next_batch(self, token: int):
        if token != self._load_token:
            return

        lw        = self.listWidget_filmstrip
        icon_size = lw.iconSize()

        for _ in range(min(self._BATCH_SIZE, len(self._pending_thumbnails))):
            item, path, _ = self._pending_thumbnails.pop(0)
            if not os.path.exists(path):
                continue
            pix = QPixmap(path)
            if pix.isNull():
                continue
            thumb = pix.scaled(icon_size, Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
            item.setIcon(QIcon(thumb))

        if self._pending_thumbnails:
            QTimer.singleShot(self._BATCH_DELAY,
                              lambda: self._load_next_batch(token))

    # =======================================================================
    # Filmstrip interaction
    # =======================================================================
    def _on_filmstrip_item_clicked(self, item: QListWidgetItem):
        """Toggle positive label on click; update badge and display."""
        idx      = item.data(ROLE_INDEX)
        was_pos  = item.data(ROLE_POSITIVE)
        is_pos   = not was_pos
        item.setData(ROLE_POSITIVE, is_pos)
        self._redraw_badge(item)
        self._update_positive_count()
        self._display_frame(idx)

    # -----------------------------------------------------------------------
    def _on_selection_changed(self):
        self._update_positive_count()

    # -----------------------------------------------------------------------
    def _display_frame(self, idx: int):
        """Load and display original + mask overlay for the given pair index."""
        if idx >= len(self._pairs):
            return

        orig_path, mask_path = self._pairs[idx]

        img_bgr = cv2.imread(orig_path)
        mask    = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img_bgr is None:
            return

        orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._raw_pixmap_orig = self._np_to_pixmap(orig_rgb)

        if mask is not None:
            overlay = img_bgr.copy()
            colored = np.zeros_like(img_bgr)
            colored[mask > 0] = (0, 180, 0)   # green tint over ROI
            overlay = cv2.addWeighted(overlay, 0.7, colored, 0.3, 0)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            self._raw_pixmap_overlay = self._np_to_pixmap(overlay_rgb)
        else:
            self._raw_pixmap_overlay = None

        self._refresh_images()

        # Update frame info bar
        self.label_frame_filename.setText(
            f"Filename: {os.path.basename(orig_path)}")

        lw   = self.listWidget_filmstrip
        item = lw.item(idx)
        if item:
            is_pos = item.data(ROLE_POSITIVE)
            result = item.data(ROLE_RESULT)
            if result is None:
                status = "Positive" if is_pos else "Unlabeled"
            else:
                status = f"Classified: {'Positive' if result else 'Negative'}"
            self.label_frame_status.setText(f"Status: {status}")

    # -----------------------------------------------------------------------
    def _refresh_images(self):
        for label, pix in [
            (self.label_image_original,    self._raw_pixmap_orig),
            (self.label_image_mask_overlay, self._raw_pixmap_overlay),
        ]:
            if pix is None:
                label.clear()
                continue
            w, h = label.width(), label.height()
            if w > 0 and h > 0:
                label.setPixmap(pix.scaled(w, h, Qt.KeepAspectRatio,
                                           Qt.SmoothTransformation))

    # -----------------------------------------------------------------------
    @staticmethod
    def _np_to_pixmap(arr: np.ndarray) -> QPixmap:
        arr = arr if arr.flags["C_CONTIGUOUS"] else arr.copy()
        h, w = arr.shape[:2]
        bpl  = w * arr.shape[2]
        fmt  = (QImage.Format_RGBA8888 if arr.shape[2] == 4
                else QImage.Format_RGB888)
        return QPixmap.fromImage(QImage(arr.data, w, h, bpl, fmt))

    # -----------------------------------------------------------------------
    def _redraw_badge(self, item: QListWidgetItem):
        """
        Overlay a small colored square badge in the top-left corner of the
        thumbnail to indicate labeling / classification status.
        """
        idx = item.data(ROLE_INDEX)
        if idx >= len(self._pairs):
            return

        orig_path = self._pairs[idx][0]
        pix = QPixmap(orig_path)
        if pix.isNull():
            return

        icon_size = self.listWidget_filmstrip.iconSize()
        thumb = pix.scaled(icon_size, Qt.KeepAspectRatio,
                           Qt.SmoothTransformation)

        from PyQt5.QtGui import QPainter
        painter = QPainter(thumb)
        is_pos = item.data(ROLE_POSITIVE)
        result = item.data(ROLE_RESULT)

        if result is not None:
            color = COLOR_POS_RESULT if result else COLOR_NEG_RESULT
        elif is_pos:
            color = COLOR_POSITIVE
        else:
            color = COLOR_UNCLASSIFIED

        painter.fillRect(0, 0, BADGE_SIZE, BADGE_SIZE, color)
        painter.end()

        item.setIcon(QIcon(thumb))

    # -----------------------------------------------------------------------
    def _redraw_all_badges(self):
        lw = self.listWidget_filmstrip
        for i in range(lw.count()):
            self._redraw_badge(lw.item(i))

    # =======================================================================
    # Select / Unselect All
    # =======================================================================
    def _select_all(self):
        lw = self.listWidget_filmstrip
        for i in range(lw.count()):
            item = lw.item(i)
            item.setData(ROLE_POSITIVE, True)
            self._redraw_badge(item)
        self._update_positive_count()

    def _unselect_all(self):
        lw = self.listWidget_filmstrip
        for i in range(lw.count()):
            item = lw.item(i)
            item.setData(ROLE_POSITIVE, False)
            self._redraw_badge(item)
        self._update_positive_count()

    # =======================================================================
    # Filmstrip filter
    # =======================================================================
    def _apply_filmstrip_filter(self):
        mode = self.comboBox_filmstrip_filter.currentText()
        lw   = self.listWidget_filmstrip
        for i in range(lw.count()):
            item   = lw.item(i)
            is_pos = item.data(ROLE_POSITIVE)
            result = item.data(ROLE_RESULT)

            if mode == "All":
                item.setHidden(False)
            elif mode == "Labeled (Positive)":
                item.setHidden(not is_pos)
            elif mode == "Unlabeled":
                item.setHidden(is_pos)
            elif mode == "Classified Positive":
                item.setHidden(result is not True)
            elif mode == "Classified Negative":
                item.setHidden(result is not False)

    # =======================================================================
    # Positive frame count
    # =======================================================================
    def _update_positive_count(self):
        lw    = self.listWidget_filmstrip
        count = sum(
            1 for i in range(lw.count())
            if lw.item(i).data(ROLE_POSITIVE)
        )
        self.label_positive_count.setText(f"Positive frames selected: {count}")

    # =======================================================================
    # Classifier radio button toggle
    # =======================================================================
    def _on_classifier_toggled(self):
        use_rf = self.radioButton_rf.isChecked()
        # SVM params
        self.label_svm_c.setEnabled(not use_rf)
        self.doubleSpinBox_svm_c.setEnabled(not use_rf)
        self.label_svm_kernel.setEnabled(not use_rf)
        self.comboBox_svm_kernel.setEnabled(not use_rf)
        # RF params
        self.label_rf_estimators.setEnabled(use_rf)
        self.spinBox_rf_estimators.setEnabled(use_rf)
        self.label_rf_depth.setEnabled(use_rf)
        self.spinBox_rf_max_depth.setEnabled(use_rf)

    # =======================================================================
    # Workflow: Extract Features
    # =======================================================================
    def _on_extract_features(self):
        if not self._pairs:
            QMessageBox.warning(self, "Surface State Classifier",
                                "No image/mask pairs loaded.")
            return

        total = len(self._pairs)
        self._progress_wheel = QProgressWheel(
            title="Extracting features…",
            total=total,
        )

        self._extract_worker = FeatureExtractionWorker(self._pairs, parent=self)
        self._extract_worker.progress.connect(self._progress_wheel.setValue)
        self._extract_worker.finished.connect(self._on_features_ready)
        self._extract_worker.error.connect(self._on_worker_error)
        self._extract_worker.start()

    # -----------------------------------------------------------------------
    def _on_features_ready(self, df: pd.DataFrame):
        try:
            self._progress_wheel.close()
        except Exception:
            pass

        if df.empty:
            QMessageBox.warning(self, "Surface State Classifier",
                                "Feature extraction produced no results.")
            return

        self._feature_df = df

        # Auto-save to segmented images folder
        csv_path = self._feature_csv_path()
        if csv_path:
            try:
                df.to_csv(csv_path, index=False)
                print(f"[SSC] Features saved to: {csv_path}")
            except Exception as e:
                QMessageBox.warning(self, "Surface State Classifier",
                                    f"Features extracted but could not save CSV:\n{e}")

        print(f"[SSC] Feature extraction complete: {len(df)} frames, "
              f"{len(df.columns)} columns.")
        self._set_button_states()

    # -----------------------------------------------------------------------
    def _load_features_from_csv(self, csv_path: str):
        """Load a previously saved feature CSV into self._feature_df."""
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                QMessageBox.warning(self, "Surface State Classifier",
                                    "Feature CSV is empty.")
                return
            self._feature_df = df
            print(f"[SSC] Features loaded from: {csv_path} ({len(df)} frames)")
            self._set_button_states()
            QMessageBox.information(self, "Features Loaded",
                                    f"Loaded {len(df)} frames from:\n{os.path.basename(csv_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))

    # -----------------------------------------------------------------------
    def _on_load_features(self):
        """Manual load — opens file dialog defaulting to segmented images folder."""
        folder = self.lineEdit_segmented_images_folder.text().strip()
        start  = folder if folder and os.path.isdir(folder) else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Feature CSV", start,
            "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self._load_features_from_csv(path)

    # =======================================================================
    # Workflow: Train Classifier
    # =======================================================================
    def _on_train_classifier(self):
        if self._feature_df is None:
            QMessageBox.warning(self, "Surface State Classifier",
                                "Extract features first.")
            return

        lw            = self.listWidget_filmstrip
        positive_idxs = [
            lw.item(i).data(ROLE_INDEX)
            for i in range(lw.count())
            if lw.item(i).data(ROLE_POSITIVE)
        ]

        if len(positive_idxs) == 0:
            QMessageBox.warning(self, "Surface State Classifier",
                                "Select at least one positive example in the filmstrip.")
            return

        use_rf = self.radioButton_rf.isChecked()

        self._classify_worker = ClassifierWorker(
            feature_df      = self._feature_df,
            positive_indices = positive_idxs,
            use_rf          = use_rf,
            rf_trees        = self.spinBox_rf_estimators.value(),
            rf_max_depth    = self.spinBox_rf_max_depth.value(),
            svm_c           = self.doubleSpinBox_svm_c.value(),
            svm_kernel      = self.comboBox_svm_kernel.currentText(),
            parent          = self,
        )
        self._classify_worker.finished.connect(self._on_classifier_ready)
        self._classify_worker.error.connect(self._on_worker_error)
        self._classify_worker.start()

        QMessageBox.information(self, "Surface State Classifier",
                                "Training classifier…")

    # -----------------------------------------------------------------------
    def _on_classifier_ready(self, pipeline: dict, predictions: pd.Series):
        self._trained_pipeline = pipeline
        # Store predictions but don't classify all frames yet —
        # Classify All Frames applies them to the filmstrip.
        self._predictions = predictions
        print(f"[SSC] Classifier trained. "
              f"Positive predictions: {int(predictions.sum())} / {len(predictions)}")
        self._set_button_states()
        QMessageBox.information(self, "Surface State Classifier",
                                "Classifier trained successfully.\n"
                                "Click 'Classify All Frames' to apply.")

    # =======================================================================
    # Workflow: Classify All Frames
    # =======================================================================
    def _on_classify_frames(self):
        if self._predictions is None:
            QMessageBox.warning(self, "Surface State Classifier",
                                "Train the classifier first.")
            return

        lw = self.listWidget_filmstrip
        for i in range(lw.count()):
            item = lw.item(i)
            idx  = item.data(ROLE_INDEX)
            if idx < len(self._predictions):
                result = bool(self._predictions.iloc[idx])
                item.setData(ROLE_RESULT, result)
            self._redraw_badge(item)

        self._apply_filmstrip_filter()
        self._set_button_states()

        pos = int(self._predictions.sum())
        neg = len(self._predictions) - pos
        QMessageBox.information(
            self, "Classification Complete",
            f"Positive: {pos}  |  Negative: {neg}  |  Total: {len(self._predictions)}"
        )

    # =======================================================================
    # Workflow: Export Model
    # =======================================================================
    def _on_export_model(self):
        if self._trained_pipeline is None:
            QMessageBox.warning(self, "Surface State Classifier",
                                "No trained model to export.")
            return

        condition = self.lineEdit_surface_condition.text().strip() or "surface_state"
        default_name = f"ssc_{condition}.joblib"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Classifier Model",
            str(Path.home() / "Documents" / "GRIME-AI" / default_name),
            "Joblib Model (*.joblib);;All Files (*)"
        )
        if not path:
            return

        try:
            joblib.dump(self._trained_pipeline, path)
            QMessageBox.information(self, "Export Complete",
                                    f"Model saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    # =======================================================================
    # Workflow: Export Results CSV
    # =======================================================================
    def _on_export_results(self):
        if self._feature_df is None or self._predictions is None:
            QMessageBox.warning(self, "Surface State Classifier",
                                "No classification results to export.")
            return

        condition   = self.lineEdit_surface_condition.text().strip() or "surface_state"
        default_name = f"ssc_{condition}_results.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results CSV",
            str(Path.home() / "Documents" / "GRIME-AI" / default_name),
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            out_df = self._feature_df[["image_path", "mask_path"]].copy()
            out_df["surface_condition"]  = condition
            out_df["classified_positive"] = self._predictions.values.astype(bool)

            # Add labeled ground-truth column
            lw = self.listWidget_filmstrip
            labeled = [lw.item(i).data(ROLE_POSITIVE) for i in range(lw.count())]
            out_df["labeled_positive"] = labeled[: len(out_df)]

            out_df.to_csv(path, index=False)
            QMessageBox.information(self, "Export Complete",
                                    f"Results saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    # =======================================================================
    # Button state management
    # =======================================================================
    def _set_button_states(self):
        has_pairs      = bool(self._pairs)
        has_features   = self._feature_df is not None
        has_classifier = self._trained_pipeline is not None
        has_results    = self._predictions is not None

        self.pushButton_extract_features.setEnabled(has_pairs)
        self.pushButton_load_features.setEnabled(has_pairs)
        self.pushButton_train_classifier.setEnabled(has_features)
        self.pushButton_classify_frames.setEnabled(has_classifier)
        self.pushButton_export_model.setEnabled(has_classifier)
        self.pushButton_export_results.setEnabled(has_results)

    # =======================================================================
    # Error handler
    # =======================================================================
    def _on_worker_error(self, msg: str):
        try:
            if self._progress_wheel:
                self._progress_wheel.close()
        except Exception:
            pass
        QMessageBox.critical(self, "Surface State Classifier",
                             f"Worker error:\n{msg}")
