#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ROIAnalyzerTab.py
# Direct port of the ROI Analyzer functionality and signal wiring from ML_ImageProcessingDlg
# Author: John Edward Stranzl, Jr.
# License: Apache License, Version 2.0

import os
import re
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pathlib import Path
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtWidgets import QWidget, QFileDialog, QListWidgetItem, QMessageBox
from PyQt5.QtGui import QPixmap, QIcon, QImage, QPainter, QColor, QFont

from GRIME_AI import PROJECT_ROOT
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor
from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_CSS_Styles import BUTTON_CSS_STEEL_BLUE


class ROIAnalyzerTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # EXPECTED UI WIDGETS (assign before calling wire_connections()):
        # - self.lineEdit_ROI_images_folder
        # - self.pushButton_browse_ROI_images_folder
        # - self.pushButton_analyze
        # - self.pushButton_extract_ROI_features
        # - self.listWidget_filmstrip
        # - self.spinBox_numClusters
        # - self.label_displayImages
        # - self.lineEdit_intensity
        # - self.lineEdit_entropy
        # - self.lineEdit_Texture
        # - self.lineEdit_GLI
        # - self.lineEdit_GCC
        # - self.buttonBox_close  (optional; used to close in original dialog)

        self._pairs = []
        self._pendingThumbnails = []
        self._batchSize = 10
        self._batchDelay = 50
        self._loadToken = 0
        self.num_clusters = None

        self.roi_metrics_df = None
        self.related_csv_df = None
        self.aligned_df = None

        self._analyzer     = None
        self._capture_date = None
        self._capture_time = None
        self._raw_pixmaps  = []

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def configure_filmstrip(self):
        """
        Force the filmstrip to be exactly one thumbnail high,
        no wrapping, no extra margins or space.
        """
        lw = self.listWidget_filmstrip

        # No wrapping, no spacing
        lw.setWrapping(False)
        lw.setSpacing(0)

        # Remove margins
        lw.setContentsMargins(0, 0, 0, 0)
        lw.setViewportMargins(0, 0, 0, 0)

        # Remove frame
        from PyQt5.QtWidgets import QFrame
        lw.setFrameShape(QFrame.NoFrame)

        # Fix the height to one icon row plus the index label under it
        icon_h = lw.iconSize().height()
        lw.setFixedHeight(icon_h + 16)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _update_texture_fields(self, image):
        """Compute and display GLCM metrics if the GLCM checkbox is checked."""
        self.lineEdit_glcm_contrast.clear()
        self.lineEdit_glcm_homogeneity.clear()
        self.lineEdit_glcm_correlation.clear()

        if self.checkBox_texture_GLCM.isChecked():
            try:
                from GRIME_AI.GRIME_AI_Texture import GLCMTexture
                features = GLCMTexture().compute_features(image)
                self.lineEdit_glcm_contrast.setText(f"{features['contrast']:.4f}")
                self.lineEdit_glcm_homogeneity.setText(f"{features['homogeneity']:.4f}")
                self.lineEdit_glcm_correlation.setText(f"{features['correlation']:.4f}")
            except Exception as e:
                print(f"[Texture] GLCM computation failed: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------------------------------------------------------
    def wire_connections(self):
        # Folder text field
        self.lineEdit_ROI_images_folder.editingFinished.connect(self._on_roi_images_folder_changed)

        # Browse button
        self.pushButton_browse_ROI_images_folder.clicked.connect(self.browse_ROI_images_folder)
        self.pushButton_browse_ROI_images_folder.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        # Analyze button
        self.pushButton_analyze.clicked.connect(self.analyze_roi)
        self.pushButton_analyze.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        # Extract features button
        self.pushButton_extract_ROI_features.clicked.connect(self.extract_ROI_features)
        self.pushButton_extract_ROI_features.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        # Filmstrip click
        self.listWidget_filmstrip.itemClicked.connect(self.on_filmstrip_item_clicked)

        # Clustering method radio buttons
        self.radioButton_kmeans.toggled.connect(self._on_clustering_method_changed)
        self.radioButton_gmm.toggled.connect(self._on_clustering_method_changed)
        self.radioButton_meanshift.toggled.connect(self._on_clustering_method_changed)

        # Cluster count spin boxes
        self.num_clusters = self.spinBox_numClusters.value()
        self.spinBox_numClusters.valueChanged.connect(self._on_num_clusters_changed)
        self.spinBox_numClusters_gmm.valueChanged.connect(self._on_num_clusters_changed)

        # Mean-shift: auto-estimate checkbox toggles quantile/bandwidth fields
        self.checkBox_autoEstimate.toggled.connect(self._on_auto_estimate_toggled)

        # Mean-shift: re-run when quantile or bandwidth changes
        self.doubleSpinBox_quantile.valueChanged.connect(self._on_meanshift_param_changed)
        self.doubleSpinBox_bandwidth.valueChanged.connect(self._on_meanshift_param_changed)

        # Button fonts
        self.pushButton_browse_ROI_images_folder.setFont(QFont("Arial", 11))
        self.pushButton_analyze.setFont(QFont("Arial", 11, QFont.Bold))
        self.pushButton_extract_ROI_features.setFont(QFont("Arial", 11, QFont.Bold))

        # Optional close hook (matches original)
        if hasattr(self, "buttonBox_close") and self.buttonBox_close is not None:
            self.buttonBox_close.rejected.connect(self.reject)

        # Texture Analysis checkbox defaults
        self.checkBox_texture_GLCM.setChecked(True)
        self.checkBox_texture_Gabor.setChecked(False)
        self.checkBox_texture_LBP.setChecked(False)
        self.checkBox_texture_Wavelet.setChecked(False)
        self.checkBox_texture_Fourier.setChecked(False)

        # Splitter stretch factors and image label stretch
        self.splitter_display.setStretchFactor(0, 4)
        self.splitter_display.setStretchFactor(1, 1)
        self.horizontalLayoutImages.setStretch(0, 1)
        self.horizontalLayoutImages.setStretch(1, 1)
        self.horizontalLayoutImages.setStretch(2, 1)

        # Populate folder from config (matches original behavior)
        roi_analyzer_image_folder = JsonEditor().getValue("ROI_Analyzer_Images_Folder")
        if roi_analyzer_image_folder:
            self.lineEdit_ROI_images_folder.setText(roi_analyzer_image_folder)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        if self._pairs and self.listWidget_filmstrip.count() > 0:
            if self._analyzer is None:
                self.listWidget_filmstrip.setCurrentRow(0)
                self.on_filmstrip_item_clicked(self.listWidget_filmstrip.item(0))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_images()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _refresh_images(self, *args):
        if not self._raw_pixmaps:
            return
        labels = [self.label_image_original,
                  self.label_image_features,
                  self.label_image_overlay]
        for label, pix in zip(labels, self._raw_pixmaps):
            w, h = label.width(), label.height()
            if w > 0 and h > 0:
                label.setPixmap(pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _display_analysis(self, analyzer, capture_date=None, capture_time=None):
        self._analyzer     = analyzer
        self._capture_date = capture_date
        self._capture_time = capture_time

        def np_to_pixmap(arr):
            c = arr if arr.flags['C_CONTIGUOUS'] else arr.copy()
            h, w = c.shape[:2]
            bpl = w * c.shape[2]
            fmt = QImage.Format_RGBA8888 if c.shape[2] == 4 else QImage.Format_RGB888
            return QPixmap.fromImage(QImage(c.data, w, h, bpl, fmt))

        orig_rgb       = cv2.cvtColor(analyzer.image,     cv2.COLOR_BGR2RGB)
        composite_rgba = cv2.cvtColor(analyzer.composite, cv2.COLOR_BGRA2RGBA)
        pix_orig     = np_to_pixmap(orig_rgb)
        pix_features = np_to_pixmap(composite_rgba)

        base, _ = os.path.splitext(analyzer.image_filename)
        pix_overlay = None
        for ext in ('.png', '.jpg', '.jpeg'):
            candidate = f"{base}_overlay{ext}"
            if os.path.exists(candidate):
                ov = cv2.imread(candidate)
                pix_overlay = np_to_pixmap(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
                break

        if pix_overlay:
            self._raw_pixmaps = [pix_orig, pix_features, pix_overlay]
            self.label_image_overlay.setVisible(True)
        else:
            self._raw_pixmaps = [pix_orig, pix_features]
            self.label_image_overlay.setVisible(False)

        self._refresh_images()
        QTimer.singleShot(50, self._refresh_images)
        self._set_swatch_pixmap()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _set_swatch_pixmap(self):
        if not self._analyzer or not self._analyzer.dominant_rgb_list:
            return
        W, H = 900, 60
        canvas = QPixmap(W, H)
        painter = QPainter(canvas)
        painter.setFont(QFont("Arial", 9))
        rgb_list = self._analyzer.dominant_rgb_list
        pct_list = self._analyzer.percentages_list
        x, n = 0, len(rgb_list)
        for i, (rgb, pct) in enumerate(zip(rgb_list, pct_list)):
            seg_w = int(round(W * pct / 100.0))
            if i == n - 1:
                seg_w = W - x
            seg_w = max(seg_w, 1)
            painter.fillRect(x, 0, seg_w, H, QColor(*rgb))
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            painter.setPen(QColor(255, 255, 255) if lum < 128 else QColor(0, 0, 0))
            painter.drawText(QRect(x, 0, seg_w, H), Qt.AlignCenter,
                             f"{rgb}\n{pct:.1f}%")
            x += seg_w
        painter.end()
        self.label_swatches.setPixmap(canvas)


    # ******************************************************************************************************************
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _get_clustering_method(self):
        """Return the currently selected clustering method string."""
        if self.radioButton_gmm.isChecked():
            return 'gmm'
        if self.radioButton_meanshift.isChecked():
            return 'meanshift'
        return 'kmeans'

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _on_clustering_method_changed(self):
        """Enable/disable per-method parameter widgets; re-run analysis on current image."""
        is_kmeans = self.radioButton_kmeans.isChecked()
        is_gmm = self.radioButton_gmm.isChecked()
        is_meanshift = self.radioButton_meanshift.isChecked()

        # K-Means controls
        self.label_numClusters.setEnabled(is_kmeans)
        self.spinBox_numClusters.setEnabled(is_kmeans)

        # GMM controls
        self.label_numClusters_gmm.setEnabled(is_gmm)
        self.spinBox_numClusters_gmm.setEnabled(is_gmm)

        # Mean-shift controls
        self.checkBox_autoEstimate.setEnabled(is_meanshift)
        auto = self.checkBox_autoEstimate.isChecked()
        self.label_quantile.setEnabled(is_meanshift and auto)
        self.doubleSpinBox_quantile.setEnabled(is_meanshift and auto)
        self.label_bandwidth.setEnabled(is_meanshift and not auto)
        self.doubleSpinBox_bandwidth.setEnabled(is_meanshift and not auto)

        current = self.listWidget_filmstrip.currentItem()
        if current:
            self.on_filmstrip_item_clicked(current)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _on_auto_estimate_toggled(self, checked):
        """Toggle quantile vs manual bandwidth fields; re-run analysis."""
        self.label_quantile.setEnabled(checked)
        self.doubleSpinBox_quantile.setEnabled(checked)
        self.label_bandwidth.setEnabled(not checked)
        self.doubleSpinBox_bandwidth.setEnabled(not checked)

        current = self.listWidget_filmstrip.currentItem()
        if current:
            self.on_filmstrip_item_clicked(current)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _on_meanshift_param_changed(self):
        """Re-run analysis when quantile or bandwidth value changes."""
        if not self.radioButton_meanshift.isChecked():
            return
        current = self.listWidget_filmstrip.currentItem()
        if current:
            self.on_filmstrip_item_clicked(current)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _get_n_clusters(self):
        """Return cluster count for the active method."""
        if self.radioButton_gmm.isChecked():
            return self.spinBox_numClusters_gmm.value()
        return self.spinBox_numClusters.value()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _get_meanshift_kwargs(self):
        """Return bandwidth kwargs dict for Mean-shift instantiation."""
        if self.checkBox_autoEstimate.isChecked():
            return {'bandwidth': None, 'quantile': self.doubleSpinBox_quantile.value()}
        return {'bandwidth': self.doubleSpinBox_bandwidth.value(), 'quantile': None}

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _extract_datetime_from_path(self, filepath):
        """
        Try multiple datetime formats in filenames.
        Returns (date_str, time_str) as 'YYYY-MM-DD' / 'HH:MM:SS', or 'n/a' for each part not found.
        """
        stem = os.path.splitext(os.path.basename(filepath))[0]
        patterns = [
            (r'(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})Z',
             lambda m: (f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
                        f"{m.group(4)}:{m.group(5)}:{m.group(6)}")),
            (r'(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})(\d{2})',
             lambda m: (f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
                        f"{m.group(4)}:{m.group(5)}:{m.group(6)}")),
            (r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',
             lambda m: (f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
                        f"{m.group(4)}:{m.group(5)}:{m.group(6)}")),
            (r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})',
             lambda m: (f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
                        f"{m.group(4)}:{m.group(5)}:{m.group(6)}")),
            (r'(\d{4})_(\d{2})_(\d{2})',
             lambda m: (f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "n/a")),
            (r'(\d{4})(\d{2})(\d{2})',
             lambda m: (f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "n/a")),
        ]
        for pattern, extractor in patterns:
            m = re.search(pattern, stem)
            if m:
                try:
                    return extractor(m)
                except Exception:
                    continue
        return "n/a", "n/a"

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _on_num_clusters_changed(self, value):
        self.num_clusters = value

        current = self.listWidget_filmstrip.currentItem()
        if current:
            # re-run analysis on the highlighted image
            self.on_filmstrip_item_clicked(current)
        else:
            # fallback: rerun the initial analysis sequence
            self.analyze_roi()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def on_filmstrip_item_clicked(self, item: QListWidgetItem):
        """
        When a thumbnail is clicked, re-run ROI analysis on that image/mask pair
        and update label_displayImages with the composite plot + top-3 swatches.
        """
        # Retrieve the index we stored in populate_filmstrip
        idx = item.data(Qt.UserRole)
        orig_path, mask_path = self._pairs[idx]

        # grab the user-selected cluster count
        n_clusters = self._get_n_clusters()

        # Run analysis for this specific pair
        from GRIME_AI.GRIME_AI_ROI_Analyzer import GRIME_AI_ROI_Analyzer
        analyzer = GRIME_AI_ROI_Analyzer(orig_path, mask_path, clusters=n_clusters, clustering_method=self._get_clustering_method(), **self._get_meanshift_kwargs())
        analyzer.run_analysis()

        # ─── populate metric fields ───
        self.lineEdit_intensity.setText(f"{analyzer.roi_intensity:.2f}")
        self.lineEdit_entropy.setText(f"{analyzer.roi_entropy:.4f}")
        self.lineEdit_Texture.setText(f"{analyzer.roi_texture:.4f}")
        self.lineEdit_GLI.setText(f"{analyzer.mean_gli:.6f}")
        self.lineEdit_GCC.setText(f"{analyzer.mean_gcc:.6f}")

        import cv2 as _cv2
        _img = _cv2.imread(str(orig_path))
        if _img is not None:
            self._update_texture_fields(_img)

        capture_date, capture_time = self._extract_datetime_from_path(orig_path)
        self._display_analysis(analyzer, capture_date, capture_time)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def populate_filmstrip(self, image_paths):
        """
        1) Clear old thumbnails.
        2) Enqueue placeholders + paths.
        3) Kick off _loadNextBatch via QTimer.
        """
        lw = self.listWidget_filmstrip
        lw.clear()

        # Invalidate any previous loader
        self._loadToken += 1
        token = self._loadToken
        self._pendingThumbnails.clear()

        # Create one blank item per image path
        iconSize = lw.iconSize()
        from PyQt5.QtCore import QSize
        label_h = 16  # room for the index label under each thumbnail
        for idx, path in enumerate(image_paths):
            # 1-based index label: matches the data-row numbering in the
            # metrics/report spreadsheets so chart anomalies map directly
            # to filmstrip positions.
            item = QListWidgetItem(QIcon(), str(idx + 1))
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
            item.setToolTip(f"#{idx + 1} - {os.path.basename(path)}")
            item.setData(Qt.UserRole, idx)
            item.setSizeHint(QSize(iconSize.width(), iconSize.height() + label_h))
            lw.addItem(item)
            self._pendingThumbnails.append((item, path, token))

        # Highlight first by default
        if lw.count():
            lw.setCurrentRow(0)

        # Schedule the first batch load
        QTimer.singleShot(self._batchDelay, lambda: self._loadNextBatch(token))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _loadNextBatch(self, token):
        """
        Pop up to self._batchSize thumbnails from self._pendingThumbnails,
        assign icons, then reschedule if more remain.
        """
        # Cancel if stale
        if token != self._loadToken:
            return

        lw = self.listWidget_filmstrip
        iconSize = lw.iconSize()

        for _ in range(min(self._batchSize, len(self._pendingThumbnails))):
            item, path, _ = self._pendingThumbnails.pop(0)
            if not os.path.exists(path):
                continue
            pix = QPixmap(path)
            if pix.isNull():
                continue

            thumb = pix.scaled(
                iconSize,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            item.setIcon(QIcon(thumb))

        # More to do?
        if self._pendingThumbnails:
            QTimer.singleShot(self._batchDelay, lambda: self._loadNextBatch(token))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def browse_ROI_images_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", str(PROJECT_ROOT))
        if not folder:
            return
        self.lineEdit_ROI_images_folder.setText(folder)

        self._on_roi_images_folder_changed()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _on_roi_images_folder_changed(self):
        folder = self.lineEdit_ROI_images_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            return

        JsonEditor().update_json_entry("ROI_Analyzer_Images_Folder", folder)

        from GRIME_AI.GRIME_AI_ROI_Analyzer import GRIME_AI_ROI_Analyzer
        temp = GRIME_AI_ROI_Analyzer("", "")
        pairs = temp.generate_file_pairs(folder)
        if not pairs:
            return

        self._pairs = pairs
        self.populate_filmstrip([orig for orig, _ in pairs])

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def analyze_roi(self):
        """
        1) Generate file pairs and populate filmstrip
        2) Run analysis on the first (or clicked) pair
        3) Display composite plot, fallback to original if empty
        4) Overlay top-3 color swatches
        """
        folder = self.lineEdit_ROI_images_folder.text().strip()
        if not folder:
            QMessageBox.warning(self, "ROI Analyzer", "Please specify a folder path.")
            return

        try:
            from GRIME_AI.GRIME_AI_ROI_Analyzer import GRIME_AI_ROI_Analyzer
        except ImportError:
            QMessageBox.warning(self, "ROI Analyzer", "Unable to import ROI Analyzer module.")
            return

        # 1) generate pairs + batched filmstrip population
        temp = GRIME_AI_ROI_Analyzer("", "")
        pairs = temp.generate_file_pairs(folder)
        if not pairs:
            QMessageBox.warning(self, "ROI Analyzer", "No image/mask pairs found.")
            return
        self._pairs = pairs

        # Replace inline loop with batched loader
        image_paths = [orig for orig, _ in pairs]
        self.populate_filmstrip(image_paths)

        # 2) analyze the first pair (index 0) by default
        orig_path, mask_path = pairs[0]
        n_clusters = self._get_n_clusters()
        analyzer = GRIME_AI_ROI_Analyzer(orig_path, mask_path, clusters=n_clusters, clustering_method=self._get_clustering_method(), **self._get_meanshift_kwargs())
        analyzer.run_analysis()

        # ─── populate metric fields ───
        self.lineEdit_intensity.setText(f"{analyzer.roi_intensity:.2f}")
        self.lineEdit_entropy.setText(f"{analyzer.roi_entropy:.4f}")
        self.lineEdit_Texture.setText(f"{analyzer.roi_texture:.4f}")
        self.lineEdit_GLI.setText(f"{analyzer.mean_gli:.6f}")
        self.lineEdit_GCC.setText(f"{analyzer.mean_gcc:.6f}")

        import cv2 as _cv2
        _img = _cv2.imread(str(orig_path))
        if _img is not None:
            self._update_texture_fields(_img)

        capture_date, capture_time = self._extract_datetime_from_path(orig_path)
        self._display_analysis(analyzer, capture_date, capture_time)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def analyze_responses(self, df, predictors, responses, output_folder):
        """
        Analyze correlations and regressions between predictors and responses,
        with numeric coercion and plot skipping for insufficient data.
        Saves plots and correlation tables to the specified output folder.
        """
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Coerce predictors and responses to numeric ---
        df1 = df.copy()
        df1[predictors + responses] = df1[predictors + responses].apply(pd.to_numeric, errors='coerce')

        # --- 2. Distribution plots ---
        for col in predictors + responses:
            plt.figure()
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(output_dir / f"hist_{col}.png")
            plt.close()

        # --- 3. Correlations ---
        pearson_results = {}
        spearman_results = {}
        for resp in responses:
            pearson_results[resp] = df1[predictors + [resp]].corr(method='pearson')[resp].drop(resp)
            spearman_results[resp] = df1[predictors + [resp]].corr(method='spearman')[resp].drop(resp)

        pearson_df = pd.DataFrame(pearson_results)
        spearman_df = pd.DataFrame(spearman_results)

        # --- 4. Scatterplots with regression lines ---
        for resp in responses:
            for pred in predictors:
                x = df1[pred]
                y = df1[resp]
                valid = x.notna() & y.notna()
                if valid.sum() < 2:
                    print(f"Skipping plot for {pred} vs {resp}: insufficient numeric data")
                    continue

                plt.figure()
                sns.regplot(x=x[valid], y=y[valid], scatter_kws={'alpha': 0.5})
                plt.xlabel(pred)
                plt.ylabel(resp)
                plt.title(f"{pred} vs {resp}")
                plt.tight_layout()
                plt.savefig(output_dir / f"scatter_{pred}_vs_{resp}.png")
                plt.close()

        # --- 5. Multiple linear regression models ---
        model_summaries = {}
        for resp in responses:
            X = df1[predictors]
            y = df1[resp]
            valid = X.notna().all(axis=1) & y.notna()
            if valid.sum() < 2:
                print(f"Skipping regression for {resp}: insufficient numeric data")
                continue
            X_valid = sm.add_constant(X[valid])
            y_valid = y[valid]
            model = sm.OLS(y_valid, X_valid).fit()
            model_summaries[resp] = model.summary().as_text()
            print(f"\n=== Regression for {resp} ===")
            print(model.summary())

        # --- 6. Correlation heatmap ---
        plt.figure(figsize=(10, 8))
        corr_matrix = df1[predictors + responses].corr(method='pearson')
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap (Pearson)")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png")
        plt.close()

        # --- 7. Save correlation tables and model summaries to Excel ---
        excel_path = output_dir / "analysis_results.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            pearson_df.to_excel(writer, sheet_name="Pearson_Corr")
            spearman_df.to_excel(writer, sheet_name="Spearman_Corr")
            summary_df = pd.DataFrame.from_dict(model_summaries, orient='index', columns=['Summary'])
            summary_df.to_excel(writer, sheet_name="Model_Summaries")

        print(f"\nAnalysis complete. Results saved to {output_dir}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def extract_ROI_features(self):
        # GET SETTINGS FROM UI CONTROLS
        n_clusters = self._get_n_clusters()

        # Legacy pattern still used for output file prefix naming
        dt_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2}-\d{2}-\d{2})Z")

        # 1) Get and validate output folder path
        output_folder = self.lineEdit_ROI_images_folder.text().strip()
        if not output_folder:
            QMessageBox.warning(self, "No Output Folder", "Please specify an output folder.")
            return

        os.makedirs(output_folder, exist_ok=True)

        # Ensure we have image/mask pairs
        if not getattr(self, "_pairs", None):
            QMessageBox.warning(self, "No Image Pairs", "No (image, mask) pairs found.")
            return

        # Resolve first image path and image_folder early
        first_img_path = self._pairs[0][0]
        image_folder = os.path.dirname(first_img_path)

        # Optional: correlate NWIS sensor data into the feature file.
        # Cancel simply skips correlation and extraction proceeds as before.
        sensor_file, _sf_filter = QFileDialog.getOpenFileName(
            self,
            "Correlate sensor data (optional - press Cancel to extract without it)",
            image_folder,
            "NWIS sensor data (*.txt *.csv);;All files (*.*)")

        # Filename prefix is the run date/time, per the GRIME AI convention
        # used by all analysis outputs (e.g. ImageTriage_YYYYMMDD_HHMMSS).
        from datetime import datetime as _dt
        file_dt_prefix = _dt.now().strftime("%Y%m%d_%H%M%S")

        # All correlation outputs (csv, xlsx, pngs) go into a 'correlation'
        # subfolder so they don't pollute the image/data folder.
        correlation_folder = os.path.join(output_folder, "correlation")
        os.makedirs(correlation_folder, exist_ok=True)

        # Output file paths with datetime prefix
        csv_path = os.path.join(correlation_folder, f"{file_dt_prefix}_roi_metrics.csv")
        xlsx_path = os.path.join(correlation_folder, f"{file_dt_prefix}_roi_metrics.xlsx")
        aligned_xlsx_path = os.path.join(correlation_folder, f"{file_dt_prefix}_aligned_results.xlsx")

        # 2) Prepare header and container for all rows
        clustering_method = self._get_clustering_method()
        meanshift_kwargs = self._get_meanshift_kwargs()

        # Build a human-readable params string for the header
        if clustering_method == 'kmeans':
            clustering_params = f"clusters={n_clusters}"
        elif clustering_method == 'gmm':
            clustering_params = f"clusters={n_clusters}"
        else:  # meanshift
            if meanshift_kwargs['bandwidth'] is None:
                clustering_params = f"auto=True, quantile={meanshift_kwargs['quantile']}"
            else:
                clustering_params = f"auto=False, bandwidth={meanshift_kwargs['bandwidth']}"

        header = [
            "Image Path",
            "Mask Path",
            "Capture Date",
            "Capture Time",
            "Clustering Method",
            "Clustering Params",
            "ROI Intensity",
            "ROI Entropy",
            "ROI Texture",
            "Mean GLI",
            "Mean GCC",
            "GLCM Contrast",
            "GLCM Homogeneity",
            "GLCM Correlation",
            "ROI Pixel Count",
            "ROI Area",
            "Image Height",
            "Image Width",
            "Image Total Pixels",
            "ROI Area Percentage",
        ]

        # Add HSV columns for each cluster
        for i in range(1, n_clusters + 1):
            header.extend([
                f"Cluster {i} H",
                f"Cluster {i} S",
                f"Cluster {i} V",
            ])

        # (Optional) Add RGB columns for each cluster
        for i in range(1, n_clusters + 1):
            header.extend([
                f"Cluster {i} R",
                f"Cluster {i} G",
                f"Cluster {i} B",
            ])

        rows = [header]

        # --- Find related CSV file (CSV has " - ", JPGs don't). Prefix is JPG stem before the datetime stamp. ---
        first_image_stem = os.path.splitext(os.path.basename(first_img_path))[0]
        match_dt = dt_pattern.search(first_image_stem)
        if match_dt:
            common_prefix = first_image_stem[:match_dt.start()].rstrip("_ ")
        else:
            common_prefix = first_image_stem

        matching_csv_path = None
        try:
            for fname in os.listdir(image_folder):
                # Case-insensitive check; CSV must start with "<common_prefix> - "
                if fname.lower().endswith(".csv") and fname.lower().startswith((common_prefix + " - ").lower()):
                    matching_csv_path = os.path.join(image_folder, fname)
                    break
        except Exception as e:
            print(f"Error listing directory {image_folder}: {e}")
            matching_csv_path = None

        if matching_csv_path:
            try:
                self.related_csv_df = pd.read_csv(matching_csv_path)

                # --- Split 'datetime' column into Date and Time (24-hour) ---
                if 'datetime' in self.related_csv_df.columns:
                    self.related_csv_df['datetime'] = pd.to_datetime(
                        self.related_csv_df['datetime'],
                        format="%m/%d/%Y %H:%M",
                        errors="coerce"
                    )
                    self.related_csv_df['Date'] = self.related_csv_df['datetime'].dt.strftime("%Y-%m-%d")
                    self.related_csv_df['Time'] = self.related_csv_df['datetime'].dt.strftime("%H:%M:%S")

            except Exception as e:
                print(f"Error reading related CSV {matching_csv_path}: {e}")
                self.related_csv_df = None
        else:
            self.related_csv_df = None

        progress_bar_closed = False
        total_iterations = len(self._pairs)
        progressBar = QProgressWheel(
            title="Extracting features in-progress...",
            total=total_iterations,
            on_close=lambda: setattr(self, "progress_bar_closed", True)
        )
        # Keep the wheel visible above the main window.
        # NOTE: QProgressWheel shows itself in its constructor, and changing
        # window flags on a shown window hides it - so show() again after.
        progressBar.setWindowFlags(progressBar.windowFlags() | Qt.WindowStaysOnTopHint)
        progressBar.show()

        # 3) Iterate pairs, run analysis, collect results
        try:
            for i, (orig_path, mask_path) in enumerate(self._pairs):
                progressBar.setValue(i)

                # Extract datetime from image filename
                filename = os.path.basename(orig_path)
                capture_date, capture_time = self._extract_datetime_from_path(orig_path)

                from GRIME_AI.GRIME_AI_ROI_Analyzer import GRIME_AI_ROI_Analyzer
                analyzer = GRIME_AI_ROI_Analyzer(orig_path, mask_path, clusters=n_clusters, clustering_method=self._get_clustering_method(), **self._get_meanshift_kwargs())

                try:
                    analyzer.run_analysis()
                except Exception as e:
                    print(f"Failed on {orig_path}, {mask_path}: {e}")
                    continue

                # Compute GLCM if checked
                glcm_contrast = glcm_homogeneity = glcm_correlation = ""
                if self.checkBox_texture_GLCM.isChecked():
                    try:
                        import cv2 as _cv2
                        from GRIME_AI.GRIME_AI_Texture import GLCMTexture
                        _img = _cv2.imread(str(orig_path))
                        if _img is not None:
                            glcm_features = GLCMTexture().compute_features(_img)
                            glcm_contrast    = f"{glcm_features['contrast']:.4f}"
                            glcm_homogeneity = f"{glcm_features['homogeneity']:.4f}"
                            glcm_correlation = f"{glcm_features['correlation']:.4f}"
                    except Exception as e:
                        print(f"[Texture] GLCM export failed for {orig_path}: {e}")

                # Build row: fixed ROI metrics first
                data_row = [
                    orig_path,
                    mask_path,
                    capture_date,
                    capture_time,
                    clustering_method,
                    clustering_params,
                    f"{analyzer.roi_intensity:.2f}",
                    f"{analyzer.roi_entropy:.4f}",
                    f"{analyzer.roi_texture:.4f}",
                    f"{analyzer.mean_gli:.6f}",
                    f"{analyzer.mean_gcc:.6f}",
                    glcm_contrast,
                    glcm_homogeneity,
                    glcm_correlation,
                    f"{analyzer.ROI_total_pixels:.2f}",
                    f"{analyzer.ROI_total_area:.2f}",
                    f"{analyzer.image_height:.2f}",
                    f"{analyzer.image_width:.2f}",
                    f"{analyzer.image_total_pixels:.2f}",
                    f"{analyzer.ROI_percentage:.2f}",
                ]

                # Append HSV values for each cluster
                for (h, s, v) in analyzer.dominant_hsv_list:
                    data_row.extend([f"{h:.4f}", f"{s:.4f}", f"{v:.4f}"])

                # Append RGB values for each cluster
                for (r, g, b) in analyzer.dominant_rgb_list:
                    data_row.extend([f"{r:.4f}", f"{g:.4f}", f"{b:.4f}"])

                rows.append(data_row)

        finally:
            try:
                progressBar.close()
            except Exception:
                pass
            del progressBar

        # 3.5) Convert to DataFrame for in-memory use
        df = pd.DataFrame(rows[1:], columns=header)
        self.roi_metrics_df = df  # Keep in memory for subsequent steps

        # Correlate the selected sensor file (if any) and append its values.
        # Failure here degrades gracefully to an uncorrelated feature file.
        sensor_df = None
        if sensor_file:
            try:
                from GRIME_AI.GRIME_AI_SensorImageCorrelator import GRIME_AI_SensorImageCorrelator
                sensor_df = GRIME_AI_SensorImageCorrelator().sensor_values_for_images(
                    df["Image Path"].tolist(), sensor_file)
                df = pd.concat([df.reset_index(drop=True),
                                sensor_df.reset_index(drop=True)], axis=1)
                header = header + list(sensor_df.columns)
                self.roi_metrics_df = df
            except Exception as e:
                sensor_df = None
                QMessageBox.warning(
                    self, "Sensor Correlation",
                    f"Could not correlate sensor data - features will be "
                    f"exported without it:\n{e}")

        # Print first 5 rows of each DataFrame
        print("\nFirst 5 rows of ROI Metrics DataFrame:")
        print(self.roi_metrics_df.head())
        if self.related_csv_df is not None:
            print("\nFirst 5 rows of Related CSV DataFrame:")
            print(self.related_csv_df.head())
        else:
            print("\nNo related CSV DataFrame to display.")

        # 4) Write out CSV of ROI metrics
        try:
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error writing CSV to {csv_path}: {e}")

        # 5) Write out XLSX with hyperlinks for ROI metrics
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font

            wb = Workbook()
            ws = wb.active
            ws.title = "ROI Metrics"

            # Write header row
            for col_idx, title in enumerate(header, start=1):
                ws.cell(row=1, column=col_idx, value=title)

            hyperlink_style = Font(color="0000FF", underline="single")

            for row_idx, data in enumerate(rows[1:], start=2):
                (
                    orig_full, mask_full, capture_date, capture_time,
                    method_name, method_params,
                    intensity, entropy, texture, gli, gcc,
                    glcm_contrast, glcm_homogeneity, glcm_correlation,
                    pixel_count, pixel_area, image_height,
                    image_width, image_total_pixels, roi_area_percentage,
                    *cluster_values
                ) = data

                img_name = os.path.basename(orig_full)
                cell_img = ws.cell(row=row_idx, column=1, value=img_name)
                cell_img.hyperlink = orig_full
                cell_img.font = hyperlink_style

                mask_name = os.path.basename(mask_full)
                cell_mask = ws.cell(row=row_idx, column=2, value=mask_name)
                cell_mask.hyperlink = mask_full
                cell_mask.font = hyperlink_style

                ws.cell(row=row_idx, column=3,  value=capture_date)
                ws.cell(row=row_idx, column=4,  value=capture_time)
                ws.cell(row=row_idx, column=5,  value=method_name)
                ws.cell(row=row_idx, column=6,  value=method_params)
                ws.cell(row=row_idx, column=7,  value=float(intensity))
                ws.cell(row=row_idx, column=8,  value=float(entropy))
                ws.cell(row=row_idx, column=9,  value=float(texture))
                ws.cell(row=row_idx, column=10, value=float(gli))
                ws.cell(row=row_idx, column=11, value=float(gcc))
                ws.cell(row=row_idx, column=12, value=float(glcm_contrast)    if glcm_contrast    != "" else "")
                ws.cell(row=row_idx, column=13, value=float(glcm_homogeneity) if glcm_homogeneity != "" else "")
                ws.cell(row=row_idx, column=14, value=float(glcm_correlation) if glcm_correlation != "" else "")
                ws.cell(row=row_idx, column=15, value=float(pixel_count))
                ws.cell(row=row_idx, column=16, value=float(pixel_area))
                ws.cell(row=row_idx, column=17, value=float(image_height))
                ws.cell(row=row_idx, column=18, value=float(image_width))
                ws.cell(row=row_idx, column=19, value=float(image_total_pixels))
                ws.cell(row=row_idx, column=20, value=float(roi_area_percentage))

                for offset, value in enumerate(cluster_values, start=21):
                    ws.cell(row=row_idx, column=offset, value=float(value))

                if sensor_df is not None:
                    sensor_base = 21 + len(cluster_values)
                    for s_off, s_col in enumerate(sensor_df.columns):
                        raw = sensor_df.iloc[row_idx - 2][s_col]
                        try:
                            cell_value = float(raw)
                        except (TypeError, ValueError):
                            cell_value = "" if raw is None else raw
                        ws.cell(row=row_idx, column=sensor_base + s_off, value=cell_value)

            wb.save(xlsx_path)

        except ImportError:
            QMessageBox.warning(
                self,
                "XLSX Export Skipped",
                "The 'openpyxl' library is not installed. Install it via 'pip install openpyxl' to enable XLSX export."
            )
        except Exception as e:
            print(f"Error writing XLSX to {xlsx_path}: {e}")

        # Sensor-vs-ROI analysis report: per-parameter chart tabs + assessment
        analysis_path = None
        if sensor_df is not None:
            try:
                from GRIME_AI.GRIME_AI_SensorROIAnalysis import GRIME_AI_SensorROIAnalysis
                analysis_path = os.path.join(correlation_folder, f"{file_dt_prefix}_sensor_analysis.xlsx")
                png_dir = os.path.join(correlation_folder, f"{file_dt_prefix}_figures")
                GRIME_AI_SensorROIAnalysis().write_report(df, analysis_path, png_dir=png_dir)
            except Exception as e:
                analysis_path = None
                print(f"Error writing sensor analysis report: {e}")

        # 6) Align ROI metrics with related CSV by nearest timestamp and export aligned results
        aligned_written = False
        aligned_df = []

        try:
            if self.related_csv_df is not None and not self.related_csv_df.empty:
                roi_dt = pd.to_datetime(
                    self.roi_metrics_df["Capture Date"].astype(str).str.strip() + " " +
                    self.roi_metrics_df["Capture Time"].astype(str).str.strip(),
                    errors="coerce",
                    format="%Y-%m-%d %H:%M:%S"
                )
                roi_df = self.roi_metrics_df.copy()
                roi_df["datetime"] = roi_dt

                csv_df = self.related_csv_df.copy()
                if {"Date", "Time"}.issubset(csv_df.columns):
                    csv_dt = pd.to_datetime(
                        csv_df["Date"].astype(str).str.strip() + " " +
                        csv_df["Time"].astype(str).str.strip(),
                        errors="coerce",
                        format="%Y-%m-%d %H:%M:%S"
                    )
                    csv_df["datetime"] = csv_dt
                elif "datetime" in csv_df.columns:
                    csv_df["datetime"] = pd.to_datetime(csv_df["datetime"], errors="coerce")
                else:
                    csv_df["datetime"] = pd.NaT

                roi_df = roi_df.dropna(subset=["datetime"]).sort_values("datetime")
                csv_df = csv_df.dropna(subset=["datetime"]).sort_values("datetime")

                if not roi_df.empty and not csv_df.empty:
                    aligned_df = pd.merge_asof(
                        roi_df,
                        csv_df,
                        on="datetime",
                        direction="nearest",
                        tolerance=pd.Timedelta("6H")
                    )
                    aligned_df["Image Path"] = aligned_df["Image Path"].apply(os.path.basename)
                    aligned_df["Mask Path"] = aligned_df["Mask Path"].apply(os.path.basename)

                    #self.aligned_df = aligned_df

                    aligned_df.to_excel(aligned_xlsx_path, index=False)
                    aligned_written = True
                else:
                    print("Alignment skipped: one of the DataFrames has no valid datetime rows.")
            else:
                print("Alignment skipped: related_csv_df is missing or empty.")
        except Exception as e:
            print(f"Error during alignment or writing aligned XLSX: {e}")

        predictors = []
        for i in range(1, n_clusters + 1):
            predictors.extend([f'Cluster {i} H', f'Cluster {i} S', f'Cluster {i} V'])

        responses = ['Gage Height', 'Discharge']

        if aligned_df:
            self.analyze_responses(aligned_df, predictors=predictors, responses=responses, output_folder=output_folder)

        from os.path import exists as _exists
        QMessageBox.information(
            self,
            "Export Complete",
            f"Metrics written to:\n{csv_path if _exists(csv_path) else '(CSV write failed)'}\n"
            + (f"{xlsx_path}" if _exists(xlsx_path) else "(XLSX skipped or write failed)")
            + ("\nAligned results written to:\n" + aligned_xlsx_path if aligned_written and _exists(aligned_xlsx_path)
               else "\nAligned results skipped or write failed")
            + ("\nRelated CSV loaded into self.related_csv_df" if self.related_csv_df is not None else "\nNo related CSV found")
            + (f"\nSensor data correlated from:\n{sensor_file}" if sensor_df is not None
               else "\nNo sensor data correlated")
            + (f"\nSensor analysis report:\n{analysis_path}" if analysis_path else "")
        )

    def reject(self):
        pass