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

from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtWidgets import QWidget, QFileDialog, QListWidgetItem, QMessageBox
from PyQt5.QtGui import QPixmap, QIcon, QImage, QPainter, QColor, QFont

from appcore import PROJECT_ROOT
from appcore.JSON_Editor import JsonEditor
from appcore.QProgressWheel import QProgressWheel
from appcore.CSS_Styles import BUTTON_CSS_STEEL_BLUE
from appcore.dialogs.ML_image_processing.roi_feature_extraction import (
    load_settings, save_settings, clustering_kwargs, cluster_count, extract_datetime_from_path,
    RoiFeatureExtractor, masked_glcm, write_xlsx, run_prefix, STATUS_OK,
    sister_data_folder, sensor_files_in, SENSOR_DATA_FOLDER_NAME)


class ROIAnalyzerTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # EXPECTED UI WIDGETS (assign before calling wire_connections()):
        # - self.lineEdit_ROI_images_folder
        # - self.pushButton_browse_ROI_images_folder
        # - self.pushButton_analyze
        # - self.pushButton_extract_ROI_features
        # - self.listWidget_filmstrip
        # - self.pushButton_feature_options
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
    def _update_texture_fields(self, analyzer):
        """Display GLCM metrics inside the mask, if GLCM is enabled in the feature options."""
        self.lineEdit_glcm_contrast.clear()
        self.lineEdit_glcm_homogeneity.clear()
        self.lineEdit_glcm_correlation.clear()

        glcm = load_settings()["texture"]["glcm"]
        if glcm["enabled"]:
            try:
                gray = cv2.cvtColor(analyzer.image, cv2.COLOR_BGR2GRAY)
                features = masked_glcm(gray, analyzer.mask_bin > 0, glcm["distance"])
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

        # Feature extraction options (clustering and texture), shared with the Segment Images tab
        self.pushButton_feature_options.clicked.connect(self._open_feature_options)
        self.pushButton_feature_options.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        # Sensor correlation: opt in with the checkbox, set the file with the button.
        # Nothing prompts unless the checkbox is on and the file cannot be found.
        self.checkBox_correlate_sensor_data.setChecked(load_settings()["sensor"]["enabled"])
        self.checkBox_correlate_sensor_data.toggled.connect(self._on_correlate_sensor_toggled)
        self.pushButton_sensor_options.clicked.connect(self._open_sensor_options)
        self.pushButton_sensor_options.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self._on_correlate_sensor_toggled(self.checkBox_correlate_sensor_data.isChecked())

        # Button fonts
        self.pushButton_browse_ROI_images_folder.setFont(QFont("Arial", 11))
        self.pushButton_feature_options.setFont(QFont("Arial", 11))
        self.pushButton_analyze.setFont(QFont("Arial", 11, QFont.Bold))
        self.pushButton_extract_ROI_features.setFont(QFont("Arial", 11, QFont.Bold))

        # Optional close hook (matches original)
        if hasattr(self, "buttonBox_close") and self.buttonBox_close is not None:
            self.buttonBox_close.rejected.connect(self.reject)

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
    def _on_correlate_sensor_toggled(self, checked):
        self.pushButton_sensor_options.setEnabled(checked)
        settings = load_settings()
        if settings["sensor"]["enabled"] != bool(checked):
            settings["sensor"]["enabled"] = bool(checked)
            try:
                save_settings(settings)
            except Exception as e:
                print(f"[ROIAnalyzerTab] Could not save the sensor setting: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _open_sensor_options(self):
        from appcore.dialogs.ML_image_processing.SensorDataOptionsDlg import SensorDataOptionsDlg
        SensorDataOptionsDlg(self, images_folder=self.lineEdit_ROI_images_folder.text().strip()).exec_()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _resolve_sensor_file(self, images_folder, settings):
        """
        The NWIS file to correlate, or None to skip the correlation.

        Auto-detect looks in the data folder beside the images folder and asks only
        when it must: several files to choose from, or the folder is missing.
        """
        sensor = settings["sensor"]
        if not sensor["auto_detect"]:
            path = sensor["sensor_file"]
            if path and os.path.isfile(path):
                return path
            path, _ = QFileDialog.getOpenFileName(
                self, "Select NWIS Sensor File", images_folder,
                "NWIS sensor data (*.txt *.csv);;All files (*.*)")
            return path or None

        folder = sister_data_folder(images_folder)
        files = sensor_files_in(folder)
        if not files:
            folder = QFileDialog.getExistingDirectory(
                self, f"Select the folder holding the NWIS sensor data "
                      f"(no '{SENSOR_DATA_FOLDER_NAME}' folder beside the images)",
                images_folder)
            if not folder:
                return None
            files = sensor_files_in(folder)
            if not files:
                QMessageBox.warning(self, "Sensor Data",
                                    f"No sensor CSV files in:\n{folder}")
                return None
        if len(files) == 1:
            return files[0]

        from appcore.dialogs.ML_image_processing.SensorDataOptionsDlg import SensorFileChooserDlg
        chooser = SensorFileChooserDlg(self, files, folder)
        return chooser.selected_file() if chooser.exec_() else None

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _open_feature_options(self):
        """Edit the feature extraction options, then re-analyze the shown image with them."""
        from appcore.dialogs.ML_image_processing.FeatureExtractionOptionsDlg import FeatureExtractionOptionsDlg
        if FeatureExtractionOptionsDlg(self).exec_():
            current = self.listWidget_filmstrip.currentItem()
            if current:
                self.on_filmstrip_item_clicked(current)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _new_analyzer(self, orig_path, mask_path):
        """ROI_Analyzer configured with the saved clustering options."""
        from appcore.ROI_Analyzer import ROI_Analyzer
        return ROI_Analyzer(orig_path, mask_path, **clustering_kwargs(load_settings()))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _extract_datetime_from_path(self, filepath):
        """(date, time) from the filename; see roi_feature_extraction.extract_datetime_from_path."""
        return extract_datetime_from_path(filepath)

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

        # Run analysis for this specific pair
        analyzer = self._new_analyzer(orig_path, mask_path)
        try:
            analyzer.run_analysis()
        except Exception as e:
            QMessageBox.warning(self, "ROI Analyzer", f"Could not analyze this image:\n{e}")
            return

        # ─── populate metric fields ───
        self.lineEdit_intensity.setText(f"{analyzer.roi_intensity:.2f}")
        self.lineEdit_entropy.setText(f"{analyzer.roi_entropy:.4f}")
        self.lineEdit_Texture.setText(f"{analyzer.roi_texture:.4f}")
        self.lineEdit_GLI.setText(f"{analyzer.mean_gli:.6f}")
        self.lineEdit_GCC.setText(f"{analyzer.mean_gcc:.6f}")

        self._update_texture_fields(analyzer)

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

        from appcore.ROI_Analyzer import ROI_Analyzer
        temp = ROI_Analyzer("", "")
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
            from appcore.ROI_Analyzer import ROI_Analyzer
        except ImportError:
            QMessageBox.warning(self, "ROI Analyzer", "Unable to import ROI Analyzer module.")
            return

        # 1) generate pairs + batched filmstrip population
        temp = ROI_Analyzer("", "")
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
        analyzer = self._new_analyzer(orig_path, mask_path)
        try:
            analyzer.run_analysis()
        except Exception as e:
            QMessageBox.warning(self, "ROI Analyzer", f"Could not analyze this image:\n{e}")
            return

        # ─── populate metric fields ───
        self.lineEdit_intensity.setText(f"{analyzer.roi_intensity:.2f}")
        self.lineEdit_entropy.setText(f"{analyzer.roi_entropy:.4f}")
        self.lineEdit_Texture.setText(f"{analyzer.roi_texture:.4f}")
        self.lineEdit_GLI.setText(f"{analyzer.mean_gli:.6f}")
        self.lineEdit_GCC.setText(f"{analyzer.mean_gcc:.6f}")

        self._update_texture_fields(analyzer)

        capture_date, capture_time = self._extract_datetime_from_path(orig_path)
        self._display_analysis(analyzer, capture_date, capture_time)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def extract_ROI_features(self):
        # Features and columns come from the shared extractor, so this export and the
        # Segment Images tab's "Export ROI features" produce identical files.
        settings = load_settings()
        n_clusters = cluster_count(settings)

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

        # Correlate NWIS sensor data only when the user asked for it. The file comes
        # from the sensor options; nothing is asked unless it cannot be found.
        sensor_file = None
        if settings["sensor"]["enabled"]:
            sensor_file = self._resolve_sensor_file(image_folder, settings)
            if not sensor_file:
                print("[ROIAnalyzerTab] Sensor correlation skipped: no sensor file selected.")

        # Filename prefix is the run date/time, per the GRIME AI convention
        # used by all analysis outputs (e.g. ImageTriage_YYYYMMDD_HHMMSS).
        file_dt_prefix = run_prefix()

        # All correlation outputs (csv, xlsx, pngs) go into a 'correlation'
        # subfolder so they don't pollute the image/data folder.
        correlation_folder = os.path.join(output_folder, "correlation")
        os.makedirs(correlation_folder, exist_ok=True)

        # Output file paths with datetime prefix
        csv_path = os.path.join(correlation_folder, f"{file_dt_prefix}_roi_metrics.csv")
        xlsx_path = os.path.join(correlation_folder, f"{file_dt_prefix}_roi_metrics.xlsx")

        # 2) Header from the shared extractor
        extractor = RoiFeatureExtractor(settings)
        header = extractor.header()
        rows = []

        total_iterations = len(self._pairs)
        self.progress_bar_closed = False
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

        # 3) Iterate pairs. A failed image still gets a row, with the reason in Status.
        n_failed = 0
        try:
            for i, (orig_path, mask_path) in enumerate(self._pairs):
                if self.progress_bar_closed:
                    break
                progressBar.setValue(i)
                row = extractor.compute_row(orig_path, mask_path)
                if row[-1] != STATUS_OK:
                    n_failed += 1
                    print(f"Failed on {orig_path}, {mask_path}: {row[-1]}")
                rows.append(row)
        finally:
            try:
                progressBar.close()
            except Exception:
                pass
            del progressBar

        # 3.5) Convert to DataFrame for in-memory use
        df = pd.DataFrame(rows, columns=header)
        self.roi_metrics_df = df  # Keep in memory for subsequent steps

        # Correlate the selected sensor file (if any) and append its values.
        # Failure here degrades gracefully to an uncorrelated feature file.
        sensor_df = None
        if sensor_file:
            try:
                from appcore.SensorImageCorrelator import SensorImageCorrelator
                tolerance = settings["sensor"]["tolerance_minutes"] or None   # 0: automatic
                sensor_df = SensorImageCorrelator().sensor_values_for_images(
                    df["Image Path"].tolist(), sensor_file, tolerance_minutes=tolerance)
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

        print("\nFirst 5 rows of ROI Metrics DataFrame:")
        print(self.roi_metrics_df.head())

        # 4) Write out CSV of ROI metrics
        try:
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error writing CSV to {csv_path}: {e}")

        # 5) Write out XLSX with hyperlinks for ROI metrics
        if settings["output"]["excel"]:
            try:
                write_xlsx(xlsx_path, header, df.astype(object).where(df.notna(), "").values.tolist())
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
                from appcore.SensorROIAnalysis import SensorROIAnalysis
                analysis_path = os.path.join(correlation_folder, f"{file_dt_prefix}_sensor_analysis.xlsx")
                png_dir = os.path.join(correlation_folder, f"{file_dt_prefix}_figures")
                SensorROIAnalysis().write_report(df, analysis_path, png_dir=png_dir)
            except Exception as e:
                analysis_path = None
                print(f"Error writing sensor analysis report: {e}")

        from os.path import exists as _exists
        QMessageBox.information(
            self,
            "Export Complete",
            f"Metrics written to:\n{csv_path if _exists(csv_path) else '(CSV write failed)'}\n"
            + (f"{xlsx_path}" if _exists(xlsx_path) else "(XLSX skipped or write failed)")
            + (f"\nSensor data correlated from:\n{sensor_file}" if sensor_df is not None
               else "\nNo sensor data correlated")
            + (f"\nSensor analysis report:\n{analysis_path}" if analysis_path else "")
            + (f"\n\n{n_failed} image(s) could not be analyzed; see the Status column." if n_failed else "")
        )

    def reject(self):
        pass