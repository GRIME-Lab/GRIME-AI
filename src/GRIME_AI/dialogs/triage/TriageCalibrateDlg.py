#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: April 2026
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# TriageCalibrateDlg.py
#
# Dialog for calibrating triage parameters from labelled image folders.
# Delegates all calibration logic to TriageCalibrator (no Qt there).
# Saves results to GRIME-AI.json and emits calibrated params to caller.
#
# The right-hand panel lets the user browse images from the triage source
# folder and draw a rubber-band rectangle to define the focus scoring ROI.
# The ROI is stored as normalised [x, y, w, h] floats in GRIME-AI.json.

import os
import json

import cv2
from PyQt5.QtWidgets import (QDialog, QWidget, QFileDialog, QApplication,
                              QMessageBox)
from PyQt5.QtCore import Qt, QThread, QRect, QEvent, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.uic import loadUi

from GRIME_AI.dialogs.triage.TriageCalibrator import TriageCalibrator, CalibrationResult
from GRIME_AI.Save_Utils import Save_Utils
from GRIME_AI.QLabel_drawing_modes import DrawingMode

BUTTON_CSS_STEEL_BLUE = 'QPushButton {background-color: steelblue; color: white;}'
APP_CONFIG_FILENAME = "GRIME-AI.json"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


# ──────────────────────────────────────────────────────────────────────────────
# Background worker thread
# ──────────────────────────────────────────────────────────────────────────────

class CalibrationWorker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(object)

    def __init__(self, calibrator, good_folder, blurry_folder, exposure_folder, color_imbalance_folder, focus_roi):
        super().__init__()
        self.calibrator             = calibrator
        self.good_folder            = good_folder
        self.blurry_folder          = blurry_folder
        self.exposure_folder        = exposure_folder
        self.color_imbalance_folder = color_imbalance_folder
        self.focus_roi              = focus_roi

    def run(self):
        result = self.calibrator.calibrate(
            good_folder            = self.good_folder,
            blurry_folder          = self.blurry_folder,
            exposure_folder        = self.exposure_folder,
            color_imbalance_folder = self.color_imbalance_folder,
            focus_roi              = self.focus_roi,
            progress_callback = lambda msg, pct: self.progress.emit(msg, pct)
        )
        self.finished.emit(result)


# ──────────────────────────────────────────────────────────────────────────────
# Dialog
# ──────────────────────────────────────────────────────────────────────────────

class TriageCalibrateDlg(QDialog):
    """
    Calibration dialog.

    Parameters
    ----------
    triage_folder : str
        The image folder selected by the user before opening triage options.
        Used to populate the focus-region image browser on the right panel.

    Emits calibration_applied(CalibrationResult) on Apply and Close.
    """

    calibration_applied = pyqtSignal(object)

    def __init__(self, parent=None, triage_folder: str = ''):
        super().__init__(parent)

        ui_dir_name      = os.path.dirname(__file__)
        ui_file_absolute = os.path.join(ui_dir_name, 'QDialog_TriageCalibrate.ui')
        loadUi(ui_file_absolute, self)

        normal_font = QFont("Arial", 9)
        normal_font.setStyleHint(QFont.SansSerif)
        self.setFont(normal_font)
        for widget in self.findChildren(QWidget):
            widget.setFont(normal_font)

        self.adjustSize()

        self._calibration_result = None
        self._worker             = None

        # ── Focus ROI state ───────────────────────────────────────────────
        self._triage_folder      = triage_folder
        self._image_files        = []
        self._image_index        = 0
        self._focus_roi          = None     # normalised [x, y, w, h]
        self._current_image_path = None

        self._setup_connections()
        self._apply_styles()
        self._init_image_panel()
        self._load_focus_roi_from_config()

        # ── Warn if existing calibration will be overwritten ──────────────
        if self._existing_calibration_found():
            response = QMessageBox.warning(
                self,
                "Existing Calibration Found",
                "Existing calibration settings were found.\n\n"
                "Proceeding will overwrite them. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if response == QMessageBox.No:
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(0, self.reject)

    # ──────────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_connections(self):
        self.pushButton_BrowseGood.clicked.connect(
            lambda: self._browse_folder(self.lineEdit_GoodFolder))
        self.pushButton_BrowseBlurry.clicked.connect(
            lambda: self._browse_folder(self.lineEdit_BlurryFolder))
        self.pushButton_BrowseExposure.clicked.connect(
            lambda: self._browse_folder(self.lineEdit_ExposureFolder))
        self.pushButton_BrowseColorImbalance.clicked.connect(
            lambda: self._browse_folder(self.lineEdit_ColorImbalanceFolder))

        self.lineEdit_GoodFolder.textChanged.connect(self._update_run_button_state)
        self.lineEdit_BlurryFolder.textChanged.connect(self._update_run_button_state)
        self.lineEdit_ExposureFolder.textChanged.connect(self._update_run_button_state)
        self.lineEdit_ColorImbalanceFolder.textChanged.connect(self._update_run_button_state)

        self.pushButton_RunCalibration.clicked.connect(self._run_calibration)
        self.pushButton_ApplyAndClose.clicked.connect(self._apply_and_close)
        self.pushButton_Cancel.clicked.connect(self.reject)

        # Focus region panel
        self.pushButton_PrevImage.clicked.connect(self._prev_image)
        self.pushButton_NextImage.clicked.connect(self._next_image)
        self.pushButton_ClearROI.clicked.connect(self._clear_roi)

        # Event filter to capture ROI after mouse release on the label
        self.label_FocusImage.installEventFilter(self)

    def _apply_styles(self):
        self.pushButton_BrowseGood.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_BrowseBlurry.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_BrowseExposure.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_BrowseColorImbalance.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_RunCalibration.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_ApplyAndClose.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

    # ──────────────────────────────────────────────────────────────────────────
    # Event filter — capture ROI after mouse release on label
    # ──────────────────────────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        if obj is self.label_FocusImage and event.type() == QEvent.MouseButtonRelease:
            # Let App_QLabel process the event first
            result = super().eventFilter(obj, event)
            self._capture_roi_from_label()
            return result
        return super().eventFilter(obj, event)

    def _capture_roi_from_label(self):
        """Read drawn ROI from App_QLabel and convert to normalised coords."""
        label_rect = self.label_FocusImage.getROI()
        if label_rect is None or not label_rect.isValid():
            return
        if label_rect.width() < 5 or label_rect.height() < 5:
            return

        norm = self._label_rect_to_normalised(label_rect)
        if norm is None:
            return

        self._focus_roi = norm
        self.pushButton_ClearROI.setEnabled(True)
        self._save_focus_roi_to_config()

    def _label_rect_to_normalised(self, label_rect: QRect):
        """
        Convert a QRect in label-widget space to normalised [x, y, w, h].
        Accounts for aspect-ratio-preserving pixmap placement inside the label.
        """
        pm = self.label_FocusImage.pixmap()
        if pm is None or pm.isNull():
            return None

        lw, lh = self.label_FocusImage.width(), self.label_FocusImage.height()
        pw, ph = pm.width(), pm.height()

        # Pixmap is centred inside the label
        x_off = (lw - pw) // 2
        y_off = (lh - ph) // 2

        # Convert label coords → pixmap coords
        rx = label_rect.x() - x_off
        ry = label_rect.y() - y_off
        rw = label_rect.width()
        rh = label_rect.height()

        # Clamp to pixmap bounds
        rx = max(0, min(pw - 1, rx))
        ry = max(0, min(ph - 1, ry))
        rw = max(1, min(pw - rx, rw))
        rh = max(1, min(ph - ry, rh))

        # Scale from pixmap coords to original image coords, then normalise
        if self._current_image_path:
            orig = cv2.imread(self._current_image_path)
            if orig is not None:
                ih_orig, iw_orig = orig.shape[:2]
                scale_x = iw_orig / pw
                scale_y = ih_orig / ph
                x  = (rx * scale_x) / iw_orig
                y  = (ry * scale_y) / ih_orig
                nw = (rw * scale_x) / iw_orig
                nh = (rh * scale_y) / ih_orig
                x  = max(0.0, min(1.0, x))
                y  = max(0.0, min(1.0, y))
                nw = max(0.0, min(1.0 - x, nw))
                nh = max(0.0, min(1.0 - y, nh))
                if nw > 0.01 and nh > 0.01:
                    return [x, y, nw, nh]
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Image panel
    # ──────────────────────────────────────────────────────────────────────────

    def _init_image_panel(self):
        self.label_FocusImage.setDrawingMode(DrawingMode.COLOR_SEGMENTATION)

        if not self._triage_folder or not os.path.isdir(self._triage_folder):
            self.label_FocusImage.setText("No triage folder selected")
            return

        self._image_files = sorted([
            os.path.join(self._triage_folder, f)
            for f in os.listdir(self._triage_folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])

        if not self._image_files:
            self.label_FocusImage.setText("No images found in triage folder")
            return

        self._image_index = 0
        self._display_current_image()
        self._update_nav_buttons()

    def _display_current_image(self):
        if not self._image_files:
            return

        path = self._image_files[self._image_index]
        self._current_image_path = path

        img = cv2.imread(path)
        if img is None:
            self.label_FocusImage.setText("Could not load image")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.label_FocusImage.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.label_FocusImage.width(),
                self.label_FocusImage.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
        self.label_FocusImage.setAlignment(Qt.AlignCenter)
        self.label_FocusImage.clearROIs()

        self.label_FocusImageName.setText(os.path.basename(path))
        self.label_ImageIndex.setText(
            f"{self._image_index + 1} / {len(self._image_files)}"
        )

        if self._focus_roi is not None:
            self._restore_roi_overlay()

    def _update_nav_buttons(self):
        n = len(self._image_files)
        self.pushButton_PrevImage.setEnabled(n > 0 and self._image_index > 0)
        self.pushButton_NextImage.setEnabled(n > 0 and self._image_index < n - 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Navigation
    # ──────────────────────────────────────────────────────────────────────────

    def _prev_image(self):
        if self._image_index > 0:
            self._image_index -= 1
            self._display_current_image()
            self._update_nav_buttons()

    def _next_image(self):
        if self._image_index < len(self._image_files) - 1:
            self._image_index += 1
            self._display_current_image()
            self._update_nav_buttons()

    # ──────────────────────────────────────────────────────────────────────────
    # ROI overlay restore
    # ──────────────────────────────────────────────────────────────────────────

    def _restore_roi_overlay(self):
        """Convert saved normalised ROI back to label-space and push into savedROIs."""
        if self._focus_roi is None:
            return

        pm = self.label_FocusImage.pixmap()
        if pm is None or pm.isNull():
            return

        lw, lh = self.label_FocusImage.width(), self.label_FocusImage.height()
        pw, ph = pm.width(), pm.height()
        x_off  = (lw - pw) // 2
        y_off  = (lh - ph) // 2

        x, y, nw, nh = self._focus_roi
        lx  = int(x  * pw) + x_off
        ly  = int(y  * ph) + y_off
        lrw = int(nw * pw)
        lrh = int(nh * ph)

        self.label_FocusImage.savedROIs = [QRect(lx, ly, lrw, lrh)]
        self.label_FocusImage.update()

    # ──────────────────────────────────────────────────────────────────────────
    # ROI clear
    # ──────────────────────────────────────────────────────────────────────────

    def _clear_roi(self):
        self._focus_roi = None
        self.label_FocusImage.clearROIs()
        self.pushButton_ClearROI.setEnabled(False)
        self._save_focus_roi_to_config()

    # ──────────────────────────────────────────────────────────────────────────
    # Public accessor
    # ──────────────────────────────────────────────────────────────────────────

    def getFocusROI(self):
        return self._focus_roi

    # ──────────────────────────────────────────────────────────────────────────
    # Folder browsing
    # ──────────────────────────────────────────────────────────────────────────

    def _browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            line_edit.setText(folder)

    # ──────────────────────────────────────────────────────────────────────────
    # Button state
    # ──────────────────────────────────────────────────────────────────────────

    def _update_run_button_state(self):
        good_set     = bool(self.lineEdit_GoodFolder.text().strip())
        blurry_set   = bool(self.lineEdit_BlurryFolder.text().strip())
        exposure_set = bool(self.lineEdit_ExposureFolder.text().strip())
        self.pushButton_RunCalibration.setEnabled(good_set and (blurry_set or exposure_set))

    # ──────────────────────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────────────────────

    def _run_calibration(self):
        self.pushButton_RunCalibration.setEnabled(False)
        self.pushButton_ApplyAndClose.setEnabled(False)
        self.progressBar.setValue(0)
        self.labelStatus.setText("Starting calibration...")
        self._clear_results()

        calibrator = TriageCalibrator(resize_percent=50.0)

        self._worker = CalibrationWorker(
            calibrator             = calibrator,
            good_folder            = self.lineEdit_GoodFolder.text().strip(),
            blurry_folder          = self.lineEdit_BlurryFolder.text().strip(),
            exposure_folder        = self.lineEdit_ExposureFolder.text().strip(),
            color_imbalance_folder = self.lineEdit_ColorImbalanceFolder.text().strip(),
            focus_roi              = self._focus_roi,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_calibration_finished)
        self._worker.start()

    def _on_progress(self, message: str, percent: int):
        self.progressBar.setValue(percent)
        self.labelStatus.setText(message)
        QApplication.processEvents()

    def _on_calibration_finished(self, result: CalibrationResult):
        self._calibration_result = result
        self.pushButton_RunCalibration.setEnabled(True)

        if not result.success:
            self.labelStatus.setText(f"Error: {result.error_message}")
            QMessageBox.critical(self, "Calibration Failed", result.error_message)
            return

        self.lineEdit_ResultFftThreshold.setText(f"{result.fft_blur_threshold:.2f}")
        self.lineEdit_ResultLaplacianThreshold.setText(f"{result.laplacian_threshold:.1f}")
        self.lineEdit_ResultFftRadius.setText(str(result.fft_shift_radius))
        self.lineEdit_ResultBrightnessMin.setText(f"{result.brightness_min:.1f}")
        self.lineEdit_ResultBrightnessMax.setText(f"{result.brightness_max:.1f}")

        # Color imbalance threshold — ask user if computed or keep current
        if result.n_color_imbalance > 0:
            current_thr = self._read_color_imbalance_threshold_from_config()
            msg = (f"Computed color imbalance threshold: {result.color_imbalance_threshold:.3f}\n\n"
                   f"Current value: {current_thr:.3f}\n\n"
                   "Use the computed value?")
            response = QMessageBox.question(self, "Color Imbalance Threshold", msg,
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if response == QMessageBox.No:
                result.color_imbalance_threshold = current_thr
        self.lineEdit_ResultColorImbalance.setText(f"{result.color_imbalance_threshold:.3f}")

        summary = (f"Done. Images: {result.n_good} good, {result.n_blurry} blurry, "
                   f"{result.n_exposure} exposure.")
        if result.n_blurry > 0:
            summary += f"  Blur F1: {result.blur_f1:.3f}."
        self.labelStatus.setText(summary)

        self.pushButton_ApplyAndClose.setEnabled(True)
        self._save_to_config(result)

    # ──────────────────────────────────────────────────────────────────────────
    # Apply and close
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_and_close(self):
        if self._calibration_result and self._calibration_result.success:
            self.calibration_applied.emit(self._calibration_result)
        self.accept()

    # ──────────────────────────────────────────────────────────────────────────
    # Config persistence
    # ──────────────────────────────────────────────────────────────────────────

    def _existing_calibration_found(self) -> bool:
        """Return True if GRIME-AI.json already contains triage calibration data."""
        try:
            config = self._load_config()
            triage = config.get("triage", {})
            return any(k in triage for k in ("laplacian_threshold", "fft_blur_threshold", "focus_roi"))
        except Exception:
            return False

    def _get_config_path(self):
        settings_folder = os.path.normpath(Save_Utils().get_settings_folder())
        return os.path.join(settings_folder, APP_CONFIG_FILENAME)

    def _load_config(self) -> dict:
        config_path = self._get_config_path()
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _write_config(self, config: dict):
        try:
            with open(self._get_config_path(), "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"[TriageCalibrateDlg] Could not write config: {e}")

    def _save_to_config(self, result: CalibrationResult):
        try:
            config = self._load_config()
            config.setdefault("triage", {})
            config["triage"].update(result.to_dict())
            if self._triage_folder:
                config["triage"]["calibration_folder"] = self._triage_folder
            self._write_config(config)
        except Exception as e:
            print(f"[TriageCalibrateDlg] Could not save calibration: {e}")

    def _save_focus_roi_to_config(self):
        try:
            config = self._load_config()
            config.setdefault("triage", {})
            config["triage"]["focus_roi"] = self._focus_roi
            if self._triage_folder:
                config["triage"]["calibration_folder"] = self._triage_folder
            self._write_config(config)
        except Exception as e:
            print(f"[TriageCalibrateDlg] Could not save focus ROI: {e}")

    def _load_focus_roi_from_config(self):
        try:
            config = self._load_config()
            roi = config.get("triage", {}).get("focus_roi")
            if roi and len(roi) == 4:
                self._focus_roi = roi
                self.pushButton_ClearROI.setEnabled(True)
                if self._image_files:
                    self._restore_roi_overlay()
        except Exception as e:
            print(f"[TriageCalibrateDlg] Could not load focus ROI: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _read_color_imbalance_threshold_from_config(self) -> float:
        """Return saved color_imbalance_threshold or 0.5 if not found."""
        try:
            config = self._load_config()
            return float(config.get("triage", {}).get("color_imbalance_threshold", 0.5))
        except Exception:
            return 0.5

    def _clear_results(self):
        for widget in [self.lineEdit_ResultFftThreshold,
                       self.lineEdit_ResultLaplacianThreshold,
                       self.lineEdit_ResultFftRadius,
                       self.lineEdit_ResultBrightnessMin,
                       self.lineEdit_ResultBrightnessMax,
                       self.lineEdit_ResultColorImbalance]:
            widget.clear()
