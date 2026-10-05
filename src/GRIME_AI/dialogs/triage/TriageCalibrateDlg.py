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
# Saves results to application json and emit calibrated params to caller.
#
# The right-hand panel lets the user browse images from the triage source
# folder and draw, move or resize a rectangle (RoiEditorLabel) to define the
# focus scoring ROI.
# The ROI is stored as normalised [x, y, w, h] floats in the application json.

import os
import json

import cv2
from PyQt5.QtWidgets import (QDialog, QWidget, QFileDialog, QApplication,
                              QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.uic import loadUi

from appcore.dialogs.triage.TriageCalibrator import TriageCalibrator, CalibrationResult
from appcore.Save_Utils import Save_Utils
from ...app_identity import APP_CONFIG_FILENAME

BUTTON_CSS_STEEL_BLUE = (
    'QPushButton {background-color: steelblue; color: white;}'
    'QPushButton:hover {background-color: #5a93c2;}'
    'QPushButton:disabled {background-color: gray; color: black;}'
)
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
        # An exception raised inside QThread.run() is only printed to the console;
        # without this the dialog stays stuck with no feedback.
        try:
            result = self.calibrator.calibrate(
                good_folder            = self.good_folder,
                blurry_folder          = self.blurry_folder,
                exposure_folder        = self.exposure_folder,
                color_imbalance_folder = self.color_imbalance_folder,
                focus_roi              = self.focus_roi,
                progress_callback = lambda msg, pct: self.progress.emit(msg, pct)
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = CalibrationResult()
            result.success = False
            result.error_message = f"{type(e).__name__}: {e}"
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
        self._update_run_button_state()
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

        # ROI drawn, moved or resized on the image
        self.label_FocusImage.roiChanged.connect(self._on_roi_changed)

    def _apply_styles(self):
        self.pushButton_BrowseGood.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_BrowseBlurry.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_BrowseExposure.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_BrowseColorImbalance.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_RunCalibration.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_ApplyAndClose.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

    # ──────────────────────────────────────────────────────────────────────────
    # Focus ROI edits
    # ──────────────────────────────────────────────────────────────────────────

    def _on_roi_changed(self, roi):
        """ROI drawn, moved, resized (normalised [x, y, w, h]) or cleared (None)."""
        self._focus_roi = roi
        self.pushButton_ClearROI.setEnabled(roi is not None)
        self._save_focus_roi_to_config()

    # ──────────────────────────────────────────────────────────────────────────
    # Image panel
    # ──────────────────────────────────────────────────────────────────────────

    def _init_image_panel(self):
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
        self.label_FocusImage.setImage(QPixmap.fromImage(qimg))     # label scales to fit

        self.label_FocusImageName.setText(os.path.basename(path))
        self.label_ImageIndex.setText(
            f"{self._image_index + 1} / {len(self._image_files)}"
        )

        self.label_FocusImage.setNormalizedRoi(self._focus_roi)

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
    # ROI clear
    # ──────────────────────────────────────────────────────────────────────────

    def _clear_roi(self):
        self.label_FocusImage.clearRoi()    # emits roiChanged(None) -> _on_roi_changed

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

    def _missing_required_folders(self) -> list:
        """Required folder selections that are still empty."""
        missing = []
        if not self.lineEdit_GoodFolder.text().strip():
            missing.append("Good images folder.")
        if not (self.lineEdit_BlurryFolder.text().strip() or self.lineEdit_ExposureFolder.text().strip()):
            missing.append("A Blurry images folder or an Exposure images folder (at least one).")
        return missing

    def _invalid_selected_folders(self) -> list:
        """Selected folders that do not exist or contain no images."""
        problems = []
        for label, line_edit in [("Good images",            self.lineEdit_GoodFolder),
                                 ("Blurry images",          self.lineEdit_BlurryFolder),
                                 ("Exposure images",        self.lineEdit_ExposureFolder),
                                 ("Color imbalance images", self.lineEdit_ColorImbalanceFolder)]:
            folder = line_edit.text().strip()
            if not folder:
                continue
            if not os.path.isdir(folder):
                problems.append(f"{label}: folder does not exist ({folder}).")
            elif not any(os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS for f in os.listdir(folder)):
                problems.append(f"{label}: folder contains no images ({folder}).")
        return problems

    def _update_run_button_state(self):
        missing = self._missing_required_folders()
        self.pushButton_RunCalibration.setEnabled(not missing)
        if missing:
            self.pushButton_RunCalibration.setToolTip("Required before running:\n" + "\n".join(missing))
        else:
            self.pushButton_RunCalibration.setToolTip("Run the calibration.")

    # ──────────────────────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────────────────────

    def _run_calibration(self):
        problems = self._missing_required_folders() + self._invalid_selected_folders()
        if problems:
            QMessageBox.warning(self, "Cannot Run Calibration",
                                "Please fix the following before running calibration:\n\n"
                                + "\n".join(f"\u2022 {p}" for p in problems))
            return

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

        self.lineEdit_ResultLaplacianThreshold.setText(f"{result.laplacian_threshold:.1f}")
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

        if result.n_blurry == 0:
            self.lineEdit_ResultFftBlur.setText("Not run")
        elif result.fft_calibration is None:
            self.lineEdit_ResultFftBlur.setText("Not calibrated")
        elif result.fft_calibration["overlap"]:
            self.lineEdit_ResultFftBlur.setText("Overlap")
        else:
            self.lineEdit_ResultFftBlur.setText("Calibrated")

        summary = (f"Done. Images: {result.n_good} good, {result.n_blurry} blurry, "
                   f"{result.n_exposure} exposure.")
        if result.n_blurry > 0:
            summary += f"  Blur F1: {result.blur_f1:.3f}."
            if result.fft_calibration is None:
                summary += f"\n{result.fft_note}"
            elif result.fft_calibration["overlap"]:
                summary += ("\nFFT blur: the Good and Blurry scores overlap, so triage will use "
                            "one fixed threshold halfway between them.")
            else:
                summary += "\nFFT blur calibrated."
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
        """Return True if the application json already contains triage calibration data."""
        try:
            config = self._load_config()
            triage = config.get("triage", {})
            return any(k in triage for k in ("laplacian_threshold", "focus_roi"))
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
            if result.fft_calibration is not None:
                config["triage"]["fft_calibration"] = result.fft_calibration
            elif result.n_blurry > 0:
                # Blurry images were given but could not be separated: drop any old calibration.
                config["triage"].pop("fft_calibration", None)
            # FFT blur is now chosen automatically per folder; drop the old fixed settings.
            config["triage"].pop("fft_blur_threshold", None)
            config["triage"].pop("fft_shift_radius", None)
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
                self.label_FocusImage.setNormalizedRoi(roi)
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
        for widget in [self.lineEdit_ResultLaplacianThreshold,
                       self.lineEdit_ResultBrightnessMin,
                       self.lineEdit_ResultBrightnessMax,
                       self.lineEdit_ResultColorImbalance,
                       self.lineEdit_ResultFftBlur]:
            widget.clear()
