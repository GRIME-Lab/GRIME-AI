#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os
import json

from PyQt5.QtWidgets import QDialog, QWidget, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIntValidator, QDoubleValidator
from PyQt5.uic import loadUi

from appcore.dialogs.triage.TriageCalibrateDlg import TriageCalibrateDlg
from ...app_identity import APP_CONFIG_FILENAME

BUTTON_CSS_STEEL_BLUE = 'QPushButton {background-color: steelblue; color: white;}'

SLIDER_RANGES = {
    "lap":  (0,   500, 150),
    "bmin": (0,   255,  40),
    "bmax": (0,   255, 215),
}


class TriageOptionsDlg(QDialog):

    def __init__(self, parent=None, folder: str = ''):
        super().__init__(parent)

        ui_dir_name      = os.path.dirname(__file__)
        ui_file_absolute = os.path.join(ui_dir_name, 'QDialog_TriageOptions.ui')
        loadUi(ui_file_absolute, self)

        normal_font = QFont("Arial", 9)
        normal_font.setStyleHint(QFont.SansSerif)
        self.setFont(normal_font)
        for widget in self.findChildren(QWidget):
            widget.setFont(normal_font)

        self.adjustSize()
        self.referenceImageFilename = ''
        self._triage_folder = folder

        self._setup_validators()
        self._setup_connections()
        self._apply_styles()
        self._sync_all_displays()
        self._load_calibration_from_config()

    # ------------------------------------------------------------------------------------------------------------------
    def _setup_validators(self):
        self.lineEdit_lap_value.setValidator(QIntValidator(0, 500))
        self.lineEdit_bmin_value.setValidator(QIntValidator(0, 255))
        self.lineEdit_bmax_value.setValidator(QIntValidator(0, 255))

    # ------------------------------------------------------------------------------------------------------------------
    def _setup_connections(self):
        self.pushButton_SelectReferenceImage.clicked.connect(self.selectReferenceImage)
        self.pushButton_Calibrate.clicked.connect(self._open_calibration_dialog)

        self.checkBox_UseLaplacian.toggled.connect(self._on_laplacian_toggled)

        self.sliderLaplacian.valueChanged.connect(self._on_lap_slider_changed)
        self.sliderBrightnessMin.valueChanged.connect(self._on_bmin_slider_changed)
        self.sliderBrightnessMax.valueChanged.connect(self._on_bmax_slider_changed)

        self.lineEdit_lap_value.editingFinished.connect(self._on_lap_edit_finished)
        self.lineEdit_bmin_value.editingFinished.connect(self._on_bmin_edit_finished)
        self.lineEdit_bmax_value.editingFinished.connect(self._on_bmax_edit_finished)

        self._on_laplacian_toggled(self.checkBox_UseLaplacian.isChecked())
        self.checkBox_UseFocusROI.toggled.connect(self._on_focus_roi_toggled)
        self.checkBox_UseColorImbalance.toggled.connect(self._on_color_imbalance_toggled)
        self.doubleSpinBox_ColorImbalanceThreshold.valueChanged.connect(self._on_color_imbalance_value_changed)

    # ------------------------------------------------------------------------------------------------------------------
    def _apply_styles(self):
        self.pushButton_SelectReferenceImage.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_Calibrate.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

    # ------------------------------------------------------------------------------------------------------------------
    def _sync_all_displays(self):
        self._on_lap_slider_changed(self.sliderLaplacian.value())
        self._on_bmin_slider_changed(self.sliderBrightnessMin.value())
        self._on_bmax_slider_changed(self.sliderBrightnessMax.value())

    # ------------------------------------------------------------------------------------------------------------------
    def _on_laplacian_toggled(self, checked):
        for w in [self.sliderLaplacian,
                  self.labelLapLeft, self.labelLapMiddle, self.labelLapRight,
                  self.lineEdit_lap_value,
                  self.doubleSpinBoxLaplacianThreshold]:
            w.setEnabled(checked)

    # ------------------------------------------------------------------------------------------------------------------
    def _on_lap_slider_changed(self, val):
        self.lineEdit_lap_value.setText(str(val))
        self.doubleSpinBoxLaplacianThreshold.setValue(float(val))

    def _on_bmin_slider_changed(self, val):
        self.lineEdit_bmin_value.setText(str(val))
        self.doubleSpinBoxBrightnessMin.setValue(float(val))

    def _on_bmax_slider_changed(self, val):
        self.lineEdit_bmax_value.setText(str(val))
        self.doubleSpinBoxBrightnessMax.setValue(float(val))

    # ------------------------------------------------------------------------------------------------------------------
    def _clamp(self, val, min_val, max_val):
        return max(min_val, min(max_val, val))

    def _on_lap_edit_finished(self):
        try:
            val = self._clamp(int(self.lineEdit_lap_value.text()), 0, 500)
            self.sliderLaplacian.setValue(val)
        except ValueError:
            self._on_lap_slider_changed(self.sliderLaplacian.value())

    def _on_bmin_edit_finished(self):
        try:
            val = self._clamp(int(self.lineEdit_bmin_value.text()), 0, 255)
            self.sliderBrightnessMin.setValue(val)
        except ValueError:
            self._on_bmin_slider_changed(self.sliderBrightnessMin.value())

    def _on_bmax_edit_finished(self):
        try:
            val = self._clamp(int(self.lineEdit_bmax_value.text()), 0, 255)
            self.sliderBrightnessMax.setValue(val)
        except ValueError:
            self._on_bmax_slider_changed(self.sliderBrightnessMax.value())

    # ------------------------------------------------------------------------------------------------------------------
    def selectReferenceImage(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff)"
        )
        if filename:
            self.referenceImageFilename = filename
            self.lineEdit_ReferenceImagePath.setText(filename)

    # ------------------------------------------------------------------------------------------------------------------
    def _open_calibration_dialog(self):
        dlg = TriageCalibrateDlg(parent=self, triage_folder=self._triage_folder)
        dlg.calibration_applied.connect(self._apply_calibration)
        dlg.exec_()
        # After calibration dialog closes, refresh checkbox state in case ROI was defined/cleared
        self._update_focus_roi_checkbox()

    # ------------------------------------------------------------------------------------------------------------------
    def _apply_calibration(self, result):
        self.sliderLaplacian.setValue(int(round(result.laplacian_threshold)))
        self.sliderBrightnessMin.setValue(int(round(result.brightness_min)))
        self.sliderBrightnessMax.setValue(int(round(result.brightness_max)))

    # ------------------------------------------------------------------------------------------------------------------
    def _on_focus_roi_toggled(self, checked):
        if not checked:
            return
        try:
            from appcore.Save_Utils import Save_Utils
            config_path = os.path.join(
                Save_Utils().get_settings_folder(), APP_CONFIG_FILENAME)
            if not os.path.exists(config_path):
                return
            with open(config_path, "r") as f:
                config = json.load(f)
            calibration_folder = config.get("triage", {}).get("calibration_folder", "")
            if not calibration_folder:
                return
            if os.path.normpath(calibration_folder) != os.path.normpath(self._triage_folder):
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Focus Region Mismatch",
                    f"The focus region was defined for a different image folder:\n\n"
                    f"Calibrated: {calibration_folder}\n\n"
                    f"Current:      {self._triage_folder}\n\n"
                    "Results may be inaccurate."
                )
        except Exception as e:
            print(f"[TriageOptionsDlg] Could not check calibration folder: {e}")

    def _on_color_imbalance_toggled(self, checked):
        self.doubleSpinBox_ColorImbalanceThreshold.setEnabled(checked)
        self._save_color_imbalance_to_config()

    def _on_color_imbalance_value_changed(self, value):
        self._save_color_imbalance_to_config()

    def _save_color_imbalance_to_config(self):
        try:
            config_path = os.path.join(
                self._get_settings_folder(), APP_CONFIG_FILENAME)
            config = {}
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
            config.setdefault("triage", {})
            config["triage"]["use_color_imbalance"]       = self.checkBox_UseColorImbalance.isChecked()
            config["triage"]["color_imbalance_threshold"] = self.doubleSpinBox_ColorImbalanceThreshold.value()
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"[TriageOptionsDlg] Could not save color imbalance settings: {e}")

    def _get_settings_folder(self):
        from appcore.Save_Utils import Save_Utils
        return Save_Utils().get_settings_folder()

    def _update_focus_roi_checkbox(self):
        """Enable checkbox only when a focus ROI is saved in config."""
        roi = self._read_focus_roi_from_config()
        has_roi = roi is not None and len(roi) == 4
        self.checkBox_UseFocusROI.setEnabled(has_roi)
        if not has_roi:
            self.checkBox_UseFocusROI.setChecked(False)

    # ------------------------------------------------------------------------------------------------------------------
    def _read_focus_roi_from_config(self):
        """Return the saved focus_roi list from <application>.json or None."""
        try:
            from appcore.Save_Utils import Save_Utils
            config_path = os.path.join(
                Save_Utils().get_settings_folder(), APP_CONFIG_FILENAME)
            if not os.path.exists(config_path):
                return None
            with open(config_path, "r") as f:
                config = json.load(f)
            return config.get("triage", {}).get("focus_roi")
        except Exception:
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def _load_calibration_from_config(self):
        try:
            from appcore.Save_Utils import Save_Utils
            settings_folder = Save_Utils().get_settings_folder()
            config_path     = os.path.join(settings_folder, APP_CONFIG_FILENAME)
            if not os.path.exists(config_path):
                return
            with open(config_path, "r") as f:
                config = json.load(f)
            triage = config.get("triage", {})
            if not triage:
                return
            if "laplacian_threshold" in triage:
                self.sliderLaplacian.setValue(int(round(triage["laplacian_threshold"])))
            if "brightness_min"      in triage:
                self.sliderBrightnessMin.setValue(int(round(triage["brightness_min"])))
            if "brightness_max"      in triage:
                self.sliderBrightnessMax.setValue(int(round(triage["brightness_max"])))

            # Restore focus ROI checkbox state
            self._update_focus_roi_checkbox()

            # Restore color imbalance settings
            if "use_color_imbalance" in triage:
                self.checkBox_UseColorImbalance.setChecked(triage["use_color_imbalance"])
            if "color_imbalance_threshold" in triage:
                self.doubleSpinBox_ColorImbalanceThreshold.setValue(
                    float(triage["color_imbalance_threshold"]))
            if triage.get("use_focus_roi") and self.checkBox_UseFocusROI.isEnabled():
                self.checkBox_UseFocusROI.setChecked(True)

        except Exception as e:
            print(f"[TriageOptionsDlg] Could not load calibration: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------------------------------------------------------
    def getCreateReport(self):
        return self.checkBoxTriageImagesReport.isChecked()

    def getMoveImages(self):
        return self.checkBoxTriageImagesMove.isChecked()

    def getUseFftBlur(self):
        return self.checkBox_UseFftBlur.isChecked()

    def getUseLaplacian(self):
        return self.checkBox_UseLaplacian.isChecked()

    def getLaplacianThreshold(self):
        return self.doubleSpinBoxLaplacianThreshold.value()

    def getBrightnessMin(self):
        return self.doubleSpinBoxBrightnessMin.value()

    def getBrightnessMax(self):
        return self.doubleSpinBoxBrightnessMax.value()

    def getSavePolylines(self):
        return self.checkBox_TriageImages_SavePolylines.isChecked()

    def getCorrectAlignment(self):
        return self.checkBox_TriageImages_CorrectAlignment.isChecked()

    def getReferenceImageFilename(self):
        return self.referenceImageFilename

    def getRotationThreshold(self):
        return self.doubleSpinBox_TriageOptions_RotationThreshold.value()

    def getUseColorImbalance(self):
        return self.checkBox_UseColorImbalance.isChecked()

    def getColorImbalanceThreshold(self):
        return self.doubleSpinBox_ColorImbalanceThreshold.value()

    def getFftCalibration(self):
        """Return the saved FFT blur calibration dict, or None. ImageQualityAnalyzer
        uses it only if it matches the current focus region."""
        try:
            from appcore.Save_Utils import Save_Utils
            config_path = os.path.join(
                Save_Utils().get_settings_folder(), APP_CONFIG_FILENAME)
            if not os.path.exists(config_path):
                return None
            with open(config_path, "r") as f:
                config = json.load(f)
            return config.get("triage", {}).get("fft_calibration")
        except Exception:
            return None

    def getFocusROI(self):
        """Return normalised [x, y, w, h] if Use Focus Region is checked, else None."""
        if not self.checkBox_UseFocusROI.isChecked():
            return None
        roi = self._read_focus_roi_from_config()
        # Persist the checked state
        try:
            from appcore.Save_Utils import Save_Utils
            config_path = os.path.join(
                Save_Utils().get_settings_folder(), APP_CONFIG_FILENAME)
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                config.setdefault("triage", {})["use_focus_roi"] = True
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)
        except Exception:
            pass
        return roi if roi and len(roi) == 4 else None
