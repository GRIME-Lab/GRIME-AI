#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# SensorDataOptionsDlg.py
#
# Where the NWIS sensor file comes from, and how close in time a sensor reading
# must be to an image to count as a match. Opened from the ROI Analyzer tab's
# "Sensor Data Options" button; nothing here prompts during extraction.

import os

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QCheckBox,
                             QLabel, QLineEdit, QPushButton, QSpinBox, QFileDialog,
                             QMessageBox, QListWidget, QDialogButtonBox)
from PyQt5.QtGui import QFont

from appcore.CSS_Styles import BUTTON_CSS_STEEL_BLUE
from appcore.dialogs.ML_image_processing.roi_feature_extraction import (
    load_settings, save_settings, merged_settings, sister_data_folder, sensor_files_in,
    SENSOR_DATA_FOLDER_NAME)

HELP_TEXT = (
    "Each image's timestamp is matched to the nearest reading in the NWIS file, and every "
    "sensor parameter in that file is added to the feature table.\n\n"
    "Auto-detect: the NWIS data is taken from the \"data\" folder beside the images folder, "
    "where downloads put it. Downloads arrive as USGS text and are converted to CSV, and only "
    "the CSV is used. If repeated downloads left more than one CSV you are asked which to use, "
    "newest first. If the folder is missing you are asked for the file, and cancelling skips "
    "the correlation.\n\n"
    "With auto-detect off, the file below is used, for a file downloaded or moved elsewhere.\n\n"
    "Maximum time difference: how far from an image's time a reading may be and still count "
    "as a match. Automatic uses half the sensor's own sampling interval. Images with no "
    "reading inside the limit keep empty sensor columns; every row also reports the offset "
    "of its match."
)


class SensorDataOptionsDlg(QDialog):

    def __init__(self, parent=None, images_folder: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Sensor Data Options")
        self.setFont(QFont("Arial", 9))
        self._images_folder = images_folder

        self.check_auto = QCheckBox("Auto-detect the NWIS file in the data folder beside the images")
        self.check_auto.toggled.connect(self._update_enabled)

        self.edit_file = QLineEdit()
        self.edit_file.setPlaceholderText("NWIS sensor file (.csv, or the USGS .txt)")
        self.button_browse = QPushButton("Browse")
        self.button_browse.clicked.connect(self._browse)

        self.spin_tolerance = QSpinBox()
        self.spin_tolerance.setRange(0, 24 * 60)
        self.spin_tolerance.setSuffix(" min")
        self.spin_tolerance.setSpecialValueText("Automatic")

        self.label_detected = QLabel()
        self.label_detected.setWordWrap(True)

        grid = QGridLayout()
        grid.addWidget(self.check_auto, 0, 0, 1, 3)
        grid.addWidget(QLabel("Sensor file:"), 1, 0)
        grid.addWidget(self.edit_file, 1, 1)
        grid.addWidget(self.button_browse, 1, 2)
        grid.addWidget(QLabel("Maximum time difference:"), 2, 0)
        grid.addWidget(self.spin_tolerance, 2, 1)
        grid.addWidget(self.label_detected, 3, 0, 1, 3)
        grid.setColumnStretch(1, 1)

        self.button_help = QPushButton("Help")
        self.button_help.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.button_help.clicked.connect(
            lambda: QMessageBox.information(self, "Help: Sensor Data", HELP_TEXT))
        self.button_ok = QPushButton("OK")
        self.button_ok.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.button_ok.clicked.connect(self._save_and_accept)
        self.button_cancel = QPushButton("Cancel")
        self.button_cancel.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addWidget(self.button_help)
        buttons.addStretch(1)
        buttons.addWidget(self.button_ok)
        buttons.addWidget(self.button_cancel)

        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addLayout(buttons)

        self._set_values(load_settings())
        self.resize(560, 180)

    # ------------------------------------------------------------------------------------------------------------------
    def _set_values(self, settings):
        s = settings["sensor"]
        self.check_auto.setChecked(bool(s["auto_detect"]))
        self.edit_file.setText(s["sensor_file"] or "")
        self.spin_tolerance.setValue(int(s["tolerance_minutes"] or 0))
        self._update_enabled()

    def _update_enabled(self, *_):
        auto = self.check_auto.isChecked()
        self.edit_file.setEnabled(not auto)
        self.button_browse.setEnabled(not auto)
        self.label_detected.setText(self._detected_text() if auto else "")

    def _detected_text(self):
        folder = sister_data_folder(self._images_folder) if self._images_folder else ""
        if not folder:
            return f"The {SENSOR_DATA_FOLDER_NAME} folder is found once an images folder is set."
        if not os.path.isdir(folder):
            return f"Not found yet: {folder}\nYou will be asked for it when features are extracted."
        files = sensor_files_in(folder)
        if not files:
            return f"No sensor CSV in {folder}"
        if len(files) == 1:
            return f"Found: {files[0]}"
        return f"Found {len(files)} sensor CSVs in {folder}; you will be asked which to use."

    def _browse(self):
        start = os.path.dirname(self.edit_file.text()) or self._images_folder
        path, _ = QFileDialog.getOpenFileName(self, "Select NWIS Sensor File", start,
                                              "NWIS sensor data (*.txt *.csv);;All files (*.*)")
        if path:
            self.edit_file.setText(path)

    # ------------------------------------------------------------------------------------------------------------------
    def get_settings(self) -> dict:
        settings = load_settings()
        settings["sensor"].update({
            "auto_detect": self.check_auto.isChecked(),
            "sensor_file": self.edit_file.text().strip(),
            "tolerance_minutes": self.spin_tolerance.value(),
        })
        return merged_settings(settings)

    def _save_and_accept(self):
        if not self.check_auto.isChecked():
            path = self.edit_file.text().strip()
            if not path or not os.path.isfile(path):
                QMessageBox.warning(self, "Sensor Data Options",
                                    "Select an NWIS sensor file, or turn auto-detect on.")
                return
        try:
            save_settings(self.get_settings())
        except Exception as e:
            QMessageBox.critical(self, "Sensor Data Options", f"Could not save the options:\n{e}")
            return
        self.accept()


class SensorFileChooserDlg(QDialog):
    """Pick one NWIS file when the data folder holds several. Newest first."""

    def __init__(self, parent, files, folder):
        super().__init__(parent)
        self.setWindowTitle("Select Sensor File")
        self.setFont(QFont("Arial", 9))
        self._files = list(files)

        self.list = QListWidget()
        for path in self._files:
            self.list.addItem(os.path.basename(path))
        self.list.setCurrentRow(0)
        self.list.itemDoubleClicked.connect(lambda _: self.accept())

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"More than one sensor CSV in {folder}.\nNewest first:"))
        layout.addWidget(self.list)
        layout.addWidget(box)
        self.resize(520, 260)

    def selected_file(self):
        row = self.list.currentRow()
        return self._files[row] if 0 <= row < len(self._files) else None
