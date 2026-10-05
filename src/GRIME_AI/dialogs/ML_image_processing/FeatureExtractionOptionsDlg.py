#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# FeatureExtractionOptionsDlg.py
#
# Options for ROI feature extraction, opened from both the ROI Analyzer tab and
# the Segment Images tab. OK saves the options (roi_feature_extraction.save_settings)
# so both tabs export with the same settings.

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
                             QWidget, QRadioButton, QButtonGroup, QSpinBox, QDoubleSpinBox,
                             QCheckBox, QLabel, QLineEdit, QPushButton, QComboBox, QMessageBox)
from PyQt5.QtGui import QFont

from appcore.CSS_Styles import BUTTON_CSS_STEEL_BLUE
from appcore.dialogs.ML_image_processing.roi_feature_extraction import (
    load_settings, save_settings, merged_settings)

WAVELETS = ["db1", "db2", "db4", "haar", "sym2", "coif1"]

# Help text per tab, shown by the Help button for the tab that is open.
HELP_TEXT = {
    "Color Clusters": (
        "Finds the dominant colors inside the segmentation mask and adds HSV and RGB "
        "columns for each cluster, largest cluster first.\n\n"
        "K-Means and GMM: the number of clusters is set directly.\n\n"
        "Mean-Shift: finds its own number of clusters. Auto estimates the bandwidth "
        "from the Quantile; otherwise the Bandwidth is used. The K-Means cluster count "
        "sets how many Mean-Shift clusters are reported, so every row has the same columns."),
    "Texture": (
        "All texture features are computed inside the segmentation mask only.\n\n"
        "GLCM: contrast, homogeneity and correlation of gray-level pairs the given pixel "
        "distance apart, averaged over four directions (0, 45, 90 and 135 degrees).\n\n"
        "Gabor: mean and variance of the filter response for each frequency (cycles per "
        "pixel, 0 to 0.5), averaged over the same four directions.\n\n"
        "LBP: histogram of uniform local binary patterns (Points + 2 bins).\n\n"
        "Wavelet: variance of the horizontal, vertical and diagonal detail at each level. "
        "Level 1 is the finest detail.\n\n"
        "Fourier: radial profile of the region's frequency content in equal rings from "
        "low to high frequency."),
    "Output": (
        "The CSV is always written. The Excel workbook has the same columns, with the "
        "image and mask names as links to the files.\n\n"
        "Every run writes new files named with the run date and time, "
        "e.g. 20260922_101500_roi_metrics.csv.\n\n"
        "The Status column is OK, or explains why an image could not be analyzed."),
}


class FeatureExtractionOptionsDlg(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Feature Extraction Options")
        font = QFont("Arial", 9)
        self.setFont(font)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(self._build_clusters_tab(), "Color Clusters")
        self.tabs.addTab(self._build_texture_tab(), "Texture")
        self.tabs.addTab(self._build_output_tab(), "Output")

        self.pushButton_help = QPushButton("Help")
        self.pushButton_help.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_help.clicked.connect(self._show_help)
        self.pushButton_defaults = QPushButton("Restore Defaults")
        self.pushButton_ok = QPushButton("OK")
        self.pushButton_cancel = QPushButton("Cancel")
        for b in (self.pushButton_defaults, self.pushButton_ok):
            b.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.pushButton_defaults.clicked.connect(lambda: self._set_values(merged_settings()))
        self.pushButton_ok.clicked.connect(self._save_and_accept)
        self.pushButton_cancel.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addWidget(self.pushButton_defaults)
        buttons.addWidget(self.pushButton_help)
        buttons.addStretch(1)
        buttons.addWidget(self.pushButton_ok)
        buttons.addWidget(self.pushButton_cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addLayout(buttons)

        self._set_values(load_settings())
        self.resize(460, 320)

    # ------------------------------------------------------------------------------------------------------------------
    def _build_clusters_tab(self):
        w = QWidget()
        g = QGridLayout(w)
        self.radio_kmeans = QRadioButton("K-Means")
        self.radio_gmm = QRadioButton("GMM")
        self.radio_meanshift = QRadioButton("Mean-Shift")
        self._method_group = QButtonGroup(w)
        for r in (self.radio_kmeans, self.radio_gmm, self.radio_meanshift):
            self._method_group.addButton(r)
            r.toggled.connect(self._update_enabled)

        self.spin_kmeans = QSpinBox(); self.spin_kmeans.setRange(1, 10)
        self.spin_gmm = QSpinBox(); self.spin_gmm.setRange(1, 10)
        self.check_auto = QCheckBox("Auto")
        self.check_auto.toggled.connect(self._update_enabled)
        self.spin_quantile = QDoubleSpinBox(); self.spin_quantile.setRange(0.05, 0.95); self.spin_quantile.setSingleStep(0.05)
        self.spin_bandwidth = QDoubleSpinBox(); self.spin_bandwidth.setRange(1.0, 200.0); self.spin_bandwidth.setSingleStep(0.5)

        g.addWidget(self.radio_kmeans, 0, 0); g.addWidget(QLabel("Clusters:"), 0, 1); g.addWidget(self.spin_kmeans, 0, 2)
        g.addWidget(self.radio_gmm, 1, 0); g.addWidget(QLabel("Clusters:"), 1, 1); g.addWidget(self.spin_gmm, 1, 2)
        g.addWidget(self.radio_meanshift, 2, 0); g.addWidget(self.check_auto, 2, 1)
        g.addWidget(QLabel("Quantile:"), 3, 1); g.addWidget(self.spin_quantile, 3, 2)
        g.addWidget(QLabel("Bandwidth:"), 4, 1); g.addWidget(self.spin_bandwidth, 4, 2)
        g.setRowStretch(5, 1)
        return w

    # ------------------------------------------------------------------------------------------------------------------
    def _build_texture_tab(self):
        w = QWidget()
        g = QGridLayout(w)
        self.check_glcm = QCheckBox("GLCM")
        self.spin_glcm_distance = QSpinBox(); self.spin_glcm_distance.setRange(1, 32)
        self.check_gabor = QCheckBox("Gabor")
        self.edit_gabor_freqs = QLineEdit()
        self.edit_gabor_freqs.setToolTip("Frequencies in cycles per pixel, separated by commas (0 to 0.5).")
        self.check_lbp = QCheckBox("LBP")
        self.spin_lbp_points = QSpinBox(); self.spin_lbp_points.setRange(4, 24)
        self.spin_lbp_radius = QDoubleSpinBox(); self.spin_lbp_radius.setRange(1.0, 8.0); self.spin_lbp_radius.setSingleStep(1.0)
        self.check_wavelet = QCheckBox("Wavelet")
        self.combo_wavelet = QComboBox(); self.combo_wavelet.addItems(WAVELETS)
        self.spin_wavelet_levels = QSpinBox(); self.spin_wavelet_levels.setRange(1, 6)
        self.check_fourier = QCheckBox("Fourier")
        self.spin_fourier_rings = QSpinBox(); self.spin_fourier_rings.setRange(4, 128)

        rows = [
            (self.check_glcm,    [("Pixel distance:", self.spin_glcm_distance)]),
            (self.check_gabor,   [("Frequencies:", self.edit_gabor_freqs)]),
            (self.check_lbp,     [("Points:", self.spin_lbp_points), ("Radius:", self.spin_lbp_radius)]),
            (self.check_wavelet, [("Wavelet:", self.combo_wavelet), ("Levels:", self.spin_wavelet_levels)]),
            (self.check_fourier, [("Rings:", self.spin_fourier_rings)]),
        ]
        for r, (check, params) in enumerate(rows):
            check.toggled.connect(self._update_enabled)
            g.addWidget(check, r, 0)
            col = 1
            for label, widget in params:
                g.addWidget(QLabel(label), r, col); g.addWidget(widget, r, col + 1)
                col += 2
        g.setRowStretch(len(rows), 1)
        return w

    # ------------------------------------------------------------------------------------------------------------------
    def _build_output_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        csv_box = QCheckBox("CSV (always written)")
        csv_box.setChecked(True)
        csv_box.setEnabled(False)
        self.check_excel = QCheckBox("Excel workbook with image hyperlinks")
        v.addWidget(csv_box)
        v.addWidget(self.check_excel)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------------------------------------------------------
    def _show_help(self):
        tab = self.tabs.tabText(self.tabs.currentIndex())
        QMessageBox.information(self, f"Help: {tab}", HELP_TEXT.get(tab, ""))

    # ------------------------------------------------------------------------------------------------------------------
    def _update_enabled(self, *_):
        ms = self.radio_meanshift.isChecked()
        self.spin_kmeans.setEnabled(self.radio_kmeans.isChecked() or ms)
        self.spin_gmm.setEnabled(self.radio_gmm.isChecked())
        self.check_auto.setEnabled(ms)
        self.spin_quantile.setEnabled(ms and self.check_auto.isChecked())
        self.spin_bandwidth.setEnabled(ms and not self.check_auto.isChecked())
        self.spin_glcm_distance.setEnabled(self.check_glcm.isChecked())
        self.edit_gabor_freqs.setEnabled(self.check_gabor.isChecked())
        self.spin_lbp_points.setEnabled(self.check_lbp.isChecked())
        self.spin_lbp_radius.setEnabled(self.check_lbp.isChecked())
        self.combo_wavelet.setEnabled(self.check_wavelet.isChecked())
        self.spin_wavelet_levels.setEnabled(self.check_wavelet.isChecked())
        self.spin_fourier_rings.setEnabled(self.check_fourier.isChecked())

    # ------------------------------------------------------------------------------------------------------------------
    def _set_values(self, s):
        c, t = s["clustering"], s["texture"]
        {"gmm": self.radio_gmm, "meanshift": self.radio_meanshift}.get(c["method"], self.radio_kmeans).setChecked(True)
        self.spin_kmeans.setValue(int(c["kmeans_clusters"]))
        self.spin_gmm.setValue(int(c["gmm_clusters"]))
        self.check_auto.setChecked(bool(c["meanshift_auto"]))
        self.spin_quantile.setValue(float(c["meanshift_quantile"]))
        self.spin_bandwidth.setValue(float(c["meanshift_bandwidth"]))

        self.check_glcm.setChecked(bool(t["glcm"]["enabled"]))
        self.spin_glcm_distance.setValue(int(t["glcm"]["distance"]))
        self.check_gabor.setChecked(bool(t["gabor"]["enabled"]))
        self.edit_gabor_freqs.setText(", ".join(str(f) for f in t["gabor"]["frequencies"]))
        self.check_lbp.setChecked(bool(t["lbp"]["enabled"]))
        self.spin_lbp_points.setValue(int(t["lbp"]["points"]))
        self.spin_lbp_radius.setValue(float(t["lbp"]["radius"]))
        self.check_wavelet.setChecked(bool(t["wavelet"]["enabled"]))
        if t["wavelet"]["wavelet"] not in WAVELETS:
            self.combo_wavelet.addItem(t["wavelet"]["wavelet"])
        self.combo_wavelet.setCurrentText(t["wavelet"]["wavelet"])
        self.spin_wavelet_levels.setValue(int(t["wavelet"]["levels"]))
        self.check_fourier.setChecked(bool(t["fourier"]["enabled"]))
        self.spin_fourier_rings.setValue(int(t["fourier"]["rings"]))

        self.check_excel.setChecked(bool(s["output"]["excel"]))
        self._update_enabled()

    # ------------------------------------------------------------------------------------------------------------------
    def _gabor_frequencies(self):
        """Parse the frequency list; None if it is not a comma-separated list of 0 < f <= 0.5."""
        try:
            freqs = [float(x) for x in self.edit_gabor_freqs.text().replace(";", ",").split(",") if x.strip()]
        except ValueError:
            return None
        return freqs if freqs and all(0.0 < f <= 0.5 for f in freqs) else None

    def get_settings(self) -> dict:
        method = "gmm" if self.radio_gmm.isChecked() else "meanshift" if self.radio_meanshift.isChecked() else "kmeans"
        return merged_settings({
            "clustering": {
                "method": method,
                "kmeans_clusters": self.spin_kmeans.value(),
                "gmm_clusters": self.spin_gmm.value(),
                "meanshift_auto": self.check_auto.isChecked(),
                "meanshift_quantile": round(self.spin_quantile.value(), 4),
                "meanshift_bandwidth": round(self.spin_bandwidth.value(), 4),
            },
            "texture": {
                "glcm":    {"enabled": self.check_glcm.isChecked(), "distance": self.spin_glcm_distance.value()},
                "gabor":   {"enabled": self.check_gabor.isChecked(),
                            "frequencies": self._gabor_frequencies() or []},
                "lbp":     {"enabled": self.check_lbp.isChecked(), "points": self.spin_lbp_points.value(),
                            "radius": self.spin_lbp_radius.value()},
                "wavelet": {"enabled": self.check_wavelet.isChecked(),
                            "wavelet": self.combo_wavelet.currentText(),
                            "levels": self.spin_wavelet_levels.value()},
                "fourier": {"enabled": self.check_fourier.isChecked(), "rings": self.spin_fourier_rings.value()},
            },
            "output": {"excel": self.check_excel.isChecked()},
        })

    def _save_and_accept(self):
        if self.check_gabor.isChecked() and self._gabor_frequencies() is None:
            self.tabs.setCurrentIndex(1)
            QMessageBox.warning(self, "Gabor Frequencies",
                                "Enter one or more Gabor frequencies separated by commas, "
                                "each greater than 0 and at most 0.5 (cycles per pixel).")
            return
        try:
            save_settings(self.get_settings())
        except Exception as e:
            QMessageBox.critical(self, "Feature Extraction Options", f"Could not save the options:\n{e}")
            return
        self.accept()
