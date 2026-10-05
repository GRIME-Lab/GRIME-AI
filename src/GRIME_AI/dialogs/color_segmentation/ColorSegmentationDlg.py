#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from appcore.utils.resource_utils import ui_path
from appcore.utils import theme
from appcore.JSON_Editor import JsonEditor

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi


# ======================================================================================================================
class roiParameters:
    def __init__(self, parent=None):
        self.strROIName       = ''
        self.roiShape         = 0   # ROIShape value: 0=RECTANGLE, 1=POLYGON, 2=FREEFORM
        self.numColorClusters = 4
        self.bDisplayROIs     = True
        self.bDisplayROIColors = True


# ======================================================================================================================
class ColorSegmentationDlg(QDialog):

    # ------------------------------------------------------------------------------------------------------------------
    # SIGNALS
    # ------------------------------------------------------------------------------------------------------------------
    colorSegmentation_Signal   = pyqtSignal(int)
    addROI_Signal              = pyqtSignal(roiParameters)
    deleteAllROI_Signal        = pyqtSignal()
    close_signal               = pyqtSignal()
    buildFeatureFile_Signal    = pyqtSignal()
    exportROIMasks_Signal      = pyqtSignal()
    importROIMasks_Signal      = pyqtSignal()
    universalTestButton_Signal = pyqtSignal(int)
    greenness_index_signal     = pyqtSignal()
    refresh_rois_signal        = pyqtSignal(roiParameters)
    roiShapeChanged_signal     = pyqtSignal(int)   # 0=RECTANGLE, 1=POLYGON, 2=FREEFORM

    returnROIParameters = roiParameters()

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, parent=None):
        super(ColorSegmentationDlg, self).__init__(parent)
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        loadUi(ui_path("color_segmentation/QDialog_ColorSegmentation.ui"), self)

        # Light-mode styles come from the .ui; these are their dark-mode versions.
        self._bind_theme()


        # ------------------------------------------------------------------
        # ROI CONTROLS
        # ------------------------------------------------------------------
        self.pushButtonAddROI.clicked.connect(self.addROI)
        # Shape selector -> notify the canvas which shape to draw.
        self.radioButton_ROIShapeRectangle.setChecked(True)
        self.radioButton_ROIShapeRectangle.toggled.connect(self._on_shape_changed)
        self.radioButton_ROIShapePolygon.toggled.connect(self._on_shape_changed)
        self.radioButton_ROIShapeFreeForm.toggled.connect(self._on_shape_changed)
        self.pushButton_deleteAllROIs.clicked.connect(self.deleteAllROI)
        self.buttonBox_Close.clicked.connect(self.closeClicked)
        self.pushButton_Dlg_BuildFeatureFile.clicked.connect(self.buildFeatureFile)
        self.pushButton_ExportROIMasks.clicked.connect(self.exportROIMasks)
        self.pushButton_ImportROIMasks.clicked.connect(self.importROIMasks)
        self.spinBoxColorClusters.valueChanged[int].connect(self.colorClusterValueChanged)

        # Region Select: at least one of Whole Image / ROI must be checked to build.
        self.checkBoxScalarRegion_WholeImage.toggled.connect(self._update_build_enabled)
        self.checkBoxScalarRegion_ROI.toggled.connect(self._update_build_enabled)

        # ------------------------------------------------------------------
        # GREENNESS INDEX
        # ------------------------------------------------------------------
        self.checkBox_GCC.clicked.connect(self.GCC_Clicked)
        self.checkBox_GLI.clicked.connect(self.GLI_Clicked)
        self.checkBox_ExG.clicked.connect(self.ExG_Clicked)
        self.checkBox_RGI.clicked.connect(self.RGI_Clicked)
        self.checkBox_NDVI.clicked.connect(self.NDVI_Clicked)

        # ------------------------------------------------------------------
        # DEVELOPER-ONLY TEST BUTTON
        # ------------------------------------------------------------------
        import getpass
        if getpass.getuser() in ("johns", "tgilmore10"):
            self.pushButton_Dlg_TEST.clicked.connect(self.universalTestButton)
        else:
            self.pushButton_Dlg_TEST.setEnabled(False)
            self.pushButton_Dlg_TEST.hide()

        # ------------------------------------------------------------------

        # Restore persisted control states (everything except ROI name).
        self._load_settings()
        self._update_build_enabled()
        # Size to the fully-built, settings-loaded content.
        self.adjustSize()
        self.setMinimumSize(self.sizeHint())

    # ------------------------------------------------------------------
    # DARK MODE
    # ------------------------------------------------------------------
    _DARK_SHAPE_BUTTON = (
        "QPushButton { border: 1px solid #6B7680; border-radius: 6px; padding: 3px 8px; } "
        "QPushButton:checked { border: 2px solid #6FA8DC; color: #8FC1EC; font-weight: 500; "
        "background-color: #24394D; }")
    _DARK_GHOST_BUTTON = (
        "QPushButton { background: transparent; color: #8FC1EC; border: 1px solid #6FA8DC; "
        "border-radius: 8px; padding: 6px 12px; } "
        "QPushButton:hover { background: rgba(111,168,220,0.15); } "
        "QPushButton:pressed { background: rgba(111,168,220,0.25); } "
        "QPushButton:disabled { color: #5A6B7A; border-color: #455564; }")
    # (fill, border, title) for each feature group in dark mode
    _DARK_GROUPS = {
        "groupBox_15":      ("#0E2A44", "#378ADD", "#9CC7F0"),   # Feature Select
        "groupBox_14":      ("#221F4D", "#7F77DD", "#C3BEF5"),   # Region Select
        "groupBox_Texture": ("#3A2708", "#EF9F27", "#F5C27A"),   # Texture Methods
        "groupBox_16":      ("#1E3310", "#7AAE3E", "#B7DC8C"),   # Greenness
    }

    def _bind_theme(self):
        theme.bind_ui(self.label_ShapeHeading, "color: #9DA9B5; font-weight: 500;")
        for b in (self.radioButton_ROIShapeRectangle, self.radioButton_ROIShapePolygon,
                  self.radioButton_ROIShapeFreeForm):
            theme.bind_ui(b, self._DARK_SHAPE_BUTTON)
        theme.bind_ui(self.pushButtonAddROI,
                      "QPushButton { background: transparent; border: 1px solid #6FA8DC; color: #8FC1EC; "
                      "border-radius: 5px; padding: 0px 8px; } "
                      "QPushButton:hover { background: rgba(111,168,220,0.15); } "
                      "QPushButton:pressed { background: rgba(111,168,220,0.25); }")
        theme.bind_ui(self.pushButton_deleteAllROIs,
                      "QPushButton { background: transparent; border: 1px solid #FF6B5E; color: #FF6B5E; "
                      "border-radius: 5px; padding: 0px 8px; } "
                      "QPushButton:hover { background: rgba(255,107,94,0.12); }")
        theme.bind_ui(self.pushButton_ExportROIMasks, self._DARK_GHOST_BUTTON)
        theme.bind_ui(self.pushButton_ImportROIMasks, self._DARK_GHOST_BUTTON)
        for name, (fill, border, title) in self._DARK_GROUPS.items():
            box = getattr(self, name)
            theme.bind_ui(box, (
                f"QGroupBox#{name} {{ background-color: {fill}; border: 1px solid {border}; "
                f"border-radius: 4px; margin-top: 10px; padding-top: 6px; }} "
                f"QGroupBox#{name}::title {{ subcontrol-origin: margin; subcontrol-position: top left; "
                f"left: 8px; padding: 0px 3px; color: {title}; background-color: {fill}; }} "
                f"QGroupBox#{name}::title:disabled {{ color: gray; }} "
                # qdarkstyle paints check boxes with the window color; let the group fill show through
                f"QGroupBox#{name} QCheckBox {{ background-color: transparent; }}"))

    # ------------------------------------------------------------------
    def get_texture_options(self) -> dict:
        """
        Returns which texture methods are selected.
        Texture is enabled when at least one method is checked.
        Call this from buildFeatureFile before passing options to ExtractFeatures.
        """
        opts = {
            'glcm':    self.checkBox_Texture_GLCM.isChecked(),
            'gabor':   self.checkBox_Texture_Gabor.isChecked(),
            'lbp':     self.checkBox_Texture_LBP.isChecked(),
            'wavelet': self.checkBox_Texture_Wavelet.isChecked(),
            'fourier': self.checkBox_Texture_Fourier.isChecked(),
        }
        opts['enabled'] = any(opts.values())
        return opts

    # ------------------------------------------------------------------
    def colorClusterValueChanged(self):
        self.returnROIParameters.numColorClusters = self.spinBoxColorClusters.value()
        self.refresh_rois_signal.emit(self.returnROIParameters)

    def _update_build_enabled(self, *_):
        ok = (self.checkBoxScalarRegion_WholeImage.isChecked()
              or self.checkBoxScalarRegion_ROI.isChecked())
        self.pushButton_Dlg_BuildFeatureFile.setEnabled(ok)
        self.pushButton_Dlg_BuildFeatureFile.setToolTip(
            "" if ok else "Select Whole Image, ROI, or both under Region Select.")

    def buildFeatureFile(self):
        self.buildFeatureFile_Signal.emit()

    def universalTestButton(self):
        self.universalTestButton_Signal.emit(1)

    # ------------------------------------------------------------------
    # Persisted settings (everything except the ROI name), stored in the
    # GRIME-AI settings JSON via JsonEditor.
    # ------------------------------------------------------------------
    _SETTINGS = {
        "ColorSeg_NumClusters":        ("spinBoxColorClusters", "int"),
        "ColorSeg_Shape_Rectangle":    ("radioButton_ROIShapeRectangle", "bool"),
        "ColorSeg_Shape_Polygon":      ("radioButton_ROIShapePolygon", "bool"),
        "ColorSeg_Shape_FreeForm":     ("radioButton_ROIShapeFreeForm", "bool"),
        "ColorSeg_Texture_GLCM":       ("checkBox_Texture_GLCM", "bool"),
        "ColorSeg_Texture_Gabor":      ("checkBox_Texture_Gabor", "bool"),
        "ColorSeg_Texture_LBP":        ("checkBox_Texture_LBP", "bool"),
        "ColorSeg_Texture_Wavelet":    ("checkBox_Texture_Wavelet", "bool"),
        "ColorSeg_Texture_Fourier":    ("checkBox_Texture_Fourier", "bool"),
        "ColorSeg_GCC":                ("checkBox_GCC", "bool"),
        "ColorSeg_GLI":                ("checkBox_GLI", "bool"),
        "ColorSeg_ExG":                ("checkBox_ExG", "bool"),
        "ColorSeg_RGI":                ("checkBox_RGI", "bool"),
        "ColorSeg_NDVI":               ("checkBox_NDVI", "bool"),
        "ColorSeg_Intensity":          ("checkBox_Intensity", "bool"),
        "ColorSeg_ShannonEntropy":     ("checkBox_ShannonEntropy", "bool"),
    }

    def _load_settings(self):
        """Restore saved control states from the GRIME-AI settings JSON."""
        try:
            je = JsonEditor()
        except Exception:
            return
        for key, (widget_name, kind) in self._SETTINGS.items():
            w = getattr(self, widget_name, None)
            if w is None:
                continue
            val = je.getValue(key)
            if val is None:
                continue
            try:
                if kind == "int":
                    w.setValue(int(val))
                elif kind == "bool":
                    w.setChecked(bool(val))
            except Exception:
                pass

    def _save_settings(self):
        """Persist current control states to the GRIME-AI settings JSON."""
        try:
            je = JsonEditor()
        except Exception:
            return
        for key, (widget_name, kind) in self._SETTINGS.items():
            w = getattr(self, widget_name, None)
            if w is None:
                continue
            try:
                if kind == "int":
                    je.update_json_entry(key, int(w.value()))
                elif kind == "bool":
                    je.update_json_entry(key, bool(w.isChecked()))
            except Exception:
                pass

    def exportROIMasks(self):
        self.exportROIMasks_Signal.emit()

    def importROIMasks(self):
        self.importROIMasks_Signal.emit()

    def showEvent(self, event):
        super().showEvent(event)
        # Ensure the dialog opens tall enough for all groups.
        self.adjustSize()

    def closeEvent(self, event):
        self._save_settings()
        super(ColorSegmentationDlg, self).closeEvent(event)
        self.close_signal.emit()

    def closeClicked(self):
        self.close_signal.emit()

    def colorSegmentationClicked(self):
        self.colorSegmentation_Signal.emit(1)

    def addROI(self):
        self.returnROIParameters.strROIName        = self.lineEdit_roiName.text()
        self.returnROIParameters.numColorClusters  = self.spinBoxColorClusters.value()
        self.returnROIParameters.bDisplayROIs      = True
        self.returnROIParameters.bDisplayROIColors = True
        self.returnROIParameters.roiShape = self.get_roi_shape()
        self.addROI_Signal.emit(self.returnROIParameters)

    def get_roi_shape(self):
        """Currently selected ROI shape: 0=RECTANGLE, 1=POLYGON, 2=FREEFORM."""
        return (2 if self.radioButton_ROIShapeFreeForm.isChecked()
                else 1 if self.radioButton_ROIShapePolygon.isChecked() else 0)

    def _on_shape_changed(self, checked=True):
        # toggled fires for the button being unchecked too; emit once, for the newly checked one.
        if not checked:
            return
        self.roiShapeChanged_signal.emit(self.get_roi_shape())

    def deleteAllROI(self):
        self.deleteAllROI_Signal.emit()

    def GCC_Clicked(self):
        self.greenness_index_signal.emit()

    def GLI_Clicked(self):
        self.greenness_index_signal.emit()

    def ExG_Clicked(self):
        self.greenness_index_signal.emit()

    def RGI_Clicked(self):
        self.greenness_index_signal.emit()

    def NDVI_Clicked(self):
        self.greenness_index_signal.emit()

    def disable_spinbox_color_clusters(self, disable_spinbox=True):
        self.spinBoxColorClusters.setDisabled(disable_spinbox)

    def get_num_color_clusters(self):
        return self.spinBoxColorClusters.value()
