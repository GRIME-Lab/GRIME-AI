#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from GRIME_AI.utils.resource_utils import ui_path
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor

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
class GRIME_AI_ColorSegmentationDlg(QDialog):

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
        super(GRIME_AI_ColorSegmentationDlg, self).__init__(parent)
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        loadUi(ui_path("color_segmentation/QDialog_ColorSegmentation.ui"), self)


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
        super(GRIME_AI_ColorSegmentationDlg, self).closeEvent(event)
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
