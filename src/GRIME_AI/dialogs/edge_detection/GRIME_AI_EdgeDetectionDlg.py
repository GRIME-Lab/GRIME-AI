#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import copy

from GRIME_AI.utils.resource_utils import ui_path

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi

from GRIME_AI.constants import edgeMethodsClass, featureMethodsClass


# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_EdgeDetectionDlg(QDialog):

    # SIGNALS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    edgeDetectionSignal = pyqtSignal(edgeMethodsClass)
    featureDetectionSignal = pyqtSignal(featureMethodsClass)

    # Legal OpenCV aperture / kernel sizes
    _LEGAL_KERNELS = (3, 5, 7)

    # -----------------------------------------------------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self, parent=None):
        super().__init__(parent)

        # INSTANCE state -- NOT class attributes. Class attributes are shared across every
        # instance of this dialog and leak mutations between them.
        self.returnEdgeData = edgeMethodsClass()
        self.returnFeatureData = featureMethodsClass()

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)

        loadUi(ui_path("edge_detection/QDialog_EdgeDetection.ui"), self)

        # Suppress emits while we wire up and seed initial state.
        self._loading = True

        # CONNECT THE SIGNALS TO THE FUNCTIONS IN THE PARENT ("CALLING") THREAD
        self.featureDetectionSignal.connect(parent.featureDetectionMethod)
        self.edgeDetectionSignal.connect(parent.edgeDetectionMethod)

        # Do not emit on every intermediate keystroke ("150" -> 1, 15, 150).
        for sb in (self.spinBoxCannyHighThreshold,
                   self.spinBoxCannyLowThreshold,
                   self.spinBoxCannyKernel,
                   self.spinBoxSobelKernel,
                   self.spinBoxOrbMaxFeatures):
            sb.setKeyboardTracking(False)

        # Radio buttons
        self.radioButtonCanny.clicked.connect(self.clicked_Canny)
        self.radioButtonSobelX.clicked.connect(self.clicked_SobelX)
        self.radioButtonSobelY.clicked.connect(self.clicked_SobelY)
        self.radioButtonSobelXY.clicked.connect(self.clicked_SobelXY)
        self.radioButtonLaplacian.clicked.connect(self.clicked_Laplacian)
        self.radioButtonSIFT.clicked.connect(self.clicked_SIFT)
        self.radioButtonORB.clicked.connect(self.clicked_ORB)

        # Spin boxes
        self.spinBoxCannyHighThreshold.valueChanged.connect(self.spinBoxCannyHighThresholdChanged)
        self.spinBoxCannyLowThreshold.valueChanged.connect(self.spinBoxCannyLowThresholdChanged)
        self.spinBoxCannyKernel.valueChanged.connect(self.spinBoxCannyKernelChanged)
        self.spinBoxSobelKernel.valueChanged.connect(self.spinBoxSobelKernelChanged)
        self.spinBoxOrbMaxFeatures.valueChanged.connect(self.spinBoxOrbMaxFeaturesChanged)

        # ---- Seed sane defaults ----------------------------------------------------------
        # A 1.4:1 low/high ratio produces fragmented, noisy Canny output. Canny's own
        # recommendation is 2:1 to 3:1.
        self.spinBoxCannyHighThreshold.setValue(150)
        self.spinBoxCannyLowThreshold.setValue(50)
        self.spinBoxCannyKernel.setValue(3)
        self.spinBoxSobelKernel.setValue(3)

        # ORB's useful range is hundreds, not hundreds of thousands.
        self.spinBoxOrbMaxFeatures.setMaximum(20000)
        self.spinBoxOrbMaxFeatures.setSingleStep(100)
        self.spinBoxOrbMaxFeatures.setValue(500)

        self._syncEdgeParams()
        self._syncFeatureParams()

        # The .ui marks radioButtonCanny checked=true, so the model must agree on open.
        self._setEdgeMethod(edgeMethodsClass.CANNY)

        self._loading = False
        self._emitEdge()

    # ==================================================================================================================
    # INTERNAL HELPERS
    # ==================================================================================================================
    def _coerceOdd(self, spinbox):
        """Force a kernel spinbox onto a legal OpenCV aperture size (3, 5, 7)."""
        v = spinbox.value()
        if v in self._LEGAL_KERNELS:
            return v
        legal = min(self._LEGAL_KERNELS, key=lambda k: abs(k - v))
        spinbox.blockSignals(True)
        spinbox.setValue(legal)
        spinbox.blockSignals(False)
        return legal

    # -----------------------------------------------------------------------------------------------------------------
    def _syncEdgeParams(self):
        """Copy every edge parameter from the widgets, enforcing OpenCV's constraints."""
        high = self.spinBoxCannyHighThreshold.value()
        low = self.spinBoxCannyLowThreshold.value()

        # cv2.Canny hysteresis is undefined-ish when low > high; keep the invariant explicit.
        if low >= high:
            low = max(0, high - 1)
            self.spinBoxCannyLowThreshold.blockSignals(True)
            self.spinBoxCannyLowThreshold.setValue(low)
            self.spinBoxCannyLowThreshold.blockSignals(False)

        self.returnEdgeData.canny_threshold_high = high
        self.returnEdgeData.canny_threshold_low = low

        # NOTE: add `canny_kernel` to edgeMethodsClass in constants.py -- it does not
        # currently exist, which is why the Canny kernel spinbox was inert.
        self.returnEdgeData.canny_kernel = self._coerceOdd(self.spinBoxCannyKernel)
        self.returnEdgeData.sobelKernel = self._coerceOdd(self.spinBoxSobelKernel)

    # -----------------------------------------------------------------------------------------------------------------
    def _syncFeatureParams(self):
        self.returnFeatureData.orbMaxFeatures = self.spinBoxOrbMaxFeatures.value()

    # -----------------------------------------------------------------------------------------------------------------
    def _setEdgeMethod(self, method):
        """Select an edge method and deactivate feature detection."""
        self.returnEdgeData.method = method
        self.returnEdgeData.selected = (method != edgeMethodsClass.NONE)

        self.returnFeatureData.method = featureMethodsClass.NONE
        self.returnFeatureData.selected = False

    # -----------------------------------------------------------------------------------------------------------------
    def _setFeatureMethod(self, method):
        """Select a feature method and deactivate edge detection."""
        self.returnFeatureData.method = method
        self.returnFeatureData.selected = (method != featureMethodsClass.NONE)

        self.returnEdgeData.method = edgeMethodsClass.NONE
        self.returnEdgeData.selected = False

    # -----------------------------------------------------------------------------------------------------------------
    def _emitEdge(self):
        """Emit a DEEP COPY so the receiver cannot be mutated out from under it."""
        if self._loading:
            return
        self.edgeDetectionSignal.emit(copy.copy(self.returnEdgeData))

    # -----------------------------------------------------------------------------------------------------------------
    def _emitFeature(self):
        if self._loading:
            return
        self.featureDetectionSignal.emit(copy.copy(self.returnFeatureData))

    # ==================================================================================================================
    # EDGE METHOD SELECTION
    # ==================================================================================================================
    def clicked_Canny(self):
        self._setEdgeMethod(edgeMethodsClass.CANNY)
        self._syncEdgeParams()
        self._emitEdge()

    # -----------------------------------------------------------------------------------------------------------------
    def clicked_SobelX(self):
        self._setEdgeMethod(edgeMethodsClass.SOBEL_X)
        self._syncEdgeParams()
        self._emitEdge()

    # -----------------------------------------------------------------------------------------------------------------
    def clicked_SobelY(self):
        self._setEdgeMethod(edgeMethodsClass.SOBEL_Y)
        self._syncEdgeParams()
        self._emitEdge()

    # -----------------------------------------------------------------------------------------------------------------
    def clicked_SobelXY(self):
        self._setEdgeMethod(edgeMethodsClass.SOBEL_XY)
        self._syncEdgeParams()
        self._emitEdge()

    # -----------------------------------------------------------------------------------------------------------------
    def clicked_Laplacian(self):
        self._setEdgeMethod(edgeMethodsClass.LAPLACIAN)
        self._syncEdgeParams()
        self._emitEdge()

    # ==================================================================================================================
    # FEATURE METHOD SELECTION
    # ==================================================================================================================
    def clicked_SIFT(self):
        self._setFeatureMethod(featureMethodsClass.SIFT)
        self._syncFeatureParams()
        self._emitFeature()

    # -----------------------------------------------------------------------------------------------------------------
    def clicked_ORB(self):
        self._setFeatureMethod(featureMethodsClass.ORB)
        self._syncFeatureParams()
        self._emitFeature()

    # ==================================================================================================================
    # PARAMETER CHANGES
    #
    # Each handler re-syncs ALL parameters, then emits only if the corresponding family of
    # method is actually the active one. Previously a Canny slider would fire an edge-detect
    # signal even while SIFT was selected, carrying a stale method/selected pair.
    # ==================================================================================================================
    def spinBoxCannyHighThresholdChanged(self):
        self._syncEdgeParams()
        if self.returnEdgeData.selected:
            self._emitEdge()

    # -----------------------------------------------------------------------------------------------------------------
    def spinBoxCannyLowThresholdChanged(self):
        self._syncEdgeParams()
        if self.returnEdgeData.selected:
            self._emitEdge()

    # -----------------------------------------------------------------------------------------------------------------
    def spinBoxCannyKernelChanged(self):
        self._syncEdgeParams()
        if self.returnEdgeData.selected:
            self._emitEdge()

    # -----------------------------------------------------------------------------------------------------------------
    def spinBoxSobelKernelChanged(self):
        self._syncEdgeParams()
        if self.returnEdgeData.selected:
            self._emitEdge()

    # -----------------------------------------------------------------------------------------------------------------
    def spinBoxOrbMaxFeaturesChanged(self):
        # Was: self.spinBoxOrbMaxFeaturesChanged().value()  -- infinite recursion.
        self._syncFeatureParams()
        if self.returnFeatureData.selected:
            self._emitFeature()

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def onCancel(self):
        self.close()
