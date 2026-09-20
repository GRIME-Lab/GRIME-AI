#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from appcore.utils.resource_utils import ui_path

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi

# ======================================================================================================================
#
# ======================================================================================================================
class MaskEditorDlg(QDialog):

    # SIGNALS TO MAIN APP TO PERFORM MASK EDITOR RELATED FUNCTIONS
    addMask_Signal = pyqtSignal()
    generateMask_Signal = pyqtSignal()
    drawingColorChange_Signal = pyqtSignal(int)
    reset_Signal = pyqtSignal()
    close_signal = pyqtSignal()
    polygonFill_Signal = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        loadUi(ui_path("mask_editor/QDialog_MaskEditor.ui"), self)

        self.pushButton_AddMask.setStyleSheet('QPushButton {background-color: steelblue;}')
        self.pushButton_GenerateMask.setStyleSheet('QPushButton {background-color: steelblue;}')
        self.pushButton_ResetMask.setStyleSheet('QPushButton {background-color: cyan; color: black;}')

        self.pushButton_AddMask.clicked.connect(self.pushButtonAddMaskClicked)
        self.pushButton_GenerateMask.clicked.connect(self.pushButtonGenerateMaskClicked)
        self.pushButton_ResetMask.clicked.connect(self.pushButtonResetMask)
        self.checkBox_FillPolygon.clicked.connect(self.fillPolygonClicked)
        self.buttonBox_Close.clicked.connect(self.closeClicked)

        # CHANGE CHECKBOX COLOR TO THE COLOR THEY REPRESENT
        self.radioButton_DrawingColor_Red.clicked.connect(self.pushButtonRed)
        self.radioButton_DrawingColor_Orange.clicked.connect(self.pushButtonOrange)
        self.radioButton_DrawingColor_Yellow.clicked.connect(self.pushButtonYellow)
        self.radioButton_DrawingColor_Green.clicked.connect(self.pushButtonGreen)
        self.radioButton_DrawingColor_Blue.clicked.connect(self.pushButtonBlue)
        self.radioButton_DrawingColor_Purple.clicked.connect(self.pushButtonPurple)

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    def closeEvent(self, event):
        super(MaskEditorDlg, self).closeEvent(event)
        self.close_signal.emit()

    def closeClicked(self):
        self.close_signal.emit()

    def pushButtonAddMaskClicked(self):
        self.addMask_Signal.emit()

    def pushButtonGenerateMaskClicked(self):
        self.generateMask_Signal.emit()

    def pushButtonResetMask(self):
        self.reset_Signal.emit()

    def fillPolygonClicked(self):
        self.polygonFill_Signal.emit(self.checkBox_FillPolygon.isChecked())

    # ------------------------------------------------------------------------------------------------------------------------
    # CHANGE CHECKBOX COLOR WHEN CHECKED
    # ------------------------------------------------------------------------------------------------------------------------
    def pushButtonRed(self):
        self.radioButton_DrawingColor_Red.setStyleSheet("QRadioButton::indicator:checked" "{" "background-color : red" "}")
        self.drawingColorChange_Signal.emit(Qt.red)

    def pushButtonOrange(self):
        self.radioButton_DrawingColor_Orange.setStyleSheet("QRadioButton::indicator:checked" "{" "background-color : magenta" "}")
        self.drawingColorChange_Signal.emit(Qt.magenta)

    def pushButtonYellow(self):
        self.radioButton_DrawingColor_Yellow.setStyleSheet("QRadioButton::indicator:checked" "{" "background-color : yellow" "}")
        self.drawingColorChange_Signal.emit(Qt.yellow)

    def pushButtonGreen(self):
        self.radioButton_DrawingColor_Green.setStyleSheet("QRadioButton::indicator:checked" "{" "background-color : green" "}")
        self.drawingColorChange_Signal.emit(Qt.green)

    def pushButtonBlue(self):
        self.radioButton_DrawingColor_Blue.setStyleSheet("QRadioButton::indicator:checked" "{" "background-color : blue" "}")
        self.drawingColorChange_Signal.emit(Qt.blue)

    def pushButtonPurple(self):
        self.radioButton_DrawingColor_Purple.setStyleSheet("QRadioButton::indicator:checked" "{" "background-color : purple" "}")
        self.drawingColorChange_Signal.emit(Qt.blue)

    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    def getCheckBox_Save(self):
        return self.checkBox_MaskAutoSave.isChecked()

    def getEnablePolygonFill(self):
        return self.checkBox_FillPolygon.isChecked()