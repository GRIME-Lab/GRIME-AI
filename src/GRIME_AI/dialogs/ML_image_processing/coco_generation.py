#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ======================================================================================================================
# File: COCOGeneration.py
# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Nov 26, 2025
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
# ======================================================================================================================

from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox
from PyQt5.uic import loadUi

from appcore.CSS_Styles import BUTTON_CSS_STEEL_BLUE
from appcore.coco_generator import CocoGenerator
from appcore.utils.resource_utils import ui_path


class COCOGeneration(QDialog):
    """
    Direct port of COCO 1.0 Generation functionality.
    Loads its own UI and handles folder selection, mask file selection,
    button state updates, and annotation generation using CocoGenerator.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Load the dedicated COCO Generation UI
        loadUi(ui_path("ML_image_processing/coco_generation.ui"), self)

        # Connect signals exactly as in the original snippet
        self.lineEdit_cocoFolder.textChanged.connect(self.updateCOCOButtonState)
        self.pushButton_cocoBrowse.clicked.connect(self.selectCocoFolder)
        self.pushButton_cocoBrowse.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.lineEdit_maskFile.textChanged.connect(self.updateCOCOButtonState)
        self.pushButton_maskBrowse.clicked.connect(self.selectMaskFile)
        self.pushButton_maskBrowse.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.checkBox_singleMask.toggled.connect(self.updateMaskFieldState)
        self.pushButton_generateCOCO.clicked.connect(self.generateCOCOAnnotations)
        self.pushButton_generateCOCO.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.updateMaskFieldState(self.checkBox_singleMask.isChecked())
        self.updateCOCOButtonState()

        self.buttonBox_close.rejected.connect(self.reject)

    # ..........................................................................
    def updateCOCOButtonState(self):
        """Enable the Generate COCO button only if both folder and mask file are provided."""
        coco_folder = self.lineEdit_cocoFolder.text().strip()
        mask_file = self.lineEdit_maskFile.text().strip()
        self.pushButton_generateCOCO.setEnabled(bool(coco_folder and mask_file))

    def updateMaskFieldState(self, single_mask: bool):
        """Enable/disable mask file field depending on singleMask checkbox."""
        self.lineEdit_maskFile.setEnabled(single_mask)
        self.pushButton_maskBrowse.setEnabled(single_mask)

    def selectCocoFolder(self):
        """Open QFileDialog to select COCO output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select COCO Output Folder")
        if folder:
            self.lineEdit_cocoFolder.setText(folder)

    def selectMaskFile(self):
        """Open QFileDialog to select mask file."""
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Mask File",
            "",
            "Image Files (*.png *.jpg *.jpeg)"
        )
        if file:
            self.lineEdit_maskFile.setText(file)

    def generateCOCOAnnotations(self):
        """
        Generate COCO 1.0 annotations from mask files.
        Uses CocoGenerator to produce JSON in the selected folder.
        """
        coco_folder = self.lineEdit_cocoFolder.text().strip()
        mask_file = self.lineEdit_maskFile.text().strip()
        single_mask = self.checkBox_singleMask.isChecked()

        if not coco_folder:
            QMessageBox.warning(self, "Missing Folder", "You must select a COCO output folder.")
            return
        if single_mask and not mask_file:
            QMessageBox.warning(self, "Missing Mask File", "You must select a mask file when singleMask is checked.")
            return

        try:
            generator = CocoGenerator()
            if single_mask:
                generator.generate_from_single_mask(mask_file, coco_folder)
            else:
                generator.generate_from_folder(coco_folder)
            QMessageBox.information(self, "COCO Generation", "COCO annotations generated successfully.")
        except Exception as e:
            QMessageBox.critical(self, "COCO Generation Error", f"Failed to generate COCO annotations:\n{e}")
