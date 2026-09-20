#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os

from appcore.utils.resource_utils import ui_path

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi
import promptlib

from appcore.App_QMessageBox import App_QMessageBox

# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===   class ExportCOCOMasksDlg    ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class ExportCOCOMasksDlg(QDialog):

    # SIGNALS
    # ------------------------------------------------------------------------------------------------------------------
    COCO_signal_ok = pyqtSignal()
    COCO_signal_cancel = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)

        loadUi(ui_path("extract_coco_masks/QDialog_ExtractCOCOMasks.ui"), self)

        self.annotationImagesFolder = ''

        self.pushButton_BrowseAnnotationImages.clicked.connect(self.pushButton_BrowseAnnotationImagesClicked)

        self.lineEdit_AnnotationImagesFolder.textChanged.connect(self.annotation_images_folder_changed)

        self.buttonBox_ok_cancel.accepted.connect(self.accepted)
        self.buttonBox_ok_cancel.rejected.connect(self.rejected)

        self.pushButton_BrowseAnnotationImages.setStyleSheet('QPushButton {background-color: steelblue;}')

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def annotation_images_folder_changed(self):
        """
        Called whenever the text in lineEdit_AnnotationImagesFolder changes.
        Updates self.annotationImagesFolder and populates listWidget_Folders
        with any subfolders containing an instances_default.json file.
        """
        self.annotationImagesFolder = self.lineEdit_AnnotationImagesFolder.text().strip()

        # Clear the listbox first
        self.listWidget_Folders.clear()

        if not self.annotationImagesFolder or not os.path.isdir(self.annotationImagesFolder):
            return

        # Walk the directory tree
        for root, dirs, files in os.walk(self.annotationImagesFolder):
            if "instances_default.json" in files:
                # Add the folder path to the list widget
                self.listWidget_Folders.addItem(root)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def pushButton_BrowseAnnotationImagesClicked(self):
        prompter = promptlib.Files()

        folder = prompter.dir()

        if os.path.exists(folder):
            if os.listdir(folder):
                self.lineEdit_AnnotationImagesFolder.setText(folder)
                self.annotationImagesFolder = self.lineEdit_AnnotationImagesFolder.text()
            else:
                msgBox = App_QMessageBox('COCO Error', 'No files or no supported files types in folder.')
                response = msgBox.displayMsgBox()


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def accepted(self):
        #msgBox = App_QMessageBox('COCO Error', 'No files or no supported files types in folder.')
        #msgBox.displayMsgBox()

        if self.annotationImagesFolder:
            self.COCO_signal_ok.emit()
            super().accept()
        else:
            if not os.path.exists(self.annotationImagesFolder):
                msgBox = App_QMessageBox('Images Folder', 'Mask extraction aborted! No images folder specified or no images found.')
                msgBox.displayMsgBox()

            self.COCO_signal_cancel.emit()
            super().reject()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def rejected(self):
        #super(ExportCOCOMasksDlg, self).closeEvent(event)
        self.COCO_signal_cancel.emit()
        super().reject()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @pyqtSlot()
    def image_folder_changed(self):
        pass
        # LET THE MAIN APPLICATION KNOW THAT THE FOLDER PATH FOR THE IMAGES HAS CHANGED BY SENDING IT THE PATH
        #image_folder = self.lineEditImageFolder.text()
        #JsonEditor().update_json_entry("Local_Image_Folder", image_folder)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def getAnnotationImagesFolder(self):
        """
        Return a list of all folder paths currently in listWidget_Folders.
        """
        folders = []
        for i in range(self.listWidget_Folders.count()):
            folders.append(self.listWidget_Folders.item(i).text())

        return folders

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def pushButtonBrowseSaveImagesOutputFolderClicked(self):
        prompter = promptlib.Files()
        folder = prompter.dir()

        if os.path.exists(folder):
            self.EditSaveImagesOutputFolder(prompter.dir())

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def setImageFolderPath(self, image_folder):
        self.lineEditImageFolder.setText(image_folder)

