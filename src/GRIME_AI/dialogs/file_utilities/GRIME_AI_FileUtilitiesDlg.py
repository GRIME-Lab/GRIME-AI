#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os
import promptlib
import datetime

from GRIME_AI.utils.resource_utils import ui_path

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi

from GRIME_AI.GRIME_AI_Utils import GRIME_AI_Utils
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor
from GRIME_AI.GRIME_AI_Video import GRIME_AI_Video


# ======================================================================================================================
#
# ======================================================================================================================
class Datapaths():

    def __init__(self, parent=None):
        self.imageInputFolder = ""
        self.imageOutputFolder = ""
        self.videoInputFolder = ""
        self.videoOutputFolder = ""
        self.gifInputFolder = ""
        self.gifOutputFolder = ""


# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_FileUtilitiesDlg(QDialog):

    # ------------------------------------------------------------------------------------------------------------------
    # SIGNALS
    # ------------------------------------------------------------------------------------------------------------------
    fetchImageList_Signal = pyqtSignal(str, int)
    create_composite_slice_signal = pyqtSignal()
    triage_images_signal = pyqtSignal(str)
    # Emitted when the USER commits an image-folder change (Browse or
    # edit-finished) so the main window can offer to update the active recipe.
    imageFolderCommitted_Signal = pyqtSignal(str)

    # ------------------------------------------------------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)

        loadUi(ui_path("file_utilities/QDialog_FileUtilities.ui"), self)

        self.accepted.connect(self.closeFileFolderDlg)
        self.rejected.connect(self.closeFileFolderDlg)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.fetchImageList_Signal.connect(parent.fetchImageList)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # CREATE VIDEO OR GIF FROM INDIVIDUAL IMAGES
        # ----------------------------------------------------------------------------------------------------
        # CREATE A VIDEO
        self.pushButton_CreateVideo.clicked.connect(self.pushButton_create_video_clicked)
        self.pushButton_CreateVideo.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')


        # CREATE A GIF *** NOTE!!! ADD A DIALOG BOX TO INPUT A DELAY BETWEEN IMAGES ***
        self.pushButton_CreateGIF.clicked.connect(self.pushButtonCreateGIFClicked)
        self.pushButton_CreateGIF.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')

        self.pushButton_BrowseImageFolder.clicked.connect(self.pushButtonBrowseImageFolderClicked)
        self.lineEdit_images_folder.textChanged.connect(self.image_folder_changed)
        # A committed edit (Enter / focus-out) is a user action; programmatic
        # setText (e.g. recipe activation) does NOT fire editingFinished.
        self.lineEdit_images_folder.editingFinished.connect(self._image_folder_committed)
        if hasattr(parent, "on_images_folder_committed"):
            self.imageFolderCommitted_Signal.connect(parent.on_images_folder_committed)

        self.pushButton_triage_images.clicked.connect(self.pushButton_triage_images_clicked)
        self.pushButton_triage_images.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')

        self.pushButton_CreateEXIFFile.clicked.connect(self.MSTExtractFrames)
        self.pushButton_CreateEXIFFile.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')


        # VIDEO FUNCTIONS
        self.pushButton_ExtractFrames.clicked.connect(self.MSTExtractFrames)
        self.pushButton_ExtractFrames.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')

        self.pushButton_FetchImageList.clicked.connect(self.pushButtonFetchImageListClicked)
        self.pushButton_FetchImageList.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')

        image_folder = JsonEditor().getValue("Local_Image_Folder")
        self.setImageFolderPath(image_folder)

        self.buttonBox.setStyleSheet('QPushButton {background-color: lightblue; color: white;}')

        self.pushButton_create_composite_slice.clicked.connect(self.pushButton_create_composite_slice_clicked)
        self.pushButton_create_composite_slice.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')


    def closeFileFolderDlg(self):
        # BEFORE CLOSING THE DIALOG BOX, SAVE THE FOLDERS TO THE SETTINGS FILE
        image_folder = self.lineEdit_images_folder.text()
        JsonEditor().update_json_entry("Local_Image_Folder", image_folder)
        # A pending, uncommitted edit (typed then closed without focus-out) still
        # needs to reconcile with the active recipe.
        self._image_folder_committed()


    @pyqtSlot()
    def image_folder_changed(self):
        # LET THE MAIN APPLICATION KNOW THAT THE FOLDER PATH FOR THE IMAGES HAS CHANGED BY SENDING IT THE PATH
        image_folder = self.lineEdit_images_folder.text()
        JsonEditor().update_json_entry("Local_Image_Folder", image_folder)

    def _image_folder_committed(self):
        # Fired on a user-committed edit; ask the main window to reconcile
        # the change with the active recipe (if any).
        self.imageFolderCommitted_Signal.emit(self.lineEdit_images_folder.text())


    def pushButtonFetchImageListClicked(self):
        if len(self.lineEdit_images_folder.text()) > 0:
            self.fetchImageList_Signal.emit(self.lineEdit_images_folder.text(), self.checkBox_FetchRecursive.isChecked())
        else:
            self.pushButtonBrowseImageFolderClicked()
            self.fetchImageList_Signal.emit(self.lineEdit_images_folder.text(), self.checkBox_FetchRecursive.isChecked())

    def checkboxNEONSitesClicked(self):
        pass

    def MSTExtractFrames(self):
        pass

    def pushButtonBrowseSaveImagesOutputFolderClicked(self):
        pass

    def pushButton_create_video_clicked(self):
        myGRIMEAI_video = GRIME_AI_Video()
        myGRIMEAI_video.createVideo(self.lineEdit_images_folder.text())

    def pushButtonCreateGIFClicked(self):
        myGRIMEAI_video = GRIME_AI_Video()
        myGRIMEAI_video.createGIF(self.lineEdit_images_folder.text())

    def pushButton_create_composite_slice_clicked(self):
        self.create_composite_slice_signal.emit()

    def pushButton_triage_images_clicked(self):
        self.triage_images_signal.emit(self.lineEdit_images_folder.text())


    # --------------------------------------------------
    # IMAGE INPUT FOLDER
    # --------------------------------------------------
    def pushButtonBrowseImageFolderClicked(self):
        prompter = promptlib.Files()
        folder = prompter.dir()

        if os.path.exists(folder):
            self.lineEdit_images_folder.setText(folder)
            self._image_folder_committed()
            #self.radioButtonHardDriveImages.setChecked(True)
            #self.checkBoxCreateEXIFFile.setEnabled(True)

            # RECURSE AND TRAVERSE FOLDERS FROM ROOT DOWNWARD TO GET A LIST OF
            #JES startDate, endDate = getLocalFileDates(folder, self.checkBox_FetchRecursive.isChecked())

            #JES updateProductTableDateRange(self.tableProducts, 0, startDate, endDate)

            #JES processLocalImage(self)


    def setImageFolderPath(self, image_folder):
        self.lineEdit_images_folder.setText(image_folder)


# ======================================================================================================================
#
# ======================================================================================================================
def getLocalFileDates(filePath, bFetchRecursive):
    # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
    extensions = ('.jpg', '.jpeg', '.png')

    # INITIALIZE VARIABLES TO A KNOWN VALUE
    startDate = datetime.date(2500, 12, 31)
    endDate   = datetime.date(1970, 1, 1)

    myGRIME_AI_Utils = GRIME_AI_Utils()
    file_count, files = myGRIME_AI_Utils.getFileList(filePath, extensions, bFetchRecursive)

    for file in files:
        ext = os.path.splitext(file)[-1].lower()

        if ext in extensions:
            fileDate, fileTime = myGRIME_AI_Utils.extractDateFromFilename(file)

            # use the date in the filenames to determine the start and end acquisition dates for the images
            if fileDate < startDate:
                startDate = fileDate

            if fileDate > endDate:
                endDate = fileDate

    return startDate, endDate

