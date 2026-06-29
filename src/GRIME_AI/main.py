#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# !/usr/bin/env python
# !/usr/bin/env python
# coding: utf-8

# THIS TO INVESTIGATE
# https://github.com/smhassanerfani/atlantis/tree/master/aquanet
# https://github.com/smhassanerfani/atlantis/tree/master



# 1. SCRAPE FIELD SITE TABLE (CSV) FILE FROM https://www.neonscience.org/field-sites/explore-field-sites
#
# 2. PERSIST A COPY OF THE CONTENTS OF THE FIELD SITE TABLE
#
# 2. POPULATE LISTBOX WITH SITES
#
# 3. SELECT A SITE
#
# 4. QUERY NEON SITE FOR INFO ON SELECTED SITE
#
# 5. DOWNLOAD DATA
#

# cv2.mahalanobis
# matplotlib.use('Qt5Agg')

# import pycurl
# main.py – at the very top

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import os
# Configure Qt High DPI behavior for consistent UI scaling across displays.
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

try:
    # TRY TO IMPORT THE VERSION CONSTANT FROM VERSION.PY
    from GRIME_AI.version import SW_VERSION
except ImportError:
    # FALLBACK IF VERSION.PY DOES NOT EXIST. 0.0.0.0 IS AN INVALID VERSION NUMBER
    SW_VERSION = "Ver. 0.0.0.0"

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import logging

# Optional: set up logging so you can track initialization
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# torch imported lazily at first use (CLI segmentation path)

try:
    import sklearn.neighbors._typedefs
except Exception:
    pass

try:
    import sklearn.neighbors._partition_nodes
except Exception:
    pass

try:
    import sklearn.utils._weight_vector
except Exception:
    pass

try:
    import sklearn.neighbors._quad_tree
except Exception:
    pass

try:
    import skimage
except Exception:
    pass

try:
    import imageio
except Exception:
    pass

try:
    import imageio_ffmpeg
except Exception:
    pass


# ------------------------------------------------------------------------------
# Configure environment variables first. Especially those that imports rely upon
# ------------------------------------------------------------------------------
import os
# Must be set before Hydra starts to get the full stack trace if an error occurs.
# This works programmtically in case the user does not have privileges to modify their operating system environment
# because it only modifies the environment of the current Python process and its children.
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

import sys
import argparse
import shutil

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def excepthook(exc_type, exc_value, exc_tb):
    print("Uncaught exception:")
    traceback.print_exception(exc_type, exc_value, exc_tb)

sys.excepthook = excepthook

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
import datetime
from datetime import date
import time

import csv

import urllib.request

import promptlib

import traceback

from dataclasses import dataclass

from GRIME_AI.utils.resource_utils import icon_path

from pathlib import Path

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
from PyQt5 import QtCore, QtWidgets, QtGui, uic
from PyQt5.QtCore import QCoreApplication, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QToolBar, QDateTimeEdit, \
    QMessageBox, QAction, QHeaderView, QDialog, QFileDialog, QSplashScreen
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMenu
from PyQt5.QtWidgets import QTreeWidgetItem

from GRIME_AI.GRIME_AI_SplashScreen import GRIME_AI_SplashScreen

# pandas imported lazily inside functions

from GRIME_AI.usgs.usgs_client import USGSClient

# ----------------------------------------------------------------------------
# POP-UP/MODELESS DIALOG BOXES
# ----------------------------------------------------------------------------
# lazy: from GRIME_AI.dialogs.color_segmentation.GRIME_AI_ColorSegmentationDlg import GRIME_AI_ColorSegmentationDlg
# lazy: from GRIME_AI.dialogs.edge_detection.GRIME_AI_EdgeDetectionDlg import GRIME_AI_EdgeDetectionDlg
# lazy: from GRIME_AI.dialogs.image_navigation.GRIME_AI_ImageNavigationDlg import GRIME_AI_ImageNavigationDlg
# lazy: from GRIME_AI.dialogs.file_utilities.GRIME_AI_FileUtilitiesDlg import GRIME_AI_FileUtilitiesDlg
# lazy: from GRIME_AI.dialogs.mask_editor.GRIME_AI_MaskEditorDlg import GRIME_AI_MaskEditorDlg
# lazy: from GRIME_AI.dialogs.composite_slice.GRIME_AI_CompositeSliceDlg import GRIME_AI_CompositeSliceDlg
# lazy: from GRIME_AI.dialogs.release_notes.GRIME_AI_ReleaseNotesDlg import GRIME_AI_ReleaseNotesDlg
# lazy: from GRIME_AI.dialogs.extract_coco_masks.GRIME_AI_ExportCOCOMasksDlg import GRIME_AI_ExportCOCOMasksDlg
# lazy: from GRIME_AI.dialogs.image_organizer.GRIME_AI_ImageOrganizerDlg import GRIME_AI_ImageOrganizerDlg
# lazy: from GRIME_AI.dialogs.temporal_averaging.GRIME_AI_TemporalAveragingDlg import GRIME_AI_TemporalAveragingDlg
from GRIME_AI.dialogs.triage.GRIME_AI_TriageOptionsDlg import GRIME_AI_TriageOptionsDlg
from GRIME_AI.GRIME_AI_Color import GRIME_AI_Color
from GRIME_AI.GRIME_AI_Vegetation_Indices import GRIME_AI_Vegetation_Indices, GreennessIndex
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor
from GRIME_AI.GRIME_AI_ImageData import imageData
from GRIME_AI.GRIME_AI_ProductTable import GRIME_AI_ProductTable
from GRIME_AI.GRIME_AI_QLabel import DrawingMode
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_Utils import GRIME_AI_Utils
from GRIME_AI.GRIME_AI_roiData import GRIME_AI_roiData, ROIShape
from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI.GRIME_AI_Resize_Controls import GRIME_AI_Resize_Controls
from GRIME_AI.GRIME_AI_TimeStamp_Utils import GRIME_AI_TimeStamp_Utils
from GRIME_AI.GRIME_AI_ImageTriage import GRIME_AI_ImageTriage
from GRIME_AI.GRIME_AI_GreenImageGenerator import GreenImageGenerator
from GRIME_AI.GRIME_AI_COCO_Utils import GRIME_AI_COCO_Utils

# ----------------------------------------------------------------------------
# PHENOCAM
# ----------------------------------------------------------------------------
from GRIME_AI.phenocam.GRIME_AI_PhenoCam import GRIME_AI_PhenoCam, dailyList
from GRIME_AI.phenocam.GRIME_AI_Phenocam_API import GRIME_AI_Phenocam_API

from GRIME_AI.colorSegmentationParams import colorSegmentationParamsClass

# ----------------------------------------------------------------------------
# DIGITAL MAPPING PLATFORMS
# ----------------------------------------------------------------------------
from GRIME_AI.geomaps.google_maps_viewer import GoogleMapWidget
from GRIME_AI.geomaps.openstreetmap_viewer import OpenStreetMapWidget

# ----------------------------------------------------------------------------------------------------------------------
# HYDRA (for SAM2)
# ----------------------------------------------------------------------------------------------------------------------
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from GRIME_AI.neon.NEON_API import NEON_API
from GRIME_AI.dialogs.api_keys import APIKeyManager, APIKeyDialog

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from GRIME_AI.constants import edgeMethodsClass, featureMethodsClass, modelSettingsClass

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from GRIME_AI.exifData import EXIFData

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
global bStartupComplete
bStartupComplete = False

global bShow_GUI
bShow_GUI = False

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
import os
import cv2
import numpy as np

# ======================================================================================================================
#
# ======================================================================================================================
SITECODE = 'ARIK'
DOMAINCODE = 'D10'
originalImg = []
dailyImagesList = dailyList([], [])
currentImage = []
currentImageIndex = -1
siteList = []
nStop = 0
gWebImageCount = 0
gWebImagesAvailable = 0
gFrameCount = 0
gProcessClick = 0
currentImageFilename = ""
frame = []
# Define the maximum number of gray levels
gray_level = 16

# URLS
# url = "http://maps.googleapis.com/maps/api/geocode/json?address=googleplex&sensor=false"
url = 'https://www.neonscience.org/field-sites/explore-field-sites'
root_url = 'https://www.neonscience.org'
SERVER = 'http://data.neonscience.org/api/v0/'

class displayOptions():
    displayROIs = True

g_displayOptions   = displayOptions()
g_edgeMethodSettings = edgeMethodsClass()
g_featureMethodSettings = featureMethodsClass()

g_modelSettings = modelSettingsClass()

hyperparameterDlg = None

# ======================================================================================================================
# PHENOCAM BACKGROUND WORKERS
# ======================================================================================================================
class PhenocamPreviewFetcher(QtCore.QThread):
    result = QtCore.pyqtSignal(object, str)
    def __init__(self, site_name, parent=None, roi_name=None):
        super().__init__(parent)
        self.site_name = site_name
        self.roi_name  = roi_name  # e.g. "NEON.D01.BART.DP1.00033_DB_1000"
    def run(self):
        import urllib.request, datetime as dt_mod
        from GRIME_AI.phenocam.GRIME_AI_PhenoCam import GRIME_AI_PhenoCam
        today = dt_mod.date.today()
        t0, t1 = dt_mod.time(11, 0), dt_mod.time(13, 0)
        # Extract ROI type prefix for filename filtering (e.g. "DB" from "..._DB_1000")
        roi_filter = None
        if self.roi_name:
            parts = self.roi_name.split("_")
            # roi_name format: sitename_TYPE_NNNN — TYPE is second-to-last segment
            if len(parts) >= 2:
                roi_filter = parts[-2]  # e.g. "DB", "EN", "SH", etc.
        img_url = None
        for offset in range(30):
            check = today - dt_mod.timedelta(days=offset)
            url = (f"https://phenocam.nau.edu/webcam/browse/{self.site_name}/"
                   f"{check.year}/{str(check.month).zfill(2)}/{str(check.day).zfill(2)}")
            try:
                imgs = GRIME_AI_PhenoCam().getVisibleImages(url, t0, t1).getVisibleList()
                if imgs:
                    if roi_filter:
                        # Filter to images whose filename contains the ROI type
                        filtered = [i for i in imgs
                                    if f"_{roi_filter}_" in i.fullPathAndFilename
                                    or f"_{roi_filter}." in i.fullPathAndFilename]
                        imgs = filtered if filtered else imgs
                    img_url = imgs[-1].fullPathAndFilename
                    break
            except Exception:
                pass
        if img_url is None:
            self.result.emit(None, "No recent images found.")
            return
        try:
            data = urllib.request.urlopen(img_url, timeout=15).read()
            qimg = QtGui.QImage()
            qimg.loadFromData(data)
            self.result.emit(QtGui.QPixmap.fromImage(qimg) if not qimg.isNull() else None,
                             "" if not qimg.isNull() else "Could not decode image.")
        except Exception as e:
            self.result.emit(None, f"Error: {e}")


class PhenocamDownloadWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(int)
    def __init__(self, site_name, start_dt, end_dt, save_folder, parent=None):
        super().__init__(parent)
        self.site_name, self.start_dt = site_name, start_dt
        self.end_dt, self.save_folder = end_dt, save_folder
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import urllib.request, os, datetime as dt_mod
        from GRIME_AI.phenocam.GRIME_AI_PhenoCam import GRIME_AI_PhenoCam

        # All PhenoCam filename timestamps are local time (both NEON via PhenoCam
        # and strictly PhenoCam sites), so the user's local start/end times are
        # compared directly — no UTC conversion needed.
        start_date, end_date = self.start_dt.date(), self.end_dt.date()
        start_time, end_time = self.start_dt.time(), self.end_dt.time()

        image_list = []
        total_days = (end_date - start_date).days + 1

        current, day_idx = start_date, 0
        while current <= end_date:
            if self._cancelled:
                self.finished.emit(-1)
                return

            day_idx += 1
            t0, t1 = start_time, end_time

            url = (
                f"https://phenocam.nau.edu/webcam/browse/{self.site_name}/"
                f"{current.year}/{str(current.month).zfill(2)}/{str(current.day).zfill(2)}"
            )

            try:
                image_list.extend(
                    GRIME_AI_PhenoCam().getVisibleImages(url, t0, t1).getVisibleList()
                )
            except Exception as e:
                print(f"[PhenocamDownloadWorker] Error scanning {current}: {e}")

            self.progress.emit(day_idx, total_days, f"Scanning {current.strftime('%Y-%m-%d')}...")
            current += dt_mod.timedelta(days=1)

        if not image_list:
            self.finished.emit(0)
            return

        os.makedirs(self.save_folder, exist_ok=True)

        total, downloaded = len(image_list), 0
        for i, img in enumerate(image_list):
            if self._cancelled:
                self.finished.emit(-1)
                return

            filename = os.path.basename(img.fullPathAndFilename)
            dest = os.path.join(self.save_folder, filename)

            if not os.path.isfile(dest):
                try:
                    urllib.request.urlretrieve(img.fullPathAndFilename, dest)
                    downloaded += 1
                except Exception as e:
                    print(f"[PhenocamDownloadWorker] Error downloading {filename}: {e}")

            self.progress.emit(i + 1, total, f"Downloading {filename}")

        self.finished.emit(downloaded)


class NEONPreviewFetcher(QtCore.QThread):
    result = QtCore.pyqtSignal(object, str)
    def __init__(self, site_code, domain_code, product_id=20002, parent=None):
        super().__init__(parent)
        self.site_code = site_code
        self.domain_code = domain_code
        self.product_id = product_id
    def run(self):
        from GRIME_AI.neon.NEON_API import NEON_API as _NEON_API
        try:
            nErrorCode, pixmap, count = _NEON_API().DownloadLatestImage(
                self.site_code, self.domain_code, self.product_id)
            if nErrorCode == 200 and count > 0 and pixmap and not pixmap.isNull():
                self.result.emit(pixmap, "")
            else:
                self.result.emit(None, "No latest image available.")
        except Exception as e:
            self.result.emit(None, f"Error: {e}")


class USGSLatestImageFetcher(QtCore.QThread):
    """Fetches the midday USGS camera image in a background thread."""
    result = QtCore.pyqtSignal(int, object, bool)  # (error_code, QPixmap or None, is_midday)

    def __init__(self, usgs_client, cam_id: str, parent=None):
        super().__init__(parent)
        self._usgs   = usgs_client
        self._cam_id = cam_id

    def run(self):
        try:
            code, pix, is_midday = self._usgs.get_midday_pixmap(self._cam_id)
            self.result.emit(code, pix, is_midday)
        except Exception:
            self.result.emit(404, None, False)


class PhenocamStartupFetcher(QtCore.QThread):
    """Fetches the PhenoCam site table in a background thread at startup."""
    result = QtCore.pyqtSignal(object)  # dict or None

    def __init__(self, neon_api, parent=None):
        super().__init__(parent)
        self._neon_api = neon_api

    def run(self):
        try:
            data = self._neon_api.scrape_phenocam_table()
            self.result.emit(data)
        except Exception as e:
            print(f"[PhenocamStartupFetcher] Error: {e}")
            self.result.emit(None)


class USGSStartupFetcher(QtCore.QThread):
    """Fetches USGS HIVIS camera dictionary in a background thread at startup."""
    result = QtCore.pyqtSignal(object, object, object)  # (hivis, cameraDictionary, cameraList)

    def __init__(self, usgs_client, parent=None):
        super().__init__(parent)
        self._usgs = usgs_client

    def run(self):
        try:
            from GRIME_AI.usgs.usgs_hivis import USGS_HIVIS
            hivis    = USGS_HIVIS()
            cam_dict = hivis.get_camera_dictionary()
            cam_list = hivis.get_camera_list()
            self.result.emit(hivis, cam_dict, cam_list)
        except Exception as e:
            print(f"[USGSStartupFetcher] Error: {e}")
            self.result.emit(None, {}, [])


class NEONStartupFetcher(QtCore.QThread):
    """Fetches NEON field site table in a background thread at startup."""
    result = QtCore.pyqtSignal(object, object)  # (status, siteList)

    def __init__(self, neon_api, parent=None):
        super().__init__(parent)
        self._neon_api = neon_api

    def run(self):
        try:
            status, site_list = self._neon_api.readFieldSiteTable()
            self.result.emit(status, site_list)
        except Exception as e:
            print(f"[NEONStartupFetcher] Error: {e}")
            self.result.emit(None, [])


class NWISParameterFetcher(QtCore.QThread):
    """
    Fetches available NWIS time series parameters for a given nwisId
    in a background thread so the UI remains responsive.
    """
    finished = QtCore.pyqtSignal(list, str)   # (params, cam_id)

    def __init__(self, usgs_service, nwis_id: str, cam_id: str, parent=None):
        super().__init__(parent)
        self._svc    = usgs_service
        self._nwis_id = nwis_id
        self._cam_id  = cam_id

    def run(self):
        params = self._svc.get_available_parameters(self._nwis_id)
        self.finished.emit(params, self._cam_id)


# ======================================================================================================================
#
# ======================================================================================================================
class MainWindow(QMainWindow):
    xStart = 0
    yStart = 0
    roiList = []
    imageStatsList = []

    # INITIALIZE POP-UP DIALOG BOXES
    fileFolderDlg        = None
    edgeDetectionDlg     = None
    colorSegmentationDlg = None
    TriageDlg            = None
    maskEditorDlg        = None
    compositeSliceDlg    = None
    imageNavigationDlg   = None
    releaseNotesDlg      = None
    buildModelDlg        = None


    imageFileFolder = None

    global dailyImagesList
    dailyImagesList = dailyList([], [])

    NEON_siteList = []
    NEON_latestImage = []


    USGS_latestImage = []

    # def eventFilter(self, source, event):
    #     if (event.type() == QtCore.QEvent.MouseMove and source is self.label):
    #         pos = event.pos()
    #         print('mouse move: (%d, %d)' % (pos.x(), pos.y()))
    #
    #     if (event.type() == QtCore.QEvent.MouseButtonDblClick and source is self.label):
    #         print('Double click')
    #
    #     return QtGui.QWidget.eventFilter(self, source, event)


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def resizeEvent(self, event):

        # PARENT CLASS WHICH CONTAINS ALL FUNCTIONS TO RESIZE ALL CONTROLS ON THE GUI, AS NEEDED
        resizeControls = GRIME_AI_Resize_Controls()

        # TAB 0 - NEON SITES
        resizeControls.resizeTab_0(self, event)
        self.NEON_DisplayLatestImage()

        # TAB 2 - USGS SITES — keep horizontal splitter 50/50 until user moves it
        if not getattr(self, '_usgs_splitter_moved_flag', False):
            half = event.size().width() // 2
            self.splitter_USGS_Horizontal.setSizes([half, half])

        # TAB 2 - USGS SITES — keep vertical splitter 75/25 until user moves it
        if not getattr(self, '_usgs_vertical_splitter_moved_flag', False):
            h = event.size().height()
            self.splitter_USGS_Vertical.setSizes([int(h * 0.65), int(h * 0.35)])

        # TAB 0 - NEON SITES — keep splitters 50/50 until user moves them
        if not getattr(self, '_neon_splitter_moved_flag', False):
            half = event.size().width() // 2
            self.splitter_NEON_Top.setSizes([half, half])
            self.splitter_NEON_Bottom.setSizes([half, half])

        # TAB 1 - NEON DOWNLOAD MANAGER
        resizeControls.resizeTab_1(self, event)

        # TAB 2 - USGS SITES
        resizeControls.resizeTab_2(self, event)

        # TAB 3 - USGS DOWNLOAD MANAGER
        resizeControls.resizeTab_3(self, event)

        # TAB 4 - IMAGE ANALYSIS
        resizeControls.resizeTab_4(self, event)

        # TAB 5 - SENSOR DATA GRAPHS

        self._phenocam_fit_preview_height()

        #QtWidgets.resizeEvent(self, event)

    def showEvent(self, event):
        super().showEvent(event)
        QtCore.QTimer.singleShot(100, self._phenocam_fit_preview_height)
        QtCore.QTimer.singleShot(200, self.NEON_DisplayLatestImage)

    # ------------------------------------------------------------------------------------------------------------------
    # CLASS INITIALIZATION
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, parent=None, win=None, session=None, splash=None):
        super(MainWindow, self).__init__(parent)
        self.mainwin = win
        self.session = session
        self._splash = splash
        self._usgs_startup_ready = False
        self._neon_startup_ready = False
        self._phenocam_startup_ready = False
        ui_path = os.path.join(os.path.dirname(__file__), "resources", "ui", "neonAIgui.ui")
        uic.loadUi(ui_path, self)

        self.setWindowTitle("GRIME AI" + " " + SW_VERSION + " - John E. Stranzl Jr.")
        self.tabWidget.setTabVisible(1, False)
        #self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowStaysOnTopHint)

        # Initialize a variable to hold the current NEON site information
        self.current_site_info = ["No site info available."]
        # Set the tooltip generator on your GRIME_AI_QLabel widget(s). For example, if your widget is named NEON_labelLatestImage:
        if 0:
            if hasattr(self.NEON_labelLatestImage, "tooltipGenerator"):
                self.NEON_labelLatestImage.tooltipGenerator = self.siteInfoTooltip
            else:
                print("Warning: NEON_labelLatestImage is not an instance of GRIME_AI_QLabel.")

        # Set stylesheet for the tabs to change color when a tab is selected.
        self.tabWidget.setStyleSheet("""
            QTabBar::tab {
                background-color: white;
                color: black;
            }
            QTabBar::tab:selected {
                background-color: steelblue;
                color: white;
            }
        """)


        # ------------------------------------------------------------------------------------------------------------------
        # INITIALIZE VARIABLES
        # ------------------------------------------------------------------------------------------------------------------
        self.myHIVIS = None
        self.cameraDictionary = {}
        self.phenocam_site_dictionary = {}
        self.NEON_siteList = []

        # ------------------------------------------------------------------------------------------------------------------
        # CREATE REQUIRED FOLDERS IN THE USER'S DOCUMENTS FOLDER
        # ------------------------------------------------------------------------------------------------------------------
        utils = GRIME_AI_Utils()
        utils.create_GRIME_folders()

        self.populate_controls()

        # ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------
        #JES file_utils = GRIME_AI_Save_Utils()
        #JES file_utils.read_config_file()

        global imageFileFolder
        imageFileFolder = JsonEditor().getValue("Local_Image_Folder")

        #JES folderPath = GRIME_AI_Save_Utils().NEON_getSaveFolderPath()
        #JES self.edit_NEONSaveFilePath.1setText(folderPath)

        #JES folderPath = GRIME_AI_Save_Utils().USGS_getSaveFolderPath()
        #JES self.edit_USGSSaveFilePath.setText(folderPath)


        # ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------
        self.greenness_index_list = []
        self.colorSegmentationParams = colorSegmentationParamsClass()
        self.getColorSegmentationParams()

        # ----------------------------------------------------------------------------------------------------
        # GET DATA, POPULATE WIDGETS, ETC.
        # ----------------------------------------------------------------------------------------------------
        self.USGS_InitProductTable()

        self.USGS_FormatProductTable(self.table_USGS_Sites)

        self.NEON_FormatProductTableHeader()

        self.initROITable(self.greenness_index_list)

        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #JES - REVISIT DOUBLE CLICKING ON IMAGES
        self.NEON_labelLatestImage.mouseDoubleClickEvent = NEON_labelMouseDoubleClickEvent

        self.NEON_labelLatestImage.installEventFilter(self)
        self.labelEdgeImage.installEventFilter(self)
        self.labelOriginalImage.installEventFilter(self)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        self.pushButton_RetrieveNEONData.clicked.connect(self.pushbutton_NEONDownloadClicked)

        self.pushButton_USGS_BrowseImageFolder.clicked.connect(self.pushButton_USGS_BrowseImageFolder_Clicked)
        self.pushButton_NEON_BrowseImageFolder.clicked.connect(self.pushButton_NEON_BrowseImageFolder_Clicked)

        # Tab 0 browse and download buttons
        self.pushButton_NEON_Browse.clicked.connect(self.pushButton_NEON_Browse_Clicked)
        self.pushButton_NEON_Download.clicked.connect(self.pushbutton_NEONDownloadClicked)

        self.pushButton_SyncDates.setChecked(True)
        self._update_sync_dates_icon()
        self.pushButton_SyncDates.toggled.connect(self._on_sync_dates_toggled)

        # Connect Check Availability button (will be added to UI)
        try:
            self.pushButton_USGSCheckAvailability.clicked.connect(self.pushButton_USGSCheckAvailability_Clicked)
        except AttributeError:
            pass  # Button doesn't exist yet in UI file

        # INITIALIZE WIDGETS
        maxRows = self.tableWidget_ROIList.rowCount()
        for i in range(0, maxRows):
            self.tableWidget_ROIList.removeRow(0)

        # SAVE AND RECALL SETTINGS
        self.action_SaveSettings.triggered.connect(self.menubarSaveSettings)
        self.action_ReleaseNotes.triggered.connect(self.toolbarButtonReleaseNotes)
        self.action_CompositeSlices.triggered.connect(self.menubarCompositeSlices)
        self.action_TriageImages.triggered.connect(self.toolbarButtonImageTriage_2)
        self.action_Generate_Greenness_Test_Images.triggered.connect(self.menubar_Generate_Greenness_Test_Images)

        self.action_RefreshNEON.triggered.connect(self.menubar_RefreshNEON)

        self.action_CreateJSON.triggered.connect(self.menubar_CreateJSON)
        self.action_ExtractCOCOMasks.triggered.connect(self.menubarExtractCOCOMasks)
        self.action_Sync_JSON_Annotations.triggered.connect(self.menubar_sync_json_annotations)
        self.action_Inspect_Annotations.triggered.connect(self.menubar_inspect_annotations)

        self.action_ImageOrganizer.triggered.connect(self.menubar_ImageOrganizer)

        try:
            self.action_TemporalAveraging = QAction("Temporal Averaging", self)
            self.action_TemporalAveraging.setStatusTip("Average a folder of images together")
            self.action_TemporalAveraging.triggered.connect(self.toolbarButtonTemporalAveraging)
            self.menuTools.addSeparator()
            self.menuTools.addAction(self.action_TemporalAveraging)
            print("[INFO] Temporal Averaging added to Tools menu successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to add Temporal Averaging to Tools menu: {e}")
            traceback.print_exc()

        try:
            self.action_LaunchSAGE = QAction("SAGE", self)
            self.action_LaunchSAGE.setStatusTip("Open the SAGE annotation and segmentation tool")
            self.action_LaunchSAGE.triggered.connect(self.menubar_launch_sage)
            self.menuTools.addSeparator()
            self.menuTools.addAction(self.action_LaunchSAGE)
            print("[INFO] SAGE added to Tools menu successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to add SAGE to Tools menu: {e}")
            traceback.print_exc()

        try:
            self._action_api_keys = QAction("API Keys\u2026", self)
            self._action_api_keys.setStatusTip("Configure NEON and USGS API keys")
            self._action_api_keys.triggered.connect(self.menubar_api_keys)
            self.menuTools.addSeparator()
            self.menuTools.addAction(self._action_api_keys)
            print("[INFO] API Keys added to Tools menu successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to add API Keys to Tools menu: {e}")
            traceback.print_exc()

        # ------------------------------------------------------------------------------------------------------------------
        # VIEW MENU — dark/light mode toggle
        # ------------------------------------------------------------------------------------------------------------------
        self._is_dark_mode = False
        self._whole_image_ref_set = False
        self._roi_ref_set = set()

        _help_action = self.menuBar().actions()[-1] if self.menuBar().actions() else None
        self._menu_view = QMenu("View", self)
        self.menuBar().insertMenu(_help_action, self._menu_view)

        self._action_toggle_theme = QAction("Dark Mode", self)
        self._action_toggle_theme.setStatusTip("Toggle between dark and light application theme")
        self._action_toggle_theme.triggered.connect(self._toggle_dark_mode)
        self._menu_view.addAction(self._action_toggle_theme)

        try:
            self.action_About.triggered.connect(self._show_about_dialog)
        except AttributeError:
            pass

        # GRAPH TAB(S)
        self.NEON_labelLatestImage.setScaledContents(False)
        self.NEON_labelLatestImage.setAlignment(QtCore.Qt.AlignCenter)
        self.NEON_labelLatestImage.setMinimumSize(0, 0)
        self.NEON_labelLatestImage.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        self.USGS_labelLatestImage.setScaledContents(False)
        self.USGS_labelLatestImage.setAlignment(QtCore.Qt.AlignCenter)
        self.USGS_labelLatestImage.setMinimumSize(0, 0)
        self.USGS_labelLatestImage.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        # Title label shown above the USGS image
        self._usgs_image_title_label = QtWidgets.QLabel("", self.USGS_labelLatestImage.parent())
        self._usgs_image_title_label.setAlignment(QtCore.Qt.AlignCenter)
        self._usgs_image_title_label.setFont(QFont("Arial", 10, QFont.Bold))
        parent_layout = self.USGS_labelLatestImage.parent().layout()
        if parent_layout:
            idx = parent_layout.indexOf(self.USGS_labelLatestImage)
            if idx >= 0:
                parent_layout.insertWidget(idx, self._usgs_image_title_label)

        self.NEON_labelLatestImage.resized.connect(self.NEON_DisplayLatestImage)
        self.USGS_labelLatestImage.resized.connect(self.USGS_DisplayLatestImage)
        self.splitter_NEON_Bottom.splitterMoved.connect(self._neon_splitter_moved)
        self.splitter_NEON_Top.splitterMoved.connect(self._neon_splitter_moved)
        self.splitter_USGS_Horizontal.splitterMoved.connect(self._usgs_splitter_moved)
        self.splitter_USGS_Vertical.splitterMoved.connect(self._usgs_vertical_splitter_moved)
        # self.ui.labelLatestImage.setScaledContents(True)
        # self.ui.labelOriginalImage.setScaledContents(True)
        # self.ui.labelEdgeImage.setScaledContents(True)

        # ------------------------------------------------------------------------------------------------------------------
        # NEON
        # ------------------------------------------------------------------------------------------------------------------
        self.NEON_listboxSites.currentItemChanged.connect(self.NEON_SiteClicked)
        self.NEON_listboxSites.itemClicked.connect(self._neon_tree_item_clicked)
        self.NEON_listboxSiteProducts.itemClicked.connect(self.NEON_ProductClicked)

        # ------------------------------------------------------------------------------------------------------------------
        # USGS
        # ------------------------------------------------------------------------------------------------------------------
        self.usgs = USGSClient()
        self.usgs.initialize()
        self._nwis_fetcher = None   # holds the active NWISParameterFetcher thread
        self._usgs_image_fetcher = None  # holds the active USGSLatestImageFetcher thread
        self.USGS_listboxSites.itemClicked.connect(self._usgs_tree_item_clicked)
        self.pushButton_USGSDownload.clicked.connect(self.pushButton_USGSDownloadClicked)

        # ============================================================================
        # DEBOUNCE TIMER FOR IMAGE COUNT CHECKING
        # AUTOMATICALLY CHECKS AVAILABILITY 2 SECONDS AFTER USER STOPS CHANGING DATES
        # ============================================================================
        self.usgs_check_timer = QTimer()
        self.usgs_check_timer.setSingleShot(True)
        self.usgs_check_timer.timeout.connect(self.USGS_check_availability)
        self.usgs_checking = False  # Track if currently checking

        # ------------------------------------------------------------------------------------------------------------------
        # NIMS — fetch camera data in background thread
        self.USGS_listboxSites.clear()
        self._usgs_startup_fetcher = USGSStartupFetcher(self.usgs, parent=self)
        self._usgs_startup_fetcher.result.connect(self._on_usgs_startup_result)
        self._usgs_startup_fetcher.start()

        #self.edit_USGSSaveFilePath.setText("C:\\Users\\Astrid Haugen\\Documents\\GRIME-AI\\Downloads\\USGS_Test")

        # ------------------------------------------------------------------------------------------------------------------
        # USGS
        # ------------------------------------------------------------------------------------------------------------------
        #exif = EXIFData().extractEXIFdata('F:/000 - Hydrology Images/Reconyx/RCNX0009.jpg')

        #x = 1

        print("Create toolbar...")
        self.createToolBar()
        print("Toolbar create...")

        # ------------------------------------------------------------------------------------------------------------------
        # MENU
        # ------------------------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------------------------
        # SET THE BACKGROUND COLORS OF SPECIFIC BUTTONS
        # ------------------------------------------------------------------------------------------------------------------
        self.pushButton_RetrieveNEONData.setStyleSheet('QPushButton {background-color: steelblue;}')
        self.pushButton_NEON_BrowseImageFolder.setStyleSheet('QPushButton {background-color: steelblue;}')

        self.pushButton_USGSDownload.setStyleSheet('QPushButton {background-color: steelblue;}')
        self.pushButton_USGS_BrowseImageFolder.setStyleSheet('QPushButton {background-color: steelblue;}')

        # INITIALIZE GUI CONTROLS
        # frame.NEON_listboxSites.setCurrentRow(1)

        geojson_str = None
        self.init_openstreetmap(self.cameraDictionary, redList=[], geojson_str=geojson_str)

        # Fetch NEON field site table in background — populates NEON tab and triggers PhenoCam fetch
        print("Fetching NEON Field Site Table in background...")
        self._neon_api = NEON_API()
        self._neon_startup_fetcher = NEONStartupFetcher(self._neon_api, parent=self)
        self._neon_startup_fetcher.result.connect(self._on_neon_startup_result)
        self._neon_startup_fetcher.start()

        # Load persisted API keys and inject into clients at startup
        try:
            _key_mgr  = APIKeyManager()
            _neon_tok = _key_mgr.get_neon_token()
            _usgs_key = _key_mgr.get_usgs_key()
            if _neon_tok and hasattr(self._neon_api, "set_token"):
                self._neon_api.set_token(_neon_tok)
            _neon_ep = _key_mgr.get_neon_endpoint()
            if _neon_ep and hasattr(self._neon_api, "set_server"):
                self._neon_api.set_server(_neon_ep)
            if _usgs_key and hasattr(self.usgs, "set_api_key"):
                self.usgs.set_api_key(_usgs_key)
            _usgs_ep = _key_mgr.get_usgs_endpoint()
            if _usgs_ep and hasattr(self.usgs, "set_endpoint"):
                self.usgs.set_endpoint(_usgs_ep)
        except Exception as _e:
            print(f"[WARN] Could not load API keys at startup: {_e}")

        # Warn if NEON token is missing (mandatory as of June 30 2026)
        try:
            if not APIKeyManager().get_neon_token():
                QTimer.singleShot(1500, self._warn_neon_token_missing)
        except Exception as _e:
            print(f"[WARN] Could not check NEON token at startup: {_e}")

        # Show main window
        self.show()

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def _toggle_dark_mode(self):
        self._is_dark_mode = not self._is_dark_mode
        app = QApplication.instance()
        if self._is_dark_mode:
            try:
                import qdarkstyle
                app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
            except ImportError:
                pass
            self._action_toggle_theme.setText("Light Mode")
        else:
            app.setStyleSheet("")
            self._action_toggle_theme.setText("Dark Mode")

    def _show_about_dialog(self):
        try:
            from GRIME_AI.version import SW_VERSION, RELEASE, BUILD_DATE, SHA
        except ImportError:
            SW_VERSION = globals().get('SW_VERSION', '0.0.0.0')
            RELEASE    = 'N/A'
            BUILD_DATE = 'N/A'
            SHA        = 'N/A'

        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtCore import Qt
        from pathlib import Path

        dlg = QDialog(self)
        dlg.setWindowTitle("About GRIME AI")
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dlg.setFixedSize(480, 420)

        layout = QVBoxLayout(dlg)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(12)

        splash_dir = Path(__file__).resolve().parent / "resources" / "splash_screens"
        logo_path  = splash_dir / "GRIME-AI Logo with Tagline.png"
        lbl_logo   = QLabel()
        lbl_logo.setAlignment(Qt.AlignCenter)
        if logo_path.exists():
            pix = QPixmap(str(logo_path)).scaledToWidth(320, Qt.SmoothTransformation)
            lbl_logo.setPixmap(pix)
        else:
            lbl_logo.setText("GRIME AI")
            lbl_logo.setStyleSheet("font-size: 28px; font-weight: bold;")
        layout.addWidget(lbl_logo)

        lbl_version = QLabel(f"Version: {SW_VERSION}")
        lbl_version.setAlignment(Qt.AlignCenter)
        lbl_version.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(lbl_version)

        lbl_release = QLabel(f"Release: {RELEASE}")
        lbl_release.setAlignment(Qt.AlignCenter)
        lbl_release.setStyleSheet("font-size: 12px;")
        layout.addWidget(lbl_release)

        lbl_date = QLabel(f"Build Date: {BUILD_DATE}")
        lbl_date.setAlignment(Qt.AlignCenter)
        lbl_date.setStyleSheet("font-size: 11px; color: gray;")
        layout.addWidget(lbl_date)

        if SHA != 'N/A':
            commit_url = f"https://github.com/JohnStranzl/GRIME-AI/commit/{SHA}"
            lbl_sha = QLabel(f'Commit: <a href="{commit_url}" style="color: gray;">{SHA[:12]}</a>')
        else:
            lbl_sha = QLabel("Commit: N/A")
        lbl_sha.setAlignment(Qt.AlignCenter)
        lbl_sha.setStyleSheet("font-size: 11px; color: gray;")
        lbl_sha.setOpenExternalLinks(True)
        lbl_sha.setTextFormat(Qt.RichText)
        layout.addWidget(lbl_sha)

        btn = QPushButton("Close")
        btn.setFixedWidth(100)
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn, alignment=Qt.AlignCenter)

        dlg.exec_()

    def _on_usgs_startup_result(self, hivis, camera_dict, camera_list):
        """Callback when USGS HIVIS data has been fetched in background."""
        if hivis is None:
            msgBox = GRIME_AI_QMessageBox('USGS NIMS Error', 'Unable to access USGS NIMS/HIVIS Database!')
            msgBox.displayMsgBox()
            return

        self.myHIVIS         = hivis
        self.cameraDictionary = camera_dict
        self.cameraList       = camera_list

        # Connect USGS service to HIVIS for cache access
        try:
            self.usgs._svc.hivis = self.myHIVIS
            print("Connected USGS service to HIVIS for cache access")
        except Exception as e:
            print(f"[USGS startup] Could not connect HIVIS to USGS service: {e}")

        self.USGS_listboxSites.clear()
        for camID in self.cameraList:
            site_item = QTreeWidgetItem([camID])
            site_item.setData(0, QtCore.Qt.UserRole, camID)
            for line in self.myHIVIS.get_camera_info(camID):
                child = QTreeWidgetItem([line])
                child.setFlags(child.flags() & ~QtCore.Qt.ItemIsSelectable)
                site_item.addChild(child)
            placeholder = QTreeWidgetItem(["  Time series: loading..."])
            placeholder.setData(0, QtCore.Qt.UserRole, "__nwis_placeholder__")
            placeholder.setFlags(placeholder.flags() & ~QtCore.Qt.ItemIsSelectable)
            site_item.addChild(placeholder)
            self.USGS_listboxSites.addTopLevelItem(site_item)

        self.USGS_listboxSites.itemExpanded.connect(self._on_usgs_site_expanded)
        self.USGS_listboxSites.collapseAll()
        self.USGS_listboxSites.show()

        cameraIndex = 1
        default_item = self.USGS_listboxSites.topLevelItem(cameraIndex)
        if default_item:
            self.USGS_listboxSites.setCurrentItem(default_item)

        self.USGS_updateSiteInfo(1)

        # Add USGS pins to map now that we have the camera dictionary
        for name, coords in self.cameraDictionary.items():
            self.osm_widget.add_pin(coords["lat"], coords["lng"], color="usgs_green", label=name)

        self._usgs_startup_ready = True
        self._check_and_dismiss_splash()

    def _on_neon_startup_result(self, status, site_list):
        """Callback when NEON field site table has been fetched in background."""
        self.NEON_siteList = site_list if site_list else []

        if not self.NEON_siteList:
            print("NEON Field Site Table from NEON website FAILED...")
        else:
            print("Populate NEON Sites tab on GUI...")
            self.NEON_listboxSites.clear()

            for site in self.NEON_siteList:
                site_item = QTreeWidgetItem([f"{site.siteID} - {site.siteName}"])
                site_item.setData(0, QtCore.Qt.UserRole, site.siteID)
                for label, value in [("Site ID", site.siteID),
                                     ("Site Name", site.siteName),
                                     ("Latitude", site.latitude),
                                     ("Longitude", site.longitude),
                                     ("PhenoCams", site.phenocamSite)]:
                    child = QTreeWidgetItem([f"{label}: {value}"])
                    child.setFlags(child.flags() & ~QtCore.Qt.ItemIsSelectable)
                    url = next((w for w in str(value).split() if w.startswith("http")), None)
                    if url:
                        child.setData(0, QtCore.Qt.UserRole + 1, url)
                        child.setForeground(0, QtGui.QBrush(QtGui.QColor("#1a6fc4")))
                        font = child.font(0)
                        font.setUnderline(True)
                        child.setFont(0, font)
                    site_item.addChild(child)
                self.NEON_listboxSites.addTopLevelItem(site_item)

            self.NEON_listboxSites.collapseAll()

            # Add NEON map pins
            for coords in self.NEON_siteList:
                self.osm_widget.add_pin(coords.latitude, coords.longitude, color="gold", label=coords.siteName)

            try:
                default_item = self.NEON_listboxSites.topLevelItem(2)
                if default_item:
                    self.NEON_listboxSites.setCurrentItem(default_item)
                    self.NEON_listboxSites.show()
                    self.NEON_SiteClicked(default_item)
            except Exception:
                pass

        # Chain: now fetch PhenoCam in background
        self._phenocam_startup_fetcher = PhenocamStartupFetcher(self._neon_api, parent=self)
        self._phenocam_startup_fetcher.result.connect(self._on_phenocam_startup_result)
        self._phenocam_startup_fetcher.start()

        self._neon_startup_ready = True
        self._check_and_dismiss_splash()
        #JES self._check_startup_complete()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _on_phenocam_startup_result(self, data):
        """Callback when PhenoCam site table has been fetched in background."""
        if not data:
            print("ERROR: Unable to access Phenocam site information on Phenocam server!")
            return
        self.phenocam_site_dictionary = data
        for site_id, info in self.phenocam_site_dictionary.items():
            self.osm_widget.add_pin(info["lat"], info["lon"], color="yellow", label=site_id)
        self.populate_phenocam_tree()
        self.setup_phenocam_right_panel()

        self._phenocam_startup_ready = True
        self._check_and_dismiss_splash()

    def _check_and_dismiss_splash(self):
        """Dismiss splash only when USGS, NEON, and PhenoCam have all completed."""
        if self._usgs_startup_ready and self._neon_startup_ready and self._phenocam_startup_ready:
            if self._splash is not None:
                self._splash.dismiss()
                self._splash = None

    def init_openstreetmap(self, cameraDictionary, redList=None, geojson_str=None):
        osm_tab = self.findChild(QWidget, "tab_GoogleMaps")

        self.osm_widget = OpenStreetMapWidget(parent=osm_tab)

        layout = QVBoxLayout(osm_tab)
        layout.addWidget(self.osm_widget)

        # POPULATE THE MAP WITH PINS FOR USGS HIVIS CAMERA SITES
        for name, coords in self.cameraDictionary.items():
            self.osm_widget.add_pin(coords["lat"], coords["lng"], color="usgs_green", label=name)

        # POPULATE THE MAP WITH PINS FOR PHENOCAM CAMERA SITES REFERENCED BY NEON
        for site_id, info in self.phenocam_site_dictionary.items():
            self.osm_widget.add_pin(info["lat"], info["lon"], color="gold", label=site_id)

        # POPULATE THE MAP WITH PINS FOR PHENOCAM CAMERA SITES REFERENCED BY NEON
        for coords in self.NEON_siteList:
            self.osm_widget.add_pin(coords.latitude, coords.longitude, color="yellow")

        # ADD A PIN FOR FLAGSTAFF, AZ WHERE NORTHERN ARIZONA UNIVERSITY IS LOCATED.
        self.osm_widget.add_pin(35.1878, -111.6528, color="blue", label="Northern Arizona University")

        # CENTER MAP ON LINCOLN, NE WHERE THE UNIVERSITY OF NEBRASKA-LINCOLN IS LOCATED.
        self.osm_widget.set_center(40.8136, -96.7026, zoom=12, add_marker=True, label="Lincoln, NE", color="red")

    def populate_phenocam_tree(self):
        """
        Populate the Phenocam tree widget.
          Top-level items : site name only
          └─ Detail fields (Lat, Lon, Elev, active, utc_offset,
                            date_first, date_last, infrared)
          └─ ROI name
             └─ Hyperlink items (opened in browser on click)
        """
        import pandas as pd
        api = GRIME_AI_Phenocam_API()
        cameras_df = api.get_cameras(all_records=True)
        rois_df    = api.get_roilists(all_records=True)

        def get_col(*candidates):
            for c in candidates:
                if c in cameras_df.columns:
                    return c
            return None

        site_col = get_col("Sitename", "sitename")
        if cameras_df.empty or site_col is None:
            print("Camera data missing expected site column.")
            return

        roi_site_col = "site" if "site" in rois_df.columns else None

        CAMERA_FIELDS = [
            ("Lat",        "Latitude"),
            ("Lon",        "Longitude"),
            ("Elev",       "Elevation (m)"),
            ("active",     "Active"),
            ("utc_offset", "UTC Offset"),
            ("date_first", "First Date"),
            ("date_last",  "Last Date"),
            ("infrared",   "Infrared"),
        ]

        ROI_LINK_FIELDS = [
            ("roi_page",                   "ROI Page"),
            ("roi_stats_file",             "ROI Stats File"),
            ("one_day_summary",            "1-Day Summary"),
            ("three_day_summary",          "3-Day Summary"),
            ("one_day_transition_dates",   "1-Day Transition Dates"),
            ("three_day_transition_dates", "3-Day Transition Dates"),
        ]

        self.treeWidget_Phenocam.clear()

        for site_name, group in cameras_df.groupby(site_col):
            # ── Top-level: site name only ──────────────────────────────────
            site_item = QTreeWidgetItem([str(site_name)])
            site_item.setData(0, QtCore.Qt.UserRole, str(site_name))

            row = group.iloc[0]

            # Store utc_offset in UserRole+3 for timezone-aware download filtering
            # on strictly PhenoCam sites (non-NEON), whose filenames are UTC.
            utc_offset_val = row.get("utc_offset", None)
            if utc_offset_val is not None and pd.notna(utc_offset_val):
                try:
                    site_item.setData(0, QtCore.Qt.UserRole + 3, float(utc_offset_val))
                except (ValueError, TypeError):
                    pass

            # ── Camera detail fields directly under site ───────────────────
            for col, label in CAMERA_FIELDS:
                val = row.get(col, None)
                if val is not None and pd.notna(val):
                    detail_item = QTreeWidgetItem([f"{label}: {val}"])
                    site_item.addChild(detail_item)

            # ── ROI links: one bold ROI-name label + its links, all under site ──
            if roi_site_col is not None and not rois_df.empty:
                site_rois = rois_df[rois_df[roi_site_col] == site_name]
                for _, roi_row in site_rois.iterrows():
                    # Collect valid links for this ROI first
                    roi_links = []
                    for col, label in ROI_LINK_FIELDS:
                        link = roi_row.get(col, None)
                        if link and pd.notna(link) and str(link).startswith("http"):
                            roi_links.append((label, str(link)))

                    if not roi_links:
                        continue

                    # Bold ROI name as a non-clickable separator label
                    roi_name = str(roi_row.get("roi_name", "ROI"))
                    roi_label_item = QTreeWidgetItem([roi_name])
                    roi_label_item.setData(0, QtCore.Qt.UserRole,     str(site_name))
                    roi_label_item.setData(0, QtCore.Qt.UserRole + 2, roi_name)
                    roi_label_item_font = roi_label_item.font(0)
                    roi_label_item_font.setBold(True)
                    roi_label_item.setFont(0, roi_label_item_font)
                    site_item.addChild(roi_label_item)

                    # Hyperlinks directly under site (same level as camera fields)
                    for label, link in roi_links:
                        link_item = QTreeWidgetItem([f"{label}: {link}"])
                        link_item.setData(0, QtCore.Qt.UserRole + 1, link)
                        link_item.setForeground(0, QtGui.QBrush(QtGui.QColor("#1a6fc4")))
                        font = link_item.font(0)
                        font.setUnderline(True)
                        link_item.setFont(0, font)
                        site_item.addChild(link_item)

            self.treeWidget_Phenocam.addTopLevelItem(site_item)

        self.treeWidget_Phenocam.collapseAll()

        ###JES - THIS INTERACTION WITH THE PHENOCAM SERVERS TAKES ON THE ORDER OF 4 MINUTES; SO I
        if 0:
            print("START: Phenocam Meta Data Export...")
            api.export_all_to_excel(output_dir=GRIME_AI_Save_Utils().get_phenocam_folder())
            print("COMPLETED: Phenocam Meta Data Export.")

    # ------------------------------------------------------------------------------------------------------------------
    # PHENOCAM RIGHT PANEL
    # ------------------------------------------------------------------------------------------------------------------
    def _make_date_time_row(self, default_qdate, default_hour, default_minute):
        """Returns (row_widget, date_edit, hour_spin, minute_spin)."""
        row = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        date_edit = QtWidgets.QDateEdit(default_qdate)
        date_edit.setCalendarPopup(True)
        date_edit.setDisplayFormat("yyyy-MM-dd")
        date_edit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        hour_spin = QtWidgets.QSpinBox()
        hour_spin.setRange(0, 23)
        hour_spin.setValue(default_hour)
        hour_spin.setFixedWidth(48)
        hour_spin.setAlignment(QtCore.Qt.AlignRight)
        minute_spin = QtWidgets.QSpinBox()
        minute_spin.setRange(0, 59)
        minute_spin.setValue(default_minute)
        minute_spin.setFixedWidth(48)
        minute_spin.setAlignment(QtCore.Qt.AlignRight)
        hl.addWidget(date_edit)
        hl.addWidget(hour_spin)
        hl.addWidget(QtWidgets.QLabel("h"))
        hl.addWidget(minute_spin)
        hl.addWidget(QtWidgets.QLabel("m"))
        return row, date_edit, hour_spin, minute_spin

    def setup_phenocam_right_panel(self):
        # ── LEFT: hard-cap the tree width; let it expand vertically ─────────────
        self.treeWidget_Phenocam.setMaximumWidth(300)
        self.treeWidget_Phenocam.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        # Expanding vertical policy lets the layout stretch the tree to fill the
        # full tab height rather than leaving dead space below it.
        self.treeWidget_Phenocam.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        # Remove AlignTop so the layout distributes space rather than pinning to top
        self.verticalLayout_PhenocamLeft.setAlignment(QtCore.Qt.Alignment())
        self.horizontalLayout_Phenocam.setStretch(0, 1)
        self.horizontalLayout_Phenocam.setStretch(1, 3)

        # ── RIGHT: clear verticalLayout_PhenocamRight and rebuild ───────────────
        rl = self.verticalLayout_PhenocamRight
        while rl.count():
            item = rl.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.setSpacing(6)

        # Preview fixed at half the known tab content height (800px tab - ~30px tabbar = 770px)
        self.phenocam_preview_label = self.labelPhenocamImage
        self.phenocam_preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.phenocam_preview_label.setStyleSheet(
            "QLabel { background-color: #1a1a1a; color: #888888; }")
        self.phenocam_preview_label.setText("Select a site to preview the latest image")
        self.phenocam_preview_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.phenocam_preview_label.setFixedHeight(385)
        rl.addWidget(self.phenocam_preview_label, stretch=0)

        # (no maximum height cap — size policy handles vertical expansion)

        # Date / Time Range
        dt_group = QtWidgets.QGroupBox("Date / Time Range")
        dt_form  = QtWidgets.QFormLayout(dt_group)
        dt_form.setContentsMargins(8, 4, 8, 4)
        dt_form.setVerticalSpacing(4)
        now   = QtCore.QDate.currentDate()
        (start_row, self.phenocam_start_date,
         self.phenocam_start_hour, self.phenocam_start_minute) = \
            self._make_date_time_row(now.addDays(-30), 0, 0)
        (end_row, self.phenocam_end_date,
         self.phenocam_end_hour, self.phenocam_end_minute) = \
            self._make_date_time_row(now, 23, 59)
        dt_form.addRow("Start:", start_row)
        dt_form.addRow("End:",   end_row)
        rl.addWidget(dt_group, stretch=0)

        # Download Folder
        folder_group  = QtWidgets.QGroupBox("Download Folder")
        folder_layout = QtWidgets.QHBoxLayout(folder_group)
        folder_layout.setContentsMargins(8, 4, 8, 4)
        self.phenocam_folder_edit = QtWidgets.QLineEdit()
        self.phenocam_folder_edit.setPlaceholderText("Select output folder...")
        try:
            saved = JsonEditor().getValue("Phenocam_Root_Folder")
            if saved:
                self.phenocam_folder_edit.setText(saved)
        except Exception:
            pass
        self.phenocam_browse_btn = QtWidgets.QPushButton("Browse...")
        self.phenocam_browse_btn.setFixedWidth(80)
        self.phenocam_browse_btn.setStyleSheet(
            "QPushButton { background-color: steelblue; color: white; }")
        self.phenocam_browse_btn.clicked.connect(self._phenocam_browse_folder)
        folder_layout.addWidget(self.phenocam_folder_edit)
        folder_layout.addWidget(self.phenocam_browse_btn)
        rl.addWidget(folder_group, stretch=0)

        # Progress + status
        self.phenocam_progress_bar = QtWidgets.QProgressBar()
        self.phenocam_progress_bar.setRange(0, 100)
        self.phenocam_progress_bar.setFixedHeight(16)
        self.phenocam_progress_bar.setVisible(False)
        rl.addWidget(self.phenocam_progress_bar, stretch=0)
        self.phenocam_status_label = QtWidgets.QLabel("")
        self.phenocam_status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.phenocam_status_label.setVisible(False)
        rl.addWidget(self.phenocam_status_label, stretch=0)

        # Download button
        self.phenocam_download_btn = QtWidgets.QPushButton("Download Images")
        self.phenocam_download_btn.setStyleSheet(
            "QPushButton { background-color: steelblue; color: white; font-weight: bold; }")
        self.phenocam_download_btn.setMinimumHeight(32)
        self.phenocam_download_btn.clicked.connect(self._phenocam_download_clicked)
        rl.addWidget(self.phenocam_download_btn, stretch=0)

        # Trailing spacer so controls pin to top
        rl.addStretch(1)

        # Connect tree selection
        self.treeWidget_Phenocam.currentItemChanged.connect(self._phenocam_site_selected)
        # Open hyperlink items in the browser on click
        self.treeWidget_Phenocam.itemClicked.connect(self._phenocam_tree_item_clicked)
        self._phenocam_worker          = None
        self._phenocam_preview_fetcher = None

    def _phenocam_fit_preview_height(self):
        if not hasattr(self, 'phenocam_preview_label'):
            return
        tab_h = self.tab_phenocam_sites.height()
        if tab_h < 50:
            return
        # Tree height is managed by its Expanding size policy; no setFixedHeight needed.
        self.phenocam_preview_label.setFixedHeight(tab_h // 2)

    # ------------------------------------------------------------------------------------------------------------------
    def _phenocam_get_start_datetime(self):
        import datetime as dt_mod
        d = self.phenocam_start_date.date()
        return dt_mod.datetime(d.year(), d.month(), d.day(),
                               self.phenocam_start_hour.value(),
                               self.phenocam_start_minute.value())

    def _phenocam_get_end_datetime(self):
        import datetime as dt_mod
        d = self.phenocam_end_date.date()
        return dt_mod.datetime(d.year(), d.month(), d.day(),
                               self.phenocam_end_hour.value(),
                               self.phenocam_end_minute.value())

    def _phenocam_get_selected_sitename(self):
        item = self.treeWidget_Phenocam.currentItem()
        if item is None:
            return None
        # Walk up to find an item that has a sitename stored
        while item is not None:
            site = item.data(0, QtCore.Qt.UserRole)
            if site:
                return site
            item = item.parent()
        return None

    def _phenocam_get_selected_roi_name(self):
        """Return the roi_name string if an ROI node (or its child) is selected."""
        item = self.treeWidget_Phenocam.currentItem()
        if item is None:
            return None
        # Check the item itself and its parent for UserRole+2 (roi_name)
        for candidate in [item, item.parent()]:
            if candidate is None:
                continue
            roi = candidate.data(0, QtCore.Qt.UserRole + 2)
            if roi:
                return roi
        return None

    def _phenocam_get_selected_utc_offset(self):
        """Return the utc_offset (float hours) for the currently selected site, or None.
        Walks up to the top-level site item where UserRole+3 is stored."""
        item = self.treeWidget_Phenocam.currentItem()
        while item is not None:
            if item.parent() is None:  # top-level site item
                return item.data(0, QtCore.Qt.UserRole + 3)
            item = item.parent()
        return None

    def _phenocam_tree_item_clicked(self, item, column):
        """Open URL in the browser if the clicked item is a hyperlink."""
        url = item.data(0, QtCore.Qt.UserRole + 1)
        if url:
            from PyQt5.QtGui import QDesktopServices
            from PyQt5.QtCore import QUrl
            QDesktopServices.openUrl(QUrl(url))

    def _phenocam_site_selected(self, current, previous):
        if current is None:
            return
        site_name = self._phenocam_get_selected_sitename()
        if site_name:
            roi_name = self._phenocam_get_selected_roi_name()
            self._phenocam_load_latest_preview(site_name, roi_name=roi_name)

    def _phenocam_load_latest_preview(self, site_name, roi_name=None):
        label = roi_name if roi_name else site_name
        self.phenocam_preview_label.setPixmap(QtGui.QPixmap())
        self.phenocam_preview_label.setText(f"Loading latest image for {label}...")

        def _on_result(pixmap, err):
            if pixmap is None:
                self.phenocam_preview_label.setText(err or "No image available.")
            else:
                lbl = self.phenocam_preview_label
                scaled = pixmap.scaled(lbl.width(), lbl.height(),
                                       QtCore.Qt.KeepAspectRatio,
                                       QtCore.Qt.SmoothTransformation)
                lbl.setPixmap(scaled)
            self._phenocam_preview_fetcher = None

        fetcher = PhenocamPreviewFetcher(site_name, self, roi_name=roi_name)
        fetcher.result.connect(_on_result)
        self._phenocam_preview_fetcher = fetcher
        fetcher.start()

    def _phenocam_browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Download Folder", self.phenocam_folder_edit.text() or "")
        if folder:
            self.phenocam_folder_edit.setText(folder)
            try:
                JsonEditor().update_json_entry("Phenocam_Root_Folder", folder)
            except Exception:
                pass

    def _phenocam_download_clicked(self):
        # If a PhenoCam download is already running, use the same button to cancel it.
        if (
                hasattr(self, "_phenocam_worker")
                and self._phenocam_worker is not None
                and self._phenocam_worker.isRunning()
        ):
            self._phenocam_worker.cancel()
            self.phenocam_status_label.setText("Cancelling download...")
            self.phenocam_download_btn.setEnabled(False)
            return

        site_name = self._phenocam_get_selected_sitename()
        if not site_name:
            GRIME_AI_QMessageBox("PhenoCam Download",
                                 "Please select a site first.").displayMsgBox()
            return

        save_folder = self.phenocam_folder_edit.text().strip()
        if not save_folder:
            GRIME_AI_QMessageBox("PhenoCam Download",
                                 "Please specify a download folder.").displayMsgBox()
            return

        start_dt = self._phenocam_get_start_datetime()
        end_dt = self._phenocam_get_end_datetime()

        if start_dt >= end_dt:
            GRIME_AI_QMessageBox("PhenoCam Download",
                                 "Start must be before End.").displayMsgBox()
            return

        try:
            JsonEditor().update_json_entry("Phenocam_Root_Folder", save_folder)
        except Exception:
            pass

        self.phenocam_download_btn.setText("Cancel Download")
        self.phenocam_download_btn.setEnabled(True)
        self.phenocam_progress_bar.setValue(0)
        self.phenocam_progress_bar.setVisible(True)
        self.phenocam_status_label.setVisible(True)
        self.phenocam_status_label.setText("Starting...")

        worker = PhenocamDownloadWorker(
            site_name, start_dt, end_dt, save_folder, parent=self
        )
        worker.progress.connect(self._phenocam_download_progress)
        worker.finished.connect(self._phenocam_download_finished)
        self._phenocam_worker = worker
        worker.start()

    def _phenocam_download_progress(self, current, total, label):
        if total > 0:
            self.phenocam_progress_bar.setValue(int(current / total * 100))
        self.phenocam_status_label.setText(label)

    def _phenocam_download_finished(self, count):
        self.phenocam_download_btn.setText("Download Images")
        self.phenocam_download_btn.setEnabled(True)
        self._phenocam_worker = None

        if count == -1:
            self.phenocam_status_label.setText("Download cancelled.")
            GRIME_AI_QMessageBox("PhenoCam Download", "Download cancelled.").displayMsgBox()
            return

        self.phenocam_progress_bar.setValue(100)

        msg = (
            f"Download complete. {count} new image(s) saved."
            if count > 0
            else "No new images found for the selected date range."
        )
        self.phenocam_status_label.setText(msg)
        GRIME_AI_QMessageBox("PhenoCam Download", msg).displayMsgBox()

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def _usgs_progress(self, idx: int, total: int, label: str | None) -> None:
        """Progress callback for USGSClient operations."""
        if not hasattr(self, "_usgsProgress"):
            self._usgsProgress = QProgressWheel(parent=self)
            self._usgsProgress.setRange(0, total)
            self._usgsProgress.setWindowTitle("USGS Operation")
            self._usgsProgress.show()

        self._usgsProgress.setMaximum(total)
        self._usgsProgress.setValue(idx)

        if label:
            self._usgsProgress.setWindowTitle(str(label))

        # Clean up when finished
        if idx >= total:
            self._usgsProgress.close()
            del self._usgsProgress

    def force_close_progress(self):
        """Forcefully close and clean up the progress wheel if it exists."""
        if hasattr(self, "_usgsProgress"):
            try:
                self._usgsProgress.close()  # triggers closeEvent
                self._usgsProgress.deleteLater()  # ensure cleanup
            finally:
                del self._usgsProgress

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def init_google_maps(self, cameraDictionary):
        google_tab = self.findChild(QWidget, "tab_GoogleMaps")
        self.google_maps_widget = GoogleMapWidget(cameraDictionary, parent=google_tab)

        layout = QVBoxLayout(google_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.google_maps_widget)
        google_tab.setLayout(layout)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def shapefile_to_geojson_string(self, shapefile_path):
        import pandas as pd
        import geopandas as gpd

        gdf = gpd.read_file(shapefile_path)
        gdf = gdf.to_crs(epsg=4326)  # ensure WGS84
        geojson_str = gdf.to_json()  # returns a JSON string
        return geojson_str

    def siteInfoTooltip(self):
        """
        This function returns the dynamic tooltip string for GRIME_AI_QLabel.
        It uses self.current_site_info (a list of strings) to create the tooltip.
        """
        if self.current_site_info:
            return "NEON Site Info:\n" + "\n".join(self.current_site_info)
        else:
            return "No site info available."

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def populate_controls(self):

        # NEON CONTROLS
        NEON_download_file_path = JsonEditor().getValue("NEON_Root_Folder")
        self.edit_NEONSaveFilePath.setText(NEON_download_file_path)
        self.edit_NEON_TableInput.setText(NEON_download_file_path)

        # USGS CONTROLS
        USGS_download_file_path = JsonEditor().getValue("USGS_Root_Folder")
        self.edit_USGSSaveFilePath.setText(USGS_download_file_path)


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def buildFeatureFile(self):
        global dailyImagesList
        from GRIME_AI.GRIME_AI_Feature_Export import GRIME_AI_Feature_Export
        myFeatureExport = GRIME_AI_Feature_Export()
        imagesList = dailyImagesList.getVisibleList()

        if self.colorSegmentationDlg != None:
            self.getColorSegmentationParams()

        # Get texture options from dialog if available
        texture_options = {}
        if self.colorSegmentationDlg is not None:
            try:
                texture_options = self.colorSegmentationDlg.get_texture_options()
            except Exception:
                pass

        global imageFileFolder
        myFeatureExport.ExtractFeatures(imagesList, imageFileFolder, self.roiList, self.colorSegmentationParams, self.greenness_index_list, texture_options=texture_options)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def universalTestButton(self, testFunction):
        if testFunction == 1:
            print('This is Test Function 1.')

            # KMeans EXPECTS THE BYTE ORDER TO BE RGB
            img1 = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())
            #img1 = GRIME_AI_Utils().convertQImageToMat(myImage.toImage())

            rgb = cv2.blur(img1, ksize=(11, 11))

            # convert image to HSV
            hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)

            if len(self.roiList) > 0:
                # DIAGNOSTICS
                #if self.checkBoxColorDiagnostics.checkState():
                from GRIME_AI.GRIME_AI_Diagnostics import GRIME_AI_Diagnostics
                GRIME_AI_Diagnostics.RGB3DPlot(rgb)
                GRIME_AI_Diagnostics.plotHSVChannelsGray(hsv)
                GRIME_AI_Diagnostics.plotHSVChannelsColor(hsv)

                # segment colors
                rgb1 = myGRIMe_Color.segmentColors(rgb, hsv, self.roiList)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def getColorSegmentationParams(self):
        if self.colorSegmentationDlg is not None:
            self.colorSegmentationParams.GCC            = self.colorSegmentationDlg.checkBox_GCC.isChecked()
            self.colorSegmentationParams.GLI            = self.colorSegmentationDlg.checkBox_GLI.isChecked()
            self.colorSegmentationParams.NDVI           = self.colorSegmentationDlg.checkBox_NDVI.isChecked()
            self.colorSegmentationParams.ExG            = self.colorSegmentationDlg.checkBox_ExG.isChecked()
            self.colorSegmentationParams.RGI            = self.colorSegmentationDlg.checkBox_RGI.isChecked()

            self.colorSegmentationParams.Intensity      = self.colorSegmentationDlg.checkBox_Intensity.isChecked()
            self.colorSegmentationParams.ShannonEntropy = self.colorSegmentationDlg.checkBox_ShannonEntropy.isChecked()
            self.colorSegmentationParams.Texture        = self.colorSegmentationDlg.checkBox_Texture.isChecked()

            self.colorSegmentationParams.wholeImage     = self.colorSegmentationDlg.checkBoxScalarRegion_WholeImage.isChecked()

            self.colorSegmentationParams.numColorClusters = self.colorSegmentationDlg.get_num_color_clusters()

        self.greenness_index_list.clear()

        if self.colorSegmentationParams.GCC:
            self.greenness_index_list.append(GreennessIndex(GRIME_AI_Vegetation_Indices.GCC))
        if self.colorSegmentationParams.GLI:
            self.greenness_index_list.append(GreennessIndex(GRIME_AI_Vegetation_Indices.GLI))
        if self.colorSegmentationParams.NDVI:
            self.greenness_index_list.append(GreennessIndex(GRIME_AI_Vegetation_Indices.NDVI))
        if self.colorSegmentationParams.ExG:
            self.greenness_index_list.append(GreennessIndex(GRIME_AI_Vegetation_Indices.ExG))
        if self.colorSegmentationParams.RGI:
            self.greenness_index_list.append(GreennessIndex(GRIME_AI_Vegetation_Indices.RGI))

        self.initROITable(self.greenness_index_list)

    # ------------------------------------------------------------------------------------------------------------------
    # TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR
    # ------------------------------------------------------------------------------------------------------------------
    def createToolBar(self):
        toolbar = QToolBar("GRIME-AI Toolbar")
        self.addToolBar(toolbar)
        toolbar.setIconSize(QtCore.QSize(48, 48))

        parent_path = os.path.abspath(os.path.dirname(__file__))
        icons_dir = os.path.join(parent_path, "resources", "toolbar_icons")
        print("Toolbar Initialization: Icons directory:", icons_dir)

        # Define all toolbar buttons in one place
        buttons = [
            ("FileFolder_1.png", "Data Exploration",
             "Select input and output folder locations", self.onMyToolBarFileFolder),
            ("Triage_2.png", "Image Triage",
             "Move images that are of poor quality", self.toolbarButtonImageTriage),
            ("ImageNav_3.png", "Image Navigation",
             "Navigate (scroll) through images", self.onMyToolBarImageNavigation),
            ("Mask.png", "Create Masks",
             "Draw polygons to create image masks", self.onMyToolBarCreateMask),
            ("ColorWheel_4.png", "Color Segmentation",
             "Create ROIs to segment regions by color", self.onMyToolBarColorSegmentation),
            ("EdgeFilters_2.png", "Edge and Feature Detection",
             "Edge Detection Filters", self.toolbarButtonEdgeDetection),
            ("Settings_1.png", "Settings",
             "Change options and settings", self.onMyToolBarSettings),
            ("Green Brain Icon.png", "Deep Learning",
             "Deep Learning - EXPERIMENTAL", self.menubar_CreateJSON),
            ("grime2_StopSign.png", "GRIME2",
             "GRIME2 - Water Level Measurement", self.toolbarButtonGRIME2),
            ("Help_2.png", "Help",
             "Help and Release Notes", self.toolbarButtonReleaseNotes),
            ("TemporalAvg.png", "Temporal Averaging",
             "Average a folder of images together", self.toolbarButtonTemporalAveraging),
        ]

        # Generic creation loop
        for filename, text, tip, slot in buttons:
            icon_file = icon_path("toolbar_icons", filename)
            action = QAction(QIcon(icon_file), text, self)
            action.setStatusTip(tip)
            action.triggered.connect(slot)
            toolbar.addAction(action)
            print(f"Toolbar Initialization: {text} icon path: {icon_file}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PROCESS NEON SITE CHANGE
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def NEON_SiteClicked(self, item, previous=None):
        global SITECODE
        global gWebImagesAvailable
        global gProcessClick
        global gWebImageCount

        if item is None or item.parent() is not None:
            return

        print("NEON Site selected...")
        try:
            # gProcessClick is checked to see if another process is already handling a click event (gProcessClick == 0).
            # If not, it sets gProcessClick to 1 to prevent concurrent clicks.

            if gProcessClick == 0:
                gProcessClick = 1

                # --------------------------------------------------------------------------------
                # --------------------------------------------------------------------------------
                print("Updating site info...")

                start_time = time.time()
                SITECODE = NEON_updateSiteInfo(self)
                end_time = time.time()
                print ("NEON Site Info Elapsed Time: ", end_time - start_time)

                # --------------------------------------------------------------------------------
                # --------------------------------------------------------------------------------
                print("Updating site products...")
                time.sleep(2.0)

                start_time = time.time()
                num_matches, matches = self.NEON_updateSiteProducts(item)
                end_time = time.time()
                print ("NEON Site Products Elapsed Time: ", end_time - start_time)

                # --------------------------------------------------------------------------------
                # --------------------------------------------------------------------------------
                if num_matches > 0:
                    strFirstProductID = matches[0]
                    strProductID = strFirstProductID.split('.')[1]
                    self.NEON_labelLatestImage.setText("Loading latest image...")
                    self._neon_preview_fetcher = NEONPreviewFetcher(SITECODE, DOMAINCODE, strProductID)
                    self._neon_preview_fetcher.result.connect(self._neon_preview_received)
                    self._neon_preview_fetcher.start()
                else:
                    gWebImagesAvailable = 0
                    self.NEON_labelLatestImage.setText("No Images Available")

                gProcessClick = 0
        except Exception:
            gProcessClick = 0

        print("NEON site selection complete.")

    # ------------------------------------------------------------------------------------------------------------------
    # UPDATE NEON SITE PRODUCT INFORMATION
    # ------------------------------------------------------------------------------------------------------------------
    def NEON_ProductClicked(self, item):
        NEON_updateProductTable(self, item)

    # ------------------------------------------------------------------------------------------------------------------
    # DOWNLOAD NEON PRODUCT FILES
    # ------------------------------------------------------------------------------------------------------------------
    def pushbutton_NEONDownloadClicked(self, item):
        downloadProductDataFiles(self, item)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # PROCESS USGS DOWNLOAD MANAGER ACTIONS
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ==================================================================================================================
    #
    # ==================================================================================================================
    def USGS_InitProductTable(self):
        # HEADER TITLES
        headerList = ['Site', 'Image Count', ' min Date ', ' max Date ', 'Start Date', 'End Date', 'Start Time', 'End Time']

        # DEFINE HEADER STYLE
        stylesheet = "::section{Background-color:rgb(116,175,80);border-radius:14px;}"

        # POINTER TO HEADER
        header = self.table_USGS_Sites.horizontalHeader()

        # SET DEFAULT HEADER SETTINGS
        header.setMinimumSectionSize(120)
        header.setDefaultSectionSize(140)
        header.setHighlightSections(False)
        header.setStretchLastSection(False)

        # INSERT TITLES INTO HEADER AND FORMAT HEADER
        # MAKE COLUMNS 1 THRU 'n' SIZE TO CONTENTS
        # MAKE COLUMN 0 STRETCH TO FILL UP REMAINING EMPTY SPACE IN THE TABLE
        for i, item in enumerate(headerList):
            headerItem = QTableWidgetItem(item)
            headerItem.setTextAlignment(QtCore.Qt.AlignCenter)

            self.table_USGS_Sites.setHorizontalHeaderItem(i, headerItem)
            self.table_USGS_Sites.setStyleSheet(stylesheet)

            header.setSectionResizeMode(i, QHeaderView.Interactive)

        header.setSectionResizeMode(0, QHeaderView.Stretch)

        # SET THE HEADER FONT
        font = QFont()
        font.setBold(True)
        self.table_USGS_Sites.horizontalHeader().setFont(font)
        #self.table_USGS_Sites.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        #self.table_USGS_Sites.resizeColumnsToContents()

        date_widget = QtWidgets.QDateEdit(QtCore.QDate(date.today().year, date.today().month, date.today().day))
        self.table_USGS_Sites.setCellWidget(0, 4, date_widget)
        self.table_USGS_Sites.setCellWidget(0, 5, date_widget)

        # START TIME - defaults to 00:00:00
        time_widget = QDateTimeEdit()
        time_widget.setDisplayFormat("hh:mm:ss")
        time_widget.setTime(QtCore.QTime(0, 0, 0))  # 00:00:00
        time_widget.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(time_widget, self.table_USGS_Sites))
        time_widget.setKeyboardTracking(False)
        time_widget.setFrame(False)
        self.table_USGS_Sites.setCellWidget(0, 6, time_widget)

        # END TIME - defaults to 23:59:59 (full day)
        time_widget = QDateTimeEdit()
        time_widget.setDisplayFormat("hh:mm:ss")
        time_widget.setTime(QtCore.QTime(23, 59, 59))  # 23:59:59
        time_widget.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(time_widget, self.table_USGS_Sites))
        time_widget.setKeyboardTracking(False)
        time_widget.setFrame(False)
        self.table_USGS_Sites.setCellWidget(0, 7, time_widget)

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def USGS_dateChangeMethod(self, date_widget, tableWidget):
        # ============================================================================
        # DEBOUNCE: RESTART TIMER ON EACH DATE/TIME CHANGE
        # THIS PREVENTS API SPAM WHILE USER IS STILL TYPING/SELECTING DATES
        # ============================================================================
        self.usgs_check_timer.stop()  # Cancel any pending check
        self.usgs_check_timer.start(2000)  # Wait 2 seconds after last change

        # Show visual feedback that check is pending
        tableWidget.setItem(0, 1, QTableWidgetItem("Checking..."))

    # ======================================================================================================================
    # THIS FUNCTION WILL UPDATE THE PRODUCT TABLE IN THE GUI WITH THE PRODUCTS THAT ARE AVAILABLE FOR A SPECIFIC SITE.
    # ======================================================================================================================
    def USGS_FormatProductTable(self, tableProducts):
        maxRows = 1

        #JES: MUST MAKE CODE DYNAMIC TO ONLY DELETE UNSELECTED ITEMS
        for i in range(tableProducts.rowCount()):
            tableProducts.removeRow(0)

        tableProducts.insertRow(0)

        for i in range(maxRows):
            m = 0
            tableProducts.setItem(i, m, QTableWidgetItem(''))

            m += 1
            tableProducts.setItem(i, m, QTableWidgetItem(''))

            # CONFIGURE DATES FOR SPECIFIC PRODUCT
            m += 1
            date_widget = QtWidgets.QDateEdit()
            date_widget.setDisabled(True)
            tableProducts.setCellWidget(i, m, date_widget)

            m += 1
            date_widget = QtWidgets.QDateEdit()
            date_widget.setDisabled(True)
            tableProducts.setCellWidget(i, m, date_widget)

            # date_widget = QtWidgets.QDateEdit(calendarPopup=True)
            # date_widget.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))
            # tableProducts.setCellWidget(i, m, date_widget)

            # date_widget = QtWidgets.QDateEdit(calendarPopup=True)
            # date_widget.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))
            # tableProducts.setCellWidget(i, m, date_widget)

            m += 1
            date_widget = QtWidgets.QDateEdit(calendarPopup=True)
            date_widget.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))
            date_widget.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(date_widget, self.table_USGS_Sites))
            date_widget.setKeyboardTracking(False)
            self.table_USGS_Sites.setCellWidget(i, m, date_widget)

            m += 1
            date_widget = QtWidgets.QDateEdit(calendarPopup=True)
            date_widget.setDate(QtCore.QDate(date.today().year, date.today().month, date.today().day))
            date_widget.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(date_widget, self.table_USGS_Sites))
            date_widget.setKeyboardTracking(False)
            self.table_USGS_Sites.setCellWidget(i, m, date_widget)

            m += 1
            # START TIME - defaults to 00:00:00
            dateTime = QDateTimeEdit()
            dateTime.setDisplayFormat("hh:mm:ss")  # Show seconds for clarity
            dateTime.setTime(QtCore.QTime(0, 0, 0))  # 00:00:00
            dateTime.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(date_widget, self.table_USGS_Sites))
            dateTime.setKeyboardTracking(False)
            dateTime.setFrame(False)
            tableProducts.setCellWidget(i, m, dateTime)

            m += 1
            # END TIME - defaults to 23:59:59 (full day)
            dateTime = QDateTimeEdit()
            dateTime.setDisplayFormat("hh:mm:ss")  # Show seconds for clarity
            dateTime.setTime(QtCore.QTime(23, 59, 59))  # 23:59:59
            dateTime.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(date_widget, self.table_USGS_Sites))
            dateTime.setKeyboardTracking(False)
            dateTime.setFrame(False)
            tableProducts.setCellWidget(i, m, dateTime)

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def USGS_updateSiteInfo(self, item):
        currentItem = self.USGS_listboxSites.currentItem()
        if currentItem is None:
            return

        strCamID = currentItem.data(0, QtCore.Qt.UserRole)

        try:
            # Refresh camera info children
            currentItem.takeChildren()
            for line in self.usgs.get_camera_info_lines(strCamID):
                child = QTreeWidgetItem([line])
                child.setFlags(child.flags() & ~QtCore.Qt.ItemIsSelectable)
                currentItem.addChild(child)

            # Re-add placeholder so _on_usgs_site_expanded will fetch NWIS params
            placeholder = QTreeWidgetItem(["  Time series: loading..."])
            placeholder.setData(0, QtCore.Qt.UserRole, "__nwis_placeholder__")
            placeholder.setFlags(placeholder.flags() & ~QtCore.Qt.ItemIsSelectable)
            currentItem.addChild(placeholder)

            # Trigger fetch immediately since this site is now selected/visible
            self._on_usgs_site_expanded(currentItem)

            # Fetch latest image in background thread
            self.USGS_labelLatestImage.setText("Loading midday image...")
            if self._usgs_image_fetcher and self._usgs_image_fetcher.isRunning():
                self._usgs_image_fetcher.quit()
                self._usgs_image_fetcher.wait()
            self._usgs_image_fetcher = USGSLatestImageFetcher(self.usgs, strCamID, parent=self)
            self._usgs_image_fetcher.result.connect(self._on_usgs_latest_image_received)
            self._usgs_image_fetcher.start()

            # Update table
            self.table_USGS_Sites.setItem(0, 0, QTableWidgetItem(strCamID))

        except Exception as e:
            print("Error in USGS_updateSiteInfo:", e)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def _neon_splitter_moved(self):
        self._neon_splitter_moved_flag = True
        self.NEON_DisplayLatestImage()
        self.NEON_DisplayLatestImage()

    def _on_usgs_latest_image_received(self, code: int, pix, is_midday: bool):
        if code == 404 or pix is None:
            self.USGS_latestImage = []
            self.USGS_labelLatestImage.setText("No Image Available")
            if hasattr(self, '_usgs_image_title_label'):
                self._usgs_image_title_label.setText("")
        else:
            self.USGS_latestImage = pix
            if hasattr(self, '_usgs_image_title_label'):
                self._usgs_image_title_label.setText(
                    "Midday Image:" if is_midday else "Last Available Image:")
            self.USGS_DisplayLatestImage()

    def _on_usgs_site_expanded(self, item):
        """
        Fired when a USGS site item is expanded.  If the placeholder child
        is still present, kick off a background NWIS parameter fetch.
        """
        if item.parent() is not None:
            return  # only handle top-level items

        # Check whether placeholder is still there
        has_placeholder = False
        for i in range(item.childCount()):
            child = item.child(i)
            if child.data(0, QtCore.Qt.UserRole) == "__nwis_placeholder__":
                has_placeholder = True
                break
        if not has_placeholder:
            return  # already fetched

        cam_id = item.data(0, QtCore.Qt.UserRole)
        info_lines = self.usgs.get_camera_info_lines(cam_id)
        nwis_id = None
        for line in info_lines:
            if line.startswith("nwisId:"):
                nwis_id = line.split(":", 1)[1].strip()
                break

        if not nwis_id:
            # Replace placeholder with "no NWIS ID"
            for i in range(item.childCount()):
                child = item.child(i)
                if child.data(0, QtCore.Qt.UserRole) == "__nwis_placeholder__":
                    child.setText(0, "  No NWIS ID for this site")
                    child.setData(0, QtCore.Qt.UserRole, None)
                    break
            return

        # Cancel any running fetch and start a new one
        if self._nwis_fetcher and self._nwis_fetcher.isRunning():
            self._nwis_fetcher.quit()
            self._nwis_fetcher.wait()

        self._nwis_fetcher = NWISParameterFetcher(
            self.usgs._svc, nwis_id, cam_id, parent=self)
        self._nwis_fetcher.finished.connect(self._on_nwis_params_fetched)
        self._nwis_fetcher.start()

    def _on_nwis_params_fetched(self, params: list, cam_id: str):
        """
        Slot called when NWISParameterFetcher finishes.
        Replaces the placeholder child with real time series data.
        """
        root = self.USGS_listboxSites.invisibleRootItem()
        target = None
        for i in range(root.childCount()):
            item = root.child(i)
            if item.data(0, QtCore.Qt.UserRole) == cam_id:
                target = item
                break
        if target is None:
            return

        # Remove placeholder
        for i in range(target.childCount()):
            child = target.child(i)
            if child.data(0, QtCore.Qt.UserRole) == "__nwis_placeholder__":
                target.removeChild(child)
                break

        if params:
            header = QTreeWidgetItem(["Time Series Available:"])
            header.setFlags(header.flags() & ~QtCore.Qt.ItemIsSelectable)
            font = header.font(0)
            font.setBold(True)
            header.setFont(0, font)
            target.addChild(header)
            for p in params:
                label = f"  {p['code']} — {p['description']}"
                ts_child = QTreeWidgetItem([label])
                ts_child.setFlags(ts_child.flags() & ~QtCore.Qt.ItemIsSelectable)
                target.addChild(ts_child)
        else:
            no_ts = QTreeWidgetItem(["  No time series data available"])
            no_ts.setFlags(no_ts.flags() & ~QtCore.Qt.ItemIsSelectable)
            target.addChild(no_ts)

    def _usgs_vertical_splitter_moved(self):
        self._usgs_vertical_splitter_moved_flag = True
        self.USGS_DisplayLatestImage()

    def _usgs_splitter_moved(self):
        self._usgs_splitter_moved_flag = True
        self.USGS_DisplayLatestImage()

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def USGS_DisplayLatestImage(self):
        if self.USGS_latestImage != [] and self.USGS_labelLatestImage.width() > 1:
            self.USGS_labelLatestImage.clear()
            self.USGS_labelLatestImage.setPixmap(
                self.USGS_latestImage.scaled(
                    self.USGS_labelLatestImage.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation))

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def _usgs_tree_item_clicked(self, item, column):
        """Only trigger site selection for top-level items."""
        if item.parent() is None:
            self.USGS_SiteClicked(item)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def USGS_SiteClicked(self, item):
        USGSSiteIndex = self.USGS_updateSiteInfo(item)

        imageCount = self.USGS_get_image_count()

        self.table_USGS_Sites.setItem(0, 1, QTableWidgetItem(imageCount.__str__()))


    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def _update_sync_dates_icon(self):
        path = icon_path("toolbar_icons", "linked.png" if self.pushButton_SyncDates.isChecked() else "unlinked.png")
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.pushButton_SyncDates.setIcon(QIcon(pixmap))
        self.pushButton_SyncDates.setIconSize(QtCore.QSize(28, 28))

    def _on_sync_dates_toggled(self, checked):
        self._update_sync_dates_icon()

    def pushButton_NEON_BrowseImageFolder_Clicked(self, item):
        # PROMPT USER FOR FOLDER INTO WHICH TO DOWNLOAD THE IMAGES/FILES
        folder =  promptlib.Files().dir()

        if os.path.exists(folder):
            self.edit_NEONSaveFilePath.setText(folder)
        else:
            os.makedirs(folder)

        JsonEditor().update_json_entry("NEON_Image_Folder", folder)

    def pushButton_NEON_Browse_Clicked(self, item=None):
        folder = promptlib.Files().dir()

        if folder:
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.edit_NEON_TableInput.setText(folder)
            self.edit_NEONSaveFilePath.setText(folder)
            JsonEditor().update_json_entry("NEON_Image_Folder", folder)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def pushButton_USGS_BrowseImageFolder_Clicked(self, item):
        # PROMPT USER FOR FOLDER INTO WHICH TO DOWNLOAD THE IMAGES/FILES
        folder =  promptlib.Files().dir()

        if os.path.exists(folder):
            self.edit_USGSSaveFilePath.setText(folder)
        else:
            os.makedirs(folder)

        JsonEditor().update_json_entry("USGS_Root_Folder", folder)


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def trainROI(self, roiParameters):
        global currentImage

        if currentImage:
            myGRIME_Color = GRIME_AI_Color()

            # CREATE AN ROI OBJECT
            roiObj = GRIME_AI_roiData()

            # POPULATE ROI OBJECT WITH ROI INFORMATION
            if len(roiParameters.strROIName) > 0:
                roiObj.setROIName(roiParameters.strROIName)
            else:
                msgBox = GRIME_AI_QMessageBox('ROI Error', 'A name for the ROI is required!', buttons=QMessageBox.Close)
                response = msgBox.displayMsgBox()
                return

            # --------------------------------------------------
            rectROI = self.labelOriginalImage.getROI()

            if rectROI != None:
                roiObj.setDisplayROI(rectROI)
            else:
                msgBox = GRIME_AI_QMessageBox('ROI Error', 'Please draw the ROI on the image!', buttons=QMessageBox.Close)
                response = msgBox.displayMsgBox()
                return

            # --------------------------------------------------
            try:
                roiObj.setImageSize(currentImage.size())
                scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                roiObj.setDisplaySize(scaledCurrentImage.size())
                roiObj.calcROI()

                roiObj.setROIShape(ROIShape.RECTANGLE)
                #if self.radioButton_ROIShapeRectangle.isChecked():
                #roiObj.setROIShape(ROIShape.RECTANGLE)
                #else:
                #roiObj.setROIShape(ROIShape.ELLIPSE)
            except Exception:
                msgBox = GRIME_AI_QMessageBox('ROI Error',
                                           'An unexpected error occurred calculating the ROI of the full resolution image!', buttons=QMessageBox.Close)
                response = msgBox.displayMsgBox()

                return

            # ----------------------------------------------------------------------------------------------------------
            # CALCULATE COLOR CLUSTERS FOR THE ROI AND SAVE THEM TO THE ROI LIST
            # ----------------------------------------------------------------------------------------------------------
            # EXTRACT THE ROI FROM THE ORIGINAL IMAGE
            roiObj.setNumColorClusters(roiParameters.numColorClusters)

            # EXTRACT DOMINANT RGB COLORS AND ADD THEM TO THE ROI OBJECT
            #JES - PROVISIONAL - RGB CLUSTERS ARE NOT CURRENTLY USED.
            #JES qImg, clusterCenters, hist = myGRIME_Color.KMeans(rgb, roiObj.getNumColorClusters())
            #JES roiObj.setClusterCenters(clusterCenters, hist)

            # EXTRACT DOMINANT HSV COLORS AND ADD THEM TO THE ROI OBJECT
            img1 = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())
            rgb = extractROI(roiObj.getImageROI(), img1)
            hist, colorClusters = myGRIME_Color.extractDominant_HSV(rgb, roiObj.getNumColorClusters())
            roiObj.setHSVClusterCenters(colorClusters, hist)

            roiObj.setTrainingImageName(currentImageFilename)

            self.roiList.append(roiObj)

            # ----------------------------------------------------------------------------------------------------------
            # DISPLAY IN FEATURE TABLE
            # ----------------------------------------------------------------------------------------------------------
            #if (nRow == 0):
            #    numToAdd = 3
            #else:
            #    numToAdd = 1

            #for i in range(numToAdd):
            #    self.tableWidget_ROIList.insertRow(nRow)
            #    nRow += 1

            # CREATE COLOR BAR TO DISPLAY CLUSTER COLORS
            colorBar = GRIME_AI_Color.create_color_bar(hist, colorClusters)

            # CONVERT colorBar TO A QImage FOR USE IN DISPLAYING IN QT GUI
            qImg = QImage(colorBar.data, colorBar.shape[1], colorBar.shape[0], QImage.Format_BGR888)

            # INSERT THE DOMINANT COLORS INTO A QLabel IN ORDER TO ADD IT TO THE FEATURE TABLE
            nRow = self.tableWidget_ROIList.rowCount()
            self.tableWidget_ROIList.insertRow(nRow)

            self.label = QtWidgets.QLabel()
            self.label.setPixmap(QPixmap(qImg.scaled(100, 50)))
            self.tableWidget_ROIList.setCellWidget(nRow, 1, self.label)

            # INSERT ROI NAME INTO TABLE
            nCol = 0
            self.tableWidget_ROIList.setItem(nRow, nCol, QTableWidgetItem(roiParameters.strROIName))

            self.tableWidget_ROIList.resizeColumnsToContents()

            #JES pix = QPixmap(QImage(colorBar.data, colorBar.shape[1], colorBar.shape[0], QImage.Format_BGR888))
            #JES self.labelDominantColors.setPixmap(pix.scaled(self.labelEdgeImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

            # self.tableWidget_ROIList.setCellWidget(nRow, nCol, QCheckBox())

            # ----------------------------------------------------------------------------------------------------------
            #
            # ----------------------------------------------------------------------------------------------------------
            global currentImageIndex

            self.labelOriginalImage.clearROIs()
            self.labelOriginalImage.setROIs(self.roiList)

            processLocalImage(self, currentImageIndex)
            self.refreshImage()

            # ----------------------------------------------------------------------------------------------------------
            # ONCE AN ROI IS DEFINED FOR A SPECIFIC NUMBER OF COLOR CLUSTERS, DISABLE THE CONTROL SO THAT THE USER
            # CANNOT CHANGE THE VALUE FOR SUBSEQUENT TRAINED ROIs.
            # ----------------------------------------------------------------------------------------------------------
            if self.colorSegmentationDlg != None:
                if len(self.roiList) > 0:
                    self.colorSegmentationDlg.disable_spinbox_color_clusters(True)
                else:
                    self.colorSegmentationDlg.disable_spinbox_color_clusters(False)

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def calcEntropy(self, img):
        entropy = []

        hist = cv2.calcHist([img], [0], None, [256], [0, 255])
        total_pixel = img.shape[0] * img.shape[1]

        for item in hist.flatten():
            probability = item / total_pixel
            if probability == 0:
                en = 0
            else:
                en = -1 * probability * (np.log(probability) / np.log(2))
            entropy.append(en)

        try:
            sum_en = sum(entropy)
        except Exception:
            sum_en = 0.0

        #from scipy.stats import entropy
        #base = 2
        #H = entropy(hist, base=base)

        return sum_en

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def displayROIs(self, roiParameters):
        if self.colorSegmentationDlg != None:
            self.getColorSegmentationParams()

        processLocalImage(self)
        self.refreshImage()

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def NEON_FormatProductTableHeader(self):

        # HEADER TITLES
        headerList = ['Site', "Image Count", 'Start Date', 'End Date', 'Start Time', 'End Time']

        # DEFINE HEADER STYLE
        stylesheet = "::section{Background-color:rgb(116,175,80);border-radius:14px;}"

        # POINTER TO HEADER
        header = self.NEON_selected_products.horizontalHeader()

        # DEFAULT HEADER SETTINGS
        header.setMinimumSectionSize(120)
        header.setDefaultSectionSize(140)
        header.setHighlightSections(False)
        header.setStretchLastSection(False)

        # INSERT TITLES INTO HEADER AND FORMAT HEADER
        # MAKE COLUMNS 1 THRU 'n' SIZE TO CONTENTS
        # MAKE COLUMN 0 STRETCH TO FILL UP REMAINING EMPTY SPACE IN THE TABLE
        for i, item in enumerate(headerList):
            headerItem = QTableWidgetItem(item)
            headerItem.setTextAlignment(QtCore.Qt.AlignCenter)
            self.NEON_selected_products.setHorizontalHeaderItem(i, headerItem)
            self.NEON_selected_products.setStyleSheet(stylesheet)

            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        header.setSectionResizeMode(0, QHeaderView.Stretch)

        # SET THE HEADER FONT
        font = QFont()
        font.setBold(True)
        self.NEON_selected_products.horizontalHeader().setFont(font)
        self.NEON_selected_products.resizeColumnsToContents()

        date_widget = QtWidgets.QDateEdit(QtCore.QDate(date.today().year, date.today().month, date.today().day))
        self.NEON_selected_products.setCellWidget(1, 4, date_widget)
        self.NEON_selected_products.setCellWidget(1, 5, date_widget)
        self.NEON_selected_products.setCellWidget(1, 6, date_widget)
        self.NEON_selected_products.setCellWidget(1, 7, date_widget)

        try:
            self.tableWidget_ROIList.horizontalHeader().setVisible(True)
        except Exception:
             pass


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def initROITable(self, greenness_list=None):

        if greenness_list == []:
            headerList = ['ROI Name', 'Ref. Image', 'Cur. Image', 'Intensity', 'Entropy']
        else:
            headerList = ['ROI Name', 'Ref. Image', 'Cur. Image']

            for greenness_name in greenness_list:
                headerList.append(greenness_name.get_name())

            headerList.append('Intensity')
            headerList.append('Entropy')


        stylesheet = "::section{Background-color:rgb(116,175,80);border-radius:14px;}"
        header = self.tableWidget_ROIList.horizontalHeader()
        font = QFont()
        font.setBold(True)
        self.tableWidget_ROIList.horizontalHeader().setFont(font)
        self.tableWidget_ROIList.horizontalHeader().setVisible(True)
        self.tableWidget_ROIList.setStyleSheet(stylesheet)
        self.tableWidget_ROIList.setColumnCount(len(headerList))

        for i, item in enumerate(headerList):
            headerItem = QTableWidgetItem(item)
            headerItem.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tableWidget_ROIList.setHorizontalHeaderItem(i, headerItem)

            # Automatically resize columns when text is inserted
            header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)

        # Set the stretch mode to adapt to any remaining space
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def USGS_get_image_count(self):

        currentItem = self.USGS_listboxSites.currentItem()
        if currentItem is not None:
            site = currentItem.data(0, QtCore.Qt.UserRole)

        try:
            startDateCol = 4
            nYear, nMonth, nDay = self.separateDate(self.table_USGS_Sites.cellWidget(0, startDateCol).date())
            startDate = datetime.date(nYear, nMonth, nDay)

            endDateCol = 5
            nYear, nMonth, nDay = self.separateDate(self.table_USGS_Sites.cellWidget(0, endDateCol).date())
            endDate = datetime.date(nYear, nMonth, nDay)

            startTimeCol = 6
            nHour, nMinute, nSecond = self.separateTime(self.table_USGS_Sites.cellWidget(0, startTimeCol).dateTime().time())
            startTime = datetime.time(nHour, nMinute, nSecond)

            endTimeCol = 7
            nHour, nMinute, nSecond = self.separateTime(self.table_USGS_Sites.cellWidget(0, endTimeCol).dateTime().time())
            endTime = datetime.time(nHour, nMinute, nSecond)

            nwisID = self.myHIVIS.get_nwisID()

            imageCount = self.myHIVIS.get_image_count(siteName=site, nwisID=nwisID, startDate=startDate, endDate=endDate, startTime=startTime, endTime=endTime)
        except Exception:
            imageCount = 0

        return imageCount

    # ============================================================================
    # CHECK AVAILABILITY - CALLED BY DEBOUNCE TIMER OR MANUAL BUTTON
    # ============================================================================
    def USGS_check_availability(self):
        """
        Fetch image count and update table. Called by:
        1. Debounce timer (2 seconds after date/time changes)
        2. Manual "Check Availability" button click
        """
        if self.usgs_checking:
            return  # Already checking, avoid duplicate calls
        
        self.usgs_checking = True
        
        try:
            # Show loading indicator
            self.table_USGS_Sites.setItem(0, 1, QTableWidgetItem("⟳ Checking..."))
            QApplication.processEvents()  # Force UI update
            
            # Fetch image count (this may take time)
            imageCount = self.USGS_get_image_count()
            
            # Update table with result
            self.table_USGS_Sites.setItem(0, 1, QTableWidgetItem(str(imageCount)))
            
        except Exception as e:
            self.table_USGS_Sites.setItem(0, 1, QTableWidgetItem("Error"))
            print(f"Error checking availability: {e}")
        finally:
            self.usgs_checking = False

    # ============================================================================
    # MANUAL CHECK AVAILABILITY BUTTON HANDLER
    # ============================================================================
    def pushButton_USGSCheckAvailability_Clicked(self):
        """
        Manual trigger for checking image availability.
        Cancels any pending auto-check and immediately checks.
        """
        self.usgs_check_timer.stop()  # Cancel pending auto-check
        self.USGS_check_availability()  # Check immediately

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def pushButton_USGSDownloadClicked(self):

        # VERIFY THAT THE FOLDER HAS BEEN SPECIFIED
        USGS_download_file_path = self.edit_USGSSaveFilePath.text()
        JsonEditor().update_json_entry("USGS_Root_Folder", USGS_download_file_path)

        if len(USGS_download_file_path) == 0:
            strMessage = 'A download folder has not been specified. Would you like to use the last GRIME-AI USGS download folder used?'
            msgBox = GRIME_AI_QMessageBox('USGS Root Download Folder', strMessage, QMessageBox.Yes | QMessageBox.No)
            response = msgBox.displayMsgBox()

            if response == QMessageBox.Yes:
                #USGS_download_file_path = os.path.expanduser('~')
                #USGS_download_file_path = os.path.join(USGS_download_file_path, 'Documents')
                #USGS_download_file_path = os.path.join(USGS_download_file_path, 'GRIME-AI')

                USGS_download_file_path = JsonEditor().getValue("USGS_Root_Folder")

                if not os.path.exists(USGS_download_file_path):
                    os.makedirs(USGS_download_file_path)
                #NEON_download_file_path = os.path.join(USGS_download_file_path, 'Downloads')
                #if not os.path.exists(USGS_download_file_path):
                #    os.makedirs(USGS_download_file_path)

                self.edit_USGSSaveFilePath.setText(USGS_download_file_path)
                JsonEditor().update_json_entry("USGS_Root_Folder", USGS_download_file_path)
        else:
            # MAKE SURE THE PATH EXISTS. IF IT DOES NOT, THEN CREATE IT.
            if not os.path.exists(USGS_download_file_path):
                os.makedirs(USGS_download_file_path)

        currentItem = self.USGS_listboxSites.currentItem()

        if currentItem is not None:
            site = currentItem.data(0, QtCore.Qt.UserRole)

            startDateCol = 4
            nYear, nMonth, nDay = self.separateDate(self.table_USGS_Sites.cellWidget(0, startDateCol).date())
            startDate = datetime.date(nYear, nMonth, nDay)

            endDateCol = 5
            nYear, nMonth, nDay = self.separateDate(self.table_USGS_Sites.cellWidget(0, endDateCol).date())
            endDate = datetime.date(nYear, nMonth, nDay)

            startTimeCol = 6
            nHour, nMinute, nSecond = self.separateTime(self.table_USGS_Sites.cellWidget(0, startTimeCol).dateTime().time())
            startTime = datetime.time(nHour, nMinute, nSecond)

            endTimeCol = 7
            nHour, nMinute, nSecond = self.separateTime(self.table_USGS_Sites.cellWidget(0, endTimeCol).dateTime().time())
            endTime = datetime.time(nHour, nMinute, nSecond)

            nwisID = self.myHIVIS.get_nwisID()

            #downloadsFilePath = os.path.join(self.edit_USGSSaveFilePath.text(), 'Images')
            downloadsFilePath = self.edit_USGSSaveFilePath.text()
            if not os.path.exists(downloadsFilePath):
                os.makedirs(downloadsFilePath)

            saveFolder = os.path.join(downloadsFilePath, "Images")
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)

            self._usgs_cancel_requested = False

            downloaded, missing = self.usgs.download_images(
                site,
                startDate,
                endDate,
                startTime,
                endTime,
                saveFolder,
                progress=self._usgs_progress,
                cancel_check = lambda: self._usgs_cancel_requested
            )

            saveFolder = os.path.join(downloadsFilePath, "data")
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)
            self.myHIVIS.fetchStageAndDischarge(nwisID, site, startDate, endDate, startTime, endTime, saveFolder)

            self.force_close_progress()

        #fetchUSGSImages(self.table_USGS_Sites, self.edit_USGSSaveFilePath)


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def separateDate(self, date):
        nYear     = date.year()
        nMonth    = date.month()
        nDay      = date.day()

        return nYear, nMonth, nDay

    def separateTime(self, time):
        nHour   = time.hour()
        nMinute = time.minute()
        nSecond = time.second()

        return nHour, nMinute, nSecond


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def menubar_CreateJSON(self):
        global hyperparameterDlg

        # If it’s already alive, just raise & activate it
        '''
        if hyperparameterDlg is not None and hyperparameterDlg.isVisible():
            hyperparameterDlg.raise_()
            hyperparameterDlg.activateWindow()
            return None
        '''

        settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
        CONFIG_FILENAME = "site_config.json"
        site_configuration_file = os.path.normpath(os.path.join(settings_folder, CONFIG_FILENAME))

        # Show a splash screen while the ML dialog loads
        from PyQt5.QtGui import QPixmap, QColor
        from PyQt5.QtCore import Qt
        _splash_pix = QPixmap(480, 120)
        _splash_pix.fill(QColor("#2b2b2b"))
        _splash = QSplashScreen(_splash_pix, Qt.WindowStaysOnTopHint)
        _splash.showMessage("Loading ML Image Processing...",
                            Qt.AlignCenter | Qt.AlignVCenter,
                            QColor("#ffffff"))
        _splash.show()
        QApplication.processEvents()

        from GRIME_AI.dialogs.ML_image_processing.GRIME_AI_ML_ImageProcessingDlg import GRIME_AI_ML_ImageProcessingDlg
        hyperparameterDlg = GRIME_AI_ML_ImageProcessingDlg(frame)

        _splash.finish(hyperparameterDlg)

        hyperparameterDlg.ml_train_signal.connect(train_main)
        hyperparameterDlg.ml_segment_signal.connect(segment_main)

        #hyperparameterDlg.accepted.connect(closehyperparameterDlg)
        #hyperparameterDlg.rejected.connect(closehyperparameterDlg)

        hyperparameterDlg.finished.connect(closehyperparameterDlg)

        #config = hyperparameterDlg.load_config_from_json(site_configuration_file)
        #hyperparameterDlg.initialize_dialog_from_config(config)

        # Show the dialog and capture user response.
        if hyperparameterDlg.exec_() == QDialog.Accepted:
            hyperparameters = hyperparameterDlg.get_values()
            #config = dialog.getValues()
            print("Configuration Options:")
            for key, value in hyperparameters.items():
                print(f"  {key}: {value}")


    def menubar_RefreshNEON(self):
        # INITIALIZE GUI CONTROLS
        # frame.NEON_listboxSites.setCurrentRow(1)

        # GET LIST OF ALL SITES ON NEON
        # if frame.checkBoxNEONSites.isChecked():
        myNEON_API = NEON_API()
        _, siteList = myNEON_API.readFieldSiteTable()
        # else:
        # NEON_FormatProductTable(frame.tableProducts)

        if len(siteList) == 0:
            pass
            # frame.radioButtonHardDriveImages.setChecked(True)
            # frame.radioButtonHardDriveImages.setDisabled(False)
        # IF THERE ARE FIELD SITE TABLES AVAILABLE, ENABLE GUI WIDGETS PERTAINING TO WEB SITE DATA/IMAGES
        else:
            self.NEON_listboxSites.clear()

            for site in siteList:
                site_item = QTreeWidgetItem([f"{site.siteID} - {site.siteName}"])
                site_item.setData(0, QtCore.Qt.UserRole, site.siteID)
                for label, value in [("Site ID", site.siteID),
                                     ("Site Name", site.siteName),
                                     ("Latitude", site.latitude),
                                     ("Longitude", site.longitude),
                                     ("PhenoCams", site.phenocamSite)]:
                    child = QTreeWidgetItem([f"{label}: {value}"])
                    child.setFlags(child.flags() & ~QtCore.Qt.ItemIsSelectable)
                    url = next((w for w in str(value).split() if w.startswith("http")), None)
                    if url:
                        child.setData(0, QtCore.Qt.UserRole + 1, url)
                        child.setForeground(0, QtGui.QBrush(QtGui.QColor("#1a6fc4")))
                        font = child.font(0)
                        font.setUnderline(True)
                        child.setFont(0, font)
                    site_item.addChild(child)
                self.NEON_listboxSites.addTopLevelItem(site_item)

            self.NEON_listboxSites.collapseAll()

            #JES - TEMPORARILY SET BARCO LAKE AS THE DEFAULT SELECTION
            try:
                default_item = self.NEON_listboxSites.topLevelItem(2)
                if default_item:
                    self.NEON_listboxSites.setCurrentItem(default_item)
                    self.NEON_listboxSites.show()
                    self.NEON_SiteClicked(default_item)
            except Exception:
                pass
                pass

        print("Initialize USGS product table...")
        self.USGS_InitProductTable()
        self.USGS_FormatProductTable(self.table_USGS_Sites)
        self.NEON_FormatProductTableHeader()

        self.show()


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def menubarCompositeSlices(self):
        global dailyImagesList
        global imageFileFolder
        global currentImageIndex

        if len(dailyImagesList.getVisibleList()) == 0:
            strMessage = 'You must first create a list of images to operate on. Use the FETCH files feature of GRIME AI.'
            msgBox = GRIME_AI_QMessageBox('Composite Slice Error', strMessage, QMessageBox.Close)
            response = msgBox.displayMsgBox()
        else:
            imageFilename = dailyImagesList.getVisibleList()[currentImageIndex].fullPathAndFilename

            if self.compositeSliceDlg is None:
                from GRIME_AI.dialogs.composite_slice.GRIME_AI_CompositeSliceDlg import GRIME_AI_CompositeSliceDlg
                self.compositeSliceDlg = GRIME_AI_CompositeSliceDlg()
                self.compositeSliceDlg.compositeSliceGenerateSignal.connect(self.generateCompositeSlices)
                self.compositeSliceDlg.compositeSliceCancelSignal.connect(self.closeCompositeSlices)
                self.compositeSliceDlg.label_Image.setDrawingMode(DrawingMode.SLICE)

            self.compositeSliceDlg.loadImage(imageFilename)
            self.compositeSliceDlg.show()
            self.compositeSliceDlg.raise_()

    def generateCompositeSlices(self):
        print("Generating composite slices image(s)...")

        global imageFileFolder
        composite_slices_folder = GRIME_AI_Save_Utils().create_composite_slices_folder(imageFileFolder)

        slice_rect = self.compositeSliceDlg.label_Image.getSliceRectInOriginal()

        from GRIME_AI.GRIME_AI_CompositeSlices import GRIME_AI_CompositeSlices
        compositeSlices = GRIME_AI_CompositeSlices(slice_rect)
        compositeSlices.create_composite_image(dailyImagesList.visibleList, composite_slices_folder)

    def closeCompositeSlices(self):
        if self.compositeSliceDlg != None:
            self.compositeSliceDlg.close()
            self.compositeSliceDlg    = None

    def menubar_Generate_Greenness_Test_Images(self):
        # initialize with default settings

        rootFolder = os.path.expanduser('~')
        rootFolder = os.path.join(rootFolder, 'Documents', 'GRIME-AI', 'Test Images')
        gen = GreenImageGenerator(out_dir=rootFolder)

        # generate all images (solids, splotches, masks)
        gen.generate_all()

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def toolbarButtonImageTriage_1(self, folder_path=None):
        self.toolbarButtonImageTriage()

    def toolbarButtonImageTriage_2(self):
        self.toolbarButtonImageTriage()

    def toolbarButtonImageTriage(self, folder_path=None, checkBox_FetchRecursive=False):
        strMessage = 'You are about to perform Image Triage. Would you like to continue?'
        msgBox = GRIME_AI_QMessageBox('Download Image Files', strMessage, QMessageBox.Yes | QMessageBox.No)
        response = msgBox.displayMsgBox()

        if response == QMessageBox.Yes:
            prompter = promptlib.Files()
            folder = prompter.dir()

            if len(folder) == 0:
                strMessage = 'ERROR! Please specify an image folder containing images to triage.'
                msgBox = GRIME_AI_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
                response = msgBox.displayMsgBox()
            else:
                TriageDlg = GRIME_AI_TriageOptionsDlg(folder=folder)

                response = TriageDlg.exec_()

                if response == 1:

                    if len(TriageDlg.getReferenceImageFilename()) == 0 and TriageDlg.getCorrectAlignment() == True:
                        strMessage = 'Please select reference image if you want to correct image alignment.'
                        msgBox = GRIME_AI_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
                        response = msgBox.displayMsgBox()
                    else:
                        myTriage = GRIME_AI_ImageTriage()
                        myTriage.cleanImages(folder, \
                                             False, \
                                             TriageDlg.getBlurThreshold(), TriageDlg.getShiftSize(), \
                                             TriageDlg.getBrightnessMin(), TriageDlg.getBrightnessMax(), \
                                             TriageDlg.getCreateReport(), TriageDlg.getMoveImages(), \
                                             TriageDlg.getCorrectAlignment(), TriageDlg.getSavePolylines(),
                                             TriageDlg.getReferenceImageFilename(), TriageDlg.getRotationThreshold(),
                                             focus_roi=TriageDlg.getFocusROI(),
                                             use_color_imbalance=TriageDlg.getUseColorImbalance(),
                                             color_imbalance_threshold=TriageDlg.getColorImbalanceThreshold())

                        strMessage = 'Image triage is complete!'
                        msgBox = GRIME_AI_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
                        response = msgBox.displayMsgBox()
                else:
                    strMessage = 'ABORT! You cancelled the triage operation.'
                    msgBox = GRIME_AI_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
                    response = msgBox.displayMsgBox()
        else:
            strMessage = 'ABORT! You cancelled the triage operation.'
            msgBox = GRIME_AI_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
            response = msgBox.displayMsgBox()


    def menubarExtractCOCOMasks(self):
        from GRIME_AI.dialogs.extract_coco_masks.GRIME_AI_ExportCOCOMasksDlg import GRIME_AI_ExportCOCOMasksDlg
        self.COCOdlg = GRIME_AI_ExportCOCOMasksDlg(self)

        self.COCOdlg.COCO_signal_ok.connect(self.accepted_COCODlg)
        self.COCOdlg.COCO_signal_cancel.connect(self.rejected_COCODlg)

        self.COCOdlg.show()


    def accepted_COCODlg(self):
        image_dir = self.COCOdlg.getAnnotationImagesFolder()
        output_dir = os.path.join(image_dir, "training_masks")

        #JES CLEANUP TASK: THE ANNOTATION FILE IS NOT REQUIRED TO BE IN THE TRAINING IMAGES FOLDER.
        utils = GRIME_AI_COCO_Utils(image_dir)
        utils.extract_masks(image_dir, output_dir)

    def rejected_COCODlg(self):
        pass


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def menubar_sync_json_annotations(self):
        selected_dir = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select a Folder",
            directory="C:/",  # initial directory
            options=QFileDialog.ShowDirsOnly
        )

        if selected_dir:
            util = GRIME_AI_COCO_Utils(selected_dir)
            print("Selected folder:", selected_dir)

            """Execute full validation and cleaning pipeline."""
            util.find_json_file()
            util.load_json()

            present, missing = util.check_images()
            if not missing:
                print("All images in JSON are present.")
                return
            else:
                # Create the message box
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Confirmation")
                msg_box.setText("Do you want to sync the JSON annotations with the available images?")

                # Use Question icon and add buttons
                msg_box.setIcon(QMessageBox.Question)
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

                if msg_box.exec_() == QMessageBox.Yes:
                    util.backup_original()
                    cleaned = util.clean_data(present)
                    util.write_json(cleaned)

                    msg_box.setWindowTitle("Completion")
                    msg_box.setText("JSON annotations sync'ed with the available images.")

                    # Use Question icon and add buttons
                    msg_box.setIcon(QMessageBox.Question)
                    msg_box.setStandardButtons(QMessageBox.Ok)

                    # Execute and capture response
                    if msg_box.exec_() == QMessageBox.Ok:
                        print("New JSON created for the images available in the folder.")


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def menubar_inspect_annotations(self):
        selected_dir = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select a Folder",
            options=QFileDialog.ShowDirsOnly
        )
        #directory = "C:/",  # initial directory

        if selected_dir:
            utils = GRIME_AI_COCO_Utils(selected_dir)
            print("Selected folder:", selected_dir)
            utils.load_coco()

            now = datetime.datetime.now()
            self.formatted_time = now.strftime('%Y%m%d_%H%M%S')
            inspection_file = f"{self.formatted_time}_Inspect_Annotations.xlsx"
            inspection_file = os.path.join(selected_dir, inspection_file)

            utils.write_image_label_counts_to_xlsx(inspection_file)
            print(f"Annotations Inspection: {inspection_file}")


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def menubarSaveSettings(self):
        utils = GRIME_AI_Save_Utils()
        utils.saveSettings()


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def menubar_api_keys(self):
        """Open the API Key Manager dialog for NEON and USGS keys."""
        dlg = APIKeyDialog(parent=self)
        if dlg.exec_() == APIKeyDialog.Accepted:
            mgr       = APIKeyManager()
            neon_tok  = mgr.get_neon_token()
            usgs_key  = mgr.get_usgs_key()
            usgs_ep   = mgr.get_usgs_endpoint()
            # Propagate new keys/endpoint into live clients without requiring a restart
            neon_ep   = mgr.get_neon_endpoint()
            if hasattr(self, "_neon_api") and self._neon_api is not None:
                if hasattr(self._neon_api, "set_token"):
                    self._neon_api.set_token(neon_tok)
                if hasattr(self._neon_api, "set_server"):
                    self._neon_api.set_server(neon_ep)
            if hasattr(self, "usgs") and self.usgs is not None:
                if hasattr(self.usgs, "set_api_key"):
                    self.usgs.set_api_key(usgs_key)
                if hasattr(self.usgs, "set_endpoint"):
                    self.usgs.set_endpoint(usgs_ep)


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def _warn_neon_token_missing(self):
        """Show a non-blocking startup warning when no NEON API token is configured."""
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("NEON API Token Required")
        msg.setText(
            "<b>No NEON API token is configured.</b><br><br>"
            "As of June 30, 2026, a NEON account token is required "
            "to download NEON data products and imagery.<br><br>"
            "NEON data access will be unavailable until a token is added."
        )
        msg.setInformativeText(
            'Get a free token at <a href="https://data.neonscience.org/myaccount">'  
            'data.neonscience.org</a>, then add it via <b>Tools → API Keys…</b>'
        )
        msg.setTextFormat(QtCore.Qt.RichText)
        open_btn  = msg.addButton("Open API Keys…", QMessageBox.AcceptRole)
        msg.addButton("Dismiss", QMessageBox.RejectRole)
        msg.exec_()
        if msg.clickedButton() == open_btn:
            self.menubar_api_keys()


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def menubar_ImageOrganizer(self):

        # --------------------------------------------------------------------------------------------------------------
        # BEGIN PROCESSING
        # --------------------------------------------------------------------------------------------------------------
        # Close any existing instance first
        try:
            if hasattr(self, "imageOrganizerDlg") and self.imageOrganizerDlg is not None:
                print("[DEBUG] Closing existing Image Organizer dialog")
                self.imageOrganizerDlg.close()
        except Exception as e:
            print(f"[WARN] Could not close existing dialog: {e}")

        # Create and show new dialog
        try:
            print("[DEBUG] Creating new Image Organizer dialog")
            from GRIME_AI.dialogs.image_organizer.GRIME_AI_ImageOrganizerDlg import GRIME_AI_ImageOrganizerDlg
            self.imageOrganizerDlg = GRIME_AI_ImageOrganizerDlg(self)
            print("[DEBUG] Showing Image Organizer dialog")
            self.imageOrganizerDlg.show()
            print("[INFO] Image Organizer dialog launched successfully")
        except Exception as e:
            print(f"[ERROR] Failed to launch Image Organizer dialog: {e}")
            traceback.print_exc()  # full traceback to terminal


    # ======================================================================================================================
    # TEMPORAL AVERAGING
    # ======================================================================================================================
    def toolbarButtonTemporalAveraging(self):
        """Launch the Temporal Averaging dialog."""
        if not hasattr(self, "_temporalAvgDlg") or self._temporalAvgDlg is None:
            from GRIME_AI.dialogs.temporal_averaging.GRIME_AI_TemporalAveragingDlg import GRIME_AI_TemporalAveragingDlg
            self._temporalAvgDlg = GRIME_AI_TemporalAveragingDlg(self)
        self._temporalAvgDlg.show()
        self._temporalAvgDlg.raise_()
        self._temporalAvgDlg.activateWindow()

    # ==================================================================================================================
    # SAGE — SEGMENTATION & ANNOTATION FOR GEOSPATIAL ECOHYDROLOGY
    # ==================================================================================================================
    def menubar_launch_sage(self):
        """Launch SAGE as a separate process."""
        try:
            import subprocess
            subprocess.Popen([sys.executable, "-m", "SAGE.main"])
        except Exception as e:
            print(f"[ERROR] Failed to launch SAGE: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "SAGE Launch Failed", str(e))


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def toolbarButtonReleaseNotes(self):
        global frame
        from GRIME_AI.dialogs.release_notes.GRIME_AI_ReleaseNotesDlg import GRIME_AI_ReleaseNotesDlg
        releaseNotesDlg = GRIME_AI_ReleaseNotesDlg(frame)

        releaseNotesDlg.show()


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def toolbarButtonGRIME2(self):
        strMessage = 'Potential future home for GRIME2 Water Level/Stage measurement functionality.'
        msgBox = GRIME_AI_QMessageBox('Water Level Measurement', strMessage, QMessageBox.Close)
        response = msgBox.displayMsgBox()


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def toolbarButtonEdgeDetection(self):
        global frame
        from GRIME_AI.dialogs.edge_detection.GRIME_AI_EdgeDetectionDlg import GRIME_AI_EdgeDetectionDlg
        self.edgeDetectionDlg = GRIME_AI_EdgeDetectionDlg(frame)

        self.edgeDetectionDlg.edgeDetectionSignal.connect(self.edgeDetectionMethod)
        self.edgeDetectionDlg.featureDetectionSignal.connect(self.featureDetectionMethod)

        self.edgeDetectionDlg.show()

    # ==================================================================================================================
    #
    # ==================================================================================================================
    # Function to get prompts for each image
    def get_prompts(self, image_name):
        # Customize this function to return prompts based on the image name or content
        # For example:
        if 'cat' in image_name:
            return {'texts': ['cat']}
        elif 'dog' in image_name:
            return {'texts': ['dog']}
        # Add more conditions as needed
        return {'texts': ['default object description']}  # Default prompt


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def edgeDetectionMethod(self, edgeMethod):
        global g_edgeMethodSettings
        g_edgeMethodSettings = edgeMethod

        processLocalImage(self)

        self.refreshImage()

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def featureDetectionMethod(self, featureMethod):
        global g_edgeMethodSettings
        global g_featureMethodSettings

        g_edgeMethodSettings.method = edgeMethodsClass.NONE

        g_featureMethodSettings = featureMethod

        processLocalImage(self)

        self.refreshImage()


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def onMyToolBarFileFolder(self):
        global frame
        from GRIME_AI.dialogs.file_utilities.GRIME_AI_FileUtilitiesDlg import GRIME_AI_FileUtilitiesDlg
        self.fileFolderDlg = GRIME_AI_FileUtilitiesDlg(frame)

        self.fileFolderDlg.create_composite_slice_signal.connect(self.menubarCompositeSlices)
        self.fileFolderDlg.triage_images_signal.connect(self.toolbarButtonImageTriage_1)

        self.fileFolderDlg.accepted.connect(self.closeFilefolderDlg)
        self.fileFolderDlg.rejected.connect(self.closeFilefolderDlg)

        self.fileFolderDlg.show()

        try:
            global gFrameCount
            self.imageNavigationDlg.setImageCount(gFrameCount)
            self.imageNavigationDlg.reset()
        except Exception:
            pass


    # ------------------------------------------------------------------------------------------
    def closeFilefolderDlg(self):
            pass


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    def onMyToolBarImageNavigation(self):

        if self.imageNavigationDlg == None:
            global gFrameCount
            global currentImageCount

            if gFrameCount > 0:
                from GRIME_AI.dialogs.image_navigation.GRIME_AI_ImageNavigationDlg import GRIME_AI_ImageNavigationDlg
                self.imageNavigationDlg = GRIME_AI_ImageNavigationDlg(frame)
                self.imageNavigationDlg.imageIndexSignal.connect(self.getImageIndex)

                self.imageNavigationDlg.accepted.connect(self.closeNavigationDlg)
                self.imageNavigationDlg.rejected.connect(self.closeNavigationDlg)

                self.imageNavigationDlg.setImageList(dailyImagesList.getVisibleList())
                self.imageNavigationDlg.setImageIndex(currentImageIndex)
                self.imageNavigationDlg.setImageCount(gFrameCount)
                #self.imageNavigationDlg.reset()

                self.imageNavigationDlg.show()
            else:
                strMessage = 'You must first fetch images to navigate and/or operate on.'
                msgBox = GRIME_AI_QMessageBox('Image Navigation', strMessage, QMessageBox.Close)
                response = msgBox.displayMsgBox()

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def getImageIndex(self, imageIndex):
        global currentImageIndex

        currentImageIndex = imageIndex

        processLocalImage(self, imageIndex)
        self.refreshImage()


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def closeNavigationDlg(self):
        del self.imageNavigationDlg

        self.imageNavigationDlg = None


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    #@pyqtSlot()
    def onMyToolBarColorSegmentation(self):
        if self.colorSegmentationDlg == None:
            if self.maskEditorDlg == None:
                self.labelOriginalImage.setDrawingMode(DrawingMode.COLOR_SEGMENTATION)

                from GRIME_AI.dialogs.color_segmentation.GRIME_AI_ColorSegmentationDlg import GRIME_AI_ColorSegmentationDlg
                self.colorSegmentationDlg = GRIME_AI_ColorSegmentationDlg()

                self.colorSegmentationDlg.colorSegmentation_Signal.connect(self.colorSegmentation)
                self.colorSegmentationDlg.addROI_Signal.connect(self.trainROI)
                self.colorSegmentationDlg.deleteAllROI_Signal.connect(self.deleteAllROI)
                self.colorSegmentationDlg.buildFeatureFile_Signal.connect(self.buildFeatureFile)
                self.colorSegmentationDlg.universalTestButton_Signal.connect(self.universalTestButton)
                self.colorSegmentationDlg.refresh_rois_signal.connect(self.displayROIs)

                self.colorSegmentationDlg.greenness_index_signal.connect(self.greenness_index_changed)

                self.colorSegmentationDlg.close_signal.connect(self.closeColorSegmentationDlg)
                self.colorSegmentationDlg.accepted.connect(self.closeColorSegmentationDlg)
                self.colorSegmentationDlg.rejected.connect(self.closeColorSegmentationDlg)

                self.getColorSegmentationParams()

                self.colorSegmentationDlg.show()
            else:
                strMessage = 'Please close the Mask Editor toolbox if you want to use the Mask Editor toolbox.\nThis will be resolved in a future design change.'
                msgBox = GRIME_AI_QMessageBox('Tool Conflict', strMessage, QMessageBox.Yes | QMessageBox.No)
                response = msgBox.displayMsgBox()


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def greenness_index_changed(self):
        if self.colorSegmentationDlg != None:
            self.getColorSegmentationParams()

            # UPDATE THE FEATURE TABLE
            self.initROITable(self.greenness_index_list)

            processLocalImage(self, currentImageIndex)
            #JES - TROUBLESHOOT DISPLAY ISSUE
            # self.refreshImage()


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def closeColorSegmentationDlg(self):
        if self.colorSegmentationDlg != None:
            self.getColorSegmentationParams()

            self.colorSegmentationDlg.close()
            del self.colorSegmentationDlg
            self.colorSegmentationDlg = None

        self.labelOriginalImage.setDrawingMode(DrawingMode.OFF)


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def colorSegmentation(self, int):
        global dailyImagesList
        videoFileList = dailyImagesList.getVisibleList()

        myGRIMe_Color = GRIME_AI_Color()

        nImageIndex = 1

        if len(videoFileList) > 0:
            if nImageIndex > gFrameCount:
                nImageIndex = gFrameCount

            inputFrame = videoFileList[nImageIndex - 1].fullPathAndFilename  # zero based index

            if os.path.isfile(inputFrame):
                global currentImageFilename
                currentImageFilename = inputFrame
                numpyImage = myGRIMe_Color.loadColorImage(inputFrame)

                hsv = cv2.cvtColor(numpyImage, cv2.COLOR_BGR2HSV)

                # Threshold of blue in HSV space
                lower_blue = np.array([60, 35, 140])
                upper_blue = np.array([180, 255, 255])

                # preparing the mask to overlay
                mask = cv2.inRange(hsv, lower_blue, upper_blue)

                # The black region in the mask has the value of 0,
                # so when multiplied with original image removes all non-blue regions
                result = cv2.bitwise_and(numpyImage, numpyImage, mask=mask)

                cv2.imshow('frame', numpyImage)
                cv2.imshow('mask', mask)
                cv2.imshow('result', result)

                tempCurrentImage = QImage(numpyImage, numpyImage.shape[1], numpyImage.shape[0], QImage.Format_RGB888)
                currentImage = QPixmap(tempCurrentImage)


    # ==================================================================================================================
    # ==================================================================================================================
    # IMAGE MASK FUNCTIONALITY
    # ==================================================================================================================
    # ==================================================================================================================
    def onMyToolBarCreateMask(self):

        # JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES   JES
        # JES - PROVISIONAL - MASK CREATION IS NOT AVAILABLE FOR THE USGS SOFTWARE RELEASE
        strMessage = 'Mask Creation is not available in this software release.\nThis functionality may be consumed into other pre-existing functionality at some later date.'
        msgBox = GRIME_AI_QMessageBox('Tool Conflict', strMessage, QMessageBox.Yes | QMessageBox.No)
        response = msgBox.displayMsgBox(on_top=True)
        return
        # ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^

        if self.maskEditorDlg == None:
            if self.colorSegmentationDlg == None:
                self.labelOriginalImage.setDrawingMode(DrawingMode.MASK)

                from GRIME_AI.dialogs.mask_editor.GRIME_AI_MaskEditorDlg import GRIME_AI_MaskEditorDlg
                self.maskEditorDlg = GRIME_AI_MaskEditorDlg()

                self.maskEditorDlg.addMask_Signal.connect(self.addMask)
                self.maskEditorDlg.generateMask_Signal.connect(self.generateMask)
                self.maskEditorDlg.drawingColorChange_Signal.connect(self.changePolygonColor)
                self.maskEditorDlg.reset_Signal.connect(self.resetMask)
                self.maskEditorDlg.polygonFill_Signal.connect(self.fillPolygonChanged)

                self.maskEditorDlg.close_signal.connect(self.maskDialogClose)
                self.maskEditorDlg.close_signal.connect(self.maskDialogClose)
                self.maskEditorDlg.accepted.connect(self.maskDialogClose)
                self.maskEditorDlg.rejected.connect(self.maskDialogClose)

                self.maskEditorDlg.show()
            else:
                strMessage = 'Please close the Color Segmentatoin toolbox if you want to use the Mask Editor toolbox.\nThis will be resolved in a future design change.'
                msgBox = GRIME_AI_QMessageBox('Tool Conflict', strMessage, QMessageBox.Yes | QMessageBox.No)
                response = msgBox.displayMsgBox(on_top=True)


    # ------------------------------------------------------------------------------------------------------------------
    def maskDialogClose(self):
        if self.maskEditorDlg != None:
            self.maskEditorDlg.close()
            del self.maskEditorDlg
            self.maskEditorDlg = None

        self.labelOriginalImage.setDrawingMode(DrawingMode.OFF)

    # ------------------------------------------------------------------------------------------------------------------
    def fillPolygonChanged(self, bFill):
        self.labelOriginalImage.enablePolygonFill(bFill)

    # ------------------------------------------------------------------------------------------------------------------
    def resetMask(self):
        self.labelOriginalImage.resetMask()
        self.labelOriginalImage.update()

    # ------------------------------------------------------------------------------------------------------------------
    def addMask(self):
        self.labelOriginalImage.incrementPolygon()

    # ------------------------------------------------------------------------------------------------------------------
    def generateMask(self):
        global currentImageFilename

        scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio,
                                                 QtCore.Qt.SmoothTransformation)

        widthMultiplier = currentImage.size().width() / scaledCurrentImage.size().width()
        heightMultiplier = currentImage.size().height() / scaledCurrentImage.size().height()

        # CONVERT IMAGE TO A MAT FORMAT TO USE ITS PARAMETERS TO CREATE A MASK IMAGE TEMPLATE
        # --------------------------------------------------------------------------------------------------------------
        img1 = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())

        # CREATE A MASK IMAGE
        mask = np.zeros(img1.shape[:2], np.uint8)

        # ITERATE THROUGH EACH ONE OF THE POLYGONS
        # --------------------------------------------------------------------------------------------------------------
        polygonList = self.labelOriginalImage.getPolygon()

        for polygon in polygonList:
            myPoints = []
            for i in range(polygon.count()):
                myPoints.append([polygon.point(i).x() * widthMultiplier, polygon.point(i).y() * heightMultiplier])

            if len(myPoints) > 0:
                cv2.fillPoly(mask, np.int32([myPoints]), color=(255, 255, 255))

        masked = cv2.bitwise_and(img1, img1, mask=mask)

        # DISPLAY THE MASK IN THE GUI
        # --------------------------------------------------------------------------------------------------------------
        qImg = QImage(masked.data, masked.shape[1], masked.shape[0], QImage.Format_BGR888)
        pix = QPixmap(qImg)
        self.labelColorSegmentation.setPixmap(pix.scaled(self.labelColorSegmentation.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        # SAVE THE MASK RASTER AND POLYGON TO FILE
        if self.maskEditorDlg.getCheckBox_Save():
            # Extract image folder path to create a mask subfolder
            maskFolderPath = os.path.join(os.path.dirname(currentImageFilename), 'Masks')

            # Check for the existence of the filename path and create if it doesn't exist
            if not os.path.exists(maskFolderPath):
                os.makedirs(maskFolderPath)

            # Extract image filename to be used for creating the mask and polygon filenames
            filename = os.path.basename(currentImageFilename)
            filename_without_ext = filename[:filename.rindex('.')]
            extension = filename[filename.rindex('.'):]

            # Create filenames with fully qualified paths
            mask_filename = os.path.join(maskFolderPath, (filename_without_ext+'.mask.bmp'))
            poly_filename = os.path.join(maskFolderPath, (filename_without_ext+'.poly.csv'))

            bSave = True
            # Check for the existence of the files. If they exist, display overwrite option dialog box
            if os.path.isfile(mask_filename) or os.path.isfile(mask_filename):
                strMessage = 'The mask and/or polygon file exist. Overwrite files?'
                msgBox = GRIME_AI_QMessageBox('Save Mask Files', strMessage, QMessageBox.Yes | QMessageBox.No)
                response = msgBox.displayMsgBox()

                if response == QMessageBox.No:
                    bSave = False

            if bSave:
                # Write the mask to a file
                cv2.imwrite(mask_filename, mask)

                csvFile = open(poly_filename, 'w', newline='')

                # Write the polygon(s) vertices to a file
                for polygon in polygonList:
                    csvFile.write('mask\n')
                    csvFile.write('x, y\n')

                    for myPoints in polygon:
                        x = (int)(myPoints.x() * widthMultiplier)
                        y = (int)(myPoints.y() * heightMultiplier)
                        outputString = "{0}, {1}\n".format(x, y)
                        csvFile.write(outputString)

                # WRITE ALL POLYGONS BEFORE CLOSING FILE
                csvFile.close()

                # EXTRACT DOMINANT RGB COLORS
                myGRIMe_Color = GRIME_AI_Color()

                _, _, hist = myGRIMe_Color.KMeans(masked, 6)

                # EXTRACT DOMINANT HSV COLORS
                hist, colorClusters = myGRIMe_Color.extractDominant_HSV(masked, 6)

                # CREATE COLOR BAR TO DISPLAY CLUSTER COLORS
                colorBar = GRIME_AI_Color.create_color_bar(hist, colorClusters[0:5])

                # CONVERT colorBar TO A QImage FOR USE IN DISPLAYING IN QT GUI
                qImg = QImage(colorBar.data, colorBar.shape[1], colorBar.shape[0], QImage.Format_BGR888)

    # ------------------------------------------------------------------------------------------------------------------
    def changePolygonColor(self, polygonColor):
        self.labelOriginalImage.setBrushColor(polygonColor)
        self.labelOriginalImage.drawPolygon()

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def deleteAllROI(self):
        del self.roiList[:]

        self.tableWidget_ROIList.clearContents()
        self.tableWidget_ROIList.setRowCount(0)
        self._whole_image_ref_set = False
        self._roi_ref_set = set()

        # Clear ROI overlays from the image display immediately
        self.labelOriginalImage.setROIs(self.roiList)

        self.colorSegmentationDlg.disable_spinbox_color_clusters(False)

        # Pass current image index so we stay on the same image
        processLocalImage(self, currentImageIndex)

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def onMyToolBarSettings(self):
        pass

    # ==================================================================================================================
    # THESE EVENT FILTERS WILL BE USED TO TRACK MOUSE MOVEMENT AND MOUSE BUTTON CLICKS FOR DISPLAYING ADDITIONAL
    # INFORMATION, VIEWS, POP-UP MENUS AND DRAWING REGIONS-OF-INTEREST (ROI) AROUND SPECIFIC AREAS OF AN IMAGE.
    # ==================================================================================================================
    def eventFilter(self, source, event):

        if event.type() == QtCore.QEvent.MouseMove and source is self.labelEdgeImage:
            # print("A")
            pass

        if event.type() == QtCore.QEvent.MouseMove and source is self.labelOriginalImage:
            if 0:
                x, y = pyautogui.position()
                pixelColor = pyautogui.screenshot().getpixel((x, y))
                ss = 'Screen Pos - X:' + str(x).rjust(4) + ' Y:' + str(y).rjust(4)
                ss += ' RGB: (' + str(pixelColor[0]).rjust(3)
                ss += ', ' + str(pixelColor[1]).rjust(3)
                ss += ', ' + str(pixelColor[2]).rjust(3) + ')'
                print(ss)
                # print("B")
            pass

        ###JES if event.type() == QtCore.QEvent.MouseButtonDblClick and source is self.labelOriginalImage:
            # labelEdgeImageDoubleClickEvent(self)
            # labelMouseDoubleClickEvent(self)
            ###JES NEON_labelOriginalImageDoubleClickEvent(self)

        return super(MainWindow, self).eventFilter(source, event)

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def closeEvent(self, event):
        # DESTROY ANY MODELESS DIALOG BOXES THAT ARE OPEN
        if self.edgeDetectionDlg != None:
            self.edgeDetectionDlg.close()

        if self.colorSegmentationDlg != None:
            self.colorSegmentationDlg.close()

        if self.TriageDlg != None:
            self.TriageDlg.close()

        if self.fileFolderDlg != None:
            self.fileFolderDlg.close()

        if self.maskEditorDlg != None:
            self.maskEditorDlg.close()

        if self.imageNavigationDlg != None:
            self.imageNavigationDlg.close()

        if self.releaseNotesDlg != None:
            self.releaseNotesDlg.close()

        global hyperparameterDlg
        if hyperparameterDlg != None:
            hyperparameterDlg.close()

        #webdriver.Chrome.quit()

        QMainWindow.closeEvent(self, event)

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def fetchImageList(self, imageFolder, bRecursive=False):
        global imageFileFolder
        imageFileFolder = imageFolder
        fetchLocalImageList(self, imageFileFolder, bRecursive, False) #, start_date, end_date, start_time, end_time)

        try:
            global gFrameCount
            global dailyImagesList

            self.onMyToolBarImageNavigation()
            self.imageNavigationDlg.setImageCount(gFrameCount)
            self.imageNavigationDlg.setImageList(dailyImagesList.getVisibleList())
            self.imageNavigationDlg.reset()
        except Exception:
            pass

    # ======================================================================================================================
    # THIS FUNCTION WILL CALL THE FUNCTION THAT PROCESSES THE IMAGE BASED UPON THE SETTINGS SELECTED BY THE
    # END-USER AND THEN UPDATE THE GUI TO DISPLAY THE PROCESSED IMAGE.
    # ======================================================================================================================
    def refreshImage(self):
        global currentImage

        '''// PROCESS THE ORIGINAL IMAGE //'''
        pix = processImage(self, currentImage)

        '''// DISPLAY PROCESSED ORIGINAL IMAGE //'''
        if not pix == []:
            self.labelEdgeImage.setPixmap(
                pix.scaled(self.labelEdgeImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            img = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())

        # CALL PROCESSEVENTS IN ORDER TO UPDATE GUI
        QCoreApplication.processEvents()

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def NEON_updateSiteProducts(self, item):
        site_json = NEON_API().FetchSiteInfoFromNEON(SERVER, SITECODE)

        self.NEON_listboxSiteProducts.clear()

        for product in site_json['data']['dataProducts']:
            strText = product['dataProductCode'] + ": " + product['dataProductTitle']
            assert isinstance(strText, object)
            self.NEON_listboxSiteProducts.addItem(strText)

        self.NEON_listboxSiteProducts.show()

        #JES - TEMPORARILY SET NITRATE DATA ('should only be one nitrate product') AS THE DEFAULT SELECTION
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        itemNitrate = self.NEON_listboxSiteProducts.findItems('Nitrate', QtCore.Qt.MatchContains)
        nIndex = 0
        if len(itemNitrate) > 0:
            for item in itemNitrate:
                nIndex = self.NEON_listboxSiteProducts.row(item)
                self.NEON_listboxSiteProducts.setCurrentRow(nIndex)

            NEON_updateProductTable(self, nIndex)

        #JES - TEMPORARILY SET NITRATE DATA ('should only be one nitrate product') AS THE DEFAULT SELECTION
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        item20002 = self.NEON_listboxSiteProducts.findItems('20002', QtCore.Qt.MatchContains)
        nIndex = 0
        if len(item20002) > 0:
            for item in item20002:
                nIndex = self.NEON_listboxSiteProducts.row(item)
                self.NEON_listboxSiteProducts.setCurrentRow(nIndex)

            NEON_updateProductTable(self, nIndex)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        targets = ["20002", "00042", "00033"]

        matches = []
        for t in targets:
            found = self.NEON_listboxSiteProducts.findItems(t, QtCore.Qt.MatchContains)
            if found:
                matches.extend(item.text() for item in found)

        num_matches = len(matches)
        print("Matches found:", num_matches)
        print("Matched values:", matches)

        #self.NEON_listboxSiteProducts.item(0).setToolTip("Hello?")


        return num_matches, matches

    # ======================================================================================================================
    # THIS FUNCTION WILL DISPLAY THE LATEST IMAGE ON THE GUI.
    # ======================================================================================================================
    def _neon_preview_received(self, pixmap, error_msg):
        global gWebImagesAvailable
        if pixmap and not pixmap.isNull():
            gWebImagesAvailable = 1
            self.NEON_latestImage = pixmap
            self.NEON_DisplayLatestImage()
        else:
            gWebImagesAvailable = 0
            self.NEON_labelLatestImage.setText(error_msg or "No Images Available")

    def _neon_tree_item_clicked(self, item, column):
        """Open URL in browser if the clicked NEON tree item is a hyperlink."""
        url = item.data(0, QtCore.Qt.UserRole + 1)
        if url:
            from PyQt5.QtGui import QDesktopServices
            from PyQt5.QtCore import QUrl
            QDesktopServices.openUrl(QUrl(url))

    def NEON_DisplayLatestImage(self):

        if self.NEON_latestImage == []:
            self.NEON_labelLatestImage.setText("No Images Available")
        elif self.NEON_labelLatestImage.width() > 1:
            self.NEON_labelLatestImage.clear()
            self.NEON_labelLatestImage.setPixmap(self.NEON_latestImage.scaled(self.NEON_labelLatestImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def WholeImage_ExtractFeatures(self, img, bWholeImageCalc):
        if bWholeImageCalc:
            # BLUR THE IMAGE
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # IMAGE INTENSITY CALCULATIONS
            intensity = cv2.mean(gray)[0]  # The range for a pixel's value in grayscale is (0-255), 127 lies midway
            strIntensity = '%3.3f' % (intensity)

            # COMPUTE ENTROPY FOR ENTIRE IMAGE
            entropyValue = self.calcEntropy(gray)
            strEntropy = '%3.3f' % (entropyValue)
        else:
            strIntensity = '---'
            strEntropy = '---'

        return strIntensity, strEntropy

    # ==================================================================================================================
    #  WHOLE IMAGE - EXTRACT FEATURES (GREENNESS INDEX, INTENSITY, ENTROPY, ETC.)
    # ==================================================================================================================0
    def whole_image_feature_extraction(self, img):
        features = self.compute_whole_image_features(img)
        self.display_whole_image_features(features)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_whole_image_features(self, img):
        """Return computed features for the whole image without touching the UI."""
        strIntensity, strEntropy = self.WholeImage_ExtractFeatures(
            img, self.colorSegmentationParams.wholeImage
        )

        # Extract dominant HSV colors
        hist, colorClusters = GRIME_AI_Color.extractDominant_HSV(
            img, self.colorSegmentationParams.numColorClusters
        )
        colorBar = GRIME_AI_Color.create_color_bar(hist, colorClusters)

        # Compute greenness values
        greenness_values = []
        for index, greenness in enumerate(self.greenness_index_list):
            greenness_updated = GRIME_AI_Vegetation_Indices().get_greenness(greenness, img)
            self.greenness_index_list[index] = greenness_updated
            greenness_values.append(greenness_updated.get_value())

        return {
            "intensity": strIntensity,
            "entropy": strEntropy,
            "colorBar": colorBar,
            "greenness": greenness_values,
        }

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def display_whole_image_features(self, features):
        """Update the Qt table with the computed features."""
        nRow = self.tableWidget_ROIList.rowCount()
        if nRow == 0:
            self.tableWidget_ROIList.insertRow(nRow)

        # Column 0: Label
        wholeImageLabel = QtWidgets.QLabel("Whole Image")
        self.tableWidget_ROIList.setCellWidget(0, 0, wholeImageLabel)

        # Column 1: Color bar
        qImg = QImage(
            features["colorBar"].data,
            features["colorBar"].shape[1],
            features["colorBar"].shape[0],
            QImage.Format_BGR888,
        )
        pixmap = QPixmap.fromImage(qImg)

        # Scale to fit cell
        cell_width = self.tableWidget_ROIList.columnWidth(0)
        cell_height = self.tableWidget_ROIList.rowHeight(0)
        pixmap_scaled = pixmap.scaledToHeight(cell_height, QtCore.Qt.SmoothTransformation)
        if pixmap_scaled.width() != cell_width:
            h_scale = cell_width / pixmap_scaled.width()
            transform = QtGui.QTransform()
            transform.scale(h_scale, 1)
            pixmap_scaled = pixmap_scaled.transformed(transform, QtCore.Qt.SmoothTransformation)

        label = QtWidgets.QLabel()
        label.setPixmap(pixmap_scaled)

        if not self._whole_image_ref_set:
            self.tableWidget_ROIList.setCellWidget(0, 1, label)
            self._whole_image_ref_set = True
            na_Label = QtWidgets.QLabel("n/a")
            self.tableWidget_ROIList.setCellWidget(0, 2, na_Label)
        else:
            self.tableWidget_ROIList.setCellWidget(0, 2, label)

        # Columns 3+: Greenness values
        col = 3
        for g in features["greenness"]:
            greennessLabel = QtWidgets.QLabel("{:.3f}".format(g))
            self.tableWidget_ROIList.setCellWidget(0, col, greennessLabel)
            col += 1

        # Intensity
        intensityLabel = QtWidgets.QLabel(features["intensity"])
        self.tableWidget_ROIList.setCellWidget(0, col, intensityLabel)
        col += 1

        # Entropy
        entropyLabel = QtWidgets.QLabel(features["entropy"])
        self.tableWidget_ROIList.setCellWidget(0, col, entropyLabel)

    # ==================================================================================================================
    #  ROI (region-of-interest) - EXTRACT FEATURES (GREENNESS INDEX, INTENSITY, ENTROPY, ETC.)
    # ==================================================================================================================
    def roi_feature_extraction(self, img, roiList):
        progressBar = QProgressWheel()
        progressBar.setRange(0, len(roiList))

        for row, roiObj in enumerate(roiList, start=1):
            progressBar.setValue(row)

            try:
                features = self.compute_roi_features(roiObj, img)
                self.display_roi_features(row, features)
                # ROI overlays drawn by GRIME_AI_QLabel.paintEvent via savedROIs
            except Exception as e:
                import traceback
                print(f"Error processing ROI {roiObj.getROIName()}: {e}")
                traceback.print_exc()

        progressBar.close()
        del progressBar

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @dataclass
    class ROIFeatures:
        roi_name: str
        color_bar: np.ndarray
        greenness: list[float]
        intensity: float
        entropy: float

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_roi_features(self, roiObj, img) -> 'MainWindow.ROIFeatures':
        rgb = extractROI(roiObj.getImageROI(), img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Dominant HSV colors
        hist, colorClusters = GRIME_AI_Color.extractDominant_HSV(rgb, roiObj.getNumColorClusters())
        colorBar = GRIME_AI_Color.create_color_bar(hist, colorClusters)

        # Greenness indices
        greenness_values = []
        for index, greenness in enumerate(self.greenness_index_list):
            greenness_updated = GRIME_AI_Vegetation_Indices().get_greenness(greenness, rgb)
            self.greenness_index_list[index] = greenness_updated
            greenness_values.append(greenness_updated.get_value())

        # Intensity
        intensity = cv2.mean(gray)[0]

        # Entropy
        entropy = self.calcEntropy(gray)

        return MainWindow.ROIFeatures(
            roi_name=roiObj.getROIName(),
            color_bar=colorBar,
            greenness=greenness_values,
            intensity=intensity,
            entropy=entropy,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def display_roi_features(self, row: int, features: 'MainWindow.ROIFeatures'):
        # Column 0: ROI name set once at draw time — do not overwrite here

        # Column 2: Color bar
        qImg = QImage(features.color_bar.data,
                      features.color_bar.shape[1],
                      features.color_bar.shape[0],
                      QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qImg)

        cell_width = self.tableWidget_ROIList.columnWidth(2)
        cell_height = self.tableWidget_ROIList.rowHeight(row)
        pixmap_scaled = pixmap.scaled(cell_width, cell_height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        label = QtWidgets.QLabel()
        label.setPixmap(pixmap_scaled)
        # Column 2: Cur. Image — always update; col 1 (Ref.) set at ROI draw time
        self.tableWidget_ROIList.setCellWidget(row, 2, label)

        # Columns 3+: Greenness
        col = 3
        for g in features.greenness:
            self.tableWidget_ROIList.setCellWidget(row, col, QtWidgets.QLabel(f"{g:.3f}"))
            col += 1

        # Intensity
        self.tableWidget_ROIList.setCellWidget(row, col, QtWidgets.QLabel(f"{features.intensity:.4f}"))
        col += 1

        # Entropy
        self.tableWidget_ROIList.setCellWidget(row, col, QtWidgets.QLabel(f"{features.entropy:.4f}"))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def draw_roi_overlay(self, painter, roiObj):
        pen = QPen(QtCore.Qt.red, 1, QtCore.Qt.SolidLine)
        painter.setPen(pen)

        if roiObj.getROIShape() == ROIShape.RECTANGLE:
            painter.drawRect(roiObj.getDisplayROI())
        elif roiObj.getROIShape() == ROIShape.ELLIPSE:
            painter.drawEllipse(roiObj.getDisplayROI())

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(
            roiObj.getDisplayROI().x(),
            roiObj.getDisplayROI().y() - 16,
            50, 16,
            QtCore.Qt.AlignLeft,
            roiObj.getROIName()
        )


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
def fetchLocalImageList(self, filePath, bFetchRecursive, bCreateEXIFFile, start_date=datetime.date(1970, 1, 1),
                        end_date=datetime.date(2099, 12, 31), start_time='000000', end_time='000000'):
    global gWebImageCount
    global dailyImagesList

    # CLEAR THE PREVIOUSLY DOWNLOADED IMAGE LIST, IF ANY
    dailyImagesList.clear()

    # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # INITIALIZE VARIABLES TO A KNOWN VALUE
    nStartDate = 99999999
    nEndDate = -1
    strStartDate = ""
    strEndDate = ""
    List = []

    # THE SOFTWARE IS NOW DESIGNED TO REQUIRE THE IMAGES TO BE DOWNLOADED FIRST FOR A VARIETY OF REASONS
    # bSaveImages = True
    bSaveImages = False     #JES - Is this flag needed any longer?
    imageOutputFolder = self.fileFolderDlg.lineEdit_images_folder.text()

    # if self.checkBoxSaveImages.isChecked():
    #    bSaveImages = True
    #    imageOutputFolder = self.EditSaveImagesOutputFolder.text()

    # create the csv writer
    if bCreateEXIFFile:
        # open the file in the write mode
        EXIFFolder = self.EditEXIFOutputFolder.text()
        csvFile = open(EXIFFolder + '/' + 'EXIFData.csv', 'w', newline='')

        writer = csv.writer(csvFile)

        bWriteHeader = True

    # count the number of images that will potentially be processed and possibly saved with the specified extension
    # to display an "hourglass" to give an indication as to how long the process will take. Furthermore, the number
    # of images will help determine whether or not there is enough disk space to accomodate storing the images.
    imageCount = GRIME_AI_Utils().get_image_count(filePath, extensions)

    # RECURSE AND TRAVERSE FROM THE SPECIFIED FOLDER DOWN TO DETERMINE THE DATE RANGE FOR THE IMAGES FOUND
    file_count, files = GRIME_AI_Utils().getFileList(filePath, extensions, bFetchRecursive)

    if bShow_GUI:
        progressBar = QProgressWheel()
        progressBar.setRange(0, file_count + 1)
        progressBar.show()

    # traverse all files in folder that meet the criteria for retrieval
    # 1. does the file have the specified file extension
    # 2. extract the date from the filename
    # 3. does the file meet the date criteria? if "yes," then continue; if "no" check next file in the list
    # 4. if the files meets the date criteria, extract the EXIF data from the file to ascertain the time at
    #    which the image was acquired
    # 5. if the file meets the time range criteria, add the file to a list of images to be used for the session
    #    and also add the file's EXIF data to a CSV EXIF log file if the option is selected by the user. Last but
    #    not least, if the user selects the option to copy the image to a separate folder, then copy the file to
    #    the folder specified by the user
    print("File Count: ", file_count)

    image_index = 0
    while image_index < file_count:
        file = files[image_index]
        if bShow_GUI:
            progressBar.setWindowTitle(file)
            progressBar.setValue(image_index)
            progressBar.repaint()

        ext = os.path.splitext(file)[-1].lower()

        if ext in extensions:
            fileDate, fileTime = GRIME_AI_TimeStamp_Utils().extractDateFromFilename(file)

            if fileDate >= start_date and fileDate <= end_date:
                fullPathAndFilename = file

                try:
                    # extract EXIF info to determine what time the image was acquired. If EXIF info is not found,
                    # throw an exception and see if the information is embedded in the filename. Currently, we are
                    # working with images from NEON and PBT. The PBT images have EXIF data and the NEON/PhenoCam
                    # do not appear to have EXIF data.
                    myEXIFData = EXIFData()
                    myEXIFData.extractEXIFData(fullPathAndFilename)

                    strTemp = str(myEXIFData.getEXIF()[8])
                    timeOriginal = re.search(' \d{2}:\d{2}:\d{2}', strTemp).group(0)

                    nHours = int(str(timeOriginal[1:3]))
                    nMins = int(str(timeOriginal[4:6]))
                    nSecs = int(str(timeOriginal[7:9]))

                    bEXIFDataFound = True
                except Exception:
                    # assume the filename contains the timestamp for the image (assumes the image file is a PBT image)
                    bEXIFDataFound = False

                    try:
                        nHours = int(str(strTime[0:2]))
                        nMins = int(str(strTime[2:4]))
                        nSecs = int(str(strTime[4:6]))
                    except Exception:
                        nHours = 0
                        nMins = 0
                        nSecs = 0

                image_time = datetime.time(nHours, nMins, nSecs)

                # if ((start_time == datetime.time(0, 0, 0)) and (end_time == datetime.time(0, 0, 0))) or \
                #        ((image_time >= start_time) and (image_time <= end_time)):

                # WRITE THE HEADER ONLY ONCE WHEN THE FIRST FILE IS PROCESSED
                if bCreateEXIFFile and bEXIFDataFound:
                    if bWriteHeader:
                        writer.writerow(myEXIFData.getHeader())
                        bWriteHeader = False
                    else:
                        writer.writerow(myEXIFData.getEXIF())

                List.append(imageData(fullPathAndFilename, 0, 0, 0))

                if bSaveImages:
                    shutil.copy(fullPathAndFilename, imageOutputFolder)

                # delete EXIFData object
                del myEXIFData

        image_index += 1

    dailyImagesList.setVisibleList(List)

    global gFrameCount
    gFrameCount = len(dailyImagesList.getVisibleList())
    gWebImageCount = len(dailyImagesList.getVisibleList())

    # INIT SPINBOX CONTROLS BASED UPON NUMBER OF IMAGES AVAILABLE
    # dailyURLvisible = []

    # clean-up before exiting function
    # 1. close and delete the progress bar
    # 2. close the EXIF log file, if opened
    if bShow_GUI:
        progressBar.close()
        del progressBar

    if bCreateEXIFFile:
        csvFile.close()

    if len(files) > 0:
        processLocalImage(self)
        #refreshImage(self)

# ======================================================================================================================
#
# ======================================================================================================================
def processLocalImage(self, nImageIndex=0, imageFileFolder=''):
    global currentImage

    myGRIMe_Color = GRIME_AI_Color()

    # videoFilePath = Path(frameFolder)
    ##JES videoFileList = [str(pp) for pp in videoFilePath.glob("**/*.jpg")]
    # videoFileList = [str(pp) for pp in videoFilePath.glob("*.jpg")]

    global dailyImagesList
    videoFileList = dailyImagesList.getVisibleList()

    if len(videoFileList) > 0:
        if nImageIndex > gFrameCount:
            nImageIndex = gFrameCount

        inputFrame = videoFileList[nImageIndex - 1].fullPathAndFilename  # zero based index

        if os.path.isfile(inputFrame):
            global currentImageFilename
            currentImageFilename = inputFrame
            numpyImage = myGRIMe_Color.loadColorImage(inputFrame)

            numpyImage = np.ascontiguousarray(numpyImage)
            bytes_per_line = numpyImage.strides[0]
            tempCurrentImage = QImage(numpyImage.data, numpyImage.shape[1], numpyImage.shape[0], bytes_per_line, QImage.Format_RGB888)
            currentImage = QPixmap.fromImage(tempCurrentImage.copy())

    # ------------------------------------------------------------------------------------------------------------------
    # DISPLAY IMAGE FROM NEON SITE
    # ------------------------------------------------------------------------------------------------------------------
    if currentImage:
        numpyImg = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())

        scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        width = scaledCurrentImage.width()
        height = scaledCurrentImage.height()

        currentImageRescaled = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        img = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())

        self.whole_image_feature_extraction(img)

        self.roi_feature_extraction(img, self.roiList)

        self.labelOriginalImage.setPixmap(currentImageRescaled)

        pix = processImage(self, currentImage)
        gray = cv2.cvtColor(numpyImage, cv2.COLOR_BGR2GRAY)

        '''
        entropy_image = entropy(gray, disk(7))
        npa = np.asarray(entropy_image, dtype=np.float64) * 255
        npa = npa.astype(np.uint8)
        colorImg = cv2.applyColorMap(npa, cv2.COLORMAP_MAGMA)
        qImg = QImage(colorImg.data, colorImg.shape[1], colorImg.shape[0], QImage.Format_RGB888)
        pix = QPixmap(qImg)

        self.labelEdgeImage.setPixmap(pix.scaled(self.labelEdgeImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        '''

        del currentImageRescaled
        del scaledCurrentImage

    # CALL PROCESSEVENTS IN ORDER TO UPDATE GUI
    QCoreApplication.processEvents()


# ======================================================================================================================
# THIS FUNCTION WILL PROCESS THE CURRENT IMAGE BASED UPON THE SETTINGS SELECTED BY THE END-USER.
# THE IMAGE STORAGE TYPE IS QImage
# ======================================================================================================================
def processImage(self, myImage):
    global g_edgeMethodSettings
    global g_featureMethodSettings

    pix = []

    if not myImage == []:
        # CONVERT IMAGE FROM QImage FORMAT TO Mat FORMAT (BYTE ORDER IS R, G, B)
        img1 = GRIME_AI_Utils().convertQImageToMat(myImage.toImage())

        #JES if self.checkboxKMeans.isChecked():
        #    myGRIMe_Color = GRIMe_Color()
        #    qImg, clusterCenters, hist = myGRIMe_Color.KMeans(img1, self.spinBoxColorClusters.value())

        # CONVERT COLOR IMAGE TO GRAY SCALE
        gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

        # REMOVE NOISE FROM THE IMAGE
        if len(gray) != 0:
            grayEQ = cv2.equalizeHist(gray)
            grayBlur = cv2.GaussianBlur(grayEQ, (15, 15), 0)
            cv2.erode(grayBlur, (7,7), gray)

        # EDGE DETECTION METHODS
        if len(gray) != 0:

            from GRIME_AI.GRIME_AI_ProcessImage import GRIME_AI_ProcessImage
            myProcessImage = GRIME_AI_ProcessImage()

            if g_edgeMethodSettings.method == edgeMethodsClass.CANNY:
                pix = myProcessImage.processCanny(img1, gray, g_edgeMethodSettings)

            elif g_edgeMethodSettings.method == edgeMethodsClass.LAPLACIAN:
                pix = myProcessImage.processLaplacian(img1)

            elif g_edgeMethodSettings.method == edgeMethodsClass.SOBEL_X or g_edgeMethodSettings.method == edgeMethodsClass.SOBEL_Y or g_edgeMethodSettings.method == edgeMethodsClass.SOBEL_XY:
                pix = myProcessImage.processSobel(gray, g_edgeMethodSettings.getSobelKernel(), g_edgeMethodSettings.method)

            elif g_featureMethodSettings.method == featureMethodsClass.SIFT:
                pix = myProcessImage.processSIFT(img1, gray)

            elif g_featureMethodSettings.method == featureMethodsClass.ORB:
                pix = myProcessImage.processORB(img1, gray, g_featureMethodSettings)

    return pix


# ======================================================================================================================
#
# ======================================================================================================================
def closehyperparameterDlg():

    global hyperparameterDlg
    del hyperparameterDlg
    hyperparameterDlg = None


# ======================================================================================================================
#
# ======================================================================================================================
def extractROI(rect, image):
    return(image[rect.y():rect.y() + rect.height(), rect.x():rect.x() + rect.width()])


# ======================================================================================================================
#
# ======================================================================================================================
'''
def closest_colour(requested_colour):
    min_colours = {}

    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name

    return min_colours[min(min_colours.keys())]


# ======================================================================================================================
#
# ======================================================================================================================
def top_colors(image, n):
    import pandas as pd
    # convert the image to rgb
    # image = image.convert('RGB')

    # resize the image to 300 x 300
    # image = image.resize((300, 300))

    detected_colors = []
    for x in range(image.width):
        for y in range(image.height):
            detected_colors.append(closest_colour(image.getpixel((x, y))))
    Series_Colors = pd.Series(detected_colors)
    output = Series_Colors.value_counts() / len(Series_Colors)
    return (output.head(n))
'''


# ======================================================================================================================
# THIS FUNCTION UPDATES THE GUI WITH THE INFO FOR A NEON SITE SELECTED BY THE END-USER.
# ======================================================================================================================
def NEON_updateSiteInfo(self):

    # EXTRACT THE SITE ID FROM UserRole (avoids parsing the display label)
    item = self.NEON_listboxSites.currentItem()
    if item is None:
        return None
    top = item if item.parent() is None else item.parent()

    global SITECODE
    SITECODE = top.data(0, QtCore.Qt.UserRole) or top.text(0).split(' - ')[0]
    print (f'Fetching site info from NEON server for {SITECODE}...')
    siteInfo = NEON_API().FetchSiteInfoFromNEON(SERVER, SITECODE)

    global DOMAINCODE
    DOMAINCODE = siteInfo['data']['domainCode']

    # Enrich the selected tree item with API fields (state, domain, deimsId)
    data = siteInfo.get('data', {})
    extra_fields = [
        ("State",       data.get('stateCode', '')),
        ("Domain Code", data.get('domainCode', '')),
        ("Domain Name", data.get('domainName', '')),
    ]
    deims_url = data.get('deimsId', '')

    # Remove any previously-added API children (marked with UserRole+3)
    i = 0
    while i < top.childCount():
        if top.child(i).data(0, QtCore.Qt.UserRole + 3):
            top.removeChild(top.child(i))
        else:
            i += 1

    for label, value in extra_fields:
        if value:
            child = QTreeWidgetItem([f"{label}: {value}"])
            child.setFlags(child.flags() & ~QtCore.Qt.ItemIsSelectable)
            child.setData(0, QtCore.Qt.UserRole + 3, True)
            top.addChild(child)

    if deims_url:
        link_item = QTreeWidgetItem([f"DEIMS ID: {deims_url}"])
        link_item.setData(0, QtCore.Qt.UserRole + 1, deims_url)
        link_item.setData(0, QtCore.Qt.UserRole + 3, True)
        link_item.setForeground(0, QtGui.QBrush(QtGui.QColor("#1a6fc4")))
        font = link_item.font(0)
        font.setUnderline(True)
        link_item.setFont(0, font)
        top.addChild(link_item)

    return (SITECODE)


# ======================================================================================================================
# THIS FUNCTION WILL UPDATE THE PRODUCT TABLE IN THE GUI WITH THE PRODUCTS THAT ARE AVAILABLE FOR A SPECIFIC SITE.
# ======================================================================================================================
def NEON_updateProductTable(self, item):
    products = self.NEON_listboxSiteProducts.selectedItems()

    #JES: FUTURE CONSIDERATION - MUST MAKE CODE DYNAMIC TO ONLY DELETE UNSELECTED ITEMS
    for i in range(self.NEON_selected_products.rowCount()):
        self.NEON_selected_products.removeRow(0)

    for i in range(len(products)):
        strText = self.NEON_listboxSiteProducts.selectedItems()[i].text()
        self.NEON_selected_products.insertRow(i)

        productID = strText.split(':')[0]
        availableMonthList = findAvailableMonths(productID)

        length = len(availableMonthList['availableMonths'])
        strFirstMonth = availableMonthList['availableMonths'][0]
        strLastMonth = availableMonthList['availableMonths'][length - 1]

        monthFields = strFirstMonth.split('-')
        firstMonth = int(monthFields[1])
        firstYear = int(monthFields[0])

        monthFields = strLastMonth.split('-')
        lastMonth = int(monthFields[1])
        lastYear = int(monthFields[0])

        m = 0
        self.NEON_selected_products.setItem(i, m, QTableWidgetItem(strText))

        # CONFIGURE DATES - columns: 0=Site, 1=Image Count, 2=Start Date, 3=End Date, 4=Start Time, 5=End Time
        nYear, nMonth, nDay = GRIME_AI_PhenoCam().getEndDate()

        m += 2
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(nYear, nMonth, nDay))
        date_widget.dateTimeChanged.connect(lambda: NEON_dateChangeMethod(date_widget, self.NEON_selected_products, not self.pushButton_SyncDates.isChecked()))
        date_widget.setKeyboardTracking(False)
        self.NEON_selected_products.setCellWidget(i, m, date_widget)

        m += 1
        date_widget = QtWidgets.QDateEdit(calendarPopup=True)
        date_widget.setDate(QtCore.QDate(nYear, nMonth, nDay))
        date_widget.dateTimeChanged.connect(lambda: NEON_dateChangeMethod(date_widget, self.NEON_selected_products, not self.pushButton_SyncDates.isChecked()))
        date_widget.setKeyboardTracking(False)
        self.NEON_selected_products.setCellWidget(i, m, date_widget)

        m += 1
        dateTime = QDateTimeEdit()
        dateTime.setDisplayFormat("hh:mm")
        dateTime.setFrame(False)
        dateTime.dateTimeChanged.connect(lambda: NEON_dateChangeMethod(date_widget, self.NEON_selected_products, not self.pushButton_SyncDates.isChecked()))
        self.NEON_selected_products.setCellWidget(i, m, dateTime)

        m += 1
        dateTime = QDateTimeEdit()
        dateTime.setDisplayFormat("hh:mm")
        dateTime.setFrame(False)
        dateTime.dateTimeChanged.connect(lambda: NEON_dateChangeMethod(date_widget, self.NEON_selected_products, not self.pushButton_SyncDates.isChecked()))
        self.NEON_selected_products.setCellWidget(i, m, dateTime)

        self.NEON_selected_products.resizeColumnsToContents()

# ======================================================================================================================
#
# ======================================================================================================================
def NEON_dateChangeMethod(date_widget, tableWidget, bUniqueDates):
    global SITECODE
    global DOMAINCODE

    nRow = tableWidget.currentIndex().row()

    strProductIDCell = tableWidget.item(nRow, 0).text().upper()

    # FETCH DATE THAT CHANGED FOR THE SPECIFIC ROW
    start_date, start_time, end_date, end_time = GRIME_AI_ProductTable().fetchTableDates(tableWidget, nRow)

    if bUniqueDates == False:
        for i in range(tableWidget.rowCount()):
            if tableWidget.cellWidget(i, 2):
                tableWidget.cellWidget(i, 2).setDate(start_date)
            if tableWidget.cellWidget(i, 3):
                tableWidget.cellWidget(i, 3).setDate(end_date)
            if tableWidget.cellWidget(i, 4):
                tableWidget.cellWidget(i, 4).setDateTime(QtCore.QDateTime(QtCore.QDate(1970, 1, 1), QtCore.QTime(start_time.hour, start_time.minute)))
            if tableWidget.cellWidget(i, 5):
                tableWidget.cellWidget(i, 5).setDateTime(QtCore.QDateTime(QtCore.QDate(1970, 1, 1), QtCore.QTime(end_time.hour, end_time.minute)))
    else:
        if tableWidget.cellWidget(nRow, 2):
            tableWidget.cellWidget(nRow, 2).setDate(start_date)
        if tableWidget.cellWidget(nRow, 3):
            tableWidget.cellWidget(nRow, 3).setDate(end_date)

    #imageCount = GRIME_AI_PhenoCam.getPhenocamImageCount(SITECODE, DOMAINCODE, start_date, end_date, start_time, end_time)

    #tableWidget.setItem(nRow, 2, QTableWidgetItem(str(imageCount)))


# ======================================================================================================================
#
# ======================================================================================================================
def DP1_20002_fetchImageList(self, nProductID, nRow, start_date, end_date, start_time, end_time, downloadsFilePath):

    imageList = DP1_20002_buildImageList(self, nProductID, nRow, start_date, end_date, start_time, end_time)

    DP1_20002_downloadImages(self, imageList, downloadsFilePath)

# ======================================================================================================================
#
# ======================================================================================================================
def DP1_20002_buildImageList(self, nProductID, nRow, start_date, end_date, start_time, end_time):
    global SITECODE, DOMAINCODE, dailyImagesList, gWebImageCount

    if nRow <= -1:
        return []

    delta = end_date - start_date

    # CREATE PROGRESS BAR
    progressBarDates = QProgressWheel()
    progressBarDates.setRange(0, delta.days + 1)
    progressBarDates.setWindowTitle('Build image list...')
    progressBarDates.show()

    # CLEAR PREVIOUS LIST
    dailyImagesList.clear()

    i = 1
    while start_date <= end_date:
        if not progressBarDates or not progressBarDates.isVisible():
            progressBarDates = None
            break
        ymd = start_date.strftime("%Y-%d-%b")
        progressBarDates.setWindowTitle(ymd)
        progressBarDates.setValue(float(i) / float(delta.days + 1) * delta.days)
        progressBarDates.repaint()
        i += 1

        QCoreApplication.processEvents()

        product_str = str(nProductID).zfill(5)

        # Build URL
        dailyURLvisible = (
            f"https://phenocam.nau.edu/webcam/browse/NEON.{DOMAINCODE}.{SITECODE}.DP1.{product_str}/"
            f"{start_date.year}/{str(start_date.month).zfill(2)}/{str(start_date.day).zfill(2)}"
        )

        phenoCam = GRIME_AI_PhenoCam()
        tmpList = phenoCam.getVisibleImages(dailyURLvisible, start_time, end_time)

        dailyImagesList.setVisibleList(tmpList.getVisibleList())

        start_date += datetime.timedelta(days=1)

    if progressBarDates and progressBarDates.isVisible():
        progressBarDates.close()

    gWebImageCount = len(dailyImagesList.getVisibleList())
    return dailyImagesList.getVisibleList()

def DP1_20002_downloadImages(self, imageList, downloadsFilePath):
    if not imageList:
        return

    cancel_requested = {"value": False}

    def request_cancel():
        cancel_requested["value"] = True

    progressBarDownloads = QProgressWheel(on_close=request_cancel)
    progressBarDownloads.setRange(0, len(imageList) + 1)
    progressBarDownloads.setWindowTitle('Download & Save Images...')
    progressBarDownloads.show()

    for i, image in enumerate(imageList):
        if cancel_requested["value"]:
            print("NEON image download cancelled by user.")
            break

        progressBarDownloads.setValue(float(i) / float(len(imageList) + 1) * len(imageList))

        filename = os.path.basename(image.fullPathAndFilename)
        if not os.path.exists(downloadsFilePath):
            os.makedirs(downloadsFilePath)

        completeFilename = os.path.join(downloadsFilePath, filename)

        if not os.path.isfile(completeFilename):
            urllib.request.urlretrieve(image.fullPathAndFilename, completeFilename)

        QCoreApplication.processEvents()

    if progressBarDownloads and progressBarDownloads.isVisible():
        progressBarDownloads.close()

#jes LET THE CALLING FUNCTION BE RESPONSIBLE FOR REPORTING DOWNLOAD COMPLETION.
#jes MODIFY THIS IN A FUTURE RELEASE TO RETURN A PASS/FAIL MESSAGE TO THE FUNCTION THAT INVOKED THIS FUNCTION.
#jes strMessage = 'Data download is complete!'
#jes msgBox = GRIME_AI_QMessageBox('Data Download', strMessage)
#jes response = msgBox.displayMsgBox()


# ======================================================================================================================
# DOWNLOAD THE PRODUCT FILES SELECTED IN THE GUI BY THE END-USER.
# ======================================================================================================================
def downloadProductDataFiles(self, item):
    global dailyImagesList
    global currentImageIndex

    missing_data_message = ""
    nitrateList = []
    nError = 0;

    myNEON_API = NEON_API()

    # ----------------------------------------------------------------------------------------------------
    # SAVE DOWNLOADED DATA TO THE USER GRIME-AI FOLDER THAT IS AUTOMATICALLY CREATED, IF IT DOES NOT EXIST,
    # CREATE IT IN THE USER'S DOCUMENT FOLDER
    # ----------------------------------------------------------------------------------------------------
    NEON_download_file_path = self.edit_NEON_TableInput.text().strip() or self.edit_NEONSaveFilePath.text().strip()
    self.edit_NEONSaveFilePath.setText(NEON_download_file_path)
    self.edit_NEON_TableInput.setText(NEON_download_file_path)
    JsonEditor().update_json_entry("NEON_Root_Folder", NEON_download_file_path)

    if len(NEON_download_file_path) == 0:
        strMessage = 'A download folder has not been specified. Would you like to use the last GRIME-AI NEON download folder?'
        msgBox = GRIME_AI_QMessageBox('NEON Root Download Folder', strMessage, QMessageBox.Yes | QMessageBox.No)
        response = msgBox.displayMsgBox()

        if response == QMessageBox.Yes:
            #NEON_download_file_path = os.path.expanduser('~')
            #NEON_download_file_path = os.path.join(NEON_download_file_path, 'Documents')
            #NEON_download_file_path = os.path.join(NEON_download_file_path, 'GRIME-AI')

            NEON_download_file_path = JsonEditor().getValue("NEON_Root_Folder")

            if not os.path.exists(NEON_download_file_path):
                os.makedirs(NEON_download_file_path)
            self.edit_NEONSaveFilePath.setText(NEON_download_file_path)
            JsonEditor().update_json_entry("NEON_Root_Folder", NEON_download_file_path)
    else:
        # MAKE SURE THE PATH EXISTS. IF IT DOES NOT, THEN CREATE IT.
        if not os.path.exists(NEON_download_file_path):
            os.makedirs(NEON_download_file_path)


    # --------------------------------------------------------------------------------
    # FIND IMAGE PRODUCT (20002) ROW TO GET DATE RANGE
    # --------------------------------------------------------------------------------
    rowRange = range(self.NEON_selected_products.rowCount())

    for nRow in rowRange:
        GRIME_AI_ProductTableObj = GRIME_AI_ProductTable()
        start_date, start_time, end_date, end_time = GRIME_AI_ProductTableObj.fetchTableDates(self.NEON_selected_products, nRow)

        # EXTRACT THE PRODUCT ID
        prodIDCol = 0
        strProductIDCell = self.NEON_selected_products.item(nRow, prodIDCol).text()
        nProductID = int(strProductIDCell.split('.')[1])
        #else:
        #    nProductID = -999

        if nProductID > 0:
            PRODUCTCODE = strProductIDCell.split(':')[0]

            # PHENOCAM IMAGES
            # ----------------------------------------------------------------------------------------------------------
            if nProductID == 20002 or nProductID == 42 or nProductID == 33:
                downloadsFilePath = os.path.join(self.edit_NEONSaveFilePath.text(), 'Images')
                if not os.path.exists(downloadsFilePath):
                    os.makedirs(downloadsFilePath)

                DP1_20002_fetchImageList(self, nProductID, nRow, start_date, end_date, start_time, end_time, downloadsFilePath)

                processLocalImage(self, imageFileFolder=downloadsFilePath)

            # ALL OTHER NEON DATA
            # ----------------------------------------------------------------------------------------------------------
            if nProductID != 20002:
                strStartYearMonth = str(start_date.year) + '-' + str(start_date.month).zfill(2)
                strEndYearMonth = str(end_date.year) + '-' + str(end_date.month).zfill(2)

                PRODUCTCODE = strProductIDCell.split(':')[0]

                # GET THE RANGE OF MONTHS FROM THE START DATE TO THE END DATE
                dateRange = GRIME_AI_Utils().getRangeOfDates(strStartYearMonth, strEndYearMonth)

                # GET THE AVAILABLE MONTHS FOR THE SELECTED DATA SET
                availableMonths = NEON_API().getAvailableMonths(SITECODE, PRODUCTCODE)

                monthCount = 0
                missingMonths = []
                for month in dateRange:
                    if month in availableMonths:
                        monthCount += 1
                    else:
                        missingMonths.append(month)

                if monthCount == 0:
                    missing_data_message = missing_data_message + 'NEON Error!  ' + strProductIDCell + 'Data is not available for some or all of the dates selected!\n'
                elif (monthCount < len(dateRange)):
                    strMsg = '%d of %d months unavailable: %s' % (len(missingMonths), len(dateRange), missingMonths)
                    missing_data_message = missing_data_message + 'Partial Download!\n   ' + strProductIDCell + strMsg + '\n'

                if monthCount > 0:
                    downloadsFilePath = os.path.normpath(os.path.join(NEON_download_file_path, 'data'))
                    if not os.path.exists(downloadsFilePath):
                        os.makedirs(downloadsFilePath)

                    nError = myNEON_API.FetchData(SITECODE, strProductIDCell, strStartYearMonth, strEndYearMonth, downloadsFilePath)
        else:
            missing_data_message = missing_data_message + 'NEON Error!\n  ' + strProductIDCell + 'Product not available!' + '\n'

    if missing_data_message != "":
        msgBox = GRIME_AI_QMessageBox('Download Error!', missing_data_message, buttons=QMessageBox.Close)
    else:
        msgBox = GRIME_AI_QMessageBox('Download Complete!', 'Download Complete!', buttons=QMessageBox.Close)
    response = msgBox.displayMsgBox()

        # ----------------------------------------------------------------------------------------------------------
        # NITRATE DATA
        # ----------------------------------------------------------------------------------------------------------
        # if nProductID == 20033:
        #     nitrateList = myNEON_API.parseNitrateCSV()
        #
        #     if len(nitrateList) > 0:
        #         #JES - USE NITRATE DATA FOR DEVELOPING GENERIC CSV READING AND DATA GRAPHING CAPABILITIES
        #         scene = QGraphicsScene()
        #         self.scene = scene
        #         nWidth = self.graphicsView.width()
        #         nHeight = self.graphicsView.height()
        #         nX = self.graphicsView.x()
        #         nY = self.graphicsView.y()
        #         self.scene.setSceneRect(0, 0, nWidth, nHeight)
        #         # self.graphicsView.setWindowTitle('Nitrate Data')
        #         self.graphicsView.setScene(self.scene)
        #         figure = Figure()
        #         axes = figure.gca()
        #         axes.set_title("Nitrate Data")
        #
        #         i = 0
        #         for i, nitrateData in enumerate(nitrateList):
        #             y = float(nitrateData.getNitrateMean())
        #             axes.plot(i, y, '.', markersize=2)
        #
        #         canvas = FigureCanvas(figure)
        #         canvas.resize(nWidth, nHeight)
        #         self.scene.addWidget(canvas)
        #         self.graphicsView.show()


# ======================================================================================================================
#
# ======================================================================================================================
def NEON_labelOriginalImageDoubleClickEvent(self):
    global currentImage

    if currentImage != []:
        img = GRIME_AI_Utils().convertQImageToMat(currentImage.toImage())

        self.setMouseTracking(False)

        cv2.imshow('Original', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.setMouseTracking(True)


# ======================================================================================================================
#
# ======================================================================================================================
def NEON_labelMouseDoubleClickEvent(self, event):
    img = GRIME_AI_Utils().convertQImageToMat(self.NEON_labelLatestImage.toImage())
    cv2.imshow('Original', img)

    # ----------
    if 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        cv2.imshow('Mask', mask)

        res = cv2.bitwise_and(img, img, mask=mask)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        cv2.imshow('laplacian', laplacian)

        mySobel = sobelData()
        mySobel.setSobelX(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5))
        mySobel.setSobelY(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5))

        cv2.imshow('Sobel: x-axis', mySobel.sobelX)
        cv2.imshow('Sobel: y-axis', mySobel.sobelY)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ----------
    if 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow('Original', img)
        edges = cv2.Canny(img, 100, 200)
        cv2.imshow('Canny Edges', edges)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ----------
    if 0:
        cv2.imshow("Latest Image (Color)", img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Latest Image (Gray)", gray)

        cv2.waitKey(0)

        cv2.destroyAllWindows()


# ======================================================================================================================
#
# ======================================================================================================================
def retranslateUi(self, MainWindow):

    szWindowsTitle = "GRIME AI" + " " + SW_VERSION + " - John E. Stranzl Jr."

    _translate = QtCore.QCoreApplication.translate
    MainWindow.setWindowTitle(_translate(szWindowsTitle, szWindowsTitle))

# ======================================================================================================================
# FIND THE MONTHS THAT DATA IS AVAILABLE FOR A PARTICULAR PRODUCT FOR A PARTICULAR SITE
# ======================================================================================================================
def findAvailableMonths(item):
    global SERVER
    global SITECODE

    PRODUCTCODE = item
    monthList = {}

    # RETRIEVE INFORMATION FROM THE NEON WEBSITE FOR THE PARTICULAR SITE
    site_json = NEON_API().FetchSiteInfoFromNEON(SERVER, SITECODE)

    if site_json:
        # EXTRACT THE AVAILABLE MONTH AND THE URL FOR THE DATA FOR EACH AVAILABLE MONTH
        data_products = site_json.get('data', {}).get('dataProducts', [])
        for product in data_products:
            if (product['dataProductCode'] == PRODUCTCODE):
                monthList['availableMonths'] = product['availableMonths']
                monthList['availableDataUrls'] = product['availableDataUrls']
                break

    return (monthList)


# ======================================================================================================================
#
# ======================================================================================================================
def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape

    print("The height and width of the image are: height,width", height, width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
                print("max_gray_level:", max_gray_level)

    return max_gray_level + 1


def run_gui():
    global frame

    # If Hydra is already initialized, clear it
    from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils

    settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
    print(settings_folder)
    hydra_working_folder = os.path.normpath(os.path.join(settings_folder, "MyHydraOutputs"))
    sys.argv.append(f"hydra.run.dir={hydra_working_folder}")

    if 0:
        if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.core.global_hydra.GlobalHydra.instance().clear()
        else:
            hydra.initialize(config_path=None)

    # CREATE MAIN APP WINDOW
    app = QApplication(sys.argv)

    # Light mode is the default — dark mode toggled via View menu
    app.setStyleSheet("")

    # ------------------------------------------------------------------------------------------------------------------
    # SHOW SPLASH IN SEPARATE PROCESS — stays visible throughout all of MainWindow init
    # ------------------------------------------------------------------------------------------------------------------
    _splash_dir = Path(__file__).resolve().parent / "resources" / "splash_screens"
    _splash = GRIME_AI_SplashScreen(
        image_path=os.path.join(_splash_dir, "GRIME-AI Logo with Tagline.png")
    )
    _splash.show()   # blocks briefly until splash is painted, then returns

    frame = MainWindow(splash=_splash)

    frame.move(app.desktop().screen().rect().center() - frame.rect().center())

    # ------------------------------------------------------------------------------------------------------------------
    # PROCESS ANY EVENTS THAT WERE DELAYED BECAUSE OF THE SPLASH SCREEN
    # ------------------------------------------------------------------------------------------------------------------
    app.processEvents()

    frame.graphicsView.setVisible(True)

    # ------------------------------------------------------------------------------------------------------------------
    # http://localhost:8888/notebooks/intro-seg.ipynb
    # ------------------------------------------------------------------------------------------------------------------
    bStartupComplete = True

    # Run the program
    sys.exit(app.exec())


def my_main():
    global bShow_GUI

    # Main parser
    parser = argparse.ArgumentParser(description='CLI for GRIME AI')
    parser.add_argument("-v", "--version", action=version_action(print_version),
                        nargs=0, help="Show version info (-v for short, --version for full details)")

    # Subparsers
    subparsers = parser.add_subparsers(dest='command')

    # Triage parser
    triage_parser = subparsers.add_parser('triage', help='Perform Image Triage')
    triage_parser.add_argument("-m", "--min", type=float, required=False, default=65.0,
                               help="Minimum brightness (default: 60.0).")
    triage_parser.add_argument("-x", "--max", type=float, required=False, default=180.0,
                               help="Maximum brightness (default: 180.0).")
    triage_parser.add_argument("-r", "--report", type=bool, required=False, default=True, help="Generate a report.")
    triage_parser.add_argument("-v", "--move", type=bool, required=False, default=True,
                               help="Move images to subfolder.")
    triage_parser.add_argument("-p", "--poly", type=bool, required=False, default=False, help="Save polyline images.")
    triage_parser.add_argument("-a", "--alignment", type=bool, required=False, default=False,
                               help="Correct rotated images.")
    triage_parser.add_argument("-d", "--delta", type=float, required=False, default=0.25,
                               help="Rotation tolerance for considering an image over- or under-rotated.")
    triage_parser.add_argument("-i", "--image", type=str, required=False, default=None,
                               help="Image to use as ground truth for rotation angle.")
    triage_parser.add_argument("-b", "--blurthreshold", type=float, required=False, default=17.50,
                               help="Blur threshold for FFT.")
    triage_parser.add_argument("-s", "--shift", type=int, required=False, default=60,
                               help="FFT Shift size for blur estimation.")
    triage_parser.add_argument("-f", "--folder", type=str, required=True, help="A folder must be specified.")

    # Slice parser
    slice_parser = subparsers.add_parser('slice', help='Perform Image Slicing')
    slice_parser.add_argument("-c", "--center", type=int, required=True, default=0, help="Center of slice (in pixels).")
    slice_parser.add_argument("-w", "--width", type=int, required=True, default=20,
                              help="Width of the slice (in pixels).")
    slice_parser.add_argument("-f", "--folder", type=str, required=True, help="A folder must be specified.")

    # COCO parser
    coco_parser = subparsers.add_parser('coco', help='Generate COCO annotation file from images and masks')
    coco_parser.add_argument("--folder", required=True,
                             help="Folder containing image files (and optional masks).")
    coco_parser.add_argument("--shared-mask", required=False,
                             help="If provided, uses this single mask file for all images in the folder.")
    coco_parser.add_argument("--output", required=False,
                             help="(Optional) Override output JSON file path.")

    # Segment parser
    segment_parser = subparsers.add_parser('segment', help='Segment a single image using a trained GRIME AI model')
    segment_parser.add_argument('--model',          required=True,  help='Path to trained model checkpoint (.torch)')
    segment_parser.add_argument('--image',          required=False, default=None,
                                help='Path to a single input image')
    segment_parser.add_argument('--folder',         required=False, default=None,
                                help='Path to a folder of images')
    segment_parser.add_argument('--output',         required=True,  help='Output folder for results')
    segment_parser.add_argument('--mode',           required=True,  choices=['sam2', 'segformer'],
                                help='Segmentation model type: sam2 or segformer')
    segment_parser.add_argument('--category-id',   type=int, default=1,
                                help='Label category ID to segment (default: 1)')
    segment_parser.add_argument('--category-name', default='Vegetation',
                                help='Label category name (default: Vegetation)')
    segment_parser.add_argument('--model-cfg',     default='sam2.1_hiera_l.yaml',
                                help='SAM2 model config YAML (default: sam2.1_hiera_l.yaml)')
    segment_parser.add_argument('--threshold',     type=float, default=0.2,
                                help='Probability threshold for SegFormer mask (default: 0.2)')
    segment_parser.add_argument('--no-mask',       action='store_true',
                                help='Skip saving binary mask PNG')
    segment_parser.add_argument('--no-copy',       action='store_true',
                                help='Skip copying original image to output folder')

    # ROI Analyzer parser
    roi_parser = subparsers.add_parser(
        'roi',
        help='Run ROI feature extraction on image/mask pairs'
    )
    roi_parser.add_argument(
        '--folder', required=True,
        help='Folder containing image/mask pairs.'
    )
    roi_parser.add_argument(
        '--mode', required=True, choices=['analyze', 'extract'],
        help=(
            'analyze: run analysis on the first pair (or the pair matching --image) '
            'and print metrics to stdout. '
            'extract: batch all pairs and write CSV, XLSX, and optional aligned XLSX.'
        )
    )
    roi_parser.add_argument(
        '--image', required=False, default=None,
        help='(analyze mode only) Filename of a specific image in --folder to analyze. '
             'If omitted, the first pair is used.'
    )
    roi_parser.add_argument(
        '--output', required=False, default=None,
        help='Output folder for CSV/XLSX results (extract mode). '
             'Defaults to --folder if not specified.'
    )
    roi_parser.add_argument(
        '--clustering', required=False, default='kmeans',
        choices=['kmeans', 'gmm', 'meanshift'],
        help='Color clustering method (default: kmeans).'
    )
    roi_parser.add_argument(
        '--clusters', required=False, type=int, default=3,
        help='Number of clusters for kmeans or gmm (default: 3).'
    )
    roi_parser.add_argument(
        '--auto-estimate', required=False, action='store_true',
        help='(meanshift only) Auto-estimate bandwidth from data using --quantile.'
    )
    roi_parser.add_argument(
        '--quantile', required=False, type=float, default=0.3,
        help='(meanshift + --auto-estimate) Quantile for bandwidth estimation (default: 0.3).'
    )
    roi_parser.add_argument(
        '--bandwidth', required=False, type=float, default=1.0,
        help='(meanshift, manual) Fixed bandwidth value (default: 1.0).'
    )

    # Custom help handling
    if '-h' in sys.argv or '--help' in sys.argv:
        print("Global Help: CLI for GRIME AI")
        parser.print_help()  # General help for the main parser
        print("\nHelp for 'triage' command:")
        triage_parser.print_help()  # Help for triage subparser
        print("\nHelp for 'slice' command:")
        slice_parser.print_help()  # Help for slice subparser
        print("\nHelp for 'segment' command:")
        segment_parser.print_help()  # Help for segment subparser
        print("\nHelp for 'roi' command:")
        roi_parser.print_help()  # Help for roi subparser
        sys.exit(0)  # Exit after displaying help

    # API key arguments (session-only; not written to api_keys.ini)
    neon_grp = parser.add_mutually_exclusive_group()
    neon_grp.add_argument(
        "--neon-token", metavar="TOKEN",
        help="NEON API token (session-only, not saved to disk)"
    )
    neon_grp.add_argument(
        "--neon-token-file", metavar="PATH",
        help="Path to a .txt file containing your NEON API token"
    )
    usgs_grp = parser.add_mutually_exclusive_group()
    usgs_grp.add_argument(
        "--usgs-api-key", metavar="KEY",
        help="USGS Water Data API key (session-only, not saved to disk)"
    )
    usgs_grp.add_argument(
        "--usgs-api-key-file", metavar="PATH",
        help="Path to a .txt file containing your USGS Water Data API key"
    )

    args = parser.parse_args()

    # Apply any CLI-supplied API key overrides (session-only)
    try:
        APIKeyManager().apply_cli_overrides(args)
    except Exception as _e:
        print(f"[WARN] Could not apply CLI API key overrides: {_e}")

    # Check if no arguments were provided
    if len(sys.argv) == 1:
        bShow_GUI = True
        run_gui()
    else:
        bShow_GUI = False
        run_cli(args)

    # SHOW MAIN WINDOW
    #frame.show()

def version_action(fn):
    class _Action(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            fn(long=option_string == "--version")
            parser.exit()
    return _Action

def print_version(long: bool):
    try:
        from GRIME_AI.version import SW_VERSION, RELEASE, BUILD_DATE, SHA
    except ImportError:
        SW_VERSION = globals().get('SW_VERSION', '0.0.0.0')
        RELEASE    = 'N/A'
        BUILD_DATE = 'N/A'
        SHA        = 'N/A'

    if not long:
        print(SW_VERSION)
    else:
        print(
            f"Version: v{SW_VERSION}\n"
            f"Release: {RELEASE}\n"
            f"Build date: {BUILD_DATE}\n"
            f"Commit: {SHA}\n"
            f"Python: {sys.version.split()[0]}"
        )

def run_cli(args):
    import re
    import os
    import sys
    from pathlib import Path

    if args.command == 'triage':
        create_report = True
        move_images = True
        fetch_recursive = False
        blur_threshold = 17.50
        shift_size = 60
        #min_val = 65.0
        #max_val = 180.0
        correct_alignment = False
        save_poly_lines = False
        reference_image_filename = ""
        rotation_threshold = 0.15

        print(args.command)

        print("These are the Triage parameters:", args.folder, args.min, args.max)
        myTriage = GRIME_AI_ImageTriage(False)
        myTriage.cleanImages(args.folder, \
                             fetch_recursive, \
                             blur_threshold, shift_size, \
                             args.min, args.max, \
                             create_report, move_images, \
                             correct_alignment, save_poly_lines,
                             reference_image_filename, rotation_threshold)

        print('Image triage is complete!')

    elif args.command == 'slice':
        filenames = cli_fetchLocalImageList(args.folder)

        from PyQt5.QtCore import QRect
        from PIL import Image
        from GRIME_AI.GRIME_AI_CompositeSlices import GRIME_AI_CompositeSlices
        first_image = Image.open(filenames[0].fullPathAndFilename)
        slice_rect = QRect(
            int(args.center - args.width / 2),
            0,
            int(args.width),
            first_image.height
        )

        compositeSlices = GRIME_AI_CompositeSlices(slice_rect, False)
        compositeSlices.create_composite_image(filenames, args.folder+'\compositeSlices')

        print("Composite slice complete!")

    elif args.command == "coco":
        # Import your CocoGenerator class from coco_generator.py
        try:
            from .coco_generator import CocoGenerator  # package mode
        except ImportError:
            from coco_generator import CocoGenerator  # direct script mode
        print("[INFO] Running COCO generation command...")
        folder = Path(args.folder)
        output_path = Path(args.output) if args.output else folder / "instances_default.json"
        if args.shared_mask:
            shared_mask = Path(args.shared_mask)
            if not shared_mask.exists():
                print(f"[ERROR] Shared mask file not found: {shared_mask}")
                sys.exit(1)
            # Instantiate in shared mask mode
            generator = CocoGenerator(folder=folder, shared_mask=shared_mask, output_path=output_path)
        else:
            # Instantiate for one-to-one mode
            generator = CocoGenerator(folder=folder, output_path=output_path)
        generator.generate_annotations()

    elif args.command == 'segment':
        from GRIME_AI.GRIME_AI_segment import run_sam2, run_segformer

        # Validate inputs
        if not os.path.isfile(args.model):
            print(f"[ERROR] Model file not found: {args.model}", file=sys.stderr)
            sys.exit(1)
        os.makedirs(args.output, exist_ok=True)

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        category = {"id": args.category_id, "name": args.category_name}

        print(f"[GRIME AI] Mode:      {args.mode.upper()}")
        print(f"[GRIME AI] Model:     {args.model}")
        print(f"[GRIME AI] Image:     {args.image}")
        print(f"[GRIME AI] Output:    {args.output}")
        print(f"[GRIME AI] Category:  {args.category_name} (ID: {args.category_id})")

        if args.mode == 'sam2':
            run_sam2(args, device, category, progressBar=None)
        elif args.mode == 'segformer':
            run_segformer(args, device, category, progressBar=None)

        print("[GRIME AI] Segmentation complete.")

    elif args.command == 'roi':

        # ------------------------------------------------------------------
        # Force matplotlib to use a non-interactive backend so plt calls
        # work headlessly on servers and HPC clusters with no display.
        # This must happen before any matplotlib.pyplot import.
        # ------------------------------------------------------------------
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        import pandas as pd
        import seaborn as sns
        import statsmodels.api as sm
        from pathlib import Path
        from openpyxl import Workbook
        from openpyxl.styles import Font

        from GRIME_AI.GRIME_AI_ROI_Analyzer import GRIME_AI_ROI_Analyzer

        # ------------------------------------------------------------------
        # Validate folder
        # ------------------------------------------------------------------
        folder = args.folder
        if not os.path.isdir(folder):
            print(f"[ERROR] Folder not found: {folder}", file=sys.stderr)
            sys.exit(1)

        output_folder = args.output if args.output else folder
        os.makedirs(output_folder, exist_ok=True)

        # ------------------------------------------------------------------
        # Build meanshift kwargs from CLI args, mirroring _get_meanshift_kwargs()
        # ------------------------------------------------------------------
        if args.clustering == 'meanshift':
            if args.auto_estimate:
                meanshift_kwargs = {'bandwidth': None, 'quantile': args.quantile}
            else:
                meanshift_kwargs = {'bandwidth': args.bandwidth, 'quantile': None}
        else:
            meanshift_kwargs = {}

        n_clusters = args.clusters
        clustering = args.clustering

        # ------------------------------------------------------------------
        # Datetime extraction helper — mirrors _extract_datetime_from_path()
        # ------------------------------------------------------------------
        def _extract_datetime_from_path(filepath):
            stem = os.path.splitext(os.path.basename(filepath))[0]
            patterns = [
                (r'(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})Z',
                 lambda m: (f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
                            f"{m.group(4)}:{m.group(5)}:{m.group(6)}")),
                (r'(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})(\d{2})',
                 lambda m: (f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
                            f"{m.group(4)}:{m.group(5)}:{m.group(6)}")),
                (r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',
                 lambda m: (f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
                            f"{m.group(4)}:{m.group(5)}:{m.group(6)}")),
                (r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})',
                 lambda m: (f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
                            f"{m.group(4)}:{m.group(5)}:{m.group(6)}")),
                (r'(\d{4})_(\d{2})_(\d{2})',
                 lambda m: (f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "n/a")),
                (r'(\d{4})(\d{2})(\d{2})',
                 lambda m: (f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "n/a")),
            ]
            for pattern, extractor in patterns:
                m = re.search(pattern, stem)
                if m:
                    try:
                        return extractor(m)
                    except Exception:
                        continue
            return "n/a", "n/a"

        # ------------------------------------------------------------------
        # analyze_responses() — mirrors tab method; uses Agg matplotlib
        # ------------------------------------------------------------------
        def _analyze_responses(df, predictors, responses, out_dir):
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            df1 = df.copy()
            df1[predictors + responses] = df1[predictors + responses].apply(
                pd.to_numeric, errors='coerce'
            )

            # Distribution plots
            for col in predictors + responses:
                plt.figure()
                sns.histplot(df1[col].dropna(), kde=True)
                plt.title(f"Distribution of {col}")
                plt.tight_layout()
                plt.savefig(out_path / f"hist_{col}.png")
                plt.close()

            # Correlations
            pearson_results = {}
            spearman_results = {}
            for resp in responses:
                pearson_results[resp] = df1[predictors + [resp]].corr(method='pearson')[resp].drop(resp)
                spearman_results[resp] = df1[predictors + [resp]].corr(method='spearman')[resp].drop(resp)

            pearson_df = pd.DataFrame(pearson_results)
            spearman_df = pd.DataFrame(spearman_results)

            # Scatterplots with regression lines
            for resp in responses:
                for pred in predictors:
                    x = df1[pred]
                    y = df1[resp]
                    valid = x.notna() & y.notna()
                    if valid.sum() < 2:
                        print(f"[ROI] Skipping scatter {pred} vs {resp}: insufficient data")
                        continue
                    plt.figure()
                    sns.regplot(x=x[valid], y=y[valid], scatter_kws={'alpha': 0.5})
                    plt.xlabel(pred)
                    plt.ylabel(resp)
                    plt.title(f"{pred} vs {resp}")
                    plt.tight_layout()
                    plt.savefig(out_path / f"scatter_{pred}_vs_{resp}.png")
                    plt.close()

            # OLS regressions
            model_summaries = {}
            for resp in responses:
                X = df1[predictors]
                y = df1[resp]
                valid = X.notna().all(axis=1) & y.notna()
                if valid.sum() < 2:
                    print(f"[ROI] Skipping regression for {resp}: insufficient data")
                    continue
                X_valid = sm.add_constant(X[valid])
                model = sm.OLS(y[valid], X_valid).fit()
                model_summaries[resp] = model.summary().as_text()
                print(f"\n[ROI] Regression for {resp}:")
                print(model.summary())

            # Correlation heatmap
            plt.figure(figsize=(10, 8))
            corr_matrix = df1[predictors + responses].corr(method='pearson')
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Heatmap (Pearson)")
            plt.tight_layout()
            plt.savefig(out_path / "correlation_heatmap.png")
            plt.close()

            # Excel export
            excel_path = out_path / "analysis_results.xlsx"
            with pd.ExcelWriter(excel_path) as writer:
                pearson_df.to_excel(writer, sheet_name="Pearson_Corr")
                spearman_df.to_excel(writer, sheet_name="Spearman_Corr")
                summary_df = pd.DataFrame.from_dict(
                    model_summaries, orient='index', columns=['Summary']
                )
                summary_df.to_excel(writer, sheet_name="Model_Summaries")

            print(f"[ROI] Response analysis saved to {out_path}")

        # ------------------------------------------------------------------
        # Generate image/mask pairs
        # ------------------------------------------------------------------
        temp = GRIME_AI_ROI_Analyzer("", "")
        pairs = temp.generate_file_pairs(folder)

        if not pairs:
            print(f"[ERROR] No image/mask pairs found in: {folder}", file=sys.stderr)
            sys.exit(1)

        print(f"[ROI] Found {len(pairs)} image/mask pair(s).")

        # ==================================================================
        # MODE: analyze
        # ==================================================================
        if args.mode == 'analyze':

            # Select the target pair
            target_pair = None
            if args.image:
                target_name = os.path.basename(args.image)
                for orig, mask in pairs:
                    if os.path.basename(orig) == target_name:
                        target_pair = (orig, mask)
                        break
                if target_pair is None:
                    print(f"[ERROR] Image not found in pairs: {args.image}", file=sys.stderr)
                    sys.exit(1)
            else:
                target_pair = pairs[0]

            orig_path, mask_path = target_pair
            print(f"[ROI] Analyzing: {orig_path}")
            print(f"[ROI] Mask:      {mask_path}")

            analyzer = GRIME_AI_ROI_Analyzer(
                orig_path, mask_path,
                clusters=n_clusters,
                clustering_method=clustering,
                **meanshift_kwargs
            )
            analyzer.run_analysis()

            capture_date, capture_time = _extract_datetime_from_path(orig_path)

            print(f"\n[ROI] Results for: {os.path.basename(orig_path)}")
            print(f"  Capture Date     : {capture_date}")
            print(f"  Capture Time     : {capture_time}")
            print(f"  ROI Intensity    : {analyzer.roi_intensity:.2f}")
            print(f"  ROI Entropy      : {analyzer.roi_entropy:.4f}")
            print(f"  ROI Texture      : {analyzer.roi_texture:.4f}")
            print(f"  Mean GLI         : {analyzer.mean_gli:.6f}")
            print(f"  Mean GCC         : {analyzer.mean_gcc:.6f}")
            print(f"  ROI Pixel Count  : {analyzer.ROI_total_pixels:.0f}")
            print(f"  ROI Area         : {analyzer.ROI_total_area:.2f}")
            print(f"  ROI Area %       : {analyzer.ROI_percentage:.2f}%")
            print(f"  Image Dimensions : {analyzer.image_width:.0f} x {analyzer.image_height:.0f}")
            print(f"  Clustering       : {clustering}, clusters={n_clusters}")
            for i, (rgb, hsv, pct) in enumerate(zip(
                    analyzer.dominant_rgb_list,
                    analyzer.dominant_hsv_list,
                    analyzer.percentages_list), start=1):
                print(f"  Cluster {i}        : RGB={rgb}  HSV={hsv}  Coverage={pct:.1f}%")

            print("\n[ROI] Analysis complete.")

        # ==================================================================
        # MODE: extract
        # ==================================================================
        elif args.mode == 'extract':

            # Build output filename prefix from first image datetime
            dt_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})T(\d{2}-\d{2}-\d{2})Z')
            file_dt_prefix = "no_datetime"
            for img_path, _ in pairs:
                m = dt_pattern.search(os.path.basename(img_path))
                if m:
                    file_dt_prefix = f"{m.group(1)}_{m.group(2).replace('-', '')}"
                    break

            csv_path = os.path.join(output_folder, f"{file_dt_prefix}_roi_metrics.csv")
            xlsx_path = os.path.join(output_folder, f"{file_dt_prefix}_roi_metrics.xlsx")
            aligned_xlsx_path = os.path.join(output_folder, f"{file_dt_prefix}_aligned_results.xlsx")

            # Build clustering params string for the output
            if clustering in ('kmeans', 'gmm'):
                clustering_params = f"clusters={n_clusters}"
            else:
                if meanshift_kwargs.get('bandwidth') is None:
                    clustering_params = f"auto=True, quantile={meanshift_kwargs['quantile']}"
                else:
                    clustering_params = f"auto=False, bandwidth={meanshift_kwargs['bandwidth']}"

            # Build CSV header
            header = [
                "Image Path", "Mask Path",
                "Capture Date", "Capture Time",
                "Clustering Method", "Clustering Params",
                "ROI Intensity", "ROI Entropy", "ROI Texture",
                "Mean GLI", "Mean GCC",
                "ROI Pixel Count", "ROI Area",
                "Image Height", "Image Width", "Image Total Pixels",
                "ROI Area Percentage",
            ]
            for i in range(1, n_clusters + 1):
                header.extend([f"Cluster {i} H", f"Cluster {i} S", f"Cluster {i} V"])
            for i in range(1, n_clusters + 1):
                header.extend([f"Cluster {i} R", f"Cluster {i} G", f"Cluster {i} B"])

            rows = []

            # ----------------------------------------------------------
            # Look for a related sensor CSV in the same folder as the
            # images, matching the naming convention: "<prefix> - *.csv"
            # ----------------------------------------------------------
            first_img_path = pairs[0][0]
            image_folder = os.path.dirname(first_img_path)
            first_stem = os.path.splitext(os.path.basename(first_img_path))[0]
            match_dt = dt_pattern.search(first_stem)
            common_prefix = first_stem[:match_dt.start()].rstrip("_ ") if match_dt else first_stem
            related_csv_df = None
            matching_csv_path = None

            try:
                for fname in os.listdir(image_folder):
                    if (fname.lower().endswith(".csv") and
                            fname.lower().startswith((common_prefix + " - ").lower())):
                        matching_csv_path = os.path.join(image_folder, fname)
                        break
            except Exception as e:
                print(f"[ROI] Warning: could not scan image folder for related CSV: {e}")

            if matching_csv_path:
                print(f"[ROI] Found related sensor CSV: {matching_csv_path}")
                try:
                    related_csv_df = pd.read_csv(matching_csv_path)
                    if 'datetime' in related_csv_df.columns:
                        related_csv_df['datetime'] = pd.to_datetime(
                            related_csv_df['datetime'],
                            format="%m/%d/%Y %H:%M",
                            errors="coerce"
                        )
                        related_csv_df['Date'] = related_csv_df['datetime'].dt.strftime("%Y-%m-%d")
                        related_csv_df['Time'] = related_csv_df['datetime'].dt.strftime("%H:%M:%S")
                except Exception as e:
                    print(f"[ROI] Warning: could not read related CSV: {e}")
                    related_csv_df = None
            else:
                print("[ROI] No related sensor CSV found. Alignment step will be skipped.")

            # ----------------------------------------------------------
            # Iterate all pairs
            # ----------------------------------------------------------
            total = len(pairs)
            for i, (orig_path, mask_path) in enumerate(pairs):
                print(f"[ROI] Processing {i + 1}/{total}: {os.path.basename(orig_path)}")

                capture_date, capture_time = _extract_datetime_from_path(orig_path)

                analyzer = GRIME_AI_ROI_Analyzer(
                    orig_path, mask_path,
                    clusters=n_clusters,
                    clustering_method=clustering,
                    **meanshift_kwargs
                )
                try:
                    analyzer.run_analysis()
                except Exception as e:
                    print(f"[ROI] Warning: analysis failed for {orig_path}: {e}")
                    continue

                data_row = [
                    orig_path, mask_path,
                    capture_date, capture_time,
                    clustering, clustering_params,
                    f"{analyzer.roi_intensity:.2f}",
                    f"{analyzer.roi_entropy:.4f}",
                    f"{analyzer.roi_texture:.4f}",
                    f"{analyzer.mean_gli:.6f}",
                    f"{analyzer.mean_gcc:.6f}",
                    f"{analyzer.ROI_total_pixels:.2f}",
                    f"{analyzer.ROI_total_area:.2f}",
                    f"{analyzer.image_height:.2f}",
                    f"{analyzer.image_width:.2f}",
                    f"{analyzer.image_total_pixels:.2f}",
                    f"{analyzer.ROI_percentage:.2f}",
                ]
                for (h, s, v) in analyzer.dominant_hsv_list:
                    data_row.extend([f"{h:.4f}", f"{s:.4f}", f"{v:.4f}"])
                for (r, g, b) in analyzer.dominant_rgb_list:
                    data_row.extend([f"{r:.4f}", f"{g:.4f}", f"{b:.4f}"])

                rows.append(data_row)

            if not rows:
                print("[ERROR] No results produced. Check image/mask pairs.", file=sys.stderr)
                sys.exit(1)

            roi_metrics_df = pd.DataFrame(rows, columns=header)

            # ----------------------------------------------------------
            # Write CSV
            # ----------------------------------------------------------
            try:
                roi_metrics_df.to_csv(csv_path, index=False)
                print(f"[ROI] CSV written: {csv_path}")
            except Exception as e:
                print(f"[ROI] Warning: could not write CSV: {e}")

            # ----------------------------------------------------------
            # Write XLSX with hyperlinks
            # ----------------------------------------------------------
            try:
                wb = Workbook()
                ws = wb.active
                ws.title = "ROI Metrics"
                hyperlink_style = Font(color="0000FF", underline="single")

                for col_idx, title in enumerate(header, start=1):
                    ws.cell(row=1, column=col_idx, value=title)

                for row_idx, data in enumerate(rows, start=2):
                    (
                        orig_full, mask_full, capture_date, capture_time,
                        method_name, method_params,
                        intensity, entropy, texture, gli, gcc,
                        pixel_count, pixel_area, image_height,
                        image_width, image_total_pixels, roi_area_percentage,
                        *cluster_values
                    ) = data

                    cell_img = ws.cell(row=row_idx, column=1, value=os.path.basename(orig_full))
                    cell_img.hyperlink = orig_full
                    cell_img.font = hyperlink_style

                    cell_mask = ws.cell(row=row_idx, column=2, value=os.path.basename(mask_full))
                    cell_mask.hyperlink = mask_full
                    cell_mask.font = hyperlink_style

                    ws.cell(row=row_idx, column=3, value=capture_date)
                    ws.cell(row=row_idx, column=4, value=capture_time)
                    ws.cell(row=row_idx, column=5, value=method_name)
                    ws.cell(row=row_idx, column=6, value=method_params)
                    ws.cell(row=row_idx, column=7, value=float(intensity))
                    ws.cell(row=row_idx, column=8, value=float(entropy))
                    ws.cell(row=row_idx, column=9, value=float(texture))
                    ws.cell(row=row_idx, column=10, value=float(gli))
                    ws.cell(row=row_idx, column=11, value=float(gcc))
                    ws.cell(row=row_idx, column=12, value=float(pixel_count))
                    ws.cell(row=row_idx, column=13, value=float(pixel_area))
                    ws.cell(row=row_idx, column=14, value=float(image_height))
                    ws.cell(row=row_idx, column=15, value=float(image_width))
                    ws.cell(row=row_idx, column=16, value=float(image_total_pixels))
                    ws.cell(row=row_idx, column=17, value=float(roi_area_percentage))

                    for offset, value in enumerate(cluster_values, start=18):
                        ws.cell(row=row_idx, column=offset, value=float(value))

                wb.save(xlsx_path)
                print(f"[ROI] XLSX written: {xlsx_path}")

            except ImportError:
                print("[ROI] Warning: openpyxl not installed. XLSX export skipped.")
            except Exception as e:
                print(f"[ROI] Warning: could not write XLSX: {e}")

            # ----------------------------------------------------------
            # Align ROI metrics with sensor CSV by nearest timestamp
            # ----------------------------------------------------------
            aligned_df = None
            aligned_written = False

            try:
                if related_csv_df is not None and not related_csv_df.empty:
                    roi_dt = pd.to_datetime(
                        roi_metrics_df["Capture Date"].astype(str).str.strip() + " " +
                        roi_metrics_df["Capture Time"].astype(str).str.strip(),
                        errors="coerce",
                        format="%Y-%m-%d %H:%M:%S"
                    )
                    roi_df = roi_metrics_df.copy()
                    roi_df["datetime"] = roi_dt

                    csv_df = related_csv_df.copy()
                    if {"Date", "Time"}.issubset(csv_df.columns):
                        csv_df["datetime"] = pd.to_datetime(
                            csv_df["Date"].astype(str).str.strip() + " " +
                            csv_df["Time"].astype(str).str.strip(),
                            errors="coerce",
                            format="%Y-%m-%d %H:%M:%S"
                        )
                    elif "datetime" in csv_df.columns:
                        csv_df["datetime"] = pd.to_datetime(csv_df["datetime"], errors="coerce")
                    else:
                        csv_df["datetime"] = pd.NaT

                    roi_df = roi_df.dropna(subset=["datetime"]).sort_values("datetime")
                    csv_df = csv_df.dropna(subset=["datetime"]).sort_values("datetime")

                    if not roi_df.empty and not csv_df.empty:
                        aligned_df = pd.merge_asof(
                            roi_df, csv_df,
                            on="datetime",
                            direction="nearest",
                            tolerance=pd.Timedelta("6H")
                        )
                        aligned_df["Image Path"] = aligned_df["Image Path"].apply(os.path.basename)
                        aligned_df["Mask Path"] = aligned_df["Mask Path"].apply(os.path.basename)
                        aligned_df.to_excel(aligned_xlsx_path, index=False)
                        aligned_written = True
                        print(f"[ROI] Aligned XLSX written: {aligned_xlsx_path}")
                    else:
                        print("[ROI] Alignment skipped: no valid datetime rows in one or both DataFrames.")
                else:
                    print("[ROI] Alignment skipped: no related sensor CSV.")

            except Exception as e:
                print(f"[ROI] Warning: alignment step failed: {e}")

            # ----------------------------------------------------------
            # Run response analysis if alignment produced results
            # ----------------------------------------------------------
            if aligned_df is not None and not aligned_df.empty:
                predictors = []
                for i in range(1, n_clusters + 1):
                    predictors.extend([f"Cluster {i} H", f"Cluster {i} S", f"Cluster {i} V"])
                responses = ["Gage Height", "Discharge"]
                try:
                    _analyze_responses(aligned_df, predictors, responses, output_folder)
                except Exception as e:
                    print(f"[ROI] Warning: response analysis failed: {e}")

            # ----------------------------------------------------------
            # Summary
            # ----------------------------------------------------------
            print("\n[ROI] Extract complete.")
            print(f"  Pairs processed : {len(rows)}")
            print(f"  CSV             : {csv_path}")
            print(f"  XLSX            : {xlsx_path}")
            print(f"  Aligned XLSX    : {aligned_xlsx_path if aligned_written else 'skipped'}")
            print(f"  Related CSV     : {matching_csv_path if matching_csv_path else 'not found'}")

        else:
            print(f"[ERROR] Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
def cli_fetchLocalImageList(filePath, bFetchRecursive=False):

    # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    List = []

    # RECURSE AND TRAVERSE FROM THE SPECIFIED FOLDER DOWN TO DETERMINE THE DATE RANGE FOR THE IMAGES FOUND
    file_count, files = GRIME_AI_Utils().getFileList(filePath, extensions, bFetchRecursive)

    for image_index, file in enumerate(files):
        ext = os.path.splitext(file)[-1].lower()

        if ext in extensions:
            List.append(imageData(file, 0, 0, 0))

    return List


# ======================================================================================================================
#
# ======================================================================================================================
from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()

'''
dirname = os.path.dirname(__file__)
myconfig_path=os.path.normpath(os.path.join(dirname, "sam2\\sam2\\configs\\sam2.1"))
print(myconfig_path)
myconfig_name=os.path.normpath(os.path.join(dirname, "sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_l.yaml"))
print(myconfig_name)
'''

import importlib.util
dirname = os.path.dirname(importlib.util.find_spec('sam2').origin)
myconfig_path = os.path.join(dirname, "configs", "sam2.1")
myconfig_name = "sam2.1_hiera_l"
###JES myconfig_name = os.path.join(dirname, "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_l")

# ======================================================================================================================
#
# ======================================================================================================================
@hydra.main(config_path=myconfig_path, config_name=myconfig_name, version_base="1.3")
def train_main(cfg: DictConfig) -> None:
    global hyperparameterDlg
    if hyperparameterDlg is not None:
        # FETCH ANY DATA WE NEED FROM THE MACHINE LEARNING DIALOG
        selected_training_model = hyperparameterDlg.get_selected_training_model()

        # WE ONLY USE HYDRA WITH SEGMENT ANYTHING MODEL (SAM, SAM2, SAM3)
        if selected_training_model.lower().startswith("sam"):
            print("Config path:", myconfig_path)
            print("Config name:", myconfig_name)

            # If Hydra is already initialized, clear it
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            print("Hydra SAM2 training config:")
            print(OmegaConf.to_yaml(cfg))

        # Keep the ML dialog open — disable Train button during training
        try:
            hyperparameterDlg.training_tab.pushButton_train.setEnabled(False)
        except Exception:
            pass

        # BEGIN TRAINING THE MODEL
        print("Instantiate MLModelTraining class...")
        from GRIME_AI.ml_core.ml_model_training import MLModelTraining
        myML_Dispatcher = MLModelTraining(cfg, parent_widget=hyperparameterDlg)

        print("Execute ML training...")
        trainer = myML_Dispatcher.Model_Training_Dispatcher(cfg=cfg, mode=selected_training_model)

        # Re-enable Train button
        try:
            hyperparameterDlg.training_tab.pushButton_train.setEnabled(True)
        except Exception:
            pass

        # Prompt user about suggested blob radius if available
        if trainer is not None and selected_training_model.lower().startswith("sam"):
            try:
                hyperparameterDlg.training_tab.prompt_blob_radius_update(trainer)
            except Exception as e:
                print(f"[Blob Radius] Could not show prompt: {e}")

# ======================================================================================================================
#
# ======================================================================================================================
@hydra.main(config_path=myconfig_path, config_name=myconfig_name, version_base="1.3")
def segment_main(cfg: DictConfig) -> None:
    # If Hydra is already initialized, clear it
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    print("Hydra SAM2 training config:")
    print(OmegaConf.to_yaml(cfg))

    global hyperparameterDlg
    if hyperparameterDlg is not None:
        copy_original_image = hyperparameterDlg.get_copy_original_image()
        save_masks = hyperparameterDlg.get_saved_masks()
        save_probability_maps = hyperparameterDlg.get_save_probability_maps()
        save_diagnostic_panels = hyperparameterDlg.get_save_diagnostic_panels()
        selected_label_categories = hyperparameterDlg.get_selected_label_categories()
        selected_segment_model = hyperparameterDlg.get_selected_segment_model()

        # Keep the ML dialog open — disable Segment button during inference
        try:
            hyperparameterDlg.segment_tab.pushButton_Segment.setEnabled(False)
        except Exception:
            pass

        from GRIME_AI.ml_core.ml_image_segmentation import MLImageSegmentation
        mySegmentation = MLImageSegmentation(cfg, parent_widget=hyperparameterDlg)
        mySegmentation.ML_Segmentation_Dispatcher(copy_original_image, save_masks, selected_label_categories,
                                                    mode=selected_segment_model,
                                                    save_probability_maps=save_probability_maps,
                                                    save_diagnostic_panels=save_diagnostic_panels)

        # Re-enable Segment button
        try:
            hyperparameterDlg.segment_tab.pushButton_Segment.setEnabled(True)
        except Exception:
            pass

# ======================================================================================================================
#
# ======================================================================================================================
if __name__ == '__main__':

    my_main()


