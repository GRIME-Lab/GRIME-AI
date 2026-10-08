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
try:
    # TRY TO IMPORT THE VERSION CONSTANT FROM VERSION.PY
    from appcore.version import SW_VERSION
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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

from appcore.utils.resource_utils import icon_path

from pathlib import Path

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
from PyQt5 import QtCore, QtWidgets, QtGui, uic
from PyQt5.QtCore import QCoreApplication, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QIcon, QPolygon
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QToolBar, QDateTimeEdit, \
    QMessageBox, QAction, QHeaderView, QDialog, QFileDialog, QSplashScreen
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMenu
from PyQt5.QtWidgets import QTreeWidgetItem

from appcore.SplashScreen import SplashScreen
from appcore.utils.window_utils import (
    install_window_placement_guard,
    reset_window_layout,
)

# pandas imported lazily inside functions

from appcore.usgs.usgs_client import USGSClient

# ----------------------------------------------------------------------------
# POP-UP/MODELESS DIALOG BOXES
# ----------------------------------------------------------------------------
# lazy: from appcore.dialogs.color_segmentation.ColorSegmentationDlg import ColorSegmentationDlg
# lazy: from appcore.dialogs.edge_detection.EdgeDetectionDlg import EdgeDetectionDlg
# lazy: from appcore.dialogs.image_navigation.ImageNavigationDlg import ImageNavigationDlg
# lazy: from appcore.dialogs.file_utilities.FileUtilitiesDlg import FileUtilitiesDlg
# lazy: from appcore.dialogs.mask_editor.MaskEditorDlg import MaskEditorDlg
# lazy: from appcore.dialogs.composite_slice.CompositeSliceDlg import CompositeSliceDlg
# lazy: from appcore.dialogs.release_notes.ReleaseNotesDlg import ReleaseNotesDlg
# lazy: from appcore.dialogs.extract_coco_masks.ExportCOCOMasksDlg import ExportCOCOMasksDlg
# lazy: from appcore.dialogs.image_organizer.ImageOrganizerDlg import ImageOrganizerDlg
from appcore.dialogs.triage.TriageOptionsDlg import TriageOptionsDlg
from appcore.Color import Color
from appcore.vegetation_indices import Vegetation_Indices, GreennessIndex
from appcore.JSON_Editor import JsonEditor
from appcore.ImageData import imageData
from appcore.ProductTable import ProductTable
from appcore.App_QLabel import DrawingMode
from appcore.App_QMessageBox import App_QMessageBox
from appcore.QProgressWheel import QProgressWheel
from appcore.App_Utils import App_Utils
from appcore.dialogs.color_segmentation.color_seg_roi_data import roiData, ROIShape
from appcore.Save_Utils import Save_Utils
from appcore.Resize_Controls import Resize_Controls
from appcore.TimeStamp_Utils import TimeStamp_Utils
from appcore.ImageTriage import ImageTriage
from appcore.GreenImageGenerator import GreenImageGenerator
from appcore.COCO_Utils import COCO_Utils

# ----------------------------------------------------------------------------
# PHENOCAM
# ----------------------------------------------------------------------------
from appcore.phenocam.PhenoCam import PhenoCam, dailyList
from appcore.phenocam.Phenocam_API import Phenocam_API

from appcore.colorSegmentationParams import colorSegmentationParamsClass

# ----------------------------------------------------------------------------
# DIGITAL MAPPING PLATFORMS
# ----------------------------------------------------------------------------
from appcore.geomaps.google_maps_viewer import GoogleMapWidget
from appcore.geomaps.openstreetmap_viewer import OpenStreetMapWidget

# ----------------------------------------------------------------------------------------------------------------------
# HYDRA (for SAM2)
# ----------------------------------------------------------------------------------------------------------------------
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from appcore.neon.NEON_API import NEON_API
from appcore.dialogs.api_keys import APIKeyManager, APIKeyDialog

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from appcore.constants import edgeMethodsClass, featureMethodsClass, modelSettingsClass

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
from appcore.exifData import EXIFData
from appcore.app_identity import APP_NAME, APP_DISPLAY_NAME, USER_ROOT, APP_LOGO_FILENAME, APP_REPO_URL

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
# PHENOCAM TAB: SETTINGS, SITE SEARCH, HOVER HELP
# ======================================================================================================================
import re

# Tuning values for the PhenoCam tab. Each can be overridden by a JsonEditor entry
# of the same name (e.g. "Phenocam_Search_Help_Delay_ms").
PHENOCAM_TAB_DEFAULTS = {
    "Phenocam_Search_Help_Delay_ms": 3000,   # hover time before the search syntax popup appears
    "Phenocam_Search_Debounce_ms":   250,    # pause after typing before the list is filtered
    "Phenocam_Default_Range_Days":   30,     # default download range ending today
    "Phenocam_Button_Radius_px":     12,
    "Phenocam_Preview_Min_Height_px": 120,
    "Phenocam_Table_Blank_Rows":     3,      # table rows shown (blank) when fewer sites are checked
    "Phenocam_Table_Max_Rows":       10,     # site rows shown before the table scrolls
    "Phenocam_Left_Panel_Width_px":  300,    # initial site-list width; saved when the splitter is dragged
    "Phenocam_Placeholder_Color":    "#4d4d4d",
    "Phenocam_Error_Color":          "#b00020",
    "Phenocam_Primary_Button_Color": "steelblue",
    "Phenocam_Table_Header_Color":   "lightsteelblue",   # header fill; text is black for readability
    "Phenocam_Show_Inactive":        False,  # inactive sites hidden until the user shows them
    "Phenocam_Preview_Height_px":    None,   # saved splitter position; None = use the default fraction
    "Phenocam_Preview_Fraction":     2 / 3,  # preview share of the panel height when none is saved
    "Phenocam_Preview_Search_Days":  30,     # days searched back from a site's last image date
    "Phenocam_Count_Max_Days":       366,    # most browse pages fetched per site for a narrow time window
    "Phenocam_Count_Parallel_Pages": 4,      # browse pages fetched at once for narrow-window counts
    "Phenocam_Count_Debounce_ms":    800,    # pause after a date/time edit before counting starts
}


# NEON tab layout values (its look shares the PhenoCam tab's colors and button styles).
NEON_TAB_DEFAULTS = {
    "NEON_Left_Panel_Width_px":     400,    # initial site-list width; saved when the splitter is dragged
    "NEON_Left_Panel_Min_Width_px": 150,    # the site list can't be dragged narrower than this
    "NEON_Preview_Height_px":       None,   # saved image height; None = image takes what the table leaves
    "NEON_Table_Blank_Rows":        3,      # table rows shown (blank) when fewer products are checked
    "NEON_Table_Max_Rows":          10,     # product rows shown before the table scrolls
}

# USGS tab layout values (its look shares the PhenoCam tab's colors and button styles).
USGS_TAB_DEFAULTS = {
    "USGS_Left_Panel_Width_px":  400,    # initial site-list width; saved when the splitter is dragged
    "USGS_Left_Panel_Min_Width_px": 150, # the site list can't be dragged narrower than this
    "USGS_Preview_Height_px":    None,   # saved image height; None = image takes the space the table leaves
    "USGS_Table_Rows_Shown":     4,      # table height in rows (filled plus blank) when few cameras are checked
    "USGS_Table_Max_Rows":       10,     # rows shown before the table scrolls
    "USGS_Count_Debounce_ms":    2000,   # pause after a date/time edit before image counting starts
}


def _phenocam_setting(key):
    """JsonEditor value for key if one is stored, else the PhenoCam or USGS tab default."""
    default = PHENOCAM_TAB_DEFAULTS.get(key, USGS_TAB_DEFAULTS.get(key, NEON_TAB_DEFAULTS.get(
        key, IMAGE_VIEW_DEFAULTS.get(key))))
    try:
        value = JsonEditor().getValue(key)
    except Exception:
        value = None
    if value in (None, ""):
        return default
    if isinstance(default, bool):
        return str(value).strip().lower() == "true"
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, int) or (default is None and str(value).lstrip("-").isdigit()):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default
    return value


def _data_table_header_style():
    """Header style shared by the PhenoCam and USGS download tables."""
    edge = _phenocam_setting("Phenocam_Primary_Button_Color")
    return (f"QHeaderView::section {{ background-color: {_phenocam_setting('Phenocam_Table_Header_Color')};"
            f" color: #000000; font-weight: bold; padding: 4px;"
            f" border: none; border-right: 1px solid {edge}; border-bottom: 1px solid {edge}; }}")


def _button_styles():
    """(primary, ghost) button style sheets shared by the PhenoCam and USGS tabs."""
    radius = int(_phenocam_setting("Phenocam_Button_Radius_px"))
    primary = _phenocam_setting("Phenocam_Primary_Button_Color")
    filled = (f"QPushButton {{ background-color: {primary}; color: white; font-weight: bold;"
              f" border: none; border-radius: {radius}px; padding: 4px 14px; }}"
              f"QPushButton:disabled {{ background-color: palette(mid); }}")
    ghost = (f"QPushButton {{ background: transparent; color: palette(text);"
             f" border: 1px solid {primary}; border-radius: {radius}px; padding: 4px 14px; }}"
             f"QPushButton:hover {{ background: palette(midlight); }}")
    return filled, ghost


# Search fields: name -> (description, kind). kind is "text", "number", or "date".
PHENOCAM_SEARCH_FIELDS = {
    "site":      ("Site name", "text"),
    "location":  ("Site description", "text"),
    "group":     ("Group", "text"),
    "type":      ("Site type (I, II, III)", "text"),
    "veg":       ("Primary or secondary vegetation type", "text"),
    "species":   ("Dominant species", "text"),
    "camera":    ("Camera description", "text"),
    "orient":    ("Camera orientation", "text"),
    "contact":   ("Either site contact", "text"),
    "roi":       ("ROI names", "text"),
    "ecoregion": ("North America ecoregion", "text"),
    "koppen":    ("Köppen-Geiger climate", "text"),
    "igbp":      ("IGBP land cover", "text"),
    "flux":      ("Flux data (true/false)", "text"),
    "active":    ("Active (true/false)", "text"),
    "lat":       ("Latitude", "number"),
    "lon":       ("Longitude", "number"),
    "elev":      ("Elevation (m)", "number"),
    "mat":       ("Daymet mean annual temperature (°C)", "number"),
    "map":       ("Daymet mean annual precipitation (mm)", "number"),
    "first":     ("First image date (YYYY-MM-DD)", "date"),
    "last":      ("Last image date (YYYY-MM-DD)", "date"),
}

# USGS HIVIS camera fields (from the NIMS camera record)
USGS_SEARCH_FIELDS = {
    "camera":  ("Camera ID", "text"),
    "name":    ("Camera name", "text"),
    "desc":    ("Camera description", "text"),
    "nwis":    ("NWIS site number", "text"),
    "state":   ("State abbreviation", "text"),
    "tz":      ("Time zone", "text"),
    "pcode":   ("Default parameter code", "text"),
    "hidden":  ("Hidden camera (true/false)", "text"),
    "lat":     ("Latitude", "number"),
    "lon":     ("Longitude", "number"),
    "newest":  ("Newest image date (YYYY-MM-DD)", "date"),
    "created": ("Camera added to NIMS (YYYY-MM-DD)", "date"),
}

# NEON field site fields; "product" also narrows the products listed under each site
NEON_SEARCH_FIELDS = {
    "site":     ("Site code", "text"),
    "name":     ("Site name", "text"),
    "state":    ("State", "text"),
    "domain":   ("Domain code or name", "text"),
    "phenocam": ("PhenoCam link", "text"),
    "product":  ("Data product code or title (also narrows the product list)", "text"),
    "lat":      ("Latitude", "number"),
    "lon":      ("Longitude", "number"),
}

# Help popup examples per tab: (search text, what it does)
PHENOCAM_SEARCH_EXAMPLES = [
    ("prairie", "a bare word matches any field"),
    ("group:NEON", "matches one field"),
    ("site:NEON.D10.*", "* any characters, ? one character (a wildcard pattern must match the whole value)"),
    ('location:"Nine Mile"', "quotes for values with spaces"),
    ("elev>1000 &nbsp; last<2025-01-01 &nbsp; lat>=40", "comparisons on numbers and dates"),
    ("-group:NEON", "a leading - excludes"),
]
PHENOCAM_SEARCH_EXAMPLE = "group:NEON veg:GR elev>500 -active:false"

USGS_SEARCH_EXAMPLES = [
    ("platte", "a bare word matches any field"),
    ("state:NE", "matches one field"),
    ("camera:NE_Platte*", "* any characters, ? one character (a wildcard pattern must match the whole value)"),
    ('name:"Platte River"', "quotes for values with spaces"),
    ("lat>=40 &nbsp; newest>2026-10-01", "comparisons on numbers and dates"),
    ("-hidden:true", "a leading - excludes"),
]
USGS_SEARCH_EXAMPLE = "state:NE nwis:06* newest>2026-10-01 -hidden:true"

NEON_SEARCH_EXAMPLES = [
    ("harvard", "a bare word matches any field, including product titles"),
    ("domain:D10", "matches one field"),
    ("product:DP1.2*", "* any characters, ? one character (a wildcard pattern must match the whole value)"),
    ('product:"water quality"', "quotes for values with spaces"),
    ("lat>40 &nbsp; lon<-100", "comparisons on numbers"),
    ("-state:AK", "a leading - excludes"),
]
NEON_SEARCH_EXAMPLE = "domain:D01 product:DP1.20002 -state:ME"


class SiteSearchQuery:
    """
    Parses and applies site search text against a set of search fields.

      prairie                 bare word: matches any field
      group:NEON              field:value
      site:NEON.D10.*         * and ? wildcards (pattern must match the whole value)
      location:"Nine Mile"    quotes for values with spaces
      elev>1000  last<2025-01-01  lat>=40   comparisons on number and date fields
      -group:NEON             leading - excludes
    Terms are ANDed and case-insensitive. Without wildcards a value matches
    anywhere inside the field.

    A record is {"fields": {name: [lowercase strings]}, "numbers": {name: float},
    "dates": {name: "YYYY-MM-DD"}}.
    """

    _FIELD_TERM = re.compile(r"^([A-Za-z_]+)(>=|<=|:|>|<|=)(.*)$")

    def __init__(self, text, fields):
        import shlex
        self.fields = fields
        self.error = ""
        self.terms = []   # (negate, field or None, op, value)
        text = (text or "").strip()
        if not text:
            return
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.replace('"', " ").split()
        for tok in tokens:
            negate = tok.startswith("-") and len(tok) > 1
            if negate:
                tok = tok[1:]
            m = self._FIELD_TERM.match(tok)
            if m and m.group(1).lower() in self.fields:
                field, op, value = m.group(1).lower(), m.group(2), m.group(3)
                kind = self.fields[field][1]
                if op in (">", "<", ">=", "<=") and kind == "text":
                    self.error = f"{field} can't be compared with {op}"
                    continue
                if kind == "number" and op != ":":
                    try:
                        float(value)
                    except ValueError:
                        self.error = f"{field}{op} needs a number"
                        continue
                self.terms.append((negate, field, op, value.lower()))
            elif m and m.group(2) in (":", ">=", "<=", ">", "<", "=") and not m.group(3).startswith("//"):
                self.error = f"Unknown field: {m.group(1)}"
            else:
                self.terms.append((negate, None, ":", tok.lower()))

    @staticmethod
    def text_match(pattern, values):
        import fnmatch
        if any(c in pattern for c in "*?"):
            return any(fnmatch.fnmatchcase(v, pattern) for v in values)
        return any(pattern in v for v in values)

    _text_match = text_match

    @staticmethod
    def _compare(a, op, b):
        return {"=": a == b, ">": a > b, "<": a < b, ">=": a >= b, "<=": a <= b}[op]

    def _term_match(self, record, field, op, value):
        fields = record["fields"]
        if field is None:
            return any(self.text_match(value, vals) for vals in fields.values())
        kind = self.fields[field][1]
        if op == ":" or (op == "=" and kind == "text"):
            return self.text_match(value, fields.get(field, []))
        if kind == "number":
            x = record.get("numbers", {}).get(field)
            return x is not None and self._compare(x, op, float(value))
        x = record.get("dates", {}).get(field)   # ISO date strings compare correctly as text
        return bool(x) and self._compare(x, op, value)

    def term_matches(self, record, term):
        """True if the record satisfies the term, ignoring its negation."""
        _negate, field, op, value = term
        return self._term_match(record, field, op, value)

    def matches(self, record):
        for negate, field, op, value in self.terms:
            if self._term_match(record, field, op, value) == negate:
                return False
        return True


class PhenocamSearchQuery(SiteSearchQuery):
    def __init__(self, text):
        super().__init__(text, PHENOCAM_SEARCH_FIELDS)


def _search_help_html(fields, examples, example):
    import html as _html
    rows = "".join(f"<tr><td><b>{name}</b></td><td>{desc}</td></tr>" for name, (desc, _) in fields.items())
    lines = "".join(f"<b>{_html.escape(code).replace('&amp;nbsp;', '&nbsp;')}</b> &nbsp; {text}<br>"
                    for code, text in examples)
    return (
        "<div style='color:#000000'>"
        "<b>Search syntax</b><br>"
        f"{lines}"
        "All terms must match. Case is ignored.<br><br>"
        f"<table cellspacing='0' cellpadding='2'>{rows}</table><br>"
        f"<b>Example:</b> {_html.escape(example)}"
        "</div>"
    )


def _phenocam_search_help_html():
    return _search_help_html(PHENOCAM_SEARCH_FIELDS, PHENOCAM_SEARCH_EXAMPLES, PHENOCAM_SEARCH_EXAMPLE)


class SearchHelpFilter(QtCore.QObject):
    """Shows a search syntax popup after the mouse rests on the search field;
    hides it when the mouse leaves or the user types. Used on all site tabs."""

    def __init__(self, line_edit, help_html, parent=None):
        super().__init__(parent)
        self._edit = line_edit
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(int(_phenocam_setting("Phenocam_Search_Help_Delay_ms")))
        self._timer.timeout.connect(self._show)
        self._popup = QtWidgets.QFrame(line_edit, QtCore.Qt.ToolTip)
        self._popup.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #000000; }"
                                  "QLabel { color: #000000; border: none; }")
        lay = QtWidgets.QVBoxLayout(self._popup)
        lbl = QtWidgets.QLabel(help_html)
        lbl.setTextFormat(QtCore.Qt.RichText)
        lay.addWidget(lbl)
        line_edit.installEventFilter(self)
        line_edit.textEdited.connect(lambda _t: self._hide())

    def _show(self):
        self._popup.adjustSize()
        pos = self._edit.mapToGlobal(QtCore.QPoint(0, self._edit.height()))
        screen = QtWidgets.QApplication.screenAt(pos)
        if screen is not None:
            avail = screen.availableGeometry()
            pos.setY(min(pos.y(), avail.bottom() - self._popup.height()))
            pos.setX(min(pos.x(), avail.right() - self._popup.width()))
        self._popup.move(pos)
        self._popup.show()

    def _hide(self):
        try:
            self._timer.stop()
            self._popup.hide()
        except RuntimeError:
            pass   # popup already destroyed during application shutdown

    def eventFilter(self, obj, event):
        if obj is self._edit:
            t = event.type()
            if t == QtCore.QEvent.Enter:
                self._timer.start()
            elif t in (QtCore.QEvent.Leave, QtCore.QEvent.KeyPress, QtCore.QEvent.Hide):
                self._hide()
        return False


class PhenocamSearchHelpFilter(SearchHelpFilter):
    def __init__(self, line_edit, parent=None):
        super().__init__(line_edit, _phenocam_search_help_html(), parent)


def _search_error_label():
    """Red line under a search field for messages like "Unknown field: grp"."""
    lbl = QtWidgets.QLabel("")
    lbl.setStyleSheet(f"QLabel {{ color: {_phenocam_setting('Phenocam_Error_Color')}; }}")
    lbl.setVisible(False)
    return lbl


# ======================================================================================================================
# ZOOMABLE IMAGE PANEL (latest/midday site images on the NEON, USGS and PhenoCam tabs)
# ======================================================================================================================
# Tuning values for the image panel. Each can be overridden by a JsonEditor entry of the same name.
IMAGE_VIEW_DEFAULTS = {
    "Image_Zoom_Step":           1.25,   # zoom factor per + / - click or mouse-wheel notch
    "Image_Zoom_Min_Percent":    5.0,    # smallest zoom allowed
    "Image_Zoom_Max_Percent":    800.0,  # largest zoom allowed
    "Image_Zoom_Button_Size_px": 30,     # width and height of the zoom buttons
}


class _ZoomGraphicsView(QtWidgets.QGraphicsView):
    """Graphics view that zooms with the mouse wheel (around the cursor) and pans by dragging."""
    wheelZoom = QtCore.pyqtSignal(float)   # zoom factor requested by the wheel

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setStyleSheet("QGraphicsView { background: transparent; }")
        self.setAlignment(QtCore.Qt.AlignCenter)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if steps:
            step = float(_phenocam_setting("Image_Zoom_Step"))
            self.wheelZoom.emit(step ** steps)
        event.accept()


class ZoomImagePanel(QtWidgets.QWidget):
    """
    Image panel with stacked + / - / 100% / Fit buttons on the left, mouse-wheel zoom,
    drag-to-pan, and scroll bars when the image is larger than the panel.

    The tab's existing QLabel is kept and shown for text messages ("Loading...",
    "No image available."); the zoomable view is shown when an image is set.
    A new image opens in Fit; re-setting the same image keeps the current zoom.
    """

    def __init__(self, message_label, parent=None):
        super().__init__(parent)
        self._label = message_label
        self._pixmap = None
        self._fit = True

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        _filled, ghost = _button_styles()
        size = int(_phenocam_setting("Image_Zoom_Button_Size_px"))
        bar = QtWidgets.QVBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(4)
        self.btn_zoom_in = QtWidgets.QPushButton("+")
        self.btn_zoom_out = QtWidgets.QPushButton("−")
        self.btn_actual = QtWidgets.QPushButton("100%")
        self.btn_fit = QtWidgets.QPushButton("Fit")
        tips = ("Zoom in", "Zoom out", "Show the image at its original size", "Fit the image to the panel")
        for btn, tip in zip((self.btn_zoom_in, self.btn_zoom_out, self.btn_actual, self.btn_fit), tips):
            btn.setToolTip(tip)
            btn.setStyleSheet(ghost + "QPushButton { padding: 0px; }")
            btn.setFixedHeight(size)
            btn.setMinimumWidth(size)
            bar.addWidget(btn)
        f = self.btn_zoom_in.font()
        f.setBold(True)
        f.setPointSizeF(f.pointSizeF() * 1.4)
        self.btn_zoom_in.setFont(f)
        self.btn_zoom_out.setFont(f)
        # All buttons share one width, wide enough for "100%" with some breathing room.
        width = max(size, self.btn_actual.fontMetrics().horizontalAdvance("100%") + size // 2)
        for btn in (self.btn_zoom_in, self.btn_zoom_out, self.btn_actual, self.btn_fit):
            btn.setFixedWidth(width)
        bar.addStretch(1)
        lay.addLayout(bar, 0)

        self._view = _ZoomGraphicsView()
        self._item = QtWidgets.QGraphicsPixmapItem()
        self._item.setTransformationMode(QtCore.Qt.SmoothTransformation)
        self._view.scene().addItem(self._item)

        self._stack = QtWidgets.QStackedWidget()
        self._label.setParent(None)
        self._stack.addWidget(self._label)
        self._stack.addWidget(self._view)
        lay.addWidget(self._stack, 1)

        self.btn_zoom_in.clicked.connect(lambda: self.zoom_by(float(_phenocam_setting("Image_Zoom_Step"))))
        self.btn_zoom_out.clicked.connect(lambda: self.zoom_by(1.0 / float(_phenocam_setting("Image_Zoom_Step"))))
        self.btn_actual.clicked.connect(self.actual_size)
        self.btn_fit.clicked.connect(self.fit)
        self._view.wheelZoom.connect(self.zoom_by)
        self._set_buttons_enabled(False)

    # ---- content ------------------------------------------------------------------------------------------------
    def setPixmap(self, pixmap):
        """Show an image. A different image opens in Fit; the same image keeps the current zoom."""
        if pixmap is None or pixmap.isNull():
            return
        same = self._pixmap is not None and self._pixmap.cacheKey() == pixmap.cacheKey()
        if not same:
            self._pixmap = pixmap
            self._item.setPixmap(pixmap)
            self._view.setSceneRect(QtCore.QRectF(pixmap.rect()))
            self._fit = True
        self._stack.setCurrentWidget(self._view)
        self._set_buttons_enabled(True)
        if self._fit:
            QtCore.QTimer.singleShot(0, self._apply_fit)

    def showMessage(self, text):
        """Show a text message in place of the image."""
        self._label.clear()
        self._label.setText(text)
        self._stack.setCurrentWidget(self._label)
        self._set_buttons_enabled(False)

    def pixmap(self):
        return self._pixmap

    # ---- zoom ---------------------------------------------------------------------------------------------------
    def _scale(self):
        return self._view.transform().m11()

    def zoom_by(self, factor):
        if self._pixmap is None or self._stack.currentWidget() is not self._view:
            return
        lo = float(_phenocam_setting("Image_Zoom_Min_Percent")) / 100.0
        hi = float(_phenocam_setting("Image_Zoom_Max_Percent")) / 100.0
        target = max(lo, min(hi, self._scale() * factor))
        self._fit = False
        self._set_scrollbars(True)
        self._view.scale(target / self._scale(), target / self._scale())

    def actual_size(self):
        if self._pixmap is None:
            return
        self._fit = False
        self._set_scrollbars(True)
        self._view.resetTransform()
        self._view.centerOn(self._item)

    def fit(self):
        self._fit = True
        self._apply_fit()

    def _apply_fit(self):
        if self._pixmap is None or not self._fit:
            return
        self._set_scrollbars(False)
        self._view.resetTransform()
        self._view.fitInView(self._item, QtCore.Qt.KeepAspectRatio)

    def _set_scrollbars(self, on):
        policy = QtCore.Qt.ScrollBarAsNeeded if on else QtCore.Qt.ScrollBarAlwaysOff
        self._view.setHorizontalScrollBarPolicy(policy)
        self._view.setVerticalScrollBarPolicy(policy)

    def _set_buttons_enabled(self, on):
        for b in (self.btn_zoom_in, self.btn_zoom_out, self.btn_actual, self.btn_fit):
            b.setEnabled(on)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit:
            self._apply_fit()


# ======================================================================================================================
# PHENOCAM BACKGROUND WORKERS
# ======================================================================================================================
class PhenocamPreviewFetcher(QtCore.QThread):
    result = QtCore.pyqtSignal(object, str)
    def __init__(self, site_name, parent=None, roi_name=None, last_date=None, search_days=30):
        super().__init__(parent)
        self.site_name = site_name
        self.roi_name  = roi_name  # e.g. "NEON.D01.BART.DP1.00033_DB_1000"
        # Search backwards from the site's last image date (from the PhenoCam API),
        # so inactive sites go straight to days that have images instead of
        # walking back from today through days that never will.
        self.last_date   = last_date
        self.search_days = search_days
        self._cancelled  = False
    def cancel(self):
        self._cancelled = True
    def run(self):
        import urllib.request, datetime as dt_mod
        from appcore.phenocam.PhenoCam import PhenoCam
        today = dt_mod.date.today()
        if self.last_date is not None and self.last_date < today:
            today = self.last_date
        t0, t1 = dt_mod.time(11, 0), dt_mod.time(13, 0)
        # Extract ROI type prefix for filename filtering (e.g. "DB" from "..._DB_1000")
        roi_filter = None
        if self.roi_name:
            parts = self.roi_name.split("_")
            # roi_name format: sitename_TYPE_NNNN — TYPE is second-to-last segment
            if len(parts) >= 2:
                roi_filter = parts[-2]  # e.g. "DB", "EN", "SH", etc.
        img_url = None
        for offset in range(self.search_days):
            if self._cancelled:
                return
            check = today - dt_mod.timedelta(days=offset)
            url = (f"https://phenocam.nau.edu/webcam/browse/{self.site_name}/"
                   f"{check.year}/{str(check.month).zfill(2)}/{str(check.day).zfill(2)}")
            try:
                imgs = PhenoCam().getVisibleImages(url, t0, t1).getVisibleList()
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
        if self._cancelled:
            return
        if img_url is None:
            self.result.emit(None, "No image available.")
            return
        try:
            data = urllib.request.urlopen(img_url, timeout=15).read()
            qimg = QtGui.QImage()
            qimg.loadFromData(data)
            self.result.emit(QtGui.QPixmap.fromImage(qimg) if not qimg.isNull() else None,
                             "" if not qimg.isNull() else "Could not decode image.")
        except Exception as e:
            self.result.emit(None, f"Error: {e}")


class PhenocamDailyCountsWorker(QtCore.QThread):
    """
    Fetches each site's per-day RGB image counts from the PhenoCam API
    (/api/dailycounts/?site=...). Two requests per site: one to read the total
    number of days, one to fetch them all.
    """
    siteDone = QtCore.pyqtSignal(str, object)   # site, {ISO date: rgb_count} or None on error
    done = QtCore.pyqtSignal()

    URL = "https://phenocam.nau.edu/api/dailycounts/"

    def __init__(self, sites, parent=None):
        super().__init__(parent)
        self.sites = list(sites)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from urllib.parse import urlencode
        from appcore.http_client import TimeoutSession
        session = TimeoutSession()
        for site in self.sites:
            if self._cancelled:
                return
            try:
                first = session.get(f"{self.URL}?{urlencode({'site': site, 'limit': 1, 'format': 'json'})}")
                first.raise_for_status()
                total = int(first.json().get("count", 0))
                counts = {}
                if total:
                    resp = session.get(f"{self.URL}?{urlencode({'site': site, 'limit': total, 'format': 'json'})}")
                    resp.raise_for_status()
                    for rec in resp.json().get("results", []):
                        if rec.get("site") == site and rec.get("local_date"):
                            counts[str(rec["local_date"])[:10]] = int(rec.get("rgb_count") or 0)
            except Exception as e:
                print(f"[PhenocamDailyCountsWorker] {site}: {e}")
                counts = None
            if self._cancelled:
                return
            self.siteDone.emit(site, counts)
        self.done.emit()


class PhenocamDayPagesWorker(QtCore.QThread):
    """
    Lists image times for (site, day) pairs from PhenoCam's daily browse pages
    (no image downloads), a few pages at a time. Used only when the time window
    is narrower than the whole day, and only for days known to have images.
    """
    dayDone = QtCore.pyqtSignal(str, str, object)   # site, ISO date, [(h, m, s), ...] or None on error
    done = QtCore.pyqtSignal()

    def __init__(self, site_days, parallel=4, parent=None):
        super().__init__(parent)
        self.site_days = list(site_days)              # [(site, datetime.date), ...]
        self.parallel = max(1, int(parallel))
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _list_day(self, site, day):
        import datetime as dt_mod
        from appcore.phenocam.PhenoCam import PhenoCam
        if self._cancelled:
            return site, day, None
        midnight = dt_mod.time(0, 0, 0)
        # Trailing slash avoids a 301 redirect on every request.
        url = (f"https://phenocam.nau.edu/webcam/browse/{site}/"
               f"{day.year}/{str(day.month).zfill(2)}/{str(day.day).zfill(2)}/")
        try:
            # start == end == 00:00 lists every image of the day
            images = PhenoCam().getVisibleImages(url, midnight, midnight).getVisibleList()
            times = []
            for img in images:
                stamp = os.path.basename(img.fullPathAndFilename).split("_")[-1]
                times.append((int(stamp[0:2]), int(stamp[2:4]), int(stamp[4:6])))
            return site, day, times
        except Exception as e:
            print(f"[PhenocamDayPagesWorker] {site} {day}: {e}")
            return site, day, None

    def run(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            futures = [pool.submit(self._list_day, site, day) for site, day in self.site_days]
            for f in as_completed(futures):
                if self._cancelled:
                    for other in futures:
                        other.cancel()
                    return
                site, day, times = f.result()
                self.dayDone.emit(site, day.isoformat(), times)
        if not self._cancelled:
            self.done.emit()


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
        import urllib.request, os, datetime as dt_mod, threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from appcore.phenocam.PhenoCam import PhenoCam

        # All PhenoCam filename timestamps are local time (both NEON via PhenoCam
        # and strictly PhenoCam sites), so the user's local start/end times are
        # compared directly - no UTC conversion needed.
        start_date, end_date = self.start_dt.date(), self.end_dt.date()
        start_time, end_time = self.start_dt.time(), self.end_dt.time()

        os.makedirs(self.save_folder, exist_ok=True)

        # ------------------------------------------------------------------
        # Pipelined scan + download: each day's images are handed to a small
        # download pool the moment that day's scan completes, so downloading
        # overlaps the remaining scanning instead of waiting for all of it.
        # Scanning stays sequential (one browse-page request at a time) and
        # the pool is kept small to stay polite to the PhenoCam server.
        # ------------------------------------------------------------------
        counters = {"downloaded": 0, "done": 0}
        lock = threading.Lock()

        def _fetch(img):
            if self._cancelled:
                return
            filename = os.path.basename(img.fullPathAndFilename)
            dest = os.path.join(self.save_folder, filename)
            got = 0
            if not os.path.isfile(dest):
                try:
                    urllib.request.urlretrieve(img.fullPathAndFilename, dest)
                    got = 1
                except Exception as e:
                    print(f"[PhenocamDownloadWorker] Error downloading {filename}: {e}")
            with lock:
                counters["done"] += 1
                counters["downloaded"] += got

        futures = []
        total_days = (end_date - start_date).days + 1
        pool = ThreadPoolExecutor(max_workers=4)
        try:
            current, day_idx = start_date, 0
            while current <= end_date:
                if self._cancelled:
                    break

                day_idx += 1
                url = (
                    f"https://phenocam.nau.edu/webcam/browse/{self.site_name}/"
                    f"{current.year}/{str(current.month).zfill(2)}/{str(current.day).zfill(2)}"
                )

                try:
                    day_images = PhenoCam().getVisibleImages(
                        url, start_time, end_time
                    ).getVisibleList()
                except Exception as e:
                    print(f"[PhenocamDownloadWorker] Error scanning {current}: {e}")
                    day_images = []

                for img in day_images:
                    futures.append(pool.submit(_fetch, img))

                with lock:
                    dl = counters["downloaded"]
                self.progress.emit(
                    day_idx, total_days,
                    f"Scanning {current.strftime('%Y-%m-%d')}... "
                    f"({len(futures)} images found, {dl} downloaded)"
                )
                current += dt_mod.timedelta(days=1)

            # Scanning finished (or cancelled): report on the remaining
            # downloads as they complete.
            total = len(futures)
            for f in as_completed(futures):
                if self._cancelled:
                    break
                with lock:
                    done = counters["done"]
                self.progress.emit(done, max(total, 1),
                                   f"Downloading... {done}/{total}")
        finally:
            pool.shutdown(wait=not self._cancelled, cancel_futures=self._cancelled)

        if self._cancelled:
            self.finished.emit(-1)
            return

        if not futures:
            self.finished.emit(0)
            return

        # Generate the completeness/gap report (HTML + CSV + optional PDF)
        # in the download folder.  PhenoCam filename timestamps are already
        # site-local, so no timezone conversion is applied.  Never fatal.
        try:
            from appcore.reporting.gap_report import generate_gap_report
            generate_gap_report(self.save_folder)
        except Exception as e:
            print(f"[PhenocamDownloadWorker] Gap report skipped: {e}")

        self.finished.emit(counters["downloaded"])


class NEONPreviewFetcher(QtCore.QThread):
    result = QtCore.pyqtSignal(object, str)
    def __init__(self, site_code, domain_code, product_id=20002, parent=None):
        super().__init__(parent)
        self.site_code = site_code
        self.domain_code = domain_code
        self.product_id = product_id
    def run(self):
        from appcore.neon.NEON_API import NEON_API as _NEON_API
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
            from appcore.usgs.usgs_hivis import USGS_HIVIS
            hivis    = USGS_HIVIS()
            cam_dict = hivis.get_camera_dictionary()
            cam_list = hivis.get_camera_list()
            self.result.emit(hivis, cam_dict, cam_list)
        except Exception as e:
            print(f"[USGSStartupFetcher] Error: {e}")
            self.result.emit(None, {}, [])


def _fix_mojibake(text):
    """Repair UTF-8 text that was decoded as Latin-1 (e.g. 'GuÃ¡nica' -> 'Guánica'); other text is unchanged."""
    if not isinstance(text, str) or not any(ch in text for ch in "ÃÂ"):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


class NEONSitesProductsFetcher(QtCore.QThread):
    """
    Fetches every NEON site's data products in one request (GET {server}sites), in the
    background. Emits {site code: {"domain", "state", "domainName", "products": [
    {"code", "title", "months"}]}} or None on failure. Fields are the same ones the
    per-site request (FetchSiteInfoFromNEON) already reads.
    """
    result = QtCore.pyqtSignal(object)

    def __init__(self, server, parent=None):
        super().__init__(parent)
        self.server = server

    def run(self):
        try:
            from appcore.http_client import TimeoutSession
            resp = TimeoutSession().get(f"{self.server}sites")
            resp.raise_for_status()
            out = {}
            for site in resp.json().get("data", []):
                code = site.get("siteCode")
                if not code:
                    continue
                products = [{"code": p.get("dataProductCode", ""),
                             "title": _fix_mojibake(p.get("dataProductTitle", "")),
                             "months": list(p.get("availableMonths") or [])}
                            for p in site.get("dataProducts", []) or []]
                products.sort(key=lambda p: p["code"])
                out[code] = {"domain": site.get("domainCode", ""), "state": site.get("stateCode", ""),
                             "domainName": site.get("domainName", ""), "products": products}
            self.result.emit(out)
        except Exception as e:
            print(f"[NEONSitesProductsFetcher] {e}")
            self.result.emit(None)


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
        resizeControls = Resize_Controls()

        # TAB 0 - NEON SITES
        resizeControls.resizeTab_0(self, event)
        self.NEON_DisplayLatestImage()

        # TAB 2 - USGS SITES — the code-built layout keeps the site list at its
        # width and gives the image whatever height the table doesn't need.
        if hasattr(self, "_usgs_bottom_panel"):
            self._usgs_fit_layout()

        # TAB 0 - NEON SITES — code-built layout sizes itself like the USGS tab
        if getattr(self, "_neon_layout_built", False):
            self._neon_fit_layout()
        elif not getattr(self, '_neon_splitter_moved_flag', False):
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

        self.setWindowTitle(f"{APP_DISPLAY_NAME}" + " " + SW_VERSION + " - John E. Stranzl Jr., PhD")

        # Menu text carrying the product name is set here, not in the .ui file,
        # so the .ui stays product-neutral.
        self.action_About.setText(f"&About {APP_DISPLAY_NAME}")
        self.tabWidget.setTabVisible(1, False)
        #self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowStaysOnTopHint)

        # Initialize a variable to hold the current NEON site information
        self.current_site_info = ["No site info available."]
        # Set the tooltip generator on your App_QLabel widget(s). For example, if your widget is named NEON_labelLatestImage:
        if 0:
            if hasattr(self.NEON_labelLatestImage, "tooltipGenerator"):
                self.NEON_labelLatestImage.tooltipGenerator = self.siteInfoTooltip
            else:
                print("Warning: NEON_labelLatestImage is not an instance of App_QLabel.")

        # Set stylesheet for the tabs to change color when a tab is selected.
        # Theme-aware: re-applied whenever light/dark mode is toggled.
        self._apply_tab_style()


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
        utils = App_Utils()
        utils.create_GRIME_folders()

        self.populate_controls()

        # On startup, re-apply the active recipe so its saved folders are the
        # source of truth. Any per-session override from the previous run (e.g.
        # an image folder changed in Data Exploration but not saved back to the
        # recipe) is discarded — persisting it requires saving it to the recipe.
        try:
            _active_recipe = self._get_recipe_store().get_active()
            if _active_recipe is not None:
                self.apply_recipe(_active_recipe)
        except Exception as _e:
            print(f"[WARN] Startup recipe re-apply skipped: {_e}")
            traceback.print_exc()

        # ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------
        #JES file_utils = Save_Utils()
        #JES file_utils.read_config_file()

        global imageFileFolder
        imageFileFolder = JsonEditor().getValue("Local_Image_Folder")

        #JES folderPath = Save_Utils().NEON_getSaveFolderPath()
        #JES self.edit_NEONSaveFilePath.1setText(folderPath)

        #JES folderPath = Save_Utils().USGS_getSaveFolderPath()
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

        # Double-clicking the USGS Sites *tab* (the tab itself, not the page)
        # toggles the display of hidden cameras in the site list.
        self.tabWidget.tabBar().installEventFilter(self)
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


        try:
            self._action_recipe_manager = QAction("Recipe Manager\u2026", self)
            self._action_recipe_manager.setStatusTip("Manage per-site folder recipes (root, composites, videos, USGS, NEON)")
            self._action_recipe_manager.triggered.connect(self.menubar_recipe_manager)
            self.menuTools.addSeparator()
            self.menuTools.addAction(self._action_recipe_manager)
            print("[INFO] Recipe Manager added to Tools menu successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to add Recipe Manager to Tools menu: {e}")
            traceback.print_exc()

        try:
            self._action_site_config_editor = QAction("Site Config Editor\u2026", self)
            self._action_site_config_editor.setStatusTip("View, edit, or save a site config JSON under a new filename")
            self._action_site_config_editor.triggered.connect(self.menubar_site_config_editor)
            self.menuTools.addSeparator()
            self.menuTools.addAction(self._action_site_config_editor)
            print("[INFO] Site Config Editor added to Tools menu successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to add Site Config Editor to Tools menu: {e}")
            traceback.print_exc()

        try:
            # Tools > Plugins: drop-in plugins from <user root>/plugins/ that ask
            # for the "tools" surface. The list is rebuilt each time the menu is
            # opened, so a plugin can be added or removed without restarting.
            self._menu_plugins = QMenu("Plugins", self)
            self._menu_plugins.aboutToShow.connect(self._build_plugins_menu)
            self.menuTools.addSeparator()
            self.menuTools.addMenu(self._menu_plugins)
            print("[INFO] Plugins added to Tools menu successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to add Plugins to Tools menu: {e}")
            traceback.print_exc()


        # ------------------------------------------------------------------------------------------------------------------
        # VIEW MENU — dark/light mode toggle
        # ------------------------------------------------------------------------------------------------------------------
        self._is_dark_mode = False
        self._whole_image_ref_set = False
        self._roi_ref_set = set()

        _help_action = self.menuBar().actions()[-1] if self.menuBar().actions() else None
        self._menu_view = QMenu("View", self)
        # Place View immediately after File (before the second top-level menu).
        _bar_actions = self.menuBar().actions()
        _after_file = _bar_actions[1] if len(_bar_actions) > 1 else _help_action
        self.menuBar().insertMenu(_after_file, self._menu_view)

        self._action_follow_system = QAction("Follow System Theme", self)
        self._action_follow_system.setCheckable(True)
        self._action_follow_system.setStatusTip(
            "Automatically match the operating system's light/dark theme")
        self._action_follow_system.toggled.connect(self._on_follow_system_toggled)
        self._menu_view.addAction(self._action_follow_system)

        self._action_toggle_theme = QAction("Dark Mode", self)
        self._action_toggle_theme.setStatusTip("Toggle between dark and light application theme")
        self._action_toggle_theme.triggered.connect(self._toggle_dark_mode)
        self._menu_view.addAction(self._action_toggle_theme)

        # Re-apply the persisted theme settings from the previous run.
        # Fresh install (nothing saved): light mode, Follow System off.
        self._theme_poll_timer = None
        self._restore_theme_settings()

        # Escape hatch for users whose window ends up mispositioned or
        # off-screen (e.g. after a VNC resolution change). Restores a sane
        # default size and re-centers inside the usable screen area.
        self._menu_view.addSeparator()
        self._action_reset_layout = QAction("Reset Window Layout", self)
        self._action_reset_layout.setStatusTip(
            f"Resize and re-center the {APP_DISPLAY_NAME} window inside the visible screen area")
        self._action_reset_layout.triggered.connect(self._reset_window_layout)
        self._menu_view.addAction(self._action_reset_layout)

        # ------------------------------------------------------------------------------------------------------------------
        # MENU REORGANIZATION — split the crowded Tools menu into topic menus.
        # Actions already exist (from the .ui or created above); here they are
        # re-parented into Data Explorer / Annotations / Connectivity /
        # Productivity / Test menus. Tools keeps SAGE and the plugins.
        # ------------------------------------------------------------------------------------------------------------------
        try:
            def _move_action(attr_name, target_menu):
                action = getattr(self, attr_name, None)
                if action is None:
                    return
                try:
                    self.menuTools.removeAction(action)
                except Exception:
                    pass
                target_menu.addAction(action)

            self._menu_data_explorer = QMenu("Data Explorer", self)
            self._menu_annotations   = QMenu("Annotations", self)
            self._menu_connectivity  = QMenu("Connectivity", self)
            self._menu_productivity  = QMenu("Productivity", self)
            self._menu_test          = QMenu("Test", self)

            # Data Explorer
            _move_action("action_CompositeSlices", self._menu_data_explorer)
            _move_action("action_TriageImages",    self._menu_data_explorer)
            _move_action("action_ImageOrganizer",  self._menu_data_explorer)
            # Annotations
            _move_action("action_CreateJSON",           self._menu_annotations)
            _move_action("action_ExtractCOCOMasks",     self._menu_annotations)
            _move_action("action_Inspect_Annotations",  self._menu_annotations)
            _move_action("action_Sync_JSON_Annotations", self._menu_annotations)
            # Connectivity
            _move_action("_action_api_keys",  self._menu_connectivity)
            _move_action("action_RefreshNEON", self._menu_connectivity)
            # Productivity
            _move_action("_action_recipe_manager",     self._menu_productivity)
            _move_action("_action_site_config_editor", self._menu_productivity)
            # Test
            _move_action("action_Generate_Greenness_Test_Images", self._menu_test)

            # Collapse separators left dangling in Tools after moving items out.
            _prev_sep = True
            for _a in list(self.menuTools.actions()):
                if _a.isSeparator():
                    if _prev_sep:
                        self.menuTools.removeAction(_a)
                    else:
                        _prev_sep = True
                else:
                    _prev_sep = False
            _left = self.menuTools.actions()
            if _left and _left[-1].isSeparator():
                self.menuTools.removeAction(_left[-1])

            # Place content menus just before Tools; put Test immediately left of Help.
            _tools_action = self.menuTools.menuAction()
            for _m in (self._menu_data_explorer, self._menu_annotations,
                       self._menu_connectivity, self._menu_productivity):
                self.menuBar().insertMenu(_tools_action, _m)
            _help_action_bar = self.menuBar().actions()[-1] if self.menuBar().actions() else None
            self.menuBar().insertMenu(_help_action_bar, self._menu_test)
            print("[INFO] Tools menu reorganized into topic menus.")
        except Exception as e:
            print(f"[ERROR] Failed to reorganize menus: {e}")
            traceback.print_exc()

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

        # List/tree row text size. These views set no font in the .ui, so they
        # inherited the default (too large). Pin to Arial 10 to match the app.
        _list_font = QFont("Arial", 10)
        for _view_name in ("NEON_listboxSites", "NEON_listboxSiteProducts",
                           "USGS_listboxSites"):
            _view = getattr(self, _view_name, None)
            if _view is not None:
                _view.setFont(_list_font)

        # ------------------------------------------------------------------------------------------------------------------
        # USGS
        # ------------------------------------------------------------------------------------------------------------------
        self.usgs = USGSClient()
        self.usgs.initialize()
        self._nwis_fetcher = None   # holds the active NWISParameterFetcher thread
        self._usgs_image_fetcher = None  # holds the active USGSLatestImageFetcher thread
        self.USGS_listboxSites.itemClicked.connect(self._usgs_tree_item_clicked)
        self.pushButton_USGSDownload.clicked.connect(self.pushButton_USGSDownloadClicked)

        # Correlate Sensor Data button - added programmatically next to the
        # USGS Download button (no .ui change required).
        try:
            self.pushButton_USGSCorrelate = QtWidgets.QPushButton("Correlate Sensor Data")
            self.pushButton_USGSCorrelate.setToolTip(
                "Correlate downloaded image timestamps with co-located NWIS sensor data")
            _dl_layout = self.pushButton_USGSDownload.parentWidget().layout()
            if _dl_layout is not None:
                _idx = _dl_layout.indexOf(self.pushButton_USGSDownload)
                _dl_layout.insertWidget(_idx, self.pushButton_USGSCorrelate)
            self.pushButton_USGSCorrelate.clicked.connect(self.pushButton_USGSCorrelate_Clicked)
        except Exception as _e:
            print(f"[USGS] Could not add Correlate button: {_e}")

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

        # USGS tab layout is built in code to match the PhenoCam tab (sites on the
        # left; image over the table on the right). Restyles the buttons above.
        self.setup_usgs_layout()

        # NEON tab layout, built in code the same way (sites with details and products on
        # the left; image over the checked-products table on the right).
        self.setup_neon_layout()

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
    def _reset_window_layout(self):
        """View -> Reset Window Layout. Puts the window back on screen."""
        try:
            reset_window_layout(self)
        except Exception as _e:
            print(f"[WARN] Reset Window Layout failed: {_e}")

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def _apply_tab_style(self):
        """Main tab bar colors. Light: white tabs, black text. Dark: dark tabs,
        white text, white border. Selected tab is steelblue in both."""
        if getattr(self, "_is_dark_mode", False):
            self.tabWidget.setStyleSheet("""
                QTabBar::tab {
                    background-color: #19232D;
                    color: white;
                    border: 1px solid white;
                    padding: 3px 10px;
                    margin-right: 2px;
                    font-size: 10pt;
                }
                QTabBar::tab:selected {
                    background-color: steelblue;
                    color: white;
                }
                QTabBar::tab:hover:!selected {
                    background-color: #32414B;
                }
            """)
        else:
            self.tabWidget.setStyleSheet("""
                QTabBar::tab {
                    background-color: white;
                    color: black;
                    font-size: 10pt;
                }
                QTabBar::tab:selected {
                    background-color: steelblue;
                    color: white;
                }
            """)

    def _toggle_dark_mode(self):
        app = QApplication.instance()

        # Snapshot the light-mode look once so it can be fully restored
        if not hasattr(self, "_light_palette"):
            self._light_palette = QtGui.QPalette(app.palette())
            self._light_style   = app.style().objectName()

        if not self._is_dark_mode:
            if not self._apply_dark_theme(app):
                # Nothing was applied - stay in light mode and tell the user
                # instead of silently doing nothing.
                try:
                    self.statusBar().showMessage(
                        "Dark mode unavailable: could not apply a dark theme "
                        "(install 'qdarkstyle' for the full theme).", 8000)
                except Exception:
                    pass
                return
            self._is_dark_mode = True
            self._action_toggle_theme.setText("Light Mode")
        else:
            # Restore the saved light-mode style, palette, and stylesheet
            app.setStyleSheet("")
            try:
                app.setStyle(self._light_style)
            except Exception:
                pass
            app.setPalette(self._light_palette)
            self._is_dark_mode = False
            self._action_toggle_theme.setText("Dark Mode")

        self._save_theme_setting()
        try:
            self._apply_tab_style()
        except Exception as e:
            print(f"[WARN] Tab style not applied: {e}")
        # Let dialogs and panels switch their own colors.
        try:
            from appcore.utils import theme
            theme.set_dark(self._is_dark_mode)
        except Exception as e:
            print(f"[WARN] Theme change not broadcast: {e}")

    def _apply_dark_theme(self, app) -> bool:
        """Apply a dark theme to the whole application.
        Prefers qdarkstyle; falls back to Qt's built-in Fusion style with a
        dark palette when qdarkstyle is not installed. Returns True if a
        dark theme was applied.
        """
        # Preferred: qdarkstyle stylesheet
        try:
            import qdarkstyle
            app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
            return True
        except Exception as e:
            print(f"[Theme] qdarkstyle unavailable ({e}); using built-in dark palette")

        # Fallback: Fusion style + dark QPalette (no external dependency)
        try:
            app.setStyle("Fusion")
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Window,          QtGui.QColor(53, 53, 53))
            palette.setColor(QtGui.QPalette.WindowText,      QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Base,            QtGui.QColor(35, 35, 35))
            palette.setColor(QtGui.QPalette.AlternateBase,   QtGui.QColor(53, 53, 53))
            palette.setColor(QtGui.QPalette.ToolTipBase,     QtGui.QColor(53, 53, 53))
            palette.setColor(QtGui.QPalette.ToolTipText,     QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Text,            QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Button,          QtGui.QColor(53, 53, 53))
            palette.setColor(QtGui.QPalette.ButtonText,      QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.BrightText,      QtCore.Qt.red)
            palette.setColor(QtGui.QPalette.Link,            QtGui.QColor(42, 130, 218))
            palette.setColor(QtGui.QPalette.Highlight,       QtGui.QColor(42, 130, 218))
            palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
            palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text,       QtGui.QColor(127, 127, 127))
            palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(127, 127, 127))
            app.setPalette(palette)
            app.setStyleSheet("")
            return True
        except Exception as e:
            print(f"[Theme] Could not apply fallback dark palette: {e}")
            return False

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def _restore_theme_settings(self):
        """Apply the theme settings persisted from the previous run.
        Follow System takes precedence over the manual Dark_Mode setting.
        With nothing saved (fresh install), the app starts in light mode."""
        if self._json_flag("Dark_Mode_Follow_System"):
            # setChecked fires _on_follow_system_toggled, which applies the
            # OS theme and starts the polling timer.
            self._action_follow_system.setChecked(True)
        elif self._json_flag("Dark_Mode"):
            self._toggle_dark_mode()

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _json_flag(key) -> bool:
        """Read a boolean setting from the application settings JSON."""
        try:
            value = JsonEditor().getValue(key)
        except Exception:
            return False
        return str(value).strip().lower() in ("true", "1", "yes")

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def _save_theme_setting(self):
        """Persist the current manual dark/light choice."""
        try:
            JsonEditor().update_json_entry("Dark_Mode", str(bool(self._is_dark_mode)))
        except Exception as e:
            print(f"[Theme] Could not save Dark_Mode setting: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def _detect_os_theme(self):
        """Return True if the OS theme is dark, False if light, or None when
        detection is unavailable (darkdetect missing or unsupported platform)."""
        try:
            import darkdetect
            result = darkdetect.isDark()
            return None if result is None else bool(result)
        except Exception:
            return None

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def _on_follow_system_toggled(self, checked):
        """Enable/disable following the OS theme. While enabled, the manual
        toggle is greyed out and a timer re-checks the OS theme every 2 s."""
        try:
            JsonEditor().update_json_entry("Dark_Mode_Follow_System", str(bool(checked)))
        except Exception as e:
            print(f"[Theme] Could not save Dark_Mode_Follow_System setting: {e}")

        if checked:
            if self._detect_os_theme() is None:
                try:
                    self.statusBar().showMessage(
                        "Follow System Theme requires the 'darkdetect' package "
                        "(pip install darkdetect).", 8000)
                except Exception:
                    pass
                self._action_follow_system.blockSignals(True)
                self._action_follow_system.setChecked(False)
                self._action_follow_system.blockSignals(False)
                try:
                    JsonEditor().update_json_entry("Dark_Mode_Follow_System", "False")
                except Exception:
                    pass
                return

            self._action_toggle_theme.setEnabled(False)
            self._sync_to_os_theme()
            if self._theme_poll_timer is None:
                self._theme_poll_timer = QtCore.QTimer(self)
                self._theme_poll_timer.timeout.connect(self._sync_to_os_theme)
            self._theme_poll_timer.start(2000)
        else:
            if self._theme_poll_timer is not None:
                self._theme_poll_timer.stop()
            self._action_toggle_theme.setEnabled(True)

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def _sync_to_os_theme(self):
        """Switch the application theme to match the OS theme if they differ."""
        os_dark = self._detect_os_theme()
        if os_dark is not None and os_dark != self._is_dark_mode:
            self._toggle_dark_mode()

    def _show_about_dialog(self):
        try:
            from appcore.version import SW_VERSION, RELEASE, BUILD_DATE, SHA
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
        dlg.setWindowTitle(f"About {APP_DISPLAY_NAME}")
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dlg.setFixedSize(480, 420)

        layout = QVBoxLayout(dlg)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(12)

        splash_dir = Path(__file__).resolve().parent / "resources" / "splash_screens"
        logo_path  = splash_dir / APP_LOGO_FILENAME
        lbl_logo   = QLabel()
        lbl_logo.setAlignment(Qt.AlignCenter)
        if logo_path.exists():
            pix = QPixmap(str(logo_path)).scaledToWidth(320, Qt.SmoothTransformation)
            lbl_logo.setPixmap(pix)
        else:
            lbl_logo.setText(APP_DISPLAY_NAME)
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
            commit_url = f"{APP_REPO_URL}/commit/{SHA}"
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
            msgBox = App_QMessageBox('USGS NIMS Error', 'Unable to access USGS NIMS/HIVIS Database!')
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

        self._populate_usgs_sites_tree()

        self.USGS_listboxSites.itemExpanded.connect(self._on_usgs_site_expanded)
        self.USGS_listboxSites.show()

        cameraIndex = 1
        default_item = self.USGS_listboxSites.topLevelItem(cameraIndex)
        if default_item:
            self.USGS_listboxSites.setCurrentItem(default_item)

        self.USGS_updateSiteInfo(1)

        # Add USGS pins to map now that we have the camera dictionary
        for name, coords in self.cameraDictionary.items():
            try:
                self.osm_widget.add_pin(
                    coords["lat"], coords["lng"], color="usgs_green",
                    label=self._osm_pin_label(name, coords, href=self._usgs_href(coords)), group="USGS")
            except Exception as _e:
                print(f"[WARN] USGS pin {name!r} skipped: {_e}")

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
                self.NEON_listboxSites.addTopLevelItem(self._neon_build_site_item(site))

            self.NEON_listboxSites.collapseAll()
            self._neon_apply_filter()

            # Add NEON map pins
            for coords in self.NEON_siteList:
                self.osm_widget.add_pin(coords.latitude, coords.longitude, color="gold", label=coords.siteName, group="NEON")

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
            # DEGRADE, DON'T BLOCK. PhenoCam being unreachable must not hold the
            # whole app at the splash screen: NEON and every other feature are
            # independent of it. Mark this startup stage complete (with an empty
            # site table) so the splash can dismiss and the app proceeds with
            # PhenoCam features simply unavailable.
            self.phenocam_site_dictionary = {}
            self._phenocam_startup_ready = True
            self._check_and_dismiss_splash()
            return
        self.phenocam_site_dictionary = data
        for site_id, info in self.phenocam_site_dictionary.items():
            try:
                from urllib.parse import urljoin
                _pc_base = "https://phenocam.nau.edu"
                _link = info.get("link") if hasattr(info, "get") else None
                href = urljoin(_pc_base, _link) if _link else f"{_pc_base}/webcam/sites/{site_id}/"
                self.osm_widget.add_pin(
                    info["lat"], info["lon"], color="yellow",
                    label=self._osm_pin_label(site_id, info, href=href, base_url=_pc_base), group="PhenoCam")
            except Exception as _e:
                print(f"[WARN] PhenoCam pin {site_id!r} skipped: {_e}")
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

    def _osm_pin_label(self, title, data=None, href=None, base_url=None):
        """Build a MESONET-style HTML popup label. `title` becomes a hyperlink
        when `href` is given; `data` (any mapping) adds prettified metadata rows.
        Field values that are URLs (or site-relative paths, resolved against
        `base_url`) are rendered as clickable links."""
        from urllib.parse import urljoin
        key_labels = {
            "lat": "Latitude", "lon": "Longitude", "lng": "Longitude",
            "nwsli": "NWSLI", "elv_ft": "Elevation (ft)", "utc_offset": "Time Zone",
            "site_no": "Site Number", "site_number": "Site Number",
            "state": "State", "county": "County", "status": "Status",
            "description": "Description", "url": "URL", "name": "Name",
            "sitename": "Site Name", "siteid": "Site ID",
            "elevation": "Elevation", "network": "Network", "link": "Link",
        }
        def pretty(k):
            return key_labels.get(str(k).lower(),
                                  str(k).replace("_", " ").strip().title())

        def as_link(s):
            # Only linkify when a base_url is supplied (e.g. PhenoCam). This keeps
            # other sources' labels exactly as before and avoids breaking the popup
            # JS on odd values.
            if not base_url:
                return None
            if s.startswith(("http://", "https://")):
                full = s
            elif s.startswith("/"):
                full = urljoin(base_url, s)
            else:
                return None
            # Reject anything that could break the anchor / popup JS.
            if any(c in full for c in ' "\'<>\n\r'):
                return None
            return f'<a href="{full}" title="Open link">{full}</a>'

        head = (f'<a href="{href}" title="Open site page">{title}</a>'
                if href else str(title))
        rows = [f"<b>{head}</b>"]
        if data is not None and hasattr(data, "items"):
            for k, v in data.items():
                if v in ("", None):
                    continue
                # Collapse newlines so a value can never break the popup JS string.
                s = str(v).replace("\r", " ").replace("\n", " ").strip()
                if not s or s.lower() == "nan":
                    continue
                rows.append(f"<b>{pretty(k)}:</b> {as_link(s) or s}")
        return "<br>".join(rows)

    def _usgs_href(self, coords):
        """Best-effort NWIS monitoring-location link from a USGS camera record."""
        try:
            get = coords.get if hasattr(coords, "get") else (lambda *_: None)
            for k in ("site_no", "site_number", "siteNumber", "nwis_id"):
                v = get(k)
                if v:
                    return f"https://waterdata.usgs.gov/monitoring-location/{str(v).strip()}/"
        except Exception:
            pass
        return None

    def init_openstreetmap(self, cameraDictionary, redList=None, geojson_str=None):
        osm_tab = self.findChild(QWidget, "tab_GoogleMaps")

        self.osm_widget = OpenStreetMapWidget(parent=osm_tab)

        layout = QVBoxLayout(osm_tab)
        layout.addWidget(self.osm_widget)

        # POPULATE THE MAP WITH PINS FOR USGS HIVIS CAMERA SITES
        try:
            for name, coords in self.cameraDictionary.items():
                self.osm_widget.add_pin(
                    coords["lat"], coords["lng"], color="usgs_green",
                    label=self._osm_pin_label(name, coords, href=self._usgs_href(coords)), group="USGS")
        except Exception as _e:
            print(f"[WARN] USGS pins skipped: {_e}")

        # POPULATE THE MAP WITH PINS FOR PHENOCAM CAMERA SITES REFERENCED BY NEON
        try:
            for site_id, info in self.phenocam_site_dictionary.items():
                from urllib.parse import urljoin
                _pc_base = "https://phenocam.nau.edu"
                _link = info.get("link") if hasattr(info, "get") else None
                href = urljoin(_pc_base, _link) if _link else f"{_pc_base}/webcam/sites/{site_id}/"
                self.osm_widget.add_pin(
                    info["lat"], info["lon"], color="gold",
                    label=self._osm_pin_label(site_id, info, href=href, base_url=_pc_base), group="PhenoCam")
        except Exception as _e:
            print(f"[WARN] PhenoCam pins skipped: {_e}")

        # POPULATE THE MAP WITH PINS FOR NEON FIELD SITES
        try:
            for coords in self.NEON_siteList:
                self.osm_widget.add_pin(coords.latitude, coords.longitude, color="yellow", group="NEON")
        except Exception as _e:
            print(f"[WARN] NEON pins skipped: {_e}")

        # POPULATE THE MAP WITH BLUE PINS FOR SD MESONET STATIONS
        try:
            from appcore.geomaps.SDMESONET import SDMesonet
            sd_df = SDMesonet().get_dataframe()
            self.osm_widget.add_sdmesonet_pins(sd_df, group="MESONET")
        except Exception as e:
            print(f"SD Mesonet pins skipped: {e}")

        # POPULATE THE MAP WITH EAR-OF-CORN PINS (RED HUSKS, YELLOW CORN) FOR NE MESONET STATIONS
        try:
            from appcore.geomaps.NEMESONET import NEMesonet, CORN_SVG
            ne_df = NEMesonet().get_dataframe()
            self.osm_widget.add_nemesonet_pins(ne_df, CORN_SVG, group="MESONET")
        except Exception as e:
            print(f"NE Mesonet pins skipped: {e}")

        # POPULATE THE MAP WITH SAGUARO PINS FOR ARIZONA AZMET STATIONS
        try:
            from appcore.geomaps.AZMET import AZMet, SAGUARO_SVG
            az_df = AZMet().get_dataframe()
            self.osm_widget.add_azmet_pins(az_df, SAGUARO_SVG, group="MESONET")
        except Exception as e:
            print(f"AZMet pins skipped: {e}")

        # POPULATE THE MAP WITH SUNFLOWER PINS FOR KANSAS MESONET STATIONS
        try:
            from appcore.geomaps.KSMESONET import KSMesonet, SUNFLOWER_SVG
            ks_df = KSMesonet().get_dataframe()
            self.osm_widget.add_kansas_pins(ks_df, SUNFLOWER_SVG, group="MESONET")
        except Exception as e:
            print(f"Kansas Mesonet pins skipped: {e}")

        # POPULATE THE MAP WITH BLACK-AND-GOLD CORN PINS FOR ISU SOIL MOISTURE (IOWA) STATIONS
        try:
            from appcore.geomaps.ISUSM import ISUSoilMoisture, IOWA_CORN_SVG
            ia_df = ISUSoilMoisture().get_dataframe()
            self.osm_widget.add_isusm_pins(ia_df, IOWA_CORN_SVG, group="MESONET")
        except Exception as e:
            print(f"ISU Soil Moisture pins skipped: {e}")

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
        api = Phenocam_API()
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
            ("contact1",   "Contact 1"),
            ("contact2",   "Contact 2"),
            # Site metadata (nested under "sitemetadata" in the API; json_normalize flattens it)
            ("sitemetadata.site_description",      "Location"),
            ("sitemetadata.group",                 "Group"),
            ("sitemetadata.camera_description",    "Camera Description"),
            ("sitemetadata.camera_orientation",    "Camera Orientation"),
            ("sitemetadata.site_type",             "Site Type"),
            ("sitemetadata.site_meteorology",      "Site Meteorology"),
            ("sitemetadata.flux_data",             "Flux Data"),
            ("sitemetadata.flux_networks",         "Flux Networks"),
            ("sitemetadata.flux_sitenames",        "Flux Site Names"),
            ("sitemetadata.MAT_site",              "Mean Annual Temperature, Site (\u00b0C)"),
            ("sitemetadata.MAP_site",              "Mean Annual Precipitation, Site (mm)"),
            ("sitemetadata.MAT_daymet",            "Mean Annual Temperature, Daymet (\u00b0C)"),
            ("sitemetadata.MAP_daymet",            "Mean Annual Precipitation, Daymet (mm)"),
            ("sitemetadata.MAT_worldclim",         "Mean Annual Temperature, WorldClim (\u00b0C)"),
            ("sitemetadata.MAP_worldclim",         "Mean Annual Precipitation, WorldClim (mm)"),
            ("sitemetadata.dominant_species",      "Dominant Species"),
            ("sitemetadata.primary_veg_type",      "Primary Vegetation Type"),
            ("sitemetadata.secondary_veg_type",    "Secondary Vegetation Type"),
            ("sitemetadata.ecoregion",             "North America Ecoregion"),
            ("sitemetadata.koeppen_geiger",        "K\u00f6ppen-Geiger Climate"),
            ("sitemetadata.landcover_igbp",        "IGBP Land Cover"),
            ("sitemetadata.site_acknowledgements", "Acknowledgements"),
            ("sitemetadata.modified",              "Metadata Last Modified"),
        ]

        ROI_LINK_FIELDS = [
            ("roi_page",                   "ROI Page"),
            ("roi_stats_file",             "ROI Stats File"),
            ("one_day_summary",            "1-Day Summary"),
            ("three_day_summary",          "3-Day Summary"),
            ("one_day_transition_dates",   "1-Day Transition Dates"),
            ("three_day_transition_dates", "3-Day Transition Dates"),
        ]

        self.treeWidget_Phenocam.blockSignals(True)
        self.treeWidget_Phenocam.clear()
        self._phenocam_records = {}   # site -> searchable fields (see PhenocamSearchQuery)

        def _text(val):
            if isinstance(val, list):
                return ", ".join(str(v.get("Name", v)) if isinstance(v, dict) else str(v) for v in val).lower()
            if val is None or (not isinstance(val, str) and pd.isna(val)):
                return ""
            return str(val).strip().lower()

        def _number(val):
            try:
                x = float(val)
                return None if pd.isna(x) else x
            except (TypeError, ValueError):
                return None

        for site_name, group in cameras_df.groupby(site_col):
            # ── Top-level: site name only, with a checkbox to queue it for download ──
            site_item = QTreeWidgetItem([str(site_name)])
            site_item.setData(0, QtCore.Qt.UserRole, str(site_name))
            site_item.setFlags(site_item.flags() | QtCore.Qt.ItemIsUserCheckable)
            site_item.setCheckState(0, QtCore.Qt.Unchecked)

            row = group.iloc[0]

            # Gray out inactive cameras, using the theme's disabled-text color.
            # The item stays selectable so archived imagery can still be downloaded.
            active_val = row.get("active", None)
            if active_val is False or str(active_val).strip().lower() == "false":
                site_item.setForeground(0, QtGui.QBrush(
                    self.treeWidget_Phenocam.palette().color(QtGui.QPalette.Disabled, QtGui.QPalette.Text)))
                site_item.setToolTip(0, "Inactive camera")

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
                if isinstance(val, list):
                    # flux_networks is a list of {"Name", "NetworkURL", "Description"}
                    val = ", ".join(str(v.get("Name", v)) if isinstance(v, dict) else str(v) for v in val)
                    if not val:
                        continue
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

            # ── Searchable record for this site ────────────────────────────
            roi_names = []
            if roi_site_col is not None and not rois_df.empty:
                roi_names = [_text(n) for n in rois_df.loc[rois_df[roi_site_col] == site_name, "roi_name"]] \
                    if "roi_name" in rois_df.columns else []
            g = lambda col: _text(row.get(col, None))
            self._phenocam_records[str(site_name)] = {
                "fields": {
                    "site":      [str(site_name).lower()],
                    "location":  [g("sitemetadata.site_description")],
                    "group":     [g("sitemetadata.group")],
                    "type":      [g("sitemetadata.site_type")],
                    "veg":       [g("sitemetadata.primary_veg_type"), g("sitemetadata.secondary_veg_type")],
                    "species":   [g("sitemetadata.dominant_species")],
                    "camera":    [g("sitemetadata.camera_description")],
                    "orient":    [g("sitemetadata.camera_orientation")],
                    "contact":   [g("contact1"), g("contact2")],
                    "roi":       roi_names,
                    "ecoregion": [g("sitemetadata.ecoregion")],
                    "koppen":    [g("sitemetadata.koeppen_geiger")],
                    "igbp":      [g("sitemetadata.landcover_igbp")],
                    "flux":      [g("sitemetadata.flux_data")],
                    "active":    [g("active")],
                },
                "numbers": {
                    "lat":  _number(row.get("Lat", None)),
                    "lon":  _number(row.get("Lon", None)),
                    "elev": _number(row.get("Elev", None)),
                    "mat":  _number(row.get("sitemetadata.MAT_daymet", None)),
                    "map":  _number(row.get("sitemetadata.MAP_daymet", None)),
                },
                "dates": {
                    "first": g("date_first")[:10],
                    "last":  g("date_last")[:10],
                },
                "active": g("active") != "false",
            }

            self.treeWidget_Phenocam.addTopLevelItem(site_item)

        self.treeWidget_Phenocam.collapseAll()
        self.treeWidget_Phenocam.blockSignals(False)
        if hasattr(self, "phenocam_table"):
            self._phenocam_rebuild_table()
            self._phenocam_apply_filter()

        ###JES - THIS INTERACTION WITH THE PHENOCAM SERVERS TAKES ON THE ORDER OF 4 MINUTES; SO I
        if 0:
            print("START: Phenocam Meta Data Export...")
            api.export_all_to_excel(output_dir=Save_Utils().get_phenocam_folder())
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
        # ── LEFT: tree fills the panel; the panel width is set by a splitter ─────
        self.treeWidget_Phenocam.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        # Expanding vertical policy lets the layout stretch the tree to fill the
        # full tab height rather than leaving dead space below it.
        self.treeWidget_Phenocam.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        # Remove AlignTop so the layout distributes space rather than pinning to top
        self.verticalLayout_PhenocamLeft.setAlignment(QtCore.Qt.Alignment())

        self._pc_primary_btn_style, self._pc_ghost_btn_style = _button_styles()

        # ── LEFT: search row above the tree, count + inactive toggle below it ───
        left = self.verticalLayout_PhenocamLeft
        tree_idx = max(left.indexOf(self.treeWidget_Phenocam), 0)

        search_row = QtWidgets.QHBoxLayout()
        search_row.setSpacing(4)
        self.phenocam_search_edit = QtWidgets.QLineEdit()
        self.phenocam_search_edit.setPlaceholderText("Enter search terms here")
        self.phenocam_search_edit.setClearButtonEnabled(True)
        pal = self.phenocam_search_edit.palette()
        if hasattr(QtGui.QPalette, "PlaceholderText"):
            pal.setColor(QtGui.QPalette.PlaceholderText,
                         QtGui.QColor(_phenocam_setting("Phenocam_Placeholder_Color")))
            self.phenocam_search_edit.setPalette(pal)
        self.phenocam_search_btn = QtWidgets.QPushButton("Search")
        self.phenocam_search_btn.setStyleSheet(self._pc_ghost_btn_style)
        search_row.addWidget(self.phenocam_search_edit, 1)
        search_row.addWidget(self.phenocam_search_btn, 0)
        left.insertLayout(tree_idx, search_row)

        self.phenocam_search_error = QtWidgets.QLabel("")
        self.phenocam_search_error.setStyleSheet(
            f"QLabel {{ color: {_phenocam_setting('Phenocam_Error_Color')}; }}")
        self.phenocam_search_error.setVisible(False)
        left.insertWidget(tree_idx + 1, self.phenocam_search_error)

        # Footer: site count on one line, the inactive toggle below it (the list is too
        # narrow for both on one line without the count wrapping).
        footer = QtWidgets.QVBoxLayout()
        footer.setSpacing(2)
        self.phenocam_count_label = QtWidgets.QLabel("")
        self.phenocam_count_label.setWordWrap(True)
        self.phenocam_inactive_btn = QtWidgets.QPushButton()
        self.phenocam_inactive_btn.setStyleSheet(self._pc_ghost_btn_style)
        self.phenocam_inactive_btn.clicked.connect(self._phenocam_toggle_inactive)
        footer.addWidget(self.phenocam_count_label)
        footer.addWidget(self.phenocam_inactive_btn, 0, QtCore.Qt.AlignRight)
        left.insertLayout(left.indexOf(self.treeWidget_Phenocam) + 1, footer)

        self._pc_show_inactive = bool(_phenocam_setting("Phenocam_Show_Inactive"))
        self._pc_query = PhenocamSearchQuery("")
        self._pc_search_timer = QtCore.QTimer(self)
        self._pc_search_timer.setSingleShot(True)
        self._pc_search_timer.setInterval(int(_phenocam_setting("Phenocam_Search_Debounce_ms")))
        self._pc_search_timer.timeout.connect(self._phenocam_run_search)
        self.phenocam_search_edit.textChanged.connect(lambda _t: self._pc_search_timer.start())
        self.phenocam_search_edit.returnPressed.connect(self._phenocam_run_search)
        self.phenocam_search_btn.clicked.connect(self._phenocam_run_search)
        self._pc_search_help = PhenocamSearchHelpFilter(self.phenocam_search_edit, parent=self)

        # ── RIGHT: clear verticalLayout_PhenocamRight and rebuild ───────────────
        rl = self.verticalLayout_PhenocamRight
        while rl.count():
            item = rl.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.setSpacing(6)

        # Preview (top of a vertical splitter). It takes Phenocam_Preview_Fraction of the
        # height until the user drags the splitter; the dragged height is then kept.
        self.phenocam_preview_label = self.labelPhenocamImage
        self.phenocam_preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.phenocam_preview_label.setStyleSheet("QLabel { background: transparent; color: palette(text); }")
        self.phenocam_preview_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        # Zoomable panel around the label (label shows messages; the panel shows the image)
        self.phenocam_image_panel = ZoomImagePanel(self.phenocam_preview_label)
        self.phenocam_image_panel.setMinimumHeight(int(_phenocam_setting("Phenocam_Preview_Min_Height_px")))
        self.phenocam_image_panel.showMessage("Select a site to preview the latest image")
        self._phenocam_pixmap = None

        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(6)

        # Download table: row 0 sets the range for all checked sites; one row per checked site
        self.phenocam_table = QtWidgets.QTableWidget(0, 7)
        self.phenocam_table.setHorizontalHeaderLabels(
            ["Site", "First Date", "Last Date", "Start Date", "End Date", "Time Window", "Image Count"])
        hdr = self.phenocam_table.horizontalHeader()
        hdr_font = QFont()
        hdr_font.setBold(True)
        hdr.setFont(hdr_font)
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setStyleSheet(_data_table_header_style())
        self.phenocam_table.verticalHeader().setVisible(False)
        self.phenocam_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.phenocam_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        bl.addWidget(self.phenocam_table, 0)
        self._pc_bottom_panel = bottom   # its height follows the table (see _phenocam_size_table)
        self._pc_rows = {}          # site -> (start_edit, end_edit, time_start, time_end)
        self._pc_daily = {}         # site -> {ISO date: rgb_count} from /api/dailycounts/, or None on error
        self._pc_day_cache = {}     # (site, ISO date) -> [(h, m, s), ...] image times from a browse page
        self._pc_day_failed = set() # (site, ISO date) browse pages that could not be read
        self._pc_daily_worker = None
        self._pc_count_worker = None
        self._pc_count_timer = QtCore.QTimer(self)
        self._pc_count_timer.setSingleShot(True)
        self._pc_count_timer.setInterval(int(_phenocam_setting("Phenocam_Count_Debounce_ms")))
        self._pc_count_timer.timeout.connect(self._phenocam_start_count)
        self._pc_all_row = None
        self._phenocam_add_all_row()
        self._phenocam_size_table()

        # Folder row: [path][Browse][Download Images]
        folder_row = QtWidgets.QHBoxLayout()
        folder_row.setSpacing(4)
        self.phenocam_folder_edit = QtWidgets.QLineEdit()
        self.phenocam_folder_edit.setPlaceholderText("Select output folder...")
        try:
            saved = JsonEditor().getValue("Phenocam_Root_Folder")
            if saved:
                self.phenocam_folder_edit.setText(saved)
        except Exception:
            pass
        self.phenocam_browse_btn = QtWidgets.QPushButton("Browse...")
        self.phenocam_browse_btn.setStyleSheet(self._pc_ghost_btn_style)
        self.phenocam_browse_btn.clicked.connect(self._phenocam_browse_folder)
        self.phenocam_download_btn = QtWidgets.QPushButton("Download Images")
        self.phenocam_download_btn.setStyleSheet(self._pc_primary_btn_style)
        self.phenocam_download_btn.clicked.connect(self._phenocam_download_clicked)
        folder_row.addWidget(self.phenocam_folder_edit, 1)
        folder_row.addWidget(self.phenocam_browse_btn, 0)
        folder_row.addWidget(self.phenocam_download_btn, 0)
        bl.addLayout(folder_row)

        # Progress + status
        self.phenocam_progress_bar = QtWidgets.QProgressBar()
        self.phenocam_progress_bar.setRange(0, 100)
        self.phenocam_progress_bar.setFixedHeight(16)
        self.phenocam_progress_bar.setVisible(False)
        bl.addWidget(self.phenocam_progress_bar)
        self.phenocam_status_label = QtWidgets.QLabel("")
        self.phenocam_status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.phenocam_status_label.setVisible(False)
        bl.addWidget(self.phenocam_status_label)

        self.phenocam_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.phenocam_splitter.setChildrenCollapsible(False)
        self.phenocam_splitter.addWidget(self.phenocam_image_panel)
        self.phenocam_splitter.addWidget(bottom)
        self.phenocam_splitter.setStretchFactor(0, 0)
        self.phenocam_splitter.setStretchFactor(1, 1)
        self.phenocam_splitter.splitterMoved.connect(self._phenocam_splitter_moved)
        saved_h = _phenocam_setting("Phenocam_Preview_Height_px")
        self._pc_preview_user_height = int(saved_h) if saved_h else None
        rl.addWidget(self.phenocam_splitter, 1)

        # ── Resizable left panel: move both columns into a horizontal splitter ──
        # Each column becomes a widget, so the search row and footer buttons are
        # bounded by the panel width and shrink with it.
        hl = self.horizontalLayout_Phenocam
        panels = []
        for lay in (self.verticalLayout_PhenocamLeft, self.verticalLayout_PhenocamRight):
            hl.removeItem(lay)
            lay.setParent(None)
            panel = QtWidgets.QWidget()
            panel.setLayout(lay)
            panels.append(panel)
        self.phenocam_left_panel, self.phenocam_right_panel = panels
        self.phenocam_h_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.phenocam_h_splitter.setChildrenCollapsible(False)
        self.phenocam_h_splitter.addWidget(self.phenocam_left_panel)
        self.phenocam_h_splitter.addWidget(self.phenocam_right_panel)
        self.phenocam_h_splitter.setStretchFactor(0, 0)
        self.phenocam_h_splitter.setStretchFactor(1, 1)
        hl.addWidget(self.phenocam_h_splitter)
        left_w = int(_phenocam_setting("Phenocam_Left_Panel_Width_px"))
        self.phenocam_h_splitter.setSizes([left_w, max(self.width() - left_w, left_w)])
        self.phenocam_h_splitter.splitterMoved.connect(self._phenocam_h_splitter_moved)

        # Connect tree selection
        self.treeWidget_Phenocam.currentItemChanged.connect(self._phenocam_site_selected)
        # Open hyperlink items in the browser on click
        self.treeWidget_Phenocam.itemClicked.connect(self._phenocam_tree_item_clicked)
        # Checking a site adds it to the download table
        self.treeWidget_Phenocam.itemChanged.connect(self._phenocam_item_changed)
        self._phenocam_worker          = None
        self._phenocam_preview_fetcher = None
        self._pc_queue                 = []
        self._pc_queue_total           = 0
        self._pc_downloaded            = 0

        self._pc_restoring = False
        self._pc_save_timer = QtCore.QTimer(self)
        self._pc_save_timer.setSingleShot(True)
        self._pc_save_timer.setInterval(int(_phenocam_setting("Phenocam_Search_Debounce_ms")))
        self._pc_save_timer.timeout.connect(self._phenocam_save_state)

        self._phenocam_rebuild_table()
        self._phenocam_apply_filter()
        self._phenocam_restore_state()

    # ------------------------------------------------------------------------------------------------------------------
    # PHENOCAM SEARCH AND INACTIVE FILTER
    # ------------------------------------------------------------------------------------------------------------------
    def _phenocam_run_search(self):
        self._pc_search_timer.stop()
        self._pc_query = PhenocamSearchQuery(self.phenocam_search_edit.text())
        self.phenocam_search_error.setText(self._pc_query.error)
        self.phenocam_search_error.setVisible(bool(self._pc_query.error))
        self._phenocam_apply_filter()

    def _phenocam_toggle_inactive(self):
        self._pc_show_inactive = not self._pc_show_inactive
        try:
            JsonEditor().update_json_entry("Phenocam_Show_Inactive", str(self._pc_show_inactive))
        except Exception:
            pass
        self._phenocam_apply_filter()

    def _phenocam_apply_filter(self):
        if not hasattr(self, "phenocam_count_label"):
            return
        records = getattr(self, "_phenocam_records", {})
        query = getattr(self, "_pc_query", None)
        tree = self.treeWidget_Phenocam
        total = shown = inactive_hidden = checked = 0
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            site = item.data(0, QtCore.Qt.UserRole)
            rec = records.get(site)
            total += 1
            if item.checkState(0) == QtCore.Qt.Checked:
                checked += 1
            visible = True
            if rec is not None and not rec["active"] and not self._pc_show_inactive:
                visible = False
                inactive_hidden += 1
            if visible and query is not None and rec is not None and not query.matches(rec):
                visible = False
            item.setHidden(not visible)
            shown += visible
        parts = [f"{shown:,} of {total:,} sites shown" if shown != total else f"{total:,} sites"]
        if inactive_hidden:
            parts.append(f"{inactive_hidden:,} inactive hidden")
        parts.append(f"{checked:,} checked")
        self.phenocam_count_label.setText(", ".join(parts))
        self.phenocam_inactive_btn.setText(
            "Hide inactive sites" if self._pc_show_inactive else "Show inactive sites")

    # ------------------------------------------------------------------------------------------------------------------
    # PHENOCAM DOWNLOAD TABLE
    # ------------------------------------------------------------------------------------------------------------------
    def _phenocam_range_widgets(self, start_qdate, end_qdate, t_start, t_end):
        start_edit = QtWidgets.QDateEdit(start_qdate)
        end_edit = QtWidgets.QDateEdit(end_qdate)
        for e in (start_edit, end_edit):
            e.setCalendarPopup(True)
            e.setDisplayFormat("yyyy-MM-dd")
            e.setFrame(False)
        time_cell = QtWidgets.QWidget()
        tl = QtWidgets.QHBoxLayout(time_cell)
        tl.setContentsMargins(2, 0, 2, 0)
        tl.setSpacing(4)
        time_start = QtWidgets.QTimeEdit(t_start)
        time_end = QtWidgets.QTimeEdit(t_end)
        for e in (time_start, time_end):
            e.setDisplayFormat("HH:mm")
            e.setFrame(False)
        tl.addWidget(time_start)
        tl.addWidget(QtWidgets.QLabel("to"))
        tl.addWidget(time_end)
        for e in (start_edit, end_edit):
            e.dateChanged.connect(lambda _d: self._phenocam_schedule_save())
            e.dateChanged.connect(lambda _d: self._phenocam_schedule_count())
        for e in (time_start, time_end):
            e.timeChanged.connect(lambda _t: self._phenocam_schedule_save())
            e.timeChanged.connect(lambda _t: self._phenocam_schedule_count())  # cached days recount instantly
        return start_edit, end_edit, time_cell, time_start, time_end

    def _phenocam_add_all_row(self):
        t = self.phenocam_table
        t.insertRow(0)
        today = QtCore.QDate.currentDate()
        days = int(_phenocam_setting("Phenocam_Default_Range_Days"))
        start_edit, end_edit, time_cell, ts, te = self._phenocam_range_widgets(
            today.addDays(-days), today, QtCore.QTime(0, 0), QtCore.QTime(23, 59))
        label = QTableWidgetItem("All checked sites")
        f = label.font()
        f.setBold(True)
        label.setFont(f)
        t.setItem(0, 0, label)
        t.setItem(0, 1, QTableWidgetItem(""))
        t.setItem(0, 2, QTableWidgetItem(""))
        t.setCellWidget(0, 3, start_edit)
        t.setCellWidget(0, 4, end_edit)
        t.setCellWidget(0, 5, time_cell)
        t.setItem(0, 6, self._phenocam_count_item("", bold=True))
        self._pc_all_row = (start_edit, end_edit, ts, te)
        for e in (start_edit, end_edit):
            e.dateChanged.connect(lambda _d: self._phenocam_apply_all_row())
        for e in (ts, te):
            e.timeChanged.connect(lambda _t: self._phenocam_apply_all_row())

    def _phenocam_fit_range(self, site, start_q, end_q):
        """Shift a date range to fall inside the site's image record, keeping its length."""
        rec = getattr(self, "_phenocam_records", {}).get(site, {})
        first = QtCore.QDate.fromString(rec.get("dates", {}).get("first", ""), "yyyy-MM-dd")
        last = QtCore.QDate.fromString(rec.get("dates", {}).get("last", ""), "yyyy-MM-dd")
        span = start_q.daysTo(end_q)
        # Trim the range to the site's record where they overlap.
        if first.isValid() and start_q < first:
            start_q = first
        if last.isValid() and end_q > last:
            end_q = last
        # No overlap: keep the range length, placed at the nearer end of the record.
        if end_q < start_q:
            if last.isValid() and start_q > last:
                end_q = last
                start_q = end_q.addDays(-span)
                if first.isValid() and start_q < first:
                    start_q = first
            else:
                start_q = first
                end_q = start_q.addDays(span)
                if last.isValid() and end_q > last:
                    end_q = last
        return start_q, end_q

    def _phenocam_apply_all_row(self):
        a_start, a_end, a_ts, a_te = self._pc_all_row
        for site, (s, e, ts, te) in self._pc_rows.items():
            sq, eq = self._phenocam_fit_range(site, a_start.date(), a_end.date())
            s.setDate(sq)
            e.setDate(eq)
            ts.setTime(a_ts.time())
            te.setTime(a_te.time())

    def _phenocam_add_site_row(self, site):
        if site in self._pc_rows:
            return
        t = self.phenocam_table
        row = t.rowCount()
        t.insertRow(row)
        rec = getattr(self, "_phenocam_records", {}).get(site, {})
        a_start, a_end, a_ts, a_te = self._pc_all_row
        sq, eq = self._phenocam_fit_range(site, a_start.date(), a_end.date())
        start_edit, end_edit, time_cell, ts, te = self._phenocam_range_widgets(sq, eq, a_ts.time(), a_te.time())
        t.setItem(row, 0, QTableWidgetItem(site))
        t.setItem(row, 1, QTableWidgetItem(rec.get("dates", {}).get("first", "")))
        t.setItem(row, 2, QTableWidgetItem(rec.get("dates", {}).get("last", "")))
        t.setCellWidget(row, 3, start_edit)
        t.setCellWidget(row, 4, end_edit)
        t.setCellWidget(row, 5, time_cell)
        t.setItem(row, 6, self._phenocam_count_item(""))
        self._pc_rows[site] = (start_edit, end_edit, ts, te)
        self._phenocam_size_table()
        self._phenocam_schedule_count()

    # ------------------------------------------------------------------------------------------------------------------
    # PHENOCAM IMAGE COUNTS
    # Counts come from PhenoCam's daily browse pages (image timestamps only, nothing is
    # downloaded). Each day's times are cached per site, so changing the time window
    # recounts instantly and changing dates only fetches days not seen before.
    # ------------------------------------------------------------------------------------------------------------------
    def _phenocam_count_item(self, text, bold=False):
        item = QTableWidgetItem(text)
        item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        if bold:
            f = item.font()
            f.setBold(True)
            item.setFont(f)
        return item

    def _phenocam_row_days(self, site):
        """Days in the site's row range, trimmed to the site's image record."""
        import datetime as dt_mod
        s, e, _ts, _te = self._pc_rows[site]
        start = s.date().toPyDate()
        end = e.date().toPyDate()
        rec = getattr(self, "_phenocam_records", {}).get(site, {}).get("dates", {})
        try:
            if rec.get("first"):
                start = max(start, dt_mod.date.fromisoformat(rec["first"]))
            if rec.get("last"):
                end = min(end, dt_mod.date.fromisoformat(rec["last"]))
        except ValueError:
            pass
        n = (end - start).days + 1
        return [start + dt_mod.timedelta(days=i) for i in range(max(n, 0))]

    @staticmethod
    def _phenocam_full_day(t_start, t_end):
        """True if the window covers the whole day (minute resolution, as the downloader uses)."""
        import datetime as dt_mod
        return t_start == dt_mod.time(0, 0) and t_end in (dt_mod.time(0, 0), dt_mod.time(23, 59))

    def _phenocam_pages_needed(self, site, days, t_start, t_end):
        """
        Days whose browse page must be read to count this row. A full-day window
        needs none where daily counts exist. A narrow window needs only days the
        daily counts say have images. Days missing from the daily counts (not yet
        tallied by PhenoCam, or the daily counts failed) always need their page.
        """
        daily = self._pc_daily.get(site) or {}
        full = self._phenocam_full_day(t_start, t_end)
        needed = []
        for d in days:
            iso = d.isoformat()
            if iso in daily:
                if full or daily[iso] == 0:
                    continue
            needed.append(d)
        return needed

    def _phenocam_schedule_count(self):
        if hasattr(self, "_pc_count_timer"):
            self._phenocam_update_counts()
            self._pc_count_timer.start()

    def _phenocam_start_count(self):
        # Stage 1: daily counts for any checked site that doesn't have them yet.
        missing = [site for site in self._pc_rows if site not in self._pc_daily]
        if missing:
            if self._pc_daily_worker is not None and self._pc_daily_worker.isRunning():
                return   # its done signal restarts the count
            worker = PhenocamDailyCountsWorker(missing, parent=self)
            worker.siteDone.connect(self._phenocam_daily_done)
            worker.done.connect(lambda w=worker: self._phenocam_daily_finished(w))
            self._pc_daily_worker = worker
            worker.start()
            return

        # Stage 2: browse pages, only where the daily counts can't answer.
        needed = []
        for site, (_s, _e, ts, te) in self._pc_rows.items():
            days = self._phenocam_row_days(site)
            pages = [d for d in self._phenocam_pages_needed(site, days, ts.time().toPyTime(), te.time().toPyTime())
                     if (site, d.isoformat()) not in self._pc_day_cache]
            if len(pages) <= int(_phenocam_setting("Phenocam_Count_Max_Days")):
                needed += [(site, d) for d in pages]
        if self._pc_count_worker is not None:
            self._pc_count_worker.cancel()
            self._pc_count_worker = None
        self._pc_day_failed.clear()
        if not needed:
            self._phenocam_update_counts()
            return
        worker = PhenocamDayPagesWorker(needed, int(_phenocam_setting("Phenocam_Count_Parallel_Pages")), parent=self)
        worker.dayDone.connect(self._phenocam_count_day_done)
        worker.done.connect(lambda w=worker: self._phenocam_count_finished(w))
        self._pc_count_worker = worker
        worker.start()

    def _phenocam_daily_done(self, site, counts):
        self._pc_daily[site] = counts

    def _phenocam_daily_finished(self, worker):
        if self._pc_daily_worker is worker:
            self._pc_daily_worker = None
        self._phenocam_start_count()

    def _phenocam_count_day_done(self, site, iso_day, times):
        if times is None:
            self._pc_day_failed.add((site, iso_day))
        else:
            self._pc_day_cache[(site, iso_day)] = times
        self._phenocam_update_counts()

    def _phenocam_count_finished(self, worker):
        if self._pc_count_worker is worker:
            self._pc_count_worker = None
        self._phenocam_update_counts()

    @staticmethod
    def _phenocam_in_window(hms, t_start, t_end):
        """Same rule as PhenoCam.getVisibleImages: minute resolution; 00:00 to 00:00 means all day."""
        import datetime as dt_mod
        if t_start == dt_mod.time(0, 0) and t_end == dt_mod.time(0, 0):
            return True
        t = dt_mod.time(hms[0], hms[1], 0)
        return t_start <= t <= t_end

    def _phenocam_row_count(self, site):
        """
        (count, status) for a row. status is "ok", "pending" (still being read),
        "unavailable" (a needed page or the daily counts failed), or "too_long".
        """
        _s, _e, ts, te = self._pc_rows[site]
        t_start, t_end = ts.time().toPyTime(), te.time().toPyTime()
        if site not in self._pc_daily:
            return None, "pending"
        days = self._phenocam_row_days(site)
        daily = self._pc_daily.get(site) or {}
        pages = set(self._phenocam_pages_needed(site, days, t_start, t_end))
        if len(pages) > int(_phenocam_setting("Phenocam_Count_Max_Days")):
            return None, "too_long"
        n = 0
        for d in days:
            iso = d.isoformat()
            if d not in pages:
                n += daily.get(iso, 0)          # full day from daily counts, or a known empty day
                continue
            times = self._pc_day_cache.get((site, iso))
            if times is None:
                return None, "unavailable" if (site, iso) in self._pc_day_failed else "pending"
            n += sum(1 for hms in times if self._phenocam_in_window(hms, t_start, t_end))
        return n, "ok"

    def _phenocam_update_counts(self):
        """Counts appear only when final; a row shows "Computing..." while it is being worked out."""
        if not hasattr(self, "phenocam_table"):
            return
        t = self.phenocam_table
        total, all_ok, any_pending = 0, True, False
        max_days = int(_phenocam_setting("Phenocam_Count_Max_Days"))
        for r in range(1, t.rowCount()):
            site = t.item(r, 0).text()
            n, status = self._phenocam_row_count(site)
            if status == "ok":
                text = f"{n:,}"
                total += n
            else:
                all_ok = False
                any_pending = any_pending or status == "pending"
                text = {"pending": "Computing...",
                        "unavailable": "Unavailable",
                        "too_long": f"Over {max_days} days"}[status]
            t.item(r, 6).setText(text)
        all_item = t.item(0, 6)
        if all_item is not None:
            if t.rowCount() == 1:
                all_item.setText("")
            else:
                # "+" marks a total missing rows that couldn't be counted
                all_item.setText("Computing..." if any_pending else f"{total:,}" + ("" if all_ok else "+"))
        t.resizeColumnToContents(6)

    def _phenocam_size_table(self):
        """
        Size the table to its rows: the "All checked sites" row plus one row per
        checked site, with Phenocam_Table_Blank_Rows empty rows shown when fewer
        sites are checked, and scrolling beyond Phenocam_Table_Max_Rows. The panel
        below the preview is capped at what it needs, so the preview gets the rest.
        """
        if not hasattr(self, "phenocam_table"):
            return
        t = self.phenocam_table
        n_sites = t.rowCount() - 1
        blank = int(_phenocam_setting("Phenocam_Table_Blank_Rows"))
        max_rows = int(_phenocam_setting("Phenocam_Table_Max_Rows"))
        slots = max(n_sites, blank)
        default_h = t.verticalHeader().defaultSectionSize()
        shown_rows = min(slots, max_rows) + 1            # + the "All checked sites" row
        h = t.horizontalHeader().height() or t.horizontalHeader().sizeHint().height()
        for r in range(shown_rows):
            h += t.rowHeight(r) if r < t.rowCount() else default_h
        h += 2 * t.frameWidth()
        if t.horizontalScrollBar().maximum() > 0:
            h += t.horizontalScrollBar().sizeHint().height()
        t.setFixedHeight(h)
        self._phenocam_size_bottom_panel()

    def _phenocam_size_bottom_panel(self):
        """Cap the panel under the preview at the height its contents need."""
        bottom = getattr(self, "_pc_bottom_panel", None)
        if bottom is None:
            return
        bottom.setMaximumHeight(bottom.sizeHint().height())
        self._phenocam_fit_preview_height()

    def _phenocam_remove_site_row(self, site):
        if site not in self._pc_rows:
            return
        t = self.phenocam_table
        for r in range(1, t.rowCount()):
            it = t.item(r, 0)
            if it is not None and it.text() == site:
                t.removeRow(r)
                break
        del self._pc_rows[site]
        self._phenocam_size_table()
        self._phenocam_update_counts()

    def _phenocam_rebuild_table(self):
        """Sync the table with the checked sites in the tree (used after the tree is rebuilt)."""
        if not hasattr(self, "phenocam_table"):
            return
        checked = []
        tree = self.treeWidget_Phenocam
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            if item.checkState(0) == QtCore.Qt.Checked:
                checked.append(item.data(0, QtCore.Qt.UserRole))
        for site in list(self._pc_rows):
            if site not in checked:
                self._phenocam_remove_site_row(site)
        for site in checked:
            self._phenocam_add_site_row(site)

    def _phenocam_item_changed(self, item, column):
        if column != 0 or item.parent() is not None:
            return
        site = item.data(0, QtCore.Qt.UserRole)
        if item.checkState(0) == QtCore.Qt.Checked:
            self._phenocam_add_site_row(site)
        else:
            self._phenocam_remove_site_row(site)
        self._phenocam_apply_filter()
        self._phenocam_schedule_save()

    # ------------------------------------------------------------------------------------------------------------------
    # PHENOCAM SAVED STATE
    # Stored in the settings file as "Phenocam_Selections" (a JSON string):
    #   {"all_sites_range": {"start", "end", "time_start", "time_end"},
    #    "sites": [{"site", "start", "end", "time_start", "time_end"}, ...]}   (table order)
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _phenocam_range_to_dict(widgets):
        s, e, ts, te = widgets
        return {"start": s.date().toString("yyyy-MM-dd"), "end": e.date().toString("yyyy-MM-dd"),
                "time_start": ts.time().toString("HH:mm"), "time_end": te.time().toString("HH:mm")}

    @staticmethod
    def _phenocam_range_from_dict(widgets, d):
        s, e, ts, te = widgets
        for w, key in ((s, "start"), (e, "end")):
            q = QtCore.QDate.fromString(str(d.get(key, "")), "yyyy-MM-dd")
            if q.isValid():
                w.setDate(q)
        for w, key in ((ts, "time_start"), (te, "time_end")):
            q = QtCore.QTime.fromString(str(d.get(key, "")), "HH:mm")
            if q.isValid():
                w.setTime(q)

    def _phenocam_schedule_save(self):
        if getattr(self, "_pc_restoring", True) or not hasattr(self, "_pc_save_timer"):
            return
        self._pc_save_timer.start()

    def _phenocam_save_state(self):
        import json
        t = self.phenocam_table
        sites = []
        for r in range(1, t.rowCount()):
            site = t.item(r, 0).text()
            sites.append({"site": site, **self._phenocam_range_to_dict(self._pc_rows[site])})
        state = {"all_sites_range": self._phenocam_range_to_dict(self._pc_all_row),
                 "sites": sites}
        try:
            JsonEditor().update_json_entry("Phenocam_Selections", json.dumps(state))
        except Exception as e:
            print(f"[PhenoCam] Could not save selections: {e}")

    def _phenocam_restore_state(self):
        import json
        try:
            raw = JsonEditor().getValue("Phenocam_Selections")
            state = json.loads(raw) if isinstance(raw, str) and raw else (raw if isinstance(raw, dict) else None)
        except Exception as e:
            print(f"[PhenoCam] Could not read saved selections: {e}")
            state = None
        self._pc_restoring = True
        try:
            if state:
                # "All checked sites" row first, without pushing its range onto site rows
                for w in self._pc_all_row:
                    w.blockSignals(True)
                self._phenocam_range_from_dict(self._pc_all_row, state.get("all_sites_range", {}))
                for w in self._pc_all_row:
                    w.blockSignals(False)

                # Re-check saved sites that still exist, then restore each row's own range
                tree = self.treeWidget_Phenocam
                items = {tree.topLevelItem(i).data(0, QtCore.Qt.UserRole): tree.topLevelItem(i)
                         for i in range(tree.topLevelItemCount())}
                skipped = []
                for entry in state.get("sites", []):
                    site = entry.get("site")
                    item = items.get(site)
                    if item is None:
                        skipped.append(site)
                        continue
                    item.setCheckState(0, QtCore.Qt.Checked)   # adds the table row
                    if site in self._pc_rows:
                        self._phenocam_range_from_dict(self._pc_rows[site], entry)
                if skipped:
                    print(f"[PhenoCam] Saved sites no longer listed by PhenoCam: {', '.join(map(str, skipped))}")
                # The search filter is deliberately not restored: each session starts with the full list.
        finally:
            self._pc_restoring = False

    # ------------------------------------------------------------------------------------------------------------------
    # PHENOCAM PREVIEW SIZING
    # ------------------------------------------------------------------------------------------------------------------
    def _phenocam_splitter_moved(self, pos, index):
        self._pc_preview_user_height = self.phenocam_splitter.sizes()[0]
        self._phenocam_scale_preview()
        # Save once the drag pauses rather than on every pixel of movement.
        if not hasattr(self, "_pc_splitter_save_timer"):
            self._pc_splitter_save_timer = QtCore.QTimer(self)
            self._pc_splitter_save_timer.setSingleShot(True)
            self._pc_splitter_save_timer.setInterval(int(_phenocam_setting("Phenocam_Search_Debounce_ms")))
            self._pc_splitter_save_timer.timeout.connect(self._phenocam_save_preview_height)
        self._pc_splitter_save_timer.start()

    def _phenocam_h_splitter_moved(self, pos, index):
        """Save the left panel width once the drag pauses."""
        if not hasattr(self, "_pc_hsplit_save_timer"):
            self._pc_hsplit_save_timer = QtCore.QTimer(self)
            self._pc_hsplit_save_timer.setSingleShot(True)
            self._pc_hsplit_save_timer.setInterval(int(_phenocam_setting("Phenocam_Search_Debounce_ms")))
            self._pc_hsplit_save_timer.timeout.connect(
                lambda: JsonEditor().update_json_entry(
                    "Phenocam_Left_Panel_Width_px", str(self.phenocam_h_splitter.sizes()[0])))
        self._pc_hsplit_save_timer.start()

    def _phenocam_save_preview_height(self):
        try:
            JsonEditor().update_json_entry("Phenocam_Preview_Height_px", str(self._pc_preview_user_height))
        except Exception:
            pass

    def _phenocam_scale_preview(self):
        if self._phenocam_pixmap is None:
            return
        self.phenocam_image_panel.setPixmap(self._phenocam_pixmap)   # keeps the user's zoom for the same image

    def _phenocam_fit_preview_height(self):
        if not hasattr(self, 'phenocam_splitter'):
            return
        sp = self.phenocam_splitter
        total = sum(sp.sizes())
        if total < 50:
            return
        bottom_min = sp.widget(1).minimumSizeHint().height()
        bottom_max = sp.widget(1).maximumHeight()
        lbl_min = self.phenocam_image_panel.minimumHeight()
        if self._pc_preview_user_height:
            h = self._pc_preview_user_height
        else:
            h = int(total * float(_phenocam_setting("Phenocam_Preview_Fraction")))
        h = max(lbl_min, min(h, total - bottom_min), total - bottom_max)
        sp.setSizes([h, total - h])
        self._phenocam_scale_preview()

    # ------------------------------------------------------------------------------------------------------------------
    def _phenocam_get_start_datetime(self):
        """Start of the range in the 'All checked sites' row."""
        import datetime as dt_mod
        s, _e, ts, _te = self._pc_all_row
        d, t = s.date(), ts.time()
        return dt_mod.datetime(d.year(), d.month(), d.day(), t.hour(), t.minute())

    def _phenocam_get_end_datetime(self):
        """End of the range in the 'All checked sites' row."""
        import datetime as dt_mod
        _s, e, _ts, te = self._pc_all_row
        d, t = e.date(), te.time()
        return dt_mod.datetime(d.year(), d.month(), d.day(), t.hour(), t.minute())

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
        import datetime as dt_mod
        label = roi_name if roi_name else site_name
        self._phenocam_pixmap = None
        self.phenocam_image_panel.showMessage(f"Loading latest image for {label}...")

        # Stop any preview search still running for a previously clicked site.
        previous = getattr(self, "_phenocam_preview_fetcher", None)
        if previous is not None:
            previous.cancel()

        last_date = None
        last_text = getattr(self, "_phenocam_records", {}).get(site_name, {}).get("dates", {}).get("last", "")
        try:
            last_date = dt_mod.date.fromisoformat(last_text) if last_text else None
        except ValueError:
            last_date = None

        fetcher = PhenocamPreviewFetcher(site_name, self, roi_name=roi_name, last_date=last_date,
                                         search_days=int(_phenocam_setting("Phenocam_Preview_Search_Days")))

        def _on_result(pixmap, err):
            if self._phenocam_preview_fetcher is not fetcher:
                return   # result from a site the user has since moved away from
            if pixmap is None:
                self._phenocam_pixmap = None
                self.phenocam_image_panel.showMessage("No image available.")
                if err and err != "No image available.":
                    print(f"[PhenoCam] Preview for {label}: {err}")
            else:
                self._phenocam_pixmap = pixmap
                self._phenocam_fit_preview_height()
            self._phenocam_preview_fetcher = None

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

    def _phenocam_row_datetimes(self, widgets):
        import datetime as dt_mod
        s, e, ts, te = widgets
        sd, ed, st, et = s.date(), e.date(), ts.time(), te.time()
        return (dt_mod.datetime(sd.year(), sd.month(), sd.day(), st.hour(), st.minute()),
                dt_mod.datetime(ed.year(), ed.month(), ed.day(), et.hour(), et.minute()))

    def _phenocam_download_clicked(self):
        # If a PhenoCam download is already running, use the same button to cancel it.
        if (
                hasattr(self, "_phenocam_worker")
                and self._phenocam_worker is not None
                and self._phenocam_worker.isRunning()
        ):
            self._pc_queue = []
            self._phenocam_worker.cancel()
            self.phenocam_status_label.setText("Cancelling download...")
            self.phenocam_download_btn.setEnabled(False)
            return

        # Checked sites (table order); if none are checked, the highlighted site
        # with the "All checked sites" range, as before.
        jobs = []
        t = self.phenocam_table
        for r in range(1, t.rowCount()):
            site = t.item(r, 0).text()
            jobs.append((site, *self._phenocam_row_datetimes(self._pc_rows[site])))
        if not jobs:
            site_name = self._phenocam_get_selected_sitename()
            if not site_name:
                App_QMessageBox("PhenoCam Download",
                                "Please check one or more sites, or select a site.").displayMsgBox()
                return
            jobs.append((site_name, self._phenocam_get_start_datetime(), self._phenocam_get_end_datetime()))

        save_folder = self.phenocam_folder_edit.text().strip()
        if not save_folder:
            App_QMessageBox("PhenoCam Download",
                                 "Please specify a download folder.").displayMsgBox()
            return

        bad = [site for site, start_dt, end_dt in jobs if start_dt >= end_dt]
        if bad:
            App_QMessageBox("PhenoCam Download",
                            "Start must be before End for: " + ", ".join(bad)).displayMsgBox()
            return

        try:
            JsonEditor().update_json_entry("Phenocam_Root_Folder", save_folder)
        except Exception:
            pass

        # One site downloads into the folder itself, as before; several sites
        # each get a subfolder named for the site.
        multi = len(jobs) > 1
        self._pc_queue = [(site, s, e, os.path.join(save_folder, site) if multi else save_folder)
                          for site, s, e in jobs]
        self._pc_queue_total = len(self._pc_queue)
        self._pc_downloaded = 0

        self.phenocam_download_btn.setText("Cancel Download")
        self.phenocam_download_btn.setEnabled(True)
        self.phenocam_progress_bar.setValue(0)
        self.phenocam_progress_bar.setVisible(True)
        self.phenocam_status_label.setVisible(True)
        self._phenocam_size_bottom_panel()
        self.phenocam_status_label.setText("Starting...")
        self._phenocam_start_next_job()

    def _phenocam_start_next_job(self):
        site_name, start_dt, end_dt, folder = self._pc_queue.pop(0)
        self._pc_current_site = site_name
        worker = PhenocamDownloadWorker(
            site_name, start_dt, end_dt, folder, parent=self
        )
        worker.progress.connect(self._phenocam_download_progress)
        worker.finished.connect(self._phenocam_download_finished)
        self._phenocam_worker = worker
        worker.start()

    def _phenocam_download_progress(self, current, total, label):
        if total > 0:
            self.phenocam_progress_bar.setValue(int(current / total * 100))
        if self._pc_queue_total > 1:
            n = self._pc_queue_total - len(self._pc_queue)
            label = f"{self._pc_current_site} (site {n} of {self._pc_queue_total}): {label}"
        self.phenocam_status_label.setText(label)

    def _phenocam_download_finished(self, count):
        if count == -1:
            self._pc_queue = []
            self.phenocam_download_btn.setText("Download Images")
            self.phenocam_download_btn.setEnabled(True)
            self._phenocam_worker = None
            self.phenocam_status_label.setText("Download cancelled.")
            App_QMessageBox("PhenoCam Download", "Download cancelled.").displayMsgBox()
            return

        self._pc_downloaded += max(count, 0)
        if self._pc_queue:
            self._phenocam_start_next_job()
            return

        self.phenocam_download_btn.setText("Download Images")
        self.phenocam_download_btn.setEnabled(True)
        self._phenocam_worker = None
        self.phenocam_progress_bar.setValue(100)

        count = self._pc_downloaded
        sites = f" from {self._pc_queue_total} sites" if self._pc_queue_total > 1 else ""
        msg = (
            f"Download complete. {count} new image(s) saved{sites}."
            if count > 0
            else "No new images found for the selected date range."
        )
        self.phenocam_status_label.setText(msg)
        App_QMessageBox("PhenoCam Download", msg).displayMsgBox()

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
        This function returns the dynamic tooltip string for App_QLabel.
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
    def exportROIMasks(self):
        """Export the current Color Segmentation ROIs to COCO 1.0 as
        ROI_Masks_<YYYYMMDD_HHMMSS>.json in the image folder. ROI names become categories;
        ROI shapes become polygon segmentations; ROIs are applied across
        every image in the folder."""
        global dailyImagesList, imageFileFolder
        from appcore.dialogs.color_segmentation.color_seg_roi_coco_export import export_roi_masks
        try:
            images_list = dailyImagesList.getVisibleList()
        except Exception:
            images_list = []
        if not self.roiList:
            msgBox = App_QMessageBox('Export ROI Masks', 'Draw and add at least one ROI first.', buttons=QMessageBox.Close)
            msgBox.displayMsgBox(); return
        if not images_list:
            msgBox = App_QMessageBox('Export ROI Masks', 'No images in the current folder.', buttons=QMessageBox.Close)
            msgBox.displayMsgBox(); return
        try:
            out_path = export_roi_masks(self.roiList, images_list, imageFileFolder)
        except Exception as e:
            msgBox = App_QMessageBox('Export ROI Masks', f'Export failed: {e}', buttons=QMessageBox.Close)
            msgBox.displayMsgBox(); return
        msgBox = App_QMessageBox('Export ROI Masks', f'Saved COCO 1.0 ROI masks to:\n{out_path}', buttons=QMessageBox.Close)
        msgBox.displayMsgBox()

    def buildFeatureFile(self):
        global dailyImagesList
        from appcore.dialogs.color_segmentation.color_seg_feature_export import ColorSegFeatureExport
        myFeatureExport = ColorSegFeatureExport()
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
            img1 = App_Utils().convertQImageToMat(currentImage.toImage())
            #img1 = App_Utils().convertQImageToMat(myImage.toImage())

            rgb = cv2.blur(img1, ksize=(11, 11))

            # convert image to HSV
            hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)

            if len(self.roiList) > 0:
                # DIAGNOSTICS
                #if self.checkBoxColorDiagnostics.checkState():
                from appcore.Diagnostics import Diagnostics
                Diagnostics.RGB3DPlot(rgb)
                Diagnostics.plotHSVChannelsGray(hsv)
                Diagnostics.plotHSVChannelsColor(hsv)

                # segment colors
                rgb1 = myColor.segmentColors(rgb, hsv, self.roiList)

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
            self.colorSegmentationParams.Texture        = self.colorSegmentationDlg.get_texture_options()['enabled']

            self.colorSegmentationParams.wholeImage     = self.colorSegmentationDlg.checkBoxScalarRegion_WholeImage.isChecked()
            self.colorSegmentationParams.ROI            = self.colorSegmentationDlg.checkBoxScalarRegion_ROI.isChecked()
            self.colorSegmentationParams.HSV            = self.colorSegmentationDlg.checkBoxColor_HSV.isChecked()

            self.colorSegmentationParams.numColorClusters = self.colorSegmentationDlg.get_num_color_clusters()

        self.greenness_index_list.clear()

        if self.colorSegmentationParams.GCC:
            self.greenness_index_list.append(GreennessIndex(Vegetation_Indices.GCC))
        if self.colorSegmentationParams.GLI:
            self.greenness_index_list.append(GreennessIndex(Vegetation_Indices.GLI))
        if self.colorSegmentationParams.NDVI:
            self.greenness_index_list.append(GreennessIndex(Vegetation_Indices.NDVI))
        if self.colorSegmentationParams.ExG:
            self.greenness_index_list.append(GreennessIndex(Vegetation_Indices.ExG))
        if self.colorSegmentationParams.RGI:
            self.greenness_index_list.append(GreennessIndex(Vegetation_Indices.RGI))

        self.initROITable(self.greenness_index_list)

    # ------------------------------------------------------------------------------------------------------------------
    # TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR   TOOLBAR
    # ------------------------------------------------------------------------------------------------------------------
    def createToolBar(self):
        toolbar = QToolBar(f"{APP_NAME} Toolbar")
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
    # ------------------------------------------------------------------------------------------------------------------
    # NEON TAB LAYOUT (built in code; same look as the PhenoCam and USGS tabs)
    #
    #   splitter_NEON_Horizontal
    #   ├── left:  search row, NEON_listboxSites, count line
    #   │          each site: "Site details" branch + "Data products (N)" branch (checkable products)
    #   └── right: splitter_NEON_Vertical
    #       ├── top:    labelLatestImageTitle + NEON_labelLatestImage
    #       └── bottom: NEON_selected_products (one row per checked site/product)
    #                   [folder path] [Browse] [Download]
    #
    # The product list and Sync Dates button from the .ui are hidden: the tree and the
    # "All checked products" row replace them. Existing handlers are unchanged.
    # ------------------------------------------------------------------------------------------------------------------
    def setup_neon_layout(self):
        filled_style, ghost_style = _button_styles()
        self._neon_products = {}        # site code -> {"domain", "state", "domainName", "products": [...]}
        self._neon_rows = {}            # (site, product code) -> row info
        self._neon_all_row = None

        # Left: search, site tree, count
        left = QtWidgets.QWidget()
        left.setMinimumWidth(int(_phenocam_setting("NEON_Left_Panel_Min_Width_px")))
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)
        search_row = QtWidgets.QHBoxLayout()
        search_row.setSpacing(4)
        self.neon_search_edit = QtWidgets.QLineEdit()
        self.neon_search_edit.setPlaceholderText("Enter search terms here")
        self.neon_search_edit.setClearButtonEnabled(True)
        if hasattr(QtGui.QPalette, "PlaceholderText"):
            pal = self.neon_search_edit.palette()
            pal.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(_phenocam_setting("Phenocam_Placeholder_Color")))
            self.neon_search_edit.setPalette(pal)
        self.neon_search_btn = QtWidgets.QPushButton("Search")
        self.neon_search_btn.setStyleSheet(ghost_style)
        search_row.addWidget(self.neon_search_edit, 1)
        search_row.addWidget(self.neon_search_btn, 0)
        ll.addLayout(search_row)
        self.neon_search_error = _search_error_label()
        ll.addWidget(self.neon_search_error)
        self._neon_search_help = SearchHelpFilter(
            self.neon_search_edit,
            _search_help_html(NEON_SEARCH_FIELDS, NEON_SEARCH_EXAMPLES, NEON_SEARCH_EXAMPLE), parent=self)
        ll.addWidget(self.NEON_listboxSites, 1)
        self.NEON_listboxSites.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.neon_count_label = QtWidgets.QLabel("")
        self.neon_count_label.setWordWrap(True)
        ll.addWidget(self.neon_count_label)

        self._neon_search_timer = QtCore.QTimer(self)
        self._neon_search_timer.setSingleShot(True)
        self._neon_search_timer.setInterval(int(_phenocam_setting("Phenocam_Search_Debounce_ms")))
        self._neon_search_timer.timeout.connect(self._neon_apply_filter)
        self.neon_search_edit.textChanged.connect(lambda _t: self._neon_search_timer.start())
        self.neon_search_edit.returnPressed.connect(self._neon_apply_filter)
        self.neon_search_btn.clicked.connect(self._neon_apply_filter)

        # Right top: image title over the image
        preview = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(preview)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(2)
        pl.addWidget(self.labelLatestImageTitle, 0)
        self.NEON_labelLatestImage.setStyleSheet("QLabel { background: transparent; color: palette(text); }")
        self.neon_image_panel = ZoomImagePanel(self.NEON_labelLatestImage)
        pl.addWidget(self.neon_image_panel, 1)
        preview.setMinimumHeight(int(_phenocam_setting("Phenocam_Preview_Min_Height_px")))

        # Right bottom: table, folder row
        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(6)
        bl.addWidget(self.NEON_selected_products, 0)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        self.edit_NEON_TableInput.setPlaceholderText("Select output folder...")
        row.addWidget(self.edit_NEON_TableInput, 1)
        for btn, style in ((self.pushButton_NEON_Browse, ghost_style), (self.pushButton_NEON_Download, filled_style)):
            btn.setStyleSheet(style)
            btn.setMinimumSize(0, 0)
            btn.setMaximumSize(QtWidgets.QWIDGETSIZE_MAX, QtWidgets.QWIDGETSIZE_MAX)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
            row.addWidget(btn, 0)
        bl.addLayout(row)
        self._neon_bottom_panel = bottom

        vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vertical.setChildrenCollapsible(False)
        vertical.addWidget(preview)
        vertical.addWidget(bottom)
        vertical.setStretchFactor(0, 1)
        vertical.setStretchFactor(1, 0)
        horizontal = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        horizontal.setChildrenCollapsible(False)
        horizontal.addWidget(left)
        horizontal.addWidget(vertical)
        horizontal.setStretchFactor(0, 0)
        horizontal.setStretchFactor(1, 1)

        # Replace the .ui arrangement. The old containers (with the product list,
        # their labels and the Sync Dates button) are hidden, not deleted, because
        # existing code still fills the product list when a site is clicked.
        main = self.layout_NEON_Main
        old = self.splitter_NEON_Vertical
        main.removeWidget(old)
        old.hide()
        self._neon_ui_layout_unused = old
        main.addWidget(horizontal, 1)
        self.splitter_NEON_Horizontal = horizontal
        self.splitter_NEON_RightVertical = vertical
        horizontal.splitterMoved.connect(lambda _p, _i: self._neon_splitter_moved())
        vertical.splitterMoved.connect(lambda _p, _i: self._neon_splitter_moved())
        horizontal.splitterMoved.connect(lambda _p, _i: self._neon_save_splitters())
        vertical.splitterMoved.connect(lambda _p, _i: self._neon_save_splitters())
        saved_h = _phenocam_setting("NEON_Preview_Height_px")
        self._neon_preview_user_height = int(saved_h) if saved_h else None
        self._neon_layout_sized = False

        self.tabWidget.currentChanged.connect(
            lambda _i: QtCore.QTimer.singleShot(0, self._neon_fit_layout)
            if self.tabWidget.currentWidget() is self.tab_NEONSites else None)

        self.NEON_listboxSites.itemChanged.connect(self._neon_item_changed)
        self._neon_setup_table()
        self._neon_layout_built = True

        # Products for every site in one request, in the background
        self._neon_products_fetcher = NEONSitesProductsFetcher(SERVER, parent=self)
        self._neon_products_fetcher.result.connect(self._neon_products_received)
        self._neon_products_fetcher.start()
        self._neon_apply_filter()

    # ------------------------------------------------------------------------------------------------------------------
    def _neon_fit_layout(self):
        h_split, v_split = self.splitter_NEON_Horizontal, self.splitter_NEON_RightVertical
        if not h_split.isVisible():
            return
        too_narrow = h_split.sizes()[0] < h_split.widget(0).minimumWidth()
        if too_narrow or (not getattr(self, "_neon_splitter_moved_flag", False) and not self._neon_layout_sized):
            total_w = sum(h_split.sizes())
            if total_w > 50:
                left_w = int(_phenocam_setting("NEON_Left_Panel_Width_px"))
                h_split.setSizes([left_w, max(total_w - left_w, 1)])
                self._neon_layout_sized = True
        total_h = sum(v_split.sizes())
        if total_h > 50:
            bottom_max = v_split.widget(1).maximumHeight()
            if self._neon_preview_user_height:
                top = max(self._neon_preview_user_height, total_h - bottom_max)
            else:
                top = total_h - min(bottom_max, total_h)
            top = max(v_split.widget(0).minimumHeight(), min(top, total_h))
            v_split.setSizes([top, total_h - top])
        self.NEON_DisplayLatestImage()

    def _neon_save_splitters(self):
        if not hasattr(self, "_neon_split_save_timer"):
            self._neon_split_save_timer = QtCore.QTimer(self)
            self._neon_split_save_timer.setSingleShot(True)
            self._neon_split_save_timer.setInterval(int(_phenocam_setting("Phenocam_Search_Debounce_ms")))

            def _save():
                try:
                    JsonEditor().update_json_entry(
                        "NEON_Left_Panel_Width_px", str(self.splitter_NEON_Horizontal.sizes()[0]))
                    self._neon_preview_user_height = self.splitter_NEON_RightVertical.sizes()[0]
                    JsonEditor().update_json_entry("NEON_Preview_Height_px", str(self._neon_preview_user_height))
                except Exception as e:
                    print(f"[NEON] Could not save splitter positions: {e}")
            self._neon_split_save_timer.timeout.connect(_save)
        self._neon_split_save_timer.start()

    # ------------------------------------------------------------------------------------------------------------------
    # NEON SITE TREE: site -> "Site details" + "Data products (N)"
    # ------------------------------------------------------------------------------------------------------------------
    NEON_BRANCH_ROLE = QtCore.Qt.UserRole + 4    # "details" or "products" on the two branch items
    NEON_PRODUCT_ROLE = QtCore.Qt.UserRole + 5   # product dict on each product item

    def _neon_branch(self, site_item, kind):
        for i in range(site_item.childCount()):
            if site_item.child(i).data(0, self.NEON_BRANCH_ROLE) == kind:
                return site_item.child(i)
        return None

    def _neon_build_site_item(self, site):
        """Tree item for one NEON field site (used at startup and by Refresh NEON)."""
        site_item = QTreeWidgetItem([f"{site.siteID} - {_fix_mojibake(site.siteName)}"])
        site_item.setData(0, QtCore.Qt.UserRole, site.siteID)

        details = QTreeWidgetItem(["Site details"])
        details.setData(0, self.NEON_BRANCH_ROLE, "details")
        details.setFlags(details.flags() & ~QtCore.Qt.ItemIsSelectable)
        for label, value in [("Site ID", site.siteID),
                             ("Site Name", _fix_mojibake(site.siteName)),
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
            details.addChild(child)
        site_item.addChild(details)

        products = QTreeWidgetItem(["Data products (loading...)"])
        products.setData(0, self.NEON_BRANCH_ROLE, "products")
        products.setFlags(products.flags() & ~QtCore.Qt.ItemIsSelectable)
        site_item.addChild(products)
        if site.siteID in getattr(self, "_neon_products", {}):
            self._neon_fill_products(site_item)
        return site_item

    def _neon_fill_products(self, site_item):
        site = site_item.data(0, QtCore.Qt.UserRole)
        branch = self._neon_branch(site_item, "products")
        info = self._neon_products.get(site)
        if branch is None or info is None:
            return
        tree = self.NEON_listboxSites
        tree.blockSignals(True)
        branch.takeChildren()
        for prod in info["products"]:
            item = QTreeWidgetItem([f"{prod['code']}  {prod['title']}"])
            item.setData(0, QtCore.Qt.UserRole, site)
            item.setData(0, self.NEON_PRODUCT_ROLE, prod)
            item.setFlags((item.flags() | QtCore.Qt.ItemIsUserCheckable) & ~QtCore.Qt.ItemIsSelectable)
            item.setCheckState(0, QtCore.Qt.Checked if (site, prod["code"]) in self._neon_rows
                               else QtCore.Qt.Unchecked)
            branch.addChild(item)
        branch.setText(0, f"Data products ({len(info['products'])})")
        tree.blockSignals(False)

    def _neon_products_received(self, data):
        if data is None:
            print("[NEON] Could not load the data product lists; product branches stay empty.")
            tree = self.NEON_listboxSites
            for i in range(tree.topLevelItemCount()):
                branch = self._neon_branch(tree.topLevelItem(i), "products")
                if branch is not None:
                    branch.setText(0, "Data products (unavailable)")
            return
        self._neon_products = data
        tree = self.NEON_listboxSites
        for i in range(tree.topLevelItemCount()):
            self._neon_fill_products(tree.topLevelItem(i))
        self._neon_apply_filter()

    def _neon_checked_count(self):
        return len(self._neon_rows)

    # ------------------------------------------------------------------------------------------------------------------
    # NEON SEARCH
    # ------------------------------------------------------------------------------------------------------------------
    def _neon_site_record(self, site_item):
        """Searchable fields for one NEON site (see NEON_SEARCH_FIELDS), without its products."""
        code = str(site_item.data(0, QtCore.Qt.UserRole) or "")
        label = site_item.text(0)
        name = label.split(" - ", 1)[1] if " - " in label else label
        details = {}
        branch = self._neon_branch(site_item, "details")
        for c in range(branch.childCount() if branch else 0):
            text = branch.child(c).text(0)
            if ":" in text:
                k, v = text.split(":", 1)
                details[k.strip().lower()] = v.strip()
        info = getattr(self, "_neon_products", {}).get(code, {})

        def num(key):
            try:
                return float(details.get(key, ""))
            except ValueError:
                return None

        return {
            "fields": {
                "site":     [code.lower()],
                "name":     [name.lower()],
                "state":    [str(info.get("state") or details.get("state", "")).lower()],
                "domain":   [str(info.get("domain") or details.get("domain code", "")).lower(),
                             str(info.get("domainName") or details.get("domain name", "")).lower()],
                "phenocam": [details.get("phenocams", "").lower()],
            },
            "numbers": {"lat": num("latitude"), "lon": num("longitude")},
            "dates": {},
        }

    def _neon_apply_filter(self):
        """
        Filter NEON sites with the search text (see NEON_SEARCH_FIELDS).
        A site is shown when every term matches the site, except that product: terms, and bare
        words the site itself doesn't match, must match at least one of its products; those terms
        also narrow the products listed under the site. Opens the products branch when products
        matched, the details branch when a site detail matched.
        """
        if not hasattr(self, "neon_count_label"):
            return
        self._neon_search_timer.stop()
        query = SiteSearchQuery(self.neon_search_edit.text(), NEON_SEARCH_FIELDS)
        self.neon_search_error.setText(query.error)
        self.neon_search_error.setVisible(bool(query.error))
        on_label = ("site", "name")   # fields shown in the site's own label

        tree = self.NEON_listboxSites
        total = shown = 0
        for i in range(tree.topLevelItemCount()):
            site_item = tree.topLevelItem(i)
            total += 1
            rec = self._neon_site_record(site_item)
            products = self._neon_branch(site_item, "products")
            product_items = [products.child(c) for c in range(products.childCount())] if products else []
            product_recs = [{"fields": {"product": [p.text(0).lower()]}} for p in product_items]

            ok, in_details, product_terms = True, False, []
            for term in query.terms:
                negate, field, _op, value = term
                if field == "product":
                    product_terms.append(term)
                    continue
                site_hit = query.term_matches(rec, term)
                if field is None:
                    if negate:
                        if site_hit:
                            ok = False
                            break
                        product_terms.append(term)      # also drop products that mention it
                    elif site_hit:
                        if not any(query.text_match(value, rec["fields"][f]) for f in on_label):
                            in_details = True
                    else:
                        product_terms.append(term)      # must be found in a product
                    continue
                if site_hit == negate:
                    ok = False
                    break
                if not negate and field not in on_label:
                    in_details = True

            visible = [all(query.term_matches(r, t) != t[0] for t in product_terms) for r in product_recs]
            if ok and any(not t[0] for t in product_terms) and not any(visible):
                ok = False
            for p, v in zip(product_items, visible):
                p.setHidden(not v)
            site_item.setHidden(not ok)
            shown += ok
            if ok and query.terms:
                if products is not None and any(not t[0] for t in product_terms):
                    site_item.setExpanded(True)
                    products.setExpanded(True)
                details = self._neon_branch(site_item, "details")
                if details is not None and in_details:
                    site_item.setExpanded(True)
                    details.setExpanded(True)

        text = f"{shown:,} of {total:,} sites shown" if shown != total else f"{total:,} sites"
        n = self._neon_checked_count()
        text += f", {n:,} product{'s' if n != 1 else ''} checked"
        self.neon_count_label.setText(text)

    # ------------------------------------------------------------------------------------------------------------------
    # NEON DOWNLOAD TABLE: row 0 sets the range for all checked products; one row per checked site/product
    # ------------------------------------------------------------------------------------------------------------------
    NEON_IMAGE_PRODUCT_IDS = (20002, 42, 33)   # products downloaded as images (same rule as the downloader)

    @staticmethod
    def _neon_product_id(code):
        try:
            return int(code.split('.')[1])
        except (IndexError, ValueError):
            return -1

    def _neon_setup_table(self):
        t = self.NEON_selected_products
        t.clear()
        t.setRowCount(0)
        t.setColumnCount(6)
        t.setHorizontalHeaderLabels(["Site", "Product", "Available", "Start Date", "End Date", "Time Window"])
        t.setStyleSheet("")
        hdr = t.horizontalHeader()
        hdr.setStyleSheet(_data_table_header_style())
        hdr.setMinimumSectionSize(0)
        hdr.setHighlightSections(False)
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        t.verticalHeader().setVisible(False)
        t.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._neon_rows = {}

        t.insertRow(0)
        today = QtCore.QDate.currentDate()
        days = int(_phenocam_setting("Phenocam_Default_Range_Days"))
        s, e, cell, ts, te = self._neon_range_widgets(today.addDays(-days), today, True)
        label = QTableWidgetItem("All checked products")
        f = label.font()
        f.setBold(True)
        label.setFont(f)
        t.setItem(0, 0, label)
        t.setSpan(0, 0, 1, 3)
        t.setCellWidget(0, 3, s)
        t.setCellWidget(0, 4, e)
        t.setCellWidget(0, 5, cell)
        self._neon_all_row = (s, e, ts, te)
        for w in (s, e):
            w.dateChanged.connect(lambda _d: self._neon_apply_all_row())
        for w in (ts, te):
            w.timeChanged.connect(lambda _t: self._neon_apply_all_row())
        self._neon_size_table()

    def _neon_range_widgets(self, start_q, end_q, with_time, t_start=None, t_end=None):
        s = QtWidgets.QDateEdit(start_q)
        e = QtWidgets.QDateEdit(end_q)
        for w in (s, e):
            w.setCalendarPopup(True)
            w.setDisplayFormat("yyyy-MM-dd")
            w.setFrame(False)
        if not with_time:
            return s, e, None, None, None
        cell = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(cell)
        hl.setContentsMargins(2, 0, 2, 0)
        hl.setSpacing(4)
        ts = QtWidgets.QTimeEdit(t_start or QtCore.QTime(0, 0))
        te = QtWidgets.QTimeEdit(t_end or QtCore.QTime(23, 59))
        for w in (ts, te):
            w.setDisplayFormat("HH:mm")
            w.setFrame(False)
        hl.addWidget(ts)
        hl.addWidget(QtWidgets.QLabel("to"))
        hl.addWidget(te)
        return s, e, cell, ts, te

    @staticmethod
    def _neon_available_range(months):
        """(first day, last day) covered by NEON 'YYYY-MM' availability strings."""
        if not months:
            return QtCore.QDate(), QtCore.QDate()
        first = QtCore.QDate.fromString(sorted(months)[0] + "-01", "yyyy-MM-dd")
        last = QtCore.QDate.fromString(sorted(months)[-1] + "-01", "yyyy-MM-dd")
        if last.isValid():
            last = last.addMonths(1).addDays(-1)
        return first, last

    def _neon_fit_range(self, months, start_q, end_q):
        """Trim a range to the product's available months; if they don't overlap, keep the
        length and place it at the nearer end of the availability (as on the PhenoCam tab)."""
        first, last = self._neon_available_range(months)
        span = start_q.daysTo(end_q)
        if first.isValid() and start_q < first:
            start_q = first
        if last.isValid() and end_q > last:
            end_q = last
        if end_q < start_q:
            if last.isValid() and start_q > last:
                end_q = last
                start_q = end_q.addDays(-span)
                if first.isValid() and start_q < first:
                    start_q = first
            else:
                start_q = first
                end_q = start_q.addDays(span)
                if last.isValid() and end_q > last:
                    end_q = last
        return start_q, end_q

    def _neon_apply_all_row(self):
        a_s, a_e, a_ts, a_te = self._neon_all_row
        for info in self._neon_rows.values():
            sq, eq = self._neon_fit_range(info["product"].get("months", []), a_s.date(), a_e.date())
            info["start"].setDate(sq)
            info["end"].setDate(eq)
            if info["time_start"] is not None:
                info["time_start"].setTime(a_ts.time())
                info["time_end"].setTime(a_te.time())

    def _neon_add_row(self, site, prod):
        key = (site, prod["code"])
        if key in self._neon_rows:
            return
        t = self.NEON_selected_products
        r = t.rowCount()
        t.insertRow(r)
        a_s, a_e, a_ts, a_te = self._neon_all_row
        months = prod.get("months", [])
        sq, eq = self._neon_fit_range(months, a_s.date(), a_e.date())
        is_image = self._neon_product_id(prod["code"]) in self.NEON_IMAGE_PRODUCT_IDS
        s, e, cell, ts, te = self._neon_range_widgets(sq, eq, is_image, a_ts.time(), a_te.time())
        first, last = self._neon_available_range(months)
        avail = (f"{sorted(months)[0]} to {sorted(months)[-1]}" if months else "")
        site_item = QTableWidgetItem(site)
        site_item.setData(QtCore.Qt.UserRole, key)
        t.setItem(r, 0, site_item)
        t.setItem(r, 1, QTableWidgetItem(f"{prod['code']}  {prod['title']}"))
        t.setItem(r, 2, QTableWidgetItem(avail))
        t.setCellWidget(r, 3, s)
        t.setCellWidget(r, 4, e)
        if cell is not None:
            t.setCellWidget(r, 5, cell)
        else:
            na = QTableWidgetItem("n/a")
            na.setTextAlignment(QtCore.Qt.AlignCenter)
            t.setItem(r, 5, na)
        self._neon_rows[key] = {"site": site, "product": prod, "start": s, "end": e,
                                "time_start": ts, "time_end": te}
        self._neon_size_table()

    def _neon_remove_row(self, key):
        if key not in self._neon_rows:
            return
        t = self.NEON_selected_products
        for r in range(1, t.rowCount()):
            it = t.item(r, 0)
            if it is not None and it.data(QtCore.Qt.UserRole) == key:
                t.removeRow(r)
                break
        del self._neon_rows[key]
        self._neon_size_table()

    def _neon_item_changed(self, item, column):
        prod = item.data(0, self.NEON_PRODUCT_ROLE)
        if column != 0 or not prod:
            return
        site = item.data(0, QtCore.Qt.UserRole)
        if item.checkState(0) == QtCore.Qt.Checked:
            self._neon_add_row(site, prod)
        else:
            self._neon_remove_row((site, prod["code"]))
        self._neon_apply_filter()

    def _neon_size_table(self):
        t = self.NEON_selected_products
        blank = int(_phenocam_setting("NEON_Table_Blank_Rows"))
        max_rows = int(_phenocam_setting("NEON_Table_Max_Rows"))
        shown = min(max(t.rowCount() - 1, blank), max_rows) + 1
        h = t.horizontalHeader().height() or t.horizontalHeader().sizeHint().height()
        for r in range(shown):
            h += t.rowHeight(r) if r < t.rowCount() else t.verticalHeader().defaultSectionSize()
        h += 2 * t.frameWidth()
        if t.horizontalScrollBar().maximum() > 0:
            h += t.horizontalScrollBar().sizeHint().height()
        t.setFixedHeight(h)
        bottom = getattr(self, "_neon_bottom_panel", None)
        if bottom is not None:
            bottom.setMaximumHeight(bottom.sizeHint().height())
            if hasattr(self, "splitter_NEON_RightVertical"):
                self._neon_fit_layout()

    def _neon_download_jobs(self):
        """[(site, domain, 'CODE: title', start_date, start_time, end_date, end_time), ...] in table order."""
        import datetime as dt_mod
        jobs = []
        t = self.NEON_selected_products
        for r in range(1, t.rowCount()):
            key = t.item(r, 0).data(QtCore.Qt.UserRole)
            info = self._neon_rows[key]
            prod = info["product"]
            sd, ed = info["start"].date().toPyDate(), info["end"].date().toPyDate()
            if info["time_start"] is not None:
                st, et = info["time_start"].time().toPyTime(), info["time_end"].time().toPyTime()
            else:
                st, et = dt_mod.time(0, 0), dt_mod.time(23, 59)
            domain = self._neon_products.get(info["site"], {}).get("domain", "")
            jobs.append((info["site"], domain, f"{prod['code']}: {prod['title']}", sd, st, ed, et))
        return jobs

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
                    self._show_image_message("neon", "Loading latest image...")
                    self._neon_preview_fetcher = NEONPreviewFetcher(SITECODE, DOMAINCODE, strProductID)
                    self._neon_preview_fetcher.result.connect(self._neon_preview_received)
                    self._neon_preview_fetcher.start()
                else:
                    gWebImagesAvailable = 0
                    self._show_image_message("neon", "No image available.")

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
    # ------------------------------------------------------------------------------------------------------------------
    # USGS TAB LAYOUT (built in code; same look as the PhenoCam tab)
    #
    #   splitter_USGS_Horizontal
    #   ├── left:  USGS_listboxSites
    #   └── right: splitter_USGS_Vertical
    #       ├── top:    image title + USGS_labelLatestImage
    #       └── bottom: table_USGS_Sites
    #                   [folder path] [Browse...] [Correlate Sensor Data] [Download]
    #
    # The existing USGS widgets are moved into new splitters; their handlers are unchanged.
    # ------------------------------------------------------------------------------------------------------------------
    def setup_usgs_layout(self):
        main = self.layout_USGS_Main
        old_vertical = self.splitter_USGS_Vertical
        filled_style, ghost_style = _button_styles()

        # Left: search row, site list, count + hidden-camera toggle (as on the PhenoCam tab)
        left = QtWidgets.QWidget()
        left.setMinimumWidth(int(_phenocam_setting("USGS_Left_Panel_Min_Width_px")))
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)

        search_row = QtWidgets.QHBoxLayout()
        search_row.setSpacing(4)
        self.usgs_search_edit = QtWidgets.QLineEdit()
        self.usgs_search_edit.setPlaceholderText("Enter search terms here")
        self.usgs_search_edit.setClearButtonEnabled(True)
        if hasattr(QtGui.QPalette, "PlaceholderText"):
            pal = self.usgs_search_edit.palette()
            pal.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(_phenocam_setting("Phenocam_Placeholder_Color")))
            self.usgs_search_edit.setPalette(pal)
        self.usgs_search_btn = QtWidgets.QPushButton("Search")
        self.usgs_search_btn.setStyleSheet(ghost_style)
        search_row.addWidget(self.usgs_search_edit, 1)
        search_row.addWidget(self.usgs_search_btn, 0)
        ll.addLayout(search_row)
        self.usgs_search_error = _search_error_label()
        ll.addWidget(self.usgs_search_error)
        self._usgs_search_help = SearchHelpFilter(
            self.usgs_search_edit,
            _search_help_html(USGS_SEARCH_FIELDS, USGS_SEARCH_EXAMPLES, USGS_SEARCH_EXAMPLE), parent=self)

        ll.addWidget(self.USGS_listboxSites, 1)
        self.USGS_listboxSites.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)

        self.usgs_count_label = QtWidgets.QLabel("")
        self.usgs_count_label.setWordWrap(True)
        self.usgs_hidden_btn = QtWidgets.QPushButton("Show hidden cameras")
        self.usgs_hidden_btn.setStyleSheet(ghost_style)
        self.usgs_hidden_btn.clicked.connect(self._usgs_toggle_hidden_clicked)
        ll.addWidget(self.usgs_count_label)
        ll.addWidget(self.usgs_hidden_btn, 0, QtCore.Qt.AlignRight)

        self._usgs_search_timer = QtCore.QTimer(self)
        self._usgs_search_timer.setSingleShot(True)
        self._usgs_search_timer.setInterval(int(_phenocam_setting("Phenocam_Search_Debounce_ms")))
        self._usgs_search_timer.timeout.connect(self._usgs_apply_filter)
        self.usgs_search_edit.textChanged.connect(lambda _t: self._usgs_search_timer.start())
        self.usgs_search_edit.returnPressed.connect(self._usgs_apply_filter)
        self.usgs_search_btn.clicked.connect(self._usgs_apply_filter)

        # Right top: image title over the image
        preview = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(preview)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(2)
        title = getattr(self, "_usgs_image_title_label", None)
        if title is not None:
            pl.addWidget(title, 0)
        self.USGS_labelLatestImage.setStyleSheet("QLabel { background: transparent; color: palette(text); }")
        self.usgs_image_panel = ZoomImagePanel(self.USGS_labelLatestImage)
        pl.addWidget(self.usgs_image_panel, 1)
        preview.setMinimumHeight(int(_phenocam_setting("Phenocam_Preview_Min_Height_px")))

        # Right bottom: table, then the folder row
        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(6)
        table = self.table_USGS_Sites
        table.verticalHeader().setVisible(False)
        bl.addWidget(table, 0)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        self.edit_USGSSaveFilePath.setPlaceholderText("Select output folder...")
        row.addWidget(self.edit_USGSSaveFilePath, 1)
        buttons = [(self.pushButton_USGS_BrowseImageFolder, ghost_style)]
        correlate = getattr(self, "pushButton_USGSCorrelate", None)
        if correlate is not None:
            buttons.append((correlate, ghost_style))
        buttons.append((self.pushButton_USGSDownload, filled_style))
        for btn, style in buttons:
            btn.setStyleSheet(style)
            btn.setMinimumSize(0, 0)
            btn.setMaximumSize(QtWidgets.QWIDGETSIZE_MAX, QtWidgets.QWIDGETSIZE_MAX)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
            row.addWidget(btn, 0)
        bl.addLayout(row)
        self._usgs_bottom_panel = bottom

        vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vertical.setChildrenCollapsible(False)
        vertical.addWidget(preview)
        vertical.addWidget(bottom)
        vertical.setStretchFactor(0, 1)
        vertical.setStretchFactor(1, 0)

        horizontal = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        horizontal.setChildrenCollapsible(False)
        horizontal.addWidget(left)
        horizontal.addWidget(vertical)
        horizontal.setStretchFactor(0, 0)
        horizontal.setStretchFactor(1, 1)

        # Replace the .ui splitters, now empty of the widgets moved above. They are
        # hidden rather than deleted so any code still holding references to the
        # old containers (e.g. Resize_Controls) keeps working.
        main.removeWidget(old_vertical)
        old_vertical.hide()
        self._usgs_ui_layout_unused = old_vertical
        main.addWidget(horizontal, 1)

        self.splitter_USGS_Horizontal = horizontal
        self.splitter_USGS_Vertical = vertical
        horizontal.splitterMoved.connect(self._usgs_splitter_moved)
        vertical.splitterMoved.connect(self._usgs_vertical_splitter_moved)
        horizontal.splitterMoved.connect(lambda _p, _i: self._usgs_save_splitters())
        vertical.splitterMoved.connect(lambda _p, _i: self._usgs_save_splitters())

        saved_h = _phenocam_setting("USGS_Preview_Height_px")
        self._usgs_preview_user_height = int(saved_h) if saved_h else None
        self._usgs_layout_sized = False
        self._usgs_size_table()

        # A hidden tab has no size yet, so set the site-list width the first time the tab is shown.
        self.tabWidget.currentChanged.connect(
            lambda _i: QtCore.QTimer.singleShot(0, self._usgs_fit_layout)
            if self.tabWidget.currentWidget() is self.tab_USGSSites else None)

        # Multi-site selection: checkboxes on the cameras, one table row per checked camera
        self._usgs_rows = {}          # camera ID -> (start date, end date, start time, end time) widgets
        self._usgs_count_cache = {}   # (camera ID, start date, end date, start time, end time) -> image count
        self.USGS_listboxSites.itemChanged.connect(self._usgs_item_changed)
        self._usgs_layout_built = True
        tree = self.USGS_listboxSites
        tree.blockSignals(True)
        for i in range(tree.topLevelItemCount()):
            it = tree.topLevelItem(i)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(0, QtCore.Qt.Unchecked)
        tree.blockSignals(False)
        self._usgs_reset_rows()

    # ------------------------------------------------------------------------------------------------------------------
    # USGS SITE SEARCH AND HIDDEN-CAMERA TOGGLE
    # ------------------------------------------------------------------------------------------------------------------
    def _usgs_toggle_hidden_clicked(self):
        self._toggle_usgs_hidden_cameras()   # rebuilds the tree, which re-applies the search

    def _usgs_showing_hidden(self):
        try:
            return any(self.myHIVIS.is_hidden(cam) for cam in self.cameraList)
        except Exception:
            return False

    def _usgs_apply_filter(self):
        """Filter the USGS site tree with the search text (see USGS_SEARCH_FIELDS)."""
        if not hasattr(self, "usgs_count_label"):
            return
        if hasattr(self, "_usgs_search_timer"):
            self._usgs_search_timer.stop()
        query = SiteSearchQuery(self.usgs_search_edit.text(), USGS_SEARCH_FIELDS)
        self.usgs_search_error.setText(query.error)
        self.usgs_search_error.setVisible(bool(query.error))
        tree = self.USGS_listboxSites
        total = shown = 0
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            ok = not query.terms or query.matches(self._usgs_search_record(item.data(0, QtCore.Qt.UserRole)))
            item.setHidden(not ok)
            total += 1
            shown += ok
        text = f"{shown:,} of {total:,} cameras shown" if shown != total else f"{total:,} cameras"
        showing_hidden = self._usgs_showing_hidden()
        if not showing_hidden:
            text += ", hidden cameras not listed"
        n = len(getattr(self, "_usgs_rows", {}))
        text += f", {n:,} checked"
        self.usgs_count_label.setText(text)
        self.usgs_hidden_btn.setText("Hide hidden cameras" if showing_hidden else "Show hidden cameras")

    def _usgs_size_table(self):
        """Table height: at least USGS_Table_Rows_Shown rows, growing with checked cameras up to
        USGS_Table_Max_Rows, then scrolling. The panel is capped to its contents."""
        t = self.table_USGS_Sites
        rows = min(max(t.rowCount(), int(_phenocam_setting("USGS_Table_Rows_Shown"))),
                   int(_phenocam_setting("USGS_Table_Max_Rows")))
        h = t.horizontalHeader().height() or t.horizontalHeader().sizeHint().height()
        for r in range(rows):
            h += t.rowHeight(r) if r < t.rowCount() else t.verticalHeader().defaultSectionSize()
        h += 2 * t.frameWidth()
        if t.horizontalScrollBar().maximum() > 0:
            h += t.horizontalScrollBar().sizeHint().height()
        t.setFixedHeight(h)
        self._usgs_bottom_panel.setMaximumHeight(self._usgs_bottom_panel.sizeHint().height())
        if hasattr(self, "splitter_USGS_Vertical") and self.splitter_USGS_Vertical.widget(1) is self._usgs_bottom_panel:
            self._usgs_fit_layout()

    def _usgs_fit_layout(self):
        """Initial site-list width, and image height (saved, or whatever the table leaves)."""
        h_split, v_split = self.splitter_USGS_Horizontal, self.splitter_USGS_Vertical
        if not h_split.isVisible():
            return   # sizes of a hidden tab aren't real yet; sized when the tab is shown
        too_narrow = h_split.sizes()[0] < h_split.widget(0).minimumWidth()
        if too_narrow or (not getattr(self, "_usgs_splitter_moved_flag", False)
                          and not getattr(self, "_usgs_layout_sized", False)):
            total_w = sum(h_split.sizes())
            if total_w > 50:
                left_w = int(_phenocam_setting("USGS_Left_Panel_Width_px"))
                h_split.setSizes([left_w, max(total_w - left_w, 1)])
                self._usgs_layout_sized = True
        total_h = sum(v_split.sizes())
        if total_h > 50:
            bottom_max = v_split.widget(1).maximumHeight()
            if self._usgs_preview_user_height:
                top = max(self._usgs_preview_user_height, total_h - bottom_max)
            else:
                top = total_h - min(bottom_max, total_h)
            top = max(v_split.widget(0).minimumHeight(), min(top, total_h))
            v_split.setSizes([top, total_h - top])

    def _usgs_save_splitters(self):
        """Save the site-list width and image height once a drag pauses."""
        if not hasattr(self, "_usgs_split_save_timer"):
            self._usgs_split_save_timer = QtCore.QTimer(self)
            self._usgs_split_save_timer.setSingleShot(True)
            self._usgs_split_save_timer.setInterval(int(_phenocam_setting("Phenocam_Search_Debounce_ms")))

            def _save():
                try:
                    JsonEditor().update_json_entry(
                        "USGS_Left_Panel_Width_px", str(self.splitter_USGS_Horizontal.sizes()[0]))
                    self._usgs_preview_user_height = self.splitter_USGS_Vertical.sizes()[0]
                    JsonEditor().update_json_entry(
                        "USGS_Preview_Height_px", str(self._usgs_preview_user_height))
                except Exception as e:
                    print(f"[USGS] Could not save splitter positions: {e}")
            self._usgs_split_save_timer.timeout.connect(_save)
        self._usgs_split_save_timer.start()

    # ------------------------------------------------------------------------------------------------------------------
    # USGS SEARCH RECORDS
    # ------------------------------------------------------------------------------------------------------------------
    def _usgs_camera_record(self, cam_id):
        """NIMS camera dict for a camera ID ({} if unknown)."""
        try:
            return self.myHIVIS.get_camera_dictionary().get(cam_id, {}) or {}
        except Exception:
            return {}

    def _usgs_nwis_id(self, cam_id):
        nwis = self._usgs_camera_record(cam_id).get("nwisId")
        return str(nwis).strip() if nwis else None

    def _usgs_search_record(self, cam_id):
        """Searchable fields for one camera (see SiteSearchQuery and USGS_SEARCH_FIELDS)."""
        cam = self._usgs_camera_record(cam_id)

        def t(key):
            v = cam.get(key)
            return "" if v is None else str(v).strip().lower()

        def num(key):
            import math
            try:
                x = float(cam.get(key))
                return x if math.isfinite(x) else None
            except (TypeError, ValueError):
                return None

        try:
            hidden = bool(self.myHIVIS.is_hidden(cam_id))
        except Exception:
            hidden = cam.get("hideCam", True) is not False
        return {
            "fields": {
                "camera": [str(cam_id).lower()],
                "name":   [t("camName")],
                "desc":   [t("camDesc")],
                "nwis":   [t("nwisId")],
                "state":  [t("stateAbrv")],
                "tz":     [t("tz")],
                "pcode":  [t("defaultPCode")],
                "hidden": ["true" if hidden else "false"],
            },
            "numbers": {"lat": num("lat"), "lon": num("lng")},
            "dates": {"newest": t("newestImageDT")[:10], "created": t("createdDate")[:10]},
        }

    # ------------------------------------------------------------------------------------------------------------------
    # USGS DOWNLOAD TABLE: row 0 ("All checked sites") sets the range for every checked camera;
    # one row per checked camera, in the existing column layout.
    # ------------------------------------------------------------------------------------------------------------------
    USGS_COL_SITE, USGS_COL_COUNT = 0, 1
    USGS_COL_START_DATE, USGS_COL_END_DATE, USGS_COL_START_TIME, USGS_COL_END_TIME = 4, 5, 6, 7

    def _usgs_all_row_widgets(self):
        t = self.table_USGS_Sites
        return tuple(t.cellWidget(0, c) for c in (self.USGS_COL_START_DATE, self.USGS_COL_END_DATE,
                                                   self.USGS_COL_START_TIME, self.USGS_COL_END_TIME))

    def _usgs_reset_rows(self):
        """Label row 0 "All checked sites", drop site rows, uncheck every camera.
        Called after USGS_FormatProductTable rebuilds the table."""
        t = self.table_USGS_Sites
        while t.rowCount() > 1:
            t.removeRow(1)
        self._usgs_rows = {}
        label = QTableWidgetItem("All checked sites")
        f = label.font()
        f.setBold(True)
        label.setFont(f)
        t.setItem(0, self.USGS_COL_SITE, label)
        t.setItem(0, self.USGS_COL_COUNT, QTableWidgetItem(""))
        s, e, ts, te = self._usgs_all_row_widgets()
        for w in (s, e, ts, te):
            if w is not None:
                w.dateTimeChanged.connect(lambda _d: self._usgs_apply_all_row())
        tree = self.USGS_listboxSites
        tree.blockSignals(True)
        for i in range(tree.topLevelItemCount()):
            it = tree.topLevelItem(i)
            if it.flags() & QtCore.Qt.ItemIsUserCheckable:
                it.setCheckState(0, QtCore.Qt.Unchecked)
        tree.blockSignals(False)
        self._usgs_size_table()
        self._usgs_apply_filter()

    def _usgs_apply_all_row(self):
        s, e, ts, te = self._usgs_all_row_widgets()
        if s is None:
            return
        for rs, re_, rts, rte in self._usgs_rows.values():
            rs.setDate(s.date())
            re_.setDate(e.date())
            rts.setTime(ts.time())
            rte.setTime(te.time())

    def _usgs_add_row(self, cam_id):
        if cam_id in self._usgs_rows:
            return
        t = self.table_USGS_Sites
        r = t.rowCount()
        t.insertRow(r)
        a_s, a_e, a_ts, a_te = self._usgs_all_row_widgets()
        item = QTableWidgetItem(cam_id)
        item.setData(QtCore.Qt.UserRole, cam_id)
        t.setItem(r, self.USGS_COL_SITE, item)
        t.setItem(r, self.USGS_COL_COUNT, QTableWidgetItem(""))
        for c in (2, 3):                       # min/max Date, disabled as on row 0
            w = QtWidgets.QDateEdit()
            w.setDisabled(True)
            t.setCellWidget(r, c, w)
        s = QtWidgets.QDateEdit(calendarPopup=True)
        e = QtWidgets.QDateEdit(calendarPopup=True)
        s.setDate(a_s.date() if a_s is not None else QtCore.QDate.currentDate())
        e.setDate(a_e.date() if a_e is not None else QtCore.QDate.currentDate())
        ts, te = QDateTimeEdit(), QDateTimeEdit()
        ts.setTime(a_ts.time() if a_ts is not None else QtCore.QTime(0, 0, 0))
        te.setTime(a_te.time() if a_te is not None else QtCore.QTime(23, 59, 59))
        for w in (s, e):
            w.setKeyboardTracking(False)
        for w in (ts, te):
            w.setDisplayFormat("hh:mm")
            w.setKeyboardTracking(False)
            w.setFrame(False)
        for w in (s, e, ts, te):
            w.dateTimeChanged.connect(lambda _d: self._usgs_schedule_count())
        t.setCellWidget(r, self.USGS_COL_START_DATE, s)
        t.setCellWidget(r, self.USGS_COL_END_DATE, e)
        t.setCellWidget(r, self.USGS_COL_START_TIME, ts)
        t.setCellWidget(r, self.USGS_COL_END_TIME, te)
        self._usgs_rows[cam_id] = (s, e, ts, te)
        self._usgs_size_table()
        self._usgs_schedule_count()

    def _usgs_remove_row(self, cam_id):
        if cam_id not in self._usgs_rows:
            return
        t = self.table_USGS_Sites
        for r in range(1, t.rowCount()):
            it = t.item(r, self.USGS_COL_SITE)
            if it is not None and it.data(QtCore.Qt.UserRole) == cam_id:
                t.removeRow(r)
                break
        del self._usgs_rows[cam_id]
        self._usgs_size_table()
        self._usgs_refresh_total()

    def _usgs_item_changed(self, item, column):
        if column != 0 or item.parent() is not None or not (item.flags() & QtCore.Qt.ItemIsUserCheckable):
            return
        cam_id = item.data(0, QtCore.Qt.UserRole)
        if item.checkState(0) == QtCore.Qt.Checked:
            self._usgs_add_row(cam_id)
        else:
            self._usgs_remove_row(cam_id)
        self._usgs_apply_filter()

    # ------------------------------------------------------------------------------------------------------------------
    # USGS IMAGE COUNTS (one NIMS request per camera, cached per camera and range)
    # ------------------------------------------------------------------------------------------------------------------
    def _usgs_row_range(self, cam_id):
        s, e, ts, te = self._usgs_rows[cam_id]
        return (s.date().toPyDate(), e.date().toPyDate(),
                ts.time().toPyTime().replace(second=0), te.time().toPyTime().replace(second=59))

    def _usgs_schedule_count(self):
        t = self.table_USGS_Sites
        for r in range(1, t.rowCount()):
            cam_id = t.item(r, self.USGS_COL_SITE).data(QtCore.Qt.UserRole)
            if (cam_id, *self._usgs_row_range(cam_id)) not in self._usgs_count_cache:
                t.item(r, self.USGS_COL_COUNT).setText("Computing...")
        self._usgs_refresh_total()
        self.usgs_check_timer.stop()
        self.usgs_check_timer.start(int(_phenocam_setting("USGS_Count_Debounce_ms")))

    def _usgs_refresh_total(self):
        t = self.table_USGS_Sites
        if t.rowCount() <= 1:
            t.item(0, self.USGS_COL_COUNT).setText("")
            return
        total, done = 0, True
        for cam_id in self._usgs_rows:
            n = self._usgs_count_cache.get((cam_id, *self._usgs_row_range(cam_id)))
            if n is None:
                done = False
            else:
                total += n
        t.item(0, self.USGS_COL_COUNT).setText(f"{total:,}" if done else "Computing...")

    def _usgs_count_rows(self):
        """Count images for every checked camera whose range isn't cached yet. Runs on the UI thread,
        one camera at a time, because the HIVIS count can show message boxes on network errors."""
        if self.usgs_checking:
            return
        self.usgs_checking = True
        try:
            t = self.table_USGS_Sites
            for cam_id in list(self._usgs_rows):
                if cam_id not in self._usgs_rows:
                    continue   # unchecked while counting
                key = (cam_id, *self._usgs_row_range(cam_id))
                if key not in self._usgs_count_cache:
                    sd, ed, st, et = key[1:]
                    try:
                        n = self.myHIVIS.get_image_count(siteName=cam_id, nwisID=self._usgs_nwis_id(cam_id),
                                                         startDate=sd, endDate=ed, startTime=st, endTime=et)
                    except Exception as e:
                        print(f"[USGS] Image count for {cam_id}: {e}")
                        n = None
                    if n is not None:
                        self._usgs_count_cache[key] = int(n)
                for r in range(1, t.rowCount()):
                    it = t.item(r, self.USGS_COL_SITE)
                    if it is not None and it.data(QtCore.Qt.UserRole) == cam_id:
                        n = self._usgs_count_cache.get((cam_id, *self._usgs_row_range(cam_id)))
                        t.item(r, self.USGS_COL_COUNT).setText(f"{n:,}" if n is not None else "Unavailable")
                        break
                self._usgs_refresh_total()
                QApplication.processEvents()
            self._usgs_refresh_total()
        finally:
            self.usgs_checking = False

    # ------------------------------------------------------------------------------------------------------------------
    # USGS MULTI-SITE DOWNLOAD
    # ------------------------------------------------------------------------------------------------------------------
    def _usgs_download_checked(self, root_folder):
        """Download every checked camera. One camera downloads into the folder itself; several
        cameras each get <folder>/<camera ID>/Images and /data."""
        self.usgs_check_timer.stop()
        t = self.table_USGS_Sites
        jobs = []
        for r in range(1, t.rowCount()):
            cam_id = t.item(r, self.USGS_COL_SITE).data(QtCore.Qt.UserRole)
            jobs.append((cam_id, *self._usgs_row_range(cam_id)))
        bad = [j[0] for j in jobs if datetime.datetime.combine(j[1], j[3]) >= datetime.datetime.combine(j[2], j[4])]
        if bad:
            App_QMessageBox("USGS Download", "Start must be before End for: " + ", ".join(bad),
                            QMessageBox.Close).displayMsgBox()
            return
        multi = len(jobs) > 1
        self._usgs_cancel_requested = False
        results = []
        for i, (cam_id, sd, ed, st, et) in enumerate(jobs, start=1):
            if self._usgs_cancel_requested:
                results.append(f"{cam_id}: not downloaded (cancelled)")
                continue
            folder = os.path.join(root_folder, cam_id) if multi else root_folder
            image_folder = os.path.join(folder, "Images")
            data_folder = os.path.join(folder, "data")
            os.makedirs(image_folder, exist_ok=True)
            os.makedirs(data_folder, exist_ok=True)
            nwis = self._usgs_nwis_id(cam_id)
            try:
                self.statusBar().showMessage(f"USGS download: {cam_id} (site {i} of {len(jobs)})")
            except Exception:
                pass
            # Builds this camera's time-zone-aware image list, which download_images then uses.
            self.myHIVIS.get_image_count(siteName=cam_id, nwisID=nwis, startDate=sd, endDate=ed,
                                         startTime=st, endTime=et)
            downloaded, missing = self.usgs.download_images(
                cam_id, sd, ed, st, et, image_folder,
                progress=self._usgs_progress,
                cancel_check=lambda: self._usgs_cancel_requested)
            self.force_close_progress()
            line = f"{cam_id}: {downloaded} new image(s)" + (f", {missing} failed" if missing else "")
            if nwis:
                self.myHIVIS.fetchStageAndDischarge(nwis, cam_id, sd, ed, st, et, data_folder)
            else:
                line += ", no NWIS site number so no sensor data"
            results.append(line)
        try:
            self.statusBar().clearMessage()
        except Exception:
            pass
        App_QMessageBox("USGS Download", "Download complete.\n\n" + "\n".join(results),
                        QMessageBox.Close).displayMsgBox()

    # ------------------------------------------------------------------------------------------------------------------
    # USGS CORRELATION: a site folder (Images + data), a folder of site folders, or (as before) an image folder
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _usgs_subfolder(path, name):
        """Subfolder of path named name, ignoring case; None if absent."""
        try:
            for d in os.listdir(path):
                if d.lower() == name.lower() and os.path.isdir(os.path.join(path, d)):
                    return os.path.join(path, d)
        except OSError:
            pass
        return None

    def _usgs_correlation_sites(self, folder):
        """[(site name, images folder, data folder)] for a site folder or a folder of site folders."""
        images, data = self._usgs_subfolder(folder, "images"), self._usgs_subfolder(folder, "data")
        if images and data:
            return [(os.path.basename(os.path.normpath(folder)), images, data)]
        sites = []
        try:
            children = sorted(os.listdir(folder))
        except OSError:
            children = []
        for child in children:
            path = os.path.join(folder, child)
            if os.path.isdir(path):
                images, data = self._usgs_subfolder(path, "images"), self._usgs_subfolder(path, "data")
                if images and data:
                    sites.append((child, images, data))
        return sites

    @staticmethod
    def _usgs_sensor_file(data_folder):
        """Newest NWIS sensor file in a data folder: .csv preferred, else .txt. None if there is none."""
        for ext in (".csv", ".txt"):
            files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.lower().endswith(ext)]
            if files:
                return max(files, key=os.path.getmtime)
        return None


    def USGS_InitProductTable(self):
        # HEADER TITLES
        headerList = ['Site', 'Image Count', ' min Date ', ' max Date ', 'Start Date', 'End Date', 'Start Time', 'End Time']

        # DEFINE HEADER STYLE (shared with the PhenoCam table)
        stylesheet = _data_table_header_style()

        # POINTER TO HEADER
        header = self.table_USGS_Sites.horizontalHeader()

        # SET DEFAULT HEADER SETTINGS (columns fit their contents, as in the PhenoCam table)
        header.setHighlightSections(False)
        header.setStretchLastSection(False)

        # INSERT TITLES INTO HEADER AND FORMAT HEADER
        # MAKE COLUMNS 1 THRU 'n' SIZE TO CONTENTS
        # MAKE COLUMN 0 STRETCH TO FILL UP REMAINING EMPTY SPACE IN THE TABLE
        for i, item in enumerate(headerList):
            headerItem = QTableWidgetItem(item)
            headerItem.setTextAlignment(QtCore.Qt.AlignCenter)

            self.table_USGS_Sites.setHorizontalHeaderItem(i, headerItem)
            header.setStyleSheet(stylesheet)

            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

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
        time_widget.setDisplayFormat("hh:mm")
        time_widget.setTime(QtCore.QTime(0, 0, 0))  # 00:00:00
        time_widget.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(time_widget, self.table_USGS_Sites))
        time_widget.setKeyboardTracking(False)
        time_widget.setFrame(False)
        self.table_USGS_Sites.setCellWidget(0, 6, time_widget)

        # END TIME - defaults to 23:59:59 (full day)
        time_widget = QDateTimeEdit()
        time_widget.setDisplayFormat("hh:mm")
        time_widget.setTime(QtCore.QTime(23, 59, 59))  # 23:59:59
        time_widget.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(time_widget, self.table_USGS_Sites))
        time_widget.setKeyboardTracking(False)
        time_widget.setFrame(False)
        self.table_USGS_Sites.setCellWidget(0, 7, time_widget)

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def USGS_dateChangeMethod(self, date_widget, tableWidget):
        if getattr(self, "_usgs_layout_built", False):
            return   # row 0 is "All checked sites"; its changes go through _usgs_apply_all_row
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
            dateTime.setDisplayFormat("hh:mm")
            dateTime.setTime(QtCore.QTime(0, 0, 0))  # 00:00:00
            dateTime.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(date_widget, self.table_USGS_Sites))
            dateTime.setKeyboardTracking(False)
            dateTime.setFrame(False)
            tableProducts.setCellWidget(i, m, dateTime)

            m += 1
            # END TIME - defaults to 23:59:59 (full day)
            dateTime = QDateTimeEdit()
            dateTime.setDisplayFormat("hh:mm")
            dateTime.setTime(QtCore.QTime(23, 59, 59))  # 23:59:59
            dateTime.dateTimeChanged.connect(lambda: self.USGS_dateChangeMethod(date_widget, self.table_USGS_Sites))
            dateTime.setKeyboardTracking(False)
            dateTime.setFrame(False)
            tableProducts.setCellWidget(i, m, dateTime)

        if getattr(self, "_usgs_layout_built", False):
            self._usgs_reset_rows()

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def _populate_usgs_sites_tree(self):
        """(Re)build the USGS site tree from self.cameraList.
        Hidden cameras (when shown) are rendered in gray italics.
        """
        self.USGS_listboxSites.blockSignals(True)
        self.USGS_listboxSites.clear()
        checked = getattr(self, "_usgs_rows", {})
        for camID in self.cameraList:
            site_item = QTreeWidgetItem([camID])
            site_item.setData(0, QtCore.Qt.UserRole, camID)
            if getattr(self, "_usgs_layout_built", False):
                site_item.setFlags(site_item.flags() | QtCore.Qt.ItemIsUserCheckable)
                site_item.setCheckState(0, QtCore.Qt.Checked if camID in checked else QtCore.Qt.Unchecked)

            # Visually distinguish hidden cameras when they are displayed
            try:
                if self.myHIVIS.is_hidden(camID):
                    font = site_item.font(0)
                    font.setItalic(True)
                    site_item.setFont(0, font)
                    site_item.setForeground(0, QtGui.QBrush(QtGui.QColor("gray")))
                    site_item.setToolTip(0, "Hidden camera (hideCam=True)")
            except Exception:
                pass

            for line in self.myHIVIS.get_camera_info(camID):
                child = QTreeWidgetItem([line])
                child.setFlags(child.flags() & ~QtCore.Qt.ItemIsSelectable)
                site_item.addChild(child)
            placeholder = QTreeWidgetItem(["  Time series: loading..."])
            placeholder.setData(0, QtCore.Qt.UserRole, "__nwis_placeholder__")
            placeholder.setFlags(placeholder.flags() & ~QtCore.Qt.ItemIsSelectable)
            site_item.addChild(placeholder)
            self.USGS_listboxSites.addTopLevelItem(site_item)

        self.USGS_listboxSites.collapseAll()
        self.USGS_listboxSites.blockSignals(False)
        if hasattr(self, "_usgs_apply_filter"):
            self._usgs_apply_filter()

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def _usgs_sites_tab_index(self):
        """Return the tabWidget index of the tab containing the USGS site list."""
        widget = self.USGS_listboxSites
        while widget is not None:
            idx = self.tabWidget.indexOf(widget)
            if idx != -1:
                return idx
            widget = widget.parentWidget()
        return -1

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def _toggle_usgs_hidden_cameras(self):
        """Toggle display of hidden USGS cameras and rebuild the site tree.
        Triggered by double-clicking the USGS Sites tab.
        """
        if not getattr(self, "_usgs_startup_ready", False) or self.myHIVIS is None:
            return

        show_hidden = self.myHIVIS.toggle_show_hidden()

        # Refresh the cached dictionary/list to reflect the new filter
        self.cameraDictionary = self.myHIVIS.get_camera_dictionary()
        self.cameraList       = self.myHIVIS.get_camera_list()

        self._populate_usgs_sites_tree()

        state = "SHOWN" if show_hidden else "HIDDEN"
        print(f"[USGS] Hidden cameras {state} - {len(self.cameraList)} cameras listed")
        try:
            self.statusBar().showMessage(
                f"USGS hidden cameras {state.lower()} ({len(self.cameraList)} cameras)", 4000)
        except Exception:
            pass

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
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
            self._show_image_message("usgs", "Loading midday image...")
            if self._usgs_image_fetcher and self._usgs_image_fetcher.isRunning():
                self._usgs_image_fetcher.quit()
                self._usgs_image_fetcher.wait()
            self._usgs_image_fetcher = USGSLatestImageFetcher(self.usgs, strCamID, parent=self)
            self._usgs_image_fetcher.result.connect(self._on_usgs_latest_image_received)
            self._usgs_image_fetcher.start()

            # Update table (row 0 is "All checked sites" in the code-built layout)
            if not getattr(self, "_usgs_layout_built", False):
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
            self._show_image_message("usgs", "No image available.")
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
    def _show_image_message(self, tab, text):
        """Text in place of the image on the "neon" or "usgs" tab (zoom panel if built, else the label)."""
        panel = getattr(self, f"{tab}_image_panel", None)
        if panel is not None:
            panel.showMessage(text)
        else:
            (self.NEON_labelLatestImage if tab == "neon" else self.USGS_labelLatestImage).setText(text)

    def USGS_DisplayLatestImage(self):
        panel = getattr(self, "usgs_image_panel", None)
        if panel is not None:
            if self.USGS_latestImage != []:
                panel.setPixmap(self.USGS_latestImage)   # same image keeps the user's zoom
            return
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
        if getattr(self, "_usgs_layout_built", False):
            return   # counts are shown per checked camera in the table

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
            myApp_Color = Color()

            # CREATE AN ROI OBJECT
            roiObj = roiData()

            # POPULATE ROI OBJECT WITH ROI INFORMATION
            if len(roiParameters.strROIName) > 0:
                roiObj.setROIName(roiParameters.strROIName)
            else:
                msgBox = App_QMessageBox('ROI Error', 'A name for the ROI is required!', buttons=QMessageBox.Close)
                response = msgBox.displayMsgBox()
                return

            # --------------------------------------------------
            try:
                roiShape = ROIShape(getattr(roiParameters, 'roiShape', 0))
            except ValueError:
                roiShape = ROIShape.RECTANGLE

            if roiShape == ROIShape.RECTANGLE:
                rectROI = self.labelOriginalImage.getROI()
            else:
                # Polygon / free-form: use the last closed outline drawn on the canvas.
                poly = self.labelOriginalImage.getLastColorSegPolygon()
                if poly and len(poly) >= 3:
                    qpts = [QPoint(int(x), int(y)) for (x, y) in poly]
                    roiObj.setDisplayPolygon(qpts)
                    rectROI = QPolygon(qpts).boundingRect()
                else:
                    rectROI = None

            if rectROI != None:
                roiObj.setDisplayROI(rectROI)
            else:
                msgBox = App_QMessageBox('ROI Error', 'Please draw the ROI on the image!', buttons=QMessageBox.Close)
                response = msgBox.displayMsgBox()
                return

            # --------------------------------------------------
            try:
                roiObj.setImageSize(currentImage.size())
                scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                roiObj.setDisplaySize(scaledCurrentImage.size())
                roiObj.calcROI()

                roiObj.setROIShape(roiShape)
            except Exception:
                msgBox = App_QMessageBox('ROI Error',
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
            #JES qImg, clusterCenters, hist = myApp_Color.KMeans(rgb, roiObj.getNumColorClusters())
            #JES roiObj.setClusterCenters(clusterCenters, hist)

            img1 = App_Utils().convertQImageToMat(currentImage.toImage())
            if not self._addTrainedROI(roiObj, img1):
                msgBox = App_QMessageBox('ROI Error', 'The ROI contains too few pixels. Please draw a larger ROI.', buttons=QMessageBox.Close)
                response = msgBox.displayMsgBox()
                return

            self._refreshROIDisplay()

    # ==================================================================================================================
    # Train a fully-positioned ROI on the current image and add it to the ROI list and table.
    # Returns False (nothing added) if the ROI has too few pixels.
    # ==================================================================================================================
    def _addTrainedROI(self, roiObj, img1):
        myApp_Color = Color()

        # EXTRACT DOMINANT HSV COLORS AND ADD THEM TO THE ROI OBJECT
        # Only pixels inside the ROI (rectangle, polygon or free-form) are used.
        rgb = roiObj.insidePixels(img1)
        if rgb is None or rgb.shape[0] < roiObj.getNumColorClusters():
            return False
        hist, colorClusters = myApp_Color.extractDominant_HSV(rgb, roiObj.getNumColorClusters())
        roiObj.setHSVClusterCenters(colorClusters, hist)

        roiObj.setTrainingImageName(currentImageFilename)

        self.roiList.append(roiObj)

        # ----------------------------------------------------------------------------------------------------------
        # DISPLAY IN FEATURE TABLE
        # ----------------------------------------------------------------------------------------------------------
        # CREATE COLOR BAR TO DISPLAY CLUSTER COLORS
        colorBar = Color.create_color_bar(hist, colorClusters)

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
        self.tableWidget_ROIList.setItem(nRow, nCol, QTableWidgetItem(roiObj.getROIName()))

        self.tableWidget_ROIList.resizeColumnsToContents()
        return True

    # ==================================================================================================================
    # Redraw ROI overlays and features after ROIs were added.
    # ==================================================================================================================
    def _refreshROIDisplay(self):
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
    # Import ROIs from an ROI_Masks (COCO 1.0) file. The ROIs are placed on the current image and
    # their color clusters are trained on it, exactly as if they had been drawn.
    # ==================================================================================================================
    def importROIMasks(self):
        global currentImage, imageFileFolder
        from appcore.dialogs.color_segmentation.color_seg_roi_coco_export import load_roi_masks

        parent = self.colorSegmentationDlg if self.colorSegmentationDlg is not None else self

        if not currentImage:
            App_QMessageBox('Import ROI Masks', 'Open an image first.', buttons=QMessageBox.Close).displayMsgBox()
            return

        path, _ = QFileDialog.getOpenFileName(parent, 'Import ROI Masks', imageFileFolder or '',
                                              'ROI mask files (*.json);;All files (*)')
        if not path:
            return

        try:
            data = load_roi_masks(path, os.path.basename(currentImageFilename or ''))
        except Exception as e:
            App_QMessageBox('Import ROI Masks', f'Could not read the file:\n{e}', buttons=QMessageBox.Close).displayMsgBox()
            return

        # The ROI coordinates only make sense on images of the same size.
        img_w, img_h = currentImage.width(), currentImage.height()
        if data['width'] and data['height'] and (data['width'], data['height']) != (img_w, img_h):
            App_QMessageBox('Import ROI Masks',
                                 f"The ROI file was made for {data['width']} x {data['height']} images, "
                                 f"but the current image is {img_w} x {img_h}. Nothing was imported.",
                            buttons=QMessageBox.Close).displayMsgBox()
            return

        # Replace or add to existing ROIs.
        if self.roiList:
            box = QMessageBox(parent)
            box.setWindowTitle('Import ROI Masks')
            box.setText(f'There are already {len(self.roiList)} ROI(s). Replace them or add the imported ROIs?')
            replace_btn = box.addButton('Replace', QMessageBox.AcceptRole)
            add_btn = box.addButton('Add', QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.exec_()
            if box.clickedButton() == replace_btn:
                self.deleteAllROI()
            elif box.clickedButton() != add_btn:
                return

        nClusters = (self.colorSegmentationDlg.get_num_color_clusters()
                     if self.colorSegmentationDlg is not None else self.colorSegmentationParams.numColorClusters)
        scaled = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        disp_w, disp_h = scaled.width(), scaled.height()
        sx = disp_w / img_w if img_w else 1.0
        sy = disp_h / img_h if img_h else 1.0
        shapes = {'rectangle': ROIShape.RECTANGLE, 'polygon': ROIShape.POLYGON, 'freeform': ROIShape.FREEFORM}

        img1 = App_Utils().convertQImageToMat(currentImage.toImage())
        added, too_small = 0, []
        for r in data['rois']:
            roiObj = roiData()
            roiObj.setROIName(r['name'])
            roiObj.setROIShape(shapes.get(r['shape'], ROIShape.POLYGON))
            roiObj.setImageSize(currentImage.size())
            roiObj.setDisplaySize(scaled.size())

            # Image coordinates come straight from the file; display coordinates are derived for drawing.
            x, y, w, h = r['bbox']
            roiObj.setImageROI(QtCore.QRect(x, y, w, h))
            roiObj.setDisplayROI(QtCore.QRect(int(round(x * sx)), int(round(y * sy)),
                                              int(round(w * sx)), int(round(h * sy))))
            if roiObj.getROIShape() != ROIShape.RECTANGLE:
                roiObj.setImagePolygon([QPoint(px, py) for (px, py) in r['polygon']])
                roiObj.setDisplayPolygon([QPoint(int(round(px * sx)), int(round(py * sy)))
                                          for (px, py) in r['polygon']])

            roiObj.setNumColorClusters(nClusters)
            if self._addTrainedROI(roiObj, img1):
                added += 1
            else:
                too_small.append(r['name'])

        self._refreshROIDisplay()

        msg = f'Imported {added} ROI(s) from:\n{path}'
        if too_small:
            msg += f"\n\nSkipped (too few pixels): {', '.join(too_small)}"
        if data['skipped']:
            msg += f"\n\nSkipped {data['skipped']} annotation(s) that are not polygons or boxes."
        App_QMessageBox('Import ROI Masks', msg, buttons=QMessageBox.Close).displayMsgBox()

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
        # Once the code-built NEON layout exists, its table replaces this format.
        if getattr(self, "_neon_layout_built", False):
            self._neon_setup_table()
            tree = self.NEON_listboxSites
            tree.blockSignals(True)
            for i in range(tree.topLevelItemCount()):
                branch = self._neon_branch(tree.topLevelItem(i), "products")
                for c in range(branch.childCount() if branch else 0):
                    branch.child(c).setCheckState(0, QtCore.Qt.Unchecked)
            tree.blockSignals(False)
            self._neon_apply_filter()
            return

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
            startTime = datetime.time(nHour, nMinute, 0)          # times are entered to the minute

            endTimeCol = 7
            nHour, nMinute, nSecond = self.separateTime(self.table_USGS_Sites.cellWidget(0, endTimeCol).dateTime().time())
            endTime = datetime.time(nHour, nMinute, 59)           # the end minute is inclusive

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
        if getattr(self, "_usgs_layout_built", False):
            self._usgs_count_rows()
            return
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
    def pushButton_USGSCorrelate_Clicked(self):
        """Correlate downloaded USGS images with the sidecar NWIS sensor file and write a report of
        matches, misalignments, and coverage gaps. The chosen folder can be:
          - a site folder holding Images and data      -> that site
          - a folder of site folders, each with both   -> every site, plus a summary of all of them
          - a folder of images (as before)             -> asks for the sensor file
        """
        start_dir = self.edit_USGSSaveFilePath.text().strip() or (JsonEditor().getValue("USGS_Root_Folder") or "")

        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select a site folder (Images + data), a folder of site folders, or a folder of images",
            start_dir)
        if not folder:
            return

        sites = self._usgs_correlation_sites(folder)
        jobs = []   # (site name, image folder, sensor file or None)
        if sites:
            for name, images, data in sites:
                jobs.append((name, images, self._usgs_sensor_file(data)))
            if len(jobs) == 1 and jobs[0][2] is None:
                sensor_file, _filter = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Select the NWIS sensor data file (.txt or .csv)", sites[0][2],
                    "NWIS sensor data (*.txt *.csv);;All files (*.*)")
                if not sensor_file:
                    return
                jobs[0] = (jobs[0][0], jobs[0][1], sensor_file)
        else:
            # The chosen folder is the image folder itself (original behavior)
            sensor_file, _filter = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select the NWIS sensor data file (.txt or .csv)",
                os.path.dirname(folder),
                "NWIS sensor data (*.txt *.csv);;All files (*.*)")
            if not sensor_file:
                return
            jobs.append((os.path.basename(os.path.normpath(folder)), folder, sensor_file))

        # Match tolerance: seconds entered directly; 0 = automatic
        # (half the sensor sampling interval, e.g. 450 s for 15-min data).
        tol_seconds, ok = QtWidgets.QInputDialog.getDouble(
            self, "Match tolerance",
            "Maximum time difference between an image and a sensor reading\n"
            "for them to be considered aligned, in SECONDS\n"
            "(e.g. 90 = 1.5 minutes; 0 = automatic: half the sensor interval):",
            0.0, 0.0, 86400.0, 1)
        if not ok:
            return
        tolerance_minutes = (tol_seconds / 60.0) if tol_seconds > 0 else None

        from appcore.QProgressWheel import QProgressWheel
        from appcore.SensorImageCorrelator import SensorImageCorrelator
        from appcore.reporting.gap_report import reports_folder_for
        results = []   # (site, status, csv path, xlsx path, sensor file)
        for i, (name, image_folder, sensor_file) in enumerate(jobs, start=1):
            prefix = f"{name} (site {i} of {len(jobs)}): " if len(jobs) > 1 else ""
            if sensor_file is None:
                results.append((name, "No sensor file in data folder", "", "", ""))
                continue

            progressBar = QProgressWheel(0, 1000)
            progressBar.setWindowTitle(prefix + "Correlating images with sensor data...")
            # Keep the wheel visible above the main window
            progressBar.setWindowFlags(progressBar.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
            progressBar.show()

            def _progress(done, total, label, _bar=progressBar, _prefix=prefix):
                _bar.setValue(int(done * 1000 / max(total, 1)))
                if label:
                    _bar.setWindowTitle(_prefix + str(label))
                QApplication.processEvents()

            try:
                # Reports go in a "correlation" folder: the sister of Images (<site>/correlation),
                # or a subfolder of any other image folder, never among the images.
                csv_path, xlsx_path = SensorImageCorrelator().correlate(
                    image_folder, sensor_file,
                    output_folder=reports_folder_for(image_folder, "correlation"),
                    tolerance_minutes=tolerance_minutes, progress=_progress)
                results.append((name, "OK", csv_path, xlsx_path, sensor_file))
            except Exception as e:
                results.append((name, f"Failed: {e}", "", "", sensor_file))
            finally:
                # Always dismiss the progress wheel, whatever happened above
                progressBar.close()

        if len(results) == 1:
            name, status, csv_path, xlsx_path, _sensor = results[0]
            if status == "OK":
                QtWidgets.QMessageBox.information(
                    self, "Sensor/Image Correlation",
                    "Correlation report written:\n\n"
                    f"{csv_path}\n{xlsx_path}\n\n"
                    "The xlsx contains Image Correlation, Sensor Coverage, Gaps, "
                    "and Summary worksheets. Unmatched rows are highlighted.")
            else:
                QtWidgets.QMessageBox.critical(self, "Sensor/Image Correlation",
                                               f"Correlation failed for {name}:\n{status}")
            return

        # Several sites: one summary of all of them in the chosen folder, named by run time
        import csv as _csv
        summary_path = os.path.join(folder, f"CorrelationSummary_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
        try:
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["Site", "Result", "Correlation CSV", "Correlation Workbook", "Sensor File"])
                w.writerows(results)
        except Exception as e:
            summary_path = f"(could not write summary: {e})"
        n_ok = sum(1 for r in results if r[1] == "OK")
        lines = [f"{r[0]}: {r[1]}" for r in results]
        QtWidgets.QMessageBox.information(
            self, "Sensor/Image Correlation",
            f"Correlated {n_ok} of {len(results)} sites.\n\n" + "\n".join(lines) +
            f"\n\nSummary of all sites:\n{summary_path}\n\n"
            "The summary lists each site's report files.")

    def pushButton_USGSDownloadClicked(self):

        # VERIFY THAT THE FOLDER HAS BEEN SPECIFIED
        USGS_download_file_path = self.edit_USGSSaveFilePath.text()
        JsonEditor().update_json_entry("USGS_Root_Folder", USGS_download_file_path)

        if len(USGS_download_file_path) == 0:
            strMessage = f'A download folder has not been specified. Would you like to use the last {APP_NAME} USGS download folder used?'
            msgBox = App_QMessageBox('USGS Root Download Folder', strMessage, QMessageBox.Yes | QMessageBox.No)
            response = msgBox.displayMsgBox()

            if response == QMessageBox.Yes:
                #USGS_download_file_path = os.path.expanduser('~')
                #USGS_download_file_path = os.path.join(USGS_download_file_path, 'Documents')

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

        # Checked cameras: download each (code-built layout)
        if getattr(self, "_usgs_layout_built", False) and self._usgs_rows:
            self._usgs_download_checked(USGS_download_file_path)
            return

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
            startTime = datetime.time(nHour, nMinute, 0)          # times are entered to the minute

            endTimeCol = 7
            nHour, nMinute, nSecond = self.separateTime(self.table_USGS_Sites.cellWidget(0, endTimeCol).dateTime().time())
            endTime = datetime.time(nHour, nMinute, 59)           # the end minute is inclusive

            nwisID = self.myHIVIS.get_nwisID()
            if getattr(self, "_usgs_layout_built", False):
                # Image counts for other cameras change HIVIS's "last camera", so take this
                # camera's NWIS ID from its own record, and build its time-zone-aware image
                # list (counting no longer happens on a site click).
                nwisID = self._usgs_nwis_id(site)
                self.myHIVIS.get_image_count(siteName=site, nwisID=nwisID, startDate=startDate, endDate=endDate,
                                             startTime=startTime, endTime=endTime)

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

        settings_folder = Save_Utils().get_settings_folder()
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

        from appcore.dialogs.ML_image_processing.ML_ImageProcessingDlg import ML_ImageProcessingDlg
        hyperparameterDlg = ML_ImageProcessingDlg(frame)

        _splash.finish(hyperparameterDlg)

        hyperparameterDlg.ml_train_signal.connect(train_main)
        hyperparameterDlg.ml_segment_signal.connect(segment_main)

        # Training-images folder change -> offer to update the active recipe
        try:
            hyperparameterDlg.training_tab.trainingImagesCommitted_Signal.connect(
                frame.on_training_images_committed)
            hyperparameterDlg.finished.connect(
                lambda *_: frame.on_training_images_committed(
                    hyperparameterDlg.training_tab.lineEdit_model_training_images_path.text()))
        except Exception as _e:
            print(f"[WARN] Could not wire training-images recipe prompt: {_e}")

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
                self.NEON_listboxSites.addTopLevelItem(self._neon_build_site_item(site))

            self.NEON_listboxSites.collapseAll()
            self._neon_apply_filter()

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
            strMessage = f'You must first create a list of images to operate on. Use the FETCH files feature of {APP_DISPLAY_NAME}.'
            msgBox = App_QMessageBox('Composite Slice Error', strMessage, QMessageBox.Close)
            response = msgBox.displayMsgBox()
        else:
            imageFilename = dailyImagesList.getVisibleList()[currentImageIndex].fullPathAndFilename

            if self.compositeSliceDlg is None:
                from appcore.dialogs.composite_slice.CompositeSliceDlg import CompositeSliceDlg
                self.compositeSliceDlg = CompositeSliceDlg()
                self.compositeSliceDlg.compositeSliceGenerateSignal.connect(self.generateCompositeSlices)
                self.compositeSliceDlg.compositeSliceCancelSignal.connect(self.closeCompositeSlices)
                self.compositeSliceDlg.label_Image.setDrawingMode(DrawingMode.SLICE)

            self.compositeSliceDlg.loadImage(imageFilename)
            self.compositeSliceDlg.show()
            self.compositeSliceDlg.raise_()

    def generateCompositeSlices(self):
        print("Generating composite slices image(s)...")

        global imageFileFolder
        composite_slices_folder = Save_Utils().create_composite_slices_folder(imageFileFolder)

        slice_rect = self.compositeSliceDlg.label_Image.getSliceRectInOriginal()

        from appcore.CompositeSlices import CompositeSlices
        compositeSlices = CompositeSlices(slice_rect)
        compositeSlices.create_composite_image(dailyImagesList.visibleList, composite_slices_folder)

    def closeCompositeSlices(self):
        if self.compositeSliceDlg != None:
            self.compositeSliceDlg.close()
            self.compositeSliceDlg    = None

    def menubar_Generate_Greenness_Test_Images(self):
        # initialize with default settings

        rootFolder = os.path.join(str(USER_ROOT), 'Test Images')
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
        msgBox = App_QMessageBox('Download Image Files', strMessage, QMessageBox.Yes | QMessageBox.No)
        response = msgBox.displayMsgBox()

        if response == QMessageBox.Yes:
            prompter = promptlib.Files()
            folder = prompter.dir()

            if len(folder) == 0:
                strMessage = 'ERROR! Please specify an image folder containing images to triage.'
                msgBox = App_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
                response = msgBox.displayMsgBox()
            else:
                TriageDlg = TriageOptionsDlg(folder=folder)

                response = TriageDlg.exec_()

                if response == 1:

                    if len(TriageDlg.getReferenceImageFilename()) == 0 and TriageDlg.getCorrectAlignment() == True:
                        strMessage = 'Please select reference image if you want to correct image alignment.'
                        msgBox = App_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
                        response = msgBox.displayMsgBox()
                    else:
                        myTriage = ImageTriage()
                        myTriage.cleanImages(folder, \
                                             False, \
                                             TriageDlg.getBrightnessMin(), TriageDlg.getBrightnessMax(), \
                                             TriageDlg.getCreateReport(), TriageDlg.getMoveImages(), \
                                             TriageDlg.getCorrectAlignment(), TriageDlg.getSavePolylines(),
                                             TriageDlg.getReferenceImageFilename(), TriageDlg.getRotationThreshold(),
                                             use_fft_blur=TriageDlg.getUseFftBlur(),
                                             use_laplacian=TriageDlg.getUseLaplacian(),
                                             laplacian_threshold=TriageDlg.getLaplacianThreshold(),
                                             focus_roi=TriageDlg.getFocusROI(),
                                             use_color_imbalance=TriageDlg.getUseColorImbalance(),
                                             color_imbalance_threshold=TriageDlg.getColorImbalanceThreshold(),
                                             fft_calibration=TriageDlg.getFftCalibration())

                        strMessage = 'Image triage is complete!'
                        msgBox = App_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
                        response = msgBox.displayMsgBox()
                else:
                    strMessage = 'ABORT! You cancelled the triage operation.'
                    msgBox = App_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
                    response = msgBox.displayMsgBox()
        else:
            strMessage = 'ABORT! You cancelled the triage operation.'
            msgBox = App_QMessageBox('Image Triage', strMessage, buttons=QMessageBox.Close)
            response = msgBox.displayMsgBox()


    def menubarExtractCOCOMasks(self):
        from appcore.dialogs.extract_coco_masks.ExportCOCOMasksDlg import ExportCOCOMasksDlg
        self.COCOdlg = ExportCOCOMasksDlg(self)

        self.COCOdlg.COCO_signal_ok.connect(self.accepted_COCODlg)
        self.COCOdlg.COCO_signal_cancel.connect(self.rejected_COCODlg)

        self.COCOdlg.show()


    def accepted_COCODlg(self):
        image_dir = self.COCOdlg.getAnnotationImagesFolder()
        output_dir = os.path.join(image_dir, "training_masks")

        #JES CLEANUP TASK: THE ANNOTATION FILE IS NOT REQUIRED TO BE IN THE TRAINING IMAGES FOLDER.
        utils = COCO_Utils(image_dir)
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
            util = COCO_Utils(selected_dir)
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
            utils = COCO_Utils(selected_dir)
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
        utils = Save_Utils()
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
    # RECIPE MANAGER — per-site folder recipes
    # ==================================================================================================================
    def menubar_recipe_manager(self):
        """Open the Recipe Manager. A recipe bundles the per-site folders
        (root, composite slices, videos/GIFs, USGS, NEON downloads) so you can
        switch study sites without editing folder paths by hand."""
        try:
            from appcore.recipe_manager import RecipeManagerDialog
            dlg = RecipeManagerDialog(self._get_recipe_store(), self, dark_mode=self._is_dark_mode)
            dlg.recipeActivated.connect(self.apply_recipe)
            dlg.recipeDeactivated.connect(self.clear_recipe)
            dlg.exec_()
        except Exception as e:
            print(f"[ERROR] Failed to open Recipe Manager: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Recipe Manager", str(e))

    def menubar_site_config_editor(self):
        """Open the standalone Site Config editor (Tools -> Site Config Editor)."""
        try:
            from appcore.utils.site_config_manager import open_editor
            open_editor(parent=self)
        except Exception as e:
            print(f"[ERROR] Failed to open Site Config Editor: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Site Config Editor", str(e))

    def _build_plugins_menu(self):
        """Fill Tools > Plugins with the plugins that asked for the Tools surface.

        Nothing here is allowed to take the application down: a missing plugin
        system or an unreadable plugins folder just leaves the menu empty."""
        self._menu_plugins.clear()
        try:
            from appcore.plugins import discover, open_in_window, plugins_folder, SURFACE_TOOLS
        except Exception as e:
            print(f"[ERROR] Plugin support unavailable: {e}")
            traceback.print_exc()
            action = self._menu_plugins.addAction("Plugins unavailable")
            action.setEnabled(False)
            return

        try:
            found = discover(surface=SURFACE_TOOLS)
        except Exception as e:
            print(f"[ERROR] Could not read the plugins folder: {e}")
            traceback.print_exc()
            found = []

        if not found:
            action = self._menu_plugins.addAction("No plugins installed")
            action.setEnabled(False)
        else:
            for info in found:
                action = self._menu_plugins.addAction(info.title)
                action.setStatusTip(info.meta.get("description", f"Open {info.title}"))
                action.triggered.connect(
                    lambda _checked=False, plugin=info: open_in_window(plugin, parent=self))

        self._menu_plugins.addSeparator()
        folder_action = self._menu_plugins.addAction("Open Plugins Folder\u2026")
        folder_action.setStatusTip(plugins_folder())
        folder_action.triggered.connect(self._open_plugins_folder)

    def _open_plugins_folder(self):
        """Show the plugins folder in the file manager, creating it if needed."""
        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl
        folder = ""
        try:
            from appcore.plugins import plugins_folder
            folder = plugins_folder()
            os.makedirs(folder, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
        except Exception as e:
            print(f"[ERROR] Could not open the plugins folder: {e}")
            QMessageBox.warning(self, "Plugins", f"Could not open:\n{folder}\n\n{e}")

    def _get_recipe_store(self):
        """Return the single shared RecipeStore instance (created lazily), so
        the Recipe Manager and the image-folder write-back use the same store."""
        from appcore.recipe_manager import RecipeStore
        if not hasattr(self, "recipe_store") or self.recipe_store is None:
            self.recipe_store = RecipeStore()
        return self.recipe_store

    def _reconcile_folder_with_recipe(self, new_path, attr, label):
        """Shared prompt: if a recipe is active and `new_path` differs from the
        recipe's `attr`, offer to write it back (persist) or keep it for this
        session only. Used by both the Data Exploration image folder and the
        ML training-images folder."""
        try:
            new_path = (new_path or "").strip()
            if not new_path:
                return
            last = getattr(self, "_last_prompt_paths", None)
            if last is None:
                last = self._last_prompt_paths = {}
            if last.get(attr) == new_path:
                print(f"[INFO] {label} commit: already reconciled this path; skipping.")
                return
            store = self._get_recipe_store()
            active = store.get_active()
            if active is None:
                print(f"[INFO] {label} commit: no active recipe; nothing to reconcile.")
                return
            current = (getattr(active, attr, "") or "").strip()
            if os.path.normpath(new_path) == os.path.normpath(current or "."):
                print(f"[INFO] {label} commit: matches active recipe; no prompt.")
                return
            print(f"[INFO] {label} commit: prompting to update recipe "
                  f"'{active.name}' ({current!r} -> {new_path!r}).")
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Update Recipe?")
            box.setText(f"You changed the {label} folder while recipe "
                        f"\u201c{active.name}\u201d is active.")
            box.setInformativeText(
                "Update the recipe to use this folder, or keep the change just "
                "for this session?")
            update_btn = box.addButton("Update Recipe", QMessageBox.AcceptRole)
            box.addButton("Just This Session", QMessageBox.RejectRole)
            box.exec_()
            last[attr] = new_path  # remember either way so we do not nag again
            if box.clickedButton() is update_btn:
                import datetime
                setattr(active, attr, new_path)
                active.modified = datetime.datetime.now().isoformat(timespec="seconds")
                store.save()
                try:
                    self.statusBar().showMessage(
                        f"Recipe '{active.name}' updated with the new {label} folder.", 5000)
                except Exception:
                    pass
        except Exception as e:
            print(f"[ERROR] reconcile {label} folder failed: {e}")
            traceback.print_exc()

    def on_images_folder_committed(self, new_path):
        """Data Exploration image folder was changed by the user."""
        self._reconcile_folder_with_recipe(new_path, "image_input", "image")

    def on_training_images_committed(self, new_path):
        """ML training-images folder was changed by the user (Training tab)."""
        self._reconcile_folder_with_recipe(new_path, "ml_images", "training images")

    # JSON entries that exist only because a recipe wrote them. With no recipe
    # active they are cleared, so outputs fall back to the default folders.
    # Image, data, download and ML folders are kept as the user's normal settings.
    _RECIPE_ONLY_KEYS = ("Composite_Slices_Folder", "Videos_Folder", "GIFs_Folder", "Recipe_Site_Root")

    def clear_recipe(self):
        """No recipe is active (the recipes themselves are kept): stop applying
        recipe folders and never prompt about recipes on folder changes."""
        try:
            self._last_prompt_paths = {}
            for key in self._RECIPE_ONLY_KEYS:
                JsonEditor().update_json_entry(key, "")
            try:
                self.statusBar().showMessage("No recipe active. Using the normal folder settings.", 5000)
            except Exception:
                pass
            print("[INFO] Recipe cleared: no recipe active.")
        except Exception as e:
            print(f"[ERROR] Failed to clear recipe: {e}")
            traceback.print_exc()

    def apply_recipe(self, recipe):
        """Push an activated recipe's folder paths into the live UI and JSON
        config so downloads and outputs land in the selected site's folders."""
        try:
            # A freshly activated recipe re-enables folder-change prompting.
            self._last_prompt_paths = {}
            # Image input folder -> Data Exploration images folder
            if recipe.image_input:
                JsonEditor().update_json_entry("Local_Image_Folder", recipe.image_input)
                if hasattr(self, "fileFolderDlg") and self.fileFolderDlg is not None:
                    try:
                        self.fileFolderDlg.setImageFolderPath(recipe.image_input)
                    except Exception:
                        pass

            # Data input folder
            if recipe.data_input:
                JsonEditor().update_json_entry("Data_Input_Folder", recipe.data_input)

            # Machine Learning training images -> Training tab
            if recipe.ml_images:
                JsonEditor().update_json_entry("Model_Training_Images_Folder", recipe.ml_images)
                try:
                    if hyperparameterDlg is not None and hasattr(hyperparameterDlg, "training_tab"):
                        hyperparameterDlg.training_tab.lineEdit_model_training_images_path.setText(recipe.ml_images)
                except Exception:
                    pass

            # Segmentation predictions output -> Segment Images tab
            if recipe.predictions:
                JsonEditor().update_json_entry("Model_Segmentation_Output_Folder", recipe.predictions)
                try:
                    if hyperparameterDlg is not None:
                        from appcore.dialogs.ML_image_processing.segment_images_tab import SegmentImagesTab
                        seg_tab = hyperparameterDlg.findChild(SegmentImagesTab)
                        if seg_tab is not None:
                            seg_tab.set_output_folder(recipe.predictions)
                except Exception:
                    pass

            # USGS download root
            if recipe.usgs:
                if hasattr(self, "edit_USGSSaveFilePath"):
                    self.edit_USGSSaveFilePath.setText(recipe.usgs)
                JsonEditor().update_json_entry("USGS_Root_Folder", recipe.usgs)

            # NEON download root
            if recipe.neon:
                if hasattr(self, "edit_NEONSaveFilePath"):
                    self.edit_NEONSaveFilePath.setText(recipe.neon)
                if hasattr(self, "edit_NEON_TableInput"):
                    self.edit_NEON_TableInput.setText(recipe.neon)
                JsonEditor().update_json_entry("NEON_Root_Folder", recipe.neon)

            # PhenoCam download root -> PhenoCam tab
            if recipe.phenocam:
                if hasattr(self, "phenocam_folder_edit"):
                    self.phenocam_folder_edit.setText(recipe.phenocam)
                JsonEditor().update_json_entry("Phenocam_Root_Folder", recipe.phenocam)

            # Composite / video / GIF outputs have no dedicated widgets yet;
            # persist to JSON so the output pipeline (Save_Utils)
            # can redirect them. Videos and GIFs use separate folders.
            if recipe.composites:
                JsonEditor().update_json_entry("Composite_Slices_Folder", recipe.composites)
            if recipe.videos:
                JsonEditor().update_json_entry("Videos_Folder", recipe.videos)
            if recipe.gifs:
                JsonEditor().update_json_entry("GIFs_Folder", recipe.gifs)

            # Site root (informational; downstream code may key off this).
            if recipe.root:
                JsonEditor().update_json_entry("Recipe_Site_Root", recipe.root)

            try:
                self.statusBar().showMessage(f"Recipe '{recipe.name}' applied.", 5000)
            except Exception:
                pass
            print(f"[INFO] Recipe applied: {recipe.name}")
        except Exception as e:
            print(f"[ERROR] Failed to apply recipe: {e}")
            traceback.print_exc()


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
            from appcore.dialogs.image_organizer.ImageOrganizerDlg import ImageOrganizerDlg
            self.imageOrganizerDlg = ImageOrganizerDlg(self)
            print("[DEBUG] Showing Image Organizer dialog")
            self.imageOrganizerDlg.show()
            print("[INFO] Image Organizer dialog launched successfully")
        except Exception as e:
            print(f"[ERROR] Failed to launch Image Organizer dialog: {e}")
            traceback.print_exc()  # full traceback to terminal


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
        from appcore.dialogs.release_notes.ReleaseNotesDlg import ReleaseNotesDlg
        releaseNotesDlg = ReleaseNotesDlg(frame)

        releaseNotesDlg.show()


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def toolbarButtonGRIME2(self):
        strMessage = 'Potential future home for GRIME2 Water Level/Stage measurement functionality.'
        msgBox = App_QMessageBox('Water Level Measurement', strMessage, QMessageBox.Close)
        response = msgBox.displayMsgBox()


    # ======================================================================================================================
    # ======================================================================================================================
    # ======================================================================================================================
    def toolbarButtonEdgeDetection(self):
        global frame
        from appcore.dialogs.edge_detection.EdgeDetectionDlg import EdgeDetectionDlg
        self.edgeDetectionDlg = EdgeDetectionDlg(frame)

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
        global g_featureMethodSettings

        # Symmetry with featureDetectionMethod(), which already clears the edge method.
        # Without this, selecting SIFT and then Canny left the feature method set to SIFT.
        g_featureMethodSettings.method = featureMethodsClass.NONE

        g_edgeMethodSettings = edgeMethod

        # refreshImage() calls processImage() itself; the former processLocalImage() call
        # here made that happen twice per parameter change.
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
        from appcore.dialogs.file_utilities.FileUtilitiesDlg import FileUtilitiesDlg
        self.fileFolderDlg = FileUtilitiesDlg(frame)

        self.fileFolderDlg.create_composite_slice_signal.connect(self.menubarCompositeSlices)
        self.fileFolderDlg.triage_images_signal.connect(self.toolbarButtonImageTriage_1)

        self.fileFolderDlg.accepted.connect(self.closeFilefolderDlg)
        self.fileFolderDlg.rejected.connect(self.closeFilefolderDlg)

        # An active recipe is the source of truth for the image folder. The
        # dialog's constructor seeds itself from the last-saved
        # Local_Image_Folder, which drifts whenever the user browses elsewhere;
        # override it here so re-opening Data Exploration reflects the active
        # recipe. Blank image_input leaves the JSON-derived value untouched.
        try:
            active = self._get_recipe_store().get_active()
            if active is not None and active.image_input:
                self.fileFolderDlg.setImageFolderPath(active.image_input)
        except Exception:
            pass

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
                from appcore.dialogs.image_navigation.ImageNavigationDlg import ImageNavigationDlg
                self.imageNavigationDlg = ImageNavigationDlg(frame)
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
                msgBox = App_QMessageBox('Image Navigation', strMessage, QMessageBox.Close)
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

                from appcore.dialogs.color_segmentation.ColorSegmentationDlg import ColorSegmentationDlg
                self.colorSegmentationDlg = ColorSegmentationDlg()

                self.colorSegmentationDlg.colorSegmentation_Signal.connect(self.colorSegmentation)
                self.colorSegmentationDlg.addROI_Signal.connect(self.trainROI)
                self.colorSegmentationDlg.deleteAllROI_Signal.connect(self.deleteAllROI)
                self.colorSegmentationDlg.buildFeatureFile_Signal.connect(self.buildFeatureFile)
                self.colorSegmentationDlg.exportROIMasks_Signal.connect(self.exportROIMasks)
                self.colorSegmentationDlg.importROIMasks_Signal.connect(self.importROIMasks)
                self.colorSegmentationDlg.universalTestButton_Signal.connect(self.universalTestButton)
                self.colorSegmentationDlg.refresh_rois_signal.connect(self.displayROIs)

                self.colorSegmentationDlg.greenness_index_signal.connect(self.greenness_index_changed)

                self.colorSegmentationDlg.close_signal.connect(self.closeColorSegmentationDlg)
                self.colorSegmentationDlg.accepted.connect(self.closeColorSegmentationDlg)
                self.colorSegmentationDlg.rejected.connect(self.closeColorSegmentationDlg)

                # ROI shape (rectangle / polygon / free-form) -> canvas drawing tool
                self.colorSegmentationDlg.roiShapeChanged_signal.connect(self.roiShapeChanged)
                self.roiShapeChanged(self.colorSegmentationDlg.get_roi_shape())

                self.getColorSegmentationParams()

                self.colorSegmentationDlg.show()
            else:
                strMessage = 'Please close the Mask Editor toolbox if you want to use the Mask Editor toolbox.\nThis will be resolved in a future design change.'
                msgBox = App_QMessageBox('Tool Conflict', strMessage, QMessageBox.Yes | QMessageBox.No)
                response = msgBox.displayMsgBox()


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def roiShapeChanged(self, shape):
        try:
            self.labelOriginalImage.setROIShape(ROIShape(shape))
        except ValueError:
            self.labelOriginalImage.setROIShape(ROIShape.RECTANGLE)

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

        myColor = Color()

        nImageIndex = 1

        if len(videoFileList) > 0:
            if nImageIndex > gFrameCount:
                nImageIndex = gFrameCount

            inputFrame = videoFileList[nImageIndex - 1].fullPathAndFilename  # zero based index

            if os.path.isfile(inputFrame):
                global currentImageFilename
                currentImageFilename = inputFrame
                numpyImage = myColor.loadColorImage(inputFrame)

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
        msgBox = App_QMessageBox('Tool Conflict', strMessage, QMessageBox.Yes | QMessageBox.No)
        response = msgBox.displayMsgBox(on_top=True)
        return
        # ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^     ^

        if self.maskEditorDlg == None:
            if self.colorSegmentationDlg == None:
                self.labelOriginalImage.setDrawingMode(DrawingMode.MASK)

                from appcore.dialogs.mask_editor.MaskEditorDlg import MaskEditorDlg
                self.maskEditorDlg = MaskEditorDlg()

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
                msgBox = App_QMessageBox('Tool Conflict', strMessage, QMessageBox.Yes | QMessageBox.No)
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
        img1 = App_Utils().convertQImageToMat(currentImage.toImage())

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
                msgBox = App_QMessageBox('Save Mask Files', strMessage, QMessageBox.Yes | QMessageBox.No)
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
                myColor = Color()

                _, _, hist = myColor.KMeans(masked, 6)

                # EXTRACT DOMINANT HSV COLORS
                hist, colorClusters = myColor.extractDominant_HSV(masked, 6)

                # CREATE COLOR BAR TO DISPLAY CLUSTER COLORS
                colorBar = Color.create_color_bar(hist, colorClusters[0:5])

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
        self.labelOriginalImage.clearColorSegPolygons()
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

        # Double-click on the USGS Sites tab (the tab itself) toggles hidden cameras
        if (event.type() == QtCore.QEvent.MouseButtonDblClick
                and source is self.tabWidget.tabBar()):
            idx = self.tabWidget.tabBar().tabAt(event.pos())
            if idx != -1 and idx == self._usgs_sites_tab_index():
                self._toggle_usgs_hidden_cameras()
                return True

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
            img = App_Utils().convertQImageToMat(currentImage.toImage())

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

        # With the code-built layout, products are chosen by checking them in the site
        # tree, so clicking a site no longer fills the table with default products.
        auto_select = not getattr(self, "_neon_layout_built", False)

        #JES - TEMPORARILY SET NITRATE DATA ('should only be one nitrate product') AS THE DEFAULT SELECTION
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        itemNitrate = self.NEON_listboxSiteProducts.findItems('Nitrate', QtCore.Qt.MatchContains)
        nIndex = 0
        if auto_select and len(itemNitrate) > 0:
            for item in itemNitrate:
                nIndex = self.NEON_listboxSiteProducts.row(item)
                self.NEON_listboxSiteProducts.setCurrentRow(nIndex)

            NEON_updateProductTable(self, nIndex)

        #JES - TEMPORARILY SET NITRATE DATA ('should only be one nitrate product') AS THE DEFAULT SELECTION
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        item20002 = self.NEON_listboxSiteProducts.findItems('20002', QtCore.Qt.MatchContains)
        nIndex = 0
        if auto_select and len(item20002) > 0:
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
            self._show_image_message("neon", "No image available.")
            if error_msg:
                print(f"[NEON] Latest image: {error_msg}")

    def _neon_tree_item_clicked(self, item, column):
        """Open URL in browser if the clicked NEON tree item is a hyperlink."""
        url = item.data(0, QtCore.Qt.UserRole + 1)
        if url:
            from PyQt5.QtGui import QDesktopServices
            from PyQt5.QtCore import QUrl
            QDesktopServices.openUrl(QUrl(url))

    def NEON_DisplayLatestImage(self):
        panel = getattr(self, "neon_image_panel", None)
        if panel is not None:
            if self.NEON_latestImage == []:
                panel.showMessage("No image available.")
            else:
                panel.setPixmap(self.NEON_latestImage)   # same image keeps the user's zoom
            return

        if self.NEON_latestImage == []:
            self.NEON_labelLatestImage.setText("No image available.")
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
        hist, colorClusters = Color.extractDominant_HSV(
            img, self.colorSegmentationParams.numColorClusters
        )
        colorBar = Color.create_color_bar(hist, colorClusters)

        # Compute greenness values
        greenness_values = []
        for index, greenness in enumerate(self.greenness_index_list):
            greenness_updated = Vegetation_Indices().get_greenness(greenness, img)
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
                # ROI overlays drawn by App_QLabel.paintEvent via savedROIs
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
        # Only pixels inside the ROI (rectangle, polygon or free-form) are used.
        rgb = roiObj.insidePixels(img)
        if rgb is None:
            raise ValueError('ROI does not overlap the image')
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Dominant HSV colors
        hist, colorClusters = Color.extractDominant_HSV(rgb, roiObj.getNumColorClusters())
        colorBar = Color.create_color_bar(hist, colorClusters)

        # Greenness indices
        greenness_values = []
        for index, greenness in enumerate(self.greenness_index_list):
            greenness_updated = Vegetation_Indices().get_greenness(greenness, rgb)
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
    imageCount = App_Utils().get_image_count(filePath, extensions)

    # RECURSE AND TRAVERSE FROM THE SPECIFIED FOLDER DOWN TO DETERMINE THE DATE RANGE FOR THE IMAGES FOUND
    file_count, files = App_Utils().getFileList(filePath, extensions, bFetchRecursive)

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
            fileDate, fileTime = TimeStamp_Utils().extractDateFromFilename(file)

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
                    timeOriginal = re.search(r' \d{2}:\d{2}:\d{2}', strTemp).group(0)

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

    myColor = Color()

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
            numpyImage = myColor.loadColorImage(inputFrame)

            numpyImage = np.ascontiguousarray(numpyImage)
            bytes_per_line = numpyImage.strides[0]
            tempCurrentImage = QImage(numpyImage.data, numpyImage.shape[1], numpyImage.shape[0], bytes_per_line, QImage.Format_RGB888)
            currentImage = QPixmap.fromImage(tempCurrentImage.copy())

    # ------------------------------------------------------------------------------------------------------------------
    # DISPLAY IMAGE FROM NEON SITE
    # ------------------------------------------------------------------------------------------------------------------
    if currentImage:
        numpyImg = App_Utils().convertQImageToMat(currentImage.toImage())

        scaledCurrentImage = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        width = scaledCurrentImage.width()
        height = scaledCurrentImage.height()

        currentImageRescaled = currentImage.scaled(self.labelOriginalImage.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        img = App_Utils().convertQImageToMat(currentImage.toImage())

        self.whole_image_feature_extraction(img)

        self.roi_feature_extraction(img, self.roiList)

        self.labelOriginalImage.setPixmap(currentImageRescaled)

        # REMOVED: `pix = processImage(self, currentImage)` and its companion
        # grayscale conversion. The result was discarded -- the display block below
        # that consumed it is commented out -- while refreshImage() recomputes the
        # identical thing. Every parameter change was running the pipeline twice.

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
# PREPROCESSING CONFIGURATION FOR EDGE / FEATURE DETECTION
#
# Set USE_CLAHE True to apply locally-adaptive contrast enhancement before edge detection.
# Unlike the global cv2.equalizeHist() that was used here previously, CLAHE does not rescale
# the intensity distribution on a per-frame basis, so threshold values remain meaningful and
# reproducible across frames. Recommended for low-contrast water / sandbar boundaries.
# ======================================================================================================================
USE_CLAHE  = False
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# Pre-blur kernel size. Must be odd. 5 is the standard choice ahead of Canny.
# This was formerly 15, which destroyed the very gradients being detected.
PREBLUR_KSIZE = 5


# ======================================================================================================================
# THIS FUNCTION WILL PROCESS THE CURRENT IMAGE BASED UPON THE SETTINGS SELECTED BY THE END-USER.
# THE IMAGE STORAGE TYPE IS QImage
# ======================================================================================================================
def processImage(self, myImage):
    global g_edgeMethodSettings
    global g_featureMethodSettings

    pix = []

    if myImage is None or myImage == []:
        return pix

    # CONVERT IMAGE FROM QImage FORMAT TO Mat FORMAT.
    #
    # convertQImageToMat() returns channel order R, G, B. If that ever changes, change it
    # HERE and nowhere else. The original code disagreed with itself: this function used
    # COLOR_RGB2GRAY while processLocalImage() used COLOR_BGR2GRAY on the same source. The
    # luma weights are 0.299R + 0.587G + 0.114B, so swapping R and B is not cosmetic for
    # river imagery -- water is blue-dominant and sandbars are red/tan-dominant, meaning the
    # wrong conversion directly weakens contrast at the boundary of interest.
    img_rgb = App_Utils().convertQImageToMat(myImage.toImage())

    if img_rgb is None or img_rgb.size == 0:
        return pix

    # NOTE: len(arr) returns the ROW COUNT, not emptiness. The former `len(gray) != 0`
    # guard was always true for any real image.
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # ------------------------------------------------------------------------------------
    # PREPROCESSING
    #
    # Order is: contrast (optional, local only) -> denoise -> detect.
    #
    # Removed from this block:
    #   cv2.equalizeHist()  - global equalization rescaled intensities per frame, which made
    #                         every threshold value frame-dependent and non-reproducible.
    #   (15,15) blur        - erased the gradients the operators are looking for.
    #   cv2.erode(...)      - (7,7) was a shape-(2,) array, NOT a 7x7 structuring element;
    #                         it also wrote into `gray` through the dst argument as a hidden
    #                         side effect. Morphological erosion ahead of an edge operator
    #                         displaces edge LOCATION by roughly the kernel radius, which is
    #                         unacceptable when the edge is a measurement (waterline).
    # ------------------------------------------------------------------------------------
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (PREBLUR_KSIZE, PREBLUR_KSIZE), 0)

    # ------------------------------------------------------------------------------------
    # DISPATCH
    #
    # Edge and feature detection are now separate branches. Previously SIFT and ORB sat in
    # the same elif chain gated on g_edgeMethodSettings.method, so they were only reachable
    # by accident of the dialog zeroing out the edge method first.
    # ------------------------------------------------------------------------------------
    from appcore.ProcessImage import ProcessImage
    myProcessImage = ProcessImage()

    edge_method    = g_edgeMethodSettings.method
    feature_method = g_featureMethodSettings.method

    if edge_method != edgeMethodsClass.NONE:

        if edge_method == edgeMethodsClass.CANNY:
            pix = myProcessImage.processCanny(img_rgb, gray, g_edgeMethodSettings)

        elif edge_method == edgeMethodsClass.LAPLACIAN:
            # WAS: processLaplacian(img1) -- passed the 3-channel color image while every
            # other branch passed grayscale.
            pix = myProcessImage.processLaplacian(gray)

        elif edge_method in (edgeMethodsClass.SOBEL_X,
                             edgeMethodsClass.SOBEL_Y,
                             edgeMethodsClass.SOBEL_XY):
            pix = myProcessImage.processSobel(gray,
                                              g_edgeMethodSettings.getSobelKernel(),
                                              edge_method)

    elif feature_method != featureMethodsClass.NONE:

        if feature_method == featureMethodsClass.SIFT:
            pix = myProcessImage.processSIFT(img_rgb, gray)

        elif feature_method == featureMethodsClass.ORB:
            pix = myProcessImage.processORB(img_rgb, gray, g_featureMethodSettings)

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

    # API fields go under the "Site details" branch when the code-built layout is in use
    details = self._neon_branch(top, "details") if getattr(self, "_neon_layout_built", False) else None
    target = details if details is not None else top

    # Remove any previously-added API children (marked with UserRole+3)
    i = 0
    while i < target.childCount():
        if target.child(i).data(0, QtCore.Qt.UserRole + 3):
            target.removeChild(target.child(i))
        else:
            i += 1

    for label, value in extra_fields:
        if value:
            child = QTreeWidgetItem([f"{label}: {value}"])
            child.setFlags(child.flags() & ~QtCore.Qt.ItemIsSelectable)
            child.setData(0, QtCore.Qt.UserRole + 3, True)
            target.addChild(child)

    if deims_url:
        link_item = QTreeWidgetItem([f"DEIMS ID: {deims_url}"])
        link_item.setData(0, QtCore.Qt.UserRole + 1, deims_url)
        link_item.setData(0, QtCore.Qt.UserRole + 3, True)
        link_item.setForeground(0, QtGui.QBrush(QtGui.QColor("#1a6fc4")))
        font = link_item.font(0)
        font.setUnderline(True)
        link_item.setFont(0, font)
        target.addChild(link_item)

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
        nYear, nMonth, nDay = PhenoCam().getEndDate()

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
    start_date, start_time, end_date, end_time = ProductTable().fetchTableDates(tableWidget, nRow)

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

    #imageCount = PhenoCam.getPhenocamImageCount(SITECODE, DOMAINCODE, start_date, end_date, start_time, end_time)

    #tableWidget.setItem(nRow, 2, QTableWidgetItem(str(imageCount)))


# ======================================================================================================================
#
# ======================================================================================================================
def DP1_20002_fetchImageList(self, nProductID, nRow, start_date, end_date, start_time, end_time, downloadsFilePath):
    """
    Pipelined NEON (via PhenoCam) image fetch: each day's browse-page scan
    hands its images straight to a small download pool, so downloading runs
    in parallel with -- lagging just behind -- the remaining scanning.  The
    function itself stays synchronous: it returns only when all images are
    on disk, because the caller immediately processes the download folder.
    """
    global SITECODE, DOMAINCODE, dailyImagesList, gWebImageCount

    import threading
    from concurrent.futures import ThreadPoolExecutor

    if nRow <= -1:
        return []

    if not os.path.exists(downloadsFilePath):
        os.makedirs(downloadsFilePath)

    total_days = (end_date - start_date).days + 1

    cancel_requested = {"value": False}
    counters = {"downloaded": 0, "done": 0}
    lock = threading.Lock()

    def request_cancel():
        cancel_requested["value"] = True

    progressBar = QProgressWheel(on_close=request_cancel)
    progressBar.setRange(0, total_days)
    progressBar.setWindowTitle('Scanning & downloading images...')
    progressBar.show()

    def _fetch(img):
        if cancel_requested["value"]:
            return
        filename = os.path.basename(img.fullPathAndFilename)
        dest = os.path.join(downloadsFilePath, filename)
        got = 0
        if not os.path.isfile(dest):
            try:
                urllib.request.urlretrieve(img.fullPathAndFilename, dest)
                got = 1
            except Exception as e:
                print(f"[NEON DP1.20002] Error downloading {filename}: {e}")
        with lock:
            counters["done"] += 1
            counters["downloaded"] += got

    dailyImagesList.clear()
    product_str = str(nProductID).zfill(5)
    futures = []
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        current, day_idx = start_date, 0
        while current <= end_date:
            if cancel_requested["value"] or not progressBar.isVisible():
                cancel_requested["value"] = True
                break

            day_idx += 1
            dailyURLvisible = (
                f"https://phenocam.nau.edu/webcam/browse/NEON.{DOMAINCODE}.{SITECODE}.DP1.{product_str}/"
                f"{current.year}/{str(current.month).zfill(2)}/{str(current.day).zfill(2)}"
            )

            try:
                day_images = PhenoCam().getVisibleImages(
                    dailyURLvisible, start_time, end_time
                ).getVisibleList()
            except Exception as e:
                print(f"[NEON DP1.20002] Error scanning {current}: {e}")
                day_images = []

            dailyImagesList.setVisibleList(day_images)
            for img in day_images:
                futures.append(pool.submit(_fetch, img))

            with lock:
                dl = counters["downloaded"]
            progressBar.setWindowTitle(
                f"{current.strftime('%Y-%m-%d')} - {len(futures)} found, {dl} downloaded"
            )
            progressBar.setValue(day_idx)
            QCoreApplication.processEvents()

            current += datetime.timedelta(days=1)

        # Scanning finished: wait for the download tail, keeping the UI alive.
        total = len(futures)
        while not cancel_requested["value"]:
            with lock:
                done = counters["done"]
            if done >= total:
                break
            progressBar.setWindowTitle(f"Downloading... {done}/{total}")
            QCoreApplication.processEvents()
            time.sleep(0.05)
    finally:
        pool.shutdown(wait=not cancel_requested["value"],
                      cancel_futures=cancel_requested["value"])
        if progressBar and progressBar.isVisible():
            progressBar.close()

    gWebImageCount = len(dailyImagesList.getVisibleList())

    # Generate the completeness/gap report (HTML + CSV + optional PDF) in the
    # download folder.  Skipped if the user cancelled; never fatal.
    if not cancel_requested["value"]:
        try:
            from appcore.reporting.gap_report import generate_gap_report
            generate_gap_report(downloadsFilePath)
        except Exception as e:
            print(f"[NEON DP1.20002] Gap report skipped: {e}")

    return dailyImagesList.getVisibleList()

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

        phenoCam = PhenoCam()
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

    # Generate the completeness/gap report (HTML + CSV + optional PDF) in the
    # download folder.  NEON-via-PhenoCam filenames are site-local timestamps,
    # same as PhenoCam proper, so no timezone conversion is applied.  Skipped
    # if the user cancelled mid-download; never fatal.
    if not cancel_requested["value"]:
        try:
            from appcore.reporting.gap_report import generate_gap_report
            generate_gap_report(downloadsFilePath)
        except Exception as e:
            print(f"[NEON DP1.20002] Gap report skipped: {e}")

#jes LET THE CALLING FUNCTION BE RESPONSIBLE FOR REPORTING DOWNLOAD COMPLETION.
#jes MODIFY THIS IN A FUTURE RELEASE TO RETURN A PASS/FAIL MESSAGE TO THE FUNCTION THAT INVOKED THIS FUNCTION.
#jes strMessage = 'Data download is complete!'
#jes msgBox = App_QMessageBox('Data Download', strMessage)
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
    # SAVE DOWNLOADED DATA TO THE USER'S APPLICATION FOLDER THAT IS AUTOMATICALLY CREATED, IF IT DOES NOT EXIST,
    # CREATE IT IN THE USER'S DOCUMENT FOLDER
    # ----------------------------------------------------------------------------------------------------
    NEON_download_file_path = self.edit_NEON_TableInput.text().strip() or self.edit_NEONSaveFilePath.text().strip()
    self.edit_NEONSaveFilePath.setText(NEON_download_file_path)
    self.edit_NEON_TableInput.setText(NEON_download_file_path)
    JsonEditor().update_json_entry("NEON_Root_Folder", NEON_download_file_path)

    if len(NEON_download_file_path) == 0:
        strMessage = f'A download folder has not been specified. Would you like to use the last {APP_NAME} NEON download folder?'
        msgBox = App_QMessageBox('NEON Root Download Folder', strMessage, QMessageBox.Yes | QMessageBox.No)
        response = msgBox.displayMsgBox()

        if response == QMessageBox.Yes:
            #NEON_download_file_path = os.path.expanduser('~')
            #NEON_download_file_path = os.path.join(NEON_download_file_path, 'Documents')

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
    # CODE-BUILT LAYOUT: one row per checked site/product, each with its own site
    # --------------------------------------------------------------------------------
    if getattr(self, "_neon_layout_built", False):
        global SITECODE, DOMAINCODE
        jobs = self._neon_download_jobs()
        if not jobs:
            App_QMessageBox('NEON Download', 'Check one or more data products in the site list first.',
                            buttons=QMessageBox.Close).displayMsgBox()
            return
        multi_site = len({j[0] for j in jobs}) > 1
        for site, domain, strProductIDCell, start_date, start_time, end_date, end_time in jobs:
            SITECODE, DOMAINCODE = site, domain
            # One site downloads into the folder itself, as before; several sites each get a subfolder.
            site_root = os.path.join(NEON_download_file_path, site) if multi_site else NEON_download_file_path
            nProductID = self._neon_product_id(strProductIDCell.split(':')[0])
            if nProductID <= 0:
                missing_data_message += 'NEON Error!\n  ' + strProductIDCell + 'Product not available!' + '\n'
                continue
            PRODUCTCODE = strProductIDCell.split(':')[0]

            if nProductID in self.NEON_IMAGE_PRODUCT_IDS:
                downloadsFilePath = os.path.join(site_root, 'Images')
                if not os.path.exists(downloadsFilePath):
                    os.makedirs(downloadsFilePath)
                DP1_20002_fetchImageList(self, nProductID, 0, start_date, end_date, start_time, end_time,
                                         downloadsFilePath)
                processLocalImage(self, imageFileFolder=downloadsFilePath)

            if nProductID != 20002:
                strStartYearMonth = str(start_date.year) + '-' + str(start_date.month).zfill(2)
                strEndYearMonth = str(end_date.year) + '-' + str(end_date.month).zfill(2)
                dateRange = App_Utils().getRangeOfDates(strStartYearMonth, strEndYearMonth)
                availableMonths = NEON_API().getAvailableMonths(SITECODE, PRODUCTCODE)
                monthCount = 0
                missingMonths = []
                for month in dateRange:
                    if month in availableMonths:
                        monthCount += 1
                    else:
                        missingMonths.append(month)
                if monthCount == 0:
                    missing_data_message = missing_data_message + 'NEON Error!  ' + site + ' ' + strProductIDCell + 'Data is not available for some or all of the dates selected!\n'
                elif monthCount < len(dateRange):
                    strMsg = '%d of %d months unavailable: %s' % (len(missingMonths), len(dateRange), missingMonths)
                    missing_data_message = missing_data_message + 'Partial Download!\n   ' + site + ' ' + strProductIDCell + strMsg + '\n'
                if monthCount > 0:
                    downloadsFilePath = os.path.normpath(os.path.join(site_root, 'data'))
                    if not os.path.exists(downloadsFilePath):
                        os.makedirs(downloadsFilePath)
                    nError = myNEON_API.FetchData(SITECODE, strProductIDCell, strStartYearMonth, strEndYearMonth, downloadsFilePath)
        rowRange = range(0)   # the .ui-table loop below has nothing left to do
    else:
        rowRange = range(self.NEON_selected_products.rowCount())

    # --------------------------------------------------------------------------------
    # FIND IMAGE PRODUCT (20002) ROW TO GET DATE RANGE  (.ui table, used before the code-built layout)
    # --------------------------------------------------------------------------------

    for nRow in rowRange:
        ProductTableObj = ProductTable()
        start_date, start_time, end_date, end_time = ProductTableObj.fetchTableDates(self.NEON_selected_products, nRow)

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
                dateRange = App_Utils().getRangeOfDates(strStartYearMonth, strEndYearMonth)

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
        msgBox = App_QMessageBox('Download Error!', missing_data_message, buttons=QMessageBox.Close)
    else:
        msgBox = App_QMessageBox('Download Complete!', 'Download Complete!', buttons=QMessageBox.Close)
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
        img = App_Utils().convertQImageToMat(currentImage.toImage())

        self.setMouseTracking(False)

        cv2.imshow('Original', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.setMouseTracking(True)


# ======================================================================================================================
#
# ======================================================================================================================
def NEON_labelMouseDoubleClickEvent(self, event):
    img = App_Utils().convertQImageToMat(self.NEON_labelLatestImage.toImage())
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

    szWindowsTitle = f"{APP_DISPLAY_NAME}" + " " + SW_VERSION + " - John E. Stranzl Jr., PhD"

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
    from appcore.Save_Utils import Save_Utils

    settings_folder = Save_Utils().get_settings_folder()
    print(settings_folder)
    hydra_working_folder = os.path.normpath(os.path.join(settings_folder, "MyHydraOutputs"))
    sys.argv.append(f"hydra.run.dir={hydra_working_folder}")

    if 0:
        if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.core.global_hydra.GlobalHydra.instance().clear()
        else:
            hydra.initialize(config_path=None)

    # CREATE MAIN APP WINDOW
    # High-DPI attributes MUST be set before QApplication is constructed or they
    # have no effect. The env vars set at import time (QT_ENABLE_HIGHDPI_SCALING)
    # are not sufficient on all PyQt5/Qt builds; without these attributes, text
    # renders un-antialiased / jagged on scaled displays (the "dithered" look).
    try:
        from PyQt5.QtCore import Qt as _Qt
        if hasattr(_Qt, "AA_EnableHighDpiScaling"):
            QApplication.setAttribute(_Qt.AA_EnableHighDpiScaling, True)
        if hasattr(_Qt, "AA_UseHighDpiPixmaps"):
            QApplication.setAttribute(_Qt.AA_UseHighDpiPixmaps, True)
    except Exception as _e:
        print(f"[HIDPI] Could not set high-DPI attributes: {_e}")

    app = QApplication(sys.argv)

    # Light mode is the default — dark mode toggled via View menu
    app.setStyleSheet("")

    # ------------------------------------------------------------------------------------------------------------------
    # SHOW SPLASH IN SEPARATE PROCESS — stays visible throughout all of MainWindow init
    # ------------------------------------------------------------------------------------------------------------------
    _splash_dir = Path(__file__).resolve().parent / "resources" / "splash_screens"
    _splash = SplashScreen(
        image_path=os.path.join(_splash_dir, APP_LOGO_FILENAME)
    )
    _splash.show()   # blocks briefly until splash is painted, then returns

    frame = MainWindow(splash=_splash)

    # ------------------------------------------------------------------------------------------------------------------
    # WINDOW PLACEMENT
    #
    # Was:  frame.move(app.desktop().screen().rect().center() - frame.rect().center())
    #
    # That centered against the FULL screen rect (ignoring the desktop panel)
    # and offset by the CLIENT rect (ignoring the title bar), so on Linux/X11
    # the title bar ended up hidden underneath the panel and the user had to
    # right-click -> Move to drag the window back into view. It also used the
    # deprecated QApplication.desktop().
    #
    # install_window_placement_guard() shrinks the window to fit the usable
    # work area first, then centers and clamps it, and re-checks whenever the
    # screen resolution changes (e.g. reconnecting to a VNC session).
    # ------------------------------------------------------------------------------------------------------------------
    install_window_placement_guard(frame)

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
    parser = argparse.ArgumentParser(description=f'CLI for {APP_DISPLAY_NAME}')
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
    segment_parser = subparsers.add_parser('segment', help=f'Segment a single image using a trained {APP_DISPLAY_NAME} model')
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
    # ------------------------------------------------------------------
    # OUTPUT OPTIONS -- these mirror the 'Output Options' group box in the
    # ML Image Processing dialog's Segment tab. The GUI path passes all four
    # through segment_main() -> ML_Segmentation_Dispatcher(); the CLI path
    # goes through run_sam2()/run_segformer(), which historically supported
    # only masks and image copying. Defaults below match the GUI defaults
    # (all four checked).
    #
    # --no-mask / --no-copy are retained as deprecated aliases so existing
    # scripts and batch jobs keep working.
    # ------------------------------------------------------------------
    segment_parser.add_argument('--save-masks', dest='save_masks',
                                action='store_true', default=True,
                                help='Save predicted binary mask PNGs (default: on)')
    segment_parser.add_argument('--no-save-masks', dest='save_masks',
                                action='store_false',
                                help='Do not save predicted binary mask PNGs')

    segment_parser.add_argument('--save-probability-maps', dest='save_probability_maps',
                                action='store_true', default=True,
                                help='Save per-pixel probability maps (default: on)')
    segment_parser.add_argument('--no-save-probability-maps', dest='save_probability_maps',
                                action='store_false',
                                help='Do not save per-pixel probability maps')

    segment_parser.add_argument('--copy-original-images', dest='copy_original_image',
                                action='store_true', default=True,
                                help='Copy each source image into the output folder (default: on)')
    segment_parser.add_argument('--no-copy-original-images', dest='copy_original_image',
                                action='store_false',
                                help='Do not copy source images into the output folder')

    segment_parser.add_argument('--save-diagnostic-panels', dest='save_diagnostic_panels',
                                action='store_true', default=True,
                                help='Save side-by-side diagnostic panel images (default: on)')
    segment_parser.add_argument('--no-save-diagnostic-panels', dest='save_diagnostic_panels',
                                action='store_false',
                                help='Do not save diagnostic panel images')

    segment_parser.add_argument('--use-tta', dest='use_tta',
                                action='store_true', default=False,
                                help='SegFormer only: test-time augmentation (flip + average). Slower, cleaner boundaries.')
    segment_parser.add_argument('--no-use-tta', dest='use_tta',
                                action='store_false',
                                help='Disable test-time augmentation (default)')

    # DEPRECATED aliases -- resolved in the handler below.
    segment_parser.add_argument('--no-mask',       action='store_true',
                                help='DEPRECATED: use --no-save-masks')
    segment_parser.add_argument('--no-copy',       action='store_true',
                                help='DEPRECATED: use --no-copy-original-images')

    # Train / fine-tune parser
    train_parser = subparsers.add_parser('train', help=f'Train / fine-tune a {APP_DISPLAY_NAME} model from a site configuration (headless)')
    train_parser.add_argument('--config', required=True,
                              help='Path to a site_config.json (as produced by the Training tab)')
    train_parser.add_argument('--label',  required=False, default=None,
                              help="Training category to train against, overriding the config's "
                                   "train_model.TRAINING_CATEGORIES. Accepts 'id - name' "
                                   "(e.g. '1 - water'), a bare name, or a bare id. "
                                   "Required when the config does not already specify one.")
    train_parser.add_argument('--mode',   required=True, choices=['sam2', 'segformer'],
                              help='Model type to train: sam2 or segformer')
    train_parser.add_argument('--validation-overlay-mode',
                              dest='validation_overlay_mode',
                              required=False, default=None,
                              choices=['last', 'every', 'interval'],
                              help="When to write validation overlay PNGs. "
                                   "'last' = final epoch only (default), 'every' = every "
                                   "epoch, 'interval' = every N epochs (see "
                                   "--validation-overlay-interval). Overlays are written at "
                                   "least once regardless, even if training stops early. "
                                   "Overrides the config.")
    train_parser.add_argument('--validation-overlay-interval',
                              dest='validation_overlay_interval',
                              type=int, required=False, default=None,
                              help='Write overlays every N epochs. Only used when '
                                   "--validation-overlay-mode is 'interval'. Overrides the config.")
    train_parser.add_argument('--lr-scheduler', dest='lr_scheduler_enabled',
                              action='store_true', default=None,
                              help='Enable the ReduceLROnPlateau scheduler. Overrides the config.')
    train_parser.add_argument('--no-lr-scheduler', dest='lr_scheduler_enabled',
                              action='store_false', default=None,
                              help='Disable the scheduler and hold the learning rate fixed. '
                                   'Overrides the config.')
    train_parser.add_argument('--lr-scheduler-factor', dest='lr_scheduler_factor',
                              type=float, required=False, default=None,
                              help='Multiply the learning rate by this on each plateau. '
                                   'Overrides the config.')
    train_parser.add_argument('--lr-scheduler-patience', dest='lr_scheduler_patience',
                              type=int, required=False, default=None,
                              help='Epochs without improvement before the learning rate drops. '
                                   'Keep below --patience. Overrides the config.')
    train_parser.add_argument('--lr-scheduler-min-lr', dest='lr_scheduler_min_lr',
                              type=float, required=False, default=None,
                              help='Floor for the learning rate (e.g. 1e-7). Overrides the config.')
    train_parser.add_argument('--validation-overlay-samples',
                              dest='validation_overlay_samples',
                              type=int, required=False, default=None,
                              help='Number of validation images to write overlays for on each '
                                   'overlay-writing epoch. Overrides the config.')
    train_parser.add_argument('--max-best-checkpoints', dest='max_best_checkpoints',
                              type=int, required=False, default=None,
                              help='How many best-scoring checkpoints to retain. Overrides the config.')

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

    # Color Segmentation parser
    colorseg_parser = subparsers.add_parser(
        'colorseg',
        help='Run Color Segmentation feature extraction (headless) and write the CSV/XLSX feature file'
    )
    colorseg_parser.add_argument(
        '--folder', required=True,
        help='Folder containing the images to analyze.'
    )
    colorseg_parser.add_argument(
        '--recursive', action='store_true',
        help='Include images in sub-folders.'
    )
    colorseg_parser.add_argument(
        '--rois', required=False, default=None,
        help='ROI_Masks (COCO 1.0) JSON file holding fixed ROIs used for every image. '
             'Required unless --region whole or --masks is given.'
    )
    colorseg_parser.add_argument(
        '--masks', required=False, default=None,
        help='Folder of per-image masks, one mask per image, named <image>_mask.<ext> '
             '(png, jpg, tif or bmp). Use instead of --rois when the region changes from image to image.'
    )
    colorseg_parser.add_argument(
        '--mask-name', dest='mask_name', required=False, default=None,
        help='Column prefix for the masked region (required with --masks), e.g. water.'
    )
    colorseg_parser.add_argument(
        '--mask-value', dest='mask_value', required=False, type=int, default=None,
        help='Class index to select in a class-index mask. Omit for a binary mask, where every '
             'non-zero pixel is inside the region.'
    )
    colorseg_parser.add_argument(
        '--region', required=False, default=None, choices=['whole', 'roi', 'both'],
        help='Which regions to measure (default: both when --rois is given, otherwise whole).'
    )
    colorseg_parser.add_argument(
        '--features', required=False, default='intensity,entropy,hsv',
        help="Comma-separated list of intensity, entropy, hsv. 'all' or 'none' also accepted "
             "(default: intensity,entropy,hsv)."
    )
    colorseg_parser.add_argument(
        '--texture', required=False, default='none',
        help="Comma-separated list of glcm, gabor, lbp, wavelet, fourier. 'all' or 'none' also "
             "accepted (default: none)."
    )
    colorseg_parser.add_argument(
        '--greenness', required=False, default='all',
        help="Comma-separated list of gcc, gli, ndvi, exg, rgi. 'all' or 'none' also accepted "
             "(default: all)."
    )
    colorseg_parser.add_argument(
        '--clusters', required=False, type=int, default=4,
        help='Number of dominant HSV color clusters (default: 4).'
    )
    colorseg_parser.add_argument(
        '--output', required=False, default=None,
        help='Output folder for the CSV/XLSX feature files (default: --folder).'
    )

    # Custom help handling
    if '-h' in sys.argv or '--help' in sys.argv:
        print(f"Global Help: CLI for {APP_DISPLAY_NAME}")
        parser.print_help()  # General help for the main parser
        print("\nHelp for 'triage' command:")
        triage_parser.print_help()  # Help for triage subparser
        print("\nHelp for 'slice' command:")
        slice_parser.print_help()  # Help for slice subparser
        print("\nHelp for 'segment' command:")
        segment_parser.print_help()  # Help for segment subparser
        print("\nHelp for 'roi' command:")
        roi_parser.print_help()  # Help for roi subparser
        print("\nHelp for 'colorseg' command:")
        colorseg_parser.print_help()  # Help for colorseg subparser
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
        from appcore.version import SW_VERSION, RELEASE, BUILD_DATE, SHA
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
        #min_val = 65.0
        #max_val = 180.0
        correct_alignment = False
        save_poly_lines = False
        reference_image_filename = ""
        rotation_threshold = 0.15

        print(args.command)

        print("These are the Triage parameters:", args.folder, args.min, args.max)
        myTriage = ImageTriage(False)
        myTriage.cleanImages(args.folder, \
                             fetch_recursive, \
                             args.min, args.max, \
                             create_report, move_images, \
                             correct_alignment, save_poly_lines,
                             reference_image_filename, rotation_threshold)

        print('Image triage is complete!')

    elif args.command == 'slice':
        filenames = cli_fetchLocalImageList(args.folder)

        from PyQt5.QtCore import QRect
        from PIL import Image
        from appcore.CompositeSlices import CompositeSlices
        first_image = Image.open(filenames[0].fullPathAndFilename)
        slice_rect = QRect(
            int(args.center - args.width / 2),
            0,
            int(args.width),
            first_image.height
        )

        compositeSlices = CompositeSlices(slice_rect, False)
        compositeSlices.create_composite_image(filenames, os.path.join(args.folder, 'compositeSlices'))

        print("Composite slice complete!")

    elif args.command == 'colorseg':
        # Headless Color Segmentation: same feature extraction as the dialog's
        # "Build Feature File", with ROIs read from an ROI_Masks (COCO) file.
        import cv2
        from PyQt5.QtCore import QRect, QPoint

        from appcore.colorSegmentationParams import colorSegmentationParamsClass
        from appcore.vegetation_indices import Vegetation_Indices, GreennessIndex
        from appcore.dialogs.color_segmentation.color_seg_feature_export import ColorSegFeatureExport
        from appcore.dialogs.color_segmentation.color_seg_roi_coco_export import load_roi_masks
        from appcore.dialogs.color_segmentation.color_seg_roi_data import roiData, ROIShape

        def _csv_option(value, valid, label):
            """Parse a comma-separated option, allowing 'all' and 'none'."""
            items = [v.strip().lower() for v in (value or '').split(',') if v.strip()]
            if items == ['all']:
                return set(valid)
            if items == ['none'] or not items:
                return set()
            bad = [v for v in items if v not in valid]
            if bad:
                print(f"[ERROR] Unknown {label}: {', '.join(bad)}. Valid: {', '.join(valid)}", file=sys.stderr)
                sys.exit(1)
            return set(items)

        folder = args.folder
        if not os.path.isdir(folder):
            print(f"[ERROR] Folder not found: {folder}", file=sys.stderr)
            sys.exit(1)

        output_folder = args.output if args.output else folder
        os.makedirs(output_folder, exist_ok=True)

        images = cli_fetchLocalImageList(folder, args.recursive)
        if not images:
            print(f"[ERROR] No images found in: {folder}", file=sys.stderr)
            sys.exit(1)

        if args.rois and args.masks:
            print("[ERROR] Use --rois (fixed ROIs) or --masks (one mask per image), not both.", file=sys.stderr)
            sys.exit(1)
        if args.masks:
            if not os.path.isdir(args.masks):
                print(f"[ERROR] Mask folder not found: {args.masks}", file=sys.stderr)
                sys.exit(1)
            if not args.mask_name:
                print("[ERROR] --mask-name is required with --masks.", file=sys.stderr)
                sys.exit(1)

        region = args.region or ('both' if (args.rois or args.masks) else 'whole')
        if region in ('roi', 'both') and not (args.rois or args.masks):
            print("[ERROR] --rois or --masks is required unless --region whole.", file=sys.stderr)
            sys.exit(1)

        features = _csv_option(args.features, ('intensity', 'entropy', 'hsv'), 'feature')
        texture = _csv_option(args.texture, ('glcm', 'gabor', 'lbp', 'wavelet', 'fourier'), 'texture method')
        greenness = _csv_option(args.greenness, ('gcc', 'gli', 'ndvi', 'exg', 'rgi'), 'greenness index')

        params = colorSegmentationParamsClass()
        params.Intensity = 'intensity' in features
        params.ShannonEntropy = 'entropy' in features
        params.HSV = 'hsv' in features
        params.Texture = bool(texture)
        params.numColorClusters = args.clusters
        params.wholeImage = region in ('whole', 'both')
        params.ROI = region in ('roi', 'both')
        params.GCC = 'gcc' in greenness
        params.GLI = 'gli' in greenness
        params.NDVI = 'ndvi' in greenness
        params.ExG = 'exg' in greenness
        params.RGI = 'rgi' in greenness

        # Order here sets the column order, matching the dialog.
        greenness_index_list = []
        for key, index in (('gcc', Vegetation_Indices.GCC),
                           ('gli', Vegetation_Indices.GLI),
                           ('ndvi', Vegetation_Indices.NDVI),
                           ('exg', Vegetation_Indices.ExG),
                           ('rgi', Vegetation_Indices.RGI)):
            if key in greenness:
                greenness_index_list.append(GreennessIndex(index))

        texture_options = {m: (m in texture) for m in ('glcm', 'gabor', 'lbp', 'wavelet', 'fourier')}
        texture_options['enabled'] = bool(texture)

        # ------------------------------------------------------------------
        # ROIs from an ROI_Masks (COCO 1.0) file, as exported by the dialog.
        # ------------------------------------------------------------------
        # Per-image mask region (one mask per image), as an alternative to fixed ROIs.
        mask_spec = None
        if params.ROI and args.masks:
            mask_spec = {'name': args.mask_name, 'folder': args.masks, 'value': args.mask_value}
            kind = 'binary' if args.mask_value is None else f'class index {args.mask_value}'
            print(f"[INFO] Per-image masks ({kind}) from {args.masks}, region name '{args.mask_name}'")

        roiList = []
        if params.ROI and args.rois:
            try:
                roi_data = load_roi_masks(args.rois)
            except Exception as e:
                print(f"[ERROR] Could not read ROI file: {e}", file=sys.stderr)
                sys.exit(1)

            first = cv2.imread(images[0].fullPathAndFilename)
            if first is not None and roi_data['width'] and roi_data['height']:
                if (roi_data['width'], roi_data['height']) != (first.shape[1], first.shape[0]):
                    print(f"[WARN] The ROI file was made for {roi_data['width']} x {roi_data['height']} "
                          f"images but the first image is {first.shape[1]} x {first.shape[0]}. "
                          f"ROI coordinates may not line up.")

            shapes = {'rectangle': ROIShape.RECTANGLE, 'polygon': ROIShape.POLYGON, 'freeform': ROIShape.FREEFORM}
            for r in roi_data['rois']:
                roiObj = roiData()
                roiObj.setROIName(r['name'])
                roiObj.setROIShape(shapes.get(r['shape'], ROIShape.POLYGON))
                roiObj.setNumColorClusters(args.clusters)
                x, y, w, h = r['bbox']
                roiObj.setImageROI(QRect(x, y, w, h))
                if roiObj.getROIShape() != ROIShape.RECTANGLE:
                    roiObj.setImagePolygon([QPoint(px, py) for (px, py) in r['polygon']])
                roiList.append(roiObj)

            print(f"[INFO] Loaded {len(roiList)} ROI(s) from {args.rois}: "
                  f"{', '.join(r['name'] for r in roi_data['rois'])}")
            if roi_data['skipped']:
                print(f"[WARN] Skipped {roi_data['skipped']} annotation(s) that are not polygons or boxes.")

        print(f"[INFO] Color Segmentation: {len(images)} image(s), region={region}, "
              f"clusters={args.clusters}, features={sorted(features) or ['none']}, "
              f"texture={sorted(texture) or ['none']}, greenness={sorted(greenness) or ['none']}")

        ColorSegFeatureExport().ExtractFeatures(images, output_folder, roiList, params,
                                                greenness_index_list, texture_options=texture_options,
                                                mask_spec=mask_spec)

        print('Color segmentation feature extraction complete!')

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
        from appcore.cli_segment import run_sam2, run_segformer

        # Validate inputs
        if not os.path.isfile(args.model):
            print(f"[ERROR] Model file not found: {args.model}", file=sys.stderr)
            sys.exit(1)
        os.makedirs(args.output, exist_ok=True)

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        category = {"id": args.category_id, "name": args.category_name}

        print(f"[{APP_DISPLAY_NAME}] Mode:      {args.mode.upper()}")
        print(f"[{APP_DISPLAY_NAME}] Model:     {args.model}")
        print(f"[{APP_DISPLAY_NAME}] Image:     {args.image}")
        print(f"[{APP_DISPLAY_NAME}] Output:    {args.output}")
        print(f"[{APP_DISPLAY_NAME}] Category:  {args.category_name} (ID: {args.category_id})")

        # ------------------------------------------------------------------
        # Resolve the deprecated negative flags onto the new positive ones.
        # An explicit --no-mask/--no-copy still wins, so old scripts behave
        # exactly as before.
        # ------------------------------------------------------------------
        if getattr(args, 'no_mask', False):
            print(f"[{APP_DISPLAY_NAME}] NOTE: --no-mask is deprecated; use --no-save-masks.")
            args.save_masks = False
        if getattr(args, 'no_copy', False):
            print(f"[{APP_DISPLAY_NAME}] NOTE: --no-copy is deprecated; use --no-copy-original-images.")
            args.copy_original_image = False

        # Aliases so run_sam2()/run_segformer() can read whichever attribute
        # name they were written against.
        args.no_mask = not args.save_masks
        args.no_copy = not args.copy_original_image
        args.save_probability_map = args.save_probability_maps
        args.save_diagnostic_panel = args.save_diagnostic_panels

        print(f"[{APP_DISPLAY_NAME}] Output options:")
        print(f"[{APP_DISPLAY_NAME}]   Save predicted masks:    {args.save_masks}")
        print(f"[{APP_DISPLAY_NAME}]   Save probability maps:   {args.save_probability_maps}")
        print(f"[{APP_DISPLAY_NAME}]   Copy original images:    {args.copy_original_image}")
        print(f"[{APP_DISPLAY_NAME}]   Save diagnostic panels:  {args.save_diagnostic_panels}")

        if args.mode == 'sam2':
            run_sam2(args, device, category, progressBar=None)
        elif args.mode == 'segformer':
            run_segformer(args, device, category, progressBar=None)

        print(f"[{APP_DISPLAY_NAME}] Segmentation complete.")

    elif args.command == 'train':
        import json
        if not os.path.isfile(args.config):
            print(f"[ERROR] Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        with open(args.config, 'r', encoding='utf-8') as f:
            site_config = json.load(f)

        print(f"[{APP_DISPLAY_NAME}] Training mode: {args.mode.upper()}")
        print(f"[{APP_DISPLAY_NAME}] Config:        {args.config}")
        print(f"[{APP_DISPLAY_NAME}] Site:          {site_config.get('siteName', '?')}")

        # ------------------------------------------------------------------
        # CLI overrides. Only applied when the flag was passed, so an
        # unflagged run uses whatever the config already specifies.
        # ------------------------------------------------------------------
        for _key in ('validation_overlay_mode', 'validation_overlay_interval',
                     'validation_overlay_samples', 'lr_scheduler_enabled',
                     'lr_scheduler_factor', 'lr_scheduler_patience',
                     'lr_scheduler_min_lr', 'max_best_checkpoints'):
            _val = getattr(args, _key, None)
            if _val is not None:
                site_config[_key] = _val
                print(f"[{APP_DISPLAY_NAME}] Override:      {_key} = {_val}")

        for _key, _label in (('validation_overlay_interval', '--validation-overlay-interval'),
                             ('validation_overlay_samples', '--validation-overlay-samples'),
                             ('max_best_checkpoints', '--max-best-checkpoints')):
            if site_config.get(_key) is not None and int(site_config[_key]) < 1:
                print(f'[ERROR] {_label} must be >= 1.', file=sys.stderr)
                sys.exit(1)

        print(f"[{APP_DISPLAY_NAME}] Overlays:      "
              f"mode={site_config.get('validation_overlay_mode', 'last')} "
              f"interval={site_config.get('validation_overlay_interval', 5)} "
              f"samples={site_config.get('validation_overlay_samples', 5)}")

        # ------------------------------------------------------------------
        # Resolve the training label. Reuses the site config editor's helpers
        # so the CLI, the editor, and the Training tab all agree on the
        # 'id - name' format and on train_model.TRAINING_CATEGORIES.
        # ------------------------------------------------------------------
        from appcore.utils.site_config_manager import (
            collect_training_labels, get_training_categories,
            set_training_categories, parse_label,
        )

        images_root = site_config.get('segmentation_images_path', '')
        selected_folders = [str(s) for s in site_config.get('selected_folders', [])]

        if args.label:
            available, _coverage, _conflicts = collect_training_labels(
                images_root, selected_folders)
            wanted = str(args.label).strip()
            resolved = None
            for candidate in available:
                parsed = parse_label(candidate)
                if parsed is None:
                    continue
                label_id, label_name = parsed
                if wanted in (candidate, label_id, label_name) or \
                        wanted.lower() == label_name.lower():
                    resolved = candidate
                    break

            if resolved is None:
                print(f"[ERROR] Label '{wanted}' was not found in the selected datasets.",
                      file=sys.stderr)
                if not images_root or not selected_folders:
                    print("        The config has no images root or no selected folders; "
                          "open it in the Site Config Editor first.", file=sys.stderr)
                else:
                    print(f"        Available: {', '.join(available) or '(none)'}",
                          file=sys.stderr)
                sys.exit(1)

            set_training_categories(site_config, resolved)
            print(f"[{APP_DISPLAY_NAME}] Label:         {resolved}  (--label overrides the config)")
        else:
            current_label = get_training_categories(site_config)
            if not current_label:
                print("[ERROR] No training label specified and the config does not set "
                      "train_model.TRAINING_CATEGORIES.", file=sys.stderr)
                print("        Pass --label, or set one in the Site Config Editor.",
                      file=sys.stderr)
                sys.exit(1)
            print(f"[{APP_DISPLAY_NAME}] Label:         {current_label}")

        # Build the SAM2 model config headlessly (the GUI path uses
        # @hydra.main); SegFormer does not need a Hydra cfg.
        cfg = None
        if args.mode == 'sam2':
            import importlib.util
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            sam2_dir = os.path.dirname(importlib.util.find_spec('sam2').origin)
            cfg_dir = os.path.join(sam2_dir, "configs", "sam2.1")
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
                cfg = compose(config_name="sam2.1_hiera_l")

        from appcore.ml_core.ml_model_training import MLModelTraining
        dispatcher = MLModelTraining(cfg, parent_widget=None, site_config=site_config)
        dispatcher.Model_Training_Dispatcher(cfg=cfg, mode=args.mode)

        print(f"[{APP_DISPLAY_NAME}] Training complete.")

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

        from appcore.ROI_Analyzer import ROI_Analyzer

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
        temp = ROI_Analyzer("", "")
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

            analyzer = ROI_Analyzer(
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

                analyzer = ROI_Analyzer(
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
    file_count, files = App_Utils().getFileList(filePath, extensions, bFetchRecursive)

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
        # SAM2 uses Hydra; SAM3 (LoRA subprocess) does not. Match SAM2 specifically.
        if selected_training_model.lower() in ("sam", "sam2"):
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
        from appcore.ml_core.ml_model_training import MLModelTraining
        myML_Dispatcher = MLModelTraining(cfg, parent_widget=hyperparameterDlg)

        print("Execute ML training...")
        trainer = myML_Dispatcher.Model_Training_Dispatcher(cfg=cfg, mode=selected_training_model)

        # Re-enable Train button
        try:
            hyperparameterDlg.training_tab.pushButton_train.setEnabled(True)
        except Exception:
            pass

        # Prompt user about suggested blob radius if available
        # Blob-radius prompt is a SAM2 artifact; SAM3 produces none.
        if trainer is not None and selected_training_model.lower() in ("sam", "sam2"):
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
        use_tta = (hyperparameterDlg.get_use_tta()
                   if hasattr(hyperparameterDlg, "get_use_tta") else False)
        selected_label_categories = hyperparameterDlg.get_selected_label_categories()
        selected_segment_model = hyperparameterDlg.get_selected_segment_model()

        # Keep the ML dialog open — disable Segment button during inference
        try:
            hyperparameterDlg.segment_tab.pushButton_Segment.setEnabled(False)
        except Exception:
            pass

        from appcore.ml_core.ml_image_segmentation import MLImageSegmentation
        mySegmentation = MLImageSegmentation(cfg, parent_widget=hyperparameterDlg)
        mySegmentation.ML_Segmentation_Dispatcher(copy_original_image, save_masks, selected_label_categories,
                                                    mode=selected_segment_model,
                                                    save_probability_maps=save_probability_maps,
                                                    save_diagnostic_panels=save_diagnostic_panels,
                                                    use_tta=use_tta)

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

