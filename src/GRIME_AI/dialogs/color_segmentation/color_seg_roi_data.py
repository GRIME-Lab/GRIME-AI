#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import cv2
import numpy as np
from enum import Enum

from PyQt5.QtCore import QRect, QSize


# ======================================================================================================================
#
# ======================================================================================================================
class ROIShape(Enum):
    RECTANGLE = 0
    POLYGON = 1
    FREEFORM = 2

# ======================================================================================================================
#
# ======================================================================================================================
class GRIME_AI_roiData:
    def __init__(self):
        self.trainingImageName = ""
        self.roiName = ""
        self.displayROI = QRect()
        self.imageROI = QRect()
        self.shape = ROIShape.RECTANGLE
        self.displayPolygon = []   # QPoint list in display coords
        self.imagePolygon = []     # QPoint list in image coords
        self.imageSize = QSize()
        self.displaySize = QSize()
        self.clusterCenters = []
        self.hist = []
        self.hsvClusterCenters = []
        self.hsvHist = []
        self.intensity = []
        self.shannonEntropy = []
        self.nNumColorClusters = []

    # --------------------------------------------------------------------------------
    def setTrainingImageName(self, trainingImageName):
        self.trainingImageName = trainingImageName
    def getTrainingImageName(self):
        return(self.trainingImageName)

    # --------------------------------------------------------------------------------
    def setNumColorClusters(self, nNumColorClusters):
        self.nNumColorClusters = nNumColorClusters
    def getNumColorClusters(self):
        return self.nNumColorClusters

    # --------------------------------------------------------------------------------
    def setROIName(self, roiName):
        self.roiName = roiName
    def getROIName(self):
        return self.roiName

    # --------------------------------------------------------------------------------
    def setDisplayROI(self, rect):
        self.displayROI = rect
    def getDisplayROI(self):
        return self.displayROI

    # --------------------------------------------------------------------------------
    def setImageROI(self, imageROI):
        self.imageROI = imageROI
    def getImageROI(self):
        temp = self.imageROI
        #temp.setY(temp.y() - 25)
        return temp
        #return self.imageROI

    # --------------------  ------------------------------------------------------------
    def setImageSize(self, imageSize):
        self.imageSize = imageSize

    # --------------------------------------------------------------------------------
    def setDisplaySize(self, displaySize):
        self.displaySize = displaySize

    # --------------------------------------------------------------------------------
    def calcROI(self):
        # GET ROI, LABEL and ORIGINAL IMAGE SIZE
        widthMultiplier = self.imageSize.width() / self.displaySize.width()
        heightMultiplier = self.imageSize.height() / self.displaySize.height()

        # CALCULATE THE ROI OF THE ACTUAL IMAGE SIZE
        x = int(self.displayROI.x() * widthMultiplier)
        y = int(self.displayROI.y() * heightMultiplier)
        roiWidth = int(self.displayROI.width() * widthMultiplier)
        roiHeight = int(self.displayROI.height() * heightMultiplier)

        rect = QRect(x, y, roiWidth, roiHeight)
        self.setImageROI(rect)

        if self.displayPolygon:
            self.imagePolygon = [type(pt)(int(pt.x() * widthMultiplier),
                                          int(pt.y() * heightMultiplier))
                                 for pt in self.displayPolygon]

    # --------------------------------------------------------------------------------
    def setDisplayPolygon(self, pts):
        self.displayPolygon = list(pts)
    def getDisplayPolygon(self):
        return self.displayPolygon
    def setImagePolygon(self, pts):
        self.imagePolygon = list(pts)
    def getImagePolygon(self):
        return self.imagePolygon

    # --------------------------------------------------------------------------------
    def setROIShape(self, shape):
        self.shape = shape
    def getROIShape(self):
        return self.shape

    # --------------------------------------------------------------------------------
    def setClusterCenters(self, clusterCenters, hist):
        self.clusterCenters = clusterCenters
        self.hist = hist
    def getClusterCenters(self):
        return self.clusterCenters, self.hist

    def setHSVClusterCenters(self, clusterCenters, hist):
        self.hsvClusterCenters = clusterCenters
        self.hsvHist = hist
    def getHSVClusterCenters(self):
        return self.hsvClusterCenters, self.hsvHist

    # --------------------------------------------------------------------------------
    def extractROI(self, rect, image):
        return (image[rect.y():rect.y() + rect.height(), rect.x():rect.x() + rect.width()])

    # --------------------------------------------------------------------------------
    def patchAndMask(self, image):
        """Bounding-box patch of this ROI and a boolean mask of the pixels inside it."""
        poly = []
        if self.shape != ROIShape.RECTANGLE and self.imagePolygon:
            poly = [(int(p.x()), int(p.y())) for p in self.imagePolygon]
        r = self.imageROI
        return roi_patch_and_mask(image, (r.x(), r.y(), r.width(), r.height()), poly)

    # --------------------------------------------------------------------------------
    def insidePixels(self, image):
        """Only the pixels inside this ROI, as an (N, 1, C) array. None if the ROI is empty."""
        patch, mask = self.patchAndMask(image)
        if patch is None:
            return None
        return patch[mask].reshape(-1, 1, patch.shape[2]) if patch.ndim == 3 else patch[mask].reshape(-1, 1)


# ======================================================================================================================
# Shared by training, the on-screen feature table and the feature-file export so every path
# uses exactly the same pixels. Plain Python/NumPy arguments so it also runs in worker processes.
# ======================================================================================================================
def roi_patch_and_mask(image, rect, polygon=None):
    """
    image   : H x W [x C] array
    rect    : (x, y, w, h) in image coordinates (used for rectangles)
    polygon : list of (x, y) image-coordinate vertices; if 3 or more are given the ROI is that polygon

    Returns (patch, mask): the ROI's bounding box clipped to the image and a boolean mask that is
    True only for pixels inside the ROI. Returns (None, None) if the ROI does not overlap the image.
    """
    H, W = image.shape[:2]
    if polygon and len(polygon) >= 3:
        pts = np.asarray(polygon, dtype=np.int32)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0) + 1
    else:
        x, y, w, h = rect
        x0, y0, x1, y1 = x, y, x + w, y + h
        pts = None

    cx0, cy0 = max(0, int(x0)), max(0, int(y0))
    cx1, cy1 = min(W, int(x1)), min(H, int(y1))
    if cx1 <= cx0 or cy1 <= cy0:
        return None, None

    patch = image[cy0:cy1, cx0:cx1]
    if pts is None:
        mask = np.ones(patch.shape[:2], dtype=bool)
    else:
        m = np.zeros(patch.shape[:2], dtype=np.uint8)
        cv2.fillPoly(m, [pts - np.array([cx0, cy0], dtype=np.int32)], 1)
        mask = m.astype(bool)
    if not mask.any():
        return None, None
    return patch, mask

