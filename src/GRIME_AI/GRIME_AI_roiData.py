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
        x = int(self.displayROI.x() * heightMultiplier)
        y = int(self.displayROI.y() * widthMultiplier)
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

