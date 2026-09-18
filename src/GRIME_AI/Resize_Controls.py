#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# ======================================================================================================================
#
# ======================================================================================================================
class Resize_Controls:
    def __init__(self):
        self.className = "Resize_Controls"
        self.instance = 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RESIZE TAB 0 - NEON SITES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def resizeTab_0(self, selfControl, event):
        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RESIZE TAB 1 - NEON DOWNLOAD MANAGER
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def resizeTab_1(self, selfControl, event):
        windowSize = event.size()
        oldWindowSize = event.oldSize()

        gap = 10
        border = 25
        vertical_margin = 50
        newWidth = (int)(windowSize.width() - gap - (2 * border))
        newHeight = (int)(windowSize.height() * 0.50)

        topLeftCorner = selfControl.NEON_tableProducts.pos()

        # SPREADSHEET
        left = topLeftCorner.x()
        top  = topLeftCorner.y()
        width  = newWidth
        height = newHeight
        selfControl.NEON_tableProducts.setGeometry(left, top, width, height)

        # DOWNLOAD GROUPBOX
        left = left
        top  = top + newHeight + gap
        width  = selfControl.groupBox_NEONDownloadManager_OutputFolders.width()
        height = selfControl.groupBox_NEONDownloadManager_OutputFolders.height()
        selfControl.groupBox_NEONDownloadManager_OutputFolders.setGeometry(left, top, width, height)

        # DOWNLOAD BUTTON
        #DOWNLOAD GROUPBOX VERTICAL CENTER
        vertCenter = top + (int)(height / 2.0)

        width = selfControl.pushButton_RetrieveNEONData.width()
        height = selfControl.pushButton_RetrieveNEONData.height()
        left = left + selfControl.groupBox_NEONDownloadManager_OutputFolders.width() + gap
        top  = vertCenter - (int)(height / 2.0)
        selfControl.pushButton_RetrieveNEONData.setGeometry(left, top, width, height)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RESIZE TAB 2 - USGS SITES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def resizeTab_2(self, selfControl, event):
        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RESIZE TAB 3 - USGS DOWNLOAD MANAGER (merged into Tab 2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def resizeTab_3(self, selfControl, event):
        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RESIZE TAB 4 - NEON SITES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def resizeTab_4(self, selfControl, event):
        windowSize = event.size()
        oldWindowSize = event.oldSize()

        gap = 10
        border = 25
        vertical_margin = 50
        newWidthAvailable = (int)(windowSize.width() - (gap * (2 + 1)) - (2*border))

        one   = selfControl.labelOriginalImage.pos()
        paneWidth  = (int)(newWidthAvailable * 0.50)

        # --------------------------------------------------------------------------------
        # TAB WIDGET - RESIZE THE TAB WIDGET
        # --------------------------------------------------------------------------------
        left = selfControl.tabWidget.x()
        left = gap
        top = selfControl.tabWidget.y()
        top = 0
        width = (int)(windowSize.width() - (2.0 * gap))
        height = windowSize.height()
        selfControl.tabWidget.setGeometry(left, top, width, height)

        # --------------------------------------------------------------------------------
        # ROW 1
        # --------------------------------------------------------------------------------
        # ROW 1, COL 1 - ORIGINAL IMAGE
        left   = one.x()
        top    = one.y()
        width  = paneWidth
        height = (int)((windowSize.height() / 2.25) - vertical_margin)
        selfControl.labelOriginalImage.setGeometry(left, top, width, height)

        # ROW 1, COL 2 - EDGE IMAGE
        left   = left + paneWidth + gap
        selfControl.labelEdgeImage.setGeometry(left, top, width, height)

        # --------------------------------------------------------------------------------
        # ROW 2
        # --------------------------------------------------------------------------------
        # ROW 2, COL 1 - MASK IMAGE
        left   = one.x()
        top    = (int)(one.y() + gap + (windowSize.height() / 2.25) - vertical_margin)
        width  = paneWidth
        height = (int)((windowSize.height() / 2.25) - vertical_margin)
        selfControl.labelColorSegmentation.setGeometry(left, top, width, height)

        # ROW 2, COL 2 - ROI LIST
        left   = left + paneWidth + gap
        selfControl.tableWidget_ROIList.setGeometry(left, top, width, height)
