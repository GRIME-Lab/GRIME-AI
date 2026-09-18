#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os

from GRIME_AI.utils.resource_utils import ui_path

from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QIcon, QPen
from PyQt5.QtWidgets import (
    QDialog,
    QListWidgetItem,
    QScrollBar,
    QAbstractItemView,
    QStyledItemDelegate,
    QStyle
)
from PyQt5.uic import loadUi


class BorderDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._border_color = Qt.darkBlue    # or Qt.GlobalColor.steelblue
        self._border_width = 2

    def paint(self, painter, option, index):
        # 1) Let Qt draw the icon (and text if any)
        super().paint(painter, option, index)

        # 2) If selected, draw a solid border
        if option.state & QStyle.State_Selected:
            rect = option.rect.adjusted(
                self._border_width//2,
                self._border_width//2,
                -self._border_width//2,
                -self._border_width//2
            )
            pen = QPen(self._border_color, self._border_width)
            painter.save()
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)
            painter.restore()


class ImageNavigationDlg(QDialog):
    imageIndexSignal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        loadUi(ui_path("image_navigation/QDialog_ImageNavigation.ui"), self)

        # ── Micro‐batch loader state ───────────────────────────────
        # these control how many thumbs we load per cycle,
        # and how long to wait before the next cycle.
        self._pendingThumbnails = []   # list of (QListWidgetItem, filePath)
        self._batchSize = 5            # load 5 thumbnails at a time
        self._batchDelay = 50          # ms between batches

        # ── SpinBox setup ─────────────────────────────────────────
        sb = self.spinBoxImageIndex
        sb.setMinimum(1)
        sb.setMaximum(1)
        sb.setValue(1)
        sb.setKeyboardTracking(False)
        sb.valueChanged.connect(self.onImageIndexChanged)

        self.pushButtonResetImageIndex.clicked.connect(self.resetImageIndex)

        # remove padding inside the scrollArea’s contents
        scroll_contents = self.scrollAreaFilmstrip.widget()
        if scroll_contents and scroll_contents.layout():
            scroll_contents.layout().setContentsMargins(0, 0, 0, 0)
            scroll_contents.layout().setSpacing(0)


        # ── Filmstrip (QListWidget) setup ──────────────────────────
        lw = self.listWidgetFilmstrip

        # remove any frame/margins on the list widget
        lw.setContentsMargins(0, 0, 0, 0)
        lw.setFrameShape(self.listWidgetFilmstrip.NoFrame)
        lw.setFrameShadow(self.listWidgetFilmstrip.Plain)
        #lw.setItemDelegate( BorderDelegate(lw) )

        lw.setStyleSheet("""
            QListWidget::item {
                border: none;             /* no default border */
                padding: 0px;             /* kill any padding */
                margin: 0px;              /* kill any margin */
                background: transparent;  /* no default background */
            }
            QListWidget::item:selected {
                background: transparent;          /* no grey overlay */
                border: 2px solid steelblue;     /* your 2 px solid color */
            }
        """)

        lw.setViewMode(lw.IconMode)
        lw.setFlow(lw.LeftToRight)
        lw.setWrapping(False)
        lw.setMovement(lw.Static)
        lw.setResizeMode(lw.Adjust)
        lw.setSpacing(5)

        # Enforce a sensible icon size
        enforced = QSize(100, 100)
        lw.setIconSize(enforced)
        lw.setUniformItemSizes(True)

        # make each “cell” exactly as big as the icon
        lw.setGridSize(enforced + QSize(4, 4))

        # Connect row changes back to your spinBox
        lw.currentRowChanged.connect(self.onFilmstripSelectionChanged)

        # Wire up the Close button
        self.buttonBox.rejected.connect(self.close)

        self.spinBoxImageIndex.editingFinished.connect(
            self._onSpinboxIndexEntered
        )


    def _onSpinboxIndexEntered(self):
        # get the 0-based row
        row = self.spinBoxImageIndex.value() - 1
        lw  = self.listWidgetFilmstrip

        # guard against out-of-range
        if row < 0 or row >= lw.count():
            return

        # select it and scroll it into center
        lw.setCurrentRow(row)
        lw.scrollToItem(
            lw.item(row),
            QAbstractItemView.PositionAtCenter
        )


    # ───────────────────────────────────────────────────────────────
    def onImageIndexChanged(self, value):
        if self.imageCount > 0 and value != getattr(self, 'lastEmittedValue', None):
            self.lastEmittedValue = value
            self.imageIndexSignal.emit(value)

    # ───────────────────────────────────────────────────────────────
    def resetImageIndex(self):
        self.spinBoxImageIndex.setValue(1)

    # ───────────────────────────────────────────────────────────────
    def setImageIndex(self, idx):
        self.spinBoxImageIndex.blockSignals(True)
        self.spinBoxImageIndex.setValue(idx)
        self.spinBoxImageIndex.blockSignals(False)

    # ───────────────────────────────────────────────────────────────
    def setImageCount(self, count):
        self.imageCount = count
        self.labelImageCountNumber.setText(str(count))
        self.spinBoxImageIndex.setMaximum(max(1, count))

    # ───────────────────────────────────────────────────────────────
    def reset(self):
        self.spinBoxImageIndex.blockSignals(True)
        self.spinBoxImageIndex.setValue(1)
        self.spinBoxImageIndex.blockSignals(False)
        self.lastEmittedValue = 1
        self.imageIndexSignal.emit(1)

    # ───────────────────────────────────────────────────────────────
    def onCancel(self):
        self.close()

    # ─── FILMSTRIP POPULATION ─────────────────────────────────────
    def setImageList(self, dailyImageList):
        """
        1) Clear the old thumbnails.
        2) Populate placeholders in the QListWidget.
        3) Kick off _loadNextBatch() via a QTimer.
        """
        # update our internal list and UI controls
        self.dailyImageList = dailyImageList
        lw = self.listWidgetFilmstrip

        lw.clear()
        self._pendingThumbnails.clear()
        self.setImageCount(len(dailyImageList))

        # queue one blank item per path
        iconSize = lw.iconSize()
        extra = 4  # padding around each icon
        for entry in dailyImageList:
            path = str(entry.fullPathAndFilename)
            item = QListWidgetItem(QIcon(), "")
            # sizeHint must be a QSize, not QtCore.QSize+QSize

            #w = iconSize.width() + extra
            #h = iconSize.height() + extra
            #item.setSizeHint(QSize(w, h))
            item.setSizeHint(iconSize + QSize(2, 2))

            lw.addItem(item)
            self._pendingThumbnails.append((item, path))

        # start loading in micro‐batches
        QTimer.singleShot(self._batchDelay, self._loadNextBatch)

        # select & scroll to the first immediately
        if lw.count():
            lw.setCurrentRow(0)

        # adjust fixed heights so the filmstrip stays the correct size
        icon_h = iconSize.height()
        scroll_h = (lw.horizontalScrollBar().sizeHint().height()
                    if isinstance(lw.horizontalScrollBar(), QScrollBar) else 0)
        frame = lw.frameWidth() * 2
        total_h = icon_h + scroll_h + frame

        lw.setFixedHeight(total_h)
        # keep your QScrollArea in sync:
        self.scrollAreaFilmstrip.setFixedHeight(
            total_h + self.scrollAreaFilmstrip.frameWidth() * 2
        )

    def _loadNextBatch(self):
        """
        Load up to self._batchSize thumbnails, assign them to items,
        then reschedule if there are more pending.
        """
        lw = self.listWidgetFilmstrip
        iconSize = lw.iconSize()

        for _ in range(min(self._batchSize, len(self._pendingThumbnails))):
            item, path = self._pendingThumbnails.pop(0)
            if not os.path.exists(path):
                continue
            pix = QPixmap(path)
            if pix.isNull():
                continue
            thumb = pix.scaled(
                iconSize,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            item.setIcon(QIcon(thumb))

        if self._pendingThumbnails:
            QTimer.singleShot(self._batchDelay, self._loadNextBatch)

    # ───────────────────────────────────────────────────────────────
    @pyqtSlot(int)
    def onFilmstripSelectionChanged(self, row):
        if row >= 0:
            self.setImageIndex(row + 1)
            self.onImageIndexChanged(row + 1)
