#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import cv2
import numpy as np

from GRIME_AI.utils.resource_utils import ui_path

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QMessageBox
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi


class CompositeSliceDlg(QtWidgets.QDialog):
    compositeSliceCancelSignal = pyqtSignal()
    compositeSliceGenerateSignal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setModal(False)
        self.setWindowModality(Qt.NonModal)

        loadUi(ui_path("composite_slice/QDialog_CompositeSlice.ui"), self)

        self.label_Image.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.widthMultiplier = 0
        self.heightMultiplier = 0

        # Initialize from UI
        self.sliceCenter = int(self.horizontalSlider.value())
        self.sliceWidth = int(self.spinBox_Width.value())
        self.lineEdit_HorizontalPosition.setText(str(self.sliceCenter))

        # Push initial values into label (label setters trigger repaint)
        self.label_Image.setSliceCenter(self.sliceCenter)
        self.label_Image.setSliceWidth(self.sliceWidth)

        # Connect signals with correct signatures
        self.horizontalSlider.valueChanged.connect(self.valuechange)          # int -> slot(int)
        self.spinBox_Width.valueChanged.connect(self.spinBox_WidthChanged)    # int -> slot(int)

        self.pushButton_Generate.clicked.connect(self.pushButton_Generate_Clicked)
        self.pushButton_Generate.setStyleSheet('QPushButton {background-color: steelblue; color: white;}')

        self.pushButton_Cancel.clicked.connect(self.pushButton_Cancel_Clicked)

        self.label_Image.resized.connect(self.updateMultipliers)

        # Remove monkey-patching of label mouse events.
        # The label class already implements slice dragging/resizing.
        # If you need dialog-side coordination, use signals from the label instead.

        self.dragStartPosition = None
        self.draggingCenter = False
        self.draggingLeftEdge = False
        self.draggingRightEdge = False
        self._originalPixmap = None

    def showExtractedSlice(self, left_orig: int, right_orig: int):
        if not hasattr(self, "_originalImage"):
            return

        slice_array = self._originalImage[:, left_orig:right_orig, :]
        if slice_array.size == 0:
            return

        # Ensure contiguous memory and uint8 dtype
        slice_array = np.ascontiguousarray(slice_array, dtype=np.uint8)

        h, w, ch = slice_array.shape
        bytes_per_line = ch * w

        # Convert to bytes for QImage
        qimg = QImage(slice_array.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        slice_pixmap = QPixmap.fromImage(qimg)

        popup = QDialog(self)
        popup.setWindowTitle("Extracted Slice")
        layout = QVBoxLayout(popup)

        slice_label = QLabel(popup)
        slice_label.setPixmap(slice_pixmap.scaled(
            slice_label.sizeHint(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        layout.addWidget(slice_label)
        popup.resize(w, h)
        popup.show()

    def loadImage(self, filename: str):
        # Load with OpenCV
        numpy_bgr = cv2.imread(filename)
        if numpy_bgr is None:
            return

        # Convert to RGB and keep NumPy array for slicing later
        numpy_rgb = cv2.cvtColor(numpy_bgr, cv2.COLOR_BGR2RGB)
        self._originalImage = numpy_rgb
        self.label_Image.setOriginalImageShape(self._originalImage.shape)

        # Build a QImage from the NumPy array
        h, w, ch = numpy_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(numpy_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Wrap in QPixmap for display
        pixmap = QPixmap.fromImage(qimg)

        # Store original pixmap for rescaling/multipliers
        self._originalPixmap = pixmap

        # Compute multipliers relative to current label size
        self.widthMultiplier = pixmap.width() / max(1, self.label_Image.width())
        self.heightMultiplier = pixmap.height() / max(1, self.label_Image.height())

        # Show scaled image in the label
        self.label_Image.setPixmap(
            pixmap.scaled(
                self.label_Image.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

        # Update control ranges now that the label size is known
        max_center = max(0, self.label_Image.width())
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(max_center)
        self.horizontalSlider.setSingleStep(1)
        self.horizontalSlider.setPageStep(5)

        self.spinBox_Width.setMinimum(self.label_Image._minSliceWidth)
        self.spinBox_Width.setMaximum(max(1, self.label_Image.width()))
        self.spinBox_Width.setSingleStep(2)

        # Clamp current values to new ranges and push to label
        self.sliceCenter = max(0, min(max_center, self.sliceCenter))
        self.sliceWidth = max(self.label_Image._minSliceWidth,
                              min(self.label_Image.width(), self.sliceWidth))
        self.horizontalSlider.setValue(self.sliceCenter)
        self.spinBox_Width.setValue(self.sliceWidth)
        self.label_Image.setSliceCenter(self.sliceCenter)
        self.label_Image.setSliceWidth(self.sliceWidth)

        # Ensure label pixmap is rescaled consistently
        self._rescaleLabelPixmap()

    def updateMultipliers(self):
        if not self._originalPixmap or not self.label_Image.pixmap():
            return

        # Original image dimensions
        orig_w = self._originalPixmap.width()
        orig_h = self._originalPixmap.height()

        # Drawn pixmap dimensions (after scaling)
        drawn_pm = self.label_Image.pixmap()
        draw_w = drawn_pm.width()
        draw_h = drawn_pm.height()

        # Multipliers: how many original pixels per drawn pixel
        self.widthMultiplier = orig_w / max(1, draw_w)
        self.heightMultiplier = orig_h / max(1, draw_h)

        # Offsets: how far the drawn pixmap is inset inside the label
        self._xOffset = (self.label_Image.width() - draw_w) // 2
        self._yOffset = (self.label_Image.height() - draw_h) // 2

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # --- Margins ---
        left_margin = 20
        top_margin = 100
        right_margin = 20
        bottom_margin = 60  # spacing between image and buttons

        # Available box for the image
        target_w = max(1, self.width() - (left_margin + right_margin))
        target_h = max(1, self.height() - (top_margin + bottom_margin + 80))  # leave room for controls/buttons

        # --- 1) Resize label_Image preserving aspect ratio ---
        if self._originalPixmap:
            img_w = self._originalPixmap.width()
            img_h = self._originalPixmap.height()
            img_ar = img_w / img_h
            box_ar = target_w / target_h

            if img_ar > box_ar:
                fit_w = target_w
                fit_h = int(fit_w / img_ar)
            else:
                fit_h = target_h
                fit_w = int(fit_h * img_ar)

            x = left_margin + (target_w - fit_w) // 2
            y = top_margin

            self.label_Image.setGeometry(x, y, fit_w, fit_h)

            scaled = self._originalPixmap.scaled(
                fit_w, fit_h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.label_Image.setPixmap(scaled)

        # --- 2) Stretch horizontalSlider width with dialog ---
        slider_geom = self.horizontalSlider.geometry()
        new_slider_width = max(1, self.width() - 40)
        self.horizontalSlider.setGeometry(
            slider_geom.x(),
            slider_geom.y(),
            new_slider_width,
            slider_geom.height()
        )

        # --- 3) Center spinbox + labels as a group ---
        group_width = (
                self.label_2.width() +
                10 + self.spinBox_Width.width() +
                10 + self.label_3.width()
        )
        group_x = (self.width() - group_width) // 2
        group_y = self.spinBox_Width.y()  # keep same vertical position

        self.label_2.move(group_x, group_y)
        self.spinBox_Width.move(group_x + self.label_2.width() + 10, group_y)
        self.label_3.move(self.spinBox_Width.x() + self.spinBox_Width.width() + 10, group_y)

        # --- 4) Move and center buttons horizontally below the image ---
        button_y = self.label_Image.geometry().bottom() + bottom_margin
        spacing = 20
        total_width = self.pushButton_Generate.width() + spacing + self.pushButton_Cancel.width()
        start_x = (self.width() - total_width) // 2

        self.pushButton_Generate.move(start_x, button_y)
        self.pushButton_Cancel.move(start_x + self.pushButton_Generate.width() + spacing, button_y)

    def _rescaleLabelPixmap(self):
        if self._originalPixmap:
            scaled = self._originalPixmap.scaled(
                self.label_Image.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.label_Image.setPixmap(scaled)

    def getMultipliers(self):
        return (self.widthMultiplier, self.heightMultiplier, self.label_Image.getSliceCenter(), self.label_Image.getSliceWidth())

    def spinBox_WidthChanged(self, value: int):
        # Update width via label setter; repaint is automatic
        self.sliceWidth = int(value)
        self.label_Image.setSliceWidth(self.sliceWidth)

    def valuechange(self, value: int):
        # Update center via label setter; repaint is automatic
        self.sliceCenter = int(value)
        self.label_Image.setSliceCenter(self.sliceCenter)
        self.lineEdit_HorizontalPosition.setText(str(self.sliceCenter))

    def closeEvent(self, event):
        self.compositeSliceCancelSignal.emit()
        super(CompositeSliceDlg, self).closeEvent(event)

    def getSliceCenter(self):
        return self.sliceCenter

    def pushButton_Generate_Clicked(self):
        self.compositeSliceGenerateSignal.emit()

    def pushButton_Cancel_Clicked(self):
        self.compositeSliceCancelSignal.emit()
        self.close()

    def onCancel(self):
        self.compositeSliceCancelSignal.emit()
        self.close()

    # If you decide to keep dialog-driven dragging instead of label’s internal handling,
    # update the label only via setters (do NOT call drawCompositeSlice directly).
    # Leaving these here commented as reference.

    # def imageLabelMousePress(self, event):
    #     if event.buttons() == Qt.LeftButton:
    #         self.dragStartPosition = event.pos()
    #         self.draggingCenter = self.isWithinCenterLine(event.pos())
    #     elif event.buttons() == Qt.RightButton:
    #         self.dragStartPosition = event.pos()
   #         self.draggingLeftEdge = self.isWithinLeftEdgeLine(event.pos())
    #         self.draggingRightEdge = self.isWithinRightEdgeLine(event.pos())

    # def imageLabelMouseMove(self, event):
    #     if event.buttons() == Qt.LeftButton and self.draggingCenter:
    #         newCenter = self.sliceCenter + (event.pos().x() - self.dragStartPosition.x())
    #         self.horizontalSlider.setValue(int(newCenter))
    #         self.dragStartPosition = event.pos()
    #     elif event.buttons() == Qt.RightButton:
    #         if self.draggingLeftEdge:
    #             newWidth = (self.sliceCenter - event.pos().x()) * 2
    #             self.spinBox_Width.setValue(int(newWidth))
    #             self.dragStartPosition = event.pos()
    #         elif self.draggingRightEdge:
    #             newWidth = (event.pos().x() - self.sliceCenter) * 2
    #             self.spinBox_Width.setValue(int(newWidth))
    #             self.dragStartPosition = event.pos()

    # def imageLabelMouseRelease(self, event):
    #     self.draggingCenter = False
    #     self.draggingLeftEdge = False
    #     self.draggingRightEdge = False

    def isWithinCenterLine(self, pos):
        centerLineX = self.sliceCenter
        return abs(pos.x() - centerLineX) < 10  # 10 pixels tolerance

    def isWithinLeftEdgeLine(self, pos):
        leftEdgeX = self.sliceCenter - self.sliceWidth / 2
        return abs(pos.x() - leftEdgeX) < 10  # 10 pixels tolerance

    def isWithinRightEdgeLine(self, pos):
        rightEdgeX = self.sliceCenter + self.sliceWidth / 2
        return abs(pos.x() - rightEdgeX) < 10  # 10 pixels tolerance
