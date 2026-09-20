#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from typing import List, Dict

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from appcore.AnnotatorLabel import AnnotatorLabel


class ImageAnnotatorDialog(QtWidgets.QDialog):
    """
    Wraps AnnotatorLabel in a scroll area; the label always expands
    or shrinks the image to fill its area, up to widget bounds.
    Enter accepts.
    """

    def __init__(self, pixmap, mode='drag', label=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Annotate Image")

        # remember the label to apply to every shape
        self._current_label = label or {"id": -1, "name": "unknown"}

        # optionally half-screen if too large
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        iw, ih = pixmap.width(), pixmap.height()
        sw, sh = screen.width(), screen.height()
        if iw > sw or ih > sh:
            pixmap = pixmap.scaled(
                sw // 2, sh // 2,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

        # build the annotator widget and pass in the shape label
        self.label = AnnotatorLabel(
            pixmap,
            mode=mode,
            shape_label=self._current_label,
            parent=self
        )

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.label)
        scroll.setWidgetResizable(True)
        scroll.setMouseTracking(True)
        self.label.setMouseTracking(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(scroll)

        # start within screen bounds
        self.resize(min(iw, sw), min(ih, sh))
        self.setSizeGripEnabled(True)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.accept()
        else:
            super().keyPressEvent(ev)

    def getAnnotations(self) -> List[Dict]:
        """
        Returns a list of shapes drawn:
          - type: 'bbox' | 'click' | 'drag'
          - points: list of (x, y) tuples
          - label: {id:int, name:str}
        """
        output = []
        for shape in self.label.shapes:
            pts = [(pt.x(), pt.y()) for pt in shape["points"]]
            output.append({
                "type": shape["type"],
                "points": pts,
                "label": shape["label"]
            })
        return output

    def setAnnotations(self, annotations: list):
        """
        Sets annotation data to the label for display.
        Each shape in `annotations` includes type, label, and points.
        """
        self.label.shapes = [
            {
                "type": shape["type"],
                "label": shape["label"],
                "points": shape["points"]  # ← 🔧 use points as-is
            }
            for shape in annotations
        ]
        self.label.update()
