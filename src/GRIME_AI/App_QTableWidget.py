#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from PyQt5 import QtGui
import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QTableWidget, QCheckBox, QHeaderView

# ======================================================================================================================
#
# ======================================================================================================================
class App_QTableWidget(QTableWidget):

    def __init__(self, parent=None):
        QTableWidget.__init__(self, parent)
        #self.chkbox1 = QCheckBox(self.horizontalHeader())

    def resizeEvent(self, event=None):
        super().resizeEvent(event)
        #self.chkbox1.setGeometry(QtCore.QRect((self.columnWidth(0)/2), 2, 16, 17))
        #self.chkbox1.setGeometry(QtCore.QRect(0, 2, 16, 17))
