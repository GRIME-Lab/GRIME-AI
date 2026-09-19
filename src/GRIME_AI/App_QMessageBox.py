#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

# ======================================================================================================================
#
# ======================================================================================================================
class App_QMessageBox(QMessageBox):

    def __init__(self, strTitle, strMsg, buttons=QMessageBox.Ok, icon=QMessageBox.Information):
        QMessageBox.__init__(self)

        self.strTitle = strTitle
        self.strMsg = strMsg
        self.setIcon(icon)
        self.setStandardButtons(buttons)

        self.setWindowTitle(self.strTitle)
        self.setText(strMsg)


    def displayMsgBox(self, on_top=True):
        if on_top:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, True);
        else:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, False);

        response = self.exec_()

        return response
