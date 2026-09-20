#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from appcore.utils.resource_utils import ui_path

from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi

userCancelled = "Cancelled"
userOk = "OK"

# ======================================================================================================================
#
# ======================================================================================================================
class ReleaseNotesDlg(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)

        loadUi(ui_path("release_notes/QDialog_ReleaseNotes.ui"), self)