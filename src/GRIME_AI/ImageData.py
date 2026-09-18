#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# ======================================================================================================================
# THIS CLASS WILL HOLD THE FULLY QUALIFIED URL PATH AND THE TIME STAMP FOR A PARTICULAR IMAGE SELECTED
# BY THE END-USER.
# ======================================================================================================================
class imageData():

    def __init__(self, fullPathAndFilename, hours, minutes, seconds):
        self.fullPathAndFilename = fullPathAndFilename
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds


