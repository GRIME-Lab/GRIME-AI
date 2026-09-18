#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import datetime

class ProductTable():
    def __init__(self):
        self.nStartYear = 0
        self.nStartMonth = 0
        self.nStartDay = 0
        self.strStartDate = ''
        self.start_date = datetime.date(1970, 1, 1)
        self.startTime = 0

        self.nEndYear = 0
        self.nEndMonth = 0
        self.nEndDay = 0
        self.strEndDate = ''
        self.end_date = datetime.date(1970, 1, 1)
        self.endTime = 0

        self.delta = self.end_date - self.start_date

    def fetchTableDates(self, productTable, nRow):
        startDateCol = 2
        self.nStartYear = productTable.cellWidget(nRow, startDateCol).date().year()
        self.nStartMonth = productTable.cellWidget(nRow, startDateCol).date().month()
        self.nStartDay = productTable.cellWidget(nRow, startDateCol).date().day()
        self.strStartDate = str(self.nStartYear) + '-' + str(self.nStartMonth).zfill(2)
        self.start_date = datetime.date(self.nStartYear, self.nStartMonth, self.nStartDay)

        startTimeCol = 4
        self.startTime = productTable.cellWidget(nRow, startTimeCol).dateTime().toPyDateTime().time()

        endDateCol = 3
        self.nEndYear = productTable.cellWidget(nRow, endDateCol).date().year()
        self.nEndMonth = productTable.cellWidget(nRow, endDateCol).date().month()
        self.nEndDay = productTable.cellWidget(nRow, endDateCol).date().day()
        self.strEndDate = str(self.nEndYear) + '-' + str(self.nEndMonth).zfill(2)
        self.end_date = datetime.date(self.nEndYear, self.nEndMonth, self.nEndDay)

        endTimeCol = 5
        self.endTime = productTable.cellWidget(nRow, endTimeCol).dateTime().toPyDateTime().time()

        self.delta = self.end_date - self.start_date

        return self.start_date, self.startTime, self.end_date, self.endTime

    def getStartDate(self):
        return self.strStartDate
    def getStartDate(self):
        return self.start_date

    def getEndDate(self):
        return self.strEndDate
    def getEndDate(self):
        return self.end_date

    def getDelta(self):
        return self.delta

    def getStartTime(self):
        return self.startTime

    def getEndTime(self):
        return self.endTime