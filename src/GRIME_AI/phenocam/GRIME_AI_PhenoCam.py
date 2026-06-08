#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# --- Standard library imports ---
import datetime
import re
import ssl
import urllib.request

# --- Third-party imports ---
import requests
from bs4 import BeautifulSoup

# --- Local application imports ---
from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_ImageData import imageData

# --- Module-level globals ---
bChromeDriverLoaded = False
driver = None

# When False, time filtering ignores seconds and compares at HH:MM resolution,
# consistent with PhenoCam API behavior. When True, filtering is applied at
# full HH:MM:SS precision.
strict_time_filter = False


# ======================================================================================================================
# ======================================================================================================================
#  =====     =====     =====     =====     =====  class GRIME_AI_PhenoCam  =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class GRIME_AI_PhenoCam:

    def __init__(self):
        self.phenoCamURL = 'https://phenocam.nau.edu'

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def closeChromeDriver(self):
        global bChromeDriverLoaded
        global driver

        bChromeDriverLoaded = False
        driver.close()
        driver.quit()

    # ==================================================================================================================
    # THIS FUNCTION GETS THE LIST OF ALL AVAILABLE VISIBLE IMAGES FOR A GIVEN DATE.
    # ==================================================================================================================
    def getVisibleImages(self, dailyURLvisible, start_time, end_time):

        global bChromeDriverLoaded
        global driver

        dailyImagesList = dailyList([], [])

        # ----------
        # DETERMINE NUMBER OF IMAGES AVAILABLE FOR THE SELECTED DATE AND BUILT A LIST OF FILENAMES
        r = requests.get(dailyURLvisible)

        if r.status_code != 404:
            List = []

            # ----------
            # GET THE FILENAMES OF EACH VISIBLE LIGHT IMAGE FOR THE SELECTED DATE
            # ----------
            response = requests.get(dailyURLvisible)
            soup = BeautifulSoup(response.text, 'html5lib')

            links=[]
            for img in soup.find_all("img"):
                if img['src'].endswith("jpg"):
                    links.append(img['src'])

            listlength = len(links)

            for link in links:
                fullPathAndFilename = link
                fullPathAndFilename = fullPathAndFilename.replace('thumbnails', 'archive')
                filename = fullPathAndFilename.split('/')[-1]
                timestamp = filename.split('_')[-1]
                hour = int(timestamp[0:2])
                minute = int(timestamp[2:4])
                second = int(timestamp[4:6])
                if strict_time_filter:
                    photo_time = datetime.time(int(hour), int(minute), int(second))
                else:
                    photo_time = datetime.time(int(hour), int(minute), 0)

                bDownload = False
                if (photo_time >= start_time) and (photo_time <= end_time):
                    bDownload = True
                elif (start_time == datetime.time(0, 0, 0)) and (end_time == datetime.time(0, 0, 0)):
                    bDownload = True

                if bDownload:
                    fullPathAndFilename = self.phenoCamURL + fullPathAndFilename
                    List.append(imageData(fullPathAndFilename, timestamp[0:2], timestamp[2:4], timestamp[4:6]))

            dailyImagesList.setVisibleList(List)
        else:
            print("404: Update Daily Images")

        return dailyImagesList


    # ==================================================================================================================
    #
    # ==================================================================================================================
    def getStartEndDate(self):
        # providing url
        url = "https://phenocam.nau.edu/webcam/sites/NEON.D03.BARC.DP1.20002/"

        print("PHENOCAM URL Request: Get start and end dates...")

        # opening the url for reading
        ssl._create_default_https_context = ssl._create_unverified_context
        html = urllib.request.urlopen(url)

        # parsing the html file
        htmlParse = BeautifulSoup(html, 'html5lib')

        delimiter = '-'

        strTemp = htmlParse.text
        nIndex = strTemp.find("Start")
        strStartDate = strTemp[nIndex:nIndex+50]
        arrStartDate = re.search('\d{4}-\d{2}-\d{2}', strStartDate)
        nStartYear = int(arrStartDate.group().split(delimiter)[0])
        nStartMonth = int(arrStartDate.group().split(delimiter)[1])
        nStartDay = int(arrStartDate.group().split(delimiter)[2])

        nIndex = strTemp.find("Last")
        strLastDate = strTemp[nIndex:nIndex+50]
        arrLastDate = re.search('\d{4}-\d{2}-\d{2}', strLastDate)
        nEndYear = int(arrLastDate.group().split(delimiter)[0])
        nEndMonth = int(arrLastDate.group().split(delimiter)[1])
        nEndDay = int(arrLastDate.group().split(delimiter)[2])

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def getStartDate(self):

        # providing url
        url = "https://phenocam.nau.edu/webcam/sites/NEON.D03.BARC.DP1.20002/"

        print("PHENOCAM URL Request: Get start date...")

        try:
            # opening the url for reading
            ssl._create_default_https_context = ssl._create_unverified_context
            html = urllib.request.urlopen(url)

            # parsing the html file
            htmlParse = BeautifulSoup(html, 'html5lib')

            delimiter = '-'

            strTemp = htmlParse.text
            nIndex = strTemp.find("Start")
            strStartDate = strTemp[nIndex:nIndex+50]
            arrStartDate = re.search('\d{4}-\d{2}-\d{2}', strStartDate)
            nYear = int(arrStartDate.group().split(delimiter)[0])
            nMonth = int(arrStartDate.group().split(delimiter)[1])
            nDay = int(arrStartDate.group().split(delimiter)[2])
        except Exception:
            nYear = 1970
            nMonth = 1
            nDay = 1

        return nYear, nMonth, nDay

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def getEndDate(self):
        nYear = 1970
        nMonth = 1
        nDay = 1

        # providing url
        url = "https://phenocam.nau.edu/webcam/sites/NEON.D03.BARC.DP1.20002/"

        print("PHENOCAM URL Request: Get end date...")

        try:
            # opening the url for reading
            ssl._create_default_https_context = ssl._create_unverified_context
            html = urllib.request.urlopen(url)

            # parsing the html file
            htmlParse = BeautifulSoup(html, 'html5lib')

            delimiter = '-'

            strTemp = htmlParse.text
            nIndex = strTemp.find("Last")
            strLastDate = strTemp[nIndex:nIndex+50]
            arrLastDate = re.search('\d{4}-\d{2}-\d{2}', strLastDate)
            nYear = int(arrLastDate.group().split(delimiter)[0])
            nMonth = int(arrLastDate.group().split(delimiter)[1])
            nDay = int(arrLastDate.group().split(delimiter)[2])
        except Exception:
            pass

        return nYear, nMonth, nDay

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def getPhenocamImageCount(self, siteCode, domainCode, start_date, end_date, start_time, end_time):

        imageList = dailyList([], [])

        date = start_date

        progressBar = QProgressWheel()
        progressBar.setRange(0, (end_date-start_date).days)
        progressBar.setWindowTitle("Culling images by Date/Time")
        progressBar.show()

        i = 1
        while date <= end_date:
            progressBar.setValue(i)

            PhenocamURL = 'https://phenocam.nau.edu/webcam/browse/NEON.D10.ARIK.DP1.20002' + '/' + str(date.year) + '/' + str(date.month).zfill(2) + '/' + str(date.day).zfill(2)

            PhenocamURL = PhenocamURL.replace('ARIK', siteCode); PhenocamURL = PhenocamURL.replace('D10', domainCode)

            tmpList = GRIME_AI_PhenoCam().getVisibleImages(PhenocamURL, start_time, end_time)

            imageList.setVisibleList(tmpList.getVisibleList())

            date += datetime.timedelta(days=1)

            i += 1

        progressBar.close()
        del progressBar

        try:
            GRIME_AI_PhenoCam.closeChromeDriver()
        except Exception:
            pass

        nImageCount = len(imageList.visibleList)

        del imageList

        return nImageCount

# ======================================================================================================================
# THIS CLASS WILL HOLD THE NAMES OF ALL THE VISIBLE AND INFRARED (IR) IMAGES FOR A SPECIFIED DATE RANGE
# ======================================================================================================================
class dailyList:
    def __init__(self, visibleList, IRList):
        self.visibleList = visibleList
        self.IRList = IRList

    def clear(self):
        self.visibleList = []
        self.IRList = []

    def setVisibleList(self, visibleList):
        self.visibleList.extend(visibleList)

    def setIRList(self, IRList):
        self.IRList = IRList

    def getVisibleList(self):
        return self.visibleList

    def getIRList(self):
        return self.IRList


