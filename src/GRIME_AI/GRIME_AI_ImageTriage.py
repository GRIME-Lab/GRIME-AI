#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os
import json
import shutil

import cv2
import numpy as np

from GRIME_AI.GRIME_AI_Utils import GRIME_AI_Utils
from GRIME_AI.GRIME_AI_Color import GRIME_AI_Color
from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI.GRIME_AI_ImageQualityCheck import ImageQualityAnalyzer
from datetime import datetime


class GRIME_AI_ImageTriage:

    def __init__(self, show_gui=True):
        self.className = "GRIME_AI_ImageTriage"
        self.show_gui  = show_gui

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def cleanImages(self, folder, bFetchRecursive, blurThreshhold, shiftSize, brightnessMin, brightnessMAX,
                    bCreateReport, bMoveImages, bCorrectAlignment, bSavePolylines,
                    strReferenceImageFilename, rotationThreshold,
                    laplacian_threshold=150.0, blur_logic="OR",
                    focus_roi=None,
                    use_color_imbalance=False,
                    color_imbalance_threshold=0.5):
        """
        Triage images in folder, flagging those that are blurry, too dark, or too light.

        Parameters
        ----------
        focus_roi : list or None
            Normalised [x, y, w, h] defining the subregion used for blur scoring.
            Brightness is always computed over the full frame. If None, the full
            frame is used for blur scoring. Must match the region used during
            calibration for thresholds to be valid.
        """

        badImageCount    = 0
        rotationAngle    = 0.0
        horizontal_shift = 0.0
        vertical_shift   = 0.0

        extensions    = ('.jpg', '.jpeg', '.png')
        myGRIMe_Color = GRIME_AI_Color()

        analyzer = ImageQualityAnalyzer(
            fft_shift_radius    = shiftSize,
            fft_blur_threshold  = blurThreshhold,
            laplacian_threshold = laplacian_threshold,
            blur_logic          = blur_logic,
            brightness_min      = brightnessMin,
            brightness_max      = brightnessMAX,
            use_contrast        = False,
            use_exposure_clipping = False,
            resize_percent               = 50.0,
            focus_roi                    = focus_roi,
            use_color_imbalance          = use_color_imbalance,
            color_imbalance_threshold    = color_imbalance_threshold,
        )

        reportRows = []   # accumulated for the xlsx report

        if bCreateReport:
            runTimestamp     = datetime.now()
            reportStem       = 'ImageTriage_' + runTimestamp.strftime("%Y%m%d_%H%M%S")
            csvFilename      = reportStem + '.csv'
            imageQualityFile = os.path.join(folder, csvFilename)
            csvFile          = open(imageQualityFile, 'a', newline='')
            csvFile.write('Focus Value, Focus Attrib, Laplacian Value, Intensity Value, Intensity Attrib., '
                          'Color Imbalance, Color Imbalance Attrib., '
                          'Quality, '
                          'Rotation, H. Shift, V. Shift, Filename, Moved\n')

        imageCount = GRIME_AI_Utils().get_image_count(folder, extensions)
        if self.show_gui:
            progressBar = QProgressWheel(0, imageCount + 1)
            progressBar.show()

        imageIndex = 0

        if bSavePolylines:
            polyFolder = os.path.join(folder, "poly")
            os.makedirs(polyFolder, exist_ok=True)

        if bCorrectAlignment:
            warpFolder = os.path.join(folder, "warp")
            os.makedirs(warpFolder, exist_ok=True)

        refImage = None
        if bCorrectAlignment and len(strReferenceImageFilename) > 0:
            refImage = myGRIMe_Color.loadColorImage(strReferenceImageFilename)

        file_count, files = GRIME_AI_Utils().getFileList(folder, extensions, bFetchRecursive)

        for file in files:
            if self.show_gui and progressBar._is_closed:
                break

            if self.show_gui:
                progressBar.setWindowTitle(file)
                progressBar.setValue(imageIndex)
                progressBar.repaint()
            imageIndex += 1

            ext = os.path.splitext(file)[-1].lower()
            if ext not in extensions:
                continue

            filename   = os.path.join(folder, file)
            numpyImage = myGRIMe_Color.loadColorImage(filename)

            quality = analyzer.analyze(numpyImage)

            blur_fft       = quality.get("blur_fft",      0.0)
            blur_laplacian = quality.get("blur_laplacian", 0.0)
            brightness     = quality.get("brightness",     0.0)
            is_blurry         = quality.get("is_blurry",          False)
            is_too_dark       = quality.get("is_too_dark",        False)
            is_too_light      = quality.get("is_too_light",       False)
            is_color_imbal    = quality.get("is_color_imbalanced", False)
            color_imbalance   = quality.get("color_imbalance",     0.0)

            bBad           = False
            strFocusMetric = "Nominal"
            strIntensity   = "Nominal"

            if is_blurry:
                strFocusMetric = "Blurry"
                bBad           = True

            if is_too_dark:
                strIntensity = "Too Dark"
                bBad         = True
            elif is_too_light:
                strIntensity = "Too Light"
                bBad         = True

            strColorImbalance = "Nominal"
            if is_color_imbal:
                strColorImbalance = "Color Imbalance"
                bBad              = True

            if refImage is not None and bCorrectAlignment:
                rotationAngle, poly_img, warp_img = self.checkImageAlignment(refImage, numpyImage)

                baseFilename = os.path.basename(file)

                if bSavePolylines:
                    cv2.imwrite(os.path.join(polyFolder, baseFilename + "_poly.jpg"), poly_img)

                if bCorrectAlignment and rotationAngle > rotationThreshold:
                    cv2.imwrite(os.path.join(warpFolder, baseFilename + "_align.jpg"), warp_img)

                grayImage = cv2.cvtColor(numpyImage, cv2.COLOR_RGB2GRAY)
                horizontal_shift, vertical_shift = self.checkImageShift(refImage, grayImage)

            bActuallyMoved = False
            if bMoveImages and bBad:
                filepath       = os.path.dirname(file)
                tempFolder     = os.path.join(filepath, "MovedImages")
                os.makedirs(tempFolder, exist_ok=True)
                shutil.move(file, tempFolder)
                filename       = os.path.join(tempFolder, os.path.basename(file))
                bActuallyMoved = True

            if bBad:
                badImageCount += 1

            if bCreateReport:
                formula = f'=HYPERLINK("{filename}", "{os.path.basename(filename)}")'
                if ',' in formula:
                    safe_formula = formula.replace('"', '""')
                    formula      = f'"{safe_formula}"'
                quotedHyperlink = f'"{formula}"'

                strQuality = 'Review' if bBad else 'Nominal'
                strOutputString = '%3.2f,%s,%3.2f,%3.2f,%s,%3.3f,%s,%s,%3.2f,%3.2f,%3.2f,%s,%s\n' % (
                    blur_fft, strFocusMetric,
                    blur_laplacian,
                    brightness, strIntensity,
                    color_imbalance, strColorImbalance,
                    strQuality,
                    rotationAngle, horizontal_shift, vertical_shift,
                    formula,
                    'Y' if bActuallyMoved else 'N'
                )
                csvFile.write(strOutputString)

                reportRows.append((
                    round(blur_fft, 2), strFocusMetric,
                    round(blur_laplacian, 2),
                    round(brightness, 2), strIntensity,
                    round(color_imbalance, 3), strColorImbalance,
                    strQuality,
                    round(rotationAngle, 2), round(horizontal_shift, 2), round(vertical_shift, 2),
                    filename,
                    'Y' if bActuallyMoved else 'N'
                ))

        if badImageCount == 0:
            strMessage = 'No bad images found.'
            print(strMessage)
            if self.show_gui:
                msgBox   = GRIME_AI_QMessageBox('Image Triage', strMessage)
                response = msgBox.displayMsgBox()

        if bCreateReport:
            csvFile.close()

            settings = self._build_settings_dict(
                runTimestamp, folder, bFetchRecursive,
                blurThreshhold, shiftSize, laplacian_threshold, blur_logic,
                brightnessMin, brightnessMAX,
                focus_roi, use_color_imbalance, color_imbalance_threshold,
                bMoveImages, bCorrectAlignment, bSavePolylines,
                strReferenceImageFilename, rotationThreshold,
                images_processed=len(reportRows), images_flagged=badImageCount)

            # JSON sidecar: same stem as the CSV, '_settings.json' suffix
            try:
                sidecarPath = os.path.join(folder, reportStem + '_settings.json')
                with open(sidecarPath, 'w') as jsonFile:
                    json.dump(settings, jsonFile, indent=4)
            except Exception as e:
                print(f"[{self.className}] Could not write settings sidecar: {e}")

            # Excel workbook: results and settings on separate worksheets
            try:
                xlsxPath = os.path.join(folder, reportStem + '.xlsx')
                self._write_xlsx_report(xlsxPath, reportRows, settings)
            except Exception as e:
                print(f"[{self.className}] Could not write xlsx report: {e}")

        if self.show_gui:
            progressBar.close()
            del progressBar

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def _build_settings_dict(self, runTimestamp, folder, bFetchRecursive,
                             blurThreshhold, shiftSize, laplacian_threshold, blur_logic,
                             brightnessMin, brightnessMAX,
                             focus_roi, use_color_imbalance, color_imbalance_threshold,
                             bMoveImages, bCorrectAlignment, bSavePolylines,
                             strReferenceImageFilename, rotationThreshold,
                             images_processed=0, images_flagged=0):
        """Assemble the complete run configuration for the settings sidecar
        and the xlsx Triage Settings worksheet."""
        try:
            from GRIME_AI.version import SW_VERSION
        except Exception:
            SW_VERSION = "unknown"

        return {
            "software":                  "GRIME AI Image Triage",
            "software_version":          SW_VERSION,
            "run_timestamp":             runTimestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "image_folder":              folder,
            "fetch_recursive":           bool(bFetchRecursive),
            "fft_blur_threshold":        blurThreshhold,
            "fft_shift_radius":          shiftSize,
            "laplacian_threshold":       laplacian_threshold,
            "blur_logic":                blur_logic,
            "brightness_min":            brightnessMin,
            "brightness_max":            brightnessMAX,
            "focus_roi":                 focus_roi,
            "use_color_imbalance":       bool(use_color_imbalance),
            "color_imbalance_threshold": color_imbalance_threshold,
            "resize_percent":            50.0,
            "move_images":               bool(bMoveImages),
            "correct_alignment":         bool(bCorrectAlignment),
            "save_polylines":            bool(bSavePolylines),
            "reference_image":           strReferenceImageFilename,
            "rotation_threshold":        rotationThreshold,
            "images_processed":          images_processed,
            "images_flagged":            images_flagged,
        }

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def _write_xlsx_report(self, xlsxPath, reportRows, settings):
        """Write the triage results and run settings to an Excel workbook,
        one worksheet each ('Triage Results', 'Triage Settings')."""
        from openpyxl import Workbook
        from openpyxl.styles import Font

        wb = Workbook()

        # --- Sheet 1: Triage Results ---
        ws = wb.active
        ws.title = "Triage Results"
        headers = ['Focus Value', 'Focus Attrib', 'Laplacian Value',
                   'Intensity Value', 'Intensity Attrib.',
                   'Color Imbalance', 'Color Imbalance Attrib.',
                   'Quality', 'Rotation', 'H. Shift', 'V. Shift',
                   'Filename', 'Moved']
        ws.append(headers)
        for cell in ws[1]:
            cell.font = Font(bold=True)

        for row in reportRows:
            ws.append(list(row))
            # Render the filename as a clickable hyperlink (basename shown)
            filenameCell           = ws.cell(row=ws.max_row, column=12)
            fullPath               = row[11]
            filenameCell.value     = os.path.basename(fullPath)
            filenameCell.hyperlink = fullPath
            filenameCell.style     = "Hyperlink"

        ws.freeze_panes = "A2"

        # --- Sheet 2: Triage Settings ---
        wsSettings = wb.create_sheet("Triage Settings")
        wsSettings.append(["Setting", "Value"])
        for cell in wsSettings[1]:
            cell.font = Font(bold=True)
        for key, value in settings.items():
            if isinstance(value, (list, tuple, dict)):
                value = json.dumps(value)
            wsSettings.append([key, value])

        # Reasonable column widths
        for sheet, widths in ((ws, {12: 45}), (wsSettings, {1: 28, 2: 60})):
            for colIdx, width in widths.items():
                colLetter = sheet.cell(row=1, column=colIdx).column_letter
                sheet.column_dimensions[colLetter].width = width

        wb.save(xlsxPath)

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def checkImageShift(self, refImage, image):
        refImageGray = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)
        imageGray    = image

        orb                      = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(refImageGray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(imageGray,    None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(descriptors1, descriptors2), key=lambda x: x.distance)

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, _             = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        horizontal_shift = M[0, 2]
        vertical_shift   = M[1, 2]

        return horizontal_shift, vertical_shift

    # ==================================================================================================================
    #
    # ==================================================================================================================
    def checkImageAlignment(self, refImage, image):
        refImageGray = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)
        imageGray    = cv2.cvtColor(image,    cv2.COLOR_RGB2GRAY)

        orb_detector             = cv2.ORB_create(5000)
        keypoints1, descriptors1 = orb_detector.detectAndCompute(imageGray,    None)
        keypoints2, descriptors2 = orb_detector.detectAndCompute(refImageGray, None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(descriptors1, descriptors2), key=lambda x: x.distance)
        matches = matches[:int(len(matches) * 0.9)]
        no_of_matches = len(matches)

        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))
        for i, m in enumerate(matches):
            p1[i, :] = keypoints1[m.queryIdx].pt
            p2[i, :] = keypoints2[m.trainIdx].pt

        homography, _ = cv2.findHomography(p1, p2, cv2.RANSAC)

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _    = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        rotation_angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

        width, height = image.shape[1], image.shape[0]
        swapImage     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        warp_img      = cv2.warpPerspective(swapImage, homography, (width, height))

        h, w  = refImage.shape[:2]
        pts   = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst   = cv2.perspectiveTransform(pts, M)
        img2  = cv2.polylines(image, [np.int32(dst)], True, (0, 0, 255), 1, cv2.LINE_AA)
        poly_img = cv2.drawMatches(refImage, keypoints1, image, keypoints2, matches[:20], None, flags=2)

        return rotation_angle, poly_img, warp_img
