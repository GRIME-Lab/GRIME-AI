#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# This Python file uses the following encoding: utf-8

import numpy as np

#from PySide6.QtGui import QPixmap, QImage

class Image_Conversion:
    def __init__(self):
        pass


    def qimage_to_opencv(self, qimage):
        """
        Converts a QImage into an OpenCV-compatible format (numpy array).

        This method takes a QImage, converts it to the RGB888 format if not already in that format,
        and then converts it into a numpy array which can be used with OpenCV for further image
        processing tasks.

        Parameters
        ----------
        qimage : QImage
            The QImage to be converted.

        Returns
        -------
        arr : numpy.ndarray
            The converted image in a numpy array format, compatible with OpenCV.
        """

        # Convert QImage to an OpenCV image format.
        qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
        width = qimage.width()
        height = qimage.height()

        # Get the byte array of the image data
        byte_str = qimage.bits().tobytes()

        # Create a 1D numpy array from the byte string
        arr = np.frombuffer(byte_str, dtype=np.uint8)

        # Reshape the numpy array to the shape that OpenCV expects
        arr = arr.reshape((height, width, 3))

        return arr


    def qpixmap_to_opencv(self, qpixmap):
        """
        Converts a QPixmap to a NumPy array in OpenCV format.

        This method first converts the QPixmap to a QImage, and then uses the `qimage_to_opencv`
        method to convert the QImage to a NumPy array that is compatible with OpenCV.

        Parameters
        ----------
        qpixmap : QPixmap
            The QPixmap to be converted.

        Returns
        -------
        cv_img : numpy.ndarray
            The converted image in a numpy array format, compatible with OpenCV.
        """

        # Convert QPixmap to a NumPy array (OpenCV format).
        qimage = qpixmap.toImage()
        cv_img = self.qimage_to_opencv(qimage)

        return cv_img


    def opencv_to_qpixmap(self, cv_img):
        """
        Converts an OpenCV image (numpy array) back to a QPixmap.

        This method is useful for displaying OpenCV-processed images in a PyQt or PySide application,
        as it allows the numpy array to be converted into a format that can be easily displayed in the GUI.

        Parameters
        ----------
        cv_img : numpy.ndarray
            The OpenCV image to be converted.

        Returns
        -------
        qpixmap : QPixmap
            The QPixmap representation of the OpenCV image.
        """

        # Convert an OpenCV image back to QPixmap.
        #height, width, channels = cv_img.shape
        height, width = cv_img.shape
        #bytes_per_line = channels * width
        bytes_per_line = width
        #qimage = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB32)
        qimage = QImage(cv_img.data, cv_img.cols, cv_img.rows, cv_img.step, QImage.Format_RGB888);

        qpixmap = QPixmap(qimage)

        return qpixmap


    def opencv_to_qimage(self, opencv_img):
        """
        Converts an OpenCV image (numpy array) to a QImage.

        This method facilitates the use of OpenCV-processed images in PyQt or PySide applications by
        converting the numpy array into a QImage. This is particularly useful for displaying images
        in the GUI or further processing them using Qt's image handling capabilities.

        Parameters
        ----------
        opencv_img : numpy.ndarray
            The OpenCV image to be converted.

        Returns
        -------
        QImage
            The QImage representation of the OpenCV image.
        """

        # Convert an OpenCV image format to QImage.
        height, width, channels = opencv_img.shape
        bytes_per_line = channels * width

        return QImage(opencv_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)


