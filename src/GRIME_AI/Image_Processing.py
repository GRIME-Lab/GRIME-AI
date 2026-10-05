#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# This Python file uses the following encoding: utf-8
import cv2
import numpy as np
import re

#from PySide6.QtGui import QPixmap

from appcore.Image_Conversion import Image_Conversion

class Image_Processing:
    def __init__(self):

        self.myImageConversion = Image_Conversion()

        pass

    # *****************************************************************************************
    # IMAGE PROCESSING  ***  IMAGE PROCESSING  ***  IMAGE PROCESSING  ***  IMAGE PROCESSING ***
    # *****************************************************************************************
    def dilate_and_erode_qpixmap(self, qpixmap):
        """
        Dilates and then erodes an image contained in a QPixmap.

        This method first converts a QPixmap to an OpenCV-compatible format, applies dilation and erosion
        operations to process the image, and then converts the result back to a QPixmap for display or
        further use in a PyQt or PySide application.

        Parameters
        ----------
        qpixmap : QPixmap
            The QPixmap containing the original image to be processed with dilation and erosion.

        Returns
        -------
        result_qpixmap : QPixmap
            The QPixmap containing the image after dilation and erosion have been applied.
        """

        # Convert QPixmap to OpenCV image
        cv_img = self.myImageConversion.qpixmap_to_opencv(qpixmap)

        # Convert to grayscale for simplicity
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        myImg = gray_img

        # Erode the image
        kernel = np.ones((5, 5), np.uint8)
        myImg = cv2.erode(myImg, kernel, iterations=1)

        # Dilate the image
        #kernel = np.ones((11, 11), np.uint8)
        #myImg = cv2.dilate(myImg, kernel, iterations=1)


        # Convert back to QPixmap
        result_qpixmap = self.myImageConversion.opencv_to_qpixmap(cv2.cvtColor(myImg, cv2.COLOR_GRAY2BGR))

        return result_qpixmap


    def apply_sobel_filter(self, cv_img_gray, myksize):
        """
        Applies a Sobel filter to an image contained in a QPixmap.

        This method converts a QPixmap to an OpenCV-compatible format, applies a Sobel filter to
        detect edges in the image, and then converts the result back to a QPixmap for display or
        further use in a PyQt or PySide application.

        Parameters
        ----------
        qpixmap : QPixmap
            The QPixmap containing the original image to which the Sobel filter will be applied.

        Returns
        -------
        qpixmap_result : QPixmap
            The QPixmap containing the image after the Sobel filter has been applied.
        """

        # Apply Sobel filter
        sobelx = cv2.Sobel(cv_img_gray, cv2.CV_64F, 1, 0, ksize=myksize)
        sobely = cv2.Sobel(cv_img_gray, cv2.CV_64F, 0, 1, ksize=myksize)
        sobel = cv2.magnitude(sobelx, sobely)

        # Convert back to uint8
        sobel_uint8 = np.uint8(sobel)

        return sobel_uint8


    def apply_canny_filter(self, qpixmap):
        """
        Applies a Canny filter to an image contained in a QPixmap.

        This method converts a QPixmap to an OpenCV-compatible format, applies a Canny filter to
        detect edges in the image, and then converts the result back to a QPixmap for display or
        further use in a PyQt or PySide application.

        Parameters
        ----------
        qpixmap : QPixmap
            The QPixmap containing the original image to which the Canny filter will be applied.

        Returns
        -------
        qpixmap_result : QPixmap
            The QPixmap containing the image after the Canny filter has been applied.
        """

        # Convert QPixmap to QImage
        qimage = qpixmap.toImage()

        # Convert QImage to OpenCV format
        cv_img = self.myImageConversion.qimage_to_opencv(qimage)

        # Convert to grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lowThresh = 0.5*high_thresh

        kernel = np.ones((5,5), np.float32) / 25
        gray = cv2.filter2D(gray, -1, kernel)

        # Apply Canny filter
        edges = cv2.Canny(gray, 25, 150)

        # Convert the edges to a 3-channel image for QImage compatibility
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Convert the OpenCV image back to QImage, then to QPixmap
        qimage_result = self.myImageConversion.opencv_to_qimage(edges_colored)
        qpixmap_result = QPixmap.fromImage(qimage_result)

        return qpixmap_result


    def apply_prewitt_filter(self, qpixmap):
        """
        Applies a Prewitt filter to an image contained in a QPixmap.

        This method converts a QPixmap to an OpenCV-compatible format, applies Prewitt filters to
        detect horizontal and vertical edges in the image, and then converts the result back to a QPixmap
        for display or further use in a PyQt or PySide application.

        Parameters
        ----------
        qpixmap : QPixmap
            The QPixmap containing the original image to which the Prewitt filter will be applied.

        Returns
        -------
        qpixmap_result : QPixmap
            The QPixmap containing the image after the Prewitt filter has been applied.
        """

        # Convert QPixmap to QImage
        qimage = qpixmap.toImage()

        # Convert QImage to OpenCV format
        cv_img = self.myImageConversion.qimage_to_opencv(qimage)

        # Convert to grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        #JES - EXPERIMENTAL
        # Define Prewitt kernels
        #kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        #kernely = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)

        #kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        #kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=int)

        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)

        # Apply Prewitt kernels
        x = cv2.filter2D(gray, -1, kernelx)
        y = cv2.filter2D(gray, -1, kernely)

        # Combine the horizontal and vertical edges
        prewitt = cv2.addWeighted(x, 0.5, y, 0.5, 0)

        prewitt_colored = cv2.cvtColor(prewitt, cv2.COLOR_GRAY2BGR)

        # Convert the OpenCV image back to QImage, then to QPixmap
        qimage_result = self.myImageConversion.opencv_to_qimage(prewitt_colored)
        qpixmap_result = QPixmap.fromImage(qimage_result)

        return qpixmap_result


    def apply_laplacian_filter(self, qpixmap):
        """
        Applies a Laplacian filter to an image contained in a QPixmap.

        This method converts a QPixmap to an OpenCV-compatible format, applies a Laplacian filter to
        enhance the edges in the image, and then converts the result back to a QPixmap for display or
        further use in a PyQt or PySide application.

        Parameters
        ----------
        qpixmap : QPixmap
            The QPixmap containing the original image to which the Laplacian filter will be applied.

        Returns
        -------
        qpixmap_result : QPixmap
            The QPixmap containing the image after the Laplacian filter has been applied.
        """

        # Convert QPixmap to QImage
        qimage = qpixmap.toImage()

        # Convert QImage to OpenCV format
        cv_img = self.myImageConversion.qimage_to_opencv(qimage)

        # Convert to grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # Apply Laplacian filter
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Convert back to uint8
        laplacian = np.uint8(np.absolute(laplacian))

        laplacian_colored = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

        # Convert the OpenCV image back to QImage, then to QPixmap
        qimage_result = self.myImageConversion.opencv_to_qimage(laplacian_colored)
        qpixmap_result = QPixmap.fromImage(qimage_result)

        return qpixmap_result


    def detect_edges_in_roi(self, file_name, input_image):
        """
        Detects edges within a specified Region of Interest (ROI) in an image.

        Parameters:
        - input_image: The input image as a numpy array.
        - points: The points defining the ROI as a numpy array of shape (n, 1, 2).

        Returns:
        - An image with edges detected within the ROI.
        """

        pattern = r"_(.*?)_"

        # Iterate over the list of filenames
        match = re.search(pattern, file_name)
        match = match.group()[1:-1]
        print("Filename:", file_name)

        points = self.fetch_camera_ROI(match)

        # Extract the ROI from the input image
        cvImage = self.myImageConversion.qpixmap_to_opencv(input_image)
        roi = self.extractRegion(cvImage, points)


        # Convert the ROI to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_roi, 100, 200)

        # Create a color image to display the edges within the original ROI
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Use bitwise_or to merge the edge image with the original ROI to highlight edges
        result = cv2.bitwise_or(roi, edges_colored)

        qimage_result = self.myImageConversion.opencv_to_qimage(result)
        qpixmap_result = QPixmap.fromImage(qimage_result)

        return qpixmap_result

