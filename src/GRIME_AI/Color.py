#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from urllib.request import urlopen

import cv2
import numpy as np

from GRIME_AI.App_Utils import App_Utils


class Color:
    """Class for processing color images using GRIME AI utilities."""

    # ==================================================================================================================
    # EXTRACT DOMINANT HSV COLORS
    # ==================================================================================================================
    @staticmethod
    def extractDominant_HSV(rgb, nNumClusters):
        """
        Extracts dominant HSV colors from the image using KMeans clustering.

        Args:
            rgb (numpy.ndarray): The color image in RGB color space.
            nNumClusters (int): The number of dominant colors to extract.

        Returns:
            tuple: A tuple containing the sorted histogram and sorted cluster centers.
        """

        if 0:
            rgb = cv2.blur(rgb, ksize=(5, 5))

        # Downsample for speed before clustering. A normal 2D image is
        # halved with cv2.resize. A polygon/freeform ROI arrives as a flat
        # (N,1,3) list of inside-mask pixels with no spatial layout, which
        # cv2.resize cannot handle (width 1 -> dsize empty). For that case
        # subsample every 4th pixel instead (same ~0.25 area reduction).
        if rgb.ndim == 3 and rgb.shape[0] > 1 and rgb.shape[1] > 1:
            resized_image = cv2.resize(rgb, None, fx=0.5, fy=0.5)
        else:
            flat = rgb.reshape(-1, 3)
            if flat.shape[0] > 4:
                flat = flat[::4]
            resized_image = flat.reshape(-1, 1, 3)

        hsv = cv2.cvtColor(resized_image, cv2.COLOR_RGB2HSV)

        _, hsvClusterCenters, hist = Color.KMeans(hsv, nNumClusters)

        if nNumClusters > 1 or len(hist) > nNumClusters:
            inds = hist.argsort()[::-1]
            sortedClusters = hsvClusterCenters[inds]
            sortedHist = hist[inds]

            sortedHist = sortedHist[0:nNumClusters]
            sortedClusters = sortedClusters[0:nNumClusters]
        else:
            sortedHist = hist
            sortedClusters = hsvClusterCenters

        return sortedHist, sortedClusters

    # ======================================================================================================================
    #
    # ======================================================================================================================
    @staticmethod
    def centroid_histogram(labels_):
        """
        Creates a histogram based on the number of pixels assigned to each cluster.

        Args:
            labels_ (numpy.ndarray): The array of labels from KMeans clustering.

        Returns:
            numpy.ndarray: The normalized histogram.
        """

        # grab the number of different clusters and create a histogram based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(labels_)) + 1)
        (hist, _) = np.histogram(labels_, bins=numLabels)

        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram
        return hist


    # ======================================================================================================================
    #
    # ======================================================================================================================
    @staticmethod
    def KMeans(img1, nClusters):
        """
        Applies KMeans clustering to the image for color quantization.

        Args:
            img1 (numpy.ndarray): The input image.
            nClusters (int): The number of clusters to form.

        Returns:
            tuple: A tuple containing the color bar, cluster centers, and histogram.
        """

        # --------------------------------------------------------------------------------
        # reshape the image to be a list of pixels
        # --------------------------------------------------------------------------------
        width = int(img1.shape[1])
        height = int(img1.shape[0])
        channels = int(img1.shape[2])

        dim = (width, height)

        pixel_values = img1.reshape((-1, channels))

        pixel_values = np.float32(pixel_values)

        # --------------------------------------------------------------------------------
        # COLOR CLUSTERING
        # --------------------------------------------------------------------------------
        # kmeans = KMeans(n_clusters=nClusters, n_jobs=-1)

        # define criteria, number of clusters(K) and apply kmeans()
        # We are going to cluster with k = 2, because the image will have just two colours ,a white background and the colour of the patch
        attempts = 10
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_PP_CENTERS
        _, labels, (centers) = cv2.kmeans(pixel_values, nClusters, None, criteria, 10, flags)

        # convert back to 8 bit values
        centers = np.uint8(centers)

        # flatten the labels array
        labels = labels.flatten()
        hist = Color.centroid_histogram(labels)

        # convert all pixels to the color of the centroids
        segmented_image = centers[labels.flatten()]

        # reshape back to the original image dimension
        segmented_image = segmented_image.reshape(img1.shape)

        bar = Color.plot_colors(hist, centers)

        return bar, centers, hist


    # ==================================================================================================================
    #
    # ==================================================================================================================
    @staticmethod
    def plot_colors(hist, centroids):
        """
        Plots a color bar representing the relative frequency of each color.

        Args:
            hist (numpy.ndarray): The histogram of color frequencies.
            centroids (numpy.ndarray): The color centroids.

        Returns:
            numpy.ndarray: The color bar chart.
        """

        # initialize the bar chart representing the relative frequency of each of the colors
        # bar = np.zeros((50, 300, 3), dtype="uint8")
        bar = np.zeros((50, 400, 3), dtype="uint8")

        startX = 0

        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * 100)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)

            startX = endX

        # return the bar chart
        return bar


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def __init__(self):
        """Initializes the Color class with default values."""

        self.className = "Color"


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def loadColorImage(self, filename):
        """
        Loads a color image from the specified file.

        Args:
            filename (str): The path to the image file.

        Returns:
            numpy.ndarray: The loaded image in RGB color space, or an empty list if loading fails.
        """

        img = cv2.imread(filename)

        try:
            origImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            origImg = []

        return origImg


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def segmentColors(self, rgb, hsv, roiList):
        """
        Segments colors in an image based on HSV color space and ROI color clusters.

        Args:
            rgb (numpy.ndarray): The color image in RGB color space.
            hsv (numpy.ndarray): The color image in HSV color space.
            roiList (list): A list of regions-of-interest and their color clusters.

        Returns:
            numpy.ndarray: The segmented color image in HSV color space.
        """

        # initialize return image to null
        rgb1 = []

        # Each ROI contains one or more colors extracted via kMeans clustering.
        # These color clusters are used as masks to extract colors that match the color clusters.
        i = 0

        for roi in roiList:
            hsv0 = roi.getHSVClusterCenters()

            for hsv1 in hsv0[0]:
                # refer to hue channel (in the colorbar)
                lower_mask = hsv[:, :, 0] > hsv1[0] - 8
                # refer to hue channel (in the colorbar)
                upper_mask = hsv[:, :, 0] < hsv1[0] + 15
                # refer to transparency channel (in the colorbar)
                saturation_mask = hsv[:, :, 1] > hsv1[1] - 3

                mask = upper_mask * lower_mask * saturation_mask
                red = rgb[:, :, 0] * mask
                green = rgb[:, :, 1] * mask
                blue = rgb[:, :, 2] * mask
                bags_masked = np.dstack((red, green, blue))
                if i > 0:
                    final_bags_masked = cv2.add(bags_masked, final_bags_masked)
                else:
                    final_bags_masked = bags_masked
                i = i + 1

        rgb1 = cv2.cvtColor(final_bags_masked, cv2.COLOR_RGB2BGR)

        return(rgb1)


    # ==================================================================================================================
    #
    # ==================================================================================================================
    @staticmethod
    def create_color_bar(hist, centroids, rows=50, cols=100, channels=3):
        # initialize the bar chart representing the relative frequency of each of the colors
        colorBar = np.zeros((rows, cols, channels), dtype="uint8")

        startX = 0

        # if only some of the clusters are used, we must rescale their size based upon their percentages to fill the bitmap
        multiplier = 1.0 / hist.sum()

        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * multiplier * 100)
            cv2.rectangle(colorBar, (int(startX), 0), (int(endX), 100), color.astype("uint8").tolist(), -1)

            startX = endX

        # CONVERT HSV TO RGB
        colorBar = cv2.cvtColor(colorBar, cv2.COLOR_HSV2RGB)

        # RETURN THE COLOR BAR
        return colorBar

