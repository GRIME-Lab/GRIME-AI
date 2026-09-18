#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


class Diagnostics:
    def __init__(self):
        self.className = "Diagnostics"

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def plotHSVChannelsGray(hsv):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(hsv[:, :, 0], cmap='gray')
        ax[0].set_title('Hue')
        ax[1].imshow(hsv[:, :, 1], cmap='gray')
        ax[1].set_title('Saturation')
        ax[2].imshow(hsv[:, :, 2], cmap='gray')
        ax[2].set_title('Value')

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def plotHSVChannelsColor(hsv):

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(hsv[:, :, 0], cmap='hsv')
        ax[0].set_title('hue')
        ax[1].imshow(hsv[:, :, 1], cmap='hsv')
        ax[1].set_title('transparency')
        ax[2].imshow(hsv[:, :, 2], cmap='hsv')
        ax[2].set_title('value')
        fig.colorbar(plt.imshow(hsv[:, :, 0], cmap='hsv'))
        fig.tight_layout()


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def RGB3DPlot(rgb):
        r, g, b = cv2.split(rgb)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")

        pixel_colors = rgb.reshape((np.shape(rgb)[0] * np.shape(rgb)[1], 3))
        norm = colors.Normalize(vmin=-1., vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()

        axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("Red")
        axis.set_ylabel("Green")
        axis.set_zlabel("Blue")
        plt.show()
