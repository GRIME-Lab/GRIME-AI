#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os
import datetime
from PIL import Image

from PyQt5.QtCore import QRect

from GRIME_AI.QProgressWheel import QProgressWheel


# ============================================================================
# ============================================================================
#                          class SegmentImagesTab
# ============================================================================
# ============================================================================
class CompositeSlices:
    def __init__(self, slice_rect: QRect, show_gui=True):
        """
        Args:
            slice_rect (QRect): The region of interest in original image coordinates.
        """

        self.slice_rect = slice_rect

        self.show_gui = show_gui

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def crop_side(self, image):
        """
        Crops the region defined by slice_rect from the given image.
        """
        return image.crop((
            self.slice_rect.left(),
            self.slice_rect.top(),
            self.slice_rect.right(),
            self.slice_rect.bottom()
        ))

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def create_composite_image(self, imageList, output_path):
        """
        Creates a composite image from a list of images by extracting a region of interest (ROI) from each image and
        sequentially pasting it into the composite image.

        Args:
            imageList (list): A list of image files. Each element in the list is an object with a `fullPathAndFilename` attribute.
            output_path (str): The path where the composite image will be saved.

        Returns:
            None

        Raises:
            IOError: If an error occurs while opening, processing, or saving the images.
        """

        if os.path.exists(output_path) == False:
            os.makedirs(output_path, exist_ok=True)

        imageCount = len(imageList)

        # DISPLAY PROGRESS BAR
        # ----------------------------------------------------------------------------------------------------
        if self.show_gui:
            progressBar = QProgressWheel()
            progressBar.setRange(0, imageCount + 1)
            progressBar.show()

        # CREATE A BASE FILENAME TO WHICH AN INDEX NUMBER WILL BE APPENDED AT THE TIME WHEN THE IMAGE IS SAVED
        # ----------------------------------------------------------------------------------------------------
        # Get the current time in ISO format
        current_time = datetime.datetime.now().replace(microsecond=0).isoformat()
        # Replace colons with underscores to avoid issues with filename
        outputFilename = os.path.join(output_path, ("CompositeImage - " + current_time.replace(':', '_')))

        # OPEN THE FIRST IMAGE TO GET ITS DIMENSIONS
        # ----------------------------------------------------------------------------------------------------
        first_image = Image.open(imageList[0].fullPathAndFilename)
        output_height = first_image.height

        maxWidthRequired = imageCount * self.slice_rect.width()

        numImages, imageWidths, filesPerImage = self.check_image_width(maxWidthRequired, self.slice_rect.width(), option=1)

        composite_image = Image.new('RGB', (imageWidths[0], output_height))

        slice_count = 0
        current_image_index = 0

        for i, image_file in enumerate(imageList):
            if self.show_gui:
                progressBar.setWindowTitle(image_file.fullPathAndFilename)
                progressBar.setValue(i)
                progressBar.repaint()

            # Open the current image
            with Image.open(image_file.fullPathAndFilename) as image:
                cropped_image = self.crop_side(image)

            # Paste the slice into the composite
            paste_x = (i % filesPerImage[current_image_index]) * cropped_image.width
            composite_image.paste(cropped_image, (paste_x, 0))
            slice_count += 1

            if slice_count == filesPerImage[current_image_index]:
                # Save the composite image
                compFilename = f"{outputFilename}-{current_image_index}.jpg"
                composite_image.save(compFilename)

                print("\nComposite Filename:", compFilename)
                print("Composite Image", current_image_index + 1, "of", numImages)
                print("Files Used in this Composite Image:", filesPerImage)
                print("Composite Image Width:", imageWidths)

                current_image_index += 1

                if current_image_index < len(imageWidths):
                    adjusted_max_width = imageWidths[current_image_index]
                    composite_image = Image.new('RGB', (adjusted_max_width, image.height))
                    slice_count = 0

        # clean-up before exiting function
        # 1. close and delete the progress bar
        # 2. close the EXIF log file, if opened
        if self.show_gui:
            progressBar.close()
            del progressBar

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def check_image_width(self, width, strip_width, option=1):
        MAX_JPG_WIDTH = 65500
        MAX_WIDTH = MAX_JPG_WIDTH - strip_width

        if width > MAX_WIDTH:
            num_images = width // MAX_WIDTH
            remaining_width = width % MAX_WIDTH
            if remaining_width > 0:
                num_images += 1

            if option == 1:
                if remaining_width > 0:
                    image_widths = [MAX_WIDTH] * (num_images - 1) + [remaining_width]
                else:
                    image_widths = [MAX_WIDTH] * num_images

                # Adjust widths to be multiples of strip_width
                image_widths = [((w // strip_width) * strip_width) for w in image_widths]
                if remaining_width % strip_width != 0:
                    image_widths[-1] += strip_width - (image_widths[-1] % strip_width)
            elif option == 2:
                equal_width = width // num_images
                remainders = width % num_images
                image_widths = [equal_width] * num_images

                for i in range(remainders):
                    image_widths[i] += 1

                # Adjust widths to be multiples of strip_width
                image_widths = [((w // strip_width) * strip_width) for w in image_widths]
                for i in range(remainders):
                    if image_widths[i] % strip_width != 0:
                        image_widths[i] += strip_width - (image_widths[i] % strip_width)

            num_strips_per_image = [w // strip_width for w in image_widths]

            return num_images, image_widths, num_strips_per_image  # Return number of images, image widths, and number of strips per image
        else:
            return 1, [width], [width // strip_width]  # Return 1 image with the given width and number of strips if within limit

