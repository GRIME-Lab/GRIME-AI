#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import cv2
import numpy as np

from PyQt5.QtGui import QPixmap, QImage

# WAS: `import GRIME_AI.sobelData`, which binds only the name `GRIME_AI`. The module then
# referenced a bare `sobelData`, raising NameError on every Sobel X / Sobel Y request.
from GRIME_AI.sobelData import sobelData

from GRIME_AI.constants import edgeMethodsClass, featureMethodsClass

from GRIME_AI.GRIME_AI_Image_Processing import GRIME_AI_Image_Processing
from GRIME_AI.GRIME_AI_Image_Conversion import GRIME_AI_Image_Conversion


# ======================================================================================================================
# DISPLAY HELPERS
#
# Every QPixmap in this module used to be built with QImage(ndarray.data, ...), which does
# NOT copy the buffer -- the QImage aliases the NumPy memory. If the array is garbage
# collected (or explicitly deleted, as processORB did) the QImage points at freed memory.
# These helpers copy defensively and compute the stride instead of assuming it.
# ======================================================================================================================
def _to_u8(arr):
    """Convert any float / signed gradient image to a displayable 8-bit image.

    Gradient operators produce SIGNED output, and there are two distinct wrong ways to
    get that onto the screen:

      arr.astype(np.uint8)          WRAPS modulo 256. A true gradient of 800 displays as
                                    32 -- a strong edge renders as near-black. This is
                                    what the commented-out line in processSobel did.
      cv2.Sobel(..., cv2.CV_8U)     SATURATES, so every negative response becomes 0 and
                                    the dark-to-light half of every edge disappears.

    convertScaleAbs takes the absolute value first, so both polarities survive.
    """
    if arr is None:
        return None

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    # Preserve magnitude of negative gradients rather than clipping them.
    return np.ascontiguousarray(cv2.convertScaleAbs(arr))


def _gray_to_pixmap(arr):
    """Build a QPixmap from a single-channel image, copying the buffer."""
    img = _to_u8(arr)
    if img is None or img.size == 0:
        return QPixmap()

    if img.ndim != 2:
        return _rgb_to_pixmap(img)

    h, w = img.shape
    q_img = QImage(img.data, w, h, img.strides[0], QImage.Format_Grayscale8)
    return QPixmap.fromImage(q_img.copy())      # .copy() detaches from the NumPy buffer


def _rgb_to_pixmap(arr):
    """Build a QPixmap from a 3-channel image in R, G, B order, copying the buffer."""
    img = np.ascontiguousarray(arr)
    if img is None or img.size == 0:
        return QPixmap()

    if img.ndim == 2:
        return _gray_to_pixmap(img)

    h, w = img.shape[:2]
    # Format_RGB888, not Format_BGR888. Everything upstream of this module is RGB
    # (processLocalImage builds its QImage with Format_RGB888). The old code declared
    # Format_BGR888 here, which swapped the red and blue channels on screen.
    q_img = QImage(img.data, w, h, img.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(q_img.copy())


class GRIME_AI_ProcessImage:
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.className = "GRIME_AI_ProcessImage"

    # ------------------------------------------------------------------------------------------------------------------
    # CANNY
    # ------------------------------------------------------------------------------------------------------------------
    def processCanny(self, img1, gray, edgeMethodSettings, overlay_contours=False):
        """Canny edge detection.

        Set overlay_contours True to draw contours over the source image instead of
        returning the edge map itself.
        """
        highThreshold = edgeMethodSettings.getCannyThresholdHigh()
        lowThreshold  = edgeMethodSettings.getCannyThresholdLow()

        # Aperture must be 3, 5 or 7. Falls back to 3 if edgeMethodsClass does not yet
        # carry the field (add `canny_kernel` to constants.py to make the spinbox live).
        kernelSize = getattr(edgeMethodSettings, "canny_kernel", 3)
        if kernelSize not in (3, 5, 7):
            kernelSize = 3

        # Enforce the hysteresis invariant explicitly rather than relying on OpenCV's
        # internal swap, so the values mean what the dialog labels say they mean.
        if lowThreshold > highThreshold:
            lowThreshold, highThreshold = highThreshold, lowThreshold

        # NOTE: main.py already blurred `gray`. A second blur here would compound the
        # smoothing and further erode the gradients, so it has been removed.
        #
        # WAS: cv2.Canny(img_blur, highThreshold, lowThreshold, kernelSize)
        #   - argument order was (high, low); the signature is (image, low, high)
        #   - the 4th POSITIONAL parameter of cv2.Canny is `edges` (the output array),
        #     NOT apertureSize. Passing kernelSize there was silently ignored, so the
        #     aperture was permanently stuck at the default of 3.
        edges = cv2.Canny(gray,
                          lowThreshold,
                          highThreshold,
                          apertureSize=kernelSize,
                          L2gradient=True)

        # REMOVED: two full-image cv2.threshold() calls computing Otsu and Triangle
        # thresholds. Both results were fed through getThresholdRange() and then never
        # used -- the lines that consumed them are commented out. They also passed
        # lowThreshold/highThreshold as thresh/maxval, which THRESH_OTSU ignores entirely.

        if not overlay_contours:
            return _gray_to_pixmap(edges)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw on a COPY. cv2.drawContours mutates in place, and img1 may be a view onto
        # the caller's QImage buffer -- the old code was scribbling on the source image.
        #
        # RETR_EXTERNAL rather than RETR_TREE: a Canny edge map is a set of thin lines, so
        # RETR_TREE returns an inner AND an outer contour for every line, doubling them.
        canvas = np.ascontiguousarray(img1.copy())
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

        return _rgb_to_pixmap(canvas)

    # ------------------------------------------------------------------------------------------------------------------
    def getThresholdRange(self, threshold, sigma=0.33):
        return (1 - sigma) * threshold, (1 + sigma) * threshold

    # ------------------------------------------------------------------------------------------------------------------
    # SOBEL
    # ------------------------------------------------------------------------------------------------------------------
    def processSobel(self, gray, sobelKernelSize, method):

        if sobelKernelSize not in (1, 3, 5, 7):
            sobelKernelSize = 3

        mySobel = sobelData()

        # CV_64F is correct here -- gradients are signed. The failure was downstream: the
        # float64 result was handed straight to QImage as Format_Grayscale8, so Qt read an
        # 8-bytes-per-pixel buffer as 1 byte per pixel. That displayed roughly one eighth
        # of the image as noise. _to_u8() now does the magnitude conversion properly.
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobelKernelSize)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobelKernelSize)

        mySobel.setSobelX(gx)
        mySobel.setSobelY(gy)

        if method == edgeMethodsClass.SOBEL_X:
            return _gray_to_pixmap(gx)

        elif method == edgeMethodsClass.SOBEL_Y:
            return _gray_to_pixmap(gy)

        elif method == edgeMethodsClass.SOBEL_XY:
            # True gradient magnitude, sqrt(gx^2 + gy^2). Not cv2.Sobel(..., 1, 1, ...),
            # which is the mixed second derivative and is not an edge strength measure.
            mag = cv2.magnitude(gx, gy)
            mySobel.setSobelXY(mag)
            return _gray_to_pixmap(mag)

        # Previously `pix` was simply unbound here, raising UnboundLocalError.
        return QPixmap()

    # ------------------------------------------------------------------------------------------------------------------
    def getGradientMagnitude(self, im):
        ddepth = cv2.CV_32F
        dx = cv2.Sobel(im, ddepth, 1, 0)
        dy = cv2.Sobel(im, ddepth, 0, 1)
        return cv2.magnitude(dx, dy)

    # ------------------------------------------------------------------------------------------------------------------
    # LAPLACIAN
    # ------------------------------------------------------------------------------------------------------------------
    def processLaplacian(self, gray, use_log=True, sigma=1.0):
        """Laplacian / Laplacian-of-Gaussian edge detection.

        SIGNATURE CHANGE: this now takes the GRAYSCALE image. It previously received the
        3-channel color image while every sibling method received grayscale.

        Removed from the old implementation:
          - RGB -> YCrCb -> equalizeHist(Y) -> `cvtColor(ycrcb, COLOR_RGB2GRAY)`, which
            reinterpreted YCrCb data as if it were RGB. If you want luma, take channel 0.
          - equalizeHist() applied to the LoG response, which amplifies noise in flat
            regions (sky, open water) far more than it helps real edges.
          - cv2.dilate(gray,(5,5),gray) followed by cv2.erode(gray,(5,5),gray). The (5,5)
            tuple is not a structuring element -- NumPy turns it into a shape-(2,) array --
            and the pair amounts to a morphological closing that thickens and displaces
            every edge. Both also wrote back into `gray` through the dst argument.
        """
        if gray is None or gray.size == 0:
            return QPixmap()

        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

        if use_log:
            return _gray_to_pixmap(laplace_of_gaussian(gray, sigma=sigma))

        # CV_64F, then magnitude -- the Laplacian is signed like any other derivative.
        return _gray_to_pixmap(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))

    # ------------------------------------------------------------------------------------------------------------------
    # SIFT
    # ------------------------------------------------------------------------------------------------------------------
    def processSIFT(self, img1, gray):
        # The old version built `pix` and then fell off the end of the function, returning
        # None. main.py's `if not pix == []` passed, then pix.scaled(...) raised
        # AttributeError -- selecting SIFT crashed the application every time.
        edges = calcSIFT(img1, gray)
        return _rgb_to_pixmap(edges)

    # ------------------------------------------------------------------------------------------------------------------
    # ORB
    # ------------------------------------------------------------------------------------------------------------------
    def processORB(self, img1, gray, featureMethodSettings):
        # REMOVED: the YCrCb / equalizeHist / laplace_of_gaussian / dilate / erode block.
        # Its entire result was discarded on the very next line, which recomputed `gray`
        # from ycrcb_img before calling calcOrb. It was pure wasted work per frame.
        #
        # ORB also wants ordinary image structure to find corners. Running it on a
        # Laplacian-of-Gaussian zero-crossing map would give meaningless keypoints anyway.
        nMaxFeatures = getattr(featureMethodSettings, "orbMaxFeatures", 500)

        edges = calcOrb(img1, gray, nMaxFeatures)

        # The old code did `del gray; del edges` BEFORE constructing the QPixmap, while the
        # QImage still aliased the `edges` buffer -- a use-after-free. _rgb_to_pixmap()
        # copies, so the arrays can be released safely by normal refcounting.
        return _rgb_to_pixmap(edges)


# ======================================================================================================================
#
# ======================================================================================================================
def laplace_of_gaussian(gray_img, sigma=1., kappa=0.75, pad=False):
    """
    Applies Laplacian of Gaussians to grayscale image.

    :param gray_img: image to apply LoG to
    :param sigma:    Gauss sigma of Gaussian applied to image, <= 0. for none
    :param kappa:    difference threshold as factor to mean of image values, <= 0 for none
    :param pad:      flag to pad output w/ zero border, keeping input image size
    """
    assert len(gray_img.shape) == 2

    img = cv2.GaussianBlur(gray_img, (0, 0), sigma) if 0. < sigma else gray_img
    img = cv2.Laplacian(img, cv2.CV_64F)
    rows, cols = img.shape[:2]

    # min/max of 3x3-neighbourhoods
    min_map = np.minimum.reduce(list(img[r:rows - 2 + r, c:cols - 2 + c]
                                     for r in range(3) for c in range(3)))
    max_map = np.maximum.reduce(list(img[r:rows - 2 + r, c:cols - 2 + c]
                                     for r in range(3) for c in range(3)))

    # bool matrix for image value positive (w/out border pixels)
    pos_img = 0 < img[1:rows - 1, 1:cols - 1]

    # bool matrix for min < 0 and 0 < image pixel
    neg_min = min_map < 0
    # WAS: neg_min[1 - pos_img] = 0
    # `1 - pos_img` on a bool array yields an INT array of 0s and 1s. NumPy then treats it
    # as integer row indexing, not as a logical-NOT mask -- so this zeroed rows 0 and 1 of
    # the array instead of the intended pixels, silently and without error. Same below.
    neg_min[~pos_img] = 0

    # bool matrix for 0 < max and image pixel < 0
    pos_max = 0 < max_map
    pos_max[pos_img] = 0

    # sign change at pixel?
    zero_cross = neg_min | pos_max

    # values: max - min, scaled to 0--255; set to 0 for no sign change
    value_scale = 255. / max(1., img.max() - img.min())
    values = value_scale * (max_map - min_map)
    values[~zero_cross] = 0.

    # optional thresholding
    if 0. <= kappa:
        thresh = float(np.absolute(img).mean()) * kappa
        values[values < thresh] = 0.

    log_img = np.clip(values, 0, 255).astype(np.uint8)

    if pad:
        log_img = np.pad(log_img, pad_width=1, mode='constant', constant_values=0)

    return log_img


# ======================================================================================================================
# THIS FUNCTION WILL USE THE SIFT FEATURE DETECTION ALGORITHM TO FIND FEATURES IN THE IMAGE THAT IS PASSED TO THIS FUNCTION.
# ======================================================================================================================
def calcSIFT(image, gray):
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    # Draw onto a COLOR copy of the source. The old code passed `img1` -- which it had
    # just rebound to the blurred GRAYSCALE image -- as drawKeypoints' outImg, then
    # displayed the result as Format_RGB888. That read a 1-byte-per-pixel buffer as 3.
    canvas = np.ascontiguousarray(image.copy())
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    return cv2.drawKeypoints(canvas, kp, None,
                             color=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# ======================================================================================================================
# THIS FUNCTION WILL USE THE ORB FEATURE DETECTION ALGORITHM TO FIND FEATURES IN THE IMAGE THAT IS
# PASSED TO THIS FUNCTION.
# ======================================================================================================================
def calcOrb(image, gray, nMaxFeatures):

    if gray is None:
        gray = image

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    # ORB's own default is 500. The dialog previously defaulted this to 100000, which
    # carpets the image with keypoints and makes the detector crawl.
    nMaxFeatures = max(1, int(nMaxFeatures))

    # edgeThreshold must be >= patchSize or the border exclusion zone silently discards
    # keypoints the patch extractor would otherwise have accepted.
    orb = cv2.ORB_create(nfeatures=nMaxFeatures,
                         nlevels=8,
                         edgeThreshold=31,
                         patchSize=31)

    kp = orb.detect(gray, None)
    kp, des = orb.compute(gray, kp)

    canvas = np.ascontiguousarray(image.copy())
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    return cv2.drawKeypoints(canvas, kp, None, color=(0, 255, 0), flags=0)


# ======================================================================================================================
# ALIGN IMAGES USING A REFERENCE IMAGE
# ======================================================================================================================
def imageAlignment(referenceImageFilename, imageFilename):

    img1_color = cv2.imread(imageFilename)              # Image to be aligned.
    img2_color = cv2.imread(referenceImageFilename)     # Reference image.

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images using Hamming distance.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color, homography, (width, height))

    cv2.imwrite('output.jpg', transformed_img)

    return transformed_img
