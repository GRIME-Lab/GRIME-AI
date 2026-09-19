#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

'''
Gray-Level Co-Occurrence Matrix (GLCM): This statistical method analyzes the spatial relationship between pixels in an
                                        image. It computes texture features like contrast, entropy, homogeneity,
                                        and correlation.

Gabor Filters: These are used for texture segmentation and feature extraction. Gabor filters analyze the frequency and
               orientation of textures, making them effective for identifying patterns.

Local Binary Patterns (LBP): This technique is used for texture classification. It works by comparing each pixel with
                             its neighbors and encoding the results into a binary pattern.

Wavelet Transform: Wavelet-based methods decompose the image into different frequency components, allowing for
                   multi-resolution texture analysis.

Fourier Transform: This method analyzes the frequency domain of an image to identify repetitive patterns or textures.
'''

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from scipy.ndimage import convolve
from skimage.color import rgb2gray
import pywt

class GLCMTexture:
    """
    Computes texture features from the Gray-Level Co-occurrence Matrix (GLCM).

    The class computes statistics such as contrast, dissimilarity, homogeneity,
    energy, correlation, and ASM based on the co-occurrence of gray levels.

    Attributes:
        distances (list): Pixel pair distance offsets.
        angles (list): List of angles in radians.
        levels (int): Number of gray levels in the image.
        symmetric (bool): If True, the GLCM is symmetric.
        normed (bool): If True, normalize the GLCM.
    """

    def __init__(self, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.symmetric = symmetric
        self.normed = normed

    def compute_features(self, image: np.ndarray) -> dict:
        # Convert to grayscale and proper uint8 if necessary.
        if image.ndim == 3:
            image = rgb2gray(image)
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Compute the GLCM and then extract texture properties.
        glcm = graycomatrix(image, distances=self.distances, angles=self.angles,
                            levels=self.levels, symmetric=self.symmetric, normed=self.normed)
        features = {}
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for prop in properties:
            features[prop] = graycoprops(glcm, prop).mean()
        return features


class GaborTexture:
    """
    Computes texture features using Gabor filters.

    The class applies Gabor filters at different frequencies and orientations.
    For each filtered response, it computes the mean and variance (as a simple measure
    of texture energy) from which a feature vector is built.
    """

    def __init__(self, frequencies=[0.1, 0.3], thetas=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
        self.frequencies = frequencies
        self.thetas = thetas

    def compute_features(self, image: np.ndarray) -> dict:
        # Ensure image is grayscale.
        if image.ndim == 3:
            image = rgb2gray(image)
        features = {}
        for frequency in self.frequencies:
            for theta in self.thetas:
                filt_real, filt_imag = gabor(image, frequency=frequency, theta=theta)
                key = f'freq_{frequency}_theta_{theta:.2f}'
                features[f'{key}_mean'] = filt_real.mean()
                features[f'{key}_var'] = filt_real.var()
        return features


class LBPTexture:
    """
    Computes texture features using Local Binary Patterns (LBP).

    Applies the LBP operator to generate a code image and then computes a normalized
    histogram of the resulting patterns.

    Attributes:
        P (int): Number of circularly symmetric neighbor set points.
        R (float): Radius of circle.
        method (str): Method to determine the pattern ('default', 'ror', 'uniform', etc.).
    """

    def __init__(self, P=8, R=1, method='uniform'):
        self.P = P
        self.R = R
        self.method = method

    def compute_features(self, image: np.ndarray) -> np.ndarray:
        # Convert image to grayscale if necessary.
        if image.ndim == 3:
            image = rgb2gray(image)
        lbp = local_binary_pattern(image, self.P, self.R, method=self.method)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        return hist


class WaveletTexture:
    """
    Computes texture features using the Discrete Wavelet Transform (DWT).

    The image is decomposed into approximation and detail coefficients
    using wavelets. This class then computes statistics (mean and variance)
    from the detail coefficients at each level as the texture feature descriptor.

    Attributes:
        wavelet (str): The type of wavelet to use (e.g., 'db1').
        level (int): The number of decomposition levels.
    """

    def __init__(self, wavelet='db1', level=2):
        self.wavelet = wavelet
        self.level = level

    def compute_features(self, image: np.ndarray) -> dict:
        # Convert to grayscale if necessary.
        if image.ndim == 3:
            image = rgb2gray(image)
        image = image.astype(np.float32)
        coeffs = pywt.wavedec2(image, wavelet=self.wavelet, level=self.level)
        features = {}
        # Skip the approximation coefficients (coeffs[0]) and compute statistics on the detail coefficients.
        for i, detail_coeffs in enumerate(coeffs[1:], start=1):
            cH, cV, cD = detail_coeffs
            features[f'level_{i}_cH_mean'] = np.mean(cH)
            features[f'level_{i}_cH_var'] = np.var(cH)
            features[f'level_{i}_cV_mean'] = np.mean(cV)
            features[f'level_{i}_cV_var'] = np.var(cV)
            features[f'level_{i}_cD_mean'] = np.mean(cD)
            features[f'level_{i}_cD_var'] = np.var(cD)
        return features


class FourierTexture:
    """
    Computes texture features using the Fourier Transform.

    This class calculates the 2D Fourier transform of the image,
    shifts the zero-frequency component to the center, then computes a
    radial profile of the magnitude spectrum. Such a radial profile can be used
    as a feature vector characterizing the spatial frequency distribution.

    Attributes:
        num_bins (int): The number of radial bins to partition the spectrum.
    """

    def __init__(self, num_bins=32):
        self.num_bins = num_bins

    def compute_features(self, image: np.ndarray) -> np.ndarray:
        # Ensure image is grayscale.
        if image.ndim == 3:
            image = rgb2gray(image)
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # Calculate a radial profile.
        rows, cols = image.shape
        center_row, center_col = rows // 2, cols // 2
        y, x = np.indices((rows, cols))
        distances = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)
        max_distance = np.max(distances)
        bin_edges = np.linspace(0, max_distance, self.num_bins + 1)
        radial_profile = np.zeros(self.num_bins)

        for i in range(self.num_bins):
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
            if np.any(mask):
                radial_profile[i] = np.mean(magnitude_spectrum[mask])
            else:
                radial_profile[i] = 0
        return radial_profile



def laws_texture_features(image: np.ndarray) -> dict:
    """
    Computes Laws texture features using convolution masks and energy maps.
    Nine texture feature planes are extracted based on Laws' masks.

    Masks: The function uses predefined Laws 1D masks (L5, E5, S5, R5, W5) to compute nine energy planes.

    Convolution: Each combination of masks creates a 2D filter applied to the image. Squared energies from these
                 filtered responses are averaged to produce feature values.

    Args:
        image (np.ndarray): Input grayscale or RGB image.

    Returns:
        dict: A dictionary containing the energy for each texture plane.
    """
    # Convert to grayscale if the image is RGB
    if image.ndim == 3:
        image = rgb2gray(image)

    # Normalize the image
    image = (image - np.mean(image)) / np.std(image)

    # Define Laws 1D masks
    L5 = np.array([1, 4, 6, 4, 1])  # Level
    E5 = np.array([-1, -2, 0, 2, 1])  # Edge
    S5 = np.array([-1, 0, 2, 0, -1])  # Spot
    R5 = np.array([1, -4, 6, -4, 1])  # Ripple
    W5 = np.array([-1, 2, 0, -2, 1])  # Wave

    # Generate 2D masks from combinations of 1D masks
    masks = [np.outer(m1, m2) for m1 in [L5, E5, S5, R5, W5] for m2 in [L5, E5, S5, R5, W5]]

    # Compute energy maps for each mask
    energy_maps = [convolve(image, mask) ** 2 for mask in masks]

    # Extract energies (mean of each energy map)
    feature_planes = {f'Plane_{i + 1}': np.mean(energy_map) for i, energy_map in enumerate(energy_maps)}

    return feature_planes


def haralick_features(image: np.ndarray, distances=[1], angles=[0]) -> dict:
    """
    Computes Haralick texture features using the Gray-Level Co-occurrence Matrix (GLCM).

    GLCM: The Gray-Level Co-occurrence Matrix represents the spatial relationships between pixels.

    Statistics: Properties like contrast, homogeneity, and correlation are computed from the GLCM. These describe
                different aspects of texture, such as smoothness or randomness.

    Args:
        image (np.ndarray): Input grayscale or RGB image.
        distances (list): List of distances for GLCM computation.
        angles (list): List of angles in radians for GLCM computation.

    Returns:
        dict: A dictionary containing Haralick features like contrast, correlation, etc.
    """
    # Convert to grayscale if the image is RGB
    if image.ndim == 3:
        image = rgb2gray(image)
        image = (image * 255).astype(np.uint8)  # Convert to uint8 for GLCM computation

    # Compute GLCM
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Extract Haralick features
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = {prop: graycoprops(glcm, prop).mean() for prop in properties}

    return features
