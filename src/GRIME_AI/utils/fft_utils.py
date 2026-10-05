#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

"""
FFT utilities shared by image-quality triage (blur) and texture feature extraction.

All frequencies are in cycles per pixel: 0 is the DC term, NYQUIST (0.5) the
highest frequency along an axis. Tuning values default to the named constants
below; callers may override them.
"""

import numpy as np

# Highest frequency a sampled image can represent along an axis, in cycles per pixel.
NYQUIST = 0.5

# Default number of rings in fft_radial_profile. Callers may override it from settings.
DEFAULT_RADIAL_RINGS = 32

# Default frequency, in cycles per pixel, separating coarse from fine detail in
# fft_blur_score. Blur is judged relative to the same camera's other images, all
# scored with this cutoff, so the exact value is not critical.
DEFAULT_BLUR_CUTOFF = 0.05

# Automatic blur calibration (choose_blur_cutoff, flag_blurry).
MIN_IMAGES_FOR_AUTO_CUTOFF = 30   # fewer images than this: statistics are unreliable
BLUR_TAIL_PERCENTILE = 5.0        # low tail used to measure separation from the median
DEFAULT_BLUR_MAD_K = 3.0          # flag scores below median - k * MAD


# ======================================================================================================================
# Core
# ======================================================================================================================
def fft_spectrum(gray, window=True):
    """
    Magnitude spectrum of a grayscale image, normalized and with DC removed.

    Parameters
    ----------
    gray : 2-D array
        Grayscale image, any numeric dtype.
    window : bool
        Apply a 2-D Hann window before the FFT. This suppresses the false
        high-frequency energy caused by the image borders (the FFT treats the
        image as repeating, so opposite edges meet at a hard seam).

    Returns
    -------
    magnitude : 2-D float array
        |FFT| divided by the pixel count, shifted so frequency 0 is at the center.
        DC is 0.
    radius : 2-D float array, same shape
        Frequency of each coefficient in cycles per pixel.
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"fft_spectrum expects a 2-D grayscale image, got shape {g.shape}")
    rows, cols = g.shape

    g = g - g.mean()
    if window:
        g = g * np.outer(np.hanning(rows), np.hanning(cols))

    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(g))) / g.size

    fy = np.fft.fftshift(np.fft.fftfreq(rows))
    fx = np.fft.fftshift(np.fft.fftfreq(cols))
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    # Windowing gives the image a small nonzero mean again, so zero DC explicitly.
    magnitude[radius == 0.0] = 0.0

    return magnitude, radius


# ======================================================================================================================
# Blur (triage)
# ======================================================================================================================
def fft_blur_score(gray, cutoff=DEFAULT_BLUR_CUTOFF, window=True):
    """
    Fraction of the image's spectral power above `cutoff` cycles per pixel,
    within the Nyquist disc.

    Returns a value from 0 to 1. Sharp images keep more fine detail and score
    higher; blurred images score lower. The score does not depend on image size,
    brightness, or overall contrast.

    Parameters
    ----------
    cutoff : float
        Frequency, in cycles per pixel (0 to NYQUIST), separating coarse from
        fine detail.
    """
    magnitude, radius = fft_spectrum(gray, window=window)
    power = magnitude ** 2
    disc = radius <= NYQUIST     # corners beyond Nyquist cover only diagonal directions
    total = power[disc].sum()
    if total <= 0.0:
        return 0.0   # flat image: no detail at any frequency
    return float(power[disc & (radius > cutoff)].sum() / total)


# ======================================================================================================================
# Texture
# ======================================================================================================================
def fft_radial_profile(gray, num_bins=DEFAULT_RADIAL_RINGS, window=True):
    """
    Mean spectrum magnitude in `num_bins` equal-width rings from DC to Nyquist.

    Ring i covers frequencies (i / num_bins * NYQUIST, (i + 1) / num_bins * NYQUIST]
    cycles per pixel. DC is excluded, and frequencies beyond Nyquist (the
    corners of the spectrum) are excluded, so every ring covers all directions.

    Returns a 1-D float array of length num_bins, low frequency first. A ring
    with no coefficients (possible only for very small images) is 0.
    """
    magnitude, radius = fft_spectrum(gray, window=window)
    ring, inside = _ring_index(radius, num_bins)

    sums = np.bincount(ring[inside], weights=magnitude[inside], minlength=num_bins)
    counts = np.bincount(ring[inside], minlength=num_bins)
    profile = np.zeros(num_bins, dtype=np.float64)
    nonzero = counts > 0
    profile[nonzero] = sums[nonzero] / counts[nonzero]
    return profile


def fft_ring_power(gray, num_bins=DEFAULT_RADIAL_RINGS, window=True):
    """
    Total spectral power in each of `num_bins` rings, same rings as fft_radial_profile.

    Kept per image during a triage run so the blur cutoff can be chosen after all
    images are scored, without repeating the FFTs.
    """
    magnitude, radius = fft_spectrum(gray, window=window)
    ring, inside = _ring_index(radius, num_bins)
    return np.bincount(ring[inside], weights=(magnitude ** 2)[inside], minlength=num_bins)


def _ring_index(radius, num_bins):
    """Ring number of each coefficient and a mask of those inside (0, NYQUIST]."""
    edges = np.linspace(0.0, NYQUIST, num_bins + 1)
    ring = np.digitize(radius, edges, right=True) - 1   # right=True puts DC (radius 0) in ring -1
    inside = (ring >= 0) & (ring < num_bins)
    return ring, inside


# ======================================================================================================================
# Automatic blur calibration (one folder / camera)
# ======================================================================================================================
def blur_scores_at(ring_powers, cutoff_ring):
    """
    Blur scores for many images from their ring powers, with the cutoff at the
    inner edge of ring `cutoff_ring` (cutoff = cutoff_ring / num_bins * NYQUIST).

    ring_powers : 2-D array, one row per image (from fft_ring_power).
    Returns a 1-D array; an image with no power scores 0.
    """
    p = np.asarray(ring_powers, dtype=np.float64)
    total = p.sum(axis=1)
    high = p[:, cutoff_ring:].sum(axis=1)
    return np.divide(high, total, out=np.zeros_like(total), where=total > 0)


def _median_mad(values):
    med = float(np.median(values))
    return med, float(np.median(np.abs(values - med)))


def choose_blur_cutoff(ring_powers):
    """
    Choose the blur cutoff for one camera's images.

    Every inner ring boundary is tried. At each, the separation of the scores is
    (median - low-tail percentile) / MAD: large when most images form a tight group
    and some fall well below it. The boundary with the largest separation wins.

    Returns (cutoff, cutoff_ring, separations). Falls back to DEFAULT_BLUR_CUTOFF
    (cutoff_ring None) when there are fewer than MIN_IMAGES_FOR_AUTO_CUTOFF images
    or the scores have no spread at any boundary.
    """
    p = np.asarray(ring_powers, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < MIN_IMAGES_FOR_AUTO_CUTOFF:
        return DEFAULT_BLUR_CUTOFF, None, None

    num_bins = p.shape[1]
    separations = np.full(num_bins, np.nan)
    for i in range(1, num_bins):
        scores = blur_scores_at(p, i)
        med, mad = _median_mad(scores)
        if mad > 0.0:
            separations[i] = (med - np.percentile(scores, BLUR_TAIL_PERCENTILE)) / mad

    if np.all(np.isnan(separations)):
        return DEFAULT_BLUR_CUTOFF, None, separations
    best = int(np.nanargmax(separations))
    return best / num_bins * NYQUIST, best, separations


def calibrate_blur_bounds(good_ring_powers, blurry_ring_powers, k=DEFAULT_BLUR_MAD_K):
    """
    Choose the blur cutoff and score bounds from labeled Good and Blurry images.

    Every inner ring boundary is tried. At each, the separation of the two sets is
    (Good median - Blurry median) / (Good MAD + Blurry MAD); the boundary with the
    largest separation wins. At that boundary:

        good_bound   = Good median   - k * Good MAD     (sharp images score above this)
        blurry_bound = Blurry median + k * Blurry MAD   (blurry images score below this)

    Medians and MADs keep a single unusual or mislabeled image from moving the bounds.
    When blurry_bound >= good_bound the sets overlap and triage uses a fixed threshold
    halfway between the two medians.

    Returns a dict, or None when the labeled sets cannot be separated at all
    (Blurry scores are not lower than Good scores at any cutoff).
    """
    g = np.asarray(good_ring_powers, dtype=np.float64)
    b = np.asarray(blurry_ring_powers, dtype=np.float64)
    if g.ndim != 2 or b.ndim != 2 or len(g) == 0 or len(b) == 0 or g.shape[1] != b.shape[1]:
        return None

    num_bins = g.shape[1]
    best = None
    for i in range(1, num_bins):
        g_med, g_mad = _median_mad(blur_scores_at(g, i))
        b_med, b_mad = _median_mad(blur_scores_at(b, i))
        gap = g_med - b_med
        spread = g_mad + b_mad
        if spread > 0.0:
            separation = gap / spread
        else:
            separation = np.inf if gap > 0.0 else -np.inf
        key = (separation, gap)     # ties (e.g. no spread) go to the larger gap
        if best is None or key > best[0]:
            best = (key, i, g_med, g_mad, b_med, b_mad)

    (_, gap), ring, g_med, g_mad, b_med, b_mad = best
    if gap <= 0.0:
        return None

    good_bound = g_med - k * g_mad
    blurry_bound = b_med + k * b_mad
    return {
        "num_bins":      int(num_bins),
        "cutoff_ring":   int(ring),
        "cutoff":        float(ring / num_bins * NYQUIST),
        "good_median":   float(g_med),
        "blurry_median": float(b_med),
        "good_bound":    float(good_bound),
        "blurry_bound":  float(blurry_bound),
        "overlap":       bool(blurry_bound >= good_bound),
    }


def calibrated_blur_threshold(scores, calibration, k=DEFAULT_BLUR_MAD_K):
    """
    Threshold for one folder's scores (at the calibrated cutoff) using the bounds
    from calibrate_blur_bounds(). Returns (threshold, description).

    The folder's own threshold (median - k * MAD, as flag_blurry) adapts to the
    scene; it is kept between blurry_bound and good_bound so it can never flag the
    kind of image labeled Good or pass the kind labeled Blurry. With too few images
    for reliable folder statistics, the midpoint of the bounds is used. When the
    labeled sets overlapped, the fixed midpoint of the two medians is used.
    """
    if calibration["overlap"]:
        return ((calibration["good_median"] + calibration["blurry_median"]) / 2.0,
                "calibrated, fixed threshold (labeled Good and Blurry scores overlap)")

    low, high = calibration["blurry_bound"], calibration["good_bound"]
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) >= MIN_IMAGES_FOR_AUTO_CUTOFF:
        _, folder_threshold = flag_blurry(scores, k=k)
        if folder_threshold is not None:
            return (float(min(max(folder_threshold, low), high)),
                    "calibrated, folder threshold kept within calibrated bounds")
    return (low + high) / 2.0, "calibrated, midpoint of calibrated bounds (too few images for folder statistics)"


def flag_blurry(scores, k=DEFAULT_BLUR_MAD_K):
    """
    Flag scores below median - k * MAD. Returns (flags, threshold).
    With no spread (MAD 0) nothing is flagged and threshold is None.
    """
    scores = np.asarray(scores, dtype=np.float64)
    med, mad = _median_mad(scores)
    if mad <= 0.0:
        return np.zeros(scores.shape, dtype=bool), None
    threshold = med - k * mad
    return scores < threshold, threshold
