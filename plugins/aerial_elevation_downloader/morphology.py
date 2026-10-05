#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# morphology.py
#
# Reach-scale measurements from an elevation model, after three studies:
#
#   Pattern (Schuurman et al. 2013, JGR Earth Surface): the distribution of
#       detrended bed level, bar height as the top 5% minus the bottom 5% of it,
#       the total braiding index from channel counts in cross sections, the
#       dominant bar length from a wavelet density along the flow, and bar
#       aspect and perimeter-to-area ratios.
#
#   Superimposed bedforms (Reesink and Bridge 2011, JSR): for each bar, the
#       height of the bedforms riding on it against the bar's own height.
#       Above 25%, superimposed bedforms lower the host's lee slope and build
#       inclined sets instead of angle-of-repose cross strata.
#
#   Inundation (Korus et al. 2020, Hydrological Processes): how often each
#       cell is under water over a period, from the daily gage height, the gage
#       datum and the reach slope. Their finding was that infrequent inundation
#       of an otherwise immobile bar reduced its hydraulic conductivity, so the
#       cells that are only sometimes wet are the ones that map it.
#
# The analyses are plain functions; the tab at the bottom drives them.

import math
import datetime

import numpy as np
import cv2
import requests

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate, QObject, QTimer, QEvent
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
                             QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QDateEdit,
                             QComboBox, QCheckBox, QPlainTextEdit, QScrollArea, QFormLayout,
                             QSplitter, QMessageBox, QToolTip)

FEET_TO_METRES = 0.3048
VOID_BELOW = -9000.0

# Schuurman et al. 2013 define bar height from these percentiles of detrended
# bed level, and use the cross-section mean as the bar/channel threshold.
BAR_HEIGHT_LOW_PERCENTILE = 5.0
BAR_HEIGHT_HIGH_PERCENTILE = 95.0

# Reesink and Bridge 2011: superimposed bedforms taller than this share of the
# host bar's height change how the host is built.
SUPERIMPOSED_RATIO_THRESHOLD = 0.25

# A bar picked from the report is drawn this much wider, in this colour.
HIGHLIGHT_WIDTH = 2
NEVER_WET_COLOR = (40, 110, 160)       # tan, BGR
DEM_LAYERS = ["Hillshade", "Elevation (grey)"]
COLOR_BASE_STEPS = 1000                # slider resolution for the colour base
# The scale sliders run logarithmically between these limits, so each end of a
# range several decades wide gets the same fine control.
BEDFORM_CUTOFF_RANGE_M = (1.0, 1000.0)
MIN_BAR_AREA_RANGE_M2 = (1.0, 1.0e6)
SCALE_SLIDER_STEPS = 1000
# Slider values are rounded to this many significant figures, so they read as
# round numbers (25 m, 500 m²) rather than the raw log steps (25.03, 501).
SCALE_SLIDER_SIGNIFICANT_FIGURES = 2
# Moving a scale slider reruns the analysis on screen once the slider has been
# still this long, so a drag does not queue a run for every step it passes.
LIVE_RERUN_DELAY_MS = 250
PICK_ZOOM_FILL = 0.35                  # a picked item fills this share of the view
MOUND_COLOR = (0, 165, 255)            # orange, BGR
POCKET_COLOR = (255, 255, 0)           # cyan
BAND_COLOR = (150, 150, 150)           # the plus and minus two sigma band
HIGHLIGHT_COLOR = (0, 255, 255)        # yellow in BGR

DEFAULTS = {
    "bedform_cutoff_m": 25.0,        # features shorter than this are bedforms, not bars
    "min_bar_area_m2": 500.0,        # smaller patches above the threshold are not bars
    "wavelet_min_m": 20.0,           # shortest bar length searched for
    "wavelet_max_m": 2000.0,         # longest bar length searched for
    "wavelet_scales": 40,
}

# The background for each analysis appears after the pointer rests on its
# heading this long, rather than cluttering the heading itself.
HOVER_HELP_DELAY_MS = 3000

SUMMARIES = {
    "Pattern": (
        "Schuurman, Marra and Kleinhans (2013), Journal of Geophysical Research: "
        "Earth Surface.\n\n"
        "Bar height is the top 5% minus the bottom 5% of the detrended bed level. The "
        "braiding index counts channels per cross section, taking each section's mean "
        "as the bar/channel threshold. The dominant bar length is the peak of a "
        "wavelet density of bed wavelengths along the flow. Bar shape is given by "
        "width-to-length and perimeter-to-area ratios."),
    "Bedforms": (
        "Reesink and Bridge (2011), Journal of Sedimentary Research.\n\n"
        "Bedforms riding on a bar that are taller than 25% of the bar's own height "
        "lower its lee slope, so the bar builds inclined sets instead of "
        "angle-of-repose cross strata. Bars past that ratio are outlined in red."),
    "Mounds and pockets": (
        "Elongated and ramped features, found from their margins: the steep edges "
        "traced as lines in the slope, and the areas they enclose. Short breaks are "
        "bridged, and the open up-ramp end of a sloped feature is closed across.\n\n"
        "A feature is kept only when its margin is significantly steeper than the "
        "ground on both sides, by a rank test whose sample sizes allow for the DEM's "
        "spatial correlation, corrected for the number of candidates tested.\n\n"
        "Each outline is then refined by an extended Kalman filter walking along the "
        "margin by arc length, which weighs each measurement by how sharp the margin "
        "is and carries the outline across faint or missing stretches. The band is two "
        "standard deviations either side, and the centre's error ellipse comes from the "
        "centroid's propagated uncertainty, reported along and across the feature's "
        "long axis. Lee and ramp are the directions the "
        "steepest and gentlest sides face."),
    "Local rank": (
        "Each cell's height is replaced by its rank in an elliptical window centred on "
        "it: the share of the window's cells lower than it. Cells within the DEM's "
        "noise of it count as ties, half lower and half higher, so flat ground and "
        "noise read 50%. A small low bar then ranks as high in its window as a large "
        "tall bar does in its own.\n\n"
        "The window is longer along the flow than across it, turned at each cell to "
        "the direction the topography runs there, or to the reach's downhill "
        "direction where it has none. Several window lengths are run; a bar that is "
        "one high at a long window and several at the next shorter one is compound.\n\n"
        "Bar cells are those ranking above the chosen share. Each bar's height above "
        "the ground around it, and the test of whether it stands above that ground, "
        "come from the DEM, not the rank."),
    "Inundation": (
        "Korus, Fraundorfer, Gilmore and Karnik (2020), Hydrological Processes.\n\n"
        "On the Loup River, infrequent inundation of an otherwise immobile bar "
        "reduced its hydraulic conductivity by about 20% through pore clogging. The "
        "cells that are only sometimes wet show where bar tops see those flows."),
}

NWIS_DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
NCAT_URL = "https://geodesy.noaa.gov/api/ncat/llh"
USER_AGENT = "GRIME-AI morphology"
REQUEST_TIMEOUT = 60


# ======================================================================================================================
# Shared preparation
# ======================================================================================================================
def clean(elevation):
    """Elevation as float64 with voids as NaN."""
    values = np.asarray(elevation, np.float64).copy()
    values[values < VOID_BELOW] = np.nan
    return values


def fit_plane(values):
    """(offset, per column, per row) of the least-squares plane through the valid cells."""
    good = np.isfinite(values)
    rows, columns = np.nonzero(good)
    design = np.column_stack([np.ones(len(columns)), columns, rows])
    coefficients, *_ = np.linalg.lstsq(design, values[good], rcond=None)
    return coefficients


def detrend(values):
    """Values with the fitted plane removed, and the plane itself."""
    offset, per_column, per_row = fit_plane(values)
    rows, columns = np.indices(values.shape)
    plane = offset + per_column * columns + per_row * rows
    return values - plane, (offset, per_column, per_row)


def flow_aligned(values, plane):
    """
    The grid rotated so the downhill direction of the plane runs left to right.
    Cross sections are then columns and the flow runs along rows, which is how
    the braiding index and bar length are measured.
    """
    _, per_column, per_row = plane
    # Downhill in image coordinates, and the rotation that lines it up with +x.
    angle = math.degrees(math.atan2(-per_row, -per_column))
    height, width = values.shape
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    matrix[0, 2] += new_width / 2.0 - centre[0]
    matrix[1, 2] += new_height / 2.0 - centre[1]
    filled = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
    mask = np.isfinite(values).astype(np.uint8)
    rotated = cv2.warpAffine(filled, matrix, (new_width, new_height),
                             flags=cv2.INTER_LINEAR, borderValue=0)
    rotated_mask = cv2.warpAffine(mask, matrix, (new_width, new_height),
                                  flags=cv2.INTER_NEAREST, borderValue=0)
    rotated = rotated.astype(np.float64)
    rotated[rotated_mask == 0] = np.nan
    return rotated, angle


# ======================================================================================================================
# Pattern (Schuurman et al. 2013)
# ======================================================================================================================
def pattern_metrics(elevation, cell_size, settings=None):
    """
    Bed-level distribution, bar height, braiding index, dominant bar length and
    bar shapes for one elevation patch. cell_size is ground metres per cell.
    """
    settings = {**DEFAULTS, **(settings or {})}
    values = clean(elevation)
    if np.isfinite(values).sum() < 100:
        return {"error": "Too few valid cells in this region."}

    detrended, plane = detrend(values)
    finite = detrended[np.isfinite(detrended)]
    low = float(np.percentile(finite, BAR_HEIGHT_LOW_PERCENTILE))
    high = float(np.percentile(finite, BAR_HEIGHT_HIGH_PERCENTILE))

    aligned, angle = flow_aligned(detrended, plane)
    braiding, section_counts, bars_aligned = _braiding_index(aligned, cell_size, settings)
    length, lengths, density = _dominant_bar_length(aligned, cell_size, settings)
    bars = _bar_shapes(detrended, cell_size, settings)

    counts, edges = np.histogram(finite, bins=60)
    return {
        "bar_height": high - low,
        "percentile_low": low,
        "percentile_high": high,
        "histogram": (counts, edges),
        "braiding_index": braiding,
        "braiding_per_section": section_counts,
        "dominant_bar_length": length,
        "wavelet_lengths": lengths,
        "wavelet_density": density,
        "bars": bars,
        "flow_rotation_degrees": angle,
        "detrended": detrended,
        "cell_size": cell_size,
    }


def _braiding_index(aligned, cell_size, settings):
    """
    Channels per cross section, averaged along the reach. Each section's own
    mean is the threshold, which removes the longitudinal trend.

    The surface is smoothed at the bedform cutoff first, and a channel has to be
    at least that wide: otherwise ripples and noise flicker across the mean and
    every flicker counts as a channel.
    """
    cutoff_cells = max(settings["bedform_cutoff_m"] / cell_size, 1.0)
    valid_all = np.isfinite(aligned)
    filled = np.where(valid_all, aligned, 0.0).astype(np.float32)
    weights = cv2.GaussianBlur(valid_all.astype(np.float32), (0, 0), cutoff_cells / 3.0)
    smooth = cv2.GaussianBlur(filled, (0, 0), cutoff_cells / 3.0) / np.maximum(weights, 1e-6)
    smooth = np.where(valid_all, smooth, np.nan)
    minimum_run = int(max(round(cutoff_cells), 1))

    valid = np.isfinite(smooth)
    usable = valid.sum(axis=0) >= 3 * minimum_run
    means = np.nanmean(np.where(valid, smooth, np.nan), axis=0)
    below = valid & (smooth < means[None, :])
    channels = np.where(usable[None, :], below, False)
    # A channel is a run of below-mean cells at least one cutoff wide. Runs are
    # found from where each column steps into and out of the below-mean state.
    padded = np.vstack([np.zeros((1, below.shape[1]), bool), below,
                        np.zeros((1, below.shape[1]), bool)]).astype(np.int8)
    steps = np.diff(padded, axis=0)
    counts = []
    for column in np.flatnonzero(usable):
        starts = np.flatnonzero(steps[:, column] == 1)
        ends = np.flatnonzero(steps[:, column] == -1)
        counts.append(int(np.count_nonzero((ends - starts) >= minimum_run)))
    return (float(np.mean(counts)) if counts else float("nan")), counts, channels


def _dominant_bar_length(aligned, cell_size, settings):
    """
    Bed wavelengths along the flow from a derivative-of-Gaussian wavelet, the
    density averaged over every row. The peak is the dominant bar length.
    """
    minimum = max(settings["wavelet_min_m"] / cell_size, 2.0)
    maximum = max(settings["wavelet_max_m"] / cell_size, minimum * 2)
    scales_count = int(settings["wavelet_scales"])
    # The second derivative of a Gaussian (Mexican hat): its Fourier wavelength
    # is 2 pi s / sqrt(2.5), so the scale range is set from the lengths wanted.
    to_wavelength = 2.0 * math.pi / math.sqrt(2.5)
    scales = np.geomspace(minimum / to_wavelength, maximum / to_wavelength, scales_count)

    # Every row at once per scale, with gaps filled by the row's mean so they
    # contribute nothing, and each row's power taken over its own valid cells.
    from scipy.signal import fftconvolve
    valid = np.isfinite(aligned)
    counts = valid.sum(axis=1)
    usable = counts >= 16
    rows_used = int(usable.sum())
    power = np.zeros(scales_count)
    if rows_used:
        means = np.where(counts > 0, np.nansum(np.where(valid, aligned, 0.0), axis=1)
                         / np.maximum(counts, 1), 0.0)
        signal = np.where(valid, aligned - means[:, None], 0.0)[usable]
        valid_rows = valid[usable]
        # Kernels are limited by the row width, not by the shortest valid row:
        # rows clipped by the rotation are short, and capping every scale to
        # them would cut the long wavelengths short everywhere.
        width = aligned.shape[1]
        for index, scale in enumerate(scales):
            half = int(min(5 * scale, width // 2))
            if half < 2:
                continue
            x = np.arange(-half, half + 1) / scale
            kernel = (1.0 - x * x) * np.exp(-x * x / 2.0) / math.sqrt(scale)
            # By FFT: the long kernels at large scales would make a direct
            # convolution grow with the kernel's length.
            response = fftconvolve(signal, kernel[None, :], mode="same", axes=1)
            squared = np.where(valid_rows, response * response, 0.0)
            power[index] = float(np.sum(squared.sum(axis=1) / valid_rows.sum(axis=1)))

    if rows_used == 0 or not power.any():
        return float("nan"), [], []
    lengths = scales * to_wavelength * cell_size
    density = power / power.sum()
    return float(lengths[int(np.argmax(density))]), lengths.tolist(), density.tolist()


def _bar_shapes(detrended, cell_size, settings, level=None, split=False, alpha=0.01):
    """
    Bars as the ground above a level, smoothed at bedform scale: length, width,
    aspect ratio and perimeter-to-area ratio of each. The level is the mean bed
    by default, as Schuurman et al. define it; a higher one separates the
    platforms a bar is built from.

    With split on, each bar is also divided along its channels: one region per
    summit, with the boundaries settling in the low lines between them. That
    separates subsections standing at different heights, which no single level
    can, since a level high enough to part the tall ones has already lost the
    low ones.
    """
    filled = np.where(np.isfinite(detrended), detrended, np.nanmean(detrended)).astype(np.float32)
    cutoff = max(int(round(settings["bedform_cutoff_m"] / cell_size)) | 1, 3)
    smooth = cv2.GaussianBlur(filled, (0, 0), cutoff / 3.0)
    threshold = float(np.nanmean(smooth)) if level is None else float(level)
    above = (smooth > threshold) & np.isfinite(detrended)
    minimum_cells = settings["min_bar_area_m2"] / (cell_size * cell_size)

    if split:
        labels = _split_along_channels(smooth, above, cell_size, settings, alpha)
    else:
        _, labels = cv2.connectedComponents(above.astype(np.uint8), connectivity=8)

    from scipy import ndimage
    bars = []
    rows, columns = labels.shape
    for label, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        top, left = max(box[0].start - 1, 0), max(box[1].start - 1, 0)
        window = (slice(top, min(box[0].stop + 1, rows)), slice(left, min(box[1].stop + 1, columns)))
        mask = (labels[window] == label).astype(np.uint8)
        area_cells = int(mask.sum())
        if area_cells < minimum_cells:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
                                       offset=(left, top))
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(contour)
        length, width = max(side_a, side_b) * cell_size, min(side_a, side_b) * cell_size
        area = float(area_cells) * cell_size * cell_size
        perimeter = cv2.arcLength(contour, True) * cell_size
        bars.append({"label": int(label), "length_m": length, "width_m": width,
                     "aspect": width / length if length else float("nan"),
                     "perimeter_area": perimeter / area if area else float("nan"),
                     "area_m2": area, "outline": contour.reshape(-1, 2).tolist()})
    bars.sort(key=lambda bar: -bar["area_m2"])
    return bars


def _split_along_channels(smooth, above, cell_size, settings, alpha):
    """
    One region per significant summit, grown down to the channels between them
    by watershed. A summit counts when it stands above the saddle to its
    neighbour by more than chance allows for this surface's own roughness at
    the significance level: the robust spread of the detail finer than the
    smallest bar, times the one-sided normal quantile, times the square root of
    two for a difference between two points.
    """
    from scipy.stats import norm
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    scale = max(math.sqrt(settings["min_bar_area_m2"]) / cell_size / 2.0, 1.0)
    detail = smooth - cv2.GaussianBlur(smooth, (0, 0), scale)
    values = detail[above]
    if values.size < 10:
        _, labels = cv2.connectedComponents(above.astype(np.uint8), connectivity=8)
        return labels
    spread = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    height = float(norm.ppf(1.0 - alpha)) * spread * SQRT2

    summits = h_maxima(np.where(above, smooth, float(smooth[above].min())).astype(np.float64),
                       max(height, 1e-6)) & above
    markers, _ = ndimage_label(summits)
    if markers.max() == 0:
        _, labels = cv2.connectedComponents(above.astype(np.uint8), connectivity=8)
        return labels
    return watershed(-smooth, markers, mask=above)


def ndimage_label(mask):
    from scipy import ndimage
    return ndimage.label(mask, structure=np.ones((3, 3), int))


# ======================================================================================================================
# Superimposed bedforms (Reesink and Bridge 2011)
# ======================================================================================================================
def superimposed_bedforms(elevation, cell_size, settings=None):
    """
    For each bar, the height of the bedforms on it against the bar's height.
    The bar is the surface smoothed at the bedform cutoff; the bedforms are what
    the smoothing removed.
    """
    settings = {**DEFAULTS, **(settings or {})}
    values = clean(elevation)
    if np.isfinite(values).sum() < 100:
        return {"error": "Too few valid cells in this region."}
    detrended, _ = detrend(values)
    valid = np.isfinite(detrended)
    filled = np.where(valid, detrended, np.nanmean(detrended)).astype(np.float32)

    cutoff = max(settings["bedform_cutoff_m"] / cell_size, 2.0)
    host = cv2.GaussianBlur(filled, (0, 0), cutoff / 3.0).astype(np.float64)
    residual = filled - host

    bars = _bar_shapes(detrended, cell_size, settings)
    shape = detrended.shape
    results = []
    for bar in bars:
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [np.asarray(bar["outline"], np.int32)], 1)
        inside = (mask > 0) & valid
        if inside.sum() < 25:
            continue
        # Bar height: the bar's own relief above the surrounding bed.
        host_inside = host[inside]
        host_height = float(np.percentile(host_inside, BAR_HEIGHT_HIGH_PERCENTILE)
                            - np.percentile(filled[valid], BAR_HEIGHT_LOW_PERCENTILE))
        bedform = residual[inside]
        bedform_height = float(np.percentile(bedform, BAR_HEIGHT_HIGH_PERCENTILE)
                               - np.percentile(bedform, BAR_HEIGHT_LOW_PERCENTILE))
        ratio = bedform_height / host_height if host_height > 0 else float("nan")
        results.append({**{k: bar[k] for k in ("label", "length_m", "width_m", "area_m2",
                                                 "outline")},
                        "host_height": host_height, "bedform_height": bedform_height,
                        "ratio": ratio,
                        "above_threshold": bool(ratio > SUPERIMPOSED_RATIO_THRESHOLD)})
    return {"bars": results, "residual": residual, "cutoff_m": settings["bedform_cutoff_m"]}


# ======================================================================================================================
# Inundation (Korus et al. 2020)
# ======================================================================================================================
def daily_gage_height(site, start, end):
    """[(date, feet)] of daily mean gage height (parameter 00065) from NWIS."""
    params = {"format": "json", "sites": site, "parameterCd": "00065",
              "startDT": start.isoformat(), "endDT": end.isoformat(), "statCd": "00003"}
    response = requests.get(NWIS_DV_URL, params=params, headers={"User-Agent": USER_AGENT},
                            timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    series = response.json().get("value", {}).get("timeSeries", [])
    if not series:
        return []
    readings = []
    for entry in series[0]["values"][0]["value"]:
        try:
            value = float(entry["value"])
        except (TypeError, ValueError):
            continue
        if value <= -999999:            # NWIS marks missing values this way
            continue
        readings.append((entry["dateTime"][:10], value))
    return readings


def datum_shift_to_navd88(latitude, longitude, datum_code):
    """
    Metres to add to a gage datum to put it on NAVD88. Zero when the datum is
    already NAVD88; for NGVD29 the shift comes from NOAA's VERTCON through NCAT.
    Returns (shift, note); shift is None when it could not be obtained.
    """
    code = (datum_code or "").upper().replace(" ", "")
    if code in ("NAVD88", "NAVD1988"):
        return 0.0, "gage datum is already NAVD88"
    if code not in ("NGVD29", "NGVD1929"):
        return None, f"gage datum {datum_code or 'unknown'} cannot be converted automatically"
    params = {"lat": latitude, "lon": longitude, "inDatum": "nad83(2011)",
              "outDatum": "nad83(2011)", "inVertDatum": "ngvd29",
              "outVertDatum": "navd88", "orthoHt": 0.0}
    try:
        response = requests.get(NCAT_URL, params=params, headers={"User-Agent": USER_AGENT},
                                timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        shift = float(data.get("destOrthoht", data.get("destOrthoHt")))
        return shift, "NGVD29 to NAVD88 by VERTCON (NOAA NCAT)"
    except Exception as err:
        return None, f"NGVD29 to NAVD88 conversion failed: {type(err).__name__}: {err}"


def inundation_days(elevation, gage_cell, cell_size, plane, water_surfaces, slope_override=None):
    """
    Days each cell is under water. The water surface at a cell is the gage's
    water-surface elevation lowered by the reach slope times the distance
    downstream of the gage; slope and direction come from the fitted plane
    unless slope_override (metres per metre) is given.
    """
    values = clean(elevation)
    _, per_column, per_row = plane
    gradient = math.hypot(per_column, per_row)
    slope = slope_override if slope_override else gradient / cell_size
    if gradient == 0:
        return np.zeros(values.shape), slope
    # Unit vector pointing downhill, in cells.
    down_column, down_row = -per_column / gradient, -per_row / gradient
    rows, columns = np.indices(values.shape)
    gage_row, gage_column = gage_cell
    downstream = ((columns - gage_column) * down_column + (rows - gage_row) * down_row) * cell_size

    # A cell is wet on a day when that day's gage water surface exceeds the
    # cell's elevation plus the fall to it. Sorting the days once, the number of
    # wet days is how many sorted surfaces lie above that, a binary search.
    ordered = np.sort(np.asarray(water_surfaces, np.float64))
    needed = values + slope * downstream
    finite = np.isfinite(needed)
    wet_days = np.full(values.shape, -1, np.int32)
    wet_days[finite] = len(ordered) - np.searchsorted(ordered, needed[finite], side="right")
    return wet_days, slope


# ======================================================================================================================
# Mounds and pockets: slope-ridge enclosures and an arc-length Kalman tracker
# ======================================================================================================================
# The features are elongated and often ramped, so they are found from their
# margins rather than from their shape. A margin is a ridge in the slope field:
# the bright lines of the slope view. Areas those ridges enclose are the
# candidates, whatever their elongation or curvature.
#
# A ramped feature's up-ramp end usually has no margin, so its ring is open,
# like a U. An open ridge whose ends curve back toward each other is closed
# across that gap, and short breaks elsewhere are bridged up to a set distance.
#
# Each enclosure's boundary is then refined by an extended Kalman filter that
# walks along it by arc length. The state is the margin's offset along the
# local normal and that offset's rate of change along the boundary; the
# measurement on each normal is where the slope peaks, with a variance from the
# peak's sharpness. Where the margin is faint or absent, as at a ramp, the
# filter carries the outline on from its prediction. Two laps and a
# Rauch-Tung-Striebel smoother close the loop. The outline's centroid error
# comes from propagating each point's uncertainty through the centroid, with
# neighbouring errors correlated as the smoother makes them, and gives the
# centre's error ellipse. Features are elongated, so a circle would hide the
# direction the centre is least certain in; the ellipse is reported by how far
# it reaches along and across the feature's long axis.

# Probabilities of the centre's error ellipses in the report.
ERROR_ELLIPSE_PROBABILITIES = (0.50, 0.95)

FEATURE_DEFAULTS = {
    "min_size_m": 3.0,            # smallest length searched for
    "max_size_m": 100.0,          # largest length searched for
    "alpha": 0.01,                # significance level for a feature's margin and height
    "fdr_correction": True,       # control the false discovery rate across all candidates
    "margin_continue_fraction": 0.5,  # a traced margin continues while its slope stays above
                                      # this share of the starting threshold
    "margin_high_sigma": 12.0,    # a margin starts where the slope is this many noise sigmas
    "margin_low_sigma": 6.0,      # and continues while it stays above this many
    "edge_sigma_cells": 1.0,      # smoothing before margins are traced, in cells
    "gap_m": 2.0,                 # breaks in a margin up to this length are bridged
    "ramp_closure": 0.6,          # margin ends are joined when their gap is under this share
                                  # of the shorter line's length (closes ramp ends)
    "ramp_gap_m": 15.0,           # the widest open ramp end that is closed
    "shape_freedom": 0.5,         # process noise: how quickly the margin offset may change
    "gate": 9.0,                  # Mahalanobis gate, chi-square with one degree of freedom
    "relative_peak": 0.5,         # a margin candidate must reach this share of the normal's strongest
    "search_m": 4.0,              # how far either side of the boundary each normal is searched
    "min_cells_across": 3,        # narrower than this the margin is not resolved
    "error_correlation_points": 3.0,  # how far along the loop smoothed errors stay correlated
    "asymmetry_min": 1.3,         # a lee direction is reported only above this lee/stoss ratio
    "boundary_spacing_cells": 1.0,  # distance between points walked along the boundary
    "min_measured_fraction": 0.3, # outlines with fewer measured points are not reported
}
SQRT2 = math.sqrt(2.0)
ENCLOSURE_RING_CELLS = 3      # width of the ring a feature is compared with, in cells
MARGIN_BAND_CELLS = 2.0       # at least this far either side of the outline, the slope is
                              # compared with the outline's own; wider where the riser is
MAX_CORRELATION_SAMPLE = 1024  # the correlation length is measured on at most this many cells
                               # along each side, subsampled; it varies slowly
# Directions the margin can face, for comparing opposite sides.
ASYMMETRY_SECTORS = 8


def find_features(elevation, cell_size, settings=None, mounds=True, pockets=True):
    """Mounds and pockets in one elevation patch, each with a tracked outline."""
    settings = {**FEATURE_DEFAULTS, **(settings or {})}
    values = clean(elevation)
    if np.isfinite(values).sum() < 100:
        return {"error": "Too few valid cells in this region."}
    detrended, _ = detrend(values)
    valid = np.isfinite(detrended)
    surface = np.where(valid, detrended, np.nanmedian(detrended)).astype(np.float32)
    sigma_z = _vertical_noise(surface)

    smooth = cv2.GaussianBlur(surface, (0, 0), 1.0)
    dy, dx = np.gradient(smooth)
    slope = np.hypot(dx, dy).astype(np.float32)                 # rise per cell

    ridges = _slope_ridges(surface, valid, settings, cell_size)
    enclosures = _enclosures(ridges, valid, cell_size, settings)

    features, counts = [], {"below_resolution": 0, "unconfirmed": 0, "outside_range": 0,
                            "not_raised_or_sunk": 0, "not_significant": 0}
    correlation_cells = correlation_length(surface, valid, settings["min_size_m"] / cell_size)
    candidates = []
    for enclosure in enclosures:
        kind, p_height, difference = _classify(surface, enclosure, correlation_cells)
        if kind is None or (kind == "mound" and not mounds) or (kind == "pocket" and not pockets):
            counts["not_raised_or_sunk"] += kind is None
            continue
        feature = _track_boundary(enclosure, surface, slope, kind, cell_size, sigma_z, settings,
                                  correlation_cells)
        if feature is None:
            counts["unconfirmed"] += 1
            continue
        if feature["width_m"] < settings["min_cells_across"] * cell_size:
            counts["below_resolution"] += 1
            continue
        if not (settings["min_size_m"] <= feature["length_m"] <= settings["max_size_m"] * 1.5):
            counts["outside_range"] += 1
            continue
        if feature["points_measured"] < settings["min_measured_fraction"] * feature["points"]:
            counts["unconfirmed"] += 1
            continue
        feature["p_height"] = p_height
        feature["relief_m"] = abs(difference)
        # The margin decides: a ramped feature's interior can barely stand
        # above the bar while its margin is plain. The height test sets mound
        # or pocket and is reported, but does not reject.
        feature["p"] = feature["p_margin"]
        candidates.append(feature)

    # Significance across all candidates at once: with hundreds tested, a
    # per-feature alpha would pass several by chance. Benjamini-Hochberg keeps
    # the expected share of false features at alpha.
    alpha = settings["alpha"]
    if candidates:
        p_values = np.array([feature["p"] for feature in candidates])
        if settings["fdr_correction"]:
            order = np.argsort(p_values)
            ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
            q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
            q_values = np.empty_like(q_sorted)
            q_values[order] = np.minimum(q_sorted, 1.0)
        else:
            q_values = p_values
        for feature, q in zip(candidates, q_values):
            feature["q"] = float(q)
            if q <= alpha:
                features.append(feature)
            else:
                counts["not_significant"] += 1

    features.sort(key=lambda feature: -feature["area_m2"])
    for index, feature in enumerate(features, start=1):
        feature["index"] = index
    return {"features": features, **counts, "noise_m": sigma_z, "cell_size": cell_size,
            "surface": surface, "settings": settings, "ridges": ridges,
            "correlation_m": correlation_cells * cell_size}


def correlation_length(surface, valid, scale_cells, limit=MAX_CORRELATION_SAMPLE):
    """
    Distance, in cells, over which the surface's fine-scale variation stays
    correlated: where its autocorrelation falls to 1/e. Neighbouring cells of a
    DEM are not independent, an interpolated one least of all, so statistics
    counted per cell overstate the evidence; dividing by this corrects that.
    Measured on the variation finer than scale_cells, which is what a feature
    is tested against.
    """
    residual = surface.astype(np.float64) - cv2.GaussianBlur(
        surface.astype(np.float32), (0, 0), max(float(scale_cells), 1.0)).astype(np.float64)
    residual = np.where(valid, residual, 0.0)
    rows, columns = residual.shape
    step = max(int(math.ceil(max(rows, columns) / float(limit))), 1)
    residual = residual[::step, ::step]
    weights = valid[::step, ::step].astype(np.float64)
    residual -= residual[weights > 0].mean() if weights.any() else 0.0
    residual *= weights
    padded = (2 * residual.shape[0], 2 * residual.shape[1])
    spectrum = np.fft.rfft2(residual, padded)
    counts = np.fft.irfft2(np.abs(np.fft.rfft2(weights, padded)) ** 2, padded)
    autocov = np.fft.irfft2(np.abs(spectrum) ** 2, padded) / np.maximum(counts, 1.0)
    autocov = np.fft.fftshift(autocov)
    centre_row, centre_column = autocov.shape[0] // 2, autocov.shape[1] // 2
    zero = autocov[centre_row, centre_column]
    if zero <= 0:
        return 1.0
    for lag in range(1, min(centre_row, centre_column)):
        ring = [autocov[centre_row + lag, centre_column], autocov[centre_row - lag, centre_column],
                autocov[centre_row, centre_column + lag], autocov[centre_row, centre_column - lag]]
        if np.mean(ring) / zero < math.exp(-1.0):
            return float(lag * step)
    return float(min(centre_row, centre_column) * step)


def _effective(count, cells_per_independent):
    """Independent samples among count correlated ones."""
    return max(count / max(cells_per_independent, 1.0), 2.0)


def _vertical_noise(surface):
    """The flattest decile's local spread: the elevation model's noise floor."""
    centred = (surface - float(np.mean(surface))).astype(np.float32)
    mean = cv2.blur(centred, (9, 9))
    mean_square = cv2.blur(centred * centred, (9, 9))
    local = np.sqrt(np.maximum(mean_square - mean * mean, 0))
    quiet = local[local <= np.percentile(local, 10)]
    return float(np.median(quiet)) if quiet.size else 0.01


def _slope_ridges(surface, valid, settings, cell_size):
    """
    The margins as thin lines, by edge detection on the elevation: the slope's
    crest across each riser (non-maximum suppression), kept where it is among
    the steepest in the region and continued along weaker stretches that join
    it (hysteresis). A plain slope threshold does not work here: on a narrow
    feature the riser on both sides and the textured top all clear it, and
    its skeleton is the feature's centreline rather than its margin.
    """
    from skimage.feature import canny
    from scipy import ndimage
    image = surface.astype(np.float64)
    sigma = settings["edge_sigma_cells"]
    # Thresholds in units of the gradient's own noise, measured on the flattest
    # ground. A percentile of the region's slopes would depend on whatever else
    # is in view, so one strong feature could hide a weak one beside it.
    smooth = ndimage.gaussian_filter(image, sigma)
    magnitude = np.hypot(ndimage.sobel(smooth, 0), ndimage.sobel(smooth, 1))
    # A margin is a slope that stands out from the surface's own slopes: above
    # their median by the one-sided normal quantile for alpha, in units of their
    # robust spread. Nothing is fixed: a smooth DEM, a rough one, a textured
    # bar or a plain one each set their own threshold.
    from scipy.stats import norm
    values = magnitude[valid]
    centre = float(np.median(values))
    spread = 1.4826 * float(np.median(np.abs(values - centre)))
    high = centre + float(norm.ppf(1.0 - settings["alpha"])) * max(spread, 1e-12)
    low = centre + settings["margin_continue_fraction"] * (high - centre)
    edges = canny(image, sigma=sigma, low_threshold=low, high_threshold=high)
    edges &= valid
    minimum = max(int(round(settings["min_size_m"] / cell_size)), 3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(edges.astype(np.uint8),
                                                               connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum
    return keep[labels]


def _enclosures(ridges, valid, cell_size, settings):
    """
    Areas enclosed by the ridge network, after bridging short breaks and
    closing open ramp ends. Each is returned as a boolean mask.
    """
    from skimage.morphology import disk, dilation as binary_dilation
    closed = ridges.copy()
    closed |= _close_open_ends(ridges, settings, cell_size)

    gap_cells = settings["gap_m"] / cell_size
    radius = max(int(round(gap_cells / 2.0)), 0)
    thick = binary_dilation(closed, disk(radius)) if radius else closed

    # Enclosed means not reachable from the region's edge without crossing a margin.
    free = (~thick & valid).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0],
                                           labels[:, -1]])))
    minimum_area = (settings["min_size_m"] / cell_size) ** 2 * 0.25
    maximum_area = (settings["max_size_m"] * 1.5 / cell_size) ** 2
    rows, columns = labels.shape
    # Each enclosure in its own window, padded for the growing below and for
    # the ring the classification looks at, rather than as a full-size mask.
    pad = radius + 1 + ENCLOSURE_RING_CELLS + 1
    enclosures = []
    for label in range(1, count):
        if label in border:
            continue
        area = stats[label, cv2.CC_STAT_AREA]
        if not (minimum_area <= area <= maximum_area):
            continue
        x, y, w, h = (int(v) for v in stats[label, :4])
        top, left = max(y - pad, 0), max(x - pad, 0)
        window = (slice(top, min(y + h + pad, rows)), slice(left, min(x + w + pad, columns)))
        mask = labels[window] == label
        # Grown back out to the margin's centreline, which the thickening covered.
        if radius:
            mask = binary_dilation(mask, disk(radius + 1)) & valid[window]
        enclosures.append((window, mask))
    return enclosures


def _close_open_ends(ridges, settings, cell_size):
    """
    Lines across the open ends of U-shaped margins: a ramped feature's up-ramp
    end has no margin, so its ring would otherwise never close. A margin
    qualifies when its two ends are closer together than ramp_closure times
    its length, so a straight or gently curved margin is left open.
    """
    from skimage.measure import label as label_components
    from skimage.morphology import skeletonize
    lines = np.zeros(ridges.shape, np.uint8)
    # Thinned first: edge lines step diagonally in pairs of cells, which gives
    # an end cell two neighbours, and it would not be found as an end.
    ridges = skeletonize(ridges)
    neighbours = cv2.filter2D(ridges.astype(np.uint8), -1, np.ones((3, 3), np.float32),
                              borderType=cv2.BORDER_CONSTANT)
    endpoints = np.argwhere(ridges & (neighbours == 2))       # itself plus one neighbour
    if len(endpoints) < 2:
        return lines.astype(bool)
    components = label_components(ridges, connectivity=2)
    sizes = np.bincount(components.ravel())
    owner = components[endpoints[:, 0], endpoints[:, 1]]
    length = sizes[owner]

    # The two sides of a ramped feature are often traced as separate lines, so
    # ends are paired across lines as well as within one. Only long lines take
    # part, and the gap has to be short relative to them, so the short lines of
    # bedform texture are not stitched into false enclosures.
    reach = settings["ramp_gap_m"] / cell_size
    pairs = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            separation = float(np.hypot(*(endpoints[i] - endpoints[j])))
            if separation < 1.5 or separation > reach:
                continue
            shorter = min(length[i], length[j])
            if separation < settings["ramp_closure"] * shorter:
                # Ranked by the gap relative to the lines it joins, so the ends
                # of two long margins are paired before either is taken by a
                # short texture line that happens to lie closer.
                pairs.append((separation / shorter, i, j))
    used = set()
    for _, i, j in sorted(pairs):
        if i in used or j in used:
            continue
        used.update((i, j))
        (r1, c1), (r2, c2) = endpoints[i], endpoints[j]
        cv2.line(lines, (int(c1), int(r1)), (int(c2), int(r2)), 1, 1)
    return lines.astype(bool)


def _classify(surface, enclosure, correlation_cells):
    """
    Mound or pocket, and the one-sided p-value that the interior differs from
    its ring in that direction: a Welch t-test whose sample sizes are the
    effective numbers of independent cells, not the raw counts.
    Returns (kind, p, difference in elevation units).
    """
    from skimage.morphology import dilation as binary_dilation, disk
    from scipy.stats import t as student
    window, mask = enclosure
    local = surface[window]
    ring = binary_dilation(mask, disk(ENCLOSURE_RING_CELLS)) & ~mask
    if mask.sum() < 3 or ring.sum() < 3:
        return None, 1.0, 0.0
    inside, outside = local[mask].astype(np.float64), local[ring].astype(np.float64)
    difference = float(inside.mean() - outside.mean())
    kind = "mound" if difference > 0 else "pocket"
    area = math.pi * correlation_cells * correlation_cells
    n1, n2 = _effective(inside.size, area), _effective(outside.size, area)
    v1, v2 = inside.var(ddof=1), outside.var(ddof=1)
    error = math.sqrt(v1 / n1 + v2 / n2)
    if error == 0:
        return kind, 0.0 if difference else 1.0, difference
    statistic = abs(difference) / error
    freedom = (v1 / n1 + v2 / n2) ** 2 / max((v1 / n1) ** 2 / max(n1 - 1, 1)
                                             + (v2 / n2) ** 2 / max(n2 - 1, 1), 1e-300)
    return kind, float(student.sf(statistic, freedom)), difference


def _track_boundary(enclosure, surface, slope, kind, cell_size, sigma_z, settings,
                    correlation_cells=1.0):
    """
    The arc-length Kalman outline of one enclosure. The walk follows the
    enclosure's boundary; on each normal the slope peak is the measurement.
    """
    window, mask = enclosure
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE,
                                   offset=(window[1].start, window[0].start))
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(contour) < 8:
        return None

    # Resampled at even spacing, and smoothed so normals are stable.
    spacing = float(settings["boundary_spacing_cells"])
    closed = np.vstack([contour, contour[:1]])
    lengths = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    perimeter = lengths[-1]
    count = max(int(perimeter / spacing), 12)
    s_values = np.linspace(0.0, perimeter, count, endpoint=False)
    base = np.column_stack([np.interp(s_values, lengths, closed[:, 0]),
                            np.interp(s_values, lengths, closed[:, 1])])
    window = max(count // 40, 1)
    kernel = np.ones(2 * window + 1) / (2 * window + 1)
    base = np.column_stack([np.convolve(np.concatenate([base[-window:, i], base[:, i],
                                                         base[:window, i]]), kernel, "valid")
                            for i in range(2)])
    tangent = np.roll(base, -1, axis=0) - np.roll(base, 1, axis=0)
    tangent /= np.maximum(np.hypot(tangent[:, 0], tangent[:, 1])[:, None], 1e-9)
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    # Outward normals: point away from the enclosure's interior.
    centre = base.mean(axis=0)
    if np.mean(np.sum((base - centre) * normal, axis=1)) < 0:
        normal = -normal

    # Samples along each normal, inside to outside.
    search = max(settings["search_m"] / cell_size, 2.0)
    offsets = np.arange(-search, search + 0.5, 0.5, dtype=np.float32)
    map_x = (base[:, 0][:, None] + normal[:, 0][:, None] * offsets[None]).astype(np.float32)
    map_y = (base[:, 1][:, None] + normal[:, 1][:, None] * offsets[None]).astype(np.float32)
    along_slope = cv2.remap(slope, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    along_height = cv2.remap(surface, map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    if kind == "pocket":
        along_height = -along_height                        # a pocket's rim is its high side

    sigma_slope = SQRT2 * sigma_z
    floor = 2.0 * sigma_slope
    step = 0.5
    measurements = []
    for k in range(count):
        profile = along_slope[k]
        candidates = []
        for i in range(1, len(profile) - 1):
            if profile[i] < floor or not (profile[i] >= profile[i - 1] and profile[i] > profile[i + 1]):
                continue
            curvature = (profile[i - 1] - 2 * profile[i] + profile[i + 1]) / (step * step)
            sigma = sigma_slope / max(abs(float(curvature)), 1e-9)
            sigma = float(min(max(sigma, step), search))
            candidates.append((float(offsets[i]), sigma * sigma, float(profile[i])))
        if candidates:
            strongest = max(c[2] for c in candidates)
            candidates = [c for c in candidates if c[2] >= settings["relative_peak"] * strongest]
        measurements.append(candidates)

    # The filter: state [offset along the normal, its rate per unit arc length].
    q = float(settings["shape_freedom"]) * 0.1
    F = np.array([[1.0, spacing], [0.0, 1.0]])
    Q = q * np.array([[spacing ** 3 / 3, spacing ** 2 / 2], [spacing ** 2 / 2, spacing]])
    H = np.array([1.0, 0.0])
    steps = 2 * count
    x = np.array([0.0, 0.0])
    P = np.diag([search ** 2, 1.0])
    predicted, filtered, chosen, innovations = [], [], [None] * steps, []
    for n in range(steps):
        k = n % count
        if n:
            x = F @ x
            P = F @ P @ F.T + Q
        predicted.append((x.copy(), P.copy()))
        best = None
        strongest = max((c[2] for c in measurements[k]), default=1.0)
        for offset, variance, peak in measurements[k]:
            S = float(H @ P @ H) + variance
            innovation = offset - x[0]
            d2 = innovation * innovation / S
            if d2 <= settings["gate"]:
                score = d2 - 2.0 * math.log(max(peak / strongest, 1e-9))
                if best is None or score < best[0]:
                    best = (score, innovation, S, peak, offset)
        if best is not None:
            _, innovation, S, peak, offset = best
            K = (P @ H) / S
            x = x + K * innovation
            P = (np.eye(2) - np.outer(K, H)) @ P
            chosen[n] = (offset, peak)
            if n >= count:
                innovations.append(innovation * innovation / S)
        filtered.append((x.copy(), P.copy()))

    smoothed = [None] * steps
    smoothed[-1] = filtered[-1]
    for n in range(steps - 2, -1, -1):
        x_f, P_f = filtered[n]
        x_p, P_p = predicted[n + 1]
        gain = P_f @ F.T @ np.linalg.pinv(P_p)
        smoothed[n] = (x_f + gain @ (smoothed[n + 1][0] - x_p),
                       P_f + gain @ (smoothed[n + 1][1] - P_p) @ gain.T)

    # The second lap, indexed by boundary point.
    lap = [smoothed[count + k] for k in range(count)]
    lap_chosen = [chosen[count + k] for k in range(count)]
    nis = max(float(np.mean(innovations)) if innovations else 1.0, 1.0)
    offset = np.array([s[0][0] for s in lap])
    sigma = np.sqrt(np.maximum([s[1][0, 0] for s in lap], 0.0)) * math.sqrt(nis)

    margin = base + normal * offset[:, None]

    # Toe and crest on each normal, either side of the tracked margin.
    toe, crest = np.zeros(count), np.zeros(count)
    for k in range(count):
        bend = np.gradient(np.gradient(along_height[k], step), step)
        at = int(np.argmin(np.abs(offsets - offset[k])))
        inside, outside = bend[:max(at, 1)], bend[at:]
        crest[k] = float(offsets[int(np.argmin(inside))]) if inside.size else offset[k]
        toe[k] = float(offsets[at + int(np.argmax(outside))]) if outside.size else offset[k]

    # Margin test: is the slope on the outline higher than just inside it and
    # just outside it? A one-sided rank test against each side, with the
    # outline's samples counted as the independent ones they are: neighbours
    # along it are correlated over the surface's correlation length.
    from scipy.stats import mannwhitneyu
    band = max(float(np.median(toe - crest)), MARGIN_BAND_CELLS)
    sample = lambda shift: np.array([np.interp(offset[k] + shift, offsets, along_slope[k])
                                     for k in range(count)])
    on_margin, inside_band, outside_band = sample(0.0), sample(-band), sample(band)
    shrink = math.sqrt(min(spacing / max(correlation_cells, spacing), 1.0))
    p_sides = []
    for other in (inside_band, outside_band):
        result = mannwhitneyu(on_margin, other, alternative="greater")
        n1, n2 = len(on_margin), len(other)
        mean = n1 * n2 / 2.0
        deviation = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        z = (float(result.statistic) - mean) / max(deviation, 1e-12)
        from scipy.stats import norm
        p_sides.append(float(norm.sf(z * shrink)))
    p_margin = max(p_sides)            # steeper than both sides, so the weaker of the two counts

    def points(values):
        return (base + normal * values[:, None]).round().astype(int).tolist()

    area = abs(_shoelace(margin))
    outer = abs(_shoelace(base + normal * (offset + sigma)[:, None]))
    inner = abs(_shoelace(base + normal * (offset - sigma)[:, None]))

    covariance = _centroid_covariance_normals(base, normal, offset, sigma,
                                              settings["error_correlation_points"])
    covariance *= cell_size * cell_size
    centroid = _polygon_centroid(margin)

    # Shape: principal axes of the outline.
    centred = margin - margin.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centred.T))
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    projected_long = centred @ major
    projected_short = centred @ np.array([-major[1], major[0]])
    length = float(projected_long.max() - projected_long.min()) * cell_size
    width = float(projected_short.max() - projected_short.min()) * cell_size
    orientation = (math.degrees(math.atan2(major[0], -major[1])) + 360.0) % 180.0
    minor = np.array([-major[1], major[0]])
    centre_error = {}
    for probability in ERROR_ELLIPSE_PROBABILITIES:
        percent = int(round(probability * 100))
        centre_error[f"centre_along{percent}_m"] = _ellipse_reach(covariance, major, probability)
        centre_error[f"centre_across{percent}_m"] = _ellipse_reach(covariance, minor, probability)
    centre_error["centre_error_tilt_degrees"] = _ellipse_tilt(covariance, major)

    # Lee and ramp: margin steepness by the direction the margin faces, compared
    # with the side facing the opposite way. Comparing the steepest stretch with
    # the weakest would call every lens asymmetric, its sides against its ends.
    peaks = np.array([c[1] if c else 0.0 for c in lap_chosen])
    facing = (np.degrees(np.arctan2(normal[:, 0], -normal[:, 1])) + 360.0) % 360.0
    sectors = int(ASYMMETRY_SECTORS)
    sector_width = 360.0 / sectors
    steepness = np.full(sectors, np.nan)
    for sector in range(sectors):
        centre_angle = sector * sector_width
        difference = np.abs((facing - centre_angle + 180.0) % 360.0 - 180.0)
        members = difference <= sector_width / 2.0
        if members.any():
            steepness[sector] = float(np.mean(peaks[members]))
    asymmetry, lee_bearing, ramp_bearing = 1.0, float("nan"), float("nan")
    ratios = []
    floor = 2.0 * sigma_slope        # a side with no margin (a ramp) is as steep as the noise
    for sector in range(sectors):
        opposite = (sector + sectors // 2) % sectors
        if np.isfinite(steepness[sector]) and np.isfinite(steepness[opposite]):
            ratios.append((max(steepness[sector], floor) / max(steepness[opposite], floor),
                           sector, opposite))
    if ratios:
        asymmetry, lee, ramp = max(ratios)
        if asymmetry >= settings["asymmetry_min"]:
            lee_bearing, ramp_bearing = lee * sector_width, ramp * sector_width

    return {
        "kind": kind,
        "centre": [float(centroid[1]), float(centroid[0])],            # row, column
        "centre_cov_m2": covariance.tolist(),
        **centre_error,
        "area_m2": area * cell_size * cell_size,
        "area_sigma_m2": 0.5 * (outer - inner) * cell_size * cell_size,
        "length_m": length, "width_m": width,
        "elongation": length / max(width, 1e-9),
        "orientation_degrees": orientation,
        "diameter_m": 2.0 * math.sqrt(area / math.pi) * cell_size,
        "riser_width_m": float(np.median(toe - crest)) * cell_size,
        "lee_bearing": lee_bearing, "ramp_bearing": ramp_bearing, "asymmetry": asymmetry,
        "points_measured": sum(1 for c in lap_chosen if c), "points": count,
        "p_margin": p_margin,
        "status": "ok",
        "outline": points(offset),
        "outline_toe": points(toe),
        "outline_crest": points(crest),
        "band_outer": points(offset + 2 * sigma),
        "band_inner": points(offset - 2 * sigma),
    }


def _shoelace(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_centroid(points):
    points = np.asarray(points, np.float64)
    x, y = points[:, 0], points[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return points.mean(axis=0)
    return np.array([((x + np.roll(x, -1)) * cross).sum() / (6 * area),
                     ((y + np.roll(y, -1)) * cross).sum() / (6 * area)])


def _centroid_covariance_normals(base, normal, offset, sigma, correlation_points):
    """
    Covariance of the outline centroid from each point's uncertainty along its
    normal, with errors correlated as exp(-separation / correlation_points)
    around the loop.
    """
    count = len(offset)
    reference = _polygon_centroid(base + normal * offset[:, None])
    jacobian = np.zeros((2, count))
    for k in range(count):
        bumped = offset.copy()
        bumped[k] += 0.01
        jacobian[:, k] = (_polygon_centroid(base + normal * bumped[:, None]) - reference) / 0.01
    separation = np.abs(np.subtract.outer(np.arange(count), np.arange(count)))
    separation = np.minimum(separation, count - separation)
    correlation = np.exp(-separation / max(float(correlation_points), 1e-6))
    return jacobian @ (correlation * np.outer(sigma, sigma)) @ jacobian.T


def _ellipse_scale(probability):
    """
    How many standard deviations out the error ellipse containing a 2-D normal
    position with this probability sits: sqrt(-2 ln(1 - P)).
    """
    return math.sqrt(-2.0 * math.log(1.0 - probability))


def _ellipse_reach(covariance, direction, probability):
    """
    How far the error ellipse of this probability reaches from the centre along
    a unit direction: the half-width of its shadow on that direction.
    """
    covariance = np.asarray(covariance, float)
    if not np.all(np.isfinite(covariance)) or np.trace(covariance) <= 0:
        return float("nan")
    variance = float(direction @ covariance @ direction)
    return _ellipse_scale(probability) * math.sqrt(max(variance, 0.0))


def _ellipse_tilt(covariance, direction):
    """
    Angle, 0 to 90 degrees, between the error ellipse's long axis and a
    direction. Near 0 or 90 the along and across reaches are the ellipse's own
    semi-axes; in between, the ellipse is skewed relative to the feature.
    """
    covariance = np.asarray(covariance, float)
    if not np.all(np.isfinite(covariance)) or np.trace(covariance) <= 0:
        return float("nan")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    cosine = abs(float(axis @ direction)) / max(float(np.linalg.norm(axis) * np.linalg.norm(direction)), 1e-12)
    return math.degrees(math.acos(min(cosine, 1.0)))


# ======================================================================================================================
# Local rank
# ======================================================================================================================
# Each cell's height is replaced by its rank in a window centred on it: the
# share of the window's cells lower than it. Cells within the DEM's noise of
# it count as ties, half lower and half higher, so flat ground and noise read
# 0.5 instead of a random scatter of high and low. A small low bar beside a channel
# then ranks as high as a large tall one does in its own window, so one rank
# level finds both. The rank is counted at every cell from a running histogram
# of quantized heights, which the window updates as it slides rather than
# rebuilding.
#
# The window is an ellipse, longer along the flow than across it, turned at
# each cell to the direction its local topography runs (the structure tensor's
# along-contour direction), or to the reach's downhill direction where the
# topography has no clear direction. It is run at several lengths: a bar that
# is one high at a long window and several at a shorter one is compound.
#
# The rank decides where bars are. Their heights and the test of whether they
# stand above their surroundings come from the DEM itself, since the rank's
# scale is not the same everywhere.

LOCAL_RANK_DEFAULTS = {
    "window_min_m": 10.0,             # shortest window length along the flow
    "window_max_m": 320.0,            # longest
    "window_count": 6,                # lengths between them, evenly spaced in log
    "window_width_ratio": 0.5,        # window width across the flow, as a share of its length
    "orientation_steps": 8,           # window directions tried over 180 degrees
    "orientation_smoothing_ratio": 0.25,  # the structure tensor is averaged over this share
                                          # of the window length
    "min_direction_coherence": 0.2,   # below this the topography has no clear direction,
                                      # and the reach's downhill direction is used
    "max_window_cells": 41,           # longer windows run on a coarsened DEM so they span
                                      # at most this many cells, then are resampled back
    "tie_alpha": 0.05,                # two cells closer in height than noise allows at this
                                      # two-sided significance count as a tie
    "bins_per_tie": 4,                # height histogram bins across the tie tolerance
    "height_clip_percent": 0.1,       # heights beyond this percent at either end are clipped,
                                      # so a tree or a void edge cannot widen the bins
    "bar_rank": 0.6,                  # a bar cell is higher than this share of its window
}
# skimage's rank filters slow sharply past this many histogram bins.
MAX_RANK_HISTOGRAM_BINS = 4096
# Rank surfaces are stored as 16-bit integers over 0 to 1.
RANK_STORE_MAX = np.iinfo(np.uint16).max


def local_rank_lengths(settings):
    """The window lengths along the flow, in metres, shortest first."""
    count = max(int(settings["window_count"]), 1)
    if count == 1:
        return [float(settings["window_min_m"])]
    return np.geomspace(settings["window_min_m"], settings["window_max_m"], count).tolist()


def local_rank_surfaces(elevation, cell_size, settings=None):
    """
    The local rank of every cell at each window length, with what is needed to
    find bars on them. Ranks are stored as uint16 over 0 to RANK_STORE_MAX.
    """
    settings = {**LOCAL_RANK_DEFAULTS, **(settings or {})}
    values = clean(elevation)
    valid = np.isfinite(values)
    if valid.sum() < 100:
        return {"error": "Too few valid cells in this region."}
    started = datetime.datetime.now()
    detrended, plane = detrend(values)
    surface = np.where(valid, detrended, np.nanmedian(detrended)).astype(np.float32)
    noise = _rank_noise(surface, valid)

    # The tie tolerance: a height difference between two cells smaller than
    # this is within what the DEM's noise produces at the tie significance.
    from scipy.stats import norm
    tie_m = float(norm.ppf(1.0 - float(settings["tie_alpha"]) / 2.0)) * SQRT2 * noise
    # The height bins, several across the tolerance, so quantizing moves no
    # cell across it by more than a fraction of it.
    clip = float(settings["height_clip_percent"])
    low = float(np.percentile(detrended[valid], clip))
    high = float(np.percentile(detrended[valid], 100.0 - clip))
    bin_width = max(tie_m / max(float(settings["bins_per_tie"]), 1.0), 1e-6)
    bins = int(math.ceil((high - low) / bin_width)) + 1
    if bins > MAX_RANK_HISTOGRAM_BINS:
        bins = MAX_RANK_HISTOGRAM_BINS
        bin_width = (high - low) / (bins - 1)
    tie_bins = max(int(round(tie_m / bin_width)), 0)
    # The rank noise alone exceeds with probability tie_alpha: a cell whose own
    # noise is at that quantile, against neighbours that are noise only.
    excursion = float(norm.ppf(1.0 - float(settings["tie_alpha"]))) * noise
    tie_actual = tie_bins * bin_width
    if noise > 0:
        lower = norm.cdf((excursion - tie_actual) / noise)
        upper = norm.cdf((excursion + tie_actual) / noise)
        noise_rank = float(lower + 0.5 * (upper - lower))
    else:
        noise_rank = 0.5

    # The reach's downhill direction, used where the topography has no direction.
    _, per_column, per_row = plane
    reach_direction = math.atan2(-per_row, -per_column) if (per_row or per_column) else 0.0

    lengths = local_rank_lengths(settings)
    ranks, scales = [], []
    for length_m in lengths:
        length_cells = length_m / cell_size
        factor = max(int(math.ceil(length_cells / float(settings["max_window_cells"]))), 1)
        coarse_values, coarse_valid = _coarsen(detrended, valid, factor)
        direction, coherence = _topography_direction(
            coarse_values, coarse_valid,
            float(settings["orientation_smoothing_ratio"]) * length_cells / factor)
        unclear = coherence < float(settings["min_direction_coherence"])
        direction = np.where(unclear, reach_direction, direction)
        rank = _oriented_rank(coarse_values, coarse_valid, low, bin_width, bins, tie_bins,
                              length_cells / factor,
                              float(settings["window_width_ratio"]),
                              int(settings["orientation_steps"]), direction)
        if factor > 1:
            rank = cv2.resize(rank, (values.shape[1], values.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        rank = np.where(valid, rank, np.nan)
        ranks.append(np.round(np.nan_to_num(rank, nan=0.0) * RANK_STORE_MAX).astype(np.uint16))
        scales.append({"length_m": float(length_m),
                       "width_m": float(length_m) * float(settings["window_width_ratio"]),
                       "coarsening": factor,
                       "unclear_direction_share": float(np.mean(unclear[coarse_valid]))
                       if coarse_valid.any() else float("nan")})

    correlation_cells = correlation_length(surface, valid,
                                           max(lengths[0] / cell_size / 2.0, 1.0))
    return {"ranks": ranks, "scales": scales, "valid": valid, "surface": surface,
            "cell_size": cell_size, "noise_m": noise, "bin_width_m": bin_width,
            "tie_m": tie_actual, "noise_rank": noise_rank,
            "bins": bins, "correlation_cells": correlation_cells, "settings": settings,
            "seconds": (datetime.datetime.now() - started).total_seconds()}


def _rank_noise(surface, valid):
    """
    The DEM's vertical noise for the tie tolerance: the robust spread of each
    cell's difference from the mean of its four neighbours, which on smooth
    ground is the noise alone, scaled by sqrt(1 + 1/4) for the neighbours' own
    share. The median ignores the bar edges and channel banks, where the
    difference is real relief.
    """
    padded = np.pad(surface.astype(np.float64), 1, mode="edge")
    neighbours = (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
    difference = (surface - neighbours)[valid]
    if difference.size < 10:
        return 0.0
    spread = 1.4826 * float(np.median(np.abs(difference - np.median(difference))))
    return spread / math.sqrt(1.25)


def _coarsen(values, valid, factor):
    """Block means of the valid cells, factor cells on a side; NaN where none were valid."""
    if factor <= 1:
        return values, valid
    rows, columns = values.shape
    size = (max(columns // factor, 1), max(rows // factor, 1))
    filled = np.where(valid, values, 0.0).astype(np.float32)
    weight = valid.astype(np.float32)
    total = cv2.resize(filled, size, interpolation=cv2.INTER_AREA)
    share = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)
    coarse_valid = share > 0.5
    coarse = np.where(coarse_valid, total / np.maximum(share, 1e-6), np.nan)
    return coarse, coarse_valid


def _topography_direction(values, valid, smoothing_cells):
    """
    The direction, in radians in image coordinates, that the topography runs
    at each cell (along its contours), and how clearly: the structure tensor's
    coherence, 0 for no preferred direction to 1 for parallel contours.
    """
    surface = np.where(valid, values, np.nanmedian(values[valid]) if valid.any() else 0.0)
    surface = cv2.GaussianBlur(surface.astype(np.float32), (0, 0), 1.0)
    dy, dx = np.gradient(surface)
    sigma = max(float(smoothing_cells), 1.0)
    xx = cv2.GaussianBlur(dx * dx, (0, 0), sigma)
    yy = cv2.GaussianBlur(dy * dy, (0, 0), sigma)
    xy = cv2.GaussianBlur(dx * dy, (0, 0), sigma)
    across = 0.5 * np.arctan2(2.0 * xy, xx - yy)     # the direction of steepest change
    coherence = np.sqrt((xx - yy) ** 2 + 4.0 * xy * xy) / np.maximum(xx + yy, 1e-12)
    return across + math.pi / 2.0, coherence


def _ellipse_footprint(length_cells, width_ratio, angle):
    """An ellipse of the given length and width share, turned to angle (radians, image axes)."""
    semi_major = max(length_cells / 2.0, 1.0)
    semi_minor = max(semi_major * width_ratio, 1.0)
    reach = int(math.ceil(semi_major))
    y, x = np.mgrid[-reach:reach + 1, -reach:reach + 1].astype(np.float64)
    along = x * math.cos(angle) + y * math.sin(angle)
    across = -x * math.sin(angle) + y * math.cos(angle)
    return (along / semi_major) ** 2 + (across / semi_minor) ** 2 <= 1.0


def _band_below_arguments(tie_bins):
    """
    pop_bilateral arguments that count the band [centre - tie_bins, centre].
    It counts the open band between the centre minus one argument and plus the
    other; which of s0 and s1 is the lower side differs from its documentation
    in some skimage versions, so it is found by a probe whose two answers differ.
    """
    from skimage.filters import rank as rank_filters
    probe = np.array([[0, 1, 2, 2, 2]], np.uint16)
    counted = int(rank_filters.pop_bilateral(probe, np.ones((1, 5), bool), s0=3, s1=1)[0, 2])
    s0_is_lower = counted == 5        # (2 - 3, 2 + 1) holds all five; (2 - 1, 2 + 3) three
    return ({"s0": tie_bins + 1, "s1": 1} if s0_is_lower
            else {"s0": 1, "s1": tie_bins + 1})


def _oriented_rank(values, valid, low, bin_width, bins, tie_bins, length_cells, width_ratio,
                   steps, direction):
    """
    The local rank at every cell, each taken in the window direction nearest
    its own. One rank pass per direction, run in parallel; each cell keeps the
    pass for its direction.

    rank = (cells lower by more than the tie tolerance + half the cells within
    it) / cells in the window, all counted from running histograms.
    """
    from concurrent.futures import ThreadPoolExecutor
    from skimage.filters import rank as rank_filters

    quantized = np.clip(np.round((np.nan_to_num(values, nan=low) - low) / bin_width),
                        0, bins - 1).astype(np.uint16)
    mask = valid.astype(np.uint8)
    top = max(int(quantized.max()), 1)
    steps = max(int(steps), 1)
    angles = [math.pi * index / steps for index in range(steps)]
    nearest = np.round((np.mod(direction, math.pi) / math.pi) * steps).astype(np.int64) % steps
    below = _band_below_arguments(tie_bins)

    def one_direction(angle):
        footprint = _ellipse_footprint(length_cells, width_ratio, angle)
        total = rank_filters.pop(quantized, footprint, mask=mask).astype(np.float32)
        total = np.maximum(total, 1.0)
        # skimage's equalize is floor(top x share at or below the centre).
        at_or_below = rank_filters.equalize(quantized, footprint, mask=mask).astype(np.float32) / top
        # Its bilateral count takes an open band, hence the extra bin on each
        # side to include the tolerance itself.
        tied_low = rank_filters.pop_bilateral(quantized, footprint, mask=mask,
                                              **below).astype(np.float32)
        tied = rank_filters.pop_bilateral(quantized, footprint, mask=mask,
                                          s0=tie_bins + 1, s1=tie_bins + 1).astype(np.float32)
        lower = np.maximum(at_or_below - tied_low / total, 0.0)
        return np.clip(lower + 0.5 * tied / total, 0.0, 1.0)

    result = np.zeros(values.shape, np.float32)
    with ThreadPoolExecutor() as pool:
        for index, rank in enumerate(pool.map(one_direction, angles)):
            chosen = nearest == index
            result[chosen] = rank[chosen]
    return np.where(valid, result, np.nan)


def local_rank_bars(result, bar_rank, min_area_m2):
    """
    Bars at each window length: connected cells ranking above bar_rank, at
    least min_area_m2. Each is measured on the DEM, tested for standing above
    the ground around it, and linked to the bars of the next shorter window it
    contains and the bar of the next longer window that contains it.
    """
    from scipy import ndimage
    cell = result["cell_size"]
    surface, valid = result["surface"], result["valid"]
    minimum_cells = min_area_m2 / (cell * cell)
    threshold = int(round(bar_rank * RANK_STORE_MAX))
    rows, columns = surface.shape
    margin = ENCLOSURE_RING_CELLS + 1

    per_scale, label_images = [], []
    for scale_index, rank in enumerate(result["ranks"]):
        above = (rank > threshold) & valid
        _, labels = cv2.connectedComponents(above.astype(np.uint8), connectivity=8)
        kept = np.zeros_like(labels)
        bars = []
        for label, box in enumerate(ndimage.find_objects(labels), start=1):
            if box is None:
                continue
            top, left = max(box[0].start - margin, 0), max(box[1].start - margin, 0)
            window = (slice(top, min(box[0].stop + margin, rows)),
                      slice(left, min(box[1].stop + margin, columns)))
            mask = labels[window] == label
            area_cells = int(mask.sum())
            if area_cells < minimum_cells:
                continue
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE, offset=(left, top))
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            (_, _), (side_a, side_b), _ = cv2.minAreaRect(contour)
            length, width = max(side_a, side_b) * cell, min(side_a, side_b) * cell
            ys, xs = np.nonzero(mask)
            if xs.size > 1:
                eigenvalues, eigenvectors = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
                major = eigenvectors[:, int(np.argmax(eigenvalues))]
                bearing = (math.degrees(math.atan2(major[0], -major[1])) + 360.0) % 180.0
            else:
                bearing = float("nan")
            _, p_value, difference = _classify(surface, (window, mask),
                                               result["correlation_cells"])
            inside = surface[window][mask]
            kept[window][mask] = len(bars) + 1
            bars.append({"scale": scale_index, "area_m2": area_cells * cell * cell,
                         "length_m": length, "width_m": width,
                         "length_width": length / width if width else float("nan"),
                         "bearing_degrees": bearing,
                         "height_above_surroundings_m": difference,
                         "p_raised": p_value,
                         "top_height_m": float(np.percentile(inside, BAR_HEIGHT_HIGH_PERCENTILE)),
                         "outline": contour.reshape(-1, 2).tolist(),
                         "parts": [], "within": None})
        # Largest first, numbered within the window length.
        order = sorted(range(len(bars)), key=lambda index: -bars[index]["area_m2"])
        renumber = np.zeros(len(bars) + 1, np.int32)
        for new, old in enumerate(order, start=1):
            renumber[old + 1] = new
        bars = [bars[old] for old in order]
        for number, bar in enumerate(bars, start=1):
            bar["id"] = f"{scale_index + 1}.{number}"
        per_scale.append(bars)
        label_images.append(renumber[kept])

    # Nesting: each bar belongs to the next longer window's bar that covers most of it.
    for scale_index in range(len(per_scale) - 1):
        shorter, longer = per_scale[scale_index], per_scale[scale_index + 1]
        shorter_labels, longer_labels = label_images[scale_index], label_images[scale_index + 1]
        for number, bar in enumerate(shorter, start=1):
            covering = longer_labels[shorter_labels == number]
            covering = covering[covering > 0]
            if not covering.size:
                continue
            parent = int(np.bincount(covering).argmax())
            if np.count_nonzero(covering == parent) * 2 < np.count_nonzero(shorter_labels == number):
                continue
            bar["within"] = longer[parent - 1]["id"]
            longer[parent - 1]["parts"].append(bar["id"])
    return per_scale


# ======================================================================================================================
# Georeferencing
# ======================================================================================================================
def georeference(path):
    """
    (transform, crs, cell size in metres) for a GeoTIFF, or None without
    rasterio. The gage position and ground distances depend on it.
    """
    try:
        import rasterio
    except ImportError:
        return None
    with rasterio.open(path) as source:
        return source.transform, source.crs, float(abs(source.transform.a))


def lonlat_to_cell(transform, crs, longitude, latitude):
    """(row, column) of a longitude and latitude in the raster's own grid."""
    from rasterio.warp import transform as reproject
    from rasterio.transform import rowcol
    xs, ys = reproject("EPSG:4326", crs, [longitude], [latitude])
    row, column = rowcol(transform, xs[0], ys[0])
    return int(row), int(column)


# ======================================================================================================================
# Tab
# ======================================================================================================================
class HoverHelp(QObject):
    """
    Shows text beside a widget once the pointer has rested on it for a while,
    and hides it when the pointer leaves. Qt's own tooltip delay is global and
    short; this keeps a long explanation from appearing on every pass.
    """

    def __init__(self, widget, text, delay_ms=HOVER_HELP_DELAY_MS):
        super().__init__(widget)
        self._widget = widget
        self._text = text
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(int(delay_ms))
        self._timer.timeout.connect(self._show)
        widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self._widget:
            if event.type() == QEvent.Enter:
                self._timer.start()
            elif event.type() in (QEvent.Leave, QEvent.MouseButtonPress, QEvent.Hide):
                self._timer.stop()
                QToolTip.hideText()
        return False

    def _show(self):
        QToolTip.showText(QCursor.pos(), self._text, self._widget)


def _log_slider_value(position, value_range):
    """The value a log-scale slider position stands for."""
    low, high = value_range
    value = low * (high / low) ** (position / float(SCALE_SLIDER_STEPS))
    digits = SCALE_SLIDER_SIGNIFICANT_FIGURES - 1 - int(math.floor(math.log10(value)))
    return min(max(round(value, digits), low), high)


def _log_slider_position(value, value_range):
    """The log-scale slider position nearest a value."""
    low, high = value_range
    value = min(max(float(value), low), high)
    return int(round(SCALE_SLIDER_STEPS * math.log(value / low) / math.log(high / low)))


class _Worker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, job):
        super().__init__()
        self._job = job

    def run(self):
        try:
            self.finished.emit(self._job(), "")
        except Exception as err:
            import traceback
            traceback.print_exc()
            self.finished.emit(None, f"{type(err).__name__}: {err}")


class MorphologyTab(QWidget):
    """
    Pattern metrics, superimposed bedforms and inundation frequency for the
    elevation model loaded on the Elevation tab, over the same region.
    """

    def __init__(self, dem_tab, area_source, settings=None, parent=None):
        super().__init__(parent)
        self._dem_tab = dem_tab
        self._area_source = area_source
        self._settings = settings
        self._worker = None
        self._build_ui()
        self._bind_settings()
        self._apply_resolution()        # the restored choice takes effect straight away

    # ------------------------------------------------------------------------------------------------------------------
    def _build_ui(self):
        from downloader_viewers import ZoomableView

        def form(title):
            box = QGroupBox(title)
            layout = QFormLayout(box)
            layout.setHorizontalSpacing(6)
            layout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
            return box, layout

        # Shared settings. Sliders, so the analysis on screen can follow them.
        from PyQt5.QtWidgets import QSlider

        def scale_slider(value_range, default, text, tooltip):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, SCALE_SLIDER_STEPS)
            slider.setValue(_log_slider_position(default, value_range))
            slider.setToolTip(tooltip)
            label = QLabel("")
            label.setMinimumWidth(label.fontMetrics().horizontalAdvance(
                text(value_range[1])))
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            row_layout.addWidget(slider, 1)
            row_layout.addWidget(label)
            return slider, label, row

        self._cutoff_text = lambda value: f"{value:,g} m"
        self._min_bar_text = lambda value: f"{value:,.0f} m\u00b2"
        self._slider_cutoff, self._label_cutoff, cutoff_row = scale_slider(
            BEDFORM_CUTOFF_RANGE_M, DEFAULTS["bedform_cutoff_m"], self._cutoff_text,
            "Horizontal length, not height. Features shorter than this across are "
            "treated as bedforms riding on a bar; longer ones as part of the bar. "
            "Set it between the dune length and the bar size, typically 10 to 20 m.")
        self._slider_min_bar, self._label_min_bar, min_bar_row = scale_slider(
            MIN_BAR_AREA_RANGE_M2, DEFAULTS["min_bar_area_m2"], self._min_bar_text,
            "Patches above the bar level smaller than this are not counted as bars.")
        self._rerun_timer = QTimer(self)
        self._rerun_timer.setSingleShot(True)
        self._rerun_timer.setInterval(LIVE_RERUN_DELAY_MS)
        self._rerun_timer.timeout.connect(self._rerun_on_screen)
        self._rerun_pending = False    # a slider moved while a run was in progress
        self._keep_view = False        # a rerun from a slider keeps the zoom
        self._slider_cutoff.valueChanged.connect(self._scale_changed)
        self._slider_min_bar.valueChanged.connect(self._scale_changed)
        self._check_numbers = QCheckBox("Number the bars")
        self._check_numbers.setChecked(True)
        self._check_numbers.setToolTip("Label each outlined bar with its row in the table.")
        self._check_numbers.toggled.connect(self._redraw)
        self._last = None          # (kind, result) of the analysis on screen
        self._highlight = None     # bar number picked in the report
        shared_box, shared = form("Scales")
        shared.addRow("Max bedform length:", cutoff_row)
        shared.addRow("Min bar area:", min_bar_row)
        self._show_scale_values()
        shared.addRow(self._check_numbers)

        # Analysis resolution, shared with Compound Bars through the Elevation tab.
        from downloader_viewers import ANALYSIS_COARSENING, MAX_ANALYSIS_CELLS
        self._coarsening_choices = ANALYSIS_COARSENING
        self._combo_resolution = QComboBox()
        self._combo_resolution.addItems([label for label, _ in ANALYSIS_COARSENING])
        self._combo_resolution.setToolTip(
            "The cell size analyses work at, here and in Compound Bars. File resolution "
            "is the DEM's own cells; coarser averages blocks of them, which is faster on "
            "a large region and blurs features smaller than the new cell.")
        self._combo_resolution.currentIndexChanged.connect(self._apply_resolution)
        self._spin_max_cells = QDoubleSpinBox()
        self._spin_max_cells.setRange(0.5, 500.0)
        self._spin_max_cells.setDecimals(1)
        self._spin_max_cells.setSuffix(" million cells")
        self._spin_max_cells.setValue(MAX_ANALYSIS_CELLS / 1e6)
        self._spin_max_cells.setToolTip("Regions larger than this, at the chosen resolution, "
                                        "are refused rather than left running for minutes.")
        self._spin_max_cells.valueChanged.connect(self._apply_resolution)
        shared.addRow("Analysis resolution:", self._combo_resolution)
        shared.addRow("Largest region:", self._spin_max_cells)

        # Pattern
        self._spin_wave_min = QDoubleSpinBox()
        self._spin_wave_min.setRange(1.0, 1e5)
        self._spin_wave_min.setValue(DEFAULTS["wavelet_min_m"])
        self._spin_wave_min.setSuffix(" m")
        self._spin_wave_max = QDoubleSpinBox()
        self._spin_wave_max.setRange(10.0, 1e6)
        self._spin_wave_max.setValue(DEFAULTS["wavelet_max_m"])
        self._spin_wave_max.setSuffix(" m")
        button_pattern = QPushButton("Pattern Metrics")
        button_pattern.clicked.connect(self._run_pattern)
        pattern_box, pattern = form("Pattern")
        self._spin_wave_min.setToolTip("The dominant bar length is searched for between these "
                                       "two lengths. They do not filter which bars are found.")
        self._spin_wave_max.setToolTip(self._spin_wave_min.toolTip())
        pattern.addRow("Search lengths from:", self._spin_wave_min)
        pattern.addRow("to:", self._spin_wave_max)
        pattern.addRow(button_pattern)

        # Colour base for the bed-level map: heights at or below it take the
        # bottom colour, so the colours spread over what stands above it.
        from PyQt5.QtWidgets import QSlider
        self._slider_base = QSlider(Qt.Horizontal)
        self._slider_base.setRange(0, COLOR_BASE_STEPS)
        self._slider_base.setValue(0)
        self._slider_base.setToolTip("The bed level shown in the lowest colour. Raise it to "
                                     "spread the colours over the higher ground.")
        self._slider_base.valueChanged.connect(self._base_changed)
        self._label_base = QLabel("")
        self._label_base.setMinimumWidth(70)
        base_row = QWidget()
        base_layout = QHBoxLayout(base_row)
        base_layout.setContentsMargins(0, 0, 0, 0)
        base_layout.setSpacing(4)
        base_layout.addWidget(self._slider_base, 1)
        base_layout.addWidget(self._label_base)
        pattern.addRow("Colour base:", base_row)
        self._check_base_bars = QCheckBox("Outline ground above the base")
        self._check_base_bars.setToolTip(
            "Bars are the ground above the mean bed, as Schuurman et al. (2013) define "
            "them, which keeps a whole bar as one. Ticked, they are the ground above the "
            "colour base instead, which separates the higher platforms a bar is built "
            "from. The braiding index and bar height keep the mean either way.")
        self._check_base_bars.toggled.connect(self._reoutline_bars)
        self._slider_base.sliderReleased.connect(self._reoutline_bars)
        pattern.addRow(self._check_base_bars)
        self._check_split_bars = QCheckBox("Split bars along channels")
        self._check_split_bars.setToolTip(
            "Divide each bar along the low lines between its higher patches, one region "
            "per summit. A summit counts only when it stands above the saddle to its "
            "neighbour by more than this surface's roughness allows by chance, at the "
            "significance level set under Mounds and pockets.")
        self._check_split_bars.toggled.connect(self._reoutline_bars)
        pattern.addRow(self._check_split_bars)

        # Superimposed bedforms
        button_bedforms = QPushButton("Superimposed Bedforms")
        button_bedforms.clicked.connect(self._run_bedforms)
        bedform_box, bedform = form("Bedforms")
        bedform.addRow(QLabel(f"Flags bedforms taller than "
                              f"{SUPERIMPOSED_RATIO_THRESHOLD:.0%} of their bar."))
        bedform.addRow(button_bedforms)

        # Mounds and pockets
        self._spin_feature_min = QDoubleSpinBox()
        self._spin_feature_min.setRange(0.5, 1000.0)
        self._spin_feature_min.setValue(FEATURE_DEFAULTS["min_size_m"])
        self._spin_feature_min.setSuffix(" m")
        self._spin_feature_max = QDoubleSpinBox()
        self._spin_feature_max.setRange(1.0, 5000.0)
        self._spin_feature_max.setValue(FEATURE_DEFAULTS["max_size_m"])
        self._spin_feature_max.setSuffix(" m")
        for spin in (self._spin_feature_min, self._spin_feature_max):
            spin.setToolTip("Lengths searched for, along each feature's long axis. Features "
                            "narrower than three DEM cells are reported as below resolution.")
        self._spin_alpha = QDoubleSpinBox()
        self._spin_alpha.setRange(0.0001, 0.2)
        self._spin_alpha.setDecimals(4)
        self._spin_alpha.setSingleStep(0.005)
        self._spin_alpha.setValue(FEATURE_DEFAULTS["alpha"])
        self._spin_alpha.setToolTip(
            "Significance level. A feature is kept only when its margin is steeper than the "
            "ground on both sides of it at this level. It also sets how far a slope must "
            "stand out from the surface's own slopes to be traced as a margin.")
        self._check_fdr = QCheckBox("Correct for many candidates")
        self._check_fdr.setChecked(FEATURE_DEFAULTS["fdr_correction"])
        self._check_fdr.setToolTip(
            "Benjamini-Hochberg: keeps the expected share of false features at the "
            "significance level when hundreds of candidates are tested together.")
        self._spin_gap = QDoubleSpinBox()
        self._spin_gap.setRange(0.0, 50.0)
        self._spin_gap.setValue(FEATURE_DEFAULTS["gap_m"])
        self._spin_gap.setSuffix(" m")
        self._spin_gap.setToolTip("Breaks in a margin up to this length are bridged.")
        self._spin_ramp = QDoubleSpinBox()
        self._spin_ramp.setRange(0.0, 200.0)
        self._spin_ramp.setValue(FEATURE_DEFAULTS["ramp_gap_m"])
        self._spin_ramp.setSuffix(" m")
        self._spin_ramp.setToolTip("The widest open end closed across, as at the up-ramp end "
                                   "of a sloped mound, where there is no margin.")
        self._spin_freedom = QDoubleSpinBox()
        self._spin_freedom.setRange(0.01, 10.0)
        self._spin_freedom.setDecimals(2)
        self._spin_freedom.setSingleStep(0.1)
        self._spin_freedom.setValue(FEATURE_DEFAULTS["shape_freedom"])
        self._spin_freedom.setToolTip("How quickly the outline may bend away from the traced "
                                      "margin. Low keeps it smooth; high follows every "
                                      "irregularity.")
        self._check_mounds = QCheckBox("Mounds")
        self._check_mounds.setChecked(True)
        self._check_pockets = QCheckBox("Pockets")
        self._check_pockets.setChecked(True)
        self._combo_boundary = QComboBox()
        self._combo_boundary.addItems(["Margin (steepest)", "Toe (base)", "Crest (top)"])
        self._combo_boundary.setToolTip("Which edge of each feature's riser is drawn.")
        self._combo_boundary.currentIndexChanged.connect(self._redraw)
        self._check_band = QCheckBox("Uncertainty band")
        self._check_band.setChecked(True)
        self._check_band.setToolTip("Two standard deviations either side of the outline.")
        self._check_band.toggled.connect(self._redraw)
        button_features = QPushButton("Find Mounds and Pockets")
        button_features.clicked.connect(self._run_features)
        feature_box, feature = form("Mounds and pockets")
        feature.addRow("Size from:", self._spin_feature_min)
        feature.addRow("to:", self._spin_feature_max)
        feature.addRow("Significance:", self._spin_alpha)
        feature.addRow(self._check_fdr)
        feature.addRow("Bridge gaps to:", self._spin_gap)
        feature.addRow("Close ramps to:", self._spin_ramp)
        feature.addRow("Shape freedom:", self._spin_freedom)
        kinds = QWidget()
        kinds_row = QHBoxLayout(kinds)
        kinds_row.setContentsMargins(0, 0, 0, 0)
        kinds_row.addWidget(self._check_mounds)
        kinds_row.addWidget(self._check_pockets)
        feature.addRow("Find:", kinds)
        feature.addRow("Draw:", self._combo_boundary)
        feature.addRow(self._check_band)
        feature.addRow(button_features)

        # Local rank
        self._spin_rank_min = QDoubleSpinBox()
        self._spin_rank_min.setRange(1.0, 5000.0)
        self._spin_rank_min.setValue(LOCAL_RANK_DEFAULTS["window_min_m"])
        self._spin_rank_min.setSuffix(" m")
        self._spin_rank_min.setToolTip("Shortest window length along the flow.")
        self._spin_rank_max = QDoubleSpinBox()
        self._spin_rank_max.setRange(1.0, 5000.0)
        self._spin_rank_max.setValue(LOCAL_RANK_DEFAULTS["window_max_m"])
        self._spin_rank_max.setSuffix(" m")
        self._spin_rank_max.setToolTip("Longest window length along the flow.")
        self._spin_rank_count = QSpinBox()
        self._spin_rank_count.setRange(1, 20)
        self._spin_rank_count.setValue(LOCAL_RANK_DEFAULTS["window_count"])
        self._spin_rank_count.setToolTip("How many window lengths, evenly spaced in log "
                                         "between the shortest and longest.")
        self._spin_rank_width = QDoubleSpinBox()
        self._spin_rank_width.setRange(0.05, 1.0)
        self._spin_rank_width.setDecimals(2)
        self._spin_rank_width.setSingleStep(0.05)
        self._spin_rank_width.setValue(LOCAL_RANK_DEFAULTS["window_width_ratio"])
        self._spin_rank_width.setToolTip("Window width across the flow as a share of its "
                                         "length. 1 is a circle.")
        self._spin_rank_directions = QSpinBox()
        self._spin_rank_directions.setRange(1, 36)
        self._spin_rank_directions.setValue(LOCAL_RANK_DEFAULTS["orientation_steps"])
        self._spin_rank_directions.setToolTip(
            "Window directions tried over 180 degrees; each cell uses the one nearest the "
            "direction its topography runs. 1 uses the reach's downhill direction everywhere. "
            "Run time grows with this.")
        from PyQt5.QtWidgets import QSlider
        self._slider_bar_rank = QSlider(Qt.Horizontal)
        self._slider_bar_rank.setRange(0, SCALE_SLIDER_STEPS)
        self._slider_bar_rank.setValue(int(round(LOCAL_RANK_DEFAULTS["bar_rank"]
                                                 * SCALE_SLIDER_STEPS)))
        self._slider_bar_rank.setToolTip(
            "A bar cell is higher than this share of the cells in its window. The report "
            "gives the share noise alone exceeds, as a lower limit.")
        self._label_bar_rank = QLabel("")
        self._label_bar_rank.setMinimumWidth(
            self._label_bar_rank.fontMetrics().horizontalAdvance("100.0% of window"))
        bar_rank_row = QWidget()
        bar_rank_layout = QHBoxLayout(bar_rank_row)
        bar_rank_layout.setContentsMargins(0, 0, 0, 0)
        bar_rank_layout.setSpacing(4)
        bar_rank_layout.addWidget(self._slider_bar_rank, 1)
        bar_rank_layout.addWidget(self._label_bar_rank)
        self._slider_bar_rank.valueChanged.connect(self._bar_rank_changed)
        self._combo_rank_window = QComboBox()
        self._combo_rank_window.addItem("All")
        self._combo_rank_window.setToolTip("Which window length's rank and bars are shown.")
        self._combo_rank_window.currentIndexChanged.connect(self._rank_window_changed)
        button_rank = QPushButton("Find Bars by Local Rank")
        button_rank.clicked.connect(self._run_local_rank)
        rank_box, rank_form = form("Local rank")
        rank_form.addRow("Window length from:", self._spin_rank_min)
        rank_form.addRow("to:", self._spin_rank_max)
        rank_form.addRow("Window lengths:", self._spin_rank_count)
        rank_form.addRow("Window width/length:", self._spin_rank_width)
        rank_form.addRow("Window directions:", self._spin_rank_directions)
        rank_form.addRow("Bar cells higher than:", bar_rank_row)
        rank_form.addRow("Show window:", self._combo_rank_window)
        rank_form.addRow(button_rank)
        self._show_bar_rank()

        # Inundation
        today = QDate.currentDate()
        self._edit_gage = QLineEdit()
        self._edit_gage.setPlaceholderText("from the Download tab")
        self._date_start = QDateEdit(today.addYears(-1))
        self._date_end = QDateEdit(today)
        for edit in (self._date_start, self._date_end):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd")
        self._spin_slope = QDoubleSpinBox()
        self._spin_slope.setRange(0.0, 0.1)
        self._spin_slope.setDecimals(6)
        self._spin_slope.setSingleStep(0.0001)
        self._spin_slope.setSpecialValueText("From the DEM")
        self._spin_slope.setToolTip("Water-surface slope, metres per metre. Left at zero, "
                                    "the slope of the plane fitted to the DEM is used.")
        self._spin_shift = QDoubleSpinBox()
        self._spin_shift.setRange(-10.0, 10.0)
        self._spin_shift.setDecimals(3)
        self._spin_shift.setSuffix(" m")
        self._spin_shift.setToolTip("Added to the gage datum to put it on NAVD88. Filled in "
                                    "from the NWIS record when the conversion succeeds.")
        button_inundation = QPushButton("Inundation Frequency")
        button_inundation.clicked.connect(self._run_inundation)
        inundation_box, inundation = form("Inundation")
        inundation.addRow("Gage:", self._edit_gage)
        inundation.addRow("From:", self._date_start)
        inundation.addRow("To:", self._date_end)
        inundation.addRow("Slope:", self._spin_slope)
        inundation.addRow("Datum to NAVD88:", self._spin_shift)
        inundation.addRow(button_inundation)

        # Display: the analysis map drawn over the DEM itself, at an opacity.
        from PyQt5.QtWidgets import QSlider
        self._slider_opacity = QSlider(Qt.Horizontal)
        self._slider_opacity.setRange(0, 100)
        self._slider_opacity.setValue(100)
        self._slider_opacity.setToolTip("How strongly the analysis colours are drawn over the "
                                        "DEM. At zero only the DEM shows.")
        self._slider_opacity.valueChanged.connect(self._opacity_changed)
        self._label_opacity = QLabel("100%")
        self._label_opacity.setMinimumWidth(40)
        opacity_row = QWidget()
        opacity_layout = QHBoxLayout(opacity_row)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.setSpacing(4)
        opacity_layout.addWidget(self._slider_opacity, 1)
        opacity_layout.addWidget(self._label_opacity)
        self._combo_under = QComboBox()
        self._combo_under.addItems(DEM_LAYERS)
        self._combo_under.setToolTip("How the DEM under the colours is drawn.")
        self._combo_under.currentIndexChanged.connect(self._redraw_layers)
        # The sun for the hillshade: where it shines from and how high it is.
        from downloader_viewers import HILLSHADE_AZIMUTH, HILLSHADE_ALTITUDE
        self._spin_azimuth = QSpinBox()
        self._spin_azimuth.setRange(0, 359)
        self._spin_azimuth.setWrapping(True)
        self._spin_azimuth.setValue(int(HILLSHADE_AZIMUTH))
        self._spin_azimuth.setSuffix("\u00b0")
        self._spin_azimuth.setToolTip("Direction the light comes from, clockwise from the top "
                                      "of the image. Features parallel to the light fade; "
                                      "turn it to bring them out.")
        self._spin_sun_height = QSpinBox()
        self._spin_sun_height.setRange(1, 90)
        self._spin_sun_height.setValue(int(HILLSHADE_ALTITUDE))
        self._spin_sun_height.setSuffix("\u00b0")
        self._spin_sun_height.setToolTip("Height of the light above the horizon. Lower "
                                         "exaggerates low relief such as bar margins.")
        self._spin_exaggerate = QDoubleSpinBox()
        self._spin_exaggerate.setRange(1.0, 100.0)
        self._spin_exaggerate.setDecimals(1)
        self._spin_exaggerate.setSingleStep(1.0)
        self._spin_exaggerate.setValue(1.0)
        self._spin_exaggerate.setSuffix(" \u00d7")
        self._spin_exaggerate.setToolTip("Vertical exaggeration for the shading only. Bar relief "
                                         "of a few tenths of a metre is nearly invisible at "
                                         "true scale; the analysis is unaffected.")
        for spin in (self._spin_azimuth, self._spin_sun_height, self._spin_exaggerate):
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._redraw_layers)
        display_box, display = form("Display")
        display.addRow("Colour opacity:", opacity_row)
        display.addRow("DEM shown as:", self._combo_under)
        display.addRow("Sun from:", self._spin_azimuth)
        display.addRow("Sun height:", self._spin_sun_height)
        display.addRow("Exaggeration:", self._spin_exaggerate)
        self._dem_layer = None          # the drawn DEM for the last run, cached

        # The source and method for each analysis, after a pause on its heading.
        self._help = [HoverHelp(box, SUMMARIES[name])
                      for box, name in ((pattern_box, "Pattern"), (bedform_box, "Bedforms"),
                                        (feature_box, "Mounds and pockets"),
                                        (rank_box, "Local rank"),
                                        (inundation_box, "Inundation"))]

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        for box in (display_box, shared_box, pattern_box, bedform_box, feature_box, rank_box,
                    inundation_box):
            panel_layout.addWidget(box)
        panel_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(panel.sizeHint().width() + 20)

        self._view = ZoomableView()
        self._view.set_message("Run an analysis on the DEM open in the Elevation tab.")
        self._plot = QLabel("")
        self._plot.setAlignment(Qt.AlignCenter)
        self._plot.setMinimumHeight(180)
        self._plot.setVisible(False)           # shown by the analyses that draw a chart
        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        self._report.setToolTip("Click a bar's row to outline it on the map.")
        # Clicking a row puts the cursor on it, which is the signal used here.
        self._report.cursorPositionChanged.connect(self._pick_from_report)
        font = self._report.font()
        font.setFamily("Courier New")
        self._report.setFont(font)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._view)
        splitter.addWidget(self._plot)
        splitter.addWidget(self._report)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 2)

        layout = QHBoxLayout(self)
        layout.addWidget(scroll)
        layout.addWidget(splitter, 1)

    def _bind_settings(self):
        if self._settings is None or not hasattr(self._settings, "bind"):
            return
        for widget, key in ((self._slider_cutoff, "cutoff_slider"),
                            (self._slider_min_bar, "min_bar_slider"),
                            (self._spin_wave_min, "wave_min"), (self._spin_wave_max, "wave_max"),
                            (self._edit_gage, "gage"), (self._spin_slope, "slope"),
                            (self._spin_shift, "datum_shift"),
                            (self._check_numbers, "number_bars"),
                            (self._slider_opacity, "colour_opacity"),
                            (self._combo_under, "dem_layer"),
                            (self._spin_azimuth, "sun_azimuth"),
                            (self._spin_sun_height, "sun_height"),
                            (self._spin_exaggerate, "shade_exaggeration"),
                            (self._slider_base, "pattern_colour_base"),
                            (self._check_base_bars, "pattern_bars_above_base"),
                            (self._check_split_bars, "pattern_split_bars"),
                            (self._combo_resolution, "analysis_resolution"),
                            (self._spin_max_cells, "max_analysis_cells_millions"),
                            (self._spin_feature_min, "feature_min"),
                            (self._spin_feature_max, "feature_max"),
                            (self._spin_alpha, "feature_alpha"),
                            (self._check_fdr, "feature_fdr"),
                            (self._spin_gap, "feature_gap_m"),
                            (self._spin_ramp, "feature_ramp_m"),
                            (self._spin_freedom, "feature_freedom"),
                            (self._check_mounds, "find_mounds"),
                            (self._check_pockets, "find_pockets"),
                            (self._combo_boundary, "feature_boundary"),
                            (self._check_band, "feature_band"),
                            (self._spin_rank_min, "rank_window_min_m"),
                            (self._spin_rank_max, "rank_window_max_m"),
                            (self._spin_rank_count, "rank_window_count"),
                            (self._spin_rank_width, "rank_window_width_ratio"),
                            (self._spin_rank_directions, "rank_window_directions"),
                            (self._slider_bar_rank, "rank_bar_cells_higher_than")):
            self._settings.bind(widget, f"Morphology.{key}")

    # ------------------------------------------------------------------------------------------------------------------
    def _patch(self):
        """The Elevation tab's DEM over its region, and ground metres per cell."""
        dem = self._dem_tab
        if getattr(dem, "_elevation", None) is None:
            QMessageBox.information(self, "Morphology", "Open a DEM on the Elevation tab first.")
            return None, None, None
        # At the file's own resolution: the view is downscaled, and on a large
        # tile that makes a cell several metres across.
        full = dem.region_full_resolution()
        if full is None:
            QMessageBox.information(self, "Morphology",
                                    "This region is too large to analyse at full resolution. "
                                    "Draw a smaller region on the Elevation tab or zoom in.")
            return None, None, None
        patch, cell_size, (left, top, right, bottom), source = full
        geo = georeference(dem._path) if dem._path else None
        return patch, cell_size, (left, top, source, geo)

    def _settings_dict(self):
        return {"bedform_cutoff_m": self._cutoff_m(),
                "min_bar_area_m2": self._min_bar_m2(),
                "wavelet_min_m": self._spin_wave_min.value(),
                "wavelet_max_m": self._spin_wave_max.value(),
                "wavelet_scales": DEFAULTS["wavelet_scales"]}

    def _cutoff_m(self):
        return _log_slider_value(self._slider_cutoff.value(), BEDFORM_CUTOFF_RANGE_M)

    def _min_bar_m2(self):
        return _log_slider_value(self._slider_min_bar.value(), MIN_BAR_AREA_RANGE_M2)

    def _show_scale_values(self):
        self._label_cutoff.setText(self._cutoff_text(self._cutoff_m()))
        self._label_min_bar.setText(self._min_bar_text(self._min_bar_m2()))

    def _scale_changed(self, *_):
        """A scale slider moved: show its value, and rerun what uses it once it settles."""
        self._show_scale_values()
        if self._last is not None and self._last[0] in ("pattern", "bedforms", "local_rank"):
            self._rerun_timer.start()

    def _rerun_on_screen(self):
        """Run the analysis on screen again with the current scales, keeping the view."""
        if self._last is None or self._last[0] not in ("pattern", "bedforms", "local_rank"):
            return
        if self._worker is not None and self._worker.isRunning():
            self._rerun_pending = True
            return
        self._keep_view = True
        {"pattern": self._run_pattern, "bedforms": self._run_bedforms,
         "local_rank": self._find_local_rank_bars}[self._last[0]]()
        if self._worker is None or not self._worker.isRunning():
            self._keep_view = False    # nothing started, so the next run fits as usual

    def _worker_done(self, *_):
        """A slider that moved during the run gets its turn now."""
        if self._rerun_pending:
            self._rerun_pending = False
            self._rerun_timer.start()

    def _apply_resolution(self, *_):
        """Hand the analysis resolution to the Elevation tab, which reads the regions."""
        index = max(self._combo_resolution.currentIndex(), 0)
        self._dem_tab.analysis_coarsening = self._coarsening_choices[index][1]
        self._dem_tab.max_analysis_cells = self._spin_max_cells.value() * 1e6

    def _draw_inundation(self, result, fit=False):
        from downloader_viewers import to_pixmap
        days, total = result["days"], result["count"]
        valid = days >= 0
        fraction = np.where(valid, days / max(total, 1), 0.0)
        image = cv2.applyColorMap((fraction * 255).astype(np.uint8), cv2.COLORMAP_OCEAN)
        image[~valid] = 0
        image[(days == 0) & valid] = NEVER_WET_COLOR
        self._view.set_pixmap(to_pixmap(self._composite(image)), keep_view=not fit)
        self._view.clear_overlays()

    def _remember_dem(self, patch, cell_size):
        """The DEM the next result will be drawn over; its drawing is made on demand."""
        self._dem_patch = np.asarray(patch, np.float32)
        self._dem_cell = cell_size
        self._dem_layer = None

    def _dem_image(self):
        """The DEM as hillshade or grey elevation, cached for the run."""
        from downloader_viewers import stretch, _hillshade_image
        patch = getattr(self, "_dem_patch", None)
        if patch is None:
            return None
        kind = (self._combo_under.currentIndex(), self._spin_azimuth.value(),
                self._spin_sun_height.value(), self._spin_exaggerate.value())
        if self._dem_layer is not None and self._dem_layer[0] == kind:
            return self._dem_layer[1]
        values = patch.copy()
        values[~np.isfinite(values)] = np.nan
        filled = np.where(np.isfinite(values), values, np.nanmedian(values)).astype(np.float32)
        if kind[0] == 0:
            # In ground units, so the light falls at the true angle on the slopes.
            image = _hillshade_image(filled * float(kind[3])
                                     / max(getattr(self, "_dem_cell", 1.0), 1e-9),
                                     azimuth=float(kind[1]), altitude=float(kind[2]))
        else:
            image = cv2.cvtColor(stretch(values, 1, 99), cv2.COLOR_GRAY2BGR)
        image[~np.isfinite(values)] = 0
        self._dem_layer = (kind, image)
        return image

    def _composite(self, colours):
        """The analysis colours over the DEM, at the chosen opacity."""
        under = self._dem_image()
        opacity = self._slider_opacity.value() / 100.0
        if under is None or under.shape != colours.shape or opacity >= 1.0:
            return colours
        return cv2.addWeighted(colours, opacity, under, 1.0 - opacity, 0)

    def _opacity_changed(self, value):
        self._label_opacity.setText(f"{value}%")
        self._redraw_layers()

    def _redraw_layers(self, *_):
        """Redraw the current result with the new opacity or DEM layer, keeping the view."""
        if self._last is None:
            return
        kind, result = self._last
        if kind == "inundation":
            self._draw_inundation(result)
        else:
            self._redraw()

    def _redraw(self):
        if self._last is None:
            return
        kind, result = self._last
        if kind == "pattern":
            self._draw_pattern(result)
        elif kind == "bedforms":
            self._draw_bedforms(result)
        elif kind == "features":
            self._draw_features(result)
        elif kind == "local_rank":
            self._draw_local_rank(result)

    def _listed(self):
        """The numbered items of the analysis on screen, in report order."""
        if self._last is None:
            return []
        kind, result = self._last
        if kind in ("pattern", "bedforms"):
            return result["bars"]
        if kind == "features":
            return self._feature_outlines(result)
        if kind == "local_rank":
            return self._shown_rank_bars(result)
        return []

    def _pick_from_report(self):
        """A click on a table row picks that bar: it is outlined and centred."""
        if self._last is None or self._last[0] not in ("pattern", "bedforms", "features",
                                                       "local_rank"):
            return
        line = self._report.textCursor().block().text().strip()
        first = line.split(maxsplit=1)[0] if line else ""
        if not first.isdigit():
            return
        index = int(first)
        bars = self._listed()
        if not 1 <= index <= len(bars) or index == self._highlight:
            return
        self._highlight = index
        self._redraw()
        # Zoom to the picked item, so a feature a few metres across is readable.
        self._view.zoom_to(bars[index - 1]["outline"], PICK_ZOOM_FILL)

    def _overlay(self, bars, color_for, bands=None):
        """
        Outlines and numbers as overlays on the view: one screen pixel wide and a
        fixed text size at any zoom, so small features stay visible when the
        whole region is on screen. The picked item is drawn thicker, in yellow.
        """
        view = self._view
        view.clear_overlays()
        to_rgb = lambda bgr: (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        if bands:
            for points in bands:
                view.add_outline(points, to_rgb(BAND_COLOR), 1.0, dashed=True, z=1.5)
        numbered = self._check_numbers.isChecked()
        for index, bar in enumerate(bars, start=1):
            points = bar["outline"]
            if len(points) < 2:
                continue
            picked = index == self._highlight
            color = HIGHLIGHT_COLOR if picked else color_for(bar)
            view.add_outline(points, to_rgb(color), HIGHLIGHT_WIDTH if picked else 1.0,
                             z=2.5 if picked else 2.0)
            if numbered or picked:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                view.add_label(max(xs) + 1, min(ys) - 1, index, to_rgb(color))

    def _draw_outlines(self, image, bars, color_for):
        """Each bar's outline, numbered to match its row in the report."""
        numbered = self._check_numbers.isChecked()
        for index, bar in enumerate(bars, start=1):
            outline = np.asarray(bar["outline"], np.int32)
            if not outline.size:
                continue
            color = color_for(bar)
            cv2.polylines(image, [outline], True, color, 1, cv2.LINE_AA)
        # The picked bar last, so it sits on top of its neighbours.
        if self._highlight and 1 <= self._highlight <= len(bars):
            outline = np.asarray(bars[self._highlight - 1]["outline"], np.int32)
            if outline.size:
                cv2.polylines(image, [outline], True, HIGHLIGHT_COLOR, HIGHLIGHT_WIDTH,
                              cv2.LINE_AA)
        for index, bar in enumerate(bars, start=1):
            outline = np.asarray(bar["outline"], np.int32)
            if not outline.size:
                continue
            color = HIGHLIGHT_COLOR if index == self._highlight else color_for(bar)
            if numbered:
                # Inside the bar, at the point farthest from its edge: a
                # centroid can fall outside an irregular or curved outline.
                mask = np.zeros(image.shape[:2], np.uint8)
                cv2.fillPoly(mask, [outline], 1)
                distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
                y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
                x, y = int(x), int(y)
                label = str(index)
                # A dark halo under the number keeps it readable on any colour.
                cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1, cv2.LINE_AA)
        return image

    def _start(self, job, on_done, message):
        if self._worker is not None and self._worker.isRunning():
            return
        self._report.setPlainText(message)
        self._worker = _Worker(job)
        self._worker.finished.connect(on_done)
        self._worker.finished.connect(self._worker_done)
        self._worker.start()

    # ------------------------------------------------------------------------------------------------------------------
    def _run_pattern(self):
        patch, cell_size, context = self._patch()
        self._remember_dem(patch, cell_size)
        if patch is None:
            return
        settings = self._settings_dict()
        self._last_context = context
        self._start(lambda: pattern_metrics(patch, cell_size, settings),
                    lambda result, error: self._after_pattern(result, error, context),
                    "Computing pattern metrics\u2026")

    def _after_pattern(self, result, error, context):
        """A fresh run outlines at the chosen level straight away, if one is chosen."""
        if not error and result and not result.get("error") and (
                self._check_base_bars.isChecked() or self._check_split_bars.isChecked()):
            self._last = ("pattern", result)
            self._reoutline_bars()
            return
        self._show_pattern(result, error, context)

    def _show_pattern(self, result, error, context):
        fit, self._keep_view = not self._keep_view, False
        if error or result.get("error"):
            self._report.setPlainText(error or result["error"])
            return
        from downloader_viewers import stretch, to_pixmap, plot_histogram
        bars = result["bars"]
        aspects = [bar["aspect"] for bar in bars]
        perimeters = [bar["perimeter_area"] for bar in bars]
        lines = [
            f"Region: {context[2]}; {result['cell_size']:,.2f} m per cell.",
            ("Bars here are the ground above the mean bed, after smoothing at the max "
             "bedform length" if result.get("bar_level") is None else
             f"Bars here are the ground above the colour base, {result['bar_level']:+.2f} m "
             "from the mean bed, after smoothing at the max bedform length")
            + ("; each is split along its channels, one region per significant summit."
               if result.get("bars_split") else "."),
            f"Reach bar relief (top 5% minus bottom 5% of the bed, reach slope removed): "
            f"{result['bar_height']:,.2f} m",
            f"Total braiding index: {result['braiding_index']:,.2f} channels per cross section "
            f"({len(result['braiding_per_section'])} sections)",
            f"Dominant bar length: {result['dominant_bar_length']:,.0f} m",
            f"Bars found: {len(bars)}",
        ]
        if bars:
            lines.append(f"Aspect ratio (width/length): median {np.median(aspects):.2f}, "
                         f"range {min(aspects):.2f} to {max(aspects):.2f}")
            lines.append(f"Perimeter/area: median {np.median(perimeters):.4f} per m")
            lines.append("")
            lines.append("The numbers on the map match the bar column, largest first.")
            lines.append("bar   length (m)   width (m)   aspect   perimeter/area   area (m2)")
            for index, bar in enumerate(bars, start=1):
                lines.append(f"{index:>3}   {bar['length_m']:>10,.0f}   {bar['width_m']:>9,.0f}   "
                             f"{bar['aspect']:>6.2f}   {bar['perimeter_area']:>14.4f}   "
                             f"{bar['area_m2']:>9,.0f}")
        self._highlight = None                 # a new run renumbers the bars
        self._report.setPlainText("\n".join(lines))
        self._last = ("pattern", result)
        self._draw_pattern(result, fit=fit)
        counts, edges = result["histogram"]
        values = np.repeat((edges[:-1] + edges[1:]) / 2.0, counts)
        self._plot.setVisible(True)
        self._plot.setPixmap(to_pixmap(plot_histogram(values, "Detrended bed level (m)")).scaled(
            self._plot.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _base_value(self, values):
        """The bed level the slider points at, between the lowest and highest present."""
        finite = values[np.isfinite(values)]
        if not finite.size:
            return 0.0, 0.0, 1.0
        low, high = float(np.percentile(finite, 0.5)), float(np.percentile(finite, 99.5))
        fraction = self._slider_base.value() / float(COLOR_BASE_STEPS)
        return low + fraction * (high - low), low, high

    def _base_changed(self, *_):
        if self._last is not None and self._last[0] == "pattern":
            self._draw_pattern(self._last[1])
            # The outlines follow too, but only once the slider is let go or
            # stepped with the keys, not on every pixel of a drag.
            if self._check_base_bars.isChecked() and not self._slider_base.isSliderDown():
                self._reoutline_bars()

    def _reoutline_bars(self, *_):
        """Outline the bars again at the chosen level, keeping the rest of the run."""
        if self._last is None or self._last[0] != "pattern":
            return
        result = self._last[1]
        level = self._base_value(result["detrended"])[0] if self._check_base_bars.isChecked() \
            else None
        split = self._check_split_bars.isChecked()
        result["bars"] = _bar_shapes(result["detrended"], result["cell_size"],
                                     self._settings_dict(), level, split,
                                     self._spin_alpha.value())
        result["bar_level"] = level
        result["bars_split"] = split
        self._show_pattern(result, "", self._last_context)

    def _draw_pattern(self, result, fit=False):
        from downloader_viewers import to_pixmap
        values = result["detrended"]
        base, low, high = self._base_value(values)
        self._label_base.setText(f"{base:+.2f} m")
        # Everything at or below the base takes the lowest colour; the full
        # colour range spans the base to the top of the bed.
        span = max(high - base, 1e-6)
        scaled = np.clip((np.nan_to_num(values, nan=base) - base) / span, 0.0, 1.0)
        background = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        background[~np.isfinite(values)] = 0
        self._view.set_pixmap(to_pixmap(self._composite(background)), keep_view=not fit)
        self._overlay(result["bars"], lambda bar: (255, 255, 255))

    # ------------------------------------------------------------------------------------------------------------------
    def _run_bedforms(self):
        patch, cell_size, context = self._patch()
        self._remember_dem(patch, cell_size)
        if patch is None:
            return
        settings = self._settings_dict()
        self._start(lambda: superimposed_bedforms(patch, cell_size, settings),
                    lambda result, error: self._show_bedforms(result, error),
                    "Measuring superimposed bedforms\u2026")

    def _show_bedforms(self, result, error):
        fit, self._keep_view = not self._keep_view, False
        if error or result.get("error"):
            self._report.setPlainText(error or result["error"])
            return
        from downloader_viewers import stretch, to_pixmap
        bars = result["bars"]
        flagged = [bar for bar in bars if bar["above_threshold"]]
        lines = [f"Max bedform length {result['cutoff_m']:,.0f} m. {len(bars)} bar(s); "
                 f"{len(flagged)} carry bedforms taller than "
                 f"{SUPERIMPOSED_RATIO_THRESHOLD:.0%} of the bar.",
                 "Bars here are the ground above the mean bed, after smoothing at the max "
                 "bedform length. Bar height is each bar's own relief.",
                 "",
                 "bar   bar height (m)   bedform height (m)   ratio   above 25%"]
        for index, bar in enumerate(bars, start=1):
            lines.append(f"{index:>3}   {bar['host_height']:>14.2f}   "
                         f"{bar['bedform_height']:>18.2f}   {bar['ratio']:>5.2f}   "
                         f"{'yes' if bar['above_threshold'] else 'no'}")
        self._highlight = None
        self._report.setPlainText("\n".join(lines))
        self._last = ("bedforms", result)
        self._draw_bedforms(result, fit=fit)
        # No chart for this analysis, so the pane goes rather than sitting empty.
        self._plot.clear()
        self._plot.setVisible(False)

    def _draw_bedforms(self, result, fit=False):
        from downloader_viewers import stretch, to_pixmap
        image = cv2.applyColorMap(stretch(result["residual"], 1, 99), cv2.COLORMAP_VIRIDIS)
        self._view.set_pixmap(to_pixmap(self._composite(image)), keep_view=not fit)
        self._overlay(result["bars"],
                      lambda bar: (0, 0, 255) if bar["above_threshold"] else (255, 255, 255))

    # ------------------------------------------------------------------------------------------------------------------
    def _run_features(self):
        patch, cell_size, context = self._patch()
        self._remember_dem(patch, cell_size)
        if patch is None:
            return
        if not (self._check_mounds.isChecked() or self._check_pockets.isChecked()):
            QMessageBox.information(self, "Mounds and pockets", "Tick Mounds, Pockets, or both.")
            return
        settings = {"min_size_m": self._spin_feature_min.value(),
                    "max_size_m": max(self._spin_feature_max.value(),
                                      self._spin_feature_min.value() * 1.5),
                    "alpha": self._spin_alpha.value(),
                    "fdr_correction": self._check_fdr.isChecked(),
                    "shape_freedom": self._spin_freedom.value(),
                    "gap_m": self._spin_gap.value(),
                    "ramp_gap_m": self._spin_ramp.value()}
        mounds, pockets = self._check_mounds.isChecked(), self._check_pockets.isChecked()
        self._start(lambda: find_features(patch, cell_size, settings, mounds, pockets),
                    lambda result, error: self._show_features(result, error, context),
                    "Finding mounds and pockets\u2026")

    def _show_features(self, result, error, context):
        if error or result.get("error"):
            self._report.setPlainText(error or result["error"])
            return
        features = result["features"]
        mounds = sum(1 for f in features if f["kind"] == "mound")
        lines = [
            f"Region: {context[2]}; {result['cell_size']:,.2f} m per cell; vertical noise about "
            f"{result['noise_m'] * 100:,.1f} cm.",
            f"{len(features)} significant feature(s): {mounds} mound(s), "
            f"{len(features) - mounds} pocket(s). {result.get('not_significant', 0)} candidates "
            f"had no significant margin, {result['below_resolution']} were too narrow to "
            f"resolve, {result['unconfirmed']} were not confirmed once tracked, "
            f"{result.get('outside_range', 0)} were outside the size range.",
            f"Significance level {result['settings']['alpha']:g}"
            + (", Benjamini-Hochberg corrected" if result['settings']['fdr_correction'] else "")
            + f". The surface's fine detail stays correlated over about "
            f"{result.get('correlation_m', 0):.1f} m, so samples closer than that are not "
            "counted as independent evidence.",
            "Click a row to zoom to that feature.",
            "Length and width are along and across the long axis; orientation is the long "
            "axis's bearing (0 = up the image). The centre's error ellipse containing it with "
            "50% and 95% probability is given by how far it reaches along and across the long "
            "axis (along50, across50, along95, across95); tilt is the angle between the "
            "ellipse's own long axis and the feature's. Area is given with one standard "
            "deviation.",
            "Lee is the direction the steepest side faces, ramp the direction the gentlest "
            f"faces, shown when they differ by at least {result['settings']['asymmetry_min']:.1f}:1.",
            "",
            "p margin: chance of a margin this much steeper than its surroundings without one "
            "being there. p height: the same for the interior standing above (or below) its "
            "ring, reported only. q: p margin after the correction for many candidates.",
            "",
            "  #  kind    length  width  elong  orient  area (m\u00b2)      along50  across50  along95  across95  tilt  "
            "riser   lee   ramp   asym  p margin  p height        q",
        ]
        for feature in features:
            def bearing(value):
                return "-" if not np.isfinite(value) else f"{value:.0f}\u00b0"
            lines.append(
                f"{feature['index']:>3}  {feature['kind']:<6}  {feature['length_m']:>6.1f}  "
                f"{feature['width_m']:>5.1f}  {feature['elongation']:>5.1f}  "
                f"{feature['orientation_degrees']:>5.0f}\u00b0  "
                f"{feature['area_m2']:>7,.0f} \u00b1 {feature['area_sigma_m2']:<5,.0f}  "
                f"{feature['centre_along50_m']:>7.2f}  {feature['centre_across50_m']:>8.2f}  "
                f"{feature['centre_along95_m']:>7.2f}  {feature['centre_across95_m']:>8.2f}  "
                f"{bearing(feature['centre_error_tilt_degrees']):>4}  "
                f"{feature['riser_width_m']:>5.1f}  {bearing(feature['lee_bearing']):>5}  "
                f"{bearing(feature['ramp_bearing']):>5}  {feature['asymmetry']:>5.1f}  "
                f"{feature['p_margin']:>8.1e}  {feature['p_height']:>8.1e}  "
                f"{feature.get('q', feature['p_margin']):>7.1e}")
        self._highlight = None
        self._report.setPlainText("\n".join(lines))
        self._last = ("features", result)
        self._draw_features(result, fit=True)
        self._plot.clear()
        self._plot.setVisible(False)

    def _feature_outlines(self, result):
        """The features with the chosen edge as their outline, for drawing and picking."""
        key = ("outline", "outline_toe", "outline_crest")[self._combo_boundary.currentIndex()]
        return [{**feature, "outline": feature[key]} for feature in result["features"]]

    def _draw_features(self, result, fit=False):
        from downloader_viewers import stretch, to_pixmap
        surface = result["surface"]
        dy, dx = np.gradient(surface)
        slope = np.degrees(np.arctan(np.hypot(dx, dy) / result["cell_size"]))
        image = cv2.applyColorMap(stretch(slope, 1, 99), cv2.COLORMAP_INFERNO)
        features = self._feature_outlines(result)
        self._view.set_pixmap(to_pixmap(self._composite(image)), keep_view=not fit)
        bands = None
        if self._check_band.isChecked():
            bands = [feature[key] for feature in features for key in ("band_outer", "band_inner")]
        self._overlay(features, lambda f: MOUND_COLOR if f["kind"] == "mound" else POCKET_COLOR,
                      bands)

    # ------------------------------------------------------------------------------------------------------------------
    def _local_rank_settings(self):
        return {**LOCAL_RANK_DEFAULTS,
                "window_min_m": min(self._spin_rank_min.value(), self._spin_rank_max.value()),
                "window_max_m": max(self._spin_rank_min.value(), self._spin_rank_max.value()),
                "window_count": self._spin_rank_count.value(),
                "window_width_ratio": self._spin_rank_width.value(),
                "orientation_steps": self._spin_rank_directions.value()}

    def _bar_rank(self):
        return self._slider_bar_rank.value() / float(SCALE_SLIDER_STEPS)

    def _show_bar_rank(self):
        self._label_bar_rank.setText(f"{self._bar_rank():.1%} of window")

    def _bar_rank_changed(self, *_):
        """The bar rank moved: show it, and find the bars again once it settles."""
        self._show_bar_rank()
        if self._last is not None and self._last[0] == "local_rank":
            self._rerun_timer.start()

    def _run_local_rank(self):
        patch, cell_size, context = self._patch()
        self._remember_dem(patch, cell_size)
        if patch is None:
            return
        settings = self._local_rank_settings()
        bar_rank, min_area = self._bar_rank(), self._min_bar_m2()

        def job():
            result = local_rank_surfaces(patch, cell_size, settings)
            if not result.get("error"):
                result["bars"] = local_rank_bars(result, bar_rank, min_area)
                result["bar_rank"], result["min_area_m2"] = bar_rank, min_area
            return result

        self._last_context = context
        self._start(job, lambda result, error: self._show_local_rank(result, error, context),
                    "Ranking every cell in its window at each length\u2026")

    def _find_local_rank_bars(self):
        """The bars again from the ranks already computed, at the current bar rank and area."""
        result = self._last[1]
        bar_rank, min_area = self._bar_rank(), self._min_bar_m2()

        def job():
            result["bars"] = local_rank_bars(result, bar_rank, min_area)
            result["bar_rank"], result["min_area_m2"] = bar_rank, min_area
            return result

        context = self._last_context
        self._start(job, lambda done, error: self._show_local_rank(done, error, context),
                    "Finding bars on the ranks\u2026")

    def _rank_window_changed(self, *_):
        if self._last is not None and self._last[0] == "local_rank":
            self._highlight = None
            self._report_local_rank(self._last[1], self._last_context)
            self._draw_local_rank(self._last[1])

    def _shown_rank_scales(self, result):
        """Indices of the window lengths shown: one, or all of them."""
        chosen = self._combo_rank_window.currentIndex() - 1
        count = len(result["scales"])
        return [chosen] if 0 <= chosen < count else list(range(count))

    def _shown_rank_bars(self, result):
        return [bar for index in self._shown_rank_scales(result) for bar in result["bars"][index]]

    def _show_local_rank(self, result, error, context):
        fit, self._keep_view = not self._keep_view, False
        if error or result.get("error"):
            self._report.setPlainText(error or result["error"])
            return
        # The window list follows the run, keeping the length shown if it is still there.
        previous = self._combo_rank_window.currentText()
        labels = ["All"] + [f"{scale['length_m']:,.0f} m" for scale in result["scales"]]
        self._combo_rank_window.blockSignals(True)
        self._combo_rank_window.clear()
        self._combo_rank_window.addItems(labels)
        self._combo_rank_window.setCurrentIndex(labels.index(previous) if previous in labels else 0)
        self._combo_rank_window.blockSignals(False)
        self._highlight = None
        self._last = ("local_rank", result)
        self._report_local_rank(result, context)
        self._draw_local_rank(result, fit=fit)
        self._plot.clear()
        self._plot.setVisible(False)

    def _report_local_rank(self, result, context):
        bars = result["bars"]
        settings = result["settings"]
        lines = [
            f"Drawn region: {context[2]}; {result['cell_size']:,.2f} m per cell. "
            f"Ranked in {result['seconds']:,.0f} s.",
            f"Vertical noise {result['noise_m'] * 100:.2f} cm; heights within "
            f"{result['tie_m'] * 100:.2f} cm of each other count as tied "
            f"({settings['tie_alpha']:.0%} two-sided), in {result['bins']} height bins of "
            f"{result['bin_width_m'] * 100:.2f} cm.",
            f"Bar cells are higher than {result['bar_rank']:.1%} of their window; noise alone "
            f"exceeds {result['noise_rank']:.1%} with {settings['tie_alpha']:.0%} chance. "
            f"Smallest bar area {result['min_area_m2']:,.0f} m\u00b2.",
            "",
            "window   length x width (m)   coarsened   no clear direction   bars   compound",
        ]
        for index, scale in enumerate(result["scales"]):
            compound = sum(1 for bar in bars[index] if len(bar["parts"]) >= 2)
            lines.append(f"{index + 1:>6}   {scale['length_m']:>8,.0f} x {scale['width_m']:<8,.0f}"
                         f"   {scale['coarsening']:>6} x   {scale['unclear_direction_share']:>17.0%}"
                         f"   {len(bars[index]):>4}   {compound:>8}")
        lines += [
            "",
            "id is window.bar. A bar is compound when it holds two or more bars of the next "
            "shorter window (parts); within is the bar of the next longer window that holds it.",
            "Height above surroundings is the bar's mean minus the mean of a ring "
            f"{ENCLOSURE_RING_CELLS} cells wide around it, on the DEM with the reach slope "
            "removed; p raised is the one-sided test of that difference, allowing for the "
            "DEM's spatial correlation. Click a row to zoom to that bar.",
            "",
            "  #  id        area (m\u00b2)  length  width  length/width  long axis bearing  "
            "height above surroundings (m)  p raised  parts  within",
        ]
        for number, bar in enumerate(self._shown_rank_bars(result), start=1):
            bearing = bar["bearing_degrees"]
            bearing_text = f"{bearing:.0f}\u00b0" if np.isfinite(bearing) else "-"
            within_text = bar["within"] or "-"
            lines.append(
                f"{number:>3}  {bar['id']:<8}  {bar['area_m2']:>10,.0f}  {bar['length_m']:>6,.0f}  "
                f"{bar['width_m']:>5,.0f}  {bar['length_width']:>12.1f}  "
                f"{bearing_text:>17}  "
                f"{bar['height_above_surroundings_m']:>29.2f}  {bar['p_raised']:>8.1e}  "
                f"{len(bar['parts']):>5}  {within_text:>6}")
        self._report.setPlainText("\n".join(lines))

    def _rank_scale_color(self, index, count):
        """BGR colour for one window length, shortest to longest along a colour map."""
        position = int(round(255 * index / max(count - 1, 1)))
        return tuple(int(v) for v in cv2.applyColorMap(
            np.array([[position]], np.uint8), cv2.COLORMAP_SPRING)[0, 0])

    def _draw_local_rank(self, result, fit=False):
        from downloader_viewers import to_pixmap
        shown = self._shown_rank_scales(result)
        count = len(result["scales"])
        if len(shown) == 1:
            rank = result["ranks"][shown[0]].astype(np.float32) / RANK_STORE_MAX
            image = cv2.applyColorMap((rank * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            image[~result["valid"]] = 0
            image = self._composite(image)
        else:
            # Several lengths at once: their ranks would hide each other, so the DEM alone.
            image = self._dem_image()
            if image is None:
                image = np.zeros(result["valid"].shape + (3,), np.uint8)
        self._view.set_pixmap(to_pixmap(image), keep_view=not fit)
        self._overlay(self._shown_rank_bars(result),
                      lambda bar: self._rank_scale_color(bar["scale"], count))

    def _run_inundation(self):
        patch, cell_size, context = self._patch()
        self._remember_dem(patch, cell_size)
        if patch is None:
            return
        left, top, _, geo = context
        if geo is None:
            QMessageBox.warning(self, "Inundation", "The gage's position in the DEM needs the "
                                                    "file's georeferencing: pip install rasterio.")
            return
        site = self._edit_gage.text().strip() or self._area_source._edit_gage.text().strip()
        if not site:
            QMessageBox.warning(self, "Inundation", "Enter the NWIS gage number.")
            return
        start = self._date_start.date().toPyDate()
        end = self._date_end.date().toPyDate()
        slope_override = self._spin_slope.value() or None
        manual_shift = self._spin_shift.value()
        transform, crs, _ = geo

        def job():
            import tnm_download
            record = tnm_download.gage_record(site)
            latitude, longitude = float(record["dec_lat_va"]), float(record["dec_long_va"])
            datum_ft = float(record["alt_va"])
            shift, note = datum_shift_to_navd88(latitude, longitude,
                                                record.get("alt_datum_cd", ""))
            if shift is None:
                shift = manual_shift
                note += f"; using the entered shift of {manual_shift:+.3f} m"
            readings = daily_gage_height(site, start, end)
            if not readings:
                raise ValueError("NWIS returned no daily gage height for these dates.")
            surfaces = [(datum_ft + feet) * FEET_TO_METRES + shift for _, feet in readings]

            row, column = lonlat_to_cell(transform, crs, longitude, latitude)
            gage_cell = (row - top, column - left)            # both in file cells
            values = clean(patch)
            plane = fit_plane(values)
            days, slope = inundation_days(values, gage_cell, cell_size, plane, surfaces,
                                          slope_override)
            return {"days": days, "count": len(readings), "slope": slope, "note": note,
                    "shift": shift, "datum_ft": datum_ft, "gage_cell": gage_cell,
                    "station": record.get("station_nm", site),
                    "surfaces": surfaces, "readings": readings}

        self._start(job, self._show_inundation, "Fetching the gage record and daily stage\u2026")

    def _show_inundation(self, result, error):
        if error:
            self._report.setPlainText(error)
            return
        from downloader_viewers import to_pixmap
        days, total = result["days"], result["count"]
        valid = days >= 0
        always = int(((days == total) & valid).sum())
        never = int(((days == 0) & valid).sum())
        sometimes = int(((days > 0) & (days < total) & valid).sum())
        cells = max(int(valid.sum()), 1)
        lines = [
            f"{result['station']}: {total} day(s) of stage.",
            f"Gage datum {result['datum_ft']:,.2f} ft; {result['note']}.",
            f"Water-surface slope used: {result['slope']:.6f} m/m.",
            f"Gage at row {result['gage_cell'][0]:,.0f}, column {result['gage_cell'][1]:,.0f} "
            "of the region" + ("" if 0 <= result['gage_cell'][0] < days.shape[0]
                                and 0 <= result['gage_cell'][1] < days.shape[1]
                                else " (outside it; the water surface is extrapolated)") + ".",
            "",
            f"Always wet:    {always / cells:6.1%} of cells",
            f"Sometimes wet: {sometimes / cells:6.1%} of cells",
            f"Never wet:     {never / cells:6.1%} of cells",
        ]
        self._report.setPlainText("\n".join(lines))
        self._last = ("inundation", result)

        self._draw_inundation(result, fit=True)

        from downloader_viewers import plot_profile
        self._plot.setVisible(True)
        self._plot.setPixmap(to_pixmap(plot_profile(
            np.asarray(result["surfaces"], np.float32),
            "Water surface at the gage (m, NAVD88)", "day", "elevation")).scaled(
            self._plot.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
