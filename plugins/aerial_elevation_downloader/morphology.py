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
        "Each outline is then refined by an extended Kalman filter walking along the "
        "margin by arc length, which weighs each measurement by how sharp the margin "
        "is and carries the outline across faint or missing stretches. The band is two "
        "standard deviations either side, and the CEP50 and CEP95 come from the "
        "centroid's propagated uncertainty. Lee and ramp are the directions the "
        "steepest and gentlest sides face."),
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

    counts = []
    channels = np.zeros(aligned.shape, bool)
    for column in range(aligned.shape[1]):
        section = smooth[:, column]
        valid = np.isfinite(section)
        if valid.sum() < 3 * minimum_run:
            continue
        below = np.zeros(section.shape, bool)
        below[valid] = section[valid] < np.nanmean(section[valid])
        channels[:, column] = below
        # A channel is a run of below-mean cells at least one cutoff wide.
        runs, length = 0, 0
        for is_below in below:
            if is_below:
                length += 1
            else:
                runs += length >= minimum_run
                length = 0
        runs += length >= minimum_run
        counts.append(runs)
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

    power = np.zeros(scales_count)
    rows_used = 0
    for row in aligned:
        valid = np.isfinite(row)
        if valid.sum() < 16:
            continue
        signal = row[valid] - np.mean(row[valid])
        for index, scale in enumerate(scales):
            half = int(min(5 * scale, len(signal) // 2))
            if half < 2:
                continue
            x = np.arange(-half, half + 1) / scale
            kernel = (1.0 - x * x) * np.exp(-x * x / 2.0) / math.sqrt(scale)
            response = np.convolve(signal, kernel, mode="same")
            power[index] += float(np.mean(response * response))
        rows_used += 1

    if rows_used == 0 or not power.any():
        return float("nan"), [], []
    lengths = scales * to_wavelength * cell_size
    density = power / power.sum()
    return float(lengths[int(np.argmax(density))]), lengths.tolist(), density.tolist()


def _bar_shapes(detrended, cell_size, settings):
    """Bars as the ground above the mean, smoothed at bedform scale: length,
    width, aspect ratio and perimeter-to-area ratio of each."""
    filled = np.where(np.isfinite(detrended), detrended, np.nanmean(detrended)).astype(np.float32)
    cutoff = max(int(round(settings["bedform_cutoff_m"] / cell_size)) | 1, 3)
    smooth = cv2.GaussianBlur(filled, (0, 0), cutoff / 3.0)
    above = ((smooth > np.nanmean(smooth)) & np.isfinite(detrended)).astype(np.uint8)
    minimum_cells = settings["min_bar_area_m2"] / (cell_size * cell_size)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(above, connectivity=8)
    bars = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < minimum_cells:
            continue
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(contour)
        length, width = max(side_a, side_b) * cell_size, min(side_a, side_b) * cell_size
        area = float(stats[label, cv2.CC_STAT_AREA]) * cell_size * cell_size
        perimeter = cv2.arcLength(contour, True) * cell_size
        bars.append({"label": int(label), "length_m": length, "width_m": width,
                     "aspect": width / length if length else float("nan"),
                     "perimeter_area": perimeter / area if area else float("nan"),
                     "area_m2": area, "outline": contour.reshape(-1, 2).tolist()})
    bars.sort(key=lambda bar: -bar["area_m2"])
    return bars


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

    wet_days = np.zeros(values.shape, np.int32)
    for surface in water_surfaces:
        wet_days += (values < surface - slope * downstream).astype(np.int32)
    wet_days[~np.isfinite(values)] = -1
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
# CEP50 and CEP95.

FEATURE_DEFAULTS = {
    "min_size_m": 3.0,            # smallest length searched for
    "max_size_m": 100.0,          # largest length searched for
    "detect_sigma": 3.0,          # interior must differ from its ring by this many noise sigmas
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
                            "not_raised_or_sunk": 0}
    for mask in enclosures:
        kind = _classify(surface, mask, settings["detect_sigma"] * sigma_z)
        if kind is None or (kind == "mound" and not mounds) or (kind == "pocket" and not pockets):
            counts["not_raised_or_sunk"] += kind is None
            continue
        feature = _track_boundary(mask, surface, slope, kind, cell_size, sigma_z, settings)
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
        features.append(feature)

    features.sort(key=lambda feature: -feature["area_m2"])
    for index, feature in enumerate(features, start=1):
        feature["index"] = index
    return {"features": features, **counts, "noise_m": sigma_z, "cell_size": cell_size,
            "surface": surface, "settings": settings, "ridges": ridges}


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
    noise = float(np.median(magnitude[valid & (magnitude <= np.percentile(magnitude[valid], 10))]))
    noise = max(noise, 1e-9)
    edges = canny(image, sigma=sigma, low_threshold=settings["margin_low_sigma"] * noise,
                  high_threshold=settings["margin_high_sigma"] * noise)
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
    masks = []
    for label in range(1, count):
        if label in border:
            continue
        area = stats[label, cv2.CC_STAT_AREA]
        if not (minimum_area <= area <= maximum_area):
            continue
        mask = labels == label
        # Grown back out to the margin's centreline, which the thickening covered.
        if radius:
            mask = binary_dilation(mask, disk(radius + 1)) & valid
        masks.append(mask)
    return masks


def _close_open_ends(ridges, settings, cell_size):
    """
    Lines across the open ends of U-shaped margins: a ramped feature's up-ramp
    end has no margin, so its ring would otherwise never close. A margin
    qualifies when its two ends are closer together than ramp_closure times
    its length, so a straight or gently curved margin is left open.
    """
    from skimage.measure import label as label_components
    lines = np.zeros(ridges.shape, np.uint8)
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


def _classify(surface, mask, minimum):
    """Mound if the interior stands above its ring, pocket if below, else None."""
    from skimage.morphology import dilation as binary_dilation, disk
    ring = binary_dilation(mask, disk(3)) & ~mask
    if mask.sum() == 0 or ring.sum() == 0:
        return None
    difference = float(np.median(surface[mask]) - np.median(surface[ring]))
    if difference > minimum:
        return "mound"
    if difference < -minimum:
        return "pocket"
    return None


def _track_boundary(mask, surface, slope, kind, cell_size, sigma_z, settings):
    """
    The arc-length Kalman outline of one enclosure. The walk follows the
    enclosure's boundary; on each normal the slope peak is the measurement.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
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

    def points(values):
        return (base + normal * values[:, None]).round().astype(int).tolist()

    area = abs(_shoelace(margin))
    outer = abs(_shoelace(base + normal * (offset + sigma)[:, None]))
    inner = abs(_shoelace(base + normal * (offset - sigma)[:, None]))

    covariance = _centroid_covariance_normals(base, normal, offset, sigma,
                                              settings["error_correlation_points"])
    covariance *= cell_size * cell_size
    cep50, cep95 = _cep(covariance)
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
        "cep50_m": cep50, "cep95_m": cep95,
        "area_m2": area * cell_size * cell_size,
        "area_sigma_m2": 0.5 * (outer - inner) * cell_size * cell_size,
        "length_m": length, "width_m": width,
        "elongation": length / max(width, 1e-9),
        "orientation_degrees": orientation,
        "diameter_m": 2.0 * math.sqrt(area / math.pi) * cell_size,
        "riser_width_m": float(np.median(toe - crest)) * cell_size,
        "lee_bearing": lee_bearing, "ramp_bearing": ramp_bearing, "asymmetry": asymmetry,
        "points_measured": sum(1 for c in lap_chosen if c), "points": count,
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


def _cep(covariance, draws=20000):
    """CEP50 and CEP95 of a 2-D position with this covariance, by sampling."""
    covariance = np.asarray(covariance, float)
    if not np.all(np.isfinite(covariance)) or np.trace(covariance) <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    samples = rng.multivariate_normal([0.0, 0.0], covariance, size=draws, check_valid="ignore")
    distance = np.hypot(samples[:, 0], samples[:, 1])
    return float(np.percentile(distance, 50)), float(np.percentile(distance, 95))


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

        # Shared settings
        self._spin_cutoff = QDoubleSpinBox()
        self._spin_cutoff.setRange(1.0, 1000.0)
        self._spin_cutoff.setValue(DEFAULTS["bedform_cutoff_m"])
        self._spin_cutoff.setSuffix(" m")
        self._spin_cutoff.setToolTip(
            "Horizontal length, not height. Features shorter than this across are "
            "treated as bedforms riding on a bar; longer ones as part of the bar. "
            "Set it between the dune length and the bar size, typically 10 to 20 m.")
        self._spin_min_bar = QDoubleSpinBox()
        self._spin_min_bar.setRange(1.0, 1e6)
        self._spin_min_bar.setValue(DEFAULTS["min_bar_area_m2"])
        self._spin_min_bar.setSuffix(" m\u00b2")
        self._check_numbers = QCheckBox("Number the bars")
        self._check_numbers.setChecked(True)
        self._check_numbers.setToolTip("Label each outlined bar with its row in the table.")
        self._check_numbers.toggled.connect(self._redraw)
        self._last = None          # (kind, result) of the analysis on screen
        self._highlight = None     # bar number picked in the report
        shared_box, shared = form("Scales")
        shared.addRow("Max bedform length:", self._spin_cutoff)
        shared.addRow("Min bar area:", self._spin_min_bar)
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
        self._spin_sensitivity = QDoubleSpinBox()
        self._spin_sensitivity.setRange(0.5, 20.0)
        self._spin_sensitivity.setSingleStep(0.5)
        self._spin_sensitivity.setValue(FEATURE_DEFAULTS["detect_sigma"])
        self._spin_sensitivity.setSuffix(" \u03c3")
        self._spin_sensitivity.setToolTip("How far a feature's top (or floor) must differ from "
                                          "its surroundings, in noise standard deviations.")
        self._spin_margin = QDoubleSpinBox()
        self._spin_margin.setRange(2.0, 50.0)
        self._spin_margin.setSingleStep(1.0)
        self._spin_margin.setValue(FEATURE_DEFAULTS["margin_high_sigma"])
        self._spin_margin.setSuffix(" \u03c3")
        self._spin_margin.setToolTip(
            "How steep a margin must be to be traced, in noise standard deviations of the "
            "slope. Traced margins continue along stretches half this steep. Lower finds "
            "faint and ramped margins, and splits features along bedform texture.")
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
        feature.addRow("Sensitivity:", self._spin_sensitivity)
        feature.addRow("Margin strength:", self._spin_margin)
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

        # The source and method for each analysis, after a pause on its heading.
        self._help = [HoverHelp(box, SUMMARIES[name])
                      for box, name in ((pattern_box, "Pattern"), (bedform_box, "Bedforms"),
                                        (feature_box, "Mounds and pockets"),
                                        (inundation_box, "Inundation"))]

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        for box in (shared_box, pattern_box, bedform_box, feature_box, inundation_box):
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
        for widget, key in ((self._spin_cutoff, "cutoff"), (self._spin_min_bar, "min_bar"),
                            (self._spin_wave_min, "wave_min"), (self._spin_wave_max, "wave_max"),
                            (self._edit_gage, "gage"), (self._spin_slope, "slope"),
                            (self._spin_shift, "datum_shift"),
                            (self._check_numbers, "number_bars"),
                            (self._combo_resolution, "analysis_resolution"),
                            (self._spin_max_cells, "max_analysis_cells_millions"),
                            (self._spin_feature_min, "feature_min"),
                            (self._spin_feature_max, "feature_max"),
                            (self._spin_sensitivity, "feature_sensitivity"),
                            (self._spin_margin, "feature_margin_sigma"),
                            (self._spin_gap, "feature_gap_m"),
                            (self._spin_ramp, "feature_ramp_m"),
                            (self._spin_freedom, "feature_freedom"),
                            (self._check_mounds, "find_mounds"),
                            (self._check_pockets, "find_pockets"),
                            (self._combo_boundary, "feature_boundary"),
                            (self._check_band, "feature_band")):
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
        return {"bedform_cutoff_m": self._spin_cutoff.value(),
                "min_bar_area_m2": self._spin_min_bar.value(),
                "wavelet_min_m": self._spin_wave_min.value(),
                "wavelet_max_m": self._spin_wave_max.value(),
                "wavelet_scales": DEFAULTS["wavelet_scales"]}

    def _apply_resolution(self, *_):
        """Hand the analysis resolution to the Elevation tab, which reads the regions."""
        index = max(self._combo_resolution.currentIndex(), 0)
        self._dem_tab.analysis_coarsening = self._coarsening_choices[index][1]
        self._dem_tab.max_analysis_cells = self._spin_max_cells.value() * 1e6

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

    def _listed(self):
        """The numbered items of the analysis on screen, in report order."""
        if self._last is None:
            return []
        kind, result = self._last
        if kind in ("pattern", "bedforms"):
            return result["bars"]
        if kind == "features":
            return self._feature_outlines(result)
        return []

    def _pick_from_report(self):
        """A click on a table row picks that bar: it is outlined and centred."""
        if self._last is None or self._last[0] not in ("pattern", "bedforms", "features"):
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
        self._worker.start()

    # ------------------------------------------------------------------------------------------------------------------
    def _run_pattern(self):
        patch, cell_size, context = self._patch()
        if patch is None:
            return
        settings = self._settings_dict()
        self._start(lambda: pattern_metrics(patch, cell_size, settings),
                    lambda result, error: self._show_pattern(result, error, context),
                    "Computing pattern metrics\u2026")

    def _show_pattern(self, result, error, context):
        if error or result.get("error"):
            self._report.setPlainText(error or result["error"])
            return
        from downloader_viewers import stretch, to_pixmap, plot_histogram
        bars = result["bars"]
        aspects = [bar["aspect"] for bar in bars]
        perimeters = [bar["perimeter_area"] for bar in bars]
        lines = [
            f"Region: {context[2]}; {result['cell_size']:,.2f} m per cell.",
            "Bars here are the ground above the mean bed, after smoothing at the max "
            "bedform length.",
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
        self._draw_pattern(result, fit=True)
        counts, edges = result["histogram"]
        values = np.repeat((edges[:-1] + edges[1:]) / 2.0, counts)
        self._plot.setVisible(True)
        self._plot.setPixmap(to_pixmap(plot_histogram(values, "Detrended bed level (m)")).scaled(
            self._plot.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _draw_pattern(self, result, fit=False):
        from downloader_viewers import stretch, to_pixmap
        background = cv2.applyColorMap(stretch(result["detrended"], 1, 99), cv2.COLORMAP_TURBO)
        self._view.set_pixmap(to_pixmap(background), keep_view=not fit)
        self._overlay(result["bars"], lambda bar: (255, 255, 255))

    # ------------------------------------------------------------------------------------------------------------------
    def _run_bedforms(self):
        patch, cell_size, context = self._patch()
        if patch is None:
            return
        settings = self._settings_dict()
        self._start(lambda: superimposed_bedforms(patch, cell_size, settings),
                    lambda result, error: self._show_bedforms(result, error),
                    "Measuring superimposed bedforms\u2026")

    def _show_bedforms(self, result, error):
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
        self._draw_bedforms(result, fit=True)
        # No chart for this analysis, so the pane goes rather than sitting empty.
        self._plot.clear()
        self._plot.setVisible(False)

    def _draw_bedforms(self, result, fit=False):
        from downloader_viewers import stretch, to_pixmap
        image = cv2.applyColorMap(stretch(result["residual"], 1, 99), cv2.COLORMAP_VIRIDIS)
        self._view.set_pixmap(to_pixmap(image), keep_view=not fit)
        self._overlay(result["bars"],
                      lambda bar: (0, 0, 255) if bar["above_threshold"] else (255, 255, 255))

    # ------------------------------------------------------------------------------------------------------------------
    def _run_features(self):
        patch, cell_size, context = self._patch()
        if patch is None:
            return
        if not (self._check_mounds.isChecked() or self._check_pockets.isChecked()):
            QMessageBox.information(self, "Mounds and pockets", "Tick Mounds, Pockets, or both.")
            return
        settings = {"min_size_m": self._spin_feature_min.value(),
                    "max_size_m": max(self._spin_feature_max.value(),
                                      self._spin_feature_min.value() * 1.5),
                    "detect_sigma": self._spin_sensitivity.value(),
                    "shape_freedom": self._spin_freedom.value(),
                    "margin_high_sigma": self._spin_margin.value(),
                    "margin_low_sigma": self._spin_margin.value() / 2.0,
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
            f"{len(features)} feature(s): {mounds} mound(s), {len(features) - mounds} pocket(s). "
            f"{result['below_resolution']} too narrow to resolve, "
            f"{result['not_raised_or_sunk']} enclosures neither raised nor sunk, "
            f"{result['unconfirmed']} not confirmed once tracked, "
            f"{result.get('outside_range', 0)} outside the size range.",
            "Click a row to zoom to that feature.",
            "Length and width are along and across the long axis; orientation is the long "
            "axis's bearing (0 = up the image). CEP50 and CEP95 are the radii containing the "
            "centre with 50% and 95% probability. Area is given with one standard deviation.",
            "Lee is the direction the steepest side faces, ramp the direction the gentlest "
            f"faces, shown when they differ by at least {result['settings']['asymmetry_min']:.1f}:1.",
            "",
            "  #  kind    length  width  elong  orient  area (m\u00b2)      CEP50  CEP95  "
            "riser   lee   ramp   asym  margin traced",
        ]
        for feature in features:
            def bearing(value):
                return "-" if not np.isfinite(value) else f"{value:.0f}\u00b0"
            lines.append(
                f"{feature['index']:>3}  {feature['kind']:<6}  {feature['length_m']:>6.1f}  "
                f"{feature['width_m']:>5.1f}  {feature['elongation']:>5.1f}  "
                f"{feature['orientation_degrees']:>5.0f}\u00b0  "
                f"{feature['area_m2']:>7,.0f} \u00b1 {feature['area_sigma_m2']:<5,.0f}  "
                f"{feature['cep50_m']:>5.2f}  {feature['cep95_m']:>5.2f}  "
                f"{feature['riser_width_m']:>5.1f}  {bearing(feature['lee_bearing']):>5}  "
                f"{bearing(feature['ramp_bearing']):>5}  {feature['asymmetry']:>5.1f}  "
                f"{feature['points_measured'] / max(feature['points'], 1):>6.0%}")
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
        self._view.set_pixmap(to_pixmap(image), keep_view=not fit)
        bands = None
        if self._check_band.isChecked():
            bands = [feature[key] for feature in features for key in ("band_outer", "band_inner")]
        self._overlay(features, lambda f: MOUND_COLOR if f["kind"] == "mound" else POCKET_COLOR,
                      bands)

    # ------------------------------------------------------------------------------------------------------------------
    def _run_inundation(self):
        patch, cell_size, context = self._patch()
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

        fraction = np.where(valid, days / max(total, 1), 0.0)
        image = cv2.applyColorMap((fraction * 255).astype(np.uint8), cv2.COLORMAP_OCEAN)
        image[~valid] = 0
        image[(days == 0) & valid] = (40, 110, 160)          # never wet: tan
        self._view.set_pixmap(to_pixmap(image))
        self._view.clear_overlays()

        from downloader_viewers import plot_profile
        self._plot.setVisible(True)
        self._plot.setPixmap(to_pixmap(plot_profile(
            np.asarray(result["surfaces"], np.float32),
            "Water surface at the gage (m, NAVD88)", "day", "elevation")).scaled(
            self._plot.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
