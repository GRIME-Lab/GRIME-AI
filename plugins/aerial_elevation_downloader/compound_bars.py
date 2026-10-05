#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# compound_bars.py
#
# Whether a bar in an elevation model is one unit or several amalgamated ones.
#
# The test is nesting. Slicing the surface at a series of levels and following
# the connected regions upward gives a merge tree: regions that are separate
# high on the bar join lower down. A simple bar has one summit and one nest. A
# compound bar has two or more summits that stay separate until they merge at
# the elevation of the accretion surface between the units, which is why
# nested closed contours read as compound in a slope view.
#
# Everything is in the elevation model's own vertical units, typically metres
# NAVD88 for 3DEP. No stage and no time: this is one surface, analysed as it
# stands. The same analysis on an inundation-threshold map, which is in
# gage-height units, would give the same quantities for comparison.
#
# Reported per unit: summit elevation, prominence (summit minus the level at
# which it merges with a neighbour), area, and an ellipse fit giving
# orientation and elongation. Bar units are lobate rather than elliptical, so
# the fit summarises orientation and elongation; its residual is itself
# informative.
#
# No Qt here; the viewer draws what this returns.

import numpy as np
import cv2

# Defaults, all editable in the dialog.
DEFAULT_LEVELS = 50            # slices, when no thickness is given
DEFAULT_SLICE_INTERVAL = 0.10  # metres; the thickness of one slice
MAX_SLICES = 2000              # a thin slice over a tall region is still bounded
DEFAULT_PROMINENCE_FACTOR = 3.0    # times the estimated vertical noise
# A unit has to stand this far above the level where it merges, as a fraction of
# the patch's relief, however quiet the surface is. Without it, a noise speck on
# a smooth bar counts as a unit.
# On a real elevation model the noise floor is centimetres while bedforms are
# decimetres, so a noise-based threshold alone finds hundreds of "units". The
# floor is a fraction of the patch's relief instead.
MIN_PROMINENCE_FRACTION = 0.10
DEFAULT_MIN_AREA_CELLS = 2000      # a unit smaller than this is a bedform, not a bar
DEFAULT_MAX_UNITS = 10             # keep the most prominent; the rest are detail
DEFAULT_DETREND_ORDER = 1          # remove the downstream slope before slicing
# Wide enough to remove bedforms, which are what turn one bar platform into a
# row of "units". Raise it further on a rippled bar.
DEFAULT_SMOOTHING_CELLS = 9
VOID_BELOW = -9000.0
NOISE_WINDOW = 9               # window for the local roughness estimate


# ======================================================================================================================
# Surface preparation
# ======================================================================================================================
def detrend(surface, order=1):
    """
    Remove the downstream slope. A reach falls continuously, so slicing raw
    elevation cuts bands across the region rather than closed lobes around bar
    platforms, and the outlines end at the region's edges. Fitting a plane
    (order 1) or a quadratic (order 2) and subtracting it leaves the relief
    relative to the local surface, which is what a bar platform is.

    Returns (detrended, trend) with the trend kept so elevations can be
    reported on the real datum rather than as residuals.
    """
    if not order:
        return surface, np.zeros_like(surface)

    rows, columns = surface.shape
    y, x = np.mgrid[0:rows, 0:columns]
    y = (y / max(rows - 1, 1)).astype(np.float32)
    x = (x / max(columns - 1, 1)).astype(np.float32)

    terms = [np.ones_like(x), x, y]
    if order >= 2:
        terms += [x * x, x * y, y * y]
    design = np.stack([term.ravel() for term in terms], axis=1)

    values = surface.ravel()
    good = np.isfinite(values)
    if good.sum() < design.shape[1]:
        return surface, np.zeros_like(surface)
    coefficients, *_ = np.linalg.lstsq(design[good], values[good], rcond=None)
    trend = (design @ coefficients).reshape(surface.shape).astype(np.float32)
    return (surface - trend).astype(np.float32), trend


def prepare(elevation, smoothing_cells=DEFAULT_SMOOTHING_CELLS):
    """Voids filled and the surface lightly smoothed, since contours on raw 1 m
    lidar follow noise rather than form."""
    surface = np.asarray(elevation, np.float32).copy()
    surface[surface < VOID_BELOW] = np.nan
    if np.all(np.isnan(surface)):
        return None
    surface = np.where(np.isnan(surface), np.nanmedian(surface), surface)
    if smoothing_cells and smoothing_cells > 1:
        size = int(smoothing_cells) | 1          # the kernel has to be odd
        if size <= 5:
            # medianBlur on float32 only supports 3 and 5, and a median keeps
            # the breaks of slope crisp, so it is preferred at those widths.
            surface = cv2.medianBlur(surface, size)
        else:
            # Wider than that, blur instead: the point is to remove bedforms
            # shorter than the width, and a box mean does that without the
            # float32 restriction.
            surface = cv2.blur(surface, (size, size))
    return surface


def estimate_noise(surface) -> float:
    """
    Vertical noise, from the flattest part of the surface: the median of the
    local standard deviation over its lowest decile. Water and smooth sand give
    the floor, so anything above a few times this is form rather than noise.
    """
    # Centred first: elevations are hundreds of metres and the noise is
    # centimetres, so squaring them in place loses the signal to rounding.
    centred = (surface - float(np.mean(surface))).astype(np.float32)
    mean = cv2.blur(centred, (NOISE_WINDOW, NOISE_WINDOW))
    mean_square = cv2.blur(centred * centred, (NOISE_WINDOW, NOISE_WINDOW))
    local = np.sqrt(np.maximum(mean_square - mean * mean, 0))
    quiet = local[local <= np.percentile(local, 10)]
    return float(np.median(quiet)) if quiet.size else 0.0


# ======================================================================================================================
# Merge tree
# ======================================================================================================================
def analyze(elevation, levels=DEFAULT_LEVELS, slice_interval=0.0, prominence=None,
            prominence_factor=DEFAULT_PROMINENCE_FACTOR,
            min_area_cells=DEFAULT_MIN_AREA_CELLS,
            smoothing_cells=DEFAULT_SMOOTHING_CELLS,
            max_units=DEFAULT_MAX_UNITS,
            detrend_order=DEFAULT_DETREND_ORDER):
    """
    Find the bar units in one elevation patch.

    levels            how many slices to cut, used when slice_interval is 0
    slice_interval    slice thickness in the elevation model's own units, which
                      is the quantity that means something: 0.10 is a
                      ten-centimetre slice whatever the relief happens to be.
                      The count follows from the relief, and both are reported.
    max_units         report at most this many, the most prominent first
    prominence        minimum summit prominence for a unit, in elevation units;
                      None derives it from the noise estimate
    min_area_cells    smallest region counted as a unit
    smoothing_cells   filter width applied before slicing, in cells; bedforms
                      shorter than this are removed
    detrend_order     0 keeps the raw surface, 1 removes the downstream slope,
                      2 also removes curvature across the reach

    Returns a dict: units (one entry each), interval, noise, prominence used,
    classification, and the merge level where the units join.
    """
    # Noise is measured on the raw surface: smoothing is what removes it, so an
    # estimate taken afterwards reads as zero and prunes nothing.
    raw = prepare(elevation, smoothing_cells=0)
    if raw is None:
        return {"error": "No elevation values in this region."}
    noise = estimate_noise(raw)

    surface = prepare(elevation, smoothing_cells)
    # Relative to the local surface, so a platform reads as a closed lobe
    # rather than a band across a sloping reach.
    surface, trend = detrend(surface, detrend_order)
    trend_drop = float(trend.max() - trend.min()) if detrend_order else 0.0
    low, high = float(surface.min()), float(surface.max())
    relief = high - low
    if relief <= 0:
        return {"error": "This region is flat."}

    if prominence is None:
        prominence = max(prominence_factor * noise, relief * MIN_PROMINENCE_FRACTION)

    # A thickness in metres beats a count: it is the same cut on every bar,
    # while a count means a different cut on every one.
    if slice_interval and slice_interval > 0:
        levels = int(max(round(relief / float(slice_interval)), 2))
        levels = min(levels, MAX_SLICES)

    slice_levels = np.linspace(high, low, int(levels))    # downward, summits first
    interval = float(slice_levels[0] - slice_levels[1]) if len(slice_levels) > 1 else relief

    # Follow each region down. A region that appears on its own is a new summit;
    # when two summits land in the same region, they have merged, and that level
    # is the top of the surface joining them.
    units = []                       # {"summit", "peak_cell", "merge_level", "mask"}
    previous_labels = None
    previous_owner = {}              # label at the previous level -> unit index

    for level in slice_levels:
        mask = (surface >= level).astype(np.uint8)
        count, labels = cv2.connectedComponents(mask, connectivity=8)
        owner = {}

        for label in range(1, count):
            region = labels == label
            if region.sum() < 1:
                continue
            claimants = set()
            if previous_labels is not None:
                for previous_label in np.unique(previous_labels[region]):
                    if previous_label and previous_label in previous_owner:
                        claimants.add(previous_owner[previous_label])

            if not claimants:
                # A new summit: record where and how high.
                peak = np.unravel_index(np.argmax(np.where(region, surface, -np.inf)),
                                        surface.shape)
                units.append({"summit": float(surface[peak]), "peak_cell": peak,
                              "merge_level": None, "mask": region.copy(),
                              "merged_into": None})
                owner[label] = len(units) - 1
            else:
                keep = max(claimants, key=lambda index: units[index]["summit"])
                merging = len(claimants) > 1
                for index in claimants:
                    # A unit's own extent is what it covered while it was still
                    # separate. After it merges, the region belongs to its
                    # neighbour, so its mask and area must stop growing.
                    if units[index]["merge_level"] is None and not merging:
                        units[index]["mask"] = region.copy()
                    if merging and index != keep and units[index]["merge_level"] is None:
                        units[index]["merge_level"] = float(level)
                        units[index]["merged_into"] = keep
                owner[label] = keep

        previous_labels, previous_owner = labels, owner

    result = _summarize(surface, units, low, relief, interval, noise,
                        prominence, min_area_cells, max_units)
    # Report elevations on the real datum: the residual surface is what the
    # units were found on, but its numbers mean nothing to a reader.
    if detrend_order and not result.get("error"):
        for unit in result["units"]:
            row, column = unit["peak_cell"]
            offset = float(trend[row, column])
            unit["summit"] += offset
            unit["merge_level"] += offset
        if result.get("amalgamation_level") is not None:
            result["amalgamation_level"] += float(np.mean(trend))
    result["detrend_order"] = detrend_order
    result["trend_drop"] = trend_drop
    result["levels_used"] = int(levels)
    result["requested_interval"] = float(slice_interval or 0.0)
    return result


def _basins(surface, summits):
    """
    Each cell assigned to the summit it belongs to, by watershed on the
    inverted surface. A unit's extent is its basin, not the sliver it held
    while it was still a separate region, which is only a few slices deep when
    two platforms are amalgamated.
    """
    markers = np.zeros(surface.shape, np.int32)
    for index, (row, column) in enumerate(summits, start=1):
        markers[row, column] = index
    try:
        from skimage.segmentation import watershed
        return watershed(-surface, markers)
    except ImportError:
        # Without skimage, fall back to the nearest summit, which is the same
        # answer for well separated units and a rougher one for the rest.
        rows, columns = np.indices(surface.shape)
        best = np.zeros(surface.shape, np.int32)
        nearest = np.full(surface.shape, np.inf)
        for index, (row, column) in enumerate(summits, start=1):
            distance = (rows - row) ** 2 + (columns - column) ** 2
            closer = distance < nearest
            best[closer] = index
            nearest[closer] = distance[closer]
        return best


def _summarize(surface, units, low, relief, interval, noise, prominence, min_area_cells,
               max_units=DEFAULT_MAX_UNITS):
    """Turn the raw units into the reported ones, dropping noise."""
    kept = []
    for unit in units:
        # A unit that never merged runs to the bottom of the patch, so its
        # prominence is measured from there.
        merge_level = unit["merge_level"] if unit["merge_level"] is not None else low
        if unit["summit"] - merge_level < prominence:
            continue
        kept.append((unit, merge_level))

    # The most prominent first, then cut: a bar has a handful of platforms, and
    # everything past that is bedforms and vegetation.
    found = len(kept)
    kept.sort(key=lambda pair: -(pair[0]["summit"] - pair[1]))
    if max_units:
        kept = kept[:int(max_units)]

    # Which unit each one merged into, translated from the raw indices into the
    # reported numbering. A unit that merged into another sits inside it: that
    # is a platform on a platform, rather than two bars side by side.
    raw_position = {id(unit): position for position, (unit, _) in enumerate(kept, start=1)}
    parents = {}
    for position, (unit, _) in enumerate(kept, start=1):
        parent = unit.get("merged_into")
        if parent is not None and 0 <= parent < len(units):
            parents[position] = raw_position.get(id(units[parent]))

    basins = _basins(surface, [unit["peak_cell"] for unit, _ in kept]) if kept else None

    # A unit's footprint is its basin above the level where the units join,
    # which is the bar itself rather than the whole patch: below that level the
    # ground is the channel and the floodplain, and including it makes every
    # outline the edge of the ROI.
    merges = [level for _, level in kept if level > low]
    base_level = min(merges) if merges else low + interval

    reported = []
    for position, (unit, merge_level) in enumerate(kept, start=1):
        basin = (basins == position) if basins is not None else unit["mask"]
        crest = basin & (surface >= base_level)      # the part standing above the junction
        crest_area = int(crest.sum())
        if crest_area < min_area_cells:
            continue

        # The crest outline is what shows the nesting; the basin is the unit's
        # full footprint, which is the area worth comparing between units.
        outline, ellipse = _outline_and_ellipse(crest)
        reported.append({
            "index": len(reported) + 1,
            "raw_position": position,
            "summit": unit["summit"],
            "peak_cell": [int(unit["peak_cell"][0]), int(unit["peak_cell"][1])],
            "merge_level": float(merge_level),
            "prominence": float(unit["summit"] - merge_level),
            "area_cells": int(basin.sum()),
            "crest_area_cells": crest_area,
            "outline": outline,
            "ellipse": ellipse,
        })

    reported.sort(key=lambda entry: -entry["summit"])
    renumber = {entry["raw_position"]: position
                for position, entry in enumerate(reported, start=1)}
    for position, entry in enumerate(reported, start=1):
        entry["index"] = position
        parent = parents.get(entry["raw_position"])
        entry["parent"] = renumber.get(parent) if parent else None
    _add_stack_level(reported)

    # The level where the units join: the highest merge among the units that
    # merged into another one.
    merge_levels = [entry["merge_level"] for entry in reported
                    if entry["merge_level"] > low]
    compound = len(reported) > 1
    return {
        "units": reported,
        "unit_count": len(reported),
        "candidates": found,
        "relief": float(relief),
        "interval": float(interval),
        "noise": float(noise),
        "prominence_used": float(prominence),
        "amalgamation_level": float(max(merge_levels)) if merge_levels else None,
        "classification": "compound" if compound else "simple",
        "surface": surface,
    }


def _outline_and_ellipse(mask):
    """The unit's largest outline, and an ellipse fitted to it."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], None
    contour = max(contours, key=cv2.contourArea)
    outline = contour.reshape(-1, 2).tolist()
    ellipse = None
    if len(contour) >= 5:
        (centre_x, centre_y), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        major, minor = max(axis_a, axis_b), max(min(axis_a, axis_b), 1e-6)
        ellipse = {"centre": [float(centre_x), float(centre_y)],
                   "major": float(major), "minor": float(minor),
                   "elongation": float(major / minor),
                   "angle_degrees": float(angle)}
    return outline, ellipse


# ======================================================================================================================
# Margins: units from the slope field
# ======================================================================================================================
# The merge tree needs vertical separation, so two platforms that abut at almost
# the same height read as one even when their margins are plain in a slope view.
# This finds the margins instead.
#
# The flat interiors seed a watershed on the slope field. Boundaries grow from
# the seeds and settle on the slope ridges where those are strong, and
# interpolate where they are broken, so nothing has to be bridged and no
# enclosure is invented by a gap-closing distance. Every boundary is then
# measured: the median slope along it says how real it is, and boundaries below
# a threshold are dissolved, merging the regions they separated.
#
# Elevation is sampled afterwards, which restores what the slope field alone
# cannot give: each unit's summit and median height, and the drop across each
# margin.

DEFAULT_FLAT_PERCENTILE = 40.0     # slope below this counts as a platform interior
DEFAULT_MARGIN_PERCENTILE = 80.0   # a boundary this steep or steeper is a real margin
DEFAULT_MIN_SEED_CELLS = 400       # smaller flat patches are not platforms
# How many margin thresholds to run, and how far apart. One level answers "which
# bars are here"; more answer "what stands on them", because a weaker threshold
# subdivides a bar into the platforms it is built from.
DEFAULT_DETAIL_STEPS = 3
DEFAULT_DETAIL_STEP = 12.0
# Outlines share their margins, so a unit counts as inside another when nearly
# all of it falls within, rather than every cell.
CONTAINMENT_FRACTION = 0.9


def slope_degrees(surface) -> np.ndarray:
    dy, dx = np.gradient(surface.astype(np.float32))
    return np.degrees(np.arctan(np.hypot(dx, dy)))


def valid_cells(elevation) -> np.ndarray:
    """Cells that hold data. Drilling into a unit passes the rest as NaN, and
    those must not count toward the percentiles or become part of a unit."""
    values = np.asarray(elevation, np.float32)
    return np.isfinite(values) & (values > VOID_BELOW)


def analyze_margins(elevation, smoothing_cells=DEFAULT_SMOOTHING_CELLS,
                    detrend_order=DEFAULT_DETREND_ORDER,
                    flat_percentile=DEFAULT_FLAT_PERCENTILE,
                    margin_percentile=DEFAULT_MARGIN_PERCENTILE,
                    min_seed_cells=DEFAULT_MIN_SEED_CELLS,
                    min_area_cells=DEFAULT_MIN_AREA_CELLS,
                    max_units=DEFAULT_MAX_UNITS,
                    detail_steps=DEFAULT_DETAIL_STEPS,
                    detail_step_percent=DEFAULT_DETAIL_STEP):
    """
    Bar units from their margins.

    flat_percentile    slope percentile below which ground seeds a platform
    margin_percentile  slope percentile a boundary must reach to be kept; the
                       regions either side of a weaker one are merged
    min_seed_cells     smallest flat patch that can seed a unit
    detail_steps       how many thresholds to run, from margin_percentile down.
                       One gives a flat partition of the region into bars; more
                       give the stacking, because a weaker threshold subdivides
                       what a stronger one kept whole, and those subdivisions
                       are the platforms the bar is built from.
    detail_step_percent  spacing of those thresholds, in percentile points
    """
    inside = valid_cells(elevation)
    if not inside.any():
        return {"error": "No elevation values in this region."}

    surface = prepare(elevation, smoothing_cells)
    surface, trend = detrend(surface, detrend_order)

    slope = slope_degrees(surface)
    # Percentiles over the region's own cells only, so a drill-down is judged
    # against the unit it opened rather than against the padding around it.
    flat_level = float(np.percentile(slope[inside], flat_percentile))
    margin_level = float(np.percentile(slope[inside], margin_percentile))

    seeds = ((slope <= flat_level) & inside).astype(np.uint8)
    seed_count, seed_labels, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    markers = np.zeros(slope.shape, np.int32)
    index = 0
    for label in range(1, seed_count):
        if stats[label, cv2.CC_STAT_AREA] < min_seed_cells:
            continue
        index += 1
        markers[seed_labels == label] = index
    if index == 0:
        return {"error": "No flat ground found: raise the flat percentile."}

    base = _watershed(slope, markers)
    base[~inside] = 0

    # Coarse to fine. Dissolving only ever merges regions, so each finer
    # partition subdivides the one above it: the nesting is exact rather than
    # inferred from geometry, and the subdivisions are the platforms a bar is
    # built from.
    percentiles = [max(float(margin_percentile) - step * float(detail_step_percent), 1.0)
                   for step in range(max(int(detail_steps), 1))]
    levels = [float(np.percentile(slope, percentile)) for percentile in percentiles]
    order = np.argsort(levels)
    by_level = _dissolve_levels(base, slope, [levels[i] for i in order])
    partitions = [None] * len(levels)
    for position, index in enumerate(order):
        partitions[index] = by_level[position]

    partitions = [np.where(inside, partition, 0) for partition in partitions]
    units, boundaries = _hierarchy(partitions, levels, surface, trend, slope,
                                   min_area_cells, detrend_order, max_units)

    return {
        "method": "margins",
        "units": units,
        "unit_count": len(units),
        "candidates": len(units),
        "boundaries": boundaries,
        "flat_level": flat_level,
        "margin_level": margin_level,
        "detail_levels": [float(level) for level in levels],
        "detail_percentiles": [float(value) for value in percentiles],
        "detrend_order": detrend_order,
        "trend_drop": float(trend.max() - trend.min()) if detrend_order else 0.0,
        "classification": "compound" if len(units) > 1 else "simple",
        "surface": surface,
        "slope": slope,
    }


def _hierarchy(partitions, levels, surface, trend, slope, min_area_cells,
               detrend_order, max_units):
    """
    Units from every threshold, each pointing at the unit above it. The coarsest
    partition gives the bars; each finer one gives what stands on them.
    """
    units, boundaries = [], []
    # The level above: its labels, and its kept units by label. Overlap with a
    # parent is read from those labels at this unit's own cells, so no
    # full-size masks are compared.
    previous_labels, previous_units = None, {}
    for stack, (labels, level) in enumerate(zip(partitions, levels)):
        found = _margin_units(labels, surface, trend, slope, min_area_cells, detrend_order)
        found.sort(key=lambda unit: -unit["area_cells"])
        if max_units:
            found = found[:int(max_units)]

        kept_here = {}
        for unit in found:
            window, mask = unit["_cells"]
            unit["stack"] = stack
            unit["level_slope"] = level
            unit["parent"] = None
            best, best_overlap = None, 0
            if previous_labels is not None and previous_units:
                above = previous_labels[window][mask]
                above = above[np.isin(above, list(previous_units))]
                if above.size:
                    ids, counts = np.unique(above, return_counts=True)
                    best_label = int(ids[int(np.argmax(counts))])
                    best, best_overlap = previous_units[best_label], int(counts.max())
            # The region above that contains this one; merging guarantees there
            # is exactly one, so the overlap test only has to find it.
            if best is not None and best_overlap >= 0.5 * unit["area_cells"]:
                unit["parent"] = best["index"] if "index" in best else None
                unit["_parent_ref"] = best
            # A finer level that repeats a region from the level above adds
            # nothing: without this the same unit appears once per level.
            if best is not None:
                same_size = abs(unit["area_cells"] - best["area_cells"]) \
                    <= 0.02 * max(best["area_cells"], 1)
                if same_size and best_overlap >= 0.98 * unit["area_cells"]:
                    unit["duplicate_of_parent"] = True
            if not unit.get("duplicate_of_parent"):
                units.append(unit)
                kept_here[unit["label"]] = unit
        # A level that added nothing must not wipe out the parents for the next
        # one, so the last level that did add something stays as the parent set.
        if kept_here:
            previous_labels, previous_units = labels, kept_here

    for unit in units:
        unit.pop("_cells", None)
    for position, unit in enumerate(units, start=1):
        unit["index"] = position
    # Resolve by identity: two units can hold equal values, so a membership test
    # on the dicts themselves finds the wrong one.
    numbering = {id(unit): unit["index"] for unit in units}
    for unit in units:
        parent = unit.pop("_parent_ref", None)
        unit["parent"] = numbering.get(id(parent)) if parent is not None else None

    for pair, strength in sorted(_boundary_strength(partitions[-1], slope).items(),
                                 key=lambda item: -item[1]):
        boundaries.append({"between": [int(pair[0]), int(pair[1])], "slope": float(strength)})
    return units, boundaries[:12]


def _watershed(slope, markers):
    """Regions grown from the seeds, meeting on the slope ridges."""
    try:
        from skimage.segmentation import watershed
        return watershed(slope, markers)
    except ImportError:
        # OpenCV's watershed works on an 8-bit colour image; the slope field is
        # scaled into one, which costs precision but not the partition.
        image = cv2.cvtColor(
            cv2.normalize(slope, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR)
        labels = markers.copy()
        cv2.watershed(image, labels)
        labels[labels < 0] = 0
        return labels


def _boundary_strength(labels, slope):
    """Median slope along each boundary, keyed by the pair of regions it separates."""
    strengths = {}
    for shift_axis in (0, 1):
        first = labels
        second = np.roll(labels, -1, axis=shift_axis)
        edge = (first != second) & (first > 0) & (second > 0)
        if shift_axis == 0:
            edge[-1, :] = False
        else:
            edge[:, -1] = False
        if not edge.any():
            continue
        pairs = np.stack([np.minimum(first[edge], second[edge]),
                          np.maximum(first[edge], second[edge])], axis=1)
        values = np.maximum(slope[edge], np.roll(slope, -1, axis=shift_axis)[edge])
        for pair, value in zip(map(tuple, pairs), values):
            strengths.setdefault(pair, []).append(float(value))
    return {pair: float(np.median(values)) for pair, values in strengths.items()}


def _edge_values(labels, slope):
    """
    Every boundary cell's slope, grouped by the pair of regions it separates:
    the boundary strength of a pair is the median of its group. Grouped once,
    with array sorting, rather than scanned again after every merge.
    """
    pairs, values = [], []
    for shift_axis in (0, 1):
        second = np.roll(labels, -1, axis=shift_axis)
        edge = (labels != second) & (labels > 0) & (second > 0)
        if shift_axis == 0:
            edge[-1, :] = False
        else:
            edge[:, -1] = False
        if not edge.any():
            continue
        first_ids, second_ids = labels[edge], second[edge]
        pairs.append(np.stack([np.minimum(first_ids, second_ids),
                               np.maximum(first_ids, second_ids)], axis=1))
        values.append(np.maximum(slope[edge], np.roll(slope, -1, axis=shift_axis)[edge]))
    if not pairs:
        return {}
    pairs = np.concatenate(pairs)
    values = np.concatenate(values).astype(np.float64)
    keys = pairs[:, 0].astype(np.int64) * (int(labels.max()) + 1) + pairs[:, 1]
    order = np.argsort(keys, kind="stable")
    keys, pairs, values = keys[order], pairs[order], values[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    groups = np.split(values, starts[1:])
    return {(int(pairs[s, 0]), int(pairs[s, 1])): group for s, group in zip(starts, groups)}


def _dissolve_levels(labels, slope, levels):
    """
    The partitions left after merging across every boundary weaker than each
    level, for several levels at once (ascending). The weakest boundary goes
    first, and its two regions become one; only the boundaries of the merged
    region change, so only those are recomputed, kept in a priority queue.
    Merging only ever continues from the previous level's partition, so one
    pass serves every level.
    """
    import heapq
    edges = _edge_values(labels, slope)
    neighbours = {}
    for a, b in edges:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    version = {}
    heap = []
    for pair, group in edges.items():
        version[pair] = 0
        heap.append((float(np.median(group)), pair[0], pair[1], 0))
    heapq.heapify(heap)
    parent = {}

    def root(region):
        while region in parent:
            region = parent[region]
        return region

    snapshots = []
    for level in levels:
        while heap and heap[0][0] < level:
            strength, a, b, stamp = heapq.heappop(heap)
            pair = (a, b)
            if version.get(pair) != stamp or pair not in edges:
                continue                                   # superseded by a merge
            keep, drop = a, b                               # the lower label survives
            del edges[pair]
            version.pop(pair, None)
            neighbours[keep].discard(drop)
            for other in list(neighbours.pop(drop, set())):
                if other == keep:
                    continue
                old = (min(drop, other), max(drop, other))
                moved = edges.pop(old)
                version.pop(old, None)
                neighbours[other].discard(drop)
                new = (min(keep, other), max(keep, other))
                merged = np.concatenate([edges[new], moved]) if new in edges else moved
                edges[new] = merged
                neighbours[keep].add(other)
                neighbours[other].add(keep)
                version[new] = version.get(new, 0) + 1
                heapq.heappush(heap, (float(np.median(merged)), new[0], new[1], version[new]))
            parent[drop] = keep

        # This level's partition: every original label mapped to its region.
        top = int(labels.max())
        mapping = np.arange(top + 1)
        for region in list(parent):
            mapping[region] = root(region)
        snapshots.append(mapping[labels])
    return snapshots


def _dissolve_to(labels, slope, margin_level):
    """Merge across every boundary weaker than margin_level; return the labels."""
    return _dissolve_levels(labels, slope, [margin_level])[0]


def _dissolve_weak_margins(labels, slope, margin_level):
    """
    Merge the regions either side of a boundary that is not steep enough to be a
    margin. A watershed always returns boundaries; this is what decides which of
    them mean anything.
    """
    labels = labels.copy()
    reported = []
    while True:
        strengths = _boundary_strength(labels, slope)
        if not strengths:
            break
        weakest = min(strengths, key=strengths.get)
        if strengths[weakest] >= margin_level:
            break
        keep, drop = weakest
        labels[labels == drop] = keep

    for pair, strength in sorted(_boundary_strength(labels, slope).items(),
                                 key=lambda item: -item[1]):
        reported.append({"between": [int(pair[0]), int(pair[1])],
                         "slope": float(strength)})
    return labels, reported


def _margin_units(labels, surface, trend, slope, min_area_cells, detrend_order):
    """
    One entry per region: its height, extent, outline and ellipse. Each region
    is measured in its own bounding box, padded by a cell so its edge is found
    against its neighbours, rather than as a mask the size of the whole patch.
    """
    from scipy import ndimage
    units = []
    rows, columns = labels.shape
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        top = max(box[0].start - 1, 0)
        left = max(box[1].start - 1, 0)
        window = (slice(top, min(box[0].stop + 1, rows)), slice(left, min(box[1].stop + 1, columns)))
        mask = labels[window] == index
        area = int(mask.sum())
        if area < min_area_cells:
            continue

        local_surface = surface[window]
        heights = local_surface[mask]
        offset = float(np.mean(trend[window][mask])) if detrend_order else 0.0
        peak_local = np.unravel_index(np.argmax(np.where(mask, local_surface, -np.inf)),
                                      mask.shape)
        peak = (peak_local[0] + top, peak_local[1] + left)
        outline, ellipse = _outline_and_ellipse(mask)
        outline = [[x + left, y + top] for x, y in outline]
        if ellipse:
            ellipse["centre"] = [ellipse["centre"][0] + left, ellipse["centre"][1] + top]
        edge = _edge_of(mask)
        units.append({
            "label": int(index),
            "summit": float(surface[peak]) + offset,
            "median_height": float(np.median(heights)) + offset,
            "relief": float(heights.max() - heights.min()),
            "peak_cell": [int(peak[0]), int(peak[1])],
            "area_cells": area,
            "crest_area_cells": area,
            "margin_slope": float(np.median(slope[window][edge])) if edge.any() else 0.0,
            "outline": outline,
            "ellipse": ellipse,
            "_cells": (window, mask),
        })
    return units


def _assign_containment(units, shape):
    """
    Which unit each one sits inside. A unit whose outline lies within another's
    is a platform on a platform; a unit that shares no enclosure with any other
    is a bar in its own right.
    """
    masks = {}
    for unit in units:
        outline = np.asarray(unit["outline"], np.int32)
        filled = np.zeros(shape, np.uint8)
        if outline.size:
            cv2.fillPoly(filled, [outline], 1)
        masks[unit["index"]] = filled.astype(bool)

    for unit in units:
        own = masks[unit["index"]]
        own_area = int(own.sum())
        best, best_area = None, None
        for other in units:
            if other is unit:
                continue
            other_mask = masks[other["index"]]
            other_area = int(other_mask.sum())
            if other_area <= own_area:
                continue
            # Inside when nearly all of this unit falls within the other one:
            # outlines touch at shared margins, so an exact test is too strict.
            covered = int((own & other_mask).sum())
            if own_area and covered / own_area >= CONTAINMENT_FRACTION:
                if best_area is None or other_area < best_area:
                    best, best_area = other["index"], other_area
        unit["parent"] = best


def _add_stack_level(units):
    """
    Each unit's stack level: 0 is a bar standing on the ground, 1 is a platform
    built on that bar, 2 a platform on that platform, and so on.
    """
    by_index = {unit["index"]: unit for unit in units}
    for unit in units:
        stack, parent, guard = 0, unit.get("parent"), 0
        while parent and parent in by_index and guard < len(units):
            stack += 1
            parent = by_index[parent].get("parent")
            guard += 1
        unit["stack"] = stack


def _edge_of(mask):
    """The one-cell border of a region, where its margin is measured."""
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    return (mask.astype(np.uint8) - eroded).astype(bool)


def stack_levels(result) -> list:
    """The stack levels present in a result, lowest first."""
    return sorted({unit.get("stack", 0) for unit in result.get("units", [])})


def area_text(result, cells) -> str:
    """An area in square metres when the cell size is known, else in cells."""
    size = result.get("cell_size")
    if size:
        return f"{cells * size * size:,.0f} m\u00b2"
    return f"{cells:,} cells"


def _area_value(result, cells):
    size = result.get("cell_size")
    return cells * size * size if size else cells


def _area_heading(result) -> str:
    return "area (m\u00b2)" if result.get("cell_size") else "area (cells)"


def describe_margins(result) -> str:
    """The margin findings as text."""
    if result.get("error"):
        return result["error"]

    lines = [f"{result['unit_count']} platform(s) found from their margins; this bar reads as "
             f"{result['classification']}."]
    lines.append(f"Flat ground: slope below {result['flat_level']:,.2f}\u00b0. A boundary counts "
                 f"as a margin at {result['margin_level']:,.2f}\u00b0 or steeper.")
    if result.get("detrend_order"):
        lines.append(f"The reach slope was removed first; it falls {result['trend_drop']:,.2f} "
                     "across the region.")
    lines.append("")
    lines.append(_nesting_summary(result["units"], result))
    lines.append("")
    lines.append(f"platform  stack  sits on  median height  summit      relief  "
                 f"{_area_heading(result):>12}  margin slope")
    for unit in result["units"]:
        parent = unit.get("parent")
        lines.append(
            f"{unit['index']:>8}  {unit.get('stack', 0) + 1:>5}  "
            f"{(str(parent) if parent else '-'):>7}  "
            f"{unit['median_height']:>13,.2f}  {unit['summit']:>10,.2f}  "
            f"{unit['relief']:>10,.2f}  {_area_value(result, unit['area_cells']):>12,.0f}  "
            f"{unit['margin_slope']:>12,.2f}")

    if result.get("boundaries"):
        lines.append("")
        lines.append("Margins kept, steepest first (median slope in degrees):")
        for boundary in result["boundaries"][:12]:
            lines.append(f"   platforms {boundary['between'][0]} and {boundary['between'][1]}: "
                         f"{boundary['slope']:,.2f}")
    return "\n".join(lines)


def combine(tree_result, margin_result) -> str:
    """
    Where the two methods agree and where they do not, which is the useful part:
    a margin with no prominence is two platforms at the same height, and
    prominence with no margin is usually vegetation or noise.
    """
    if tree_result.get("error") or margin_result.get("error"):
        return ""
    tree_count = tree_result["unit_count"]
    margin_count = margin_result["unit_count"]
    lines = ["", f"Elevation prominence finds {tree_count} platform(s); margins find "
                 f"{margin_count}."]
    if margin_count > tree_count:
        lines.append("More margins than prominences: platforms of similar height that abut, "
                     "which is what an amalgamated bar looks like.")
    elif tree_count > margin_count:
        lines.append("More prominences than margins: highs without a boundary around them, "
                     "which is usually vegetation, a spike, or one platform with an "
                     "irregular top.")
    else:
        lines.append("The two agree on the count.")
    return "\n".join(lines)


# ======================================================================================================================
# Drawing
# ======================================================================================================================
UNIT_COLORS = [(0, 0, 255), (0, 200, 255), (0, 255, 120), (255, 180, 0),
               (255, 80, 255), (255, 255, 255)]


def draw(result, background, line_width=1, show_ellipses=True, label=True,
         only_stack=None, highlight=None) -> np.ndarray:
    """
    Unit outlines, ellipses and summits over a BGR background image.

    line_width is in pixels of the image, not of the screen, so a thin line
    stays thin however far the view is zoomed out. Labels and markers are
    scaled with it for the same reason.

    highlight draws one unit in white underneath its own outline, so a clicked
    unit is unmistakable among its neighbours.

    only_stack draws one stack level on its own. A stacked unit shares most of
    its outline with the unit it sits on, since it is a subdivision of it, so
    the two are hard to tell apart when everything is drawn at once.
    """
    display = background.copy()
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    width = max(int(line_width), 1)
    marker_size = 6 * width
    font_scale = 0.35 * width

    for unit in sorted(result.get("units", []), key=lambda entry: entry.get("stack", 0)):
        if only_stack is not None and unit.get("stack", 0) != only_stack:
            continue
        color = UNIT_COLORS[(unit["index"] - 1) % len(UNIT_COLORS)]
        outline = np.asarray(unit["outline"], np.int32)
        if outline.size:
            if highlight == unit["index"]:
                cv2.polylines(display, [outline], True, (255, 255, 255), width + 4, cv2.LINE_AA)
            cv2.polylines(display, [outline], True, color, width, cv2.LINE_AA)
        if unit.get("parent"):
            # A line from this unit's summit to its parent's, so the stacking is
            # visible without reading the table.
            parent = next((other for other in result["units"]
                           if other["index"] == unit["parent"]), None)
            if parent:
                cv2.line(display,
                         (int(unit["peak_cell"][1]), int(unit["peak_cell"][0])),
                         (int(parent["peak_cell"][1]), int(parent["peak_cell"][0])),
                         color, width, cv2.LINE_AA)
        if show_ellipses and unit["ellipse"]:
            ellipse = unit["ellipse"]
            cv2.ellipse(display,
                        (int(ellipse["centre"][0]), int(ellipse["centre"][1])),
                        (int(ellipse["major"] / 2), int(ellipse["minor"] / 2)),
                        ellipse["angle_degrees"], 0, 360, color, width, cv2.LINE_AA)
        row, column = unit["peak_cell"]
        cv2.drawMarker(display, (int(column), int(row)), color, cv2.MARKER_CROSS,
                       marker_size, width)
        if label:
            cv2.putText(display, str(unit["index"]),
                        (int(column) + marker_size, int(row) - marker_size),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, width, cv2.LINE_AA)
    return display


def _nesting_summary(units, result=None) -> str:
    """
    The stacking, which is what says whether there are bars on bars. Stacks are
    numbered from 1: stack 1 sits on the ground, stack 2 on a stack-1 platform,
    and so on.
    """
    if not units:
        return ""
    result = result or {}
    roots = [unit for unit in units if not unit.get("parent")]
    stacked = [unit for unit in units if unit.get("parent")]
    by_index = {unit["index"]: unit for unit in units}

    def chain_length(unit, guard=0):
        parent = unit.get("parent")
        if not parent or parent not in by_index or guard > len(units):
            return 1
        return 1 + chain_length(by_index[parent], guard + 1)

    tallest = max((chain_length(unit) for unit in units), default=1)

    lines = ["",
             f"Stacking: {len(roots)} platform(s) sit on the ground and {len(stacked)} sit "
             f"on another platform; the tallest pile is {tallest} platform(s) high."]
    if stacked:
        lines.append("Platforms built on platforms: "
                     + ", ".join(f"{unit['index']} on {unit['parent']}" for unit in stacked))
    else:
        lines.append("Nothing is stacked: these are separate bars, not one bar built of "
                     "stacked platforms.")
        if len(result.get("detail_percentiles", [0, 0])) == 1:
            lines.append("With one stack level the margins method cannot find stacking; "
                         "raise Stack levels to look for it.")

    lines.append("")
    lines.append("Stacking (indented one step per level; stack 1 is on the ground):")

    def branch(unit, indent):
        label = (f"{'    ' * indent}stack {unit.get('stack', 0) + 1}  platform "
                 f"{unit['index']}: summit {unit['summit']:,.2f}, "
                 f"area {area_text(result, unit['area_cells'])}")
        rows = [label]
        for child in units:
            if child.get("parent") == unit["index"]:
                rows += branch(child, indent + 1)
        return rows

    for root in roots:
        lines += branch(root, 1)
    return "\n".join(lines)


def describe(result) -> str:
    """The findings as text, for the report panel."""
    if result.get("error"):
        return result["error"]

    extra = ""
    if result.get("candidates", 0) > result["unit_count"]:
        extra = (f"  {result['candidates']} summit(s) passed the prominence threshold; "
                 f"the {result['unit_count']} most prominent are reported. Raise "
                 "Prominence or Min area to be stricter.")
    lines = [f"{result['unit_count']} platform(s) found from prominence; this bar reads as "
             f"{result['classification']}.{extra}",
             f"Relief {result['relief']:,.2f} after removing the reach slope; "
             f"{result.get('levels_used', 0)} slices of {result['interval']:,.3f} each; "
             f"vertical noise about {result['noise']:,.3f}; "
             f"prominence threshold {result['prominence_used']:,.3f}."]
    if result.get("detrend_order"):
        lines.append(f"The reach slope falls {result['trend_drop']:,.2f} across the region.")
    if result.get("amalgamation_level") is not None:
        lines.append(f"The platforms join at {result['amalgamation_level']:,.2f}, the elevation "
                     "of the surface between them.")
    lines.append("")
    lines.append(_nesting_summary(result["units"], result))
    lines.append("")
    heading = "basin (m\u00b2)" if result.get("cell_size") else "basin (cells)"
    crest_heading = "crest (m\u00b2)" if result.get("cell_size") else "crest (cells)"
    lines.append(f"platform  stack  sits on  summit      prominence  merge level  "
                 f"{heading:>13}  {crest_heading:>13}")
    for unit in result["units"]:
        parent = unit.get("parent")
        lines.append(
            f"{unit['index']:>8}  {unit.get('stack', 0) + 1:>5}  "
            f"{(str(parent) if parent else '-'):>7}  "
            f"{unit['summit']:>10,.2f}  {unit['prominence']:>10,.2f}  "
            f"{unit['merge_level']:>11,.2f}  "
            f"{_area_value(result, unit['area_cells']):>13,.0f}  "
            f"{_area_value(result, unit.get('crest_area_cells', 0)):>13,.0f}")
    if result["unit_count"] > 1:
        angles = [u["ellipse"]["angle_degrees"] for u in result["units"] if u["ellipse"]]
        if len(angles) > 1:
            spread = max(angles) - min(angles)
            lines.append("")
            lines.append(f"Platform orientations differ by {spread:,.0f}\u00b0"
                         + (", which suggests accretion under different flow directions."
                            if spread > 20 else "."))
    return "\n".join(lines)
