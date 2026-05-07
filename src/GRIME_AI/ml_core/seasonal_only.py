#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation: University of Nebraska-Lincoln
# Contact: jstranzl2@huskers.unl.edu
# License: Apache License, Version 2.0

"""
Seasonal Only Utility
=====================
Provides functions to keep ONLY images from a selected season.
This is the complement of seasonal_dropout.py, which excludes a season.

Used by the segmentation pipeline when the user selects "Filter by Season"
in the Segment Images tab — only images from the selected season are
passed to the inference engine.

Supported filename format:
    {site_name}___{YYYY}-{MM}-{DD}T{HH}-{MM}-{SS}Z.jpg
    Example: NE_Platte_River_near_Grand_Island___2023-01-15T12-00-04Z.jpg

Season type options:
    "Meteorological"  - month-based boundaries (Dec 1, Mar 1, Jun 1, Sep 1)
    "Astronomical"    - fixed-date boundaries  (Dec 21, Mar 20, Jun 21, Sep 23)
"""

from GRIME_AI.ml_core.seasonal_dropout import extract_date_from_usgs_filename, get_season
from typing import Optional
from datetime import date


def is_selected_season(filename: str, selected_season: str, season_type: str) -> bool:
    """
    Return True if the image falls in the selected season.

    Args:
        filename:        Bare filename or full path of the image.
        selected_season: Season to keep: "Winter", "Spring", "Summer", or "Fall".
        season_type:     "Meteorological" or "Astronomical".

    Returns:
        True  if the image is in the selected season.
        False if it is not, or if the date cannot be parsed
              (unparseable filenames are excluded by default).
    """
    capture_date = extract_date_from_usgs_filename(filename)
    if capture_date is None:
        return False
    return get_season(capture_date, season_type) == selected_season


def filter_season_only(
    image_paths: list,
    selected_season: str,
    season_type: str,
) -> tuple:
    """
    Partition a list of image paths keeping only the selected season.
    Convenience wrapper around filter_seasons_only for a single season.
    """
    return filter_seasons_only(image_paths, [selected_season], season_type)


def filter_seasons_only(
    image_paths: list,
    selected_seasons: list,
    season_type: str,
) -> tuple:
    """
    Partition a list of image paths into kept (selected seasons only) and excluded.

    Args:
        image_paths:      List of image filenames or full paths.
        selected_seasons: List of season names to keep e.g. ["Winter", "Spring"].
        season_type:      "Meteorological" or "Astronomical".

    Returns:
        Tuple of (kept, excluded) lists.
        kept     — images from any of the selected seasons.
        excluded — all other images.
    """
    selected_set = set(selected_seasons)
    kept         = []
    excluded     = []
    for path in image_paths:
        capture_date = extract_date_from_usgs_filename(str(path))
        if capture_date is None:
            # Unparseable filenames are excluded by default for segmentation
            excluded.append(path)
            continue
        if get_season(capture_date, season_type) in selected_set:
            kept.append(path)
        else:
            excluded.append(path)
    return kept, excluded
