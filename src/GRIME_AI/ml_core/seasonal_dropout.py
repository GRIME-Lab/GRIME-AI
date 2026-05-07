#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation: University of Nebraska-Lincoln
# Contact: jstranzl2@huskers.unl.edu
# License: Apache License, Version 2.0

"""
Seasonal Dropout Utility
========================
Provides functions to determine whether a USGS HIVIS image file should be
excluded from training and validation based on its capture date and the
selected knockout season.

Supported filename format:
    {site_name}___{YYYY}-{MM}-{DD}T{HH}-{MM}-{SS}Z.jpg
    Example: MD_Lake_Serene_at_Edgewood___2024-01-01T13-00-01Z.jpg

Season type options:
    "Meteorological"  - month-based boundaries (Dec 1, Mar 1, Jun 1, Sep 1)
    "Astronomical"    - fixed-date boundaries  (Dec 21, Mar 20, Jun 21, Sep 23)
"""

import re
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Season date boundaries
# Each entry is (month, day) for the START of that season.
# Seasons are evaluated in order; Winter wraps around the year end.
# ---------------------------------------------------------------------------

_METEOROLOGICAL_STARTS = {
    "Winter": (12, 1),
    "Spring": (3,  1),
    "Summer": (6,  1),
    "Fall":   (9,  1),
}

_ASTRONOMICAL_STARTS = {
    "Winter": (12, 21),
    "Spring": (3,  20),
    "Summer": (6,  21),
    "Fall":   (9,  23),
}

# Evaluation order: Spring → Summer → Fall → Winter (Winter is the fallback)
_SEASON_EVAL_ORDER = ["Spring", "Summer", "Fall", "Winter"]


def _get_season_starts(season_type: str) -> dict:
    if season_type.lower().startswith("astro"):
        return _ASTRONOMICAL_STARTS
    return _METEOROLOGICAL_STARTS


def get_season(capture_date: date, season_type: str) -> str:
    """
    Return the season name ("Winter", "Spring", "Summer", "Fall") for a given
    date and season type.

    Args:
        capture_date: The date of image capture.
        season_type:  "Meteorological" or "Astronomical".

    Returns:
        Season name string.
    """
    starts = _get_season_starts(season_type)
    md = (capture_date.month, capture_date.day)

    # Walk Spring → Summer → Fall; if none match, it's Winter.
    for season in ["Spring", "Summer", "Fall"]:
        if md >= starts[season]:
            current = season
        else:
            break
    else:
        # All three matched — must be Fall or later
        current = "Fall"

    # Simpler and correct approach: find the latest season whose start <= md
    current = "Winter"
    for season in ["Spring", "Summer", "Fall"]:
        if md >= starts[season]:
            current = season

    # Handle winter wrap: Dec 1 (met) or Dec 21 (astro) restarts Winter
    winter_start = starts["Winter"]
    if md >= winter_start:
        current = "Winter"

    return current


def extract_date_from_usgs_filename(filename: str) -> Optional[date]:
    """
    Parse the capture date from a USGS HIVIS filename.

    Expected format: {site_name}___{YYYY}-{MM}-{DD}T{HH}-{MM}-{SS}Z.jpg

    Args:
        filename: Bare filename or full path string.

    Returns:
        datetime.date object, or None if the filename does not match.
    """
    # Strip directory components
    bare = filename.replace("\\", "/").split("/")[-1]

    # Match the timestamp after the triple underscore
    match = re.search(r"___(\d{4})-(\d{2})-(\d{2})T", bare)
    if not match:
        return None

    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def is_knockout_image(filename: str, knockout_season: str, season_type: str) -> bool:
    """
    Determine whether an image should be excluded from training and validation
    based on the seasonal dropout configuration.

    Args:
        filename:        Bare filename or full path of the image.
        knockout_season: Season to exclude: "Winter", "Spring", "Summer", or "Fall".
        season_type:     "Meteorological" or "Astronomical".

    Returns:
        True  if the image falls in the knockout season and should be excluded.
        False if the image should be kept, or if the date cannot be parsed
              (images with unparseable filenames are kept by default).
    """
    capture_date = extract_date_from_usgs_filename(filename)
    if capture_date is None:
        return False

    image_season = get_season(capture_date, season_type)
    return image_season == knockout_season


def get_season_date_range(knockout_season: str, season_type: str) -> str:
    """
    Return a human-readable date range string for the given season and type.

    Args:
        knockout_season: "Winter", "Spring", "Summer", or "Fall".
        season_type:     "Meteorological" or "Astronomical".

    Returns:
        String such as "Dec 1 – Feb 28" or "Dec 21 – Mar 19".
    """
    ranges = {
        "Meteorological": {
            "Winter": "Dec 1 – Feb 28",
            "Spring": "Mar 1 – May 31",
            "Summer": "Jun 1 – Aug 31",
            "Fall":   "Sep 1 – Nov 30",
        },
        "Astronomical": {
            "Winter": "Dec 21 – Mar 19",
            "Spring": "Mar 20 – Jun 20",
            "Summer": "Jun 21 – Sep 22",
            "Fall":   "Sep 23 – Dec 20",
        },
    }
    return ranges.get(season_type, ranges["Meteorological"]).get(knockout_season, "")


def filter_knockout_images(
    image_paths: list,
    knockout_season: str,
    season_type: str,
) -> tuple:
    """
    Partition a list of image paths into kept and excluded lists.

    Args:
        image_paths:     List of image file paths.
        knockout_season: Single season to exclude.
        season_type:     "Meteorological" or "Astronomical".

    Returns:
        Tuple of (kept, excluded) lists.
    """
    return filter_seasons(image_paths, [knockout_season], season_type)


def filter_seasons(
    image_paths: list,
    holdout_seasons: list,
    season_type: str,
) -> tuple:
    """
    Partition a list of image paths into kept and excluded lists,
    excluding one or more seasons.

    Args:
        image_paths:     List of image file paths.
        holdout_seasons: List of season names to exclude e.g. ["Winter", "Spring"].
        season_type:     "Meteorological" or "Astronomical".

    Returns:
        Tuple of (kept, excluded) lists.
        kept     — images NOT in any holdout season.
        excluded — images that ARE in at least one holdout season.
    """
    holdout_set = set(holdout_seasons)
    kept        = []
    excluded    = []
    for path in image_paths:
        capture_date = extract_date_from_usgs_filename(str(path))
        if capture_date is None:
            # Unparseable filenames are kept by default
            kept.append(path)
            continue
        if get_season(capture_date, season_type) in holdout_set:
            excluded.append(path)
        else:
            kept.append(path)
    return kept, excluded
