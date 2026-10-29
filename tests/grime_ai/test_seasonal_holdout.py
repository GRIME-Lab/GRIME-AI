"""Season assignment and season hold-out filtering used for leave-one-season-out validation."""
from datetime import date

import pytest

from GRIME_AI.ml_core.seasonal_dropout import (extract_date_from_usgs_filename, filter_seasons,
                                               get_season, is_knockout_image)

MET, ASTRO = "Meteorological", "Astronomical"


@pytest.mark.parametrize("d, season", [
    (date(2025, 1, 15), "Winter"), (date(2025, 2, 28), "Winter"),
    (date(2025, 3, 1), "Spring"), (date(2025, 5, 31), "Spring"),
    (date(2025, 6, 1), "Summer"), (date(2025, 8, 31), "Summer"),
    (date(2025, 9, 1), "Fall"), (date(2025, 11, 30), "Fall"),
    (date(2025, 12, 1), "Winter"), (date(2025, 12, 31), "Winter"),
])
def test_meteorological_seasons(d, season):
    assert get_season(d, MET) == season


@pytest.mark.parametrize("d, season", [
    (date(2025, 3, 19), "Winter"), (date(2025, 3, 20), "Spring"),
    (date(2025, 6, 20), "Spring"), (date(2025, 6, 21), "Summer"),
    (date(2025, 9, 22), "Summer"), (date(2025, 9, 23), "Fall"),
    (date(2025, 12, 20), "Fall"), (date(2025, 12, 21), "Winter"),
])
def test_astronomical_seasons(d, season):
    assert get_season(d, ASTRO) == season


def test_extract_date_from_usgs_filename():
    name = r"C:\images\NE_Platte_River_near_Grand_Island___2024-07-04T17-30-00Z.jpg"
    assert extract_date_from_usgs_filename(name) == date(2024, 7, 4)


@pytest.mark.parametrize("name", ["IMG_0001.jpg", "site___2024-13-40T00-00-00Z.jpg"])
def test_unparseable_filename_returns_none(name):
    assert extract_date_from_usgs_filename(name) is None


def test_unparseable_images_are_kept():
    assert is_knockout_image("IMG_0001.jpg", "Winter", MET) is False


def test_filter_seasons_partitions_by_season():
    files = [
        "site___2024-01-10T12-00-00Z.jpg",   # Winter
        "site___2024-04-10T12-00-00Z.jpg",   # Spring
        "site___2024-07-10T12-00-00Z.jpg",   # Summer
        "site___2024-10-10T12-00-00Z.jpg",   # Fall
        "no_date.jpg",
    ]
    kept, excluded = filter_seasons(files, ["Winter", "Summer"], MET)
    assert excluded == [files[0], files[2]]
    assert kept == [files[1], files[3], files[4]]
    assert sorted(kept + excluded) == sorted(files)
