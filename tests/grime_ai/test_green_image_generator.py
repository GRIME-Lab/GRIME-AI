"""Greenness edge-case test images: the generator's files and expected values, checked against get_greenness."""
import os

import numpy as np
import openpyxl
import pytest
from PIL import Image

from GRIME_AI.GreenImageGenerator import GreenImageGenerator
from GRIME_AI.vegetation_indices import GreennessIndex, Vegetation_Indices

EXPECTED_CASES = {
    "rgb_reference": ("RGB", 3), "rgba_4channel": ("RGBA", 4), "grayscale_1channel": ("L", 1),
    "single_row": ("RGB", 3), "single_column": ("RGB", 3), "all_black": ("RGB", 3),
    "zero_red": ("RGB", 3), "zero_red_and_green": ("RGB", 3),
}


@pytest.fixture(scope="module")
def edge_cases(tmp_path_factory):
    """Generate the edge-case images once; return (folder, list of spreadsheet rows as dicts)."""
    out = tmp_path_factory.mktemp("green_images")
    gen = GreenImageGenerator(out_dir=str(out), width=64, height=48)
    gen.generate_edge_case_images()
    folder = out / "edge_cases"
    xlsx = [f for f in os.listdir(folder) if f.endswith(".xlsx")]
    assert len(xlsx) == 1
    rows = list(openpyxl.load_workbook(folder / xlsx[0]).active.iter_rows(values_only=True))
    return folder, [dict(zip(rows[0], r)) for r in rows[1:]]


def test_every_case_is_generated(edge_cases):
    folder, rows = edge_cases
    assert {r["case"] for r in rows} == set(EXPECTED_CASES)
    for r in rows:
        assert (folder / r["filename"]).is_file()
        assert r["filename"].endswith(".png")


def test_image_modes_and_sizes(edge_cases):
    folder, rows = edge_cases
    for r in rows:
        img = Image.open(folder / r["filename"])
        mode, channels = EXPECTED_CASES[r["case"]]
        assert (img.mode, len(img.getbands())) == (mode, channels) == (r["PIL_mode"], r["channels"])
        assert img.size == (r["width"], r["height"])
    sizes = {r["case"]: (r["width"], r["height"]) for r in rows}
    assert sizes["single_row"][1] == 1 and sizes["single_column"][0] == 1


def test_expected_values_match_get_greenness(edge_cases):
    """The spreadsheet's expected GCC, NDVI and RGI equal what GRIME AI computes from each image."""
    folder, rows = edge_cases
    vi = Vegetation_Indices()
    for r in rows:
        image = np.array(Image.open(folder / r["filename"]))
        for name, column in ((Vegetation_Indices.GCC, "expected_GCC"),
                             (Vegetation_Indices.NDVI, "expected_NDVI"),
                             (Vegetation_Indices.RGI, "expected_RGI")):
            got = vi.get_greenness(GreennessIndex(name), image).get_value()
            assert got == pytest.approx(r[column], abs=1e-4), f"{r['case']} {name}"


def test_reference_color_detects_red_blue_swap():
    """Red and blue differ in the reference color, so reading it in BGR order changes NDVI and RGI."""
    r, g, b = GreenImageGenerator.EDGE_REFERENCE_RGB
    assert r != b
    assert GreenImageGenerator.expected_indices((r, g, b))[1:] != GreenImageGenerator.expected_indices((b, g, r))[1:]


@pytest.mark.parametrize("rgb, expected", [
    (None, (-999.0, -999.0, -999.0)),
    ((0, 0, 0), (0.0, -999.0, -999.0)),
    ((0, 255, 0), (1.0, 1.0, -999.0)),
    ((40, 100, 60), (0.5, 0.4286, 2.5)),
])
def test_expected_indices(rgb, expected):
    assert GreenImageGenerator.expected_indices(rgb) == pytest.approx(expected)
