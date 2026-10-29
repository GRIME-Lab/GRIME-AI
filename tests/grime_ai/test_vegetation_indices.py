"""Greenness indices computed on synthetic solid-color images, where the exact answer is known."""
import numpy as np
import pytest

from GRIME_AI.vegetation_indices import GreennessIndex, Vegetation_Indices


def solid_rgb(r, g, b, size=(32, 32)):
    img = np.zeros((*size, 3), dtype=np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = r, g, b
    return img


@pytest.fixture
def vi():
    return Vegetation_Indices()


@pytest.mark.parametrize("rgb, expected", [
    ((0, 255, 0), 1.0),            # pure green
    ((255, 0, 0), 0.0),            # pure red
    ((128, 128, 128), 1.0 / 3.0),  # neutral gray
    ((50, 100, 50), 0.5),
])
def test_gcc_solid_color(vi, rgb, expected):
    result = vi.get_greenness(GreennessIndex(Vegetation_Indices.GCC), solid_rgb(*rgb))
    assert result.get_value() == pytest.approx(expected, abs=1e-9)


def test_gcc_black_image_is_zero(vi):
    """An all-black image has no signal; GCC is defined as 0 rather than dividing by zero."""
    assert vi.compute_GCC(0.0, 0.0, 0.0) == 0.0


def test_gcc_independent_of_image_size(vi):
    small = vi.get_greenness(GreennessIndex(Vegetation_Indices.GCC), solid_rgb(30, 90, 60, (8, 8)))
    large = vi.get_greenness(GreennessIndex(Vegetation_Indices.GCC), solid_rgb(30, 90, 60, (200, 300)))
    assert small.get_value() == pytest.approx(large.get_value(), abs=1e-12)


def test_gli_formula(vi):
    r, g, b = 40.0, 120.0, 60.0
    expected = (2 * g - r - b) / (2 * g + r + b)
    assert vi.compute_GLI(r, g, b) == pytest.approx(expected)


def test_exg_formula_uses_chromatic_coordinates(vi):
    r, g, b = 40.0, 120.0, 40.0
    total = r + g + b
    expected = 2 * g / total - (r / total + b / total)
    assert vi.compute_ExG(r, g, b) == pytest.approx(expected)


def test_vari_formula(vi):
    r, g, b = 40.0, 120.0, 30.0
    expected = (g - r) / (g + r - b)
    assert vi.compute_Visible_Atmospherically_Resistant_Vegetation_Index(g, r, b) == pytest.approx(expected)



def test_grayscale_image_returns_failure_value(vi):
    """An image without three color channels records -999 instead of crashing,
    and never keeps the previous image's value."""
    gi = GreennessIndex(Vegetation_Indices.GCC, value=0.42)
    vi.get_greenness(gi, np.zeros((4, 4), dtype=np.uint8))
    assert gi.get_value() == -999.0


def test_rgba_image_ignores_alpha(vi):
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[..., :3] = (40, 100, 60)
    rgba[..., 3] = 128
    rgb = rgba[..., :3].copy()
    for name in (Vegetation_Indices.GCC, Vegetation_Indices.NDVI, Vegetation_Indices.RGI):
        assert vi.get_greenness(GreennessIndex(name), rgba).get_value() == \
            pytest.approx(vi.get_greenness(GreennessIndex(name), rgb).get_value())


@pytest.mark.parametrize("shape", [(1, 64), (64, 1), (1, 1)])
def test_one_pixel_wide_images(vi, shape):
    result = vi.get_greenness(GreennessIndex(Vegetation_Indices.GCC), solid_rgb(40, 100, 60, shape))
    assert result.get_value() == pytest.approx(0.5)


@pytest.mark.parametrize("rgb, name", [
    ((0, 0, 0), Vegetation_Indices.NDVI),       # G + R = 0
    ((0, 0, 0), Vegetation_Indices.RGI),        # R = 0
    ((0, 255, 0), Vegetation_Indices.RGI),      # R = 0
    ((0, 0, 255), Vegetation_Indices.NDVI),     # G + R = 0
])
def test_zero_denominator_records_failure_value(vi, rgb, name):
    gi = GreennessIndex(name, value=0.42)
    vi.get_greenness(gi, solid_rgb(*rgb))
    assert gi.get_value() == -999.0


def test_ndvi_and_rgi_reference_values(vi):
    img = solid_rgb(40, 100, 60)
    assert vi.get_greenness(GreennessIndex(Vegetation_Indices.NDVI), img).get_value() == pytest.approx(60 / 140)
    assert vi.get_greenness(GreennessIndex(Vegetation_Indices.RGI), img).get_value() == pytest.approx(2.5)
