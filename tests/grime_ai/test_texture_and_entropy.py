"""Texture features and Shannon entropy on images with analytically known values."""
import numpy as np
import pytest

from GRIME_AI.ROI_Analyzer import ROI_Analyzer
from GRIME_AI.Texture import FourierTexture, GLCMTexture, LBPTexture


def test_entropy_of_constant_image_is_zero():
    assert ROI_Analyzer.calculate_shannon_entropy(np.full(1000, 77, dtype=np.uint8)) == 0.0


def test_entropy_of_two_equal_levels_is_one_bit():
    pixels = np.array([0] * 500 + [255] * 500, dtype=np.uint8)
    assert ROI_Analyzer.calculate_shannon_entropy(pixels) == pytest.approx(1.0)


def test_entropy_of_all_256_levels_is_eight_bits():
    pixels = np.repeat(np.arange(256, dtype=np.uint8), 10)
    assert ROI_Analyzer.calculate_shannon_entropy(pixels) == pytest.approx(8.0)


def test_entropy_of_empty_region_is_zero():
    assert ROI_Analyzer.calculate_shannon_entropy(np.array([], dtype=np.uint8)) == 0.0


def test_glcm_features_of_constant_image():
    features = GLCMTexture().compute_features(np.full((32, 32), 100, dtype=np.uint8))
    assert set(features) == {"contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"}
    assert features["contrast"] == pytest.approx(0.0)
    assert features["dissimilarity"] == pytest.approx(0.0)
    assert features["homogeneity"] == pytest.approx(1.0)
    assert features["energy"] == pytest.approx(1.0)


def test_glcm_contrast_of_checkerboard():
    """Neighbouring pixels always differ by 255 horizontally and vertically."""
    board = (np.indices((32, 32)).sum(axis=0) % 2 * 255).astype(np.uint8)
    features = GLCMTexture(angles=[0.0, np.pi / 2]).compute_features(board)
    assert features["contrast"] == pytest.approx(255.0 ** 2)


def test_glcm_accepts_color_image(rng):
    features = GLCMTexture().compute_features(rng.random((32, 32, 3)))
    assert np.isfinite(list(features.values())).all()


def test_lbp_histogram_is_a_density(textured_image):
    hist = LBPTexture().compute_features(textured_image)
    assert hist.sum() == pytest.approx(1.0)


def test_fourier_texture_length(textured_image):
    assert FourierTexture(num_bins=8).compute_features(textured_image).shape == (8,)
