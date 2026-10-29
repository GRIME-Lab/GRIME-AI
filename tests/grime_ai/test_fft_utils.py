"""FFT utilities used for blur triage and texture features."""
import cv2
import numpy as np
import pytest

from GRIME_AI.utils import fft_utils as fu


def test_spectrum_rejects_color_image():
    with pytest.raises(ValueError):
        fu.fft_spectrum(np.zeros((16, 16, 3)))


def test_spectrum_has_zero_dc(textured_image):
    magnitude, radius = fu.fft_spectrum(textured_image)
    assert magnitude.shape == textured_image.shape
    assert np.all(magnitude[radius == 0.0] == 0.0)


def test_blur_score_of_flat_image_is_zero():
    assert fu.fft_blur_score(np.full((64, 64), 128, dtype=np.uint8)) == 0.0


def test_blur_score_is_between_zero_and_one(textured_image):
    score = fu.fft_blur_score(textured_image)
    assert 0.0 <= score <= 1.0


def test_blurred_image_scores_lower_than_sharp(textured_image):
    blurred = cv2.GaussianBlur(textured_image, (0, 0), 4)
    assert fu.fft_blur_score(blurred) < fu.fft_blur_score(textured_image)


def test_blur_score_independent_of_brightness_scaling(textured_image):
    darker = (textured_image.astype(np.float64) * 0.5)
    assert fu.fft_blur_score(darker) == pytest.approx(fu.fft_blur_score(textured_image), rel=1e-9)


def test_radial_profile_length(textured_image):
    profile = fu.fft_radial_profile(textured_image, num_bins=16)
    assert profile.shape == (16,)
    assert np.all(profile >= 0)


def test_ring_power_scores_match_direct_score(textured_image):
    """Scoring from ring powers at a ring boundary equals the direct score at that frequency."""
    num_bins = fu.DEFAULT_RADIAL_RINGS
    ring = 4
    powers = fu.fft_ring_power(textured_image, num_bins=num_bins)[None, :]
    from_rings = fu.blur_scores_at(powers, ring)[0]
    direct = fu.fft_blur_score(textured_image, cutoff=ring / num_bins * fu.NYQUIST)
    assert from_rings == pytest.approx(direct, rel=1e-6)


def test_flag_blurry_flags_outliers():
    scores = np.array([0.80, 0.82, 0.79, 0.81, 0.80, 0.83, 0.78, 0.81, 0.10])
    flags, threshold = fu.flag_blurry(scores)
    assert flags[-1]
    assert not flags[:-1].any()
    assert scores[-1] < threshold < scores[:-1].min()


def test_choose_cutoff_falls_back_with_too_few_images(textured_image):
    powers = np.array([fu.fft_ring_power(textured_image)] * (fu.MIN_IMAGES_FOR_AUTO_CUTOFF - 1))
    cutoff, cutoff_ring, _ = fu.choose_blur_cutoff(powers)
    assert cutoff == fu.DEFAULT_BLUR_CUTOFF
    assert cutoff_ring is None
