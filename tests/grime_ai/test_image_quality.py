"""Image-quality triage checks on synthetic images with known defects."""
import cv2
import numpy as np
import pytest

from GRIME_AI.ImageQualityCheck import ImageQualityAnalyzer
from GRIME_AI.utils.fft_utils import MIN_IMAGES_FOR_AUTO_CUTOFF


def brightness_only():
    return ImageQualityAnalyzer(use_fft_blur=False, use_laplacian_blur=False, use_contrast=False,
                                use_exposure_clipping=False, resize_percent=100.0)


def test_nominal_image(textured_image):
    analyzer = ImageQualityAnalyzer(use_fft_blur=False, use_exposure_clipping=False, resize_percent=100.0)
    result = analyzer.analyze(textured_image)
    assert result["is_bad"] is False
    assert result["reason"] == "Nominal"


def test_too_dark():
    result = brightness_only().analyze(np.full((64, 64), 10, dtype=np.uint8))
    assert result["is_too_dark"] and not result["is_too_light"]
    assert result["reason"] == "Too Dark"


def test_too_light():
    result = brightness_only().analyze(np.full((64, 64), 250, dtype=np.uint8))
    assert result["is_too_light"] and not result["is_too_dark"]
    assert result["reason"] == "Too Light"


def test_brightness_of_uniform_image():
    result = brightness_only().analyze(np.full((64, 64), 100, dtype=np.uint8))
    assert result["brightness"] == pytest.approx(100.0, abs=0.5)


def test_low_contrast():
    img = np.full((64, 64), 120, dtype=np.uint8)
    img[:, 32:] = 130                                  # range of 10 grey levels
    analyzer = ImageQualityAnalyzer(use_fft_blur=False, use_laplacian_blur=False, use_brightness=False,
                                    use_exposure_clipping=False, resize_percent=100.0)
    result = analyzer.analyze(img)
    assert result["contrast"] == pytest.approx(10.0)
    assert result["is_low_contrast"]


def test_exposure_clipping_fraction():
    img = np.full((100, 100), 128, dtype=np.uint8)
    img[:50, :] = 255                                  # half the frame saturated
    analyzer = ImageQualityAnalyzer(use_fft_blur=False, use_laplacian_blur=False, use_brightness=False,
                                    use_contrast=False, resize_percent=100.0)
    result = analyzer.analyze(img)
    assert result["clip_fraction"] == pytest.approx(0.5)
    assert result["is_clipped"]


def test_laplacian_flags_blurred_image(textured_image):
    analyzer = ImageQualityAnalyzer(use_fft_blur=False, use_brightness=False, use_contrast=False,
                                    use_exposure_clipping=False, resize_percent=100.0)
    sharp = analyzer.analyze(textured_image)
    blurred = analyzer.analyze(cv2.GaussianBlur(textured_image, (0, 0), 5))
    assert not sharp["is_blurry"]
    assert blurred["is_blurry"]
    assert blurred["blur_laplacian"] < sharp["blur_laplacian"]


def test_color_imbalance():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 1] = 200                                  # strongly one channel
    analyzer = ImageQualityAnalyzer(use_fft_blur=False, use_laplacian_blur=False, use_brightness=False,
                                    use_contrast=False, use_exposure_clipping=False,
                                    use_color_imbalance=True, resize_percent=100.0)
    result = analyzer.analyze(img)
    assert result["color_imbalance"] == pytest.approx(1.0)
    assert result["is_color_imbalanced"]


def test_focus_roi_crops_blur_region():
    analyzer = ImageQualityAnalyzer(resize_percent=100.0, focus_roi=[0.25, 0.25, 0.5, 0.5])
    gray = np.zeros((100, 200), dtype=np.uint8)
    assert analyzer._apply_focus_roi(gray).shape == (50, 100)


def test_fft_blur_skipped_with_too_few_images(textured_image):
    analyzer = ImageQualityAnalyzer(use_laplacian_blur=False, resize_percent=100.0)
    results = [analyzer.analyze(textured_image) for _ in range(MIN_IMAGES_FOR_AUTO_CUTOFF - 1)]
    analyzer.finalize_fft_blur(results)
    assert all(r["is_blurry_fft"] is None for r in results)
    assert "too few images" in analyzer.fft_blur_mode


def test_fft_blur_flags_blurred_images_in_folder(rng):
    """A fixed-camera folder (one scene, varying light and sensor noise) with a few blurred frames:
    every blurred frame is flagged and scores below every sharp frame."""
    analyzer = ImageQualityAnalyzer(use_laplacian_blur=False, use_brightness=False, use_contrast=False,
                                    use_exposure_clipping=False, resize_percent=100.0)
    scene = cv2.GaussianBlur(rng.integers(0, 256, size=(128, 128)).astype(np.float64), (0, 0), 1.5)
    n_sharp, n_blurred = MIN_IMAGES_FOR_AUTO_CUTOFF + 5, 3
    images = []
    for i in range(n_sharp + n_blurred):
        frame = np.clip(scene * rng.uniform(0.7, 1.2) + rng.normal(0, 3, scene.shape), 0, 255).astype(np.uint8)
        if i >= n_sharp:
            frame = cv2.GaussianBlur(frame, (0, 0), 4)
        images.append(frame)

    results = [analyzer.analyze(img) for img in images]
    analyzer.finalize_fft_blur(results)

    sharp, blurred = results[:n_sharp], results[n_sharp:]
    assert all(r["is_blurry_fft"] for r in blurred)
    assert all("Blurry" in r["reason"] for r in blurred)
    assert max(r["blur_fft"] for r in blurred) < min(r["blur_fft"] for r in sharp)
