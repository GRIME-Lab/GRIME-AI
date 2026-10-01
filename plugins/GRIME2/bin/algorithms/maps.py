from __future__ import annotations

import cv2
import numpy as np


def _validate_kernel(image: np.ndarray, kernel_size: int) -> int:
    if kernel_size < 3:
        raise ValueError("kernel size must be at least 3")
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
    if image.ndim != 2 or image.shape[0] < kernel_size or image.shape[1] < kernel_size:
        raise ValueError("image must be single-channel and larger than the kernel")
    return kernel_size


def create_variance_map(
    image: np.ndarray,
    kernel_size: int,
    mask: np.ndarray | None = None,
    floatscale: float = -1.0,
) -> np.ndarray:
    """Local variance map: E[x^2] - E[x]^2 over a box window.

    Mathematically identical to the previous per-pixel Python loop and to
    the C++ integral-image version, but vectorised. The loop ran 2.4M
    iterations on a 2048x1152 frame.
    """
    kernel_size = _validate_kernel(image, kernel_size)
    if mask is not None and mask.shape != image.shape:
        raise ValueError("mask must have the same shape as image")
    source = image.astype(np.float64, copy=False)
    if mask is not None:
        source = np.where(mask != 0, source, 0.0)

    window = (kernel_size, kernel_size)
    mean = cv2.boxFilter(source, cv2.CV_64F, window, normalize=True,
                         borderType=cv2.BORDER_REFLECT)
    mean_square = cv2.boxFilter(source * source, cv2.CV_64F, window,
                                normalize=True, borderType=cv2.BORDER_REFLECT)
    result = np.maximum(0.0, mean_square - mean * mean).astype(np.float32)

    # The border was left at zero by the loop version; keep that, since
    # callers threshold against it.
    radius = kernel_size // 2
    if radius:
        result[:radius, :] = 0.0
        result[-radius:, :] = 0.0
        result[:, :radius] = 0.0
        result[:, -radius:] = 0.0

    if floatscale > 0:
        maximum = float(result.max())
        if maximum > 0:
            return np.clip(result * (255.0 * floatscale / maximum), 0, 255).astype(np.uint8)
        return np.zeros_like(result, dtype=np.uint8)
    return result


def calc_entropy_map(
    image: np.ndarray,
    kernel_size: int,
    use_ellipse: bool = True,
) -> np.ndarray:
    """Count of occupied 5-bit intensity bins in the window, per pixel.

    Despite the historical name this is a bin count, not Shannon entropy;
    the C++ does the same. Computed as one box filter per bin over a
    binary membership image, which fills EVERY pixel. The previous port
    stepped by 4 and left three quarters of the output as zeros.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("entropy input must be an 8-bit grayscale image")
    if kernel_size < 3 or kernel_size > 1024:
        raise ValueError("kernel size must be between 3 and 1024")

    bins = image >> 3                      # 32 bins, as the C++ uses
    window = (kernel_size, kernel_size)
    occupied = np.zeros(image.shape, dtype=np.float32)
    if use_ellipse:
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, window)
        for value in range(32):
            member = (bins == value).astype(np.uint8)
            present = cv2.dilate(member, element)   # any member in the window
            occupied += present.astype(np.float32)
    else:
        for value in range(32):
            member = (bins == value).astype(np.float32)
            count = cv2.boxFilter(member, cv2.CV_32F, window, normalize=False,
                                  borderType=cv2.BORDER_REFLECT)
            occupied += (count > 0.5).astype(np.float32)
    return occupied
