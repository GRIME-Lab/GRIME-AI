from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from grime2py.algorithms.maps import calc_entropy_map


@dataclass
class PixelStats:
    centroid: tuple[float, float] = (-1.0, -1.0)
    average: float = -1.0
    sigma: float = -1.0
    vertical_gradient: float = -1.0
    horizontal_gradient: float = -1.0


@dataclass
class ImageAreaFeatures:
    name: str = ""
    image_size: tuple[int, int] = (-1, -1)
    gray_stats: PixelStats = field(default_factory=PixelStats)
    entropy_stats: PixelStats = field(default_factory=PixelStats)
    lab_stats: list[PixelStats] = field(default_factory=list)
    mask_contour: list[tuple[int, int]] = field(default_factory=list)


def _masked_stats(image: np.ndarray, mask: np.ndarray | None) -> PixelStats:
    valid = np.ones(image.shape, dtype=bool) if mask is None else mask != 0
    values = image[valid]
    if values.size == 0:
        return PixelStats()
    ys, xs = np.nonzero(valid)
    weights = values.astype(np.float64)
    total = float(weights.sum())
    centroid = (float((xs * weights).sum() / total), float((ys * weights).sum() / total)) if total else (-1.0, -1.0)
    return PixelStats(
        centroid=centroid,
        average=float(values.mean()),
        sigma=float(values.std()),
    )


def calc_image_features(image: np.ndarray, mask: np.ndarray | None = None) -> ImageAreaFeatures:
    """Port GRIME2 AreaFeatures for a grayscale/BGR image and optional mask."""
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("image must be grayscale or BGR")
    if mask is not None and mask.shape != gray.shape:
        raise ValueError("mask must match image dimensions")
    entropy = calc_entropy_map(gray.astype(np.uint8), 5, True)
    result = ImageAreaFeatures(image_size=(image.shape[1], image.shape[0]))
    result.gray_stats = _masked_stats(gray, mask)
    result.entropy_stats = _masked_stats(entropy, mask)
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        result.lab_stats = [_masked_stats(channel, mask) for channel in cv2.split(lab)]
    return result


def calc_masked_features(
    image: np.ndarray,
    polygons: Sequence[Sequence[Sequence[int]]],
    names: Sequence[str] | None = None,
) -> list[ImageAreaFeatures]:
    features = []
    for index, polygon in enumerate(polygons):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        contour = np.asarray(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [contour], 255)
        item = calc_image_features(image, mask)
        item.name = names[index] if names and index < len(names) else ""
        item.mask_contour = [tuple(map(int, point)) for point in contour]
        features.append(item)
    return features
