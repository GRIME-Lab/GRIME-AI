from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass
class AnchorMatch:
    found: bool
    angle: float = -99999.0
    offset: tuple[int, int] = (-1, -1)
    location: tuple[int, int] = (-1, -1)
    score: float = -1.0


class FindAnchor:
    """Port of GRIME2's rotated reference-anchor template matcher."""

    def __init__(self) -> None:
        self.model_rect: tuple[int, int, int, int] | None = None
        self.model_reference_path = ""
        self.rotated_models: list[tuple[np.ndarray, float]] = []
        self.last_response: np.ndarray | None = None
        self.last_match = AnchorMatch(False)

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 2:
            return image.copy()
        raise ValueError("anchor image must be grayscale or BGR")

    def set_ref(self, image: np.ndarray | str, model_roi: Sequence[int]) -> None:
        if isinstance(image, (str, Path)):
            self.model_reference_path = str(image)
            source = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if source is None:
                raise ValueError(f"could not read anchor reference image: {image}")
        else:
            source = image
        x, y, width, height = map(int, model_roi)
        if width <= 0 or height <= 0:
            raise ValueError("anchor model ROI must have positive dimensions")
        if width + 50 > source.shape[1] or height + 50 > source.shape[0]:
            raise ValueError("anchor model ROI is too large for the reference image")
        gray = cv2.GaussianBlur(self._gray(source), (5, 5), 3.0)
        self.model_rect = (x, y, width, height)
        self.rotated_models = []
        center = (gray.shape[1] / 2.0, gray.shape[0] / 2.0)
        for index in range(-15, 16):
            angle = index / 2.0
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray, matrix, (gray.shape[1] * 2, gray.shape[0] * 2), flags=cv2.INTER_CUBIC)
            self.rotated_models.append((rotated[y:y + height, x:x + width].copy(), angle))
        self.last_response = None
        self.last_match = AnchorMatch(False)

    def find(self, image: np.ndarray) -> AnchorMatch:
        if self.model_rect is None or not self.rotated_models:
            raise ValueError("anchor reference model is not initialized")
        gray = cv2.GaussianBlur(self._gray(image), (5, 5), 3.0)
        best = AnchorMatch(False)
        x, y, _, _ = self.model_rect
        for template, angle in self.rotated_models:
            response = cv2.matchTemplate(gray, template, cv2.TM_CCORR_NORMED)
            _, score, _, location = cv2.minMaxLoc(response)
            if score > best.score:
                best = AnchorMatch(
                    True,
                    float(angle),
                    (int(location[0] - x), int(location[1] - y)),
                    (int(location[0]), int(location[1])),
                    float(score),
                )
                self.last_response = response
        self.last_match = best
        return best

    def calc_move_model(self, image: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], float]:
        result = self.find(image)
        if self.model_rect is None:
            raise ValueError("anchor reference model is not initialized")
        original = (self.model_rect[0], self.model_rect[1])
        return original, result.location, result.angle

    @staticmethod
    def rotate_image(image: np.ndarray, center: Sequence[float], angle: float) -> np.ndarray:
        matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
        return cv2.warpAffine(image, matrix, (image.shape[1] * 2, image.shape[0] * 2), flags=cv2.INTER_CUBIC)
