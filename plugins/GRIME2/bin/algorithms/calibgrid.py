from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass
class TemplateBowtieItem:
    point: tuple[float, float]
    score: float


class FindCalibGrid:
    template_count = 21
    target_count = 8
    minimum_score = 0.05

    def __init__(self) -> None:
        self.templates: list[np.ndarray] = []
        self.matches: list[TemplateBowtieItem] = []
        self.move_left: tuple[int, int, int, int] | None = None
        self.move_right: tuple[int, int, int, int] | None = None

    def clear(self) -> None:
        self.templates.clear()
        self.matches.clear()
        self.move_left = None
        self.move_right = None

    def init_bowtie_template(self, template_dim: int, search_image_size: tuple[int, int]) -> None:
        if not 20 <= template_dim <= 1000:
            raise ValueError("template dimension must be between 20 and 1000")
        even = template_dim + template_dim % 2
        working = even * 2
        base = np.full((working, working), 224, dtype=np.uint8)
        left = np.asarray([[1, 1], [1, working - 2], [working // 2, working // 2]], dtype=np.int32)
        right = np.asarray([[working - 2, 1], [working - 2, working - 2], [working // 2, working // 2]], dtype=np.int32)
        cv2.fillConvexPoly(base, left, 32)
        cv2.fillConvexPoly(base, right, 32)
        center = self.template_count // 2
        x0 = even // 2
        self.templates = []
        for index in range(self.template_count):
            angle = index - center
            matrix = cv2.getRotationMatrix2D((working / 2.0, working / 2.0), angle, 1.0)
            rotated = cv2.warpAffine(base, matrix, (working, working), flags=cv2.INTER_CUBIC)
            self.templates.append(rotated[x0:x0 + even, x0:x0 + even].copy())

    @staticmethod
    def _subpixel(response: np.ndarray, point: tuple[int, int]) -> tuple[float, float]:
        x, y = point
        if x < 1 or y < 1 or x >= response.shape[1] - 1 or y >= response.shape[0] - 1:
            return float(x), float(y)
        window = response[y - 1:y + 2, x - 1:x + 2]
        weights = np.maximum(window, 0.0)
        total = float(weights.sum())
        if total <= 0:
            return float(x), float(y)
        yy, xx = np.indices(window.shape)
        return float((weights * (xx + x - 1)).sum() / total), float((weights * (yy + y - 1)).sum() / total)

    def _match_candidates(
        self,
        image: np.ndarray,
        roi: tuple[int, int, int, int],
        minimum_score: float,
        count: int,
    ) -> list[TemplateBowtieItem]:
        x, y, width, height = roi
        crop = image[y:y + height, x:x + width]
        response = cv2.matchTemplate(crop, self.templates[self.template_count // 2], cv2.TM_CCOEFF_NORMED)
        result: list[TemplateBowtieItem] = []
        for _ in range(count):
            _, score, _, maximum = cv2.minMaxLoc(response)
            if score < minimum_score:
                break
            px, py = maximum
            result.append(TemplateBowtieItem((x + px + self.templates[0].shape[1] / 2.0, y + py + self.templates[0].shape[0] / 2.0), float(score)))
            cv2.circle(response, maximum, 17, 0.0, -1)
        return result

    def _refine(self, image: np.ndarray, item: TemplateBowtieItem, minimum_score: float) -> TemplateBowtieItem:
        height, width = self.templates[0].shape
        x = max(0, round(item.point[0]) - width // 2 - width // 4)
        y = max(0, round(item.point[1]) - height // 2 - height // 4)
        x = min(x, max(0, image.shape[1] - (width + width // 2)))
        y = min(y, max(0, image.shape[0] - (height + height // 2)))
        roi = image[y:y + height + height // 2, x:x + width + width // 2]
        best = item
        for template in self.templates:
            response = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, maximum = cv2.minMaxLoc(response)
            if score > max(minimum_score, best.score):
                refined = self._subpixel(response, maximum)
                best = TemplateBowtieItem((x + refined[0] + width / 2.0, y + refined[1] + height / 2.0), float(score))
        return best

    def find_targets(
        self,
        image: np.ndarray,
        target_roi: tuple[int, int, int, int],
        min_score: float = 0.05,
    ) -> list[TemplateBowtieItem]:
        if not self.templates:
            raise ValueError("bowtie templates are not initialized")
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("calibration grid input must be 8-bit grayscale")
        if not 0.01 <= min_score <= 1.0:
            raise ValueError("minimum score must be between 0.01 and 1.0")
        candidates = self._match_candidates(image, target_roi, min_score, self.target_count * 2)
        self.matches = [self._refine(image, item, min_score) for item in candidates]
        self.matches.sort(key=lambda item: (round(item.point[1] / max(1, self.templates[0].shape[0] * 2)), item.point[0]))
        return self.matches

    def find_move_targets(
        self,
        image: np.ndarray,
        target_roi: tuple[int, int, int, int],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        matches = self.find_targets(image, target_roi, min_score=self.minimum_score)
        if len(matches) < 2:
            raise ValueError("fewer than two calibration-grid targets found")
        ordered = sorted(matches, key=lambda item: item.point[0])
        return ordered[0].point, ordered[-1].point
