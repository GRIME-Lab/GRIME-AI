from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from grime2py.core import detect_octagon_points
from grime2py.algorithms.octorefine import OctoRefine


@dataclass
class OctagonSearchResult:
    found: bool
    points: list[tuple[float, float]]
    message: str = ""


class OctagonSearch:
    """Public Python facade for GRIME2 octagon search and refinement."""

    def __init__(self) -> None:
        self.template_dim = 51
        self.rotate_count = 5
        self.last_result = OctagonSearchResult(False, [], "not run")

    def init(self, template_dim: int = 51, rotate_count: int = 5) -> None:
        if template_dim < 3 or rotate_count < 1:
            raise ValueError("invalid octagon template parameters")
        self.template_dim = template_dim
        self.rotate_count = rotate_count

    def find(
        self,
        image: np.ndarray,
        target_roi: tuple[int, int, int, int] | None = None,
        refine: bool = True,
    ) -> OctagonSearchResult:
        try:
            points = detect_octagon_points(image, target_roi)
            if refine:
                try:
                    points = OctoRefine().refine_points(image, points)
                except (ValueError, cv2.error):
                    pass
            self.last_result = OctagonSearchResult(True, points, "octagon found")
        except (ValueError, cv2.error) as error:
            self.last_result = OctagonSearchResult(False, [], str(error))
        return self.last_result

    def find_scale(self, image: np.ndarray, scale: float, target_roi: tuple[int, int, int, int] | None = None) -> OctagonSearchResult:
        if scale <= 0:
            raise ValueError("scale must be positive")
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        result = self.find(resized, None, refine=True)
        if result.found:
            result.points = [(x / scale, y / scale) for x, y in result.points]
        return result
