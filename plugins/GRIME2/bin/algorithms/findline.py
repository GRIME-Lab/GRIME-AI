from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from grime2py.core import (
    WaterLineResult,
    detect_water_level_in_roi,
    preprocess_for_line,
)
from grime2py.algorithms.searchlines import LineEnds


@dataclass
class FindPointSet:
    angle_pixel: float = -99999.0
    angle_world: float = -99999.0
    left_pixel: tuple[float, float] = (-1.0, -1.0)
    center_pixel: tuple[float, float] = (0.0, 0.0)
    right_pixel: tuple[float, float] = (-1.0, -1.0)
    left_world: tuple[float, float] = (-1.0, -1.0)
    center_world: tuple[float, float] = (-1.0, -1.0)
    right_world: tuple[float, float] = (-1.0, -1.0)


@dataclass
class FindLineResult:
    find_success: bool = False
    calib_success: bool = False
    water_level_adjusted: tuple[float, float] = (-1.0, -1.0)
    calc_line_points: FindPointSet = field(default_factory=FindPointSet)
    found_points: list[tuple[float, float]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    waterline: WaterLineResult | None = None


class FindLine:
    """Structured public facade for the GRIME2 waterline detector."""

    def __init__(self) -> None:
        self.min_line_angle = -9.0
        self.max_line_angle = 9.0

    def set_line_find_angle_bounds(self, minimum: float, maximum: float) -> None:
        if minimum > maximum:
            raise ValueError("minimum angle must not exceed maximum angle")
        self.min_line_angle = minimum
        self.max_line_angle = maximum

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Delegates to core.preprocess_for_line.

        There used to be two preprocessors: this one, which matched the
        C++ and was never called, and a weaker one inside
        detect_water_level_in_roi that actually ran. Now there is one.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return preprocess_for_line(gray)

    def find(self, image: np.ndarray, lines: Sequence[LineEnds] | Sequence[Sequence[Sequence[float]]]) -> FindLineResult:
        if not lines:
            return FindLineResult(messages=["no waterline search lines defined"])
        top_left = lines[0].top if isinstance(lines[0], LineEnds) else lines[0][0]
        top_right = lines[-1].top if isinstance(lines[-1], LineEnds) else lines[-1][0]
        bottom_left = lines[0].bottom if isinstance(lines[0], LineEnds) else lines[0][1]
        bottom_right = lines[-1].bottom if isinstance(lines[-1], LineEnds) else lines[-1][1]
        result = detect_water_level_in_roi(image, (top_left, top_right, bottom_left, bottom_right))
        if not result.found or result.endpoints is None:
            return FindLineResult(messages=result.messages, waterline=result)
        if not self.min_line_angle <= result.angle <= self.max_line_angle:
            return FindLineResult(messages=[f"line angle {result.angle:.3f} outside configured bounds"], waterline=result)
        center = result.center or ((image.shape[1] - 1) / 2.0, result.y)
        points = [center]
        line_points = FindPointSet(
            angle_pixel=result.angle,
            left_pixel=result.endpoints[0],
            center_pixel=center,
            right_pixel=result.endpoints[1],
        )
        return FindLineResult(
            find_success=True,
            water_level_adjusted=center,
            calc_line_points=line_points,
            found_points=points,
            messages=result.messages,
            waterline=result,
        )
