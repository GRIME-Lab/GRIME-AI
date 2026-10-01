from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LineEnds:
    top: tuple[float, float]
    bottom: tuple[float, float]

    @property
    def length(self) -> float:
        return float(np.hypot(self.bottom[0] - self.top[0], self.bottom[1] - self.top[1]))


def calculate_search_lines(
    search_corners: Sequence[Sequence[float]],
    minimum_length: float = 120.0,
) -> list[LineEnds]:
    """Generate dense vertical search lines from the four C++ ROI corners."""
    if len(search_corners) != 4:
        raise ValueError("search line corners must contain four points")
    corners = sorted(((float(x), float(y)) for x, y in search_corners), key=lambda point: point[1])
    top = sorted(corners[:2], key=lambda point: point[0])
    bottom = sorted(corners[2:], key=lambda point: point[0])
    left_top, right_top = top
    left_bottom, right_bottom = bottom
    top_width = right_top[0] - left_top[0]
    bottom_width = right_bottom[0] - left_bottom[0]
    width = max(int(round(top_width)), int(round(bottom_width)))
    if width <= 0:
        raise ValueError("search region must have positive width")
    lines: list[LineEnds] = []
    for index in range(width + 1):
        fraction = index / width
        top_point = (
            left_top[0] + fraction * top_width,
            left_top[1] + fraction * (right_top[1] - left_top[1]),
        )
        bottom_point = (
            left_bottom[0] + fraction * bottom_width,
            left_bottom[1] + fraction * (right_bottom[1] - left_bottom[1]),
        )
        line = LineEnds(top_point, bottom_point)
        if line.length < minimum_length:
            raise ValueError("search region is not tall enough")
        lines.append(line)
    return lines
