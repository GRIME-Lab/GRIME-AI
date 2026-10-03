from __future__ import annotations

from typing import Sequence


def bresenham(
    point0: Sequence[int],
    point1: Sequence[int],
    max_points: int = 999_999_999,
) -> list[tuple[int, int]]:
    """Port the legacy GRIME2 Bresenham helper.

    The C++ helper excludes the final endpoint and is intended for the
    rightward/upward line walks used by octagon search.
    """
    x0, y0 = int(point0[0]), int(point0[1])
    x1, y1 = int(point1[0]), int(point1[1])
    if (x0, y0) == (x1, y1):
        raise ValueError("line endpoints must differ")
    if max_points <= 0:
        return []

    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    error = dx - dy
    x, y = x0, y0
    while (x, y) != (x1, y1) and len(points) < max_points:
        points.append((x, y))
        double_error = 2 * error
        if double_error > -dy:
            error -= dy
            x += sx
        if double_error < dx:
            error += dx
            y += sy
    return points
