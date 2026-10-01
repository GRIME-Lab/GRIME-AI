"""Octagon corner refinement.

Corner ordering is delegated to core.sort_octagon_points so that this
module and core.detect_octagon_points cannot disagree. They used to use
two different orderings, and refinement runs inside a try/except, so a
refinement failure silently rotated the pixel-to-world correspondence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class LineEquation:
    direction: tuple[float, float]
    point: tuple[float, float]


class OctoRefine:
    @staticmethod
    def sort_octagon_points(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
        from grime2py.core import sort_octagon_points as _sort
        return _sort(points)

    def get_point_projection(self, point: Sequence[float], first: Sequence[float], second: Sequence[float]) -> tuple[float, float]:
        p = np.asarray(point, dtype=float)
        a = np.asarray(first, dtype=float)
        direction = np.asarray(second, dtype=float) - a
        denominator = float(np.dot(direction, direction))
        if denominator <= 1e-12:
            raise ValueError("projection line endpoints must differ")
        return tuple(map(float, a + direction * np.dot(p - a, direction) / denominator))

    def get_line_pixels(self, first: Sequence[float], second: Sequence[float], center: Sequence[float]) -> list[tuple[int, int]]:
        """Pixels along the facet edge. `center` is accepted for signature
        compatibility with the C++; the normal it used to compute was
        never applied."""
        a = np.asarray(first, dtype=float)
        b = np.asarray(second, dtype=float)
        if float(np.linalg.norm(b - a)) <= 1e-12:
            return []
        return self.get_line_coords(np.rint(a).astype(int), np.rint(b).astype(int))

    @staticmethod
    def get_line_coords(first: Sequence[int], second: Sequence[int]) -> list[tuple[int, int]]:
        x0, y0 = map(int, first)
        x1, y1 = map(int, second)
        steps = max(abs(x1 - x0), abs(y1 - y0))
        if steps == 0:
            return [(x0, y0)]
        values = []
        for fraction in np.linspace(0.0, 1.0, steps + 1):
            point = (round(x0 + fraction * (x1 - x0)), round(y0 + fraction * (y1 - y0)))
            if not values or point != values[-1]:
                values.append(point)
        return values

    def calculate_facet_lines(self, center: Sequence[float], points: Sequence[Sequence[float]], extension: int = 20) -> list[list[tuple[tuple[int, int], tuple[int, int]]]]:
        ordered = np.asarray(self.sort_octagon_points(points), dtype=float)
        center_array = np.asarray(center, dtype=float)
        facets = []
        for index in range(8):
            first = ordered[index]
            second = ordered[(index + 1) % 8]
            edge = second - first
            normal = np.asarray([-edge[1], edge[0]], dtype=float)
            if np.dot(normal, center_array - (first + second) / 2.0) < 0:
                normal = -normal
            normal /= max(1e-12, np.linalg.norm(normal))
            pixels = self.get_line_coords(np.rint(first).astype(int), np.rint(second).astype(int))
            if len(pixels) > 16:
                pixels = pixels[2:-2]
            scanlines = []
            for x, y in pixels:
                inner = np.rint(np.asarray([x, y], dtype=float) + normal * extension).astype(int)
                outer = np.rint(np.asarray([x, y], dtype=float) - normal * extension).astype(int)
                scanlines.append((tuple(map(int, inner)), tuple(map(int, outer))))
            facets.append(scanlines)
        return facets

    @staticmethod
    def _smooth(values: np.ndarray, sigma: float) -> np.ndarray:
        size = max(3, int(6 * sigma + 1) | 1)
        kernel = cv2.getGaussianKernel(size, sigma, cv2.CV_64F).reshape(-1)
        radius = size // 2
        padded = np.pad(values.astype(np.float64), (radius, radius), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def find_subpixel_falling_edge(self, image: np.ndarray, first: Sequence[int], second: Sequence[int], sigma: float = 2.0) -> tuple[float, float] | None:
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("octagon refinement requires an 8-bit grayscale image")
        coords = self.get_line_coords(first, second)
        if len(coords) < 3:
            return None
        intensities = np.asarray([image[y, x] if 0 <= x < image.shape[1] and 0 <= y < image.shape[0] else 0 for x, y in coords], dtype=float)
        smooth = self._smooth(intensities, sigma)
        gradient = np.empty_like(smooth)
        gradient[0] = smooth[1] - smooth[0]
        gradient[-1] = smooth[-1] - smooth[-2]
        gradient[1:-1] = 0.5 * (smooth[2:] - smooth[:-2])
        # argmax = strongest RISING edge along the scan. The scan runs from
        # inside the octagon outward, so the blue-to-black transition is
        # rising in this direction despite the function name (kept to match
        # the C++ symbol). Flagged rather than changed: flipping it would
        # move every refined corner.
        index = int(np.argmax(gradient))
        if index == 0 or index == len(gradient) - 1:
            return None
        denominator = gradient[index - 1] - 2.0 * gradient[index] + gradient[index + 1]
        offset = 0.0 if abs(denominator) < 1e-12 else 0.5 * (gradient[index - 1] - gradient[index + 1]) / denominator
        position = max(0.0, min(float(len(coords) - 1), index + offset))
        left = np.asarray(coords[int(np.floor(position))], dtype=float)
        right_index = min(len(coords) - 1, int(np.ceil(position)))
        right = np.asarray(coords[right_index], dtype=float)
        fraction = position - np.floor(position)
        result = left + fraction * (right - left)
        return float(result[0]), float(result[1])

    def find_line_intersection(self, first: LineEquation, second: LineEquation) -> tuple[bool, tuple[float, float] | None]:
        direction_a = np.asarray(first.direction, dtype=float)
        direction_b = np.asarray(second.direction, dtype=float)
        delta = np.asarray(second.point, dtype=float) - np.asarray(first.point, dtype=float)
        determinant = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
        if abs(determinant) < 1e-6:
            return False, None
        t = (delta[0] * direction_b[1] - delta[1] * direction_b[0]) / determinant
        point = np.asarray(first.point, dtype=float) + t * direction_a
        return True, tuple(map(float, point))

    def get_octagon_vertices(self, lines: Sequence[LineEquation]) -> list[tuple[float, float]]:
        if len(lines) != 8:
            raise ValueError("exactly eight facet lines are required")
        vertices = []
        for index in range(8):
            valid, point = self.find_line_intersection(lines[index], lines[(index + 1) % 8])
            if not valid or point is None:
                raise ValueError("octagon facet lines are parallel")
            vertices.append(point)
        return self.sort_octagon_points(vertices)

    def refine_points(self, image: np.ndarray, points: Sequence[Sequence[float]], min_facet_points: int = 8, sigma: float = 2.0) -> list[tuple[float, float]]:
        gray = cv2.medianBlur(image, 7) if image.ndim == 2 else cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 7)
        ordered = self.sort_octagon_points(points)
        center = np.mean(np.asarray(ordered), axis=0)
        facets = self.calculate_facet_lines(center, ordered)
        lines = []
        for scanlines in facets:
            edges = [self.find_subpixel_falling_edge(gray, inner, outer, sigma) for inner, outer in scanlines]
            valid = np.asarray([edge for edge in edges if edge is not None], dtype=np.float32)
            if len(valid) < min_facet_points:
                raise ValueError("not enough valid edge points on octagon facet")
            vector = cv2.fitLine(valid, cv2.DIST_L1, 0, 0.01, 0.01).reshape(-1)
            lines.append(LineEquation((float(vector[0]), float(vector[1])), (float(vector[2]), float(vector[3]))))
        return self.get_octagon_vertices(lines)
