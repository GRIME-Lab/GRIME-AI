from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

import cv2
import numpy as np


class StaffGaugeTickType(IntEnum):
    BLACK_BOTTOM_LEFT_CORNER = 0
    BLACK_TOP_RIGHT_POINT = 1


@dataclass
class TickItem:
    point: tuple[float, float]
    score: float
    y_interval: float = 0.0
    x_length: float = 0.0


@dataclass
class StaffGaugeResult:
    found: bool
    ticks: list[TickItem] = field(default_factory=list)
    line_start: tuple[float, float] | None = None
    line_end: tuple[float, float] | None = None
    message: str = ""


class FindStaffGauge:
    """OpenCV port of GRIME2's staff-gauge tick detector."""

    template_count = 21
    minimum_score = 0.1
    minimum_points = 5

    def __init__(self) -> None:
        self.templates: list[np.ndarray] = []
        self.last_result = StaffGaugeResult(False, message="not run")

    def _create_templates(self, tick_type: StaffGaugeTickType) -> None:
        template_size = 14
        working_size = template_size * 2
        if tick_type == StaffGaugeTickType.BLACK_TOP_RIGHT_POINT:
            base = np.ones((working_size, working_size), dtype=np.uint8)
            region = base[template_size // 2:template_size + template_size // 2, template_size // 2:template_size + template_size // 2]
            contour = np.asarray([
                [0, region.shape[0] // 2],
                [region.shape[1] // 2, region.shape[0] // 2],
                [0, region.shape[0] - 1],
            ], dtype=np.int32)
            cv2.drawContours(region, [contour], -1, 255, cv2.FILLED)
            base = cv2.bitwise_not(base)
        elif tick_type == StaffGaugeTickType.BLACK_BOTTOM_LEFT_CORNER:
            base = np.zeros((working_size, working_size), dtype=np.uint8)
            base[:, :template_size] = 255
            base[template_size:, :] = 255
            base = cv2.bitwise_not(base)
        else:
            raise ValueError("unsupported staff-gauge tick type")

        center = self.template_count // 2
        roi = (template_size // 2, template_size // 2, template_size, template_size)
        self.templates = []
        for index in range(self.template_count):
            angle = index - center
            matrix = cv2.getRotationMatrix2D((working_size / 2.0, working_size / 2.0), angle, 1.0)
            rotated = cv2.warpAffine(base, matrix, (working_size, working_size), flags=cv2.INTER_CUBIC)
            x, y, width, height = roi
            self.templates.append(rotated[y:y + height, x:x + width].copy())

    @staticmethod
    def _subpixel(match_space: np.ndarray, maximum: tuple[int, int]) -> tuple[float, float]:
        x, y = maximum
        if x < 1 or y < 1 or x >= match_space.shape[1] - 1 or y >= match_space.shape[0] - 1:
            return float(x), float(y)
        window = match_space[y - 1:y + 2, x - 1:x + 2]
        weights = np.maximum(window, 0.0)
        total = float(weights.sum())
        if total <= 0:
            return float(x), float(y)
        yy, xx = np.indices(window.shape)
        return float((weights * (xx + x - 1)).sum() / total), float((weights * (yy + y - 1)).sum() / total)

    @staticmethod
    def _line(points: np.ndarray, image_shape: tuple[int, int]) -> tuple[tuple[float, float], tuple[float, float]]:
        vector = cv2.fitLine(points.astype(np.float32), cv2.DIST_L12, 0, 0.01, 0.01).reshape(-1)
        vx, vy, x0, y0 = map(float, vector)
        if abs(vx) < 1e-8:
            return (x0, 0.0), (x0, float(image_shape[0] - 1))
        slope = vy / vx
        return (0.0, y0 - slope * x0), (float(image_shape[1] - 1), y0 + slope * (image_shape[1] - 1 - x0))

    @staticmethod
    def _distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
        p = np.asarray(point, dtype=float)
        a = np.asarray(start, dtype=float)
        b = np.asarray(end, dtype=float)
        direction = b - a
        denominator = float(np.linalg.norm(direction))
        if denominator == 0:
            return 0.0
        # 2-D cross product written out. np.cross on 2-D inputs is
        # deprecated in NumPy 2 and slated for removal.
        offset = a - p
        cross = float(direction[0] * offset[1] - direction[1] * offset[0])
        return abs(cross) / denominator

    def find(self, image: np.ndarray, tick_type: StaffGaugeTickType = StaffGaugeTickType.BLACK_TOP_RIGHT_POINT) -> StaffGaugeResult:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError("image must be grayscale or BGR")
        if gray.dtype != np.uint8:
            raise ValueError("staff-gauge image must be uint8")
        self._create_templates(tick_type)
        response = cv2.matchTemplate(gray, self.templates[self.template_count // 2], cv2.TM_CCOEFF_NORMED)
        detections: list[TickItem] = []
        suppressed = response.copy()
        for _ in range(26 if tick_type == StaffGaugeTickType.BLACK_TOP_RIGHT_POINT else 22):
            _, score, _, maximum = cv2.minMaxLoc(suppressed)
            if score < self.minimum_score:
                break
            x, y = maximum
            detections.append(TickItem((x + 7.0, y + 7.0), float(score)))
            cv2.circle(suppressed, maximum, 17, 0, -1)
        detections.sort(key=lambda item: item.point[1])
        refined: list[TickItem] = []
        for item in detections:
            best = item
            rect_x = max(0, round(item.point[0]) - 10)
            rect_y = max(0, round(item.point[1]) - 10)
            for template in self.templates:
                if rect_y + 21 > gray.shape[0] or rect_x + 21 > gray.shape[1]:
                    continue
                roi = gray[rect_y:rect_y + 21, rect_x:rect_x + 21]
                local = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, maximum = cv2.minMaxLoc(local)
                if score > best.score:
                    point = self._subpixel(local, maximum)
                    best = TickItem((rect_x + point[0] + 7.0, rect_y + point[1] + 7.0), float(score))
            refined.append(best)
        if len(refined) < self.minimum_points:
            self.last_result = StaffGaugeResult(False, refined, message=f"only {len(refined)} staff ticks found")
            return self.last_result
        for index in range(1, len(refined)):
            refined[index].y_interval = refined[index].point[1] - refined[index - 1].point[1]
        points = np.asarray([item.point for item in refined], dtype=np.float32)
        line_start, line_end = self._line(points, gray.shape[:2])
        for item in refined:
            item.x_length = self._distance(item.point, line_start, line_end)
        self.last_result = StaffGaugeResult(True, refined, line_start, line_end, "staff ticks found")
        return self.last_result
