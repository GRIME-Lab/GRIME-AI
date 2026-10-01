from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass
class SymbolMatch:
    point: tuple[float, float]
    score: float


class FindSymbol:
    angle_max = 17.5
    angle_increment = 0.5

    def __init__(self) -> None:
        self.templates: list[np.ndarray] = []
        self.matches: list[SymbolMatch] = []
        self.pixel_points: list[tuple[float, float]] = []
        self.world_points: list[tuple[float, float]] = []
        self.homography_pixel_to_world: np.ndarray | None = None
        self.homography_world_to_pixel: np.ndarray | None = None

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 2:
            return image.copy()
        raise ValueError("symbol image must be grayscale or BGR")

    def create_templates(
        self,
        image: np.ndarray,
        template_rect: tuple[int, int, int, int],
        search_rect: tuple[int, int, int, int],
    ) -> None:
        gray = self._gray(image)
        x, y, width, height = template_rect
        sx, sy, swidth, sheight = search_rect
        if width < 10 or height < 10 or swidth < 100 or sheight < 100:
            raise ValueError("symbol template/search region is too small")
        if x < 0 or y < 0 or x + width > gray.shape[1] or y + height > gray.shape[0]:
            raise ValueError("symbol template region is outside the image")
        template = gray[y:y + height, x:x + width].copy()
        _, template = cv2.threshold(template, 3, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        template = cv2.erode(template, None, iterations=2)
        template = cv2.dilate(template, None, iterations=3)
        template = cv2.erode(template, None, iterations=1)
        rotate_x = x - width // 2
        rotate_y = y - height // 2
        if rotate_x < 0 or rotate_y < 0 or rotate_x + width * 2 > gray.shape[1] or rotate_y + height * 2 > gray.shape[0]:
            raise ValueError("symbol rotation region is outside the image")
        source = gray[rotate_y:rotate_y + height * 2, rotate_x:rotate_x + width * 2]
        center = len(np.arange(-self.angle_max, self.angle_max + self.angle_increment / 2, self.angle_increment)) // 2
        self.templates = []
        for index, angle in enumerate(np.arange(-self.angle_max, self.angle_max + self.angle_increment / 2, self.angle_increment)):
            matrix = cv2.getRotationMatrix2D((source.shape[1] / 2.0, source.shape[0] / 2.0), float(angle), 1.0)
            rotated = cv2.warpAffine(source, matrix, source.shape[::-1], flags=cv2.INTER_CUBIC)
            self.templates.append(rotated[height // 2:height // 2 + height, width // 2:width // 2 + width].copy())
        if len(self.templates) != 71 or center != 35:
            raise ValueError("unexpected symbol template count")

    def find_targets(self, image: np.ndarray, target_roi: tuple[int, int, int, int], min_score: float = 0.05) -> list[SymbolMatch]:
        if not self.templates:
            raise ValueError("symbol templates are not initialized")
        gray = self._gray(image)
        x, y, width, height = target_roi
        crop = gray[y:y + height, x:x + width]
        response = cv2.matchTemplate(crop, self.templates[len(self.templates) // 2], cv2.TM_CCOEFF_NORMED)
        matches: list[SymbolMatch] = []
        for _ in range(len(self.templates) * 2):
            _, score, _, maximum = cv2.minMaxLoc(response)
            if score < min_score:
                break
            px, py = maximum
            matches.append(SymbolMatch((x + px + self.templates[0].shape[1] / 2.0, y + py + self.templates[0].shape[0] / 2.0), float(score)))
            cv2.circle(response, maximum, 17, 0.0, -1)
        self.matches = matches
        return matches

    def calibrate(self, pixel_points: Sequence[Sequence[float]], world_points: Sequence[Sequence[float]]) -> None:
        if len(pixel_points) != len(world_points) or len(pixel_points) < 4:
            raise ValueError("at least four matching symbol calibration points are required")
        self.pixel_points = [tuple(map(float, point)) for point in pixel_points]
        self.world_points = [tuple(map(float, point)) for point in world_points]
        self.homography_pixel_to_world, _ = cv2.findHomography(np.asarray(pixel_points, np.float32), np.asarray(world_points, np.float32))
        self.homography_world_to_pixel, _ = cv2.findHomography(np.asarray(world_points, np.float32), np.asarray(pixel_points, np.float32))
        if self.homography_pixel_to_world is None or self.homography_world_to_pixel is None:
            raise ValueError("symbol calibration homography could not be estimated")
