from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np


@dataclass
class CalibrationConfig:
    calib_type: str = "Octagon"
    facet_length: float = 1.0
    zero_offset: float = 2.0
    target_roi: tuple[int, int, int, int] = (-1, -1, -1, -1)
    search_poly: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] = (
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
    )
    calib_result_json: str = ""
    pixel_to_world_points: tuple[tuple[float, float, float, float], ...] = ()
    image_size: tuple[int, int] = (0, 0)
    search_lines: tuple[tuple[float, float, float, float], ...] = ()
    control_json: str = ""

    @classmethod
    def from_json(cls, payload: str) -> "CalibrationConfig":
        data = json.loads(payload)
        waterline = data.get("WaterlineSearchRegion", {})
        target = data.get("TargetSearchRegion", {})
        calibration_points = tuple(
            (
                float(point["pixelX"]),
                float(point["pixelY"]),
                float(point["worldX"]),
                float(point["worldY"]),
            )
            for point in data.get("PixelToWorld", {}).get("points", [])
        )
        search_poly = (
            (
                int(waterline.get("toplft_x", data.get("searchPoly_lftTop_x", -1))),
                int(waterline.get("toplft_y", data.get("searchPoly_lftTop_y", -1))),
            ),
            (
                int(waterline.get("toprgt_x", data.get("searchPoly_rgtTop_x", -1))),
                int(waterline.get("toprgt_y", data.get("searchPoly_rgtTop_y", -1))),
            ),
            (
                int(waterline.get("botlft_x", data.get("searchPoly_lftBot_x", -1))),
                int(waterline.get("botlft_y", data.get("searchPoly_lftBot_y", -1))),
            ),
            (
                int(waterline.get("botrgt_x", data.get("searchPoly_rgtBot_x", -1))),
                int(waterline.get("botrgt_y", data.get("searchPoly_rgtBot_y", -1))),
            ),
        )
        return cls(
            calib_type=data.get("calibType", "Octagon"),
            facet_length=float(data.get("facetLength", -1.0)),
            zero_offset=float(data.get("zeroOffset", 0.0)),
            target_roi=(
                int(target.get("x", data.get("targetRoi_x", -1))),
                int(target.get("y", data.get("targetRoi_y", -1))),
                int(target.get("width", data.get("targetRoi_width", -1))),
                int(target.get("height", data.get("targetRoi_height", -1))),
            ),
            search_poly=search_poly,
            calib_result_json=data.get("calibResult_json", ""),
            pixel_to_world_points=calibration_points,
            image_size=(int(data.get("imageWidth", 0)), int(data.get("imageHeight", 0))),
            search_lines=tuple(
                (
                    float(line["topX"]),
                    float(line["topY"]),
                    float(line["botX"]),
                    float(line["botY"]),
                )
                for line in data.get("SearchLines", [])
            ),
            control_json=data.get("control_json", ""),
        )


@dataclass
class CalibrationModel:
    valid: bool = False
    image_size: tuple[int, int] = (0, 0)
    pixel_points: list[tuple[float, float]] = field(default_factory=list)
    world_points: list[tuple[float, float]] = field(default_factory=list)
    facet_length: float = 1.0
    zero_offset: float = 0.0
    center_pixel: tuple[float, float] = (-1.0, -1.0)
    center_world: tuple[float, float] = (-1.0, -1.0)
    angle: float = 0.0
    homography_pixel_to_world: np.ndarray | None = None
    homography_world_to_pixel: np.ndarray | None = None
    reprojection_rms: float = float("nan")
    reprojection_max: float = float("nan")


@dataclass
class WaterLineResult:
    found: bool
    y: float
    angle: float = 0.0
    center: tuple[float, float] | None = None
    endpoints: tuple[tuple[float, float], tuple[float, float]] | None = None
    confidence: float = 0.0
    candidates: tuple[tuple[int, float], ...] = ()
    messages: list[str] = field(default_factory=list)


class CalibExecutive:
    # World units. The octagon facet is ~0.6, so 5% of a facet is a
    # generous bound that still catches a mis-ordered or spurious corner.
    max_reprojection_rms = 0.03

    def __init__(self) -> None:
        self.model = CalibrationModel()

    def _estimate_affine(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        if src.shape != dst.shape or src.shape[1] != 2:
            raise ValueError("source and destination points must share the same 2D shape")
        if len(src) < 3:
            raise ValueError("at least three point pairs are required")

        rows = []
        targets = []
        for point_src, point_dst in zip(src, dst):
            x, y = point_src
            u, v = point_dst
            rows.append([x, y, 1.0, 0.0, 0.0, 0.0])
            rows.append([0.0, 0.0, 0.0, x, y, 1.0])
            targets.extend([u, v])

        design = np.asarray(rows, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        affine = np.array([
            [coeffs[0], coeffs[1], coeffs[2]],
            [coeffs[3], coeffs[4], coeffs[5]],
        ], dtype=np.float64)
        return affine

    def calibrate_from_points(
        self,
        pixel_points: Sequence[Sequence[float]],
        world_points: Sequence[Sequence[float]],
        image_size: tuple[int, int] | None = None,
    ) -> bool:
        if len(pixel_points) != len(world_points):
            raise ValueError("pixel and world point counts must match")
        if len(pixel_points) < 4:
            raise ValueError("need at least 4 calibration points")

        src = np.asarray(pixel_points, dtype=np.float32)
        dst = np.asarray(world_points, dtype=np.float32)

        if image_size is None:
            image_size = (int(np.max(src[:, 0])) + 100, int(np.max(src[:, 1])) + 100)

        self.model.image_size = image_size
        self.model.pixel_points = [tuple(map(float, pt)) for pt in src]
        self.model.world_points = [tuple(map(float, pt)) for pt in dst]

        try:
            homography_pixel_to_world, _ = cv2.findHomography(src, dst, method=0)
            homography_world_to_pixel, _ = cv2.findHomography(dst, src, method=0)
        except (ValueError, cv2.error):
            self.model.valid = False
            return False
        if homography_pixel_to_world is None or homography_world_to_pixel is None:
            self.model.valid = False
            return False

        # Reprojection error. The octagon over-determines the homography,
        # so the residual is free and is the natural per-frame quality
        # number. The first port set valid=True unconditionally, so one bad
        # corner warped the whole mapping with no indication.
        projected = cv2.perspectiveTransform(
            src.reshape(-1, 1, 2), homography_pixel_to_world).reshape(-1, 2)
        errors = np.linalg.norm(projected - dst, axis=1)
        self.model.reprojection_rms = float(np.sqrt(np.mean(errors ** 2)))
        self.model.reprojection_max = float(errors.max())
        if self.model.reprojection_rms > self.max_reprojection_rms:
            self.model.valid = False
            return False

        self.model.valid = True
        # facet_length and zero_offset belong to the CALLER. The first port
        # recomputed them from the world points, replacing the user's facet
        # length with a point spacing and the zero offset with a centroid.

        centroid = np.mean(src, axis=0)
        self.model.center_pixel = (float(centroid[0]), float(centroid[1]))
        self.model.center_world = (float(np.mean(dst[:, 0])), float(np.mean(dst[:, 1])))

        # Angle from the first facet (points 0->1), which is a real edge of
        # the target. The first port took the principal axis of all eight
        # corners by SVD; an octagon is nearly isotropic, so the two
        # singular values are nearly equal and that angle was noise.
        edge = src[1] - src[0]
        self.model.angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
        self.model.homography_pixel_to_world = homography_pixel_to_world
        self.model.homography_world_to_pixel = homography_world_to_pixel
        return True

    def calibrate_octagon_image(
        self,
        image: np.ndarray,
        facet_length: float,
        zero_offset: float = 0.0,
        target_roi: tuple[int, int, int, int] | None = None,
    ) -> bool:
        pixel_points = detect_octagon_points(image, target_roi)
        try:
            from grime2py.algorithms.octorefine import OctoRefine

            pixel_points = OctoRefine().refine_points(image, pixel_points)
        except (ValueError, cv2.error):
            pass
        world_points = octagon_world_points(facet_length, zero_offset)
        self.model.facet_length = float(facet_length)
        self.model.zero_offset = float(zero_offset)
        return self.calibrate_from_points(
            pixel_points, world_points, (image.shape[1], image.shape[0]))

    def pixel_to_world(self, pixel: Sequence[float]) -> tuple[float, float]:
        if not self.model.valid:
            raise ValueError("calibration model is not valid")
        if self.model.homography_pixel_to_world is not None:
            point = np.asarray([[pixel]], dtype=np.float32)
            projected = cv2.perspectiveTransform(point, self.model.homography_pixel_to_world).reshape(-1)
            return float(projected[0]), float(projected[1])
        src = np.asarray(self.model.pixel_points, dtype=np.float64)
        dst = np.asarray(self.model.world_points, dtype=np.float64)
        affine = self._estimate_affine(src, dst)
        point = np.asarray([pixel[0], pixel[1], 1.0], dtype=np.float64)
        projected = point @ affine.T
        return float(projected[0]), float(projected[1])

    def world_to_pixel(self, world: Sequence[float]) -> tuple[float, float]:
        if not self.model.valid:
            raise ValueError("calibration model is not valid")
        if self.model.homography_world_to_pixel is not None:
            point = np.asarray([[world]], dtype=np.float32)
            projected = cv2.perspectiveTransform(point, self.model.homography_world_to_pixel).reshape(-1)
            return float(projected[0]), float(projected[1])
        src = np.asarray(self.model.world_points, dtype=np.float64)
        dst = np.asarray(self.model.pixel_points, dtype=np.float64)
        affine = self._estimate_affine(src, dst)
        point = np.asarray([world[0], world[1], 1.0], dtype=np.float64)
        projected = point @ affine.T
        return float(projected[0]), float(projected[1])


def octagon_world_points(facet_length: float, zero_offset: float = 0.0) -> list[tuple[float, float]]:
    corner_length = float(facet_length) / np.sqrt(2.0)
    points = [
        (0.0, 0.0),
        (facet_length, 0.0),
        (facet_length + corner_length, -corner_length),
        (facet_length + corner_length, -facet_length - corner_length),
        (facet_length, -2.0 * corner_length - facet_length),
        (0.0, -2.0 * corner_length - facet_length),
        (-corner_length, -corner_length - facet_length),
        (-corner_length, -corner_length),
    ]
    return [(x, y + zero_offset) for x, y in points]


def sort_octagon_points(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """The single canonical corner order, shared with OctoRefine.

    The first port had two different orderings: this module sorted by
    topmost-then-leftmost while OctoRefine sorted by angle about the
    centroid. Refinement runs inside a try/except, so whether it succeeded
    decided which ordering reached the world points -- a silent rotation of
    the correspondence, and a silently wrong homography.
    """
    values = np.asarray(points, dtype=np.float64)
    if len(values) != 8:
        raise ValueError("exactly eight octagon points are required")
    center = values.mean(axis=0)
    angles = np.arctan2(values[:, 1] - center[1], values[:, 0] - center[0])
    ordered = values[np.argsort(angles)]
    top = np.argsort(ordered[:, 1])[:2]
    start = int(top[np.argmin(ordered[top, 0])])
    ordered = np.roll(ordered, -start, axis=0)
    return [tuple(map(float, point)) for point in ordered]


# Shape validation for the blue blob. Aspect alone rejects the patches of
# water that were being picked up on bluish frames.
OCTAGON_MIN_AREA = 500
OCTAGON_ASPECT = (0.85, 1.18)
OCTAGON_FILL = (0.40, 0.95)


def _octagon_from_contour(contour) -> np.ndarray | None:
    """Reduce a contour to exactly eight corners.

    A single fixed approxPolyDP epsilon is brittle -- it returns 7 or 9
    vertices depending on blur, exposure and how much of the black border
    survived thresholding. Binary-searching epsilon finds the tolerance
    that yields eight for THIS contour instead of hoping one constant fits
    every frame.
    """
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0:
        return None
    low, high = 0.001, 0.12
    best = None
    for _ in range(40):
        mid = (low + high) / 2.0
        polygon = cv2.approxPolyDP(hull, mid * perimeter, True).reshape(-1, 2)
        count = len(polygon)
        if count == 8:
            return polygon.astype(np.float64)
        if count > 8:
            low = mid          # too much detail, loosen
        else:
            high = mid         # too few corners, tighten
        if best is None or abs(count - 8) < abs(len(best) - 8):
            best = polygon
    return None


def _blue_mask(hsv: np.ndarray) -> np.ndarray:
    """Blue/cyan octagon face. The range is deliberately wide: the target
    photographs anywhere from deep blue to pale cyan depending on light,
    white balance and which target revision is installed."""
    mask = cv2.inRange(hsv, np.asarray([80, 45, 30]), np.asarray([140, 255, 255]))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


# Shape validation for the blue blob. Aspect alone rejects the patches of
# water that get picked up on bluish frames.
OCTAGON_MIN_AREA = 500
OCTAGON_ASPECT = (0.70, 1.40)
OCTAGON_FILL = (0.35, 0.98)


def _detect_octagon_in(hsv: np.ndarray, offset_x: int, offset_y: int,
                       report: list[str]) -> list[tuple[float, float]] | None:
    mask = _blue_mask(hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    examined = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < OCTAGON_MIN_AREA:
            continue
        examined += 1
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(hull)
        if w <= 0 or h <= 0:
            continue
        aspect = w / float(h)
        fill = hull_area / float(w * h)
        if not OCTAGON_ASPECT[0] <= aspect <= OCTAGON_ASPECT[1]:
            report.append("blob %dx%d@(%d,%d) rejected: aspect %.2f"
                          % (w, h, x + offset_x, y + offset_y, aspect))
            continue
        if not OCTAGON_FILL[0] <= fill <= OCTAGON_FILL[1]:
            report.append("blob %dx%d@(%d,%d) rejected: fill %.2f"
                          % (w, h, x + offset_x, y + offset_y, fill))
            continue
        polygon = _octagon_from_contour(contour)
        if polygon is None:
            report.append("blob %dx%d@(%d,%d) rejected: no 8-corner fit"
                          % (w, h, x + offset_x, y + offset_y))
            continue
        if best is None or hull_area > best[0]:
            best = (hull_area, polygon)
    if best is None:
        if examined == 0:
            report.append("no blue region larger than %d px" % OCTAGON_MIN_AREA)
        return None
    polygon = best[1]
    polygon[:, 0] += offset_x
    polygon[:, 1] += offset_y
    return sort_octagon_points(polygon)


def detect_octagon_points(
    image: np.ndarray,
    target_roi: tuple[int, int, int, int] | None = None,
) -> list[tuple[float, float]]:
    """Detect the blue octagon face and return canonically ordered corners.

    When a target ROI is supplied and nothing is found inside it, the whole
    frame is searched before giving up. A caller-supplied ROI is a hint,
    not a constraint, and a slightly wrong one used to abort calibration
    outright even though the octagon was plainly visible elsewhere.
    """
    if image.ndim == 2:
        hsv = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
    elif image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("image must be grayscale or BGR")

    report: list[str] = []
    if target_roi is not None:
        x, y, width, height = (int(v) for v in target_roi)
        if width > 0 and height > 0:
            x0, y0 = max(0, x), max(0, y)
            x1 = min(hsv.shape[1], x + width)
            y1 = min(hsv.shape[0], y + height)
            if x1 - x0 > 8 and y1 - y0 > 8:
                found = _detect_octagon_in(hsv[y0:y1, x0:x1], x0, y0, report)
                if found is not None:
                    return found
                report.insert(0, "not found in target ROI (%d,%d %dx%d);"
                                 " retried whole frame" % (x, y, width, height))

    found = _detect_octagon_in(hsv, 0, 0, report)
    if found is not None:
        return found
    raise ValueError("could not detect an eight-sided octagon. "
                     + ("; ".join(report) if report else "no blue region found"))


def adjust_search_polygon_for_target(
    image: np.ndarray,
    search_poly: Sequence[Sequence[float]],
    reference_octagon_points: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    """Translate a calibrated search polygon to follow target movement in a new frame."""
    if len(reference_octagon_points) != 8:
        raise ValueError("reference octagon must contain eight points")
    current_points = detect_octagon_points(image)
    reference_center = np.mean(np.asarray(reference_octagon_points, dtype=np.float64), axis=0)
    current_center = np.mean(np.asarray(current_points, dtype=np.float64), axis=0)
    offset = current_center - reference_center
    return tuple((float(x + offset[0]), float(y + offset[1])) for x, y in search_poly)


def _search_lines(
    image_shape: tuple[int, int],
    search_poly: Sequence[Sequence[float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    height, width = image_shape
    if len(search_poly) != 4:
        raise ValueError("search polygon must contain four points")

    corners = sorted(((float(x), float(y)) for x, y in search_poly), key=lambda point: point[1])
    top = sorted(corners[:2], key=lambda point: point[0])
    bottom = sorted(corners[2:], key=lambda point: point[0])
    left_top, right_top = top
    left_bottom, right_bottom = bottom
    line_count = max(2, int(round(max(right_top[0] - left_top[0], right_bottom[0] - left_bottom[0])) + 1))
    lines = []
    for index in range(line_count):
        fraction = index / (line_count - 1)
        top_x = left_top[0] + fraction * (right_top[0] - left_top[0])
        bottom_x = left_bottom[0] + fraction * (right_bottom[0] - left_bottom[0])
        top_y = left_top[1] + fraction * (right_top[1] - left_top[1])
        bottom_y = left_bottom[1] + fraction * (right_bottom[1] - left_bottom[1])
        lines.append(((top_x, top_y), (bottom_x, bottom_y)))
    return [
        ((max(0.0, min(width - 1.0, top[0])), max(0.0, min(height - 1.0, top[1]))),
         (max(0.0, min(width - 1.0, bottom[0])), max(0.0, min(height - 1.0, bottom[1]))))
        for top, bottom in lines
    ]



# ======================================================================
# Waterline detection, restored to match GRIME2's C++ FindLine
#
# The first Python port dropped four things the C++ does. They are back:
#   1. One preprocess, including the tall-kernel morphological closing.
#   2. Row SUMS median-filtered before differencing (CalcRowSums).
#   3. Sub-pixel swath points (CalcSwathPoint).
#   4. RANSAC line fit with an angle-bound inlier test and an
#      interquartile trim, plus the RemoveOutliers retry (FitLineRANSAC).
#
# Deliberate divergences from the C++, each fixing a bug there:
#   * swaths cover the width evenly (C++ integer division left a gap and
#     overlapped the last swath)
#   * every search line is resampled to a common length (C++ used line 0's
#     height for all lines, misaligning sloped ROIs)
#   * the sub-pixel offset is real parabolic interpolation (the C++
#     expression is dimensionally inconsistent)
#   * index bounds are checked (C++ reads rowSums[index-2] with size_t
#     arithmetic, which underflows when index == 1)
# ======================================================================
MEDIAN_FILTER_KERN_SIZE = 5
TRIAGE_MAX_DEV = 17.0           # px from median Y, as the C++ uses
RANSAC_TRIES = 200
RANSAC_SAMPLE = 5
RANSAC_MIN_VALID = 9
MIN_LINE_ANGLE = -9.0           # degrees
MAX_LINE_ANGLE = 9.0


def content_bottom(image: np.ndarray, dark_threshold: float = 30.0,
                   margin: int = 12) -> int:
    """Last usable row above any solid overlay bar at the bottom of the frame.

    Trail cameras stamp a black caption strip along the bottom edge. A
    search polygon that reaches into it sees a ~100-level step into solid
    black, which outscores every real waterline: with a polygon ending at
    h-20 the detector returned row 1120 on 12 of 12 labelled frames. This
    walks up from the bottom past rows whose mean grey is below the
    threshold and returns a row safely above them.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    row_mean = gray.mean(axis=1)
    y = gray.shape[0] - 1
    while y > 0 and row_mean[y] < dark_threshold:
        y -= 1
    return max(0, y - margin)


def preprocess_for_line(gray: np.ndarray) -> np.ndarray:
    """The single preprocess used by every waterline path.

    Mirrors C++ FindLine::Preprocess: blur, median, then a closing with a
    tall 5x11 kernel. The first port had this function in findline.py but
    never called it, and did a different, weaker preprocess in the path
    that actually ran.
    """
    dst = cv2.GaussianBlur(gray, (11, 11), 3.0)
    dst = cv2.medianBlur(dst, 23)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 11))
    dst = cv2.dilate(dst, kernel, iterations=2)
    dst = cv2.erode(dst, kernel, iterations=2)
    return dst


def _median_filter_1d(values: np.ndarray, kernel: int) -> np.ndarray:
    """Median filter the row-sum profile, as C++ CalcRowSums does before
    the profile is differenced."""
    kernel = max(3, kernel | 1)
    radius = kernel // 2
    padded = np.pad(values.astype(np.float64), radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
    return np.median(windows, axis=1)


def _subpixel_peak(profile: np.ndarray, index: int) -> float:
    """Parabolic interpolation of the |difference| peak at `index`.

    The C++ computes |dd1| / (|dd1*dd1| + dd2*dd2), which squares a value
    and then takes its absolute value, and is not dimensionally a fraction
    of a sample. This is the standard three-point parabola instead.
    """
    if index <= 0 or index >= len(profile) - 1:
        return float(index)
    y0, y1, y2 = float(profile[index - 1]), float(profile[index]), float(profile[index + 1])
    denominator = y0 - 2.0 * y1 + y2
    if abs(denominator) < 1e-12:
        return float(index)
    offset = 0.5 * (y0 - y2) / denominator
    if not -1.0 < offset < 1.0:
        return float(index)
    return float(index) + offset


def _fit_line_ransac(points: np.ndarray, rng=None):
    """C++ FitLineRANSAC: random minimal fits, keep those inside the angle
    bounds, average the interquartile middle of the survivors.

    Returns (slope, intercept, spread, n_valid) or None. `spread` is the
    standard deviation of the surviving lines' centre Y -- a real per-frame
    confidence number, which the first port had no equivalent of.
    """
    if len(points) < RANSAC_SAMPLE:
        return None
    xs = points[:, 0].astype(np.float64)
    ys = points[:, 1].astype(np.float64)
    if rng is None:
        rng = np.random.default_rng(12345)      # deterministic results
    x_center = float(xs.mean())
    valid = []
    for attempt in range(RANSAC_TRIES):
        if attempt == 0 and len(xs) == RANSAC_SAMPLE:
            idx = np.arange(len(xs))
        else:
            idx = rng.choice(len(xs), RANSAC_SAMPLE, replace=False)
        sx, sy = xs[idx], ys[idx]
        if np.ptp(sx) < 1e-6:
            continue
        slope, intercept = np.polyfit(sx, sy, 1)
        angle = float(np.degrees(np.arctan(slope)))
        if not MIN_LINE_ANGLE <= angle <= MAX_LINE_ANGLE:
            continue                    # the angle bound IS the inlier test
        valid.append((slope * x_center + intercept, slope, intercept))
    if len(valid) < RANSAC_MIN_VALID:
        return None
    valid.sort(key=lambda item: item[0])
    cut = len(valid) // 4
    middle = valid[cut:len(valid) - cut] or valid
    slope = float(np.mean([item[1] for item in middle]))
    intercept = float(np.mean([item[2] for item in middle]))
    spread = float(np.std([item[0] for item in middle]))
    return slope, intercept, spread, len(valid)


def _triage_points(points: np.ndarray) -> np.ndarray:
    """Keep points within TRIAGE_MAX_DEV of the median Y (C++ TriagePoints)."""
    if len(points) < 5:
        return points
    median_y = float(np.median(points[:, 1]))
    return points[np.abs(points[:, 1] - median_y) < TRIAGE_MAX_DEV]


def _remove_outliers(points: np.ndarray, keep: int = 5) -> np.ndarray:
    """Keep the `keep` points closest to the median Y (C++ RemoveOutliers).

    This is the retry path the first port omitted entirely: the C++ falls
    back to it when triage-plus-RANSAC fails, so frames that used to be
    recoverable were being reported as failures.
    """
    if len(points) <= keep:
        return points
    median_y = float(np.median(points[:, 1]))
    order = np.argsort(np.abs(points[:, 1] - median_y))
    kept = points[order[:keep]]
    return kept[np.argsort(kept[:, 0])]


def _fit_waterline(points: np.ndarray, image_width: int, roi_x0: float = None,
                   roi_x1: float = None):
    """Fit the waterline and report Y at the CENTRE OF THE MEASURED SPAN.

    The first port extrapolated the fitted line to x=0 and x=width-1 and
    reported the midpoint of that extrapolation, so the answer changed if
    the image was cropped, and the endpoints were far outside any column
    that had been measured.
    """
    if len(points) < 2:
        raise ValueError("at least two waterline points are required")
    if roi_x0 is None:
        roi_x0 = float(points[:, 0].min())
    if roi_x1 is None:
        roi_x1 = float(points[:, 0].max())

    fit = _fit_line_ransac(points)
    if fit is not None:
        slope, intercept, spread, n_valid = fit
    else:
        vector = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        vx, vy, x0, y0 = map(float, vector)
        if abs(vx) < 1e-6:
            raise ValueError("waterline points are vertical; not a water surface")
        slope = vy / vx
        intercept = y0 - slope * x0
        spread = float("nan")
        n_valid = 0

    left_y = slope * roi_x0 + intercept
    right_y = slope * roi_x1 + intercept
    angle = float(np.degrees(np.arctan(slope)))
    center_x = (roi_x0 + roi_x1) / 2.0
    center_y = float(slope * center_x + intercept)
    endpoints = ((float(roi_x0), float(left_y)), (float(roi_x1), float(right_y)))
    return center_y, angle, endpoints, spread, n_valid


def detect_water_level_in_roi(
    image: np.ndarray,
    search_poly: Sequence[Sequence[float]],
) -> WaterLineResult:
    """GRIME2 swath search, with the C++ steps the first port dropped."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("image must be grayscale or BGR")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    filtered = preprocess_for_line(gray)
    lines = _search_lines(gray.shape[:2], search_poly)
    if len(lines) < 2:
        return WaterLineResult(found=False, y=float("nan"),
                               messages=["search polygon produced no search lines"])

    # Every line is resampled to a common length. The C++ used line 0's
    # height for all lines, which misaligns row indices on a sloped ROI.
    sample_count = min(max(2, int(round(np.hypot(bot[0] - top[0], bot[1] - top[1]))))
                       for top, bot in lines)
    if sample_count < 8:
        return WaterLineResult(found=False, y=float("nan"),
                               messages=["search region is not tall enough"])
    samples = np.zeros((len(lines), sample_count), dtype=np.float32)
    fractions = np.linspace(0.0, 1.0, sample_count)
    for line_index, ((top_x, top_y), (bottom_x, bottom_y)) in enumerate(lines):
        xs = np.clip(np.rint(top_x + fractions * (bottom_x - top_x)).astype(np.int32),
                     0, gray.shape[1] - 1)
        ys = np.clip(np.rint(top_y + fractions * (bottom_y - top_y)).astype(np.int32),
                     0, gray.shape[0] - 1)
        samples[line_index] = filtered[ys, xs]

    swath_count = min(10, len(lines))
    points: list[tuple[float, float]] = []
    bounds = np.array_split(np.arange(len(lines)), swath_count)
    for swath in bounds:
        if swath.size == 0:
            continue
        # Row SUMS, median filtered, exactly as C++ CalcRowSums does. The
        # first port differenced the raw means with no filtering.
        row_sums = _median_filter_1d(samples[swath].sum(axis=0),
                                     MEDIAN_FILTER_KERN_SIZE)
        differences = np.abs(np.diff(row_sums))
        if differences.size < 3:
            continue
        index = int(np.argmax(differences)) + 1
        if index < 2 or index > len(row_sums) - 2:
            continue            # C++ reads row_sums[index-2] unguarded here
        position = _subpixel_peak(differences, index - 1) + 1.0
        fraction = position / max(1.0, sample_count - 1.0)
        first, last = lines[int(swath[0])], lines[int(swath[-1])]
        x_center = (first[0][0] + last[0][0]) / 2.0
        top = ((first[0][0] + last[0][0]) / 2.0, (first[0][1] + last[0][1]) / 2.0)
        bottom = ((first[1][0] + last[1][0]) / 2.0, (first[1][1] + last[1][1]) / 2.0)
        points.append((x_center, top[1] + fraction * (bottom[1] - top[1])))

    if len(points) < 2:
        return WaterLineResult(found=False, y=float("nan"),
                               messages=["no waterline transition found in search polygon"])

    all_points = np.asarray(points, dtype=np.float32)
    roi_x0 = float(min(top[0] for top, _bot in lines))
    roi_x1 = float(max(top[0] for top, _bot in lines))

    # Triage then RANSAC; on failure, RemoveOutliers then RANSAC again.
    # The retry is the C++ recovery path the first port omitted.
    attempts = [("triage", _triage_points(all_points)),
                ("outlier-trim", _remove_outliers(all_points, 5))]
    for label, candidate in attempts:
        if len(candidate) < 2:
            continue
        try:
            y, angle, endpoints, spread, n_valid = _fit_waterline(
                candidate, gray.shape[1], roi_x0, roi_x1)
        except ValueError:
            continue
        if not MIN_LINE_ANGLE <= angle <= MAX_LINE_ANGLE:
            continue
        confidence = 0.0 if spread != spread else float(1.0 / (1.0 + spread))
        return WaterLineResult(
            found=True,
            y=y,
            angle=angle,
            center=((endpoints[0][0] + endpoints[1][0]) / 2.0, y),
            endpoints=endpoints,
            confidence=confidence,
            messages=[f"waterline from {len(candidate)} swath points via {label};"
                      f" ransac valid={n_valid} spread={spread:.2f}px"],
        )
    return WaterLineResult(found=False, y=float("nan"),
                           messages=["waterline points did not form a plausible line"])


def water_level_from_image(image: np.ndarray, min_contrast: float = 4.0) -> float | None:
    """Whole-image fallback. Returns None when there is no real transition.

    The first port returned a row unconditionally and the caller reported
    found=True, so a frame with no waterline at all produced a confident
    wrong answer.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    row_means = gray.mean(axis=1).astype(np.float32)
    row_diffs = np.abs(np.diff(row_means))
    if row_diffs.size == 0 or float(row_diffs.max()) < min_contrast:
        return None
    return float(int(np.argmax(row_diffs)) + 1)


def form_octagon_calib_json_string(config: CalibrationConfig) -> str:
    payload = {
        "calibType": config.calib_type,
        "imageWidth": config.image_size[0],
        "imageHeight": config.image_size[1],
        "facetLength": config.facet_length,
        "zeroOffset": config.zero_offset,
        "drawCalib": 0,
        "drawWaterLineSearchROI": 0,
        "targetRoi_x": config.target_roi[0],
        "targetRoi_y": config.target_roi[1],
        "targetRoi_width": config.target_roi[2],
        "targetRoi_height": config.target_roi[3],
        "searchPoly_lftTop_x": config.search_poly[0][0],
        "searchPoly_lftTop_y": config.search_poly[0][1],
        "searchPoly_rgtTop_x": config.search_poly[1][0],
        "searchPoly_rgtTop_y": config.search_poly[1][1],
        "searchPoly_lftBot_x": config.search_poly[2][0],
        "searchPoly_lftBot_y": config.search_poly[2][1],
        "searchPoly_rgtBot_x": config.search_poly[3][0],
        "searchPoly_rgtBot_y": config.search_poly[3][1],
        "calibResult_json": config.calib_result_json,
        "PixelToWorld": {
            "points": [
                {
                    "pixelX": pixel_x,
                    "pixelY": pixel_y,
                    "worldX": world_x,
                    "worldY": world_y,
                }
                for pixel_x, pixel_y, world_x, world_y in config.pixel_to_world_points
            ]
        },
        "SearchLines": [
            {
                "topX": top_x,
                "topY": top_y,
                "botX": bot_x,
                "botY": bot_y,
            }
            for top_x, top_y, bot_x, bot_y in config.search_lines
        ],
        "control_json": config.control_json,
    }
    return json.dumps(payload)


def detect_water_level(
    image: np.ndarray,
    search_poly: Sequence[Sequence[float]] | None = None,
    prefer: str = "ensemble",
) -> WaterLineResult:
    """Find the waterline. One detector owns the answer, the other checks it.

    prefer="ensemble" (default) uses the band-contrast detector in
    waterline.py; prefer="swath" uses the port of the C++ FindLine swath
    search. On hand-labelled KOLA frames the ensemble lands within 6px on
    most frames while the swath search can miss by hundreds, so the
    ensemble is the default and the swath is kept for comparison.

    Whichever runs second is reported as agreement or disagreement. An
    earlier version returned ONE detector's row with the OTHER's
    confidence and messages, so the reported confidence described a row
    that had been discarded.
    """
    if search_poly is None:
        y = water_level_from_image(image)
        if y is None:
            return WaterLineResult(found=False, y=float("nan"),
                                   messages=["no global row transition found"])
        return WaterLineResult(found=True, y=float(y),
                               center=(image.shape[1] / 2.0, float(y)),
                               confidence=0.0,
                               messages=["waterline by global row transition"
                                         " (uncalibrated fallback)"])

    points = np.asarray(search_poly, dtype=np.float32)
    bbox = (int(points[:, 0].min()), int(points[:, 1].min()),
            int(points[:, 0].max()), int(points[:, 1].max()))

    from grime2py.waterline import estimate_waterline

    try:
        estimate = estimate_waterline(image, bbox)
    except (ValueError, cv2.error) as error:
        estimate = None
        estimate_error = str(error)
    else:
        estimate_error = ""

    def _as_result(est) -> WaterLineResult:
        x0, y0, x1, y1 = est.line
        angle = (float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
                 if x1 != x0 else 0.0)
        return WaterLineResult(
            found=True,
            y=float((y0 + y1) / 2.0),
            angle=angle,
            center=(float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)),
            endpoints=((float(x0), float(y0)), (float(x1), float(y1))),
            confidence=est.confidence,
            candidates=est.candidates,
            messages=[est.reason],
        )

    if prefer == "ensemble":
        if estimate is not None and estimate.line is not None:
            result = _as_result(estimate)
            swath = detect_water_level_in_roi(image, search_poly)
            if swath.found:
                result.messages.append(
                    "swath cross-check y=%.1f (delta %+.1fpx)"
                    % (swath.y, swath.y - result.y))
            else:
                result.messages.append("swath cross-check found nothing")
            return result
        swath = detect_water_level_in_roi(image, search_poly)
        if swath.found:
            swath.messages = list(swath.messages) + [
                "ensemble found nothing" + (f": {estimate_error}" if estimate_error else "")]
        return swath

    # prefer == "swath"
    calibrated = detect_water_level_in_roi(image, search_poly)
    if calibrated.found:
        if estimate is not None and estimate.line is not None:
            other_y = (estimate.line[1] + estimate.line[3]) / 2.0
            calibrated.candidates = estimate.candidates
            calibrated.messages = list(calibrated.messages) + [
                "ensemble cross-check y=%.1f (delta %+.1fpx)"
                % (other_y, other_y - calibrated.y)]
        else:
            calibrated.messages = list(calibrated.messages) + ["ensemble found nothing"]
        return calibrated
    if estimate is not None and estimate.line is not None:
        result = _as_result(estimate)
        result.messages.append("swath search failed; ensemble result used")
        return result
    return calibrated
