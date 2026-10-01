from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import ExifTags, Image

from grime2py.core import (
    CalibExecutive,
    CalibrationConfig,
    adjust_search_polygon_for_target,
    content_bottom,
    detect_water_level,
    form_octagon_calib_json_string,
    _search_lines,
)


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _load_calibration(path: str) -> CalibrationConfig:
    return CalibrationConfig.from_json(Path(path).read_text(encoding="utf-8"))


def _search_poly_for_image(image, config: CalibrationConfig):
    """Search polygon for this frame, with target tracking applied.

    Shared with the GUI so both surfaces produce the same answer; the GUI
    used to skip tracking entirely.
    """
    search_poly = config.search_poly
    movement = (0.0, 0.0)
    if len(config.pixel_to_world_points) == 8:
        reference_points = [point[:2] for point in config.pixel_to_world_points]
        try:
            search_poly = adjust_search_polygon_for_target(image, search_poly, reference_points)
            movement = (
                search_poly[0][0] - config.search_poly[0][0],
                search_poly[0][1] - config.search_poly[0][1],
            )
        except (ValueError, cv2.error):
            pass
    return search_poly, movement


def _detect_image(path: Path, config: CalibrationConfig) -> dict[str, object]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not read image: {path}")
    search_poly, movement = _search_poly_for_image(image, config)
    result = detect_water_level(image, search_poly)
    record: dict[str, object] = {
        "source": str(path),
        "found": result.found,
        "waterline_pixel_y": result.y,
        "angle_pixel": result.angle,
        "confidence": result.confidence,
        "target_movement_x": movement[0],
        "target_movement_y": movement[1],
        "message": "; ".join(result.messages),
    }
    if result.center is not None and config.pixel_to_world_points:
        calibration = CalibExecutive()
        pixels = np.asarray([point[:2] for point in config.pixel_to_world_points], dtype=np.float32)
        worlds = np.asarray([point[2:] for point in config.pixel_to_world_points], dtype=np.float32)
        calibration.calibrate_from_points(pixels, worlds, image_size=(image.shape[1], image.shape[0]))
        world = calibration.pixel_to_world(result.center)
        record["waterline_world_x"] = world[0]
        record["waterline_world_y"] = world[1]
    return record


def _write_result_image(source: Path, destination: str, config: CalibrationConfig) -> None:
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"could not read image: {source}")
    search_poly, _movement = _search_poly_for_image(image, config)
    result = detect_water_level(image, search_poly)
    if result.endpoints is not None:
        start, end = result.endpoints
        cv2.line(
            image,
            (round(start[0]), round(start[1])),
            (round(end[0]), round(end[1])),
            (0, 0, 255),
            3,
        )
    if not cv2.imwrite(destination, image):
        raise ValueError(f"could not write result image: {destination}")


def _write_csv(path: str, records: list[dict[str, object]]) -> None:
    if not records:
        return
    fieldnames = list(records[0])
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _read_metadata(source: str) -> dict[str, object]:
    path = Path(source)
    with Image.open(path) as image:
        metadata: dict[str, object] = {
            "source": str(path),
            "width": image.width,
            "height": image.height,
            "format": image.format,
        }
        exif = image.getexif()
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            metadata[tag] = value
    return metadata


def _make_gif(source: str, destination: str, delay_ms: int, scale: float | None) -> int:
    paths = sorted(path for path in Path(source).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"no images found in {source}")
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frame = image.convert("RGB")
            if scale is not None:
                if scale <= 0:
                    raise ValueError("--scale must be positive")
                frame = frame.resize((max(1, round(frame.width * scale)), max(1, round(frame.height * scale))))
            frames.append(frame.copy())
    frames[0].save(
        destination,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=max(0, delay_ms),
        loop=0,
    )
    return len(frames)


def _calibrate_image(source: str, destination: str, facet_length: float, zero_offset: float) -> None:
    image = cv2.imread(source)
    if image is None:
        raise ValueError(f"could not read image: {source}")
    calibration = CalibExecutive()
    if not calibration.calibrate_octagon_image(image, facet_length, zero_offset):
        raise ValueError("octagon calibration failed")
    points = calibration.model.pixel_points
    min_x = max(0, int(min(point[0] for point in points) - 100))
    max_x = min(image.shape[1] - 1, int(max(point[0] for point in points) + 300))
    top_y = min(image.shape[0] - 2, int(max(point[1] for point in points) + 100))
    # Stop above the camera's caption banner, not at h-20. See content_bottom.
    bottom_y = content_bottom(image)
    config = CalibrationConfig(
        facet_length=facet_length,
        zero_offset=zero_offset,
        target_roi=(
            max(0, int(min(point[0] for point in points) - 40)),
            max(0, int(min(point[1] for point in points) - 40)),
            int(max(point[0] for point in points) - min(point[0] for point in points) + 80),
            int(max(point[1] for point in points) - min(point[1] for point in points) + 80),
        ),
        search_poly=((min_x, top_y), (max_x, top_y), (min_x, bottom_y), (max_x, bottom_y)),
        image_size=(image.shape[1], image.shape[0]),
        search_lines=tuple(
            (*line[0], *line[1])
            for line in _search_lines(image.shape[:2], ((min_x, top_y), (max_x, top_y), (min_x, bottom_y), (max_x, bottom_y)))
        ),
        pixel_to_world_points=tuple(
            (*pixel, *world)
            for pixel, world in zip(calibration.model.pixel_points, calibration.model.world_points)
        ),
    )
    Path(destination).write_text(form_octagon_calib_json_string(config), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRIME2 Python CLI", add_help=False)
    parser.add_argument("--version", dest="command", action="store_const", const="version")
    parser.add_argument("--help", dest="command", action="store_const", const="help")
    parser.add_argument("--calibrate", dest="command", action="store_const", const="calibrate")
    parser.add_argument("--create_calib", dest="command", action="store_const", const="create_calib")
    parser.add_argument("--find_line", dest="command", action="store_const", const="find_line")
    parser.add_argument("--run_folder", dest="command", action="store_const", const="run_folder")
    parser.add_argument("--make_gif", dest="command", action="store_const", const="make_gif")
    parser.add_argument("--show_metadata", dest="command", action="store_const", const="show_metadata")

    parser.add_argument("--source", dest="source")
    parser.add_argument("--calib_json", dest="calib_json")
    parser.add_argument("--result_image", dest="result_image")
    parser.add_argument("--csv_file", dest="csv_file")
    parser.add_argument("--timestamp_from_exif", action="store_true")
    parser.add_argument("--timestamp_from_filename", action="store_true")
    parser.add_argument("--timestamp_start_pos", type=int)
    parser.add_argument("--timestamp_format")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--zero_offset", type=float)
    parser.add_argument("--delay_ms", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_calib_save", action="store_true")
    parser.add_argument("--cache_result", action="store_true")
    parser.add_argument("--line_roi_folder")
    parser.add_argument("--logFile")
    parser.add_argument("--facet_length", type=float)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "command", None):
        args.command = "help"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "version":
        print("GRIME2 Python CLI v0.1.0")
        return 0

    if args.command == "help":
        print(
            "GRIME2 Python CLI\n"
            "Supported commands:\n"
            "  --version\n"
            "  --calibrate --source <image> --calib_json <json>\n"
            "  --find_line --source <image> --calib_json <json>\n"
            "  --run_folder --source <folder> --calib_json <json>\n"
            "  --make_gif --source <folder> --result_image <gif>\n"
            "  --show_metadata --source <image>\n"
        )
        return 0

    if args.command in {"calibrate", "create_calib"}:
        if not args.source or not args.calib_json:
            raise SystemExit("--calibrate requires --source and --calib_json")
        _calibrate_image(args.source, args.calib_json, args.facet_length or args.scale or 0.599, args.zero_offset or 0.0)
        print(f"Calibration written to {args.calib_json}")
        return 0

    if args.command == "find_line":
        if not args.source or not args.calib_json:
            raise SystemExit("--find_line requires --source and --calib_json")
        config = _load_calibration(args.calib_json)
        record = _detect_image(Path(args.source), config)
        if args.csv_file:
            _write_csv(args.csv_file, [record])
        if args.result_image:
            _write_result_image(Path(args.source), args.result_image, config)
        print(json.dumps(record))
        return 0

    if args.command == "run_folder":
        if not args.source or not args.calib_json:
            raise SystemExit("--run_folder requires --source and --calib_json")
        config = _load_calibration(args.calib_json)
        source = Path(args.source)
        records = [_detect_image(path, config) for path in sorted(source.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]
        if args.csv_file:
            _write_csv(args.csv_file, records)
        for record in records:
            print(json.dumps(record))
        return 0

    if args.command == "make_gif":
        if not args.source or not args.result_image:
            raise SystemExit("--make_gif requires --source and --result_image")
        count = _make_gif(args.source, args.result_image, args.delay_ms or 100, args.scale)
        print(f"Created GIF with {count} frames: {args.result_image}")
        return 0

    if args.command == "show_metadata":
        if not args.source:
            raise SystemExit("--show_metadata requires --source")
        print(json.dumps(_read_metadata(args.source), default=str))
        return 0

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
