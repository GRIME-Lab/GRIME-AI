from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


@dataclass
class KalmanParams:
    datetime_format: str
    output_csv_filepath: str
    input_csv_filepath: str
    first_data_row: int = 0
    datetime_column: int = 0
    measurement_column: int = 1
    time_string_start: int = 0

    @classmethod
    def from_json(cls, payload: str) -> "KalmanParams":
        data = json.loads(payload)
        return cls(
            datetime_format=data["datetime_format"],
            output_csv_filepath=data["output_csv_filepath"],
            input_csv_filepath=data["input_csv_filepath"],
            first_data_row=int(data.get("first_data_row", 0)),
            datetime_column=int(data["datetime_column"]),
            measurement_column=int(data["measurement_column"]),
            time_string_start=int(data.get("time_string_start", 0)),
        )

    def to_json(self) -> str:
        return json.dumps({
            "datetime_format": self.datetime_format,
            "output_csv_filepath": self.output_csv_filepath,
            "input_csv_filepath": self.input_csv_filepath,
            "first_data_row": self.first_data_row,
            "datetime_column": self.datetime_column,
            "measurement_column": self.measurement_column,
            "time_string_start": self.time_string_start,
        }, indent=2)


def _timestamp(value: str, params: KalmanParams) -> float:
    value = value[params.time_string_start:]
    return datetime.strptime(value, params.datetime_format).timestamp()


def apply_kalman(params: KalmanParams) -> int:
    with Path(params.input_csv_filepath,).open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.reader(stream) if row and not row[0].lstrip().startswith("#")]
    if not rows:
        raise ValueError("input CSV contains no data")

    output_rows: list[tuple[str, float, float]] = []
    first = rows[params.first_data_row]
    first_time = _timestamp(first[params.datetime_column], params)
    first_measurement = float(first[params.measurement_column])
    output_rows.append((first[params.datetime_column], first_measurement, first_measurement))

    filter_ = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
    filter_.transitionMatrix = np.asarray(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    filter_.measurementMatrix = np.eye(2, 4, dtype=np.float32)
    filter_.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-6
    filter_.measurementNoiseCov = np.eye(2, dtype=np.float32) * 20
    filter_.errorCovPost = np.eye(4, dtype=np.float32)
    # NOTE: epoch seconds in a float32 state quantise to ~128 s near 1.7e9,
    # and transitionMatrix hardcodes dt=1 while state[0] carries absolute
    # epoch seconds, so the constant-velocity term is not dimensionally
    # meaningful. Both faults are inherited from the C++ and are kept so
    # results stay comparable; fixing them changes every output value.
    filter_.statePre = np.asarray([[first_time], [first_measurement], [0], [0]], dtype=np.float32)
    filter_.statePost = filter_.statePre.copy()

    for row in rows[params.first_data_row + 1:]:
        measurement = float(row[params.measurement_column])
        if measurement <= -1.0:
            continue
        seconds = _timestamp(row[params.datetime_column], params)
        filter_.predict()
        corrected = filter_.correct(np.asarray([[seconds], [measurement]], dtype=np.float32))
        output_rows.append((row[params.datetime_column], measurement, float(corrected[1, 0])))

    with Path(params.output_csv_filepath).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Timestamp", "measured", "estimated"))
        writer.writerows(output_rows)
    return len(output_rows)


def apply_kalman_from_file(json_path: str) -> int:
    return apply_kalman(KalmanParams.from_json(Path(json_path).read_text(encoding="utf-8")))
