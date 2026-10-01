from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def create_streamflow_model(
    input_csv: str,
    output_csv: str,
    timestamp_format: str,
    timestamp_column: int,
    value_column: int,
    conversion_factor: float = 0.02832,
) -> int:
    """Port the active GRIME2 streamflow-model branch grouped by day of year."""
    groups: dict[int, list[float]] = defaultdict(list)
    seasonal: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with Path(input_csv).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    for row in rows[33:]:
        if len(row) <= max(timestamp_column, value_column):
            continue
        try:
            timestamp = datetime.strptime(row[timestamp_column].strip(), timestamp_format)
            value = float(row[value_column].strip())
        except ValueError:
            continue
        day = timestamp.timetuple().tm_yday
        if timestamp.year < 2014 or (timestamp.year == 2014 and timestamp.month <= 9):
            groups[day].append(value)
        elif (timestamp.year == 2014 and timestamp.month >= 10) or (timestamp.year == 2015 and timestamp.month <= 9):
            seasonal[2015][day].append(value)
        elif (timestamp.year == 2015 and timestamp.month >= 10) or (timestamp.year == 2016 and timestamp.month <= 9):
            seasonal[2016][day].append(value)
        elif (timestamp.year == 2016 and timestamp.month >= 10) or (timestamp.year == 2017 and timestamp.month <= 9):
            seasonal[2017][day].append(value)
    with Path(output_csv).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Day of year", "mean", "median", "2015 mean", "2015 median", "2016 mean", "2016 median", "2017 mean", "2017 median"))
        for day in range(1, 367):
            values = groups[day]
            row: list[float | int] = [day]
            row.extend(_summary(values, conversion_factor))
            for year in (2015, 2016, 2017):
                row.extend(_summary(seasonal[year][day], conversion_factor))
            writer.writerow(row)
    return 366


def _summary(values: list[float], factor: float) -> tuple[float, float]:
    # Deliberate divergence: the C++ divides by an empty day's size and
    # indexes items[size>>1] on an empty vector. It also accumulates sum17
    # inside the 2016 loop, so the 2016 column is wrong there.
    if not values:
        return 0.0, 0.0
    return float(np.mean(values) * factor), float(np.median(values) * factor)
