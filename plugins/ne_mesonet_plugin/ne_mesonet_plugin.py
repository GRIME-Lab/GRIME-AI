#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ne_mesonet_plugin.py
# EXPERIMENTAL plugin: downloads weather, soil and wind time series and the latest camera images from the
# Nebraska MESONET website, using internal, unpublished endpoints that may change without notice.
#
# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC

import os
import re
import io
import json
import time
import datetime
import traceback

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate, QAbstractTableModel, QModelIndex
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox, QCheckBox,
    QGroupBox, QProgressBar, QFileDialog, QMessageBox, QLineEdit, QListWidget, QListWidgetItem,
    QTabWidget, QTableView, QPlainTextEdit, QDateEdit, QScrollArea, QWidget, QSizePolicy, QFrame
)

try:
    from appcore.geomaps.NEMESONET import NEMesonet, PERIODS_BY_RESOLUTION, EARLIEST_DATA_DATE
except ImportError:
    from GRIME_AI.geomaps.NEMESONET import NEMesonet, PERIODS_BY_RESOLUTION, EARLIEST_DATA_DATE

# The host's settings helper, with a local stand-in so this plugin still runs
# on its own. Same behavior either way: one JSON file named after the plugin.
try:
    from appcore.plugins import plugin_settings
except ImportError:
    class _Settings(dict):
        def __init__(self, path):
            super().__init__()
            self.path = path
            try:
                with open(path, "r") as handle:
                    self.update(json.load(handle))
            except Exception:
                pass

        def save(self):
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "w") as handle:
                    json.dump(dict(self), handle, indent=4)
            except Exception as err:
                print(f"[ne_mesonet_plugin] Could not save settings: {err}")

        def bind(self, widget, key, default=None):
            from PyQt5.QtWidgets import (QLineEdit, QCheckBox, QComboBox,
                                         QSpinBox, QDoubleSpinBox, QSlider)
            pairs = {QLineEdit: ("setText", "text", "editingFinished"),
                     QCheckBox: ("setChecked", "isChecked", "toggled"),
                     QComboBox: ("setCurrentIndex", "currentIndex", "currentIndexChanged"),
                     QSpinBox: ("setValue", "value", "valueChanged"),
                     QDoubleSpinBox: ("setValue", "value", "valueChanged"),
                     QSlider: ("setValue", "value", "valueChanged")}
            for widget_type, (setter, getter, signal) in pairs.items():
                if isinstance(widget, widget_type):
                    value = self.get(key, default)
                    if value is not None:
                        getattr(widget, setter)(value)
                    getattr(widget, signal).connect(
                        lambda *_, w=widget, k=key, g=getter: (self.__setitem__(k, getattr(w, g)()),
                                                               self.save()))
                    return self
            return self

    def plugin_settings(plugin_file, defaults=None):
        folder = os.path.dirname(os.path.abspath(plugin_file))
        return _Settings(os.path.join(
            folder, os.path.splitext(os.path.basename(plugin_file))[0] + ".json"))

PLUGIN = {
    "title":       "Nebraska MESONET (UNOFFICIAL API, EXPERIMENTAL)",
    "class":       "NEMesonetPlugin",
    "description": "Download Nebraska MESONET weather, soil, wind and camera images (unofficial, experimental)",
    "surface":     "tools",
    "size":        [1280, 860],
    "api_version": 2,
}

WARNING_TEXT = "Uses internal, unpublished Nebraska MESONET website endpoints that may change without notice."

# ----------------------------------------------------------------------------------------------------------------------
# SETTINGS (prototype: module level, not saved to the settings file)
# ----------------------------------------------------------------------------------------------------------------------
REQUEST_TIMEOUT_SEC  = 30
REQUEST_DELAY_SEC    = 0.5     # pause between requests to the MESONET site
LEFT_PANEL_WIDTH     = 340
VARIABLE_LIST_CHARS  = 40      # width of the chart Variable list, in characters
THUMB_WIDTH          = 300
THUMB_COLUMNS        = 2       # camera thumbnails per row in the Camera tab
STATION_LIST_HEIGHT  = 170
TABLE_PREVIEW_ROWS   = 5000    # rows shown in the Table tab (files always get every row)

# Resolutions offered (minute is left out: untested, and 5-minute covers the need).
RESOLUTION_LABELS = {"5-minute": "5 minutes", "10-minute": "10 minutes", "hourly": "Hourly", "daily": "Daily"}
PERIOD_LABELS = {
    "past-hour": "Past hour", "today": "Today", "yesterday": "Yesterday", "last-3-days": "Last 3 days",
    "last-7-days": "Last 7 days", "last-30-days": "Last 30 days", "this-month": "This month",
    "last-month": "Last month", "last-year": "Last year", "year-to-date": "Year to date",
    "season-to-date": "Season to date", "custom": "Custom dates",
}
AGGREGATE_LABELS = {"avg": "Average", "min": "Minimum", "max": "Maximum"}
SOIL_INDENT_PX = 18      # indent of the soil measurement checkboxes under "Soil"
SOIL_LABELS = {"soil-moisture": "Soil moisture", "bare-soil-temperature": "Bare soil temperature",
               "vegetated-soil-temperature": "Vegetated soil temperature"}
WIND_LABELS = {"wind-3m": "Wind 3 m", "gust-3m": "Gust 3 m", "wind-10m": "Wind 10 m", "gust-10m": "Gust 10 m"}
TEN_METER_WIND_SOURCES = ("wind-10m", "gust-10m")
WIND_COLUMN_NAMES = {"wind": "speed", "windRange": "speed_range", "gust": "gust"}

MSG_NO_CAMERAS      = "Cameras not supported on this site."
MSG_CAMERAS_OFFLINE = "The cameras are currently not accessible."
MSG_NO_DATA_YET     = "Click Preview to load data."
MSG_NO_ROWS         = "No data returned. See the Log tab."


# ======================================================================================================================
# Helpers
# ======================================================================================================================
def snake(name):
    """camelCase -> snake_case (tenMeterTemperature -> ten_meter_temperature)."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def sheet_name(text):
    """Excel sheet names: max 31 characters, none of []:*?/\\"""
    return re.sub(r"[\[\]:*?/\\]", "-", str(text))[:31]


def image_from_bytes(data):
    """QImage from bytes; falls back to Pillow when Qt has no WebP plugin."""
    img = QImage()
    if img.loadFromData(data):
        return img
    from PIL import Image
    buf = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(buf, format="PNG")
    img.loadFromData(buf.getvalue())
    return img


# ======================================================================================================================
# Workers
# ======================================================================================================================
class CallWorker(QThread):
    """Runs one callable off the GUI thread."""
    result = pyqtSignal(object)
    error  = pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def run(self):
        try:
            self.result.emit(self.fn(*self.args, **self.kwargs))
        except Exception as e:
            self.error.emit(f"{e.__class__.__name__}: {e}")


class MesonetDataWorker(QThread):
    """
    Fetches the selected data types for every selected station, merges them on timestamp per station, and
    (for Download) writes the files and the latest camera images.
    """
    progress = pyqtSignal(int, int)
    log      = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, ne, request, write_output):
        super().__init__()
        self.ne = ne
        self.req = request
        self.write_output = write_output
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            self.finished_ok.emit(self._process())
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------------------------------------------------------
    def _process(self):
        import pandas as pd
        req = self.req
        stations = req["stations"]
        steps = len(stations) * (int(req["weather"]) + len(req["soil_measurements"]) + int(req["wind"]))
        if self.write_output and req["camera"]:
            steps += len(stations)
        step = 0
        units = {}
        station_frames = {}

        for st in stations:
            sid, name = st["station_id"], st["station"]
            frames = []
            frames_have_precip = False        # soil responses all carry the same precipitation; keep it once
            jobs = (([("weather", None)] if req["weather"] else [])
                    + [("soil", m) for m in req["soil_measurements"]]
                    + ([("wind", None)] if req["wind"] else []))
            for kind, measurement in jobs:
                label = f"{kind} ({measurement})" if measurement else kind
                if self._abort:
                    raise RuntimeError("Aborted by user.")
                step += 1
                self.progress.emit(step, steps)

                if kind == "soil" and not st["soil_sensor_depths_in"]:
                    self.log.emit(f"{name} ({sid}): no soil sensors; {measurement} columns left empty.")
                    continue
                if kind == "wind" and req["wind_source"] in TEN_METER_WIND_SOURCES and not st["has_10m_wind"]:
                    self.log.emit(f"{name} ({sid}): no 10 m wind sensor; wind columns left empty.")
                    continue

                time.sleep(REQUEST_DELAY_SEC)
                try:
                    df = self._fetch(kind, sid, measurement)
                except Exception as e:
                    self.log.emit(f"{name} ({sid}): {label} request failed: {e}")
                    continue
                for k, v in df.attrs.get("units", {}).items():
                    units[f"unit_{kind}_{snake(k[:-len('Unit')] if k.endswith('Unit') else k)}"] = v
                n = len(df)
                self.log.emit(f"{name} ({sid}): {label} {n} rows.")
                if n:
                    frames.append(self._rename(kind, df, keep_precipitation=not frames_have_precip)
                                  .set_index("timestamp_utc"))
                    frames_have_precip = frames_have_precip or kind == "soil"

            if frames:
                merged = pd.concat(frames, axis=1).sort_index()
                merged = merged.loc[:, ~merged.columns.duplicated()]
                merged.index.name = "timestamp_utc"
                merged = merged.reset_index()
                merged.insert(1, "timestamp_local", merged["timestamp_utc"].dt.tz_convert(st["timezone"]))
            else:
                merged = pd.DataFrame(columns=["timestamp_utc", "timestamp_local"])
            station_frames[sid] = merged

        all_stations = self._all_stations(stations, station_frames)
        info = self._info(stations, units)
        result = {"stations": stations, "station_frames": station_frames, "all_stations": all_stations,
                  "info": info, "output_dir": None}

        if self.write_output:
            out_dir = self._write(result)
            if req["camera"]:
                for st in stations:
                    step += 1
                    self.progress.emit(step, steps)
                    self._download_images(st, out_dir)
            result["output_dir"] = out_dir
        self.progress.emit(steps, steps)
        return result

    # ------------------------------------------------------------------------------------------------------------------
    def _fetch(self, kind, sid, measurement=None):
        r = self.req
        common = dict(resolution=r["resolution"], period=r["period"], aggregate=r["aggregate"],
                      start=r["start"], end=r["end"])
        if kind == "weather":
            return self.ne.get_station_history(sid, **common)
        if kind == "soil":
            return self.ne.get_soil_history(sid, measurement=measurement, **common)
        return self.ne.get_wind_history(sid, wind_source=r["wind_source"], **common)

    def _rename(self, kind, df, keep_precipitation=True):
        keep = ["timestamp_utc"]
        cols = {}
        for c in df.columns:
            if c in ("timestamp_utc", "timestamp_local"):
                continue
            if kind == "weather":
                cols[c] = f"weather_{snake(c)}"
            elif kind == "soil":
                if c == "precipitation":
                    if self.req["weather"] or not keep_precipitation:
                        continue                      # weather_precipitation_total, or an earlier soil set, has it
                    cols[c] = "precipitation_total"
                else:
                    cols[c] = c                       # e.g. soil_moisture_2in
            else:
                base, _, suffix = c.partition("_")
                mapped = WIND_COLUMN_NAMES.get(base, snake(base)) + (f"_{suffix}" if suffix else "")
                cols[c] = f"{self.req['wind_source'].replace('-', '_')}_{mapped}"
        return df[keep + list(cols)].rename(columns=cols)

    def _all_stations(self, stations, station_frames):
        import pandas as pd
        parts = []
        for st in stations:
            df = station_frames[st["station_id"]].copy()
            df.insert(0, "station", st["station"])
            df.insert(0, "station_id", st["station_id"])
            parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _info(self, stations, units):
        r = self.req
        rows = [
            ("generated_local", datetime.datetime.now().isoformat(timespec="seconds")),
            ("source", "Nebraska MESONET (nemesonet.unl.edu)"),
            ("warning", WARNING_TEXT),
            ("resolution", r["resolution"]),
            ("period", r["period"]),
            ("start", r["start"] or ""),
            ("end", r["end"] or ""),
            ("aggregate", r["aggregate"]),
            ("data_types", ", ".join(k for k, on in (("weather", r["weather"]), ("soil", r["soil_measurements"]),
                                                     ("wind", r["wind"])) if on)),
            ("soil_measurements", ", ".join(r["soil_measurements"])),
            ("wind_source", r["wind_source"] if r["wind"] else ""),
            ("stations", "; ".join(f"{s['station']} ({s['station_id']})" for s in stations)),
            ("timestamp_utc", "UTC"),
            ("timestamp_local", "each station's local time zone"),
            ("empty_cells", "no data, or the station lacks that sensor; see download_log.txt"),
        ]
        rows += sorted(units.items())
        return rows

    # ------------------------------------------------------------------------------------------------------------------
    def _write(self, result):
        import pandas as pd
        r = self.req
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(r["output_folder"], f"NE_MESONET_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        stations = result["stations"]
        info_df = pd.DataFrame(result["info"], columns=["item", "value"])

        def label(st):
            return f"{st['station']} ({st['station_id']})"

        if "csv" in r["formats"]:
            for st in stations:
                path = os.path.join(out_dir, f"{slug(st['station'])}_{st['station_id']}.csv")
                result["station_frames"][st["station_id"]].to_csv(path, index=False)
            result["all_stations"].to_csv(os.path.join(out_dir, "all_stations.csv"), index=False)
            info_df.to_csv(os.path.join(out_dir, "info.csv"), index=False)
            self.log.emit("Wrote CSV files.")

        if "xlsx" in r["formats"]:
            path = os.path.join(out_dir, f"NE_MESONET_{ts}.xlsx")
            try:
                with pd.ExcelWriter(path) as xw:
                    self._excel_ready(result["all_stations"]).to_excel(xw, sheet_name="All Stations", index=False)
                    used = {"All Stations", "Info"}
                    for st in stations:
                        name = sheet_name(label(st))
                        while name in used:
                            name = sheet_name(f"{st['station_id']} {name}")
                        used.add(name)
                        self._excel_ready(result["station_frames"][st["station_id"]]).to_excel(
                            xw, sheet_name=name, index=False)
                    info_df.to_excel(xw, sheet_name="Info", index=False)
                self.log.emit("Wrote XLSX workbook.")
            except Exception as e:
                self.log.emit(f"XLSX not written: {e}")

        if "json" in r["formats"]:
            def records(df):
                return json.loads(df.to_json(orient="records", date_format="iso"))
            payload = {
                "info": dict(result["info"]),
                "all_stations": records(result["all_stations"]),
                "stations": {str(st["station_id"]): {"station": st["station"],
                                                     "data": records(result["station_frames"][st["station_id"]])}
                             for st in stations},
            }
            with open(os.path.join(out_dir, f"NE_MESONET_{ts}.json"), "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1)
            self.log.emit("Wrote JSON file.")
        return out_dir

    @staticmethod
    def _excel_ready(df):
        """Excel cannot store time-zone-aware datetimes: keep the wall-clock values, drop the zone."""
        df = df.copy()
        for c in ("timestamp_utc", "timestamp_local"):
            if c in df.columns and hasattr(df[c], "dt") and df[c].dt.tz is not None:
                df[c] = df[c].dt.tz_localize(None)
        return df

    def _download_images(self, st, out_dir):
        images = self.req["camera_table"]
        sid, name = st["station_id"], st["station"]
        if images is None:
            self.log.emit(f"{name} ({sid}): {MSG_CAMERAS_OFFLINE}")
            return
        items = images.get(sid, [])
        if not items:
            self.log.emit(f"{name} ({sid}): {MSG_NO_CAMERAS}")
            return
        img_dir = os.path.join(out_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        for it in items:
            stamp = re.sub(r"[^0-9T]", "", it["last_modified_utc"].split(".")[0]) + "Z"
            path = os.path.join(img_dir, f"{slug(name)}_{sid}_{it['direction_code']}_{stamp}.webp")
            try:
                time.sleep(REQUEST_DELAY_SEC)
                with open(path, "wb") as fh:
                    fh.write(self.ne.download_bytes(it["url"]))
                self.log.emit(f"{name} ({sid}): saved {it['direction']} image.")
            except Exception as e:
                self.log.emit(f"{name} ({sid}): {it['direction']} image not downloaded ({e}). {MSG_CAMERAS_OFFLINE}")


# ======================================================================================================================
# Table model
# ======================================================================================================================
class DataFrameModel(QAbstractTableModel):
    def __init__(self, df=None):
        super().__init__()
        self._df = df

    def set_frame(self, df):
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if self._df is None else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if self._df is None else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or self._df is None:
            return None
        v = self._df.iat[index.row(), index.column()]
        return "" if v is None or (isinstance(v, float) and v != v) else str(v)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or self._df is None:
            return None
        return str(self._df.columns[section]) if orientation == Qt.Horizontal else str(section + 1)


# ======================================================================================================================
# Main dialog
# ======================================================================================================================
class NEMesonetDlg(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(PLUGIN["title"])
        self.resize(*PLUGIN["size"])

        self.ne = NEMesonet(timeout_sec=REQUEST_TIMEOUT_SEC)
        self.stations_df = None
        self.camera_table = None           # {station_id: [image, ...]}; None = not loaded or not reachable
        self.camera_table_failed = False
        self.result = None
        self._workers = []
        self._data_worker = None

        self._settings = plugin_settings(__file__)
        self._settings_ready = False          # True once saved values are restored
        self._build_ui()
        self._bind_settings()
        self._load_stations()

    # ------------------------------------------------------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        banner = QLabel(WARNING_TEXT)
        banner.setWordWrap(True)
        banner.setStyleSheet("background:#fde7b0; color:#111; padding:6px 10px; font-weight:600;"
                             "border-bottom:1px solid #d9b65a;")
        root.addWidget(banner)

        body = QHBoxLayout()
        body.setContentsMargins(8, 8, 8, 8)
        root.addLayout(body, stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(LEFT_PANEL_WIDTH)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel = self._build_left_panel()
        panel.setFixedWidth(LEFT_PANEL_WIDTH - self.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent))
        scroll.setWidget(panel)
        body.addWidget(scroll)
        body.addWidget(self._build_tabs(), stretch=1)

    @staticmethod
    def _shrinkable(combo):
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return combo

    def _form_grid(self):
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(5)
        g.setColumnStretch(1, 1)
        return g

    def _build_left_panel(self):
        panel = QWidget()
        lv = QVBoxLayout(panel)
        lv.setContentsMargins(0, 0, 4, 0)

        # Stations
        grp = QGroupBox("Stations")
        v = QVBoxLayout(grp)
        g = self._form_grid()
        g.addWidget(QLabel("Search"), 0, 0)
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Name, county, NRD, or ID")
        self.txt_search.textChanged.connect(self._filter_stations)
        g.addWidget(self.txt_search, 0, 1)
        v.addLayout(g)
        self.lst_stations = QListWidget()
        self.lst_stations.setFixedHeight(STATION_LIST_HEIGHT)
        self.lst_stations.itemChanged.connect(self._on_station_checked)
        self.lst_stations.currentItemChanged.connect(self._on_station_highlighted)
        v.addWidget(self.lst_stations)
        row = QHBoxLayout()
        self.lbl_selected = QLabel("0 selected")
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_selection)
        row.addWidget(self.lbl_selected)
        row.addStretch()
        row.addWidget(btn_clear)
        v.addLayout(row)
        self.lbl_station_info = QLabel("Loading stations…")
        self.lbl_station_info.setWordWrap(True)
        self.lbl_station_info.setOpenExternalLinks(True)
        self.lbl_station_info.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.lbl_station_info.setStyleSheet("background:#f3f5f8; border:1px solid #dde1e6; padding:6px; color:#111;")
        v.addWidget(self.lbl_station_info)
        lv.addWidget(grp)

        # Data
        grp = QGroupBox("Data")
        g = self._form_grid()
        self.chk_weather = QCheckBox("Weather")
        self.chk_weather.setChecked(True)
        g.addWidget(self.chk_weather, 0, 0, 1, 2)
        g.addWidget(QLabel("Soil"), 1, 0, 1, 2)
        soil_box = QVBoxLayout()
        soil_box.setContentsMargins(SOIL_INDENT_PX, 0, 0, 0)
        self.soil_checks = {}
        for k, lab in SOIL_LABELS.items():
            self.soil_checks[k] = QCheckBox(lab)
            soil_box.addWidget(self.soil_checks[k])
        g.addLayout(soil_box, 2, 0, 1, 2)
        self.chk_wind = QCheckBox("Wind")
        self.cmb_wind = self._shrinkable(QComboBox())
        for k, lab in WIND_LABELS.items():
            self.cmb_wind.addItem(lab, k)
        g.addWidget(self.chk_wind, 3, 0)
        g.addWidget(self.cmb_wind, 3, 1)

        g.addWidget(QLabel("Resolution"), 4, 0)
        self.cmb_resolution = self._shrinkable(QComboBox())
        for k, lab in RESOLUTION_LABELS.items():
            self.cmb_resolution.addItem(lab, k)
        self.cmb_resolution.setCurrentIndex(list(RESOLUTION_LABELS).index("hourly"))
        self.cmb_resolution.currentIndexChanged.connect(self._refresh_periods)
        g.addWidget(self.cmb_resolution, 4, 1)

        g.addWidget(QLabel("Period"), 5, 0)
        self.cmb_period = self._shrinkable(QComboBox())
        self.cmb_period.currentIndexChanged.connect(self._refresh_dates)
        g.addWidget(self.cmb_period, 5, 1)

        g.addWidget(QLabel("Aggregate"), 6, 0)
        self.cmb_aggregate = self._shrinkable(QComboBox())
        for k, lab in AGGREGATE_LABELS.items():
            self.cmb_aggregate.addItem(lab, k)
        g.addWidget(self.cmb_aggregate, 6, 1)

        earliest = QDate.fromString(EARLIEST_DATA_DATE, "yyyy-MM-dd")
        today = QDate.currentDate()
        g.addWidget(QLabel("Start"), 7, 0)
        self.date_start = QDateEdit(today.addDays(-7))
        g.addWidget(QLabel("End"), 8, 0)
        self.date_end = QDateEdit(today.addDays(-1))
        for d, row_i in ((self.date_start, 7), (self.date_end, 8)):
            d.setCalendarPopup(True)
            d.setDisplayFormat("yyyy-MM-dd")
            d.setDateRange(earliest, today)
            g.addWidget(d, row_i, 1)

        note = QLabel(f"Custom dates: daily only, from {EARLIEST_DATA_DATE}. "
                      "Sub-daily data: last 7 days (10 minutes: last 30 days).")
        note.setWordWrap(True)
        g.addWidget(note, 9, 0, 1, 2)
        grp.setLayout(g)
        lv.addWidget(grp)

        # Camera images
        self.grp_camera = QGroupBox("Camera Images")
        v = QVBoxLayout(self.grp_camera)
        self.chk_camera = QCheckBox("Download latest images")
        v.addWidget(self.chk_camera)
        self.lbl_camera = QLabel("")
        self.lbl_camera.setWordWrap(True)
        self.lbl_camera.setStyleSheet("font-weight:600;")
        v.addWidget(self.lbl_camera)
        lv.addWidget(self.grp_camera)

        # Output
        grp = QGroupBox("Output")
        g = self._form_grid()
        g.addWidget(QLabel("Folder"), 0, 0)
        self.txt_folder = QLineEdit()
        g.addWidget(self.txt_folder, 0, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        g.addWidget(btn_browse, 0, 2)
        g.addWidget(QLabel("Formats"), 1, 0)
        fmt = QHBoxLayout()
        self.chk_csv = QCheckBox("CSV")
        self.chk_csv.setChecked(True)
        self.chk_xlsx = QCheckBox("XLSX")
        self.chk_json = QCheckBox("JSON")
        for w in (self.chk_csv, self.chk_xlsx, self.chk_json):
            fmt.addWidget(w)
        fmt.addStretch()
        g.addLayout(fmt, 1, 1, 1, 2)
        grp.setLayout(g)
        lv.addWidget(grp)

        # Actions
        row = QHBoxLayout()
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.clicked.connect(lambda: self._run(write_output=False))
        self.btn_download = QPushButton("Download")
        self.btn_download.clicked.connect(lambda: self._run(write_output=True))
        self.btn_abort = QPushButton("Abort")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self._abort)
        for b in (self.btn_preview, self.btn_download, self.btn_abort):
            row.addWidget(b)
        lv.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        lv.addWidget(self.progress)
        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setWordWrap(True)
        lv.addWidget(self.lbl_status)
        lv.addStretch()

        self._refresh_periods()
        self._refresh_availability()
        return panel

    def _build_tabs(self):
        self.tabs = QTabWidget()

        # Chart
        chart = QWidget()
        cv = QVBoxLayout(chart)
        row = QHBoxLayout()
        row.addWidget(QLabel("Variable"))
        self.cmb_variable = QComboBox()
        self.cmb_variable.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.cmb_variable.setMinimumContentsLength(VARIABLE_LIST_CHARS)
        self.cmb_variable.currentIndexChanged.connect(self._draw_chart)
        row.addWidget(self.cmb_variable)
        row.addStretch()
        cv.addLayout(row)
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            self.figure = Figure()
            self.canvas = FigureCanvasQTAgg(self.figure)
            cv.addWidget(self.canvas, stretch=1)
            self._chart_message(MSG_NO_DATA_YET)
        except ImportError:
            self.figure = None
            cv.addWidget(QLabel("Charts need matplotlib, which is not installed."), stretch=1)
        self.tabs.addTab(chart, "Chart")

        # Table
        table_tab = QWidget()
        tv = QVBoxLayout(table_tab)
        self.lbl_table = QLabel(MSG_NO_DATA_YET)
        self.lbl_table.setStyleSheet("font-weight:600;")
        tv.addWidget(self.lbl_table)
        self.table_model = DataFrameModel()
        self.table = QTableView()
        self.table.setModel(self.table_model)
        tv.addWidget(self.table, stretch=1)
        self.tabs.addTab(table_tab, "Table")

        # Camera
        cam = QWidget()
        self.camera_layout = QVBoxLayout(cam)
        self.lbl_camera_title = QLabel("Select a station.")
        self.lbl_camera_title.setStyleSheet("font-weight:600;")
        self.camera_layout.addWidget(self.lbl_camera_title)
        cam_scroll = QScrollArea()
        cam_scroll.setWidgetResizable(True)
        cam_scroll.setFrameShape(QFrame.NoFrame)
        cam_body = QWidget()
        self.camera_row = QGridLayout(cam_body)
        self.camera_row.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        cam_scroll.setWidget(cam_body)
        self.camera_layout.addWidget(cam_scroll, stretch=1)
        self.tabs.addTab(cam, "Camera")

        # Log
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.tabs.addTab(self.txt_log, "Log")
        return self.tabs

    # ------------------------------------------------------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------------------------------------------------------
    def _bind_settings(self):
        """Choices come back the next time the plugin is opened. Data and images are not stored: they are live."""
        st = self._settings
        st.bind(self.chk_weather, "weather")
        for k, chk in self.soil_checks.items():
            st.bind(chk, f"soil_{k.replace('-', '_')}")
        st.bind(self.chk_wind, "wind")
        st.bind(self.cmb_wind, "wind_source_index")
        st.bind(self.cmb_resolution, "resolution_index")     # refills the Period list
        st.bind(self.cmb_aggregate, "aggregate_index")
        st.bind(self.chk_camera, "download_camera_images")
        st.bind(self.txt_folder, "output_folder")
        st.bind(self.chk_csv, "format_csv")
        st.bind(self.chk_xlsx, "format_xlsx")
        st.bind(self.chk_json, "format_json")

        # Period depends on resolution, so it is stored by value rather than list position.
        idx = self.cmb_period.findData(st.get("period"))
        if idx >= 0:
            self.cmb_period.setCurrentIndex(idx)
        self.cmb_period.currentIndexChanged.connect(lambda _: self._save("period", self.cmb_period.currentData()))

        for widget, key in ((self.date_start, "custom_start"), (self.date_end, "custom_end")):
            saved = QDate.fromString(st.get(key) or "", "yyyy-MM-dd")
            if saved.isValid():
                widget.setDate(saved)
            widget.dateChanged.connect(lambda d, k=key: self._save(k, d.toString("yyyy-MM-dd")))
        self._settings_ready = True

    def _save(self, key, value):
        self._settings[key] = value
        self._settings.save()

    def _restore_station_checks(self):
        saved = set(self._settings.get("stations", []))
        if not saved:
            return
        self.lst_stations.blockSignals(True)
        for i in range(self.lst_stations.count()):
            item = self.lst_stations.item(i)
            if item.data(Qt.UserRole) in saved:
                item.setCheckState(Qt.Checked)
        self.lst_stations.blockSignals(False)

    # ------------------------------------------------------------------------------------------------------------------
    # Background loading
    # ------------------------------------------------------------------------------------------------------------------
    def _start(self, worker, on_result, on_error):
        worker.result.connect(on_result)
        worker.error.connect(on_error)
        worker.finished.connect(lambda: self._workers.remove(worker) if worker in self._workers else None)
        self._workers.append(worker)
        worker.start()

    def _load_stations(self):
        self._start(CallWorker(self.ne.get_dataframe), self._on_stations_loaded,
                    lambda e: self._set_status(f"Station list not loaded: {e}"))

    def _on_stations_loaded(self, df):
        self.stations_df = df.sort_values("station").reset_index(drop=True)
        self.lst_stations.blockSignals(True)
        self.lst_stations.clear()
        for _, r in self.stations_df.iterrows():
            item = QListWidgetItem(f"{r['station']} ({r['station_id']})")
            item.setData(Qt.UserRole, int(r["station_id"]))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.lst_stations.addItem(item)
        self.lst_stations.blockSignals(False)
        self._restore_station_checks()
        self._on_station_checked(None)
        self.lbl_station_info.setText(f"{len(self.stations_df)} stations. Check stations to download; "
                                      "click one to see its details.")
        self._log(f"Loaded {len(self.stations_df)} stations.")
        # Camera table: any station's page carries it.
        self._start(CallWorker(self.ne.get_station_images, int(self.stations_df.iloc[0]["station_id"])),
                    self._on_camera_table_loaded, self._on_camera_table_failed)

    def _on_camera_table_loaded(self, table):
        self.camera_table = table
        n = sum(1 for v in table.values() if v)
        self._log(f"Camera table loaded: {n} station(s) with cameras.")
        self._refresh_camera_state()

    def _on_camera_table_failed(self, err):
        self.camera_table_failed = True
        self._log(f"Camera table not loaded: {err}")
        self._refresh_camera_state()

    # ------------------------------------------------------------------------------------------------------------------
    # Station selection
    # ------------------------------------------------------------------------------------------------------------------
    def _station_row(self, sid):
        return self.stations_df[self.stations_df["station_id"] == sid].iloc[0]

    def _selected_ids(self):
        return [self.lst_stations.item(i).data(Qt.UserRole) for i in range(self.lst_stations.count())
                if self.lst_stations.item(i).checkState() == Qt.Checked]

    def _filter_stations(self, text):
        text = text.strip().lower()
        for i in range(self.lst_stations.count()):
            item = self.lst_stations.item(i)
            r = self._station_row(item.data(Qt.UserRole))
            hay = f"{r['station']} {r['station_id']} {r['county']} {r['nrd']}".lower()
            item.setHidden(bool(text) and text not in hay)

    def _clear_selection(self):
        self.lst_stations.blockSignals(True)
        for i in range(self.lst_stations.count()):
            self.lst_stations.item(i).setCheckState(Qt.Unchecked)
        self.lst_stations.blockSignals(False)
        self._on_station_checked(None)

    def _on_station_checked(self, _item):
        ids = self._selected_ids()
        self.lbl_selected.setText(f"{len(ids)} selected")
        if _item is not None or not ids:
            self._save("stations", ids)
        self._refresh_availability()
        self._refresh_camera_state()

    def _on_station_highlighted(self, item, _prev):
        if item is None or self.stations_df is None:
            return
        r = self._station_row(item.data(Qt.UserRole))
        cams = self._camera_items(int(r["station_id"]))
        cam_text = ", ".join(c["direction"] for c in cams) if cams else "None"
        if self.camera_table is None:
            cam_text = "Unknown" if not self.camera_table_failed else "Not accessible"
        self.lbl_station_info.setText(
            f"<b>{r['station']} ({r['station_id']})</b><br>"
            f"<b>County:</b> {r['county']}<br>"
            f"<b>NRD:</b> {r['nrd']}<br>"
            f"<b>Soil depths (in):</b> {r['soil_sensor_depths_in'] or 'None'}<br>"
            f"<b>10 m wind:</b> {'Yes' if r['has_10m_wind'] else 'No'}<br>"
            f"<b>Cameras:</b> {cam_text}<br>"
            f"<a href=\"{r['station_url']}\">Open station page</a>")
        self._show_camera_preview(int(r["station_id"]), r["station"])

    # ------------------------------------------------------------------------------------------------------------------
    # Availability rules
    # ------------------------------------------------------------------------------------------------------------------
    def _refresh_periods(self):
        resolution = self.cmb_resolution.currentData()
        current = self.cmb_period.currentData()
        allowed = PERIODS_BY_RESOLUTION.get(resolution, tuple(PERIOD_LABELS))
        self.cmb_period.blockSignals(True)
        self.cmb_period.clear()
        for p in allowed:
            self.cmb_period.addItem(PERIOD_LABELS.get(p, p), p)
        idx = self.cmb_period.findData(current)
        self.cmb_period.setCurrentIndex(idx if idx >= 0 else 0)
        self.cmb_period.blockSignals(False)
        if getattr(self, "_settings_ready", False):
            self._save("period", self.cmb_period.currentData())
        self._refresh_dates()

    def _refresh_dates(self):
        custom = self.cmb_period.currentData() == "custom"
        self.date_start.setEnabled(custom)
        self.date_end.setEnabled(custom)

    def _refresh_availability(self):
        """A data type is disabled only when none of the selected stations has it."""
        ids = self._selected_ids()
        rows = [self._station_row(sid) for sid in ids] if self.stations_df is not None else []
        any_soil = any(bool(r["soil_sensor_depths_in"]) for r in rows)
        any_10m = any(bool(r["has_10m_wind"]) for r in rows)

        for chk in self.soil_checks.values():
            chk.setEnabled(any_soil)

        model = self.cmb_wind.model()
        for i in range(self.cmb_wind.count()):
            ten_m = self.cmb_wind.itemData(i) in TEN_METER_WIND_SOURCES
            model.item(i).setEnabled(any_10m or not ten_m)

        has_any = bool(rows)
        for w in (self.chk_weather, self.chk_wind, self.cmb_wind):
            w.setEnabled(has_any)
        if not has_any:
            for chk in self.soil_checks.values():
                chk.setEnabled(False)

    def _camera_items(self, sid):
        return (self.camera_table or {}).get(sid, [])

    def _refresh_camera_state(self):
        ids = self._selected_ids()
        if not ids:
            self.grp_camera.setEnabled(False)
            self.lbl_camera.setText("Select one or more stations.")
            return
        if self.camera_table is None:
            self.grp_camera.setEnabled(False)
            self.lbl_camera.setText(MSG_CAMERAS_OFFLINE if self.camera_table_failed else "Checking cameras…")
            return
        with_cams = [sid for sid in ids if self._camera_items(sid)]
        if not with_cams:
            self.grp_camera.setEnabled(False)
            self.lbl_camera.setText(MSG_NO_CAMERAS)
            return
        self.grp_camera.setEnabled(True)
        names = ", ".join(self._station_row(s)["station"] for s in with_cams)
        self.lbl_camera.setText(f"Latest only; no image archive is available. Cameras at: {names}.")

    # ------------------------------------------------------------------------------------------------------------------
    # Camera preview (highlighted station)
    # ------------------------------------------------------------------------------------------------------------------
    def _clear_camera_row(self):
        while self.camera_row.count():
            w = self.camera_row.takeAt(0).widget()
            if w is not None:
                w.deleteLater()

    def _show_camera_preview(self, sid, name):
        self._clear_camera_row()
        if self.camera_table is None:
            self.lbl_camera_title.setText(f"{name} ({sid}): "
                                          f"{MSG_CAMERAS_OFFLINE if self.camera_table_failed else 'Checking cameras…'}")
            return
        items = self._camera_items(sid)
        if not items:
            self.lbl_camera_title.setText(f"{name} ({sid}): {MSG_NO_CAMERAS}")
            return
        self.lbl_camera_title.setText(f"Latest camera images: {name} ({sid})")
        for n, it in enumerate(items):
            box = QVBoxLayout()
            frame = QWidget()
            frame.setLayout(box)
            pic = QLabel("Loading…")
            pic.setFixedWidth(THUMB_WIDTH)
            pic.setAlignment(Qt.AlignCenter)
            cap = QLabel(f"<b>{it['direction']}</b>, published {it['last_modified_utc'][:19].replace('T', ' ')} UTC")
            box.addWidget(pic)
            box.addWidget(cap)
            self.camera_row.addWidget(frame, n // THUMB_COLUMNS, n % THUMB_COLUMNS)
            self._start(CallWorker(self.ne.download_bytes, it["url"]),
                        lambda data, p=pic: self._set_thumb(p, data),
                        lambda e, p=pic: p.setText(MSG_CAMERAS_OFFLINE))

    @staticmethod
    def _set_thumb(label, data):
        try:
            img = image_from_bytes(data)
            label.setPixmap(QPixmap.fromImage(img).scaledToWidth(THUMB_WIDTH, Qt.SmoothTransformation))
        except Exception:
            label.setText(MSG_CAMERAS_OFFLINE)

    # ------------------------------------------------------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------------------------------------------------------
    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.txt_folder.text())
        if folder:
            self.txt_folder.setText(folder)
            self._save("output_folder", folder)

    def _request(self, write_output):
        ids = self._selected_ids()
        if not ids:
            raise ValueError("Select at least one station.")
        soil_measurements = [k for k, chk in self.soil_checks.items() if chk.isChecked() and chk.isEnabled()]
        if not (self.chk_weather.isChecked() or soil_measurements or self.chk_wind.isChecked()):
            raise ValueError("Select at least one data type.")
        formats = [f for f, c in (("csv", self.chk_csv), ("xlsx", self.chk_xlsx), ("json", self.chk_json))
                   if c.isChecked()]
        if write_output:
            if not self.txt_folder.text().strip():
                raise ValueError("Select an output folder.")
            if not formats and not self.chk_camera.isChecked():
                raise ValueError("Select at least one output format.")
        period = self.cmb_period.currentData()
        start = end = None
        if period == "custom":
            if self.date_start.date() > self.date_end.date():
                raise ValueError("Start date is after end date.")
            start = self.date_start.date().toString("yyyy-MM-dd")
            end = self.date_end.date().toString("yyyy-MM-dd")
        stations = [self._station_row(sid).to_dict() for sid in ids]
        return {
            "stations": stations,
            "weather": self.chk_weather.isChecked(),
            "soil_measurements": soil_measurements,
            "wind": self.chk_wind.isChecked(),
            "wind_source": self.cmb_wind.currentData(),
            "resolution": self.cmb_resolution.currentData(),
            "period": period,
            "aggregate": self.cmb_aggregate.currentData(),
            "start": start,
            "end": end,
            "camera": self.chk_camera.isChecked() and self.grp_camera.isEnabled(),  # disabled group = no cameras
            "camera_table": self.camera_table,
            "formats": formats,
            "output_folder": self.txt_folder.text().strip(),
        }

    def _run(self, write_output):
        try:
            req = self._request(write_output)
        except ValueError as e:
            QMessageBox.warning(self, PLUGIN["title"], str(e))
            return
        self._log_lines = []
        self._log("-" * 60)
        self._log(f"{'Download' if write_output else 'Preview'}: {len(req['stations'])} station(s), "
                  f"{req['resolution']}, {req['period']}.")
        self._set_busy(True)
        self._data_worker = w = MesonetDataWorker(self.ne, req, write_output)
        w.progress.connect(lambda i, n: (self.progress.setMaximum(max(n, 1)), self.progress.setValue(i)))
        w.log.connect(self._log)
        w.finished_ok.connect(self._on_data_done)
        w.error.connect(self._on_data_error)
        w.start()

    def _abort(self):
        if self._data_worker is not None:
            self._data_worker.abort()

    def _set_busy(self, busy):
        for b in (self.btn_preview, self.btn_download):
            b.setEnabled(not busy)
        self.btn_abort.setEnabled(busy)
        self._set_status("Working…" if busy else "Ready.")

    def _on_data_done(self, result):
        self._set_busy(False)
        self.result = result
        df = result["all_stations"]
        self.table_model.set_frame(df.head(TABLE_PREVIEW_ROWS))
        self.lbl_table.setVisible(len(df) == 0)
        self.lbl_table.setText(MSG_NO_ROWS)
        numeric = [c for c in df.columns if c not in ("station_id",) and str(df[c].dtype).startswith(("float", "int"))]
        self.cmb_variable.blockSignals(True)
        self.cmb_variable.clear()
        self.cmb_variable.addItems(numeric)
        self.cmb_variable.blockSignals(False)
        self._draw_chart()
        if result["output_dir"]:
            self._write_log_file(result["output_dir"])
            self._set_status(f"Saved to {result['output_dir']}")
            self._log(f"Saved to {result['output_dir']}")
        else:
            self._set_status(f"Preview: {len(df)} rows.")

    def _on_data_error(self, msg):
        self._set_busy(False)
        self._log(msg)
        self._set_status("Stopped. See the Log tab.")

    def _chart_message(self, text):
        self.figure.clear()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", fontsize=14, color="#111")
        self.canvas.draw()

    def _draw_chart(self):
        if self.figure is None or self.result is None:
            return
        var = self.cmb_variable.currentText()
        if not var:
            self._chart_message(MSG_NO_ROWS)
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        df = self.result["all_stations"]
        if var and var in df.columns:
            for st in self.result["stations"]:
                part = df[df["station_id"] == st["station_id"]]
                if part[var].notna().any():
                    ax.plot(part["timestamp_local"].dt.tz_localize(None), part[var],
                            label=f"{st['station']} ({st['station_id']})")
            ax.set_ylabel(var, color="#111")
            ax.set_xlabel("Local time", color="#111")
            ax.tick_params(colors="#111")
            if ax.lines:
                ax.legend()
            ax.grid(True, alpha=0.3)
        self.figure.autofmt_xdate()
        self.canvas.draw()

    # ------------------------------------------------------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------------------------------------------------------
    def _log(self, text):
        line = f"{datetime.datetime.now().strftime('%H:%M:%S')}  {text}"
        self.txt_log.appendPlainText(line)
        if not hasattr(self, "_log_lines"):
            self._log_lines = []
        self._log_lines.append(line)

    def _write_log_file(self, out_dir):
        try:
            with open(os.path.join(out_dir, "download_log.txt"), "w", encoding="utf-8") as fh:
                fh.write(WARNING_TEXT + "\n\n" + "\n".join(self._log_lines) + "\n")
        except OSError as e:
            self._log(f"Log file not written: {e}")

    def _set_status(self, text):
        self.lbl_status.setText(text)

    def closeEvent(self, event):
        if self._data_worker is not None and self._data_worker.isRunning():
            self._data_worker.abort()
            self._data_worker.wait()
        for w in list(self._workers):
            w.wait()
        super().closeEvent(event)


# ======================================================================================================================
# Plugin wrapper
# ======================================================================================================================
class NEMesonetPlugin(QtWidgets.QWidget):
    """Hosts the dialog inside a plugin window, the same way as the other plugins."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._dialog = NEMesonetDlg(self)
        self._dialog.setWindowFlags(Qt.Widget)
        layout.addWidget(self._dialog)

    @property
    def dialog(self):
        return self._dialog


def main(argv=None):
    import sys
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    dlg = NEMesonetDlg()
    dlg.show()
    return app.exec_()


if __name__ == "__main__":
    import sys
    sys.exit(main())
