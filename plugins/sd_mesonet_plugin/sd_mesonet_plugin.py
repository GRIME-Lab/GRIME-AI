#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# sd_mesonet_plugin.py
# EXPERIMENTAL plugin: downloads the last 48 hours of 5-minute weather data and the latest camera images from the
# South Dakota MESONET website, using internal, unpublished endpoints that may change without notice.
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
from email.utils import parsedate_to_datetime

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QAbstractTableModel, QModelIndex
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox, QCheckBox,
    QGroupBox, QProgressBar, QFileDialog, QMessageBox, QLineEdit, QListWidget, QListWidgetItem,
    QTabWidget, QTableView, QPlainTextEdit, QScrollArea, QWidget, QFrame
)

try:
    from appcore.geomaps.SDMESONET import SDMesonet
except ImportError:
    from GRIME_AI.geomaps.SDMESONET import SDMesonet

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
                print(f"[sd_mesonet_plugin] Could not save settings: {err}")

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
    "title":       "South Dakota MESONET (UNOFFICIAL API, EXPERIMENTAL)",
    "class":       "SDMesonetPlugin",
    "description": "Download South Dakota MESONET 48-hour weather data and camera images (unofficial, experimental)",
    "surface":     "tools",
    "size":        [1280, 860],
    "api_version": 2,
}

WARNING_TEXT = "Uses internal, unpublished South Dakota MESONET website endpoints that may change without notice."
TERMS_TEXT = ("SD Mesonet terms: credit required; redistribution not allowed; commercial use not allowed; "
              "data are provisional.")

# ----------------------------------------------------------------------------------------------------------------------
# SETTINGS (prototype: module level, not saved to the settings file)
# ----------------------------------------------------------------------------------------------------------------------
REQUEST_TIMEOUT_SEC  = 30
REQUEST_DELAY_SEC    = 0.5     # pause between requests to the MESONET site
LEFT_PANEL_WIDTH     = 340
THUMB_WIDTH          = 300
THUMB_COLUMNS        = 2       # camera thumbnails per row in the Camera tab
STATION_LIST_HEIGHT  = 220
VARIABLE_LIST_CHARS  = 40      # width of the chart Variable list, in characters
TABLE_PREVIEW_ROWS   = 5000    # rows shown in the Table tab (files always get every row)

MSG_NO_CAMERAS      = "Cameras not supported on this site."
MSG_CAMERAS_OFFLINE = "The cameras are currently not accessible."
MSG_NO_DATA_YET     = "Click Preview to load data."
MSG_NO_ROWS         = "No data returned. See the Log tab."


# ======================================================================================================================
# Helpers
# ======================================================================================================================
def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def sheet_name(text):
    """Excel sheet names: max 31 characters, none of []:*?/\\"""
    return re.sub(r"[\[\]:*?/\\]", "-", str(text))[:31]


def stamp_from_last_modified(value):
    """HTTP Last-Modified -> 20261005T222005Z, or the download time if the header is missing."""
    try:
        dt = parsedate_to_datetime(value).astimezone(datetime.timezone.utc)
    except Exception:
        dt = datetime.datetime.now(datetime.timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def wall_clock(series):
    """Time-zone-aware timestamps (one zone or several) -> naive wall-clock datetimes."""
    import pandas as pd
    return pd.to_datetime(series.apply(lambda x: x.tz_localize(None) if getattr(x, "tzinfo", None) else x))


def image_from_bytes(data):
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


class SDDataWorker(QThread):
    """Fetches 48-hour history for every selected station and, for Download, writes files and camera images."""
    progress    = pyqtSignal(int, int)
    log         = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    error       = pyqtSignal(str)

    def __init__(self, sd, request, write_output):
        super().__init__()
        self.sd = sd
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

    def _process(self):
        import pandas as pd
        req = self.req
        stations = req["stations"]
        want_images = self.write_output and (req["stills"] or req["timelapse"])
        steps = len(stations) * (int(req["weather"]) + int(want_images))
        step, units, station_frames = 0, {}, {}

        for st in stations:
            code, name = st["code"], st["station"]
            df = pd.DataFrame(columns=["timestamp_utc", "timestamp_local"])
            if req["weather"]:
                if self._abort:
                    raise RuntimeError("Aborted by user.")
                step += 1
                self.progress.emit(step, steps)
                time.sleep(REQUEST_DELAY_SEC)
                try:
                    df = self.sd.get_history(code, st["timezone"])
                    units.update(df.attrs.get("units", {}))
                    self.log.emit(f"{name} ({st['nwsli']}): {len(df)} rows.")
                except Exception as e:
                    self.log.emit(f"{name} ({st['nwsli']}): history request failed: {e}")
            station_frames[code] = df

        parts = []
        for st in stations:
            d = station_frames[st["code"]].copy()
            d.insert(0, "station", st["station"])
            d.insert(0, "nwsli", st["nwsli"])
            parts.append(d)
        all_stations = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        info = self._info(stations, units)
        result = {"stations": stations, "station_frames": station_frames, "all_stations": all_stations,
                  "info": info, "output_dir": None}

        if self.write_output:
            out_dir = self._write(result)
            if want_images:
                for st in stations:
                    step += 1
                    self.progress.emit(step, steps)
                    self._download_images(st, out_dir)
            result["output_dir"] = out_dir
        self.progress.emit(steps, steps)
        return result

    def _info(self, stations, units):
        rows = [
            ("generated_local", datetime.datetime.now().isoformat(timespec="seconds")),
            ("source", "South Dakota MESONET (climate.sdstate.edu)"),
            ("warning", WARNING_TEXT),
            ("terms", TERMS_TEXT),
            ("period", "last 48 hours"),
            ("resolution", "5 minutes"),
            ("stations", "; ".join(f"{s['station']} ({s['nwsli']})" for s in stations)),
            ("timestamp_utc", "UTC"),
            ("timestamp_local", "each station's local time zone (Central or Mountain, with daylight saving time)"),
            ("empty_cells", "no data for that time (e.g. no gust, no wind chill or heat index, night-time "
                            "clear-sky radiation); see download_log.txt"),
        ]
        rows += [(f"unit_{k}", v) for k, v in sorted(units.items())]
        return rows

    def _write(self, result):
        import pandas as pd
        r = self.req
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(r["output_folder"], f"SD_MESONET_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        stations = result["stations"]
        info_df = pd.DataFrame(result["info"], columns=["item", "value"])

        if not r["weather"]:
            return out_dir

        if "csv" in r["formats"]:
            for st in stations:
                path = os.path.join(out_dir, f"{slug(st['station'])}_{st['nwsli']}.csv")
                result["station_frames"][st["code"]].to_csv(path, index=False)
            result["all_stations"].to_csv(os.path.join(out_dir, "all_stations.csv"), index=False)
            info_df.to_csv(os.path.join(out_dir, "info.csv"), index=False)
            self.log.emit("Wrote CSV files.")

        if "xlsx" in r["formats"]:
            path = os.path.join(out_dir, f"SD_MESONET_{ts}.xlsx")
            try:
                with pd.ExcelWriter(path) as xw:
                    self._excel_ready(result["all_stations"]).to_excel(xw, sheet_name="All Stations", index=False)
                    used = {"All Stations", "Info"}
                    for st in stations:
                        name = sheet_name(f"{st['station']} ({st['nwsli']})")
                        while name in used:
                            name = sheet_name(f"{st['code']} {name}")
                        used.add(name)
                        self._excel_ready(result["station_frames"][st["code"]]).to_excel(
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
                "stations": {st["nwsli"]: {"station": st["station"],
                                           "data": records(result["station_frames"][st["code"]])}
                             for st in stations},
            }
            with open(os.path.join(out_dir, f"SD_MESONET_{ts}.json"), "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1)
            self.log.emit("Wrote JSON file.")
        return out_dir

    @staticmethod
    def _excel_ready(df):
        """Excel cannot store time-zone-aware datetimes: keep the wall-clock values, drop the zone.
        Works when a column mixes zones (e.g. Central and Mountain stations in one table)."""
        df = df.copy()
        for c in ("timestamp_utc", "timestamp_local"):
            if c in df.columns:
                df[c] = wall_clock(df[c])
        return df

    def _download_images(self, st, out_dir):
        name, nwsli = st["station"], st["nwsli"]
        views = self.req["camera_table"].get(st["code"])
        if views is None:
            try:
                views = self.sd.get_station_images(st["code"])
            except Exception as e:
                self.log.emit(f"{name} ({nwsli}): {MSG_CAMERAS_OFFLINE} ({e})")
                return
        if not views:
            self.log.emit(f"{name} ({nwsli}): {MSG_NO_CAMERAS}")
            return
        img_dir = os.path.join(out_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        kinds = (([("still_url", "jpg", "still image")] if self.req["stills"] else [])
                 + ([("timelapse_url", "gif", "time-lapse loop")] if self.req["timelapse"] else []))
        for v in views:
            for key, ext, label in kinds:
                url = v.get(key)
                if not url:
                    continue
                try:
                    time.sleep(REQUEST_DELAY_SEC)
                    data, last_modified = self.sd.download_file(url, REQUEST_TIMEOUT_SEC)
                    path = os.path.join(img_dir, f"{slug(name)}_{nwsli}_{slug(v['direction'])}_"
                                                 f"{stamp_from_last_modified(last_modified)}.{ext}")
                    with open(path, "wb") as fh:
                        fh.write(data)
                    self.log.emit(f"{name} ({nwsli}): saved {v['direction']} {label}.")
                except Exception as e:
                    self.log.emit(f"{name} ({nwsli}): {v['direction']} {label} not downloaded ({e}). "
                                  f"{MSG_CAMERAS_OFFLINE}")


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
class SDMesonetDlg(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(PLUGIN["title"])
        self.resize(*PLUGIN["size"])

        self.sd = SDMesonet()
        self.stations_df = None
        self.camera_table = {}            # code -> list of views ([] = no cameras)
        self.camera_failed = set()        # codes whose dashboard could not be read
        self.camera_pending = set()
        self.result = None
        self._workers = []
        self._data_worker = None
        self._log_lines = []
        self._settings = plugin_settings(__file__)

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
        banner = QLabel(f"{WARNING_TEXT}<br>{TERMS_TEXT}")
        banner.setWordWrap(True)
        banner.setStyleSheet("background:#fde7b0; color:#111; padding:6px 10px; font-weight:600;"
                             "border-bottom:1px solid #d9b65a;")
        root.addWidget(banner)

        body = QHBoxLayout()
        body.setContentsMargins(8, 8, 8, 8)
        root.addLayout(body, stretch=1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel = self._build_left_panel()
        panel.setFixedWidth(LEFT_PANEL_WIDTH - self.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent))
        scroll.setWidget(panel)
        scroll.setFixedWidth(LEFT_PANEL_WIDTH)
        body.addWidget(scroll)
        body.addWidget(self._build_tabs(), stretch=1)

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
        grp = QGroupBox("Stations (active)")
        v = QVBoxLayout(grp)
        g = self._form_grid()
        g.addWidget(QLabel("Search"), 0, 0)
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Name, county, or NWS ID")
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
        v = QVBoxLayout(grp)
        self.chk_weather = QCheckBox("Weather (all variables)")
        self.chk_weather.setChecked(True)
        v.addWidget(self.chk_weather)
        note = QLabel("Last 48 hours at 5-minute steps; the site offers no longer range. "
                      "Daily history needs the site's Daily Data Request form.")
        note.setWordWrap(True)
        v.addWidget(note)
        lv.addWidget(grp)

        # Camera images
        self.grp_camera = QGroupBox("Camera Images")
        v = QVBoxLayout(self.grp_camera)
        self.chk_stills = QCheckBox("Latest still images")
        self.chk_timelapse = QCheckBox("Time-lapse loops (GIF)")
        v.addWidget(self.chk_stills)
        v.addWidget(self.chk_timelapse)
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

        self._refresh_camera_state()
        return panel

    def _build_tabs(self):
        self.tabs = QTabWidget()

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

        cam = QWidget()
        self.camera_layout = QVBoxLayout(cam)
        self.lbl_camera_title = QLabel("Select a station.")
        self.lbl_camera_title.setStyleSheet("font-weight:600;")
        self.camera_layout.addWidget(self.lbl_camera_title)
        cam_scroll = QScrollArea()
        cam_scroll.setWidgetResizable(True)
        cam_scroll.setFrameShape(QFrame.NoFrame)
        cam_body = QWidget()
        self.camera_grid = QGridLayout(cam_body)
        self.camera_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        cam_scroll.setWidget(cam_body)
        self.camera_layout.addWidget(cam_scroll, stretch=1)
        self.tabs.addTab(cam, "Camera")

        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.tabs.addTab(self.txt_log, "Log")
        return self.tabs

    # ------------------------------------------------------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------------------------------------------------------
    def _bind_settings(self):
        st = self._settings
        st.bind(self.chk_weather, "weather")
        st.bind(self.chk_stills, "camera_stills")
        st.bind(self.chk_timelapse, "camera_timelapse")
        st.bind(self.txt_folder, "output_folder")
        st.bind(self.chk_csv, "format_csv")
        st.bind(self.chk_xlsx, "format_xlsx")
        st.bind(self.chk_json, "format_json")

    def _save(self, key, value):
        self._settings[key] = value
        self._settings.save()

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
        self._start(CallWorker(self.sd.get_dataframe), self._on_stations_loaded,
                    lambda e: self.lbl_status.setText(f"Station list not loaded: {e}"))

    def _on_stations_loaded(self, df):
        df = df[df["status"] == "Active"].copy()
        df["code"] = df["station_url"].apply(SDMesonet.station_code)
        df["timezone"] = df["utc_offset"].apply(SDMesonet.timezone_for)
        skipped = df[df["code"].isna() | df["timezone"].isna()]
        for _, r in skipped.iterrows():
            self._log(f"{r['station']} ({r['nwsli']}): no station code or time zone; not listed.")
        df = df.drop(skipped.index)
        df["code"] = df["code"].astype(int)
        self.stations_df = df.sort_values("station").reset_index(drop=True)

        saved = set(self._settings.get("stations", []))
        self.lst_stations.blockSignals(True)
        self.lst_stations.clear()
        for _, r in self.stations_df.iterrows():
            item = QListWidgetItem(f"{r['station'].strip()} ({r['nwsli']})")
            item.setData(Qt.UserRole, int(r["code"]))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if int(r["code"]) in saved else Qt.Unchecked)
            self.lst_stations.addItem(item)
        self.lst_stations.blockSignals(False)
        self.lbl_station_info.setText(f"{len(self.stations_df)} active stations. Check stations to download; "
                                      "click one to see its details.")
        self._log(f"Loaded {len(self.stations_df)} active stations.")
        self._on_station_checked(None)

    # ------------------------------------------------------------------------------------------------------------------
    # Stations
    # ------------------------------------------------------------------------------------------------------------------
    def _row(self, code):
        return self.stations_df[self.stations_df["code"] == code].iloc[0]

    def _selected_codes(self):
        return [self.lst_stations.item(i).data(Qt.UserRole) for i in range(self.lst_stations.count())
                if self.lst_stations.item(i).checkState() == Qt.Checked]

    def _filter_stations(self, text):
        text = text.strip().lower()
        for i in range(self.lst_stations.count()):
            item = self.lst_stations.item(i)
            r = self._row(item.data(Qt.UserRole))
            hay = f"{r['station']} {r['nwsli']} {r['county']}".lower()
            item.setHidden(bool(text) and text not in hay)

    def _clear_selection(self):
        self.lst_stations.blockSignals(True)
        for i in range(self.lst_stations.count()):
            self.lst_stations.item(i).setCheckState(Qt.Unchecked)
        self.lst_stations.blockSignals(False)
        self._on_station_checked(None)

    def _on_station_checked(self, _item):
        codes = self._selected_codes()
        self.lbl_selected.setText(f"{len(codes)} selected")
        if _item is not None or not codes:
            self._save("stations", codes)
        for code in codes:
            self._check_cameras(code)
        self._refresh_camera_state()

    def _on_station_highlighted(self, item, _prev):
        if item is None or self.stations_df is None:
            return
        code = item.data(Qt.UserRole)
        r = self._row(code)
        self.lbl_station_info.setText(
            f"<b>{r['station'].strip()} ({r['nwsli']})</b><br>"
            f"<b>County:</b> {r['county']}<br>"
            f"<b>Location:</b> {r['detail']}<br>"
            f"<b>Time zone:</b> {r['timezone']}<br>"
            f"<b>Elevation:</b> {r['elv_ft']} ft<br>"
            f"<a href=\"{r['station_url']}\">Open station page</a>")
        self._check_cameras(code)
        self._show_camera_preview(code)

    # ------------------------------------------------------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------------------------------------------------------
    def _check_cameras(self, code):
        """Read the station dashboard once to learn its camera views."""
        if code in self.camera_table or code in self.camera_pending:
            return
        self.camera_pending.add(code)
        self._start(CallWorker(self.sd.get_station_images, code),
                    lambda views, c=code: self._on_cameras(c, views, None),
                    lambda err, c=code: self._on_cameras(c, None, err))

    def _on_cameras(self, code, views, err):
        self.camera_pending.discard(code)
        if err is None:
            self.camera_table[code] = views
            self.camera_failed.discard(code)
        else:
            self.camera_failed.add(code)
            self._log(f"Camera check failed for station code {code}: {err}")
        self._refresh_camera_state()
        cur = self.lst_stations.currentItem()
        if cur is not None and cur.data(Qt.UserRole) == code:
            self._show_camera_preview(code)

    def _refresh_camera_state(self):
        codes = self._selected_codes() if self.stations_df is not None else []
        if not codes:
            self.grp_camera.setEnabled(False)
            self.lbl_camera.setText("Select one or more stations.")
            return
        with_cams = [c for c in codes if self.camera_table.get(c)]
        if with_cams:
            self.grp_camera.setEnabled(True)
            self.lbl_camera.setText(f"Latest only; no image archive is available. Cameras at "
                                    f"{len(with_cams)} of {len(codes)} selected station(s).")
            return
        self.grp_camera.setEnabled(False)
        if any(c in self.camera_pending for c in codes):
            self.lbl_camera.setText("Checking cameras…")
        elif any(c in self.camera_failed for c in codes):
            self.lbl_camera.setText(MSG_CAMERAS_OFFLINE)
        else:
            self.lbl_camera.setText(MSG_NO_CAMERAS)

    def _clear_camera_grid(self):
        while self.camera_grid.count():
            w = self.camera_grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()

    def _show_camera_preview(self, code):
        self._clear_camera_grid()
        r = self._row(code)
        label = f"{r['station'].strip()} ({r['nwsli']})"
        if code in self.camera_pending or (code not in self.camera_table and code not in self.camera_failed):
            self.lbl_camera_title.setText(f"{label}: Checking cameras…")
            return
        if code in self.camera_failed:
            self.lbl_camera_title.setText(f"{label}: {MSG_CAMERAS_OFFLINE}")
            return
        views = self.camera_table.get(code) or []
        if not views:
            self.lbl_camera_title.setText(f"{label}: {MSG_NO_CAMERAS}")
            return
        self.lbl_camera_title.setText(f"Latest camera images: {label}")
        for n, v in enumerate(views):
            frame = QWidget()
            box = QVBoxLayout(frame)
            pic = QLabel("Loading…")
            pic.setFixedWidth(THUMB_WIDTH)
            pic.setAlignment(Qt.AlignCenter)
            cap = QLabel(f"<b>{v['direction']}</b>")
            box.addWidget(pic)
            box.addWidget(cap)
            self.camera_grid.addWidget(frame, n // THUMB_COLUMNS, n % THUMB_COLUMNS)
            if v.get("still_url"):
                self._start(CallWorker(self.sd.download_file, v["still_url"], REQUEST_TIMEOUT_SEC),
                            lambda res, p=pic, c=cap, d=v["direction"]: self._set_thumb(p, c, d, res),
                            lambda e, p=pic: p.setText(MSG_CAMERAS_OFFLINE))
            else:
                pic.setText("No still image for this view.")

    @staticmethod
    def _set_thumb(label, caption, direction, result):
        data, last_modified = result
        try:
            label.setPixmap(QPixmap.fromImage(image_from_bytes(data)).scaledToWidth(THUMB_WIDTH,
                                                                                    Qt.SmoothTransformation))
            if last_modified:
                caption.setText(f"<b>{direction}</b>, published {last_modified}")
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
        codes = self._selected_codes()
        if not codes:
            raise ValueError("Select at least one station.")
        cams_on = self.grp_camera.isEnabled()
        stills = cams_on and self.chk_stills.isChecked()
        timelapse = cams_on and self.chk_timelapse.isChecked()
        if not self.chk_weather.isChecked() and not (write_output and (stills or timelapse)):
            raise ValueError("Select Weather, or camera images for Download.")
        formats = [f for f, c in (("csv", self.chk_csv), ("xlsx", self.chk_xlsx), ("json", self.chk_json))
                   if c.isChecked()]
        if write_output:
            if not self.txt_folder.text().strip():
                raise ValueError("Select an output folder.")
            if self.chk_weather.isChecked() and not formats:
                raise ValueError("Select at least one output format.")
        stations = [self._row(c).to_dict() for c in codes]
        for s in stations:
            s["station"] = s["station"].strip()
        return {"stations": stations, "weather": self.chk_weather.isChecked(), "stills": stills,
                "timelapse": timelapse, "camera_table": dict(self.camera_table), "formats": formats,
                "output_folder": self.txt_folder.text().strip()}

    def _run(self, write_output):
        try:
            req = self._request(write_output)
        except ValueError as e:
            QMessageBox.warning(self, PLUGIN["title"], str(e))
            return
        self._log_lines = []
        self._log("-" * 60)
        self._log(f"{'Download' if write_output else 'Preview'}: {len(req['stations'])} station(s), last 48 hours.")
        self._set_busy(True)
        self._data_worker = w = SDDataWorker(self.sd, req, write_output)
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
        self.lbl_status.setText("Working…" if busy else "Ready.")

    def _on_data_done(self, result):
        self._set_busy(False)
        self.result = result
        df = result["all_stations"]
        self.table_model.set_frame(df.head(TABLE_PREVIEW_ROWS))
        self.lbl_table.setVisible(len(df) == 0)
        self.lbl_table.setText(MSG_NO_ROWS)
        numeric = [c for c in df.columns if str(df[c].dtype).startswith(("float", "int"))]
        self.cmb_variable.blockSignals(True)
        self.cmb_variable.clear()
        self.cmb_variable.addItems(numeric)
        self.cmb_variable.blockSignals(False)
        self._draw_chart()
        if result["output_dir"]:
            self._log(f"Saved to {result['output_dir']}")
            self._write_log_file(result["output_dir"])
            self.lbl_status.setText(f"Saved to {result['output_dir']}")
        else:
            self.lbl_status.setText(f"Preview: {len(df)} rows.")

    def _on_data_error(self, msg):
        self._set_busy(False)
        self._log(msg)
        self.lbl_status.setText("Stopped. See the Log tab.")

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
        for st in self.result["stations"]:
            part = df[df["nwsli"] == st["nwsli"]]
            if var in part.columns and part[var].notna().any():
                ax.plot(part["timestamp_utc"].dt.tz_localize(None), part[var],
                        label=f"{st['station']} ({st['nwsli']})")
        ax.set_ylabel(var, color="#111")
        ax.set_xlabel("Time (UTC)", color="#111")
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
        self._log_lines.append(line)

    def _write_log_file(self, out_dir):
        try:
            with open(os.path.join(out_dir, "download_log.txt"), "w", encoding="utf-8") as fh:
                fh.write(f"{WARNING_TEXT}\n{TERMS_TEXT}\n\n" + "\n".join(self._log_lines) + "\n")
        except OSError as e:
            self._log(f"Log file not written: {e}")

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
class SDMesonetPlugin(QtWidgets.QWidget):
    """Hosts the dialog inside a plugin window, the same way as the other plugins."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._dialog = SDMesonetDlg(self)
        self._dialog.setWindowFlags(Qt.Widget)
        layout.addWidget(self._dialog)

    @property
    def dialog(self):
        return self._dialog


def main(argv=None):
    import sys
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    dlg = SDMesonetDlg()
    dlg.show()
    return app.exec_()


if __name__ == "__main__":
    import sys
    sys.exit(main())
