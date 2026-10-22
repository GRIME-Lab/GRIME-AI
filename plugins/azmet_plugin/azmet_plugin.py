#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# azmet_plugin.py
# EXPERIMENTAL plugin: downloads Arizona Meteorological Network (AZMet) data from the public AZMet API
# (api.azmet.arizona.edu): 15-minute, hourly, daily, leaf wetness 15-minute and leaf wetness daily.
#
# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC

import os
import re
import json
import time
import datetime
import traceback

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate, QAbstractTableModel, QModelIndex
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox, QCheckBox,
    QGroupBox, QProgressBar, QFileDialog, QMessageBox, QLineEdit, QListWidget, QListWidgetItem,
    QTabWidget, QTableView, QPlainTextEdit, QScrollArea, QWidget, QFrame, QDateEdit
)

try:
    from appcore.geomaps.AZMET import AZMet, DATA_TYPES, EARLIEST_DATA_DATE, AZMET_TIMEZONE
except ImportError:
    from GRIME_AI.geomaps.AZMET import AZMet, DATA_TYPES, EARLIEST_DATA_DATE, AZMET_TIMEZONE

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
                print(f"[azmet_plugin] Could not save settings: {err}")

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
    "title":       "Arizona AZMet (EXPERIMENTAL)",
    "class":       "AZMetPlugin",
    "description": "Download Arizona Meteorological Network (AZMet) data from the public AZMet API (experimental)",
    "surface":     "tools",
    "size":        [1280, 860],
    "api_version": 2,
}

NOTICE_TEXT = ("Uses the public AZMet API (api.azmet.arizona.edu). Rows with meta_needs_review = 1 have not been "
               "reviewed by AZMet and are provisional.")

# ----------------------------------------------------------------------------------------------------------------------
# SETTINGS (prototype: module level, not saved to the settings file)
# ----------------------------------------------------------------------------------------------------------------------
REQUEST_TIMEOUT_SEC  = 60
REQUEST_DELAY_SEC    = 0.5     # pause between requests to the AZMet API
CHUNK_HOURS          = 720     # longest span per request for 15-minute and hourly data
CHUNK_DAYS           = 365     # longest span per request for daily data
DEFAULT_DAYS_BACK    = 7       # initial date range when nothing is saved yet
LEFT_PANEL_WIDTH     = 340
STATION_LIST_HEIGHT  = 220
VARIABLE_LIST_CHARS  = 40
TABLE_PREVIEW_ROWS   = 5000

MSG_NO_DATA_YET = "Click Preview to load data."
MSG_NO_ROWS     = "No data returned. See the Log tab."


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def sheet_name(text):
    return re.sub(r"[\[\]:*?/\\]", "-", str(text))[:31]


def wall_clock(series):
    """Time-zone-aware timestamps -> naive wall-clock datetimes (Excel cannot store time zones)."""
    import pandas as pd
    return pd.to_datetime(series.apply(lambda x: x.tz_localize(None) if getattr(x, "tzinfo", None) else x))


# ======================================================================================================================
# Workers
# ======================================================================================================================
class CallWorker(QThread):
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


class AZMetDataWorker(QThread):
    """Fetches each selected data type for each selected station; for Download also writes the files."""
    progress    = pyqtSignal(int, int)
    log         = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    error       = pyqtSignal(str)

    def __init__(self, az, request, write_output):
        super().__init__()
        self.az = az
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
        stations, types = req["stations"], req["data_types"]
        steps, step = len(stations) * len(types), 0
        frames = {t: {} for t in types}        # data type -> station_id -> DataFrame

        for t in types:
            for st in stations:
                if self._abort:
                    raise RuntimeError("Aborted by user.")
                step += 1
                self.progress.emit(step, steps)
                sid, name = st["station_id"], st["station"]
                time.sleep(REQUEST_DELAY_SEC)
                try:
                    df = self.az.get_observations(t, sid, req["start_date"], req["end_date"])
                    for err in dict.fromkeys(str(e) for e in df.attrs.get("api_errors", [])):
                        self.log.emit(f"{name} ({sid}): {DATA_TYPES[t][0]}: API message: {err}")
                    self.log.emit(f"{name} ({sid}): {DATA_TYPES[t][0]}: {len(df)} rows.")
                except Exception as e:
                    df = pd.DataFrame()
                    self.log.emit(f"{name} ({sid}): {DATA_TYPES[t][0]} request failed: {e}")
                frames[t][sid] = df

        all_stations = {}
        for t in types:
            parts = []
            for st in stations:
                d = frames[t][st["station_id"]].copy()
                if d.empty:
                    continue
                d.insert(0, "station", st["station"])
                d.insert(0, "station_id", st["station_id"])
                parts.append(d)
            all_stations[t] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        result = {"stations": stations, "types": types, "frames": frames, "all_stations": all_stations,
                  "info": self._info(stations, types), "output_dir": None}
        if self.write_output:
            result["output_dir"] = self._write(result)
        self.progress.emit(steps, steps)
        return result

    def _info(self, stations, types):
        r = self.req
        return [
            ("generated_local", datetime.datetime.now().isoformat(timespec="seconds")),
            ("source", "Arizona Meteorological Network (AZMet), api.azmet.arizona.edu"),
            ("notice", NOTICE_TEXT),
            ("data_types", ", ".join(DATA_TYPES[t][0] for t in types)),
            ("start_date", r["start_date"].isoformat() if r["start_date"] else "API default"),
            ("end_date", r["end_date"].isoformat() if r["end_date"] else "API default"),
            ("stations", "; ".join(f"{s['station']} ({s['station_id']})" for s in stations)),
            ("timestamp_local", f"Arizona time ({AZMET_TIMEZONE}, UTC-7, no daylight saving time)"),
            ("timestamp_utc", "UTC"),
            ("hourly_timestamps", "end of each hour; midnight is written as 23:59:59"),
            ("units", "as provided by AZMet, in field-name suffixes (C/F, mm/in, mps/mph); "
                      "see https://azmet.arizona.edu/about"),
            ("empty_cells", "no measurement (AZMet -999/-9999/-99999) or no record; see download_log.txt"),
        ]

    def _write(self, result):
        import pandas as pd
        r = self.req
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(r["output_folder"], f"AZMET_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        info_df = pd.DataFrame(result["info"], columns=["item", "value"])
        stations = result["stations"]

        if "csv" in r["formats"]:
            info_df.to_csv(os.path.join(out_dir, "info.csv"), index=False)
            for t in result["types"]:
                type_dir = os.path.join(out_dir, t)
                if not result["all_stations"][t].empty:
                    os.makedirs(type_dir, exist_ok=True)
                for st in stations:
                    df = result["frames"][t][st["station_id"]]
                    if not df.empty:
                        df.to_csv(os.path.join(type_dir, f"{slug(st['station'])}_{st['station_id']}.csv"),
                                  index=False)
                if result["all_stations"][t].empty:
                    self.log.emit(f"{DATA_TYPES[t][0]}: no rows for the selected stations; no files written.")
                else:
                    result["all_stations"][t].to_csv(os.path.join(type_dir, "all_stations.csv"), index=False)
            self.log.emit("Wrote CSV files (one folder per data type).")

        if "xlsx" in r["formats"]:
            for t in result["types"]:
                if result["all_stations"][t].empty:
                    continue
                path = os.path.join(out_dir, f"AZMET_{ts}_{t}.xlsx")
                try:
                    with pd.ExcelWriter(path) as xw:
                        self._excel_ready(result["all_stations"][t]).to_excel(xw, sheet_name="All Stations",
                                                                              index=False)
                        used = {"All Stations", "Info"}
                        for st in stations:
                            df = result["frames"][t][st["station_id"]]
                            if df.empty:
                                continue
                            name = sheet_name(f"{st['station']} ({st['station_id']})")
                            while name in used:
                                name = sheet_name(f"{st['station_id']} {name}")
                            used.add(name)
                            self._excel_ready(df).to_excel(xw, sheet_name=name, index=False)
                        info_df.to_excel(xw, sheet_name="Info", index=False)
                    self.log.emit(f"Wrote XLSX workbook for {DATA_TYPES[t][0]}.")
                except Exception as e:
                    self.log.emit(f"XLSX for {DATA_TYPES[t][0]} not written: {e}")

        if "json" in r["formats"]:
            def records(df):
                return json.loads(df.to_json(orient="records", date_format="iso")) if not df.empty else []
            payload = {"info": dict(result["info"]), "data_types": {}}
            for t in result["types"]:
                payload["data_types"][t] = {
                    "label": DATA_TYPES[t][0],
                    "all_stations": records(result["all_stations"][t]),
                    "stations": {st["station_id"]: {"station": st["station"],
                                                    "data": records(result["frames"][t][st["station_id"]])}
                                 for st in stations},
                }
            with open(os.path.join(out_dir, f"AZMET_{ts}.json"), "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1)
            self.log.emit("Wrote JSON file.")
        return out_dir

    @staticmethod
    def _excel_ready(df):
        df = df.copy()
        for c in ("timestamp_local", "timestamp_utc"):
            if c in df.columns:
                df[c] = wall_clock(df[c])
        return df


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
class AZMetDlg(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(PLUGIN["title"])
        self.resize(*PLUGIN["size"])
        self.az = AZMet(timeout_sec=REQUEST_TIMEOUT_SEC, chunk_hours=CHUNK_HOURS, chunk_days=CHUNK_DAYS)
        self.stations_df = None
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
        banner = QLabel(NOTICE_TEXT)
        banner.setWordWrap(True)
        banner.setStyleSheet("background:#e3f0e3; color:#111; padding:6px 10px; font-weight:600;"
                             "border-bottom:1px solid #9cc79c;")
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

        grp = QGroupBox("Stations")
        v = QVBoxLayout(grp)
        g = self._form_grid()
        g.addWidget(QLabel("Search"), 0, 0)
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Name, symbol, county, or ID")
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
        self.lbl_station_info.setStyleSheet("background:#f3f5f8; border:1px solid #dde1e6; padding:6px; color:#111;")
        v.addWidget(self.lbl_station_info)
        lv.addWidget(grp)

        grp = QGroupBox("Data")
        v = QVBoxLayout(grp)
        self.type_checks = {}
        for t, (label, _field, _daily) in DATA_TYPES.items():
            self.type_checks[t] = QCheckBox(label)
            v.addWidget(self.type_checks[t])
        self.type_checks["hourly"].setChecked(True)
        g = self._form_grid()
        earliest = QDate.fromString(EARLIEST_DATA_DATE, "yyyy-MM-dd")
        today = QDate.currentDate()
        self.chk_latest = QCheckBox("Most recent (API default)")
        g.addWidget(self.chk_latest, 0, 0, 1, 2)
        g.addWidget(QLabel("Start"), 1, 0)
        self.date_start = QDateEdit(today.addDays(-DEFAULT_DAYS_BACK))
        g.addWidget(self.date_start, 1, 1)
        g.addWidget(QLabel("End"), 2, 0)
        self.date_end = QDateEdit(today.addDays(-1))
        g.addWidget(self.date_end, 2, 1)
        for d in (self.date_start, self.date_end):
            d.setCalendarPopup(True)
            d.setDisplayFormat("yyyy-MM-dd")
            d.setDateRange(earliest, today)
        self.chk_latest.toggled.connect(self._refresh_dates)
        v.addLayout(g)
        note = QLabel(f"Hourly and daily data from {EARLIEST_DATA_DATE}. 15-minute data covers only the most "
                      "recent period AZMet keeps. Leaf wetness is measured at some stations only.")
        note.setWordWrap(True)
        v.addWidget(note)
        lv.addWidget(grp)

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
        lv.addWidget(self.progress)
        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setWordWrap(True)
        lv.addWidget(self.lbl_status)
        lv.addStretch()
        return panel

    def _build_tabs(self):
        self.tabs = QTabWidget()
        chart = QWidget()
        cv = QVBoxLayout(chart)
        row = QHBoxLayout()
        row.addWidget(QLabel("Data type"))
        self.cmb_type = QComboBox()
        self.cmb_type.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.cmb_type.setMinimumContentsLength(max(len(v[0]) for v in DATA_TYPES.values()))
        self.cmb_type.currentIndexChanged.connect(self._on_type_changed)
        row.addWidget(self.cmb_type)
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

        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.tabs.addTab(self.txt_log, "Log")
        return self.tabs

    # ------------------------------------------------------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------------------------------------------------------
    def _bind_settings(self):
        st = self._settings
        for t, chk in self.type_checks.items():
            st.bind(chk, f"type_{t}")
        st.bind(self.chk_latest, "most_recent")
        st.bind(self.txt_folder, "output_folder")
        st.bind(self.chk_csv, "format_csv")
        st.bind(self.chk_xlsx, "format_xlsx")
        st.bind(self.chk_json, "format_json")
        for widget, key in ((self.date_start, "start_date"), (self.date_end, "end_date")):
            saved = QDate.fromString(st.get(key) or "", "yyyy-MM-dd")
            if saved.isValid():
                widget.setDate(saved)
            widget.dateChanged.connect(lambda d, k=key: self._save(k, d.toString("yyyy-MM-dd")))
        self._refresh_dates()

    def _save(self, key, value):
        self._settings[key] = value
        self._settings.save()

    def _refresh_dates(self, *_):
        on = not self.chk_latest.isChecked()
        self.date_start.setEnabled(on)
        self.date_end.setEnabled(on)

    # ------------------------------------------------------------------------------------------------------------------
    # Stations
    # ------------------------------------------------------------------------------------------------------------------
    def _start(self, worker, on_result, on_error):
        worker.result.connect(on_result)
        worker.error.connect(on_error)
        worker.finished.connect(lambda: self._workers.remove(worker) if worker in self._workers else None)
        self._workers.append(worker)
        worker.start()

    def _load_stations(self):
        self._start(CallWorker(self.az.get_dataframe), self._on_stations_loaded,
                    lambda e: self.lbl_status.setText(f"Station list not loaded: {e}"))

    def _on_stations_loaded(self, df):
        self.stations_df = df.sort_values("station").reset_index(drop=True)
        saved = set(self._settings.get("stations", []))
        self.lst_stations.blockSignals(True)
        self.lst_stations.clear()
        for _, r in self.stations_df.iterrows():
            item = QListWidgetItem(f"{r['station']} ({r['station_id']})")
            item.setData(Qt.UserRole, r["station_id"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if r["station_id"] in saved else Qt.Unchecked)
            self.lst_stations.addItem(item)
        self.lst_stations.blockSignals(False)
        self.lbl_station_info.setText(f"{len(self.stations_df)} stations. Check stations to download; "
                                      "click one to see its details.")
        self._log(f"Loaded {len(self.stations_df)} stations.")
        self._on_station_checked(None)

    def _row(self, sid):
        return self.stations_df[self.stations_df["station_id"] == sid].iloc[0]

    def _selected_ids(self):
        return [self.lst_stations.item(i).data(Qt.UserRole) for i in range(self.lst_stations.count())
                if self.lst_stations.item(i).checkState() == Qt.Checked]

    def _filter_stations(self, text):
        text = text.strip().lower()
        for i in range(self.lst_stations.count()):
            item = self.lst_stations.item(i)
            r = self._row(item.data(Qt.UserRole))
            hay = f"{r['station']} {r['symbol']} {r['county']} {r['station_id']}".lower()
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

    def _on_station_highlighted(self, item, _prev):
        if item is None or self.stations_df is None:
            return
        r = self._row(item.data(Qt.UserRole))
        self.lbl_station_info.setText(
            f"<b>{r['station']} ({r['station_id']}, {r['symbol']})</b><br>"
            f"<b>County:</b> {r['county']}<br>"
            f"<b>Elevation:</b> {r['elev_m']} m ({r['elev_ft']} ft)<br>"
            f"<b>Status:</b> {r['status']}<br>"
            f"<b>Data since:</b> {r['start_date']}")

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
        types = [t for t, chk in self.type_checks.items() if chk.isChecked()]
        if not types:
            raise ValueError("Select at least one data type.")
        start = end = None
        if not self.chk_latest.isChecked():
            start, end = self.date_start.date().toPyDate(), self.date_end.date().toPyDate()
            if start > end:
                raise ValueError("Start date is after end date.")
        formats = [f for f, c in (("csv", self.chk_csv), ("xlsx", self.chk_xlsx), ("json", self.chk_json))
                   if c.isChecked()]
        if write_output:
            if not self.txt_folder.text().strip():
                raise ValueError("Select an output folder.")
            if not formats:
                raise ValueError("Select at least one output format.")
        return {"stations": [self._row(s).to_dict() for s in ids], "data_types": types,
                "start_date": start, "end_date": end, "formats": formats,
                "output_folder": self.txt_folder.text().strip()}

    def _run(self, write_output):
        try:
            req = self._request(write_output)
        except ValueError as e:
            QMessageBox.warning(self, PLUGIN["title"], str(e))
            return
        self._log_lines = []
        self._log("-" * 60)
        rng = (f"{req['start_date']} to {req['end_date']}" if req["start_date"] else "most recent")
        self._log(f"{'Download' if write_output else 'Preview'}: {len(req['stations'])} station(s), "
                  f"{', '.join(DATA_TYPES[t][0] for t in req['data_types'])}, {rng}.")
        self._set_busy(True)
        self._data_worker = w = AZMetDataWorker(self.az, req, write_output)
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
        self.cmb_type.blockSignals(True)
        self.cmb_type.clear()
        for t in result["types"]:
            self.cmb_type.addItem(DATA_TYPES[t][0], t)
        self.cmb_type.blockSignals(False)
        self._on_type_changed()
        total = sum(len(df) for df in result["all_stations"].values())
        if result["output_dir"]:
            self._log(f"Saved to {result['output_dir']}")
            self._write_log_file(result["output_dir"])
            self.lbl_status.setText(f"Saved to {result['output_dir']}")
        else:
            self.lbl_status.setText(f"Preview: {total} rows.")

    def _on_data_error(self, msg):
        self._set_busy(False)
        self._log(msg)
        self.lbl_status.setText("Stopped. See the Log tab.")

    def _current_frame(self):
        if self.result is None or self.cmb_type.currentData() is None:
            return None
        return self.result["all_stations"][self.cmb_type.currentData()]

    def _on_type_changed(self, *_):
        df = self._current_frame()
        if df is None:
            return
        self.table_model.set_frame(df.head(TABLE_PREVIEW_ROWS))
        self.lbl_table.setVisible(df.empty)
        self.lbl_table.setText(MSG_NO_ROWS)
        # Calendar fields (date_doy, date_year, ...) are not measurements, so they are not offered for charting.
        numeric = [c for c in df.columns if str(df[c].dtype).startswith(("float", "int"))
                   and not c.startswith("date_")]
        self.cmb_variable.blockSignals(True)
        self.cmb_variable.clear()
        self.cmb_variable.addItems(numeric)
        preferred = [c for c in numeric if c.startswith("temp_air")]
        if preferred:
            self.cmb_variable.setCurrentText(preferred[0])
        self.cmb_variable.blockSignals(False)
        self._draw_chart()

    def _chart_message(self, text):
        self.figure.clear()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", fontsize=14, color="#111")
        self.canvas.draw()

    def _draw_chart(self, *_):
        if self.figure is None:
            return
        df = self._current_frame()
        var = self.cmb_variable.currentText()
        if df is None or df.empty or not var:
            self._chart_message(MSG_NO_ROWS if self.result is not None else MSG_NO_DATA_YET)
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x_col = "date" if "date" in df.columns else "timestamp_local"
        for st in self.result["stations"]:
            part = df[df["station_id"] == st["station_id"]]
            if var in part.columns and part[var].notna().any():
                x = wall_clock(part[x_col]) if x_col == "timestamp_local" else part[x_col]
                ax.plot(x, part[var], label=f"{st['station']} ({st['station_id']})")
        ax.set_ylabel(var, color="#111")
        ax.set_xlabel("Date" if x_col == "date" else "Arizona time", color="#111")
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
                fh.write(NOTICE_TEXT + "\n\n" + "\n".join(self._log_lines) + "\n")
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
class AZMetPlugin(QtWidgets.QWidget):
    """Hosts the dialog inside a plugin window, the same way as the other plugins."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._dialog = AZMetDlg(self)
        self._dialog.setWindowFlags(Qt.Widget)
        layout.addWidget(self._dialog)

    @property
    def dialog(self):
        return self._dialog


def main(argv=None):
    import sys
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    dlg = AZMetDlg()
    dlg.show()
    return app.exec_()


if __name__ == "__main__":
    import sys
    sys.exit(main())
