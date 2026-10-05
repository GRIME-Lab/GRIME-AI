#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# nwis_downloader.py
#
# Plugin: USGS Water Data Downloader.
#
# Tab 1 (Sites): OpenStreetMap of USGS monitoring locations in the visible map
# area. Hover a pin for site metadata, click to select or deselect, shift+drag to
# select every pin in a box, or type site IDs directly. Selecting a site fetches
# the list of time series it carries and their periods of record.
#
# Tab 2 (Data): data types, parameters, date range and output options, then
# download for every selected site.
#
# Data comes from the USGS Water Data APIs through the USGS dataretrieval
# package (waterdata module). Output goes to
#   <output folder>/NWIS_Download_YYYYMMDD_HHMMSS/raw/          dataretrieval results written straight to CSV
#   <output folder>/NWIS_Download_YYYYMMDD_HHMMSS/reformatted/  human-readable labels, optional .xlsx
# Every reformatted row carries Site ID, Site Name, Latitude and Longitude.
#
# Requires the host application (map widget, API key manager) and the
# dataretrieval package (pip install dataretrieval).

import ast
import html
import importlib
import json
import logging
import os
import re
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QDate, QSize
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
                             QProgressBar, QPlainTextEdit, QFileDialog, QMessageBox, QTabWidget,
                             QComboBox, QCheckBox, QLineEdit, QTreeWidget, QTreeWidgetItem,
                             QListWidget, QListWidgetItem, QDateEdit, QSplitter, QHeaderView,
                             QGridLayout, QSizePolicy)

from appcore.app_identity import PLUGINS_DIR
from appcore.dialogs.api_keys.api_key_manager import APIKeyManager
from appcore.dialogs.api_keys.api_key_verifier import verify_usgs_key

try:
    import dataretrieval
    from dataretrieval import waterdata as wd
    _DATARETRIEVAL_ERROR = ""
except Exception as _err:
    dataretrieval = None
    wd = None
    _DATARETRIEVAL_ERROR = f"{type(_err).__name__}: {_err}"

PLUGIN = {
    "title":       "USGS Water Data Downloader",
    "class":       "NWISDownloaderTab",
    "description": "Pick USGS gages on a map and download their data (raw and reformatted)",
    "surface":     "tools",
    "api_version": 2,
}

USER_ROOT = Path(str(PLUGINS_DIR)).parent
SETTINGS_FILE = USER_ROOT / "Settings" / "nwis_downloader.json"

# Tuning values. Every one can be overridden in the settings file.
DEFAULT_SETTINGS = {
    "output_folder":          "",
    "site_type_code":         "ST",       # USGS site type code for map pins; "" shows all types
    "continuous_only":        False,      # only show sites with continuous (instantaneous) series
    "max_pins":               2000,       # beyond this, ask the user to zoom in
    "bounds_debounce_ms":     500,        # wait after pan/zoom before querying sites
    "continuous_chunk_days":  365,        # continuous data is requested in chunks this long
    "default_span_days":      365,        # default date range when none is saved
    "initial_center":         [41.5, -99.5],
    "initial_zoom":           7,
    "selected_color":         "red",      # pin icons (used when marker_style is "pin")
    "unselected_color":       "blue",
    "marker_style":           "dot",      # "dot" or "pin"
    "dot_radius":             5,          # px
    "dot_border_width":       1,          # px
    "dot_fill_opacity":       0.9,
    "dot_selected_fill":      "#e0393e",
    "dot_selected_border":    "#7a1016",
    "dot_unselected_fill":    "#3b8fd6",
    "dot_unselected_border":  "#15426b",
    "full_record":            False,
    "start_date":             "",
    "end_date":               "",
    "write_raw":              True,
    "write_reformatted":      True,
    "write_xlsx":             False,
    "data_types":             ["site_metadata", "continuous", "daily"],
    "samples_group":          "",
    "http_log_level":         "WARNING",  # httpx/httpcore logging used by dataretrieval
    "window_size":            [1330, 650],  # starting size of the plugin window (width, height)
    "map_width_fraction":     0.7,        # share of the Sites tab width given to the map at startup
    "status_lines":           2,          # visible lines in the Status box
    "add_button_color":       "steelblue",
    "remove_button_color":    "#d9363e",  # red ghost button
    "clear_button_border":    "#a5d8f3",  # ice blue
}

# Excel's worksheet row limit, including the header row.
EXCEL_MAX_ROWS = 1048576

DATA_TYPES = [
    ("site_metadata",      "Site metadata and series list"),
    ("continuous",         "Continuous (instantaneous) values"),
    ("daily",              "Daily values"),
    ("field_measurements", "Field measurements"),
    ("samples",            "Water-quality samples"),
    ("statistics",         "Day-of-year statistics (period of record)"),
    ("peaks",              "Annual peaks"),
    ("ratings",            "Rating curves"),
]

MARKER_LAYER = "nwis_sites"


# ======================================================================================================================
# Settings
# ======================================================================================================================
def _load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            settings.update(json.load(f))
    except Exception:
        pass
    return settings


def _save_settings(settings: dict):
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"[nwis_downloader] Could not save settings: {e}")


# ======================================================================================================================
# Host map widget
# ======================================================================================================================
def _import_map_widget():
    """Locate openstreetmap_viewer.py inside the host package and return its OpenStreetMapWidget class."""
    import appcore
    root = Path(appcore.__file__).resolve().parent
    for path in root.rglob("openstreetmap_viewer.py"):
        parts = path.relative_to(root).with_suffix("").parts
        module = importlib.import_module("appcore." + ".".join(parts))
        return module.OpenStreetMapWidget
    raise ImportError("openstreetmap_viewer.py was not found in the application package.")


# ======================================================================================================================
# Helpers
# ======================================================================================================================
def _dataretrieval_config():
    """Context manager applying the stored USGS API key (if any) to dataretrieval calls on this thread."""
    kwargs = {"progress": False}
    key = APIKeyManager().get_usgs_key()
    if key:
        kwargs["api_key"] = key
    return dataretrieval.configure(dataretrieval.Configuration(**kwargs))


def normalize_site_id(text: str) -> str:
    """'06770500' -> 'USGS-06770500'; IDs that already carry an agency prefix are kept."""
    text = text.strip()
    if not text:
        return ""
    return text if "-" in text else f"USGS-{text}"


def _lat_lon(geom):
    """(lat, lon) from a shapely point, a [lon, lat] list, or 'POINT (lon lat)' text; (None, None) otherwise."""
    if geom is None:
        return None, None
    if hasattr(geom, "x") and hasattr(geom, "y"):
        return float(geom.y), float(geom.x)
    if isinstance(geom, (list, tuple)) and len(geom) >= 2:
        return float(geom[1]), float(geom[0])
    if isinstance(geom, str):
        m = re.search(r"POINT\s*\(\s*([-\d.eE]+)\s+([-\d.eE]+)", geom)
        if m:
            return float(m.group(2)), float(m.group(1))
    return None, None


def _code(value, width):
    """USGS code as text with its leading zeros restored (e.g. 60 -> '00060'). Blank stays blank."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in ("", "nan", "none"):
        return ""
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".")[0]
    return text.zfill(width) if text.isdigit() else text


def _flatten_qualifier(value):
    """"['ESTIMATED', 'ICE']" or a list -> 'ESTIMATED; ICE'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, tuple)):
        items = value
    else:
        text = str(value).strip()
        if text.lower() in ("", "nan", "none", "[]"):
            return ""
        items = re.findall(r"'([^']*)'|\"([^\"]*)\"", text)
        items = [a or b for a, b in items] or [text.strip("[]")]
    return "; ".join(str(i).strip() for i in items if str(i).strip())


def _col(df, name):
    """Column by name, or a blank column when the API did not return it."""
    return df[name] if name in df.columns else pd.Series([""] * len(df), index=df.index)


def _utc_text(series):
    return pd.to_datetime(series, utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")


def _date_text(series):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _threshold_time(text):
    """Threshold period bound as 'YYYY-MM-DD HH:MM:SS' (UTC); USGS open-ended bounds (years 0001 / 9999) are blank."""
    if not text or str(text).startswith(("0001-", "9999-")):
        return ""
    t = pd.to_datetime(str(text), utc=True, errors="coerce")
    return "" if pd.isna(t) else t.strftime("%Y-%m-%d %H:%M:%S")


def _site_tooltip(row: dict) -> str:
    fields = [
        ("Site", row.get("monitoring_location_name")),
        ("ID", row.get("monitoring_location_id")),
        ("Type", row.get("site_type")),
        ("County", row.get("county_name")),
        ("State", row.get("state_name")),
        ("HUC", row.get("hydrologic_unit_code")),
        ("Drainage area (sq mi)", row.get("drainage_area")),
        ("Altitude (ft)", row.get("altitude")),
        ("Vertical datum", row.get("vertical_datum")),
        ("Time zone", row.get("time_zone_abbreviation")),
        ("Uses DST", row.get("uses_daylight_savings")),
        ("Latitude", row.get("_lat")),
        ("Longitude", row.get("_lon")),
    ]
    return "<br>".join(f"<b>{html.escape(k)}:</b> {html.escape(str(v))}" for k, v in fields
                       if v is not None and str(v) not in ("", "nan", "None", "NaT"))


def _records_with_coords(df: pd.DataFrame) -> list:
    rows = []
    for rec in df.to_dict("records"):
        lat, lon = _lat_lon(rec.get("geometry"))
        rec["_lat"], rec["_lon"] = lat, lon
        rows.append(rec)
    return rows


# ======================================================================================================================
# Workers
# ======================================================================================================================
class SiteQueryWorker(QThread):
    """Monitoring locations inside a bounding box."""
    done = pyqtSignal(int, list, bool)   # sequence number, site records, overflowed
    failed = pyqtSignal(int, str)

    def __init__(self, seq, bbox, site_type, continuous_only, max_pins):
        super().__init__()
        self._seq, self._bbox, self._site_type = seq, bbox, site_type
        self._continuous_only, self._max_pins = continuous_only, max_pins

    def run(self):
        try:
            with _dataretrieval_config():
                kwargs = {"bbox": self._bbox, "max_rows": self._max_pins + 1}
                if self._site_type:
                    kwargs["site_type_code"] = self._site_type
                df, _ = wd.get_monitoring_locations(**kwargs)
                overflow = len(df) > self._max_pins
                if not overflow and self._continuous_only and not df.empty:
                    ts, _ = wd.get_time_series_metadata(bbox=self._bbox, computation_period_identifier="Points",
                                                        skip_geometry=True)
                    keep = set(ts["monitoring_location_id"]) if not ts.empty else set()
                    df = df[df["monitoring_location_id"].isin(keep)]
            self.done.emit(self._seq, [] if overflow else _records_with_coords(df), overflow)
        except Exception as e:
            self.failed.emit(self._seq, f"{type(e).__name__}: {e}")


class AvailabilityWorker(QThread):
    """Time series available at each site, plus site metadata for sites not already known."""
    done = pyqtSignal(object, list)   # time-series metadata DataFrame, site records for unknown sites
    failed = pyqtSignal(str)

    def __init__(self, site_ids, unknown_ids):
        super().__init__()
        self._site_ids, self._unknown_ids = list(site_ids), list(unknown_ids)

    def run(self):
        try:
            with _dataretrieval_config():
                ts, _ = wd.get_time_series_metadata(monitoring_location_id=self._site_ids)
                sites = []
                if self._unknown_ids:
                    loc, _ = wd.get_monitoring_locations(monitoring_location_id=self._unknown_ids)
                    sites = _records_with_coords(loc)
            self.done.emit(ts, sites)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class SiteTypesWorker(QThread):
    done = pyqtSignal(list)   # [(code, label), ...]
    failed = pyqtSignal(str)

    def run(self):
        try:
            with _dataretrieval_config():
                df, _ = wd.get_reference_table("site-types")
            code_col = "id" if "id" in df.columns else df.columns[0]
            label_col = next((c for c in df.columns if "name" in c.lower() or "description" in c.lower()), code_col)
            items = sorted({(str(r[code_col]), str(r[label_col])) for _, r in df.iterrows()})
            self.done.emit(items)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class QuotaWorker(QThread):
    done = pyqtSignal(bool, list)

    def run(self):
        mgr = APIKeyManager()
        ok, lines = verify_usgs_key(mgr.get_usgs_key() or "", mgr.get_usgs_endpoint())
        self.done.emit(ok, lines)


class DownloadWorker(QThread):
    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    finished_all = pyqtSignal(str)   # output folder

    def __init__(self, plan: dict):
        super().__init__()
        self._plan = plan
        self._cancelled = False

    def cancel(self):
        """Stop after the current request; files already written are kept."""
        self._cancelled = True

    # ------------------------------------------------------------------------------------------------------------------
    def run(self):
        p = self._plan
        out_root = Path(p["output_folder"]) / f"NWIS_Download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        raw_dir, fmt_dir = out_root / "raw", out_root / "reformatted"
        if p["write_raw"]:
            raw_dir.mkdir(parents=True, exist_ok=True)
        if p["write_reformatted"]:
            fmt_dir.mkdir(parents=True, exist_ok=True)

        jobs = [(site, dtype) for site in p["sites"] for dtype in p["data_types"]]
        total = len(jobs)
        try:
            with _dataretrieval_config():
                for i, (site, dtype) in enumerate(jobs, start=1):
                    if self._cancelled:
                        self.status.emit(f"Cancelled after {i - 1} of {total} download(s).")
                        break
                    self.status.emit(f"{site}: {dtype}\u2026")
                    try:
                        getattr(self, f"_do_{dtype}")(site, raw_dir, fmt_dir)
                    except Exception as e:
                        self.status.emit(f"  {site} {dtype} FAILED: {type(e).__name__}: {e}")
                    self.progress.emit(i, total)
        except Exception as e:
            self.status.emit(f"ERROR: {e}\n{traceback.format_exc()}")
        self.finished_all.emit(str(out_root))

    # ------------------------------------------------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------------------------------------------------
    def _write(self, site, dtype, raw_df, fmt_df, raw_dir, fmt_dir):
        stem = f"{site}_{dtype}"
        if raw_df is None or raw_df.empty:
            self.status.emit(f"  {site} {dtype}: no data returned")
            return
        if self._plan["write_raw"]:
            raw_df.to_csv(raw_dir / f"{stem}.csv", index=False)
        if self._plan["write_reformatted"] and fmt_df is not None:
            fmt_df.to_csv(fmt_dir / f"{stem}.csv", index=False)
            if self._plan["write_xlsx"]:
                if len(fmt_df) + 1 > EXCEL_MAX_ROWS:
                    self.status.emit(f"  {stem}: {len(fmt_df):,} rows exceeds Excel's limit; .xlsx skipped (CSV written)")
                else:
                    fmt_df.to_excel(fmt_dir / f"{stem}.xlsx", index=False)
        self.status.emit(f"  {site} {dtype}: {len(raw_df):,} rows")

    def _site_columns(self, site, n):
        info = self._plan["site_info"].get(site, {})
        return pd.DataFrame({
            "Site ID": [site] * n,
            "Site Name": [info.get("monitoring_location_name", "")] * n,
            "Latitude": [info.get("_lat")] * n,
            "Longitude": [info.get("_lon")] * n,
        })

    def _with_site(self, site, body: pd.DataFrame) -> pd.DataFrame:
        body = body.reset_index(drop=True)
        return pd.concat([self._site_columns(site, len(body)), body], axis=1)

    def _time_arg(self):
        p = self._plan
        return None if p["full_record"] else f"{p['start_date']}/{p['end_date']}"

    def _pcodes(self):
        return self._plan["parameter_codes"] or None

    # ------------------------------------------------------------------------------------------------------------------
    # Data types
    # ------------------------------------------------------------------------------------------------------------------
    def _do_site_metadata(self, site, raw_dir, fmt_dir):
        loc, _ = wd.get_monitoring_locations(monitoring_location_id=site)
        ts, _ = wd.get_time_series_metadata(monitoring_location_id=site)
        self._write(site, "site", loc, self._fmt_site(site, loc), raw_dir, fmt_dir)
        self._write(site, "series_list", ts, self._fmt_series(site, ts), raw_dir, fmt_dir)
        if self._plan["write_reformatted"]:
            thresholds = self._fmt_thresholds(site, ts)
            if thresholds is not None:
                self._write_reformatted_only(site, "thresholds", thresholds, fmt_dir)

    def _do_continuous(self, site, raw_dir, fmt_dir):
        frames = []
        for start, end in self._continuous_chunks(site):
            if self._cancelled:
                break
            self.status.emit(f"  {site} continuous {start} to {end}")
            df, _ = wd.get_continuous(monitoring_location_id=site, parameter_code=self._pcodes(),
                                      time=f"{start}/{end}")
            if not df.empty:
                frames.append(df)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not raw.empty and "continuous_id" in raw.columns:
            raw = raw.drop_duplicates(subset=["continuous_id"]).reset_index(drop=True)   # chunk edges overlap
        fmt = None
        if not raw.empty:
            fmt = self._with_site(site, pd.DataFrame({
                "Time (UTC)": _utc_text(raw["time"]),
                "Parameter Code": _col(raw, "parameter_code").map(lambda v: _code(v, 5)),
                "Parameter": _col(raw, "time_series_id").map(self._plan["series_param_name"]).fillna(""),
                "Statistic Code": _col(raw, "statistic_id").map(lambda v: _code(v, 5)),
                "Value": _col(raw, "value"),
                "Unit": _col(raw, "unit_of_measure"),
                "Approval Status": _col(raw, "approval_status"),
                "Qualifier": _col(raw, "qualifier").map(_flatten_qualifier),
                "Time Series ID": _col(raw, "time_series_id"),
            }))
        self._write(site, "continuous", raw, fmt, raw_dir, fmt_dir)

    def _continuous_chunks(self, site):
        """Date intervals covering the requested range (or the site's continuous record) in fixed-length chunks."""
        p = self._plan
        if p["full_record"]:
            ts = p["series"]
            if ts.empty:
                return []
            ts = ts[(ts["monitoring_location_id"] == site) & (ts["computation_period_identifier"] == "Points")]
            if p["parameter_codes"]:
                ts = ts[ts["parameter_code"].map(lambda v: _code(v, 5)).isin(p["parameter_codes"])]
            begins = pd.to_datetime(ts["begin_utc"], utc=True, errors="coerce").dropna()
            ends = pd.to_datetime(ts["end_utc"], utc=True, errors="coerce").dropna()
            if begins.empty or ends.empty:
                return []
            start, end = begins.min().date(), ends.max().date()
        else:
            start = date.fromisoformat(p["start_date"])
            end = date.fromisoformat(p["end_date"])
        step = timedelta(days=int(p["continuous_chunk_days"]))
        chunks, cur = [], start
        while cur <= end:
            nxt = min(cur + step, end + timedelta(days=1))
            chunks.append((cur.isoformat(), nxt.isoformat()))
            cur = nxt
        return chunks

    def _do_daily(self, site, raw_dir, fmt_dir):
        raw, _ = wd.get_daily(monitoring_location_id=site, parameter_code=self._pcodes(), time=self._time_arg())
        fmt = None
        if not raw.empty:
            fmt = self._with_site(site, pd.DataFrame({
                "Date (site local)": _date_text(raw["time"]),
                "Parameter Code": _col(raw, "parameter_code").map(lambda v: _code(v, 5)),
                "Parameter": _col(raw, "time_series_id").map(self._plan["series_param_name"]).fillna(""),
                "Statistic Code": _col(raw, "statistic_id").map(lambda v: _code(v, 5)),
                "Statistic": _col(raw, "time_series_id").map(self._plan["series_stat_name"]).fillna(""),
                "Value": _col(raw, "value"),
                "Unit": _col(raw, "unit_of_measure"),
                "Approval Status": _col(raw, "approval_status"),
                "Qualifier": _col(raw, "qualifier").map(_flatten_qualifier),
            }))
        self._write(site, "daily", raw, fmt, raw_dir, fmt_dir)

    def _do_field_measurements(self, site, raw_dir, fmt_dir):
        raw, _ = wd.get_field_measurements(monitoring_location_id=site, parameter_code=self._pcodes(),
                                           time=self._time_arg())
        fmt = None
        if not raw.empty:
            fmt = self._with_site(site, pd.DataFrame({
                "Time (UTC)": _utc_text(raw["time"]),
                "Field Visit ID": _col(raw, "field_visit_id"),
                "Reading Type": _col(raw, "reading_type"),
                "Parameter Code": _col(raw, "parameter_code").map(lambda v: _code(v, 5)),
                "Parameter": _col(raw, "parameter_code").map(lambda v: _code(v, 5))
                                 .map(self._plan["pcode_name"]).fillna(""),
                "Value": _col(raw, "value"),
                "Unit": _col(raw, "unit_of_measure"),
                "Method": _col(raw, "observing_procedure"),
                "Approval Status": _col(raw, "approval_status"),
                "Qualifier": _col(raw, "qualifier").map(_flatten_qualifier),
                "Measuring Agency": _col(raw, "measuring_agency"),
                "Control Condition": _col(raw, "control_condition"),
                "Measurement Rated": _col(raw, "measurement_rated"),
            }))
        self._write(site, "field_measurements", raw, fmt, raw_dir, fmt_dir)

    def _do_samples(self, site, raw_dir, fmt_dir):
        p = self._plan
        kwargs = {"monitoring_location_id": site}
        if not p["full_record"]:
            kwargs["activity_start_date_lower"] = p["start_date"]
            kwargs["activity_start_date_upper"] = p["end_date"]
        if p["samples_group"]:
            kwargs["characteristic_group"] = p["samples_group"]
        raw, _ = wd.get_samples(**kwargs)
        fmt = None
        if not raw.empty:
            fmt = self._with_site(site, pd.DataFrame({
                "Date (local)": _col(raw, "Activity_StartDate"),
                "Time (local)": _col(raw, "Activity_StartTime"),
                "Time Zone": _col(raw, "Activity_StartTimeZone"),
                "Time (UTC)": _utc_text(_col(raw, "Activity_StartDateTime")),
                "Characteristic": _col(raw, "Result_Characteristic"),
                "Characteristic Detail": _col(raw, "Result_CharacteristicUserSupplied"),
                "Characteristic Group": _col(raw, "Result_CharacteristicGroup"),
                "USGS Parameter Code": _col(raw, "USGSpcode").map(lambda v: _code(v, 5)),
                "Value": _col(raw, "Result_Measure"),
                "Unit": _col(raw, "Result_MeasureUnit"),
                "Qualifier": _col(raw, "Result_MeasureQualifierCode"),
                "Detection Limit": _col(raw, "DetectionLimit_MeasureA"),
                "Detection Limit Unit": _col(raw, "DetectionLimit_MeasureUnitA"),
                "Result Status": _col(raw, "Result_MeasureStatusIdentifier"),
                "Activity ID": _col(raw, "Activity_ActivityIdentifier"),
            }))
        self._write(site, "samples", raw, fmt, raw_dir, fmt_dir)

    def _do_statistics(self, site, raw_dir, fmt_dir):
        raw, _ = wd.get_stats_por(monitoring_location_id=site, parameter_code=self._pcodes())
        fmt = None
        if not raw.empty:
            fmt = self._with_site(site, pd.DataFrame({
                "Time of Year": _col(raw, "time_of_year"),
                "Time of Year Type": _col(raw, "time_of_year_type"),
                "Parameter Code": _col(raw, "parameter_code").map(lambda v: _code(v, 5)),
                "Parameter": _col(raw, "parameter_code").map(lambda v: _code(v, 5))
                                 .map(self._plan["pcode_name"]).fillna(""),
                "Statistic": _col(raw, "computation"),
                "Percentile": _col(raw, "percentile"),
                "Value": _col(raw, "value"),
                "Unit": _col(raw, "unit_of_measure"),
                "Sample Count": _col(raw, "sample_count"),
                "Approval Status": _col(raw, "approval_status"),
            }))
        self._write(site, "statistics", raw, fmt, raw_dir, fmt_dir)

    def _do_peaks(self, site, raw_dir, fmt_dir):
        # Reformatted peaks wait on a real sample of the peaks response; raw is written now.
        raw, _ = wd.get_peaks(monitoring_location_id=site, parameter_code=self._pcodes(), time=self._time_arg())
        self._write(site, "peaks", raw, None, raw_dir, fmt_dir)

    def _do_ratings(self, site, raw_dir, fmt_dir):
        rdb_dir = raw_dir if self._plan["write_raw"] else None
        result = wd.get_ratings(monitoring_location_id=site, file_path=str(rdb_dir) if rdb_dir else None)
        if not result:
            self.status.emit(f"  {site} ratings: no data returned")
            return
        for key, raw in result.items():
            fmt = self._with_site(site, pd.DataFrame({
                "Stage": _col(raw, "INDEP"),
                "Shift": _col(raw, "SHIFT"),
                "Discharge": _col(raw, "DEP"),
                "Stored Point": _col(raw, "STOR"),
            }))
            file_type = key.split(".")[1] if key.count(".") >= 2 else "rating"
            self._write(site, f"rating_{file_type}", raw, fmt, raw_dir, fmt_dir)

    def _write_reformatted_only(self, site, dtype, fmt_df, fmt_dir):
        """For tables that exist only in reformatted form (their raw content sits inside another raw file)."""
        stem = f"{site}_{dtype}"
        fmt_df.to_csv(fmt_dir / f"{stem}.csv", index=False)
        if self._plan["write_xlsx"] and len(fmt_df) + 1 <= EXCEL_MAX_ROWS:
            fmt_df.to_excel(fmt_dir / f"{stem}.xlsx", index=False)
        self.status.emit(f"  {site} {dtype}: {len(fmt_df):,} rows")

    def _fmt_thresholds(self, site, ts):
        """
        One row per threshold period from the series list's nested 'thresholds' field
        (reference lines and data-validation limits USGS attaches to each series).
        """
        if ts.empty or "thresholds" not in ts.columns:
            return None
        rows = []
        for _, r in ts.iterrows():
            value = r["thresholds"]
            if isinstance(value, str):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    continue
            if not isinstance(value, (list, tuple)):
                continue
            for t in value:
                if not isinstance(t, dict):
                    continue
                for period in t.get("Periods") or [{}]:
                    rows.append({
                        "Parameter Code": _code(r.get("parameter_code"), 5),
                        "Parameter": r.get("parameter_name", ""),
                        "Computation": r.get("computation_identifier", ""),
                        "Threshold": t.get("Name", ""),
                        "Type": t.get("Type", ""),
                        "Value": period.get("ReferenceValue"),
                        "Secondary Value": period.get("SecondaryReferenceValue"),
                        "Unit": r.get("unit_of_measure", ""),
                        "Start": _threshold_time(period.get("StartTime")),
                        "End": _threshold_time(period.get("EndTime")),
                        "Suppresses Data": period.get("SuppressData"),
                        "Description": t.get("Description", ""),
                        "Reference Code": t.get("ReferenceCode", ""),
                        "Time Series ID": r.get("time_series_id", ""),
                    })
        if not rows:
            return None
        return self._with_site(site, pd.DataFrame(rows))

    # ------------------------------------------------------------------------------------------------------------------
    def _fmt_site(self, site, loc):
        if loc.empty:
            return None
        body = loc.drop(columns=[c for c in ("geometry", "monitoring_location_id") if c in loc.columns])
        body = body.rename(columns=lambda c: c.replace("_", " ").title())
        return self._with_site(site, body)

    def _fmt_series(self, site, ts):
        if ts.empty:
            return None
        return self._with_site(site, pd.DataFrame({
            "Parameter Code": _col(ts, "parameter_code").map(lambda v: _code(v, 5)),
            "Parameter": _col(ts, "parameter_name"),
            "Description": _col(ts, "parameter_description"),
            "Statistic Code": _col(ts, "statistic_id").map(lambda v: _code(v, 5)),
            "Computation Period": _col(ts, "computation_period_identifier"),
            "Computation": _col(ts, "computation_identifier"),
            "Unit": _col(ts, "unit_of_measure"),
            "Begin (UTC)": _utc_text(_col(ts, "begin_utc")),
            "End (UTC)": _utc_text(_col(ts, "end_utc")),
            "Primary": _col(ts, "primary"),
            "Sensor": _col(ts, "web_description"),
            "Time Series ID": _col(ts, "time_series_id"),
        }))


# ======================================================================================================================
# Plugin widget
# ======================================================================================================================
class NWISDownloaderTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = _load_settings()
        for name in ("httpx", "httpcore"):
            logging.getLogger(name).setLevel(self._settings["http_log_level"])
        self._site_info = {}                      # id -> site record (with _lat/_lon)
        self._selected = []                       # ordered selected site IDs
        self._series = pd.DataFrame()             # time-series metadata for selected sites
        self._pins = {}                           # id -> site record currently on the map
        self._query_seq = 0
        self._workers = set()
        self._download_worker = None

        self._build_ui()

        if wd is None:
            import sys
            self._log.appendPlainText(
                f"The dataretrieval package is not available ({_DATARETRIEVAL_ERROR}).\n"
                f"This application is running on: {sys.executable}\n"
                f"Install it into that Python with:\n"
                f"  \"{sys.executable}\" -m pip install dataretrieval")
            self._download_btn.setEnabled(False)
            return

        self._start(SiteTypesWorker(), done=self._on_site_types, failed=lambda m: self._log.appendPlainText(
            f"Could not load USGS site types ({m}); using the saved code."))

    # ------------------------------------------------------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_sites_tab(), "Sites")
        self._tabs.addTab(self._build_data_tab(), "Data")
        layout.addWidget(self._tabs, 1)

        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier", 8))
        lines = int(self._settings["status_lines"])
        self._log.setFixedHeight(int(self._log.fontMetrics().lineSpacing() * lines
                                     + 3 * self._log.document().documentMargin()   # QPlainTextEdit needs a
                                     + 2 * self._log.frameWidth()))                # margin's worth extra to show
                                                                                   # the last N lines in full
        status_layout.addWidget(self._log)
        status_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(status_group)

    def _build_sites_tab(self):
        page = QWidget()
        outer = QHBoxLayout(page)
        splitter = QSplitter(Qt.Horizontal)
        self._splitter = splitter
        splitter.splitterMoved.connect(lambda *_: setattr(self, "_splitter_user_moved", True))
        outer.addWidget(splitter)

        MapWidget = _import_map_widget()
        self._map = MapWidget(bounds_debounce_ms=int(self._settings["bounds_debounce_ms"]))
        self._map.markerClicked.connect(self._toggle_site)
        self._map.boxSelected.connect(self._add_sites)
        self._map.boundsChanged.connect(self._on_bounds)
        lat, lon = self._settings["initial_center"]
        self._map.set_center(lat, lon, int(self._settings["initial_zoom"]))
        self._map.enable_box_select(True)
        self._map.enable_bounds_events(True)
        splitter.addWidget(self._map)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(6, 0, 0, 0)

        hint = QLabel("Hover a pin for site details. Click a pin to select or deselect it. "
                      "Shift+drag to select every pin in a box.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #444; font-size: 9pt; background: #f5f5f5; padding: 6px; border-radius: 3px;")
        side_layout.addWidget(hint)

        filt = QGroupBox("Map pins")
        grid = QGridLayout(filt)
        grid.addWidget(QLabel("Site type:"), 0, 0)
        self._site_type_combo = QComboBox()
        self._site_type_combo.addItem("All types", "")
        saved = self._settings["site_type_code"]
        if saved:
            self._site_type_combo.addItem(saved, saved)
            self._site_type_combo.setCurrentIndex(1)
        self._site_type_combo.currentIndexChanged.connect(self._on_filter_changed)
        grid.addWidget(self._site_type_combo, 0, 1)
        self._continuous_chk = QCheckBox("Only sites with continuous data")
        self._continuous_chk.setChecked(bool(self._settings["continuous_only"]))
        self._continuous_chk.toggled.connect(self._on_filter_changed)
        grid.addWidget(self._continuous_chk, 1, 0, 1, 2)
        self._pin_status = QLabel("")
        self._pin_status.setWordWrap(True)
        grid.addWidget(self._pin_status, 2, 0, 1, 2)
        side_layout.addWidget(filt)

        add_row = QHBoxLayout()
        self._id_edit = QLineEdit()
        self._id_edit.setPlaceholderText("Site IDs, e.g. 06770500, USGS-06768000")
        self._id_edit.returnPressed.connect(self._add_typed)
        add_btn = QPushButton("Add")
        c = self._settings["add_button_color"]
        add_btn.setStyleSheet(f"QPushButton {{ background-color: {c}; color: white; border: 1px solid {c}; border-radius: 3px; padding: 4px 12px; }} QPushButton:hover {{ border-color: white; }}")
        add_btn.clicked.connect(self._add_typed)
        add_row.addWidget(self._id_edit)
        add_row.addWidget(add_btn)
        side_layout.addLayout(add_row)

        sel_group = QGroupBox("Selected sites")
        sel_layout = QVBoxLayout(sel_group)
        self._sel_tree = QTreeWidget()
        self._sel_tree.setHeaderLabels(["Site / series", "Begin", "End"])
        self._sel_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        # Begin/End sized to a yyyy-mm-dd date (plus two characters of padding), leaving the rest to column 0
        date_width = self._sel_tree.fontMetrics().horizontalAdvance("0000-00-00" + "0" * 2)
        for col in (1, 2):
            self._sel_tree.header().setSectionResizeMode(col, QHeaderView.Fixed)
            self._sel_tree.header().resizeSection(col, date_width)
        self._sel_tree.header().setStretchLastSection(False)
        sel_layout.addWidget(self._sel_tree)
        btns = QHBoxLayout()
        rm_btn = QPushButton("Remove")
        c = self._settings["remove_button_color"]
        rm_btn.setStyleSheet(f"QPushButton {{ background-color: transparent; color: {c}; border: 1px solid {c}; border-radius: 3px; padding: 4px 12px; }} QPushButton:hover {{ background-color: {c}; color: white; }}")
        rm_btn.clicked.connect(self._remove_highlighted)
        clr_btn = QPushButton("Clear")
        c = self._settings["clear_button_border"]
        clr_btn.setStyleSheet(f"QPushButton {{ background-color: white; color: black; border: 2px solid {c}; border-radius: 3px; padding: 4px 12px; }} QPushButton:hover {{ background-color: {c}; }}")
        clr_btn.clicked.connect(self._clear_selection)
        btns.addWidget(rm_btn)
        btns.addWidget(clr_btn)
        sel_layout.addLayout(btns)
        side_layout.addWidget(sel_group, 1)

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        return page

    def _build_data_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        top = QHBoxLayout()

        types_group = QGroupBox("Data types")
        types_layout = QVBoxLayout(types_group)
        self._type_checks = {}
        for key, label in DATA_TYPES:
            chk = QCheckBox(label)
            chk.setChecked(key in self._settings["data_types"])
            chk.toggled.connect(self._update_estimate)
            types_layout.addWidget(chk)
            self._type_checks[key] = chk
        grp_row = QHBoxLayout()
        grp_row.addWidget(QLabel("Samples characteristic group:"))
        self._group_edit = QLineEdit(self._settings["samples_group"])
        self._group_edit.setPlaceholderText("blank = all")
        grp_row.addWidget(self._group_edit)
        types_layout.addLayout(grp_row)
        types_layout.addStretch()
        top.addWidget(types_group)

        param_group = QGroupBox("Parameters (none checked = all)")
        param_layout = QVBoxLayout(param_group)
        self._param_list = QListWidget()
        param_layout.addWidget(self._param_list)
        top.addWidget(param_group, 1)
        layout.addLayout(top)

        range_group = QGroupBox("Date range")
        range_layout = QHBoxLayout(range_group)
        self._full_chk = QCheckBox("Full period of record")
        self._full_chk.setChecked(bool(self._settings["full_record"]))
        self._full_chk.toggled.connect(self._on_full_toggled)
        self._start_edit, self._end_edit = QDateEdit(), QDateEdit()
        today = date.today()
        start_default = today - timedelta(days=int(self._settings["default_span_days"]))
        for edit, saved, fallback in ((self._start_edit, self._settings["start_date"], start_default),
                                      (self._end_edit, self._settings["end_date"], today)):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd")
            edit.setSpecialValueText(" ")   # shown blank while set to its minimum date (full period of record)
            d = date.fromisoformat(saved) if saved else fallback
            edit.setDate(QDate(d.year, d.month, d.day))
            edit.dateChanged.connect(self._update_estimate)
        range_layout.addWidget(self._full_chk)
        range_layout.addWidget(QLabel("Start:"))
        range_layout.addWidget(self._start_edit)
        range_layout.addWidget(QLabel("End:"))
        range_layout.addWidget(self._end_edit)
        note = QLabel("Statistics and ratings ignore the date range.")
        note.setStyleSheet("color: #666;")
        range_layout.addWidget(note)
        range_layout.addStretch()
        layout.addWidget(range_group)
        self._on_full_toggled(self._full_chk.isChecked())

        out_group = QGroupBox("Output")
        out_layout = QVBoxLayout(out_group)
        folder_row = QHBoxLayout()
        self._folder_lbl = QLabel(self._settings["output_folder"] or "No folder selected")
        self._folder_lbl.setWordWrap(True)
        browse = QPushButton("Browse\u2026")
        browse.setFixedWidth(90)
        browse.clicked.connect(self._browse)
        folder_row.addWidget(self._folder_lbl, 1)
        folder_row.addWidget(browse)
        out_layout.addLayout(folder_row)
        opts = QHBoxLayout()
        self._raw_chk = QCheckBox("Raw (as returned by dataretrieval)")
        self._raw_chk.setChecked(bool(self._settings["write_raw"]))
        self._fmt_chk = QCheckBox("Reformatted (readable labels)")
        self._fmt_chk.setChecked(bool(self._settings["write_reformatted"]))
        self._xlsx_chk = QCheckBox("Also write reformatted .xlsx")
        self._xlsx_chk.setChecked(bool(self._settings["write_xlsx"]))
        for chk in (self._raw_chk, self._fmt_chk, self._xlsx_chk):
            opts.addWidget(chk)
        opts.addStretch()
        out_layout.addLayout(opts)
        layout.addWidget(out_group)

        quota_row = QHBoxLayout()
        self._estimate_lbl = QLabel("")
        quota_btn = QPushButton("Check Quota")
        quota_btn.setToolTip("Uses one request to read your remaining USGS request quota.")
        quota_btn.clicked.connect(self._check_quota)
        quota_row.addWidget(self._estimate_lbl, 1)
        quota_row.addWidget(quota_btn)
        layout.addLayout(quota_row)

        run_row = QHBoxLayout()
        self._download_btn = QPushButton("Download")
        self._download_btn.setFixedHeight(32)
        self._download_btn.clicked.connect(self._download)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(32)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        run_row.addWidget(self._download_btn)
        run_row.addWidget(self._cancel_btn)
        layout.addLayout(run_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        layout.addStretch()
        return page

    def _apply_map_fraction(self):
        """Give the map its configured share of the Sites tab, until the user drags the splitter themselves."""
        if getattr(self, "_splitter_user_moved", False):
            return
        total = sum(self._splitter.sizes()) or self._splitter.width()
        map_width = int(total * float(self._settings["map_width_fraction"]))
        self._splitter.blockSignals(True)
        self._splitter.setSizes([map_width, total - map_width])
        self._splitter.blockSignals(False)

    def sizeHint(self):
        width, height = self._settings["window_size"]
        return QSize(int(width), int(height))

    def showEvent(self, event):
        super().showEvent(event)
        # Size the plugin's own window once. Only when the host gave this plugin a window of its own
        # (titled with the plugin title), never GRIME AI's main window.
        if not getattr(self, "_window_sized", False):
            self._window_sized = True
            win = self.window()
            if win is not self and win.windowTitle() == PLUGIN["title"]:
                width, height = self._settings["window_size"]
                win.resize(max(win.width(), int(width)), max(win.height(), int(height)))
        self._apply_map_fraction()

    def resizeEvent(self, event):
        # The host may show the plugin small and then enlarge its window; keep the proportion through that.
        super().resizeEvent(event)
        self._apply_map_fraction()

    # ------------------------------------------------------------------------------------------------------------------
    # Worker bookkeeping
    # ------------------------------------------------------------------------------------------------------------------
    def _start(self, worker, done, failed):
        self._workers.add(worker)
        worker.done.connect(done)
        if hasattr(worker, "failed"):
            worker.failed.connect(failed)
        worker.finished.connect(lambda w=worker: self._workers.discard(w))
        worker.start()

    # ------------------------------------------------------------------------------------------------------------------
    # Map pins
    # ------------------------------------------------------------------------------------------------------------------
    def _on_site_types(self, items):
        current = self._site_type_combo.currentData()
        self._site_type_combo.blockSignals(True)
        self._site_type_combo.clear()
        self._site_type_combo.addItem("All types", "")
        for code, label in items:
            self._site_type_combo.addItem(f"{code}  {label}" if label != code else code, code)
        idx = self._site_type_combo.findData(current)
        self._site_type_combo.setCurrentIndex(max(idx, 0))
        self._site_type_combo.blockSignals(False)

    def _current_site_type(self):
        return self._site_type_combo.currentData() or ""

    def _on_filter_changed(self, *_):
        self._settings["site_type_code"] = self._current_site_type()
        self._settings["continuous_only"] = self._continuous_chk.isChecked()
        _save_settings(self._settings)
        if hasattr(self, "_last_bounds"):
            self._on_bounds(*self._last_bounds)

    def _on_bounds(self, south, west, north, east):
        if wd is None:
            return
        self._last_bounds = (south, west, north, east)
        self._query_seq += 1
        self._pin_status.setText("Loading sites\u2026")
        worker = SiteQueryWorker(self._query_seq, [west, south, east, north], self._current_site_type(),
                                 self._continuous_chk.isChecked(), int(self._settings["max_pins"]))
        self._start(worker, done=self._on_sites_loaded, failed=self._on_sites_failed)

    def _on_sites_loaded(self, seq, sites, overflow):
        if seq != self._query_seq:
            return   # a newer pan/zoom superseded this query
        if overflow:
            self._map.clear_marker_layer(MARKER_LAYER)
            self._pins = {}
            self._pin_status.setText(f"More than {self._settings['max_pins']:,} sites in view. Zoom in to show pins.")
            return
        self._pins = {}
        markers = []
        for rec in sites:
            sid = rec.get("monitoring_location_id")
            if not sid or rec.get("_lat") is None:
                continue
            self._pins[sid] = rec
            self._site_info.setdefault(sid, rec)
            markers.append({"id": sid, "lat": rec["_lat"], "lng": rec["_lon"], "tooltip": _site_tooltip(rec),
                            **self._marker_colors(sid)})
        self._map.add_marker_layer(MARKER_LAYER, markers, self._dot_style())
        self._pin_status.setText(f"{len(markers):,} site(s) in view.")

    def _on_sites_failed(self, seq, msg):
        if seq == self._query_seq:
            self._pin_status.setText("Site query failed.")
            self._log.appendPlainText(f"Site query failed: {msg}")

    def _dot_style(self):
        s = self._settings
        if s["marker_style"] != "dot":
            return None
        return {"radius": s["dot_radius"], "weight": s["dot_border_width"], "fill_opacity": s["dot_fill_opacity"]}

    def _marker_colors(self, sid, selected=None):
        """{'color': ..., 'border': ...} for a site, as dots or pins per the marker_style setting."""
        s = self._settings
        selected = (sid in self._selected) if selected is None else selected
        if s["marker_style"] == "dot":
            key = "selected" if selected else "unselected"
            return {"color": s[f"dot_{key}_fill"], "border": s[f"dot_{key}_border"]}
        return {"color": s["selected_color"] if selected else s["unselected_color"], "border": None}

    def _recolor(self, sid, selected):
        c = self._marker_colors(sid, selected)
        self._map.set_marker_color(MARKER_LAYER, sid, c["color"], c["border"])

    # ------------------------------------------------------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------------------------------------------------------
    def _toggle_site(self, sid):
        if sid in self._selected:
            self._remove_sites([sid])
        else:
            self._add_sites([sid])

    def _add_typed(self):
        ids = [normalize_site_id(t) for t in re.split(r"[,\s;]+", self._id_edit.text())]
        self._id_edit.clear()
        self._add_sites([i for i in ids if i])

    def _add_sites(self, ids):
        new = [i for i in ids if i not in self._selected]
        if not new:
            return
        self._selected.extend(new)
        for sid in new:
            if sid in self._pins:
                self._recolor(sid, True)
        self._refresh_tree()
        unknown = [i for i in new if i not in self._site_info]
        self._log.appendPlainText(f"Fetching series available at {len(new)} site(s)\u2026")
        self._start(AvailabilityWorker(new, unknown), done=self._on_availability,
                    failed=lambda m: self._log.appendPlainText(f"Availability query failed: {m}"))

    def _remove_sites(self, ids):
        for sid in ids:
            if sid in self._selected:
                self._selected.remove(sid)
                if sid in self._pins:
                    self._recolor(sid, False)
        if not self._series.empty:
            self._series = self._series[self._series["monitoring_location_id"].isin(self._selected)]
        self._refresh_tree()
        self._refresh_params()

    def _remove_highlighted(self):
        ids = []
        for item in self._sel_tree.selectedItems():
            while item.parent() is not None:
                item = item.parent()
            ids.append(item.data(0, Qt.UserRole))
        self._remove_sites([i for i in ids if i])

    def _clear_selection(self):
        self._remove_sites(list(self._selected))

    def _on_availability(self, ts, sites):
        for rec in sites:
            sid = rec.get("monitoring_location_id")
            if sid:
                self._site_info[sid] = rec
        if ts is not None and not ts.empty:
            ts = ts[ts["monitoring_location_id"].isin(self._selected)]
            if not self._series.empty:
                ts = pd.concat([self._series, ts], ignore_index=True)
            self._series = ts.drop_duplicates(subset=["time_series_id"]).reset_index(drop=True)
        missing = [s for s in self._selected if s not in self._site_info]
        if missing:
            self._log.appendPlainText(f"No USGS site found for: {', '.join(missing)}")
            self._remove_sites(missing)
            return
        self._refresh_tree()
        self._refresh_params()

    def _refresh_tree(self):
        self._sel_tree.clear()
        for sid in self._selected:
            info = self._site_info.get(sid, {})
            top = QTreeWidgetItem([f"{sid}  {info.get('monitoring_location_name', '')}", "", ""])
            top.setData(0, Qt.UserRole, sid)
            self._sel_tree.addTopLevelItem(top)
            if self._series.empty:
                continue
            for _, r in self._series[self._series["monitoring_location_id"] == sid].iterrows():
                begin, end = _date_text(pd.Series([r.get("begin_utc")])).iloc[0], \
                             _date_text(pd.Series([r.get("end_utc")])).iloc[0]
                label = (f"{_code(r.get('parameter_code'), 5)} {r.get('parameter_name', '')}  "
                         f"({r.get('computation_period_identifier', '')} {r.get('computation_identifier', '')})")
                child = QTreeWidgetItem([label, begin if isinstance(begin, str) else "no data",
                                         end if isinstance(end, str) else ""])
                if not isinstance(begin, str):
                    child.setForeground(0, QColor("gray"))
                top.addChild(child)
            top.setExpanded(True)
        self._update_estimate()

    def _refresh_params(self):
        checked = {self._param_list.item(i).data(Qt.UserRole) for i in range(self._param_list.count())
                   if self._param_list.item(i).checkState() == Qt.Checked}
        self._param_list.clear()
        if self._series.empty:
            return
        ts = self._series[pd.to_datetime(self._series["begin_utc"], utc=True, errors="coerce").notna()].copy()
        ts["_pc"] = ts["parameter_code"].map(lambda v: _code(v, 5))
        n_sites = len(self._selected)
        for pc, grp in ts.groupby("_pc"):
            name = grp["parameter_name"].iloc[0]
            unit = grp["unit_of_measure"].iloc[0]
            item = QListWidgetItem(f"{pc}  {name} ({unit})  [{grp['monitoring_location_id'].nunique()} of "
                                   f"{n_sites} site(s)]")
            item.setData(Qt.UserRole, pc)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if pc in checked else Qt.Unchecked)
            self._param_list.addItem(item)

    # ------------------------------------------------------------------------------------------------------------------
    # Data tab
    # ------------------------------------------------------------------------------------------------------------------
    def _on_full_toggled(self, checked):
        """Full period of record blanks the start/end boxes; unchecking restores the dates they held."""
        edits = (self._start_edit, self._end_edit)
        if checked:
            if all(e.date() != e.minimumDate() for e in edits):
                self._saved_range = tuple(e.date() for e in edits)
            for e in edits:
                e.setDate(e.minimumDate())
        elif getattr(self, "_saved_range", None):
            for e, d in zip(edits, self._saved_range):
                e.setDate(d)
        for e in edits:
            e.setEnabled(not checked)
        self._update_estimate()

    def _range_text(self):
        """(start, end) as yyyy-MM-dd: the boxes' dates, or the dates they held before Full period of record."""
        if self._full_chk.isChecked() and getattr(self, "_saved_range", None):
            dates = self._saved_range
        else:
            dates = (self._start_edit.date(), self._end_edit.date())
        return tuple(d.toString("yyyy-MM-dd") for d in dates)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", self._settings["output_folder"])
        if folder:
            self._folder_lbl.setText(folder)
            self._settings["output_folder"] = folder
            _save_settings(self._settings)

    def _update_estimate(self, *_):
        if not hasattr(self, "_estimate_lbl"):
            return
        types = [k for k, c in self._type_checks.items() if c.isChecked()]
        n = len(self._selected)
        calls = 0
        for dtype in types:
            if dtype == "site_metadata":
                calls += 2 * n
            elif dtype == "continuous" and not self._full_chk.isChecked():
                days = self._start_edit.date().daysTo(self._end_edit.date()) + 1
                calls += n * max(1, -(-days // int(self._settings["continuous_chunk_days"])))
            else:
                calls += n
        self._estimate_lbl.setText(f"About {calls:,} API call(s) for {n} site(s). Large results use more "
                                   f"requests (one per page).")

    def _check_quota(self):
        self._log.appendPlainText("Checking USGS request quota\u2026")
        self._start(QuotaWorker(), done=lambda ok, lines: self._log.appendPlainText("\n".join(lines)), failed=None)

    def _download(self):
        if not self._selected:
            QMessageBox.warning(self, "No sites", "Select at least one site on the Sites tab.")
            return
        folder = self._settings["output_folder"]
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Please select an output folder.")
            return
        types = [k for k, c in self._type_checks.items() if c.isChecked()]
        if not types:
            QMessageBox.warning(self, "No data types", "Check at least one data type.")
            return
        if not self._raw_chk.isChecked() and not self._fmt_chk.isChecked():
            QMessageBox.warning(self, "No output", "Check raw, reformatted, or both.")
            return
        pending = [s for s in self._selected if s not in self._site_info]
        if pending:
            QMessageBox.warning(self, "Still loading", "Site details are still loading. Try again in a moment.")
            return

        pcodes = [self._param_list.item(i).data(Qt.UserRole) for i in range(self._param_list.count())
                  if self._param_list.item(i).checkState() == Qt.Checked]
        start, end = self._range_text()

        s = self._settings
        s.update({"data_types": types, "full_record": self._full_chk.isChecked(), "start_date": start,
                  "end_date": end, "write_raw": self._raw_chk.isChecked(),
                  "write_reformatted": self._fmt_chk.isChecked(), "write_xlsx": self._xlsx_chk.isChecked(),
                  "samples_group": self._group_edit.text().strip()})
        _save_settings(s)

        ts = self._series.copy()
        series_param_name, series_stat_name, pcode_name = {}, {}, {}
        if not ts.empty:
            series_param_name = dict(zip(ts["time_series_id"], ts["parameter_name"]))
            series_stat_name = dict(zip(ts["time_series_id"], ts["computation_identifier"]))
            pcode_name = dict(zip(ts["parameter_code"].map(lambda v: _code(v, 5)), ts["parameter_name"]))

        plan = {
            "sites": list(self._selected), "data_types": types, "parameter_codes": pcodes,
            "full_record": s["full_record"], "start_date": start, "end_date": end,
            "samples_group": s["samples_group"], "output_folder": folder,
            "write_raw": s["write_raw"], "write_reformatted": s["write_reformatted"], "write_xlsx": s["write_xlsx"],
            "continuous_chunk_days": s["continuous_chunk_days"],
            "site_info": {k: self._site_info[k] for k in self._selected},
            "series": ts, "series_param_name": series_param_name, "series_stat_name": series_stat_name,
            "pcode_name": pcode_name,
        }

        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._download_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._log.appendPlainText(f"Downloading {len(types)} data type(s) for {len(self._selected)} site(s)\u2026")

        self._download_worker = DownloadWorker(plan)
        self._download_worker.status.connect(self._log.appendPlainText)
        self._download_worker.progress.connect(
            lambda d, t: self._progress.setValue(int(d * 100 / t) if t else 0))
        self._download_worker.finished_all.connect(self._on_download_finished)
        self._download_worker.start()

    def _cancel(self):
        if self._download_worker is not None and self._download_worker.isRunning():
            self._cancel_btn.setEnabled(False)
            self._log.appendPlainText("Cancelling after the current request\u2026")
            self._download_worker.cancel()

    def _on_download_finished(self, out_root):
        self._progress.setVisible(False)
        self._download_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._log.appendPlainText(f"Done. Output: {out_root}")
