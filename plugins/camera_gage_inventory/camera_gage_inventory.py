#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# camera_sensor_inventory.py
#
# Plugin: USGS Camera and Sensor Inventory.
#
# Probes USGS HIVIS cameras (NIMS API) and USGS monitoring sites (Water Data OGC API):
#   - how many cameras exist (visible and hidden)
#   - every measurement recorded at camera sites
#   - for the measurements checked in the Co-located Sensor column, how many sites record
#     them nationally and how many cameras are co-located with one, either at their own
#     site (the camera's nwisId) or within a set distance of one
#
# A co-located sensor is a continuous ("Points") record of a checked measurement (default
# discharge 00060 and gage height 00065). It is active when it has data within the active
# window in settings.
#
# When the plugin opens, the cameras and every measurement at camera sites are loaded in
# the background (a few requests), so the measurements table is filled before any run.
# Run Inventory then adds the national query for the checked measurements and the
# co-location results.
#
# Output goes to <output folder>/CameraSensorInventory_YYYYMMDD_HHMMSS/:
#   Summary.txt, Cameras.csv, Camera_Colocated_Sensors.csv (one row per camera and
#   co-located sensor), Parameters.csv, Camera_Parameter_Matrix.csv,
#   raw/ (API responses as returned)

import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import importlib
import numpy as np
import pandas as pd
import requests

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize, QTimer, QRect
from PyQt5.QtGui import QFont, QPainter, QColor, QPen
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QPlainTextEdit,
                             QFileDialog, QMessageBox, QTabWidget, QCheckBox, QLineEdit, QFormLayout, QSpinBox,
                             QDoubleSpinBox, QGridLayout, QSplitter, QTableWidget, QTableWidgetItem,
                             QHeaderView, QSizePolicy, QAbstractItemView, QTreeWidget, QTreeWidgetItem)

from appcore.app_identity import PLUGINS_DIR
from appcore.dialogs.api_keys.api_key_manager import APIKeyManager

PLUGIN = {
    "title":       "USGS Camera and Sensor Inventory",
    "class":       "CameraSensorInventoryTab",
    "description": "Count HIVIS cameras, the sensors at camera sites, and cameras co-located with chosen sensors",
    "surface":     "tools",
    "api_version": 2,
}

USER_ROOT = Path(str(PLUGINS_DIR)).parent
SETTINGS_FILE = USER_ROOT / "Settings" / "camera_sensor_inventory.json"
OLD_SETTINGS_FILE = USER_ROOT / "Settings" / "camera_gage_inventory.json"   # earlier name, read once if present

NIMS_DEFAULT_ENDPOINT = "https://api.waterdata.usgs.gov/nims/v0"
OGC_SERIES_ITEMS = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/time-series-metadata/items"
EARTH_RADIUS_KM = 6371.0088   # IUGG mean Earth radius

# Tuning values. Every one can be overridden in the settings file.
DEFAULT_SETTINGS = {
    "output_folder":        "",
    "colocated_sensor_codes": "00060,00065", # measurements checked in the Co-located Sensor column
    "active_days":          30,              # a sensor is active if it has data within this many days
    "nearby_km":            1.0,             # an active sensor site within this distance counts as co-located
    "include_hidden":       False,           # show hidden cameras in the tables and map
    "show_sensor_sites_on_map": False,
    "site_chunk":           200,             # site IDs per camera-site series request (URL length limit)
    "page_rows":            10000,           # rows per page for paged requests
    "request_timeout_s":    120,
    "recalc_debounce_ms":   300,
    "hidden_columns": {                      # columns turned off by right-clicking a table header
        "Cameras": ["Camera Name", "Hidden"],
        "Measurements": [],
    },
    "header_color":         "#cfe2f3",       # table header background (light blue); text is bold black
    "header_border_color":  "#4682b4",
    "header_check_size":    14,              # select-all checkbox in the Co-located Sensor header (pixels)
    "header_check_colors": {                 # fill, border, mark
        "all":  ["#2a6fdb", "#2a6fdb", "#ffffff"],
        "some": ["#9e9e9e", "#7a7a7a", "#ffffff"],
        "none": ["#ffffff", "#5a5a5a", "#ffffff"],
    },             # pause after an active-window or distance edit before recalculating
    "window_size":          [1330, 700],
    "left_panel_width":     330,
    "status_lines":         2,
    "map_debounce_ms":      500,
    "initial_center":       [39.5, -98.5],
    "initial_zoom":         4,
    "dot_radius":           5,
    "dot_border_width":     1,
    "dot_fill_opacity":     0.9,
    "sensor_site_dot_radius": 2,
    "colors": {
        "Active sensor at site":       ["#1a9850", "#0b4d27"],
        "Active sensor nearby":        ["#3b8fd6", "#15426b"],
        "Discontinued sensor at site": ["#f39c12", "#7a4d05"],
        "No co-located sensor":        ["#9e9e9e", "#424242"],
        "sensor_site":                 ["#000000", "#000000"],
    },
}

# Measurements table columns, with what each means (shown as header tooltips)
COL_COLOCATED = "Co-located Sensor"
COL_CODE = "Parameter Code"
COL_NAME = "Measurement (USGS parameter)"
COL_UNIT = "Unit"
COL_CAMS_NOW = "Cameras Currently Reporting"
COL_CAMS_EVER = "Cameras Ever Recorded"
COL_SITES_EVER = "Camera Sites Ever Recorded"
PARAM_COLUMN_TIPS = {
    COL_COLOCATED: "Checked measurements are the sensors the co-location check looks for",
    COL_CODE: "USGS parameter code",
    COL_NAME: "What is measured (USGS parameter name)",
    COL_CAMS_NOW: "Cameras whose site has this measurement with data within the 'Active if data within' days "
                  "(USGS: active time series)",
    COL_CAMS_EVER: "Cameras whose site has ever recorded this measurement, including discontinued records "
                   "(USGS: any time series)",
    COL_SITES_EVER: "Distinct camera sites that have ever recorded this measurement "
                    "(several cameras can share one site)",
}

# Camera co-location status, in order of precedence
STATUSES = ["Active sensor at site", "Active sensor nearby", "Discontinued sensor at site", "No co-located sensor"]

CAMERA_MARKER_LAYER = "hivis_cameras"
SENSOR_SITE_MARKER_LAYER = "sensor_sites"


# ======================================================================================================================
# Settings
# ======================================================================================================================
def _load_settings() -> dict:
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    try:
        path = SETTINGS_FILE if SETTINGS_FILE.exists() else OLD_SETTINGS_FILE
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        # Settings saved by earlier versions used "stream gage" names
        for old, new in (("stream_gage_codes", "colocated_sensor_codes"),
                         ("show_gages_on_map", "show_sensor_sites_on_map"),
                         ("gage_dot_radius", "sensor_site_dot_radius")):
            if old in saved:
                saved.setdefault(new, saved.pop(old))
        settings["hidden_columns"].update(saved.pop("hidden_columns", {}))
        saved_colors = {k: v for k, v in saved.pop("colors", {}).items() if k in settings["colors"]}
        colors = {**settings["colors"], **saved_colors}
        settings.update(saved)
        settings["colors"] = colors
    except Exception:
        pass
    return settings


def _save_settings(settings: dict):
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"[camera_sensor_inventory] Could not save settings: {e}")


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
def _parse_utc(text):
    if not text:
        return None
    try:
        t = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _utc_text(t):
    return t.strftime("%Y-%m-%d %H:%M:%S") if t else ""


def _float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _point(geom):
    """(lat, lon) from a GeoJSON point, else (None, None)."""
    if isinstance(geom, dict) and geom.get("type") == "Point":
        c = geom.get("coordinates") or []
        if len(c) >= 2:
            return _float(c[1]), _float(c[0])
    return None, None


def _site_id(nwis_id):
    return f"USGS-{nwis_id}" if nwis_id else ""


def _is_hidden(cam):
    """GRIME AI treats a camera as hidden when hideCam is True or missing."""
    return cam.get("hideCam", True) is not False


# ======================================================================================================================
# Worker
# ======================================================================================================================
class InventoryWorker(QThread):
    status = pyqtSignal(str)
    done = pyqtSignal(object)    # result dict
    failed = pyqtSignal(str)

    def __init__(self, settings: dict, preloaded=None):
        super().__init__()
        self._s = dict(settings)
        self._preloaded = preloaded   # {"cams", "site_series"} from SiteMeasurementsWorker, or None
        mgr = APIKeyManager()
        self._key = mgr.get_usgs_key() or ""
        endpoint = (mgr.get_usgs_endpoint() or "").rstrip("/")
        self._nims = endpoint if "api.waterdata.usgs.gov" in endpoint else NIMS_DEFAULT_ENDPOINT
        self._http = requests.Session()
        if self._key:
            self._http.headers["X-Api-Key"] = self._key
        self.calls = 0
        self.rate_remaining = ""

    def _get(self, url, params=None):
        r = self._http.get(url, params=params, timeout=float(self._s["request_timeout_s"]))
        self.calls += 1
        self.rate_remaining = r.headers.get("X-RateLimit-Remaining", self.rate_remaining)
        r.raise_for_status()
        return r.json()

    def _features(self, params):
        """All features for a time-series-metadata query, following rel=next pages."""
        feats, data = [], self._get(OGC_SERIES_ITEMS, params)
        while True:
            page = data.get("features", [])
            feats += page
            nxt = next((l["href"] for l in data.get("links", []) if l.get("rel") == "next"), None)
            if not nxt or not page:
                return feats
            data = self._get(nxt)

    def _camera_site_series(self):
        """NIMS cameras, and every time series at their NWIS sites (chunked by site_chunk)."""
        page = int(self._s["page_rows"])
        self.status.emit("Fetching HIVIS cameras\u2026")
        cams = self._get(f"{self._nims}/cameras")
        site_ids = sorted({_site_id(c["nwisId"]) for c in cams if c.get("nwisId")})
        chunk = int(self._s["site_chunk"])
        site_series = []
        for i in range(0, len(site_ids), chunk):
            self.status.emit(f"Fetching series at camera sites ({i + 1}-{min(i + chunk, len(site_ids))} "
                             f"of {len(site_ids)})\u2026")
            site_series += self._features({"monitoring_location_id": ",".join(site_ids[i:i + chunk]),
                                           "limit": page, "f": "json"})
        return cams, site_series

    def run(self):
        try:
            t0 = time.perf_counter()
            if not self._key:
                self.status.emit("No USGS API key stored; using the lower anonymous rate limit.")
            codes = [c.strip() for c in str(self._s["colocated_sensor_codes"]).split(",") if c.strip()]
            page = int(self._s["page_rows"])

            if self._preloaded:
                cams, site_series = self._preloaded["cams"], self._preloaded["site_series"]
            else:
                cams, site_series = self._camera_site_series()

            national = {}
            for code in codes:
                self.status.emit(f"Fetching sites nationwide that record {code}\u2026")
                national[code] = self._features({"parameter_code": code, "computation_period_identifier": "Points",
                                                 "properties": "monitoring_location_id,parameter_code,end_utc",
                                                 "limit": page, "f": "json"})

            self.status.emit("Building inventory\u2026")
            result = _build_inventory(cams, site_series, national, codes, self._s)
            result["inputs"] = {"cams": cams, "site_series": site_series, "national": national, "codes": codes}
            result["raw"] = {"NIMS_Cameras.json": cams, "CameraSite_Series.json": site_series,
                             **{f"National_Series_{c}.json": v for c, v in national.items()}}
            result["requests"] = self.calls
            result["rate_remaining"] = self.rate_remaining
            result["elapsed_s"] = time.perf_counter() - t0
            self.done.emit(result)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# ======================================================================================================================
# Worker: cameras and measurements at camera sites (runs when the plugin opens)
# ======================================================================================================================
class SiteMeasurementsWorker(InventoryWorker):
    """Fetches the NIMS cameras and every time series at their sites. Emits done(dict) with
    "cams", "site_series" and "requests"; no national query (that needs the co-located sensor choice)."""

    def run(self):
        try:
            self.status.emit("Loading cameras and measurements at camera sites\u2026")
            cams, site_series = self._camera_site_series()
            self.done.emit({"cams": cams, "site_series": site_series, "requests": self.calls})
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# ======================================================================================================================
# Inventory
# ======================================================================================================================
def _site_measurements(site_series):
    """site -> {code: {"last", "points"}}, site -> (lat, lon), code -> (name, unit)."""
    site_params, site_loc, param_names = {}, {}, {}
    for f in site_series:
        p = f.get("properties", {})
        sid, code = p.get("monitoring_location_id"), p.get("parameter_code")
        if not sid or not code:
            continue
        code = str(code).zfill(5) if str(code).isdigit() else str(code)
        param_names.setdefault(code, (p.get("parameter_name") or "", p.get("unit_of_measure") or ""))
        entry = site_params.setdefault(sid, {}).setdefault(code, {"last": None, "points": False})
        end = _parse_utc(p.get("end_utc"))
        if end and (entry["last"] is None or end > entry["last"]):
            entry["last"] = end
        if p.get("computation_period_identifier") == "Points":
            entry["points"] = True
        if sid not in site_loc:
            lat, lon = _point(f.get("geometry"))
            if lat is not None:
                site_loc[sid] = (lat, lon)
    return site_params, site_loc, param_names


def _preload_rows(cams, site_series):
    """Per-camera rows with only what the measurements table needs (before a run)."""
    site_params, _loc, param_names = _site_measurements(site_series)
    rows = [{"Site ID": _site_id(c.get("nwisId")), "Hidden": "Yes" if _is_hidden(c) else "No",
             "_params": site_params.get(_site_id(c.get("nwisId")), {})} for c in cams]
    return rows, param_names


def _build_inventory(cams, site_series, national, codes, s):
    run_time = datetime.now(timezone.utc)
    cutoff = run_time - timedelta(days=float(s["active_days"]))
    nearby_km = float(s["nearby_km"])

    # Sites nationwide with a checked measurement: site -> last value time and location
    sensor_sites = {}
    for feats in national.values():
        for f in feats:
            p = f.get("properties", {})
            sid = p.get("monitoring_location_id")
            if not sid:
                continue
            g = sensor_sites.setdefault(sid, {"last": None, "lat": None, "lon": None})
            end = _parse_utc(p.get("end_utc"))
            if end and (g["last"] is None or end > g["last"]):
                g["last"] = end
            if g["lat"] is None:
                g["lat"], g["lon"] = _point(f.get("geometry"))
    active_sites = {sid: g for sid, g in sensor_sites.items() if g["last"] and g["last"] >= cutoff}
    act_ids = [sid for sid, g in active_sites.items() if g["lat"] is not None]
    act_lat = np.radians([active_sites[sid]["lat"] for sid in act_ids]) if act_ids else np.array([])
    act_lon = np.radians([active_sites[sid]["lon"] for sid in act_ids]) if act_ids else np.array([])

    # Per checked sensor: its active sites nationwide, for the nearest-site search
    per_code = {}
    for code, feats in national.items():
        last_by_site, loc_by_site = {}, {}
        for f in feats:
            p = f.get("properties", {})
            sid = p.get("monitoring_location_id")
            end = _parse_utc(p.get("end_utc"))
            if not sid or not end:
                continue
            if sid not in last_by_site or end > last_by_site[sid]:
                last_by_site[sid] = end
            if sid not in loc_by_site:
                lat, lon = _point(f.get("geometry"))
                if lat is not None:
                    loc_by_site[sid] = (lat, lon)
        ids = [sid for sid, end in last_by_site.items() if end >= cutoff and sid in loc_by_site]
        per_code[code] = (ids, np.radians([loc_by_site[i][0] for i in ids]) if ids else np.array([]),
                          np.radians([loc_by_site[i][1] for i in ids]) if ids else np.array([]))

    def nearest(lat, lon, ids, lats, lons):
        if lat is None or lon is None or not len(ids):
            return "", None
        la, lo = math.radians(lat), math.radians(lon)
        h = np.sin((lats - la) / 2) ** 2 + math.cos(la) * np.cos(lats) * np.sin((lons - lo) / 2) ** 2
        d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))
        i = int(np.argmin(d))
        return ids[i], float(d[i])

    # Series at camera sites: site -> {code: {last, points}}
    site_params, site_loc, param_names = _site_measurements(site_series)

    cams_per_site = {}
    for c in cams:
        if c.get("nwisId"):
            sid = _site_id(c["nwisId"])
            cams_per_site[sid] = cams_per_site.get(sid, 0) + 1

    rows = []
    for c in cams:
        sid = _site_id(c.get("nwisId"))
        params = site_params.get(sid, {})
        sensor_series = {k: v for k, v in params.items() if k in codes and v["points"]}
        sensor_last = max((v["last"] for v in sensor_series.values() if v["last"]), default=None)
        lat, lon = _float(c.get("lat")), _float(c.get("lng"))

        nearest_id, nearest_km = nearest(lat, lon, act_ids, act_lat, act_lon)

        # Each checked sensor that is co-located with this camera (shown beneath the camera)
        sensors = []
        for code in codes:
            entry = params.get(code)
            at_site = bool(entry and entry["points"])
            last = entry["last"] if at_site else None
            n_id, n_km = nearest(lat, lon, *per_code.get(code, ([], np.array([]), np.array([]))))
            if last and last >= cutoff:
                st = STATUSES[0]
                # Active at the camera's own site: report that site and its distance from the camera
                own = site_loc.get(sid)
                n_id = sid
                n_km = nearest(lat, lon, [sid], np.radians([own[0]]), np.radians([own[1]]))[1] if own else None
            elif n_km is not None and n_km <= nearby_km:
                st = STATUSES[1]
            elif at_site:
                st = STATUSES[2]
            else:
                continue   # not co-located: no row
            sensors.append({"Sensor": f"{param_names.get(code, ('', ''))[0] or code} ({code})", "Code": code,
                            "Status": st, "Last Value at Camera Site (UTC)": _utc_text(last),
                            "Nearest Active Site": n_id,
                            "Distance (km)": round(n_km, 3) if n_km is not None else None})

        if sensor_last and sensor_last >= cutoff:
            status = STATUSES[0]
        elif nearest_km is not None and nearest_km <= nearby_km:
            status = STATUSES[1]
        elif sensor_series:
            status = STATUSES[2]
        else:
            status = STATUSES[3]

        rows.append({
            "Camera ID": c.get("camId", ""),
            "Camera Name": c.get("camName", ""),
            "Hidden": "Yes" if _is_hidden(c) else "No",
            "Site ID": sid,
            "Co-location Status": status,
            "Co-located Sensors at Site": ", ".join(sorted(sensor_series)),
            "Last Co-located Sensor Value (UTC)": _utc_text(sensor_last),
            "Nearest Active Sensor Site": nearest_id,
            "Distance to Nearest Active Sensor Site (km)": round(nearest_km, 3) if nearest_km is not None else None,
            "Cameras at Site": cams_per_site.get(sid, 0) if sid else 0,
            "Measurements at Site": len(params),
            "Camera Default Parameter": c.get("defaultPCode", "") or "",
            "Newest Image (UTC)": _utc_text(_parse_utc(c.get("newestImageDT"))),
            "Camera Latitude": lat,
            "Camera Longitude": lon,
            "_params": params,
            "_sensors": sensors,
            "_measurement_names": sorted(f"{param_names.get(k, ('', ''))[0] or k} ({k})" for k in params),
        })

    # Summary counts
    visible = [r for r in rows if r["Hidden"] == "No"]
    status_counts = {st: (sum(r["Co-location Status"] == st for r in visible),
                          sum(r["Co-location Status"] == st for r in rows)) for st in STATUSES}
    summary = {
        "run_time": run_time,
        "cameras_total": len(rows),
        "cameras_visible": len(visible),
        "cameras_hidden": len(rows) - len(visible),
        "camera_sites": len(cams_per_site),
        "sensor_sites_total": len(sensor_sites),
        "sensor_sites_active": len(active_sites),
        "status_counts": status_counts,
        "codes": codes,
        "active_days": float(s["active_days"]),
        "nearby_km": nearby_km,
    }

    sensor_site_points = [(sid, g["lat"], g["lon"], g["last"]) for sid, g in active_sites.items()
                          if g["lat"] is not None]
    return {"summary": summary, "rows": rows, "param_names": param_names, "cutoff": cutoff,
            "sensor_site_points": sensor_site_points}


def _parameter_table(rows, param_names, cutoff):
    counts = {}
    for r in rows:
        for code, v in r["_params"].items():
            c = counts.setdefault(code, {"active": 0, "any": 0, "sites": set()})
            c["any"] += 1
            if v["last"] and v["last"] >= cutoff:
                c["active"] += 1
            c["sites"].add(r["Site ID"])
    table = [{COL_CODE: code,
              COL_NAME: param_names.get(code, ("", ""))[0],
              COL_UNIT: param_names.get(code, ("", ""))[1],
              COL_CAMS_NOW: c["active"],
              COL_CAMS_EVER: c["any"],
              COL_SITES_EVER: len(c["sites"])} for code, c in counts.items()]
    return sorted(table, key=lambda t: (-t[COL_CAMS_EVER], t[COL_CODE]))


def _matrix(rows, cutoff):
    codes = sorted({code for r in rows for code in r["_params"]})
    out = []
    for r in rows:
        line = {"Camera ID": r["Camera ID"], "Site ID": r["Site ID"], "Hidden": r["Hidden"]}
        for code in codes:
            v = r["_params"].get(code)
            line[code] = "" if v is None else ("Active" if v["last"] and v["last"] >= cutoff else "Discontinued")
        out.append(line)
    return pd.DataFrame(out)


def _summary_lines(summary, requests_made, elapsed_s, rate_remaining):
    s = summary
    lines = [
        f"USGS Camera and Sensor Inventory, run {_utc_text(s['run_time'])} UTC",
        f"Co-located sensors: continuous records of {', '.join(s['codes'])}; active = data within {s['active_days']:g} days; "
        f"nearby = within {s['nearby_km']:g} km",
        "",
        f"HIVIS cameras: {s['cameras_total']} ({s['cameras_visible']} visible, {s['cameras_hidden']} hidden)",
        f"Camera sites (distinct nwisId): {s['camera_sites']}",
        f"USGS sites recording a co-located sensor measurement nationally: {s['sensor_sites_total']} "
        f"({s['sensor_sites_active']} active)",
        "",
        f"{'Co-location status':30}{'Visible':>10}{'All':>10}",
    ]
    for st, (vis, allc) in s["status_counts"].items():
        lines.append(f"{st:30}{vis:>10}{allc:>10}")
    lines += ["", f"Requests: {requests_made}   Rate limit remaining: {rate_remaining}   Time: {elapsed_s:.1f} s"]
    return lines


# ======================================================================================================================
# Plugin widget
# ======================================================================================================================
class _CheckAllHeader(QHeaderView):
    """Horizontal header with a select-all checkbox drawn at the left of section 0.
    State: "all" (filled, checked), "some" (grey, dash), "none" (empty), or "hidden" (no
    box, while the table shows a message).
    Clicking the box emits toggled(); clicking the rest of the header sorts as usual."""
    toggled = pyqtSignal()
    MARGIN = 4

    def __init__(self, settings, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._settings = settings
        self._state = "none"
        self.setSectionsClickable(True)
        self.setHighlightSections(False)

    def set_state(self, state):
        if state != self._state:
            self._state = state
            self.viewport().update()

    def box_rect(self, section_rect):
        size = int(self._settings["header_check_size"])
        return QRect(section_rect.left() + self.MARGIN, section_rect.center().y() - size // 2, size, size)

    def text_indent(self):
        return int(self._settings["header_check_size"]) + 2 * self.MARGIN

    def paintSection(self, painter, rect, logical_index):
        painter.save()
        super().paintSection(painter, rect, logical_index)
        painter.restore()
        if logical_index != 0 or self._state == "hidden":
            return
        fill, border, mark = self._settings["header_check_colors"][self._state]
        box = self.box_rect(rect)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor(border), 1))
        painter.setBrush(QColor(fill))
        painter.drawRoundedRect(box, 3, 3)
        pen = QPen(QColor(mark), 2)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        w, h, x, y = box.width(), box.height(), box.left(), box.top()
        if self._state == "all":
            painter.drawLine(int(x + w * 0.22), int(y + h * 0.52), int(x + w * 0.42), int(y + h * 0.72))
            painter.drawLine(int(x + w * 0.42), int(y + h * 0.72), int(x + w * 0.78), int(y + h * 0.30))
        elif self._state == "some":
            painter.drawLine(int(x + w * 0.25), int(y + h * 0.5), int(x + w * 0.75), int(y + h * 0.5))
        painter.restore()

    def mousePressEvent(self, event):
        idx = self.logicalIndexAt(event.pos())
        if idx == 0:
            left = self.sectionViewportPosition(0)
            rect = QRect(left, 0, self.sectionSize(0), self.height())
            if self._state != "hidden" and self.box_rect(rect).adjusted(-2, -2, 2, 2).contains(event.pos()):
                self.toggled.emit()
                event.accept()
                return
        super().mousePressEvent(event)


class _SortableTreeItem(QTreeWidgetItem):
    """Sorts numeric columns by value (stored in UserRole) instead of text."""
    def __lt__(self, other):
        col = self.treeWidget().sortColumn() if self.treeWidget() else 0
        a, b = self.data(col, Qt.UserRole), other.data(col, Qt.UserRole)
        if a is not None and b is not None:
            return a < b
        if (a is None) != (b is None):
            return a is None   # blanks first
        return self.text(col).lower() < other.text(col).lower()


class CameraSensorInventoryTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = _load_settings()
        self._result = None
        self._worker = None
        self._preloaded = None      # cameras + camera-site series loaded when the plugin opens
        self._pre_rows = []
        self._param_names = {}
        self._build_ui()
        self._update_sensor_label()
        self._show_param_message("Loading measurements at camera sites\u2026")
        self._start_preload()

    # ------------------------------------------------------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        self._tabs = QTabWidget()
        self._camera_table = QTreeWidget()
        self._camera_table.setAlternatingRowColors(True)
        self._camera_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._camera_table.setUniformRowHeights(True)
        self._param_table = self._new_table()
        self._param_header = _CheckAllHeader(self._settings, self._param_table)
        self._param_table.setHorizontalHeader(self._param_header)
        self._param_header.toggled.connect(self._toggle_all_sensors)
        for view, name, fixed in ((self._camera_table, "Cameras", "Camera ID"),
                                  (self._param_table, "Measurements", COL_COLOCATED)):
            hdr = view.header() if isinstance(view, QTreeWidget) else view.horizontalHeader()
            self._style_header(hdr)
            hdr.setContextMenuPolicy(Qt.CustomContextMenu)
            hdr.customContextMenuRequested.connect(
                lambda pos, v=view, n=name, f=fixed: self._column_menu(v, n, f, pos))
        # Cameras tab: a centered message until the first run fills the table
        from PyQt5.QtWidgets import QStackedWidget
        self._camera_stack = QStackedWidget()
        self._camera_empty = QLabel('Click "Run Inventory" to list the cameras and their co-located sensors.')
        self._camera_empty.setAlignment(Qt.AlignCenter)
        self._camera_empty.setWordWrap(True)
        f = self._camera_empty.font()
        f.setBold(True)
        self._camera_empty.setFont(f)
        self._camera_stack.addWidget(self._camera_empty)
        self._camera_stack.addWidget(self._camera_table)
        self._tabs.addTab(self._camera_stack, "Cameras")
        self._tabs.addTab(self._param_table, "Measurements at Camera Sites")
        self._param_table.itemChanged.connect(self._on_param_item_changed)
        self._tabs.addTab(self._build_map(), "Map")
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([int(self._settings["left_panel_width"]), 1000])
        layout.addWidget(splitter, 1)

        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier", 8))
        lines = int(self._settings["status_lines"])
        self._log.setFixedHeight(int(self._log.fontMetrics().lineSpacing() * lines
                                     + 3 * self._log.document().documentMargin()
                                     + 2 * self._log.frameWidth()))
        status_layout.addWidget(self._log)
        status_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(status_group)

    def _build_left_panel(self):
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 6, 0)
        s = self._settings

        opts = QGroupBox("Settings")
        form = QFormLayout(opts)
        form.setLabelAlignment(Qt.AlignLeft)
        self._sensor_label = QLabel("")
        self._sensor_label.setWordWrap(True)
        self._sensor_label.setToolTip("A camera is co-located with a sensor if its site, or a site nearby, records one of "
                                      "these measurements continuously.\n"
                                    "Choose them in the Co-located Sensor column of the Measurements at Camera Sites tab.")
        form.addRow(QLabel("Co-located sensors:"))
        form.addRow(self._sensor_label)      # own row so a long list wraps under the heading
        self._active_spin = QSpinBox()
        self._active_spin.setRange(1, 36500)
        self._active_spin.setValue(int(s["active_days"]))
        self._active_spin.setSuffix(" days")
        form.addRow("Active if data within:", self._active_spin)
        self._nearby_spin = QDoubleSpinBox()
        self._nearby_spin.setDecimals(2)
        self._nearby_spin.setRange(0.0, 1000.0)
        self._nearby_spin.setValue(float(s["nearby_km"]))
        self._nearby_spin.setSuffix(" km")
        form.addRow("Nearby sensor site within:", self._nearby_spin)
        self._recalc_timer = QTimer(self)
        self._recalc_timer.setSingleShot(True)
        self._recalc_timer.setInterval(int(s["recalc_debounce_ms"]))
        self._recalc_timer.timeout.connect(self._recalculate)
        self._active_spin.valueChanged.connect(lambda _v: self._recalc_timer.start())
        self._nearby_spin.valueChanged.connect(lambda _v: self._recalc_timer.start())
        self._hidden_chk = QCheckBox("Include hidden cameras in tables and map")
        self._hidden_chk.setChecked(bool(s["include_hidden"]))
        self._hidden_chk.toggled.connect(self._refresh_views)
        form.addRow(self._hidden_chk)
        self._sensor_sites_chk = QCheckBox("Show active sensor sites on map")
        self._sensor_sites_chk.setChecked(bool(s["show_sensor_sites_on_map"]))
        self._sensor_sites_chk.toggled.connect(self._refresh_map)
        form.addRow(self._sensor_sites_chk)
        v.addWidget(opts)

        out = QGroupBox("Output folder")
        out_row = QHBoxLayout(out)
        self._folder_lbl = QLabel(s["output_folder"] or "No folder selected")
        self._folder_lbl.setWordWrap(True)
        browse = QPushButton("Browse\u2026")
        browse.clicked.connect(self._browse)
        out_row.addWidget(self._folder_lbl, 1)
        out_row.addWidget(browse)
        v.addWidget(out)

        self._run_btn = QPushButton("Run Inventory")
        self._run_btn.setFixedHeight(32)
        self._run_btn.clicked.connect(self._run)
        v.addWidget(self._run_btn)

        summ = QGroupBox("Summary")
        self._summary_grid = QGridLayout(summ)
        self._summary_grid.setHorizontalSpacing(12)
        v.addWidget(summ)
        self._show_summary(None)
        v.addStretch()
        return panel

    def _build_map(self):
        MapWidget = _import_map_widget()
        self._map = MapWidget(bounds_debounce_ms=int(self._settings["map_debounce_ms"]))
        lat, lon = self._settings["initial_center"]
        self._map.set_center(lat, lon, int(self._settings["initial_zoom"]))
        return self._map

    @staticmethod
    def _new_table():
        t = QTableWidget()
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setAlternatingRowColors(True)
        t.verticalHeader().setVisible(False)
        return t

    def sizeHint(self):
        width, height = self._settings["window_size"]
        return QSize(int(width), int(height))

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_window_sized", False):
            self._window_sized = True
            win = self.window()
            if win is not self and win.windowTitle() == PLUGIN["title"]:
                width, height = self._settings["window_size"]
                win.resize(max(win.width(), int(width)), max(win.height(), int(height)))

    # ------------------------------------------------------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------------------------------------------------------
    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Output folder", self._settings["output_folder"] or "")
        if folder:
            self._settings["output_folder"] = folder
            self._folder_lbl.setText(folder)
            _save_settings(self._settings)

    def _collect_settings(self):
        s = self._settings
        s["active_days"] = self._active_spin.value()
        s["nearby_km"] = self._nearby_spin.value()
        s["include_hidden"] = self._hidden_chk.isChecked()
        s["show_sensor_sites_on_map"] = self._sensor_sites_chk.isChecked()
        _save_settings(s)

    def _run(self):
        if not self._settings["output_folder"]:
            self._browse()
            if not self._settings["output_folder"]:
                return
        if not self._sensor_codes():
            QMessageBox.warning(self, PLUGIN["title"],
                                "Check at least one measurement in the Co-located Sensor column of the "
                                "Measurements at Camera Sites tab.")
            return
        self._collect_settings()
        self._run_btn.setEnabled(False)
        self._worker = InventoryWorker(self._settings, preloaded=self._preloaded)
        self._worker.status.connect(self._log.appendPlainText)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _start_preload(self):
        self._pre_worker = SiteMeasurementsWorker(self._settings)
        self._pre_worker.status.connect(self._log.appendPlainText)
        self._pre_worker.done.connect(self._on_preload_done)
        self._pre_worker.failed.connect(self._on_preload_failed)
        self._pre_worker.start()

    def _on_preload_done(self, data):
        self._preloaded = data
        self._pre_rows, self._param_names = _preload_rows(data["cams"], data["site_series"])
        # Keep only checked codes that camera sites actually record; any other code could
        # never be unchecked because it has no row in the table.
        codes = [c for c in self._sensor_codes() if c in self._param_names]
        if self._param_names and codes != self._sensor_codes():
            self._settings["colocated_sensor_codes"] = ",".join(codes)
            _save_settings(self._settings)
        self._update_sensor_label()
        if self._result is None:
            self._refresh_param_table()
        n_sites = len({r["Site ID"] for r in self._pre_rows if r["_params"]})
        self._log.appendPlainText(f"Loaded {len(self._param_names):,} measurements at {n_sites:,} camera sites "
                                  f"({data['requests']} requests).")

    def _on_preload_failed(self, msg):
        self._log.appendPlainText(f"Could not load measurements at camera sites: {msg}. Run Inventory will try again.")
        if self._result is None:
            self._show_param_message("Measurements could not be loaded. Run Inventory to try again.")

    def _show_param_message(self, text):
        """One-cell message in the measurements table (while loading, or if loading failed)."""
        t = self._param_table
        t.blockSignals(True)
        t.setSortingEnabled(False)
        t.clear()
        t.setColumnCount(1)
        t.setRowCount(1)
        t.setHorizontalHeaderLabels([""])
        t.setItem(0, 0, QTableWidgetItem(text))
        t.resizeColumnsToContents()
        t.blockSignals(False)
        self._param_header.set_state("hidden")

    def _recalculate(self):
        """Active window or nearby distance changed: recount from data already loaded (no new requests)."""
        self._collect_settings()
        if self._result is not None and "inputs" in self._result:
            i = self._result["inputs"]
            fresh = _build_inventory(i["cams"], i["site_series"], i["national"], i["codes"], self._settings)
            for key in ("inputs", "raw", "requests", "rate_remaining", "elapsed_s"):
                fresh[key] = self._result[key]
            self._result = fresh
            self._show_summary(fresh["summary"])
            self._refresh_views()
            self._log.appendPlainText(f"Recalculated for data within {self._active_spin.value()} days and "
                                      f"nearby within {self._nearby_spin.value():g} km. "
                                      "Run Inventory again to write output files with these settings.")
        elif self._pre_rows:
            self._refresh_param_table()

    def _sensor_codes(self):
        return [c.strip() for c in str(self._settings["colocated_sensor_codes"]).split(",") if c.strip()]

    def _update_sensor_label(self):
        names = []
        for code in self._sensor_codes():
            name = self._param_names.get(code, ("", ""))[0]
            names.append(f"{name} ({code})" if name else code)
        self._sensor_label.setText("; ".join(names) if names else "None checked")

    def _on_param_item_changed(self, item):
        if item.column() != 0 or not (item.flags() & Qt.ItemIsUserCheckable):
            return
        code = item.data(Qt.UserRole)
        codes = self._sensor_codes()
        if item.checkState() == Qt.Checked and code not in codes:
            codes.append(code)
        elif item.checkState() != Qt.Checked and code in codes:
            codes.remove(code)
        self._settings["colocated_sensor_codes"] = ",".join(codes)
        _save_settings(self._settings)
        self._update_sensor_label()
        self._update_header_check()

    # ------------------------------------------------------------------------------------------------------------------
    # Column chooser: right-click a table header to show or hide columns (saved per table)
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _header_names(view):
        if isinstance(view, QTreeWidget):
            h = view.headerItem()
            return [h.text(c) for c in range(view.columnCount())]
        return [(view.horizontalHeaderItem(c).text().strip() if view.horizontalHeaderItem(c) else "")
                for c in range(view.columnCount())]

    def _apply_hidden_columns(self, view, table_name):
        hidden = set(self._settings["hidden_columns"].get(table_name, []))
        for c, name in enumerate(self._header_names(view)):
            view.setColumnHidden(c, name in hidden)

    def _column_menu(self, view, table_name, fixed, pos):
        from PyQt5.QtWidgets import QMenu
        names = self._header_names(view)
        if not any(names):
            return
        hidden = set(self._settings["hidden_columns"].get(table_name, []))
        menu = QMenu(self)
        for name in names:
            act = menu.addAction(name)
            act.setCheckable(True)
            act.setChecked(name not in hidden)
            act.setEnabled(name != fixed)       # the first column always stays
            act.setData(name)
        hdr = view.header() if isinstance(view, QTreeWidget) else view.horizontalHeader()
        chosen = menu.exec_(hdr.mapToGlobal(pos))
        if chosen is None:
            return
        name = chosen.data()
        hidden = [n for n in self._settings["hidden_columns"].get(table_name, []) if n != name]
        if not chosen.isChecked():
            hidden.append(name)
        self._settings["hidden_columns"][table_name] = hidden
        _save_settings(self._settings)
        self._apply_hidden_columns(view, table_name)

    def _style_header(self, hdr):
        """Light blue header with bold, centered black text."""
        hdr.setStyleSheet(
            f"QHeaderView::section {{ background-color: {self._settings['header_color']}; color: #000000;"
            f" font-weight: bold; padding: 4px; border: none;"
            f" border-right: 1px solid {self._settings['header_border_color']};"
            f" border-bottom: 1px solid {self._settings['header_border_color']}; }}")
        hdr.setDefaultAlignment(Qt.AlignCenter)
        f = hdr.font()
        f.setBold(True)
        hdr.setFont(f)

    def _sensor_check_items(self):
        t = self._param_table
        return [t.item(r, 0) for r in range(t.rowCount())
                if t.item(r, 0) is not None and t.item(r, 0).flags() & Qt.ItemIsUserCheckable]

    def _update_header_check(self):
        items = self._sensor_check_items()
        n = sum(1 for it in items if it.checkState() == Qt.Checked)
        self._param_header.set_state("all" if items and n == len(items) else "some" if n else "none")

    def _toggle_all_sensors(self):
        """Header checkbox: check every measurement, or uncheck all when all are checked."""
        items = self._sensor_check_items()
        if not items:
            return
        target = Qt.Unchecked if all(it.checkState() == Qt.Checked for it in items) else Qt.Checked
        t = self._param_table
        t.blockSignals(True)
        for it in items:
            it.setCheckState(target)
        t.blockSignals(False)
        self._settings["colocated_sensor_codes"] = ",".join(it.data(Qt.UserRole) for it in items
                                                            if it.checkState() == Qt.Checked)
        _save_settings(self._settings)
        self._update_sensor_label()
        self._update_header_check()
        t.viewport().update()

    def _refresh_param_table(self):
        """Measurements table from the run result if there is one, else from the preload,
        with a Co-located Sensor checkbox column first."""
        if self._result is not None:
            rows, names = self._shown_rows(), self._result["param_names"]
            cutoff = self._result["cutoff"]
        else:
            rows = self._pre_rows if self._hidden_chk.isChecked() else [r for r in self._pre_rows
                                                                         if r["Hidden"] == "No"]
            names = self._param_names
            cutoff = datetime.now(timezone.utc) - timedelta(days=float(self._active_spin.value()))
        records = [{COL_COLOCATED: "", **r} for r in _parameter_table(rows, names, cutoff)]
        t = self._param_table
        t.blockSignals(True)
        self._fill_table(t, records)
        t.setSortingEnabled(False)
        codes = set(self._sensor_codes())
        for i, rec in enumerate(records):
            item = QTableWidgetItem("")
            item.setFlags((item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
            item.setData(Qt.UserRole, rec[COL_CODE])
            item.setCheckState(Qt.Checked if rec[COL_CODE] in codes else Qt.Unchecked)
            t.setItem(i, 0, item)
        for c in range(t.columnCount()):
            h = t.horizontalHeaderItem(c)
            if h is not None and h.text() in PARAM_COLUMN_TIPS:
                h.setToolTip(PARAM_COLUMN_TIPS[h.text()])
                if h.text() == COL_COLOCATED:
                    h.setToolTip(PARAM_COLUMN_TIPS[COL_COLOCATED] + ". The box in this header checks all, "
                                 "or unchecks all when every one is checked.")
                    # room for the select-all box drawn at the left of this header
                    space = t.fontMetrics().horizontalAdvance(" ")
                    h.setText(" " * (self._param_header.text_indent() // max(space, 1) + 1) + COL_COLOCATED)
                    h.setTextAlignment(Qt.AlignCenter)
        t.setSortingEnabled(True)
        t.resizeColumnsToContents()
        t.blockSignals(False)
        self._apply_hidden_columns(t, "Measurements")
        self._update_header_check()

    def _on_failed(self, msg):
        self._run_btn.setEnabled(True)
        self._log.appendPlainText(f"Inventory failed: {msg}")

    def _on_done(self, result):
        self._run_btn.setEnabled(True)
        self._result = result
        self._show_summary(result["summary"])
        self._refresh_views()
        try:
            folder = self._write_output(result)
            self._log.appendPlainText(f"Done: {result['requests']} requests, {result['elapsed_s']:.1f} s. "
                                      f"Wrote {folder}")
        except Exception as e:
            self._log.appendPlainText(f"Could not write output: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------------------------------------------------
    def _write_output(self, result):
        stamp = result["summary"]["run_time"].astimezone().strftime("%Y%m%d_%H%M%S")
        folder = Path(self._settings["output_folder"]) / f"CameraSensorInventory_{stamp}"
        raw = folder / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        for name, data in result["raw"].items():
            with open(raw / name, "w", encoding="utf-8") as f:
                json.dump(data, f)
        lines = _summary_lines(result["summary"], result["requests"], result["elapsed_s"], result["rate_remaining"])
        (folder / "Summary.txt").write_text("\n".join(lines), encoding="utf-8")
        cams = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in result["rows"]])
        cams.to_csv(folder / "Cameras.csv", index=False)
        pd.DataFrame([{"Camera ID": r["Camera ID"], "Site ID": r["Site ID"], "Hidden": r["Hidden"],
                       **{k: v for k, v in sen.items() if k != "Code"}}
                      for r in result["rows"] for sen in r["_sensors"]],
                     columns=["Camera ID", "Site ID", "Hidden", "Sensor", "Status",
                              "Last Value at Camera Site (UTC)", "Nearest Active Site", "Distance (km)"]
                     ).to_csv(folder / "Camera_Colocated_Sensors.csv", index=False)
        pd.DataFrame(_parameter_table(result["rows"], result["param_names"], result["cutoff"])).to_csv(
            folder / "Parameters.csv", index=False)
        _matrix(result["rows"], result["cutoff"]).to_csv(folder / "Camera_Parameter_Matrix.csv", index=False)
        return folder

    # ------------------------------------------------------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------------------------------------------------------
    def _show_summary(self, summary):
        g = self._summary_grid
        while g.count():
            w = g.takeAt(0).widget()
            if w:
                w.deleteLater()
        if summary is None:
            g.addWidget(QLabel("Run the inventory to see counts."), 0, 0)
            return
        bold = QFont()
        bold.setBold(True)

        def add(row, col, text, b=False, align=Qt.AlignLeft):
            lbl = QLabel(str(text))
            if b:
                lbl.setFont(bold)
            g.addWidget(lbl, row, col, alignment=align)

        # Name the sensors this run looked for, so every count below reads in their terms
        names = self._result["param_names"] if self._result else self._param_names
        sensors = "; ".join(f"{names.get(c, ('', ''))[0] or c} ({c})" for c in summary["codes"])
        head = QLabel(f"Co-located sensors: {sensors}")
        head.setWordWrap(True)
        head.setFont(bold)
        g.addWidget(head, 0, 0, 1, 3)
        r = 1
        for label, value in (("HIVIS cameras", f"{summary['cameras_total']:,}"),
                             ("  Visible", f"{summary['cameras_visible']:,}"),
                             ("  Hidden", f"{summary['cameras_hidden']:,}"),
                             ("USGS sites with these sensors", f"{summary['sensor_sites_total']:,}"),
                             ("  Active", f"{summary['sensor_sites_active']:,}")):
            add(r, 0, label)
            add(r, 1, value, align=Qt.AlignRight)
            r += 1
        r += 1
        add(r, 0, "Cameras by co-location", b=True)
        add(r, 1, "Visible", b=True, align=Qt.AlignRight)
        add(r, 2, "All", b=True, align=Qt.AlignRight)
        r += 1
        for st, (vis, allc) in summary["status_counts"].items():
            add(r, 0, st)
            add(r, 1, f"{vis:,}", align=Qt.AlignRight)
            add(r, 2, f"{allc:,}", align=Qt.AlignRight)
            r += 1

    def _shown_rows(self):
        if not self._result:
            return []
        rows = self._result["rows"]
        return rows if self._hidden_chk.isChecked() else [r for r in rows if r["Hidden"] == "No"]

    def _refresh_views(self):
        if not self._result:
            if self._pre_rows:
                self._refresh_param_table()
            return
        rows = self._shown_rows()
        self._fill_camera_tree(rows)
        self._refresh_param_table()
        self._refresh_map()

    # Child rows put each co-located sensor's values under the matching camera columns
    SENSOR_CHILD_COLUMNS = {
        "Camera ID": "Sensor",
        "Co-location Status": "Status",
        "Last Co-located Sensor Value (UTC)": "Last Value at Camera Site (UTC)",
        "Nearest Active Sensor Site": "Nearest Active Site",
        "Distance to Nearest Active Sensor Site (km)": "Distance (km)",
    }

    def _fill_camera_tree(self, rows):
        """One row per camera; beneath it, one row per checked sensor co-located with it."""
        tree = self._camera_table
        tree.setSortingEnabled(False)
        tree.clear()
        cols = [k for k in rows[0] if not k.startswith("_")] if rows else []
        tree.setColumnCount(len(cols))
        tree.setHeaderLabels(cols)

        def set_cell(item, col, v):
            item.setText(col, "" if v is None else (f"{v:,}" if isinstance(v, int) and not isinstance(v, bool)
                                                     else str(v)))
            item.setData(col, Qt.UserRole, v if isinstance(v, (int, float)) and v is not None else None)

        for r in rows:
            parent = _SortableTreeItem()
            for j, col in enumerate(cols):
                set_cell(parent, j, r[col])
                if col == "Measurements at Site" and r.get("_measurement_names"):
                    parent.setToolTip(j, "\n".join(r["_measurement_names"]))
            for sensor in r.get("_sensors", []):
                child = QTreeWidgetItem()
                for j, col in enumerate(cols):
                    key = self.SENSOR_CHILD_COLUMNS.get(col)
                    if key:
                        set_cell(child, j, sensor[key])
                parent.addChild(child)
            tree.addTopLevelItem(parent)
        self._camera_stack.setCurrentWidget(self._camera_table)
        tree.setSortingEnabled(True)
        tree.sortByColumn(0, Qt.AscendingOrder)
        tree.expandAll()
        for j in range(len(cols)):
            tree.resizeColumnToContents(j)
        self._apply_hidden_columns(tree, "Cameras")

    @staticmethod
    def _fill_table(table, records):
        table.setSortingEnabled(False)
        table.clear()
        cols = list(records[0].keys()) if records else []
        table.setColumnCount(len(cols))
        table.setRowCount(len(records))
        table.setHorizontalHeaderLabels(cols)
        for i, rec in enumerate(records):
            for j, col in enumerate(cols):
                v = rec[col]
                item = QTableWidgetItem()
                if isinstance(v, (int, float)) and v is not None:
                    item.setData(Qt.DisplayRole, v)
                else:
                    item.setText("" if v is None else str(v))
                table.setItem(i, j, item)
        table.setSortingEnabled(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.resizeColumnsToContents()

    def _refresh_map(self):
        if not self._result:
            return
        s = self._settings
        style = {"radius": s["dot_radius"], "weight": s["dot_border_width"], "fill_opacity": s["dot_fill_opacity"]}
        markers = []
        for r in self._shown_rows():
            if r["Camera Latitude"] is None or r["Camera Longitude"] is None:
                continue
            fill, border = s["colors"][r["Co-location Status"]]
            dist = r["Distance to Nearest Active Sensor Site (km)"]
            tip = (f"<b>{r['Camera Name']}</b><br>Camera ID: {r['Camera ID']}<br>Site ID: {r['Site ID']}<br>"
                   f"Status: {r['Co-location Status']}<br>Hidden: {r['Hidden']}<br>"
                   f"Nearest active sensor site: {r['Nearest Active Sensor Site']}"
                   + (f" ({dist:.2f} km)" if dist is not None else "")
                   + "".join(f"<br>{sen['Sensor']}: {sen['Status']}"
                             + (f", {sen['Distance (km)']:.2f} km" if sen['Distance (km)'] is not None
                                and sen['Status'] == STATUSES[1] else "")
                             for sen in r["_sensors"]))
            markers.append({"id": r["Camera ID"], "lat": r["Camera Latitude"], "lng": r["Camera Longitude"],
                            "tooltip": tip, "color": fill, "border": border})
        self._map.clear_marker_layer(SENSOR_SITE_MARKER_LAYER)
        if self._sensor_sites_chk.isChecked():
            fill, border = s["colors"]["sensor_site"]
            sites = [{"id": sid, "lat": lat, "lng": lon, "color": fill, "border": border,
                      "tooltip": f"<b>{sid}</b><br>Last value: {_utc_text(last)} UTC"}
                     for sid, lat, lon, last in self._result["sensor_site_points"]]
            self._map.add_marker_layer(SENSOR_SITE_MARKER_LAYER, sites, {**style, "radius": s["sensor_site_dot_radius"]})
        self._map.add_marker_layer(CAMERA_MARKER_LAYER, markers, style)
