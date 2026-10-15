#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# camera_sensor_links.py
#
# Plugin: Camera Sensor Links.
#
# For each camera location, finds the USGS sensors near it and links each sensor site to its
# Open Knowledge Network (Geoconnex) identifier, for building Smart Samples.
#
# Two searches, because not every sensor is on the river:
#   - Within radius: every USGS time series at sites within a straight-line distance of the
#     camera, any site type (stream gages, wells, springs, lakes, weather stations). USGS Water
#     Data API, time-series-metadata collection, queried by a box and filtered to the circle.
#   - On the river: USGS stream gages up and down the river from the camera's reach, within an
#     along-river distance. USGS NLDI: the camera's coordinates give its NHDPlus reach, and
#     navigation from that reach gives the gages.
# Every site found is then checked in Geoconnex (OKN) for its linkable URI,
#   https://geoconnex.us/usgs/monitoring-location/<Site ID>
# and the number of dataset records the graph holds for it.
#
# The Graph Explorer tab (graph_explorer.py in this folder) runs SPARQL queries against Geoconnex
# or any other OKN graph. Double-clicking a site in Sensors Found opens it there.
#
# The Measurements tab lists every measurement found, all included by default. Unchecking one
# removes it from the Sensors Found tree and from the saved output; no new requests are made.
#
# Save Results writes <output folder>/CameraSensorLinks_YYYYMMDD_HHMMSS/:
#   Camera_Sensor_Series.csv   one row per camera and time series
#   Camera_Sensor_Sites.csv    one row per camera and sensor site
#   Provenance.json            run time, settings, endpoints, queries, requests, errors
#   raw/                       API responses as returned

import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QPlainTextEdit,
                             QFileDialog, QMessageBox, QTabWidget, QLineEdit, QFormLayout, QSpinBox,
                             QDoubleSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
                             QSizePolicy, QAbstractItemView, QTreeWidget, QTreeWidgetItem)

from appcore.plugins import plugin_settings, plugins_folder

from graph_explorer import GraphExplorer      # this plugin's own module, beside this file

PLUGIN = {
    "title":       "Camera Sensor Links (USGS + OKN)",
    "class":       "CameraSensorLinks",
    "description": "Find USGS sensors near camera locations and link them to the Open Knowledge Network",
    "surface":     "tools",
    "size":        [1400, 850],
    "api_version": 2,
}

EARTH_RADIUS_KM = 6371.0088   # IUGG mean Earth radius
SPARQL_JSON = "application/sparql-results+json"
SCHEMA_SUBJECT_OF = "https://schema.org/subjectOf"

FOUND_WITHIN_RADIUS = "Within radius"
FOUND_UPSTREAM = "Upstream on river"
FOUND_DOWNSTREAM = "Downstream on river"
NLDI_DIRECTIONS = {"UM": FOUND_UPSTREAM, "DM": FOUND_DOWNSTREAM}

# Tuning values and remembered choices. Every one can be changed in the settings file.
DEFAULTS = {
    "cameras":              [],        # [[camera name, latitude, longitude], ...]
    "radius_km":            1.0,       # straight-line search radius around each camera
    "river_km":             1.0,       # along-river search distance, up and down
    "active_days":          30,        # a series is active if it has data within this many days
    "excluded_codes":       [],        # measurements unchecked in the Measurements tab
    "output_folder":        "",
    "series_url":           "https://api.waterdata.usgs.gov/ogcapi/v0/collections/time-series-metadata/items",
    "nldi_url":             "https://api.water.usgs.gov/nldi/linked-data",
    "nldi_source":          "nwissite",
    "geoconnex_sparql":     "https://apps.okn.us/geoconnex/sparql",
    "geoconnex_site_uri":   "https://geoconnex.us/usgs/monitoring-location/{site_id}",
    "site_chunk":           200,       # site IDs per series request (URL length limit)
    "geoconnex_chunk":      50,        # site URIs per Geoconnex query
    "page_rows":            10000,     # rows per page for paged requests
    "request_timeout_s":    120,
    "retries":              2,         # extra attempts for a request that fails to connect
    "retry_wait_s":         3.0,
    "left_panel_width":     380,
    "status_lines":         3,
    "header_color":         "#cfe2f3",
    "header_border_color":  "#4682b4",
}

# Panel control ranges
DISTANCE_RANGE_KM = (0.01, 1000.0)
ACTIVE_DAYS_RANGE = (1, 36500)

CAMERA_COLUMNS = ["Camera Name", "Latitude", "Longitude"]
TREE_COLUMNS = ["Camera / Site / Measurement", "How Found", "Distance (km)", "In Geoconnex",
                "Record Type", "First Value (UTC)", "Last Value (UTC)", "Active"]
COL_INCLUDE = "Include"
MEASUREMENT_COLUMNS = [COL_INCLUDE, "Parameter Code", "Measurement", "Unit", "Sites", "Series"]

STYLE = """
QWidget { color: black; }
QGroupBox { border: 1px solid #b0b0b0; border-radius: 8px; margin-top: 10px; padding-top: 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget, QTreeWidget, QPushButton {
    border: 1px solid #b0b0b0; border-radius: 6px; padding: 3px; }
QPushButton:hover { background: #e8eef7; }
"""


# ======================================================================================================================
# Helpers
# ======================================================================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def bbox_around(lat, lon, radius_km):
    """minlon,minlat,maxlon,maxlat of the box that contains the circle."""
    dlat = math.degrees(radius_km / EARTH_RADIUS_KM)
    dlon = dlat / max(math.cos(math.radians(lat)), 1e-12)
    return f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"


def parse_utc(text):
    if not text:
        return None
    try:
        t = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def utc_text(t):
    return t.strftime("%Y-%m-%d %H:%M:%S") if t else ""


def point_lat_lon(geometry):
    """(lat, lon) from a GeoJSON point, else (None, None)."""
    if isinstance(geometry, dict) and geometry.get("type") == "Point":
        c = geometry.get("coordinates") or []
        if len(c) >= 2:
            return float(c[1]), float(c[0])
    return None, None


def parameter_code(props):
    code = str(props.get("parameter_code") or "")
    return code.zfill(5) if code.isdigit() else code


def load_usgs_key():
    """The USGS API key from GRIME AI's key storage, or "" when none is stored."""
    try:
        from appcore.dialogs.api_keys.api_key_manager import APIKeyManager
        return APIKeyManager().get_usgs_key() or ""
    except Exception:
        return ""


# ======================================================================================================================
# Search (no Qt, so it can be tested on its own)
# ======================================================================================================================
class SensorSearch:
    """Runs both searches and the Geoconnex check for a list of cameras."""

    def __init__(self, settings, usgs_key="", status=print):
        self.s = dict(settings)
        self.status = status
        self.requests_made = 0
        self.raw = {}             # file name -> response as returned
        self.queries = []         # Geoconnex query texts, for provenance
        self._usgs = requests.Session()
        if usgs_key:
            # The key goes to the USGS Water Data API only, never to NLDI or Geoconnex.
            self._usgs.headers["X-Api-Key"] = usgs_key
        self._public = requests.Session()

    # ------------------------------------------------------------------------------------------------------------------
    def _get(self, session, url, params=None, headers=None):
        attempts = int(self.s["retries"]) + 1
        for attempt in range(1, attempts + 1):
            try:
                r = session.get(url, params=params, headers=headers, timeout=float(self.s["request_timeout_s"]))
                self.requests_made += 1
                r.raise_for_status()
                return r.json()
            except requests.ConnectionError:
                if attempt == attempts:
                    raise
                time.sleep(float(self.s["retry_wait_s"]))

    def _series_pages(self, params, raw_name):
        """All time-series-metadata features for a query, following rel=next pages."""
        features, pages = [], []
        data = self._get(self._usgs, self.s["series_url"], params)
        while True:
            pages.append(data)
            page = data.get("features", [])
            features += page
            nxt = next((l["href"] for l in data.get("links", []) if l.get("rel") == "next"), None)
            if not nxt or not page:
                break
            data = self._get(self._usgs, nxt)
        self.raw[raw_name] = pages
        return features

    # ------------------------------------------------------------------------------------------------------------------
    def series_within_radius(self, n, lat, lon):
        radius = float(self.s["radius_km"])
        features = self._series_pages({"bbox": bbox_around(lat, lon, radius), "limit": int(self.s["page_rows"]),
                                       "f": "json"}, f"Camera{n}_Series_In_Box.json")
        inside = []
        for f in features:
            slat, slon = point_lat_lon(f.get("geometry"))
            if slat is not None and haversine_km(lat, lon, slat, slon) <= radius:
                inside.append(f)
        return inside

    def series_for_sites(self, n, site_ids):
        chunk = int(self.s["site_chunk"])
        features = []
        for i in range(0, len(site_ids), chunk):
            features += self._series_pages({"monitoring_location_id": ",".join(site_ids[i:i + chunk]),
                                            "limit": int(self.s["page_rows"]), "f": "json"},
                                           f"Camera{n}_Series_River_Gages_{i // chunk + 1}.json")
        return features

    def river_gages(self, n, lat, lon):
        """(reach comid, {site ID: (direction label, lat, lon)}) from NLDI."""
        base = self.s["nldi_url"].rstrip("/")
        reach = self._get(self._public, f"{base}/comid/position", {"coords": f"POINT({lon} {lat})"})
        self.raw[f"Camera{n}_NLDI_Reach.json"] = reach
        feats = reach.get("features") or []
        comid = feats[0].get("properties", {}).get("comid") if feats else None
        gages = {}
        if comid is None:
            return None, gages
        for code, label in NLDI_DIRECTIONS.items():
            data = self._get(self._public, f"{base}/comid/{comid}/navigation/{code}/{self.s['nldi_source']}",
                             {"distance": float(self.s["river_km"])})
            self.raw[f"Camera{n}_NLDI_{label.replace(' ', '_')}.json"] = data
            for f in data.get("features") or []:
                sid = f.get("id") or f.get("properties", {}).get("identifier")
                if sid:
                    glat, glon = point_lat_lon(f.get("geometry"))
                    gages.setdefault(sid, (label, glat, glon))
        return comid, gages

    def geoconnex_datasets(self, site_ids):
        """{site ID: dataset records} for the sites Geoconnex holds; absent sites are left out."""
        found = {}
        chunk = int(self.s["geoconnex_chunk"])
        pattern = self.s["geoconnex_site_uri"]
        for i in range(0, len(site_ids), chunk):
            part = site_ids[i:i + chunk]
            uris = {pattern.format(site_id=sid): sid for sid in part}
            values = " ".join(f"<{u}>" for u in uris)
            query = (f"SELECT ?site (COUNT(DISTINCT ?dataset) AS ?datasets) WHERE {{\n"
                     f"  VALUES ?site {{ {values} }}\n"
                     f"  ?site ?p ?o .\n"
                     f"  OPTIONAL {{ ?site <{SCHEMA_SUBJECT_OF}> ?dataset }}\n"
                     f"}} GROUP BY ?site")
            self.queries.append(query)
            data = self._get(self._public, self.s["geoconnex_sparql"], {"query": query}, {"Accept": SPARQL_JSON})
            self.raw[f"Geoconnex_Sites_{i // chunk + 1}.json"] = data
            for b in data.get("results", {}).get("bindings", []):
                sid = uris.get(b.get("site", {}).get("value"))
                if sid:
                    found[sid] = int(b.get("datasets", {}).get("value", 0))
        return found

    # ------------------------------------------------------------------------------------------------------------------
    def run(self, cameras):
        run_time = datetime.now(timezone.utc)
        results = []
        for n, (name, lat, lon) in enumerate(cameras, start=1):
            cam = {"name": name, "lat": lat, "lon": lon, "comid": None, "errors": [], "sites": {}}
            results.append(cam)

            def site(sid, slat, slon):
                entry = cam["sites"].setdefault(sid, {"lat": slat, "lon": slon, "found_by": [], "series": {}})
                if entry["lat"] is None and slat is not None:
                    entry["lat"], entry["lon"] = slat, slon
                return entry

            self.status(f"{name}: sensors within {float(self.s['radius_km']):g} km…")
            try:
                for f in self.series_within_radius(n, lat, lon):
                    p = f.get("properties", {})
                    slat, slon = point_lat_lon(f.get("geometry"))
                    entry = site(p.get("monitoring_location_id"), slat, slon)
                    if FOUND_WITHIN_RADIUS not in entry["found_by"]:
                        entry["found_by"].append(FOUND_WITHIN_RADIUS)
                    entry["series"][f.get("id") or p.get("id")] = p
            except Exception as e:
                cam["errors"].append(f"Within-radius search: {type(e).__name__}: {e}")

            self.status(f"{name}: gages within {float(self.s['river_km']):g} km along the river…")
            try:
                cam["comid"], gages = self.river_gages(n, lat, lon)
                for sid, (label, glat, glon) in gages.items():
                    entry = site(sid, glat, glon)
                    if label not in entry["found_by"]:
                        entry["found_by"].append(label)
                missing = sorted(sid for sid in gages if not cam["sites"][sid]["series"])
                for f in self.series_for_sites(n, missing) if missing else []:
                    p = f.get("properties", {})
                    sid = p.get("monitoring_location_id")
                    if sid in cam["sites"]:
                        cam["sites"][sid]["series"][f.get("id") or p.get("id")] = p
            except Exception as e:
                cam["errors"].append(f"Along-river search: {type(e).__name__}: {e}")

            for entry in cam["sites"].values():
                entry["distance_km"] = (haversine_km(lat, lon, entry["lat"], entry["lon"])
                                        if entry["lat"] is not None else None)

        all_sites = sorted({sid for cam in results for sid in cam["sites"] if sid})
        geoconnex, geoconnex_error = {}, None
        if all_sites:
            self.status(f"Checking {len(all_sites)} sites in Geoconnex…")
            try:
                geoconnex = self.geoconnex_datasets(all_sites)
            except Exception as e:
                geoconnex_error = f"{type(e).__name__}: {e}"
        return {"run_time": run_time, "cameras": results, "geoconnex": geoconnex,
                "geoconnex_error": geoconnex_error, "requests": self.requests_made,
                "raw": self.raw, "queries": self.queries, "settings": self.s}


# ======================================================================================================================
# Results as tables
# ======================================================================================================================
def measurement_names(result):
    """code -> (measurement name, unit), sites, series counts across all cameras."""
    names, sites, series = {}, {}, {}
    for cam in result["cameras"]:
        for sid, entry in cam["sites"].items():
            for p in entry["series"].values():
                code = parameter_code(p)
                names.setdefault(code, (p.get("parameter_name") or "", p.get("unit_of_measure") or ""))
                sites.setdefault(code, set()).add(sid)
                series[code] = series.get(code, 0) + 1
    return names, {c: len(v) for c, v in sites.items()}, series


def series_rows(result, included, active_days):
    cutoff = datetime.now(timezone.utc) - timedelta(days=float(active_days))
    pattern = result["settings"]["geoconnex_site_uri"]
    url = result["settings"]["series_url"].rstrip("/")
    rows = []
    for cam in result["cameras"]:
        for sid, entry in sorted(cam["sites"].items(), key=lambda kv: (kv[1]["distance_km"] is None,
                                                                      kv[1]["distance_km"] or 0)):
            for series_id, p in entry["series"].items():
                code = parameter_code(p)
                if code not in included:
                    continue
                last = parse_utc(p.get("end_utc"))
                rows.append({
                    "Camera Name": cam["name"],
                    "Camera Latitude": cam["lat"],
                    "Camera Longitude": cam["lon"],
                    "Site ID": sid,
                    "Geoconnex Site URI": pattern.format(site_id=sid),
                    "In Geoconnex": geoconnex_text(result, sid),
                    "How Found": "; ".join(entry["found_by"]),
                    "Distance (km)": round(entry["distance_km"], 3) if entry["distance_km"] is not None else None,
                    "Parameter Code": code,
                    "Measurement": p.get("parameter_name") or "",
                    "Unit": p.get("unit_of_measure") or "",
                    "Record Type": p.get("computation_period_identifier") or "",
                    "Statistic": p.get("computation_identifier") or "",
                    "Statistic ID": p.get("statistic_id") or "",
                    "First Value (UTC)": utc_text(parse_utc(p.get("begin_utc"))),
                    "Last Value (UTC)": utc_text(last),
                    "Active": "Yes" if last and last >= cutoff else "No",
                    "Series ID": series_id,
                    "Series Metadata URL": f"{url}/{series_id}",
                })
    return rows


def site_rows(result, included):
    pattern = result["settings"]["geoconnex_site_uri"]
    rows = []
    for cam in result["cameras"]:
        for sid, entry in cam["sites"].items():
            codes = sorted({parameter_code(p) for p in entry["series"].values()} & included)
            rows.append({
                "Camera Name": cam["name"],
                "Camera Reach (NHDPlus COMID)": cam["comid"],
                "Site ID": sid,
                "Geoconnex Site URI": pattern.format(site_id=sid),
                "In Geoconnex": geoconnex_text(result, sid),
                "Geoconnex Dataset Records": result["geoconnex"].get(sid),
                "How Found": "; ".join(entry["found_by"]),
                "Distance (km)": round(entry["distance_km"], 3) if entry["distance_km"] is not None else None,
                "Site Latitude": entry["lat"],
                "Site Longitude": entry["lon"],
                "Measurements Included": ", ".join(codes),
            })
    return rows


def geoconnex_text(result, sid):
    if result.get("geoconnex_error"):
        return "Not checked"
    return "Yes" if sid in result["geoconnex"] else "No"


# ======================================================================================================================
# Workers
# ======================================================================================================================
class SearchWorker(QThread):
    status = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, settings, cameras):
        super().__init__()
        self._settings, self._cameras = settings, cameras

    def run(self):
        try:
            key = load_usgs_key()
            if not key:
                self.status.emit("No USGS API key stored; using the lower anonymous rate limit.")
            search = SensorSearch(self._settings, key, self.status.emit)
            self.done.emit(search.run(self._cameras))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class GageLookupWorker(QThread):
    """Name and coordinates of one USGS site from NLDI, to add it as a camera location."""
    done = pyqtSignal(str, float, float)
    failed = pyqtSignal(str)

    def __init__(self, settings, site_id):
        super().__init__()
        self._settings, self._site_id = settings, site_id

    def run(self):
        try:
            url = f"{self._settings['nldi_url'].rstrip('/')}/{self._settings['nldi_source']}/{self._site_id}"
            r = requests.get(url, timeout=float(self._settings["request_timeout_s"]))
            r.raise_for_status()
            feats = r.json().get("features") or []
            if not feats:
                raise ValueError(f"NLDI has no site {self._site_id}")
            lat, lon = point_lat_lon(feats[0].get("geometry"))
            name = feats[0].get("properties", {}).get("name") or self._site_id
            self.done.emit(f"{name} ({self._site_id})", lat, lon)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# ======================================================================================================================
# Plugin widget
# ======================================================================================================================
class _SortableTreeItem(QTreeWidgetItem):
    """Sorts numeric columns by value (stored in UserRole) instead of text."""
    def __lt__(self, other):
        col = self.treeWidget().sortColumn() if self.treeWidget() else 0
        a, b = self.data(col, Qt.UserRole), other.data(col, Qt.UserRole)
        if a is not None and b is not None:
            return a < b
        if (a is None) != (b is None):
            return a is None
        return self.text(col).lower() < other.text(col).lower()


class CameraSensorLinks(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLE)
        self._settings = plugin_settings(__file__, DEFAULTS)
        self._result = None
        self._worker = None
        self._lookup = None
        self._build_ui()
        self._load_cameras()

    # ------------------------------------------------------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())

        self._tabs = QTabWidget()
        self._tree = QTreeWidget()
        self._tree.setColumnCount(len(TREE_COLUMNS))
        self._tree.setHeaderLabels(TREE_COLUMNS)
        self._tree.setAlternatingRowColors(True)
        self._style_header(self._tree.header())
        self._measure_table = QTableWidget()
        self._measure_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._measure_table.setAlternatingRowColors(True)
        self._measure_table.verticalHeader().setVisible(False)
        self._style_header(self._measure_table.horizontalHeader())
        self._measure_table.itemChanged.connect(self._on_measure_checked)
        self._tree.setToolTip("Double-click a site to look it up in the Graph Explorer tab")
        self._tree.itemDoubleClicked.connect(self._explore_site)
        self._tree.itemExpanded.connect(lambda _item: self._fit_tree_columns())
        self._explorer = GraphExplorer()
        self._tabs.addTab(self._tree, "Sensors Found")
        self._tabs.addTab(self._measure_table, "Measurements")
        self._tabs.addTab(self._explorer, "Graph Explorer")
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([int(self._settings["left_panel_width"]), 1000])
        layout.addWidget(splitter, 1)

        status = QGroupBox("Status")
        sv = QVBoxLayout(status)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier", 8))
        self._log.setFixedHeight(int(self._log.fontMetrics().lineSpacing() * int(self._settings["status_lines"])
                                     + 3 * self._log.document().documentMargin()
                                     + 2 * self._log.frameWidth()))
        sv.addWidget(self._log)
        status.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(status)

    def _build_left_panel(self):
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 6, 0)
        s = self._settings

        cams = QGroupBox("Camera locations")
        cv = QVBoxLayout(cams)
        self._camera_table = QTableWidget(0, len(CAMERA_COLUMNS))
        self._camera_table.setHorizontalHeaderLabels(CAMERA_COLUMNS)
        self._camera_table.verticalHeader().setVisible(False)
        self._camera_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._style_header(self._camera_table.horizontalHeader())
        self._camera_table.itemChanged.connect(self._save_cameras)
        cv.addWidget(self._camera_table)
        row = QHBoxLayout()
        add = QPushButton("Add camera")
        add.clicked.connect(lambda: self._add_camera_row("", "", ""))
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_cameras)
        row.addWidget(add)
        row.addWidget(remove)
        cv.addLayout(row)
        gage_row = QHBoxLayout()
        self._gage_edit = QLineEdit()
        self._gage_edit.setPlaceholderText("USGS site, e.g. 06770500")
        self._gage_btn = QPushButton("Add site location")
        self._gage_btn.setToolTip("Add a camera at the coordinates of a USGS site (looked up in NLDI)")
        self._gage_btn.clicked.connect(self._lookup_gage)
        gage_row.addWidget(self._gage_edit, 1)
        gage_row.addWidget(self._gage_btn)
        cv.addLayout(gage_row)
        v.addWidget(cams, 1)

        search = QGroupBox("Search")
        form = QFormLayout(search)
        form.setLabelAlignment(Qt.AlignLeft)
        self._radius_spin = self._distance_spin("radius_km")
        self._radius_spin.setToolTip("Any USGS sensor site within this straight-line distance of the camera")
        form.addRow("Within radius:", self._radius_spin)
        self._river_spin = self._distance_spin("river_km")
        self._river_spin.setToolTip("USGS stream gages within this distance up or down the river from the camera")
        form.addRow("Along river:", self._river_spin)
        self._active_spin = QSpinBox()
        self._active_spin.setRange(*ACTIVE_DAYS_RANGE)
        self._active_spin.setSuffix(" days")
        self._settings.bind(self._active_spin, "active_days")
        self._active_spin.valueChanged.connect(lambda _v: self._refresh_tree())
        form.addRow("Active if data within:", self._active_spin)
        v.addWidget(search)

        out = QGroupBox("Output folder")
        orow = QHBoxLayout(out)
        self._folder_lbl = QLabel(s["output_folder"] or "No folder selected")
        self._folder_lbl.setWordWrap(True)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        orow.addWidget(self._folder_lbl, 1)
        orow.addWidget(browse)
        v.addWidget(out)

        self._run_btn = QPushButton("Find Sensors")
        self._run_btn.clicked.connect(self._run)
        v.addWidget(self._run_btn)
        self._save_btn = QPushButton("Save Results")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_results)
        v.addWidget(self._save_btn)
        return panel

    def _distance_spin(self, key):
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(*DISTANCE_RANGE_KM)
        spin.setSuffix(" km")
        self._settings.bind(spin, key)
        return spin

    def _style_header(self, hdr):
        hdr.setStyleSheet(
            f"QHeaderView::section {{ background-color: {self._settings['header_color']}; color: #000000;"
            f" font-weight: bold; padding: 4px; border: none;"
            f" border-right: 1px solid {self._settings['header_border_color']};"
            f" border-bottom: 1px solid {self._settings['header_border_color']}; }}")

    # ------------------------------------------------------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------------------------------------------------------
    def _load_cameras(self):
        for name, lat, lon in self._settings.get("cameras") or []:
            self._add_camera_row(name, lat, lon, save=False)

    def _add_camera_row(self, name, lat, lon, save=True):
        t = self._camera_table
        t.blockSignals(True)
        r = t.rowCount()
        t.insertRow(r)
        for c, v in enumerate((name, lat, lon)):
            t.setItem(r, c, QTableWidgetItem("" if v in (None, "") else str(v)))
        t.blockSignals(False)
        if save:
            self._save_cameras()

    def _remove_cameras(self):
        for r in sorted({i.row() for i in self._camera_table.selectedIndexes()}, reverse=True):
            self._camera_table.removeRow(r)
        self._save_cameras()

    def _camera_cells(self):
        t = self._camera_table
        return [[(t.item(r, c).text().strip() if t.item(r, c) else "") for c in range(len(CAMERA_COLUMNS))]
                for r in range(t.rowCount())]

    def _save_cameras(self, *_):
        self._settings["cameras"] = self._camera_cells()
        self._settings.save()

    def _valid_cameras(self):
        """[(name, lat, lon)] for complete rows; the row numbers of rows that could not be read."""
        good, bad = [], []
        for i, (name, lat, lon) in enumerate(self._camera_cells(), start=1):
            if not (name or lat or lon):
                continue
            try:
                la, lo = float(lat), float(lon)
                if not (-90 <= la <= 90 and -180 <= lo <= 180):
                    raise ValueError
                good.append((name or f"Camera {i}", la, lo))
            except ValueError:
                bad.append(i)
        return good, bad

    def _lookup_gage(self):
        text = self._gage_edit.text().strip()
        if not text:
            return
        site_id = text if text.upper().startswith("USGS-") else f"USGS-{text}"
        self._gage_btn.setEnabled(False)
        self._lookup = GageLookupWorker(self._settings, site_id)
        self._lookup.done.connect(self._on_gage_found)
        self._lookup.failed.connect(self._on_gage_failed)
        self._lookup.start()

    def _on_gage_found(self, name, lat, lon):
        self._gage_btn.setEnabled(True)
        self._add_camera_row(name, lat, lon)
        self._gage_edit.clear()
        self._log.appendPlainText(f"Added {name} at {lat:.6f}, {lon:.6f}.")

    def _on_gage_failed(self, msg):
        self._gage_btn.setEnabled(True)
        self._log.appendPlainText(f"Site lookup failed: {msg}")

    # ------------------------------------------------------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------------------------------------------------------
    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Output folder", self._settings["output_folder"]
                                                  or str(Path(plugins_folder()).parent))
        if folder:
            self._settings["output_folder"] = folder
            self._settings.save()
            self._folder_lbl.setText(folder)

    def _run(self):
        if self._worker is not None and self._worker.isRunning():
            return
        cameras, bad = self._valid_cameras()
        if bad:
            QMessageBox.warning(self, PLUGIN["title"], "Latitude and longitude must be decimal degrees. "
                                f"Check camera row(s): {', '.join(map(str, bad))}.")
            return
        if not cameras:
            QMessageBox.warning(self, PLUGIN["title"], "Add at least one camera location.")
            return
        self._run_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._worker = SearchWorker(dict(self._settings), cameras)
        self._worker.status.connect(self._log.appendPlainText)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_failed(self, msg):
        self._run_btn.setEnabled(True)
        self._log.appendPlainText(f"Search failed: {msg}")

    def _on_done(self, result):
        self._run_btn.setEnabled(True)
        self._result = result
        self._refresh_measurements()
        self._refresh_tree()
        self._save_btn.setEnabled(True)
        n_sites = sum(len(c["sites"]) for c in result["cameras"])
        n_series = sum(len(e["series"]) for c in result["cameras"] for e in c["sites"].values())
        self._log.appendPlainText(f"Found {n_sites} sensor sites and {n_series} time series for "
                                  f"{len(result['cameras'])} camera(s) ({result['requests']} requests). "
                                  "Uncheck measurements to leave out, then Save Results.")
        for cam in result["cameras"]:
            for err in cam["errors"]:
                self._log.appendPlainText(f"{cam['name']}: {err}")
        if result.get("geoconnex_error"):
            self._log.appendPlainText(f"Geoconnex check failed, sites marked Not checked: {result['geoconnex_error']}")

    # ------------------------------------------------------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------------------------------------------------------
    def _included_codes(self):
        if not self._result:
            return set()
        names, _sites, _series = measurement_names(self._result)
        return set(names) - set(self._settings.get("excluded_codes") or [])

    def _refresh_measurements(self):
        names, sites, series = measurement_names(self._result)
        excluded = set(self._settings.get("excluded_codes") or [])
        t = self._measure_table
        t.blockSignals(True)
        t.setSortingEnabled(False)
        t.clear()
        t.setColumnCount(len(MEASUREMENT_COLUMNS))
        t.setHorizontalHeaderLabels(MEASUREMENT_COLUMNS)
        codes = sorted(names, key=lambda c: (-series[c], c))
        t.setRowCount(len(codes))
        for r, code in enumerate(codes):
            check = QTableWidgetItem("")
            check.setFlags((check.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
            check.setData(Qt.UserRole, code)
            check.setCheckState(Qt.Unchecked if code in excluded else Qt.Checked)
            t.setItem(r, 0, check)
            for c, v in enumerate((code, names[code][0], names[code][1], sites[code], series[code]), start=1):
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, v)
                t.setItem(r, c, item)
        t.setSortingEnabled(True)
        t.resizeColumnsToContents()
        t.blockSignals(False)

    def _on_measure_checked(self, item):
        if item.column() != 0 or not (item.flags() & Qt.ItemIsUserCheckable):
            return
        code = item.data(Qt.UserRole)
        excluded = [c for c in (self._settings.get("excluded_codes") or []) if c != code]
        if item.checkState() != Qt.Checked:
            excluded.append(code)
        self._settings["excluded_codes"] = excluded
        self._settings.save()
        self._refresh_tree()

    def _refresh_tree(self):
        tree = self._tree
        tree.setSortingEnabled(False)
        tree.clear()
        if not self._result:
            return
        included = self._included_codes()
        rows = series_rows(self._result, included, self._active_spin.value())
        by_site = {}
        for r in rows:
            by_site.setdefault((r["Camera Name"], r["Site ID"]), []).append(r)
        for cam in self._result["cameras"]:
            top = _SortableTreeItem([cam["name"], "", "", "", "", "", "", ""])
            reach = f"reach {cam['comid']}" if cam["comid"] is not None else "no reach found"
            top.setText(1, f"{len(cam['sites'])} sites; {reach}")
            for sid, entry in cam["sites"].items():
                series = by_site.get((cam["name"], sid), [])
                if not series:
                    continue
                dist = entry["distance_km"]
                child = _SortableTreeItem([sid, "; ".join(entry["found_by"]),
                                           f"{dist:.3f}" if dist is not None else "",
                                           geoconnex_text(self._result, sid), "", "", "", ""])
                child.setData(2, Qt.UserRole, dist)
                child.setData(0, Qt.UserRole, self._result["settings"]["geoconnex_site_uri"].format(site_id=sid))
                for r in series:
                    child.addChild(QTreeWidgetItem([f"{r['Measurement']} ({r['Parameter Code']})", "", "", "",
                                                    r["Record Type"], r["First Value (UTC)"],
                                                    r["Last Value (UTC)"], r["Active"]]))
                top.addChild(child)
            tree.addTopLevelItem(top)
        tree.setSortingEnabled(True)
        # Expand before sizing: a column is only as wide as the rows that are showing.
        tree.expandAll()
        self._fit_tree_columns()

    def _fit_tree_columns(self):
        for c in range(self._tree.columnCount()):
            self._tree.resizeColumnToContents(c)

    def _explore_site(self, item, _column):
        """A site row holds its Geoconnex URI; describe it in the Graph Explorer tab."""
        uri = item.data(0, Qt.UserRole)
        if isinstance(uri, str) and uri:
            self._tabs.setCurrentWidget(self._explorer)
            self._explorer.explore_uri(uri)

    # ------------------------------------------------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------------------------------------------------
    def _save_results(self):
        if not self._result:
            return
        if not self._settings["output_folder"]:
            self._browse()
            if not self._settings["output_folder"]:
                return
        try:
            folder = write_output(self._result, Path(self._settings["output_folder"]),
                                  self._included_codes(), self._active_spin.value())
            self._log.appendPlainText(f"Wrote {folder}")
        except Exception as e:
            self._log.appendPlainText(f"Could not write output: {type(e).__name__}: {e}")


def write_output(result, parent, included, active_days):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = parent / f"CameraSensorLinks_{stamp}"
    raw = folder / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, data in result["raw"].items():
        with open(raw / name, "w", encoding="utf-8") as f:
            json.dump(data, f)
    pd.DataFrame(series_rows(result, included, active_days)).to_csv(folder / "Camera_Sensor_Series.csv", index=False)
    pd.DataFrame(site_rows(result, included)).to_csv(folder / "Camera_Sensor_Sites.csv", index=False)
    s = result["settings"]
    names, _sites, _series = measurement_names(result)
    provenance = {
        "search_run_utc": utc_text(result["run_time"]),
        "saved_local": stamp,
        "within_radius_km": s["radius_km"],
        "along_river_km": s["river_km"],
        "active_if_data_within_days": active_days,
        "measurements_included": sorted(included),
        "measurements_left_out": sorted(set(names) - set(included)),
        "endpoints": {"usgs_time_series_metadata": s["series_url"], "usgs_nldi": s["nldi_url"],
                      "nldi_source": s["nldi_source"], "geoconnex_sparql": s["geoconnex_sparql"],
                      "geoconnex_site_uri_pattern": s["geoconnex_site_uri"]},
        "geoconnex_queries": result["queries"],
        "geoconnex_error": result["geoconnex_error"],
        "requests": result["requests"],
        "cameras": [{"name": c["name"], "latitude": c["lat"], "longitude": c["lon"],
                     "reach_nhdplus_comid": c["comid"], "sites_found": len(c["sites"]), "errors": c["errors"]}
                    for c in result["cameras"]],
    }
    with open(folder / "Provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=4)
    return folder
