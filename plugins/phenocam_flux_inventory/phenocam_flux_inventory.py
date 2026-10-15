#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# phenocam_flux_inventory.py
#
# Plugin: PhenoCam Flux Co-location Inventory.
#
# Lists PhenoCam cameras and whether each has a co-located flux tower, using only
# what the PhenoCam network reports (sitemetadata.flux_data / flux_networks /
# flux_sitenames). For cameras co-located with an AmeriFlux site, the sensors
# (measured variables) at that site are filled in from the AmeriFlux API.
#
# Co-location is PhenoCam's own flag; it is not inferred from USGS or any other
# network. AmeriFlux is the one enrichment, kept in its own AmeriFlux_API class,
# so other flux networks (ICOS, etc.) can be added later the same way.
#
# Output goes to <output folder>/PhenoCamFluxInventory_YYYYMMDD_HHMMSS/:
#   Summary.txt, Cameras.csv, Camera_Flux_Sensors.csv (one row per camera and
#   AmeriFlux sensor).

import ast
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import importlib
import pandas as pd

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QPlainTextEdit,
                             QFileDialog, QGridLayout, QSplitter, QTabWidget, QStackedWidget,
                             QSizePolicy, QAbstractItemView, QTreeWidget, QTreeWidgetItem)

from appcore.app_identity import PLUGINS_DIR
from appcore.phenocam.Phenocam_API import Phenocam_API
from appcore.ameriflux.AmeriFlux_API import AmeriFlux_API

# ICOS is an optional add-on: if its module isn't present yet, the plugin still loads and runs,
# just without ICOS matching. Create appcore/icos/ (with __init__.py) and drop ICOS_API.py in it.
try:
    from appcore.icos.ICOS_API import ICOS_API
except Exception:
    ICOS_API = None

PLUGIN = {
    "title":       "PhenoCam Flux Co-location Inventory",
    "class":       "PhenoCamFluxInventoryTab",
    "description": "List PhenoCam cameras co-located with flux towers and the sensors at AmeriFlux sites",
    "surface":     "tools",
    "api_version": 2,
}

USER_ROOT = Path(str(PLUGINS_DIR)).parent
SETTINGS_FILE = USER_ROOT / "Settings" / "phenocam_flux_inventory.json"

# Co-location status, in order of precedence (drives the map legend and colors)
STATUS_FLUX_ACTIVE = "Flux tower, camera active"
STATUS_FLUX_INACTIVE = "Flux tower, camera inactive"
STATUS_NO_FLUX = "No flux tower"
STATUSES = [STATUS_FLUX_ACTIVE, STATUS_FLUX_INACTIVE, STATUS_NO_FLUX]

CAMERA_MARKER_LAYER = "phenocam_cameras"

DEFAULT_SETTINGS = {
    "output_folder":    "",
    "request_timeout_s": 120,
    "window_size":      [1330, 700],
    "left_panel_width": 330,
    "status_lines":     2,
    "map_debounce_ms":  500,
    "initial_center":   [20.0, 0.0],   # PhenoCam is a global network
    "initial_zoom":     2,
    "dot_radius":       5,
    "dot_border_width": 1,
    "dot_fill_opacity": 0.9,
    "colors": {                        # fill, border, per status
        STATUS_FLUX_ACTIVE:   ["#1a9850", "#0b4d27"],
        STATUS_FLUX_INACTIVE: ["#f39c12", "#7a4d05"],
        STATUS_NO_FLUX:       ["#9e9e9e", "#424242"],
    },
}


# ======================================================================================================================
# Settings
# ======================================================================================================================
def _load_settings() -> dict:
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        saved_colors = {k: v for k, v in saved.pop("colors", {}).items() if k in settings["colors"]}
        settings["colors"].update(saved_colors)
        settings.update(saved)
    except Exception:
        pass
    return settings


def _save_settings(settings: dict):
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"[phenocam_flux_inventory] Could not save settings: {e}")


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
# Helpers: parse PhenoCam's flux fields (lists or strings, either json.dumps- or repr-style)
# ======================================================================================================================
def _as_list(value):
    """PhenoCam returns flux_networks / flux_sitenames as a Python list, or as a string holding one
    (json or repr). Return a list; [] for empty / 'No' / blank."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if s == "" or s.lower() in ("no", "none", "nan", "[]"):
        return []
    if s[0] in "[{":
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(s)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                pass
    # a plain string, possibly several ids separated by comma or semicolon
    return [part.strip() for part in s.replace(";", ",").split(",") if part.strip()]


def _network_names(flux_networks):
    names = []
    for n in _as_list(flux_networks):
        name = n.get("Name") if isinstance(n, dict) else str(n)
        if name:
            names.append(str(name).strip())
    return names


def _site_id_tokens(flux_sitenames):
    tokens = []
    for s in _as_list(flux_sitenames):
        token = s.get("Name") if isinstance(s, dict) else str(s)
        token = str(token).strip()
        if token and token.lower() != "no":
            tokens.append(token)
    return tokens


def _truthy(v):
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def _country_name(cc):
    """ISO-2 country code to full name when pycountry is available, else the code."""
    if not cc:
        return ""
    try:
        import pycountry
        rec = pycountry.countries.get(alpha_2=str(cc).upper())
        return rec.name if rec else str(cc)
    except Exception:
        return str(cc)


def _reverse_geocode(cams_df):
    """Country and state for each camera from its Lat/Lon, using offline reverse geocoding (no
    per-camera web calls). Returns (countries, states) aligned to cams_df rows, or None if the
    reverse_geocoder package is not installed."""
    try:
        import reverse_geocoder as rg
    except Exception:
        return None
    lats, lons = cams_df.get("Lat"), cams_df.get("Lon")
    coords, idx = [], []
    for i in range(len(cams_df)):
        lat = lats.iloc[i] if lats is not None else None
        lon = lons.iloc[i] if lons is not None else None
        if pd.notna(lat) and pd.notna(lon):
            coords.append((float(lat), float(lon)))
            idx.append(i)
    countries = [""] * len(cams_df)
    states = [""] * len(cams_df)
    if coords:
        try:
            results = rg.search(coords, mode=1, verbose=False)
        except TypeError:
            results = rg.search(coords)   # older signature without mode/verbose
        for i, res in zip(idx, results):
            states[i] = res.get("admin1", "") or ""
            countries[i] = _country_name(res.get("cc", ""))
    return countries, states


# ======================================================================================================================
# Worker: fetch PhenoCam cameras + AmeriFlux tables and build the inventory in the background
# ======================================================================================================================
class FluxInventoryWorker(QThread):
    status = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def run(self):
        try:
            t0 = time.perf_counter()
            self.status.emit("Fetching PhenoCam cameras…")
            cams = Phenocam_API().get_cameras(all_records=True)

            # Country / state from coordinates (offline reverse geocoding), filled once for all cameras.
            self.status.emit("Resolving country/state from coordinates…")
            geo = _reverse_geocode(cams)
            if geo is None:
                self.status.emit(f"reverse_geocoder not importable in this interpreter; "
                                 f"State/Country left blank. Install it into THIS python:")
                self.status.emit(f'  "{sys.executable}" -m pip install reverse_geocoder')
                cams["_country"], cams["_state"] = "", ""
            else:
                cams["_country"], cams["_state"] = geo[0], geo[1]

            # AmeriFlux enrichment: degrade gracefully if the AmeriFlux API is unreachable.
            amf_ids, availability, variables, coverage = set(), None, None, {}
            try:
                self.status.emit("Fetching AmeriFlux sites and variable availability…")
                amf = AmeriFlux_API()
                sites = amf.get_sites()
                if "SITE_ID" in sites.columns:
                    amf_ids = set(sites["SITE_ID"].astype(str))
                availability = amf.get_variable_availability()
                variables = amf.get_fp_variables()
                cov = amf.get_data_coverage()
                if {"SITE_ID", "publish_years"}.issubset(cov.columns):
                    coverage = {str(r["SITE_ID"]): r["publish_years"] for _, r in cov.iterrows()}
                self._amf = amf
            except Exception as e:
                self.status.emit(f"AmeriFlux unavailable ({type(e).__name__}); co-location shown without sensors.")
                self._amf = None

            # ICOS matching (European hub): the id set is enough to match a camera to an ICOS flux
            # tower. Degrades to no ICOS matching if the Carbon Portal is unreachable.
            icos_ids = set()
            if ICOS_API is None:
                self.status.emit("ICOS_API module not found; ICOS matching skipped "
                                 "(add appcore/icos/ICOS_API.py).")
            else:
                try:
                    self.status.emit("Fetching ICOS stations…")
                    es = ICOS_API().ecosystem_stations()
                    if "id" in es.columns:
                        icos_ids = set(es["id"].astype(str))
                except Exception as e:
                    self.status.emit(f"ICOS unavailable ({type(e).__name__}); ICOS matching skipped.")

            self.status.emit("Building inventory…")
            rows = _build_rows(cams, amf_ids, availability, variables, coverage,
                               getattr(self, "_amf", None), icos_ids, self.status.emit)
            result = {
                "rows": rows,
                "amf_available": bool(amf_ids),
                "icos_available": bool(icos_ids),
                "elapsed_s": time.perf_counter() - t0,
            }
            self.done.emit(result)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


def _build_rows(cams_df, amf_ids, availability, variables, coverage, amf, icos_ids=frozenset(),
                log=lambda *_: None):
    rows = []
    for _, c in cams_df.iterrows():
        flux = _truthy(c.get("sitemetadata.flux_data"))
        active = _truthy(c.get("active"))
        networks = _network_names(c.get("sitemetadata.flux_networks"))
        tokens = _site_id_tokens(c.get("sitemetadata.flux_sitenames"))
        # Match a camera's flux site id against each hub's id set, ignoring the network label.
        amf_site = next((t for t in tokens if t in amf_ids), "")
        icos_site = next((t for t in tokens if t in icos_ids), "")

        if flux:
            status = STATUS_FLUX_ACTIVE if active else STATUS_FLUX_INACTIVE
        else:
            status = STATUS_NO_FLUX

        sensors = []
        if amf_site and amf is not None:
            try:
                sensors = amf.site_variables(amf_site, availability=availability, variables=variables)
            except Exception:
                sensors = []
        years = coverage.get(amf_site)
        year_range = ""
        if isinstance(years, (list, tuple)) and years:
            yrs = [str(y) for y in years]
            year_range = f"{yrs[0]}–{yrs[-1]}" if len(yrs) > 1 else yrs[0]

        lat, lon = c.get("Lat"), c.get("Lon")
        rows.append({
            "Camera": c.get("Sitename", ""),
            "Site Description": c.get("sitemetadata.site_description", "") or "",
            "State": c.get("_state", "") or "",
            "Country": c.get("_country", "") or "",
            "Co-location": status,
            "Flux Networks": ", ".join(networks),
            "Flux Site ID": ", ".join(tokens),
            "AmeriFlux Site": amf_site,
            "ICOS Site": icos_site,
            "AmeriFlux Sensors": len(sensors),
            "AmeriFlux Data Years": year_range,
            "Camera Active": "Yes" if active else "No",
            "First Image": c.get("date_first", "") or "",
            "Last Image": c.get("date_last", "") or "",
            "Infrared": c.get("infrared", "") or "",
            "Latitude": float(lat) if pd.notna(lat) else None,
            "Longitude": float(lon) if pd.notna(lon) else None,
            "_sensors": sensors,
            "_year_range": year_range,
        })
    return rows


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


class PhenoCamFluxInventoryTab(QWidget):

    # Child rows show each AmeriFlux sensor under its camera, in the camera's first column
    SENSOR_CHILD_COLUMN = "Camera"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = _load_settings()
        self._result = None
        self._worker = None
        self._build_ui()

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

        self._camera_stack = QStackedWidget()
        self._camera_empty = QLabel('Click "Run Inventory" to list PhenoCam cameras and their flux co-location.')
        self._camera_empty.setAlignment(Qt.AlignCenter)
        self._camera_empty.setWordWrap(True)
        f = self._camera_empty.font()
        f.setBold(True)
        self._camera_empty.setFont(f)
        self._camera_stack.addWidget(self._camera_empty)
        self._camera_stack.addWidget(self._camera_table)

        self._tabs.addTab(self._build_cameras_tab(), "Cameras")
        self._tabs.addTab(self._build_map(), "Map")
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([int(self._settings["left_panel_width"]), 1000])

        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier", 8))
        # One line minimum; the vertical splitter below lets the user drag it taller.
        self._log.setMinimumHeight(int(self._log.fontMetrics().lineSpacing()
                                       + 2 * self._log.document().documentMargin()
                                       + 2 * self._log.frameWidth()))
        status_layout.addWidget(self._log)

        # Vertical splitter: drag the handle to expand or shrink the Status area.
        v_split = QSplitter(Qt.Vertical)
        v_split.addWidget(splitter)
        v_split.addWidget(status_group)
        v_split.setStretchFactor(0, 1)   # the top content takes the extra space
        v_split.setStretchFactor(1, 0)
        v_split.setCollapsible(0, False)
        lines = int(self._settings["status_lines"])
        status_h = int(self._log.fontMetrics().lineSpacing() * lines
                       + 3 * self._log.document().documentMargin()
                       + 2 * self._log.frameWidth() + 30)   # room for the group-box title
        v_split.setSizes([10000, status_h])
        layout.addWidget(v_split, 1)

    def _build_left_panel(self):
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 6, 0)
        s = self._settings

        out = QGroupBox("Output folder")
        out_row = QHBoxLayout(out)
        self._folder_lbl = QLabel(s["output_folder"] or "No folder selected")
        self._folder_lbl.setWordWrap(True)
        browse = QPushButton("Browse…")
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

    def _build_cameras_tab(self):
        """Cameras tab: Expand all / Collapse all above the camera tree (disabled until a run fills it)."""
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._expand_all_btn = QPushButton("Expand all")
        self._collapse_all_btn = QPushButton("Collapse all")
        self._expand_all_btn.clicked.connect(lambda: self._camera_table.expandAll())
        self._collapse_all_btn.clicked.connect(lambda: self._camera_table.collapseAll())
        for b in (self._expand_all_btn, self._collapse_all_btn):
            b.setEnabled(False)
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(self._camera_stack)
        return page

    def _build_map(self):
        MapWidget = _import_map_widget()
        self._map = MapWidget(bounds_debounce_ms=int(self._settings["map_debounce_ms"]))
        lat, lon = self._settings["initial_center"]
        self._map.set_center(lat, lon, int(self._settings["initial_zoom"]))
        return self._map

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

    def _run(self):
        if not self._settings["output_folder"]:
            self._browse()
            if not self._settings["output_folder"]:
                return   # no folder chosen: don't write into the app's launch directory
        self._run_btn.setEnabled(False)
        self._worker = FluxInventoryWorker()
        self._worker.status.connect(self._log.appendPlainText)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_failed(self, msg):
        self._run_btn.setEnabled(True)
        self._log.appendPlainText(f"Inventory failed: {msg}")

    def _on_done(self, result):
        self._run_btn.setEnabled(True)
        self._result = result
        self._show_summary(result)
        self._fill_camera_tree(result["rows"])
        self._refresh_map()
        try:
            folder = self._write_output(result)
            self._log.appendPlainText(f"Done in {result['elapsed_s']:.1f} s. Wrote {folder}")
        except Exception as e:
            self._log.appendPlainText(f"Could not write output: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------------------------------------------------------
    def _show_summary(self, result):
        g = self._summary_grid
        while g.count():
            w = g.takeAt(0).widget()
            if w:
                w.deleteLater()
        if result is None:
            g.addWidget(QLabel("Run the inventory to see counts."), 0, 0)
            return
        rows = result["rows"]
        bold = QFont()
        bold.setBold(True)

        def add(r, col, text, b=False, align=Qt.AlignLeft):
            lbl = QLabel(str(text))
            if b:
                lbl.setFont(bold)
            g.addWidget(lbl, r, col, alignment=align)

        amf_matched = sum(1 for r in rows if r["AmeriFlux Site"])
        add(0, 0, "PhenoCam cameras", b=True)
        add(0, 1, f"{len(rows):,}", b=True, align=Qt.AlignRight)
        r = 1
        for st in STATUSES:
            add(r, 0, st)
            add(r, 1, f"{sum(1 for x in rows if x['Co-location'] == st):,}", align=Qt.AlignRight)
            r += 1
        add(r, 0, "Matched to an AmeriFlux site")
        add(r, 1, f"{amf_matched:,}", align=Qt.AlignRight)
        r += 1
        add(r, 0, "Matched to an ICOS site")
        add(r, 1, f"{sum(1 for x in rows if x['ICOS Site']):,}", align=Qt.AlignRight)
        r += 1
        if not result.get("amf_available"):
            add(r, 0, "(AmeriFlux unavailable; no sensors)")
            r += 1

        # Cameras by flux network (a camera can list more than one), to show where the cameras are
        net_counts = Counter()
        for x in rows:
            for n in (p.strip() for p in str(x["Flux Networks"]).split(",") if p.strip()):
                net_counts[n] += 1
        if net_counts:
            r += 1
            add(r, 0, "By flux network", b=True)
            add(r, 1, "Cameras", b=True, align=Qt.AlignRight)
            r += 1
            for name, cnt in net_counts.most_common():
                add(r, 0, f"  {name}")
                add(r, 1, f"{cnt:,}", align=Qt.AlignRight)
                r += 1

    def _fill_camera_tree(self, rows):
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

        for row in rows:
            parent = _SortableTreeItem()
            for j, col in enumerate(cols):
                set_cell(parent, j, row[col])
            # Every flux-tower camera is expandable, so the arrow means "has a flux tower". The
            # children are the AmeriFlux sensors when available, otherwise one line saying why not.
            is_flux = row["Co-location"] in (STATUS_FLUX_ACTIVE, STATUS_FLUX_INACTIVE)
            if row["_sensors"]:
                tip = "\n".join(f"{s['name']} ({s['units']})" if s.get("units") else s["name"]
                                for s in row["_sensors"])
                parent.setToolTip(cols.index("AmeriFlux Sensors"), tip)
                for s in row["_sensors"]:
                    child = QTreeWidgetItem()
                    label = f"{s['name']} ({s['units']})" if s.get("units") else s["name"]
                    if s.get("description"):
                        label += f" — {s['description']}"
                    child.setText(cols.index(self.SENSOR_CHILD_COLUMN), label)
                    parent.addChild(child)
            elif is_flux:
                if row["AmeriFlux Site"]:
                    note = f"AmeriFlux site {row['AmeriFlux Site']}: sensor list unavailable"
                elif row["ICOS Site"]:
                    note = f"ICOS station {row['ICOS Site']}: sensor list pending live query"
                elif row["Flux Networks"]:
                    note = f"Flux network {row['Flux Networks']}: sensor list not yet integrated"
                else:
                    note = "Flux tower (network unspecified): no sensor list"
                child = QTreeWidgetItem()
                child.setText(cols.index(self.SENSOR_CHILD_COLUMN), note)
                parent.addChild(child)
            tree.addTopLevelItem(parent)

        self._camera_stack.setCurrentWidget(self._camera_table)
        tree.setSortingEnabled(True)
        tree.sortByColumn(0, Qt.AscendingOrder)
        tree.collapseAll()
        for j in range(len(cols)):
            tree.resizeColumnToContents(j)
        for b in (self._expand_all_btn, self._collapse_all_btn):
            b.setEnabled(bool(rows))

    def _refresh_map(self):
        if not self._result:
            return
        s = self._settings
        style = {"radius": s["dot_radius"], "weight": s["dot_border_width"], "fill_opacity": s["dot_fill_opacity"]}
        by_status = {st: [] for st in STATUSES}
        for r in self._result["rows"]:
            if r["Latitude"] is None or r["Longitude"] is None:
                continue
            fill, border = s["colors"][r["Co-location"]]
            sensor_line = ""
            if r["_sensors"]:
                shown = ", ".join(x["name"] for x in r["_sensors"][:12])
                more = "…" if len(r["_sensors"]) > 12 else ""
                sensor_line = (f"<br>AmeriFlux {r['AmeriFlux Site']}"
                               + (f" ({r['_year_range']})" if r["_year_range"] else "")
                               + f": {len(r['_sensors'])} sensors<br>{shown}{more}")
            elif r["AmeriFlux Site"]:
                sensor_line = f"<br>AmeriFlux {r['AmeriFlux Site']}: sensor list unavailable"
            tip = (f"<b>{r['Camera']}</b><br>{r['Site Description']}<br>"
                   f"Status: {r['Co-location']}<br>"
                   f"Flux networks: {r['Flux Networks'] or 'none'}<br>"
                   f"Last image: {r['Last Image']}" + sensor_line)
            by_status[r["Co-location"]].append(
                {"id": r["Camera"], "lat": r["Latitude"], "lng": r["Longitude"],
                 "tooltip": tip, "color": fill, "border": border})
        for st in STATUSES:
            self._map.add_marker_layer(f"{CAMERA_MARKER_LAYER}::{st}", by_status[st], style,
                                       group=st, group_color=s["colors"][st][0])

    # ------------------------------------------------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------------------------------------------------
    def _write_output(self, result):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = Path(self._settings["output_folder"]) / f"PhenoCamFluxInventory_{stamp}"
        folder.mkdir(parents=True, exist_ok=True)
        rows = result["rows"]

        pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]).to_csv(
            folder / "Cameras.csv", index=False)

        sensor_rows = [{"Camera": r["Camera"], "AmeriFlux Site": r["AmeriFlux Site"],
                        "Sensor": s["name"], "Units": s.get("units", ""), "Description": s.get("description", "")}
                       for r in rows for s in r["_sensors"]]
        pd.DataFrame(sensor_rows,
                     columns=["Camera", "AmeriFlux Site", "Sensor", "Units", "Description"]).to_csv(
            folder / "Camera_Flux_Sensors.csv", index=False)

        counts = {st: sum(1 for r in rows if r["Co-location"] == st) for st in STATUSES}
        lines = [f"PhenoCam Flux Co-location Inventory, run {datetime.now():%Y-%m-%d %H:%M:%S}",
                 "",
                 f"PhenoCam cameras: {len(rows)}"]
        lines += [f"  {st}: {counts[st]}" for st in STATUSES]
        lines += [f"Matched to an AmeriFlux site: {sum(1 for r in rows if r['AmeriFlux Site'])}",
                  f"AmeriFlux sensors available: {'yes' if result.get('amf_available') else 'no'}"]
        (folder / "Summary.txt").write_text("\n".join(lines), encoding="utf-8")
        return folder
