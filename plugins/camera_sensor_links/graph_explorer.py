#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# graph_explorer.py
#
# The Graph Explorer tab of the Camera Sensor Links plugin: query an Open Knowledge Network
# (OKN) knowledge graph, such as Geoconnex, over SPARQL and save every run with its provenance.
# Part of the camera_sensor_links plugin folder; it is not a plugin on its own.
#
# Results are parsed only as the W3C SPARQL 1.1 Query Results JSON format, so the
# explorer works with any SPARQL endpoint. Graph-specific queries are presets that
# fill the query editor; the editor text is always what is sent.
#
# Each saved run goes in <output folder>/KnowledgeGraphQuery_YYYYMMDD_HHMMSS/:
#     query.rq        the exact query sent
#     results.json    the raw response
#     results.csv     one row per result, one column per query variable
#     provenance.json endpoint, query, run time, row count, elapsed seconds
#
# Double-clicking a URI in the results copies it to Entity URI and describes it,
# so the graph can be explored one link at a time. explore_uri() does the same for
# a URI sent from another tab.

import csv
import json
import os
import time
from datetime import datetime

import requests

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QLabel,
                             QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QPushButton, QPlainTextEdit, QTableWidget, QTableWidgetItem,
                             QFileDialog, QSplitter, QFrame, QHeaderView)

from appcore.plugins import plugin_settings, plugins_folder

SPARQL_JSON = "application/sparql-results+json"

# Settings defaults; every value here can be changed in the panel and is remembered.
DEFAULTS = {
    "endpoint": "https://apps.okn.us/geoconnex/sparql",
    "site": "06770500",
    "entity_uri": "https://geoconnex.us/usgs/monitoring-location/USGS-06770500",
    "limit": 100,
    "timeout_s": 60.0,
    "save_runs": True,
    "output_folder": "",
}

# Spin box bounds for the panel controls.
LIMIT_RANGE = (1, 100000)
TIMEOUT_RANGE_S = (1.0, 3600.0)

# Presets fill the query editor. {site}, {uri} and {limit} come from the panel.
PRESETS = {
    "Describe entity (outgoing links)":
        "SELECT ?property ?value WHERE {{\n  <{uri}> ?property ?value\n}}\nLIMIT {limit}",
    "Entities linking to entity (incoming links)":
        "SELECT ?subject ?property WHERE {{\n  ?subject ?property <{uri}>\n}}\nLIMIT {limit}",
    "Find entities by USGS site number":
        "SELECT ?subject ?property ?value WHERE {{\n  ?subject ?property ?value .\n"
        "  FILTER(isLiteral(?value) && (STR(?value) = \"{site}\" || STR(?value) = \"USGS-{site}\"))\n"
        "}}\nLIMIT {limit}",
    "Classes in graph (with counts)":
        "SELECT ?class (COUNT(?s) AS ?count) WHERE {{\n  ?s a ?class\n}}\n"
        "GROUP BY ?class ORDER BY DESC(?count)\nLIMIT {limit}",
    "Properties in graph (with counts)":
        "SELECT ?property (COUNT(*) AS ?count) WHERE {{\n  ?s ?property ?o\n}}\n"
        "GROUP BY ?property ORDER BY DESC(?count)\nLIMIT {limit}",
}
DESCRIBE_PRESET = "Describe entity (outgoing links)"

STYLE = """
QWidget { color: black; }
QFrame#panel { border: 1px solid #b0b0b0; border-radius: 8px; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget, QPushButton {
    border: 1px solid #b0b0b0; border-radius: 6px; padding: 3px; }
QPushButton:hover { background: #e8eef7; }
"""


# ======================================================================================================================
class QueryWorker(QThread):
    """Runs one SPARQL request off the UI thread."""
    done = pyqtSignal(dict)

    def __init__(self, endpoint, query, timeout_s):
        super().__init__()
        self.endpoint, self.query, self.timeout_s = endpoint, query, timeout_s

    def run(self):
        start = time.time()
        result = {"endpoint": self.endpoint, "query": self.query, "status": None,
                  "text": "", "error": None}
        try:
            r = requests.get(self.endpoint, params={"query": self.query},
                             headers={"Accept": SPARQL_JSON}, timeout=self.timeout_s)
            result["status"] = r.status_code
            result["text"] = r.text
            if not r.ok:
                result["error"] = f"HTTP {r.status_code}"
        except requests.RequestException as err:
            result["error"] = f"{type(err).__name__}: {err}"
        result["elapsed_s"] = round(time.time() - start, 2)
        self.done.emit(result)


# ======================================================================================================================
class GraphExplorer(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLE)
        self._worker = None
        self._settings = plugin_settings(__file__, DEFAULTS)
        self._build_ui()
        self._bind_settings()
        self._fill_from_preset()

    # ------------------------------------------------------------------------------------------------------------------
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # Left: controls
        panel = QFrame(objectName="panel")
        form = QFormLayout(panel)
        form.setLabelAlignment(Qt.AlignRight)

        self._edit_endpoint = QLineEdit()
        self._combo_preset = QComboBox()
        self._combo_preset.addItems(PRESETS.keys())
        self._edit_site = QLineEdit()
        self._edit_uri = QLineEdit()
        self._spin_limit = QSpinBox()
        self._spin_limit.setRange(*LIMIT_RANGE)
        self._spin_timeout = QDoubleSpinBox()
        self._spin_timeout.setRange(*TIMEOUT_RANGE_S)
        self._spin_timeout.setSuffix(" s")
        self._check_save = QCheckBox("Save each run")
        self._edit_output = QLineEdit()
        button_browse = QPushButton("Browse...")
        button_browse.clicked.connect(self._browse_output)
        self._button_fill = QPushButton("Load preset into editor")
        self._button_fill.clicked.connect(self._fill_from_preset)
        self._button_run = QPushButton("Run query")
        self._button_run.clicked.connect(self._run)
        self._label_status = QLabel("Ready.")
        self._label_status.setWordWrap(True)

        form.addRow("Endpoint", self._edit_endpoint)
        form.addRow("Preset", self._combo_preset)
        form.addRow("USGS site", self._edit_site)
        form.addRow("Entity URI", self._edit_uri)
        form.addRow("Limit", self._spin_limit)
        form.addRow("Timeout", self._spin_timeout)
        form.addRow("", self._button_fill)
        form.addRow("Output folder", self._edit_output)
        form.addRow("", button_browse)
        form.addRow("", self._check_save)
        form.addRow("", self._button_run)
        form.addRow("Status", self._label_status)

        # Right: query editor over results
        right = QSplitter(Qt.Vertical)
        self._edit_query = QPlainTextEdit()
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.cellDoubleClicked.connect(self._follow_uri)
        right.addWidget(self._edit_query)
        right.addWidget(self._table)
        right.setStretchFactor(1, 3)

        splitter.addWidget(panel)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

    def _bind_settings(self):
        s = self._settings
        s.bind(self._edit_endpoint, "endpoint")
        s.bind(self._edit_site, "site")
        s.bind(self._edit_uri, "entity_uri")
        s.bind(self._spin_limit, "limit")
        s.bind(self._spin_timeout, "timeout_s")
        s.bind(self._check_save, "save_runs")
        s.bind(self._edit_output, "output_folder")
        if not self._edit_output.text():
            self._edit_output.setText(os.path.dirname(os.path.normpath(plugins_folder())))

    # ------------------------------------------------------------------------------------------------------------------
    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Output folder", self._edit_output.text())
        if folder:
            self._edit_output.setText(folder)
            self._edit_output.editingFinished.emit()

    def _fill_from_preset(self):
        template = PRESETS[self._combo_preset.currentText()]
        self._edit_query.setPlainText(template.format(site=self._edit_site.text().strip(),
                                                      uri=self._edit_uri.text().strip(),
                                                      limit=self._spin_limit.value()))

    def _follow_uri(self, row, col):
        item = self._table.item(row, col)
        if item is None or item.data(Qt.UserRole) != "uri":
            return
        self.explore_uri(item.text())

    def explore_uri(self, uri):
        """Describe one entity: put its URI in Entity URI, load the describe preset, and run it."""
        self._edit_uri.setText(uri)
        self._edit_uri.editingFinished.emit()
        self._combo_preset.setCurrentText(DESCRIBE_PRESET)
        self._fill_from_preset()
        self._run()

    # ------------------------------------------------------------------------------------------------------------------
    def _run(self):
        if self._worker is not None and self._worker.isRunning():
            return
        query = self._edit_query.toPlainText().strip()
        endpoint = self._edit_endpoint.text().strip()
        if not query or not endpoint:
            self._label_status.setText("Enter an endpoint and a query.")
            return
        self._button_run.setEnabled(False)
        self._label_status.setText("Running...")
        self._run_started = datetime.now()
        self._worker = QueryWorker(endpoint, query, self._spin_timeout.value())
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, result):
        self._button_run.setEnabled(True)
        variables, rows, parse_error = _parse_sparql_json(result["text"])
        error = result["error"] or parse_error
        self._show_table(variables, rows)

        saved = ""
        if self._check_save.isChecked():
            saved = self._save_run(result, variables, rows, error)

        if error:
            self._label_status.setText(f"Failed after {result['elapsed_s']} s: {error}{saved}")
        else:
            self._label_status.setText(f"{len(rows)} rows in {result['elapsed_s']} s.{saved}")

    def _show_table(self, variables, rows):
        self._table.clear()
        self._table.setColumnCount(len(variables))
        self._table.setRowCount(len(rows))
        self._table.setHorizontalHeaderLabels(variables)
        for r, row in enumerate(rows):
            for c, var in enumerate(variables):
                cell = row.get(var, {})
                item = QTableWidgetItem(cell.get("value", ""))
                item.setData(Qt.UserRole, cell.get("type", ""))
                self._table.setItem(r, c, item)

    def _save_run(self, result, variables, rows, error):
        try:
            folder = os.path.join(self._edit_output.text().strip(),
                                  f"KnowledgeGraphQuery_{self._run_started:%Y%m%d_%H%M%S}")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "query.rq"), "w", encoding="utf-8") as f:
                f.write(result["query"])
            with open(os.path.join(folder, "results.json"), "w", encoding="utf-8") as f:
                f.write(result["text"])
            with open(os.path.join(folder, "results.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(variables)
                for row in rows:
                    writer.writerow([row.get(v, {}).get("value", "") for v in variables])
            with open(os.path.join(folder, "provenance.json"), "w", encoding="utf-8") as f:
                json.dump({"endpoint": result["endpoint"],
                           "query": result["query"],
                           "run_time": self._run_started.isoformat(timespec="seconds"),
                           "http_status": result["status"],
                           "rows": len(rows),
                           "elapsed_s": result["elapsed_s"],
                           "error": error}, f, indent=4)
            return f" Saved to {folder}"
        except OSError as err:
            return f" Could not save: {err}"


# ======================================================================================================================
def _parse_sparql_json(text):
    """(variables, rows, error) from a SPARQL 1.1 JSON results document."""
    try:
        data = json.loads(text)
        variables = data["head"]["vars"]
        rows = data["results"]["bindings"]
        return variables, rows, None
    except (ValueError, KeyError, TypeError):
        return [], [], "Response was not SPARQL JSON results" if text else None
