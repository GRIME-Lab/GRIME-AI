#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# TemporalAveragingDlg.py
# Dialog for configuring and running temporal image averaging.
#
# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC

import os
import json
import datetime
import traceback

import cv2
import numpy as np

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QProgressBar, QFileDialog, QComboBox,
    QTimeEdit, QDateEdit, QMessageBox, QTextEdit, QSizePolicy,
    QScrollArea, QWidget, QTabWidget, QApplication
)
from PyQt5.QtCore import QTime, QDate

# ---------------------------------------------------------------------------
# Worker thread — keeps the GUI responsive during processing
# ---------------------------------------------------------------------------
class TemporalAveragingWorker(QThread):
    """
    Runs the full temporal averaging pipeline in a background thread.
    Emits progress updates and the final results back to the dialog.
    """

    progress        = pyqtSignal(int, int, str)   # current, total, message
    finished        = pyqtSignal(dict)             # results dict
    error           = pyqtSignal(str)              # error message

    # -----------------------------------------------------------------------
    def __init__(self, image_files, config, output_dir, folder_name):
        super().__init__()
        self.image_files = image_files
        self.config      = config
        self.output_dir  = output_dir
        self.folder_name = folder_name
        self._abort      = False

    # -----------------------------------------------------------------------
    def abort(self):
        self._abort = True

    # -----------------------------------------------------------------------
    def run(self):
        try:
            self._process()
        except Exception as e:
            self.error.emit(f"Processing error: {e}\n{traceback.format_exc()}")

    # -----------------------------------------------------------------------
    def _process(self):
        cfg            = self.config
        image_files    = self.image_files
        total          = len(image_files)

        # Accumulators
        acc_mean  = None   # float64 running sum for mean
        acc_sq    = None   # float64 running sum of squares (for std dev)
        stack_med = []     # list of float32 frames for median (if enabled)
        acc_min   = None   # float32 running minimum
        acc_max   = None   # float32 running maximum

        loaded          = 0
        skipped_blur    = 0
        skipped_expose  = 0
        skipped_outlier = 0
        skipped_time    = 0
        skipped_date    = 0
        skipped_read    = 0
        included_files  = []
        skip_log        = []

        ref_shape = None  # (H, W, C) of the first successfully loaded frame

        for idx, path in enumerate(image_files):
            if self._abort:
                self.error.emit("Processing aborted by user.")
                return

            self.progress.emit(idx + 1, total, os.path.basename(path))
            QApplication.processEvents()

            # ------------------------------------------------------------------
            # DATE RANGE FILTER — based on file modification time as fallback
            # ------------------------------------------------------------------
            if cfg["filter_dates"]:
                try:
                    mtime = os.path.getmtime(path)
                    file_date = datetime.date.fromtimestamp(mtime)
                    if not (cfg["date_start"] <= file_date <= cfg["date_end"]):
                        skipped_date += 1
                        skip_log.append({"file": path, "reason": "date range"})
                        continue
                except Exception:
                    pass

            # ------------------------------------------------------------------
            # READ IMAGE
            # ------------------------------------------------------------------
            img = cv2.imread(path)
            if img is None:
                skipped_read += 1
                skip_log.append({"file": path, "reason": "unreadable"})
                print(f"[WARN] TemporalAveraging: could not read {path}, skipping.")
                continue

            # ------------------------------------------------------------------
            # TIME-OF-DAY FILTER — cascading: filename -> EXIF -> mtime
            # ------------------------------------------------------------------
            if cfg["filter_time"]:
                file_time, time_source = _extract_datetime_from_file(path)
                if file_time is not None:
                    t_start = cfg["time_start"]
                    t_end   = cfg["time_end"]
                    if not (t_start <= file_time <= t_end):
                        skipped_time += 1
                        skip_log.append({"file": path, "reason": "time of day",
                                         "time_source": time_source,
                                         "file_time": str(file_time)})
                        continue
                else:
                    # Could not resolve time at all — include frame but log it
                    skip_log.append({"file": path, "reason": "time unresolvable (included anyway)",
                                     "time_source": "unknown"})

            # ------------------------------------------------------------------
            # RESIZE to reference shape if needed
            # ------------------------------------------------------------------
            if ref_shape is None:
                ref_shape = img.shape

            if img.shape != ref_shape:
                img = cv2.resize(img, (ref_shape[1], ref_shape[0]),
                                 interpolation=cv2.INTER_AREA)

            img_f = img.astype(np.float32)

            # ------------------------------------------------------------------
            # BLUR FILTER
            # ------------------------------------------------------------------
            if cfg["filter_blur"]:
                gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian < cfg["blur_threshold"]:
                    skipped_blur += 1
                    skip_log.append({"file": path, "reason": f"blurry (score={laplacian:.2f})"})
                    continue

            # ------------------------------------------------------------------
            # EXPOSURE FILTER
            # ------------------------------------------------------------------
            if cfg["filter_exposure"]:
                mean_brightness = img_f.mean()
                if mean_brightness < cfg["expose_min"] or mean_brightness > cfg["expose_max"]:
                    skipped_expose += 1
                    skip_log.append({"file": path,
                                     "reason": f"exposure ({mean_brightness:.1f})"})
                    continue

            # ------------------------------------------------------------------
            # ACCUMULATE
            # ------------------------------------------------------------------
            if acc_mean is None:
                acc_mean  = img_f.astype(np.float64)
                acc_sq    = (img_f ** 2).astype(np.float64)
                acc_min   = img_f.copy()
                acc_max   = img_f.copy()
            else:
                acc_mean += img_f.astype(np.float64)
                acc_sq   += (img_f.astype(np.float64) ** 2)
                acc_min   = np.minimum(acc_min, img_f)
                acc_max   = np.maximum(acc_max, img_f)

            if cfg["compute_median"]:
                stack_med.append(img_f)

            included_files.append(path)
            loaded += 1

        # -----------------------------------------------------------------------
        # OUTLIER REJECTION (post-pass, applied to mean stack)
        # -----------------------------------------------------------------------
        if cfg["filter_outliers"] and loaded > 2 and acc_mean is not None:
            mean_img = (acc_mean / loaded).astype(np.float32)
            std_img  = np.sqrt(np.maximum((acc_sq / loaded) - (mean_img.astype(np.float64) ** 2), 0)).astype(np.float32)

            # Reprocess including only non-outlier frames
            acc_mean2 = None
            acc_sq2   = None
            loaded2   = 0
            removed   = []

            for path in included_files:
                img = cv2.imread(path)
                if img is None:
                    continue
                if img.shape != ref_shape:
                    img = cv2.resize(img, (ref_shape[1], ref_shape[0]),
                                     interpolation=cv2.INTER_AREA)
                img_f = img.astype(np.float32)
                diff  = np.abs(img_f - mean_img)
                if np.mean(diff) > cfg["outlier_sigma"] * np.mean(std_img) + 1e-6:
                    skipped_outlier += 1
                    removed.append(path)
                    skip_log.append({"file": path, "reason": f"outlier (sigma={cfg['outlier_sigma']})"})
                    continue
                if acc_mean2 is None:
                    acc_mean2 = img_f.astype(np.float64)
                    acc_sq2   = (img_f ** 2).astype(np.float64)
                else:
                    acc_mean2 += img_f.astype(np.float64)
                    acc_sq2   += (img_f.astype(np.float64) ** 2)
                loaded2 += 1

            if loaded2 > 0:
                acc_mean = acc_mean2
                acc_sq   = acc_sq2
                loaded   = loaded2
                included_files = [f for f in included_files if f not in removed]

        # -----------------------------------------------------------------------
        # GUARD — nothing usable
        # -----------------------------------------------------------------------
        if loaded == 0 or acc_mean is None:
            self.error.emit("No images passed the quality filters. "
                            "Try relaxing the blur/exposure thresholds.")
            return

        # -----------------------------------------------------------------------
        # COMPUTE OUTPUT IMAGES
        # -----------------------------------------------------------------------
        mean_img = (acc_mean / loaded).clip(0, 255).astype(np.uint8)

        std_img = None
        if cfg["compute_stddev"] and loaded > 1:
            variance = (acc_sq / loaded) - ((acc_mean / loaded) ** 2)
            std_raw  = np.sqrt(np.maximum(variance, 0))
            std_img  = cv2.normalize(std_raw, None, 0, 255,
                                     cv2.NORM_MINMAX).astype(np.uint8)

        med_img = None
        if cfg["compute_median"] and len(stack_med) > 0:
            med_img = np.median(np.stack(stack_med, axis=0),
                                axis=0).clip(0, 255).astype(np.uint8)

        min_img = acc_min.clip(0, 255).astype(np.uint8) if acc_min is not None else None
        max_img = acc_max.clip(0, 255).astype(np.uint8) if acc_max is not None else None

        diff_img = None
        if cfg["compute_diff"] and min_img is not None and max_img is not None:
            diff_raw = acc_max - acc_min
            diff_img = diff_raw.clip(0, 255).astype(np.uint8)

        # -----------------------------------------------------------------------
        # SAVE OUTPUTS
        # -----------------------------------------------------------------------
        os.makedirs(self.output_dir, exist_ok=True)
        ts          = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base        = f"{ts}_{self.folder_name}"
        saved_files = []

        def _save(img_array, suffix):
            if img_array is None:
                return None
            path_out = os.path.join(self.output_dir, f"{base}_{suffix}.png")
            cv2.imwrite(path_out, img_array)
            saved_files.append(path_out)
            return path_out

        path_mean   = _save(mean_img,  "mean")
        path_median = _save(med_img,   "median")
        path_std    = _save(std_img,   "stddev")
        path_min    = _save(min_img,   "min")
        path_max    = _save(max_img,   "max")
        path_diff   = _save(diff_img,  "diff")

        # -----------------------------------------------------------------------
        # SIDECAR REPORT
        # -----------------------------------------------------------------------
        # Summarise how time was resolved across all skip_log entries
        time_sources = {"filename": 0, "exif": 0, "mtime": 0, "unknown": 0}
        for entry in skip_log:
            src = entry.get("time_source")
            if src in time_sources:
                time_sources[src] += 1

        report = {
            "run_timestamp":    datetime.datetime.now().isoformat(),
            "input_folder":     os.path.dirname(self.image_files[0]) if self.image_files else "",
            "images_found":     total,
            "images_included":  loaded,
            "skipped": {
                "unreadable":   skipped_read,
                "blurry":       skipped_blur,
                "exposure":     skipped_expose,
                "outlier":      skipped_outlier,
                "time_of_day":  skipped_time,
                "date_range":   skipped_date,
            },
            "time_resolution_sources": time_sources,
            "config":           {k: str(v) for k, v in cfg.items()},
            "outputs":          saved_files,
            "skip_log":         skip_log,
        }

        report_path = os.path.join(self.output_dir, f"{base}_report.json")
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2)

        self.finished.emit({
            "mean_img":    mean_img,
            "median_img":  med_img,
            "std_img":     std_img,
            "min_img":     min_img,
            "max_img":     max_img,
            "diff_img":    diff_img,
            "report":      report,
            "report_path": report_path,
            "path_mean":   path_mean,
            "loaded":      loaded,
            "total":       total,
        })


# ---------------------------------------------------------------------------
# Helper — cascading datetime extraction: filename -> EXIF -> mtime
# ---------------------------------------------------------------------------
def _extract_datetime_from_file(path):
    """
    Attempt to extract a datetime.time using a three-level cascade:

      1. Filename  — GRIME AI / USGS HIVIS conventions (YYYYMMDD_HHMMSS, _HHMMSS_)
      2. EXIF      — DateTimeOriginal / DateTimeDigitized / DateTime via Pillow
                     (avoids relying on camera-specific EXIFData class since
                      EXIF is inconsistent across camera models)
      3. mtime     — filesystem modification time (always available, least reliable)

    Returns (datetime.time, source_str) where source_str is one of:
      'filename', 'exif', 'mtime', or 'unknown'.
    """
    import re

    # ------------------------------------------------------------------
    # Level 1 - Filename patterns
    # ------------------------------------------------------------------
    name = os.path.splitext(os.path.basename(path))[0]

    # YYYYMMDD_HHMMSS  or  YYYYMMDD-HHMMSS
    m = re.search(r'\d{8}[_\-](\d{2})(\d{2})(\d{2})', name)
    if m:
        try:
            t = datetime.time(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return t, 'filename'
        except ValueError:
            pass

    # Standalone _HHMMSS_ block
    m = re.search(r'[_\-](\d{2})(\d{2})(\d{2})[_\-]', name)
    if m:
        try:
            t = datetime.time(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return t, 'filename'
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Level 2 - EXIF via Pillow
    # Priority order: DateTimeOriginal (36867) -> DateTimeDigitized (36868) -> DateTime (306)
    # ------------------------------------------------------------------
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as pil_img:
            exif_data = pil_img._getexif()
            if exif_data:
                for tag_id in (36867, 36868, 306):
                    raw = exif_data.get(tag_id)
                    if raw:
                        # Standard EXIF format: "YYYY:MM:DD HH:MM:SS"
                        m = re.search(r'(\d{2}):(\d{2}):(\d{2})$', raw.strip())
                        if m:
                            t = datetime.time(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                            return t, 'exif'
    except Exception:
        pass  # Pillow not available, EXIF malformed, or no EXIF block

    # ------------------------------------------------------------------
    # Level 3 - Filesystem modification time (always available)
    # ------------------------------------------------------------------
    try:
        mtime = os.path.getmtime(path)
        t = datetime.datetime.fromtimestamp(mtime).time()
        return t, 'mtime'
    except Exception:
        pass

    return None, 'unknown'


# ---------------------------------------------------------------------------
# Helper — convert a numpy BGR image to a QPixmap for display
# ---------------------------------------------------------------------------
def _bgr_to_pixmap(img_bgr):
    if img_bgr is None:
        return None
    rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    qimg  = QImage(rgb.data.tobytes(), w, h, w * c, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ===========================================================================
# Main Dialog
# ===========================================================================
class TemporalAveragingDlg(QDialog):
    """
    Full-featured Temporal Averaging dialog for GRIME AI.
    Presents configuration options, runs processing in a QThread,
    and displays all output images with a sidecar JSON report.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Temporal Averaging")
        self.setMinimumSize(900, 700)
        self.resize(1000, 760)

        self._worker       = None
        self._folder       = ""
        self._image_files  = []
        self._results      = {}

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI CONSTRUCTION
    # -----------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── Folder selector ─────────────────────────────────────────────────
        folder_row = QHBoxLayout()
        self.lbl_folder = QLabel("No folder selected")
        self.lbl_folder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(QLabel("Image Folder:"))
        folder_row.addWidget(self.lbl_folder)
        folder_row.addWidget(btn_browse)
        root.addLayout(folder_row)

        # ── Tabs: Settings | Results ─────────────────────────────────────────
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        self.tabs.addTab(self._build_settings_tab(), "Settings")
        self.tabs.addTab(self._build_results_tab(),  "Results")

        # ── Progress bar ─────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.lbl_progress = QLabel("")
        root.addWidget(self.progress_bar)
        root.addWidget(self.lbl_progress)

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_run    = QPushButton("Run")
        self.btn_run.setStyleSheet("QPushButton { background-color: steelblue; color: white; }")
        self.btn_run.setEnabled(False)
        self.btn_abort  = QPushButton("Abort")
        self.btn_abort.setEnabled(False)
        self.btn_close  = QPushButton("Close")
        self.btn_run.clicked.connect(self._run)
        self.btn_abort.clicked.connect(self._abort)
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_abort)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

    # -----------------------------------------------------------------------
    def _build_settings_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ── Output options ───────────────────────────────────────────────────
        grp_out = QGroupBox("Output Images")
        g = QGridLayout(grp_out)
        self.chk_mean   = QCheckBox("Mean image");           self.chk_mean.setChecked(True)
        self.chk_median = QCheckBox("Median image");         self.chk_median.setChecked(True)
        self.chk_stddev = QCheckBox("Std. deviation image"); self.chk_stddev.setChecked(True)
        self.chk_min    = QCheckBox("Minimum image");        self.chk_min.setChecked(True)
        self.chk_max    = QCheckBox("Maximum image");        self.chk_max.setChecked(True)
        self.chk_diff   = QCheckBox("Difference image (max − min)"); self.chk_diff.setChecked(True)
        g.addWidget(self.chk_mean,   0, 0)
        g.addWidget(self.chk_median, 0, 1)
        g.addWidget(self.chk_stddev, 0, 2)
        g.addWidget(self.chk_min,    1, 0)
        g.addWidget(self.chk_max,    1, 1)
        g.addWidget(self.chk_diff,   1, 2)
        layout.addWidget(grp_out)

        # ── Quality filters ──────────────────────────────────────────────────
        grp_qc = QGroupBox("Quality Control Filters")
        g2 = QGridLayout(grp_qc)

        # Blur
        self.chk_blur = QCheckBox("Skip blurry frames  (Laplacian variance <")
        self.chk_blur.setChecked(True)
        self.spin_blur = QDoubleSpinBox()
        self.spin_blur.setRange(0.1, 500.0)
        self.spin_blur.setValue(17.5)
        self.spin_blur.setDecimals(1)
        self.spin_blur.setSuffix(")")
        g2.addWidget(self.chk_blur,  0, 0)
        g2.addWidget(self.spin_blur, 0, 1)

        # Exposure
        self.chk_expose = QCheckBox("Skip by brightness  (mean px outside")
        self.chk_expose.setChecked(True)
        exp_row = QHBoxLayout()
        self.spin_exp_min = QDoubleSpinBox(); self.spin_exp_min.setRange(0, 255); self.spin_exp_min.setValue(30)
        self.spin_exp_max = QDoubleSpinBox(); self.spin_exp_max.setRange(0, 255); self.spin_exp_max.setValue(220)
        exp_row.addWidget(self.spin_exp_min)
        exp_row.addWidget(QLabel("–"))
        exp_row.addWidget(self.spin_exp_max)
        exp_row.addWidget(QLabel(")"))
        exp_widget = QWidget(); exp_widget.setLayout(exp_row)
        g2.addWidget(self.chk_expose,  1, 0)
        g2.addWidget(exp_widget,       1, 1)

        # Outliers
        self.chk_outlier = QCheckBox("Reject outlier frames  (> N σ from mean,  N =")
        self.chk_outlier.setChecked(True)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.5, 10.0)
        self.spin_sigma.setValue(2.5)
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setSuffix(")")
        g2.addWidget(self.chk_outlier,  2, 0)
        g2.addWidget(self.spin_sigma,   2, 1)

        layout.addWidget(grp_qc)

        # ── Temporal filters ─────────────────────────────────────────────────
        grp_time = QGroupBox("Temporal Filters")
        g3 = QGridLayout(grp_time)

        # Time of day
        self.chk_time = QCheckBox("Filter by time of day  (")
        self.chk_time.setChecked(False)
        self.time_start = QTimeEdit(QTime(10, 0))
        self.time_end   = QTimeEdit(QTime(14, 0))
        self.time_start.setDisplayFormat("HH:mm")
        self.time_end.setDisplayFormat("HH:mm")
        tod_row = QHBoxLayout()
        tod_row.addWidget(self.time_start)
        tod_row.addWidget(QLabel(" – "))
        tod_row.addWidget(self.time_end)
        tod_row.addWidget(QLabel(")"))
        tod_widget = QWidget(); tod_widget.setLayout(tod_row)
        g3.addWidget(self.chk_time,  0, 0)
        g3.addWidget(tod_widget,     0, 1)

        # Date range
        self.chk_dates = QCheckBox("Filter by date range  (")
        self.chk_dates.setChecked(False)
        today = QDate.currentDate()
        self.date_start = QDateEdit(today.addYears(-1))
        self.date_end   = QDateEdit(today)
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        self.date_end.setDisplayFormat("yyyy-MM-dd")
        self.date_start.setCalendarPopup(True)
        self.date_end.setCalendarPopup(True)
        dr_row = QHBoxLayout()
        dr_row.addWidget(self.date_start)
        dr_row.addWidget(QLabel(" – "))
        dr_row.addWidget(self.date_end)
        dr_row.addWidget(QLabel(")"))
        dr_widget = QWidget(); dr_widget.setLayout(dr_row)
        g3.addWidget(self.chk_dates, 1, 0)
        g3.addWidget(dr_widget,      1, 1)

        layout.addWidget(grp_time)
        layout.addStretch()
        return widget

    # -----------------------------------------------------------------------
    def _build_results_tab(self):
        widget  = QWidget()
        layout  = QVBoxLayout(widget)

        # Image selector
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("View:"))
        self.cmb_result = QComboBox()
        self.cmb_result.addItems(["Mean", "Median", "Std. Deviation", "Minimum", "Maximum", "Difference"])
        self.cmb_result.currentIndexChanged.connect(self._show_selected_result)
        sel_row.addWidget(self.cmb_result)
        sel_row.addStretch()
        layout.addLayout(sel_row)

        # Image preview
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.lbl_preview = QLabel("Run processing to see results here.")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumSize(600, 350)
        scroll.setWidget(self.lbl_preview)
        layout.addWidget(scroll, stretch=3)

        # Report text
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setMaximumHeight(160)
        self.txt_report.setPlaceholderText("Processing report will appear here…")
        layout.addWidget(self.txt_report, stretch=1)

        return widget

    # -----------------------------------------------------------------------
    # FOLDER BROWSING
    # -----------------------------------------------------------------------
    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if not folder:
            return

        self._folder = folder
        self.lbl_folder.setText(folder)

        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        self._image_files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[-1].lower() in extensions
        ])

        n = len(self._image_files)
        if n == 0:
            QMessageBox.warning(self, "No Images Found",
                                "No supported images found in the selected folder.")
            self.btn_run.setEnabled(False)
        else:
            self.lbl_folder.setText(f"{folder}   ({n} images found)")
            self.btn_run.setEnabled(True)

    # -----------------------------------------------------------------------
    # BUILD CONFIG DICT FROM UI
    # -----------------------------------------------------------------------
    def _build_config(self):
        qs  = self.date_start.date()
        qe  = self.date_end.date()
        ts  = self.time_start.time()
        te  = self.time_end.time()
        return {
            "compute_median":  self.chk_median.isChecked(),
            "compute_stddev":  self.chk_stddev.isChecked(),
            "compute_min":     self.chk_min.isChecked(),
            "compute_max":     self.chk_max.isChecked(),
            "compute_diff":    self.chk_diff.isChecked(),

            "filter_blur":     self.chk_blur.isChecked(),
            "blur_threshold":  self.spin_blur.value(),

            "filter_exposure": self.chk_expose.isChecked(),
            "expose_min":      self.spin_exp_min.value(),
            "expose_max":      self.spin_exp_max.value(),

            "filter_outliers": self.chk_outlier.isChecked(),
            "outlier_sigma":   self.spin_sigma.value(),

            "filter_time":     self.chk_time.isChecked(),
            "time_start":      datetime.time(ts.hour(), ts.minute()),
            "time_end":        datetime.time(te.hour(), te.minute()),

            "filter_dates":    self.chk_dates.isChecked(),
            "date_start":      datetime.date(qs.year(), qs.month(), qs.day()),
            "date_end":        datetime.date(qe.year(), qe.month(), qe.day()),
        }

    # -----------------------------------------------------------------------
    # RUN
    # -----------------------------------------------------------------------
    def _run(self):
        if not self._image_files:
            QMessageBox.warning(self, "No Images", "Please select a folder first.")
            return

        output_dir  = os.path.join(self._folder, "Temporal Averaging")
        folder_name = os.path.basename(self._folder)
        config      = self._build_config()

        self.btn_run.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self._image_files))
        self.progress_bar.setValue(0)

        self._worker = TemporalAveragingWorker(
            self._image_files, config, output_dir, folder_name
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    # -----------------------------------------------------------------------
    def _abort(self):
        if self._worker:
            self._worker.abort()
        self.btn_abort.setEnabled(False)

    # -----------------------------------------------------------------------
    # WORKER CALLBACKS
    # -----------------------------------------------------------------------
    def _on_progress(self, current, total, filename):
        self.progress_bar.setValue(current)
        self.lbl_progress.setText(f"Processing {current}/{total}: {filename}")

    # -----------------------------------------------------------------------
    def _on_finished(self, results):
        self._results = results
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.lbl_progress.setText(
            f"Done. {results['loaded']} / {results['total']} images included."
        )

        # Switch to Results tab and show mean by default
        self.tabs.setCurrentIndex(1)
        self._show_selected_result()

        # Populate report
        r = results["report"]
        skipped = r["skipped"]
        ts = r.get("time_resolution_sources", {})
        lines = [
            f"Run: {r['run_timestamp']}",
            f"Input folder: {r['input_folder']}",
            f"Images found: {r['images_found']}    Included: {r['images_included']}",
            f"Skipped — blurry: {skipped['blurry']}  |  exposure: {skipped['exposure']}  "
            f"|  outlier: {skipped['outlier']}  |  time: {skipped['time_of_day']}  "
            f"|  date: {skipped['date_range']}  |  unreadable: {skipped['unreadable']}",
            f"Time resolved via — filename: {ts.get('filename', 0)}  |  "
            f"EXIF: {ts.get('exif', 0)}  |  mtime: {ts.get('mtime', 0)}  |  "
            f"unknown: {ts.get('unknown', 0)}",
            f"Outputs saved to: {os.path.dirname(results.get('path_mean', ''))}",
            f"Report: {results['report_path']}",
        ]
        self.txt_report.setPlainText("\n".join(lines))

    # -----------------------------------------------------------------------
    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.lbl_progress.setText("Error during processing.")
        QMessageBox.critical(self, "Temporal Averaging Error", msg)

    # -----------------------------------------------------------------------
    # RESULT PREVIEW
    # -----------------------------------------------------------------------
    def _show_selected_result(self):
        if not self._results:
            return

        key_map = {
            0: "mean_img",
            1: "median_img",
            2: "std_img",
            3: "min_img",
            4: "max_img",
            5: "diff_img",
        }
        key = key_map.get(self.cmb_result.currentIndex(), "mean_img")
        img = self._results.get(key)

        if img is None:
            self.lbl_preview.setText("This output was not computed\n(check Settings tab).")
            return

        pix = _bgr_to_pixmap(img)
        if pix:
            scaled = pix.scaled(self.lbl_preview.size(),
                                 Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation)
            self.lbl_preview.setPixmap(scaled)
