#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# image_motion_tracking.py
# EXPERIMENTAL plugin: measures camera rotation (tilt) and shift of every image relative to the first image,
# using features inside a user-drawn ROI. Report only; images are not modified. Nothing is saved to settings.
#
# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC

import os
import csv
import json
import math
import datetime
import traceback

import cv2
import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPolygon
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QGroupBox, QProgressBar, QFileDialog, QComboBox, QMessageBox, QSizePolicy,
    QWidget, QListWidget, QListWidgetItem, QListView, QTableWidget, QTableWidgetItem,
    QSplitter, QAbstractItemView
)

from appcore.App_QLabel import App_QLabel
from appcore.QLabel_drawing_modes import DrawingMode
from appcore.dialogs.color_segmentation.color_seg_roi_data import ROIShape

PLUGIN = {
    "title":       "Image Motion Tracking",
    "class":       "ImageMotionTrackingPlugin",
    "description": "Measure camera tilt and shift relative to the first image (experimental)",
    "surface":     "tools",
    "size":        [1200, 800],
    "api_version": 2,
}

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# ----------------------------------------------------------------------------------------------------------------------
# DISPLAY SIZES (pixels)
# ----------------------------------------------------------------------------------------------------------------------
LEFT_PANEL_WIDTH  = 290
THUMB_WIDTH       = 128
THUMB_HEIGHT      = 96
FILMSTRIP_MARGIN  = 36      # room for the scroll bar and item padding below the thumbnails
MIN_IMAGE_VIEW    = 200
DISPLAY_DECIMALS  = 3

# ----------------------------------------------------------------------------------------------------------------------
# TRACKING PARAMETERS: label, minimum, maximum, default, step, decimals (None = integer), tooltip.
# GUI only in this experimental version; not saved to the settings file.
# ----------------------------------------------------------------------------------------------------------------------
PARAM_SPEC = {
    "max_features": ("Max features", 100, 20000, 2000, 100, None,
                     "Maximum number of ORB keypoints detected inside the ROI per image."),
    "match_ratio": ("Match ratio", 0.50, 0.95, 0.75, 0.05, 2,
                    "Lowe ratio test. A match is kept only if its distance is below this fraction of the "
                    "second-best match. Lower is stricter."),
    "ransac_threshold_px": ("RANSAC threshold (px)", 0.5, 20.0, 3.0, 0.5, 1,
                            "Maximum distance, in pixels, for a matched point to count as an inlier."),
    "min_inliers": ("Min inlier features", 3, 1000, 20, 1, None,
                    "Frames with fewer inliers are reported with status 'low_inliers'."),
}

# A similarity transform (rotation, uniform scale, x/y shift) needs at least two point pairs.
MIN_POINTS_SIMILARITY = 2
# The ratio test compares the best and second-best match.
KNN_NEIGHBORS = 2

STATUS_REFERENCE     = "reference"
STATUS_OK            = "ok"
STATUS_LOW_INLIERS   = "low_inliers"
STATUS_NO_MATCH      = "no_match"
STATUS_UNREADABLE    = "unreadable"
STATUS_SIZE_MISMATCH = "size_mismatch"

CSV_COLUMNS = [
    "frame_index",
    "file_name",
    "rotation_deg_clockwise",
    "roi_center_shift_x_px",
    "roi_center_shift_y_px",
    "scale",
    "matched_features",
    "inlier_features",
    "status",
]

TABLE_HEADERS = [
    "Frame",
    "File",
    "Rotation (deg, + = clockwise)",
    "ROI center shift X (px)",
    "ROI center shift Y (px)",
    "Scale",
    "Matched features",
    "Inlier features",
    "Status",
]


# ======================================================================================================================
# Helpers
# ======================================================================================================================
def _bgr_to_qimage(img_bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()


def _build_mask(rois, shape):
    """Union of all ROIs (image coordinates) as a uint8 mask of the given (height, width)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x, y, rw, rh) in rois["rectangles"]:
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(w - 1, int(round(x + rw)))
        y1 = min(h - 1, int(round(y + rh)))
        if x1 > x0 and y1 > y0:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=cv2.FILLED)
    for poly in rois["polygons"]:
        pts = np.array([[int(round(px)), int(round(py))] for px, py in poly], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


# ======================================================================================================================
# Thumbnail loader: keeps the GUI responsive while the filmstrip fills in
# ======================================================================================================================
class ThumbnailLoader(QThread):
    thumb_ready = pyqtSignal(int, QImage)

    def __init__(self, image_files):
        super().__init__()
        self.image_files = image_files
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        for idx, path in enumerate(self.image_files):
            if self._abort:
                return
            img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
            if img is None:
                continue
            h, w = img.shape[:2]
            s = min(THUMB_WIDTH / w, THUMB_HEIGHT / h)
            thumb = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
            self.thumb_ready.emit(idx, _bgr_to_qimage(thumb))


# ======================================================================================================================
# Tracking worker
# ======================================================================================================================
class MotionTrackingWorker(QThread):
    """
    For every image, estimates the similarity transform (rotation, scale, shift) that maps the first image onto it,
    using ORB features inside the ROI mask and RANSAC. The shift is reported for the ROI center so it is independent
    of where the rotation is measured from.
    """
    progress  = pyqtSignal(int, int, str)   # current, total, file name
    row_ready = pyqtSignal(dict)
    completed = pyqtSignal(dict)
    error     = pyqtSignal(str)

    def __init__(self, image_files, mask, rois, config, output_dir, folder_name):
        super().__init__()
        self.image_files = image_files
        self.mask        = mask
        self.rois        = rois
        self.config      = config
        self.output_dir  = output_dir
        self.folder_name = folder_name
        self._abort      = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            self._process()
        except Exception as e:
            self.error.emit(f"Processing error: {e}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _row(idx, name, rotation, shift_x, shift_y, scale, matched, inliers, status):
        return {
            "frame_index":            idx,
            "file_name":              name,
            "rotation_deg_clockwise": rotation,
            "roi_center_shift_x_px":  shift_x,
            "roi_center_shift_y_px":  shift_y,
            "scale":                  scale,
            "matched_features":       matched,
            "inlier_features":        inliers,
            "status":                 status,
        }

    # ------------------------------------------------------------------------------------------------------------------
    def _process(self):
        cfg   = self.config
        files = self.image_files
        total = len(files)

        ref = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
        if ref is None:
            self.error.emit(f"Could not read the reference image:\n{files[0]}")
            return
        if self.mask.shape != ref.shape:
            self.error.emit("The ROI was drawn on an image whose size differs from the reference (first) image.")
            return

        orb = cv2.ORB_create(nfeatures=cfg["max_features"])
        kp_ref, des_ref = orb.detectAndCompute(ref, self.mask)
        if des_ref is None or len(kp_ref) < cfg["min_inliers"]:
            n = 0 if des_ref is None else len(kp_ref)
            self.error.emit(f"The reference image has only {n} features inside the ROI "
                            f"(minimum {cfg['min_inliers']}). Enlarge the ROI or include more textured, "
                            f"stable areas such as the horizon, banks, or fixed structures.")
            return
        ref_pts = np.float32([k.pt for k in kp_ref])

        m = cv2.moments(self.mask, binaryImage=True)
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        rows = []

        for idx, path in enumerate(files):
            if self._abort:
                self.error.emit("Processing aborted by user. No report was written.")
                return

            name = os.path.basename(path)
            self.progress.emit(idx + 1, total, name)

            if idx == 0:
                row = self._row(idx, name, 0.0, 0.0, 0.0, 1.0, len(kp_ref), len(kp_ref), STATUS_REFERENCE)
            else:
                row = self._track(idx, name, path, ref.shape, orb, matcher, des_ref, ref_pts, cx, cy)

            rows.append(row)
            self.row_ready.emit(row)

        self.completed.emit(self._write_outputs(rows, files[0], (cx, cy)))

    # ------------------------------------------------------------------------------------------------------------------
    def _track(self, idx, name, path, ref_shape, orb, matcher, des_ref, ref_pts, cx, cy):
        cfg = self.config

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return self._row(idx, name, None, None, None, None, 0, 0, STATUS_UNREADABLE)
        if img.shape != ref_shape:
            return self._row(idx, name, None, None, None, None, 0, 0, STATUS_SIZE_MISMATCH)

        kp, des = orb.detectAndCompute(img, self.mask)
        if des is None or len(kp) < MIN_POINTS_SIMILARITY:
            return self._row(idx, name, None, None, None, None, 0, 0, STATUS_NO_MATCH)

        pairs = matcher.knnMatch(des_ref, des, k=KNN_NEIGHBORS)
        good = [p[0] for p in pairs
                if len(p) == KNN_NEIGHBORS and p[0].distance < cfg["match_ratio"] * p[1].distance]
        if len(good) < MIN_POINTS_SIMILARITY:
            return self._row(idx, name, None, None, None, None, len(good), 0, STATUS_NO_MATCH)

        src = ref_pts[[g.queryIdx for g in good]]
        dst = np.float32([kp[g.trainIdx].pt for g in good])
        M, inlier_flags = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                      ransacReprojThreshold=cfg["ransac_threshold_px"])
        if M is None:
            return self._row(idx, name, None, None, None, None, len(good), 0, STATUS_NO_MATCH)

        # Image y axis points down, so a positive angle here is clockwise as viewed.
        rotation = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        scale    = math.hypot(M[0, 0], M[1, 0])
        new_cx   = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
        new_cy   = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
        inliers  = int(inlier_flags.sum()) if inlier_flags is not None else 0
        status   = STATUS_OK if inliers >= cfg["min_inliers"] else STATUS_LOW_INLIERS

        return self._row(idx, name, rotation, new_cx - cx, new_cy - cy, scale, len(good), inliers, status)

    # ------------------------------------------------------------------------------------------------------------------
    def _write_outputs(self, rows, reference_file, roi_center):
        os.makedirs(self.output_dir, exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{ts}_{self.folder_name}_image_motion"

        csv_path = os.path.join(self.output_dir, f"{base}.csv")
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        status_counts = {}
        for r in rows:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

        report = {
            "run_timestamp":   datetime.datetime.now().isoformat(),
            "input_folder":    os.path.dirname(reference_file),
            "reference_image": os.path.basename(reference_file),
            "images_found":    len(rows),
            "status_counts":   status_counts,
            "roi_image_coordinates": self.rois,
            "roi_center_px":   {"x": roi_center[0], "y": roi_center[1]},
            "config":          self.config,
            "conventions": {
                "rotation_deg_clockwise": "rotation of each image relative to the reference; positive = clockwise",
                "roi_center_shift_px":    "where the reference ROI center moved to in each image; "
                                          "+x = right, +y = down",
                "scale":                  "1.0 = same zoom as the reference",
            },
            "csv": csv_path,
        }
        report_path = os.path.join(self.output_dir, f"{base}_report.json")
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2)

        return {"csv_path": csv_path, "report_path": report_path, "status_counts": status_counts,
                "total": len(rows)}


# ======================================================================================================================
# Main dialog
# ======================================================================================================================
class ImageMotionTrackingDlg(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Motion Tracking (experimental)")
        self.resize(*PLUGIN["size"])

        self._folder        = ""
        self._image_files   = []
        self._rows          = []
        self._current_bgr   = None
        self._current_index = -1
        self._geom          = None    # (x_offset, y_offset, image_px_per_display_px) of the displayed pixmap
        self._roi_cache     = None    # (display ROIs as last set, same ROIs in image coords); avoids rounding drift
        self._worker        = None
        self._thumb_loader  = None
        self._param_widgets = {}

        self._build_ui()

    # ------------------------------------------------------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_center(), stretch=1)

    def _build_left_panel(self):
        panel = QWidget()
        panel.setFixedWidth(LEFT_PANEL_WIDTH)
        lv = QVBoxLayout(panel)
        lv.setContentsMargins(0, 0, 0, 0)

        # Folder
        grp_folder = QGroupBox("Images")
        fv = QVBoxLayout(grp_folder)
        btn_browse = QPushButton("Select Folder…")
        btn_browse.clicked.connect(self._browse_folder)
        self.lbl_folder = QLabel("No folder selected")
        self.lbl_folder.setWordWrap(True)
        self.lbl_reference = QLabel("Reference: first image")
        self.lbl_reference.setWordWrap(True)
        fv.addWidget(btn_browse)
        fv.addWidget(self.lbl_folder)
        fv.addWidget(self.lbl_reference)
        lv.addWidget(grp_folder)

        # ROI
        grp_roi = QGroupBox("ROI (stable part of the scene)")
        rv = QVBoxLayout(grp_roi)
        rf = QFormLayout()
        rf.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.cmb_shape = QComboBox()
        self.cmb_shape.addItem("Rectangle", ROIShape.RECTANGLE)
        self.cmb_shape.addItem("Polygon", ROIShape.POLYGON)
        self.cmb_shape.addItem("Freeform", ROIShape.FREEFORM)
        self.cmb_shape.currentIndexChanged.connect(self._on_shape_changed)
        rf.addRow("Shape", self.cmb_shape)
        rv.addLayout(rf)
        btn_clear = QPushButton("Clear ROIs")
        btn_clear.clicked.connect(self._clear_rois)
        rv.addWidget(btn_clear)
        help_lbl = QLabel("Rectangle: drag.  Polygon: click vertices, right-click to close.  "
                          "Freeform: drag to trace.  Multiple ROIs are combined.")
        help_lbl.setWordWrap(True)
        rv.addWidget(help_lbl)
        lv.addWidget(grp_roi)

        # Parameters
        grp_params = QGroupBox("Tracking Parameters")
        pf = QFormLayout(grp_params)
        pf.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        for key, (label, lo, hi, default, step, decimals, tip) in PARAM_SPEC.items():
            if decimals is None:
                w = QSpinBox()
            else:
                w = QDoubleSpinBox()
                w.setDecimals(decimals)
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setValue(default)
            w.setToolTip(tip)
            lbl = QLabel(label)
            lbl.setToolTip(tip)
            pf.addRow(lbl, w)
            self._param_widgets[key] = w
        lv.addWidget(grp_params)

        # Run
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run)
        self.btn_abort = QPushButton("Abort")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self._abort)
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_abort)
        lv.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        lv.addWidget(self.progress_bar)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        lv.addWidget(self.lbl_status)

        # Selected frame
        grp_frame = QGroupBox("Selected Image")
        sv = QVBoxLayout(grp_frame)
        self.lbl_frame_info = QLabel("")
        self.lbl_frame_info.setWordWrap(True)
        self.lbl_frame_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sv.addWidget(self.lbl_frame_info)
        lv.addWidget(grp_frame)

        lv.addStretch()
        return panel

    def _build_center(self):
        splitter = QSplitter(Qt.Vertical)

        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)

        self.lbl_image = App_QLabel(self)
        self.lbl_image.setAlignment(Qt.AlignCenter)   # App_QLabel's geometry assumes a centered pixmap
        self.lbl_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_image.setMinimumSize(MIN_IMAGE_VIEW, MIN_IMAGE_VIEW)
        self.lbl_image.setDrawingMode(DrawingMode.COLOR_SEGMENTATION)
        self.lbl_image.setROIShape(ROIShape.RECTANGLE)
        self.lbl_image.resized.connect(self._on_image_resized)
        tv.addWidget(self.lbl_image, stretch=1)

        self.filmstrip = QListWidget()
        self.filmstrip.setViewMode(QListView.IconMode)
        self.filmstrip.setFlow(QListView.LeftToRight)
        self.filmstrip.setWrapping(False)
        self.filmstrip.setMovement(QListView.Static)
        self.filmstrip.setUniformItemSizes(True)
        self.filmstrip.setIconSize(QSize(THUMB_WIDTH, THUMB_HEIGHT))
        self.filmstrip.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.filmstrip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.filmstrip.setFixedHeight(THUMB_HEIGHT + FILMSTRIP_MARGIN)
        self.filmstrip.currentRowChanged.connect(self._on_filmstrip_row)
        tv.addWidget(self.filmstrip)

        splitter.addWidget(top)

        self.table = QTableWidget(0, len(TABLE_HEADERS))
        self.table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.currentCellChanged.connect(self._on_table_row)
        splitter.addWidget(self.table)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        return splitter

    # ------------------------------------------------------------------------------------------------------------------
    # Folder and filmstrip
    # ------------------------------------------------------------------------------------------------------------------
    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", self._folder,
                                                  QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if not folder:
            return

        files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                       if os.path.splitext(f)[-1].lower() in IMAGE_EXTENSIONS)
        if not files:
            QMessageBox.warning(self, "No Images Found", "No supported images found in the selected folder.")
            return

        self._stop_thumbnail_loader()
        self._folder      = folder
        self._image_files = files
        self._rows        = []
        self._geom        = None
        self._clear_rois()
        self.table.setRowCount(0)
        self.lbl_status.setText("")

        self.lbl_folder.setText(f"{folder}   ({len(files)} images)")
        self.lbl_reference.setText(f"Reference: {os.path.basename(files[0])}")

        self.filmstrip.blockSignals(True)
        self.filmstrip.clear()
        for path in files:
            item = QListWidgetItem()
            item.setToolTip(os.path.basename(path))
            item.setSizeHint(QSize(THUMB_WIDTH, THUMB_HEIGHT))
            self.filmstrip.addItem(item)
        self.filmstrip.blockSignals(False)

        self._thumb_loader = ThumbnailLoader(files)
        self._thumb_loader.thumb_ready.connect(self._on_thumb_ready)
        self._thumb_loader.start()

        self.btn_run.setEnabled(True)
        self.filmstrip.setCurrentRow(0)

    def _stop_thumbnail_loader(self):
        if self._thumb_loader is not None:
            self._thumb_loader.thumb_ready.disconnect()
            self._thumb_loader.abort()
            self._thumb_loader.wait()
            self._thumb_loader = None

    def _on_thumb_ready(self, idx, qimg):
        item = self.filmstrip.item(idx)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(qimg)))

    def _on_filmstrip_row(self, row):
        if row < 0:
            return
        self._show_frame(row)
        if row < self.table.rowCount():
            self.table.blockSignals(True)
            self.table.selectRow(row)
            self.table.blockSignals(False)

    def _on_table_row(self, row, _col, _prev_row, _prev_col):
        if row >= 0:
            self.filmstrip.setCurrentRow(row)

    # ------------------------------------------------------------------------------------------------------------------
    # Image display and ROI coordinate mapping
    # ------------------------------------------------------------------------------------------------------------------
    def _show_frame(self, idx):
        img = cv2.imread(self._image_files[idx])
        if img is None:
            self.lbl_status.setText(f"Could not read {os.path.basename(self._image_files[idx])}")
            return
        self._current_bgr   = img
        self._current_index = idx
        self._set_pixmap(img)
        self._update_frame_info(idx)

    def _set_pixmap(self, img):
        # Keep drawn ROIs attached to the same image pixels when the display scale or offset changes.
        rois = self._current_image_rois() if self._geom is not None else None

        pm = QPixmap.fromImage(_bgr_to_qimage(img)).scaled(self.lbl_image.size(), Qt.KeepAspectRatio,
                                                           Qt.SmoothTransformation)
        self.lbl_image.setPixmap(pm)
        self.lbl_image.setOriginalImageShape(img.shape if img.ndim == 3 else (*img.shape, 1))

        x_off = (self.lbl_image.width() - pm.width()) // 2
        y_off = (self.lbl_image.height() - pm.height()) // 2
        self._geom = (x_off, y_off, img.shape[1] / pm.width())

        if rois is not None:
            self._set_display_rois(rois)

    def _on_image_resized(self):
        if self._current_bgr is not None:
            self._set_pixmap(self._current_bgr)

    def _rois_in_image_coords(self):
        x_off, y_off, s = self._geom
        rects = []
        for r in self.lbl_image.savedROIs:
            if r.width() <= 0 or r.height() <= 0:
                continue
            rect = ((r.x() - x_off) * s, (r.y() - y_off) * s, r.width() * s, r.height() * s)
            if rect not in rects:
                rects.append(rect)
        polys = [[((px - x_off) * s, (py - y_off) * s) for px, py in poly]
                 for poly in self.lbl_image.getColorSegPolygons()]
        return {"rectangles": rects, "polygons": polys}

    def _set_display_rois(self, rois):
        x_off, y_off, s = self._geom
        self.lbl_image.savedROIs = [
            QRect(int(round(x / s + x_off)), int(round(y / s + y_off)), int(round(w / s)), int(round(h / s)))
            for (x, y, w, h) in rois["rectangles"]
        ]
        self.lbl_image.savedPolygons = [
            QPolygon([QPoint(int(round(px / s + x_off)), int(round(py / s + y_off))) for px, py in poly])
            for poly in rois["polygons"]
        ]
        self._roi_cache = (self._display_roi_snapshot(), rois)
        self.lbl_image.update()

    def _display_roi_snapshot(self):
        return ([(r.x(), r.y(), r.width(), r.height()) for r in self.lbl_image.savedROIs],
                self.lbl_image.getColorSegPolygons())

    def _current_image_rois(self):
        """Image-coordinate ROIs; reuses the cached values when nothing was drawn since they were last mapped."""
        if self._roi_cache is not None and self._display_roi_snapshot() == self._roi_cache[0]:
            return self._roi_cache[1]
        return self._rois_in_image_coords()

    def _on_shape_changed(self, _index):
        self.lbl_image.setROIShape(self.cmb_shape.currentData())

    def _clear_rois(self):
        self.lbl_image.clearROIs()
        self.lbl_image.clearColorSegPolygons()
        self._roi_cache = None
        self.lbl_image.update()

    # ------------------------------------------------------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------------------------------------------------------
    def _config(self):
        return {key: w.value() for key, w in self._param_widgets.items()}

    def _run(self):
        if not self._image_files:
            QMessageBox.warning(self, "No Images", "Please select a folder first.")
            return

        rois = self._current_image_rois() if self._geom is not None else None
        if not rois or (not rois["rectangles"] and not rois["polygons"]):
            QMessageBox.warning(self, "No ROI", "Draw an ROI on a stable part of the scene "
                                                "(horizon, banks, fixed structures).")
            return

        ref = cv2.imread(self._image_files[0], cv2.IMREAD_GRAYSCALE)
        if ref is None:
            QMessageBox.warning(self, "Reference Unreadable", f"Could not read {self._image_files[0]}")
            return
        mask = _build_mask(rois, ref.shape)
        if not mask.any():
            QMessageBox.warning(self, "Empty ROI", "The ROI does not cover any part of the reference image.")
            return

        self._rows = []
        self.table.setRowCount(0)
        self.btn_run.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.progress_bar.setRange(0, len(self._image_files))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        output_dir  = os.path.join(self._folder, "Image Motion")
        folder_name = os.path.basename(self._folder)
        self._worker = MotionTrackingWorker(self._image_files, mask, rois, self._config(), output_dir, folder_name)
        self._worker.progress.connect(self._on_progress)
        self._worker.row_ready.connect(self._on_row)
        self._worker.completed.connect(self._on_completed)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _abort(self):
        if self._worker is not None:
            self._worker.abort()
        self.btn_abort.setEnabled(False)

    # ------------------------------------------------------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------------------------------------------------------
    def _on_progress(self, current, total, name):
        self.progress_bar.setValue(current)
        self.lbl_status.setText(f"Tracking {current}/{total}: {name}")

    def _on_row(self, row):
        self._rows.append(row)
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, key in enumerate(CSV_COLUMNS):
            v = row[key]
            if v is None:
                text = ""
            elif isinstance(v, float):
                text = f"{v:.{DISPLAY_DECIMALS}f}"
            else:
                text = str(v)
            item = QTableWidgetItem(text)
            if not isinstance(v, str):
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(r, c, item)
        if row["frame_index"] == self._current_index:
            self._update_frame_info(self._current_index)

    def _on_completed(self, info):
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.table.resizeColumnsToContents()
        counts = ", ".join(f"{k}: {v}" for k, v in info["status_counts"].items())
        self.lbl_status.setText(f"Done. {info['total']} images ({counts}).\nCSV: {info['csv_path']}")

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("Stopped.")
        QMessageBox.critical(self, "Image Motion Tracking", msg)

    def _update_frame_info(self, idx):
        name = os.path.basename(self._image_files[idx])
        if idx >= len(self._rows):
            self.lbl_frame_info.setText(f"{name}\nNot tracked yet.")
            return
        row = self._rows[idx]

        def fmt(v):
            return "" if v is None else f"{v:.{DISPLAY_DECIMALS}f}"

        self.lbl_frame_info.setText(
            f"{name}\n"
            f"Rotation: {fmt(row['rotation_deg_clockwise'])} deg (+ = clockwise)\n"
            f"ROI center shift: x {fmt(row['roi_center_shift_x_px'])} px, y {fmt(row['roi_center_shift_y_px'])} px\n"
            f"Scale: {fmt(row['scale'])}\n"
            f"Inliers: {row['inlier_features']} of {row['matched_features']} matches\n"
            f"Status: {row['status']}"
        )

    # ------------------------------------------------------------------------------------------------------------------
    def closeEvent(self, event):
        self._stop_thumbnail_loader()
        if self._worker is not None and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait()
        super().closeEvent(event)


# ======================================================================================================================
# Plugin wrapper
# ======================================================================================================================
class ImageMotionTrackingPlugin(QtWidgets.QWidget):
    """Hosts the dialog inside a plugin window, the same way as the Temporal Averaging plugin."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._dialog = ImageMotionTrackingDlg(self)
        self._dialog.setWindowFlags(Qt.Widget)
        layout.addWidget(self._dialog)

    @property
    def dialog(self):
        return self._dialog


# ======================================================================================================================
# Standalone use: python image_motion_tracking.py
# ======================================================================================================================
def main(argv=None):
    import sys
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    dialog = ImageMotionTrackingDlg()
    dialog.show()
    return app.exec_()


if __name__ == "__main__":
    import sys
    sys.exit(main())
