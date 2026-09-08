#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# sandbar_analyzer_tab.py
# Controller for the Sandbar Analyzer tab. Mirrors ROIAnalyzerTab's lifecycle
# (configure_filmstrip / wire_connections / populate_filmstrip /
# on_filmstrip_item_clicked / batched thumbnail loading) so it registers via
# the same _add_tab_safe(..., post=lambda t: (t.configure_filmstrip(),
# t.wire_connections())) call in ML_ImageProcessingDlg.
#
# v1 scope: internal-edge detection inside the single segmented ROI. Click a
# frame -> run the edge engine -> left panel shows the chosen display view,
# right panel shows the edge output, stats table docked beneath.
#
# Author: John Edward Stranzl, Jr.
# License: Apache License, Version 2.0

import os
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtWidgets import (QWidget, QFileDialog, QListWidgetItem, QMessageBox,
                             QTableWidgetItem)
from PyQt5.QtGui import QPixmap, QIcon, QImage

from GRIME_AI import PROJECT_ROOT
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor


class SandbarAnalyzerTab(QWidget):

    # EXPECTED UI WIDGETS (from sandbar_analyzer_tab.ui):
    #   lineEdit_sandbar_images_folder, pushButton_browse_sandbar_images_folder
    #   comboBox_operator
    #   spinBox_cannyLow, spinBox_cannyHigh, spinBox_sobelKsize,
    #   spinBox_blurKsize, spinBox_erodePx
    #   radioButton_view_segmented / _original / _overlay   (left view)
    #   radioButton_out_edges / _out_edges_on_image         (right view)
    #   label_image_left, label_image_right
    #   tableWidget_edgeStats
    #   listWidget_filmstrip
    #   pushButton_analyze, pushButton_extract_sandbar_features

    STAT_ROWS = [
        ("ROI pixels",      "roi_pixels",    "{:d}"),
        ("Edge pixels",     "edge_pixels",   "{:d}"),
        ("Edge density",    "edge_density",  "{:.4f}"),
        ("Mean gradient",   "mean_gradient", "{:.2f}"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        self._pairs = []
        self._pendingThumbnails = []
        self._batchSize = 10
        self._batchDelay = 50
        self._loadToken = 0

        self._current_idx = None        # selected filmstrip row
        self._analyzer = None           # last GRIME_AI_Sandbar_Analyzer
        self._raw_left = None           # QPixmap for left panel (unscaled)
        self._raw_right = None          # QPixmap for right panel (unscaled)

    # ------------------------------------------------------------------
    # lifecycle (called by _add_tab_safe post-hook, same as ROI tab)
    # ------------------------------------------------------------------
    def configure_filmstrip(self):
        """Force the filmstrip to exactly one thumbnail row: no wrapping,
        no vertical scroll — same as the ROI Analyzer filmstrip."""
        from PyQt5.QtWidgets import QFrame
        lw = self.listWidget_filmstrip

        lw.setWrapping(False)
        lw.setSpacing(0)
        lw.setContentsMargins(0, 0, 0, 0)
        lw.setViewportMargins(0, 0, 0, 0)
        lw.setFrameShape(QFrame.NoFrame)

        # One icon row plus the index label beneath it
        icon_h = lw.iconSize().height()
        lw.setFixedHeight(icon_h + 16)

    def wire_connections(self):
        self.pushButton_browse_sandbar_images_folder.clicked.connect(
            self.browse_sandbar_images_folder)
        self.lineEdit_sandbar_images_folder.editingFinished.connect(
            self._on_sandbar_images_folder_changed)

        self.listWidget_filmstrip.itemClicked.connect(self.on_filmstrip_item_clicked)

        self.pushButton_analyze.clicked.connect(self._reanalyze_current)
        self.pushButton_extract_sandbar_features.clicked.connect(
            self.extract_edge_features)

        # Live re-run when any edge parameter changes
        for w in (self.spinBox_cannyLow, self.spinBox_cannyHigh,
                  self.spinBox_sobelKsize, self.spinBox_blurKsize,
                  self.spinBox_erodePx):
            w.valueChanged.connect(self._reanalyze_current)
        self.comboBox_operator.currentIndexChanged.connect(self._reanalyze_current)

        # Left view selector only re-renders (no recompute)
        for rb in (self.radioButton_view_segmented, self.radioButton_view_original,
                   self.radioButton_view_overlay):
            rb.toggled.connect(self._refresh_left)

        # Right view selector re-renders; POC view builds the composite lazily
        for rb in (self.radioButton_out_edges, self.radioButton_out_edges_on_image,
                   self.radioButton_out_poc_composite):
            rb.toggled.connect(self._on_right_view_changed)

        # Clustering controls only affect the POC composite -> rebuild it
        self.comboBox_clusterMethod.currentIndexChanged.connect(self._rebuild_composite)
        self.spinBox_clusterCount.valueChanged.connect(self._rebuild_composite)
        for rb in (self.radioButton_outline_all, self.radioButton_outline_dominant):
            rb.toggled.connect(self._rebuild_composite)

        self._init_stats_table()

    def showEvent(self, event):
        """Restore the last-used folder every time the tab becomes visible —
        matches the other tabs (e.g. Segment Images), which restore in showEvent
        so the path repopulates on re-entry to the ML Image Processing dialog,
        not just on first construction.
        """
        super().showEvent(event)
        if self.lineEdit_sandbar_images_folder.text().strip():
            return   # already populated this session; don't clobber a live selection
        saved = JsonEditor().getValue("Sandbar_Analyzer_Images_Folder")
        if saved and os.path.isdir(saved):
            self.lineEdit_sandbar_images_folder.setText(saved)
            self._on_sandbar_images_folder_changed()

    # ------------------------------------------------------------------
    # folder / filmstrip
    # ------------------------------------------------------------------
    def browse_sandbar_images_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", str(PROJECT_ROOT))
        if not folder:
            return
        self.lineEdit_sandbar_images_folder.setText(folder)
        self._on_sandbar_images_folder_changed()

    def _on_sandbar_images_folder_changed(self):
        folder = self.lineEdit_sandbar_images_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            return
        JsonEditor().update_json_entry("Sandbar_Analyzer_Images_Folder", folder)

        from GRIME_AI.GRIME_AI_Sandbar_Analyzer import GRIME_AI_Sandbar_Analyzer
        self._pairs = GRIME_AI_Sandbar_Analyzer.generate_file_pairs(folder)
        if not self._pairs:
            QMessageBox.warning(self, "Sandbar Analyzer", "No image/mask pairs found.")
            return
        self.populate_filmstrip([orig for orig, _ in self._pairs])
        # Analyze the first pair immediately
        self._current_idx = 0
        self._reanalyze_current()

    def populate_filmstrip(self, image_paths):
        lw = self.listWidget_filmstrip
        lw.clear()
        self._loadToken += 1
        token = self._loadToken
        self._pendingThumbnails.clear()

        iconSize = lw.iconSize()
        label_h = 16
        for idx, path in enumerate(image_paths):
            item = QListWidgetItem(QIcon(), str(idx + 1))
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
            item.setToolTip(f"#{idx + 1} - {os.path.basename(path)}")
            item.setData(Qt.UserRole, idx)
            item.setSizeHint(QSize(iconSize.width(), iconSize.height() + label_h))
            lw.addItem(item)
            self._pendingThumbnails.append((item, path, token))

        if lw.count():
            lw.setCurrentRow(0)
        QTimer.singleShot(self._batchDelay, lambda: self._loadNextBatch(token))

    def _loadNextBatch(self, token):
        if token != self._loadToken:
            return
        lw = self.listWidget_filmstrip
        iconSize = lw.iconSize()
        for _ in range(min(self._batchSize, len(self._pendingThumbnails))):
            item, path, _ = self._pendingThumbnails.pop(0)
            if not os.path.exists(path):
                continue
            pix = QPixmap(path)
            if pix.isNull():
                continue
            item.setIcon(QIcon(pix.scaled(iconSize, Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)))
        if self._pendingThumbnails:
            QTimer.singleShot(self._batchDelay, lambda: self._loadNextBatch(token))

    def on_filmstrip_item_clicked(self, item: QListWidgetItem):
        self._current_idx = item.data(Qt.UserRole)
        self._reanalyze_current()

    # ------------------------------------------------------------------
    # analysis
    # ------------------------------------------------------------------
    def _edge_kwargs(self):
        return dict(
            operator=self.comboBox_operator.currentText().lower(),
            canny_low=self.spinBox_cannyLow.value(),
            canny_high=self.spinBox_cannyHigh.value(),
            sobel_ksize=self.spinBox_sobelKsize.value() | 1,   # force odd
            blur_ksize=self.spinBox_blurKsize.value(),
            erode_px=self.spinBox_erodePx.value(),
        )

    def _reanalyze_current(self):
        if self._current_idx is None or not self._pairs:
            return
        orig_path, mask_path = self._pairs[self._current_idx]

        from GRIME_AI.GRIME_AI_Sandbar_Analyzer import GRIME_AI_Sandbar_Analyzer
        try:
            analyzer = GRIME_AI_Sandbar_Analyzer(orig_path, mask_path)
            analyzer.run(**self._edge_kwargs())
        except Exception as err:
            QMessageBox.warning(self, "Sandbar Analyzer", f"Analysis failed:\n{err}")
            return

        self._analyzer = analyzer
        self._build_pixmaps()
        # If the POC composite is the active right view, it must be rebuilt now —
        # the edge recompute invalidated the cached composite (set to None in
        # _build_pixmaps). Ensure it exists before any refresh reads it.
        if self.radioButton_out_poc_composite.isChecked():
            self._ensure_composite()
        self._refresh_left()
        self._refresh_right()
        self._fill_stats_table(analyzer.stats)

    def _build_pixmaps(self):
        """Prepare unscaled QPixmaps for both panels from the analyzer output."""
        a = self._analyzer
        # Left panel candidates
        self._left_views = {
            'segmented': self._composite_segmented(a),
            'original':  cv2.cvtColor(a.image, cv2.COLOR_BGR2RGB),
            'overlay':   self._mask_overlay(a),
        }
        # Right panel candidates. POC composite is built lazily (it clusters,
        # which is heavier than edges) only when its view is selected.
        edges_rgb = cv2.cvtColor(a.edges, cv2.COLOR_GRAY2RGB)   # white edges on black
        self._right_views = {
            'edges':          edges_rgb,
            'edges_on_image': a.edge_overlay_rgb,
            'poc':            None,   # filled by _ensure_composite()
        }

    @staticmethod
    def _composite_segmented(a):
        """Segmented region over white (matches ROI tab's composite look)."""
        rgb = cv2.cvtColor(a.image, cv2.COLOR_BGR2RGB).copy()
        white = np.full_like(rgb, 255)
        m = a.mask_bin > 0
        out = white.copy()
        out[m] = rgb[m]
        return out

    @staticmethod
    def _mask_overlay(a):
        """Original with the ROI tinted (transparent mask overlay)."""
        rgb = cv2.cvtColor(a.image, cv2.COLOR_BGR2RGB).copy()
        tint = rgb.copy()
        m = a.mask_bin > 0
        tint[m] = (0.55 * tint[m] + 0.45 * np.array([50, 120, 255])).astype(np.uint8)
        return tint

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    @staticmethod
    def _np_to_pixmap(arr):
        c = arr if arr.flags['C_CONTIGUOUS'] else np.ascontiguousarray(arr)
        h, w = c.shape[:2]
        ch = c.shape[2]
        bpl = w * ch
        fmt = QImage.Format_RGBA8888 if ch == 4 else QImage.Format_RGB888
        return QPixmap.fromImage(QImage(c.data, w, h, bpl, fmt))

    def _selected_left_key(self):
        if self.radioButton_view_original.isChecked():
            return 'original'
        if self.radioButton_view_overlay.isChecked():
            return 'overlay'
        return 'segmented'

    def _selected_right_key(self):
        if self.radioButton_out_poc_composite.isChecked():
            return 'poc'
        if self.radioButton_out_edges_on_image.isChecked():
            return 'edges_on_image'
        return 'edges'

    def _composite_kwargs(self):
        method = self.comboBox_clusterMethod.currentText().lower().replace('-', '')
        method = {'kmeans': 'kmeans', 'gmm': 'gmm', 'meanshift': 'meanshift'}.get(
            method, 'kmeans')
        outline = 'dominant' if self.radioButton_outline_dominant.isChecked() else 'all'
        return dict(method=method, clusters=self.spinBox_clusterCount.value(),
                    outline_mode=outline)

    def _ensure_composite(self):
        """Build the POC composite for the current frame if not already built."""
        if self._analyzer is None:
            return
        if self._right_views.get('poc') is None:
            self._analyzer.build_poc_composite(**self._composite_kwargs())
            self._right_views['poc'] = self._analyzer.poc_composite_rgb

    def _on_right_view_changed(self, *args):
        if self._selected_right_key() == 'poc':
            self._ensure_composite()
        self._refresh_right()

    def _rebuild_composite(self, *args):
        """Clustering control changed: invalidate + rebuild only if POC is shown."""
        if self._analyzer is None:
            return
        self._right_views['poc'] = None
        if self.radioButton_out_poc_composite.isChecked():
            self._ensure_composite()
            self._refresh_right()

    def _refresh_left(self, *args):
        if not self._analyzer:
            return
        arr = self._left_views[self._selected_left_key()]
        self._raw_left = self._np_to_pixmap(arr)
        self._scale_into(self.label_image_left, self._raw_left)

    def _refresh_right(self, *args):
        if not self._analyzer:
            return
        key = self._selected_right_key()
        if key == 'poc':
            self._ensure_composite()
        arr = self._right_views.get(key)
        if arr is None:
            return
        self._raw_right = self._np_to_pixmap(arr)
        self._scale_into(self.label_image_right, self._raw_right)

    @staticmethod
    def _scale_into(label, pix):
        if pix is None:
            return
        w, h = label.width(), label.height()
        if w > 1 and h > 1:
            label.setPixmap(pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            label.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_into(self.label_image_left, self._raw_left)
        self._scale_into(self.label_image_right, self._raw_right)

    # ------------------------------------------------------------------
    # stats table
    # ------------------------------------------------------------------
    def _init_stats_table(self):
        t = self.tableWidget_edgeStats
        t.setColumnCount(2)
        t.setRowCount(len(self.STAT_ROWS))
        t.setHorizontalHeaderLabels(["Metric", "Value"])
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
        for r, (label, _, _) in enumerate(self.STAT_ROWS):
            t.setItem(r, 0, QTableWidgetItem(label))
            t.setItem(r, 1, QTableWidgetItem(""))

    def _fill_stats_table(self, stats):
        t = self.tableWidget_edgeStats
        for r, (_, key, fmt) in enumerate(self.STAT_ROWS):
            val = stats.get(key, 0)
            t.item(r, 1).setText(fmt.format(val))

    # ------------------------------------------------------------------
    # feature export (all frames -> CSV)
    # ------------------------------------------------------------------
    def extract_edge_features(self):
        if not self._pairs:
            QMessageBox.warning(self, "Sandbar Analyzer", "No image/mask pairs loaded.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Edge Features", str(PROJECT_ROOT), "CSV Files (*.csv)")
        if not out_path:
            return

        from GRIME_AI.GRIME_AI_Sandbar_Analyzer import GRIME_AI_Sandbar_Analyzer
        import csv
        kwargs = self._edge_kwargs()
        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "mask", "operator"] +
                            [key for _, key, _ in self.STAT_ROWS])
            for orig_path, mask_path in self._pairs:
                try:
                    a = GRIME_AI_Sandbar_Analyzer(orig_path, mask_path)
                    a.run(**kwargs)
                    writer.writerow(
                        [os.path.basename(orig_path), os.path.basename(mask_path),
                         kwargs['operator']] +
                        [a.stats.get(key, "") for _, key, _ in self.STAT_ROWS])
                except Exception as err:
                    writer.writerow([os.path.basename(orig_path),
                                     os.path.basename(mask_path), f"ERROR: {err}"])
        QMessageBox.information(self, "Sandbar Analyzer",
                                f"Edge features written to:\n{out_path}")
