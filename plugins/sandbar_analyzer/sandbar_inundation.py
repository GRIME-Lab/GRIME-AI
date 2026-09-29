#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# sandbar_inundation.py
#
# Inundation thresholds for a sandbar, from a stack of segmentation masks and
# the gage height of each image.
#
# A pixel inside the mask is exposed bar; outside it, at a higher stage, it is
# submerged. The stage at which a pixel changes from one to the other is its
# inundation threshold, which is a relative elevation in gage-height units. The
# map of those thresholds is a topographic surface of the bar as the camera
# sees it, with no survey and no photogrammetry.
#
# For each pixel the threshold is the stage that best separates its exposed
# frames from its submerged ones, rather than the last stage at which it
# happened to appear, so a few bad inference frames do not move it. The
# proportion of frames on the wrong side of that threshold is reported per
# pixel as a confidence layer.
#
# Pixels that never change state get no threshold and are labelled instead:
# never submerged (threshold above the observed range), never exposed (water or
# vegetation throughout), or too few frames to judge.

import os
import json
import datetime

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
                             QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
                             QPlainTextEdit, QProgressBar, QFileDialog, QMessageBox,
                             QComboBox)

MASK_SUFFIX = "_mask.png"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# A pixel seen in fewer frames than this has no meaningful transition to find.
DEFAULT_MIN_OBSERVATIONS = 8
# Working resolution: masks are downscaled by this factor before stacking, which
# keeps a long stack in memory. 1.0 processes at full resolution.
DEFAULT_SCALE = 0.5

LABEL_NO_THRESHOLD = {
    "never_submerged": 1,     # exposed in every frame: threshold is above the range seen
    "never_exposed":   2,     # submerged, vegetated or missed by inference throughout
    "too_few":         3,     # not enough frames covering this pixel
}


def _load_correlator():
    """
    The SensorImageCorrelator class, from the host package or from a copy beside
    this plugin when it runs standalone.
    """
    try:
        from appcore.SensorImageCorrelator import SensorImageCorrelator
    except ImportError:
        try:
            from SensorImageCorrelator import SensorImageCorrelator      # beside this file
        except ImportError as err:
            raise ImportError(
                "SensorImageCorrelator was not found. Running outside the application, "
                "place SensorImageCorrelator.py beside this plugin. Original error: "
                f"{err}") from err
    return SensorImageCorrelator


# ======================================================================================================================
# Worker
# ======================================================================================================================
class InundationWorker(QThread):
    """Computes the threshold map. One pass over the masks in stage order."""

    progress = pyqtSignal(int, int)          # done, total
    status   = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, pairs, output_dir, scale, min_observations):
        super().__init__()
        self._pairs = pairs                  # [(image_path, mask_path, stage), ...]
        self._output_dir = output_dir
        self._scale = scale
        self._min_observations = min_observations
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    # ------------------------------------------------------------------------------------------------------------------
    def run(self):
        try:
            self.finished.emit(self._compute())
        except Exception as err:
            import traceback
            traceback.print_exc()
            self.finished.emit({"error": f"{type(err).__name__}: {err}"})

    # ------------------------------------------------------------------------------------------------------------------
    def _compute(self) -> dict:
        # Ascending stage: every pixel should be exposed early and submerged late.
        pairs = sorted(self._pairs, key=lambda p: p[2])
        stages = [stage for _, _, stage in pairs]
        total = len(pairs)
        if total < self._min_observations:
            return {"error": f"Only {total} image(s) with both a mask and a stage; "
                             f"at least {self._min_observations} are needed."}

        shape = None
        exposed_count = submerged_count = None
        # running (submerged so far) - (exposed so far), and the best split seen
        running = best_score = best_index = None

        for i, (_, mask_path, stage) in enumerate(pairs):
            if self._cancelled:
                self.status.emit(f"Cancelled after {i} of {total} image(s).")
                break

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                self.status.emit(f"Unreadable mask skipped: {os.path.basename(mask_path)}")
                continue
            if self._scale != 1.0:
                mask = cv2.resize(mask, None, fx=self._scale, fy=self._scale,
                                  interpolation=cv2.INTER_NEAREST)
            exposed = mask > 0

            if shape is None:
                shape = exposed.shape
                exposed_count = np.zeros(shape, np.uint16)
                submerged_count = np.zeros(shape, np.uint16)
                running = np.zeros(shape, np.int16)
                best_score = np.zeros(shape, np.int16)
                best_index = np.zeros(shape, np.int16)
            elif exposed.shape != shape:
                self.status.emit(f"Size mismatch skipped: {os.path.basename(mask_path)}")
                continue

            exposed_count += exposed
            submerged_count += ~exposed
            # A split after this frame gains one when the frame is submerged (it
            # belongs above the threshold) and loses one when it is exposed.
            running += np.where(exposed, -1, 1).astype(np.int16)
            improved = running < best_score
            best_score = np.where(improved, running, best_score)
            best_index = np.where(improved, np.int16(i), best_index)

            self.progress.emit(i + 1, total)

        if shape is None:
            return {"error": "No masks could be read."}

        observations = exposed_count.astype(np.int32) + submerged_count.astype(np.int32)

        # The threshold sits between the frame that ends the exposed run and the
        # next one up in stage.
        stages_array = np.asarray(stages, np.float64)
        upper = np.minimum(best_index.astype(np.int32) + 1, len(stages_array) - 1)
        threshold = (stages_array[best_index.astype(np.int32)] + stages_array[upper]) / 2.0

        # Frames on the wrong side of the split, as a fraction: 0 is a clean
        # transition, 0.5 is a pixel whose state tells you nothing.
        errors = exposed_count + best_score.astype(np.int32)
        confidence = 1.0 - np.clip(errors / np.maximum(observations, 1), 0.0, 1.0)

        labels = np.zeros(shape, np.uint8)
        labels[observations < self._min_observations] = LABEL_NO_THRESHOLD["too_few"]
        labels[(submerged_count == 0) & (observations > 0)] = LABEL_NO_THRESHOLD["never_submerged"]
        labels[(exposed_count == 0) & (observations > 0)] = LABEL_NO_THRESHOLD["never_exposed"]
        valid = labels == 0
        threshold = np.where(valid, threshold, np.nan)

        return self._save(pairs, stages, threshold, observations, confidence, labels, valid)

    # ------------------------------------------------------------------------------------------------------------------
    def _save(self, pairs, stages, threshold, observations, confidence, labels, valid) -> dict:
        os.makedirs(self._output_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(self._output_dir, f"{stamp}_inundation")
        written = []

        def _write(path, array):
            cv2.imwrite(path, array)
            written.append(path)

        # Full precision for analysis, in gage-height units.
        np.save(base + "_threshold.npy", threshold.astype(np.float32))
        written.append(base + "_threshold.npy")

        # Readable version: low threshold (low on the bar) dark, high threshold
        # (high on the bar) bright, pixels without a threshold black.
        finite = threshold[valid]
        if finite.size:
            low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
            span = max(high - low, 1e-6)
            scaled = np.zeros(threshold.shape, np.uint8)
            scaled[valid] = np.clip((threshold[valid] - low) / span * 255.0, 0, 255).astype(np.uint8)
            _write(base + "_threshold.png", scaled)
            _write(base + "_threshold_color.png",
                   cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS) * valid[..., None])
        else:
            low = high = float("nan")

        _write(base + "_observations.png",
               cv2.normalize(observations.astype(np.float32), None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8))
        _write(base + "_confidence.png", (confidence * 255).astype(np.uint8))
        _write(base + "_labels.png", (labels * 60).astype(np.uint8))

        histogram = {}
        if finite.size:
            counts, edges = np.histogram(finite, bins=50)
            histogram = {"counts": counts.tolist(), "edges": edges.tolist()}
            with open(base + "_histogram.csv", "w", newline="", encoding="utf-8") as f:
                f.write("bin_low,bin_high,pixels\n")
                for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
                    f.write(f"{lo:.4f},{hi:.4f},{int(count)}\n")
            written.append(base + "_histogram.csv")

        report = {
            "run_timestamp":     datetime.datetime.now().isoformat(),
            "images_used":       len(pairs),
            "stage_min":         float(np.min(stages)),
            "stage_max":         float(np.max(stages)),
            "working_scale":     self._scale,
            "min_observations":  self._min_observations,
            "pixels_with_threshold":  int(valid.sum()),
            "pixels_never_submerged": int((labels == LABEL_NO_THRESHOLD["never_submerged"]).sum()),
            "pixels_never_exposed":   int((labels == LABEL_NO_THRESHOLD["never_exposed"]).sum()),
            "pixels_too_few":         int((labels == LABEL_NO_THRESHOLD["too_few"]).sum()),
            "threshold_min":     low,
            "threshold_max":     high,
            "histogram":         histogram,
            "outputs":           written,
        }
        with open(base + "_report.json", "w") as f:
            json.dump(report, f, indent=2)
        report["preview"] = base + "_threshold_color.png"
        return report


# ======================================================================================================================
# Tab
# ======================================================================================================================
class InundationThresholdTab(QWidget):
    """Folders in, threshold map out. The gage height of each image comes from
    the NWIS file in the data folder beside the images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._build_ui()

    # ------------------------------------------------------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        folders = QGroupBox("Folders")
        grid = QGridLayout(folders)
        self._edit_images = QLineEdit()
        self._edit_masks = QLineEdit()
        self._edit_output = QLineEdit()
        self._edit_sensor = QLineEdit()
        self._edit_sensor.setPlaceholderText("Found in the data folder beside the images")
        for row, (label, edit, picker) in enumerate([
                ("Images:", self._edit_images, self._browse_images),
                ("Masks:", self._edit_masks, lambda: self._browse_into(self._edit_masks, "Mask Folder")),
                ("Sensor file:", self._edit_sensor, self._browse_sensor),
                ("Output:", self._edit_output, lambda: self._browse_into(self._edit_output, "Output Folder"))]):
            button = QPushButton("Browse")
            button.clicked.connect(picker)
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(button, row, 2)
        grid.setColumnStretch(1, 1)

        options = QGroupBox("Options")
        opt = QHBoxLayout(options)
        self._spin_scale = QDoubleSpinBox()
        self._spin_scale.setRange(0.1, 1.0)
        self._spin_scale.setSingleStep(0.1)
        self._spin_scale.setValue(DEFAULT_SCALE)
        self._spin_scale.setToolTip("Masks are resized by this factor before stacking.")
        self._spin_min_obs = QSpinBox()
        self._spin_min_obs.setRange(2, 500)
        self._spin_min_obs.setValue(DEFAULT_MIN_OBSERVATIONS)
        self._spin_min_obs.setToolTip("Pixels seen in fewer frames than this get no threshold.")
        opt.addWidget(QLabel("Working scale:"));   opt.addWidget(self._spin_scale)
        opt.addWidget(QLabel("Minimum frames:"));  opt.addWidget(self._spin_min_obs)
        opt.addStretch(1)

        self._button_run = QPushButton("Compute Thresholds")
        self._button_run.clicked.connect(self._run)
        self._button_cancel = QPushButton("Cancel")
        self._button_cancel.setEnabled(False)
        self._button_cancel.clicked.connect(self._cancel)
        buttons = QHBoxLayout()
        buttons.addWidget(self._button_run)
        buttons.addWidget(self._button_cancel)
        buttons.addStretch(1)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._preview = QLabel("The threshold map appears here.")
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumSize(360, 270)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(140)

        layout.addWidget(folders)
        layout.addWidget(options)
        layout.addLayout(buttons)
        layout.addWidget(self._progress)
        layout.addWidget(self._preview, 1)
        layout.addWidget(self._log)

    # ------------------------------------------------------------------------------------------------------------------
    def _browse_into(self, edit, title):
        folder = QFileDialog.getExistingDirectory(self, title, edit.text() or "")
        if folder:
            edit.setText(folder)

    def _browse_images(self):
        self._browse_into(self._edit_images, "Image Folder")
        if self._edit_images.text() and not self._edit_output.text():
            self._edit_output.setText(os.path.join(self._edit_images.text(), "inundation"))

    def _browse_sensor(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select NWIS Sensor File", self._edit_images.text() or "",
            "NWIS sensor data (*.csv *.txt);;All files (*.*)")
        if path:
            self._edit_sensor.setText(path)

    # ------------------------------------------------------------------------------------------------------------------
    def _sensor_file(self, images_folder):
        """The chosen NWIS file, or the one in the data folder beside the images."""
        chosen = self._edit_sensor.text().strip()
        if chosen and os.path.isfile(chosen):
            return chosen
        data_folder = os.path.join(os.path.dirname(os.path.normpath(images_folder)), "data")
        if os.path.isdir(data_folder):
            files = sorted((os.path.join(data_folder, f) for f in os.listdir(data_folder)
                            if f.lower().endswith(".csv")),
                           key=os.path.getmtime, reverse=True)
            if files:
                return files[0]
        return ""

    def _pairs_with_stage(self, images_folder, masks_folder, sensor_file):
        """[(image, mask, stage)] for every image that has both a mask and a stage."""
        SensorImageCorrelator = _load_correlator()
        images = []
        masks = []
        for name in sorted(os.listdir(images_folder)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTENSIONS or stem.lower().endswith(("_mask", "_overlay")):
                continue
            mask = os.path.join(masks_folder, stem + MASK_SUFFIX)
            if os.path.exists(mask):
                images.append(os.path.join(images_folder, name))
                masks.append(mask)
        if not images:
            return [], "No image and mask pairs were found."

        sensor = SensorImageCorrelator().sensor_values_for_images(images, sensor_file)
        stage_columns = [c for c in sensor.columns if "00065" in str(c)]
        if not stage_columns:
            return [], ("The sensor file has no gage height (parameter 00065) column, "
                        "which is what the thresholds are measured in.")

        stages = sensor[stage_columns[0]]
        pairs = [(img, msk, float(stage))
                 for img, msk, stage in zip(images, masks, stages)
                 if stage == stage and stage is not None]     # drop unmatched frames
        return pairs, ""

    # ------------------------------------------------------------------------------------------------------------------
    def _run(self):
        images_folder = self._edit_images.text().strip()
        masks_folder = self._edit_masks.text().strip()
        if not os.path.isdir(images_folder) or not os.path.isdir(masks_folder):
            QMessageBox.warning(self, "Folders", "Select an image folder and a mask folder.")
            return

        output_folder = self._edit_output.text().strip() or os.path.join(images_folder, "inundation")
        sensor_file = self._sensor_file(images_folder)
        if not sensor_file:
            QMessageBox.warning(self, "Sensor Data",
                                "No NWIS file was found in the data folder beside the images. "
                                "Select one with Browse.")
            return

        self._log.clear()
        self._log.appendPlainText(f"Sensor file: {sensor_file}")
        try:
            pairs, problem = self._pairs_with_stage(images_folder, masks_folder, sensor_file)
        except Exception as err:
            QMessageBox.critical(self, "Sensor Data", f"Could not read the sensor file:\n{err}")
            return
        if problem:
            QMessageBox.warning(self, "Inundation Thresholds", problem)
            return

        self._log.appendPlainText(f"{len(pairs)} image(s) with a mask and a stage.")
        self._button_run.setEnabled(False)
        self._button_cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        self._worker = InundationWorker(pairs, output_folder,
                                        self._spin_scale.value(), self._spin_min_obs.value())
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._log.appendPlainText)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self._button_cancel.setEnabled(False)
            self._log.appendPlainText("Cancelling\u2026")
            self._worker.cancel()

    def _on_progress(self, done, total):
        self._progress.setMaximum(total)
        self._progress.setValue(done)

    def _on_finished(self, report):
        self._progress.setVisible(False)
        self._button_run.setEnabled(True)
        self._button_cancel.setEnabled(False)

        if report.get("error"):
            self._log.appendPlainText(f"Failed: {report['error']}")
            QMessageBox.warning(self, "Inundation Thresholds", report["error"])
            return

        self._log.appendPlainText(
            f"Stage range {report['stage_min']:.2f} to {report['stage_max']:.2f}; "
            f"{report['pixels_with_threshold']} pixel(s) with a threshold, "
            f"{report['pixels_never_submerged']} never submerged, "
            f"{report['pixels_never_exposed']} never exposed, "
            f"{report['pixels_too_few']} with too few frames.")
        for path in report["outputs"]:
            self._log.appendPlainText(f"Wrote: {path}")

        preview = report.get("preview")
        if preview and os.path.exists(preview):
            pixmap = QPixmap(preview)
            if not pixmap.isNull():
                self._preview.setPixmap(pixmap.scaled(self._preview.size(), Qt.KeepAspectRatio,
                                                      Qt.SmoothTransformation))
