#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os
import cv2
import datetime
import traceback

# VIDEO CREATION PACKAGES
# ----------------------------------------------------------------------------------------------------------------------
import imageio as iio

# ----------------------------------------------------------------------------------------------------------------------
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

# GRIME-AI Classes
# ----------------------------------------------------------------------------------------------------------------------
from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_Utils import GRIME_AI_Utils
from GRIME_AI.GRIME_AI_Color import GRIME_AI_Color
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox

# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     ===== HELPER FUNCTIONS =====     =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
def estimate_gif_write_time(filenames):
    """
    Rough Fermi estimate for total encode time (seconds):
      base_time + pixel_time * (width * height * num_frames)
    Returns a single float, never a tuple.
    """
    if not filenames:
        return 0.0
    first = iio.imread(filenames[0])
    h, w = first.shape[:2]
    n = len(filenames)
    base_time = 1.2       # fixed overhead (sec)
    pixel_time = 4.5e-7    # sec per pixel per frame
    return base_time + (w * h * pixel_time * n)

# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     ===== class ProgressWheelThread  =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class GIFWriterWorker(QThread):
    """
    Phase-1 worker: writes frames and emits `progress(frame_index)`,
    then finalizes the GIF and emits `finished()`.
    """
    progress = pyqtSignal(int)  # emits 1…num_frames
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, filenames, output_path, duration=0.25):
        super().__init__()
        self.filenames   = filenames
        self.output_path = output_path
        self.duration    = duration

    def run(self):
        try:
            writer = iio.get_writer(self.output_path, mode="I", duration=self.duration)
            total = len(self.filenames)

            # Phase-1: append each frame
            for idx, fname in enumerate(self.filenames, start=1):
                frame = iio.imread(fname)
                writer.append_data(frame)
                self.progress.emit(idx)
                QApplication.processEvents()

            # Phase-2: finalize (blocking)
            writer.close()
            self.finished.emit()

        except Exception:
            tb = traceback.format_exc()
            self.error.emit(tb)
            self.finished.emit()


# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====  class FinalizeEstimator   =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class FinalizeEstimator(QThread):
    """
    Phase-2 estimator: emits `tick(1…final_steps)` over estimated_time.
    """
    tick = pyqtSignal(int)

    def __init__(self, estimated_time, final_steps=10):
        super().__init__()
        self.estimated_time = max(0.1, estimated_time)
        self.final_steps    = final_steps

    def run(self):
        interval_ms = int(self.estimated_time * 1000 / self.final_steps)
        for i in range(1, self.final_steps + 1):
            self.msleep(interval_ms)
            self.tick.emit(i)


# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====    class GRIME_AI_Video    =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class GRIME_AI_Video:
    def __init__(self):
        self.className = "GRIME_AI_Video"

        from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
        self.myGRIMEAI_save_utils = GRIME_AI_Save_Utils()

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def createVideo(self, rootFolder):

        out = None

        myGRIMe_Color = GRIME_AI_Color()

        filePath = self.myGRIMEAI_save_utils.create_video_folder(rootFolder)

        # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
        extensions = ('.jpg', '.jpeg', '.png')

        myGRIMEAI_utils = GRIME_AI_Utils()
        imageCount = myGRIMEAI_utils.get_image_count(rootFolder, extensions)

        progressBar = QProgressWheel(0, imageCount)
        progressBar.show()

        for imageIndex, file in enumerate(sorted(os.listdir(rootFolder))):
            ext = os.path.splitext(file)[-1].lower()

            if ext in extensions:
                progressBar.setValue(imageIndex)

                img = myGRIMe_Color.loadColorImage(os.path.join(rootFolder, file))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                height, width, layers = img.shape

                # WE CAN'T OPEN THE VIDEO STREAM UNTIL WE KNOW THE SIZE OF ONE OF THE IMAGES WHICH ALSO ASSUMES THAT
                # ALL IMAGES ARE OF THE SAME SIZE.
                if out == None:
                    videoFile = 'Original_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.avi'
                    out = cv2.VideoWriter(filePath + '/' + videoFile, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

                out.write(img)

        out.release()

        progressBar.close()
        del progressBar


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def createGIF(self, rootFolder):
        """
        Scans rootFolder for .jpg/.jpeg/.png images, writes a GIF in a QThread,
        and drives a two-phase QProgressWheel:
          • Phase-1 (0 → 100 - final_steps)% by actual frames written
          • Phase-2 (100 - final_steps → 100)% by an estimated finalization timer
        """
        if len(rootFolder) == 0:
            msgBox = GRIME_AI_QMessageBox('Image Folder Error', 'Please specify an image folder!', buttons=QMessageBox.Close)
            response = msgBox.displayMsgBox()
            return

        # 1) Gather source files
        utils = GRIME_AI_Utils()
        _, files = utils.getFileList(rootFolder, ('.jpg', '.jpeg', '.png'))
        filenames = [
            os.path.join(rootFolder, f)
            for f in files
            if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png'}
        ]
        if not filenames:
            QMessageBox.information(self, "No Images", "No JPG/PNG found in folder.")
            return

        # 2) Prepare output path
        out_dir = self.myGRIMEAI_save_utils.create_gif_folder(rootFolder)
        gif_name = 'Original_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.gif'
        gif_file = os.path.join(out_dir, gif_name)

        # 3) Compute phases
        total_frames = len(filenames)
        total_estimate_sec = estimate_gif_write_time(filenames)  # float
        final_steps = 10
        frame_pct_max = 100 - final_steps

        # 4) Set up a determinate progress wheel (0…100)
        progressBar = QProgressWheel(0, 0)
        progressBar.setRange(0, 100)
        progressBar.show()

        # 5) Track completion of both phases
        done = {"writer": False, "estimate": False}

        def try_close():
            if done["writer"] and done["estimate"]:
                progressBar.setValue(100)
                progressBar.close()
                # drop references so GC can clean up
                self._gif_worker = None
                self._final_est = None

        # 6) Phase-1: GIFWriterWorker (writes frames + actual finalize())
        self._gif_worker = gw = GIFWriterWorker(filenames, gif_file, duration=0.25)
        # Map frame progress → 0…frame_pct_max
        gw.progress.connect(lambda i: progressBar.setValue(
            int(i / total_frames * frame_pct_max)
        ))
        gw.error.connect(lambda tb: QMessageBox.critical(self, "GIF Write Error", tb))

        def on_writer_finished():
            done["writer"] = True
            try_close()

        gw.finished.connect(on_writer_finished)
        gw.start()

        # 7) Phase-2: FinalizeEstimator (ticks last final_steps over estimated time)
        self._final_est = fe = FinalizeEstimator(total_estimate_sec, final_steps)
        fe.tick.connect(lambda t: progressBar.setValue(frame_pct_max + t))

        def on_estimate_finished():
            done["estimate"] = True
            try_close()

        fe.finished.connect(on_estimate_finished)
        fe.start()
