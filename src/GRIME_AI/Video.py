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
from PIL import Image

# ----------------------------------------------------------------------------------------------------------------------
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

# Application Classes
# ----------------------------------------------------------------------------------------------------------------------
from appcore.QProgressWheel import QProgressWheel
from appcore.App_Utils import App_Utils
from appcore.Color import Color
from appcore.App_QMessageBox import App_QMessageBox

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


def get_canvas_size(filenames):
    """
    Reads only the image headers and returns (width, height) of the canvas:
    the largest width and largest height found across all images.
    """
    canvas_w = 0
    canvas_h = 0
    for fname in filenames:
        with Image.open(fname) as im:
            w, h = im.size
        canvas_w = max(canvas_w, w)
        canvas_h = max(canvas_h, h)
    return canvas_w, canvas_h


def center_on_canvas(frame, canvas_w, canvas_h, border_color):
    """
    Centers frame on a canvas of (canvas_w, canvas_h), filling the border with border_color.
    Frames already at canvas size are returned unchanged.
    """
    h, w = frame.shape[:2]
    if (w, h) == (canvas_w, canvas_h):
        return frame
    top = (canvas_h - h) // 2
    left = (canvas_w - w) // 2
    bottom = canvas_h - h - top
    right = canvas_w - w - left
    return cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)

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

    def __init__(self, filenames, output_path, duration=0.25, canvas_size=None, border_color=(0, 0, 0)):
        super().__init__()
        self.filenames    = filenames
        self.output_path  = output_path
        self.duration     = duration
        self.canvas_size  = canvas_size      # (width, height); None = no centering
        self.border_color = border_color     # RGB

    def run(self):
        try:
            writer = iio.get_writer(self.output_path, mode="I", duration=self.duration)
            total = len(self.filenames)

            # Phase-1: append each frame
            for idx, fname in enumerate(self.filenames, start=1):
                frame = iio.imread(fname)
                if self.canvas_size is not None:
                    frame = center_on_canvas(frame, self.canvas_size[0], self.canvas_size[1], self.border_color)
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
# =====     =====     =====     =====     =====    class Video    =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class Video:
    def __init__(self):
        self.className = "Video"

        from appcore.Save_Utils import Save_Utils
        self.myApp_save_utils = Save_Utils()

    # ======================================================================================================================
    #
    # ======================================================================================================================
    def createVideo(self, rootFolder, border_color=(0, 0, 0)):
        """
        border_color: RGB fill for the border around images smaller than the canvas.
        """

        # Guard: a source image folder is required (mirrors createGIF).
        if not rootFolder or not os.path.isdir(rootFolder):
            msgBox = App_QMessageBox('Image Folder Error',
                                          'Please specify a valid image folder!',
                                     buttons=QMessageBox.Close)
            msgBox.displayMsgBox()
            return

        out = None

        myGRIMe_Color = Color()

        filePath = self.myApp_save_utils.create_video_folder(rootFolder)

        # ONLY LOOK FOR FILES WITH THE FOLLOWING EXTENSIONS
        extensions = ('.jpg', '.jpeg', '.png')

        myApp_utils = App_Utils()
        imageCount = myApp_utils.get_image_count(rootFolder, extensions)

        # CANVAS = LARGEST WIDTH x LARGEST HEIGHT ACROSS ALL IMAGES. SMALLER IMAGES ARE CENTERED ON IT SO THAT
        # cv2.VideoWriter DOES NOT SILENTLY DROP FRAMES WHOSE SIZE DIFFERS FROM THE SIZE IT WAS OPENED WITH.
        imageFiles = [os.path.join(rootFolder, f) for f in sorted(os.listdir(rootFolder))
                      if os.path.splitext(f)[-1].lower() in extensions]
        canvas_w, canvas_h = get_canvas_size(imageFiles)

        # IMAGES ARE CONVERTED TO BGR BEFORE WRITING, SO THE BORDER COLOR IS TOO
        border_color_bgr = tuple(reversed(border_color))

        progressBar = QProgressWheel(0, imageCount)
        progressBar.show()

        for imageIndex, file in enumerate(sorted(os.listdir(rootFolder))):
            ext = os.path.splitext(file)[-1].lower()

            if ext in extensions:
                progressBar.setValue(imageIndex)

                img = myGRIMe_Color.loadColorImage(os.path.join(rootFolder, file))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = center_on_canvas(img, canvas_w, canvas_h, border_color_bgr)

                if out == None:
                    videoFile = 'Original_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.avi'
                    out = cv2.VideoWriter(filePath + '/' + videoFile, cv2.VideoWriter_fourcc(*'mp4v'), 15, (canvas_w, canvas_h))

                out.write(img)

        out.release()

        progressBar.close()
        del progressBar


    # ======================================================================================================================
    #
    # ======================================================================================================================
    def createGIF(self, rootFolder, border_color=(0, 0, 0)):
        """
        border_color: RGB fill for the border around images smaller than the canvas.

        Scans rootFolder for .jpg/.jpeg/.png images, writes a GIF in a QThread,
        and drives a two-phase QProgressWheel:
          • Phase-1 (0 → 100 - final_steps)% by actual frames written
          • Phase-2 (100 - final_steps → 100)% by an estimated finalization timer
        """
        if len(rootFolder) == 0:
            msgBox = App_QMessageBox('Image Folder Error', 'Please specify an image folder!', buttons=QMessageBox.Close)
            response = msgBox.displayMsgBox()
            return

        # 1) Gather source files
        utils = App_Utils()
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
        out_dir = self.myApp_save_utils.create_gif_folder(rootFolder)
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
        canvas_size = get_canvas_size(filenames)
        self._gif_worker = gw = GIFWriterWorker(filenames, gif_file, duration=0.25,
                                                canvas_size=canvas_size, border_color=border_color)
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
