#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import sys
import os
import subprocess
import tempfile

# ======================================================================================================================
# FADE MODES
# ======================================================================================================================
FADE_NONE   = 'none'    # image pops up instantly, glow starts immediately
FADE_SMOOTH = 'smooth'  # window opacity fades from 0 → 1, then glow starts
FADE_SNOW   = 'snow'    # random pixel blocks revealed gradually, then glow starts

# ======================================================================================================================
# ENVIRONMENT VARIABLES used to pass settings to the splash subprocess.
# App-neutral names: the child runs as a standalone script and has no package
# context, so it cannot derive them from app_identity before reading them.
# ======================================================================================================================
ENV_FADE_MODE = 'APP_SPLASH_FADE_MODE'
ENV_DONE_FILE = 'APP_SPLASH_DONE_FILE'

# ======================================================================================================================
# Run as __main__ to display the splash screen.
# argv[1] = image_path
# argv[2] = version_str (may be empty string)
# fade_mode and done_file communicated via environment variables — nothing extra in argv
# so Qt never renders stray text on the splash image.
# ======================================================================================================================
if __name__ == '__main__':
    import math
    import random

    image_path  = sys.argv[1]
    version_str = sys.argv[2] if len(sys.argv) > 2 else ''

    fade_mode = os.environ.get(ENV_FADE_MODE, FADE_SMOOTH)
    done_file = os.environ.get(ENV_DONE_FILE, '')

    from PyQt5.QtWidgets import QApplication, QWidget
    from PyQt5.QtGui import QPixmap, QPainter, QFont, QColor, QPen
    from PyQt5.QtCore import Qt, QTimer, QRect

    app = QApplication([sys.argv[0]])   # only script name — Qt sees no extra args

    pixmap = QPixmap(image_path)

    if version_str:
        painter = QPainter(pixmap)
        painter.setFont(QFont('Arial', 8))
        painter.setPen(Qt.black)
        painter.drawText(475, 220, version_str)
        painter.end()

    BORDER = 16

    class SplashWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SplashScreen)
            w = pixmap.width()  + BORDER * 2
            h = pixmap.height() + BORDER * 2
            self.resize(w, h)
            screen = QApplication.primaryScreen().geometry()
            self.move((screen.width() - w) // 2, (screen.height() - h) // 2)

            self._phase     = 0.0
            self._opacity   = 0.0   # used for both smooth fade and border alpha during fade
            self._fade_done = False

            # ------------------------------------------------------------------
            # FADE_NONE — instant
            # ------------------------------------------------------------------
            if fade_mode == FADE_NONE:
                self._opacity   = 1.0
                self._fade_done = True

            # ------------------------------------------------------------------
            # FADE_SMOOTH — opacity 0 → 1
            # ------------------------------------------------------------------
            elif fade_mode == FADE_SMOOTH:
                self.setWindowOpacity(0.0)
                self._fade_timer = QTimer(self)
                self._fade_timer.timeout.connect(self._smooth_tick)
                self._fade_timer.start(20)

            # ------------------------------------------------------------------
            # FADE_SNOW — random pixel block reveal
            # ------------------------------------------------------------------
            elif fade_mode == FADE_SNOW:
                self._block_size = 4
                cols = (pixmap.width()  + self._block_size - 1) // self._block_size
                rows = (pixmap.height() + self._block_size - 1) // self._block_size
                self._all_blocks = [(c, r) for r in range(rows) for c in range(cols)]
                random.shuffle(self._all_blocks)
                self._revealed   = set()
                self._batch_size = max(1, len(self._all_blocks) // 80)
                self._fade_timer = QTimer(self)
                self._fade_timer.timeout.connect(self._snow_tick)
                self._fade_timer.start(40)

            # Glow timer — runs always so phase is already moving during fade-in (no color jump)
            self._glow_timer = QTimer(self)
            self._glow_timer.timeout.connect(self._glow_tick)
            self._glow_timer.start(30)

            # Done polling
            self._done_timer = QTimer(self)
            self._done_timer.timeout.connect(self._check_done)
            self._done_timer.start(100)

        # ----------------------------------------------------------------------
        def _smooth_tick(self):
            self._opacity = min(1.0, self._opacity + 0.02)
            self.setWindowOpacity(self._opacity)
            if self._opacity >= 1.0:
                self._fade_timer.stop()
                self._fade_done = True

        # ----------------------------------------------------------------------
        def _snow_tick(self):
            # Ramp opacity proxy so border fades in with the snow
            self._opacity = min(1.0, len(self._revealed) / max(1, len(self._all_blocks)))
            start = len(self._revealed)
            end   = min(start + self._batch_size, len(self._all_blocks))
            for i in range(start, end):
                self._revealed.add(self._all_blocks[i])
            self.update()
            if len(self._revealed) >= len(self._all_blocks):
                self._fade_timer.stop()
                self._fade_done = True

        # ----------------------------------------------------------------------
        def _glow_tick(self):
            self._phase += 0.05
            self.update()

        # ----------------------------------------------------------------------
        def _check_done(self):
            if done_file and os.path.exists(done_file):
                self.close()
                app.quit()

        # ----------------------------------------------------------------------
        def _draw_border(self, painter, alpha_scale=1.0):
            """Draw the glow border, scaled by alpha_scale (0.0–1.0) for fade-in."""
            t     = (math.sin(self._phase) + 1.0) / 2.0  # always driven by phase, no jump
            r     = int(0  + t * 15)
            g     = int(30 + t * 120)
            b     = int(60 + t * 195)
            al    = int((130 + t * 125) * alpha_scale)
            color = QColor(r, g, b, al)
            for i in range(BORDER, 0, -1):
                fade = i / BORDER
                pen  = QPen(QColor(color.red(), color.green(), color.blue(),
                                   int(color.alpha() * fade)), 1.5)
                painter.setPen(pen)
                m = BORDER - i
                painter.drawRect(QRect(m, m, self.width() - m*2 - 1, self.height() - m*2 - 1))

        # ----------------------------------------------------------------------
        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            if fade_mode == FADE_SNOW and not self._fade_done:
                painter.fillRect(self.rect(), Qt.black)
                bs = self._block_size
                for (c, r) in self._revealed:
                    sx, sy = c * bs, r * bs
                    painter.drawPixmap(BORDER + sx, BORDER + sy, pixmap, sx, sy, bs, bs)
            else:
                painter.drawPixmap(BORDER, BORDER, pixmap)

            # Border always drawn — scales alpha by _opacity so it fades in with the image
            self._draw_border(painter, alpha_scale=self._opacity)

    splash = SplashWidget()
    splash.show()
    app.processEvents()
    sys.exit(app.exec_())


# ======================================================================================================================
#
# ======================================================================================================================
class SplashScreen:
    """
    Shows a splash screen in a completely separate subprocess.
    Call dismiss() when initialization is complete.

    Parameters
    ----------
    image_path : Absolute path to the splash image file.
    strVersion : Optional version string painted onto the image.
    fade_mode  : FADE_SMOOTH (default), FADE_NONE, or FADE_SNOW.
    """

    def __init__(self, image_path: str = '', strVersion: str = '',
                 fade_mode: str = FADE_SMOOTH, delay: float = 2, pixmap=None):
        self._image_path  = image_path
        self._version_str = strVersion
        self._fade_mode   = fade_mode
        self._process     = None
        self._done_file   = None

    def show(self, mainWin=None):
        # Communicate fade_mode and done_file via environment — nothing extra in argv
        from .app_identity import APP_ID
        done_file = os.path.join(tempfile.gettempdir(),
                                 f'{APP_ID.lower()}_splash_{os.getpid()}.done')
        self._done_file = done_file

        env = os.environ.copy()
        env[ENV_FADE_MODE] = self._fade_mode
        env[ENV_DONE_FILE] = done_file

        script = os.path.abspath(__file__)
        self._process = subprocess.Popen(
            [sys.executable, script,
             self._image_path,
             self._version_str],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def dismiss(self):
        if self._done_file:
            try:
                open(self._done_file, 'w').close()
            except Exception:
                pass
        if self._process:
            try:
                self._process.wait(timeout=2)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        if self._done_file:
            try:
                os.remove(self._done_file)
            except Exception:
                pass
            self._done_file = None

    def finish(self, mainWin=None):
        self.dismiss()
