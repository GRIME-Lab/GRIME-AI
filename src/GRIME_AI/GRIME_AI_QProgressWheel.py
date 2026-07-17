#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import operator
import sys
from enum import Enum

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, QPointF, Qt, QRectF
from PyQt5.QtGui import (QPalette, QConicalGradient, QGradient, QRadialGradient,
                         QFontMetricsF, QFont, QPainter, QPen, QPainterPath, QImage,
                         QPaintEvent)
from PyQt5.QtWidgets import QWidget, QApplication


def _running_headless() -> bool:
    """True when there is no usable GUI. Intrinsic to Qt state, so no external
    flag is needed: with no QApplication a QWidget cannot be constructed at all,
    and an offscreen/minimal platform has no visible display."""
    try:
        app = QApplication.instance()
    except Exception:
        return True
    if app is None:
        return True
    try:
        if app.platformName() in ("offscreen", "minimal", ""):
            return True
    except Exception:
        pass
    return False


class QProgressWheel(QWidget):

    # When constructed headlessly, transparently become the console
    # subclass. isinstance() and subclassing still work, and every call
    # site keeps constructing QProgressWheel(...) unchanged. The guard
    # cls is QProgressWheel means real GUI subclasses are never swapped.
    def __new__(cls, *args, **kwargs):
        if cls is QProgressWheel and _running_headless():
            return super().__new__(_QProgressWheelConsole)
        return super().__new__(cls)

    # CONSTANTS
    PositionLeft = 180
    PositionTop = 90
    PositionRight = 0
    PositionBottom = -90

    # CONSTRUCTOR ---------------------------------------------------
    def __init__(self, startVal=0, maxVal=0, alwaysOnTop: bool = True,
                 title: str = None, total: int = None, on_close=None, parent=None):
        super(QProgressWheel, self).__init__(parent)
        self._is_closed = False  # must be set before any overridden method is called
        self._on_close_callback = on_close
        self.m_min = 0
        self.m_max = maxVal
        self.m_value = startVal
        self.m_nullPosition = QProgressWheel.PositionTop
        self.m_barStyle = self.BarStyle.DONUT
        self.m_outlinePenWidth = 1
        self.m_dataPenWidth = 1
        self.m_rebuildBrush = False
        self.m_format = '%p%'
        self.m_decimals = 1
        self.m_updateFlags = self.UpdateFlags.PERCENT
        self.m_gradientData = None

        # SET DEFAULT COLORS FOR BASE, TEXT and SHADOW
        palette = QPalette()
        palette.setBrush(QPalette.Base, QtCore.Qt.lightGray)
        palette.setColor(QPalette.Text, QtCore.Qt.magenta)
        palette.setColor(QPalette.Shadow, QtCore.Qt.green)
        self.setPalette(palette)
        self.setNullPosition(QProgressWheel.PositionLeft)
        self.setBarStyle(QProgressWheel.BarStyle.PIE)

        # SET DEFAULT COLORS FOR COLOR WHEEL
        colorList = [QtCore.Qt.red, QtCore.Qt.darkRed, QtCore.Qt.yellow, QtCore.Qt.green, QtCore.Qt.darkGreen, QtCore.Qt.cyan, QtCore.Qt.blue]
        numColors = len(colorList)
        percentage = 1.0 / float(numColors)
        colorIndex = 0
        gradientPoints = []
        for color in colorList:
            gradientPoints.append((percentage * colorIndex, color))
            colorIndex = colorIndex + 1
        self.setDataColors(gradientPoints)

        # DEFAULT POSITION AND SIZE
        self.setGeometry(800, 400, 400, 400)
        self.move(QApplication.desktop().screen().rect().center() - self.rect().center())

        self.setWindowFlag(Qt.FramelessWindowHint, False)
        flags = Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint
        self.setWindowFlags(flags)

        # Prevent closing this window from quitting the application
        self.setAttribute(Qt.WA_QuitOnClose, False)

        self.show()

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setStyleSheet("background-color: white;")

        if title:
            self.setWindowTitle(title)

        if total is not None:
            self.setRange(0, total)
            self.setValue(1)

    def setAlwaysOnTop(self, enable: bool):
        """Retained for API compatibility. Z-ordering is handled via parent window."""
        pass

    # ENUMS ---------------------------------------------------------
    class BarStyle(Enum):
        DONUT = 0,
        PIE = 1,
        LINE = 2,
        EXPAND = 3

    class UpdateFlags(Enum):
        VALUE = 0,
        PERCENT = 1,
        MAX = 2

    # GETTERS -------------------------------------------------------
    def minimum(self):
        return self.m_min

    def maximum(self):
        return self.m_max

    # SETTERS -------------------------------------------------------
    def setNullPosition(self, position: float):
        if position != self.m_nullPosition:
            self.m_nullPosition = position
            self.m_rebuildBrush = True
            self.update()

    def setBarStyle(self, style: BarStyle):
        if style != self.m_barStyle:
            self.m_barStyle = style
            self.m_rebuildBrush = True
            self.update()

    def setOutlinePenWidth(self, width: float):
        if width != self.m_outlinePenWidth:
            self.m_outlinePenWidth = width
            self.update()

    def setDataPenWidth(self, width: float):
        if width != self.m_dataPenWidth:
            self.m_dataPenWidth = width
            self.update()

    def setDataColors(self, stopPoints: list):
        if stopPoints != self.m_gradientData:
            self.m_gradientData = stopPoints
            self.m_rebuildBrush = True
            self.update()

    def setFormat(self, val: str):
        if val != self.m_format:
            self.m_format = val
            self.valueFormatChanged()

    def resetFormat(self):
        self.m_format = None
        self.valueFormatChanged()

    def setDecimals(self, count: int):
        if count >= 0 and count != self.m_decimals:
            self.m_decimals = count
            self.valueFormatChanged()

    # SLOTS ---------------------------------------------------------
    @pyqtSlot(float, float)
    def setRange(self, minval: float, maxval: float):
        self.m_min = minval
        self.m_max = maxval
        if self.m_max < self.m_min:
            self.m_min = maxval
            self.m_max = minval
        if self.m_value < self.m_min:
            self.m_value = self.m_min
        elif self.m_value > self.m_max:
            self.m_value = self.m_max
        self.m_rebuildBrush = True
        self.update()

    @pyqtSlot(float)
    def setMinimum(self, val: float):
        self.setRange(val, self.m_max)

    @pyqtSlot(float)
    def setMaximum(self, val: float):
        self.setRange(self.m_min, val)

    def setWindowTitle(self, title: str):
        if self._is_closed:
            return
        try:
            super().setWindowTitle(title)
        except Exception:
            pass

    def show(self):
        if self._is_closed:
            return
        try:
            super().show()
        except Exception:
            pass

    def isVisible(self):
        if self._is_closed:
            return False
        try:
            return super().isVisible()
        except Exception:
            return False

    def close(self):
        if self._is_closed:
            return
        try:
            super().close()
        except Exception:
            pass

    def repaint(self):
        if self._is_closed:
            return
        try:
            super().repaint()
        except Exception:
            pass

    @pyqtSlot(int)
    def setValue(self, val: int):
        if self._is_closed:
            return
        if self.m_value != val:
            if val < self.m_min:
                self.m_value = self.m_min
            elif val > self.m_max:
                self.m_value = self.m_max
            else:
                self.m_value = val
            self.update()
        QApplication.processEvents()

    @pyqtSlot()
    def getValue(self):
        return self.m_value

    # PAINTING ------------------------------------------------------
    def paintEvent(self, event: QPaintEvent):
        outerRadius = min(self.width(), self.height())
        self.resize(outerRadius, outerRadius)
        baseRect = QRectF(1, 1, outerRadius - 2, outerRadius - 2)
        buffer = QImage(outerRadius, outerRadius, QImage.Format_ARGB32_Premultiplied)
        p = QPainter(buffer)
        p.setRenderHint(QPainter.Antialiasing)
        self.rebuildDataBrushIfNeeded()
        self.drawBackground(p, buffer.rect())
        self.drawBase(p, baseRect)
        if self.m_value > 0:
            try:
                delta = (self.m_max - self.m_min) / (self.m_value - self.m_min)
            except Exception:
                delta = 0
        else:
            delta = 0
        self.drawValue(p, baseRect, self.m_value, delta)
        innerRect, innerRadius = self.calculateInnerRect(outerRadius)
        self.drawInnerBackground(p, innerRect)
        self.drawText(p, innerRect, innerRadius, self.m_value)
        p.end()
        painter = QPainter(self)
        painter.fillRect(baseRect, self.palette().window())
        painter.drawImage(0, 0, buffer)

    def drawBackground(self, p: QPainter, baseRect: QRectF):
        p.fillRect(baseRect, self.palette().window())

    def drawBase(self, p: QPainter, baseRect: QRectF):
        if self.m_barStyle == self.BarStyle.DONUT:
            p.setPen(QPen(self.palette().shadow().color(), self.m_outlinePenWidth))
            p.setBrush(self.palette().base())
            p.drawEllipse(baseRect)
        elif self.m_barStyle == self.BarStyle.LINE:
            p.setPen(QPen(self.palette().base().color(), self.m_outlinePenWidth))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(baseRect.adjusted(self.m_outlinePenWidth / 2, self.m_outlinePenWidth / 2,
                                            -self.m_outlinePenWidth / 2, -self.m_outlinePenWidth / 2))
        elif self.m_barStyle in (self.BarStyle.PIE, self.BarStyle.EXPAND):
            p.setPen(QPen(self.palette().base().color(), self.m_outlinePenWidth))
            p.setBrush(self.palette().base())
            p.drawEllipse(baseRect)

    def drawValue(self, p: QPainter, baseRect: QRectF, value: float, delta: float):
        if value == self.m_min:
            return
        if self.m_barStyle == self.BarStyle.EXPAND:
            p.setBrush(self.palette().highlight())
            p.setPen(QPen(self.palette().shadow().color(), self.m_dataPenWidth))
            radius = (baseRect.height() / 2) / delta
            p.drawEllipse(baseRect.center(), radius, radius)
            return
        if self.m_barStyle == self.BarStyle.LINE:
            p.setPen(QPen(self.palette().highlight().color(), self.m_dataPenWidth))
            p.setBrush(Qt.NoBrush)
            if value == self.m_max:
                p.drawEllipse(baseRect.adjusted(self.m_outlinePenWidth / 2, self.m_outlinePenWidth / 2,
                                                -self.m_outlinePenWidth / 2, -self.m_outlinePenWidth / 2))
            else:
                arcLength = 360 / delta
                p.drawArc(baseRect.adjusted(self.m_outlinePenWidth / 2, self.m_outlinePenWidth / 2,
                                            -self.m_outlinePenWidth / 2, -self.m_outlinePenWidth / 2),
                          int(self.m_nullPosition * 16),
                          int(-arcLength * 16))
            return
        dataPath = QPainterPath()
        dataPath.setFillRule(Qt.WindingFill)
        if value == self.m_max:
            dataPath.addEllipse(baseRect)
        else:
            arcLength = 360 / delta
            dataPath.moveTo(baseRect.center())
            dataPath.arcTo(baseRect, self.m_nullPosition, -arcLength)
            dataPath.lineTo(baseRect.center())
        p.setBrush(self.palette().highlight())
        p.setPen(QPen(self.palette().shadow().color(), self.m_dataPenWidth))
        p.drawPath(dataPath)

    def calculateInnerRect(self, outerRadius: float):
        if self.m_barStyle in (self.BarStyle.LINE, self.BarStyle.EXPAND):
            innerRadius = outerRadius - self.m_outlinePenWidth
        else:
            innerRadius = outerRadius * 0.75
        delta = (outerRadius - innerRadius) / 2
        innerRect = QRectF(delta, delta, innerRadius, innerRadius)
        return innerRect, innerRadius

    def drawInnerBackground(self, p: QPainter, innerRect: QRectF):
        if self.m_barStyle == self.BarStyle.DONUT:
            p.setBrush(self.palette().alternateBase())
            p.drawEllipse(innerRect)

    def drawText(self, p: QPainter, innerRect: QRectF, innerRadius: float, value: float):
        if not self.m_format:
            return
        f = QFont(self.font())
        f.setPixelSize(10)
        fm = QFontMetricsF(f)
        maxWidth = fm.width(self.valueToText(self.m_max))
        delta = innerRadius / maxWidth
        fontSize = f.pixelSize() * delta * 0.75
        f.setPixelSize(int(fontSize))
        p.setFont(f)
        textRect = QRectF(innerRect)
        p.setPen(self.palette().text().color())
        p.drawText(textRect, Qt.AlignCenter, self.valueToText(value))

    def valueToText(self, value: float):
        textToDraw = self.m_format
        if self.m_updateFlags == self.UpdateFlags.VALUE:
            textToDraw = textToDraw.replace('%v', str(round(value, self.m_decimals)))
        if self.m_updateFlags == self.UpdateFlags.PERCENT:
            try:
                percent = (value - self.m_min) / (self.m_max - self.m_min) * 100
            except Exception:
                percent = 0
            textToDraw = textToDraw.replace('%p', str(round(percent, self.m_decimals)))
        if self.m_updateFlags == self.UpdateFlags.MAX:
            textToDraw = textToDraw.replace('%m', str(round(self.m_max - self.m_min + 1, self.m_decimals)))
        return textToDraw

    def valueFormatChanged(self):
        if operator.contains(self.m_format, '%v'):
            self.m_updateFlags = self.UpdateFlags.VALUE
        if operator.contains(self.m_format, '%p'):
            self.m_updateFlags = self.UpdateFlags.PERCENT
        if operator.contains(self.m_format, '%m'):
            self.m_updateFlags = self.UpdateFlags.MAX
        self.update()

    def rebuildDataBrushIfNeeded(self):
        if not self.m_rebuildBrush or not self.m_gradientData or self.m_barStyle == self.BarStyle.LINE:
            return
        self.m_rebuildBrush = False
        p = self.palette()
        if self.m_barStyle == self.BarStyle.EXPAND:
            dataBrush = QRadialGradient(0.5, 0.5, 0.5, 0.5, 0.5)
            dataBrush.setCoordinateMode(QGradient.StretchToDeviceMode)
            for i in range(0, len(self.m_gradientData)):
                dataBrush.setColorAt(self.m_gradientData[i][0], self.m_gradientData[i][1])
            p.setBrush(QPalette.Highlight, dataBrush)
        else:
            dataBrush = QConicalGradient(QPointF(0.5, 0.5), self.m_nullPosition)
            dataBrush.setCoordinateMode(QGradient.StretchToDeviceMode)
            for i in range(0, len(self.m_gradientData)):
                dataBrush.setColorAt(1 - self.m_gradientData[i][0], self.m_gradientData[i][1])
            p.setBrush(QPalette.Highlight, dataBrush)
        self.setPalette(p)

    def closeEvent(self, event):
        self._is_closed = True
        if self._on_close_callback is not None:
            self._on_close_callback()
            self._on_close_callback = None
        self.deleteLater()
        event.accept()


# ======================================================================================================================
# Console (headless) variant
# ----------------------------------------------------------------------------------------------------------------------
# A real subclass of QProgressWheel, so isinstance(x, QProgressWheel) holds and
# QProgressWheel(...) can return it from __new__ with no call-site changes. It
# deliberately does NOT chain to QWidget.__init__: in a headless process there
# is no QApplication and the C++ widget cannot exist. Instead it renders to the
# terminal. Every inherited method that would touch the C++ widget is either
# overridden here or neutralised by the update() no-op below, so the console
# instance never reaches the (uninitialised) Qt side. Pure-Python inherited
# reads (getValue/minimum/maximum/valueToText) work as-is via the m_* fields.
# ======================================================================================================================
class _QProgressWheelConsole(QProgressWheel):

    _SPINNER = "|/-\\"

    def __init__(self, startVal=0, maxVal=0, alwaysOnTop: bool = True,
                 title: str = None, total: int = None, on_close=None, parent=None):
        # NOTE: intentionally no super().__init__() — see class docstring.
        self._is_closed = False
        self._on_close_callback = on_close
        self.m_min = 0
        self.m_max = total if total is not None else maxVal
        self.m_value = startVal
        self.m_format = '%p%'
        self.m_decimals = 1
        self._title = title or "Working"
        self._spin = 0
        self._last_pct = -1
        self._width = 28
        try:
            self._isatty = bool(sys.stdout.isatty())
        except Exception:
            self._isatty = False
        self._render(force=True)

    # Neutralise the one Qt call the inherited setters funnel through, so
    # cosmetic setters (setBarStyle/setFormat/setDataColors/...) stay safe.
    def update(self, *args, **kwargs):
        pass

    # --- range / value -------------------------------------------------------
    def setRange(self, minval, maxval):
        self.m_min, self.m_max = minval, maxval
        if self.m_max < self.m_min:
            self.m_min, self.m_max = self.m_max, self.m_min
        if self.m_value < self.m_min:
            self.m_value = self.m_min
        elif self.m_value > self.m_max:
            self.m_value = self.m_max
        self._render(force=True)

    def setValue(self, val):
        if self._is_closed:
            return
        if val < self.m_min:
            val = self.m_min
        elif val > self.m_max:
            val = self.m_max
        self.m_value = val
        self._render()

    # --- window-ish API ------------------------------------------------------
    def setWindowTitle(self, title):
        self._title = title or self._title
        self._render(force=True)

    def show(self):
        self._render(force=True)

    def isVisible(self):
        return not self._is_closed

    def repaint(self):
        self._render(force=True)

    def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        if self._isatty:
            try:
                sys.stdout.write("\n"); sys.stdout.flush()
            except Exception:
                pass
        if self._on_close_callback is not None:
            cb, self._on_close_callback = self._on_close_callback, None
            try:
                cb()
            except Exception:
                pass

    # --- rendering -----------------------------------------------------------
    def _pct(self):
        span = self.m_max - self.m_min
        return 0.0 if span <= 0 else (self.m_value - self.m_min) / span * 100.0

    def _render(self, force=False):
        if self._is_closed:
            return
        pct = self._pct()
        try:
            if self._isatty:
                self._spin = (self._spin + 1) % len(self._SPINNER)
                filled = int(self._width * pct / 100.0)
                bar = "\u2588" * filled + "\u2591" * (self._width - filled)
                sys.stdout.write(
                    f"\r{self._title} [{bar}] {pct:5.1f}% "
                    f"({int(self.m_value)}/{int(self.m_max)}) {self._SPINNER[self._spin]}"
                )
                sys.stdout.flush()
            else:
                ip = int(pct)
                if force or (ip != self._last_pct and ip % 5 == 0):
                    self._last_pct = ip
                    print(f"{self._title}: {ip}% ({int(self.m_value)}/{int(self.m_max)})", flush=True)
        except Exception:
            pass
