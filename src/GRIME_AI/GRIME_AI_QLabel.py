from PyQt5 import Qt, QtCore
from PyQt5.QtCore import QRect, QPoint, Qt
from PyQt5.QtGui import QPen, QBrush, QPainter, QPainterPath, QPolygon, QPolygonF
from PyQt5.QtWidgets import QLabel, QToolTip
from GRIME_AI.dialogs.color_segmentation.color_seg_roi_data import ROIShape
from GRIME_AI.QLabel_drawing_modes import DrawingMode


# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====    class GRIME_AI_QLabel    =====     =====     =====     =====    =====
# ======================================================================================================================
# ======================================================================================================================
class GRIME_AI_QLabel(QLabel):
    resized = QtCore.pyqtSignal()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def __init__(self, parent=None):
        QLabel.__init__(self, parent=parent)
        self.savedROIs = []
        self.x0 = self.y0 = self.x1 = self.y1 = -1
        self.rect = QRect(self.x0, self.y0, self.x1, self.y1)
        self.flag = False
        self.shape = ROIShape.RECTANGLE
        self.drawingMode = DrawingMode.OFF
        self.enableFill = False
        self.path = QPainterPath()
        self.points = QPolygon()
        self.brushColor = Qt.green
        self.polygonList = []
        self.savedPolygons = []   # closed color-seg polygons (QPolygon list)
        self.lastPolygon = None
        self.polygonListCount = 0  # initialized
        self.setWindowTitle("Slice Position")

        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover, True)
        self.tooltipGenerator = None

        # Slice interaction state
        self._sliceHitMargin = 6
        self._minSliceWidth = 4
        self._draggingSlice = False
        self._resize = False
        self._lastMouseX = None

        self._centerRatio = 0.5
        self._widthRatio = 0.05

        self._orig_w = None
        self._orig_h = None

        # Initialize slice from ratios; avoid hard-coded override
        w = max(1, self.size().width())
        self.sliceCenter = int(w * self._centerRatio)
        self.sliceWidth = max(self._minSliceWidth, int(w * self._widthRatio))

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def setOriginalImageShape(self, shape):
        """
        Store the original image dimensions (h, w, channels).
        Call this once from the dialog after loading the image.
        """
        self._orig_h, self._orig_w, _ = shape

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        draw_w, _, x_off, _ = self._drawn_pixmap_geometry()
        if draw_w > 0:
            self.sliceCenter = int(x_off + draw_w * self._centerRatio)
            self.sliceWidth = max(self._minSliceWidth, int(draw_w * self._widthRatio))
            half = self.sliceWidth // 2
            self.sliceCenter = max(x_off + half, min(x_off + draw_w - half, self.sliceCenter))
        self.update()
        self.resized.emit()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def mousePressEvent(self, event):
        if self.drawingMode == DrawingMode.SLICE:
            x = event.x()
            left_edge = int(self.sliceCenter - self.sliceWidth // 2)
            right_edge = int(self.sliceCenter + self.sliceWidth // 2)

            hit_left = abs(x - left_edge) <= self._sliceHitMargin
            hit_right = abs(x - right_edge) <= self._sliceHitMargin
            inside_band = left_edge < x < right_edge

            if hit_left or hit_right or inside_band:
                self._draggingSlice = True
                # Right-button near edges = resize; otherwise move
                self._resize = (event.button() == Qt.RightButton) and (hit_left or hit_right)
                self._lastMouseX = x
                event.accept()
                return
        # Fall through to ROI/mask handling only when not in SLICE mode or not hitting slice
        # COLOR_SEGMENTATION + POLYGON: left-click adds a vertex,
        # right-click closes the polygon.
        if (self.drawingMode == DrawingMode.COLOR_SEGMENTATION
                and self.shape == ROIShape.POLYGON):
            if event.button() == Qt.RightButton:
                self._closeColorSegPolygon()
                super().mousePressEvent(event)
                return
            self.flag = True
            self.points << event.pos()
            self.update()
            super().mousePressEvent(event)
            return

        # COLOR_SEGMENTATION + FREEFORM: press starts a traced path.
        if (self.drawingMode == DrawingMode.COLOR_SEGMENTATION
                and self.shape == ROIShape.FREEFORM):
            self.flag = True
            self.points = QPolygon()
            self.points << event.pos()
            self.update()
            super().mousePressEvent(event)
            return

        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()

        if self.drawingMode == DrawingMode.MASK:
            self.points << event.pos()

        self.update()
        super().mousePressEvent(event)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def mouseDoubleClickEvent(self, event):
        # Double-click closes an in-progress color-seg polygon.
        if (self.drawingMode == DrawingMode.COLOR_SEGMENTATION
                and self.shape == ROIShape.POLYGON
                and self.points.count() >= 3):
            self._closeColorSegPolygon()
            return
        self.flag = False

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def mouseReleaseEvent(self, event):
        if self._draggingSlice:
            self._draggingSlice = False
            self._resize = False
            self._lastMouseX = None
            self.update()
            event.accept()
            return

        if (self.drawingMode == DrawingMode.COLOR_SEGMENTATION
                and self.shape == ROIShape.POLYGON):
            # vertices are added on press; closed by right-click/double-click
            return
        if (self.drawingMode == DrawingMode.COLOR_SEGMENTATION
                and self.shape == ROIShape.FREEFORM):
            # mouse-up closes the traced path
            self._closeColorSegPolygon()
            return
        if self.flag:
            roi = self.getROI()
            if roi:
                self.savedROIs.append(roi)
        self.flag = False
        # Clear live coords so we don't redraw the just-saved ROI
        self.x0 = self.y0 = self.x1 = self.y1 = -1
        self.update()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _drawn_pixmap_geometry(self):
        """
        Returns (draw_w, draw_h, x_off, y_off) for the currently drawn pixmap.
        If no pixmap, returns zeros.
        """
        pm = self.pixmap()
        if not pm or pm.isNull():
            return 0, 0, 0, 0

        label_w, label_h = self.width(), self.height()
        pm_w, pm_h = pm.width(), pm.height()
        x_off = (label_w - pm_w) // 2
        y_off = (label_h - pm_h) // 2
        return pm_w, pm_h, x_off, y_off

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def mouseMoveEvent(self, event):
        # FREEFORM: while the button is held, trace the path.
        if (self.drawingMode == DrawingMode.COLOR_SEGMENTATION
                and self.shape == ROIShape.FREEFORM and self.flag):
            self.points << event.pos()
            self.update()
            return
        # Only update slice when actively dragging
        if self.drawingMode == DrawingMode.SLICE and self._draggingSlice:
            x = event.x()
            dx = x - (self._lastMouseX if self._lastMouseX is not None else x)
            self._lastMouseX = x

            draw_w, _, x_off, _ = self._drawn_pixmap_geometry()

            if self._resize:
                # half-width from center to mouse
                new_half = abs(x - self.sliceCenter)
                new_width = int(new_half * 2)  # full width
                new_width = max(self._minSliceWidth, new_width)

                left_edge = self.sliceCenter - new_width // 2
                right_edge = self.sliceCenter + new_width // 2

                if left_edge < x_off:
                    new_width -= int(2 * (x_off - left_edge))
                if right_edge > x_off + draw_w:
                    new_width -= int(2 * (right_edge - (x_off + draw_w)))

                self.sliceWidth = max(self._minSliceWidth, new_width)
                # Persist width ratio only if you want resize persistence
                # self._widthRatio = self.sliceWidth / draw_w if draw_w > 0 else self._widthRatio
            else:
                new_center = self.sliceCenter + dx
                half = self.sliceWidth // 2
                self.sliceCenter = max(x_off + half, min(x_off + draw_w - half, new_center))
                # Persist center ratio only if you want move persistence
                # self._centerRatio = (self.sliceCenter - x_off) / draw_w if draw_w > 0 else self._centerRatio

            self.update()
            event.accept()
            return

        # ROI dragging logic
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def paintEvent(self, event):
        super().paintEvent(event)

        if self.drawingMode == DrawingMode.OFF:
            return

        painter = QPainter(self)

        # Only draw saved ROIs if not in SLICE mode
        if self.drawingMode != DrawingMode.SLICE and self.savedROIs:
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            # Saved rectangles are always shown, whatever shape is currently selected.
            for roi in self.savedROIs:
                painter.drawRect(roi)
            # polygons are painted in drawColorSegmentationROI

        if self.drawingMode == DrawingMode.COLOR_SEGMENTATION:
            self.drawColorSegmentationROI(painter)
        elif self.drawingMode == DrawingMode.MASK:
            self.drawPolygon(painter)
        elif self.drawingMode == DrawingMode.SLICE:
            self.drawCompositeSlice(painter)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def enterEvent(self, event):
        if self.tooltipGenerator and callable(self.tooltipGenerator):
            try:
                tooltip_text = self.tooltipGenerator()
            except Exception as e:
                tooltip_text = f"Error retrieving tooltip: {e}"
            self.setToolTip(tooltip_text)
            self.showToolTip(event.globalPos())  # Force tooltip display
        super(GRIME_AI_QLabel, self).enterEvent(event)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def showToolTip(self, global_pos):
        QToolTip.showText(global_pos, self.toolTip(), self)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def getSliceCenterInOriginal(self) -> int:
        """
        Return the slice center X in original image coordinates.
        """
        rect = self.getSliceRectInOriginal()
        return rect.left() + rect.width() // 2

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def getSliceWidthInOriginal(self) -> int:
        """
        Return the slice width in original image coordinates.
        """
        rect = self.getSliceRectInOriginal()
        return rect.width()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def getSliceRectInOriginal(self) -> QRect:
        pm = self.pixmap()
        if not pm or pm.isNull():
            return QRect()

        draw_w, draw_h, x_off, y_off = self._drawn_pixmap_geometry()
        if draw_w <= 0 or draw_h <= 0:
            return QRect()

        # Current slice edges in label coordinates
        left_label = self.sliceCenter - self.sliceWidth // 2
        right_label = self.sliceCenter + self.sliceWidth // 2

        # Convert to pixmap-relative coords (subtract inset once)
        left_rel = left_label - x_off
        right_rel = right_label - x_off

        # Clamp to drawn pixmap bounds
        left_rel = max(0, min(draw_w, left_rel))
        right_rel = max(0, min(draw_w, right_rel))

        # Scale to original image coordinates
        scale_x = self._orig_w / draw_w
        left_orig = int(round(left_rel * scale_x))
        right_orig = int(round(right_rel * scale_x))

        # Clamp and ensure non-zero width
        left_orig = max(0, min(self._orig_w, left_orig))
        right_orig = max(left_orig + 1, min(self._orig_w, right_orig))

        return QRect(left_orig, 0, right_orig - left_orig, self._orig_h)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def drawCompositeSlice(self, painter):
        labelSize = self.size()
        left_x = int(self.sliceCenter - self.sliceWidth // 2)
        right_x = int(self.sliceCenter + self.sliceWidth // 2)

        painter.setPen(QPen(Qt.red, 1, Qt.SolidLine))
        painter.drawLine(self.sliceCenter, 0, self.sliceCenter, labelSize.height())

        painter.setPen(QPen(Qt.magenta, 1, Qt.SolidLine))
        painter.drawLine(left_x, 0, left_x, labelSize.height())
        painter.drawLine(right_x, 0, right_x, labelSize.height())

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def drawContinuous(self):
        if self.flag:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawLine(self.x0, self.y0, self.x1, self.y1)
            self.x0 = self.x1
            self.y0 = self.y1

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _closeColorSegPolygon(self):
        """Finalize the in-progress color-seg polygon/freeform path."""
        if self.points.count() >= 3:
            self.savedPolygons.append(QPolygon(self.points))
            self.lastPolygon = QPolygon(self.points)
        self.points = QPolygon()
        self.flag = False
        self.update()

    def drawColorSegmentationROI(self, painter):
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))

        # already-closed polygons are always shown, whatever shape is currently selected
        for poly in self.savedPolygons:
            painter.drawPolygon(poly)

        if self.getROIShape() in (ROIShape.POLYGON, ROIShape.FREEFORM):
            # in-progress polygon: vertices + connecting lines
            if self.points.count() > 0:
                lp = QPoint()
                for cp in self.points:
                    painter.drawEllipse(cp, 2, 2)
                    if not lp.isNull():
                        painter.drawLine(lp, cp)
                    lp = cp
            return

        # RECTANGLE (rubber-band)
        if self.flag:
            x = min(self.x0, self.x1)
            y = min(self.y0, self.y1)
            w = abs(self.x1 - self.x0)
            h = abs(self.y1 - self.y0)
            painter.drawRect(QRect(x, y, w, h))

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def drawPolygon(self, qp):
        qp.setRenderHint(QPainter.Antialiasing)

        pen = QPen(Qt.red, 1)
        qp.setPen(pen)

        brush = QBrush(self.brushColor)
        qp.setBrush(brush)

        lp = QPoint()
        for myPoint in self.points:
            cp = myPoint
            qp.drawEllipse(cp, 2, 2)
            if not lp.isNull():
                qp.drawLine(lp, cp)
            lp = cp

        if self.enableFill:
            # Fill polygon
            polyPath = QPainterPath()
            polyPath.addPolygon(QPolygonF(self.points))

            # Draw polygon
            qp.drawPolygon(QPolygonF(self.points))
            qp.fillPath(polyPath, brush)

        for myPolygon in self.polygonList:
            lp = QPoint()
            for myPoints in myPolygon:
                cp = myPoints
                qp.drawEllipse(cp, 2, 2)
                if not lp.isNull():
                    qp.drawLine(lp, cp)
                lp = cp

                if self.enableFill:
                    # Fill polygon
                    polyPath = QPainterPath()
                    polyPath.addPolygon(QPolygonF(myPoints))

                    # Draw polygon
                    qp.drawPolygon(QPolygonF(myPoints))
                    qp.fillPath(polyPath, brush)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def getPolygon(self):
        return self.polygonList

    # ------------------------------------------------------------------------
    def getColorSegPolygons(self):
        """Closed color-seg polygons as lists of (x,y) tuples (display coords)."""
        return [[(pt.x(), pt.y()) for pt in poly] for poly in self.savedPolygons]

    def getLastColorSegPolygon(self):
        if self.lastPolygon is None:
            return None
        return [(pt.x(), pt.y()) for pt in self.lastPolygon]

    def clearColorSegPolygons(self):
        self.savedPolygons = []
        self.lastPolygon = None
        self.points = QPolygon()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def incrementPolygon(self):
        self.polygonList.append(self.points)
        self.points.clear()
        self.polygonListCount = len(self.polygonList)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def getROI(self):
        # Prefer live ROI if dragging
        if self.x0 != -1 and self.x1 != -1:
            x = min(self.x0, self.x1)
            y = min(self.y0, self.y1)
            w = abs(self.x1 - self.x0)
            h = abs(self.y1 - self.y0)
            return QRect(x, y, w, h)

        # Fall back to last saved ROI
        if self.savedROIs:
            return self.savedROIs[-1]

        return None

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def setROIShape(self, shape):
        self.shape = shape
        # Discard any half-drawn shape when the user switches tools.
        self.points = QPolygon()
        self.flag = False
        self.x0 = self.y0 = self.x1 = self.y1 = -1
        self.update()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def getROIShape(self):
        return self.shape

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def setBrushColor(self, brushColor):
        self.brushColor = brushColor

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def resetMask(self):
        if self.points.count() > 0:
            self.points = QPolygon()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def setDrawingMode(self, mode):
        self.drawingMode = mode

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def enablePolygonFill(self, bFill):
        self.enableFill = bFill

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def setSliceCenter(self, sliceCenter):
        draw_w, _, x_off, _ = self._drawn_pixmap_geometry()
        sliceCenter = int(sliceCenter)

        if draw_w > 0:
            half = self.sliceWidth // 2
            # Clamp to drawn pixmap bounds
            clamped = max(x_off + half, min(x_off + draw_w - half, sliceCenter))
            self.sliceCenter = clamped
            # Persist relative center ratio
            self._centerRatio = (self.sliceCenter - x_off) / draw_w
        else:
            self.sliceCenter = sliceCenter

        self.update()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def getSliceCenter(self):
        return self.sliceCenter

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def setSliceWidth(self, sliceWidth):
        draw_w, _, x_off, _ = self._drawn_pixmap_geometry()
        sliceWidth = int(sliceWidth)

        if draw_w > 0:
            # Clamp width to available drawn pixmap width
            sliceWidth = max(self._minSliceWidth, min(draw_w, sliceWidth))
            self.sliceWidth = sliceWidth
            self._widthRatio = self.sliceWidth / draw_w

            # Ensure center remains valid for new width
            half = self.sliceWidth // 2
            self.sliceCenter = max(x_off + half, min(x_off + draw_w - half, self.sliceCenter))
        else:
            self.sliceWidth = max(self._minSliceWidth, sliceWidth)

        self.update()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def getSliceWidth(self):
        return self.sliceWidth

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def setROIs(self, roi_list):
        # Rectangles and polygon/freeform ROIs are drawn from their own lists.
        self.savedROIs = [roi.getDisplayROI() for roi in roi_list
                          if roi.getROIShape() == ROIShape.RECTANGLE]
        self.savedPolygons = [QPolygon(roi.getDisplayPolygon()) for roi in roi_list
                              if roi.getROIShape() != ROIShape.RECTANGLE and roi.getDisplayPolygon()]
        # The last drawn polygon has been consumed (or cleared); require a new one for the next ROI.
        self.lastPolygon = None
        self.update()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def clearROIs(self):
        self.savedROIs.clear()
        self.x0 = self.y0 = self.x1 = self.y1 = -1
        self.flag = False
        self.update()
