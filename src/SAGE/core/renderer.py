# sam2_gui/core/renderer.py
import numpy as np
import cv2
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QPolygonF
from PyQt5.QtCore import QPointF


class Renderer:
    def __init__(self, base_image_np: np.ndarray):
        self.base_image_np = base_image_np

    def base_pixmap(self):
        h, w, _ = self.base_image_np.shape
        qimage = QImage(
            self.base_image_np.data,
            w,
            h,
            self.base_image_np.strides[0],
            QImage.Format_RGB888,
        )
        return QPixmap.fromImage(qimage)

    def overlay_masks(self, base_pixmap, masks, opacity=120, selected_mask_id=-1,
                      show_borders=True, flash_mask_id=-1):
        """
        masks: list of dicts:
          - 'mask': np.ndarray (H, W)
          - 'color': (r, g, b)
          - 'visible': bool
        selected_mask_id: draw this mask's border in yellow instead of black
        """
        result = QPixmap(base_pixmap)
        painter = QPainter(result)

        for m in masks:
            if not m.get("visible", True):
                continue
            mask = m["mask"].astype(bool)
            color = m["color"]

            # Flash: brighten the fill and boost opacity for the flashed mask.
            is_flash = (m.get("id", -1) == flash_mask_id)
            if is_flash:
                color = (min(255, color[0] + 120),
                         min(255, color[1] + 120),
                         min(255, color[2] + 120))
                fill_opacity = min(255, opacity + 110)
            else:
                fill_opacity = opacity

            # Draw filled mask
            rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
            rgba[..., 0] = color[0]
            rgba[..., 1] = color[1]
            rgba[..., 2] = color[2]
            rgba[..., 3] = mask * fill_opacity

            qimage = QImage(
                rgba.data,
                rgba.shape[1],
                rgba.shape[0],
                rgba.strides[0],
                QImage.Format_RGBA8888,
            )
            painter.drawImage(0, 0, qimage)

            # Draw border — yellow+thicker for selected, black for others
            is_selected = (m.get("id", -1) == selected_mask_id)
            if show_borders or is_selected:
                # Yellow borders (matches GRIME AI mask viewer). Selected mask
                # gets a thicker line so it still stands out.
                if is_selected:
                    border_pen = QPen(QColor(255, 255, 0), 2)
                else:
                    border_pen = QPen(QColor(255, 255, 0), 1)

                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                painter.setPen(border_pen)

                for contour in contours:
                    if len(contour) < 3:
                        continue
                    points = [QPointF(float(pt[0][0]), float(pt[0][1])) for pt in contour]
                    polygon = QPolygonF(points)
                    painter.drawPolygon(polygon)

        painter.end()
        return result

    def overlay_single_mask(self, base_pixmap, m, opacity=120):
        """Paint ONE mask, brightened, on top of an already-composited pixmap.
        Cheap path used by the selection flash — no full recomposite, no
        per-mask contour work. Matches the brighten used in overlay_masks."""
        result = QPixmap(base_pixmap)
        painter = QPainter(result)
        mask = m["mask"].astype(bool)
        color = m["color"]
        bright = (min(255, color[0] + 120),
                  min(255, color[1] + 120),
                  min(255, color[2] + 120))
        fill_opacity = min(255, opacity + 110)
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[..., 0] = bright[0]
        rgba[..., 1] = bright[1]
        rgba[..., 2] = bright[2]
        rgba[..., 3] = mask * fill_opacity
        qimage = QImage(rgba.data, rgba.shape[1], rgba.shape[0],
                        rgba.strides[0], QImage.Format_RGBA8888)
        painter.drawImage(0, 0, qimage)
        painter.end()
        return result

    def draw_points(self, pixmap, fg_points, bg_points):
        result = QPixmap(pixmap)
        painter = QPainter(result)
        fg_pen = QPen(QColor(0, 255, 0), 2)
        bg_pen = QPen(QColor(255, 0, 0), 2)

        for x, y in fg_points:
            painter.setPen(fg_pen)
            painter.drawPoint(int(x), int(y))

        for x, y in bg_points:
            painter.setPen(bg_pen)
            painter.drawPoint(int(x), int(y))

        painter.end()
        return result