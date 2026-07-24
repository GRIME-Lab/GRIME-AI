from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsTextItem
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, QRectF
from PyQt5.QtGui import QPixmap, QPen, QPainterPath, QColor, QBrush, QFont, QImage

from SAGE.ui.mask_item import MaskItem


class Canvas(QGraphicsView):
    polygon_drawn = pyqtSignal(list)              # list of (x, y) tuples
    eraser_move = pyqtSignal(float, float)        # x,y in image coords
    edge_trace_stroke = pyqtSignal(list, list)    # (fg_points, bg_points)
    edge_trace_interval_changed = pyqtSignal(int) # new interval value for HUD
    label_required = pyqtSignal()                 # drawing attempted with no active label
    mask_clicked = pyqtSignal(int)                # Select mode: mask_id clicked on canvas
    annotation_blocked = pyqtSignal()             # drawing attempted on already-annotated pixels

    def __init__(self, on_left_click, on_right_click, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._on_left_click = on_left_click
        self._on_right_click = on_right_click

        self._panning = False
        self._pan_start = None

        # Left-button pan (drag detection)
        self._left_pan_active = False       # currently panning with left button
        self._tool_mode = "draw"            # "draw" | "select" | "pan"
        self._left_press_pos = None         # view-coords where left button went down
        self._left_press_scene_pos = None   # scene-coords at left press
        self._LEFT_PAN_THRESHOLD = 6        # pixels of movement before pan kicks in

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setFocusPolicy(Qt.StrongFocus)

        self._orig_width = None
        self._orig_height = None
        self._user_zoomed = False  # True once user has manually zoomed

        self._segmentation_mode = "points"
        self._drawing_polygon = False
        self._polygon_points = []
        self._polygon_path_item = None

        # Paint mode variables
        self._painting = False
        self._paint_points = []
        self._paint_path_item = None
        self._brush_size = 15  # pixels between paint points
        self._paint_is_negative = False  # track if current stroke is negative

        # Eraser mode variables
        self._eraser_enabled = False
        self._eraser_radius = 18  # tweakable
        self._delete_key_down = False   # 'd' held → click deletes nearest point
        self._is_erasing = False

        # Manual polygon (click-by-click) state
        self._manual_polygon_points = []
        self._manual_rubber_band = None

        # Edge Trace mode variables
        self._edge_tracing = False
        self._edge_trace_points = []        # raw path coords collected during drag
        self._edge_trace_path_item = None   # cyan center stroke
        self._edge_trace_overlay_item = None  # orange edge pixel overlay (QGraphicsPixmapItem)
        self._edge_trace_dot_items = []     # green/red point dot items
        self._edge_trace_fg_pts = []        # accumulated fg points this stroke
        self._edge_trace_bg_pts = []        # accumulated bg points this stroke
        self._edge_trace_edge_pixels = []   # accumulated (x,y) edge pixel coords
        self._edge_trace_dist_accum = 0.0   # distance accumulator for sampling
        self._edge_trace_canny = None       # precomputed canny edge map (numpy)
        self._edge_trace_interval = 40      # pixels between samples (tunable with +/-)
        self._ET_MAX_RAY = 80               # max outward ray reach in pixels
        self._ET_BG_OFFSET = 6             # pixels beyond edge for bg point
        self._et_last_tangent = (0.0, 1.0) # default vertical — interval on y-axis

        # Crosshair guide lines (two QGraphicsLineItems)
        self._et_crosshair_items = []

        # Seed point items (fg/bg, cosmetic size)
        self._seed_point_items = []

        # Left-button deferred click / pan-on-drag state
        self._left_press_pos = None        # view coords of left press
        self._left_press_scene = None      # scene coords of left press
        self._left_pan_active = False      # True once drag threshold exceeded
        _PAN_THRESHOLD = 5                 # pixels before pan kicks in
        self._PAN_THRESHOLD = _PAN_THRESHOLD

        # HUD label for edge trace interval
        self._et_hud = QGraphicsTextItem("Interval: 40px  Width: 80px")
        font = QFont("Monospace", 11)
        font.setBold(True)
        self._et_hud.setFont(font)
        self._et_hud.setDefaultTextColor(QColor(0, 220, 255))
        self._et_hud.setZValue(20_000)
        self._et_hud.setVisible(False)
        self._scene.addItem(self._et_hud)

        # Eraser preview circle (cursor)
        self._eraser_preview = QGraphicsEllipseItem()
        self._eraser_preview.setZValue(10_000)  # always on top
        self._eraser_preview.setVisible(False)

        pen = QPen(QColor(0, 255, 255, 180))  # cyan-ish
        fill = QColor(0, 255, 255, 60)  # cyan, low alpha
        self._eraser_preview.setBrush(QBrush(fill))
        pen.setWidth(2)
        pen.setCosmetic(True)  # IMPORTANT: thickness stays same while zooming
        self._eraser_preview.setPen(pen)

        self._eraser_preview.setBrush(QBrush(Qt.NoBrush))
        self._scene.addItem(self._eraser_preview)

    def set_segmentation_mode(self, mode: str):
        if mode in ("points", "polygon", "paint", "manual_polygon", "manual_draw", "manual_freeform", "mask", "edge_trace"):
            if self._segmentation_mode in ("polygon", "manual_polygon") and mode not in ("polygon", "manual_polygon"):
                self._cancel_manual_polygon()
            if self._segmentation_mode == "edge_trace" and mode != "edge_trace":
                for item in self._et_crosshair_items:
                    self._scene.removeItem(item)
                self._et_crosshair_items = []
            self._segmentation_mode = mode
            self._et_hud.setVisible(mode == "edge_trace")
            if mode == "edge_trace":
                self.setFocus()

    def set_pixmap(self, pixmap: QPixmap):
        """Update the displayed pixmap without resetting zoom."""
        self._pixmap_item.setPixmap(pixmap)
        self._orig_width = pixmap.width()
        self._orig_height = pixmap.height()
        self._scene.setSceneRect(QRectF(pixmap.rect()))

    def reset_zoom(self):
        """Fit image to view and reset user zoom state. Call only on new image load."""
        self._user_zoomed = False
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def set_eraser_enabled(self, enabled: bool):
        self._eraser_enabled = enabled
        if not enabled:
            self._is_erasing = False
        self._eraser_preview.setVisible(enabled)

    def update_seed_points(self, fg_points, bg_points):
        """Redraw fg/bg seed points as cosmetic scene items (zoom-invariant size)."""
        # Remove old items
        for item in self._seed_point_items:
            self._scene.removeItem(item)
        self._seed_point_items = []

        RADIUS = 3  # screen pixels — stays constant regardless of zoom

        fg_pen   = QPen(QColor(0, 180, 0), 1)
        fg_brush = QBrush(QColor(0, 255, 0, 220))
        bg_pen   = QPen(QColor(180, 0, 0), 1)
        bg_brush = QBrush(QColor(255, 0, 0, 220))

        for pen, brush, points in (
            (fg_pen, fg_brush, fg_points),
            (bg_pen, bg_brush, bg_points),
        ):
            for x, y in points:
                item = self._scene.addEllipse(
                    x - RADIUS, y - RADIUS, RADIUS * 2, RADIUS * 2, pen, brush
                )
                item.setFlag(item.ItemIgnoresTransformations, True)  # cosmetic — no zoom scaling
                item.setAcceptedMouseButtons(Qt.NoButton)
                item.setZValue(8_000)
                # Re-center after ItemIgnoresTransformations shifts the origin
                item.setPos(x, y)
                item.setRect(-RADIUS, -RADIUS, RADIUS * 2, RADIUS * 2)
                self._seed_point_items.append(item)

    def _update_eraser_preview(self, x, y):
        r = float(self._eraser_radius)
        self._eraser_preview.setRect(x - r, y - r, 2 * r, 2 * r)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._user_zoomed = True
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._user_zoomed and self._pixmap_item.pixmap():
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click events - MUST come before mousePressEvent logic"""
        # Double-click closes a manual polygon in progress
        if event.button() == Qt.LeftButton and self._segmentation_mode in ("polygon", "manual_polygon"):
            if len(self._manual_polygon_points) >= 3:
                self._close_manual_polygon()
            event.accept()
            return

        # Right-double-click: Delete mask — search all items at position
        if event.button() == Qt.RightButton:
            scene_pos = self.mapToScene(event.pos())
            for item in self._scene.items(scene_pos):
                if isinstance(item, MaskItem):
                    self._on_right_click(item.mask_id)
                    event.accept()
                    return

        super().mouseDoubleClickEvent(event)

    def _pixel_is_owned(self, x, y):
        """True if (x, y) already belongs to a mask. Backed by the
        `_is_owned_pixel` callback the main window installs; returns False when
        no callback is set, so the canvas stays usable standalone."""
        fn = getattr(self, "_is_owned_pixel", None)
        if fn is None:
            return False
        try:
            return bool(fn(x, y))
        except Exception:
            return False

    def set_tool_mode(self, mode: str):
        """Top-level interaction mode: 'draw' (annotate), 'select' (pick a mask),
        or 'pan' (drag the image)."""
        self._tool_mode = mode
        if mode == "pan":
            self.setCursor(Qt.OpenHandCursor)
        elif mode == "select":
            self.setCursor(Qt.ArrowCursor)
        else:
            self.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event):
        # ----------------------------------------------------
        # Tool-mode gate. Select and Pan reassign the left button and never draw.
        # ----------------------------------------------------
        if self._tool_mode == "pan":
            if event.button() == Qt.LeftButton:
                self._left_press_pos = event.pos()
                self._left_pan_active = False
                self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if self._tool_mode == "select":
            scene_pos = self.mapToScene(event.pos())
            if event.button() == Qt.LeftButton:
                for item in self._scene.items(scene_pos):
                    if isinstance(item, MaskItem):
                        self.mask_clicked.emit(item.mask_id)
                        break
                event.accept()
                return
            if event.button() == Qt.RightButton:
                for item in self._scene.items(scene_pos):
                    if isinstance(item, MaskItem):
                        self._on_right_click(item.mask_id)   # delete on right-click
                        break
                event.accept()
                return

        # ----------------------------------------------------
        # Block drawing entirely when no label is active. Deletion/selection
        # shortcuts still work. Fires the dialog once per press — never during a
        # drag — so there is no message-box loop.
        # ----------------------------------------------------
        can = getattr(self, "_can_annotate", None)
        if can is not None and not can():
            is_delete = (
                (self._delete_key_down and event.button() in (Qt.LeftButton, Qt.RightButton))
                or (event.button() == Qt.RightButton and event.modifiers() == Qt.ControlModifier)
            )
            mask_hit = False
            if event.button() == Qt.RightButton and not is_delete:
                sp = self.mapToScene(event.pos())
                mask_hit = any(isinstance(i, MaskItem) for i in self._scene.items(sp))
            if not is_delete and not mask_hit:
                self.label_required.emit()
                event.accept()
                return

        # ----------------------------------------------------
        # 'd' held: click deletes the nearest fg/bg point (either button)
        # ----------------------------------------------------
        if self._delete_key_down and event.button() in (Qt.LeftButton, Qt.RightButton):
            x, y = self._map_to_image_coords(event.pos())
            if self._is_in_bounds(x, y):
                self.eraser_move.emit(x, y)   # routes to _erase_seeds_at → refresh
            event.accept()
            return

        # ----------------------------------------------------
        # Ctrl+Right-click: Delete mask
        # ----------------------------------------------------
        if event.button() == Qt.RightButton and event.modifiers() == Qt.ControlModifier:
            scene_pos = self.mapToScene(event.pos())
            for item in self._scene.items(scene_pos):
                if isinstance(item, MaskItem):
                    self._on_right_click(item.mask_id)
                    event.accept()
                    return

        # ----------------------------------------------------
        # Regular right-click: Add negative point OR paint negative brush
        # ----------------------------------------------------
        if event.button() == Qt.RightButton:
            scene_pos = self.mapToScene(event.pos())

            # A polygon in progress owns the right button. This must come before
            # the MaskItem check below: that check defers to the base class to
            # wait for a possible right-double-click delete, which swallows the
            # press and leaves the polygon permanently open when the click lands
            # over an existing mask.
            if (
                self._segmentation_mode in ("polygon", "manual_polygon")
                and self._manual_polygon_points
            ):
                if len(self._manual_polygon_points) >= 3:
                    self._close_manual_polygon()
                else:
                    self._cancel_manual_polygon()
                event.accept()
                return

            # Refuse to drop a negative prompt point on pixels that already
            # belong to a mask. Tested per-pixel rather than by MaskItem hit:
            # the 'Other' fill is also a MaskItem, so an item-based test blocks
            # the entire frame whenever the fill is displayed. Still deferred to
            # the base class so a right-double-click can delete the mask.
            if self._pixel_is_owned(scene_pos.x(), scene_pos.y()):
                self.annotation_blocked.emit()
                super().mousePressEvent(event)
                return

            # SAM2 Polygon and Manual Polygon: right-click closes or cancels
            if self._segmentation_mode in ("polygon", "manual_polygon"):
                if len(self._manual_polygon_points) >= 3:
                    self._close_manual_polygon()
                else:
                    self._cancel_manual_polygon()
                event.accept()
                return

            # Edge Trace: right-click has no action (double-click handled separately)
            if self._segmentation_mode == "edge_trace":
                super().mousePressEvent(event)
                return

            # Paint mode: start negative brush stroke
            if self._segmentation_mode == "paint":
                pos = self.mapToScene(event.pos())
                x, y = pos.x(), pos.y()

                if (
                    self._orig_width is not None
                    and (x < 0 or y < 0 or x >= self._orig_width or y >= self._orig_height)
                ):
                    super().mousePressEvent(event)
                    return

                self._painting = True
                self._paint_is_negative = True  # Right-click = negative
                self._paint_points = [(x, y)]
                self._on_right_click(x, y)  # Add as background point
                self._init_paint_path_item()
                self._update_paint_path()
                return

            # Not paint mode: add single negative point
            pos = self.mapToScene(event.pos())
            x, y = pos.x(), pos.y()

            if (
                self._orig_width is not None
                and (x < 0 or y < 0 or x >= self._orig_width or y >= self._orig_height)
            ):
                super().mousePressEvent(event)
                return

            self._handle_click(QPointF(x, y), is_left=False)
            return

        # ----------------------------------------------------
        # Left-click: Eraser or Add positive point, draw polygon, or paint
        # ----------------------------------------------------
        if event.button() == Qt.LeftButton:
            # If eraser is enabled, DO NOT do polygon/paint/point behavior
            if self._eraser_enabled:
                self._is_erasing = True
                x, y = self._map_to_image_coords(event.pos())

                if self._is_in_bounds(x, y):
                    self._update_eraser_preview(x, y)  # optional but nice
                    self.eraser_move.emit(x, y)

                event.accept()
                return

            # Normal behavior
            else:
                pos = self.mapToScene(event.pos())
                x = pos.x()
                y = pos.y()

                if (
                        self._orig_width is not None
                        and (x < 0 or y < 0 or x >= self._orig_width or y >= self._orig_height)
                ):
                    super().mousePressEvent(event)
                    return

                else:
                    # Refuse to start or extend an annotation on pixels that
                    # already belong to a mask. Overlap is unsupported for now.
                    if self._pixel_is_owned(x, y):
                        self.annotation_blocked.emit()
                        event.accept()
                        return

                    if self._segmentation_mode in ("polygon", "manual_polygon"):
                        self._add_manual_polygon_vertex(x, y)

                    elif self._segmentation_mode in ("manual_draw", "manual_freeform"):
                        self._drawing_polygon = True
                        self._polygon_points = [(x, y)]
                        self._init_polygon_path_item()
                        self._update_polygon_path()

                    elif self._segmentation_mode == "paint":
                        self._painting = True
                        self._paint_is_negative = False  # Left-click = positive
                        self._paint_points = [(x, y)]
                        self._on_left_click(x, y)  # Add as foreground point
                        self._init_paint_path_item()
                        self._update_paint_path()
                    elif self._segmentation_mode == "edge_trace":
                        self._edge_tracing = True
                        self._edge_trace_points = [(x, y)]
                        self._edge_trace_fg_pts = []
                        self._edge_trace_bg_pts = []
                        self._edge_trace_edge_pixels = []
                        self._edge_trace_dist_accum = 0.0
                        self._precompute_canny()
                        self._init_edge_trace_path_item()
                        self._update_edge_trace_path()
                        self._update_et_hud_position(x, y)
                    else:
                        # Points mode: defer click until release to allow pan detection
                        self._left_press_pos = event.pos()
                        self._left_press_scene_pos = QPointF(x, y)
                        self._left_pan_active = False

        # ----------------------------------------------------
        # Middle-click: Pan
        # ----------------------------------------------------
        elif event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

        super().mousePressEvent(event)

    def _is_in_bounds(self, x, y):
        """Check if coordinates are within image bounds"""
        if self._orig_width is None:
            return True
        return not (x < 0 or y < 0 or x >= self._orig_width or y >= self._orig_height)

    def _map_to_image_coords(self, view_pos):
        """
        Convert a QWidget mouse position (event.pos()) into *image pixel coords*.
        Uses the scene coords since your pixmap is drawn into the scene 1:1.
        """
        scene_pos = self.mapToScene(view_pos)
        return float(scene_pos.x()), float(scene_pos.y())

    def mouseMoveEvent(self, event):
        # Left-button pan detection
        if self._left_press_pos is not None and not self._left_pan_active:
            delta = event.pos() - self._left_press_pos
            if (delta.x() ** 2 + delta.y() ** 2) ** 0.5 >= self._LEFT_PAN_THRESHOLD:
                self._left_pan_active = True
                self.setCursor(Qt.ClosedHandCursor)

        if self._left_pan_active:
            delta = event.pos() - self._left_press_pos
            self._left_press_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )

        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )

        # SAM2 Polygon and Manual Polygon: update rubber-band line to cursor
        if self._segmentation_mode in ("polygon", "manual_polygon") and self._manual_polygon_points:
            pos = self.mapToScene(event.pos())
            self._update_manual_rubber_band(pos.x(), pos.y())

        # Manual Draw: freehand drag streaming
        if self._drawing_polygon and self._segmentation_mode in ("manual_draw", "manual_freeform"):
            pos = self.mapToScene(event.pos())
            x, y = pos.x(), pos.y()
            if (
                self._orig_width is not None
                and (x < 0 or y < 0 or x >= self._orig_width or y >= self._orig_height)
            ):
                super().mouseMoveEvent(event)
                return
            self._polygon_points.append((x, y))
            self._update_polygon_path()

        # Paint mode - add points as mouse moves
        if self._painting and self._segmentation_mode == "paint":
            pos = self.mapToScene(event.pos())
            x = pos.x()
            y = pos.y()

            if (
                    self._orig_width is not None
                    and (x < 0 or y < 0 or x >= self._orig_width or y >= self._orig_height)
            ):
                super().mouseMoveEvent(event)
                return

            # Only add point if far enough from last point (brush spacing)
            if len(self._paint_points) > 0:
                last_x, last_y = self._paint_points[-1]
                dist = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
                if dist < self._brush_size:
                    super().mouseMoveEvent(event)
                    return

            self._paint_points.append((x, y))

            # Add point with correct type based on negative flag
            if self._paint_is_negative:
                self._on_right_click(x, y)  # Add as background point
            else:
                self._on_left_click(x, y)  # Add as foreground point

            self._update_paint_path()

        # Edge Trace mode - collect raw path points during drag
        if self._edge_tracing and self._segmentation_mode == "edge_trace":
            pos = self.mapToScene(event.pos())
            x, y = pos.x(), pos.y()
            if (
                self._orig_width is not None
                and (x < 0 or y < 0 or x >= self._orig_width or y >= self._orig_height)
            ):
                super().mouseMoveEvent(event)
                return
            prev = self._edge_trace_points[-1] if self._edge_trace_points else (x, y)
            self._edge_trace_points.append((x, y))
            # Smooth tangent with exponential moving average to avoid jitter
            import math as _math
            dx, dy = x - prev[0], y - prev[1]
            seg_len = _math.hypot(dx, dy)
            if seg_len >= 1.0:
                new_tx, new_ty = dx / seg_len, dy / seg_len
                old_tx, old_ty = self._et_last_tangent
                # Ensure new tangent points in same half-plane as old (avoid flip)
                if new_tx * old_tx + new_ty * old_ty < 0:
                    new_tx, new_ty = -new_tx, -new_ty
                alpha = 0.15  # smoothing factor — lower = smoother, higher = more responsive
                blended_tx = old_tx + alpha * (new_tx - old_tx)
                blended_ty = old_ty + alpha * (new_ty - old_ty)
                norm = _math.hypot(blended_tx, blended_ty)
                if norm > 1e-6:
                    self._et_last_tangent = (blended_tx / norm, blended_ty / norm)
            self._update_edge_trace_path()
            self._et_sample_segment(prev[0], prev[1], x, y)
            self._update_et_hud_position(x, y)
            self._update_et_crosshair(x, y)

        # Edge Trace hover (not dragging) — keep crosshair visible at cursor
        elif self._segmentation_mode == "edge_trace" and not self._edge_tracing:
            pos = self.mapToScene(event.pos())
            x, y = pos.x(), pos.y()
            self._update_et_hud_position(x, y)
            self._update_et_crosshair(x, y)

        # Show eraser preview following mouse
        if self._eraser_enabled:
            x, y = self._map_to_image_coords(event.pos())
            if self._is_in_bounds(x, y):
                self._update_eraser_preview(x, y)

        # Erase points when it is enabled
        if self._is_erasing and self._eraser_enabled:
            x, y = self._map_to_image_coords(event.pos())
            if self._is_in_bounds(x, y):
                self.eraser_move.emit(x, y)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # Pan tool: finish the drag, restore the open-hand cursor, draw nothing.
        if self._tool_mode == "pan":
            if event.button() == Qt.LeftButton:
                self._left_press_pos = None
                self._left_pan_active = False
                self.setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        # Select tool: left/right handled on press; nothing to resolve here.
        if self._tool_mode == "select":
            event.accept()
            return

        if event.button() == Qt.MiddleButton:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)

        # Left button release — resolve deferred point click vs pan
        if event.button() == Qt.LeftButton and self._left_press_pos is not None:
            if not self._left_pan_active:
                # No drag occurred — treat as a click and place point
                self._handle_click(self._left_press_scene_pos, is_left=True)
            self._left_press_pos = None
            self._left_press_scene_pos = None
            self._left_pan_active = False
            self.setCursor(Qt.ArrowCursor)

        if event.button() == Qt.LeftButton and self._drawing_polygon and self._segmentation_mode in ("manual_draw", "manual_freeform"):
            self._drawing_polygon = False
            if len(self._polygon_points) >= 3:
                if self._polygon_points[0] != self._polygon_points[-1]:
                    self._polygon_points.append(self._polygon_points[0])
                self.polygon_drawn.emit(self._polygon_points.copy())
            if self._polygon_path_item is not None:
                self._scene.removeItem(self._polygon_path_item)
                self._polygon_path_item = None
            self._polygon_points = []

        # Edge Trace mode release - emit stroke and clean up
        if event.button() == Qt.LeftButton and self._edge_tracing:
            self._edge_tracing = False
            if self._edge_trace_fg_pts:
                self.edge_trace_stroke.emit(
                    self._edge_trace_fg_pts.copy(),
                    self._edge_trace_bg_pts.copy(),
                )
            self._edge_trace_points = []
            self._edge_trace_fg_pts = []
            self._edge_trace_bg_pts = []
            self._edge_trace_edge_pixels = []
            self._edge_trace_dist_accum = 0.0
            self._edge_trace_canny = None
            if self._edge_trace_path_item is not None:
                self._scene.removeItem(self._edge_trace_path_item)
                self._edge_trace_path_item = None
            if self._edge_trace_overlay_item is not None:
                self._scene.removeItem(self._edge_trace_overlay_item)
                self._edge_trace_overlay_item = None
            for item in self._edge_trace_dot_items:
                self._scene.removeItem(item)
            self._edge_trace_dot_items = []
            self._et_last_tangent = (0.0, 1.0)  # reset to vertical default

        # Paint mode release - allow multiple strokes before segmentation
        # Works for BOTH left and right button releases
        if (event.button() == Qt.LeftButton or event.button() == Qt.RightButton) and self._painting:
            self._painting = False

            # Clear the path item so we can start a new stroke
            if self._paint_path_item is not None:
                self._scene.removeItem(self._paint_path_item)
                self._paint_path_item = None

            # Keep paint_points for bounding box calculation
            # They'll be cleared when segmentation runs

        # Releasing erasing
        if self._is_erasing and event.button() == Qt.LeftButton:
            self._is_erasing = False
            return

        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        """Show eraser preview when mouse enters canvas"""
        if self._eraser_enabled and self._eraser_preview is not None:
            self._eraser_preview.setVisible(True)
        # Restore edge trace crosshair and grab focus for key events
        if self._segmentation_mode == "edge_trace":
            for item in self._et_crosshair_items:
                item.setVisible(True)
            self.setFocus()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Hide eraser preview when mouse leaves canvas"""
        if self._eraser_preview is not None:
            self._eraser_preview.setVisible(False)
        # Hide edge trace crosshair
        for item in self._et_crosshair_items:
            item.setVisible(False)
        super().leaveEvent(event)

    def _handle_click(self, scene_pos: QPointF, is_left: bool):
        x, y = scene_pos.x(), scene_pos.y()
        if x < 0 or y < 0:
            return
        if is_left:
            self._on_left_click(x, y)
        else:
            self._on_right_click(x, y)

    def _init_polygon_path_item(self):
        if self._polygon_path_item is None:
            pen = QPen(QColor(255, 255, 0))  # yellow outline, easy to see
            pen.setWidth(2)
            pen.setCosmetic(True)
            self._polygon_path_item = self._scene.addPath(QPainterPath(), pen)
            self._polygon_path_item.setBrush(QBrush(Qt.NoBrush))

    def _update_polygon_path(self):
        if not self._polygon_points or self._polygon_path_item is None:
            return

        path = QPainterPath()
        first_x, first_y = self._polygon_points[0]
        path.moveTo(first_x, first_y)
        for x, y in self._polygon_points[1:]:
            path.lineTo(x, y)

        self._polygon_path_item.setPath(path)

    def _init_paint_path_item(self):
        if self._paint_path_item is None:
            # Green for positive, Red for negative
            color = QColor(255, 0, 0) if self._paint_is_negative else QColor(0, 255, 0)
            pen = QPen(color)
            pen.setWidth(self._brush_size)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            self._paint_path_item = self._scene.addPath(QPainterPath(), pen)

    def _update_paint_path(self):
        if not self._paint_points or self._paint_path_item is None:
            return

        path = QPainterPath()
        first_x, first_y = self._paint_points[0]
        path.moveTo(first_x, first_y)
        for x, y in self._paint_points[1:]:
            path.lineTo(x, y)

        self._paint_path_item.setPath(path)

    # ------------------------------------------------------------------
    # Manual polygon (click-by-click) helpers
    # ------------------------------------------------------------------

    def _add_manual_polygon_vertex(self, x, y):
        self._manual_polygon_points.append((x, y))
        self._init_polygon_path_item()
        self._update_manual_polygon_path()

    def _update_manual_polygon_path(self):
        if not self._manual_polygon_points or self._polygon_path_item is None:
            return
        path = QPainterPath()
        x0, y0 = self._manual_polygon_points[0]
        path.moveTo(x0, y0)
        for x, y in self._manual_polygon_points[1:]:
            path.lineTo(x, y)
        # Do NOT close the path here — closing happens only on double-click/right-click
        self._polygon_path_item.setPath(path)
        self._polygon_path_item.setBrush(QBrush(Qt.NoBrush))

    def _update_manual_rubber_band(self, cx, cy):
        if not self._manual_polygon_points:
            return
        lx, ly = self._manual_polygon_points[-1]
        if self._manual_rubber_band is None:
            from PyQt5.QtWidgets import QGraphicsLineItem
            self._manual_rubber_band = QGraphicsLineItem()
            pen = QPen(QColor(255, 200, 0))  # amber dashed
            pen.setWidth(1)
            pen.setStyle(Qt.DashLine)
            pen.setCosmetic(True)
            self._manual_rubber_band.setPen(pen)
            self._manual_rubber_band.setZValue(9999)
            self._scene.addItem(self._manual_rubber_band)
        self._manual_rubber_band.setLine(lx, ly, cx, cy)

    def _close_manual_polygon(self):
        pts = self._manual_polygon_points.copy()
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        self.polygon_drawn.emit(pts)
        self._cancel_manual_polygon()

    def _cancel_manual_polygon(self):
        self._manual_polygon_points = []
        if self._polygon_path_item is not None:
            self._scene.removeItem(self._polygon_path_item)
            self._polygon_path_item = None
        if self._manual_rubber_band is not None:
            self._scene.removeItem(self._manual_rubber_band)
            self._manual_rubber_band = None

    # ------------------------------------------------------------------
    # Edge Trace helpers
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        """+ / - adjust sample interval;  [ / ] adjust ray reach width.
        Hold 'd' to click-delete fg/bg points."""
        if event.key() == Qt.Key_D and not event.isAutoRepeat():
            self._delete_key_down = True
            self.setCursor(Qt.PointingHandCursor)
            event.accept()
            return

        if self._segmentation_mode == "edge_trace":
            if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
                self._edge_trace_interval = min(200, self._edge_trace_interval + 5)
                self._update_et_hud_text()
                self.edge_trace_interval_changed.emit(self._edge_trace_interval)
                event.accept()
                return
            elif event.key() == Qt.Key_Minus:
                self._edge_trace_interval = max(10, self._edge_trace_interval - 5)
                self._update_et_hud_text()
                self.edge_trace_interval_changed.emit(self._edge_trace_interval)
                event.accept()
                return
            elif event.key() == Qt.Key_BracketRight:
                self._ET_MAX_RAY = min(300, self._ET_MAX_RAY + 5)
                self._update_et_hud_text()
                event.accept()
                return
            elif event.key() == Qt.Key_BracketLeft:
                self._ET_MAX_RAY = max(5, self._ET_MAX_RAY - 5)
                self._update_et_hud_text()
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_D and not event.isAutoRepeat():
            self._delete_key_down = False
            self.setCursor(Qt.CrossCursor if self._eraser_enabled else Qt.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _update_et_hud_text(self):
        self._et_hud.setPlainText(
            f"Interval: {self._edge_trace_interval}px  Width: {self._ET_MAX_RAY}px"
        )
        # Redraw crosshair at its current position if visible
        if self._et_crosshair_items:
            # Get current position from the existing stroke arm midpoint
            item = self._et_crosshair_items[0]
            line = item.line()
            cx = (line.x1() + line.x2()) / 2.0
            cy = (line.y1() + line.y2()) / 2.0
            self._update_et_crosshair(cx, cy)

    def _precompute_canny(self):
        """Compute Canny edge map from the current image numpy array."""
        import cv2
        import numpy as np
        # Walk up to MainWindow to get image_np
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "image_np") and parent.image_np is not None:
                img = parent.image_np
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                self._edge_trace_canny = cv2.Canny(blurred, 40, 120)
                return
            parent = parent.parent()
        self._edge_trace_canny = None

    def _et_sample_segment(self, x0, y0, x1, y1):
        """
        Process a new drag segment. Fire perpendicular Canny rays at each
        sample interval hit, accumulate fg/bg points and edge pixels, and
        refresh the visual overlay.
        """
        import math as _math

        if self._edge_trace_canny is None:
            return

        edges = self._edge_trace_canny
        h, w = edges.shape
        dx = x1 - x0
        dy = y1 - y0
        seg_len = _math.hypot(dx, dy)
        if seg_len < 1e-6:
            return

        tx = dx / seg_len
        ty = dy / seg_len
        px = -ty   # perpendicular left
        py = tx
        self._et_last_tangent = (tx, ty)  # update immediately for crosshair rotation

        self._edge_trace_dist_accum += seg_len
        new_pts_added = False

        while self._edge_trace_dist_accum >= self._edge_trace_interval:
            t = 1.0 - (self._edge_trace_dist_accum - self._edge_trace_interval) / seg_len
            t = max(0.0, min(1.0, t))
            sx = x0 + t * dx
            sy = y0 + t * dy
            self._edge_trace_dist_accum -= self._edge_trace_interval

            # Cast rays and collect edge pixels on both sides
            left_dist, left_pixels = self._cast_ray(sx, sy, px, py, w, h, edges)
            right_dist, right_pixels = self._cast_ray(sx, sy, -px, -py, w, h, edges)

            if left_dist is None or right_dist is None:
                continue  # skip — no edge on one side

            # fg point at stroke center
            self._edge_trace_fg_pts.append((sx, sy))

            # bg points just outside each edge
            bd = left_dist + self._ET_BG_OFFSET
            bx = max(0.0, min(float(w - 1), sx + bd * px))
            by = max(0.0, min(float(h - 1), sy + bd * py))
            self._edge_trace_bg_pts.append((bx, by))

            bd = right_dist + self._ET_BG_OFFSET
            bx = max(0.0, min(float(w - 1), sx - bd * px))
            by = max(0.0, min(float(h - 1), sy - bd * py))
            self._edge_trace_bg_pts.append((bx, by))

            # accumulate edge pixels for overlay
            self._edge_trace_edge_pixels.extend(left_pixels)
            self._edge_trace_edge_pixels.extend(right_pixels)
            new_pts_added = True

        if new_pts_added:
            self._rebuild_et_overlays()

    def _cast_ray(self, sx, sy, dx, dy, w, h, edges):
        """
        Walk outward from (sx, sy) in direction (dx, dy).
        Returns (distance, list_of_edge_pixels) if an edge is hit, else (None, []).
        Collects ALL edge pixels along the ray up to the hit point for display.
        """
        edge_pixels = []
        for d in range(2, self._ET_MAX_RAY + 1):
            rx = int(round(sx + d * dx))
            ry = int(round(sy + d * dy))
            if rx < 0 or ry < 0 or rx >= w or ry >= h:
                break
            if edges[ry, rx] > 0:
                edge_pixels.append((rx, ry))
                return d, edge_pixels
        return None, []

    def _rebuild_et_overlays(self):
        """Rebuild the orange edge pixel overlay and green/red dot items."""
        import numpy as np

        if self._orig_width is None or self._orig_height is None:
            return

        # --- Orange edge pixel overlay via RGBA image ---
        overlay = np.zeros((self._orig_height, self._orig_width, 4), dtype=np.uint8)
        for ex, ey in self._edge_trace_edge_pixels:
            if 0 <= ey < self._orig_height and 0 <= ex < self._orig_width:
                overlay[ey, ex] = [255, 165, 0, 220]  # orange, mostly opaque

        qimage = QImage(
            overlay.data,
            self._orig_width,
            self._orig_height,
            overlay.strides[0],
            QImage.Format_RGBA8888,
        )
        pixmap = QPixmap.fromImage(qimage)

        if self._edge_trace_overlay_item is None:
            from PyQt5.QtWidgets import QGraphicsPixmapItem
            self._edge_trace_overlay_item = self._scene.addPixmap(pixmap)
            self._edge_trace_overlay_item.setZValue(9_000)
            self._edge_trace_overlay_item.setAcceptedMouseButtons(Qt.NoButton)
        else:
            self._edge_trace_overlay_item.setPixmap(pixmap)

        # --- Remove old dot items ---
        for item in self._edge_trace_dot_items:
            self._scene.removeItem(item)
        self._edge_trace_dot_items = []

        # --- Draw fg dots (green) ---
        fg_pen = QPen(QColor(0, 255, 0), 7)
        fg_pen.setCapStyle(Qt.RoundCap)
        for x, y in self._edge_trace_fg_pts:
            item = self._scene.addEllipse(-4, -4, 8, 8,
                                          QPen(QColor(0, 200, 0), 2),
                                          QBrush(QColor(0, 255, 0, 200)))
            item.setFlag(item.ItemIgnoresTransformations, True)
            item.setPos(x, y)
            item.setZValue(9_500)
            item.setAcceptedMouseButtons(Qt.NoButton)
            self._edge_trace_dot_items.append(item)

        # --- Draw bg dots (red) ---
        for x, y in self._edge_trace_bg_pts:
            item = self._scene.addEllipse(-4, -4, 8, 8,
                                          QPen(QColor(200, 0, 0), 2),
                                          QBrush(QColor(255, 0, 0, 200)))
            item.setFlag(item.ItemIgnoresTransformations, True)
            item.setPos(x, y)
            item.setZValue(9_500)
            item.setAcceptedMouseButtons(Qt.NoButton)
            self._edge_trace_dot_items.append(item)

    def _update_et_hud_position(self, x, y):
        """Keep HUD label near cursor, offset so it doesn't overlap."""
        self._et_hud.setPos(x + 14, y - 28)

    def _update_et_crosshair(self, cx, cy):
        """
        Draw a rotated crosshair at (cx, cy) showing:
          - Along-stroke arm: ± INTERVAL/2  (how far to next/prev sample)
          - Perpendicular arm: ± MAX_RAY    (ray reach width)
        Uses the last known tangent direction so it rotates with the stroke.
        """
        from PyQt5.QtWidgets import QGraphicsLineItem

        # Remove old crosshair items
        for item in self._et_crosshair_items:
            self._scene.removeItem(item)
        self._et_crosshair_items = []

        tx, ty = self._et_last_tangent          # along-stroke unit vector
        px, py = -ty, tx                        # perpendicular unit vector

        half_interval = self._edge_trace_interval / 2.0
        ray = float(self._ET_MAX_RAY)

        pen = QPen(QColor(255, 220, 0, 180), 1)  # yellow, semi-transparent
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)

        # Along-stroke arm (interval guide) — spans ± half_interval along stroke
        ax0 = cx - tx * half_interval
        ay0 = cy - ty * half_interval
        ax1 = cx + tx * half_interval
        ay1 = cy + ty * half_interval
        stroke_arm = QGraphicsLineItem(ax0, ay0, ax1, ay1)
        stroke_arm.setPen(pen)
        stroke_arm.setZValue(10_500)
        stroke_arm.setAcceptedMouseButtons(Qt.NoButton)
        self._scene.addItem(stroke_arm)
        self._et_crosshair_items.append(stroke_arm)

        # Perpendicular arm (ray reach guide) — spans ± MAX_RAY perpendicular to stroke
        bx0 = cx - px * ray
        by0 = cy - py * ray
        bx1 = cx + px * ray
        by1 = cy + py * ray
        perp_arm = QGraphicsLineItem(bx0, by0, bx1, by1)
        perp_arm.setPen(pen)
        perp_arm.setZValue(10_500)
        perp_arm.setAcceptedMouseButtons(Qt.NoButton)
        self._scene.addItem(perp_arm)
        self._et_crosshair_items.append(perp_arm)

        # Bracket tick marks at each end of the perpendicular arm
        # Each bracket is two short lines along the stroke axis (like [ and ])
        TICK = 6  # screen pixels for tick length
        for ex, ey, sign in ((bx0, by0, 1), (bx1, by1, -1)):
            # sign flips the inner direction so left end gets ] and right gets [
            t0x = ex + tx * TICK * sign
            t0y = ey + ty * TICK * sign
            t1x = ex - tx * TICK * sign
            t1y = ey - ty * TICK * sign
            for lx0, ly0, lx1, ly1 in ((ex, ey, t0x, t0y), (ex, ey, t1x, t1y)):
                tick = QGraphicsLineItem(lx0, ly0, lx1, ly1)
                tick.setPen(pen)
                tick.setZValue(10_500)
                tick.setAcceptedMouseButtons(Qt.NoButton)
                self._scene.addItem(tick)
                self._et_crosshair_items.append(tick)

    def _init_edge_trace_path_item(self):
        if self._edge_trace_path_item is None:
            pen = QPen(QColor(0, 200, 255))   # cyan-blue
            pen.setWidth(3)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            pen.setCosmetic(True)
            self._edge_trace_path_item = self._scene.addPath(QPainterPath(), pen)
            self._edge_trace_path_item.setZValue(9_800)

    def _update_edge_trace_path(self):
        if not self._edge_trace_points or self._edge_trace_path_item is None:
            return
        path = QPainterPath()
        x0, y0 = self._edge_trace_points[0]
        path.moveTo(x0, y0)
        for x, y in self._edge_trace_points[1:]:
            path.lineTo(x, y)
        self._edge_trace_path_item.setPath(path)