import os
import json
import numpy as np
import cv2

from PyQt5 import QtWidgets, QtGui, QtCore
import pycocotools.mask as maskutils

from GRIME_AI.dialogs.ML_image_processing.mask_visualizer import MaskVisualizer
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor

_JSON_KEY = "COCO_Viewer_JSON_Path"
_K_OPACITY = "COCO_Viewer_Opacity"
_K_BORDER_ON = "COCO_Viewer_BorderEnabled"
_K_BORDER_W = "COCO_Viewer_BorderWidth"
_K_SHOW_ALL = "COCO_Viewer_ShowAllMasks"

# ====================================================================================
# ====================================================================================
#   =====     =====     =====     class CocoViewerTab     =====     =====     =====
# ====================================================================================
# ====================================================================================
class CocoViewerTab(QtWidgets.QWidget):
    """
    PyQt5 widget for viewing images with COCO-style polygon and RLE masks.
    Designed to be embedded in a QTabWidget.
    """

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def __init__(self, folder=None, parent=None):
        """
        Initialize the widget, build the UI, and optionally load a folder.

        Args:
            folder (str, optional): Path containing instances_default.json and images.
            parent (QWidget, optional): Parent widget.
        """
        super().__init__(parent)

        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.installEventFilter(self)  # ensure arrow keys are captured globally

        # State
        self.folder = folder
        self.images = {}
        self.annotations = {}
        self.categories = {}
        self.image_ids = []
        self.current_image_index = 0
        self.current_mask_index = 0
        self.show_all_masks = False
        self.total_masks = 0
        self.masks_per_category = {}
        self._last_rgb = None

        # UI: File selection
        self.file_edit = QtWidgets.QLineEdit()
        self.file_edit.setPlaceholderText("Select instances_default.json...")
        self.file_edit.editingFinished.connect(self._on_file_edit_changed)
        self.browse_button = QtWidgets.QPushButton("Browse JSON")
        self.browse_button.clicked.connect(self.browse_json)

        file_layout = QtWidgets.QHBoxLayout()
        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(self.browse_button)

        # UI: Image display
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.label.setMinimumSize(400, 300)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Mask type info label
        self.mask_type_label = QtWidgets.QLabel("Mask type: N/A")
        self.mask_type_label.setAlignment(QtCore.Qt.AlignLeft)

        # Mask list box (on the right)
        self.mask_list = QtWidgets.QListWidget()
        self.mask_list.setMinimumWidth(250)
        self.mask_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.mask_list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.mask_list.currentRowChanged.connect(self.on_mask_selected)

        # Mask Options groupbox (above the mask list)
        self.toggle_button = QtWidgets.QPushButton("Show All Masks")
        self.toggle_button.clicked.connect(self.toggle_masks)
        self.toggle_button.setStyleSheet(
            "QPushButton { background-color: #4682B4; color: white; "
            "border: none; padding: 6px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3a6d99; }"
        )

        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.opacity_value_label = QtWidgets.QLabel("100%")

        opacity_row = QtWidgets.QHBoxLayout()
        opacity_row.addWidget(QtWidgets.QLabel("Opacity:"))
        opacity_row.addWidget(self.opacity_slider)
        opacity_row.addWidget(self.opacity_value_label)

        self.border_enabled_check = QtWidgets.QCheckBox()
        self.border_enabled_check.setChecked(True)
        self.border_enabled_check.setToolTip("Draw selected-mask border")
        self.border_enabled_check.stateChanged.connect(self._on_border_toggled)
        self._border_enabled = True

        self.border_spin = QtWidgets.QSpinBox()
        self.border_spin.setRange(1, 3)
        self.border_spin.setValue(2)
        self.border_spin.valueChanged.connect(self._on_border_thickness_changed)
        self.border_spin.setEnabled(False)
        self.border_enabled_check.setEnabled(False)
        self._border_thickness = 2

        border_row = QtWidgets.QHBoxLayout()
        border_row.addWidget(self.border_enabled_check)
        border_row.addWidget(QtWidgets.QLabel("Border Width:"))
        border_row.addWidget(self.border_spin)
        border_row.addStretch()

        mask_options_layout = QtWidgets.QVBoxLayout()
        mask_options_layout.addWidget(self.toggle_button)
        mask_options_layout.addLayout(opacity_row)
        mask_options_layout.addLayout(border_row)

        self.mask_options_group = QtWidgets.QGroupBox("Mask Options")
        self.mask_options_group.setLayout(mask_options_layout)

        # Stack: Mask Options, then mask_type_label, then mask_list
        list_layout = QtWidgets.QVBoxLayout()
        list_layout.addWidget(self.mask_options_group)
        list_layout.addWidget(self.mask_type_label)
        list_layout.addWidget(self.mask_list)

        # Image + list side-by-side
        image_and_list_layout = QtWidgets.QHBoxLayout()
        image_and_list_layout.addWidget(self.label, stretch=3)
        image_and_list_layout.addLayout(list_layout, stretch=1)

        # Navigation controls
        self.button_left = QtWidgets.QPushButton("Previous Mask")
        self.button_right = QtWidgets.QPushButton("Next Mask")
        self.button_left.clicked.connect(self.show_prev)
        self.button_right.clicked.connect(self.show_next)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.addWidget(self.button_left)
        controls_layout.addWidget(self.button_right)
        self._mask_opacity = 1.0

        # Diagnostics and stats
        self.status = QtWidgets.QLabel()
        self.stats_label = QtWidgets.QLabel()

        # Thumbnail filmstrip beneath the image canvas
        self.filmstrip = QtWidgets.QListWidget()
        self.filmstrip.setViewMode(QtWidgets.QListView.IconMode)
        self.filmstrip.setFlow(QtWidgets.QListView.LeftToRight)
        self.filmstrip.setWrapping(False)
        self.filmstrip.setMovement(QtWidgets.QListView.Static)
        self.filmstrip.setResizeMode(QtWidgets.QListView.Adjust)
        self.filmstrip.setIconSize(QtCore.QSize(120, 72))
        self.filmstrip.setFixedHeight(104)
        self.filmstrip.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.filmstrip.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.filmstrip.setSpacing(2)
        self.filmstrip.currentRowChanged.connect(self.on_filmstrip_selected)
        self._film_pending = []
        self._film_token = 0

        # Assemble layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(file_layout)
        layout.addLayout(image_and_list_layout)
        layout.addWidget(self.filmstrip)
        layout.addLayout(controls_layout)
        layout.addWidget(self.status)
        layout.addWidget(self.stats_label)

        self.visualizer = MaskVisualizer(category_color_fn=self.category_color, marker_color=(255, 255, 255))

    # --------------------------------------------------------------------------------
    # DELEGATIONS
    # --------------------------------------------------------------------------------
    def overlay_all_masks(self, img, anns):
        return self.visualizer.overlay_all_masks(img, anns)

    def overlay_single_mask(self, img, ann):
        return self.visualizer.overlay_single_mask(img, ann)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def category_color(self, cid):
        """
        Generate a deterministic RGB color for a category ID.

        Args:
            cid (int): Category ID.

        Returns:
            tuple: (R, G, B) color values.
        """
        np.random.seed(cid)
        return tuple(np.random.randint(0, 255, size=3).tolist())

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def eventFilter(self, obj, event):
        """
        Intercept arrow keys to prevent focus navigation and use them for browsing.

        Args:
            obj (QObject): Observed object.
            event (QEvent): Event to inspect.

        Returns:
            bool: True if event is consumed, False to pass through.
        """
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Left:
                self.show_prev()
                return True
            if event.key() == QtCore.Qt.Key_Right:
                self.show_next()
                return True
            if event.key() == QtCore.Qt.Key_Space:
                self.toggle_masks()
                return True
        return super().eventFilter(obj, event)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def browse_json(self):
        """
        Open a file dialog to select a JSON file and load its dataset.
        """
        annotation_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select instances_default.json", "", "JSON Files (*.json)"
        )
        if annotation_file:
            self.file_edit.setText(annotation_file)
            JsonEditor().update_json_entry(_JSON_KEY, annotation_file)
            self.load_json_and_images(annotation_file)

    def _on_file_edit_changed(self):
        path = self.file_edit.text().strip()
        if path and os.path.isfile(path):
            JsonEditor().update_json_entry(_JSON_KEY, path)
            self.load_json_and_images(path)

    def showEvent(self, event):
        """Restore the last-used JSON path and mask options when shown."""
        super().showEvent(event)
        if not getattr(self, "_options_restored", False):
            self._restore_mask_options()
            self._options_restored = True
        if not self.file_edit.text().strip():
            saved = JsonEditor().getValue(_JSON_KEY)
            if saved and os.path.isfile(saved):
                self.file_edit.setText(saved)
                self.load_json_and_images(saved)

    def _restore_mask_options(self):
        je = JsonEditor()
        op = je.getValue(_K_OPACITY)
        if op is not None:
            self.opacity_slider.setValue(int(op))
            self._mask_opacity = int(op) / 100.0
        bw = je.getValue(_K_BORDER_W)
        if bw is not None:
            self.border_spin.setValue(int(bw))
            self._border_thickness = int(bw)
        bo = je.getValue(_K_BORDER_ON)
        if bo is not None:
            self.border_enabled_check.setChecked(bool(bo))
            self._border_enabled = bool(bo)
        sa = je.getValue(_K_SHOW_ALL)
        if bool(sa) and not self.show_all_masks:
            self.toggle_masks()   # flips state, updates labels + gating

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def load_json_and_images(self, annotation_file):
        """
        Load JSON data and initialize images, annotations, categories, and stats.

        Args:
            annotation_file (str): annotation filename with its folder path.
        """
        json_path = annotation_file
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load JSON: {e}")
            return

        self.folder = os.path.dirname(annotation_file)
        self.images = {img["id"]: img for img in data.get("images", [])}

        self.annotations = {}
        for ann in data.get("annotations", []):
            self.annotations.setdefault(ann.get("image_id"), []).append(ann)

        self.categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

        self.image_ids = sorted(self.images.keys())
        self.current_image_index = 0
        self.current_mask_index = 0

        self.total_masks = len(data.get("annotations", []))

        self.update_stats_label()

        self.masks_per_category = {}
        for ann in data.get("annotations", []):
            cid = ann.get("category_id")
            self.masks_per_category[cid] = self.masks_per_category.get(cid, 0) + 1

        # Diagnostics: check mask coverage
        # update stats label first
        self.update_stats_label()

        # now append mask coverage and segmentation info
        mask_check = self.check_images_have_masks(data)
        if mask_check["all_have_masks"]:
            extra_info = "All images have at least one mask."
        else:
            extra_info = f"{len(mask_check['images_without_masks'])} images lack masks."

        seg_type_info = self.analyze_segmentation_types(data)

        # combine everything into one message
        self.stats_label.setText(self.stats_label.text() + " | " + extra_info + " | " + seg_type_info)

        # Warn if any images referenced in the JSON are missing from the folder
        missing_files = [
            img["file_name"]
            for img in self.images.values()
            if not os.path.isfile(os.path.join(self.folder, img["file_name"]))
        ]
        if missing_files:
            missing_list = "\n".join(missing_files[:10])
            if len(missing_files) > 10:
                missing_list += f"\n... and {len(missing_files) - 10} more."
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Image Files",
                f"{len(missing_files)} image(s) referenced in the JSON were not found in:\n"
                f"{self.folder}\n\n"
                f"{missing_list}\n\n"
                f"The JSON and image folder may not match."
            )

        if self.image_ids:
            self.populate_filmstrip()
            self.show_image()
        else:
            self.label.setText("No images found.")

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def analyze_segmentation_types(self, data):
        """
        Inspect annotations in the JSON and report whether they contain
        polygons only, RLE only, or both.
        """
        has_polygons = False
        has_rle = False

        for ann in data.get("annotations", []):
            seg = ann.get("segmentation", None)
            if isinstance(seg, list) and len(seg) > 0:
                has_polygons = True
            elif isinstance(seg, dict):
                has_rle = True

            # Early exit if both found
            if has_polygons and has_rle:
                break

        if has_polygons and has_rle:
            return "Both polygons and RLE present"
        elif has_polygons:
            return "Polygons only"
        elif has_rle:
            return "RLE only"
        else:
            return "No valid segmentations found"

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def check_images_have_masks(self, data):
        """
        Verify that each image in the COCO dataset has at least one annotation.

        Args:
            data (dict): Parsed JSON data from instances_default.json

        Returns:
            dict: {
                "all_have_masks": bool,
                "images_without_masks": list of file names,
                "total_images": int,
                "total_annotations": int
            }
        """
        images = {img["id"]: img for img in data.get("images", [])}
        annotations = data.get("annotations", [])
        annotated_ids = set(ann["image_id"] for ann in annotations)

        missing = [
            img["file_name"] for img_id, img in images.items()
            if img_id not in annotated_ids
        ]

        return {
            "all_have_masks": len(missing) == 0,
            "images_without_masks": missing,
            "total_images": len(images),
            "total_annotations": len(annotations),
        }

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def update_stats_label(self):
        """
        Update the stats label to show total masks and masks per category.
        """
        stats_text = f"Total masks: {self.total_masks}"
        if self.masks_per_category:
            parts = []
            for cid, count in sorted(self.masks_per_category.items()):
                cat_name = self.categories.get(cid, "Unknown")
                parts.append(f"{cat_name} ({cid}): {count}")
            stats_text += " | " + " | ".join(parts)
        self.stats_label.setText(stats_text)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def populate_filmstrip(self):
        """Fill the filmstrip with one thumbnail per image, in image_ids order."""
        self.filmstrip.blockSignals(True)
        self.filmstrip.clear()
        self.filmstrip.blockSignals(False)

        self._film_token += 1
        token = self._film_token
        self._film_pending = []

        for idx, img_id in enumerate(self.image_ids):
            name = self.images[img_id]["file_name"]
            item = QtWidgets.QListWidgetItem(QtGui.QIcon(), "")
            item.setToolTip(name)
            item.setSizeHint(QtCore.QSize(self.filmstrip.iconSize().width() + 8,
                                          self.filmstrip.iconSize().height() + 8))
            self.filmstrip.addItem(item)
            self._film_pending.append((item, os.path.join(self.folder, name), token))

        QtCore.QTimer.singleShot(30, lambda: self._load_film_batch(token))

    def _load_film_batch(self, token):
        if token != self._film_token:
            return
        size = self.filmstrip.iconSize()
        for _ in range(min(16, len(self._film_pending))):
            item, path, tok = self._film_pending.pop(0)
            if tok != self._film_token or not os.path.isfile(path):
                continue
            pix = QtGui.QPixmap(path)
            if pix.isNull():
                continue
            item.setIcon(QtGui.QIcon(pix.scaled(size, QtCore.Qt.KeepAspectRatio,
                                                QtCore.Qt.SmoothTransformation)))
        if self._film_pending:
            QtCore.QTimer.singleShot(30, lambda: self._load_film_batch(token))

    def on_filmstrip_selected(self, row):
        if row is None or row < 0 or row >= len(self.image_ids):
            return
        if row == self.current_image_index:
            return
        self.current_image_index = row
        self.current_mask_index = 0
        self.show_image()

    def show_image(self):
        """
        Render the current image with masks (single or all) into the label.
        Also update the status and mask list.
        """
        if not self.image_ids:
            return

        img_id = self.image_ids[self.current_image_index]
        img_info = self.images[img_id]
        img_path = os.path.join(self.folder, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            self.label.setText(f"Could not load {img_info['file_name']}")
            return

        # Keep filmstrip selection in sync without re-triggering navigation
        if self.filmstrip.currentRow() != self.current_image_index:
            self.filmstrip.blockSignals(True)
            self.filmstrip.setCurrentRow(self.current_image_index)
            self.filmstrip.blockSignals(False)
            item = self.filmstrip.item(self.current_image_index)
            if item:
                self.filmstrip.scrollToItem(item)

        anns = self.annotations.get(img_id, [])

        # Preserve current selection before rebuilding the list
        prev_selected = self.mask_list.currentRow()

        # Populate list box with mask labels; row 0 is always "No mask"
        self.mask_list.blockSignals(True)
        self.mask_list.clear()
        self.mask_list.addItem("No mask")
        for ann in anns:
            cid = ann.get("category_id")
            cat_name = self.categories.get(cid, "Unknown")
            self.mask_list.addItem(f"{cat_name} (ID {cid})")

        if anns and not getattr(self, "_show_no_mask", False):
            if self.show_all_masks:
                target = prev_selected if 1 <= prev_selected < self.mask_list.count() else 1
                self.mask_list.setCurrentRow(target)
                item = self.mask_list.item(target)
                if item:
                    self.mask_list.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
            else:
                self.mask_list.setCurrentRow(self.current_mask_index + 1)
                item = self.mask_list.item(self.current_mask_index + 1)
                if item:
                    item.setSelected(True)
                    self.mask_list.setFocus()
                    self.mask_list.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
        else:
            self.mask_list.setCurrentRow(0)
        self.mask_list.blockSignals(False)

        base_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        show_none = (not self.show_all_masks
                     and (getattr(self, "_show_no_mask", False)
                          or self.mask_list.currentRow() == 0)) or not anns

        if show_none:
            rgb = base_rgb
            self.mask_type_label.setText("Mask type: None")
        else:
            if self.show_all_masks:
                overlayed = self.overlay_all_masks(img, anns)
                selected_row = self.mask_list.currentRow() - 1
                if 0 <= selected_row < len(anns) and getattr(self, "_border_enabled", True):
                    ann = anns[selected_row]
                    overlayed = self.draw_mask_border(overlayed, ann, color=(0, 255, 255), thickness=self._border_thickness)
                    seg = ann.get("segmentation", [])
                    if isinstance(seg, list):
                        self.mask_type_label.setText("Mask type: Polygon")
                    elif isinstance(seg, dict):
                        self.mask_type_label.setText("Mask type: RLE")
                img = overlayed
            else:
                if self.current_mask_index >= len(anns):
                    self.current_mask_index = 0
                ann = anns[self.current_mask_index]
                img = self.overlay_single_mask(img, ann)
                seg = ann.get("segmentation", [])
                if isinstance(seg, list):
                    self.mask_type_label.setText("Mask type: Polygon")
                elif isinstance(seg, dict):
                    self.mask_type_label.setText("Mask type: RLE")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self._mask_opacity < 1.0:
                rgb = cv2.addWeighted(rgb, self._mask_opacity,
                                      base_rgb, 1.0 - self._mask_opacity, 0)

        # Cache once and update once
        self._last_rgb = rgb
        self._update_label_pixmap()

        # Status
        self.status.setText(
            f"Image {self.current_image_index + 1}/{len(self.image_ids)} | "
            f"Masks {self.current_mask_index + 1 if anns and not self.show_all_masks else len(anns) if anns else 0}/"
            f"{len(anns)} | Image ID: {img_id}"
        )

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def draw_mask_border(self, img, ann, color=(0, 255, 255), thickness=4):
        seg = ann.get("segmentation", [])
        if isinstance(seg, list) and len(seg) > 0:
            try:
                polys = seg if isinstance(seg[0], (list, tuple)) else [seg[0]]
                for poly in polys:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(img, [pts], True, color, thickness, lineType=cv2.LINE_8)
            except Exception:
                pass
        elif isinstance(seg, dict):
            try:
                h, w = img.shape[:2]
                counts = seg.get("counts", None)
                if isinstance(counts, list):
                    rle = maskutils.frPyObjects(seg, h, w)
                    m = maskutils.decode(rle)
                else:
                    if "size" not in seg:
                        seg = {"counts": counts, "size": [h, w]}
                    m = maskutils.decode(seg)
                if m is not None:
                    m8 = (m > 0).astype(np.uint8)
                    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cv2.drawContours(img, contours, -1, color, thickness, lineType=cv2.LINE_8)
            except Exception:
                pass
        return img

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def on_mask_selected(self, row):
        """
        Handle selection from the mask list.

        In individual mode, jump to the selected mask.
        In all masks mode, simply redraw to show the border for the selected item.
        """
        if row is None or row < 0:
            return
        anns = self.annotations.get(self.image_ids[self.current_image_index], [])
        if row == 0:            # "No mask"
            self._show_no_mask = True
            self.show_image()
            return
        self._show_no_mask = False
        if not anns:
            return
        if self.show_all_masks:
            self.show_image()
        else:
            self.current_mask_index = max(0, min(row - 1, len(anns) - 1))
            self.show_image()

    def _on_opacity_changed(self, value):
        self._mask_opacity = value / 100.0
        self.opacity_value_label.setText(f"{value}%")
        JsonEditor().update_json_entry(_K_OPACITY, value)
        if self.image_ids:
            self.show_image()

    def _on_border_thickness_changed(self, value):
        self._border_thickness = value
        JsonEditor().update_json_entry(_K_BORDER_W, value)
        if self.image_ids:
            self.show_image()

    def _on_border_toggled(self, state):
        self._border_enabled = bool(state)
        self.border_spin.setEnabled(self._border_enabled and self.show_all_masks)
        JsonEditor().update_json_entry(_K_BORDER_ON, self._border_enabled)
        if self.image_ids:
            self.show_image()

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def show_prev(self):
        """
        Navigate to the previous mask or previous image if needed.
        """
        if not self.image_ids:
            return
        anns = self.annotations.get(self.image_ids[self.current_image_index], [])
        if anns and self.current_mask_index > 0 and not self.show_all_masks:
            self.current_mask_index -= 1
        else:
            if self.current_image_index > 0:
                self.current_image_index -= 1
                anns = self.annotations.get(self.image_ids[self.current_image_index], [])
                self.current_mask_index = max(len(anns) - 1, 0)
        self.show_image()

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def show_next(self):
        """
        Navigate to the next mask or next image if needed.
        """
        if not self.image_ids:
            return
        anns = self.annotations.get(self.image_ids[self.current_image_index], [])
        if anns and self.current_mask_index < len(anns) - 1 and not self.show_all_masks:
            self.current_mask_index += 1
        else:
            if self.current_image_index < len(self.image_ids) - 1:
                self.current_image_index += 1
                self.current_mask_index = 0
        self.show_image()

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def toggle_masks(self):
        """
        Toggle between showing all masks or a single mask.
        Update the button text accordingly.
        """
        self.show_all_masks = not self.show_all_masks
        JsonEditor().update_json_entry(_K_SHOW_ALL, self.show_all_masks)
        if self.show_all_masks:
            self._show_no_mask = False
            self.toggle_button.setText("Show Individual Masks")
            self.button_left.setText("Previous Image")
            self.button_right.setText("Next Image")
        else:
            self.toggle_button.setText("Show All Masks")
            self.button_left.setText("Previous Mask")
            self.button_right.setText("Next Mask")
        self.show_image()
        self._sync_border_controls_enabled()

    def _sync_border_controls_enabled(self):
        on = self.show_all_masks
        self.border_enabled_check.setEnabled(on)
        self.border_spin.setEnabled(on and self.border_enabled_check.isChecked())

    def _update_label_pixmap(self):
        """
        Update the QLabel pixmap from the last RGB frame using current label size.
        """
        if self._last_rgb is None:
            return
        rgb = self._last_rgb
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qimg)
        self.label.setPixmap(
            pm.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    def resizeEvent(self, event):
        """
        Ensure the image scales smoothly when the widget is resized.
        """
        super().resizeEvent(event)
        self._update_label_pixmap()
