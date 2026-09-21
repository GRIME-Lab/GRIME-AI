# annotation_tab.py
# Full port of the Data Annotation tab logic into a dedicated QWidget class.

import os
import json
from pathlib import Path
from typing import List, Dict

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QFileDialog, QListWidgetItem, QMessageBox
from PyQt5.uic import loadUi

from appcore import PROJECT_ROOT
from appcore.utils.resource_utils import ui_path
from appcore.ImageAnnotatorDlg import ImageAnnotatorDialog


class AnnotationTab(QWidget):
    """
    Encapsulates the Data Annotation tab controls, signals, slots, and state.
    This is a direct port of the annotation-related code from ML_ImageProcessingDlg.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Load the dedicated UI for the annotation tab
        loadUi(ui_path("ML_image_processing/annotation_tab.ui"), self)

        # Initialize per-image annotation store
        # Keys: full image path, Values: list of {type, points, label}
        self.annotation_store: Dict[str, List[Dict]] = {}

        # Annotation tab state
        self._annotation_image_paths: List[str] = []

        # --- Filmstrip one‐row configuration (exact port) ---
        filmstrip = self.listWidget_annotationFilmstrip

        # 1. Layout: Left‐to‐right, no wrapping, static movement
        filmstrip.setFlow(QtWidgets.QListView.LeftToRight)
        filmstrip.setWrapping(False)
        filmstrip.setResizeMode(QtWidgets.QListView.Adjust)
        filmstrip.setMovement(QtWidgets.QListView.Static)

        # 2. Scroll bars: only horizontal
        filmstrip.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        filmstrip.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        # 3. Fix height to exactly one icon row
        icon_h = filmstrip.iconSize().height()
        # account for widget frame borders
        frame = filmstrip.frameWidth() * 2
        # if you want a tiny gap above/below, you can include spacing()
        spacing = filmstrip.spacing()
        filmstrip.setFixedHeight(icon_h + frame + spacing)

        # Optional: Prevent the layout from stretching it vertically
        filmstrip.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )

        # Wire annotation-specific connections
        self._wire_connections()

        # install event filter on the annotation‐image label
        self.label_imageAnnotation.installEventFilter(self)

    # ----------------------------------------------------------------------
    # Connections (direct port)
    # ----------------------------------------------------------------------
    def _wire_connections(self):
        # Enter → add to list (lineEdit_annotationCategory + listWidget_annotationCategories)
        self.lineEdit_annotationCategory.returnPressed.connect(
            lambda: self._add_category(self.lineEdit_annotationCategory, self.listWidget_annotationCategories)
        )

        # Browse button
        self.pushButton_annotationBrowse.clicked.connect(self.browse_annotation_folder)

        # Filmstrip click → display image
        self.listWidget_annotationFilmstrip.itemClicked.connect(self.display_annotation_image)

    # ----------------------------------------------------------------------
    # Category entry (direct port)
    # ----------------------------------------------------------------------
    def _add_category(self, lineedit, listwidget):
        text = lineedit.text().strip()
        if text and not listwidget.findItems(text, Qt.MatchExactly):
            listwidget.addItem(text)
        lineedit.clear()

    # ----------------------------------------------------------------------
    # Populate filmstrip (direct port)
    # ----------------------------------------------------------------------
    def populate_annotation_filmstrip(self, folder: str):
        """
        List all supported images in `folder`, fill the filmstrip.
        """
        self.listWidget_annotationFilmstrip.clear()

        # gather all image files
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        paths = sorted(
            Path(folder).glob('*')
        )
        images = [str(p) for p in paths if p.suffix.lower() in exts]
        self._annotation_image_paths = images

        # preload existing .anno.json for each image
        for img_path in images:
            self.load_annotations_for_image(img_path)

        # update the category list based on preloaded data
        self._refresh_annotation_categories()         # NEW: populate categories

        icon_size = self.listWidget_annotationFilmstrip.iconSize()
        for idx, img_path in enumerate(images):
            pix = QPixmap(img_path)
            if pix.isNull():
                continue
            thumb = pix.scaled(icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QListWidgetItem(QIcon(thumb), "")
            # store index so we can look it up on click
            item.setData(Qt.UserRole, idx)
            self.listWidget_annotationFilmstrip.addItem(item)

        # auto-select first
        if self.listWidget_annotationFilmstrip.count():
            self.listWidget_annotationFilmstrip.setCurrentRow(0)
            # display it immediately
            self.display_annotation_image(self.listWidget_annotationFilmstrip.currentItem())

    # ----------------------------------------------------------------------
    # Display image (exact line-for-line port)
    # ----------------------------------------------------------------------
    def display_annotation_image(self, item):
        """
        Show the full image corresponding to the clicked thumbnail.
        """
        idx = item.data(Qt.UserRole)
        path = self._annotation_image_paths[idx]
        pix = QPixmap(path)
        if pix.isNull():
            return

        # scale to fit the label, keep aspect ratio
        lbl = self.label_imageAnnotation
        scaled = pix.scaled(
            lbl.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        lbl.setPixmap(scaled)

    # ----------------------------------------------------------------------
    # Open annotator dialog (direct port)
    # ----------------------------------------------------------------------
    def _openAnnotator(self):
        pm = self.label_imageAnnotation.pixmap()
        if pm is None:
            QMessageBox.warning(self, "No Image", "No image is loaded.")
            return

        # Determine drawing mode
        if self.radioButton_boundingBox.isChecked():
            mode = 'bbox'
        elif self.radioButton_polylineClick.isChecked():
            mode = 'click'
        else:
            mode = 'drag'

        # Figure out which category is selected in your Annotation tab
        current = self.listWidget_annotationCategories.currentItem()
        if current:
            name = current.text()
            idx  = self.listWidget_annotationCategories.currentRow() + 1
        else:
            name = "<unknown>"
            idx  = -1

        # Load any existing shapes from disk
        current_item = self.listWidget_annotationFilmstrip.currentItem()
        img_idx = current_item.data(Qt.UserRole)
        img_path = self._annotation_image_paths[img_idx]
        img_path = os.path.normpath(img_path)  # normalizes slashes
        img_path = img_path.replace("\\", "/")  # force-forward slashes

        # DEBUG: Print resolved image path and store state
        print(f"[DEBUG] Current image index: {img_idx}")
        print(f"[DEBUG] Resolved image path:\n  {img_path}")
        print(f"[DEBUG] Annotation store keys:\n  {list(self.annotation_store.keys())}")

        # Launch dialog, passing along the label ID & name
        dlg = ImageAnnotatorDialog(
            pm,
            mode=mode,
            label={"id": idx, "name": name},
            parent=self
        )

        # Preload into the annotator
        existing = self.annotation_store.get(img_path)
        print(f"[DEBUG] Loaded {len(existing) if existing else 0} shapes for this image.")
        print(f"[DEBUG] Opening image: {img_path}")
        print(f"[DEBUG] Stored shapes: {len(existing) if existing else 0}")
        if existing:
            dlg.setAnnotations(existing)

        if dlg.exec_():
            shapes = dlg.getAnnotations()
            self.annotation_store[img_path] = shapes
            self.save_annotations_for_image(img_path)
            print(f"[DEBUG] Saved {len(shapes)} shapes for {img_path}")

    # ----------------------------------------------------------------------
    # Save annotations per image (direct port)
    # ----------------------------------------------------------------------
    def save_annotations_for_image(self, img_path: str):
        """
        Writes the annotation JSON for a single image into
        an 'annotations/' subfolder next to the image.
        """
        shapes = self.annotation_store.get(img_path, [])
        if not shapes:
            return

        img_p = Path(img_path)
        ann_folder = img_p.parent / "annotations"
        try:
            ann_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Could not create annotation folder {ann_folder}: {e}")

        ann_file = ann_folder / f"{img_p.stem}.anno.json"
        try:
            with open(ann_file, "w", encoding="utf-8") as f:
                json.dump(shapes, f, indent=2)
            print(f"Saved annotations to {ann_file}")
        except Exception as e:
            print(f"Failed to save annotations to {ann_file}: {e}")

    # ----------------------------------------------------------------------
    # Load annotations per image (direct port)
    # ----------------------------------------------------------------------
    def load_annotations_for_image(self, img_path: str):
        """
        Loads the annotation JSON for a single image from
        the 'annotations/' subfolder, if it exists.
        """
        img_p = Path(img_path)
        ann_file = img_p.parent / "annotations" / f"{img_p.stem}.anno.json"
        if not ann_file.exists():
            return

        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                shapes = json.load(f)
            # store under the image key (string)
            self.annotation_store[img_path] = shapes
            print(f"Loaded annotations from {ann_file}")
        except Exception as e:
            print(f"Failed to load annotations from {ann_file}: {e}")

    # ----------------------------------------------------------------------
    # Refresh category list from store (direct port)
    # ----------------------------------------------------------------------
    def _refresh_annotation_categories(self):
        """
        Scan self.annotation_store for all loaded shapes,
        extract unique (id, name) pairs and populate the category list.
        """
        unique = {}
        for shapes in self.annotation_store.values():
            for s in shapes:
                lab = s.get("label", {})
                cid = lab.get("id", None)
                name = lab.get("name", None)
                if cid is not None and name:
                    unique[cid] = name

        self.listWidget_annotationCategories.clear()
        for cid, name in sorted(unique.items()):
            self.listWidget_annotationCategories.addItem(name)

    # ----------------------------------------------------------------------
    # Preload from COCO (direct port)
    # ----------------------------------------------------------------------
    def preload_annotation_store_from_coco(self, folder: str):
        """
        Reads instances_default.json from `folder` and populates self.annotation_store.
        Structure:
            self.annotation_store[full_image_path] = [shape1, shape2, ...]
        Each shape includes 'type', 'points', and 'label'.
        """
        json_path = os.path.join(folder, "instances_default.json")
        if not os.path.exists(json_path):
            print(f"No annotation file found at {json_path}")
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            return

        image_id_map = {
            img["id"]: os.path.normpath(os.path.join(folder, img["file_name"])).replace("\\", "/")
            for img in data.get("images", [])
        }
        category_map = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

        self.annotation_store.clear()

        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            cat_id = ann.get("category_id")
            segmentation = ann.get("segmentation", [[]])[0]

        # unchanged logic: guard and point extraction
            if img_id not in image_id_map or cat_id not in category_map or not segmentation:
                continue

            img_path = image_id_map[img_id]
            label = {"id": cat_id, "name": category_map[cat_id]}

            pts = []
            for i in range(0, len(segmentation), 2):
                x, y = segmentation[i], segmentation[i + 1]
                pts.append(QtCore.QPointF(x, y))

            shape = {
                "type": "drag",  # ← patched type per original
                "points": pts,
                "label": label
            }

            if img_path not in self.annotation_store:
                self.annotation_store[img_path] = []

            self.annotation_store[img_path].append(shape)

        print(f"Preloaded {len(self.annotation_store)} images from instances_default.json")

    # ----------------------------------------------------------------------
    # Browse and load (direct port, self-contained)
    # ----------------------------------------------------------------------
    def browse_annotation_folder(self):
        """
        Open dialog, select a folder, populate line edit and filmstrip.
        """
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Annotation Folder",
            str(PROJECT_ROOT)
        )
        if not folder:
            return

        self.lineEdit_annotationFolder.setText(folder)

        labels = self._load_labels_from_annotation(folder)
        self.listWidget_annotationCategories.clear()
        if labels:
            self.listWidget_annotationCategories.addItems(labels)
            print(f"Loaded {len(labels)} label(s) from annotations.json.")

        # Load annotations into internal store
        self.preload_annotation_store_from_coco(folder)

        self.populate_annotation_filmstrip(folder)

    # Localized version to avoid coupling to the parent dialog
    def _load_labels_from_annotation(self, folder_path: str) -> List[str]:
        """
        Reads categories from instances_default.json under folder_path
        and returns ['id - name'] entries.
        """
        annotation_file = os.path.join(folder_path, "instances_default.json")
        if not os.path.exists(annotation_file):
            return []

        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        labels = set()
        if "annotations" in data and "categories" in data:
            for cat in data["categories"]:
                labels.add(f"{cat['id']} - {cat['name']}")
        return sorted(labels)

    # ----------------------------------------------------------------------
    # Add category via combo (direct port)
    # ----------------------------------------------------------------------
    def add_annotation_category(self):
        """
        Called when the user presses Enter in the annotation combo's line edit.
        Adds the current text as a new category if it’s non-empty and not already present.
        """
        text = self.comboBox_annotationCategories.currentText().strip()
        if not text:
            return

        # Only add unique, non-empty entries
        if self.comboBox_annotationCategories.findText(text) == -1:
            self.comboBox_annotationCategories.addItem(text)

        # Clear the editor so it's ready for the next entry
        self.comboBox_annotationCategories.lineEdit().clear()

    # ----------------------------------------------------------------------
    # Event filter (double-click opens annotator) — direct port
    # ----------------------------------------------------------------------
    def eventFilter(self, source, event):
        if source == self.label_imageAnnotation and event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            self._openAnnotator()
            return True

        return super().eventFilter(source, event)

    # ----------------------------------------------------------------------
    # Debug Helper: Dump all stored annotations — direct port
    # ----------------------------------------------------------------------
    def print_all_annotations(self):
        """
        Prints every image path and the count/details of shapes stored
        in self.annotation_store for debugging.
        """
        if not self.annotation_store:
            print("No annotations have been recorded yet.")
            return

        for img_path, shapes in self.annotation_store.items():
            print(f"\nAnnotations for {img_path}:")
            for idx, shape in enumerate(shapes, start=1):
                pts = shape['points']
                print(f"  {idx}. {shape['type']} with {len(pts)} points → {pts}")
