import os
import shutil
from datetime import datetime

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import json

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem, QAbstractItemView, QSizePolicy, QListWidget, QMessageBox, QTreeWidget, QTreeWidgetItem

from GRIME_AI import PROJECT_ROOT
from GRIME_AI.GRIME_AI_CSS_Styles import BUTTON_CSS_STEEL_BLUE, BUTTON_CSS_DARK_RED, BUTTON_CSS_YELLOW, BUTTON_CSS_RED_OUTLINE, BUTTON_CSS_YELLOW_OUTLINE
from PyQt5.QtGui import QPalette, QColor, QIcon

try:
    from GRIME_AI.dialogs.ML_image_processing.chain_link_resources import get_icon as _get_chain_icon
except Exception:
    _get_chain_icon = None
from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI.dialogs.ML_image_processing.model_config_manager import ModelConfigManager

# Optional: if there is a training entry point, import it. Replace with the actual path/class.
#try:
#    from GRIME_AI.training.trainer import Trainer
#except Exception:
#    Trainer = None


# ============================================================================
# ============================================================================
# ===                        MODULE LEVEL HELPERS                          ===
# ============================================================================
# ============================================================================
def _check_folder(folder: Path) -> Tuple[bool, List[str], object, List[str], List[str]]:
    """
    Validates a single folder by:
      1. Finding at least one .json COCO file and some .jpg/.jpeg images.
      2. Parsing the JSON images list of dicts to pull out file_name.
      3. Verifying every listed file_name exists in that folder.
      4. Checking for annotations referencing image_ids not in the images list.
      5. Checking for on-disk images with no entry in the JSON images list.

    Returns (is_valid, missing_files_list, json_path, orphan_annotations, unannotated_files).
      - missing_files_list: JSON images not found on disk (hard error).
      - orphan_annotations: annotation entries whose image_id has no matching image entry (hard error).
      - unannotated_files: on-disk files not referenced in the JSON (warning only).
    """
    # 1) List JSONs and JPGs via os.scandir
    jsons = [e.name for e in os.scandir(folder)
             if e.is_file() and e.name.lower() == "instances_default.json"]
    jpgs = {e.name for e in os.scandir(folder)
            if e.is_file() and e.name.lower().endswith((".jpg", ".jpeg"))}

    print(f"Scanning `{folder}` -> JSONs: {jsons}, JPGs: {list(jpgs)[:5]}")  # debug

    if not jsons or not jpgs:
        return False, [], None, [], []

    # 2) Load the first JSON file
    path_json = folder / jsons[0]
    try:
        data = json.loads(path_json.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"Cannot parse {jsons[0]}: {e}"], path_json, [], []

    # 3) Extract expected filenames and image_ids from COCO "images" list
    raw_images = data.get("images")
    if not isinstance(raw_images, list):
        return False, [f"'images' key missing or not a list in {jsons[0]}"], path_json, [], []

    expected_files = []
    valid_image_ids = set()
    for item in raw_images:
        if isinstance(item, dict):
            fname = item.get("file_name") or item.get("filename")
            if not fname:
                return False, [f"Missing 'file_name' in entry: {item}"], path_json, [], []
            expected_files.append(Path(fname).name)
            if "id" in item:
                valid_image_ids.add(item["id"])
        elif isinstance(item, str):
            expected_files.append(item)
        else:
            return False, [f"Unsupported image entry type: {type(item)}"], path_json, [], []

    # 4) Check JSON images vs disk
    missing = [f for f in expected_files if f not in jpgs]

    # 5) Check for orphan annotations (image_id not in images list)
    raw_annotations = data.get("annotations", [])
    orphan_ann_ids = [
        ann.get("id", "?")
        for ann in raw_annotations
        if ann.get("image_id") not in valid_image_ids
    ]
    orphan_annotations = [f"annotation id={aid}" for aid in orphan_ann_ids]

    # 6) Check for on-disk files not in JSON (warn only)
    expected_set = set(expected_files)
    unannotated_files = sorted(f for f in jpgs if f not in expected_set)

    is_valid = not missing and not orphan_annotations
    return is_valid, missing, path_json, orphan_annotations, unannotated_files


def _iter_dirs(root: Path):
    """
    Recursively yield every subdirectory under root using os.scandir.
    """
    # Skip system/Conda directories
    bad = ["anaconda3", "miniconda3", "ProgramData", "Windows"]
    if any(b in str(root).lower() for b in bad):
        return

    if not root.exists():
        return

    for entry in os.scandir(root):
        if entry.is_dir():
            sub = Path(entry.path)
            yield sub
            yield from _iter_dirs(sub)

# ============================================================================
# ============================================================================
# ===                     class DraggableTreeWidget                        ===
# ============================================================================
# ============================================================================
# Custom TreeWidget classes with drag and drop support.
# Only top-level (parent) nodes are draggable — child label nodes are not.
class DraggableTreeWidget(QTreeWidget):
    def mimeData(self, items):
        mimeData = QtCore.QMimeData()
        # Only drag top-level items (folder nodes), ignore children
        top_level_texts = sorted(set(
            item.text(0) for item in items if item.parent() is None
        ))
        mimeData.setText("\n".join(top_level_texts))
        return mimeData

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        # Only allow drag initiation on top-level items
        if item is not None and item.parent() is not None:
            # Child node — don't start a drag, just expand/collapse
            self.setDragEnabled(False)
        else:
            self.setDragEnabled(True)
        super().mousePressEvent(event)


# ============================================================================
# ============================================================================
# ===                     class DroppableTreeWidget                        ===
# ============================================================================
# ============================================================================
class DroppableTreeWidget(QTreeWidget):
    def dropEvent(self, event):
        if event.mimeData().hasText():
            text = event.mimeData().text()
            items_to_drop = [line.strip() for line in text.splitlines() if line.strip()]
            dlg = self.parent()
            if dlg is not None:
                for item_text in items_to_drop:
                    # Remove matching top-level item from available tree
                    available = dlg.listWidget_availableFolders
                    root = available.invisibleRootItem()
                    for idx in range(root.childCount()):
                        avail_item = root.child(idx)
                        if avail_item.text(0) == item_text:
                            root.removeChild(avail_item)
                            break
                    # Add to selected tree if not already there
                    if item_text not in dlg.transferred_items:
                        dlg._add_folder_to_tree(self, item_text)
                        dlg.transferred_items.add(item_text)
                        print(f"Dragged '{item_text}' from available to selected folders via drop.")
                dlg.listWidget_selectedFolders.repaint()
                dlg._refresh_annotations_from_selection()
                dlg.updateTrainButtonState()
            event.accept()
        else:
            event.ignore()


# ============================================================================
# ============================================================================
# ===                       class ClickableLabel                           ===
# ============================================================================
# ============================================================================
class ClickableLabel(QtWidgets.QLabel):
    """
    A QLabel that toggles expand/collapse on all items in a linked QTreeWidget
    when clicked. Displays a chevron (▶ collapsed, ▼ expanded) appended to the
    base text, and shows a tooltip hinting the action.
    """
    def __init__(self, base_text: str, tree: QTreeWidget, parent=None):
        super().__init__(parent)
        self._base_text = base_text
        self._tree = tree
        self._expanded = False  # start collapsed
        self._refresh()
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def _refresh(self):
        chevron = "▼" if self._expanded else "▶"
        self.setText(f"{self._base_text}  {chevron}")
        self.setToolTip("Collapse all" if self._expanded else "Expand all")

    def mousePressEvent(self, event):
        self._expanded = not self._expanded
        if self._expanded:
            self._tree.expandAll()
        else:
            self._tree.collapseAll()
        self._refresh()
        super().mousePressEvent(event)


# ============================================================================
# ============================================================================
# ===                         class TrainingTab                            ===
# ============================================================================
# ============================================================================
class TrainingTab(QtWidgets.QWidget):
    ml_train_signal = QtCore.pyqtSignal()
    # Emitted when the USER commits a change to the training images folder
    # (Browse or edit-finished) so the main window can reconcile with the recipe.
    trainingImagesCommitted_Signal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        ui_path = Path(__file__).parent / "training_tab.ui"
        uic.loadUi(str(ui_path), self)

        # Expose named layouts as attributes (uic only creates widget attrs)
        try:
            gb = self.groupBox_imageFolders
            hl = gb.layout()  # horizontalListLayout
            self.horizontalListLayout    = hl
            self.verticalLayout_available = hl.itemAt(0).layout()
            self.verticalLayout_selected  = hl.itemAt(2).layout()
        except Exception:
            pass

        # Replace default QListWidgets with drag/drop versions while preserving object names and layout positions
        self._install_drag_drop_lists()

        # State holders
        self.categories_available: bool = False
        self.annotation_list: List[Dict[str, Any]] = []
        self.unique_training_labels: List[str] = []
        self.transferred_items: Set[str] = set()
        self.original_folders = []
        self.categories_available = False
        self._folder_validation_state: Dict[str, str] = {}  # folder name -> 'ok' | 'red' | 'yellow'

        # Detect optional labels list widget
        self._init_labels_widget_reference()

        # Init config manager
        settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
        config_file = Path(settings_folder) / "site_config.json"
        self._mgr = ModelConfigManager(str(config_file))

        # Load config (creates/writes template if missing/empty/invalid)
        self.site_config: Dict[str, Any] = self._mgr.load_config(return_type="dict")

        # --- LoRA UI: keep all LoRA controls together so we can enable/disable them ---
        self._lora_widgets = []

        # Main LoRA hyperparam group
        if hasattr(self, "groupBox_loraHyperparameters"):
            self._lora_widgets.append(self.groupBox_loraHyperparameters)

        # Start with LoRA controls disabled; they will be enabled only when LoRA is selected
        self._set_lora_enabled(False)

        # Default selection — must be set before _populate_ui_from_config
        # so _update_context_sensitive_ui knows which model is active
        self.selected_training_model = "sam2"

        # Populate UI from config and initialize state
        self._populate_ui_from_config(self.site_config)
        #self.setup_custom_list_widgets()
        self.setup_ui_properties()
        self.setup_drag_and_drop()
        self.setup_connections()

        self.reset_selection()

        #JES FIX CONDA SCANNING ISSUE
        #JES self.populate_available_folders()

        self.updateTrainButtonState()

        # Apply context-sensitive UI for the initial model selection
        self._update_context_sensitive_ui()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def setup_ui_properties(self):
        """Set size policies and layout stretch factors."""
        self.listWidget_availableFolders.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.listWidget_availableFolders.setMinimumHeight(120)

        self.listWidget_selectedFolders.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.listWidget_selectedFolders.setMinimumHeight(120)

        # Device — GPU is mandatory for training; auto-detect, no user choice
        self.comboBox_device.clear()
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except Exception:
            gpu_available = False

        if gpu_available:
            self.comboBox_device.addItem("gpu")
        else:
            self.comboBox_device.addItem("gpu (not detected)")
            self.pushButton_train.setEnabled(False)
            self.pushButton_train.setToolTip(
                "No CUDA-capable GPU detected. Training requires a GPU."
            )
        try:
            self.verticalLayout_available.setStretch(0, 0)  # label — fixed
            self.verticalLayout_available.setStretch(1, 1)  # tree  — expand
            self.verticalLayout_selected.setStretch(0, 0)
            self.verticalLayout_selected.setStretch(1, 1)
            self.verticalLayout_available.setAlignment(QtCore.Qt.AlignTop)
            self.verticalLayout_selected.setAlignment(QtCore.Qt.AlignTop)
        except Exception:
            pass

        self.adjustSize()
        self.setMinimumSize(self.size())

        self.verticalTabParametersLayout.setStretch(0, 0)  # path bar — fixed
        self.verticalTabParametersLayout.setStretch(1, 1)  # folder groupbox — expand
        self.verticalTabParametersLayout.setStretch(2, 0)  # bottom cols — fixed

        self.horizontalListLayout.setStretch(0, 1)
        self.horizontalListLayout.setStretch(1, 0)
        self.horizontalListLayout.setStretch(2, 1)

        # Bottom three columns: col1 | col2 (params+blob) | col3 (lora+splits)
        self.horizontalMainLayout.setStretch(0, 1)
        self.horizontalMainLayout.setStretch(1, 2)
        self.horizontalMainLayout.setStretch(2, 1)

        # Cap spinbox widths so they don't stretch full column width
        for sb in [
            self.spinBox_epochs, self.spinBox_batchSize,
            self.spinBox_saveFrequency, self.spinBox_validationFrequency,
            self.spinBox_patience, self.spinBox_blobFilterRadius,
            self.spinBox_loraRank, self.spinBox_loraAlpha,
            self.spinBox_trainSplit, self.spinBox_valSplit,
        ]:
            sb.setMaximumWidth(90)

        self.doubleSpinBox_weightDecay.setMaximumWidth(110)
        self.doubleSpinBox_loraDropout.setMaximumWidth(110)

        # Chain link toggle for Training Splits
        self._split_linked = True
        self._split_guard  = False

        self._btn_split_link = QtWidgets.QPushButton()
        self._btn_split_link.setCheckable(True)
        self._btn_split_link.setChecked(True)
        self._btn_split_link.setFixedSize(28, 28)
        self._btn_split_link.setToolTip("Linked: changing one value sets the other to its complement (100 - n).\nClick to unlink.")
        # Load icons from resource module; fall back to text if unavailable
        if _get_chain_icon:
            self._icon_linked   = QIcon(_get_chain_icon("linked"))
            self._icon_unlinked = QIcon(_get_chain_icon("unlinked"))
            self._btn_split_link.setIcon(self._icon_linked)
            self._btn_split_link.setIconSize(QtCore.QSize(20, 20))
        else:
            self._btn_split_link.setText("🔗")
        self._btn_split_link.setStyleSheet("""
QPushButton {
    font-size: 14px;
    border: 1px solid #aaa;
    border-radius: 6px;
    background: transparent;
}
QPushButton:checked {
    border: 2px solid steelblue;
    background: rgba(70,130,180,0.15);
}
QPushButton:hover { background: rgba(128,128,128,0.15); }
""")
        self._btn_split_link.toggled.connect(self._on_split_link_toggled)
        try:
            self.horizontalLayout_splitRow.addWidget(self._btn_split_link)
        except Exception:
            pass

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def setup_drag_and_drop(self):
        """Configure drag & drop for folder lists and set style for the Segment button."""
        self.listWidget_availableFolders.setDragEnabled(True)
        self.listWidget_availableFolders.setDragDropMode(QAbstractItemView.DragOnly)
        self.listWidget_selectedFolders.setAcceptDrops(True)
        self.listWidget_selectedFolders.setDragDropMode(QAbstractItemView.DropOnly)
        self.listWidget_selectedFolders.installEventFilter(self)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def eventFilter(self, source, event):
        """
        Process drag-and-drop events on the selected folders list.
        """
        if source == self.listWidget_selectedFolders:
            if event.type() in (QtCore.QEvent.Type.DragEnter, QtCore.QEvent.Type.DragMove):
                event.accept()
                return True
            elif event.type() == QtCore.QEvent.Type.Drop:
                if event.mimeData().hasText():
                    mime_text = event.mimeData().text()
                    dragged_items = [txt.strip() for txt in mime_text.splitlines() if txt.strip()]
                    avail_root = self.listWidget_availableFolders.invisibleRootItem()
                    for txt in dragged_items:
                        for idx in range(avail_root.childCount()):
                            avail_item = avail_root.child(idx)
                            if avail_item.text(0) == txt:
                                avail_root.removeChild(avail_item)
                                break
                        if txt not in self.transferred_items:
                            self._add_folder_to_tree(self.listWidget_selectedFolders, txt)
                            self.transferred_items.add(txt)
                            print(f"Dragged '{txt}' from available to selected folders via eventFilter.")
                event.accept()
                self._refresh_annotations_from_selection()
                self.updateTrainButtonState()
                return True

        return super().eventFilter(source, event)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    '''
    def setup_ui_properties(self):
        """Set size policies and layout stretch factors."""
        self.tabWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.listWidget_availableFolders.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.listWidget_selectedFolders.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.listWidget_availableFolders.setMinimumHeight(200)
        self.listWidget_selectedFolders.setMinimumHeight(200)
        self.adjustSize()
        self.setMinimumSize(self.size())
        self.verticalTabParametersLayout.setStretch(0, 1)
        self.verticalTabParametersLayout.setStretch(1, 0)
        self.horizontalMainLayout.setStretch(0, 1)
        self.horizontalMainLayout.setStretch(1, 3)
        self.horizontalListLayout.setStretch(0, 1)
        self.horizontalListLayout.setStretch(1, 0)
        self.horizontalListLayout.setStretch(2, 1)

        # Set stylesheet for the tabs to change color when a tab is selected.
        self.tabWidget.setStyleSheet("""
            QTabBar::tab {
                background-color: white;
                color: black;
            }
            QTabBar::tab:selected {
                background-color: steelblue;
                color: white;
            }
        """)
    '''

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _install_clickable_labels(self) -> None:
        """
        Insert a plain bold QLabel above each folder tree as a title,
        and hide the tree's own column header.
        """
        for layout_name, tree, base_text in [
            ("verticalLayout_available", self.listWidget_availableFolders, "Available Image Folders"),
            ("verticalLayout_selected",  self.listWidget_selectedFolders,  "Selected Image Folders"),
        ]:
            tree.setHeaderHidden(True)
            tree.setRootIsDecorated(True)
            tree.setUniformRowHeights(False)

            layout = getattr(self, layout_name, None)
            if layout is None:
                continue

            idx = -1
            for j in range(layout.count()):
                it = layout.itemAt(j)
                if it and it.widget() == tree:
                    idx = j
                    break

            if idx >= 0:
                lbl = QtWidgets.QLabel(base_text, tree.parent())
                lbl.setAlignment(QtCore.Qt.AlignCenter)
                lbl.setStyleSheet("font: bold 10pt;")
                layout.insertWidget(idx, lbl)
                layout.setStretch(idx,     0)
                layout.setStretch(idx + 1, 1)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def setup_connections(self):
        """Connect signals with their slot methods."""

        # Replace plain labels with ClickableLabel for expand/collapse toggle
        self._install_clickable_labels()

        # Wire up signals
        self.pushButton_reset.clicked.connect(self.reset_selection)
        self.pushButton_moveRight.clicked.connect(self.move_to_right)
        self.pushButton_moveLeft.clicked.connect(self.move_to_available)
        self.pushButton_validate.clicked.connect(self.validate_label_consistency)
        self.pushButton_validate.setStyleSheet(BUTTON_CSS_YELLOW_OUTLINE)
        self.lineEdit_siteName.editingFinished.connect(self.save_site_name_to_json)

        self.pushButton_browse_model_training_images_folder.clicked.connect(self.browse_model_training_images_folder)
        model_training_image_folder = JsonEditor().getValue("Model_Training_Images_Folder")
        if model_training_image_folder:
            self.lineEdit_model_training_images_path.setText(model_training_image_folder)
        # The active recipe (if any) is authoritative for the training-images
        # root, overriding the site-config value loaded above.
        try:
            from GRIME_AI.recipe_manager import RecipeStore
            _active = RecipeStore().get_active()
            if _active is not None and getattr(_active, "ml_images", ""):
                self.lineEdit_model_training_images_path.setText(_active.ml_images)
        except Exception as _e:
            print(f"[WARN] Recipe training-images lookup skipped: {_e}")
        self.lineEdit_model_training_images_path.editingFinished.connect(self.populate_available_folders)
        self.lineEdit_model_training_images_path.editingFinished.connect(self._training_images_committed)

        self.pushButton_moveRight.clicked.connect(self.move_to_right)
        self.pushButton_moveRight.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_moveLeft.clicked.connect(self.move_to_left)
        self.pushButton_moveLeft.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.pushButton_reset.clicked.connect(self.reset_lists)
        self.pushButton_reset.setStyleSheet(BUTTON_CSS_RED_OUTLINE)

        self.pushButton_train.clicked.connect(self.train)
        self.pushButton_train.setStyleSheet(BUTTON_CSS_STEEL_BLUE)

        self.listWidget_availableFolders.itemDoubleClicked.connect(self.handle_left_item_doubleclick)

        self.listWidget_selectedFolders.itemDoubleClicked.connect(self.handle_right_item_doubleclick)
        self.listWidget_selectedFolders.itemSelectionChanged.connect(self.updateTrainButtonState)

        # Hold-out seasons: double-click moves items between Available and Hold-Out lists
        self.listWidget_availableSeasons.itemDoubleClicked.connect(self._on_available_season_doubleclick)
        self.listWidget_holdoutSeasons.itemDoubleClicked.connect(self._on_holdout_season_doubleclick)

        # Fix height of season list widgets to show exactly 4 rows (max seasons)
        _row_h = self.listWidget_availableSeasons.sizeHintForRow(0)
        if _row_h < 1:
            _row_h = 22  # fallback if no items loaded yet
        _season_list_h = (_row_h * 4) + self.listWidget_availableSeasons.frameWidth() * 2 + 4
        self.listWidget_availableSeasons.setFixedHeight(_season_list_h)
        self.listWidget_holdoutSeasons.setFixedHeight(_season_list_h)

        # DEBUG: log every model-level insert/remove on Hold-Out listbox to catch drag-drops,
        # programmatic clears, or anything else that modifies the listbox contents.
        ho_model = self.listWidget_holdoutSeasons.model()
        ho_model.rowsInserted.connect(self._debug_holdout_rows_inserted)
        ho_model.rowsAboutToBeRemoved.connect(self._debug_holdout_rows_removed)

        self.comboBox_train_label_selection.currentIndexChanged.connect(self.on_train_label_changed)

        # Blob filter radius — update companion label whenever value changes
        self.spinBox_blobFilterRadius.valueChanged.connect(self._update_blob_filter_pct_label)
        self.radioButton_blobRadiusComputed.setChecked(True)

        # Train/val split — either spinbox drives the other, always sum to 100
        self.spinBox_valSplit.valueChanged.connect(self._on_val_split_changed)
        self.spinBox_trainSplit.valueChanged.connect(self._on_train_split_changed)

        # Hold-out seasons dual listbox — no connections needed; drag-drop is handled by Qt
        self._init_holdout_season_lists()

        # YOLO base weights combobox
        self._populate_yolo_weights_combobox()
        self.pushButton_refreshYoloWeights.clicked.connect(self._populate_yolo_weights_combobox)
        self.pushButton_refreshYoloWeights.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        self.comboBox_yoloWeights.currentIndexChanged.connect(self.update_model_config)
        self.comboBox_yoloWeights.currentIndexChanged.connect(self._update_train_button_state)

        # Connect signals
        self.radioButton_train_model_SAM2.toggled.connect(lambda checked: self.set_training_model("sam2", checked))
        self.radioButton_train_model_segformer.toggled.connect(lambda checked: self.set_training_model("segformer", checked))
        # [MASKRCNN] self.radioButton_train_model_MaskRCNN.toggled.connect(lambda checked: self.set_training_model("maskrcnn", checked))
        self.radioButton_train_model_YOLO.toggled.connect(lambda checked: self.set_training_model("yolo", checked))

        # Default selection
        self.selected_training_model = self.get_selected_model()

        #self.buttonBox_close.rejected.connect(self.reject)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def handle_left_item_doubleclick(self, item, column):
        if item.parent() is not None:
            return  # ignore double-click on child label nodes
        self.move_items_to_selected([item])

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def handle_right_item_doubleclick(self, item, column):
        if item.parent() is not None:
            return  # ignore double-click on child label nodes
        self.move_items_to_available([item])

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def resizeEvent(self, event):
        """
        Ensure list widgets update their geometry on dialog resize.
        """
        super().resizeEvent(event)
        self.listWidget_availableFolders.updateGeometry()
        self.listWidget_selectedFolders.updateGeometry()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def initialize_dialog_from_config(self, config):
        self.site_config = config
        self.setup_from_config_file()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def setup_from_config_file(self):
        """
        Initialize dialog controls from a configuration dictionary.
        """
        self.lineEdit_siteName.setText(self.site_config.get("siteName", ""))
        learningRates = self.site_config.get("learningRates", [])
        lr_str = ", ".join(str(x) for x in learningRates)
        self.lineEdit_learningRates.setText(lr_str)

        optimizer = self.site_config.get("Optimizer", "")
        idx = self.comboBox_optimizer.findText(optimizer)
        if idx >= 0:
            self.comboBox_optimizer.setCurrentIndex(idx)

        self.doubleSpinBox_weightDecay.setValue(self.site_config.get("weight_decay", 0.0))
        self.spinBox_epochs.setValue(self.site_config.get("number_of_epochs", 0))
        self.spinBox_batchSize.setValue(self.site_config.get("batch_size", 0))
        self.spinBox_saveFrequency.setValue(self.site_config.get("save_model_frequency", 0))
        self.spinBox_validationFrequency.setValue(self.site_config.get("validation_frequency", 0))
        self.checkBox_earlyStopping.setChecked(self.site_config.get("early_stopping", False))
        self.spinBox_patience.setValue(self.site_config.get("patience", 0))

        device = self.site_config.get("device", "")
        idx = self.comboBox_device.findText(device)
        if idx >= 0:
            self.comboBox_device.setCurrentIndex(idx)

        self.current_path = self.site_config.get("Path", None)

        # Apply context-sensitive UI for the loaded model selection
        self._update_context_sensitive_ui()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def reset_lists(self):
        """
        Move all items from the selected tree back to the available tree (sorted).
        Does not clear the available tree.
        """
        sel_root = self.listWidget_selectedFolders.invisibleRootItem()
        avail_root = self.listWidget_availableFolders.invisibleRootItem()

        # Collect names from selected tree
        moved_names = [
            sel_root.child(i).text(0).lstrip('★ ')
            for i in range(sel_root.childCount())
        ]

        self.listWidget_selectedFolders.clear()
        self.transferred_items.clear()

        # Merge with existing available names and rebuild sorted
        existing = [avail_root.child(i).text(0).lstrip('★ ') for i in range(avail_root.childCount())]
        all_names = sorted(set(existing + moved_names))
        self.listWidget_availableFolders.clear()
        for name in all_names:
            self._add_folder_to_tree(self.listWidget_availableFolders, name)

        self._refresh_annotations_from_selection()
        self.updateTrainButtonState()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def reject(self):
        # DO NOTHING. LET IT CLOSE. IF THE CALLING PROGRAM CREATED THE DIALOG USING EXEC, THE CALLING INSTANTIATING
        # PROGRAM CAN INSPECT THE RETURN RESULT
        super().reject()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def move_to_left(self):
        """
        Move selected top-level items back from selected tree to available tree, then re-sort.
        """
        sel_root = self.listWidget_selectedFolders.invisibleRootItem()
        avail_root = self.listWidget_availableFolders.invisibleRootItem()

        # Only move top-level (folder) items, not child label nodes
        selected_items = [
            item for item in self.listWidget_selectedFolders.selectedItems()
            if item.parent() is None
        ]
        for item in selected_items:
            name = item.text(0).lstrip('★ ')
            sel_root.removeChild(item)
            if name in self.transferred_items:
                self.transferred_items.remove(name)
            print(f"Moved '{name}' from selected back to available folders (sorted, button).")

        # Rebuild available tree sorted
        existing = [avail_root.child(i).text(0).lstrip('★ ') for i in range(avail_root.childCount())]
        moved = [item.text(0).lstrip('★ ') for item in selected_items]
        all_names = sorted(set(existing + moved))
        self.listWidget_availableFolders.clear()
        for name in all_names:
            self._add_folder_to_tree(self.listWidget_availableFolders, name)

        self._refresh_annotations_from_selection()
        self.updateTrainButtonState()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def move_items_to_available(self, items):
        sel_root = self.listWidget_selectedFolders.invisibleRootItem()
        avail_root = self.listWidget_availableFolders.invisibleRootItem()

        for item in items:
            # Only act on top-level folder nodes
            if item.parent() is not None:
                continue
            name = item.text(0).lstrip('★ ')
            if name in self.transferred_items:
                self.transferred_items.remove(name)
                sel_root.removeChild(item)

                existing = [avail_root.child(i).text(0).lstrip('★ ') for i in range(avail_root.childCount())]
                all_names = sorted(set(existing + [name]))
                self.listWidget_availableFolders.clear()
                for n in all_names:
                    self._add_folder_to_tree(self.listWidget_availableFolders, n)

        self._refresh_annotations_from_selection()
        self.updateTrainButtonState()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def set_training_model(self, model_name: str, checked: bool):
        """Update selected_training_model when a radio button is toggled on."""
        if checked:  # only update when the button is checked, not unchecked
            self.selected_training_model = model_name
            print(f"Selected training model: {self.selected_training_model}")
            self._update_lora_ui_for_model()
            self._update_context_sensitive_ui()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _set_lora_enabled(self, enabled: bool):
        """Enable/disable all LoRA-specific controls in one place."""
        for w in getattr(self, "_lora_widgets", []):
            if w is not None:
                w.setEnabled(enabled)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _update_lora_ui_for_model(self):
        """
        Turn LoRA section on only when the LoRA/SegFormer training model
        is selected; keep it greyed out for SAM2 and Mask R-CNN.
        """
        is_lora = (getattr(self, "selected_training_model", "") == "segformer")
        self._set_lora_enabled(is_lora)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _update_context_sensitive_ui(self):
        """
        Show/hide UI sections that are not applicable to the selected model.
        Called whenever the model radio button changes.

        YOLO trains on all annotated categories simultaneously — label
        selection is not applicable and is hidden to avoid confusion.
        All other models require explicit label selection and show it.

        If YOLO is selected but ultralytics is not installed, warn the user
        and revert to SAM2 so GRIME AI remains fully functional.
        """
        is_yolo = (getattr(self, "selected_training_model", "") == "yolo")

        if is_yolo:
            import importlib.util
            if importlib.util.find_spec("ultralytics") is None:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "ultralytics Not Installed",
                    "The ultralytics package is required for YOLOv11-seg training "
                    "but is not installed in this environment.\n\n"
                    "Install it with:\n"
                    "  pip install ultralytics\n\n"
                    "Reverting to SAM2. All other GRIME AI features remain available."
                )
                self.radioButton_train_model_SAM2.setChecked(True)
                return  # set_training_model will re-fire and call this method again

        self.groupBox_labelSelection.setVisible(not is_yolo)
        # The .ui defines the YOLO weights row as a layout (horizontalLayout_yoloWeightsRow),
        # not a wrapper widget, so toggle each child widget; the grid row collapses when empty.
        _yolo_row = self.horizontalLayout_yoloWeightsRow
        for _i in range(_yolo_row.count()):
            _w = _yolo_row.itemAt(_i).widget()
            if _w is not None:
                _w.setVisible(is_yolo)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def on_train_label_changed(self, index: int):
        """
        Update site_config with the selected training label(s).
        Store both label_id and label_name into TRAINING_CATEGORIES.
        """
        selected_training_labels = self.get_selected_training_labels()
        if not selected_training_labels:
            return

    # ------------------------------------------------------------------------------------------------------------------
        self.site_config.setdefault("train_model", {})

        # Overwrite TRAINING_CATEGORIES with the new selection
        self.site_config["train_model"]["TRAINING_CATEGORIES"] = selected_training_labels

        # Save back to JSON
        self._mgr.update_config(self.site_config, save=True)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_selected_training_labels(self):
        """
        Fetch the selected training label from comboBox_train_label_selection
        and return it as a list of dictionaries.

        Each dictionary contains:
        - "label_id": the parsed ID from the comboBox text
        - "label_name": the parsed label name

        Returns
        -------
        list[dict]
            A list with one dictionary if selection is valid, else an empty list.
        """
        selected_text = self.comboBox_train_label_selection.currentText().strip()

        if "-" not in selected_text:
            return []  # malformed or empty selection

        label_id, label_name = map(str.strip, selected_text.split("-", 1))

        return [{
            "label_id": label_id,
            "label_name": label_name
        }]

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def setup_custom_list_widgets(self):
        """Replace default tree widgets with custom draggable/droppable ones."""
        self.listWidget_availableFolders.__class__ = DraggableTreeWidget
        self.listWidget_selectedFolders.__class__ = DroppableTreeWidget

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _populate_yolo_weights_combobox(self):
        """
        Scan the ultralytics/assets folder for *-seg.pt files and populate
        comboBox_yoloWeights. If none are found, show 'No weights installed.'
        and disable the combobox.
        """
        import importlib.util
        self.comboBox_yoloWeights.blockSignals(True)
        self.comboBox_yoloWeights.clear()

        weights_dir = None
        try:
            spec = importlib.util.find_spec('ultralytics')
            if spec and spec.origin:
                assets = Path(spec.origin).parent / 'assets'
                if assets.is_dir():
                    weights_dir = assets
        except Exception:
            pass

        found = []
        if weights_dir:
            found = sorted(p.name for p in weights_dir.glob('*-seg.pt'))

        if found:
            for name in found:
                self.comboBox_yoloWeights.addItem(name)
            self.comboBox_yoloWeights.setEnabled(True)
        else:
            self.comboBox_yoloWeights.addItem('No weights installed.')
            self.comboBox_yoloWeights.setEnabled(False)

        self.comboBox_yoloWeights.blockSignals(False)
        self._update_train_button_state()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _update_train_button_state(self):
        """Refresh train button state — delegates to existing updateTrainButtonState."""
        try:
            self.updateTrainButtonState()
        except Exception:
            pass

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def browse_model_training_images_folder(self):
        """Open a dialog to choose a folder and update the folder path field."""
        folder = QFileDialog.getExistingDirectory(self, "Select training images folder", str(PROJECT_ROOT))
        folder = os.path.normpath(folder)

        if folder:
            self.lineEdit_model_training_images_path.setText(folder)
            self.populate_available_folders()

            JsonEditor().update_json_entry("Model_Training_Images_Folder", folder)

            self.updateTrainButtonState()
            self._training_images_committed()

    def _training_images_committed(self):
        # Ask the main window to reconcile this training-images change with
        # the active recipe (prompt to update the recipe or keep for session).
        self.trainingImagesCommitted_Signal.emit(
            self.lineEdit_model_training_images_path.text())

    # ------------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------------
    def _find_container_layout_and_index(self, widget):
        """
        Return (layout, index) for the slot in which `widget` resides.
        Traverses parent().layout() and its sub-layouts to find the actual item index.
        """
        parent = widget.parent()
        if not parent:
            return None, -1

        root_layout = parent.layout()
        if not isinstance(root_layout, QtWidgets.QLayout):
            return None, -1

        # breadth-first search over nested layouts
        queue = [root_layout]
        while queue:
            lay = queue.pop(0)
            for i in range(lay.count()):
                item = lay.itemAt(i)
                # Direct widget slot
                if item and item.widget() is widget:
                    return lay, i
                # Nested layout slot
                child_layout = item.layout() if item else None
                if isinstance(child_layout, QtWidgets.QLayout):
                    queue.append(child_layout)

        return None, -1

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _add_folder_to_tree(self, tree: QTreeWidget, folder_name: str) -> QTreeWidgetItem:
        """
        Add a folder as a top-level parent node to a tree widget, with its
        annotation labels and image count as non-selectable child nodes beneath it.
        Returns the created parent item.
        """
        from PyQt5.QtGui import QFont
        parent_item = QTreeWidgetItem(tree, [folder_name])
        parent_item.setFlags(parent_item.flags() | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsDragEnabled)

        # Load labels and image count from instances_default.json
        base_path = self.lineEdit_model_training_images_path.text().strip()
        folder_path = os.path.normpath(os.path.join(base_path, folder_name))

        child_font = QFont()
        child_font.setItalic(True)

        # Image count child
        annotation_file = os.path.join(folder_path, "instances_default.json")
        if os.path.exists(annotation_file):
            try:
                with open(annotation_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                image_count = len(data.get("images", []))
            except Exception:
                image_count = 0
        else:
            image_count = 0

        count_item = QTreeWidgetItem(parent_item, [f"Image count: {image_count}"])
        count_item.setFlags(QtCore.Qt.ItemIsEnabled)
        count_item.setFont(0, child_font)

        # Label children
        cats = self._load_categories(folder_path)
        if cats:
            for cat in cats:
                label_text = f"{cat['name']} (ID={cat['id']})"
                child_item = QTreeWidgetItem(parent_item, [label_text])
                child_item.setFlags(QtCore.Qt.ItemIsEnabled)
                child_item.setFont(0, child_font)

        tree.collapseItem(parent_item)
        return parent_item

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _install_drag_drop_lists(self) -> None:
        # Replace available tree
        avail_layout, avail_index = self._find_container_layout_and_index(self.listWidget_availableFolders)
        if avail_layout is None or avail_index < 0:
            print("Could not locate layout slot for listWidget_availableFolders; aborting replacement.")
        else:
            avail_dd = DraggableTreeWidget(self.listWidget_availableFolders.parent())
            avail_dd.setObjectName("listWidget_availableFolders")
            avail_dd.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            avail_dd.setHeaderHidden(True)
            avail_dd.setRootIsDecorated(True)
            avail_dd.setUniformRowHeights(False)
            avail_dd.setDragEnabled(True)
            avail_dd.setDragDropMode(QAbstractItemView.DragOnly)

            # Replace at the exact slot index
            avail_layout.insertWidget(avail_index, avail_dd)
            old = avail_layout.itemAt(avail_index + 1).widget()
            avail_layout.removeWidget(old)
            old.hide()
            old.deleteLater()
            self.listWidget_availableFolders = avail_dd

        # Replace selected tree
        sel_layout, sel_index = self._find_container_layout_and_index(self.listWidget_selectedFolders)
        if sel_layout is None or sel_index < 0:
            print("Could not locate layout slot for listWidget_selectedFolders; aborting replacement.")
        else:
            sel_dd = DroppableTreeWidget(self.listWidget_selectedFolders.parent())
            sel_dd.setObjectName("listWidget_selectedFolders")
            sel_dd.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            sel_dd.setHeaderHidden(True)
            sel_dd.setRootIsDecorated(True)
            sel_dd.setUniformRowHeights(False)
            sel_dd.setAcceptDrops(True)
            sel_dd.setDragDropMode(QAbstractItemView.DropOnly)

            sel_layout.insertWidget(sel_index, sel_dd)
            old = sel_layout.itemAt(sel_index + 1).widget()
            sel_layout.removeWidget(old)
            old.hide()
            old.deleteLater()
            self.listWidget_selectedFolders = sel_dd

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _init_labels_widget_reference(self) -> None:
        """
        Detect whether a label selection list exists (listWidget_labels).
        If not, we default to using comboBox_train_label_selection only.
        """
        if hasattr(self, "listWidget_labels") and isinstance(self.listWidget_labels, QtWidgets.QListWidget):
            self.categories_available = True
        else:
            self.categories_available = False

    # ------------------------------------------------------------------------
    # UI population and collection
    # ------------------------------------------------------------------------
    def _populate_ui_from_config(self, cfg: Dict[str, Any]) -> None:
        # Path for training images
        self.lineEdit_model_training_images_path.setText(cfg.get("segmentation_images_path", ""))

        # Model selection radio buttons
        model = cfg.get("load_model", {}).get("MODEL", "sam2").lower()
        self.selected_training_model = model
        self.radioButton_train_model_SAM2.setChecked(model == "sam2")
        self.radioButton_train_model_segformer.setChecked(model == "segformer")
        # [MASKRCNN] self.radioButton_train_model_MaskRCNN.setChecked(model == "maskrcnn")

        # Labels/categories
        labels = cfg.get("train_model", {}).get("TRAINING_CATEGORIES", [])
        if hasattr(self, "comboBox_train_label_selection"):
            self.comboBox_train_label_selection.clear()
            for label in labels:
                self.comboBox_train_label_selection.addItem(str(label))
        if hasattr(self, "listWidget_labels") and isinstance(self.listWidget_labels, QtWidgets.QListWidget):
            self.listWidget_labels.clear()
            for label in labels:
                self.listWidget_labels.addItem(QtWidgets.QListWidgetItem(str(label)))
            self.categories_available = True if labels else False
        else:
            self.categories_available = True if labels else False

        # Training parameters
        self.lineEdit_siteName.setText(cfg.get("siteName", ""))
        self.lineEdit_learningRates.setText(",".join(str(x) for x in cfg.get("learningRates", [0.0001])))
        self.comboBox_optimizer.setCurrentText(cfg.get("optimizer", "Adam"))
        self.doubleSpinBox_weightDecay.setValue(float(cfg.get("weight_decay", 0.01) or 0.01))
        self.spinBox_epochs.setValue(int(cfg.get("number_of_epochs", 20) or 20))
        self.spinBox_batchSize.setValue(int(cfg.get("batch_size", 32) or 32))
        self.spinBox_saveFrequency.setValue(int(cfg.get("save_model_frequency", 20) or 20))
        self.spinBox_validationFrequency.setValue(int(cfg.get("validation_frequency", 20) or 20))
        self.checkBox_earlyStopping.setChecked(bool(cfg.get("early_stopping", False)))
        self.spinBox_patience.setValue(int(cfg.get("patience", 3) or 3))
        self.comboBox_device.setCurrentText(cfg.get("device", "cpu"))

        # Blob filter radius — stored as fraction, displayed as pixels
        blob_fraction = float(cfg.get("blob_filter_radius", 0.0))
        if blob_fraction > 0.0:
            # Convert fraction back to pixels using reference diagonal from first training image
            ref_px = self._blob_fraction_to_pixels(blob_fraction)
            if ref_px is not None:
                self.spinBox_blobFilterRadius.setValue(int(round(ref_px)))
        # Always refresh the companion label
        self._update_blob_filter_pct_label()

        # Train/val split — both stored as fractions, loaded independently
        val_pct   = int(round(float(cfg.get("val_split",   0.2)) * 100))
        train_pct = int(round(float(cfg.get("train_split", 0.8)) * 100))
        # Back-compat: older configs only have val_split; derive train from it
        if "train_split" not in cfg:
            train_pct = 100 - val_pct
        self.spinBox_valSplit.setValue(val_pct)
        self.spinBox_trainSplit.setValue(train_pct)

        # YOLO base weights — select saved value in combobox if present
        saved_weights = cfg.get("yolo_base_weights", "")
        if saved_weights:
            idx = self.comboBox_yoloWeights.findText(saved_weights)
            if idx >= 0:
                self.comboBox_yoloWeights.setCurrentIndex(idx)

        # Hold-out seasons — restore from config
        holdout_seasons = cfg.get("holdout_seasons", [])
        self._set_holdout_seasons(holdout_seasons)

        # Folder lists
        self.listWidget_availableFolders.clear()
        for p in cfg.get("available_folders", []):
            self._add_folder_to_tree(self.listWidget_availableFolders, str(p))
        self.listWidget_selectedFolders.clear()
        for p in cfg.get("selected_folders", []):
            self._add_folder_to_tree(self.listWidget_selectedFolders, str(p))

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_selected_model(self) -> str:

        # THIS DESIGN PATTERN WILL ALLOW US TO EASILY ADD ANY ADDITIONAL
        # RADIOBUTTONS ASSOCIATED WITH TRAINING MODELS
        button_map = {
            self.radioButton_train_model_SAM2: "sam2",
            self.radioButton_train_model_segformer: "segformer",
            # [MASKRCNN] self.radioButton_train_model_MaskRCNN: "maskrcnn",
            self.radioButton_train_model_YOLO: "yolo",
        }
        for button, model in button_map.items():
            if button.isChecked():
                return model
        return "sam2"  # default

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _collect_ui_values(self) -> Dict[str, Any]:

        self.selected_training_model = self.get_selected_model()

        # Learning rates parsing
        lr_text = self.lineEdit_learningRates.text().strip()
        learning_rates: List[float] = []
        if lr_text:
            for tok in lr_text.split(","):
                tok = tok.strip()
                if tok:
                    try:
                        learning_rates.append(float(tok))
                    except ValueError:
                        pass

        # Selected labels
        selected_labels = self.comboBox_train_label_selection.currentText()
        # Split into id and name
        label_id, label_name = selected_labels.split(" - ", 1)

        values: Dict[str, Any] = {
            "siteName": self.lineEdit_siteName.text().strip(),
            "learningRates": learning_rates or [0.0001],
            "optimizer": self.comboBox_optimizer.currentText(),
            "weight_decay": float(self.doubleSpinBox_weightDecay.value()),
            "number_of_epochs": int(self.spinBox_epochs.value()),
            "batch_size": int(self.spinBox_batchSize.value()),
            "save_model_frequency": int(self.spinBox_saveFrequency.value()),
            "validation_frequency": int(self.spinBox_validationFrequency.value()),
            "early_stopping": bool(self.checkBox_earlyStopping.isChecked()),
            "patience": int(self.spinBox_patience.value()),
            "device": self.comboBox_device.currentText(),
            "blob_filter_radius": self._blob_pixels_to_fraction(),
            "val_split":   round(self.spinBox_valSplit.value()   / 100.0, 2),
            "train_split": round(self.spinBox_trainSplit.value() / 100.0, 2),
            "yolo_base_weights": (
                self.comboBox_yoloWeights.currentText()
                if self.comboBox_yoloWeights.isEnabled() else ""
            ),
            "holdout_seasons": self._get_holdout_seasons(),
            "segmentation_images_path": self.lineEdit_model_training_images_path.text().strip(),
            "available_folders": [
                self.listWidget_availableFolders.invisibleRootItem().child(i).text(0).lstrip('★ ')
                for i in range(self.listWidget_availableFolders.invisibleRootItem().childCount())
            ],
            "selected_folders": [
                self.listWidget_selectedFolders.invisibleRootItem().child(i).text(0).lstrip('★ ')
                for i in range(self.listWidget_selectedFolders.invisibleRootItem().childCount())
            ],
            "train_model": {
                "TRAINING_CATEGORIES": [
                    {
                        "label_id": label_id.strip(),
                        "label_name": label_name.strip()
                    }
                ]
            },
            "load_model": {
                **self.site_config.get("load_model", {}),
                "MODEL": self.selected_training_model,
            }
        }

        # UPDATE PATH SECTION BASED ON SELECTED FOLDERS
        root_folder = self.lineEdit_model_training_images_path.text().strip()
        if root_folder:
            self.update_path_from_selection(root_folder, values)

        return values

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def update_path_from_selection(self, root_folder: str, values: Dict[str, Any]):
        """
        Collect selected folders from listWidget_selectedFolders and update values["Path"]
        with image and annotation paths.
        """
        # 1. Gather selected folder names (top-level nodes only)
        sel_root = self.listWidget_selectedFolders.invisibleRootItem()
        selected_folders = [
            sel_root.child(i).text(0).lstrip('★ ')
            for i in range(sel_root.childCount())
        ]

        new_folders = []
        new_annotations = []

        # 2. Build normalized paths
        # JES: no longer using CVAT's layered convention
        for folder in selected_folders:
            folder = os.path.normpath(os.path.join(root_folder, folder))
            new_folders.append(folder)

            filepath = os.path.normpath(os.path.join(root_folder, folder, "instances_default.json"))
            new_annotations.append(filepath)

        # 3. Update values["Path"]
        values["Path"] = [{
            "siteName": "custom",
            "directoryPaths": {
                "folders": new_folders,
                "annotations": new_annotations
            }
        }]

        print("Updated Path section from selected folders.")

    # ------------------------------------------------------------------------
    # Actions and slots
    # ------------------------------------------------------------------------
    def get_selected_training_model(self):
        self.selected_training_model = self.get_selected_model()
        return self.selected_training_model

    def update_model_config(self) -> None:
        values = self._collect_ui_values()
        self._mgr.update_config(values, save=True)
        self.site_config = self._mgr.load_config(return_type="dict")

    def _backup_site_config_before_run(self):
        """
        Snapshot the current on-disk site_config.json to a timestamped backup
        inside a dedicated subfolder of the Settings directory. Called from
        train() immediately before the UI values are written to disk, so the
        prior run's settings are always recoverable.

        Failures are logged but never raised; a backup error must not block
        a training run.
        """
        try:
            settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
            config_file = os.path.normpath(os.path.join(settings_folder, "site_config.json"))
            if not os.path.exists(config_file):
                print("No existing site_config.json to back up; skipping.")
                return
            backup_dir = os.path.join(settings_folder, "site_config_backups")
            os.makedirs(backup_dir, exist_ok=True)
            now_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            backup_path = os.path.join(backup_dir, f"{now_str}_site_config.json")
            shutil.copy2(config_file, backup_path)
            print(f"Backed up site_config.json to {backup_path}")
        except Exception as e:
            print(f"WARNING: site_config.json backup failed: {e}")

    def train(self):
        """
        Called when the Train button is clicked.
        Validates model path, images folder, label selections, and image presence.
        """
        msg = self.validate_training_inputs()
        if msg:
            GRIME_AI_QMessageBox(
                'Missing Parameters',
                msg,
                GRIME_AI_QMessageBox.Ok,
                icon=QMessageBox.Warning
            ).displayMsgBox()
            return

        # LABEL CONSISTENCY GATE
        state = self._check_label_consistency()
        self._folder_validation_state = state
        self._apply_folder_colors(state)

        red_folders    = [n for n, s in state.items() if s == 'red']
        yellow_folders = [n for n, s in state.items() if s == 'yellow']
        unreadable     = [n for n, s in state.items() if s == 'unreadable']

        # Hard block — mismatched IDs
        if red_folders or unreadable:
            lines = ["Training cannot proceed due to annotation errors:\n"]
            if red_folders:
                lines.append("❌  Label ID conflicts:")
                for f in red_folders:
                    lines.append(f"    • {f}")
                lines.append("")
            if unreadable:
                lines.append("❌  Annotation file missing or unreadable:")
                for f in unreadable:
                    lines.append(f"    • {f}")
            GRIME_AI_QMessageBox(
                'Annotation Errors — Training Blocked',
                "\n".join(lines),
                GRIME_AI_QMessageBox.Ok,
                icon=QMessageBox.Critical
            ).displayMsgBox()
            return

        # Soft warning — missing categories, let user decide
        if yellow_folders:
            lines = ["⚠️  The following folders are missing some categories:\n"]
            for f in yellow_folders:
                lines.append(f"    • {f}")
            lines.append("\nTraining can proceed, but results may be incomplete.\nContinue?")
            reply = QMessageBox.question(
                self,
                "Missing Categories — Continue?",
                "\n".join(lines),
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel
            )
            if reply != QMessageBox.Yes:
                return

        # All valid — proceed silently

        # DEBUG: capture exact listbox state at Train-click moment
        holdouts_at_click = self._get_holdout_seasons()
        print(f"\n[HOLDOUT DEBUG] Train click — Hold-Out listbox reads: {holdouts_at_click!r}")
        print(f"[HOLDOUT DEBUG] Train click — siteName lineEdit: "
              f"{self.lineEdit_siteName.text().strip()!r}")

        self._backup_site_config_before_run()
        self.update_model_config()
        print("\nTrain button clicked. Starting training process...")
        self.ml_train_signal.emit()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def reset_selection(self):
        self.listWidget_selectedFolders.clear()
        self.listWidget_availableFolders.clear()
        self.transferred_items.clear()
        self.annotation_list = []
        self.unique_training_labels = []
        self.populate_train_label_combobox([])
        self.updateTrainButtonState()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def move_to_selected(self):
        avail_root = self.listWidget_availableFolders.invisibleRootItem()
        for item in self.listWidget_availableFolders.selectedItems():
            if item.parent() is not None:
                continue  # skip child nodes
            name = item.text(0)
            avail_root.removeChild(item)
            self._add_folder_to_tree(self.listWidget_selectedFolders, name)
            self.transferred_items.add(name)
        self.listWidget_selectedFolders.repaint()
        self._refresh_annotations_from_selection()
        self.updateTrainButtonState()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def move_to_right(self):
        selected_items = [
            item for item in self.listWidget_availableFolders.selectedItems()
            if item.parent() is None
        ]
        self._move_items(selected_items)
        for item in selected_items:
            print(f"Moved '{item.text(0)}' from available to selected folders (button).")
        self.updateTrainButtonState()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def move_to_available(self):
        sel_root = self.listWidget_selectedFolders.invisibleRootItem()
        avail_root = self.listWidget_availableFolders.invisibleRootItem()

        items = [
            item for item in self.listWidget_selectedFolders.selectedItems()
            if item.parent() is None
        ]
        moved_names = []
        for item in items:
            name = item.text(0).lstrip('★ ')
            sel_root.removeChild(item)
            self.transferred_items.discard(name)
            moved_names.append(name)

        existing = [avail_root.child(i).text(0).lstrip('★ ') for i in range(avail_root.childCount())]
        all_names = sorted(set(existing + moved_names))
        self.listWidget_availableFolders.clear()
        for name in all_names:
            self._add_folder_to_tree(self.listWidget_availableFolders, name)

        self.listWidget_availableFolders.repaint()
        self._refresh_annotations_from_selection()
        self.updateTrainButtonState()

    # ------------------------------------------------------------------------
    # Folder population and annotation aggregation
    # ------------------------------------------------------------------------
    def populate_available_folders(self):
        """
        Safely populate the available training folders list.
        Prevents accidental scanning of system/Conda directories
        when the path field is empty or invalid.
        """

        raw = self.lineEdit_model_training_images_path.text().strip()

        # HARD STOP: EMPTY OR WHITESPACE PATH
        if not raw:
            print("Training images path is empty; skipping folder scan.")
            return

        root = Path(raw).resolve()

        # If the root folder has changed and selected folders are present, ask user
        previous_root = getattr(self, "_last_training_root", None)
        if previous_root is not None and Path(previous_root) != root:
            sel_root = self.listWidget_selectedFolders.invisibleRootItem()
            if sel_root.childCount() > 0:
                reply = QMessageBox.question(
                    self,
                    "Root Folder Changed",
                    "You are changing the root folder.\n\nWould you like to clear your currently selected folders?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.reset_lists()

        self._last_training_root = str(root)
        self.listWidget_availableFolders.clear()

        # HARD STOP: REFUSE TO SCAN SUSPICIOUS/SYSTEM ROOTS
        forbidden = ["anaconda3", "miniconda3", "programdata", "windows"]
        if any(f in str(root).lower() for f in forbidden):
            print(f"Refusing to scan suspicious root: {root}")
            return

        if not root.is_dir():
            QMessageBox.warning(
                self,
                "Invalid Folder",
                f"The selected path is not a directory:\n{root}"
            )
            return

        valid: List[Path] = []
        incomplete: Dict[str, List[str]] = {}

        # CHECK THE ROOT ITSELF
        ok, missing, json_path, orphans, unannotated = _check_folder(root)
        if ok:
            valid.append(root)
        elif missing or orphans:
            incomplete[str(root)] = (missing, orphans, unannotated, json_path)

        # RECURSE INTO SUBFOLDERS (SAFE BECAUSE ROOT IS VALIDATED)
        for folder in _iter_dirs(root):
            ok, missing, json_path, orphans, unannotated = _check_folder(folder)
            if ok:
                valid.append(folder)
            elif missing or orphans:
                incomplete[str(folder)] = (missing, orphans, unannotated, json_path)

        # POPULATE OR ALERT "NO VALID"
        if valid:
            for vf in sorted(set(valid)):
                rel = vf.relative_to(root)
                display_name = str(rel)
                self._add_folder_to_tree(self.listWidget_availableFolders, display_name)
        else:
            QMessageBox.information(
                self,
                "No Valid Training Sets",
                "No folders were found containing a COCO JSON and all its images."
            )

        # INCOMPLETE SETS POPUP
        if incomplete:
            lines = ["Folders with annotation issues:"]
            for fld, (miss, orphans, unannotated, json_path) in incomplete.items():
                json_label = f"\n  Annotation file: {json_path}" if json_path else ""
                lines.append(f"\n{fld}{json_label}")
                if miss:
                    lines.append(f"  Missing from disk ({len(miss)}):")
                    lines += [f"    - {m}" for m in miss[:10]]
                    if len(miss) > 10:
                        lines.append(f"    ... and {len(miss) - 10} more.")
                if orphans:
                    lines.append(f"  Orphan annotations ({len(orphans)}) - image_id not in images list:")
                    lines += [f"    - {a}" for a in orphans[:10]]
                    if len(orphans) > 10:
                        lines.append(f"    ... and {len(orphans) - 10} more.")
                if unannotated:
                    lines.append(f"  On-disk files with no JSON entry ({len(unannotated)}) - will be skipped during training:")
                    lines += [f"    - {u}" for u in unannotated[:5]]
                    if len(unannotated) > 5:
                        lines.append(f"    ... and {len(unannotated) - 5} more.")
            QMessageBox.information(
                self,
                "Incomplete Training Sets",
                "\n".join(lines)
            )
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _move_items(self, items):
        avail_root = self.listWidget_availableFolders.invisibleRootItem()
        for item in items:
            if item.parent() is not None:
                continue  # skip child nodes
            name = item.text(0)
            if name not in self.transferred_items:
                self._add_folder_to_tree(self.listWidget_selectedFolders, name)
                self.transferred_items.add(name)
            avail_root.removeChild(item)

        self._refresh_annotations_from_selection()
        self.updateTrainButtonState()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def move_items_to_selected(self, items):
        self._move_items(items)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _refresh_annotations_from_selection(self):
        # ENSURE THE SELECTED LIST VISUALLY UPDATES BEFORE WE COMPUTE LABELS
        self.listWidget_selectedFolders.repaint()

        # BASE PATH (ROOT FOLDER OF TRAINING IMAGES)
        base_path = self.lineEdit_model_training_images_path.text().strip()

        # CURRENT SELECTED FOLDERS (TOP-LEVEL NODES ONLY)
        sel_root = self.listWidget_selectedFolders.invisibleRootItem()
        moved_names = [
            sel_root.child(i).text(0).lstrip('★ ')
            for i in range(sel_root.childCount())
        ]

        # VALIDATE SELECTED FOLDERS: CHECK JSON vs DISK CONSISTENCY
        root_folder = os.path.normpath(os.path.abspath(base_path)) if base_path else ""
        invalid_folders = []
        for folder_name in moved_names:
            folder_path = Path(os.path.join(root_folder, folder_name))
            ok, missing, json_path, orphans, unannotated = _check_folder(folder_path)
            if not ok:
                invalid_folders.append((folder_name, missing, orphans, unannotated, json_path))

        if invalid_folders:
            lines = ["The following selected folders have annotation issues that will cause training errors:\n"]
            for folder_name, missing, orphans, unannotated, json_path in invalid_folders:
                json_label = f"\n  JSON: {json_path}" if json_path else ""
                lines.append(f"\n{folder_name}{json_label}")
                if missing:
                    lines.append(f"  Missing from disk ({len(missing)}):")
                    for m in missing[:10]:
                        lines.append(f"    - {m}")
                    if len(missing) > 10:
                        lines.append(f"    ... and {len(missing) - 10} more.")
                if orphans:
                    lines.append(f"  Orphan annotations ({len(orphans)}) - image_id not in images list:")
                    for a in orphans[:10]:
                        lines.append(f"    - {a}")
                    if len(orphans) > 10:
                        lines.append(f"    ... and {len(orphans) - 10} more.")
                if unannotated:
                    lines.append(f"  On-disk files with no JSON entry ({len(unannotated)}) - will be skipped during training:")
                    for u in unannotated[:5]:
                        lines.append(f"    - {u}")
                    if len(unannotated) > 5:
                        lines.append(f"    ... and {len(unannotated) - 5} more.")
            QMessageBox.warning(
                self,
                "JSON / Image Mismatch in Selected Folders",
                "\n".join(lines)
            )
        # BUILD ANNOTATION LIST FROM SELECTED FOLDERS
        self.annotation_list = self._build_annotation_list(base_path, moved_names)

        # COLLECT UNIQUE LABELS AND REPOPULATE COMBOBOX
        self.unique_training_labels = self.collect_unique_labels(self.annotation_list)
        self.populate_train_label_combobox(self.unique_training_labels)

        # SILENTLY RE-COLOR SELECTED FOLDERS BASED ON LABEL CONSISTENCY
        self.validate_label_consistency(silent=True)

    # ------------------------------------------------------------------------
    # Annotation helpers
    # ------------------------------------------------------------------------
    def _build_annotation_list(self, base_path: str, folder_names: list[str]) -> list[str]:
        root_folder = os.path.normpath(os.path.abspath(base_path))
        annotations = []

        for folder in folder_names:
            folder_path = os.path.normpath(os.path.join(root_folder, folder))
            annotation_file_path = os.path.normpath(folder_path)
            if os.path.exists(annotation_file_path):
                annotations.append(annotation_file_path)

        return annotations
        self.populate_train_label_combobox(self.unique_training_labels)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def load_labels_from_annotation(self, folder_path):
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

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def collect_unique_labels(self, annotation_files: list[str]) -> list[str]:
        """
        Collects all unique category labels from a list of annotation files.

        Parameters
        ----------
        annotation_files : list[str]
            Fully qualified paths to instances_default.json files.

        Returns
        -------
        list[str]
            Sorted list of unique labels in the format 'id - name'.
        """
        all_labels = set()

        for path in annotation_files:
            labels = self.load_labels_from_annotation(path)
            all_labels.update(labels)

        return sorted(all_labels)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _load_categories(self, folder_path: str) -> Optional[List[Dict]]:
        """
        Load and return sorted categories from instances_default.json in folder_path.
        Returns None if the file is missing or unreadable.
        """
        annotation_file = os.path.join(folder_path, "instances_default.json")
        if not os.path.exists(annotation_file):
            return None
        try:
            with open(annotation_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return sorted(data.get("categories", []), key=lambda c: c["id"])
        except Exception:
            return None

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _check_label_consistency(self) -> Dict[str, str]:
        """
        Compare all selected folders against the gold standard (first entry in
        listWidget_selectedFolders).

        Returns a dict mapping folder display name -> status:
            'ok'     : matches gold standard exactly
            'yellow' : subset of gold standard (missing categories, no conflicts)
            'red'    : ID/name mismatch against gold standard
            'gold'   : this IS the gold standard (first folder)
            'unreadable': annotation file missing or unparseable
        """
        base_path = self.lineEdit_model_training_images_path.text().strip()
        sel_root = self.listWidget_selectedFolders.invisibleRootItem()
        count = sel_root.childCount()
        if count == 0:
            return {}

        state: Dict[str, str] = {}

        # Collect folder names in order (top-level nodes only), stripping any ★ prefix
        folder_names = [
            sel_root.child(i).text(0).lstrip('★ ')
            for i in range(count)
        ]

        # Gold standard is always the first entry
        gold_name = folder_names[0]
        gold_path = os.path.normpath(os.path.join(base_path, gold_name))
        gold_cats = self._load_categories(gold_path)

        if gold_cats is None:
            # Can't establish a gold standard — mark everything unreadable
            for name in folder_names:
                state[name] = 'unreadable'
            return state

        # Build gold standard lookup: name -> id
        gold_by_name: Dict[str, int] = {c["name"]: c["id"] for c in gold_cats}
        gold_schema = tuple((c["id"], c["name"]) for c in gold_cats)

        state[gold_name] = 'gold'

        for name in folder_names[1:]:
            folder_path = os.path.normpath(os.path.join(base_path, name))
            cats = self._load_categories(folder_path)

            if cats is None:
                state[name] = 'unreadable'
                continue

            folder_schema = tuple((c["id"], c["name"]) for c in cats)

            if folder_schema == gold_schema:
                state[name] = 'ok'
                continue

            # Check for any ID/name conflict vs gold standard
            conflict = False
            for c in cats:
                if c["name"] in gold_by_name and gold_by_name[c["name"]] != c["id"]:
                    conflict = True
                    break
                # Also check: same ID, different name
                for gc in gold_cats:
                    if gc["id"] == c["id"] and gc["name"] != c["name"]:
                        conflict = True
                        break
                if conflict:
                    break

            if conflict:
                state[name] = 'red'
            else:
                # No conflicts — this folder is a subset (missing some categories)
                state[name] = 'yellow'

        return state

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _apply_folder_colors(self, state: Dict[str, str]) -> None:
        """
        Apply colors to top-level folder nodes in listWidget_selectedFolders.
        - Gold standard: black text + ★ prefix
        - OK: black text
        - Yellow (missing categories): dark yellow text
        - Red / unreadable: red text
        Also highlights individual child label nodes red when their ID conflicts
        with the gold standard.
        """
        from PyQt5.QtGui import QColor, QFont

        color_map = {
            'gold':        QColor('black'),
            'ok':          QColor('black'),
            'yellow':      QColor(180, 120, 0),
            'red':         QColor('red'),
            'unreadable':  QColor('red'),
        }

        # Build gold standard label lookup directly from gold parent's child nodes
        sel_root = self.listWidget_selectedFolders.invisibleRootItem()
        gold_by_name: Dict[str, int] = {}  # label name -> ID from gold standard tree node
        if sel_root.childCount() > 0:
            gold_parent = sel_root.child(0)
            for j in range(gold_parent.childCount()):
                child_text = gold_parent.child(j).text(0)  # e.g. "water (ID=2)"
                parts = child_text.split(" (ID=")
                if len(parts) == 2:
                    name = parts[0].strip()
                    try:
                        gid = int(parts[1].rstrip(')'))
                        gold_by_name[name] = gid
                    except ValueError:
                        pass

        child_normal_font = QFont()
        child_normal_font.setItalic(True)

        for i in range(sel_root.childCount()):
            parent_item = sel_root.child(i)
            raw_name = parent_item.text(0)
            base_name = raw_name.lstrip('★ ')
            status = state.get(base_name, 'ok')
            color = color_map.get(status, QColor('black'))

            # Apply color and ★ to parent node
            if status == 'gold':
                parent_item.setText(0, f"★ {base_name}")
            else:
                parent_item.setText(0, base_name)
            parent_item.setForeground(0, color)

            for j in range(parent_item.childCount()):
                child = parent_item.child(j)
                child.setFont(0, child_normal_font)
                child.setForeground(0, QColor('black'))

                if status == 'gold':
                    continue  # gold standard children always normal

                # Parse label name and ID directly from child text
                child_text = child.text(0)
                parts = child_text.split(" (ID=")
                if len(parts) != 2:
                    continue
                label_name = parts[0].strip()
                try:
                    child_id = int(parts[1].rstrip(')'))
                except ValueError:
                    continue

                gold_id = gold_by_name.get(label_name)
                if gold_id is not None and child_id != gold_id:
                    child.setForeground(0, QColor('red'))
                    parent_item.setExpanded(True)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def validate_label_consistency(self, silent: bool = False) -> bool:
        """
        Run label consistency check, color the selected folders list, and
        (unless silent=True) show a result dialog to the user.

        Returns True if no red items exist (training may proceed).
        """
        state = self._check_label_consistency()
        self._folder_validation_state = state
        self._apply_folder_colors(state)

        if not state:
            if not silent:
                GRIME_AI_QMessageBox(
                    'Validate Labels',
                    'No folders are selected.',
                    GRIME_AI_QMessageBox.Ok,
                    icon=QMessageBox.Information
                ).displayMsgBox()
            return True

        red_folders   = [n for n, s in state.items() if s == 'red']
        yellow_folders = [n for n, s in state.items() if s == 'yellow']
        unreadable    = [n for n, s in state.items() if s == 'unreadable']

        has_errors = bool(red_folders or unreadable)

        if not silent:
            base_path = self.lineEdit_model_training_images_path.text().strip()
            sel_root = self.listWidget_selectedFolders.invisibleRootItem()
            gold_name = sel_root.child(0).text(0) if sel_root.childCount() > 0 else '(none)'

            if not has_errors and not yellow_folders:
                msg = "✅  All annotation files are valid."
            else:
                lines = []

                if red_folders:
                    lines.append("❌  Label ID conflicts (training blocked):")
                    for f in red_folders:
                        lines.append(f"    • {f}")
                    lines.append("")

                if unreadable:
                    lines.append("❌  Annotation file missing or unreadable:")
                    for f in unreadable:
                        lines.append(f"    • {f}")
                    lines.append("")

                if yellow_folders:
                    lines.append("⚠️  Missing categories (training allowed):")
                    for f in yellow_folders:
                        lines.append(f"    • {f}")

                msg = "\n".join(lines)

            icon = QMessageBox.Critical if has_errors else (
                QMessageBox.Warning if yellow_folders else QMessageBox.Information
            )
            title = "Validate Labels — Issues Found" if (has_errors or yellow_folders) else "Validate Labels — All Clear"
            GRIME_AI_QMessageBox(title, msg, GRIME_AI_QMessageBox.Ok, icon=icon).displayMsgBox()

        return not has_errors

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def populate_train_label_combobox(self, labels: list[str]):
        self.comboBox_train_label_selection.clear()
        for label in sorted(labels):
            self.comboBox_train_label_selection.addItem(label)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def updateTrainButtonState(self):
        """
        Update Train button appearance and tooltip based on current inputs.
        Do NOT show any popups here; only passive feedback.
        """
        msg = self.validate_training_inputs()

        if msg:
            # Keep enabled (so clicks can be handled), but gray it out and set a helpful tooltip
            self.pushButton_train.setEnabled(True)
            #self.pushButton_train.setStyleSheet("background-color: lightgray; color: darkgray;")
            # Short tooltip, not the full message box text (less annoying)
            self.pushButton_train.setToolTip("Complete required fields before training.")
        else:
            self.pushButton_train.setEnabled(True)
            #self.pushButton_train.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
            self.pushButton_train.setToolTip("Start training with current configuration.")

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def validate_training_inputs(self) -> str:
        """
        Validate all required training inputs and return a string of missing parameters.
        If everything is valid, return an empty string.
        """
        missing = []

        # Site name
        if not self.lineEdit_siteName.text().strip():
            missing.append("Site name")

        # Training images folder path
        if not self.lineEdit_model_training_images_path.text().strip():
            missing.append("Model training images folder")

        # Selected folders
        if self.listWidget_selectedFolders.invisibleRootItem().childCount() == 0:
            missing.append("Selected training folders")

        # Labels (if categories are available)
        if self.categories_available and hasattr(self, "listWidget_labels") and isinstance(self.listWidget_labels,
                                                                                           QtWidgets.QListWidget):
            any_selected = any(
                self.listWidget_labels.item(i).isSelected() for i in range(self.listWidget_labels.count()))
            if not any_selected and self.listWidget_labels.count() > 0:
                missing.append("Training labels")

        # ComboBox train label selection
        if not self.comboBox_train_label_selection.currentText().strip():
            missing.append("Train label selection")

        # Radio buttons (mutually exclusive group: require at least one checked)
        if not (
                self.radioButton_train_model_SAM2.isChecked()
                or self.radioButton_train_model_segformer.isChecked()
                # [MASKRCNN] or self.radioButton_train_model_MaskRCNN.isChecked()
                or self.radioButton_train_model_YOLO.isChecked()
        ):
            missing.append("Training model selection")

        # Learning rates
        lr_text = self.lineEdit_learningRates.text().strip()
        if not lr_text:
            missing.append("Learning rate(s)")

        # Optimizer
        if not self.comboBox_optimizer.currentText().strip():
            missing.append("Optimizer")

        # Loss function
        # Weight decay
        if self.doubleSpinBox_weightDecay.value() == 0.0:
            missing.append("Weight decay")

        # Epochs
        if self.spinBox_epochs.value() <= 0:
            missing.append("Number of epochs")

        # Batch size
        if self.spinBox_batchSize.value() <= 0:
            missing.append("Batch size")

        # Save frequency
        if self.spinBox_saveFrequency.value() <= 0:
            missing.append("Save model frequency")

        # Validation frequency
        if self.spinBox_validationFrequency.value() <= 0:
            missing.append("Validation frequency")

        # Patience (only matters if early stopping is checked)
        if self.checkBox_earlyStopping.isChecked() and self.spinBox_patience.value() <= 0:
            missing.append("Early stopping patience")

        # Device
        if not self.comboBox_device.currentText().strip():
            missing.append("Device")

        # YOLO base weights — required when YOLO is selected
        if self.radioButton_train_model_YOLO.isChecked():
            if (not self.comboBox_yoloWeights.isEnabled() or
                    self.comboBox_yoloWeights.currentText() == "No weights installed."):
                missing.append("YOLO base weights (run download_yolo_weights.py)")

        # Build message string
        if missing:
            return "Missing required parameters: " + ", ".join(missing)
        return ""

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_training_images_root_folder(self):
        return self.lineEdit_model_training_images_path.text().strip()

    # ******************************************************************************************************************
    # *   OTHER    OTHER     OTHER     OTHER     OTHER     OTHER     OTHER     OTHER     OTHER     OTHER     OTHER     *
    # ******************************************************************************************************************
    def save_site_name_to_json(self):
        settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
        CONFIG_FILENAME = "site_config.json"
        config_file = os.path.normpath(os.path.join(settings_folder, CONFIG_FILENAME))

        # Use raw json.load (not JsonEditor) to guarantee complete dict round-trip.
        # JsonEditor may filter unknown keys, which previously stripped holdout_seasons.
        with open(config_file, "r", encoding="utf-8") as f:
            settings = json.load(f)

        # DEBUG: log holdout_seasons state before and after to confirm no fields are lost
        before_holdout = settings.get("holdout_seasons", "<MISSING>")
        print(f"[SITE_NAME DEBUG] save_site_name_to_json called. "
              f"On-disk holdout_seasons before: {before_holdout!r}")

        # Update with the current SiteName value
        settings["siteName"] = self.lineEdit_siteName.text()

        # Write back to the config file
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)

        after_holdout = settings.get("holdout_seasons", "<MISSING>")
        print(f"[SITE_NAME DEBUG] save_site_name_to_json wrote. "
              f"On-disk holdout_seasons after:  {after_holdout!r}")
        print(f"Updated siteName in {config_file} to '{settings['siteName']}'")

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _get_reference_diagonal(self) -> float:
        """
        Return the diagonal (in pixels) of the first image found in the
        configured training folder. Used to convert px ↔ fraction.
        Returns None if no image can be found.
        """
        import math
        root = self.lineEdit_model_training_images_path.text().strip()
        if not root or not os.path.isdir(root):
            return None

        valid_exts = ('.jpg', '.jpeg', '.png')
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith(valid_exts):
                    try:
                        from PIL import Image as PILImage
                        img_path = os.path.join(dirpath, fname)
                        with PILImage.open(img_path) as im:
                            w, h = im.size
                        return math.sqrt(w * w + h * h)
                    except Exception:
                        continue
        return None

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _blob_pixels_to_fraction(self) -> float:
        """
        Convert the spinBox_blobFilterRadius pixel value to a fraction of
        image diagonal using the reference image. Falls back to the default
        fraction (50 px / 2236 px ≈ 0.02236) if no reference image is found.
        """
        import math
        DEFAULT_FRACTION = 50.0 / math.sqrt(2000**2 + 1000**2)  # ~0.02236
        px = self.spinBox_blobFilterRadius.value()
        diagonal = self._get_reference_diagonal()
        if diagonal and diagonal > 0:
            return px / diagonal
        return DEFAULT_FRACTION

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _blob_fraction_to_pixels(self, fraction: float):
        """
        Convert a stored fraction back to pixels using the reference diagonal.
        Returns None if no reference image is found.
        """
        diagonal = self._get_reference_diagonal()
        if diagonal and diagonal > 0:
            return fraction * diagonal
        return None

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _on_split_link_toggled(self, linked: bool) -> None:
        """Toggle chain link — when linked spinboxes are complementary; when unlinked, independent."""
        self._split_linked = linked
        if _get_chain_icon:
            self._btn_split_link.setIcon(
                getattr(self, '_icon_linked', QIcon()) if linked
                else getattr(self, '_icon_unlinked', QIcon())
            )
        else:
            self._btn_split_link.setText("🔗" if linked else "⛓")
        self._btn_split_link.setToolTip(
            "Linked: changing one value sets the other to its complement (100 - n).\nClick to unlink."
            if linked else
            "Unlinked: spinboxes are independent.\nClick to link."
        )
        if linked:
            self._split_guard = True
            try:
                self.spinBox_valSplit.setValue(100 - self.spinBox_trainSplit.value())
            finally:
                self._split_guard = False

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _on_val_split_changed(self, val_pct: int) -> None:
        """When validation changes, update training only if linked."""
        if getattr(self, '_split_guard', False) or not getattr(self, '_split_linked', False):
            return
        self._split_guard = True
        try:
            self.spinBox_trainSplit.setValue(100 - val_pct)
        finally:
            self._split_guard = False

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _on_train_split_changed(self, train_pct: int) -> None:
        """When training changes, update validation only if linked."""
        if getattr(self, '_split_guard', False) or not getattr(self, '_split_linked', False):
            return
        self._split_guard = True
        try:
            self.spinBox_valSplit.setValue(100 - train_pct)
        finally:
            self._split_guard = False

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ── Hold-out seasons dual listbox ─────────────────────────────────────────

    _SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
    _SEASON_TYPE  = "Meteorological"  # hardcoded; no UI control needed
    _SEASON_DATES = {
        "Winter": "Dec 1 - Feb 28",
        "Spring": "Mar 1 - May 31",
        "Summer": "Jun 1 - Aug 31",
        "Fall":   "Sep 1 - Nov 30",
    }

    @classmethod
    def _season_label(cls, season: str) -> str:
        """Return display label with date range, e.g. 'Winter (Dec - Feb)'."""
        dates = cls._SEASON_DATES.get(season, "")
        return f"{season} ({dates})" if dates else season

    @staticmethod
    def _season_from_label(label: str) -> str:
        """Strip the date suffix from a display label to recover the season name."""
        return label.split(" (")[0]

    def _init_holdout_season_lists(self) -> None:
        """Ensure all four seasons are distributed between the two listboxes."""
        # Seasons already populated in UI file on left side; right starts empty.
        # Nothing to do on init beyond what the .ui file provides.
        pass

    def _get_holdout_seasons(self) -> list:
        """Return list of plain season names currently in the hold-out listbox."""
        lw = self.listWidget_holdoutSeasons
        return [self._season_from_label(lw.item(i).text()) for i in range(lw.count())]

    def _debug_holdout_rows_inserted(self, parent, first, last):
        """DEBUG: fires when items are added to the Hold-Out listbox (incl. via drag-drop)."""
        import traceback
        lw = self.listWidget_holdoutSeasons
        added = [lw.item(i).text() for i in range(first, last + 1) if lw.item(i)]
        current = [self._season_from_label(lw.item(i).text()) for i in range(lw.count())]
        # Show the call site that triggered the insert
        stack = traceback.extract_stack()
        caller = stack[-3] if len(stack) >= 3 else stack[-1]
        print(f"[HOLDOUT DEBUG] rowsInserted: added {added!r} → listbox now {current!r}  "
              f"(triggered from {caller.name}())")

    def _debug_holdout_rows_removed(self, parent, first, last):
        """DEBUG: fires when items are removed from the Hold-Out listbox."""
        import traceback
        lw = self.listWidget_holdoutSeasons
        removed = [lw.item(i).text() for i in range(first, last + 1) if lw.item(i)]
        stack = traceback.extract_stack()
        caller = stack[-3] if len(stack) >= 3 else stack[-1]
        print(f"[HOLDOUT DEBUG] rowsAboutToBeRemoved: removing {removed!r}  "
              f"(triggered from {caller.name}())")

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _set_holdout_seasons(self, holdout_seasons: list) -> None:
        """
        Distribute seasons between available and hold-out listboxes based on
        the provided holdout list. Resets both listboxes. Items are displayed
        with their meteorological date range in parentheses.
        """
        # DEBUG: log every call so we can see who clears/rebuilds the listbox
        import traceback
        caller = traceback.extract_stack()[-2]
        print(f"[HOLDOUT DEBUG] _set_holdout_seasons({holdout_seasons!r}) called from "
              f"{caller.filename.split(chr(92))[-1].split('/')[-1]}:{caller.lineno} in {caller.name}()")

        holdout_set = set(holdout_seasons)
        self.listWidget_availableSeasons.clear()
        self.listWidget_holdoutSeasons.clear()
        for season in self._SEASON_ORDER:
            label = self._season_label(season)
            if season in holdout_set:
                self.listWidget_holdoutSeasons.addItem(label)
            else:
                self.listWidget_availableSeasons.addItem(label)

        # DEBUG: confirm final state
        final = [self._season_from_label(self.listWidget_holdoutSeasons.item(i).text())
                 for i in range(self.listWidget_holdoutSeasons.count())]
        print(f"[HOLDOUT DEBUG]   → Hold-Out listbox now contains: {final!r}")

    def _on_available_season_doubleclick(self, item) -> None:
        """
        Double-clicking an item in the Available Seasons listbox moves it
        to the Hold-Out Seasons listbox. Preserves the canonical season
        ordering on the destination side.
        """
        if item is None:
            return
        season = self._season_from_label(item.text())
        current = self._get_holdout_seasons()
        if season in current:
            return
        current.append(season)
        # Re-sort to canonical season order so Winter/Spring/Summer/Fall stays consistent
        ordered = [s for s in self._SEASON_ORDER if s in current]
        self._set_holdout_seasons(ordered)

    def _on_holdout_season_doubleclick(self, item) -> None:
        """
        Double-clicking an item in the Hold-Out Seasons listbox moves it
        back to the Available Seasons listbox.
        """
        if item is None:
            return
        season = self._season_from_label(item.text())
        current = self._get_holdout_seasons()
        if season not in current:
            return
        current.remove(season)
        self._set_holdout_seasons(current)

    # ------------------------------------------------------------------------
    def _update_blob_filter_pct_label(self):
        """
        Recompute and display the blob filter radius as a percentage of image
        diagonal.

        NOTE: the standalone label_blobFilterRadiusPct widget was removed from
        the .ui in favor of writing the equivalent percent into the spinbox
        tooltip via _update_blob_radius_tooltip(). This stub exists only because
        other code paths still call this method by name; the tooltip path is
        called from those same code paths.
        """
        try:
            self._update_blob_radius_tooltip()
        except AttributeError:
            # Tooltip helper not yet wired; safe to ignore.
            pass

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def prompt_blob_radius_update(self, trainer):
        """
        Called from main.py after training completes. Automatically saves the
        active blob filter radius to all checkpoints with no user prompt.

        If Computed mode is selected, the Mahalanobis distribution (mean,
        covariance, n_sigma) from the trainer is written to all checkpoints
        alongside the scalar fallback fraction. If Manual mode is selected,
        only the scalar fraction from the spinbox is written.
        """
        use_computed = self.radioButton_blobRadiusComputed.isChecked()

        if use_computed:
            dist = getattr(trainer, "suggested_blob_radius_result", None)
            if dist is None:
                print("[Blob Filter] Computed mode selected but no result available. "
                      "Spinbox and checkpoints unchanged.")
                return
            fallback_fraction = dist.get("fallback_fraction", trainer.blob_filter_radius)
            diagonal_px       = dist.get("diagonal_px", 1.0)
            suggested_px      = fallback_fraction * diagonal_px
            trainer._update_checkpoints_blob_radius(fallback_fraction,
                                                     blob_distribution=dist)
            trainer.blob_filter_radius = fallback_fraction
            self.spinBox_blobFilterRadius.setValue(int(round(suggested_px)))
            self._update_blob_filter_pct_label()
            cov_status = "Mahalanobis" if dist.get("centroid_cov") else "scalar fallback"
            print("[Blob Filter] Computed distribution saved automatically "
                  "({}, {} px fallback).".format(cov_status, int(round(suggested_px))))
        else:
            manual_fraction = self._blob_pixels_to_fraction()
            trainer._update_checkpoints_blob_radius(manual_fraction)
            trainer.blob_filter_radius = manual_fraction
            print("[Blob Filter] Manual radius saved automatically: "
                  "{} px ({:.2f}% of diagonal).".format(
                      self.spinBox_blobFilterRadius.value(), manual_fraction * 100.0))

