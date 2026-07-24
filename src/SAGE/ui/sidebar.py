# sam2_gui/ui/sidebar.py
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QHBoxLayout,
    QPushButton,
    QButtonGroup,
    QGroupBox,
    QGridLayout,
    QCheckBox,
    QStackedWidget,
    QRadioButton,
    QSizePolicy,
    QFrame,
    QLineEdit,
    QInputDialog,
    QMenu,
    QAction,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QByteArray
from PyQt5.QtGui import QColor, QFont, QPixmap, QPainter, QIcon
try:
    from PyQt5.QtSvg import QSvgRenderer
    _HAS_SVG = True
except Exception:
    _HAS_SVG = False


_ICON_COLOR = "#8a9099"   # mid gray: legible on light and dark buttons

_TOOL_SVGS = {
    "draw": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            '<path fill="{c}" d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25z'
            'M20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0'
            'l-1.83 1.83 3.75 3.75 1.83-1.83z"/></svg>',
    "select": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
              '<path fill="{c}" d="M6 2 L6 18 L10 14 L13 20 L15 19 L12 13 L17 13 Z"/></svg>',
    "pan": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
           '<path fill="{c}" d="M23 5.5V20c0 2.2-1.8 4-4 4h-7.3c-1.08 0-2.1-.43-2.85-1.19'
           'L1 14.83c0 0 1.26-1.23 1.3-1.25.22-.19.49-.29.79-.29.22 0 .42.06.6.16'
           '.04.01 4.31 2.46 4.31 2.46V4c0-.83.67-1.5 1.5-1.5S12 3.17 12 4v7h1V1.5'
           'c0-.83.67-1.5 1.5-1.5S16 .67 16 1.5V11h1V2.5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5V11'
           'h1V5.5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5z"/></svg>',
    # Four-pointed sparkle: marks a mode whose result comes from SAM2.
    "sam2": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            '<path fill="{c}" d="M10 6 C10.6 11.2, 13.6 13.4, 18 14 C13.6 14.6, 10.6 16.8, 10 22'
            ' C9.4 16.8, 6.4 14.6, 2 14 C6.4 13.4, 9.4 11.2, 10 6 Z"/>'
            '<path fill="{c}" d="M18.5 2 C18.8 4.3, 20.2 5.2, 22 5.5 C20.2 5.8, 18.8 6.7, 18.5 9'
            ' C18.2 6.7, 16.8 5.8, 15 5.5 C16.8 5.2, 18.2 4.3, 18.5 2 Z"/></svg>',
}

_TOOL_FALLBACK = {"draw": "Draw", "select": "Select", "pan": "Pan"}


def _tool_icon(kind: str, size: int = 22) -> QIcon:
    svg = _TOOL_SVGS[kind].format(c=_ICON_COLOR)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    renderer.render(p)
    p.end()
    return QIcon(pix)


def _mark_sam2(btn) -> None:
    """Badge a mode button whose result comes from SAM2. Deliberately NOT
    applied to Manual Polygon / Manual Free-form: those fill the drawn shape
    directly, and the absence of the mark is what carries the distinction."""
    if _HAS_SVG:
        btn.setIcon(_tool_icon("sam2", 14))
        btn.setIconSize(QSize(14, 14))
    else:
        btn.setText("\u2726  " + btn.text())   # QtSvg missing → glyph fallback
import os


# ---------------------------------------------------------------------------
# Sub-panel state constants
# ---------------------------------------------------------------------------
MASK_STATE_LOCKED   = "locked"    # mode active, no mask loaded yet
MASK_STATE_LOADED   = "loaded"    # mask loaded, seeds placed, ready to segment
MASK_STATE_PERSISTS = "persists"  # image changed, mask carried forward


class Sidebar(QWidget):
    # Existing signals
    save_all_coco_requested     = pyqtSignal()
    mask_unlabeled_requested    = pyqtSignal()
    eraser_toggled              = pyqtSignal(bool)
    segmentation_mode_changed   = pyqtSignal(str)   # "points","polygon","paint","manual_polygon","mask"
    tool_mode_changed           = pyqtSignal(str)   # "draw","select","pan"
    polygon_sampling_changed    = pyqtSignal(str)   # "dense","random","poisson"
    mask_selected               = pyqtSignal(int)   # mask_id; -1 = none selected
    mask_reclicked              = pyqtSignal(int)   # mask_id; re-click of already-selected row
    mask_renamed                = pyqtSignal(int, str)  # mask_id, new_name
    label_class_renamed         = pyqtSignal(str, str)  # old_name, new_name
    label_class_id_changed      = pyqtSignal(str, int, int)  # name, old_id, new_id

    # Seed-mask signals (previously lived in seed_groupbox)
    load_seed_mask_requested    = pyqtSignal()
    clear_seed_mask_requested   = pyqtSignal()
    seed_points_now_requested   = pyqtSignal()
    auto_seed_toggled           = pyqtSignal(bool)

    def __init__(
            self,
            controller,
            on_run_segmentation,
            on_visibility_changed,
            on_clear_points,
            parent=None
    ):
        super().__init__(parent)
        self.controller           = controller
        self.on_run_segmentation  = on_run_segmentation
        self.on_visibility_changed = on_visibility_changed
        self.on_clear_points      = on_clear_points

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Internal state (must exist before lists are rebuilt)
        self._label_classes = []   # list of {"name": str, "id": int, "protected"?: bool}, ordered
        self._label_classes.append({"name": "Other", "id": 999, "protected": True})
        self._active_label  = None

        # ====================================================
        # Section 1 — Label Classes
        # ====================================================
        label_group = QGroupBox("Label Classes")
        lg = QVBoxLayout(label_group)
        lg.setContentsMargins(8, 10, 8, 8)
        lg.setSpacing(6)

        self.label_class_list = QTableWidget(0, 2)
        self.label_class_list.setHorizontalHeaderLabels(["Name", "ID"])
        self.label_class_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.label_class_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.label_class_list.verticalHeader().setVisible(False)
        self.label_class_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.label_class_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.label_class_list.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.label_class_list.setFixedHeight(150)
        self.label_class_list.itemSelectionChanged.connect(self._on_label_class_selection_changed)
        self.label_class_list.itemChanged.connect(self._on_label_class_item_changed)
        self.label_class_list.itemDoubleClicked.connect(self.label_class_list.editItem)
        self.label_class_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_class_list.customContextMenuRequested.connect(self._on_label_class_context_menu)
        lg.addWidget(self.label_class_list)

        add_label_row = QHBoxLayout()
        self.new_label_input = QLineEdit()
        self.new_label_input.setPlaceholderText("New label name…")
        self.new_label_input.returnPressed.connect(self._add_label_class)
        add_label_row.addWidget(self.new_label_input)
        add_btn = QPushButton("Add")
        add_btn.setFixedWidth(38)
        add_btn.clicked.connect(self._add_label_class)
        add_label_row.addWidget(add_btn)
        del_btn = QPushButton("Del")
        del_btn.setFixedWidth(40)
        del_btn.setToolTip("Delete selected label class")
        del_btn.clicked.connect(self._delete_label_class)
        add_label_row.addWidget(del_btn)
        clr_btn = QPushButton("Clr")
        clr_btn.setFixedWidth(40)
        clr_btn.setToolTip("Clear all label classes (keeps 'Other')")
        clr_btn.clicked.connect(self._clear_label_classes)
        add_label_row.addWidget(clr_btn)
        lg.addLayout(add_label_row)
        layout.addWidget(label_group)

        # ====================================================
        # Section 2 — Masks on this image  (+ Segment trigger)
        # ====================================================
        masks_group = QGroupBox("Masks on this image")
        mg = QVBoxLayout(masks_group)
        mg.setContentsMargins(8, 10, 8, 8)
        mg.setSpacing(6)

        self.mask_list = QListWidget()
        self.mask_list.setFixedHeight(160)
        self.mask_list.itemChanged.connect(self._on_item_changed)
        self.mask_list.itemSelectionChanged.connect(self._on_mask_selection_changed)
        # A click on the already-selected row emits mask_reclicked (used to
        # stop a mask flash). itemClicked fires on release, after any
        # itemSelectionChanged on press, so the flag below tells a genuine
        # re-click (no selection change) from a new-row click.
        self.mask_list.itemClicked.connect(self._on_mask_item_clicked)
        self._mask_selection_changed = False
        self.mask_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mask_list.customContextMenuRequested.connect(self._on_mask_context_menu)
        mg.addWidget(self.mask_list)

        self.run_btn = QPushButton("Segment (Enter)")
        self.run_btn.clicked.connect(self.on_run_segmentation)
        mg.addWidget(self.run_btn)
        layout.addWidget(masks_group)

        # ====================================================
        # Tool — Draw / Select / Pan (top-level interaction mode)
        # ====================================================
        tool_groupbox = QGroupBox("Tool")
        tool_row = QHBoxLayout(tool_groupbox)
        tool_row.setContentsMargins(8, 10, 8, 8)
        tool_row.setSpacing(6)
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)

        def _make_tool_btn(kind, tip):
            b = QPushButton()
            b.setToolTip(tip)
            b.setCheckable(True)
            if _HAS_SVG:
                b.setIcon(_tool_icon(kind))
                b.setIconSize(QSize(22, 22))
            else:
                b.setText(_TOOL_FALLBACK[kind])   # QtSvg missing → readable text
            b.setMinimumHeight(38)
            b.clicked.connect(lambda: self.tool_mode_changed.emit(kind))
            self._tool_group.addButton(b)
            tool_row.addWidget(b)
            return b

        self.draw_tool_btn   = _make_tool_btn("draw",   "Draw — annotate with the segmentation tools")
        self.select_tool_btn = _make_tool_btn("select", "Select — click a mask to highlight it in the list")
        self.pan_tool_btn    = _make_tool_btn("pan",    "Pan — drag to move the image when zoomed in")
        self.draw_tool_btn.setChecked(True)
        layout.addWidget(tool_groupbox)

        # ====================================================
        # Section 3 — Segmentation Mode  (radio sampling under Polygon)
        # ====================================================
        mode_groupbox = QGroupBox("Segmentation Mode")
        mode_outer    = QVBoxLayout(mode_groupbox)
        mode_outer.setContentsMargins(8, 10, 8, 8)
        mode_outer.setSpacing(6)

        mode_grid         = QGridLayout()
        mode_button_group = QButtonGroup(self)
        mode_button_group.setExclusive(True)

        self.points_btn = QPushButton("Click or Drag")
        self.points_btn.setToolTip("Click a point, or click-drag to add many. "
                                   "Left = foreground, right = background.")
        self.points_btn.setCheckable(True)
        self.points_btn.setChecked(True)
        self.points_btn.clicked.connect(lambda: self._on_mode_button_clicked("points"))
        _mark_sam2(self.points_btn)
        mode_button_group.addButton(self.points_btn)
        mode_grid.addWidget(self.points_btn, 0, 0, 1, 2)

        self.polygon_btn = QPushButton("Polygon")
        self.polygon_btn.setCheckable(True)
        self.polygon_btn.clicked.connect(lambda: self._on_mode_button_clicked("polygon"))
        _mark_sam2(self.polygon_btn)
        mode_button_group.addButton(self.polygon_btn)
        mode_grid.addWidget(self.polygon_btn, 1, 0)

        # Polygon sampling — radio buttons stacked directly under the Polygon button.
        samp_box = QVBoxLayout()
        samp_box.setContentsMargins(8, 2, 0, 2)
        samp_box.setSpacing(2)
        sampling_group = QButtonGroup(self)
        sampling_group.setExclusive(True)

        self.dense_radio = QRadioButton("Grid")
        self.dense_radio.setChecked(True)
        self.dense_radio.clicked.connect(lambda: self._on_sampling_button_clicked("dense"))
        sampling_group.addButton(self.dense_radio)
        samp_box.addWidget(self.dense_radio)

        self.random_radio = QRadioButton("Random")
        self.random_radio.clicked.connect(lambda: self._on_sampling_button_clicked("random"))
        sampling_group.addButton(self.random_radio)
        samp_box.addWidget(self.random_radio)

        self.poisson_radio = QRadioButton("Disc")
        self.poisson_radio.clicked.connect(lambda: self._on_sampling_button_clicked("poisson"))
        sampling_group.addButton(self.poisson_radio)
        samp_box.addWidget(self.poisson_radio)

        self.sampling_container = QWidget()
        self.sampling_container.setLayout(samp_box)
        mode_grid.addWidget(self.sampling_container, 2, 0)

        self.mask_btn = QPushButton("Import Mask")
        self.mask_btn.setCheckable(True)
        self.mask_btn.clicked.connect(lambda: self._on_mode_button_clicked("mask"))
        _mark_sam2(self.mask_btn)
        mode_button_group.addButton(self.mask_btn)
        mode_grid.addWidget(self.mask_btn, 5, 0, 1, 2)

        self.manual_draw_btn = QPushButton("Free-form")
        self.manual_draw_btn.setCheckable(True)
        self.manual_draw_btn.clicked.connect(lambda: self._on_mode_button_clicked("manual_draw"))
        _mark_sam2(self.manual_draw_btn)
        mode_button_group.addButton(self.manual_draw_btn)
        mode_grid.addWidget(self.manual_draw_btn, 3, 0)

        # Manual Free-form: freehand drag filled directly as a mask (like
        # Manual Polygon, but a traced stroke instead of clicked vertices; no SAM2).
        self.manual_freeform_btn = QPushButton("Manual Free-form")
        self.manual_freeform_btn.setCheckable(True)
        self.manual_freeform_btn.setToolTip(
            "Drag to trace a free-form outline; the enclosed shape is filled "
            "directly as a mask (no SAM2), like Manual Polygon.")
        self.manual_freeform_btn.clicked.connect(lambda: self._on_mode_button_clicked("manual_freeform"))
        mode_button_group.addButton(self.manual_freeform_btn)
        mode_grid.addWidget(self.manual_freeform_btn, 3, 1)

        self.manual_polygon_btn = QPushButton("Manual Polygon")
        self.manual_polygon_btn.setCheckable(True)
        self.manual_polygon_btn.clicked.connect(lambda: self._on_mode_button_clicked("manual_polygon"))
        mode_button_group.addButton(self.manual_polygon_btn)
        mode_grid.addWidget(self.manual_polygon_btn, 1, 1)

        self.edge_trace_btn = QPushButton("⟿  Edge Trace")
        self.edge_trace_btn.setCheckable(True)
        self.edge_trace_btn.setToolTip(
            "Drag along the center of an object. Positive points are placed along the stroke; "
            "negative points are placed just outside the detected edges on each side.\n"
            "+/- : adjust sample interval   [ / ] : adjust search width"
        )
        self.edge_trace_btn.clicked.connect(lambda: self._on_mode_button_clicked("edge_trace"))
        _mark_sam2(self.edge_trace_btn)
        mode_button_group.addButton(self.edge_trace_btn)
        mode_grid.addWidget(self.edge_trace_btn, 4, 0, 1, 2)

        mode_outer.addLayout(mode_grid)

        # ---- Mask sub-panel (QStackedWidget with 3 pages) ----
        self.mask_subpanel = QStackedWidget()
        self.mask_subpanel.setVisible(False)
        self.mask_subpanel.addWidget(self._build_subpanel_locked())
        self.mask_subpanel.addWidget(self._build_subpanel_loaded())
        self.mask_subpanel.addWidget(self._build_subpanel_persists())
        mode_outer.addWidget(self.mask_subpanel)
        layout.addWidget(mode_groupbox)

        # sampling starts disabled (default mode = points)
        self._update_sampling_enabled_state(False)

        # ====================================================
        # Section 4 — Segment (Erase / Clear Points)
        # ====================================================
        seg_groupbox = QGroupBox("Segment")
        seg_layout   = QHBoxLayout(seg_groupbox)
        seg_layout.setContentsMargins(8, 10, 8, 8)
        seg_layout.setSpacing(6)

        self.erase_btn = QPushButton("Erase")
        self.erase_btn.setCheckable(True)
        self.erase_btn.clicked.connect(self._on_erase_clicked)
        seg_layout.addWidget(self.erase_btn)

        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.clicked.connect(self.on_clear_points)
        seg_layout.addWidget(self.clear_points_btn)
        layout.addWidget(seg_groupbox)

        # ====================================================
        # Section 5 — Mask Unlabeled + Save
        # ====================================================
        self.mask_unlabeled_btn = QPushButton("Mask Unlabeled")
        self.mask_unlabeled_btn.setToolTip(
            "Assign every pixel not yet in a mask to a single 'Other' region.\n"
            "Use as a final step, after all target regions are labeled. Re-click to regenerate.")
        self.mask_unlabeled_btn.clicked.connect(self.mask_unlabeled_requested.emit)
        layout.addWidget(self.mask_unlabeled_btn)

        self.save_all_btn = QPushButton("Save COCO 1.0 (All Images)")
        self.save_all_btn.clicked.connect(self.save_all_coco_requested.emit)
        self.save_all_btn.setMinimumHeight(42)
        self.save_all_btn.setObjectName("saveButton")   # themed as the primary/commit action
        layout.addWidget(self.save_all_btn)
        layout.addStretch(1)

        self.setLayout(layout)
        self.setObjectName("sagePanel")

        # Draw the label table now so the protected 'Other' class is visible from
        # launch, rather than appearing only when the first label is added.
        self._rebuild_label_class_list()

        from SAGE.ui.theme import apply_theme
        apply_theme(self, "light")   # call self.set_theme("light"/"dark") to switch

    # =========================================================
    # Sub-panel builders
    # =========================================================

    def _build_subpanel_locked(self):
        """
        State A: SAM2 Mask mode active, no mask loaded yet.
        Only 'Load Mask File...' is active. Everything else is ghosted.
        """
        w = QFrame()
        w.setObjectName("subpanel_locked")
        w.setStyleSheet(
            "#subpanel_locked { background: #efefef;"
            " border: 1px dashed #bbb; border-radius: 4px; }"
        )
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(5)

        hint = QLabel("Load a mask to continue")
        hint.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        vbox.addWidget(hint)

        self.load_mask_btn_locked = QPushButton("Load Mask File...")
        self.load_mask_btn_locked.clicked.connect(self.load_seed_mask_requested.emit)
        vbox.addWidget(self.load_mask_btn_locked)

        ghost_label = QLabel("Sampling and auto-seed unlock after loading")
        ghost_label.setStyleSheet("color: #bbb; font-size: 10px;")
        ghost_label.setWordWrap(True)
        vbox.addWidget(ghost_label)

        return w

    def _build_subpanel_loaded(self):
        """
        State B: Mask confirmed, seeds placed. Full controls active.
        """
        w = QFrame()
        w.setObjectName("subpanel_loaded")
        w.setStyleSheet(
            "#subpanel_loaded { background: #f0faf0;"
            " border: 1px solid #a5d6a7; border-radius: 4px; }"
        )
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(5)

        self.loaded_filename_label = QLabel("No file")
        self.loaded_filename_label.setStyleSheet(
            "color: #2e7d32; font-weight: bold; font-size: 11px;"
        )
        self.loaded_filename_label.setWordWrap(True)
        vbox.addWidget(self.loaded_filename_label)

        btn_row = QHBoxLayout()
        self.load_new_btn = QPushButton("Load New...")
        self.load_new_btn.clicked.connect(self.load_seed_mask_requested.emit)
        btn_row.addWidget(self.load_new_btn)

        self.unload_btn = QPushButton("Unload")
        self.unload_btn.clicked.connect(self.clear_seed_mask_requested.emit)
        btn_row.addWidget(self.unload_btn)
        vbox.addLayout(btn_row)

        self.auto_seed_checkbox = QCheckBox("Auto-Seed: ON")
        self.auto_seed_checkbox.setChecked(True)
        self.auto_seed_checkbox.toggled.connect(self._on_auto_seed_changed)
        vbox.addWidget(self.auto_seed_checkbox)

        self.seed_now_btn = QPushButton("Apply Seeds")
        self.seed_now_btn.setEnabled(False)   # only active when Auto-Seed is OFF
        self.seed_now_btn.clicked.connect(self.seed_points_now_requested.emit)
        vbox.addWidget(self.seed_now_btn)

        return w

    def _build_subpanel_persists(self):
        """
        State C: Image changed, mask carried forward. Amber warning.
        """
        w = QFrame()
        w.setObjectName("subpanel_persists")
        w.setStyleSheet(
            "#subpanel_persists { background: #fffde7;"
            " border: 1px solid #ffe082; border-radius: 4px; }"
        )
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(5)

        title = QLabel("Mask carried forward")
        title.setStyleSheet("color: #b45309; font-weight: bold; font-size: 11px;")
        vbox.addWidget(title)

        self.persists_filename_label = QLabel("")
        self.persists_filename_label.setStyleSheet("color: #78350f; font-size: 10px;")
        self.persists_filename_label.setWordWrap(True)
        vbox.addWidget(self.persists_filename_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #ffe082;")
        vbox.addWidget(sep)

        btn_row = QHBoxLayout()
        self.load_new_btn_persists = QPushButton("Load New...")
        self.load_new_btn_persists.clicked.connect(self.load_seed_mask_requested.emit)
        btn_row.addWidget(self.load_new_btn_persists)

        self.unload_btn_persists = QPushButton("Unload")
        self.unload_btn_persists.clicked.connect(self.clear_seed_mask_requested.emit)
        btn_row.addWidget(self.unload_btn_persists)
        vbox.addLayout(btn_row)

        # Mirror the auto-seed checkbox state (read-only display; actual checkbox is in loaded panel)
        self.auto_seed_persists_label = QLabel("Auto-Seed: ON")
        self.auto_seed_persists_label.setStyleSheet("color: #78350f; font-size: 10px;")
        vbox.addWidget(self.auto_seed_persists_label)

        return w

    # =========================================================
    # Public method: MainWindow calls this on image change or
    # mask load/unload to drive sub-panel state.
    # =========================================================

    def set_mask_subpanel_state(self, state: str, mask_path: str = None):
        """
        state: MASK_STATE_LOCKED | MASK_STATE_LOADED | MASK_STATE_PERSISTS
        mask_path: full path to loaded mask (used for filename display)
        """
        name = os.path.basename(mask_path) if mask_path else ""

        if state == MASK_STATE_LOCKED:
            self.mask_subpanel.setCurrentIndex(0)
        elif state == MASK_STATE_LOADED:
            self.loaded_filename_label.setText(f"✓ {name}")
            self.mask_subpanel.setCurrentIndex(1)
        elif state == MASK_STATE_PERSISTS:
            self.persists_filename_label.setText(
                f"{name} is still loaded from the previous image. "
                "Seeds have been re-applied automatically."
            )
            # Keep auto-seed label in sync
            auto_on = self.auto_seed_checkbox.isChecked()
            self.auto_seed_persists_label.setText(
                "Auto-Seed: ON" if auto_on else "Auto-Seed: OFF"
            )
            self.mask_subpanel.setCurrentIndex(2)

    # =========================================================
    # Segment button gating
    # =========================================================

    def update_segment_button_state(self):
        if self.controller is None:
            self.run_btn.setEnabled(False)
        else:
            has_points = (
                len(self.controller.fg_points) > 0
                or len(self.controller.bg_points) > 0
            )
            self.run_btn.setEnabled(has_points)

    # =========================================================
    # Mode button handler
    # =========================================================

    def _on_mode_button_clicked(self, mode: str):
        self.segmentation_mode_changed.emit(mode)

        is_mask    = mode == "mask"
        is_polygon = mode in ("polygon", "paint", "mask", "manual_draw")

        # Show / hide the mask sub-panel
        self.mask_subpanel.setVisible(is_mask)

        # Sampling enabled for polygon-family modes
        self._update_sampling_enabled_state(is_polygon)

        # SAM2 Mask defaults to Random sampling
        if is_mask:
            self.random_radio.setChecked(True)
            self._on_sampling_button_clicked("random")

        # Selected-button appearance is handled by the theme's QSS :checked rule.

    def set_theme(self, mode: str):
        """Switch the panel between 'light' and 'dark'."""
        from SAGE.ui.theme import apply_theme
        apply_theme(self, mode)

    def _update_sampling_enabled_state(self, enabled: bool):
        for btn in (self.dense_radio, self.random_radio, self.poisson_radio):
            btn.setEnabled(enabled)

    def _on_sampling_button_clicked(self, mode: str):
        self.polygon_sampling_changed.emit(mode)

    # =========================================================
    # Erase button
    # =========================================================

    def _on_erase_clicked(self, checked: bool):
        self.eraser_toggled.emit(checked)

    # =========================================================
    # Auto-seed checkbox (lives inside the loaded sub-panel)
    # =========================================================

    def _on_auto_seed_changed(self, on: bool):
        self.auto_seed_checkbox.setText("Auto-Seed: ON" if on else "Auto-Seed: OFF")
        self.seed_now_btn.setEnabled(not on)
        self.auto_seed_toggled.emit(on)

    # =========================================================
    # Called by MainWindow on image change
    # (replaces the old set_seed_controls_state signature)
    # =========================================================

    def set_seed_controls_state(self, auto_on: bool):
        """Sync auto-seed checkbox without triggering signals."""
        self.auto_seed_checkbox.blockSignals(True)
        self.auto_seed_checkbox.setChecked(auto_on)
        self.auto_seed_checkbox.setText("Auto-Seed: ON" if auto_on else "Auto-Seed: OFF")
        self.seed_now_btn.setEnabled(not auto_on)
        self.auto_seed_checkbox.blockSignals(False)

    # =========================================================
    # Label Class management
    # =========================================================

    # Predefined distinct colors for label classes
    _LABEL_COLORS = [
        "#e53935", "#43a047", "#1e88e5", "#fb8c00", "#8e24aa",
        "#00acc1", "#f4511e", "#6d4c41", "#3949ab", "#00897b",
    ]

    def _label_color(self, index: int) -> str:
        return self._LABEL_COLORS[index % len(self._LABEL_COLORS)]

    def _names(self) -> list:
        return [lc["name"] for lc in self._label_classes]

    def _ids(self) -> list:
        return [lc["id"] for lc in self._label_classes]

    def _next_id(self) -> int:
        return (max(self._ids()) + 1) if self._label_classes else 1

    def _add_label_class(self):
        """Add a new label class from the input field."""
        name = self.new_label_input.text().strip()
        if not name or name in self._names():
            self.new_label_input.clear()
            return
        self._label_classes.append({"name": name, "id": self._next_id()})
        self._rebuild_label_class_list()
        if len(self._label_classes) == 1:
            self._set_active_label(name)
        self.new_label_input.clear()

    def add_label_classes_from_mask(self, names: list):
        """Called when SAM2 Mask file is loaded — populate classes from mask."""
        for name in names:
            if name and name not in self._names():
                self._label_classes.append({"name": name, "id": self._next_id()})
        self._rebuild_label_class_list()
        if not self._active_label and self._label_classes:
            self._set_active_label(self._label_classes[0]["name"])

    def _delete_label_class(self):
        row = self.label_class_list.currentRow()
        if row < 0:
            return
        if self._label_classes[row].get("protected"):
            return
        name = self._label_classes[row]["name"]
        self._label_classes.pop(row)
        if self._active_label == name:
            self._active_label = self._label_classes[0]["name"] if self._label_classes else None
        self._rebuild_label_class_list()

    def _clear_label_classes(self):
        """Remove all user-defined label classes, keeping the protected
        'Other'. For starting a new dataset with different labels. Does not
        delete masks already on the canvas."""
        removable = [lc for lc in self._label_classes if not lc.get("protected")]
        if not removable:
            return
        from PyQt5.QtWidgets import QMessageBox
        resp = QMessageBox.question(
            self, "Clear Labels",
            f"Remove all {len(removable)} label class(es)? 'Other' is kept.\n\n"
            "Masks already on the canvas are not deleted.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if resp != QMessageBox.Yes:
            return
        self._label_classes = [lc for lc in self._label_classes if lc.get("protected")]
        non_protected = [lc["name"] for lc in self._label_classes if not lc.get("protected")]
        self._active_label = non_protected[0] if non_protected else None
        self._rebuild_label_class_list()

    def _rebuild_label_class_list(self):
        tbl = self.label_class_list
        tbl.blockSignals(True)
        tbl.setRowCount(0)
        for i, lc in enumerate(self._label_classes):
            tbl.insertRow(i)
            protected = lc.get("protected", False)
            color = QColor("#c026d3") if protected else QColor(self._label_color(i))  # predefined = magenta
            highlight = QColor("#dbeafe") if lc["name"] == self._active_label else None
            editable = Qt.ItemIsEditable if not protected else Qt.NoItemFlags
            tip = "Predefined class — cannot be renamed, re-numbered, or deleted." if protected else None

            name_item = QTableWidgetItem(lc["name"])
            name_item.setData(Qt.UserRole, lc["name"])   # original for revert/diff
            name_item.setForeground(color)
            name_item.setFlags((name_item.flags() | Qt.ItemIsEditable) if not protected
                               else name_item.flags() & ~Qt.ItemIsEditable)
            if tip:
                name_item.setToolTip(tip)
            if highlight:
                name_item.setBackground(highlight)
            tbl.setItem(i, 0, name_item)

            id_item = QTableWidgetItem(str(lc["id"]))
            id_item.setData(Qt.UserRole, lc["id"])       # original for revert
            id_item.setTextAlignment(Qt.AlignCenter)
            id_item.setForeground(color)
            id_item.setFlags((id_item.flags() | Qt.ItemIsEditable) if not protected
                             else id_item.flags() & ~Qt.ItemIsEditable)
            if tip:
                id_item.setToolTip(tip)
            if highlight:
                id_item.setBackground(highlight)
            tbl.setItem(i, 1, id_item)
        tbl.blockSignals(False)

    def _on_label_class_selection_changed(self):
        row = self.label_class_list.currentRow()
        if 0 <= row < len(self._label_classes):
            self._set_active_label(self._label_classes[row]["name"])

    def _on_label_class_item_changed(self, item: QTableWidgetItem):
        row = item.row()
        col = item.column()
        if row < 0 or row >= len(self._label_classes):
            return
        if self._label_classes[row].get("protected"):
            self._rebuild_label_class_list()
            return

        self.label_class_list.blockSignals(True)
        try:
            if col == 0:
                new_name = item.text().strip()
                old_name = item.data(Qt.UserRole)
                if not new_name or new_name == old_name:
                    item.setText(old_name)
                    return
                if new_name in self._names():
                    item.setText(old_name)  # duplicate — revert
                    return
                self._label_classes[row]["name"] = new_name
                if self._active_label == old_name:
                    self._active_label = new_name
                self.label_class_renamed.emit(old_name, new_name)
            elif col == 1:
                old_id = item.data(Qt.UserRole)
                try:
                    new_id = int(item.text().strip())
                except ValueError:
                    item.setText(str(old_id))
                    return
                if new_id == old_id:
                    return
                if new_id in (lc["id"] for i, lc in enumerate(self._label_classes) if i != row):
                    item.setText(str(old_id))  # duplicate — revert
                    return
                self._label_classes[row]["id"] = new_id
                self.label_class_id_changed.emit(
                    self._label_classes[row]["name"], old_id, new_id
                )
        finally:
            self.label_class_list.blockSignals(False)
            self._rebuild_label_class_list()

    def _on_label_class_context_menu(self, pos):
        item = self.label_class_list.itemAt(pos)
        if item is None:
            return
        if self._label_classes[item.row()].get("protected"):
            return
        menu = QMenu(self)
        rename_action = QAction("Rename…", self)
        rename_action.triggered.connect(lambda: self._rename_label_class(item.row()))
        menu.addAction(rename_action)
        menu.exec_(self.label_class_list.mapToGlobal(pos))

    def _rename_label_class(self, row: int):
        if not (0 <= row < len(self._label_classes)):
            return
        old_name = self._label_classes[row]["name"]
        new_name, ok = QInputDialog.getText(
            self, "Rename Label", "New label name:", text=old_name
        )
        new_name = new_name.strip()
        if not ok or not new_name or new_name == old_name:
            return
        if new_name in self._names():
            return  # already exists
        self._label_classes[row]["name"] = new_name
        if self._active_label == old_name:
            self._active_label = new_name
        self._rebuild_label_class_list()
        self.label_class_renamed.emit(old_name, new_name)

    def _set_active_label(self, name: str):
        self._active_label = name
        # Recolor rows in place — do NOT rebuild (that cancels an open editor).
        tbl = self.label_class_list
        tbl.blockSignals(True)
        for i, lc in enumerate(self._label_classes):
            hl = QColor("#dbeafe") if lc["name"] == name else QColor(Qt.white)
            for c in (0, 1):
                it = tbl.item(i, c)
                if it:
                    it.setBackground(hl)
        tbl.blockSignals(False)

    def get_active_label(self) -> str | None:
        """Return active label, or None if no label classes have been defined."""
        return self._active_label or None

    def get_active_label_id(self) -> int | None:
        """Return the ID of the currently active label, or None."""
        for lc in self._label_classes:
            if lc["name"] == self._active_label:
                return lc["id"]
        return None

    def get_label_classes(self) -> list:
        """Return the current ordered list of label class names (backward-compatible)."""
        return self._names()

    def get_label_classes_with_ids(self) -> list:
        """Return ordered list of (name, id) tuples — use for COCO export."""
        return [(lc["name"], lc["id"]) for lc in self._label_classes]

    def set_label_classes(self, names, ids=None):
        """
        Replace all label classes. Accepts either:
          - list[str] (auto-assign IDs starting at 1)
          - list[(name, id)] tuples
          - list[str] + parallel `ids` list
        """
        self._label_classes = []
        if names and isinstance(names[0], tuple):
            for name, lid in names:
                self._label_classes.append({"name": name, "id": int(lid)})
        elif ids is not None:
            for name, lid in zip(names, ids):
                self._label_classes.append({"name": name, "id": int(lid)})
        else:
            for i, name in enumerate(names, start=1):
                self._label_classes.append({"name": name, "id": i})
        # A predefined 'Other' (id 999, protected) must always exist and cannot
        # be loaded away by a file's categories. Promote an existing one to
        # protected, or seed it; keep it pinned at the top.
        self._label_classes = [lc for lc in self._label_classes if lc["name"] != "Other"]
        self._label_classes.insert(0, {"name": "Other", "id": 999, "protected": True})
        # Reset active label to first class if current one no longer exists
        if self._active_label not in self._names():
            non_protected = [lc["name"] for lc in self._label_classes if not lc.get("protected")]
            self._active_label = non_protected[0] if non_protected else None
        self._rebuild_label_class_list()

    def get_color_for_label(self, name: str) -> tuple:
        """Return (r, g, b) tuple for a label class, consistent across all instances."""
        try:
            idx = self._names().index(name)
        except ValueError:
            idx = abs(hash(name)) % len(self._LABEL_COLORS)
        h = self._label_color(idx).lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    # =========================================================
    # =========================================================
    # Mask list (instances on this image)
    # =========================================================

    def select_mask_in_list(self, mask_id: int):
        """Highlight and scroll to the row for `mask_id` without re-emitting
        mask_selected (avoids a canvas→list→canvas feedback loop)."""
        for i in range(self.mask_list.count()):
            item = self.mask_list.item(i)
            if item.data(Qt.UserRole) == mask_id:
                self.mask_list.blockSignals(True)
                self.mask_list.setCurrentRow(i)
                self.mask_list.blockSignals(False)
                self.mask_list.scrollToItem(item)
                return

    def refresh_masks(self):
        self.mask_list.blockSignals(True)
        self.mask_list.clear()
        for m in self.controller.masks:
            item = QListWidgetItem(m["label"])
            item.setFlags(
                item.flags()
                | Qt.ItemIsUserCheckable
                | Qt.ItemIsEnabled
            )
            item.setCheckState(Qt.Checked if m["visible"] else Qt.Unchecked)
            item.setData(Qt.UserRole, m["id"])
            self.mask_list.addItem(item)
        self.mask_list.blockSignals(False)

    def _on_item_changed(self, item: QListWidgetItem):
        mask_id = item.data(Qt.UserRole)
        self.controller.set_mask_visibility(mask_id, item.checkState() == Qt.Checked)
        self.on_visibility_changed()

    def _on_mask_selection_changed(self):
        # Mark that this click changed the selection; consumed in
        # _on_mask_item_clicked so the paired itemClicked is not a re-click.
        self._mask_selection_changed = True
        items = self.mask_list.selectedItems()
        if items:
            self.mask_selected.emit(items[0].data(Qt.UserRole))
        else:
            self.mask_selected.emit(-1)

    def _on_mask_item_clicked(self, item: QListWidgetItem):
        """Emit mask_reclicked when the click landed on the already-selected
        row. A new-row click fires itemSelectionChanged (flag set) before this;
        a re-click fires only itemClicked, so a clear flag here means the
        selection did not change."""
        if self._mask_selection_changed:
            self._mask_selection_changed = False   # paired with a new-row click
            return
        if item is not None:
            self.mask_reclicked.emit(item.data(Qt.UserRole))

    def _on_mask_context_menu(self, pos):
        item = self.mask_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        rename_action = QAction("Rename…", self)
        rename_action.triggered.connect(lambda: self._rename_mask(item))
        menu.addAction(rename_action)
        menu.exec_(self.mask_list.mapToGlobal(pos))

    def _rename_mask(self, item: QListWidgetItem):
        if not self._label_classes:
            QInputDialog.getText(self, "No Labels", "No label classes defined.")
            return
        mask_id = item.data(Qt.UserRole)
        old_name = item.text()
        names = self._names()
        current_idx = names.index(old_name) if old_name in names else 0
        new_name, ok = QInputDialog.getItem(
            self, "Rename Mask", "Select label:",
            names, current_idx, editable=False
        )
        if not ok or new_name == old_name:
            return
        self.mask_renamed.emit(mask_id, new_name)
