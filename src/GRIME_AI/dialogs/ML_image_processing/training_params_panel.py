"""
training_params_panel.py

Builds the entire Training Parameters panel: wraps the parameters group box
in an Advanced / Guided QTabWidget, reflows the Advanced form into two
columns, and provides the Guided accordion.

  * Advanced — the existing flat form, untouched.
  * Guided   — an accordion of grouped, annotated sections aimed at first-time
               users. Each guided widget mirrors its Advanced counterpart and
               writes through immediately, so the Advanced widgets remain the
               single source of truth (train() and _collect_ui_values() read
               them unchanged).

Install with:

    from GRIME_AI.dialogs.ML_image_processing.training_params_panel import install_guided_panel
    install_guided_panel(self)   # self = TrainingTab, at the end of __init__

Model-specific controls (backbone, loss function, LoRA, blob filter, YOLO
weights) intentionally stay Advanced-only to keep the guided path simple.
"""

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QLabel, QLineEdit,
    QScrollArea, QSizePolicy, QSpinBox, QToolButton, QVBoxLayout, QWidget,
    QGridLayout, QTabWidget, QHBoxLayout,
)


# ============================================================================
# ===                       PLAIN-LANGUAGE BLURBS                          ===
# ============================================================================
SECTION_BLURBS = {
    "basics": (
        "Name this training run and decide how long it trains. An epoch is one "
        "full pass through your training images; batch size is how many images "
        "are processed at once (larger is faster but uses more GPU memory — "
        "halve it if you run out)."
    ),
    "optimization": (
        "Training uses the AdamW optimizer — the modern standard for "
        "fine-tuning — automatically. The learning rate sets how big each "
        "adjustment to the model's weights is: too high and training goes "
        "haywire, too low and it crawls; 0.0001 is a safe starting point for "
        "fine-tuning. Weight decay gently nudges weights toward zero to "
        "discourage memorizing the training images."
    ),
    "schedule": (
        "The scheduler automatically shrinks the learning rate when validation "
        "loss stops improving — almost always worth keeping on. Early stopping "
        "quits training entirely when validation stops improving, saving time "
        "and preventing overfitting. Keep the scheduler's patience below the "
        "early-stopping patience, or training stops before the rate ever drops."
    ),
    "outputs": (
        "Validation overlays are images with the model's predicted mask drawn "
        "on top, so you can see how training is going. Checkpoints are model "
        "snapshots; only the best-scoring ones are kept and the rest are "
        "deleted to save disk."
    ),
}

FIELD_HINTS = {
    "lineEdit_siteName": "A label for this run, so you can find its checkpoints later.",
    "spinBox_epochs": "Passes through the full training set. More isn't always better.",
    "spinBox_batchSize": "Images processed at once. Halve it if you run out of GPU memory.",
    "lineEdit_learningRates": "Comma-separated values train one run per rate.",
    "doubleSpinBox_weightDecay": "Mild regularization. 0.01 is a common default.",
    "checkBox_lrScheduler": "Lowers the rate when validation loss plateaus.",
    "doubleSpinBox_lrSchedulerFactor": "How much the rate shrinks (0.50 = cut in half).",
    "spinBox_lrSchedulerPatience": "Stalled epochs to wait before shrinking the rate.",
    "lineEdit_lrSchedulerMinLR": "The floor — the scheduler never goes below this.",
    "checkBox_earlyStopping": "Stop when validation stops improving.",
    "spinBox_patience": "Stalled epochs to wait before stopping.",
    "comboBox_validationOverlayMode": "When overlay images are written.",
    "spinBox_validationOverlayInterval": "Only used in \"Every N epochs\" mode.",
    "spinBox_validationOverlaySamples": "How many example images per overlay pass.",
    "spinBox_maxBestCheckpoints": "Best snapshots to keep; older/worse are deleted.",
}


# ============================================================================
# ===                        WIDGET MIRRORING                              ===
# ============================================================================
class _Mirror(QtCore.QObject):
    """
    Creates a clone of an Advanced widget and keeps the pair in sync.
    The Advanced widget remains authoritative: guided edits write through
    immediately; Advanced edits (or config loads) refresh the guided clone.
    """

    changed = QtCore.pyqtSignal()

    def __init__(self, adv, parent=None):
        super().__init__(parent)
        self.adv = adv
        self._guard = False
        self.clone = self._make_clone(adv)
        self._connect(adv, self._on_adv_changed)
        self._connect(self.clone, self._on_clone_changed)

    # -- construction --------------------------------------------------------
    def _make_clone(self, adv):
        if isinstance(adv, QDoubleSpinBox):
            w = QDoubleSpinBox()
            w.setDecimals(adv.decimals())
            w.setRange(adv.minimum(), adv.maximum())
            w.setSingleStep(adv.singleStep())
            w.setValue(adv.value())
        elif isinstance(adv, QSpinBox):
            w = QSpinBox()
            w.setRange(adv.minimum(), adv.maximum())
            w.setValue(adv.value())
        elif isinstance(adv, QComboBox):
            w = QComboBox()
            for i in range(adv.count()):
                w.addItem(adv.itemText(i))
            w.setCurrentIndex(adv.currentIndex())
        elif isinstance(adv, QCheckBox):
            w = QCheckBox()
            w.setChecked(adv.isChecked())
        elif isinstance(adv, QLineEdit):
            w = QLineEdit()
            w.setText(adv.text())
        else:
            raise TypeError(f"Unsupported widget type: {type(adv)}")
        w.setToolTip(adv.toolTip())
        return w

    # -- signal plumbing -----------------------------------------------------
    @staticmethod
    def _connect(w, slot):
        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.valueChanged.connect(slot)
        elif isinstance(w, QComboBox):
            w.currentIndexChanged.connect(slot)
        elif isinstance(w, QCheckBox):
            w.toggled.connect(slot)
        elif isinstance(w, QLineEdit):
            w.textChanged.connect(slot)

    @staticmethod
    def _get(w):
        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
            return w.value()
        if isinstance(w, QComboBox):
            return w.currentIndex()
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QLineEdit):
            return w.text()

    @staticmethod
    def _set(w, v):
        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.setValue(v)
        elif isinstance(w, QComboBox):
            w.setCurrentIndex(v)
        elif isinstance(w, QCheckBox):
            w.setChecked(v)
        elif isinstance(w, QLineEdit):
            w.setText(v)

    def _copy(self, src, dst):
        if self._guard:
            return
        self._guard = True
        try:
            dst.blockSignals(True)
            self._set(dst, self._get(src))
        finally:
            dst.blockSignals(False)
            self._guard = False
        self.changed.emit()

    def _on_adv_changed(self, *_):
        self._copy(self.adv, self.clone)

    def _on_clone_changed(self, *_):
        self._copy(self.clone, self.adv)


# ============================================================================
# ===                            ACCORDION                                 ===
# ============================================================================
class AccordionSection(QWidget):
    def __init__(self, title: str, content: QWidget, parent=None):
        super().__init__(parent)
        self.toggle = QToolButton(text=title, checkable=True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.RightArrow)
        self.toggle.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; text-align: left; padding: 4px; }"
        )
        self.toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.summary = QLabel("")
        self.summary.setStyleSheet("color: gray; font-size: 8pt;")
        self.summary.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 6, 0)
        header.addWidget(self.toggle, 1)
        header.addWidget(self.summary, 0)

        self.content = content
        self.content.setVisible(False)

        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        flay = QVBoxLayout(frame)
        flay.setContentsMargins(0, 0, 0, 0)
        flay.setSpacing(0)
        head_w = QWidget()
        head_w.setLayout(header)
        flay.addWidget(head_w)
        flay.addWidget(self.content)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 2)
        lay.addWidget(frame)

        self.toggle.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool):
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)

    def set_summary(self, text: str):
        self.summary.setText(text)


class Accordion(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._sections = []
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(4)
        self._lay.addStretch()

    def add_section(self, title: str, content: QWidget) -> AccordionSection:
        s = AccordionSection(title, content)
        s.toggle.toggled.connect(lambda c, s=s: c and self._collapse_others(s))
        self._sections.append(s)
        self._lay.insertWidget(self._lay.count() - 1, s)
        return s

    def _collapse_others(self, keep: AccordionSection):
        for s in self._sections:
            if s is not keep:
                s.toggle.setChecked(False)


# ============================================================================
# ===                       GUIDED PANEL                                   ===
# ============================================================================
class GuidedTrainingParamsPanel(QWidget):
    """Accordion of grouped, annotated mirrors of the Advanced widgets."""

    def __init__(self, tab, parent=None):
        super().__init__(parent)
        self.tab = tab
        self._mirrors = {}

        self.accordion = Accordion()

        self.sec_basics = self.accordion.add_section(
            "1.  Basics",
            self._build_section("basics", [
                ("Model name", "lineEdit_siteName"),
                ("Epochs", "spinBox_epochs"),
                ("Batch size", "spinBox_batchSize"),
            ], extra_top=self._gpu_status_row()))

        self.sec_opt = self.accordion.add_section(
            "2.  Optimization",
            self._build_section("optimization", [
                ("Learning rate(s)", "lineEdit_learningRates"),
                ("Weight decay", "doubleSpinBox_weightDecay"),
            ]))

        self.sec_sched = self.accordion.add_section(
            "3.  LR schedule && early stopping",
            self._build_section("schedule", [
                ("LR scheduler", "checkBox_lrScheduler"),
                ("Reduction factor", "doubleSpinBox_lrSchedulerFactor"),
                ("Scheduler patience", "spinBox_lrSchedulerPatience"),
                ("Minimum LR", "lineEdit_lrSchedulerMinLR"),
                ("Early stopping", "checkBox_earlyStopping"),
                ("Stopping patience", "spinBox_patience"),
            ]))

        self.sec_out = self.accordion.add_section(
            "4.  Outputs && checkpoints",
            self._build_section("outputs", [
                ("Validation overlays", "comboBox_validationOverlayMode"),
                ("Overlay interval", "spinBox_validationOverlayInterval"),
                ("Overlay samples", "spinBox_validationOverlaySamples"),
                ("Checkpoints to keep", "spinBox_maxBestCheckpoints"),
            ]))

        note = QLabel("Changes apply immediately and appear in the Advanced tab.")
        note.setStyleSheet("color: gray; font-size: 8pt;")

        scroll_body = QWidget()
        body_lay = QVBoxLayout(scroll_body)
        body_lay.setContentsMargins(6, 6, 6, 6)
        body_lay.addWidget(self.accordion)
        body_lay.addWidget(note)
        body_lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(scroll_body)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(scroll)

        # Conditional enables + summaries
        self._wire_conditionals()
        self._refresh_summaries()

        # Open the first section by default
        self.sec_basics.toggle.setChecked(True)

    # ------------------------------------------------------------------------
    def _gpu_status_row(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 4)
        try:
            import torch
            ok = torch.cuda.is_available()
            name = torch.cuda.get_device_name(0) if ok else None
        except Exception:
            ok, name = False, None
        if ok:
            lbl = QLabel(f"GPU: {name}  \u2713")
            lbl.setStyleSheet("color: green; font-size: 8pt;")
        else:
            lbl = QLabel("GPU: none detected  \u2717  (training requires a GPU)")
            lbl.setStyleSheet("color: darkred; font-size: 8pt;")
        lay.addWidget(lbl)
        lay.addStretch()
        return w

    # ------------------------------------------------------------------------
    def _build_section(self, blurb_key: str, rows, extra_top: QWidget = None) -> QWidget:
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(24, 4, 10, 10)
        lay.setSpacing(6)

        if extra_top is not None:
            lay.addWidget(extra_top)

        blurb = QLabel(SECTION_BLURBS[blurb_key])
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: #444; font-size: 8pt;")
        lay.addWidget(blurb)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        for r, (label_text, adv_name) in enumerate(rows):
            adv = getattr(self.tab, adv_name, None)
            if adv is None:
                continue
            mirror = _Mirror(adv, parent=self)
            mirror.changed.connect(self._refresh_summaries)
            self._mirrors[adv_name] = mirror

            lbl = QLabel(label_text + ":")
            grid.addWidget(lbl, r, 0)
            grid.addWidget(mirror.clone, r, 1)

            hint = FIELD_HINTS.get(adv_name, "")
            if hint:
                h = QLabel(hint)
                h.setStyleSheet("color: gray; font-size: 8pt;")
                h.setWordWrap(True)
                grid.addWidget(h, r, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 2)
        lay.addLayout(grid)
        return content

    # ------------------------------------------------------------------------
    def _wire_conditionals(self):
        m = self._mirrors

        def sync_enables(*_):
            # Overlay interval only in "Every N epochs" mode
            if "comboBox_validationOverlayMode" in m and "spinBox_validationOverlayInterval" in m:
                every_n = "every n" in m["comboBox_validationOverlayMode"].clone.currentText().lower()
                m["spinBox_validationOverlayInterval"].clone.setEnabled(every_n)
            # Scheduler sub-fields track the scheduler checkbox
            if "checkBox_lrScheduler" in m:
                on = m["checkBox_lrScheduler"].clone.isChecked()
                for name in ("doubleSpinBox_lrSchedulerFactor",
                             "spinBox_lrSchedulerPatience",
                             "lineEdit_lrSchedulerMinLR"):
                    if name in m:
                        m[name].clone.setEnabled(on)
            # Stopping patience tracks early stopping
            if "checkBox_earlyStopping" in m and "spinBox_patience" in m:
                m["spinBox_patience"].clone.setEnabled(
                    m["checkBox_earlyStopping"].clone.isChecked())

        for mirror in m.values():
            mirror.changed.connect(sync_enables)
        sync_enables()

    # ------------------------------------------------------------------------
    def _refresh_summaries(self):
        t = self.tab
        try:
            self.sec_basics.set_summary(
                f"{t.lineEdit_siteName.text() or '(unnamed)'} \u00b7 "
                f"{t.spinBox_epochs.value()} epochs \u00b7 batch {t.spinBox_batchSize.value()}")
            self.sec_opt.set_summary(
                f"AdamW \u00b7 lr {t.lineEdit_learningRates.text()} \u00b7 "
                f"wd {t.doubleSpinBox_weightDecay.value():g}")
            sched = (f"on \u00d7{t.doubleSpinBox_lrSchedulerFactor.value():g}"
                     if t.checkBox_lrScheduler.isChecked() else "off")
            stop = (f"stop @{t.spinBox_patience.value()}"
                    if t.checkBox_earlyStopping.isChecked() else "no early stop")
            self.sec_sched.set_summary(f"scheduler {sched} \u00b7 {stop}")
            self.sec_out.set_summary(
                f"{t.comboBox_validationOverlayMode.currentText()} \u00b7 "
                f"keep {t.spinBox_maxBestCheckpoints.value()}")
        except Exception:
            pass


# ============================================================================
# ===                          INSTALLER                                   ===
# ============================================================================
def _reflow_two_columns(gb: QGroupBox) -> None:
    """
    Reflow the parameters QFormLayout into a two-column grid
    (label | field | label | field), halving the form's height so the
    widgets below it fit on screen. The existing widgets are reused —
    object names, signals, tooltips, and enabled-state wiring all survive.
    """
    old = gb.layout()
    if not isinstance(old, QtWidgets.QFormLayout):
        return

    pairs = []
    for r in range(old.rowCount()):
        li = old.itemAt(r, QtWidgets.QFormLayout.LabelRole)
        fi = old.itemAt(r, QtWidgets.QFormLayout.FieldRole)
        lw = li.widget() if li else None
        fw = fi.widget() if fi else None
        if lw is not None or fw is not None:
            pairs.append((lw, fw))

    while old.count():
        item = old.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
    QWidget().setLayout(old)  # detach and discard the old layout

    grid = QGridLayout(gb)
    grid.setContentsMargins(8, 8, 8, 8)
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(5)
    for i, (lw, fw) in enumerate(pairs):
        row, col = divmod(i, 2)
        if lw is not None:
            grid.addWidget(lw, row, col * 3, Qt.AlignRight | Qt.AlignVCenter)
        if fw is not None:
            grid.addWidget(fw, row, col * 3 + 1, Qt.AlignLeft | Qt.AlignVCenter)
    # spacer column between the two halves, stretch on the far right
    grid.setColumnMinimumWidth(2, 18)
    grid.setColumnStretch(6, 1)


def install_guided_panel(tab) -> QTabWidget:
    """
    Replace tab.groupBox_trainingParameters with a group box containing an
    Advanced / Guided QTabWidget. The original group box (and every widget in
    it) is preserved inside the Advanced tab, so all existing attribute
    references, signals, and _collect_ui_values() keep working.
    """
    gb = tab.groupBox_trainingParameters
    lay, idx = tab._find_container_layout_and_index(gb)
    if lay is None:
        raise RuntimeError("Could not locate layout slot for groupBox_trainingParameters")

    # Pull the original group box out of its slot
    lay.takeAt(idx)
    gb.setParent(None)
    gb.setTitle("")  # title moves to the wrapper group box

    # Two-column reflow (mockup layout): halves the form height so the
    # Blob Filter below fits without scrolling.
    _reflow_two_columns(gb)

    tabs = QTabWidget()
    tabs.setObjectName("tabWidget_trainingParameters")

    # Advanced page scrolls vertically so a tall parameter form can compress
    # instead of pushing content off-screen. Blob Filter is moved INSIDE the
    # scroll page, directly under the parameters form — clipping is then
    # structurally impossible: worst case it scrolls, it can never be crushed.
    adv_body = QWidget()
    adv_lay = QVBoxLayout(adv_body)
    adv_lay.setContentsMargins(0, 0, 0, 0)
    adv_lay.setSpacing(4)
    adv_lay.addWidget(gb)

    blob = getattr(tab, "groupBox_blobFilter", None)
    if blob is not None:
        _blay, _bidx = tab._find_container_layout_and_index(blob)
        if _blay is not None:
            _blay.takeAt(_bidx)
            blob.setParent(None)
            adv_lay.addWidget(blob)
    adv_lay.addStretch()

    adv_scroll = QScrollArea()
    adv_scroll.setWidgetResizable(True)
    adv_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    adv_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    adv_scroll.setWidget(adv_body)
    # Request enough height for the full content (form + blob filter) so the
    # dialog's size hint accounts for it and nothing is clipped. Bounded at
    # 430px so small screens can still open the dialog; beyond that the
    # scrollbar takes over.
    adv_scroll.setMinimumHeight(min(adv_body.sizeHint().height() + 4, 430))
    tabs.addTab(adv_scroll, "Advanced")

    guided = GuidedTrainingParamsPanel(tab)
    tabs.addTab(guided, "Guided")
    tab.guided_params_panel = guided

    wrapper = QGroupBox("Training Parameters")
    wrapper.setObjectName("groupBox_trainingParametersWrapper")
    wlay = QVBoxLayout(wrapper)
    wlay.setContentsMargins(2, 2, 2, 2)
    wlay.setSpacing(0)
    wlay.addWidget(tabs)

    lay.insertWidget(idx, wrapper)

    # First-run nicety: open Guided by default when the model name is still
    # the config template / empty; returning users land on Advanced.
    if not tab.lineEdit_siteName.text().strip():
        tabs.setCurrentIndex(1)

    return tabs
