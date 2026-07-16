# SAGE/ui/filmstrip.py
"""
Horizontal thumbnail filmstrip beneath the image canvas.

Ported from the GRIME AI Image Navigator pattern: single-row, no wrapping,
batched/non-blocking thumbnail loading. Emits image_clicked(filename) on
selection so MainWindow can build the full path and load it.
"""
import os
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QListView, QFrame


class Filmstrip(QListWidget):
    image_clicked = pyqtSignal(str)   # emits filename (not full path)

    def __init__(self, parent=None, thumb_height=72, batch_size=12, batch_delay=40):
        super().__init__(parent)

        self._folder = ""
        self._names = []
        self._pending = []
        self._batch_size = batch_size
        self._batch_delay = batch_delay
        self._load_token = 0

        icon_w = int(thumb_height * 16 / 9)
        self.setIconSize(QSize(icon_w, thumb_height))
        self.setViewMode(QListView.IconMode)
        self.setFlow(QListView.LeftToRight)
        self.setWrapping(False)
        self.setSpacing(2)
        self.setMovement(QListView.Static)
        self.setResizeMode(QListView.Adjust)
        self.setContentsMargins(0, 0, 0, 0)
        self.setViewportMargins(0, 0, 0, 0)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # One icon row tall (+ scrollbar allowance).
        self.setFixedHeight(thumb_height + 26)

        self.itemClicked.connect(self._on_item_clicked)

    # -------------------------------------------------------------
    def populate(self, folder, names):
        """names: list of filenames (not full paths) in `folder`."""
        self._folder = folder
        self._names = list(names)
        self.clear()

        self._load_token += 1
        token = self._load_token
        self._pending.clear()

        for idx, name in enumerate(names):
            item = QListWidgetItem(QIcon(), "")
            item.setData(Qt.UserRole, name)
            item.setToolTip(name)
            item.setSizeHint(QSize(self.iconSize().width() + 6,
                                   self.iconSize().height() + 6))
            self.addItem(item)
            self._pending.append((item, os.path.join(folder, name), token))

        if self.count():
            self.setCurrentRow(0)
        QTimer.singleShot(self._batch_delay, lambda: self._load_batch(token))

    # -------------------------------------------------------------
    def select_name(self, name):
        """Highlight the item for `name` without emitting a click."""
        for i in range(self.count()):
            if self.item(i).data(Qt.UserRole) == name:
                self.blockSignals(True)
                self.setCurrentRow(i)
                self.blockSignals(False)
                self.scrollToItem(self.item(i))
                return

    # -------------------------------------------------------------
    def _load_batch(self, token):
        if token != self._load_token:
            return
        for _ in range(min(self._batch_size, len(self._pending))):
            item, path, tok = self._pending.pop(0)
            if tok != self._load_token or not os.path.exists(path):
                continue
            pix = QPixmap(path)
            if pix.isNull():
                continue
            item.setIcon(QIcon(pix.scaled(self.iconSize(),
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)))
        if self._pending:
            QTimer.singleShot(self._batch_delay, lambda: self._load_batch(token))

    # -------------------------------------------------------------
    def _on_item_clicked(self, item):
        name = item.data(Qt.UserRole)
        if name:
            self.image_clicked.emit(name)
