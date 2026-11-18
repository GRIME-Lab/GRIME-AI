# SAGE/ui/filmstrip.py
"""
Horizontal thumbnail filmstrip beneath the image canvas.

Ported from the GRIME AI Image Navigator pattern: single-row, no wrapping,
batched/non-blocking thumbnail loading. Emits image_clicked(filename) on
selection so MainWindow can build the full path and load it.
"""
import os
from PyQt5.QtCore import Qt, QTimer, QSize, QRectF, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QImageReader, QPalette, QPen
from PyQt5.QtWidgets import (
    QListWidget, QListWidgetItem, QListView, QFrame, QStyledItemDelegate,
)

# Per-item data roles beyond Qt.UserRole (which holds the filename).
ROLE_NAME = Qt.UserRole
ROLE_ANNOTATED = Qt.UserRole + 1     # bool: has >=1 saved annotation
ROLE_DIMS = Qt.UserRole + 2          # (w, h) or None until known


# Annotated-image colour used when the theme palette defines no "annotated"
# key. Dark orange: distinct from black text and from the (blue) selection
# highlight, avoids red and green for colour-blind viewers, and dark enough
# to read as small bold text on a light background.
ANNOTATED_ORANGE = "#C86400"


def _annotated_color() -> str:
    """Annotated-image colour from the active palette ("annotated" key),
    falling back to ANNOTATED_ORANGE."""
    from SAGE.ui.theme import colors, active_mode
    return colors(active_mode()).get("annotated", ANNOTATED_ORANGE)


class _IndexDelegate(QStyledItemDelegate):
    """Draws the thumbnail normally, then the image's 1-based index centered
    beneath it. When the image has at least one saved annotation, the number
    is drawn in the annotated colour and the thumbnail gets a border in the
    same colour; otherwise the number uses the normal text colour and there is
    no border. Painting in the delegate keeps both independent of the
    thumbnail, so they survive selection highlight and need no icon
    regeneration when the annotation state flips."""

    LABEL_PAD = 2    # gap above and below the number, px
    BORDER_W = 2     # annotated-thumbnail border width, px

    def __init__(self, color=None, parent=None):
        super().__init__(parent)
        self._color = color if color is not None else QColor(_annotated_color())

    def set_color(self, color):
        self._color = color

    def band_height(self, widget):
        """Height of the strip beneath the thumbnail that holds the number."""
        return widget.fontMetrics().height() + 2 * self.LABEL_PAD

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        r = option.rect
        band_h = option.fontMetrics.height() + 2 * self.LABEL_PAD
        band = QRectF(r.left(), r.bottom() - band_h + 1, r.width(), band_h)
        annotated = bool(index.data(ROLE_ANNOTATED))
        if annotated:
            color = self._color
        else:
            color = option.palette.color(QPalette.Text)
        font = option.font
        font.setBold(True)
        painter.save()
        if annotated:
            # Border around the thumbnail area (cell above the number band).
            half = self.BORDER_W / 2.0
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(QPen(self._color, self.BORDER_W))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(r.left() + half, r.top() + half,
                                    r.width() - self.BORDER_W,
                                    r.height() - band_h - self.BORDER_W))
        painter.setPen(color)
        painter.setFont(font)
        painter.drawText(band, Qt.AlignCenter, str(index.row() + 1))
        painter.restore()


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

        # Index number painted beneath each thumbnail; annotated images get an
        # orange number and an orange border.
        self._index_delegate = _IndexDelegate(parent=self)
        self.setItemDelegate(self._index_delegate)
        self._label_band = self._index_delegate.band_height(self)

        # Tooltips are built per-item on hover (see event()), so mouse tracking
        # must be on even when no button is held.
        self.setMouseTracking(True)

        # One icon row tall (+ number band + scrollbar allowance).
        self.setFixedHeight(thumb_height + 26 + self._label_band)

        self.itemClicked.connect(self._on_item_clicked)

    # -------------------------------------------------------------
    def populate(self, folder, names, annotated=None):
        """names: list of filenames (not full paths) in `folder`.
        annotated: optional set/container of filenames that have saved
        annotations; those get an orange number and border. Absent -> all
        numbers in the normal text colour, no borders (caller can set them later via
        set_annotated / set_annotated_set)."""
        self._folder = folder
        self._names = list(names)
        annotated = annotated or set()
        self.clear()

        self._load_token += 1
        token = self._load_token
        self._pending.clear()

        band = self._label_band
        for idx, name in enumerate(names):
            item = QListWidgetItem(QIcon(), "")
            item.setData(ROLE_NAME, name)
            item.setData(ROLE_ANNOTATED, name in annotated)
            item.setData(ROLE_DIMS, None)
            # Tooltip is composed on hover in event(); nothing static here.
            item.setSizeHint(QSize(self.iconSize().width() + 6,
                                   self.iconSize().height() + 6 + band))
            self.addItem(item)
            self._pending.append((item, os.path.join(folder, name), token))

        if self.count():
            self.setCurrentRow(0)
        QTimer.singleShot(self._batch_delay, lambda: self._load_batch(token))

    # -------------------------------------------------------------
    def select_name(self, name):
        """Highlight the item for `name` without emitting a click."""
        for i in range(self.count()):
            if self.item(i).data(ROLE_NAME) == name:
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
            # The full-res pixmap is already decoded here; record its true
            # dimensions for the hover tooltip before scaling it down, so hover
            # never has to reopen the file.
            item.setData(ROLE_DIMS, (pix.width(), pix.height()))
            item.setIcon(QIcon(pix.scaled(self.iconSize(),
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)))
        if self._pending:
            QTimer.singleShot(self._batch_delay, lambda: self._load_batch(token))

    # -------------------------------------------------------------
    def _on_item_clicked(self, item):
        name = item.data(ROLE_NAME)
        if name:
            self.image_clicked.emit(name)

    # -------------------------------------------------------------
    # Annotation-state API
    # -------------------------------------------------------------
    def set_annotated(self, name, flag=True):
        """Flip the annotated state for a single image and repaint just its cell."""
        for i in range(self.count()):
            it = self.item(i)
            if it.data(ROLE_NAME) == name:
                if bool(it.data(ROLE_ANNOTATED)) != bool(flag):
                    it.setData(ROLE_ANNOTATED, bool(flag))
                    self.update(self.indexFromItem(it))
                return

    def set_annotated_set(self, names):
        """Bulk-set which images are annotated. `names` is any container of
        filenames; everything not in it is cleared."""
        wanted = set(names or ())
        for i in range(self.count()):
            it = self.item(i)
            flag = it.data(ROLE_NAME) in wanted
            if bool(it.data(ROLE_ANNOTATED)) != flag:
                it.setData(ROLE_ANNOTATED, flag)
        self.viewport().update()

    def refresh_theme(self):
        """Re-read the annotated colour after a theme switch."""
        self._index_delegate.set_color(QColor(_annotated_color()))
        self.viewport().update()

    def set_dot_color(self, color):
        """Colour of the number and border for annotated images. Name kept so existing
        callers continue to work."""
        self._index_delegate.set_color(color)
        self.viewport().update()

    # -------------------------------------------------------------
    # Hover tooltip: image name + pixel dimensions, composed on demand.
    # -------------------------------------------------------------
    def _tooltip_for(self, item):
        name = item.data(ROLE_NAME) or ""
        dims = item.data(ROLE_DIMS)
        if dims is None:
            # Thumbnail not decoded yet: read only the header, no pixel decode.
            dims = self._read_dims(os.path.join(self._folder, name))
            if dims is not None:
                item.setData(ROLE_DIMS, dims)
        if dims is None:
            return name
        w, h = dims
        annotated = " • annotated" if item.data(ROLE_ANNOTATED) else ""
        return f"{name}\n{w} × {h} px{annotated}"

    @staticmethod
    def _read_dims(path):
        """Dimensions from the image header without decoding pixels."""
        if not os.path.exists(path):
            return None
        reader = QImageReader(path)
        size = reader.size()
        if size.isValid():
            return (size.width(), size.height())
        return None

    def event(self, e):
        from PyQt5.QtCore import QEvent
        if e.type() == QEvent.ToolTip:
            item = self.itemAt(e.pos())
            if item is not None:
                from PyQt5.QtWidgets import QToolTip
                QToolTip.showText(e.globalPos(), self._tooltip_for(item), self)
            else:
                from PyQt5.QtWidgets import QToolTip
                QToolTip.hideText()
            return True
        return super().event(e)
