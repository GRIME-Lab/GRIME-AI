"""
APIKeyDialog.py
---------------
PyQt5 dialog for managing NEON and USGS API keys in GRIME AI.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QTextEdit, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QTextCursor
from PyQt5.QtSvg import QSvgRenderer

from .api_key_manager  import APIKeyManager
from .api_key_verifier import verify_neon_token, verify_usgs_key


BUTTON_CSS_STEEL_BLUE = """
QPushButton {
  background-color: steelblue;
  color: white;
  border: 1px solid #3b6a93;
  padding: 6px 14px;
  border-radius: 6px;
}
QPushButton:hover { background-color: #5a93c2; }
QPushButton:disabled {
  background-color: gray;
  color: black;
}"""

# Same style but with minimal horizontal padding for fixed-width narrow buttons
# (Browse, Verify, Reset, Clear) where 14px side padding clips the label.
BUTTON_CSS_STEEL_BLUE_COMPACT = """
QPushButton {
  background-color: steelblue;
  color: white;
  border: 1px solid #3b6a93;
  padding: 6px 4px;
  border-radius: 6px;
}
QPushButton:hover { background-color: #5a93c2; }
QPushButton:disabled {
  background-color: gray;
  color: black;
}"""

BUTTON_CSS_RED_COMPACT = """
QPushButton {
  background-color: transparent;
  color: #c0392b;
  border: 1px solid #c0392b;
  padding: 6px 4px;
  border-radius: 6px;
}
QPushButton:hover {
  background-color: #fdecea;
  border-color: #922b21;
  color: #922b21;
}
QPushButton:disabled {
  background-color: transparent;
  color: #aaaaaa;
  border-color: #cccccc;
}"""

BUTTON_CSS_GHOST_COMPACT = """
QPushButton {
  background-color: transparent;
  color: #444444;
  border: 1px solid #aaaaaa;
  padding: 6px 4px;
  border-radius: 6px;
}
QPushButton:hover {
  background-color: #f0f0f0;
  border-color: #888888;
}
QPushButton:checked {
  background-color: #e8e8e8;
  border-color: #666666;
}
QPushButton:disabled {
  background-color: transparent;
  color: #aaaaaa;
  border-color: #cccccc;
}"""

BUTTON_CSS_STEEL_BLUE_BOLD_COMPACT = """
QPushButton {
  background-color: steelblue;
  color: white;
  font-weight: bold;
  border: 1px solid #3b6a93;
  padding: 6px 4px;
  border-radius: 6px;
}
QPushButton:hover { background-color: #5a93c2; }
QPushButton:disabled {
  background-color: gray;
  color: black;
}"""


# ---------------------------------------------------------------------------
# Background verification workers
# ---------------------------------------------------------------------------

class _NEONVerifyWorker(QThread):
    done = pyqtSignal(bool, list)   # ok, lines

    def __init__(self, token: str, endpoint: str, parent=None):
        super().__init__(parent)
        self._token    = token
        self._endpoint = endpoint

    def run(self):
        ok, lines = verify_neon_token(self._token, self._endpoint)
        self.done.emit(ok, lines)


class _USGSVerifyWorker(QThread):
    done = pyqtSignal(bool, list)   # ok, lines

    def __init__(self, key: str, endpoint: str, parent=None):
        super().__init__(parent)
        self._key      = key
        self._endpoint = endpoint

    def run(self):
        ok, lines = verify_usgs_key(self._key, self._endpoint)
        self.done.emit(ok, lines)


# ---------------------------------------------------------------------------
# Helper — inline hyperlink label
# ---------------------------------------------------------------------------

def _link_label(text: str, url: str) -> QLabel:
    lbl = QLabel(f'<a href="{url}">{text}</a>')
    lbl.setOpenExternalLinks(True)
    lbl.setTextFormat(Qt.RichText)
    return lbl


# ---------------------------------------------------------------------------
# Eye icon helper — SVG rendered to QIcon, no external files needed
# ---------------------------------------------------------------------------

def _eye_icon(visible: bool) -> "QIcon":
    """Return a QIcon of an open eye (hidden state) or slashed eye (visible state).

    SVG is rendered to a QPixmap at 36x36 so it looks sharp on HiDPI displays.
    No external files or emoji fonts required — works on Windows, Linux, macOS.
    """
    from PyQt5.QtSvg import QSvgRenderer
    from PyQt5.QtGui import QPixmap, QPainter, QIcon
    from PyQt5.QtCore import QByteArray, QSize

    if not visible:
        # Outline eye — key is currently hidden
        svg = b"""<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24'
                  viewBox='0 0 24 24' fill='none' stroke='#444444'
                  stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
          <path d='M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z'/>
          <circle cx='12' cy='12' r='3'/>
        </svg>"""
    else:
        # Slashed eye — key is currently visible
        svg = b"""<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24'
                  viewBox='0 0 24 24' fill='none' stroke='#444444'
                  stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
          <path d='M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8
                   a18.45 18.45 0 0 1 5.06-5.94'/>
          <path d='M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8
                   a18.5 18.5 0 0 1-2.16 3.19'/>
          <line x1='1' y1='1' x2='23' y2='23'/>
        </svg>"""

    renderer = QSvgRenderer(QByteArray(svg))
    pixmap   = QPixmap(QSize(36, 36))
    pixmap.fill(QtCore.Qt.transparent)
    painter  = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


# ---------------------------------------------------------------------------
# Per-service widget block
# ---------------------------------------------------------------------------

class _APIKeyBlock(QWidget):
    def __init__(self, label, signup_url, signup_text, required,
                 echo_mode=QLineEdit.Password, parent=None):
        super().__init__(parent)
        self._required = required

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Title row
        title_row = QHBoxLayout()
        title_lbl = QLabel(f"<b>{label}</b>")
        badge_html = (
            '<span style="color:#c0392b; font-size:11px;"> (required)</span>' if required
            else '<span style="color:#7f8c8d; font-size:11px;"> (optional)</span>'
        )
        required_badge = QLabel(badge_html)
        required_badge.setTextFormat(Qt.RichText)
        title_row.addWidget(title_lbl)
        title_row.addWidget(required_badge)
        title_row.addStretch()
        title_row.addWidget(_link_label(signup_text, signup_url))
        layout.addLayout(title_row)

        # Entry row
        entry_row = QHBoxLayout()
        self.edit = QLineEdit()
        self.edit.setEchoMode(echo_mode)
        self.edit.setPlaceholderText("Paste API key/token here, or click Browse…")
        self.edit.setMinimumWidth(360)

        self.btn_show   = QPushButton()
        self.btn_show.setFixedWidth(34)
        self.btn_show.setFixedHeight(34)
        self.btn_show.setCheckable(True)
        self.btn_show.setToolTip("Show / hide key")
        self.btn_show.setStyleSheet(BUTTON_CSS_GHOST_COMPACT)
        self.btn_show.setIcon(_eye_icon(visible=False))
        self.btn_show.setIconSize(QtCore.QSize(18, 18))

        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.setFixedWidth(80)
        self.btn_browse.setToolTip("Load API key from a .txt file")
        self.btn_browse.setStyleSheet(BUTTON_CSS_STEEL_BLUE_BOLD_COMPACT)

        self.btn_verify = QPushButton("Verify")
        self.btn_verify.setFixedWidth(65)
        self.btn_verify.setToolTip("Send a test request to confirm the key is valid")
        self.btn_verify.setStyleSheet(BUTTON_CSS_STEEL_BLUE_COMPACT)

        entry_row.addWidget(self.edit)
        entry_row.addWidget(self.btn_show)
        entry_row.addWidget(self.btn_browse)
        entry_row.addWidget(self.btn_verify)
        layout.addLayout(entry_row)

        self.btn_browse.clicked.connect(self._browse)
        self.btn_show.toggled.connect(self._toggle_visibility)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select API key file", "",
            "Text files (*.txt);;All files (*)",
        )
        if path:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self.edit.setText(fh.read().strip())
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "File Error", str(exc))

    def _toggle_visibility(self, checked: bool):
        self.edit.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
        self.btn_show.setIcon(_eye_icon(visible=checked))

    def value(self) -> str:
        return self.edit.text().strip()

    def set_value(self, v: str):
        self.edit.setText(v or "")

    def set_verifying(self):
        self.btn_verify.setEnabled(False)

    def set_verify_done(self):
        self.btn_verify.setEnabled(True)


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class APIKeyDialog(QDialog):
    """
    Modal dialog for entering and saving NEON and USGS API keys.
    Call  APIKeyDialog(parent=self).exec_()  from a menu action.
    """

    # Colours for listbox entries
    _COL_HEADER  = QColor("#1a73e8")   # blue  — timestamp / section header
    _COL_OK      = QColor("#27ae60")   # green — success result line
    _COL_FAIL    = QColor("#c0392b")   # red   — failure result line
    _COL_DETAIL  = QColor("#555555")   # grey  — detail lines

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API Key Manager")
        self.setMinimumWidth(680)
        self.setMinimumHeight(620)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._mgr         = APIKeyManager()
        self._neon_worker = None
        self._usgs_worker = None

        self._build_ui()
        self._load_saved_keys()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(18, 18, 18, 14)

        intro = QLabel(
            "GRIME AI uses the NEON and USGS Water Data APIs to retrieve "
            "imagery and site metadata. Register for free accounts at each "
            "service to obtain your keys."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        # ── NEON ──────────────────────────────────────────────────────
        neon_box = QGroupBox("NSF NEON  (National Ecological Observatory Network)")
        neon_layout = QVBoxLayout(neon_box)
        neon_layout.setSpacing(4)

        self._neon_block = _APIKeyBlock(
            label="API Token", signup_url="https://data.neonscience.org/myaccount",
            signup_text="Get a token →", required=True,
        )
        self._neon_block.btn_verify.clicked.connect(self._verify_neon)
        neon_layout.addWidget(self._neon_block)

        neon_ep_row = QHBoxLayout()
        neon_ep_lbl = QLabel("API Endpoint:")
        neon_ep_lbl.setFixedWidth(110)
        self._neon_endpoint_edit = QLineEdit()
        self._neon_endpoint_edit.setPlaceholderText("https://data.neonscience.org/api/v0/  (default)")
        self._neon_endpoint_edit.setToolTip("NEON API base URL. Change only if directed by NEON.")
        neon_ep_reset = QPushButton("Reset")
        neon_ep_reset.setFixedWidth(55)
        neon_ep_reset.setStyleSheet(BUTTON_CSS_RED_COMPACT)
        neon_ep_reset.clicked.connect(
            lambda: self._neon_endpoint_edit.setText("https://data.neonscience.org/api/v0/")
        )
        neon_ep_row.addWidget(neon_ep_lbl)
        neon_ep_row.addWidget(self._neon_endpoint_edit)
        neon_ep_row.addWidget(neon_ep_reset)
        neon_layout.addLayout(neon_ep_row)

        neon_note = QLabel(
            '<i>Required as of June 30 2026. Token passed as <code>X-API-Token</code> header.</i>'
        )
        neon_note.setTextFormat(Qt.RichText)
        neon_layout.addWidget(neon_note)
        root.addWidget(neon_box)

        # ── USGS ──────────────────────────────────────────────────────
        usgs_box = QGroupBox("USGS Water Data APIs  (api.waterdata.usgs.gov)")
        usgs_layout = QVBoxLayout(usgs_box)
        usgs_layout.setSpacing(4)

        self._usgs_block = _APIKeyBlock(
            label="API Key", signup_url="https://api.waterdata.usgs.gov/signup/",
            signup_text="Get a key →", required=False,
        )
        self._usgs_block.btn_verify.clicked.connect(self._verify_usgs)
        usgs_layout.addWidget(self._usgs_block)

        ep_row = QHBoxLayout()
        ep_lbl = QLabel("NIMS Endpoint:")
        ep_lbl.setFixedWidth(110)
        self._usgs_endpoint_edit = QLineEdit()
        self._usgs_endpoint_edit.setPlaceholderText("https://api.waterdata.usgs.gov/nims/v0  (default)")
        self._usgs_endpoint_edit.setToolTip("NIMS API base URL. Change only if directed by USGS.")
        ep_reset = QPushButton("Reset")
        ep_reset.setFixedWidth(55)
        ep_reset.setStyleSheet(BUTTON_CSS_RED_COMPACT)
        ep_reset.clicked.connect(
            lambda: self._usgs_endpoint_edit.setText("https://api.waterdata.usgs.gov/nims/v0")
        )
        ep_row.addWidget(ep_lbl)
        ep_row.addWidget(self._usgs_endpoint_edit)
        ep_row.addWidget(ep_reset)
        usgs_layout.addLayout(ep_row)

        usgs_note = QLabel(
            '<i>Optional — raises rate limits. Key passed as <code>X-Api-Key</code> header.</i>'
        )
        usgs_note.setTextFormat(Qt.RichText)
        usgs_layout.addWidget(usgs_note)
        root.addWidget(usgs_box)

        # ── Verification results listbox ───────────────────────────────
        results_box = QGroupBox("Verification Results")
        results_layout = QVBoxLayout(results_box)
        results_layout.setSpacing(4)

        self._results_text = QTextEdit()
        self._results_text.setFont(QFont("Courier New", 9))
        self._results_text.setMinimumHeight(130)
        self._results_text.setReadOnly(True)
        self._results_text.setToolTip("Select text and copy with Ctrl+C")

        btn_copy = QPushButton("Copy")
        btn_copy.setFixedWidth(55)
        btn_copy.setToolTip("Copy all results text to clipboard")
        btn_copy.setStyleSheet(BUTTON_CSS_GHOST_COMPACT)
        btn_copy.clicked.connect(lambda: (
            self._results_text.selectAll(),
            self._results_text.copy(),
            self._results_text.textCursor().clearSelection()
        ))

        btn_clear = QPushButton("Clear")
        btn_clear.setFixedWidth(55)
        btn_clear.setToolTip("Clear the results log")
        btn_clear.setStyleSheet(BUTTON_CSS_RED_COMPACT)
        btn_clear.clicked.connect(self._results_text.clear)

        results_header = QHBoxLayout()
        results_header.addWidget(QLabel("<i>HTTP responses from Verify — select text to copy.</i>"))
        results_header.addStretch()
        results_header.addWidget(btn_copy)
        results_header.addWidget(btn_clear)

        results_layout.addLayout(results_header)
        results_layout.addWidget(self._results_text)
        root.addWidget(results_box)

        # ── Button bar ────────────────────────────────────────────────
        btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._save_and_close)
        btn_box.rejected.connect(self.reject)
        for _btn in btn_box.buttons():
            _btn.setStyleSheet(BUTTON_CSS_STEEL_BLUE)
        root.addWidget(btn_box)

    # ------------------------------------------------------------------
    # Results listbox helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Results helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _esc(text: str) -> str:
        """HTML-escape a plain text string."""
        return (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))

    def _append_results(self, lines: list[str], ok: bool):
        """Append colour-coded HTML lines to the results QTextEdit.

        Uses insertHtml for reliable bold/colour on all platforms.
        Layout:
          - HR separator between entries
          - Line 0  : timestamp header, blue bold
          - Result: : label black bold, value green (ok) or red (fail)
          - others  : label black bold, value default grey
          - blank   : spacer
        """
        result_colour = "#27ae60" if ok else "#c0392b"

        def _row(label: str, value: str, label_colour: str = "#000000",
                 value_colour: str = "#555555") -> str:
            return (
                f'<span style="font-family:Courier New; font-size:9pt; color:{label_colour};">' 
                f'<b>{self._esc(label)}</b></span>'
                f'<span style="font-family:Courier New; font-size:9pt; color:{value_colour};">' 
                f'{self._esc(value)}</span><br>'
            )

        def _plain(text: str, colour: str = "#555555", bold: bool = False) -> str:
            weight = "bold" if bold else "normal"
            return (
                f'<span style="font-family:Courier New; font-size:9pt; ' 
                f'color:{colour}; font-weight:{weight};">' 
                f'{self._esc(text)}</span><br>'
            )

        html = (
            '<span style="font-family:Courier New; font-size:9pt; color:#aaaaaa;">' +
            self._esc("─" * 72) +
            '</span><br>'
        )

        for i, line in enumerate(lines):
            stripped = line.strip()

            if i == 0:
                # Timestamp header — blue bold
                html += _plain(line, colour="#1a73e8", bold=True)

            elif not stripped:
                html += "<br>"

            elif stripped.startswith("Result:"):
                colon = line.find(":")
                label = line[:colon + 1]
                value = line[colon + 1:]
                # Label black bold, value coloured
                html += _row(label, value,
                             label_colour="#000000",
                             value_colour=result_colour)

            else:
                colon = line.find(":")
                if colon == -1:
                    html += _plain(line)
                else:
                    label = line[:colon + 1]
                    value = line[colon + 1:]
                    html += _row(label, value,
                                 label_colour="#000000",
                                 value_colour="#555555")

        self._results_text.moveCursor(QTextCursor.End)
        self._results_text.insertHtml(html)
        self._results_text.moveCursor(QTextCursor.End)

    # ------------------------------------------------------------------
    # Key loading
    # ------------------------------------------------------------------

    def _load_saved_keys(self):
        neon          = self._mgr.get_neon_token()
        neon_endpoint = self._mgr.get_neon_endpoint()
        usgs          = self._mgr.get_usgs_key()
        usgs_endpoint = self._mgr.get_usgs_endpoint()
        if neon:
            self._neon_block.set_value(neon)
        self._neon_endpoint_edit.setText(neon_endpoint)
        if usgs:
            self._usgs_block.set_value(usgs)
        self._usgs_endpoint_edit.setText(usgs_endpoint)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_neon(self):
        token = self._neon_block.value()
        if not token:
            self._append_results(["[NEON]  No token entered — nothing to verify."], ok=False)
            return
        self._neon_block.set_verifying()
        ep = self._neon_endpoint_edit.text().strip()
        self._neon_worker = _NEONVerifyWorker(token, ep, parent=self)
        self._neon_worker.done.connect(self._on_neon_verify_done)
        self._neon_worker.start()

    def _on_neon_verify_done(self, ok: bool, lines: list):
        self._neon_block.set_verify_done()
        self._append_results(lines, ok)

    def _verify_usgs(self):
        key = self._usgs_block.value()
        self._usgs_block.set_verifying()
        ep = self._usgs_endpoint_edit.text().strip()
        self._usgs_worker = _USGSVerifyWorker(key, ep, parent=self)
        self._usgs_worker.done.connect(self._on_usgs_verify_done)
        self._usgs_worker.start()

    def _on_usgs_verify_done(self, ok: bool, lines: list):
        self._usgs_block.set_verify_done()
        self._append_results(lines, ok)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_and_close(self):
        neon = self._neon_block.value()
        usgs = self._usgs_block.value()

        if not neon:
            reply = QtWidgets.QMessageBox.question(
                self, "NEON Token Missing",
                "The NEON API token is required as of June 30 2026.\n\n"
                "Save anyway without a NEON token?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        neon_ep = self._neon_endpoint_edit.text().strip()
        usgs_ep = self._usgs_endpoint_edit.text().strip()
        self._mgr.save(
            neon_token=neon or "",
            neon_endpoint=neon_ep or "",
            usgs_key=usgs or "",
            usgs_endpoint=usgs_ep or "",
        )
        self.accept()
