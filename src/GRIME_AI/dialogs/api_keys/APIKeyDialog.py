"""
APIKeyDialog.py
---------------
PyQt5 dialog for managing NEON and USGS API keys in GRIME AI.

Layout (per service):
  [Label + "how to get a key" link]
  [QLineEdit (password echo)] [Browse...] [Verify]
  [Status label]

Launching from main.py:
    from GRIME_AI.dialogs.api_keys.APIKeyDialog import APIKeyDialog
    dlg = APIKeyDialog(parent=self)
    dlg.exec_()
    # After the dialog closes, the MainWindow (or any code) can call:
    #   APIKeyManager().get_neon_token()
    #   APIKeyManager().get_usgs_key()
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl

from .api_key_manager  import APIKeyManager
from .api_key_verifier import verify_neon_token, verify_usgs_key


# ---------------------------------------------------------------------------
# Background verification workers
# ---------------------------------------------------------------------------

class _NEONVerifyWorker(QThread):
    done = pyqtSignal(bool, str)

    def __init__(self, token: str, parent=None):
        super().__init__(parent)
        self._token = token

    def run(self):
        ok, msg = verify_neon_token(self._token)
        self.done.emit(ok, msg)


class _USGSVerifyWorker(QThread):
    done = pyqtSignal(bool, str)

    def __init__(self, key: str, parent=None):
        super().__init__(parent)
        self._key = key

    def run(self):
        ok, msg = verify_usgs_key(self._key)
        self.done.emit(ok, msg)


# ---------------------------------------------------------------------------
# Helper — inline hyperlink label
# ---------------------------------------------------------------------------

def _link_label(text: str, url: str) -> QLabel:
    lbl = QLabel(f'<a href="{url}">{text}</a>')
    lbl.setOpenExternalLinks(True)
    lbl.setTextFormat(Qt.RichText)
    return lbl


# ---------------------------------------------------------------------------
# Per-service widget block
# ---------------------------------------------------------------------------

class _APIKeyBlock(QWidget):
    """
    Reusable widget for one API key entry (paste or browse) with verify.
    """

    def __init__(
        self,
        label:        str,
        signup_url:   str,
        signup_text:  str,
        required:     bool,
        echo_mode:    QLineEdit.EchoMode = QLineEdit.Password,
        parent=None,
    ):
        super().__init__(parent)
        self._required = required

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Title row -------------------------------------------------------
        title_row = QHBoxLayout()
        title_lbl = QLabel(f"<b>{label}</b>")
        required_badge = QLabel(
            '<span style="color:#c0392b; font-size:11px;">'
            " (required)</span>" if required else
            '<span style="color:#7f8c8d; font-size:11px;">'
            " (optional)</span>"
        )
        required_badge.setTextFormat(Qt.RichText)
        title_row.addWidget(title_lbl)
        title_row.addWidget(required_badge)
        title_row.addStretch()
        title_row.addWidget(_link_label(signup_text, signup_url))
        layout.addLayout(title_row)

        # Entry row -------------------------------------------------------
        entry_row = QHBoxLayout()
        self.edit = QLineEdit()
        self.edit.setEchoMode(echo_mode)
        self.edit.setPlaceholderText("Paste API key/token here, or click Browse…")
        self.edit.setMinimumWidth(360)

        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.setFixedWidth(80)
        self.btn_browse.setToolTip("Load API key from a .txt file")

        self.btn_verify = QPushButton("Verify")
        self.btn_verify.setFixedWidth(65)
        self.btn_verify.setToolTip("Send a test request to confirm the key is valid")

        self.btn_show = QPushButton("Show")
        self.btn_show.setFixedWidth(50)
        self.btn_show.setCheckable(True)
        self.btn_show.setToolTip("Toggle key visibility")

        entry_row.addWidget(self.edit)
        entry_row.addWidget(self.btn_show)
        entry_row.addWidget(self.btn_browse)
        entry_row.addWidget(self.btn_verify)
        layout.addLayout(entry_row)

        # Status label ----------------------------------------------------
        self.lbl_status = QLabel("")
        self.lbl_status.setMinimumHeight(18)
        layout.addWidget(self.lbl_status)

        # Connections -----------------------------------------------------
        self.btn_browse.clicked.connect(self._browse)
        self.btn_show.toggled.connect(self._toggle_visibility)

    # ------------------------------------------------------------------

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select API key file",
            "",
            "Text files (*.txt);;All files (*)",
        )
        if path:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    value = fh.read().strip()
                self.edit.setText(value)
                self.set_status("", neutral=True)
            except Exception as exc:
                self.set_status(f"Could not read file: {exc}", ok=False)

    def _toggle_visibility(self, checked: bool):
        self.edit.setEchoMode(
            QLineEdit.Normal if checked else QLineEdit.Password
        )
        self.btn_show.setText("Hide" if checked else "Show")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def value(self) -> str:
        return self.edit.text().strip()

    def set_value(self, v: str):
        self.edit.setText(v or "")

    def set_status(self, msg: str, ok: bool = True, neutral: bool = False):
        if neutral or not msg:
            self.lbl_status.setText(msg)
            self.lbl_status.setStyleSheet("")
        elif ok:
            self.lbl_status.setText(f"✓  {msg}")
            self.lbl_status.setStyleSheet("color: #27ae60;")
        else:
            self.lbl_status.setText(f"✗  {msg}")
            self.lbl_status.setStyleSheet("color: #c0392b;")

    def set_verifying(self):
        self.lbl_status.setText("⟳  Verifying…")
        self.lbl_status.setStyleSheet("color: #7f8c8d;")
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
    Keys are persisted to <settings_folder>/api_keys.ini on Save.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API Key Manager")
        self.setMinimumWidth(620)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )

        self._mgr            = APIKeyManager()
        self._neon_worker    = None
        self._usgs_worker    = None

        self._build_ui()
        self._load_saved_keys()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(18, 18, 18, 14)

        # Intro blurb
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
            label="API Token",
            signup_url="https://data.neonscience.org/myaccount",
            signup_text="Get a token →",
            required=True,
        )
        self._neon_block.btn_verify.clicked.connect(self._verify_neon)
        neon_layout.addWidget(self._neon_block)

        # Endpoint row
        neon_ep_row = QHBoxLayout()
        neon_ep_lbl = QLabel("API Endpoint:")
        neon_ep_lbl.setFixedWidth(110)
        self._neon_endpoint_edit = QLineEdit()
        self._neon_endpoint_edit.setPlaceholderText(
            "https://data.neonscience.org/api/v0/  (default)"
        )
        self._neon_endpoint_edit.setToolTip(
            "NEON API base URL. Change only if directed by NEON or using a mirror endpoint."
        )
        neon_ep_reset = QPushButton("Reset")
        neon_ep_reset.setFixedWidth(55)
        neon_ep_reset.setToolTip("Reset to the default NEON API endpoint")
        neon_ep_reset.clicked.connect(
            lambda: self._neon_endpoint_edit.setText(
                "https://data.neonscience.org/api/v0/"
            )
        )
        neon_ep_row.addWidget(neon_ep_lbl)
        neon_ep_row.addWidget(self._neon_endpoint_edit)
        neon_ep_row.addWidget(neon_ep_reset)
        neon_layout.addLayout(neon_ep_row)

        neon_note = QLabel(
            '<i>Required as of June 30 2026.  Token is passed as the '
            '<code>X-API-Token</code> HTTP header.</i>'
        )
        neon_note.setTextFormat(Qt.RichText)
        neon_note.setWordWrap(True)
        neon_layout.addWidget(neon_note)
        root.addWidget(neon_box)

        # ── USGS ──────────────────────────────────────────────────────
        usgs_box = QGroupBox("USGS Water Data APIs  (api.waterdata.usgs.gov)")
        usgs_layout = QVBoxLayout(usgs_box)
        usgs_layout.setSpacing(4)

        self._usgs_block = _APIKeyBlock(
            label="API Key",
            signup_url="https://api.waterdata.usgs.gov/signup/",
            signup_text="Get a key →",
            required=False,
        )
        self._usgs_block.btn_verify.clicked.connect(self._verify_usgs)
        usgs_layout.addWidget(self._usgs_block)

        # Endpoint row
        ep_row = QHBoxLayout()
        ep_lbl = QLabel("NIMS Endpoint:")
        ep_lbl.setFixedWidth(110)
        self._usgs_endpoint_edit = QLineEdit()
        self._usgs_endpoint_edit.setPlaceholderText(
            "https://api.waterdata.usgs.gov/nims/v0  (default)"
        )
        self._usgs_endpoint_edit.setToolTip(
            "NIMS API base URL. Change only if directed by USGS or using a private endpoint."
        )
        ep_reset = QPushButton("Reset")
        ep_reset.setFixedWidth(55)
        ep_reset.setToolTip("Reset to the default NIMS endpoint")
        ep_reset.clicked.connect(
            lambda: self._usgs_endpoint_edit.setText(
                "https://api.waterdata.usgs.gov/nims/v0"
            )
        )
        ep_row.addWidget(ep_lbl)
        ep_row.addWidget(self._usgs_endpoint_edit)
        ep_row.addWidget(ep_reset)
        usgs_layout.addLayout(ep_row)

        usgs_note = QLabel(
            '<i>Optional key — required for more than a few requests/hour.  '
            'Key is passed as the <code>X-Api-Key</code> header.  '
            'Endpoint can be changed if USGS updates the URL or you have a private endpoint.</i>'
        )
        usgs_note.setTextFormat(Qt.RichText)
        usgs_note.setWordWrap(True)
        usgs_layout.addWidget(usgs_note)
        root.addWidget(usgs_box)

        # ── Button bar ────────────────────────────────────────────────
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._save_and_close)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

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
            self._neon_block.set_status("Saved key loaded.", neutral=True)
        self._neon_endpoint_edit.setText(neon_endpoint)
        if usgs:
            self._usgs_block.set_value(usgs)
            self._usgs_block.set_status("Saved key loaded.", neutral=True)
        self._usgs_endpoint_edit.setText(usgs_endpoint)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_neon(self):
        token = self._neon_block.value()
        if not token:
            self._neon_block.set_status("Enter or browse a token first.", ok=False)
            return
        self._neon_block.set_verifying()
        self._neon_worker = _NEONVerifyWorker(token, parent=self)
        self._neon_worker.done.connect(self._on_neon_verify_done)
        self._neon_worker.start()

    def _on_neon_verify_done(self, ok: bool, msg: str):
        self._neon_block.set_verify_done()
        self._neon_block.set_status(msg, ok=ok)

    def _verify_usgs(self):
        key = self._usgs_block.value()
        if not key:
            self._usgs_block.set_status("Enter or browse a key first.", ok=False)
            return
        self._usgs_block.set_verifying()
        self._usgs_worker = _USGSVerifyWorker(key, parent=self)
        self._usgs_worker.done.connect(self._on_usgs_verify_done)
        self._usgs_worker.start()

    def _on_usgs_verify_done(self, ok: bool, msg: str):
        self._usgs_block.set_verify_done()
        self._usgs_block.set_status(msg, ok=ok)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_and_close(self):
        neon = self._neon_block.value()
        usgs = self._usgs_block.value()

        # Warn if NEON token is empty (mandatory)
        if not neon:
            reply = QtWidgets.QMessageBox.question(
                self,
                "NEON Token Missing",
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
