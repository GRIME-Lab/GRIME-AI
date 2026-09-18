# ImageOrganizerDlg.py

import traceback

from datetime import datetime
from GRIME_AI.utils.resource_utils import ui_path
from GRIME_AI.utils import theme
from pathlib import Path

from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QTimer, Qt
from PyQt5.QtGui import QTextOption
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox

# ---- core (your current module) ----
from GRIME_AI.image_organizer import (
    organize_images, example_filename,
    iter_images, read_image_metadata,
    move_files_to_subfolder, revert_operation,
    estimate_minute_collision_count,  # <-- warning estimator
)

# Always Steel Blue buttons (including popups)
BUTTON_CSS = """
QPushButton {
  background-color: #4682B4;
  color: white;
  border: 1px solid #3b6a93;
  padding: 6px 14px;
  border-radius: 6px;
}
QPushButton:hover { background-color: #5a93c2; }
"""

class ImageOrganizerDlg(QDialog):
    """
    Single-screen Image Organizer dialog.

    - Counts line: Total File Count / Files w/Timestamp.
    - Options: Move Files Missing Date, Operate Recursively, Include Seconds (+ red warning).
    - Site Information wraps at word boundaries; dialog auto-expands (never shrinks).
    - Run shows completion dialog with 'Revert' button.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            loadUi(ui_path("image_organizer/QDialog_ImageOrganizer.ui"), self)
        except Exception as e:
            print("[ERROR] uic.loadUi failed:", e)
            traceback.print_exc()
            raise

        # Global button style
        self.setStyleSheet(BUTTON_CSS)

        # Keep Browse a normal width (matches UI spacer for site row; tweak if you like)
        self.src_btn.setFixedWidth(100)

        # Site Information: wrap at WORD boundaries using the widget width
        # (UI should also set these, but set here in case)
        self.site_info_edit.setLineWrapMode(self.site_info_edit.WidgetWidth)
        self.site_info_edit.setWordWrapMode(QTextOption.WordWrap)
        self.site_info_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.site_info_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Counts baseline
        self.example_lbl.setText("Example filename: (select a folder)")
        self.total_count_lbl.setText("TOTAL File Count: 0")
        self.ts_count_lbl.setText("Files w/Timestamp: 0")
        self.ts_count_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Seconds warning label (red) – ensure hidden initially
        if hasattr(self, "sec_warn_lbl"):
            theme.bind(self.sec_warn_lbl, "color:#B22222;", "color:#FF6B5E;")  # firebrick / light red
            self.sec_warn_lbl.setVisible(False)

        # Signals
        self.src_btn.clicked.connect(self.pick_src)
        self.cancel_btn.clicked.connect(self.close)
        self.run_btn.clicked.connect(self.run_organizer)

        self.src_edit.textChanged.connect(self.update_example)
        self.site_edit.textChanged.connect(self._on_inputs_changed)
        self.sec_chk.toggled.connect(self._on_inputs_changed)
        self.recurse_chk.toggled.connect(self._on_inputs_changed)

        # Sync line-edit heights & capture initial size
        self._initial_size = None
        QTimer.singleShot(0, self._sync_text_field_heights)
        QTimer.singleShot(0, self._capture_initial_size)

        # Grow dialog when Site Information grows (never shrink)
        self.site_info_edit.textChanged.connect(self._grow_for_site_info)

        # cache of images for the last-picked folder (to avoid re-walking twice)
        self._imgs_cache = []

    # ---------- sizing helpers ----------
    def _capture_initial_size(self):
        self._initial_size = self.size()
        self.setMinimumSize(self._initial_size)

    def _sync_text_field_heights(self):
        h = self.src_btn.sizeHint().height()
        for le in (self.src_edit, self.site_edit, self.copyright_edit):
            le.setMinimumHeight(h)
            le.setMaximumHeight(h)

    def _grow_for_site_info(self):
        """
        Grow the QTextEdit to fit content and enlarge the dialog if needed.
        Never shrink below the current or initial size.
        """
        doc = self.site_info_edit.document()
        doc.adjustSize()

        frame = int(self.site_info_edit.frameWidth() * 2)
        content_h = int(doc.size().height()) + frame + 12
        min_h = max(140, content_h)
        self.site_info_edit.setMinimumHeight(min_h)

        hint = self.sizeHint()
        target_w = max(self.width(), hint.width(),
                       self._initial_size.width() if self._initial_size else 0)
        target_h = max(self.height(), hint.height(),
                       self._initial_size.height() if self._initial_size else 0)
        if target_w > self.width() or target_h > self.height():
            self.resize(target_w, target_h)

    # ---------- misc helpers ----------
    def _steel_blue_msgbox(self, title: str, text: str, icon=QMessageBox.Information):
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(icon)
        box.setText(text)
        box.setStyleSheet(BUTTON_CSS)
        return box

    def _compute_example(self) -> str:
        return example_filename(
            self.site_edit.text().strip(),
            self.sec_chk.isChecked(),
            example_dt=datetime.now(),
            ext=".jpg"
        )

    @pyqtSlot()
    def update_example(self):
        if not self.src_edit.text().strip():
            self.example_lbl.setText("Example filename: (select a folder)")
        else:
            self.example_lbl.setText(f"Example filename: {self._compute_example()}")

    def _on_inputs_changed(self):
        # Update example and seconds warning whenever inputs that affect them change
        self.update_example()
        self._update_seconds_warning()

    # ---------- compute and show red warning ----------
    def _update_seconds_warning(self):
        """Show/hide the red warning next to 'Include Seconds'."""
        try:
            if not hasattr(self, "sec_warn_lbl"):
                return

            # Only warn when seconds are NOT checked
            if self.sec_chk.isChecked():
                self.sec_warn_lbl.setVisible(False)
                self.sec_warn_lbl.setText("")
                return

            folder = self.src_edit.text().strip()
            if not folder:
                self.sec_warn_lbl.setVisible(False)
                self.sec_warn_lbl.setText("")
                return

            # Use cached file list if available; else build it
            imgs = self._imgs_cache or iter_images(Path(folder), self.recurse_chk.isChecked())
            if not imgs:
                self.sec_warn_lbl.setVisible(False)
                self.sec_warn_lbl.setText("")
                return

            # Estimate how many files will need suffixes without seconds
            site = self.site_edit.text().strip()
            risky = estimate_minute_collision_count(imgs, site)

            if risky > 0:
                # succinct message (wraps nicely beside the checkbox)
                self.sec_warn_lbl.setText(
                    f"⚠ {risky} image(s) share the same minute. "
                    f"Without seconds, those will be suffixed (e.g., _02)."
                )
                self.sec_warn_lbl.setVisible(True)
            else:
                self.sec_warn_lbl.setVisible(False)
                self.sec_warn_lbl.setText("")
        except Exception as e:
            # Never break the UI on warning compute
            print("[WARN] seconds warning compute failed:", e)
            if hasattr(self, "sec_warn_lbl"):
                self.sec_warn_lbl.setVisible(False)
                self.sec_warn_lbl.setText("")

    # ---------- folder pick ----------
    @pyqtSlot()
    def pick_src(self):
        d = QFileDialog.getExistingDirectory(self, "Select folder", str(Path.home()))
        if not d:
            return

        self.src_edit.setText(d)
        print("[DEBUG] Selected folder:", d)

        # Build cache and counts
        try:
            recursive = self.recurse_chk.isChecked()
            self._imgs_cache = iter_images(Path(d), recursive)
            total = len(self._imgs_cache)
            with_ts = 0
            has_seconds_any = False

            for p in self._imgs_cache:
                m = read_image_metadata(p)
                if m.get("DateTimeFromMetadata"):
                    with_ts += 1
                if m.get("HasSeconds", False):
                    has_seconds_any = True

            self.total_count_lbl.setText(f"TOTAL File Count: {total}")
            self.ts_count_lbl.setText(f"Files w/Timestamp: {with_ts}")
            self.update_example()

            # Seconds warning next to checkbox (user in control)
            self._update_seconds_warning()

            # (Optional) early heads-up dialog about seconds presence
            if has_seconds_any and not self.sec_chk.isChecked():
                warn = self._steel_blue_msgbox(
                    "Seconds Present in Metadata",
                    ("Some images contain seconds in 'Date Taken'.\n\n"
                     "You currently have 'Include Seconds in Timestamp' unchecked, "
                     "which may create duplicate names for photos shot in the same minute. "
                     "If duplicates occur, the organizer will disambiguate using "
                     "resolution and/or numeric suffixes."),
                    QMessageBox.Warning
                )
                warn.exec_()

        except Exception as e:
            print("[ERROR] Folder scan failed:", e)
            err_box = self._steel_blue_msgbox("Folder Scan Error", str(e), QMessageBox.Critical)
            err_box.exec_()
            self._imgs_cache = []
            self.total_count_lbl.setText("TOTAL File Count: 0")
            self.ts_count_lbl.setText("Files w/Timestamp: 0")
            self._update_seconds_warning()

    # ---------- main action ----------
    @pyqtSlot()
    def run_organizer(self):
        folder = self.src_edit.text().strip()
        if not folder:
            warn = self._steel_blue_msgbox("Missing Folder", "Please choose a folder of images.", QMessageBox.Warning)
            warn.exec_()
            return

        folder_path = Path(folder)
        recursive = self.recurse_chk.isChecked()

        # Optional: Move files missing Date Taken first (per checkbox)
        try:
            if self.move_missing_chk.isChecked():
                if not self._imgs_cache:
                    self._imgs_cache = iter_images(folder_path, recursive)

                has_dt, missing_dt = [], []
                for p in self._imgs_cache:
                    m = read_image_metadata(p)
                    (has_dt if m.get("DateTimeFromMetadata") else missing_dt).append(p)

                if missing_dt:
                    sub = move_files_to_subfolder(missing_dt, base_dir=folder_path)
                    info = self._steel_blue_msgbox(
                        "Moved Files Without Date",
                        f"Moved {len(missing_dt)} image(s) without 'Date Taken' to:\n{sub}\n\n"
                        "Proceeding to rename the remaining images."
                    )
                    info.exec_()
                    # refresh cache to exclude moved files
                    self._imgs_cache = has_dt
        except Exception as e:
            print("[WARN] Could not move files missing date:", e)

        # Run the organizer
        try:
            log_csv, rows, tags, errors = organize_images(
                src_folder=folder_path,
                recursive=recursive,
                site=self.site_edit.text().strip(),
                include_seconds=self.sec_chk.isChecked(),
                copyright_holder=self.copyright_edit.text().strip(),
                site_info=self.site_info_edit.toPlainText().strip()
            )

            msg = f"Image Organizer completed.\nLog file:\n{log_csv}"
            if errors:
                msg += f"\n\nCompleted with {len(errors)} warning(s)/error(s) (see terminal)."
            else:
                msg += "\n\nAll images processed successfully."

            # Completion dialog with Revert
            done = QMessageBox(self)
            done.setWindowTitle("Done")
            done.setIcon(QMessageBox.Information)
            done.setText(msg)
            revert_btn = done.addButton("Revert", QMessageBox.ActionRole)
            close_btn  = done.addButton("Close", QMessageBox.AcceptRole)
            done.setStyleSheet(BUTTON_CSS)
            done.exec_()

            if done.clickedButton() is revert_btn:
                reverted, rev_errors = revert_operation(folder_path, Path(log_csv))
                if rev_errors:
                    er = self._steel_blue_msgbox(
                        "Revert Finished with Issues",
                        f"Reverted {reverted} file(s).\n\n" +
                        "\n".join(rev_errors[:10]) + ("\n…" if len(rev_errors) > 10 else ""),
                        QMessageBox.Warning
                    )
                    er.exec_()
                else:
                    ok = self._steel_blue_msgbox("Reverted", f"Reverted {reverted} file(s).")
                    ok.exec_()

            print("[INFO] Completed. Log:", log_csv)

        except Exception as e:
            print("[ERROR] Image Organizer failed:", e)
            err = self._steel_blue_msgbox("Error", str(e), QMessageBox.Critical)
            err.exec_()
