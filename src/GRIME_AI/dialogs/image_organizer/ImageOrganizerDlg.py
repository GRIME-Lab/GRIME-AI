# ImageOrganizerDlg.py

import logging

from datetime import datetime
from appcore.utils.resource_utils import ui_path
from appcore.utils import theme
from appcore.JSON_Editor import JsonEditor
from pathlib import Path

from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QTimer, Qt
from PyQt5.QtGui import QTextOption
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox

logger = logging.getLogger(__name__)

# ---- core (your current module) ----
from appcore.image_organizer import (
    organize_images, example_filename,
    iter_images, read_image_metadata,
    move_files_to_subfolder, revert_operation,
    estimate_minute_collision_count,  # <-- warning estimator
)

# Button style, filled from the theme so the buttons match light/dark mode.
# Background and border come from the theme's accent colors; text and hover are
# chosen per theme for contrast against the accent.
_BUTTON_CSS_TEMPLATE = """
QPushButton {{
  background-color: {bg};
  color: {text};
  border: 1px solid {border};
  padding: 6px 14px;
  border-radius: 6px;
}}
QPushButton:hover {{ background-color: {hover}; }}
"""

class ImageOrganizerDlg(QDialog):
    """
    Single-screen Image Organizer dialog.

    - Counts line: Total images / With timestamp.
    - Options: Move Files Missing Date, Operate Recursively, Include Seconds (+ red warning).
    - Site Information wraps at word boundaries; dialog auto-expands (never shrinks).
    - Run shows completion dialog with 'Revert' button.
    - Remembers the last folder, site name, copyright, site info and option
      checkboxes between sessions (stored via JsonEditor).
    """

    # Keys used to persist the dialog's fields between sessions.
    SETTING_LAST_FOLDER = "ImageOrganizer_LastFolder"
    SETTING_SITE_NAME = "ImageOrganizer_SiteName"
    SETTING_COPYRIGHT = "ImageOrganizer_Copyright"
    SETTING_TITLE = "ImageOrganizer_Title"
    SETTING_COMMENTS = "ImageOrganizer_Comments"
    SETTING_SITE_INFO = "ImageOrganizer_SiteInfo"
    SETTING_MOVE_MISSING = "ImageOrganizer_MoveMissingDate"
    SETTING_RECURSIVE = "ImageOrganizer_Recursive"
    SETTING_INCLUDE_SECONDS = "ImageOrganizer_IncludeSeconds"

    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            loadUi(ui_path("image_organizer/QDialog_ImageOrganizer.ui"), self)
        except Exception:
            logger.exception("uic.loadUi failed")
            raise

        # Buttons and group-box titles follow the theme (steelblue in light,
        # accent blue in dark) and restyle automatically when the theme is toggled.
        theme.on_change(self._apply_dialog_theme, owner=self)
        self._apply_dialog_theme()

        # Browse button width is set in the .ui (fixed 100px); no need to set it again here.

        # Site Information: wrap at WORD boundaries using the widget width
        # (UI should also set these, but set here in case)
        self.site_info_edit.setLineWrapMode(self.site_info_edit.WidgetWidth)
        self.site_info_edit.setWordWrapMode(QTextOption.WordWrap)
        self.site_info_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.site_info_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Counts baseline
        self.example_lbl.setText("Example filename: (select a folder)")
        self.total_count_lbl.setText("Total images: 0")
        self.ts_count_lbl.setText("With timestamp: 0")
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
        # Recursive affects which files are counted, so a toggle must re-scan the
        # folder, not just refresh the example/warning.
        self.recurse_chk.toggled.connect(self._on_recurse_toggled)

        # Sync line-edit heights & capture initial size
        self._initial_size = None
        QTimer.singleShot(0, self._sync_text_field_heights)
        QTimer.singleShot(0, self._capture_initial_size)

        # Grow dialog when Site Information grows (never shrink)
        self.site_info_edit.textChanged.connect(self._grow_for_site_info)

        # The batch metadata fields show a blue dot when the entered value would
        # change at least one scanned file. Each entry is
        # (label, base_caption, value_getter, metadata_key).
        self._field_indicators = [
            (self.label_copyright, self.label_copyright.text(), self.copyright_edit.text,      "CopyrightExisting"),
            (self.label_title,     self.label_title.text(),     self.title_edit.text,          "Title"),
            (self.label_comments,  self.label_comments.text(),  self.comments_edit.toPlainText, "Comments"),
            (self.label_siteinfo,  self.label_siteinfo.text(),  self.site_info_edit.toPlainText, "SiteInfoExisting"),
        ]
        self.copyright_edit.textChanged.connect(self._update_change_indicators)
        self.title_edit.textChanged.connect(self._update_change_indicators)
        self.comments_edit.textChanged.connect(self._update_change_indicators)
        self.site_info_edit.textChanged.connect(self._update_change_indicators)

        # cache of images for the last-picked folder (to avoid re-walking twice)
        self._imgs_cache = []
        # cache of parsed metadata keyed by path string, so each image's metadata
        # is read once per folder scan instead of on every pass.
        self._meta_cache = {}

        # Restore fields from the previous session.
        self._load_settings()

    # ---------- settings persistence ----------
    def _load_settings(self):
        """Restore the dialog's fields from the previous session. Best-effort:
        any failure leaves the dialog at its defaults rather than breaking it."""
        try:
            je = JsonEditor()
            folder = je.getValue(self.SETTING_LAST_FOLDER) or ""
            site = je.getValue(self.SETTING_SITE_NAME) or ""
            copyright_holder = je.getValue(self.SETTING_COPYRIGHT) or ""
            title = je.getValue(self.SETTING_TITLE) or ""
            comments = je.getValue(self.SETTING_COMMENTS) or ""
            site_info = je.getValue(self.SETTING_SITE_INFO) or ""
            move_missing = bool(je.getValue(self.SETTING_MOVE_MISSING))
            recursive = bool(je.getValue(self.SETTING_RECURSIVE))
            include_seconds = bool(je.getValue(self.SETTING_INCLUDE_SECONDS))
        except Exception as e:
            logger.warning("Could not load Image Organizer settings: %s", e)
            return

        # Set the fields with signals blocked so the restore does not trigger a
        # scan or warning per field; one refresh is done at the end.
        widgets = [self.src_edit, self.site_edit, self.copyright_edit,
                   self.title_edit, self.comments_edit, self.site_info_edit,
                   self.move_missing_chk, self.recurse_chk, self.sec_chk]
        for wdg in widgets:
            wdg.blockSignals(True)
        try:
            self.site_edit.setText(site)
            self.copyright_edit.setText(copyright_holder)
            self.title_edit.setText(title)
            self.comments_edit.setPlainText(comments)
            self.site_info_edit.setPlainText(site_info)
            self.move_missing_chk.setChecked(move_missing)
            self.recurse_chk.setChecked(recursive)
            self.sec_chk.setChecked(include_seconds)
            if folder:
                self.src_edit.setText(folder)
        finally:
            for wdg in widgets:
                wdg.blockSignals(False)

        # Refresh once. Scan the restored folder if it still exists; otherwise
        # just refresh the example, warning and site-info sizing.
        if folder and Path(folder).is_dir():
            self._scan_folder(folder, announce_seconds=False)
        else:
            self._on_inputs_changed()
            self._grow_for_site_info()
            self._update_change_indicators()

    def _save_settings(self):
        """Persist the dialog's fields for the next session. Best-effort."""
        try:
            je = JsonEditor()
            je.update_json_entry(self.SETTING_LAST_FOLDER, self.src_edit.text().strip())
            je.update_json_entry(self.SETTING_SITE_NAME, self.site_edit.text().strip())
            je.update_json_entry(self.SETTING_COPYRIGHT, self.copyright_edit.text().strip())
            je.update_json_entry(self.SETTING_TITLE, self.title_edit.text().strip())
            je.update_json_entry(self.SETTING_COMMENTS, self.comments_edit.toPlainText().strip())
            je.update_json_entry(self.SETTING_SITE_INFO, self.site_info_edit.toPlainText().strip())
            je.update_json_entry(self.SETTING_MOVE_MISSING, self.move_missing_chk.isChecked())
            je.update_json_entry(self.SETTING_RECURSIVE, self.recurse_chk.isChecked())
            je.update_json_entry(self.SETTING_INCLUDE_SECONDS, self.sec_chk.isChecked())
        except Exception as e:
            logger.warning("Could not save Image Organizer settings: %s", e)

    def closeEvent(self, event):
        # Save fields when the dialog closes (Cancel or the window close button).
        self._save_settings()
        super().closeEvent(event)

    # ---------- sizing helpers ----------
    def _capture_initial_size(self):
        self._initial_size = self.size()
        self.setMinimumSize(self._initial_size)

    def _sync_text_field_heights(self):
        h = self.src_btn.sizeHint().height()
        for le in (self.src_edit, self.site_edit, self.copyright_edit, self.title_edit):
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
    def _button_css(self):
        """Button style sheet built from the current theme's accent colors."""
        return _BUTTON_CSS_TEMPLATE.format(
            bg=theme.color("accent"),
            border=theme.color("accent_border"),
            text=theme.pick("white", "#19232D"),   # dark text on the light-blue dark accent
            hover=theme.pick("#5a93c2", "#A7D0F0"),
        )

    def _apply_dialog_theme(self, *_):
        # Buttons plus steelblue (accent) group-box titles, in one dialog sheet.
        css = self._button_css() + (
            "\nQGroupBox::title { color: %s; }" % theme.color("accent")
        )
        self.setStyleSheet(css)
        # Recolor the change dots to the current theme's accent.
        self._update_change_indicators()

    def _themed_msgbox(self, title: str, text: str, icon=QMessageBox.Information):
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(icon)
        box.setText(text)
        box.setStyleSheet(self._button_css())
        return box

    def _get_meta(self, p):
        """Return parsed metadata for a path, reading it once and caching it for
        the current folder scan."""
        key = str(p)
        m = self._meta_cache.get(key)
        if m is None:
            m = read_image_metadata(p)
            self._meta_cache[key] = m
        return m

    def _change_dot(self):
        """A blue (accent) superscript dot marking a field that would change files."""
        return ' <sup><span style="color:%s;">&#9679;</span></sup>' % theme.color("accent")

    def _update_change_indicators(self, *_):
        """Show the blue dot on Copyright, Title, Comments and Site Information
        when the entered value is non-empty and differs from at least one scanned
        file. With no folder scanned, no dot is shown."""
        if not hasattr(self, "_field_indicators"):
            return
        metas = [self._get_meta(p) for p in self._imgs_cache] if self._imgs_cache else []
        dot = self._change_dot()
        for label, base, getter, key in self._field_indicators:
            value = (getter() or "").strip()
            dirty = bool(value) and any(str(m.get(key, "")).strip() != value for m in metas)
            label.setText(base + (dot if dirty else ""))

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

    def _on_recurse_toggled(self):
        # Recursive changes the set of files, so re-scan the picked folder and
        # refresh the counts. If no folder is picked yet, just refresh inputs.
        folder = self.src_edit.text().strip()
        if folder:
            self._scan_folder(folder, announce_seconds=False)
        else:
            self._on_inputs_changed()

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
            logger.warning("seconds warning compute failed: %s", e)
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
        # Default the site name from the folder name when the user hasn't set one.
        if not self.site_edit.text().strip():
            self.site_edit.setText(Path(d).name)
        logger.debug("Selected folder: %s", d)
        self._scan_folder(d, announce_seconds=True)

    def _scan_folder(self, folder, announce_seconds=True):
        """Walk the folder, refresh the counts, rebuild the image and metadata
        caches, and update the example and seconds warning. announce_seconds
        controls the one-time heads-up dialog, shown on a fresh pick but not on a
        mere recursive toggle."""
        try:
            recursive = self.recurse_chk.isChecked()
            self._meta_cache = {}
            self._imgs_cache = iter_images(Path(folder), recursive)
            total = len(self._imgs_cache)
            with_ts = 0
            has_seconds_any = False

            for p in self._imgs_cache:
                m = self._get_meta(p)
                if m.get("DateTimeFromMetadata"):
                    with_ts += 1
                if m.get("HasSeconds", False):
                    has_seconds_any = True

            self.total_count_lbl.setText(f"Total images: {total}")
            self.ts_count_lbl.setText(f"With timestamp: {with_ts}")
            self.update_example()

            # Seconds warning next to checkbox (user in control)
            self._update_seconds_warning()

            # Blue dots on the batch metadata fields now that files are known.
            self._update_change_indicators()

            # (Optional) early heads-up dialog about seconds presence
            if announce_seconds and has_seconds_any and not self.sec_chk.isChecked():
                warn = self._themed_msgbox(
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
            logger.error("Folder scan failed: %s", e)
            err_box = self._themed_msgbox("Folder Scan Error", str(e), QMessageBox.Critical)
            err_box.exec_()
            self._imgs_cache = []
            self._meta_cache = {}
            self.total_count_lbl.setText("Total images: 0")
            self.ts_count_lbl.setText("With timestamp: 0")
            self._update_seconds_warning()
            self._update_change_indicators()

    # ---------- main action ----------
    @pyqtSlot()
    def run_organizer(self):
        folder = self.src_edit.text().strip()
        if not folder:
            warn = self._themed_msgbox("Missing Folder", "Please choose a folder of images.", QMessageBox.Warning)
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
                    m = self._get_meta(p)
                    (has_dt if m.get("DateTimeFromMetadata") else missing_dt).append(p)

                if missing_dt:
                    sub = move_files_to_subfolder(missing_dt, base_dir=folder_path)
                    info = self._themed_msgbox(
                        "Moved Files Without Date",
                        f"Moved {len(missing_dt)} image(s) without 'Date Taken' to:\n{sub}\n\n"
                        "Proceeding to rename the remaining images."
                    )
                    info.exec_()
                    # refresh cache to exclude moved files
                    self._imgs_cache = has_dt
        except Exception as e:
            logger.warning("Could not move files missing date: %s", e)

        # Run the organizer
        try:
            log_csv, rows, tags, errors = organize_images(
                src_folder=folder_path,
                recursive=recursive,
                site=self.site_edit.text().strip(),
                include_seconds=self.sec_chk.isChecked(),
                copyright_holder=self.copyright_edit.text().strip(),
                site_info=self.site_info_edit.toPlainText().strip(),
                title=self.title_edit.text().strip(),
                comments=self.comments_edit.toPlainText().strip()
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
            done.setStyleSheet(self._button_css())
            done.exec_()

            if done.clickedButton() is revert_btn:
                reverted, rev_errors = revert_operation(folder_path, Path(log_csv))
                if rev_errors:
                    er = self._themed_msgbox(
                        "Revert Finished with Issues",
                        f"Reverted {reverted} file(s).\n\n" +
                        "\n".join(rev_errors[:10]) + ("\n…" if len(rev_errors) > 10 else ""),
                        QMessageBox.Warning
                    )
                    er.exec_()
                else:
                    ok = self._themed_msgbox("Reverted", f"Reverted {reverted} file(s).")
                    ok.exec_()

            logger.info("Completed. Log: %s", log_csv)
            # Persist the fields after a successful run so they survive a later crash.
            self._save_settings()

        except Exception as e:
            logger.error("Image Organizer failed: %s", e)
            err = self._themed_msgbox("Error", str(e), QMessageBox.Critical)
            err.exec_()
