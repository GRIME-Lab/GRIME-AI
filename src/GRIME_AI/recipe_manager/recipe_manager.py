"""
recipe_manager.py  (package: GRIME_AI.recipe_manager)

Recipe Manager for GRIME AI.

A "recipe" is a named study-site configuration that bundles the output/download
folders you would otherwise have to reset by hand every time you switch sites:

    root            - the site's top-level folder
    composites      - where composite slices are saved
    videos          - where videos are saved
    gifs            - where GIFs are saved
    usgs            - where USGS files are downloaded
    neon            - where NEON files are downloaded

Recipes persist as JSON. Activating a recipe emits `recipeActivated(Recipe)`
so the main GRIME AI window can push these paths into its existing options.

Run standalone to try it:  python grime_ai_recipe_manager.py
"""

import os
import re
import json
import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QListWidget, QListWidgetItem, QLineEdit, QPushButton, QLabel,
    QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox, QFileDialog,
    QMessageBox, QWidget, QCheckBox, QDialogButtonBox, QScrollArea,
    QStackedWidget,
)

# Editable path fields, in display order, mapped to their default sub-folder name.
# Input-folder fields (set manually; NOT auto-derived/rebased from root).
INPUT_FIELDS = [
    ("image_input", "Image input"),
    ("data_input",  "Data input"),
]
# Output-folder fields, in display order, mapped to their default sub-folder name.
OUTPUT_FIELDS = [
    ("composites",  "Composite slices"),
    ("videos",      "Videos"),
    ("gifs",        "GIFs"),
    ("usgs",        "USGS downloads"),
    ("neon",        "NEON downloads"),
]
# Machine-learning fields (set manually; NOT auto-derived/rebased from root).
ML_FIELDS = [
    ("ml_images", "Training images"),
]
# All editable path fields, for load / save round-tripping.
PATH_FIELDS = INPUT_FIELDS + OUTPUT_FIELDS + ML_FIELDS
# Auto-fill / rebase apply to OUTPUT folders only.
SUBFOLDER_DEFAULTS = {
    "composites":  "composites",
    "videos":      "Videos",
    "gifs":        "GIFs",
    "usgs":        "usgs",
    "neon":        "neon",
}


def _now_iso() -> str:
    """Current local timestamp as an ISO-8601 string (second precision)."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def _os_path(path: str) -> str:
    """Return `path` with separators canonical to the host OS (backslash on
    Windows, forward slash on POSIX). Empty stays empty (avoids normpath('.')).."""
    return os.path.normpath(path) if path and path.strip() else ""


def _fmt_dt(iso: str) -> str:
    """Format a stored ISO timestamp for display, or an em dash if unset."""
    if not iso:
        return "\u2014"
    try:
        return datetime.datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return iso


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Recipe:
    name: str
    created: str = ""
    modified: str = ""
    root: str = ""
    image_input: str = ""
    data_input: str = ""
    composites: str = ""
    videos: str = ""
    gifs: str = ""
    usgs: str = ""
    neon: str = ""
    ml_images: str = ""

    def derive_from_root(self, overwrite: bool = False) -> None:
        """Fill sub-folder paths as <root>/<default>. If overwrite is False,
        only empty fields are filled."""
        if not self.root:
            return
        for attr, sub in SUBFOLDER_DEFAULTS.items():
            if overwrite or not getattr(self, attr):
                setattr(self, attr, _os_path(os.path.join(self.root, sub)))

    @staticmethod
    def _parts(path: str) -> list:
        """Split a path into components on either separator, dropping empties."""
        return [c for c in re.split(r"[\\/]+", (path or "").strip()) if c]

    @staticmethod
    def rebase_path(path: str, old_root: str, new_root: str) -> str:
        """If `path` lives under `old_root` (matched on component boundaries,
        case-insensitively on Windows), return it re-rooted under `new_root`
        with its tail preserved. Otherwise return `path` unchanged.

        Component-boundary matching avoids false hits like
        .../Junk/Missouri matching .../Junk/Missouri2/...
        """
        if not path or not old_root or not new_root:
            return path
        p, o = Recipe._parts(path), Recipe._parts(old_root)
        if len(p) < len(o):
            return path
        fold = (lambda s: s.casefold()) if os.name == "nt" else (lambda s: s)
        if [fold(c) for c in p[:len(o)]] != [fold(c) for c in o]:
            return path
        tail = p[len(o):]
        base = new_root.rstrip("/\\")
        result = base + ("/" + "/".join(tail) if tail else "")
        return _os_path(result)

    def rebase_subfolders(self, old_root: str, new_root: str) -> bool:
        """Re-root every sub-folder that currently lives under `old_root`.
        Returns True if anything changed."""
        changed = False
        for attr in ("composites", "videos", "gifs", "usgs", "neon"):
            new_val = Recipe.rebase_path(getattr(self, attr), old_root, new_root)
            if new_val != getattr(self, attr):
                setattr(self, attr, new_val)
                changed = True
        return changed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        fields = ("name", "created", "modified", "root",
                  "image_input", "data_input",
                  "composites", "videos", "gifs", "usgs", "neon",
                  "ml_images")
        out = {k: d.get(k, "") for k in fields}
        # Back-compat: recipes saved before Videos/GIFs were split into two
        # fields carry a single "videos_gifs" path — seed both from it.
        legacy = d.get("videos_gifs", "")
        if legacy:
            out["videos"] = out["videos"] or legacy
            out["gifs"] = out["gifs"] or legacy
        return cls(**out)


class RecipeStore:
    """Loads/saves recipes to a JSON file and tracks the active recipe."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or self._default_path()
        self.recipes: List[Recipe] = []
        self.active_name: Optional[str] = None
        self.load()

    @staticmethod
    def _default_path() -> str:
        # Keep recipes alongside the rest of GRIME AI's config, in the visible
        # Documents/GRIME-AI/Settings folder (NOT a hidden AppData location).
        try:
            from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
            settings = GRIME_AI_Save_Utils().get_settings_folder()
        except Exception:
            settings = os.path.join(os.path.expanduser("~"),
                                    "Documents", "GRIME-AI", "Settings")
            os.makedirs(settings, exist_ok=True)
        return os.path.join(settings, "GRIME_AI_Recipes.json")

    def load(self) -> None:
        self.recipes, self.active_name = [], None
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.recipes = [Recipe.from_dict(r) for r in data.get("recipes", [])]
                self.active_name = data.get("active")
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass  # start empty on a corrupt/unreadable file

    def save(self) -> None:
        data = {"active": self.active_name,
                "recipes": [r.to_dict() for r in self.recipes]}
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def names(self) -> List[str]:
        return [r.name for r in self.recipes]

    def get(self, name: str) -> Optional[Recipe]:
        return next((r for r in self.recipes if r.name == name), None)

    def add(self, recipe: Recipe) -> None:
        self.recipes.append(recipe)

    def remove(self, name: str) -> None:
        self.recipes = [r for r in self.recipes if r.name != name]
        if self.active_name == name:
            self.active_name = None

    def get_active(self) -> Optional[Recipe]:
        return self.get(self.active_name) if self.active_name else None

    def unique_name(self, base: str) -> str:
        existing = set(self.names())
        if base not in existing:
            return base
        i = 2
        while f"{base} ({i})" in existing:
            i += 1
        return f"{base} ({i})"


# --------------------------------------------------------------------------- #
# Dialog
# --------------------------------------------------------------------------- #
class RecipeManagerDialog(QDialog):
    """CRUD editor for study-site recipes. Emits recipeActivated(Recipe)."""

    recipeActivated = pyqtSignal(object)  # emits a Recipe

    def __init__(self, store: RecipeStore, parent=None):
        super().__init__(parent)
        self.store = store
        self._current: Optional[Recipe] = None
        self._loading = False   # guards field-change handlers during load
        self._dirty = False
        self._prev_root = ""    # root value before the current edit (for rebasing)
        self._drafting = False  # True while a New (uncommitted) recipe is in the form

        self.setWindowTitle("GRIME AI — Recipe Manager")
        self.setMinimumSize(820, 520)
        self.resize(880, 600)
        self._build_ui()
        self._refresh_list(select=self.store.active_name)

    # ---- UI construction -------------------------------------------------- #
    def _build_ui(self) -> None:
        # Shared style + uniform width so Browse (and Reset) buttons align in a
        # column and match in size.
        self._browse_style = (
            "QPushButton { background-color: #996515; color: white;"
            " border: none; padding: 4px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #7d5211; }"
            "QPushButton:pressed { background-color: #63410d; }")
        _sample = QPushButton("Browse…")
        _sample.setStyleSheet(self._browse_style)
        self._browse_w = _sample.sizeHint().width()
        _sample.deleteLater()

        # Left: recipe list + list actions
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_select)
        self.list.itemDoubleClicked.connect(lambda *_: self._set_active())

        btn_new = QPushButton("New")
        btn_add = QPushButton("Add")
        btn_dup = QPushButton("Duplicate")
        btn_del = QPushButton("Delete")
        btn_act = QPushButton("Set Active")
        btn_new.clicked.connect(self._new)
        btn_add.clicked.connect(self._add)
        btn_add.setStyleSheet(
            "QPushButton { background-color: #2E7D32; color: white;"
            " border: none; padding: 4px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #256628; }"
            "QPushButton:pressed { background-color: #1e5220; }"
            "QPushButton:disabled { background-color: #a8c3aa; color: #eef4ee; }")
        btn_dup.clicked.connect(self._duplicate)
        btn_dup.setStyleSheet(
            "QPushButton { background-color: transparent; color: #000000;"
            " border: 1px solid #000000; padding: 4px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: rgba(0, 0, 0, 0.08); }"
            "QPushButton:pressed { background-color: rgba(0, 0, 0, 0.16); }"
            "QPushButton:disabled { color: #9e9e9e; border-color: #cccccc; }")
        btn_del.clicked.connect(self._delete)
        btn_del.setStyleSheet(
            "QPushButton { background-color: transparent; color: #c0392b;"
            " border: 1px solid #c0392b; padding: 4px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: rgba(192, 57, 43, 0.12); }"
            "QPushButton:pressed { background-color: rgba(192, 57, 43, 0.22); }"
            "QPushButton:disabled { color: #d9a5a0; border-color: #e3c2be; }")
        btn_act.clicked.connect(self._set_active)
        btn_act.setStyleSheet(
            "QPushButton { background-color: #4682B4; color: white;"
            " border: none; padding: 4px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #3a6d99; }"
            "QPushButton:pressed { background-color: #2f587c; }"
            "QPushButton:disabled { background-color: #b7c7d4; color: #eef2f5; }")
        self.btn_new, self.btn_add = btn_new, btn_add
        self.btn_dup, self.btn_del, self.btn_act = btn_dup, btn_del, btn_act

        left = QVBoxLayout()
        left.addWidget(QLabel("Recipes"))
        left.addWidget(self.list, 1)
        row1 = QHBoxLayout()
        row1.addWidget(btn_new)
        row1.addWidget(btn_add)
        row2 = QHBoxLayout()
        row2.addWidget(btn_dup)
        row2.addWidget(btn_del)
        left.addLayout(row1)
        left.addLayout(row2)
        left.addWidget(btn_act)
        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(240)

        # Right: editor
        self.name_edit = QLineEdit()
        self.name_edit.textEdited.connect(self._on_name_edited)

        self.root_edit, root_row = self._path_row("Browse…", self._browse_root)
        self.root_edit.textChanged.connect(self._on_root_changed)
        self.root_edit.editingFinished.connect(self._commit_root_change)

        self.autofill = QCheckBox("Auto-fill output sub-folders from root")
        self.autofill.setChecked(True)

        self.created_label = QLabel("\u2014")
        self.modified_label = QLabel("\u2014")

        # Identity / site form (name, root, dates)
        id_form = QFormLayout()
        id_form.addRow("Name", self.name_edit)
        id_form.addRow("Root (site)", root_row)
        id_form.addRow("", self.autofill)
        id_form.addRow("Created", self.created_label)
        id_form.addRow("Modified", self.modified_label)

        # Per-path rows are built into self.path_edits and grouped below.
        self.path_edits = {}

        def _build_group(title, fields, trailing=None):
            gform = QFormLayout()
            for attr, label in fields:
                edit, row_w = self._path_row(
                    "Browse…", lambda _=False, a=attr: self._browse_field(a))
                edit.textChanged.connect(
                    lambda _=None, a=attr: self._on_field_changed(a))
                self.path_edits[attr] = edit
                gform.addRow(label, row_w)
            if trailing is not None:
                trow = QWidget()
                tlay = QHBoxLayout(trow)
                tlay.setContentsMargins(0, 0, 0, 0)
                tlay.addStretch(1)
                tlay.addWidget(trailing)
                gform.addRow("", trow)
            box = QGroupBox(title)
            box.setLayout(gform)
            return box

        input_box = _build_group("Input folders", INPUT_FIELDS)

        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Reset output sub-folders to root defaults")
        btn_reset.setFixedWidth(self._browse_w)
        btn_reset.setStyleSheet(
            "QPushButton { background-color: transparent; color: #c0392b;"
            " border: 1px solid #c0392b; padding: 4px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: rgba(192, 57, 43, 0.12); }"
            "QPushButton:pressed { background-color: rgba(192, 57, 43, 0.22); }"
            "QPushButton:disabled { color: #d9a5a0; border-color: #e3c2be; }")
        btn_reset.clicked.connect(self._reset_subfolders)
        output_box = _build_group("Output folders", OUTPUT_FIELDS, trailing=btn_reset)
        ml_box = _build_group("Machine Learning", ML_FIELDS)

        editor_layout = QVBoxLayout()
        editor_layout.addLayout(id_form)
        editor_layout.addWidget(input_box)
        editor_layout.addWidget(output_box)
        editor_layout.addWidget(ml_box)
        editor_layout.addStretch(1)

        self.editor = QGroupBox("Recipe")
        self.editor.setLayout(editor_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(self.editor)

        # Empty-state placeholder shown when there is no recipe to edit
        # (e.g. a first-time user who hasn't created any recipes yet).
        self._empty_label = QLabel()
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet("color: gray;")
        empty_new_btn = QPushButton("New Recipe")
        empty_new_btn.clicked.connect(self._new)
        empty_lay = QVBoxLayout()
        empty_lay.addStretch(1)
        empty_lay.addWidget(self._empty_label)
        empty_lay.addSpacing(8)
        empty_btn_row = QHBoxLayout()
        empty_btn_row.addStretch(1)
        empty_btn_row.addWidget(empty_new_btn)
        empty_btn_row.addStretch(1)
        empty_lay.addLayout(empty_btn_row)
        empty_lay.addStretch(2)
        empty_page = QWidget()
        empty_page.setLayout(empty_lay)

        self._stack = QStackedWidget()
        self._stack.addWidget(empty_page)   # index 0 = placeholder
        self._stack.addWidget(scroll)       # index 1 = editor

        # Bottom buttons
        bb = QDialogButtonBox()
        self.btn_save = bb.addButton("Save", QDialogButtonBox.AcceptRole)
        self.btn_close = bb.addButton("Close", QDialogButtonBox.RejectRole)
        self.btn_save.clicked.connect(self._save)
        self.btn_close.clicked.connect(self.close)

        right = QVBoxLayout()
        right.addWidget(self._stack, 1)

        top = QHBoxLayout()
        top.addWidget(left_w)
        top.addLayout(right, 1)

        outer = QVBoxLayout(self)
        outer.addLayout(top, 1)
        outer.addWidget(bb)

    def _path_row(self, browse_text, on_browse):
        edit = QLineEdit()
        btn = QPushButton(browse_text)
        btn.clicked.connect(on_browse)
        btn.setStyleSheet(self._browse_style)
        btn.setFixedWidth(self._browse_w)
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(edit, 1)
        lay.addWidget(btn)
        return edit, w

    # ---- List management -------------------------------------------------- #
    def _refresh_list(self, select: Optional[str] = None) -> None:
        self._drafting = False
        self.list.blockSignals(True)
        self.list.clear()
        for r in self.store.recipes:
            label = f"\u2605 {r.name}" if r.name == self.store.active_name else r.name
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, r.name)
            self.list.addItem(item)
        self.list.blockSignals(False)
        if select:
            self._select_by_name(select)
        elif self.list.count():
            self.list.setCurrentRow(0)
        else:
            self._load_recipe(None)

    def _select_by_name(self, name: str) -> None:
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.UserRole) == name:
                self.list.setCurrentRow(i)
                return

    def _on_select(self, current, _prev) -> None:
        # If the user clicks a list item while drafting a New recipe, confirm
        # before abandoning the unsaved draft.
        if self._drafting and not self._confirm_discard_draft():
            self.list.blockSignals(True)
            self.list.setCurrentRow(-1)
            self.list.blockSignals(False)
            return
        self._drafting = False
        name = current.data(Qt.UserRole) if current else None
        self._load_recipe(self.store.get(name) if name else None)

    # ---- Editor load / write-back ---------------------------------------- #
    def _load_recipe(self, recipe: Optional[Recipe]) -> None:
        self._current = recipe
        self._prev_root = recipe.root if recipe else ""
        self._loading = True
        enabled = recipe is not None
        self.editor.setEnabled(enabled)
        self.name_edit.setText(recipe.name if recipe else "")
        self.root_edit.setText(recipe.root if recipe else "")
        for attr, edit in self.path_edits.items():
            edit.setText(getattr(recipe, attr) if recipe else "")
        self._update_date_labels()
        self._loading = False
        self._update_editor_visibility()
        self._update_buttons()

    def _update_editor_visibility(self) -> None:
        """Show the editor when a recipe is selected or being drafted; otherwise
        show the empty-state placeholder with guidance."""
        show_editor = self._drafting or (self._current is not None)
        self._stack.setCurrentIndex(1 if show_editor else 0)
        if not show_editor:
            if not self.store.recipes:
                self._empty_label.setText(
                    "You haven't created any recipes yet.\n\n"
                    "A recipe stores a study site's input and output folders so "
                    "you can switch sites without re-entering paths.\n\n"
                    "Click \u201cNew\u201d to create your first one.")
            else:
                self._empty_label.setText(
                    "Select a recipe from the list, or click \u201cNew\u201d "
                    "to create another.")

    def _update_date_labels(self) -> None:
        r = self._current
        self.created_label.setText(_fmt_dt(r.created) if r else "\u2014")
        self.modified_label.setText(_fmt_dt(r.modified) if r else "\u2014")

    def _touch_current(self) -> None:
        """Bump the current recipe's modified time on a content edit. Draft
        recipes are timestamped at Add, so this is a no-op while drafting."""
        if self._current and not self._drafting:
            self._current.modified = _now_iso()
            self._update_date_labels()

    def _on_name_edited(self, text: str) -> None:
        if self._loading or not self._current:
            return
        text = text.strip()
        if self._drafting:
            # Draft names are validated at Add time, not while typing.
            self._current.name = text
            self._mark_dirty()
            return
        if text and text != self._current.name and text in self.store.names():
            return  # ignore collisions silently; validated on save
        self._current.name = text
        self._mark_dirty()
        self._touch_current()
        item = self.list.currentItem()
        if item:
            prefix = "\u2605 " if text == self.store.active_name else ""
            item.setText(prefix + text)
            item.setData(Qt.UserRole, text)

    def _on_root_changed(self, text: str) -> None:
        # Keep the model's root in sync while typing, but defer any sub-folder
        # rebasing until the edit is committed (editingFinished / Browse) so we
        # don't churn paths on every keystroke.
        if self._loading or not self._current:
            return
        self._current.root = text.strip()
        self._mark_dirty()
        self._touch_current()

    def _commit_root_change(self) -> None:
        """On a committed root change, re-root every sub-folder that lived under
        the OLD root (tail preserved). Then, if auto-fill is on, seed any still
        empty sub-folder from the new root."""
        if self._loading or not self._current:
            return
        new_root = _os_path(self.root_edit.text().strip())
        old_root = self._prev_root
        # Reflect the OS-canonical form back into the field.
        if new_root != self.root_edit.text():
            self._loading = True
            self.root_edit.setText(new_root)
            self._loading = False
        self._current.root = new_root

        changed = False
        if old_root and new_root and old_root != new_root:
            changed |= self._current.rebase_subfolders(old_root, new_root)
        if self.autofill.isChecked():
            before = self._current.to_dict()
            self._current.derive_from_root(overwrite=False)  # fill empties only
            changed |= (self._current.to_dict() != before)

        self._prev_root = new_root
        if changed:
            self._loading = True
            for attr, edit in self.path_edits.items():
                edit.setText(getattr(self._current, attr))
            self._loading = False
            self._mark_dirty()

    def _on_field_changed(self, attr: str) -> None:
        if self._loading or not self._current:
            return
        setattr(self._current, attr, self.path_edits[attr].text().strip())
        self._mark_dirty()
        self._touch_current()

    # ---- Actions ---------------------------------------------------------- #
    def _new(self) -> None:
        """Start a blank draft in the form. It is NOT added to the list until
        'Add' is clicked, and never becomes active until 'Set Active'."""
        if self._drafting and not self._confirm_discard_draft():
            return
        self.list.blockSignals(True)
        self.list.setCurrentRow(-1)
        self.list.blockSignals(False)
        self._drafting = True
        self._load_recipe(Recipe(name=""))
        self.name_edit.setFocus()

    def _add(self) -> None:
        """Commit the current draft into the list. Requires a unique, non-empty
        name. Does not change which recipe is active."""
        if not self._drafting or not self._current:
            return
        name = self._current.name.strip()
        if not name:
            QMessageBox.warning(self, "Add Recipe", "Give the recipe a name first.")
            self.name_edit.setFocus()
            return
        if name in self.store.names():
            QMessageBox.warning(self, "Add Recipe",
                                f"A recipe named '{name}' already exists.")
            self.name_edit.setFocus()
            return
        self._current.name = name
        now = _now_iso()
        self._current.created = now
        self._current.modified = now
        self.store.add(self._current)
        self._mark_dirty()
        self._refresh_list(select=name)  # clears drafting, selects the new item

    def _duplicate(self) -> None:
        if self._drafting or not self._current:
            return
        clone = Recipe.from_dict(self._current.to_dict())
        clone.name = self.store.unique_name(f"{self._current.name} copy")
        now = _now_iso()
        clone.created = now
        clone.modified = now
        self.store.add(clone)
        self._mark_dirty()
        self._refresh_list(select=clone.name)

    def _delete(self) -> None:
        if self._drafting or not self._current:
            return
        if QMessageBox.question(
            self, "Delete Recipe",
            f"Delete recipe '{self._current.name}'?",
        ) != QMessageBox.Yes:
            return
        self.store.remove(self._current.name)
        self._mark_dirty()
        self._refresh_list()

    def _set_active(self) -> None:
        if self._drafting or not self._current:
            return
        if not self._current.name:
            QMessageBox.warning(self, "Recipe Manager", "Give the recipe a name first.")
            return
        self.store.active_name = self._current.name
        self._mark_dirty()
        self._refresh_list(select=self._current.name)
        self.recipeActivated.emit(self._current)

    # ---- Draft / button-state helpers ------------------------------------ #
    def _confirm_discard_draft(self) -> bool:
        """Return True if it's OK to abandon the current draft. Prompts only
        when the draft actually has content."""
        d = self._current
        has_content = bool(d and (d.name.strip() or d.root
                                  or d.image_input or d.data_input or d.composites
                                  or d.videos or d.gifs or d.usgs or d.neon
                                  or d.ml_images))
        if not has_content:
            return True
        return QMessageBox.question(
            self, "Discard new recipe?",
            "The new recipe hasn't been added to the list. Discard it?",
        ) == QMessageBox.Yes

    def _update_buttons(self) -> None:
        drafting = self._drafting
        has_sel = (self._current is not None) and not drafting
        self.btn_new.setEnabled(True)
        self.btn_add.setEnabled(drafting)
        self.btn_dup.setEnabled(has_sel)
        self.btn_del.setEnabled(has_sel)
        self.btn_act.setEnabled(has_sel)

    def _reset_subfolders(self) -> None:
        if not self._current:
            return
        self._current.derive_from_root(overwrite=True)
        self._loading = True
        for attr, edit in self.path_edits.items():
            edit.setText(getattr(self._current, attr))
        self._loading = False
        self._mark_dirty()
        self._touch_current()

    # ---- Browsers --------------------------------------------------------- #
    def _browse_root(self) -> None:
        start = self.root_edit.text() or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, "Select site root folder", start)
        if d:
            self.root_edit.setText(d)
            self._commit_root_change()   # rebase sub-folders under the new root

    def _browse_field(self, attr: str) -> None:
        start = (self.path_edits[attr].text()
                 or self.root_edit.text() or os.path.expanduser("~"))
        d = QFileDialog.getExistingDirectory(self, "Select folder", start)
        if d:
            self.path_edits[attr].setText(_os_path(d))

    # ---- Save / dirty / close -------------------------------------------- #
    def _mark_dirty(self) -> None:
        self._dirty = True
        self.setWindowTitle("GRIME AI — Recipe Manager *")

    def _validate(self) -> Optional[str]:
        names = self.store.names()
        for r in self.store.recipes:
            if not r.name.strip():
                return "Every recipe needs a name."
        if len(names) != len(set(names)):
            return "Recipe names must be unique."
        return None

    def _save(self) -> bool:
        err = self._validate()
        if err:
            QMessageBox.warning(self, "Recipe Manager", err)
            return False
        self.store.save()
        self._dirty = False
        self.setWindowTitle("GRIME AI — Recipe Manager")
        return True

    def closeEvent(self, event) -> None:
        if self._dirty:
            resp = QMessageBox.question(
                self, "Unsaved changes", "Save changes before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if resp == QMessageBox.Save:
                if not self._save():
                    event.ignore()
                    return
            elif resp == QMessageBox.Cancel:
                event.ignore()
                return
        event.accept()


# --------------------------------------------------------------------------- #
# Standalone test harness
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    # Standalone: operate on the real store (Documents/GRIME-AI/Settings).
    store = RecipeStore()

    dlg = RecipeManagerDialog(store)
    dlg.recipeActivated.connect(
        lambda r: print(f"[activated] {r.name}\n"
                        f"  created:     {r.created}\n"
                        f"  modified:    {r.modified}\n"
                        f"  root:        {r.root}\n"
                        f"  image_input: {r.image_input}\n"
                        f"  data_input:  {r.data_input}\n"
                        f"  composites:  {r.composites}\n"
                        f"  videos:      {r.videos}\n"
                        f"  gifs:        {r.gifs}\n"
                        f"  usgs:        {r.usgs}\n"
                        f"  neon:        {r.neon}"))
    dlg.exec_()
    sys.exit(0)
