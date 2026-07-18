#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# License: Apache License, Version 2.0

"""
GRIME AI — Site Config editor (CLI + GUI in one script).

A *modify* tool for a site config JSON. It loads an existing file, applies only
the parameters you specify (a partial merge — everything else is preserved),
validates, and writes it back. One script, two front-ends sharing one core:

  CLI (edit in place):
      python GRIME_AI_site_config.py <path> --lr 0.003 --weight-decay 0.01
      python GRIME_AI_site_config.py <path> --lr 0.003 --lr 0.001   # sweep
      python GRIME_AI_site_config.py <path> --show

  GUI (visual editor; also does Save As under a new filename):
      python GRIME_AI_site_config.py --gui
      python GRIME_AI_site_config.py            # no args -> GUI
      (or Tools -> Site Config Editor in GRIME AI)

The core (the _PARAMS table, split validation, load/write) is stdlib-only. Qt is
imported lazily, only when the GUI is actually opened, so the CLI stays headless.
"""

import argparse
import copy
import json
import os
import sys


# ============================================================================
# CORE — editable parameters, validation, load/write (stdlib only)
# ============================================================================

# Editable scalar parameters: (flag, json_key, kind, help)
# kind in {"float", "int", "str", "bool", "float_list"}
_PARAMS = [
    ("--lr",             "learningRates",              "float_list", "Learning rate; repeat for a sweep (e.g. --lr 3e-3 --lr 1e-3)"),
    ("--weight-decay",   "weight_decay",               "float",      "Weight decay"),
    ("--epochs",         "number_of_epochs",           "int",        "Number of epochs"),
    ("--batch-size",     "batch_size",                 "int",        "Batch size"),
    ("--optimizer",      "optimizer",                  "str",        "Optimizer name (e.g. Adam, AdamW, SGD)"),
    ("--loss",           "loss_function",              "str",        "Loss function name (e.g. IOU, BCE + Dice + Score)"),
    ("--patience",       "patience",                   "int",        "Early-stopping patience (epochs)"),
    ("--early-stopping", "early_stopping",             "bool",       "Enable early stopping (true/false)"),
    ("--save-freq",      "save_model_frequency",       "int",        "Checkpoint save frequency (epochs)"),
    ("--val-freq",       "validation_frequency",       "int",        "Validation frequency (epochs)"),
    ("--device",         "device",                     "str",        "Device (gpu/cpu)"),
    ("--train-split",    "train_split",                "float",      "Training fraction, 0-1"),
    ("--val-split",      "val_split",                  "float",      "Validation fraction, 0-1"),
    ("--blob-radius",    "blob_filter_radius",         "float",      "Blob-filter radius fraction"),
    ("--blob-mode",      "blob_radius_mode",           "str",        "Blob radius mode (e.g. Manual, Computed)"),
    ("--num-clusters",   "num_clusters",               "int",        "Number of clusters"),
    ("--overlay-samples","validation_overlay_samples", "int",        "Validation overlay sample count"),
    ("--yolo-weights",   "yolo_base_weights",          "str",        "YOLO base-weights filename"),
    ("--use-lora",       "use_lora",                   "bool",       "Enable LoRA (true/false)"),
    ("--lora-rank",      "lora_rank",                  "int",        "LoRA rank"),
    ("--lora-alpha",     "lora_alpha",                 "int",        "LoRA alpha"),
    ("--lora-dropout",   "lora_dropout",               "float",      "LoRA dropout"),
    ("--lora-bias",      "lora_bias",                  "str",        "LoRA bias mode (none/all/lora_only)"),
    ("--site-name",      "siteName",                   "str",        "Site name"),
]

_SPLIT_EPS = 1e-6


def _str2bool(v):
    s = str(v).strip().lower()
    if s in ("true", "t", "yes", "y", "1", "on"):
        return True
    if s in ("false", "f", "no", "n", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {v!r}")


def _dest(flag):
    return flag.lstrip("-").replace("-", "_")


def _default_config_path():
    """Resolve the settings-folder site_config.json. Imported lazily so the
    module stays light unless the default path is actually needed."""
    from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
    settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
    return os.path.normpath(os.path.join(settings_folder, "site_config.json"))


def _detect_eol(raw_bytes):
    return "\r\n" if b"\r\n" in raw_bytes else "\n"


def _load(path):
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8")), _detect_eol(raw)


def _write(path, cfg, eol):
    text = json.dumps(cfg, indent=2, ensure_ascii=False)
    if eol == "\r\n":
        text = text.replace("\n", "\r\n")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def _validate_split(cfg):
    """Enforce the split invariant: train and val are non-negative and sum to at
    most 100%. Mirrors DatasetUtils.split_dataset so a config saved here can
    never trip the trainer's guard."""
    train = cfg.get("train_split")
    val = cfg.get("val_split")
    if train is None or val is None:
        return
    train, val = float(train), float(val)
    if train < 0 or val < 0:
        raise ValueError(
            f"train_split ({train}) and val_split ({val}) must be non-negative."
        )
    if train + val > 1.0 + _SPLIT_EPS:
        raise ValueError(
            f"train_split ({train}) + val_split ({val}) = {train + val:.4f} "
            f"exceeds 1.0. Splits may sum to at most 100% "
            f"(unlink allows less, never more)."
        )


# ============================================================================
# CLI
# ============================================================================

def _add_config_arguments(p):
    """Add the config-editor arguments to any parser (subparser or flat)."""
    p.add_argument("config_path", nargs="?", default=None,
                   help="Path to the site config JSON (optional; default: the "
                        "GRIME AI settings folder). May also be given as --config.")
    p.add_argument("--config", dest="config", default=None,
                   help="Path to the site config JSON (alternative to the positional path)")
    p.add_argument("--show", action="store_true",
                   help="Print the current editable values and exit")
    p.add_argument("--gui", action="store_true",
                   help="Open the graphical editor instead of editing on the command line")

    for flag, key, kind, help_text in _PARAMS:
        dest = _dest(flag)
        if kind == "float_list":
            p.add_argument(flag, dest=dest, action="append", type=float,
                           default=None, help=help_text)
        elif kind == "float":
            p.add_argument(flag, dest=dest, type=float, default=None, help=help_text)
        elif kind == "int":
            p.add_argument(flag, dest=dest, type=int, default=None, help=help_text)
        elif kind == "bool":
            p.add_argument(flag, dest=dest, type=_str2bool, default=None,
                           metavar="{true,false}", help=help_text)
        else:  # str
            p.add_argument(flag, dest=dest, type=str, default=None, help=help_text)

    p.add_argument("--link", dest="split_linked", action="store_const", const=True,
                   default=None, help="Mark the split as linked (complementary)")
    p.add_argument("--unlink", dest="split_linked", action="store_const", const=False,
                   help="Mark the split as unlinked (independent)")
    return p


def add_config_subparser(subparsers):
    """Register the `config` subcommand on an argparse subparsers object
    (used by main.py: `python -m GRIME_AI.main config --lr ...`)."""
    p = subparsers.add_parser(
        "config",
        help="Create/modify site config parameters (partial merge; --gui for the editor)",
    )
    return _add_config_arguments(p)


def _collect_updates(args):
    """Return {json_key: new_value} for exactly the flags the user supplied."""
    updates = {}
    for flag, key, kind, _ in _PARAMS:
        val = getattr(args, _dest(flag), None)
        if val is not None:
            updates[key] = val
    if getattr(args, "split_linked", None) is not None:
        updates["split_linked"] = args.split_linked
    return updates


def _fmt(v):
    return json.dumps(v, ensure_ascii=False)


def _resolve_path(args):
    return args.config or getattr(args, "config_path", None) or _default_config_path()


def run_config(args):
    """CLI entry point. Returns a process exit code."""
    # GUI requested from the command line -> hand off to the editor.
    if getattr(args, "gui", False):
        return open_editor(path=(args.config or getattr(args, "config_path", None)))

    path = _resolve_path(args)

    if not os.path.isfile(path):
        print(f"[ERROR] site config not found: {path}", file=sys.stderr)
        print("        This tool modifies an existing config. Create one from "
              "the Training tab (or the --gui editor's Save As) first.", file=sys.stderr)
        return 1

    cfg, eol = _load(path)

    if args.show:
        print(f"[GRIME AI] site config: {path}")
        for _, key, _, _ in _PARAMS:
            if key in cfg:
                print(f"  {key:28s} = {_fmt(cfg[key])}")
        if "split_linked" in cfg:
            print(f"  {'split_linked':28s} = {_fmt(cfg['split_linked'])}")
        return 0

    updates = _collect_updates(args)
    if not updates:
        print("[GRIME AI] No parameters supplied — nothing changed. "
              "Use --show to view, --gui to edit visually, or --help for options.")
        return 0

    before = copy.deepcopy(cfg)
    cfg.update(updates)

    try:
        _validate_split(cfg)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print("        No changes were written.", file=sys.stderr)
        return 1

    changed = [(k, before.get(k, "<unset>"), cfg[k])
               for k in updates if before.get(k, "<unset>") != cfg[k]]
    _write(path, cfg, eol)

    print(f"[GRIME AI] Updated {path}")
    if changed:
        for k, old, new in changed:
            print(f"  {k}: {_fmt(old)} -> {_fmt(new)}")
    else:
        print("  (values already matched; file rewritten unchanged)")
    return 0


# ============================================================================
# GUI — lazy (Qt imported only when the editor is opened)
# ============================================================================

def _parse_float_list(text):
    """'0.003, 0.001' -> [0.003, 0.001]. Accepts commas and/or whitespace."""
    parts = [p for chunk in str(text).split(",") for p in chunk.split()]
    return [float(p) for p in parts if p]


def _fmt_float_list(value):
    if isinstance(value, (list, tuple)):
        return ", ".join(repr(float(v)) for v in value)
    if value in (None, ""):
        return ""
    return str(value)


_EDITOR_CLASS = None


def _get_editor_class():
    """Define (once) and return the SiteConfigEditor QDialog. Qt is imported
    here, not at module load, so the CLI never depends on PyQt5."""
    global _EDITOR_CLASS
    if _EDITOR_CLASS is not None:
        return _EDITOR_CLASS

    from PyQt5.QtWidgets import (
        QDialog, QWidget, QLabel, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox,
        QPushButton, QFormLayout, QVBoxLayout, QHBoxLayout, QScrollArea,
        QFileDialog, QMessageBox, QFrame,
    )
    from PyQt5.QtCore import Qt

    _JSON_FILTER = "Site config (*.json);;All files (*)"

    class SiteConfigEditor(QDialog):
        def __init__(self, path=None, parent=None):
            super().__init__(parent)
            self.setWindowTitle("GRIME AI — Site Config Editor")
            self.setModal(False)
            self.resize(560, 640)
            self._cfg = {}
            self._eol = "\n"
            self._path = None
            self._widgets = {}
            self._build_ui()

            start = path
            if start is None:
                try:
                    cand = _default_config_path()
                    if cand and os.path.isfile(cand):
                        start = cand
                except Exception:
                    start = None
            if start and os.path.isfile(start):
                self._load_path(start)
            else:
                self._refresh_path_label()

        # ---- UI ----
        def _build_ui(self):
            outer = QVBoxLayout(self)
            top = QHBoxLayout()
            self._path_label = QLabel("No file loaded")
            self._path_label.setStyleSheet("color: #555;")
            self._path_label.setWordWrap(True)
            btn_open = QPushButton("Open\u2026")
            btn_open.clicked.connect(self._on_open)
            top.addWidget(self._path_label, 1)
            top.addWidget(btn_open, 0)
            outer.addLayout(top)

            line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)
            outer.addWidget(line)

            host = QWidget()
            self._form = QFormLayout(host)
            self._form.setLabelAlignment(Qt.AlignRight)
            for flag, key, kind, help_text in _PARAMS:
                w = self._make_widget(kind)
                w.setToolTip(help_text)
                self._widgets[key] = w
                self._form.addRow(QLabel(f"{key.replace('_', ' ')}  ({flag})"), w)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(host)
            outer.addWidget(scroll, 1)

            btns = QHBoxLayout()
            btn_save = QPushButton("Save")
            btn_save_as = QPushButton("Save As\u2026")
            btn_close = QPushButton("Close")
            btn_save.clicked.connect(self._on_save)
            btn_save_as.clicked.connect(self._on_save_as)
            btn_close.clicked.connect(self.close)
            for b in (btn_save, btn_save_as):
                b.setStyleSheet("QPushButton {background-color: steelblue; color: white; padding: 4px 12px;}")
            btns.addStretch(1)
            btns.addWidget(btn_save)
            btns.addWidget(btn_save_as)
            btns.addWidget(btn_close)
            outer.addLayout(btns)

        @staticmethod
        def _make_widget(kind):
            if kind == "int":
                w = QSpinBox(); w.setRange(0, 1_000_000); return w
            if kind == "float":
                w = QDoubleSpinBox(); w.setDecimals(6); w.setRange(0.0, 1_000_000.0)
                w.setSingleStep(0.001); return w
            if kind == "bool":
                return QCheckBox()
            return QLineEdit()

        # ---- load / populate ----
        def _load_path(self, path):
            try:
                cfg, eol = _load(path)
            except Exception as e:
                QMessageBox.critical(self, "Open", f"Could not read:\n{path}\n\n{e}")
                return
            self._cfg, self._eol, self._path = cfg, eol, path
            self._populate_from_cfg()
            self._refresh_path_label()

        def _populate_from_cfg(self):
            for flag, key, kind, _ in _PARAMS:
                w = self._widgets[key]
                present = key in self._cfg
                val = self._cfg.get(key)
                if kind == "int":
                    w.setValue(int(val) if present and val is not None else 0)
                elif kind == "float":
                    w.setValue(float(val) if present and val is not None else 0.0)
                elif kind == "bool":
                    w.setChecked(bool(val) if present else False)
                elif kind == "float_list":
                    w.setText(_fmt_float_list(val) if present else "")
                else:
                    w.setText("" if not present or val is None else str(val))

        def _refresh_path_label(self):
            self._path_label.setText(self._path if self._path else "No file loaded (Open… to begin)")

        # ---- collect ----
        def _apply_to_cfg(self):
            for flag, key, kind, _ in _PARAMS:
                w = self._widgets[key]
                if kind == "int":
                    self._cfg[key] = int(w.value())
                elif kind == "float":
                    self._cfg[key] = float(w.value())
                elif kind == "bool":
                    self._cfg[key] = bool(w.isChecked())
                elif kind == "float_list":
                    text = w.text().strip()
                    if text:
                        self._cfg[key] = _parse_float_list(text)
                else:
                    self._cfg[key] = w.text()

        # ---- actions ----
        def _on_open(self):
            start_dir = os.path.dirname(self._path) if self._path else ""
            path, _ = QFileDialog.getOpenFileName(self, "Open site config", start_dir, _JSON_FILTER)
            if path:
                self._load_path(path)

        def _save_to(self, path):
            try:
                self._apply_to_cfg()
            except ValueError as e:
                QMessageBox.warning(self, "Invalid value", str(e)); return False
            try:
                _validate_split(self._cfg)
            except ValueError as e:
                QMessageBox.warning(self, "Invalid split", str(e)); return False
            try:
                _write(path, self._cfg, self._eol)
            except Exception as e:
                QMessageBox.critical(self, "Save", f"Could not write:\n{path}\n\n{e}"); return False
            self._path = path
            self._refresh_path_label()
            return True

        def _on_save(self):
            if not self._path:
                self._on_save_as(); return
            if self._save_to(self._path):
                QMessageBox.information(self, "Saved", f"Saved:\n{self._path}")

        def _on_save_as(self):
            start_dir = os.path.dirname(self._path) if self._path else ""
            suggested = os.path.join(start_dir, "site_config.json") if start_dir else "site_config.json"
            path, _ = QFileDialog.getSaveFileName(self, "Save site config as", suggested, _JSON_FILTER)
            if not path:
                return
            if not path.lower().endswith(".json"):
                path += ".json"
            if self._save_to(path):
                QMessageBox.information(self, "Saved", f"Saved:\n{path}")

    _EDITOR_CLASS = SiteConfigEditor
    return _EDITOR_CLASS


def open_editor(parent=None, path=None):
    """Open the GUI editor. Reuses an existing QApplication (when called from
    inside GRIME AI) or creates one (standalone). Returns an exit code."""
    from PyQt5.QtWidgets import QApplication
    cls = _get_editor_class()
    existing = QApplication.instance()
    if existing is not None:
        cls(path=path, parent=parent).exec_()
        return 0
    app = QApplication(sys.argv)
    dlg = cls(path=path)
    dlg.show()
    return app.exec_()


# ============================================================================
# Entry point — one script, both front-ends
# ============================================================================

def main(argv=None):
    raw = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(
        description="Edit a GRIME AI site config JSON — CLI flags to edit in "
                    "place, or --gui (or no args) for the visual editor."
    )
    _add_config_arguments(parser)
    args = parser.parse_args(raw)

    # No args at all -> open the editor (friendly default for a bare launch).
    if not raw:
        args.gui = True

    return run_config(args)


if __name__ == "__main__":
    sys.exit(main())
