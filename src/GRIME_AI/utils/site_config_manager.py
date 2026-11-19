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
      python site_config_manager.py <path> --lr 0.003 --weight-decay 0.01
      python site_config_manager.py <path> --lr 0.003 --lr 0.001   # sweep
      python site_config_manager.py <path> --show

  CLI (training datasets — same discovery rules as the Training tab):
      python site_config_manager.py <path> --images-root D:/sites --list-folders
      python site_config_manager.py <path> --images-root D:/sites --select-all
      python site_config_manager.py <path> --select Pecos/2023 --select Pecos/2024
      python site_config_manager.py <path> --deselect Pecos/2023

  GUI (visual editor; also does Save As under a new filename):
      python site_config_manager.py --gui
      python site_config_manager.py            # no args -> GUI
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

# Editable parameters, in the order and with the labels of the Training tab's
# Advanced tab: (flag, json_key, kind, help)
# kind in {"float", "int", "str", "bool", "float_list", "sci", "pct", "choice",
#          "modules"}
#   sci     - float shown in scientific notation (e.g. 1e-07)
#   pct     - fraction in the JSON, shown as a whole percent in the editor
#   choice  - one of _CHOICES[key]; the JSON stores the value, the editor shows
#             the Training tab's label for it
#   modules - list of LoRA target modules (subset of _LORA_MODULES)
_PARAMS = [
    # Training Parameters
    ("--site-name",        "siteName",                    "str",        "A name for this training run; trained models and their files are named with it"),
    ("--lr",               "learningRates",               "float_list", "Learning rate; repeat for one run per rate (e.g. --lr 1e-4 --lr 5e-5)"),
    ("--weight-decay",     "weight_decay",                "float",      "Weight decay"),
    ("--epochs",           "number_of_epochs",            "int",        "Number of epochs"),
    ("--batch-size",       "batch_size",                  "int",        "Batch size"),
    ("--early-stopping",   "early_stopping",              "bool",       "Stop when validation loss stops improving (true/false)"),
    ("--patience",         "patience",                    "int",        "Epochs without improvement before early stopping"),
    ("--backbone",         "backbone_size",               "choice",     "SegFormer backbone size (b0 to b5)"),
    ("--loss",             "loss_function",               "choice",     "SegFormer loss function"),
    ("--overlay-mode",     "validation_overlay_mode",     "choice",     "When validation overlays are written (last, every, interval)"),
    ("--overlay-interval", "validation_overlay_interval", "int",        "Write overlays every N epochs (interval mode only)"),
    ("--overlay-samples",  "validation_overlay_samples",  "int",        "Number of validation images per overlay pass"),
    ("--lr-scheduler",     "lr_scheduler_enabled",        "bool",       "Lower the learning rate when validation loss plateaus (true/false)"),
    ("--lr-factor",        "lr_scheduler_factor",         "float",      "Learning rate is multiplied by this value on each plateau"),
    ("--lr-patience",      "lr_scheduler_patience",       "int",        "Epochs without improvement before the learning rate is reduced"),
    ("--min-lr",           "lr_scheduler_min_lr",         "sci",        "Floor for the learning rate (e.g. 1e-7)"),
    ("--max-checkpoints",  "max_best_checkpoints",        "int",        "Number of best-scoring checkpoints to keep"),
    # Blob Filter
    ("--blob-radius",      "blob_filter_radius",          "float",      "Blob filter radius, as a fraction of the image diagonal"),
    # Training Splits
    ("--train-split",      "train_split",                 "pct",        "Training fraction, 0-1 (shown as a percent in the editor)"),
    ("--val-split",        "val_split",                   "pct",        "Validation fraction, 0-1 (shown as a percent in the editor)"),
    # YOLO
    ("--yolo-weights",     "yolo_base_weights",           "str",        "YOLO base-weights filename"),
    # LoRA Hyperparameters
    ("--use-lora",         "use_lora",                    "bool",       "Use LoRA; false runs a full fine-tune (true/false)"),
    ("--lora-rank",        "lora_rank",                   "int",        "LoRA rank r"),
    ("--lora-alpha",       "lora_alpha",                  "int",        "LoRA alpha"),
    ("--lora-dropout",     "lora_dropout",                "float",      "LoRA dropout"),
    ("--lora-bias",        "lora_bias",                   "choice",     "LoRA bias (none, all, lora_only)"),
    ("--lora-target",      "lora_target_modules",         "modules",    "LoRA target module; repeat for several (query, key, value)"),
]

# Section headings shown in the editor above the first parameter of each group,
# matching the Training tab's group boxes.
_GROUP_TITLES = {
    "siteName": "Training Parameters",
    "blob_filter_radius": "Blob Filter",
    "train_split": "Training Splits",
    "yolo_base_weights": "YOLO",
    "use_lora": "LoRA Hyperparameters",
}

_SPLIT_EPS = 1e-6

# ---------------------------------------------------------------------------
# Display labels: the Training tab's labels for each key.
# ---------------------------------------------------------------------------

_LABELS = {
    "siteName": "Model Name",
    "learningRates": "Learning Rates",
    "weight_decay": "Weight Decay",
    "number_of_epochs": "Number of Epochs",
    "batch_size": "Batch Size",
    "early_stopping": "Early Stopping",
    "patience": "Patience",
    "backbone_size": "Backbone Size",
    "loss_function": "Loss Function",
    "validation_overlay_mode": "Validation Overlays",
    "validation_overlay_interval": "Overlay Interval (epochs)",
    "validation_overlay_samples": "Overlay Samples",
    "lr_scheduler_enabled": "LR Scheduler",
    "lr_scheduler_factor": "LR Reduction Factor",
    "lr_scheduler_patience": "LR Scheduler Patience",
    "lr_scheduler_min_lr": "Minimum LR",
    "max_best_checkpoints": "Best Checkpoints to Keep",
    "blob_filter_radius": "Radius (fraction)",
    "train_split": "Train (%)",
    "val_split": "Validation (%)",
    "yolo_base_weights": "YOLO Base Weights",
    "use_lora": "Use LoRA",
    "lora_rank": "Rank r",
    "lora_alpha": "Alpha",
    "lora_dropout": "Dropout",
    "lora_bias": "Bias",
    "lora_target_modules": "Target Modules",
}


def _pretty_label(key):
    return _LABELS.get(key, key)


# LoRA hyperparameters other than the Use LoRA checkbox; grayed out when Use
# LoRA is off, as in the Training tab (a full fine-tune ignores them).
_LORA_PARAM_KEYS = ("lora_rank", "lora_alpha", "lora_dropout", "lora_bias", "lora_target_modules")

# LR scheduler settings; grayed out when the scheduler is off.
_LR_SCHED_KEYS = ("lr_scheduler_factor", "lr_scheduler_patience", "lr_scheduler_min_lr")

# ---------------------------------------------------------------------------
# Constrained values: (stored JSON value, Training tab label). The first entry
# is used when a config holds no value or an unknown one.
# ---------------------------------------------------------------------------

_CHOICES = {
    "backbone_size": [(b, b) for b in ("b0", "b1", "b2", "b3", "b4", "b5")],
    "loss_function": [(v, v) for v in ("ce_dice", "ce_focal", "ce_tversky",
                                       "focal", "tversky", "dice")],
    "validation_overlay_mode": [("last", "Last epoch only"),
                                ("every", "Every epoch"),
                                ("interval", "Every N epochs")],
    "lora_bias": [("none", "None"), ("all", "All"), ("lora_only", "Lora Only")],
}

_LORA_MODULES = [("query", "Query"), ("key", "Key"), ("value", "Value")]

# Spin box limits and steps, matching the Training tab's widgets:
# key -> (minimum, maximum, decimals, step); decimals/step only for floats.
_RANGES = {
    "weight_decay": (0.0, 10.0, 4, 0.001),
    "number_of_epochs": (1, 10000, None, None),
    "batch_size": (1, 1024, None, None),
    "patience": (1, 100, None, None),
    "validation_overlay_interval": (1, 10000, None, None),
    "validation_overlay_samples": (1, 100, None, None),
    "lr_scheduler_factor": (0.01, 0.99, 2, 0.05),
    "lr_scheduler_patience": (1, 100, None, None),
    "max_best_checkpoints": (1, 50, None, None),
    "lora_dropout": (0.0, 1.0, 3, 0.001),
}


def _choice_values(key):
    return [v for v, _ in _CHOICES[key]]


def _default(key):
    """Default for a key from the app's single source of defaults, or None when
    the app package is not importable (standalone use)."""
    try:
        from appcore.dialogs.ML_image_processing.model_config_manager import ModelConfigManager
        return ModelConfigManager.get_default(key)
    except Exception:
        return None


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
    from appcore.Save_Utils import Save_Utils
    settings_folder = Save_Utils().get_settings_folder()
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
# CORE — training dataset discovery (stdlib only)
#
# Mirrors the Training tab's folder validation so the editor and the tab agree
# on what a "valid training folder" is: a directory holding instances_default.json
# plus every image that JSON references.
# ============================================================================

_ANNOTATION_FILENAME = "instances_default.json"
_IMAGE_EXTS = (".jpg", ".jpeg")
_FORBIDDEN_ROOT_PARTS = ("anaconda3", "miniconda3", "programdata", "windows")


def check_training_folder(folder):
    """Validate one folder as a training dataset.

    Returns (is_valid, missing, json_path, orphan_annotations, unannotated).
      missing    - images listed in the JSON but absent on disk (hard error)
      orphans    - annotations whose image_id has no image entry (hard error)
      unannotated- images on disk with no JSON entry (warning only)
    """
    folder = os.path.normpath(str(folder))
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return False, [], None, [], []

    jsons = [e.name for e in entries
             if e.is_file() and e.name.lower() == _ANNOTATION_FILENAME]
    jpgs = {e.name for e in entries
            if e.is_file() and e.name.lower().endswith(_IMAGE_EXTS)}

    if not jsons or not jpgs:
        return False, [], None, [], []

    path_json = os.path.join(folder, jsons[0])
    try:
        with open(path_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"Cannot parse {jsons[0]}: {e}"], path_json, [], []

    raw_images = data.get("images")
    if not isinstance(raw_images, list):
        return False, [f"'images' key missing or not a list in {jsons[0]}"], path_json, [], []

    expected_files = []
    valid_image_ids = set()
    for item in raw_images:
        if isinstance(item, dict):
            fname = item.get("file_name") or item.get("filename")
            if not fname:
                return False, [f"Missing 'file_name' in entry: {item}"], path_json, [], []
            expected_files.append(os.path.basename(str(fname).replace("\\", "/")))
            if "id" in item:
                valid_image_ids.add(item["id"])
        elif isinstance(item, str):
            expected_files.append(os.path.basename(item.replace("\\", "/")))
        else:
            return False, [f"Unsupported image entry type: {type(item)}"], path_json, [], []

    missing = [f for f in expected_files if f not in jpgs]

    orphan_annotations = [
        f"annotation id={ann.get('id', '?')}"
        for ann in data.get("annotations", [])
        if ann.get("image_id") not in valid_image_ids
    ]

    expected_set = set(expected_files)
    unannotated = sorted(f for f in jpgs if f not in expected_set)

    return (not missing and not orphan_annotations), missing, path_json, orphan_annotations, unannotated


def _iter_dirs(root):
    """Recursively yield every subdirectory under root, skipping system trees."""
    if any(b in str(root).lower() for b in _FORBIDDEN_ROOT_PARTS):
        return
    if not os.path.isdir(root):
        return
    try:
        entries = list(os.scandir(root))
    except OSError:
        return
    for entry in entries:
        if entry.is_dir():
            yield entry.path
            yield from _iter_dirs(entry.path)


def scan_training_folders(root):
    """Recurse `root` and classify every folder that looks like a dataset.

    Returns (valid, incomplete):
      valid      - sorted list of folder names relative to root ('.' for the root itself)
      incomplete - {relative_name: (missing, orphans, unannotated, json_path)}
    """
    root = os.path.normpath(os.path.abspath(str(root)))
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Not a directory: {root}")
    if any(f in root.lower() for f in _FORBIDDEN_ROOT_PARTS):
        raise ValueError(f"Refusing to scan a system/Conda root: {root}")

    valid = []
    incomplete = {}

    def _rel(p):
        r = os.path.relpath(p, root)
        return "." if r == os.curdir else r

    for folder in [root] + list(_iter_dirs(root)):
        ok, missing, json_path, orphans, unannotated = check_training_folder(folder)
        if ok:
            valid.append(_rel(folder))
        elif missing or orphans:
            incomplete[_rel(folder)] = (missing, orphans, unannotated, json_path)

    return sorted(set(valid)), incomplete


def folder_details(root, rel_name):
    """Return (image_count, categories) for one dataset folder, for display."""
    folder = os.path.normpath(os.path.join(str(root), str(rel_name)))
    ann = os.path.join(folder, _ANNOTATION_FILENAME)
    if not os.path.isfile(ann):
        return 0, []
    try:
        with open(ann, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return 0, []
    cats = [c for c in data.get("categories", []) if isinstance(c, dict)]
    return len(data.get("images", [])), cats


def load_categories(root, rel_name):
    """Categories from one folder's annotation file, sorted by id.

    Returns None when the file is missing or unparseable, matching the
    Training tab's _load_categories contract.
    """
    ann = os.path.join(os.path.normpath(str(root)), str(rel_name), _ANNOTATION_FILENAME)
    if not os.path.isfile(ann):
        return None
    try:
        with open(ann, "r", encoding="utf-8") as f:
            data = json.load(f)
        cats = [c for c in data.get("categories", []) if isinstance(c, dict) and "id" in c]
        return sorted(cats, key=lambda c: c["id"])
    except Exception:
        return None


def check_label_consistency(root, selected):
    """Compare every selected folder against the canonical one.

    Port of the Training tab's _check_label_consistency. The canonical ("gold")
    dataset is simply the first entry in `selected`; drop it and the next entry
    inherits the role, which is why the selected list must preserve insertion
    order rather than sorting.

    Returns {folder: status} where status is one of:
        'gold'       this IS the canonical dataset
        'ok'         schema matches the canonical exactly
        'yellow'     subset of the canonical (missing categories, no conflicts)
        'red'        id/name mismatch against the canonical
        'unreadable' annotation file missing or unparseable
    """
    selected = list(selected)
    if not selected:
        return {}

    gold_name = selected[0]
    gold_cats = load_categories(root, gold_name)
    if gold_cats is None:
        return {name: "unreadable" for name in selected}

    gold_by_name = {c["name"]: c["id"] for c in gold_cats}
    gold_schema = tuple((c["id"], c["name"]) for c in gold_cats)

    state = {gold_name: "gold"}
    for name in selected[1:]:
        cats = load_categories(root, name)
        if cats is None:
            state[name] = "unreadable"
            continue
        if tuple((c["id"], c["name"]) for c in cats) == gold_schema:
            state[name] = "ok"
            continue

        conflict = False
        for c in cats:
            if c["name"] in gold_by_name and gold_by_name[c["name"]] != c["id"]:
                conflict = True
                break
            for gc in gold_cats:
                if gc["id"] == c["id"] and gc["name"] != c["name"]:
                    conflict = True
                    break
            if conflict:
                break

        state[name] = "red" if conflict else "yellow"

    return state


def collect_training_labels(root, selected):
    """Union of annotation categories across the selected dataset folders.

    Label strings use the Training tab's exact format, 'id - name', so a value
    written here parses identically in get_selected_training_labels().

    Returns (labels, coverage, conflicts):
      labels    - sorted 'id - name' strings, ordered by numeric id
      coverage  - {label: set(folders containing it)}
      conflicts - human-readable id/name collisions across folders
    """
    coverage = {}
    ids_by_name = {}
    names_by_id = {}

    for rel in selected:
        _n_images, cats = folder_details(root, rel)
        for cat in cats:
            cid = cat.get("id")
            cname = str(cat.get("name", "")).strip()
            if cid is None or not cname:
                continue
            coverage.setdefault(f"{cid} - {cname}", set()).add(rel)
            ids_by_name.setdefault(cname, set()).add(cid)
            names_by_id.setdefault(cid, set()).add(cname)

    conflicts = []
    for name, ids in sorted(ids_by_name.items()):
        if len(ids) > 1:
            conflicts.append(f"'{name}' is assigned conflicting IDs {sorted(ids)}")
    for cid, names in sorted(names_by_id.items()):
        if len(names) > 1:
            conflicts.append(f"ID {cid} maps to multiple names {sorted(names)}")

    def _sort_key(lbl):
        head = lbl.split(" - ", 1)[0]
        return (0, int(head)) if head.lstrip("-").isdigit() else (1, head)

    return sorted(coverage, key=_sort_key), coverage, conflicts


def parse_label(text):
    """'1 - water' -> ('1', 'water'). Returns None when malformed."""
    text = str(text or "").strip()
    if " - " not in text:
        return None
    lid, lname = map(str.strip, text.split(" - ", 1))
    return (lid, lname) if lid and lname else None


def get_training_categories(cfg):
    """Read train_model.TRAINING_CATEGORIES as an 'id - name' string, or ''."""
    cats = (cfg.get("train_model") or {}).get("TRAINING_CATEGORIES") or []
    if not isinstance(cats, list) or not cats:
        return ""
    first = cats[0]
    if isinstance(first, dict):
        lid = str(first.get("label_id", "")).strip()
        lname = str(first.get("label_name", "")).strip()
        return f"{lid} - {lname}" if lid and lname else ""
    return str(first).strip()  # tolerate a bare 'id - name' string


def set_training_categories(cfg, label_text):
    """Write train_model.TRAINING_CATEGORIES in the Training tab's shape."""
    train_model = cfg.setdefault("train_model", {})
    parsed = parse_label(label_text)
    train_model["TRAINING_CATEGORIES"] = (
        [{"label_id": parsed[0], "label_name": parsed[1]}] if parsed else []
    )
    return train_model["TRAINING_CATEGORIES"]


def build_path_section(root, selected, site_name="custom"):
    """Build the site_config 'Path' section from a root + selected folder names.

    Byte-for-byte the same shape the Training tab writes, so a config saved
    here drops straight into the trainer.
    """
    root = os.path.normpath(str(root))
    folders, annotations = [], []
    for name in selected:
        folder = os.path.normpath(os.path.join(root, str(name)))
        folders.append(folder)
        annotations.append(os.path.normpath(os.path.join(folder, _ANNOTATION_FILENAME)))
    return [{
        "siteName": site_name,
        "directoryPaths": {"folders": folders, "annotations": annotations},
    }]


def _describe_incomplete(incomplete, limit=10):
    """Render the incomplete-folder report used by both front-ends."""
    lines = ["Folders with annotation issues:"]
    for fld, (missing, orphans, unannotated, json_path) in sorted(incomplete.items()):
        lines.append(f"\n{fld}")
        if json_path:
            lines.append(f"  Annotation file: {json_path}")
        if missing:
            lines.append(f"  Missing from disk ({len(missing)}):")
            lines += [f"    - {m}" for m in missing[:limit]]
            if len(missing) > limit:
                lines.append(f"    ... and {len(missing) - limit} more.")
        if orphans:
            lines.append(f"  Orphan annotations ({len(orphans)}) - image_id not in images list:")
            lines += [f"    - {a}" for a in orphans[:limit]]
            if len(orphans) > limit:
                lines.append(f"    ... and {len(orphans) - limit} more.")
        if unannotated:
            lines.append(f"  On-disk images with no JSON entry ({len(unannotated)}) - skipped during training:")
            lines += [f"    - {u}" for u in unannotated[:5]]
            if len(unannotated) > 5:
                lines.append(f"    ... and {len(unannotated) - 5} more.")
    return "\n".join(lines)


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
        elif kind in ("float", "sci", "pct"):
            p.add_argument(flag, dest=dest, type=float, default=None, help=help_text)
        elif kind == "modules":
            p.add_argument(flag, dest=dest, action="append", default=None,
                           choices=[v for v, _ in _LORA_MODULES], help=help_text)
        elif kind == "int":
            p.add_argument(flag, dest=dest, type=int, default=None, help=help_text)
        elif kind == "bool":
            p.add_argument(flag, dest=dest, type=_str2bool, default=None,
                           metavar="{true,false}", help=help_text)
        elif kind == "choice":
            p.add_argument(flag, dest=dest, type=str, default=None,
                           choices=_choice_values(key), help=help_text)
        else:  # str
            p.add_argument(flag, dest=dest, type=str, default=None, help=help_text)

    p.add_argument("--link", dest="split_linked", action="store_const", const=True,
                   default=None, help="Mark the split as linked (complementary)")
    p.add_argument("--unlink", dest="split_linked", action="store_const", const=False,
                   help="Mark the split as unlinked (independent)")

    # ---- training datasets -------------------------------------------------
    g = p.add_argument_group("training datasets")
    g.add_argument("--images-root", dest="images_root", default=None,
                   help="Root folder to recurse for training datasets "
                        "(sets segmentation_images_path)")
    g.add_argument("--scan", action="store_true",
                   help="Rescan the images root and refresh available_folders")
    g.add_argument("--list-folders", dest="list_folders", action="store_true",
                   help="Print the datasets found under the images root and exit")
    g.add_argument("--select", dest="select", action="append", default=None,
                   metavar="FOLDER",
                   help="Add a folder (relative to the images root) to selected_folders; repeatable")
    g.add_argument("--select-all", dest="select_all", action="store_true",
                   help="Select every valid dataset found under the images root")
    g.add_argument("--deselect", dest="deselect", action="append", default=None,
                   metavar="FOLDER", help="Remove a folder from selected_folders; repeatable")
    g.add_argument("--clear-selection", dest="clear_selection", action="store_true",
                   help="Empty selected_folders")
    g.add_argument("--list-labels", dest="list_labels", action="store_true",
                   help="Print the annotation categories in the selected folders and exit")
    g.add_argument("--label", dest="label", default=None, metavar="LABEL",
                   help="Training category, as 'id - name', a bare name, or a bare id "
                        "(sets train_model.TRAINING_CATEGORIES)")
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


def _apply_dataset_args(cfg, args):
    """Apply the training-dataset flags to cfg in place.

    Returns (changed, messages). Raises ValueError on a bad root.
    """
    msgs = []
    touched = False

    if args.images_root is not None:
        cfg["segmentation_images_path"] = os.path.normpath(os.path.abspath(args.images_root))
        touched = True

    root = cfg.get("segmentation_images_path", "")
    selected = [str(s) for s in cfg.get("selected_folders", [])]

    needs_scan = args.scan or args.select_all or args.list_folders or args.images_root is not None
    if needs_scan:
        if not root:
            raise ValueError("No images root set. Pass --images-root <folder> first.")
        valid, incomplete = scan_training_folders(root)
        cfg["available_folders"] = valid
        touched = True
        msgs.append(f"Scanned {root} -> {len(valid)} valid dataset folder(s).")
        if incomplete:
            msgs.append(_describe_incomplete(incomplete))
        if args.select_all:
            selected = list(valid)
        scanned = True
    else:
        valid = [str(s) for s in cfg.get("available_folders", [])]
        scanned = False

    for name in (args.select or []):
        name = os.path.normpath(name)
        if scanned:
            ok = name in valid
        else:
            # No rescan requested — validate the folder directly on disk.
            ok = name in valid or name in selected or (
                bool(root) and check_training_folder(os.path.join(root, name))[0])
        if not ok:
            raise ValueError(
                f"'{name}' is not a valid dataset folder under {root or '<unset>'}. "
                f"Use --list-folders to see the choices, or --scan to refresh."
            )
        if name not in selected:
            selected.append(name)

    released = []  # folders leaving the selection go back to available

    for name in (args.deselect or []):
        name = os.path.normpath(name)
        if name in selected:
            released.append(name)
        selected = [s for s in selected if s != name]

    if args.clear_selection:
        released.extend(selected)
        selected = []

    if (args.select or args.deselect or args.clear_selection or args.select_all):
        touched = True

    if args.label is not None:
        labels, _cov, _con = collect_training_labels(root, selected)
        wanted = str(args.label).strip()
        match = None
        for lbl in labels:
            lid, lname = parse_label(lbl)
            if wanted in (lbl, lname, lid) or wanted.lower() == lname.lower():
                match = lbl
                break
        if match is None:
            raise ValueError(
                f"'{wanted}' is not a category in the selected folders. "
                f"Available: {', '.join(labels) or '(none)'}. Use --list-labels."
            )
        set_training_categories(cfg, match)
        touched = True
        msgs.append(f"Training label set to '{match}'.")

    if touched:
        cfg["selected_folders"] = selected
        # Keep the two lists disjoint and complementary, like the Training tab's panes.
        pool = list(cfg.get("available_folders", [])) + released
        cfg["available_folders"] = sorted({f for f in pool if f not in selected})
        if root:
            cfg["Path"] = build_path_section(root, selected, cfg.get("siteName", "custom"))

    return touched, msgs


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

    if getattr(args, "list_folders", False):
        root = args.images_root or cfg.get("segmentation_images_path", "")
        if not root:
            print("[ERROR] No images root. Set one with --images-root <folder>.", file=sys.stderr)
            return 1
        try:
            valid, incomplete = scan_training_folders(root)
        except (OSError, ValueError) as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return 1
        sel_order = [str(s) for s in cfg.get("selected_folders", [])]
        sel = set(sel_order)
        canonical = sel_order[0] if sel_order else None
        print(f"[GRIME AI] datasets under {root}:")
        for name in valid:
            n_img, cats = folder_details(root, name)
            labels = ", ".join(c.get("name", "?") for c in cats) or "no categories"
            mark = "\u2605" if name == canonical else ("*" if name in sel else " ")
            print(f"  {mark} {name:50s} {n_img:5d} images  [{labels}]")
        print("  (* = selected, \u2605 = canonical)")
        if incomplete:
            print()
            print(_describe_incomplete(incomplete))
        return 0

    if getattr(args, "list_labels", False):
        root = args.images_root or cfg.get("segmentation_images_path", "")
        selected = [str(x) for x in cfg.get("selected_folders", [])]
        if not root or not selected:
            print("[ERROR] Need an images root and at least one selected folder.",
                  file=sys.stderr)
            return 1
        labels, coverage, conflicts = collect_training_labels(root, selected)
        current = get_training_categories(cfg)
        print(f"[GRIME AI] categories across {len(selected)} selected folder(s):")
        for lbl in labels:
            have = len(coverage.get(lbl, ()))
            mark = "*" if lbl == current else " "
            flag = "" if have == len(selected) else "   <-- missing from some folders"
            print(f"  {mark} {lbl:40s} {have}/{len(selected)} folder(s){flag}")
        print("  (* = currently selected)")

        state = check_label_consistency(root, selected)
        legend = {"gold": "CANONICAL", "ok": "matches", "yellow": "missing categories",
                  "red": "CONFLICT", "unreadable": "UNREADABLE"}
        print()
        print("Consistency vs the canonical (first selected) dataset:")
        for name in selected:
            st = state.get(name, "ok")
            star = "*" if st == "gold" else " "
            print(f"  {star} {name:44s} {legend.get(st, st)}")
        if conflicts:
            print()
            print("Conflicts:")
            for c in conflicts:
                print(f"  - {c}")
        return 0

    if args.show:
        print(f"[GRIME AI] site config: {path}")
        for _, key, _, _ in _PARAMS:
            if key in cfg:
                print(f"  {key:28s} = {_fmt(cfg[key])}")
        if "split_linked" in cfg:
            print(f"  {'split_linked':28s} = {_fmt(cfg['split_linked'])}")
        if "segmentation_images_path" in cfg:
            print(f"  {'segmentation_images_path':28s} = {_fmt(cfg['segmentation_images_path'])}")
        current_label = get_training_categories(cfg)
        if current_label:
            print(f"  {'TRAINING_CATEGORIES':28s} = {_fmt(current_label)}")
        for key in ("available_folders", "selected_folders"):
            if key in cfg:
                print(f"  {key:28s} = {len(cfg[key])} folder(s)")
                for name in cfg[key]:
                    print(f"      {name}")
        return 0

    before = copy.deepcopy(cfg)

    try:
        dataset_changed, dataset_msgs = _apply_dataset_args(cfg, args)
    except (OSError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print("        No changes were written.", file=sys.stderr)
        return 1

    updates = _collect_updates(args)
    if not updates and not dataset_changed:
        print("[GRIME AI] No parameters supplied — nothing changed. "
              "Use --show to view, --list-folders to inspect datasets, "
              "--gui to edit visually, or --help for options.")
        return 0

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
    for m in dataset_msgs:
        print(f"  {m}")
    if dataset_changed:
        for k in ("segmentation_images_path", "available_folders", "selected_folders"):
            if before.get(k, "<unset>") != cfg.get(k, "<unset>"):
                if k == "segmentation_images_path":
                    print(f"  {k}: {_fmt(before.get(k, '<unset>'))} -> {_fmt(cfg[k])}")
                else:
                    print(f"  {k}: {len(before.get(k, []) or [])} -> {len(cfg.get(k, []))} folder(s)")
        if cfg.get("selected_folders"):
            print("  selected:")
            for name in cfg["selected_folders"]:
                print(f"      {name}")
    if changed:
        for k, old, new in changed:
            print(f"  {k}: {_fmt(old)} -> {_fmt(new)}")
    elif not dataset_changed:
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
        QFileDialog, QMessageBox, QFrame, QGroupBox, QSplitter, QTreeWidget,
        QTreeWidgetItem, QAbstractItemView, QApplication, QComboBox,
    )
    from PyQt5.QtGui import QFont, QColor
    from PyQt5.QtCore import Qt

    _JSON_FILTER = "Site config (*.json);;All files (*)"

    # House button styles. Imported from the app when available so this dialog
    # matches the Training tab; falls back to equivalents when the module is run
    # standalone with the GRIME_AI package off the path.
    try:
        from appcore.CSS_Styles import (
            BUTTON_CSS_STEEL_BLUE as _BTN_CSS,
            BUTTON_CSS_RED_OUTLINE as _BTN_CSS_RED_OUTLINE,
        )
    except Exception:
        _BTN_CSS = ("QPushButton {background-color: steelblue; color: white; "
                    "padding: 4px 12px;}")
        _BTN_CSS_RED_OUTLINE = (
            "QPushButton {background: transparent; color: #c0392b; "
            "border: 1px solid #c0392b; border-radius: 3px; padding: 4px 12px;}"
            "QPushButton:hover {background: rgba(192, 57, 43, 0.08);}"
            "QPushButton:disabled {color: #b0b0b0; border-color: #d0d0d0;}"
        )

    # Rounded corners for the panel containers. Deliberately sets only borders,
    # radii, and title placement — no background or text colors — so the dialog
    # still follows the system palette under a dark theme.
    _PANEL_CSS = """
        QGroupBox {
            border: 1px solid #b0b0b0;
            border-radius: 6px;
            margin-top: 9px;
            padding-top: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 9px;
            padding: 0 4px;
        }
        QTreeWidget {
            border: 1px solid #b0b0b0;
            border-radius: 4px;
        }
    """

    # The platform default renders as a bare hairline against the form; give the
    # parameter pane a visible track, handle, and arrow buttons.
    _SCROLLBAR_CSS = """
        QScrollBar::handle:vertical {
            background: #a6a6a6;
            border-radius: 2px;
        }
        QScrollBar:vertical {
            width: 12px;
            background: #cccccc;
            margin: 12px 0 12px 0;  /* reserve space for top/bottom arrows */
        }
        QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
            width: 8px;
            height: 8px;
        }
        QScrollBar::add-line:vertical {
            height: 12px;
            subcontrol-position: bottom;
            subcontrol-origin: margin;
        }
        QScrollBar::sub-line:vertical {
            height: 12px;
            subcontrol-position: top;
            subcontrol-origin: margin;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
    """

    class SiteConfigEditor(QDialog):
        def __init__(self, path=None, parent=None):
            super().__init__(parent)
            self.setWindowTitle("GRIME AI — Site Config Editor")
            self.setModal(False)
            self.resize(1020, 720)
            self._cfg = {}
            self._eol = "\n"
            self._path = None
            self._widgets = {}
            self._last_scanned_root = None
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
            self.setStyleSheet(_PANEL_CSS)
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

            # ---- parameters (right column) ----
            host = QWidget()
            self._form = QFormLayout(host)
            self._form.setLabelAlignment(Qt.AlignRight)
            self._labels = {}
            for flag, key, kind, help_text in _PARAMS:
                if key in _GROUP_TITLES:
                    heading = QLabel(_GROUP_TITLES[key])
                    heading_font = QFont(heading.font())
                    heading_font.setBold(True)
                    heading.setFont(heading_font)
                    self._form.addRow(heading)

                w = self._make_widget(kind, key)
                w.setToolTip(help_text)
                self._widgets[key] = w

                label = QLabel(f"{_pretty_label(key)}  ({flag})")
                self._labels[key] = label
                self._form.addRow(label, w)

            # Dependent settings are grayed out as in the Training tab.
            self._widgets["validation_overlay_mode"].currentIndexChanged.connect(
                self._update_enabled_states)
            self._widgets["lr_scheduler_enabled"].toggled.connect(self._update_enabled_states)
            self._widgets["use_lora"].toggled.connect(self._update_enabled_states)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(host)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            scroll.verticalScrollBar().setStyleSheet(_SCROLLBAR_CSS)
            self._params_scroll = scroll

            params_box = QGroupBox("Training Parameters")
            params_layout = QVBoxLayout(params_box)
            params_layout.setContentsMargins(6, 6, 6, 6)
            params_layout.addWidget(scroll)

            # ---- datasets (left column) ----
            splitter = QSplitter(Qt.Horizontal)
            splitter.addWidget(self._build_dataset_panel())
            splitter.addWidget(params_box)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)
            splitter.setSizes([520, 480])
            outer.addWidget(splitter, 1)

            btns = QHBoxLayout()
            btn_save = QPushButton("Save")
            btn_save_as = QPushButton("Save As\u2026")
            btn_close = QPushButton("Close")
            btn_save.clicked.connect(self._on_save)
            btn_save_as.clicked.connect(self._on_save_as)
            btn_close.clicked.connect(self.close)
            for b in (btn_save, btn_save_as):
                b.setStyleSheet(_BTN_CSS)
            btns.addStretch(1)
            btns.addWidget(btn_save)
            btns.addWidget(btn_save_as)
            btns.addWidget(btn_close)
            outer.addLayout(btns)

        # ------------------------------------------------------------------
        # Dataset panel — root folder, available datasets, selected datasets
        # ------------------------------------------------------------------
        def _build_dataset_panel(self):
            box = QGroupBox("Training Datasets")
            col = QVBoxLayout(box)
            col.setContentsMargins(6, 6, 6, 6)
            col.setSpacing(6)

            # Root folder row
            root_row = QHBoxLayout()
            root_row.addWidget(QLabel("Root folder:"))
            self._root_edit = QLineEdit()
            self._root_edit.setPlaceholderText(
                "Folder to recurse for instances_default.json + images")
            self._root_edit.setToolTip("segmentation_images_path")
            self._root_edit.editingFinished.connect(self._on_root_committed)
            btn_browse = QPushButton("Browse")
            btn_browse.clicked.connect(self._on_browse_root)
            root_row.addWidget(self._root_edit, 1)
            root_row.addWidget(btn_browse, 0)
            col.addLayout(root_row)

            # Available
            self._avail_label = QLabel("Available Training Folders")
            col.addWidget(self._avail_label)
            self._avail_tree = self._make_tree()
            self._avail_tree.itemDoubleClicked.connect(
                lambda item, _c: self._move_items(self._avail_tree, self._sel_tree, [item]))
            col.addWidget(self._avail_tree, 1)

            # Transfer buttons
            xfer = QHBoxLayout()
            btn_rescan = QPushButton("Rescan")
            btn_add = QPushButton("Add \u25bc")
            btn_add_all = QPushButton("Add All \u25bc")
            btn_remove = QPushButton("\u25b2 Remove")
            btn_remove_all = QPushButton("\u25b2 Remove All")
            btn_rescan.setToolTip("Re-walk the root folder and rebuild the available list")
            btn_rescan.clicked.connect(lambda: self._scan_root(force=True))
            btn_add.clicked.connect(
                lambda: self._move_items(self._avail_tree, self._sel_tree,
                                         self._top_level_selection(self._avail_tree)))
            btn_add_all.clicked.connect(
                lambda: self._move_items(self._avail_tree, self._sel_tree,
                                         self._all_top_level(self._avail_tree)))
            btn_remove.clicked.connect(
                lambda: self._move_items(self._sel_tree, self._avail_tree,
                                         self._top_level_selection(self._sel_tree)))
            btn_remove_all.clicked.connect(
                lambda: self._move_items(self._sel_tree, self._avail_tree,
                                         self._all_top_level(self._sel_tree)))
            for b in (btn_rescan, btn_add, btn_add_all, btn_remove):
                b.setStyleSheet(_BTN_CSS)
            # Matches pushButton_reset in the Training tab: same operation
            # (return every selected folder to Available), same signal.
            btn_remove_all.setStyleSheet(_BTN_CSS_RED_OUTLINE)
            xfer.addWidget(btn_rescan)
            xfer.addStretch(1)
            for b in (btn_add, btn_add_all, btn_remove, btn_remove_all):
                xfer.addWidget(b)
            col.addLayout(xfer)

            # Selected
            self._sel_label = QLabel("Selected Training Folders")
            col.addWidget(self._sel_label)
            self._sel_tree = self._make_tree()
            self._sel_tree.itemDoubleClicked.connect(
                lambda item, _c: self._move_items(self._sel_tree, self._avail_tree, [item]))
            col.addWidget(self._sel_tree, 1)

            # Training label — the category SAM2 is trained against, drawn from
            # the annotation files of whatever is currently selected.
            label_row = QHBoxLayout()
            label_row.addWidget(QLabel("Training label:"))
            self._label_combo = QComboBox()
            self._label_combo.setToolTip(
                "Category to train on, read from instances_default.json in the "
                "selected folders. Saved as train_model.TRAINING_CATEGORIES.")
            self._label_combo.currentIndexChanged.connect(self._on_label_changed)
            label_row.addWidget(self._label_combo, 1)
            col.addLayout(label_row)

            self._label_status = QLabel("")
            self._label_status.setWordWrap(True)
            col.addWidget(self._label_status)

            self._refresh_list_labels()
            return box

        @staticmethod
        def _make_tree():
            t = QTreeWidget()
            t.setHeaderHidden(True)
            t.setRootIsDecorated(True)
            t.setUniformRowHeights(False)
            t.setSelectionMode(QAbstractItemView.ExtendedSelection)
            t.setMinimumHeight(140)
            return t

        # ---- tree helpers ----
        def _add_folder_to_tree(self, tree, folder_name):
            """Add a dataset as a top-level node with image count and labels beneath."""
            parent = QTreeWidgetItem(tree, [folder_name])
            # Display text may gain a ★ prefix; the canonical name lives in UserRole.
            parent.setData(0, Qt.UserRole, folder_name)
            parent.setFlags(parent.flags() | Qt.ItemIsSelectable)

            child_font = QFont()
            child_font.setItalic(True)

            root = self._root_edit.text().strip()
            n_images, cats = folder_details(root, folder_name) if root else (0, [])

            count_item = QTreeWidgetItem(parent, [f"Image count: {n_images}"])
            count_item.setFlags(Qt.ItemIsEnabled)
            count_item.setFont(0, child_font)

            for cat in cats:
                label = QTreeWidgetItem(parent, [f"{cat.get('name', '?')} (ID={cat.get('id', '?')})"])
                label.setFlags(Qt.ItemIsEnabled)
                label.setFont(0, child_font)

            tree.collapseItem(parent)
            return parent

        @staticmethod
        def _all_top_level(tree):
            root = tree.invisibleRootItem()
            return [root.child(i) for i in range(root.childCount())]

        @staticmethod
        def _top_level_selection(tree):
            root = tree.invisibleRootItem()
            top = {root.child(i) for i in range(root.childCount())}
            return [it for it in tree.selectedItems() if it in top]

        def _names_in(self, tree):
            return [(it.data(0, Qt.UserRole) or it.text(0))
                    for it in self._all_top_level(tree)]

        def _set_tree_names(self, tree, names, sort=True):
            """Rebuild a tree's contents.

            Available is sorted (it is a lookup list). Selected must NOT be:
            its first entry is the canonical dataset, so order is meaningful
            and user-controlled.
            """
            names = [str(n) for n in names]
            if sort:
                names = sorted(set(names))
            else:
                seen, ordered = set(), []
                for n in names:
                    if n not in seen:
                        seen.add(n)
                        ordered.append(n)
                names = ordered
            tree.clear()
            for name in names:
                self._add_folder_to_tree(tree, name)

        def _move_items(self, src, dst, items):
            # Only top-level dataset nodes move; the count/label children are inert.
            items = [it for it in (items or []) if it is not None and it.parent() is None]
            if not items:
                return
            moving = [(it.data(0, Qt.UserRole) or it.text(0)) for it in items]
            remaining = [n for n in self._names_in(src) if n not in moving]
            self._set_tree_names(src, remaining, sort=(src is self._avail_tree))
            self._set_tree_names(dst, self._names_in(dst) + moving,
                                 sort=(dst is self._avail_tree))
            self._refresh_list_labels()
            self._refresh_training_labels()

        # ---- canonical dataset / label consistency ----
        _FOLDER_COLORS = {
            "gold":       "black",
            "ok":         "black",
            "yellow":     (180, 120, 0),
            "red":        "red",
            "unreadable": "red",
        }

        def _apply_folder_colors(self):
            """Color the Selected tree against the canonical (first) dataset.

            Mirrors the Training tab: ★ and black on the canonical entry, black
            on an exact schema match, dark yellow on a subset, red on a conflict
            or unreadable annotation file. Individual label rows whose ID
            disagrees with the canonical turn red and their folder expands.
            """
            root = self._root_edit.text().strip()
            selected = self._names_in(self._sel_tree)
            state = check_label_consistency(root, selected) if (root and selected) else {}
            self._folder_state = state

            gold_cats = load_categories(root, selected[0]) if (root and selected) else None
            gold_by_name = {c["name"]: c["id"] for c in (gold_cats or [])}

            child_font = QFont()
            child_font.setItalic(True)

            for item in self._all_top_level(self._sel_tree):
                name = item.data(0, Qt.UserRole) or item.text(0)
                status = state.get(name, "ok")
                spec = self._FOLDER_COLORS.get(status, "black")
                color = QColor(*spec) if isinstance(spec, tuple) else QColor(spec)

                item.setText(0, f"\u2605 {name}" if status == "gold" else name)
                item.setForeground(0, color)
                item.setToolTip(0, self._status_tooltip(status, name))

                for j in range(item.childCount()):
                    child = item.child(j)
                    child.setFont(0, child_font)
                    child.setForeground(0, QColor("black"))
                    if status == "gold":
                        continue
                    parsed = self._parse_child_label(child.text(0))
                    if parsed is None:
                        continue
                    label_name, child_id = parsed
                    gold_id = gold_by_name.get(label_name)
                    if gold_id is not None and child_id != gold_id:
                        child.setForeground(0, QColor("red"))
                        item.setExpanded(True)

        @staticmethod
        def _parse_child_label(text):
            """'water (ID=2)' -> ('water', 2); None when it is not a label row."""
            parts = str(text).split(" (ID=")
            if len(parts) != 2:
                return None
            try:
                return parts[0].strip(), int(parts[1].rstrip(")"))
            except ValueError:
                return None

        @staticmethod
        def _status_tooltip(status, name):
            return {
                "gold": f"Canonical dataset — every other folder is compared against {name}.",
                "ok": "Category schema matches the canonical dataset exactly.",
                "yellow": "Subset of the canonical dataset: missing categories, but no conflicts.",
                "red": "Conflicts with the canonical dataset: a label ID or name disagrees.",
                "unreadable": "Annotation file is missing or could not be parsed.",
            }.get(status, "")

        def _refresh_list_labels(self):
            self._avail_label.setText(
                f"Available Training Folders  ({self._avail_tree.invisibleRootItem().childCount()})")
            self._sel_label.setText(
                f"Selected Training Folders  ({self._sel_tree.invisibleRootItem().childCount()})")

        # ---- training label ----
        def _refresh_training_labels(self, preferred=None):
            """Rebuild the label combo from the categories in the selected folders."""
            combo = self._label_combo
            keep = preferred if preferred is not None else combo.currentText().strip()

            root = self._root_edit.text().strip()
            selected = self._names_in(self._sel_tree)
            labels, coverage, conflicts = (
                collect_training_labels(root, selected) if (root and selected) else ([], {}, []))

            combo.blockSignals(True)
            combo.clear()
            combo.addItems(labels)
            for i, lbl in enumerate(labels):
                have = len(coverage.get(lbl, ()))
                combo.setItemData(
                    i, f"Present in {have} of {len(selected)} selected folder(s)",
                    Qt.ToolTipRole)
            if keep and keep in labels:
                combo.setCurrentIndex(labels.index(keep))
            combo.blockSignals(False)

            self._label_scan = (labels, coverage, conflicts, selected)
            self._apply_folder_colors()
            self._update_label_status(labels, coverage, conflicts, selected, keep)

        def _update_label_status(self, labels, coverage, conflicts, selected, previous=""):
            """Explain coverage gaps and id/name collisions under the combo."""
            if not selected:
                self._label_status.setText("Select at least one folder to list its labels.")
                self._label_status.setStyleSheet("color: gray;")
                return
            if not labels:
                self._label_status.setText(
                    "No categories found in the selected folders' annotation files.")
                self._label_status.setStyleSheet("color: #c0392b;")
                return

            msgs, severity = [], "ok"

            if previous and previous not in labels:
                msgs.append(f"Previous label '{previous}' is not in the current selection.")
                severity = "warn"

            current = self._label_combo.currentText().strip()
            have = len(coverage.get(current, ()))
            total = len(selected)
            if current and have < total:
                missing = sorted(set(selected) - coverage.get(current, set()))
                preview = ", ".join(missing[:3]) + ("…" if len(missing) > 3 else "")
                msgs.append(f"'{current}' is missing from {total - have} of {total} "
                            f"folder(s): {preview}")
                severity = "warn"
            elif current:
                msgs.append(f"'{current}' present in all {total} selected folder(s).")

            state = getattr(self, "_folder_state", {}) or {}
            if selected:
                msgs.insert(0, f"Canonical: {selected[0]}.")
            bad = sorted(n for n, st in state.items() if st in ("red", "unreadable"))
            subset = sorted(n for n, st in state.items() if st == "yellow")
            if subset:
                msgs.append(f"{len(subset)} folder(s) missing categories vs canonical: "
                            + ", ".join(subset[:3]) + ("…" if len(subset) > 3 else ""))
                severity = "warn"
            if bad:
                msgs.append(f"{len(bad)} folder(s) conflict with the canonical dataset: "
                            + ", ".join(bad[:3]) + ("…" if len(bad) > 3 else ""))
                severity = "error"

            if conflicts:
                msgs.append("Conflicts: " + "; ".join(conflicts))
                severity = "error"

            colors = {"ok": "gray", "warn": "#b8860b", "error": "#c0392b"}
            self._label_status.setText("  ".join(msgs))
            self._label_status.setStyleSheet(f"color: {colors[severity]};")

        def _on_label_changed(self, _index):
            # Selection changed within the existing list; no need to re-read disk.
            labels, coverage, conflicts, selected = getattr(
                self, "_label_scan", ([], {}, [], []))
            self._update_label_status(labels, coverage, conflicts, selected)

        # ---- root folder actions ----
        def _on_browse_root(self):
            start = self._root_edit.text().strip() or os.path.expanduser("~")
            folder = QFileDialog.getExistingDirectory(self, "Select training images root", start)
            if folder:
                self._root_edit.setText(os.path.normpath(folder))
                self._scan_root(force=True)

        def _on_root_committed(self):
            root = self._root_edit.text().strip()
            if root and os.path.normpath(root) != (self._last_scanned_root or ""):
                self._scan_root(force=False)

        def _scan_root(self, force=False):
            """Recurse the root folder and rebuild the available list."""
            raw = self._root_edit.text().strip()
            if not raw:
                QMessageBox.information(self, "Root Folder",
                                        "Set a root folder first (Browse).")
                return

            root = os.path.normpath(os.path.abspath(raw))
            changed_root = self._last_scanned_root is not None and root != self._last_scanned_root
            if changed_root and self._sel_tree.invisibleRootItem().childCount() > 0:
                reply = QMessageBox.question(
                    self, "Root Folder Changed",
                    "You are changing the root folder.\n\n"
                    "Would you like to clear your currently selected folders?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self._sel_tree.clear()

            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                valid, incomplete = scan_training_folders(root)
            except (OSError, ValueError) as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "Invalid Folder", str(e))
                return
            finally:
                if QApplication.overrideCursor() is not None:
                    QApplication.restoreOverrideCursor()

            self._root_edit.setText(root)
            self._last_scanned_root = root

            # Anything already selected stays selected; the rest becomes available.
            selected = [n for n in self._names_in(self._sel_tree) if n in valid]
            self._set_tree_names(self._sel_tree, selected, sort=False)
            self._set_tree_names(self._avail_tree, [n for n in valid if n not in selected])
            self._refresh_list_labels()
            self._refresh_training_labels()

            if not valid:
                QMessageBox.information(
                    self, "No Valid Training Sets",
                    "No folders were found containing a COCO JSON and all its images.")
            if incomplete:
                QMessageBox.information(self, "Incomplete Training Sets",
                                        _describe_incomplete(incomplete))

        @staticmethod
        def _make_widget(kind, key=None):
            lo, hi, decimals, step = _RANGES.get(key, (None, None, None, None))
            if kind == "int":
                w = QSpinBox()
                w.setRange(int(lo) if lo is not None else 0,
                           int(hi) if hi is not None else 1_000_000)
                return w
            if kind == "float":
                w = QDoubleSpinBox()
                w.setDecimals(decimals if decimals is not None else 6)
                w.setRange(float(lo) if lo is not None else 0.0,
                           float(hi) if hi is not None else 1_000_000.0)
                w.setSingleStep(step if step is not None else 0.001)
                return w
            if kind == "pct":
                w = QSpinBox(); w.setRange(0, 100); return w
            if kind == "bool":
                return QCheckBox()
            if kind == "choice":
                w = QComboBox()
                for value, text in _CHOICES[key]:
                    w.addItem(text, value)
                return w
            if kind == "modules":
                w = QWidget()
                col = QVBoxLayout(w)
                col.setContentsMargins(0, 0, 0, 0)
                col.setSpacing(2)
                w._boxes = {}
                for value, text in _LORA_MODULES:
                    box = QCheckBox(text)
                    w._boxes[value] = box
                    col.addWidget(box)
                return w
            return QLineEdit()

        def _update_enabled_states(self, *_args):
            """Gray out settings that do not apply, as the Training tab does."""
            def _set(keys, on):
                for k in keys:
                    self._widgets[k].setEnabled(on)
                    self._labels[k].setEnabled(on)

            mode = self._widgets["validation_overlay_mode"].currentData()
            _set(("validation_overlay_interval",), mode == "interval")
            _set(_LR_SCHED_KEYS, self._widgets["lr_scheduler_enabled"].isChecked())
            _set(_LORA_PARAM_KEYS, self._widgets["use_lora"].isChecked())

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
                # A key missing from the file shows the app's default for it.
                val = self._cfg.get(key)
                if val is None:
                    val = _default(key)
                if kind == "int":
                    w.setValue(int(val) if val is not None else w.minimum())
                elif kind == "float":
                    w.setValue(float(val) if val is not None else w.minimum())
                elif kind == "pct":
                    w.setValue(int(round(float(val) * 100)) if val is not None else 0)
                elif kind == "sci":
                    w.setText(f"{float(val):g}" if val is not None else "")
                elif kind == "bool":
                    w.setChecked(bool(val))
                elif kind == "float_list":
                    w.setText(_fmt_float_list(val) if val is not None else "")
                elif kind == "choice":
                    values = _choice_values(key)
                    wanted = str(val).strip().lower() if val is not None else ""
                    idx = next((i for i, v in enumerate(values) if v.lower() == wanted), 0)
                    w.setCurrentIndex(idx)
                elif kind == "modules":
                    chosen = {str(m).strip().lower() for m in (val or [])}
                    for value, box in w._boxes.items():
                        box.setChecked(value in chosen)
                else:
                    w.setText("" if val is None else str(val))

            self._update_enabled_states()
            self._populate_datasets_from_cfg()

        def _populate_datasets_from_cfg(self):
            """Restore the root folder and both folder lists from the config."""
            root = str(self._cfg.get("segmentation_images_path", "") or "")
            self._root_edit.setText(os.path.normpath(root) if root else "")
            self._last_scanned_root = os.path.normpath(os.path.abspath(root)) if root else None

            def _clean(seq):
                # Tolerate the Training tab's '★ ' prefix on saved names.
                return [str(p).lstrip("\u2605 ").strip() for p in (seq or []) if str(p).strip()]

            selected = _clean(self._cfg.get("selected_folders"))
            available = [n for n in _clean(self._cfg.get("available_folders")) if n not in selected]
            self._set_tree_names(self._sel_tree, selected, sort=False)
            self._set_tree_names(self._avail_tree, available)
            self._refresh_list_labels()
            self._refresh_training_labels(preferred=get_training_categories(self._cfg))

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
                elif kind == "pct":
                    self._cfg[key] = round(w.value() / 100.0, 2)
                elif kind == "sci":
                    text = w.text().strip()
                    if text:
                        self._cfg[key] = float(text)
                elif kind == "bool":
                    self._cfg[key] = bool(w.isChecked())
                elif kind == "choice":
                    self._cfg[key] = w.currentData()
                elif kind == "modules":
                    self._cfg[key] = [v for v, box in w._boxes.items() if box.isChecked()]
                elif kind == "float_list":
                    text = w.text().strip()
                    if text:
                        self._cfg[key] = _parse_float_list(text)
                else:
                    self._cfg[key] = w.text()
            self._apply_datasets_to_cfg()

        def _apply_datasets_to_cfg(self):
            """Write the root folder, both lists, and the trainer's Path section."""
            root = self._root_edit.text().strip()
            selected = self._names_in(self._sel_tree)
            self._cfg["segmentation_images_path"] = os.path.normpath(root) if root else ""
            self._cfg["available_folders"] = self._names_in(self._avail_tree)
            self._cfg["selected_folders"] = selected
            set_training_categories(self._cfg, self._label_combo.currentText())
            if root:
                self._cfg["Path"] = build_path_section(
                    root, selected, self._cfg.get("siteName", "custom"))

        def _confirm_datasets(self):
            """Warn (but do not block) on an empty or stale selection."""
            root = self._cfg.get("segmentation_images_path", "")
            selected = self._cfg.get("selected_folders", [])

            if not selected:
                return QMessageBox.question(
                    self, "No Training Folders Selected",
                    "No training folders are selected, so this config cannot start a "
                    "training run as-is.\n\nSave anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

            if not get_training_categories(self._cfg):
                if QMessageBox.question(
                        self, "No Training Label Selected",
                        "No training label is selected, so SAM2 has no target "
                        "category to train against.\n\nSave anyway?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No) != QMessageBox.Yes:
                    return False

            missing = [n for n in selected
                       if not os.path.isfile(os.path.join(root, n, _ANNOTATION_FILENAME))]
            if missing:
                preview = "\n".join(f"  - {m}" for m in missing[:10])
                more = f"\n  ... and {len(missing) - 10} more." if len(missing) > 10 else ""
                return QMessageBox.question(
                    self, "Missing Annotation Files",
                    f"{len(missing)} selected folder(s) no longer contain "
                    f"{_ANNOTATION_FILENAME}:\n\n{preview}{more}\n\nSave anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes
            return True

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
            if not self._confirm_datasets():
                return False
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
