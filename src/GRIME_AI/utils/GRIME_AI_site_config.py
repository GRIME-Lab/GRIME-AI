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

  CLI (training datasets — same discovery rules as the Training tab):
      python GRIME_AI_site_config.py <path> --images-root D:/sites --list-folders
      python GRIME_AI_site_config.py <path> --images-root D:/sites --select-all
      python GRIME_AI_site_config.py <path> --select Pecos/2023 --select Pecos/2024
      python GRIME_AI_site_config.py <path> --deselect Pecos/2023

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
    ("--site-name",      "siteName",                   "str",        "Site name"),
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
    ("--blob-mode",      "blob_radius_mode",           "choice",     "Blob radius mode (Computed or Manual)"),
    ("--num-clusters",   "num_clusters",               "int",        "Number of clusters"),
    ("--overlay-samples","validation_overlay_samples", "int",        "Validation overlay sample count"),
    ("--yolo-weights",   "yolo_base_weights",          "str",        "YOLO base-weights filename"),
    ("--use-lora",       "use_lora",                   "bool",       "Enable LoRA (true/false)"),
    ("--lora-rank",      "lora_rank",                  "int",        "LoRA rank"),
    ("--lora-alpha",     "lora_alpha",                 "int",        "LoRA alpha"),
    ("--lora-dropout",   "lora_dropout",               "float",      "LoRA dropout"),
    ("--lora-bias",      "lora_bias",                  "str",        "LoRA bias mode (none/all/lora_only)"),
]

_SPLIT_EPS = 1e-6

# ---------------------------------------------------------------------------
# Display labels
#
# Keys are snake_case or camelCase in the JSON; the UI shows them in prose with
# the acronyms cased the way they are written in the literature.
# ---------------------------------------------------------------------------

_ACRONYMS = {
    "lora": "LoRA",
    "yolo": "YOLO",
    "lr": "LR",
    "iou": "IoU",
    "gpu": "GPU",
    "cpu": "CPU",
}

# camelCase keys and anything the generic rule renders awkwardly
_LABEL_OVERRIDES = {
    "siteName": "Site name",
    "learningRates": "Learning rates",
    "num_clusters": "Number of clusters",
}


def _pretty_label(key):
    """'lora_rank' -> 'LoRA rank', 'yolo_base_weights' -> 'YOLO base weights'."""
    if key in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[key]
    words = [_ACRONYMS.get(w.lower(), w) for w in str(key).split("_")]
    if words and words[0] not in _ACRONYMS.values():
        words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Model gating
#
# SAM2 is the only training model currently supported, so the parameters that
# apply solely to the other backends are shown but locked. They stay visible so
# an existing config's values remain legible, and they are still written back
# untouched on save — nothing is silently dropped.
# ---------------------------------------------------------------------------

_LORA_KEYS = ("use_lora", "lora_rank", "lora_alpha", "lora_dropout", "lora_bias")
_YOLO_KEYS = ("yolo_base_weights",)
_MODEL_LOCKED_KEYS = frozenset(_LORA_KEYS + _YOLO_KEYS)

_MODEL_LOCK_REASON = (
    "Disabled — SAM2 is the only training model currently supported. "
    "LoRA applies to SegFormer and this setting applies to YOLO."
)
_LORA_LOCK_REASON = (
    "Disabled — SAM2 is the only training model currently supported; "
    "LoRA applies to SegFormer training."
)
_YOLO_LOCK_REASON = (
    "Disabled — SAM2 is the only training model currently supported; "
    "this setting applies to YOLO training."
)


def _lock_reason(key):
    return _LORA_LOCK_REASON if key in _LORA_KEYS else _YOLO_LOCK_REASON


# ---------------------------------------------------------------------------
# Constrained values
#
# Keys whose value is one of a fixed set. The first entry is the default and
# the value a config is normalized to when it is loaded.
# ---------------------------------------------------------------------------

_CHOICES = {
    "blob_radius_mode": ["Computed", "Manual"],
}

# Choice keys forced back to their default on load, regardless of the file.
_FORCED_CHOICES = ("blob_radius_mode",)


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
        elif kind == "float":
            p.add_argument(flag, dest=dest, type=float, default=None, help=help_text)
        elif kind == "int":
            p.add_argument(flag, dest=dest, type=int, default=None, help=help_text)
        elif kind == "bool":
            p.add_argument(flag, dest=dest, type=_str2bool, default=None,
                           metavar="{true,false}", help=help_text)
        elif kind == "choice":
            p.add_argument(flag, dest=dest, type=str, default=None,
                           choices=_CHOICES[key], help=help_text)
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
        sel = set(str(s) for s in cfg.get("selected_folders", []))
        print(f"[GRIME AI] datasets under {root}:")
        for name in valid:
            n_img, cats = folder_details(root, name)
            labels = ", ".join(c.get("name", "?") for c in cats) or "no categories"
            mark = "*" if name in sel else " "
            print(f"  {mark} {name:50s} {n_img:5d} images  [{labels}]")
        print("  (* = currently selected)")
        if incomplete:
            print()
            print(_describe_incomplete(incomplete))
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
    from PyQt5.QtGui import QFont
    from PyQt5.QtCore import Qt

    _JSON_FILTER = "Site config (*.json);;All files (*)"

    # House button styles. Imported from the app when available so this dialog
    # matches the Training tab; falls back to equivalents when the module is run
    # standalone with the GRIME_AI package off the path.
    try:
        from GRIME_AI.GRIME_AI_CSS_Styles import (
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
                w = self._make_widget(kind, key)
                w.setToolTip(help_text)
                self._widgets[key] = w

                label = QLabel(f"{_pretty_label(key)}  ({flag})")
                self._labels[key] = label

                if key in _MODEL_LOCKED_KEYS:
                    reason = _lock_reason(key)
                    for part in (label, w):
                        part.setEnabled(False)
                        part.setToolTip(reason)

                self._form.addRow(label, w)

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
            splitter.setSizes([580, 420])
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
            btn_browse = QPushButton("Browse\u2026")
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
            return [it.text(0) for it in self._all_top_level(tree)]

        def _set_tree_names(self, tree, names):
            tree.clear()
            for name in sorted(set(str(n) for n in names)):
                self._add_folder_to_tree(tree, name)

        def _move_items(self, src, dst, items):
            # Only top-level dataset nodes move; the count/label children are inert.
            items = [it for it in (items or []) if it is not None and it.parent() is None]
            if not items:
                return
            moving = [it.text(0) for it in items]
            remaining = [n for n in self._names_in(src) if n not in moving]
            self._set_tree_names(src, remaining)
            self._set_tree_names(dst, self._names_in(dst) + moving)
            self._refresh_list_labels()

        def _refresh_list_labels(self):
            self._avail_label.setText(
                f"Available Training Folders  ({self._avail_tree.invisibleRootItem().childCount()})")
            self._sel_label.setText(
                f"Selected Training Folders  ({self._sel_tree.invisibleRootItem().childCount()})")

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
                                        "Set a root folder first (Browse…).")
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
            self._set_tree_names(self._sel_tree, selected)
            self._set_tree_names(self._avail_tree, [n for n in valid if n not in selected])
            self._refresh_list_labels()

            if not valid:
                QMessageBox.information(
                    self, "No Valid Training Sets",
                    "No folders were found containing a COCO JSON and all its images.")
            if incomplete:
                QMessageBox.information(self, "Incomplete Training Sets",
                                        _describe_incomplete(incomplete))

        @staticmethod
        def _make_widget(kind, key=None):
            if kind == "int":
                w = QSpinBox(); w.setRange(0, 1_000_000); return w
            if kind == "float":
                w = QDoubleSpinBox(); w.setDecimals(6); w.setRange(0.0, 1_000_000.0)
                w.setSingleStep(0.001); return w
            if kind == "bool":
                return QCheckBox()
            if kind == "choice":
                w = QComboBox(); w.addItems(_CHOICES[key]); return w
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
                elif kind == "choice":
                    options = _CHOICES[key]
                    idx = next((i for i, o in enumerate(options)
                                if present and str(val).strip().lower() == o.lower()), 0)
                    w.setCurrentIndex(idx)
                else:
                    w.setText("" if not present or val is None else str(val))

            # SAM2-only: LoRA is never active, whatever the file said.
            self._widgets["use_lora"].setChecked(False)

            # Normalize forced choices back to their default (index 0).
            for key in _FORCED_CHOICES:
                self._widgets[key].setCurrentIndex(0)

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
            self._set_tree_names(self._sel_tree, selected)
            self._set_tree_names(self._avail_tree, available)
            self._refresh_list_labels()

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
                elif kind == "choice":
                    self._cfg[key] = w.currentText()
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
