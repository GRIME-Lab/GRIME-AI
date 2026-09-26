#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# plugins.py
#
# Discovery and loading of drop-in plugins from <user root>/plugins/, shared by
# the Tools > Plugins menu and the ML Image Processing dialog.
#
# A plugin is either a .py file directly in the plugins folder, or a subfolder
# holding its .py together with its .ui files, data and anything else it needs.
# A folder keeps a plugin's files together, so installing is copying one folder
# and uninstalling is deleting it, and two plugins cannot collide over a file
# name. In a folder, the plugin's .py is the one named like the folder, else
# plugin.py, else the only .py present; the folder is on sys.path while it is
# imported, so a plugin can import its own modules.
#
# The .py file has a module-level PLUGIN dict:
#     PLUGIN = {
#         "title":   "Image Timestamp Sidecar",   # menu item or tab label
#         "class":   "MyWidgetClass",             # QWidget class in this file
#         "ui":      "my_tab.ui" or None,         # optional, beside the .py
#         "post":    ["wire_connections"],        # methods called on the widget
#         "surface": "tools",                     # "tools", "ml", or both
#         "api_version": 2,
#     }
#
# surface says where the plugin appears:
#     "ml"                 the ML Image Processing tab bar (the default when
#                          the key is missing, so older plugins are unaffected)
#     "tools"              the Tools > Plugins menu, in its own window
#     ["tools", "ml"]      both
#
# Modules are loaded by file path, so a plugin need not be installed; its own
# "from appcore..." imports still resolve. A bad plugin is skipped and logged,
# never taking the application down with it.

import os
import sys
import importlib.util
import traceback

from PyQt5.QtWidgets import QDialog, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi

from appcore.app_identity import PLUGINS_DIR

SUPPORTED_API_VERSIONS = (1, 2)
SURFACE_ML = "ml"
SURFACE_TOOLS = "tools"
DEFAULT_SURFACE = SURFACE_ML


class PluginInfo:
    """One discovered plugin: its file, its PLUGIN dict, and where it belongs."""

    def __init__(self, path, meta):
        self.path = path
        self.meta = meta
        self.title = meta.get("title") or os.path.splitext(os.path.basename(path))[0]
        surface = meta.get("surface", DEFAULT_SURFACE)
        self.surfaces = [s.lower() for s in ([surface] if isinstance(surface, str) else list(surface))]

    def __repr__(self):
        return f"<PluginInfo {self.title} surfaces={self.surfaces}>"


def plugins_folder() -> str:
    return str(PLUGINS_DIR)


def discover(surface=None) -> list:
    """
    Plugins in the plugins folder, sorted by title. With a surface, only the
    plugins that asked for it. Nothing is instantiated here, so this is cheap
    enough to call while building a menu.
    """
    folder = plugins_folder()
    found = []
    if not os.path.isdir(folder):
        return found

    for path in _candidate_files(folder):
        name = os.path.relpath(path, folder)
        try:
            meta = _read_plugin_meta(path)
        except Exception as err:
            print(f"[plugins] '{name}' unavailable: {type(err).__name__}: {err}", flush=True)
            traceback.print_exc()
            continue
        if meta is None:
            continue
        info = PluginInfo(path, meta)
        if surface is None or surface.lower() in info.surfaces:
            found.append(info)

    found.sort(key=lambda p: p.title.lower())
    return found


def load_widget(info: PluginInfo, parent=None):
    """
    Build the plugin's widget: import the module, construct its class, load its
    .ui if it has one, and call its post methods. Raises on failure; callers
    decide whether that is fatal (it never is, in the two callers here).
    """
    module = _import_module(info.path)
    meta = info.meta
    widget = getattr(module, meta["class"])(parent)

    ui_rel = meta.get("ui")
    if ui_rel:
        ui_file = os.path.join(os.path.dirname(info.path), ui_rel)
        if not os.path.exists(ui_file):
            raise FileNotFoundError(f"UI file not found: {ui_file}")
        loadUi(ui_file, widget)

    for method in meta.get("post", []):
        getattr(widget, method)()

    return widget


def open_in_window(info: PluginInfo, parent=None):
    """
    Show the plugin in its own resizable, non-modal window and return it, or
    None if the plugin could not be loaded. The window is kept on the parent so
    it is not garbage collected while open.
    """
    try:
        widget = load_widget(info, parent=None)
    except Exception as err:
        print(f"[plugins] '{info.title}' could not be opened: "
              f"{type(err).__name__}: {err}", flush=True)
        traceback.print_exc()
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(parent, "Plugin Failed",
                             f"{info.title} could not be opened:\n\n"
                             f"{type(err).__name__}: {err}")
        return None

    window = QDialog(parent)
    window.setWindowTitle(info.title)
    window.setWindowFlags(window.windowFlags() | Qt.WindowMinMaxButtonsHint)
    window.setAttribute(Qt.WA_DeleteOnClose, True)
    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget)
    window.resize(widget.sizeHint().expandedTo(widget.minimumSizeHint()))

    if parent is not None:
        # Hold a reference so the window survives; drop it when it closes.
        open_windows = getattr(parent, "_plugin_windows", None)
        if open_windows is None:
            open_windows = []
            parent._plugin_windows = open_windows
        open_windows.append(window)
        window.destroyed.connect(lambda _=None: open_windows.remove(window)
                                 if window in open_windows else None)

    window.show()
    return window


# ======================================================================================================================
# Internals
# ======================================================================================================================
def _candidate_files(folder) -> list:
    """Every plugin .py: loose files in the plugins folder, then one per subfolder."""
    paths = []
    for entry in sorted(os.listdir(folder)):
        if entry.startswith((".", "_")):
            continue
        full = os.path.join(folder, entry)
        if os.path.isfile(full) and entry.endswith(".py"):
            paths.append(full)
        elif os.path.isdir(full):
            main = _folder_main_file(full, entry)
            if main:
                paths.append(main)
    return paths


def _folder_main_file(folder, name):
    """
    The .py to load from a plugin folder: the one named like the folder, then
    plugin.py, then the only .py present. Anything else is ambiguous and skipped,
    so a plugin with several modules names its entry point one of those two ways.
    """
    try:
        scripts = [f for f in sorted(os.listdir(folder))
                   if f.endswith(".py") and not f.startswith(("_", "."))]
    except OSError:
        return None
    if not scripts:
        return None
    for preferred in (f"{name}.py", "plugin.py"):
        if preferred in scripts:
            return os.path.join(folder, preferred)
    if len(scripts) == 1:
        return os.path.join(folder, scripts[0])
    print(f"[plugins] '{name}' skipped: several .py files and none named "
          f"{name}.py or plugin.py.")
    return None


def _import_module(path):
    mod_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)

    # A plugin in its own folder can import its own modules by name.
    own_folder = os.path.dirname(path)
    added = own_folder not in sys.path
    if added:
        sys.path.insert(0, own_folder)
    try:
        spec.loader.exec_module(module)      # executes the plugin file
    finally:
        if added and own_folder in sys.path:
            sys.path.remove(own_folder)
    return module


def _read_plugin_meta(path):
    """The plugin's PLUGIN dict, or None when it has none or its API differs."""
    fname = os.path.basename(path)
    module = _import_module(path)
    meta = getattr(module, "PLUGIN", None)
    if not isinstance(meta, dict):
        print(f"[plugins] '{fname}' skipped (no PLUGIN dict).")
        return None
    if meta.get("api_version") not in SUPPORTED_API_VERSIONS:
        print(f"[plugins] '{fname}' skipped (api_version mismatch).")
        return None
    if not meta.get("class"):
        print(f"[plugins] '{fname}' skipped (no class named).")
        return None
    return meta
