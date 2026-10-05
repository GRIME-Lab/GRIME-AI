#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRIME2 as a GRIME AI plugin, opened from Tools > Plugins.

Install layout, under ~/Documents/GRIME-AI/plugins/ :

    plugins/
        grime2/                 <- this plugin's folder
            grime2_plugin.py    <- this file, the only thing GRIME AI loads
            GRIME2/             <- the whole GRIME2 package, unmodified
                main.py  core.py  waterline.py  cli.py  gui.py  qtcompat.py
                algorithms/...

    The package folder may carry any name: every subfolder beside this file
    is searched for grime2py, so renaming GRIME2/ to bin/ changes nothing.

This file stays thin on purpose: it declares the PLUGIN dict, puts the
GRIME2 folder on sys.path, and hands back a widget. Everything else lives
in GRIME2/ and is the same code that runs standalone, so there is one
implementation, not a fork.

Qt binding: GRIME AI is PyQt5. grime2py.qtcompat detects the binding the
host has already imported and uses it, so nothing here forces a second
binding into the process.
"""
from __future__ import annotations

import os
import sys

PLUGIN = {
    "title": "GRIME2",
    "class": "GRIME2Tab",
    "description": "Stage measurement from images (the standalone GRIME2 application)",
    "ui": None,             # the widget builds its own children, no .ui file
    "post": [],             # nothing to call after construction
    "surface": "tools",     # Tools > Plugins, in its own window
    "api_version": 2,
}

# --- make the bundled GRIME2 package importable -----------------------
# Several plausible layouts are accepted, because "put it in a GRIME2
# subfolder" can reasonably mean either of the first two:
#
#   plugins/GRIME2/grime2py/core.py      <- package inside GRIME2
#   plugins/GRIME2/core.py               <- modules loose in GRIME2
#   plugins/grime2py/core.py             <- package beside the plugin file
#
# Whichever is found, the folder that CONTAINS the package goes on
# sys.path so "import grime2py" resolves.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SEARCHED = []


def _find_package_root() -> str:
    """Folder to add to sys.path so `import grime2py` works, or ""."""
    candidates = [
        os.path.join(_HERE, "GRIME2"),      # grime2/GRIME2/grime2py/
        _HERE,                              # grime2/grime2py/
    ]
    # Any other subfolder beside this file, so the package folder's name
    # (GRIME2, bin, whatever) does not have to be known here.
    try:
        candidates += [os.path.join(_HERE, name) for name in sorted(os.listdir(_HERE))
                       if os.path.isdir(os.path.join(_HERE, name))
                       and not name.startswith(("_", "."))]
    except OSError:
        pass
    for root in candidates:
        _SEARCHED.append(os.path.join(root, "grime2py", "__init__.py"))
        if os.path.isfile(os.path.join(root, "grime2py", "__init__.py")):
            return root

    # Modules sitting loose in the package folder: import that folder AS
    # grime2py by registering it under the expected name, so the package's
    # own "from grime2py.core import ..." statements still resolve.
    loose = next((root for root in candidates
                  if os.path.isfile(os.path.join(root, "core.py"))), "")
    if loose:
        _SEARCHED.append(os.path.join(loose, "core.py"))
        import importlib.util
        import types
        package = types.ModuleType("grime2py")
        package.__path__ = [loose]
        package.__spec__ = importlib.util.spec_from_loader("grime2py", loader=None,
                                                           is_package=True)
        sys.modules.setdefault("grime2py", package)
        sub = os.path.join(loose, "algorithms")
        if os.path.isdir(sub):
            algorithms = types.ModuleType("grime2py.algorithms")
            algorithms.__path__ = [sub]
            sys.modules.setdefault("grime2py.algorithms", algorithms)
        return loose
    return ""


_PACKAGE_ROOT = _find_package_root()
if _PACKAGE_ROOT and _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

try:
    from grime2py.qtcompat import QtCore, QtWidgets      # noqa: E402
except ImportError as _error:                            # noqa: N816
    # Re-raise with the paths that were tried. GRIME AI catches this and
    # prints it, so the console names the missing folder instead of just
    # saying "No module named grime2py".
    raise ImportError(
        "GRIME2 plugin could not find the grime2py package. Looked for:\n  "
        + "\n  ".join(_SEARCHED)
        + "\nExpected layout:\n"
          "  plugins/grime2_plugin.py\n"
          "  plugins/GRIME2/grime2py/...\n"
          "Original error: %s" % _error
    ) from _error


class GRIME2Tab(QtWidgets.QWidget):
    """Hosts the standalone GRIME2 window inside a GRIME AI plugin window.

    The GRIME2 UI is a QMainWindow because it owns a menu bar, a status bar
    and a filmstrip. A QMainWindow is perfectly legal as a child widget, so
    it is embedded whole rather than being taken apart. That keeps the
    standalone application and the plugin on exactly the same code path:
    a bug fixed in one is fixed in the other.

    Construction is deferred to showEvent, so nothing is built until the
    plugin window is actually shown.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._window = None
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._placeholder = QtWidgets.QLabel("Loading GRIME2...")
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._placeholder)

    # -- lifecycle ------------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        if self._window is None:
            self._build()

    def _build(self) -> None:
        try:
            from grime2py.gui import MainWindow
        except Exception as error:                      # noqa: BLE001
            # Never let a broken plugin take the host dialog down. GRIME AI
            # guards this too, but a readable message in the tab beats an
            # empty one.
            self._placeholder.setText(
                "GRIME2 could not be loaded:\n%s\n\n"
                "Check that the GRIME2 package folder sits beside this plugin file."
                % error)
            self._placeholder.setWordWrap(True)
            return
        self._window = MainWindow()
        # Embedded: no floating window, and no window-level chrome.
        self._window.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self._layout.removeWidget(self._placeholder)
        self._placeholder.hide()
        self._layout.addWidget(self._window)

    # -- convenience for host code -------------------------------------
    @property
    def window(self):
        """The embedded MainWindow, or None if it has not been built yet."""
        return self._window

    def load_folder(self, folder: str) -> None:
        """Point the tab at a folder of images. Lets GRIME AI drive GRIME2
        with the dataset the user already selected elsewhere in the app."""
        if self._window is None:
            self._build()
        if self._window is not None:
            self._window.load_folder(folder)
