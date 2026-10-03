"""Qt binding shim.

GRIME AI is PyQt5. GRIME2 standalone was written against PySide6. Two Qt
bindings cannot coexist in one process, so a plugin loaded into GRIME AI
must use the binding the host already imported.

This module picks one binding and normalises the two differences that
actually matter for this codebase:

  * QAction lives in QtWidgets under PyQt5 and in QtGui under PySide6.
    It is re-exported on QtGui either way, so callers write QtGui.QAction.
  * pyqtSignal vs Signal.

Every scoped enum this GUI uses (Qt.AlignmentFlag.AlignCenter,
QListView.ViewMode.IconMode, QKeySequence.StandardKey.Open, and the rest)
resolves identically under PyQt5 5.15 and PySide6, so nothing else needs
translating.

Selection order:
  1. whichever binding is ALREADY imported -- inside GRIME AI that is
     PyQt5, and importing the other one would crash the process
  2. PyQt5
  3. PySide6
Override with the GRIME2_QT_BINDING environment variable.
"""
from __future__ import annotations

import os
import sys

BINDING = ""


def _load(name: str):
    if name == "PyQt5":
        from PyQt5 import QtCore, QtGui, QtWidgets
        # PyQt5 keeps QAction in QtWidgets; put it where PySide6 code expects.
        if not hasattr(QtGui, "QAction"):
            QtGui.QAction = QtWidgets.QAction
        QtCore.Signal = QtCore.pyqtSignal
        QtCore.Slot = QtCore.pyqtSlot
        return QtCore, QtGui, QtWidgets
    if name == "PySide6":
        from PySide6 import QtCore, QtGui, QtWidgets
        return QtCore, QtGui, QtWidgets
    raise ImportError(f"unknown Qt binding: {name}")


def _choose() -> str:
    forced = os.environ.get("GRIME2_QT_BINDING", "").strip()
    if forced:
        return forced
    # Never import a second binding into a process that already has one.
    for name in ("PyQt5", "PySide6"):
        if name in sys.modules:
            return name
    for name in ("PyQt5", "PySide6"):
        try:
            __import__(name)
            return name
        except ImportError:
            continue
    raise ImportError(
        "No Qt binding found. Install PyQt5 (to match GRIME AI) or PySide6."
    )


BINDING = _choose()
QtCore, QtGui, QtWidgets = _load(BINDING)


def exec_app(app) -> int:
    """QApplication.exec() across bindings (PyQt5 5.x prefers exec_)."""
    runner = getattr(app, "exec", None) or getattr(app, "exec_")
    return int(runner())
