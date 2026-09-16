#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

"""
theme.py  (GRIME_AI.utils.theme)

One place that knows whether GRIME AI is in dark mode, and the colors to use
in each theme.

    from GRIME_AI.utils import theme

    theme.is_dark()                        # current theme
    theme.color("text_secondary")          # a named color for the current theme
    theme.pick(light_css, dark_css)        # choose between two values
    theme.bind(widget, light_css, dark_css)
        # apply now, and again automatically whenever the theme is toggled
    theme.bind_ui(widget, dark_css)
        # same, keeping the widget's current (.ui) style sheet as the light one
    theme.on_change(callback)              # callback(is_dark) on every toggle

The main window calls theme.set_dark(True/False) when the user toggles the
theme. qdarkstyle only installs a style sheet (it does not change QPalette),
so widgets cannot detect dark mode from the palette; this module is the
source of truth instead.
"""

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication

_APP_PROPERTY = "grime_ai_dark_mode"

# Named colors: (light, dark). Dark values are chosen for the qdarkstyle
# background (#19232D) and text (#DFE1E2).
COLORS = {
    "window_bg":      ("white",     "#19232D"),
    "text":           ("black",     "#DFE1E2"),
    "text_secondary": ("#444444",   "#C9CED4"),
    "text_muted":     ("gray",      "#9DA9B5"),
    "text_path":      ("#555555",   "#AEB6BF"),
    "success":        ("green",     "#6CC070"),
    "warning":        ("#b8860b",   "#E0B040"),
    "error":          ("#c0392b",   "#FF6B5E"),
    "error_strong":   ("darkred",   "#FF7B72"),
    "accent":         ("steelblue", "#8FC1EC"),
    "accent_border":  ("steelblue", "#6FA8DC"),
}


class _Notifier(QObject):
    changed = pyqtSignal(bool)


_notifier = None


def notifier():
    global _notifier
    if _notifier is None:
        _notifier = _Notifier()
    return _notifier


def is_dark():
    app = QApplication.instance()
    return bool(app is not None and app.property(_APP_PROPERTY))


def set_dark(dark):
    """Called by the main window when the theme is toggled."""
    app = QApplication.instance()
    if app is not None:
        app.setProperty(_APP_PROPERTY, bool(dark))
    notifier().changed.emit(bool(dark))


def pick(light, dark):
    return dark if is_dark() else light


def color(name):
    light, dark = COLORS[name]
    return pick(light, dark)


def on_change(callback, owner=None):
    """Call callback(is_dark) on every theme toggle. If owner (a QObject) is
    given, the connection is dropped when the owner is destroyed."""
    sig = notifier().changed

    def _safe(dark):
        try:
            callback(dark)
        except RuntimeError:
            # the Qt object behind the callback was deleted
            _disconnect(sig, _safe)

    sig.connect(_safe)
    if owner is not None:
        owner.destroyed.connect(lambda *_: _disconnect(sig, _safe))
    return _safe


def bind(widget, light_css, dark_css):
    """Apply light_css or dark_css to widget now and on every theme toggle."""
    def apply(_dark=None):
        widget.setStyleSheet(dark_css if is_dark() else light_css)
    apply()
    on_change(apply, owner=widget)
    return apply


def bind_ui(widget, dark_css):
    """Like bind(), using the widget's current style sheet (e.g. from the .ui) as the light one."""
    return bind(widget, widget.styleSheet(), dark_css)


def _disconnect(sig, slot):
    try:
        sig.disconnect(slot)
    except (TypeError, RuntimeError):
        pass
