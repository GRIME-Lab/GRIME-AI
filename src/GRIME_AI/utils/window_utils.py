#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

"""
Window placement helpers for application.

PROBLEM THIS SOLVES
-------------------
On Linux/X11 (notably HCC Open OnDemand and VNC desktops), the application main
window came up with its title bar tucked underneath the desktop panel/menu bar.
The user's only recourse was right-click -> Move -> drag the window down.

Root cause was in run_gui():

    frame.move(app.desktop().screen().rect().center() - frame.rect().center())

Three separate defects in that one line:

  1. screen().rect() is the FULL screen rectangle. It does not exclude the
     panel/dock/menu bar. Centering against it puts the top of the window
     under the panel whenever the window is tall.
  2. frame.rect() is the CLIENT rectangle -- it excludes the window manager's
     title bar. move() positions the client area, so the title bar lands
     roughly 25-40 px ABOVE the computed y. That is the part that disappears.
  3. If the window is taller than the screen, the centered y is NEGATIVE and
     the entire top of the window goes off-screen.

Additionally app.desktop() is deprecated in Qt5 and removed in Qt6.

THE FIX
-------
fix_window_placement() replaces that call. It works in this order:

    read work area -> SHRINK to fit -> re-read actual size -> position -> clamp

Shrinking happens BEFORE positioning, and the resulting size is re-read from
Qt rather than assumed, because a layout's minimumSizeHint() can refuse the
requested resize. Positioning uses the post-resize frame size, so the bottom
of the window cannot be pushed off the bottom edge as a side effect of
pulling the top down out from under the panel.

All decisions are logged to the console so a bug report is readable without
asking the user to run diagnostic commands.
"""

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

import logging
log = logging.getLogger(f"{__package__.split('.')[0]}.window")

# Used ONLY when the window manager fails to publish _NET_WORKAREA. Minimal
# WMs common in VNC sessions (IceWM, Fluxbox, twm) sometimes omit it, in which
# case Qt reports availableGeometry() == geometry() and we have no idea how
# tall the panel is. 48 px is a conservative guess for a single top panel.
FALLBACK_TOP_MARGIN = 48

# Default gap between the window frame and the edge of the usable work area.
DEFAULT_MARGIN = 8


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _r(rect):
    """Format a QRect the way X11 tools do: WxH+X+Y."""
    return "%dx%d+%d+%d" % (rect.width(), rect.height(), rect.left(), rect.top())


def _screen_for(win):
    """
    Return the QScreen the window is on.

    QWidget.screen() only exists in Qt 5.14+, so go through windowHandle()
    and fall back to the primary screen.
    """
    try:
        handle = win.windowHandle()
        if handle is not None and handle.screen() is not None:
            return handle.screen()
    except Exception:
        pass

    try:
        scr = win.screen()
        if scr is not None:
            return scr
    except Exception:
        pass

    return QApplication.primaryScreen()


def _work_area(win):
    """
    Return (screen, full_geometry, usable_work_area).

    Falls back to a synthetic top margin when the WM publishes no work area.
    """
    scr = _screen_for(win)
    full = scr.geometry()
    avail = scr.availableGeometry()

    try:
        dpr = scr.devicePixelRatio()
    except Exception:
        dpr = 1.0

    log.info("screen=%s  full=%s  avail=%s  dpr=%.2f",
             scr.name(), _r(full), _r(avail), dpr)

    if avail == full:
        avail = full.adjusted(0, FALLBACK_TOP_MARGIN, 0, 0)
        log.warning("window manager published no usable work area "
                    "(_NET_WORKAREA missing or full-screen); assuming a %d px "
                    "top panel -> avail=%s", FALLBACK_TOP_MARGIN, _r(avail))

    return scr, full, avail


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
def fix_window_placement(win, margin=DEFAULT_MARGIN, center=True):
    """
    Make sure `win` is fully inside the usable screen area.

    Shrinks the window if it is larger than the work area, then positions it.
    Safe to call repeatedly. Must be called AFTER win.show(), otherwise the
    window manager has not applied decorations yet and the title-bar height
    reads as zero.

    Parameters
    ----------
    win : QWidget
        The window to place (normally the application main window).
    margin : int
        Gap to leave between the window frame and the work-area edges.
    center : bool
        True  -> center the window in the work area (matches the original
                 startup behaviour, minus the bugs).
        False -> leave the window where it is and only clamp it into range.
                 Use this when re-fixing after a screen-resolution change so
                 a window the user deliberately moved is not yanked back to
                 the middle.
    """
    if win is None:
        log.warning("fix_window_placement() called with win=None; ignoring.")
        return

    scr, full, avail = _work_area(win)

    geo = win.geometry()             # client area (no title bar)
    frame = win.frameGeometry()      # client area + WM decorations

    deco_w = frame.width() - geo.width()      # left + right border
    deco_h = frame.height() - geo.height()    # title bar + borders
    dx = geo.left() - frame.left()            # left border thickness
    dy = geo.top() - frame.top()              # title bar height

    log.info("before: client=%s  frame=%s  decorations=(%d x %d)  titlebar=%d px",
             _r(geo), _r(frame), deco_w, deco_h, dy)

    if dy == 0:
        log.warning("title-bar height reads as 0 -- window decorations are not "
                    "applied yet. fix_window_placement() should be called after "
                    "show(), ideally via QTimer.singleShot(0, ...). Placement "
                    "may be off by the height of the title bar.")

    # ------------------------------------------------------------------------
    # STEP 1 -- SHRINK FIRST so the whole frame fits inside the work area.
    #           Doing this before positioning is what stops the bottom of the
    #           window from being pushed off-screen when we pull the top down
    #           out from under the panel.
    # ------------------------------------------------------------------------
    max_frame_w = avail.width() - 2 * margin
    max_frame_h = avail.height() - 2 * margin

    want_client_w = min(frame.width(), max_frame_w) - deco_w
    want_client_h = min(frame.height(), max_frame_h) - deco_h

    # Never ask for a degenerate size on a tiny virtual desktop.
    want_client_w = max(want_client_w, 200)
    want_client_h = max(want_client_h, 150)

    if (want_client_w, want_client_h) != (geo.width(), geo.height()):
        log.info("resizing client %d x %d -> %d x %d "
                 "(frame must fit within %d x %d)",
                 geo.width(), geo.height(),
                 want_client_w, want_client_h,
                 max_frame_w, max_frame_h)
        win.resize(want_client_w, want_client_h)
    else:
        log.info("no resize needed; window already fits %d x %d",
                 max_frame_w, max_frame_h)

    # ------------------------------------------------------------------------
    # STEP 2 -- RE-READ the geometry. Qt may have refused the resize because
    #           a layout, a minimumSize, or a minimumSizeHint set a floor.
    #           Never assume the resize took effect.
    # ------------------------------------------------------------------------
    geo = win.geometry()
    frame = win.frameGeometry()
    frame_w, frame_h = frame.width(), frame.height()

    if frame_w > max_frame_w or frame_h > max_frame_h:
        try:
            hint = win.minimumSizeHint()
            hint_w, hint_h = hint.width(), hint.height()
        except Exception:
            hint_w = hint_h = -1
        log.warning("window will not shrink to fit: frame=%d x %d exceeds "
                    "usable %d x %d  (minimumSize=%d x %d, "
                    "minimumSizeHint=%d x %d). Part of the window will "
                    "overflow the screen edge. Consider lowering the main "
                    "window minimum size or putting content in a QScrollArea.",
                    frame_w, frame_h, max_frame_w, max_frame_h,
                    win.minimumWidth(), win.minimumHeight(), hint_w, hint_h)
    else:
        log.info("post-resize: client=%s  frame=%s", _r(geo), _r(frame))

    # ------------------------------------------------------------------------
    # STEP 3 -- POSITION using the POST-resize frame size.
    # ------------------------------------------------------------------------
    min_fx = avail.left() + margin
    min_fy = avail.top() + margin
    max_fx = avail.left() + avail.width() - margin - frame_w
    max_fy = avail.top() + avail.height() - margin - frame_h

    if center:
        target_fx = avail.left() + (avail.width() - frame_w) // 2
        target_fy = avail.top() + (avail.height() - frame_h) // 2
        log.info("centering in work area -> frame target (%d, %d)",
                 target_fx, target_fy)
    else:
        target_fx, target_fy = frame.left(), frame.top()
        log.info("preserving current position -> frame target (%d, %d)",
                 target_fx, target_fy)

    # Clamp. When the window is too big to fit (max < min), pin to the
    # top-left of the work area so the menu bar and title bar stay reachable;
    # overflow goes off the bottom/right where it does the least harm.
    if max_fx < min_fx:
        new_fx = min_fx
        log.warning("window is wider than the work area; pinning to left edge.")
    else:
        new_fx = max(min_fx, min(target_fx, max_fx))

    if max_fy < min_fy:
        new_fy = min_fy
        log.warning("window is taller than the work area; pinning to top edge "
                    "so the title bar and menu bar remain reachable.")
    else:
        new_fy = max(min_fy, min(target_fy, max_fy))

    log.info("moving frame (%d, %d) -> (%d, %d)   [x allowed %d..%d, "
             "y allowed %d..%d]",
             frame.left(), frame.top(), new_fx, new_fy,
             min_fx, max_fx, min_fy, max_fy)

    # move() positions the CLIENT area, so add the border/title-bar offsets
    # to make the FRAME land where we intend. This is the correction the
    # original code was missing.
    win.move(new_fx + dx, new_fy + dy)

    # Report the settled result once the WM has processed the request.
    QTimer.singleShot(0, lambda: log.info(
        "after:  client=%s  frame=%s",
        _r(win.geometry()), _r(win.frameGeometry())))


# ----------------------------------------------------------------------------
# convenience wrappers
# ----------------------------------------------------------------------------
def install_window_placement_guard(win, margin=DEFAULT_MARGIN):
    """
    Place the window correctly at startup and keep it correct afterwards.

    Call this once, immediately after the main window has been shown.

    - Runs fix_window_placement() on the next event-loop turn, which is when
      the window manager has finished decorating the window.
    - Re-runs it (without re-centering) whenever the work area changes, so
      reconnecting to a VNC session at a different resolution cannot strand
      the window off-screen.
    """
    QTimer.singleShot(0, lambda: fix_window_placement(win, margin=margin,
                                                      center=True))

    scr = _screen_for(win)
    if scr is None:
        return

    def _on_work_area_changed(*_args):
        log.info("screen work area changed; re-checking window placement.")
        fix_window_placement(win, margin=margin, center=False)

    try:
        scr.availableGeometryChanged.connect(_on_work_area_changed)
        scr.geometryChanged.connect(_on_work_area_changed)
        log.info("window placement guard installed on screen '%s'.", scr.name())
    except Exception as exc:
        log.warning("could not connect screen-change signals: %s", exc)

    # Keep a reference so the closures are not garbage collected.
    win._grime_placement_handler = _on_work_area_changed


def reset_window_layout(win, margin=DEFAULT_MARGIN):
    """
    User-facing escape hatch, wired to View -> Reset Window Layout.

    Un-maximizes if needed, restores a sane default size, then re-centers
    inside the usable work area.
    """
    log.info("Reset Window Layout requested by user.")

    try:
        if win.isMaximized() or win.isFullScreen():
            win.showNormal()
    except Exception:
        pass

    scr, full, avail = _work_area(win)

    # A comfortable default: 85% of the usable area, capped at 1600x1000.
    default_w = min(int(avail.width() * 0.85), 1600)
    default_h = min(int(avail.height() * 0.85), 1000)
    log.info("resetting to default size %d x %d", default_w, default_h)
    win.resize(default_w, default_h)

    fix_window_placement(win, margin=margin, center=True)
