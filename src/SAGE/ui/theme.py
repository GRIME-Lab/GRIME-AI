# SAGE/ui/theme.py
"""
Light / dark QSS theme for the SAGE panel.

Separation comes from card-style QGroupBoxes on a recessed page: each section
is a raised surface with a hairline border, and the gaps between them do the
separating - no heavy divider lines. One accent (green) marks selection and the
Save commit; everything else stays neutral.

The accent moved from blue to green so the app reads with GRIME's identity
colour. Selection, focus rings, checked mode buttons, and Save all draw from the
same green ramp; status colours (info / ok / warn used by the seed sub-panels)
are separate tokens so they adapt between light and dark instead of being
hard-coded per widget.

Usage:
    from SAGE.ui.theme import apply_theme
    apply_theme(widget, "dark")          # or "light"
    # status colours for code that styles widgets directly:
    from SAGE.ui.theme import colors
    c = colors("dark")
    label.setStyleSheet(f"color: {c['warn_tx']};")
"""

# Corner radius of filmstrip thumbnails and their item cells, px. Shared by the
# QSS below and by filmstrip.py so the cell, the image and the annotated border
# all round the same way.
THUMB_RADIUS = 6

_LIGHT = {
    "page":        "#e8eae6",
    "card":        "#fbfcfa",
    "surface":     "#ffffff",
    "border":      "#dfe3dc",
    "border_soft": "#eef0ec",
    "text":        "#2b302b",
    "text_muted":  "#6b7269",
    "btn":         "#f3f5f1",
    "btn_hover":   "#e9ece6",
    "btn_border":  "#dfe3dc",
    # accent = green
    "accent_bg":   "#e2f2e5",
    "accent_bd":   "#5cb27a",
    "accent_tx":   "#1f7a44",
    "sel_row":     "#e9f4ec",
    "header":      "#f4f6f2",
    "disabled_tx": "#a5aaa0",
    "save":        "#2e9e6b",
    "save_hover":  "#28885b",
    # ---- status tokens (seed sub-panels: info / ok / warn) ----
    "info_bg":     "#f0f1ef",
    "info_bd":     "#c3c8bf",
    "info_tx":     "#7b8175",
    "ok_bg":       "#eef8ef",
    "ok_bd":       "#a5d6a7",
    "ok_tx":       "#2e7d32",
    "warn_bg":     "#fbf6e6",
    "warn_bd":     "#e6cf87",
    "warn_tx":     "#8a5a12",
    "warn_tx2":    "#6b4610",
    # ---- added: chrome + control tokens ----
    "toolbar":     "#f4f6f2",
    "icon":        "#6b7269",
    "icon_on":     "#1f7a44",
    "pill":        "#eef0ec",
    "pill_tx":     "#6b7269",
    "swatch_bd":   "#cfd4cb",
    "canvas_bg":   "#20241f",
    # ---- filmstrip: annotated-image number + border (orange; avoids red
    #      and green for colour-blind viewers) ----
    "annotated":   "#c86400",
}

_DARK = {
    "page":        "#191c1a",
    "card":        "#262a27",
    "surface":     "#1f231f",
    "border":      "#343a34",
    "border_soft": "#2b302b",
    "text":        "#d5dad3",
    "text_muted":  "#8b928a",
    "btn":         "#2f342f",
    "btn_hover":   "#373d37",
    "btn_border":  "#343a34",
    # accent = green
    "accent_bg":   "#234630",
    "accent_bd":   "#4caf72",
    "accent_tx":   "#8fd6a4",
    "sel_row":     "#223a2a",
    "header":      "#2b302b",
    "disabled_tx": "#5f665f",
    "save":        "#2e9e6b",
    "save_hover":  "#37b07b",
    # ---- status tokens ----
    "info_bg":     "#2a2f2a",
    "info_bd":     "#3a413a",
    "info_tx":     "#8b928a",
    "ok_bg":       "#1e3524",
    "ok_bd":       "#3f7a4e",
    "ok_tx":       "#7fc98d",
    "warn_bg":     "#332d1c",
    "warn_bd":     "#6b5a2c",
    "warn_tx":     "#d9b25f",
    "warn_tx2":    "#c39a45",
    # ---- added: chrome + control tokens ----
    "toolbar":     "#222623",
    "icon":        "#8b928a",
    "icon_on":     "#8fd6a4",
    "pill":        "#2b302b",
    "pill_tx":     "#8b928a",
    "swatch_bd":   "#3d443d",
    "canvas_bg":   "#141613",
    # ---- filmstrip: annotated-image number + border (lighter orange so it
    #      reads on the dark background) ----
    "annotated":   "#e69f00",
}


def colors(mode: str = "dark") -> dict:
    """Return the raw token dict for `mode`. Use for widgets styled in code
    (e.g. the seed sub-panels) so their colours track the active theme instead
    of being hard-coded."""
    return dict(_DARK if str(mode).lower() == "dark" else _LIGHT)


def _qss(c):
    return f"""
    QWidget {{
        color: {c['text']};
        font-size: 12px;
    }}
QMainWindow {{ background: {c['page']}; }}
    QScrollArea {{ background: {c['page']}; border: none; }}
    QScrollArea > QWidget > QWidget {{ background: {c['page']}; }}
    QMenuBar {{ background: {c['card']}; color: {c['text']}; border-bottom: 1px solid {c['border']}; }}
    QMenuBar::item {{ background: transparent; padding: 5px 10px; border-radius: 5px; }}
    QMenuBar::item:selected {{ background: {c['accent_bg']}; color: {c['accent_tx']}; }}
    QMenu {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 6px; padding: 4px; }}
    QMenu::item {{ padding: 5px 20px 5px 12px; border-radius: 4px; color: {c['text']}; }}
    QMenu::item:selected {{ background: {c['accent_bg']}; color: {c['accent_tx']}; }}
    QSpinBox, QDoubleSpinBox {{
        background: {c['surface']};
        color: {c['text']};
        border: 1px solid {c['border']};
        border-radius: 5px;
        padding: 3px 6px;
    }}
    QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {c['accent_bd']}; }}
    QComboBox {{
        background: {c['surface']};
        color: {c['text']};
        border: 1px solid {c['border']};
        border-radius: 5px;
        padding: 4px 8px;
    }}
    QComboBox:focus {{ border: 1px solid {c['accent_bd']}; }}
    QWidget#sagePanel {{ background: {c['page']}; }}
        QGroupBox {{
        background: {c['card']};
        border: 1px solid {c['border']};
        border-radius: 7px;
        margin-top: 16px;
        padding-top: 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 8px;
        top: 0px;
        padding: 1px 5px;
        background: {c['page']};
        color: {c['text']};
        font-size: 11px;
    }}
    QPushButton {{
        background: {c['btn']};
        color: {c['text']};
        border: 1px solid {c['btn_border']};
        border-radius: 5px;
        padding: 6px 8px;
    }}
    QPushButton:hover {{ background: {c['btn_hover']}; }}
    QPushButton:pressed {{ background: {c['btn_hover']}; }}
    QPushButton:checked {{
        background: {c['accent_bg']};
        border: 1px solid {c['accent_bd']};
        color: {c['accent_tx']};
        font-weight: 500;
    }}
    QPushButton:disabled {{ color: {c['disabled_tx']}; }}
    QPushButton#saveButton {{
        background: {c['save']};
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 500;
    }}
    QPushButton#saveButton:hover {{ background: {c['save_hover']}; }}
    QLineEdit {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 5px;
        padding: 5px 8px;
        color: {c['text']};
    }}
    QLineEdit:focus {{ border: 1px solid {c['accent_bd']}; }}
    QListWidget, QTableWidget {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 5px;
        outline: none;
    }}
    QListWidget::item:selected, QTableWidget::item:selected {{
        background: {c['sel_row']};
        color: {c['text']};
    }}
    QHeaderView::section {{
        background: {c['header']};
        color: {c['text_muted']};
        border: none;
        border-bottom: 1px solid {c['border_soft']};
        padding: 4px;
    }}
    QTableWidget {{ gridline-color: {c['border_soft']}; }}
    QRadioButton {{ color: {c['text']}; spacing: 6px; padding: 1px 0; }}
    QRadioButton:disabled {{ color: {c['disabled_tx']}; }}
    QCheckBox {{ color: {c['text']}; spacing: 6px; }}
    QCheckBox:disabled {{ color: {c['disabled_tx']}; }}
    QToolTip {{
        background: {c['card']};
        color: {c['text']};
        border: 1px solid {c['border']};
        border-radius: 4px;
        padding: 4px 6px;
    }}
    QScrollBar:vertical {{ background: {c['card']}; width: 10px; margin: 0; }}
    QScrollBar::handle:vertical {{
        background: {c['btn_border']};
        border-radius: 5px;
        min-height: 24px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    /* ---- seed sub-panel states (objectName-scoped) ---- */
    QFrame#subpanel_locked {{
        background: {c['info_bg']};
        border: 1px dashed {c['info_bd']};
        border-radius: 4px;
    }}
    QFrame#subpanel_loaded {{
        background: {c['ok_bg']};
        border: 1px solid {c['ok_bd']};
        border-radius: 4px;
    }}
    QFrame#subpanel_persists {{
        background: {c['warn_bg']};
        border: 1px solid {c['warn_bd']};
        border-radius: 4px;
    }}

    /* =======================================================
       Toolbar chrome
       ======================================================= */
    QToolBar {{
        background: {c['toolbar']};
        border: none;
        border-bottom: 1px solid {c['border']};
        padding: 4px 8px;
        spacing: 8px;
    }}
    QToolBar::separator {{
        background: {c['border']};
        width: 1px;
        margin: 3px 6px;
    }}
    QLabel#toolbarCaption {{
        color: {c['text_muted']};
        font-size: 11px;
        padding-right: 2px;
    }}

    /* Checkboxes were colliding in the toolbar because they had no spacing
       between the indicator and the label, and none between siblings. The
       native indicator is deliberately left unstyled so the checkmark glyph
       is preserved. */
    QCheckBox {{
        color: {c['text']};
        spacing: 6px;
        padding: 2px 4px;
    }}
    QCheckBox:disabled {{ color: {c['disabled_tx']}; }}

    /* =======================================================
       Panel section headings
       ======================================================= */
    QLabel#sectionCaption {{
        color: {c['text_muted']};
        font-size: 11px;
        font-weight: 500;
        padding: 0 2px 2px 2px;
    }}
    QLabel#hintText {{
        color: {c['text_muted']};
        font-size: 11px;
    }}
    QLabel#okText   {{ color: {c['ok_tx']};   font-size: 11px; font-weight: 500; }}
    QLabel#warnText {{ color: {c['warn_tx']}; font-size: 11px; font-weight: 500; }}
    QLabel#warnSub  {{ color: {c['warn_tx2']}; font-size: 10px; }}

    /* =======================================================
       Segmentation-mode buttons
       ======================================================= */
    QPushButton#modeButton {{
        background: {c['btn']};
        border: 1px solid {c['btn_border']};
        border-radius: 5px;
        padding: 6px 6px;
        color: {c['text']};
        text-align: center;
    }}
    QPushButton#modeButton:hover {{ background: {c['btn_hover']}; }}
    QPushButton#modeButton:checked {{
        background: {c['accent_bg']};
        border: 1px solid {c['accent_bd']};
        color: {c['accent_tx']};
        font-weight: 500;
    }}

    /* =======================================================
       Tool rail: Draw / Select / Pan (icon over caption)
       ======================================================= */
    QToolButton#toolButton {{
        background: {c['btn']};
        border: 1px solid {c['btn_border']};
        border-radius: 6px;
        padding: 5px 2px;
        color: {c['text_muted']};
        font-size: 10px;
    }}
    QToolButton#toolButton:hover {{ background: {c['btn_hover']}; }}
    QToolButton#toolButton:checked {{
        background: {c['accent_bg']};
        border: 1px solid {c['accent_bd']};
        color: {c['accent_tx']};
        font-weight: 500;
    }}

    /* =======================================================
       Seeding pills (replaces the loose radio column)
       ======================================================= */
    QPushButton#seedPill {{
        background: {c['pill']};
        border: 1px solid transparent;
        border-radius: 9px;
        padding: 3px 10px;
        color: {c['pill_tx']};
        font-size: 11px;
    }}
    QPushButton#seedPill:hover {{ background: {c['btn_hover']}; }}
    QPushButton#seedPill:checked {{
        background: {c['accent_bg']};
        border: 1px solid {c['accent_bd']};
        color: {c['accent_tx']};
        font-weight: 500;
    }}
    QPushButton#seedPill:disabled {{ color: {c['disabled_tx']}; background: {c['pill']}; }}
    QWidget#seedRow {{
        background: {c['surface']};
        border: 1px solid {c['border_soft']};
        border-radius: 6px;
    }}
    QLabel#seedCaption {{ color: {c['text_muted']}; font-size: 10px; }}

    /* =======================================================
       Filmstrip: rounded item cells to match the rounded thumbnails
       ======================================================= */
    QListWidget#filmstrip::item {{
        border-radius: {THUMB_RADIUS}px;
    }}
    QListWidget#filmstrip::item:selected {{
        background: {c['sel_row']};
        border-radius: {THUMB_RADIUS}px;
    }}
    """


_ACTIVE_MODE = "light"


def active_mode() -> str:
    """The mode passed to the most recent apply_theme() call. Lets widgets that
    paint themselves in code (icons, swatches) pick colours without each one
    having to track the theme separately."""
    return _ACTIVE_MODE


def icon_color(mode: str = None, on: bool = False) -> str:
    """Colour for hand-rendered icons. `on` selects the checked/accent variant."""
    c = colors(mode or _ACTIVE_MODE)
    return c["icon_on"] if on else c["icon"]


def apply_theme(widget, mode: str = "dark"):
    """Apply the light or dark QSS to `widget` (and, via cascade, its children)."""
    global _ACTIVE_MODE
    _ACTIVE_MODE = "dark" if str(mode).lower() == "dark" else "light"
    palette = _DARK if _ACTIVE_MODE == "dark" else _LIGHT
    widget.setStyleSheet(_qss(palette))