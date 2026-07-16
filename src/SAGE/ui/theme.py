# SAGE/ui/theme.py
"""
Light / dark QSS theme for the SAGE panel.

Separation comes from card-style QGroupBoxes on a recessed page: each section
is a raised surface with a hairline border, and the gaps between them do the
separating — no heavy divider lines. One accent (blue) marks selection; one
commit color (green) marks Save. Everything else stays neutral.

Usage:
    from SAGE.ui.theme import apply_theme
    apply_theme(widget, "dark")   # or "light"
"""

_LIGHT = {
    "page":        "#e8eaee",
    "card":        "#fbfcfd",
    "surface":     "#ffffff",
    "border":      "#e0e3e8",
    "border_soft": "#eef0f3",
    "text":        "#2b3038",
    "text_muted":  "#6b7280",
    "btn":         "#f4f5f7",
    "btn_hover":   "#eaecef",
    "btn_border":  "#e0e3e8",
    "accent_bg":   "#e6f0fd",
    "accent_bd":   "#7fb0ee",
    "accent_tx":   "#1c5fa8",
    "sel_row":     "#eaf1fb",
    "header":      "#f5f6f8",
    "disabled_tx": "#a2a7ad",
    "save":        "#2e9e6b",
    "save_hover":  "#28885b",
}

_DARK = {
    "page":        "#191c21",
    "card":        "#262a31",
    "surface":     "#1f232a",
    "border":      "#343943",
    "border_soft": "#2b3038",
    "text":        "#d5d9df",
    "text_muted":  "#8b929c",
    "btn":         "#2f343c",
    "btn_hover":   "#373d46",
    "btn_border":  "#343943",
    "accent_bg":   "#22405f",
    "accent_bd":   "#4a90e2",
    "accent_tx":   "#9cc4f5",
    "sel_row":     "#243447",
    "header":      "#2b3038",
    "disabled_tx": "#5f6672",
    "save":        "#2e9e6b",
    "save_hover":  "#37b07b",
}


def _qss(c):
    return f"""
    QWidget {{
        color: {c['text']};
        font-size: 12px;
    }}
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
    QScrollBar:vertical {{ background: {c['card']}; width: 10px; margin: 0; }}
    QScrollBar::handle:vertical {{
        background: {c['btn_border']};
        border-radius: 5px;
        min-height: 24px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    """


def apply_theme(widget, mode: str = "dark"):
    """Apply the light or dark QSS to `widget` (and, via cascade, its children)."""
    palette = _DARK if str(mode).lower() == "dark" else _LIGHT
    widget.setStyleSheet(_qss(palette))
