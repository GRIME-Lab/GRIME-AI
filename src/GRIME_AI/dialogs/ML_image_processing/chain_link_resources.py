# chain_link_resources.py
# Icons are stored in src/GRIME_AI/resources/app_icons/
# Drop chain_linked.png and chain_unlinked.png there.

import os
from PyQt5.QtGui import QPixmap

_ICONS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "app_icons")

_PIXMAPS: dict = {}

def get_icon(name: str) -> QPixmap:
    """Return QPixmap for 'linked' or 'unlinked'. Loads from resources/app_icons/."""
    if name not in _PIXMAPS:
        path = os.path.normpath(os.path.join(_ICONS_DIR, f"chain_{name}.png"))
        pm = QPixmap(path)
        _PIXMAPS[name] = pm
    return _PIXMAPS[name]
