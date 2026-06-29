# chain_link_resources.py
# Auto-generated — do not edit by hand.
# Embeds linked/unlinked chain icon PNGs as base64 and registers them
# with Qt so they are accessible via QPixmap(":/icons/linked.png") etc.

import base64
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QByteArray

_LINKED_B64 = b"""iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAj0lEQVR4nO2VQQ6AIAwEwfA2T7yJg2/yxOf0RKKlNG1NxMSdm7ojTbbGEAAA4O9E6eG67cf1upYs5j0ue5PKFGkQq7tYXyBlPG6SwnRazQFWN2lCHJYs57YBuwoaXM/aJbS4wwHe4rsDcB1re7e4tyWsJUftIVynHlf8DJ9uuibXVaDZ9FHG407/FwAAwHRORrxXkTUj/DQAAAAASUVORK5CYII="""
_UNLINKED_B64 = b"""iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAoklEQVR4nO2VsRGAIAxFkXMiK4tMwggZKCM4iQWVK2GFx0FEQiHemVdhzMN/gmiMoijK35lqN4kopNeIWO3vcdliLufUgkhdK50g7fEAwQOEvN7iRmZJ2pYHSF3b0sRBROFw7hpLXTZACrfOd2u/bFu3exvgLb4bgFtTrrbue/FqW90igOSgQcQp7c/3wZMbx9XPULKze93hJ+Hwf4GiKMpwTpk9Zh9gNdVsAAAAAElFTkSuQmCC"""

def _load(b64_data: bytes, qt_path: str):
    raw = base64.b64decode(b64_data)
    ba  = QByteArray(raw)
    pm  = QPixmap()
    pm.loadFromData(ba, "PNG")
    # Cache on the module so it isn't garbage collected
    _PIXMAPS[qt_path] = pm

_PIXMAPS: dict = {}

def load_resources():
    """Call once at startup to register the icons."""
    _load(_LINKED_B64,   "linked")
    _load(_UNLINKED_B64, "unlinked")

def get_icon(name: str) -> QPixmap:
    """Return QPixmap for 'linked' or 'unlinked'."""
    if name not in _PIXMAPS:
        load_resources()
    return _PIXMAPS[name]
