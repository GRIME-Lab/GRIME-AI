# SAGE/main.py

print(">>> RUNNING SAGE.main FROM:", __file__)

import sam2
print("SAM2 is being imported from:", sam2.__file__)

import os
import sys
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import argparse
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from SAGE.ui.main_window import MainWindow
from SAGE.core.model_manager import ModelManager
from SAGE.backends.sam2_backend import SAM2Backend


def parse_args():
    # Locate SAM2 the same way GRIME AI's sam2_trainer.py does: from the
    # installed sam2 package, so no machine-specific paths are needed.
    sam2_dir = os.path.dirname(importlib.util.find_spec("sam2").origin)

    parser = argparse.ArgumentParser(description="Segmentation & Annotation for Geospatial Ecohydrology (SAGE)")
    parser.add_argument(
        "--checkpoint",
        default=os.path.normpath(os.path.join(sam2_dir, "checkpoints", "sam2.1_hiera_large.pt")),
        help="Path to SAM2.1 checkpoint",
    )
    parser.add_argument(
        "--config_dir",
        default=os.path.normpath(os.path.join(sam2_dir, "configs", "sam2.1")),
        help="Directory containing SAM2.1 configs",
    )
    parser.add_argument(
        "--config_name",
        default="sam2.1_hiera_l.yaml",
        help="Config filename inside config_dir",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device string (e.g. 'cuda', 'cuda:0', 'cpu')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    app = QApplication(sys.argv)

    # Theme the ENTIRE application once, here, so every widget - main window,
    # menu bar, toolbar, canvas, filmstrip, sidebar - inherits it. Applying on
    # the QApplication is the only scope a child widget cannot override or miss.
    from SAGE.ui.theme import apply_theme
    apply_theme(app, "light")   # switch to "dark" for dark mode

    app.processEvents()

    # Load splash image
    base_path = Path(__file__).resolve().parent
    logo_dir = base_path / "resources"
    RESOURCE_PATH = os.path.join(logo_dir, "sage_logo.png")

    full_pixmap = QPixmap(RESOURCE_PATH)

    scaled_pixmap = full_pixmap.scaled(full_pixmap.width() // 2, full_pixmap.height() // 2, Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation)
    splash = QSplashScreen(scaled_pixmap, Qt.WindowStaysOnTopHint)
    splash.setMask(scaled_pixmap.mask())
    splash.show()

    # Optional: delay for visual effect
    time.sleep(1.5)  # adjust duration as needed

    model_manager = ModelManager(
        backend_cls=SAM2Backend,
        checkpoint_path=args.checkpoint,
        config_dir=args.config_dir,
        config_name=args.config_name,
        device=args.device,
    )

    def launch_main():
        window = MainWindow(model_manager=model_manager)
        window.show()
        splash.finish(window)

    QTimer.singleShot(1500, launch_main)  # auto-close after 1.5 seconds

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
