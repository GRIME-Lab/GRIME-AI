"""Shared test setup: make the src/ packages importable and keep Qt headless."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import GRIME_AI  # noqa: E402  registers the appcore import alias used inside the package


@pytest.fixture
def rng():
    """Seeded random generator so every run sees the same synthetic images."""
    return np.random.default_rng(12345)


@pytest.fixture
def textured_image(rng):
    """Sharp, mid-brightness, full-contrast 8-bit grayscale image (256 x 256)."""
    return rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
