# sam2_gui/core/segmentation_backend.py
from abc import ABC, abstractmethod
import numpy as np


class SegmentationBackend(ABC):
    """Abstract interface for segmentation backends."""

    @abstractmethod
    def set_image(self, image_np: np.ndarray):
        pass

    @abstractmethod
    def segment_from_points(self, fg_points, bg_points):
        """
        fg_points, bg_points: list of (x, y) tuples in image coordinates.
        Returns: mask (H, W) bool or uint8.
        """
        pass

    # Optional capability — backends that support automatic-mask + anchor
    # classification (Smart Select) override this.
    def segment_smart_select(self, fg_points, bg_points, exclude_mask=None):
        raise NotImplementedError(
            f"{type(self).__name__} does not support Smart Select segmentation."
        )
