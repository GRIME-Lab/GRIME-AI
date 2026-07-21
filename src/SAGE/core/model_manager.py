# sam2_gui/core/model_manager.py
import numpy as np
from SAGE.core.segmentation_backend import SegmentationBackend


class ModelManager:
    def __init__(
        self,
        backend_cls,
        checkpoint_path: str,
        config_dir: str,
        config_name: str,
        device: str = "cuda",
    ):
        assert issubclass(backend_cls, SegmentationBackend)
        self.backend_cls = backend_cls
        self.checkpoint_path = checkpoint_path
        self.config_dir = config_dir
        self.config_name = config_name
        self.device = device
        self.backend = None

    def initialize_backend(self, image_np: np.ndarray):
        # Build the SAM2 backend (loads the checkpoint) ONCE, then only swap the
        # image on subsequent calls. Reconstructing per image reloaded the full
        # ~900 MB checkpoint on every filmstrip click — the load-time bottleneck.
        if self.backend is None:
            self.backend = self.backend_cls(
                checkpoint_path=self.checkpoint_path,
                config_dir=self.config_dir,
                config_name=self.config_name,
                device=self.device,
            )
        self.backend.set_image(image_np)

    def segment_from_points(self, fg_points, bg_points, exclude_mask=None):
        if self.backend is None:
            raise RuntimeError("Model backend not initialized with image.")

        # >>> ADDED: pass exclude_mask to backend if provided <<<
        return self.backend.segment_from_points(
            fg_points,
            bg_points,
            exclude_mask=exclude_mask,
        )

    def segment_smart_select(self, fg_points, bg_points, exclude_mask=None):
        if self.backend is None:
            raise RuntimeError("Model backend not initialized with image.")
        return self.backend.segment_smart_select(
            fg_points,
            bg_points,
            exclude_mask=exclude_mask,
        )
