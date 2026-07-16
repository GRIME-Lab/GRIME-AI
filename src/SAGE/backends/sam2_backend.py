# sam2_gui/backends/sam2_backend.py
import sys
import os
import numpy as np
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_module

import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from SAGE.core.segmentation_backend import SegmentationBackend
from SAGE.utils.anchor_classifier import classify_by_anchors


class SAM2Backend(SegmentationBackend):
    def __init__(self, checkpoint_path, config_dir, config_name, device="cuda",
                 use_texture_filter=False, use_mahalanobis=False, use_boundary_refinement=False):
        self.checkpoint_path = checkpoint_path
        self.config_dir = config_dir
        self.config_name = config_name
        self.device = device
        self.use_texture_filter = use_texture_filter
        self.use_mahalanobis = use_mahalanobis
        self.use_boundary_refinement = use_boundary_refinement

        # Hydra init for SAM2.1 configs
        GlobalHydra.instance().clear()
        # hydra.initialize_config_dir(config_dir=self.config_dir, version_base="1.1")
        #
        # sam2_model = build_sam2(self.config_name, self.checkpoint_path, device=self.device)
        initialize_config_module("sam2", version_base="1.2")  # match your sam2/__init__.py

        # --- Normalize config_name to match sam2 package layout ---
        cn = (self.config_name or "").replace("\\", "/").strip()

        # If user passed only filename (e.g., "sam2.1_hiera_l.yaml"), expand it
        if cn.endswith(".yaml") and "/" not in cn:
            if cn.startswith("sam2.1_"):
                cn = f"configs/sam2.1/{cn}"
            elif cn.startswith("sam2_"):
                cn = f"configs/sam2/{cn}"

        # If user passed "configs/..." but missing the subfolder, fix it
        if cn.startswith("configs/") and "sam2.1_" in cn and "/sam2.1/" not in cn:
            fname = cn.split("/")[-1]
            cn = f"configs/sam2.1/{fname}"

        self.config_name = cn
        print("Using SAM2 config:", self.config_name)

        # --- Normalize checkpoint_path (remove John absolute path) ---
        ckpt = (self.checkpoint_path or "").replace("\\", "/").strip()

        # If it's an absolute path but doesn't exist, try to recover by using just the filename
        if ckpt and (":" in ckpt) and (not os.path.exists(ckpt)):
            ckpt = os.path.basename(ckpt)

        # If it's just a filename, assume it's in GRIME_AI/sam2/checkpoints/
        package_dir = os.path.dirname(sam2.__file__)
        default_ckpt_dir = os.path.join(package_dir, "checkpoints")
        ckpt = os.path.join(default_ckpt_dir, os.path.basename(ckpt))

        self.checkpoint_path = ckpt
        print("Using SAM2 checkpoint:", self.checkpoint_path)

        sam2_model = build_sam2(self.config_name, self.checkpoint_path, device=self.device)
        self.sam2_model = sam2_model
        self.predictor = SAM2ImagePredictor(sam2_model)

        # Lazily-built automatic mask generator (Smart Select mode only).
        self._auto_generator = None
        self._auto_masks = None          # cached candidate masks for current image
        self.image_np = None

    def set_image(self, image_np: np.ndarray):
        self.image_np = image_np
        self._auto_masks = None          # invalidate candidate cache on image change
        self.predictor.set_image(image_np)

    def segment_from_points(self, fg_points, bg_points, exclude_mask=None):
        points = []
        labels = []

        for x, y in fg_points:
            points.append([x, y])
            labels.append(1)

        for x, y in bg_points:
            points.append([x, y])
            labels.append(0)

        if not points:
            return None

        input_point = np.array(points)
        input_label = np.array(labels)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        mask = masks[0].astype(bool)

        # Apply exclusion mask if provided
        if exclude_mask is not None:
            # Ensure boolean, same shape; if not, this will raise visibly instead of silently corrupting
            exclude_bool = exclude_mask.astype(bool)
            if exclude_bool.shape == mask.shape:
                mask = np.logical_and(mask, np.logical_not(exclude_bool))

        return mask

    # =====================================================================
    # Smart Select: automatic mask generation + anchor classify
    # =====================================================================
    def _get_auto_generator(self):
        if self._auto_generator is None:
            # Tuned for large, sparse water regions rather than tiny parts.
            self._auto_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2_model,
                points_per_side=32,
                pred_iou_thresh=0.80,
                stability_score_thresh=0.90,
                min_mask_region_area=200,
            )
        return self._auto_generator

    def _get_candidate_masks(self):
        """Generate (once per image) and cache class-agnostic candidate masks."""
        if self._auto_masks is None:
            if self.image_np is None:
                return []
            gen = self._get_auto_generator()
            records = gen.generate(self.image_np)
            self._auto_masks = [r["segmentation"].astype(bool) for r in records]
        return self._auto_masks

    def segment_smart_select(self, fg_points, bg_points, exclude_mask=None):
        """
        Smart Select: auto-generate + anchor classify.
          - auto-generate all candidate masks (zero per-thread prompts)
          - fg_points = water anchors, bg_points = exposed-bed anchors
          - two-class Mahalanobis assignment over per-region features
          - return union of water-assigned candidates

        Returns (mask_bool_or_None, info_dict).
        """
        if self.image_np is None:
            return None, {"reason": "no image set"}

        candidates = self._get_candidate_masks()
        if not candidates:
            return None, {"reason": "auto generator produced no masks"}

        target, info = classify_by_anchors(
            self.image_np, candidates, fg_points, bg_points
        )

        if target is not None and exclude_mask is not None:
            exclude_bool = exclude_mask.astype(bool)
            if exclude_bool.shape == target.shape:
                target = np.logical_and(target, np.logical_not(exclude_bool))
                if not target.any():
                    target = None

        return target, info


# sam2_gui/backends/sam2_backend.py
'''
class SAM2Backend(SegmentationBackend):
    def __init__(self, checkpoint_path, config_dir, config_name, device="cuda",
                 use_texture_filter=False, use_mahalanobis=False, use_boundary_refinement=False):
        self.checkpoint_path = checkpoint_path
        self.config_dir = config_dir
        self.config_name = config_name
        self.device = device
        self.use_texture_filter = use_texture_filter
        self.use_mahalanobis = use_mahalanobis
        self.use_boundary_refinement = use_boundary_refinement

        # Hydra init for SAM2.1 configs
        GlobalHydra.instance().clear()
        hydra.initialize_config_dir(config_dir=self.config_dir)

        sam2_model = build_sam2(self.config_name, self.checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.image_np = None

    def set_image(self, image_np: np.ndarray):
        self.image_np = image_np
        self.predictor.set_image(image_np)

    def _compute_texture_features(self, image_np, window_size=15):
        """
        Compute texture features: Shannon entropy and edge density
        High values indicate grass-like texture (many edges, high complexity)
        Low values indicate soil-like texture (smooth, homogeneous)
        """
        import cv2
        from scipy.ndimage import generic_filter
        from skimage.filters import sobel

        # Convert to grayscale
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # 1. Shannon Entropy (local complexity)
        def local_entropy(patch):
            """Compute Shannon entropy of a patch"""
            hist, _ = np.histogram(patch, bins=16, range=(0, 256))
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # Remove zeros
            return -np.sum(hist * np.log2(hist))

        entropy_map = generic_filter(gray, local_entropy, size=window_size)

        # 2. Edge density (Sobel gradient magnitude)
        edges = sobel(gray)
        edge_density = generic_filter(edges, np.mean, size=window_size)

        # Normalize both to [0, 1]
        entropy_norm = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-8)
        edge_norm = (edge_density - edge_density.min()) / (edge_density.max() - edge_density.min() + 1e-8)

        # Combine: texture score (higher = more grass-like)
        texture_score = 0.5 * entropy_norm + 0.5 * edge_norm

        return texture_score

    def _refine_boundary(self, mask, image_np, fg_points):
        """
        Refine mask boundaries using color similarity and morphological operations
        Particularly useful for water/vegetation boundaries
        """
        import cv2
        from skimage import morphology

        # 1. Get color statistics from foreground points
        fg_colors = []
        for x, y in fg_points:
            x_int, y_int = int(x), int(y)
            if 0 <= y_int < image_np.shape[0] and 0 <= x_int < image_np.shape[1]:
                fg_colors.append(image_np[y_int, x_int])

        if len(fg_colors) == 0:
            return mask

        fg_colors = np.array(fg_colors)
        mean_color = fg_colors.mean(axis=0)
        std_color = fg_colors.std(axis=0) + 1e-6

        # 2. Color similarity map (Mahalanobis-like distance in RGB space)
        h, w = mask.shape
        color_diff = image_np.reshape(-1, 3) - mean_color
        color_dist = np.sqrt(np.sum((color_diff / std_color) ** 2, axis=1))
        color_dist = color_dist.reshape(h, w)

        # Adaptive threshold based on foreground color variance
        color_threshold = 3.0  # Standard deviations
        color_mask = color_dist <= color_threshold

        # 3. Combine with original mask (conservative - only keep pixels that pass both tests)
        refined_mask = mask & color_mask

        # 4. Morphological operations to clean up boundaries
        # Remove small holes
        refined_mask = morphology.remove_small_holes(refined_mask, area_threshold=100)

        # Remove small objects (isolated pixels)
        refined_mask = morphology.remove_small_objects(refined_mask, min_size=50)

        # 5. Edge-aware smoothing using bilateral filter on mask boundary
        # This preserves sharp water/vegetation edges while smoothing noise
        mask_uint8 = refined_mask.astype(np.uint8) * 255

        # Find boundary region (dilate then subtract)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
        boundary_region = dilated - mask_uint8

        # Apply bilateral filter only to boundary region
        smoothed = cv2.bilateralFilter(mask_uint8, d=5, sigmaColor=50, sigmaSpace=50)

        # Blend: use smoothed version in boundary region, original elsewhere
        result = np.where(boundary_region > 0, smoothed, mask_uint8)

        # Threshold back to binary
        refined_mask = result > 127

        return refined_mask

    def segment_from_points(self, fg_points, bg_points, exclude_mask=None):
        points = []
        labels = []

        for x, y in fg_points:
            points.append([x, y])
            labels.append(1)

        for x, y in bg_points:
            points.append([x, y])
            labels.append(0)

        if not points:
            return None

        input_point = np.array(points)
        input_label = np.array(labels)

        # KEEP ORIGINAL: multimask_output=False for better accuracy
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        mask = masks[0].astype(bool)

        # Optional: Mahalanobis distance-based spatial constraint
        if self.use_mahalanobis and len(fg_points) >= 3 and self.image_np is not None:
            fg_array = np.array(fg_points)

            # Compute mean and covariance of foreground points
            mean = fg_array.mean(axis=0)
            cov = np.cov(fg_array.T)

            # Add regularization to avoid singular covariance
            cov += np.eye(2) * 1e-6

            try:
                # Compute inverse covariance
                cov_inv = np.linalg.inv(cov)

                # Create grid of all pixel coordinates
                h, w = mask.shape
                y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                pixel_coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)

                # Compute Mahalanobis distance for each pixel
                diff = pixel_coords - mean
                mahal_dist = np.sqrt(np.sum((diff @ cov_inv) * diff, axis=1))
                mahal_dist = mahal_dist.reshape(h, w)

                # Adaptive threshold based on point distribution
                scale_factor = np.sqrt(np.trace(cov)) / 50
                threshold = 6.0 * max(1.0, scale_factor)

                distance_mask = mahal_dist <= threshold
                mask = mask & distance_mask

            except np.linalg.LinAlgError:
                pass

        # Optional: Texture-based refinement
        if self.use_texture_filter and self.image_np is not None and len(fg_points) > 0:
            # Compute texture features
            texture_score = self._compute_texture_features(self.image_np, window_size=15)

            # Sample texture at foreground points to establish "grass texture threshold"
            fg_textures = []
            for x, y in fg_points:
                x_int, y_int = int(x), int(y)
                if 0 <= y_int < texture_score.shape[0] and 0 <= x_int < texture_score.shape[1]:
                    fg_textures.append(texture_score[y_int, x_int])

            if len(fg_textures) > 0:
                # Threshold: mean of foreground textures minus some margin
                texture_threshold = np.mean(fg_textures) - 0.15

                # Create texture mask: keep only high-texture areas
                texture_mask = texture_score >= texture_threshold

                # Apply texture constraint to mask
                mask = mask & texture_mask

        # Optional: Boundary refinement for water/vegetation edges
        if self.use_boundary_refinement and self.image_np is not None and len(fg_points) > 0:
            mask = self._refine_boundary(mask, self.image_np, fg_points)

        # Apply exclusion mask if provided
        if exclude_mask is not None:
            # Ensure boolean, same shape; if not, this will raise visibly instead of silently corrupting
            exclude_bool = exclude_mask.astype(bool)
            if exclude_bool.shape == mask.shape:
                mask = np.logical_and(mask, np.logical_not(exclude_bool))

        return mask
'''