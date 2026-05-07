#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# sam2_inference_engine.py

import os
import sys
import cv2
import shutil
import torch
import numpy as np
import importlib.util

from PIL import Image

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, prevents GUI windows
import matplotlib.pyplot as plt

# SAM2 imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../sam2'))
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from GRIME_AI.ml_core.ml_helpers import (get_color_for_category, init_coco_structure,
                                          add_coco_entries, save_coco_json,
                                          build_centroid_point_prompts)
from PyQt5.QtWidgets import QMessageBox


# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===        class SAM2InferenceEngine       ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class SAM2InferenceEngine:

    def __init__(self, device, model_cfg, trained_checkpoint_path, input_dir, output_dir):
        # =========================================================================
        # TEST MODE CONFIGURATION
        # Set self.TEST_MODE = True to run inference on hardcoded water + sky with colors
        # Set self.TEST_MODE = False to use normal configuration
        # =========================================================================
        self.TEST_MODE = False  # CHANGE THIS to True to enable test mode
        self.TEST_CATEGORIES = [
        {"id": 1, "name": "water", "color": (255, 255, 0)},  # Yellow
        {"id": 2, "name": "sky", "color": (255, 165, 0)}     # Orange
        ]
        # =========================================================================

        # =========================================================================
        # LABEL DISPLAY CONFIGURATION
        # Set self.DISPLAY_LABELS = True to show category names on overlay images
        # Set self.DISPLAY_LABELS = False to hide labels
        # =========================================================================
        self.DISPLAY_LABELS = True  # CHANGE THIS to False to hide labels
        # =========================================================================

        """
        Args:
            device: torch device (cuda/cpu)
            model_cfg: path to SAM2 model config yaml (e.g., "sam2.1_hiera_l.yaml")
            trained_checkpoint_path: path to YOUR TRAINED checkpoint (.torch file)
            input_dir: directory with input images
            output_dir: directory for output predictions
        """
        self.device = device
        self.MODEL_CFG = model_cfg
        self.TRAINED_CHECKPOINT = trained_checkpoint_path  # ← Your trained model
        self.segmentation_images_path = input_dir
        self.predictions_output_path = output_dir + " (sam2)"

        self.category_centroids = {}
        self.target_category_name = None

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def load_sam2_model(self):
        """Load SAM2 model architecture and trained weights from .torch checkpoint."""

        # 1. Setup paths
        main_dir = os.path.dirname(importlib.util.find_spec('sam2').origin)
        config_file = os.path.join(main_dir, "configs", "sam2.1", "sam2.1_hiera_l.yaml")

        print(f"Model config: {config_file}")
        print(f"Trained checkpoint: {self.TRAINED_CHECKPOINT}")

        # 2. Load model architecture from config
        cfg_intern = OmegaConf.load(config_file)
        raw_model_cfg = OmegaConf.to_container(cfg_intern.model, resolve=True)

        for key in ["no_obj_embed_spatial", "use_signed_tpos_enc_to_obj_ptrs", "device"]:
            raw_model_cfg.pop(key, None)

        new_cfg = OmegaConf.create(raw_model_cfg)
        model = instantiate(new_cfg, _recursive_=True)

        # 3. Move model to device
        # Use self.device as passed by the caller — do NOT re-resolve here.
        # Re-resolving via torch.cuda.is_available() can return False under lazy
        # CUDA initialisation, silently falling back to CPU.
        device = self.device
        print(f"Inference device: {device}")
        sam2_model = model.to(device).eval()
        predictor = SAM2ImagePredictor(sam2_model)

        # 4. Load trained checkpoint with metadata
        # Safe loading with proper error handling
        try:
            checkpoint = torch.load(
                self.TRAINED_CHECKPOINT,
                map_location=device,
                weights_only=False  # Required for checkpoints with metadata
            )
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            raise

        # 5. Display checkpoint info
        print("\n=== Checkpoint Information ===")
        print(f"Site: {checkpoint.get('site_name', 'N/A')}")
        print(f"Created: {checkpoint.get('creation_UTC', 'N/A')}")
        print(f"Epochs trained: {checkpoint.get('epochs', 'N/A')}")
        print(f"Learning rate: {checkpoint.get('learning_rate', 'N/A')}")
        print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"Validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
        print(f"Mean IoU: {checkpoint.get('miou', 'N/A')}")
        print(f"Categories: {len(checkpoint.get('categories', []))} classes")

        # 6. Load model weights
        if "model_state_dict" in checkpoint:
            predictor.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print("✓ Loaded trained model weights successfully")
        else:
            raise ValueError("Checkpoint missing 'model_state_dict' key!")

        # LOAD CATEGORY CENTROIDS METADATA
        self.category_centroids = checkpoint.get("category_centroids", {})
        print(f"Loaded {sum(len(v) for v in self.category_centroids.values())} centroids across "
              f"{len(self.category_centroids)} categories")

        # Store target category name from checkpoint
        self.target_category_name = checkpoint.get("target_category_name", None)
        if self.target_category_name:
            print(f"Model trained on category: {self.target_category_name}")

        # Resolve blob_filter_radius using three-tier fallback:
        #   1. .torch checkpoint metadata
        #   2. site_config.json
        #   3. GUI/default value (50px / 2236px diagonal ≈ 0.02236)
        self.blob_filter_radius = self._resolve_blob_filter_radius(checkpoint)
        print(f"Blob filter radius: {self.blob_filter_radius:.5f} (fraction of diagonal)")

        print("=== Model ready for inference ===\n")
        return predictor

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _resolve_blob_filter_radius(self, checkpoint: dict) -> float:
        """
        Resolve blob_filter_radius using a three-tier fallback:
          1. .torch checkpoint metadata  (most authoritative — matches training exactly)
          2. site_config.json            (fallback for checkpoints predating this feature)
          3. Default of 50px / 2236px   (last resort, ~0.02236 fraction of diagonal)
        Returns the fraction of image diagonal to use as the radius threshold.
        """
        import math as _math
        DEFAULT_FRACTION = 50.0 / _math.sqrt(2000**2 + 1000**2)  # ~0.02236

        # Tier 1: checkpoint metadata
        ckpt_value = checkpoint.get("blob_filter_radius")
        if ckpt_value is not None:
            try:
                val = float(ckpt_value)
                if val > 0:
                    print(f"  blob_filter_radius: loaded from checkpoint ({val:.5f})")
                    return val
            except (TypeError, ValueError):
                pass

        # Tier 2: site_config.json
        try:
            from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
            from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor
            import os
            settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
            config_file = os.path.join(settings_folder, "site_config.json")
            cfg = JsonEditor().load_json_file(config_file)
            cfg_value = cfg.get("blob_filter_radius")
            if cfg_value is not None:
                val = float(cfg_value)
                if val > 0:
                    print(f"  blob_filter_radius: loaded from site_config.json ({val:.5f})")
                    return val
        except Exception as e:
            print(f"  blob_filter_radius: could not read site_config.json ({e})")

        # Tier 3: default
        print(f"  blob_filter_radius: using default ({DEFAULT_FRACTION:.5f})")
        return DEFAULT_FRACTION

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def predict_sam2(self, predictor, image_array, multimask_output=False):
        """
        Run SAM2 prediction on an image.
        If multimask_output=True, return the highest-scoring mask among multiple candidates.
        If multimask_output=False, return the single mask directly.
        """
        predictor.set_image(image_array)

        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=multimask_output
        )

        if len(scores) == 0:
            return None, None, None

        if multimask_output:
            # Multiple masks returned, pick the best one
            best_idx = int(np.argmax(scores))
            return masks[best_idx], scores[best_idx], logits[best_idx]
        else:
            # Single mask returned, take it directly
            return masks[0], scores[0], logits[0]

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # RUN PREDICTION WITH CENTROID PROMPTS
    def predict_with_centroids(self, predictor, image_array, category_id, multimask_output=False):
        """
        Run prediction using centroids as prompts.
        Uses POSITIVE prompts from target category and NEGATIVE prompts from all other categories.
        Delegates prompt construction to the shared build_centroid_point_prompts helper
        so training, validation, and inference all use identical logic.
        """
        predictor.set_image(image_array)

        h, w = image_array.shape[:2]

        point_coords, point_labels = build_centroid_point_prompts(
            category_id=category_id,
            category_centroids=self.category_centroids,
            image_w=w,
            image_h=h,
            device=self.device,
            random_seed=42
        )

        if point_coords is None:
            print(f"Warning: No centroids found for category ID {category_id}. "
                  f"Model may not be trained for this category.")
            return None, None, None

        # Run prediction with both positive and negative prompts
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output
        )

        if len(scores) == 0:
            return None, None, None
        if multimask_output:
            best_idx = int(np.argmax(scores))
            return masks[best_idx], scores[best_idx], logits[best_idx]
        return masks[0], scores[0], logits[0]

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def show_masks(self, output_file_with_path, image, mask, borders=True, category_id=None, category_name=None, display_label=True):
        """
        Compose and save an overlay image using OpenCV — replaces the previous
        matplotlib-based implementation for a 5–20× speedup per frame.

        Args:
            output_file_with_path: full path for the output PNG
            image: RGB numpy array (H, W, 3) — PIL image converted via np.array
            mask:  binary numpy array (H, W) — 1=foreground, 0=background
            borders: if True, draw white contours around the mask
            category_id: used to look up the RGBA colour for this category
            category_name: text label drawn at the mask centroid when display_label=True
            display_label: whether to render the category name on the overlay
        """
        # image arrives as RGB from PIL — convert to BGR for OpenCV
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        mask_u8 = mask.astype(np.uint8)

        # Resolve category colour (RGBA float 0-1) → BGR uint8
        rgba = get_color_for_category(category_id)          # shape (4,) float32 0-1
        b = int(rgba[2] * 255)
        g = int(rgba[1] * 255)
        r = int(rgba[0] * 255)
        alpha = float(rgba[3]) if len(rgba) > 3 else 0.45  # blend strength

        # Blend category colour into masked region
        overlay = bgr.copy()
        colour_layer = np.zeros_like(bgr)
        colour_layer[mask_u8 == 1] = (b, g, r)
        overlay = cv2.addWeighted(overlay, 1.0 - alpha, colour_layer, alpha, 0)

        # Contours
        if borders:
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

        # Category label at centroid of largest blob
        if display_label and category_name:
            num_labels, labels = cv2.connectedComponents(mask_u8)
            if num_labels > 1:
                largest_lbl, largest_size = 1, 0
                for lbl_id in range(1, num_labels):
                    sz = int(np.sum(labels == lbl_id))
                    if sz > largest_size:
                        largest_size = sz
                        largest_lbl = lbl_id
                ys, xs = np.where(labels == largest_lbl)
                if xs.size > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    text = category_name.upper()
                    font       = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness  = 2
                    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    # Dark rounded background rectangle
                    pad = 4
                    cv2.rectangle(overlay,
                                  (cx - tw // 2 - pad, cy - th - pad),
                                  (cx + tw // 2 + pad, cy + baseline + pad),
                                  (0, 0, 0), cv2.FILLED)
                    cv2.putText(overlay, text,
                                (cx - tw // 2, cy),
                                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        cv2.imwrite(output_file_with_path, overlay)

    # ------------------------------------------------------------------------------------------------------------------
    # show_mask is retained as a thin wrapper for any callers that pass an ax —
    # internally it delegates to show_masks so there is one code path.
    # ------------------------------------------------------------------------------------------------------------------
    def show_mask(self, mask, ax, category_id=None, borders=True, category_name=None, display_label=True):
        """Legacy matplotlib-ax signature — kept for API compatibility but not called by save_outputs."""
        # This path is only hit if something outside the normal inference loop
        # calls show_mask directly with a matplotlib axis.  In that case fall
        # back to the original behaviour so callers don't break.
        color = get_color_for_category(category_id)
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = np.zeros((h, w, 4), dtype=np.float32)
        rgba_color = color.reshape((1, 1, -1))
        mask_image += mask.reshape(h, w, 1) * rgba_color
        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.squeeze()
                if contour.ndim != 2 or contour.shape[1] != 2:
                    continue
                ax.plot(contour[:, 0], contour[:, 1], linewidth=0.5, color="white")
        ax.imshow(mask_image)
        if display_label and category_name:
            num_labels, labels = cv2.connectedComponents(mask)
            if num_labels > 1:
                largest_component, largest_size = 1, 0
                for label_id in range(1, num_labels):
                    size = np.sum(labels == label_id)
                    if size > largest_size:
                        largest_size = size
                        largest_component = label_id
                ys, xs = np.where(labels == largest_component)
                if len(xs) > 0:
                    ax.text(xs.mean(), ys.mean(), category_name.upper(),
                            fontsize=7, fontweight='bold', color='white',
                            ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.25',
                                      facecolor='black', alpha=0.25, edgecolor='none'))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1],
                   color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1],
                   color='red', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h,
                                   edgecolor='green',
                                   facecolor=(0, 0, 0, 0),
                                   lw=2))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_outputs(self, image_path, pil_image, mask, prob_map, score, save_masks,
                     copy_original_image, category_id, category_name=None, display_label=True,
                     save_probability_maps=True, save_diagnostic_panels=False):
        base = os.path.splitext(os.path.basename(image_path))[0]

        # Overlay — pure OpenCV path via show_masks
        overlay_path = os.path.join(self.predictions_output_path, f"{base}_overlay.png")
        self.show_masks(overlay_path, np.array(pil_image), mask, borders=False,
                        category_id=category_id, category_name=category_name,
                        display_label=display_label)

        # Binary mask
        if save_masks:
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                    np.ones((3, 3), np.uint8))
            mask_path = os.path.join(self.predictions_output_path, f"{base}_mask.png")
            cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)

        # Original image copy
        if copy_original_image:
            try:
                shutil.copy(image_path,
                            os.path.join(self.predictions_output_path,
                                         os.path.basename(image_path)))
            except Exception as e:
                print(f"[SAM2] Could not copy original image: {e}")

        # Probability maps
        if save_probability_maps and prob_map is not None:
            self._save_heatmap(prob_map, self.predictions_output_path, base)

        # Diagnostic panel
        if save_diagnostic_panels:
            if prob_map is None:
                prob_map = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float32)
            self._save_panel(np.array(pil_image), mask, prob_map,
                             self.predictions_output_path, base)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def run_sam2_inference(self, copy_original_image, save_masks, selected_label_categories, progressBar,
                           save_probability_maps=True, save_diagnostic_panels=False):
        predictor = self.load_sam2_model()
        
        # Validate that selected categories exist in model
        if not self.TEST_MODE and selected_label_categories:
            requested_category = selected_label_categories[0]
            requested_id = requested_category.get("id")
            requested_name = requested_category.get("name", "unknown")
            
            # Check if this category has centroids in the model
            if int(requested_id) not in self.category_centroids:
                # Close progress bar before showing dialog
                if progressBar is not None and progressBar.isVisible():
                    progressBar.close()
                
                # Build available categories message
                if self.target_category_name:
                    available_msg = f"This model was trained on: '{self.target_category_name}'"
                else:
                    available_msg = f"Available category IDs in model: {list(self.category_centroids.keys())}"
                
                QMessageBox.critical(
                    None,
                    "Category Not Found in Model",
                    f"The selected category '{requested_name}' (ID: {requested_id}) was not found in the trained model.\n\n"
                    f"{available_msg}\n\n"
                    f"Please select a different model or category."
                )
                return None
            
            # Check if centroids list is empty
            if not self.category_centroids.get(int(requested_id)):
                # Close progress bar before showing dialog
                if progressBar is not None and progressBar.isVisible():
                    progressBar.close()
                
                QMessageBox.critical(
                    None,
                    "No Training Data for Category",
                    f"The selected category '{requested_name}' exists in the model but has no centroid data.\n\n"
                    f"This indicates the model was not properly trained on this category.\n\n"
                    f"Please use a different model."
                )
                return None
        
        if self.TEST_MODE:
            # === TEST MODE: Multi-category inference with colored output ===
            print("\n" + "=" * 70)
            print("TEST MODE ENABLED: Running inference on", [c["name"] for c in self.TEST_CATEGORIES])
            print("=" * 70 + "\n")
            
            return self._run_test_mode_inference(predictor, copy_original_image, save_masks, progressBar)
        else:
            # === NORMAL MODE: Standard single-category inference ===
            return self._run_normal_inference(predictor, copy_original_image, save_masks,
                                             selected_label_categories, progressBar,
                                             save_probability_maps, save_diagnostic_panels)
    
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def run_inference_on_folder(self, predictor, input_dir, output_dir,
                                copy_original_image, save_masks,
                                selected_label_categories, progressBar,
                                save_probability_maps=True, save_diagnostic_panels=False,
                                image_filter=None):
        """
        Run inference using an already-loaded predictor on a specific folder.
        Avoids reloading the model for each folder in multi-folder segmentation.

        image_filter: optional set of filenames to include. When provided only
                      those filenames are segmented; all others are skipped.
        """
        self.segmentation_images_path = input_dir
        self.predictions_output_path = output_dir
        return self._run_normal_inference(
            predictor, copy_original_image, save_masks,
            selected_label_categories, progressBar,
            save_probability_maps, save_diagnostic_panels,
            image_filter=image_filter
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _run_normal_inference(self, predictor, copy_original_image, save_masks,
                              selected_label_categories, progressBar,
                              save_probability_maps=True, save_diagnostic_panels=False,
                              image_filter=None):
        """Normal inference mode - processes single category."""
        coco_data = init_coco_structure(selected_label_categories)

        os.makedirs(self.predictions_output_path, exist_ok=True)
        VALID_EXTS = ('.jpg', '.jpeg')
        images_list = [f for f in os.listdir(self.segmentation_images_path)
                       if f.lower().endswith(VALID_EXTS)]
        if image_filter is not None:
            images_list = [f for f in images_list if f in image_filter]
            print(f"[SAM2InferenceEngine] Season filter applied: {len(images_list)} images")
        if progressBar is not None:
            progressBar.setRange(0, len(images_list) + 1)

        image_id = 1
        annotation_id = 1

        # Track success/failure counts
        images_found = 0
        images_not_found = 0
        images_processed = 0
        total_images_in_folder = len(images_list)
        target_category_name = selected_label_categories[0].get("name", "unknown") if selected_label_categories else "unknown"
        cancelled = False

        # ── Pre-loop constants (OPT 1, 2) ────────────────────────────────────
        # Centroid denormalization and radius/diagonal are invariant across
        # frames — compute once here rather than repeating per image.
        category_id   = selected_label_categories[0]["id"] if selected_label_categories else 2
        category_name = selected_label_categories[0].get("name", "unknown") if selected_label_categories else "unknown"

        positive_centroids = self.category_centroids.get(int(category_id), [])

        # Centroid pixel coords are resolved lazily on the first image so we
        # know the actual image dimensions; cached after that.
        centroid_coords_cache = None   # list of [cx_px, cy_px] or None
        radius_threshold_cache = None  # int or None

        for img_index, image in enumerate(images_list):
            # Update progress at start of iteration
            if progressBar is not None and progressBar.isVisible():
                progressBar.setValue(img_index)

            # Check for cancellation
            if progressBar is not None:
                if hasattr(progressBar, 'wasCanceled') and progressBar.wasCanceled():
                    print("Inference cancelled by user")
                    cancelled = True
                    break

            image_path = os.path.join(self.segmentation_images_path, image)
            base       = os.path.splitext(os.path.basename(image_path))[0]
            prob_map   = None

            try:
                with Image.open(image_path) as _img:
                    pil_image = _img.convert("RGB")
            except Exception as e:
                print(f"Failed to open {image_path}: {e}")
                images_not_found += 1
                images_processed += 1
                continue

            image_array = np.array(pil_image)
            prob_map    = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.float32)

            mask, score, logits = self.predict_with_centroids(
                predictor, image_array, category_id, multimask_output=False
            )
            if mask is None:
                images_not_found += 1
                images_processed += 1
                continue

            # ENSURE THE MASK MATCHES THE IMAGE SIZE
            target_h, target_w = image_array.shape[:2]
            if mask.shape != (target_h, target_w):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST)

            # ── OPT 1 & 2: resolve centroid coords + radius once ──────────────
            if positive_centroids and centroid_coords_cache is None:
                centroid_coords_cache = []
                for entry in positive_centroids:
                    if isinstance(entry, dict):
                        cx_norm, cy_norm = entry["centroid_norm"]
                    else:
                        cx_norm, cy_norm = entry
                    centroid_coords_cache.append([
                        int(round(cx_norm * (target_w - 1))),
                        int(round(cy_norm * (target_h - 1))),
                    ])
                img_diagonal         = np.sqrt(target_h ** 2 + target_w ** 2)
                radius_threshold_cache = max(10, int(self.blob_filter_radius * img_diagonal))

            # ── FILTER BLOBS NOT NEAR CENTROID PROMPTS ────────────────────────
            if centroid_coords_cache:
                labels_np = mask.astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(labels_np)
                valid_mask = np.zeros_like(labels, dtype=np.uint8)

                for lbl in range(1, num_labels):
                    ys, xs = np.nonzero(labels == lbl)
                    if xs.size == 0:
                        continue
                    cx_blob, cy_blob = xs.mean(), ys.mean()
                    for cx, cy in centroid_coords_cache:
                        if np.linalg.norm([cx - cx_blob, cy - cy_blob]) < radius_threshold_cache:
                            valid_mask[labels == lbl] = 1
                            break

                mask = valid_mask

            # OPT 5: mask.any() is faster than np.sum(mask) == 0
            if not mask.any():
                images_not_found += 1
                images_processed += 1
                continue

            # Resize logits to image size for probability map
            if logits is not None:
                prob_map = cv2.resize(logits.astype(np.float32), (target_w, target_h),
                                      interpolation=cv2.INTER_LINEAR)

            # Save overlay/mask/probability map/panel outputs
            self.save_outputs(image_path, pil_image, mask, prob_map, score, save_masks,
                              copy_original_image, category_id, category_name=category_name,
                              display_label=self.DISPLAY_LABELS,
                              save_probability_maps=save_probability_maps,
                              save_diagnostic_panels=save_diagnostic_panels)

            # COCO bookkeeping
            add_coco_entries(coco_data, image_path, mask, image_array, image_id, annotation_id)

            image_id       += 1
            annotation_id  += 1
            images_found   += 1
            images_processed += 1

            # OPT 4: periodic empty_cache removed — not needed in inference;
            # only the final cleanup below is retained.

        
        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_coco_json(coco_data, self.predictions_output_path)

        # Return stats dict — caller (MLImageSegmentation) shows the final summary
        return {
            "predictor": predictor,
            "cancelled": cancelled,
            "target_category_name": target_category_name,
            "total_images_in_folder": total_images_in_folder,
            "images_processed": images_processed,
            "images_found": images_found,
            "images_not_found": images_not_found,
        }

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _run_test_mode_inference(self, predictor, copy_original_image, save_masks, progressBar):
        """Test mode inference - processes multiple categories with colored output."""
        coco_data = init_coco_structure(self.TEST_CATEGORIES)

        os.makedirs(self.predictions_output_path, exist_ok=True)
        VALID_EXTS = ('.jpg', '.jpeg')
        images_list = [f for f in os.listdir(self.segmentation_images_path) if f.lower().endswith(VALID_EXTS)]
        if progressBar is not None:
            progressBar.setRange(0, len(images_list) + 1)

        image_id = 1
        annotation_id = 1

        for img_index, image in enumerate(images_list):
            # Check for cancellation
            if progressBar is not None:
                if hasattr(progressBar, 'wasCanceled') and progressBar.wasCanceled():
                    print("Inference cancelled by user")
                    break
                # If progress bar is closed/hidden, treat as cancellation
            image_path = os.path.join(self.segmentation_images_path, image)

            try:
                with Image.open(image_path) as _img:
                    pil_image = _img.convert("RGB")
            except Exception as e:
                print(f"Failed to open {image_path}: {e}")
                continue

            image_array = np.array(pil_image)
            
            # Create composite image for both categories
            composite_overlay = image_array.copy()
            
            # Process each category
            for category in self.TEST_CATEGORIES:
                category_id = category["id"]
                category_name = category["name"]
                category_color = category["color"]
                
                print(f"  Processing {category_name} for {os.path.basename(image_path)}")
                
                multimask_output = False
                mask, score, logits = self.predict_with_centroids(
                    predictor, image_array, category_id, multimask_output=multimask_output
                )
                
                if mask is None:
                    print(f"    No mask for {category_name}")
                    continue

                # Ensure mask matches image size
                target_h, target_w = image_array.shape[:2]
                if mask.shape != (target_h, target_w):
                    mask = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                
                # Apply category-specific color to overlay
                composite_overlay[mask == 1] = category_color
                
                # Add label text at center of largest blob
                labels_np = mask.astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(labels_np)
                if num_labels > 1:
                    # Find largest blob
                    largest_blob = 0
                    largest_size = 0
                    for lbl in range(1, num_labels):
                        size = np.sum(labels == lbl)
                        if size > largest_size:
                            largest_size = size
                            largest_blob = lbl
                    
                    if largest_blob > 0:
                        # Get center of largest blob
                        ys, xs = np.nonzero(labels == largest_blob)
                        center_x = int(xs.mean())
                        center_y = int(ys.mean())
                        
                        # Add text label
                        cv2.putText(
                            composite_overlay,
                            category_name.upper(),
                            (center_x - 50, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255, 255, 255),  # White text
                            3,
                            cv2.LINE_AA
                        )
            
            # Save the composite overlay image
            base = os.path.splitext(os.path.basename(image_path))[0]
            composite_path = os.path.join(self.predictions_output_path, f"{base}_composite.png")
            cv2.imwrite(composite_path, cv2.cvtColor(composite_overlay, cv2.COLOR_RGB2BGR))
            
            if copy_original_image:
                shutil.copy(image_path, os.path.join(self.predictions_output_path, os.path.basename(image_path)))

            image_id += 1
            annotation_id += 1
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and img_index % 10 == 0:
                torch.cuda.empty_cache()

        
        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_coco_json(coco_data, self.predictions_output_path)

        return predictor


    def _save_heatmap(self, prob_map, out_dir, base):
        prob_map = prob_map.astype(np.float32)
        min_val, max_val = prob_map.min(), prob_map.max()
        if max_val > min_val:
            norm_map = ((prob_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            norm_map = (prob_map * 255).astype(np.uint8)

        heatmap_dir = os.path.normpath(os.path.join(out_dir, "probability_maps"))
        os.makedirs(heatmap_dir, exist_ok=True)

        cv2.imwrite(os.path.join(heatmap_dir, f"{base}_heatmap_gray.png"), norm_map)
        cv2.imwrite(os.path.join(heatmap_dir, f"{base}_heatmap_jet.png"),
                    cv2.applyColorMap(norm_map, cv2.COLORMAP_JET))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_panel(self, img, pred, prob_map, out_dir, base):
        """
        2×2 composite panel using OpenCV — replaces the matplotlib version.
          [0,0] Original   [0,1] Overlay
          [1,0] Binary mask  [1,1] Probability heatmap (jet)
        """
        try:
            h, w = img.shape[:2]
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # [0,1] Overlay
            overlay = img_bgr.copy()
            colour_layer = np.zeros_like(img_bgr)
            colour_layer[pred == 1] = (255, 150, 0)   # BGR: same blue-orange as original
            overlay = cv2.addWeighted(overlay, 0.6, colour_layer, 0.4, 0)

            # [1,0] Binary mask — convert to 3-channel grey
            mask_vis = (pred.astype(np.uint8) * 255)
            mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

            # [1,1] Heatmap
            prob_f = prob_map.astype(np.float32)
            mn, mx = prob_f.min(), prob_f.max()
            norm = ((prob_f - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            heatmap_bgr = cv2.resize(heatmap_bgr, (w, h))

            # Stack into 2×2 grid
            top    = np.hstack([img_bgr, overlay])
            bottom = np.hstack([mask_bgr, heatmap_bgr])
            panel  = np.vstack([top, bottom])

            panel_dir = os.path.normpath(os.path.join(out_dir, "panels"))
            os.makedirs(panel_dir, exist_ok=True)
            cv2.imwrite(os.path.join(panel_dir, f"{base}_panel.png"), panel)

        except Exception as e:
            print(f"[SAM2] _save_panel failed for {base}: {e}")
