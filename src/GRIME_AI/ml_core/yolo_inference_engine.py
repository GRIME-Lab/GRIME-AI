#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# yolo_inference_engine.py

import os
import cv2
import random
import shutil
import numpy as np

from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from PyQt5.QtWidgets import QMessageBox

from GRIME_AI.ml_core.ml_helpers import (
    init_coco_structure, add_coco_entries, save_coco_json
)
from ..app_identity import APP_DISPLAY_NAME


# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===     class YOLOInferenceEngine          ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class YOLOInferenceEngine:
    """
    YOLOv11-seg inference engine for GRIME AI.

    Mirrors SAM2InferenceEngine and SegFormerInferenceEngine in interface and output
    structure so ML_Segmentation_Dispatcher can use all three identically.

    Output folder: <output_dir> (yolo)
    Per-image outputs:
        <base>_overlay.png
        <base>_mask.png          (if save_masks)
        panels/<base>_panel.png
        heatmaps/<base>_heatmap_gray.png
        heatmaps/<base>_heatmap_jet.png
        mask components/<base>_components.png
        <original image>         (if copy_original_image)
    COCO JSON: instances_default.json in output folder
    """

    def __init__(self, device, yolo_model_path: str, input_dir: str, output_dir: str,
                 confidence_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Args:
            device:               torch device (cuda/cpu)
            yolo_model_path:      path to the GRIME AI-trained YOLO best.pt checkpoint
            input_dir:            directory containing input images
            output_dir:           base output directory (engine appends ' (yolo)')
            confidence_threshold: minimum confidence for accepting a detection
            iou_threshold:        NMS IoU threshold
        """
        self.device              = device
        self.YOLO_MODEL          = yolo_model_path
        self.segmentation_images_path  = input_dir
        self.predictions_output_path   = output_dir + " (yolo)"
        self.confidence_threshold      = confidence_threshold
        self.iou_threshold             = iou_threshold
        self.cat_id_to_yolo            = {}   # populated by _load_model from checkpoint

        self.model = self._load_model()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _load_model(self):
        """
        Load the trained YOLO model from the GRIME AI .torch checkpoint.

        The .torch file contains GRIME AI metadata (categories, site_name,
        training stats, etc.) plus the path to best.pt which is the complete
        fine-tuned Ultralytics model. We load best.pt directly with YOLO()
        since it is already the fully trained model with the correct
        architecture and custom class count baked in.
        """
        if not self.YOLO_MODEL or not os.path.exists(self.YOLO_MODEL):
            raise FileNotFoundError(
                f"[YOLOInferenceEngine] Model checkpoint not found: {self.YOLO_MODEL}"
            )

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "[YOLOInferenceEngine] ultralytics is not installed. "
                "Run: pip install ultralytics"
            )

        # PyTorch 2.6 compatibility patch
        try:
            import ultralytics.nn.tasks as _ult_tasks

            def _patched_torch_safe_load(weight):
                ckpt = torch.load(weight, map_location="cpu", weights_only=False)
                return ckpt, weight

            _ult_tasks.torch_safe_load = _patched_torch_safe_load
        except Exception as e:
            print(f"[YOLOInferenceEngine] Could not apply torch_safe_load patch: {e}")

        # Load GRIME AI .torch file for metadata
        grime_ckpt = torch.load(self.YOLO_MODEL, map_location="cpu", weights_only=False)

        if not isinstance(grime_ckpt, dict) or "categories" not in grime_ckpt:
            raise ValueError(
                f"[YOLOInferenceEngine] {self.YOLO_MODEL} does not appear to be a "
                f"{APP_DISPLAY_NAME} YOLO checkpoint."
            )

        # Print metadata
        print("\n=== YOLO Checkpoint Information ===")
        print(f"Model path:    {self.YOLO_MODEL}")
        print(f"Site:          {grime_ckpt.get('site_name', 'N/A')}")
        print(f"Created:       {grime_ckpt.get('creation_UTC', 'N/A')}")
        print(f"Epochs:        {grime_ckpt.get('epochs', 'N/A')}")
        print(f"Learning rate: {grime_ckpt.get('learning_rate', 'N/A')}")
        print(f"Val loss:      {grime_ckpt.get('val_loss', 'N/A')}")
        print(f"Val accuracy:  {grime_ckpt.get('val_accuracy', 'N/A')}")
        print(f"mIoU:          {grime_ckpt.get('miou', 'N/A')}")
        cats = grime_ckpt.get("categories", [])
        print(f"Categories:    {[c.get('name') for c in cats] if cats else 'N/A'}")

        # Store metadata
        self.categories     = cats
        self.class_names    = grime_ckpt.get("class_names", {})
        self.num_classes    = grime_ckpt.get("num_classes", len(cats))
        self.cat_id_to_yolo = grime_ckpt.get("cat_id_to_yolo", {})

        # Reconstruct categories from class_names if missing
        if not self.categories and self.class_names:
            yolo_to_coco = {v: k for k, v in self.cat_id_to_yolo.items()}
            self.categories = [
                {"id": yolo_to_coco.get(idx, idx + 1), "name": name}
                for idx, name in sorted(self.class_names.items())
            ]
            print(f"[YOLOInferenceEngine] Reconstructed categories: {self.categories}")

        # Resolve best.pt — stored in metadata or sits alongside the .torch file
        best_pt_path = grime_ckpt.get("best_pt_path", "")
        if not best_pt_path or not os.path.exists(best_pt_path):
            # Fall back: best.pt is always in the same weights/ folder as the .torch
            torch_dir = Path(self.YOLO_MODEL).parent
            best_pt_path = str(torch_dir / "best.pt")

        if not os.path.exists(best_pt_path):
            raise FileNotFoundError(
                f"[YOLOInferenceEngine] best.pt not found at: {best_pt_path}\n"
                f"Expected alongside the .torch file in: {Path(self.YOLO_MODEL).parent}"
            )

        print(f"Loading best.pt: {best_pt_path}")
        model = YOLO(best_pt_path)
        print("=== Model ready for inference ===\n")
        return model

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def segment_image(self, image_path: str):
        """
        Run YOLO inference on a single image.

        Returns:
            combined_mask:  H x W uint8 binary mask (1 = any foreground class)
            conf_map:       H x W float32 confidence heatmap (max confidence per pixel)
            class_mask:     H x W int32 class index map (-1 = background)
            results:        raw ultralytics Results object
        """
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"[YOLOInferenceEngine] Image not found: {image_path}")
        h, w = img_bgr.shape[:2]

        results = self.model(
            image_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )[0]

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        conf_map      = np.zeros((h, w), dtype=np.float32)
        class_mask    = np.full((h, w), -1, dtype=np.int32)

        if results.masks is not None and len(results.masks) > 0:
            masks_data = results.masks.data.cpu().numpy()   # N x H_m x W_m
            confs      = results.boxes.conf.cpu().numpy()   # N
            classes    = results.boxes.cls.cpu().numpy().astype(int)  # N

            for seg_mask, conf, cls_idx in zip(masks_data, confs, classes):
                # Resize mask to original image dimensions
                seg_resized = cv2.resize(
                    seg_mask.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR
                )
                binary = (seg_resized > 0.5).astype(np.uint8)

                # Update combined mask and confidence/class maps
                update = (binary == 1) & (seg_resized > conf_map)
                combined_mask = np.maximum(combined_mask, binary)
                conf_map[update] = seg_resized[update] * float(conf)
                class_mask[update] = cls_idx

        return combined_mask, conf_map, class_mask, results

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def run_yolo_inference(self, copy_original_image: bool, save_masks: bool,
                           selected_label_categories: list, progressBar) -> "YOLOInferenceEngine":
        """
        Main inference loop. Mirrors SAM2InferenceEngine.run_sam2_inference and
        SegFormerInferenceEngine.run_segformer_inference in signature and return value.

        Returns self so ML_Segmentation_Dispatcher receives the engine instance.
        """
        coco_data = init_coco_structure(selected_label_categories)

        os.makedirs(self.predictions_output_path, exist_ok=True)
        VALID_EXTS = ('.jpg', '.jpeg', '.png')
        images_list = sorted([
            f for f in os.listdir(self.segmentation_images_path)
            if f.lower().endswith(VALID_EXTS)
        ])

        if progressBar is not None:
            progressBar.setRange(0, len(images_list) + 1)

        image_id      = 1
        annotation_id = 1
        images_found     = 0
        images_not_found = 0
        cancelled = False

        # Build set of YOLO class indices for selected categories
        # cat_id_to_yolo maps COCO ID → YOLO 0-based index
        # class_mask contains YOLO indices so we must compare in YOLO space
        if selected_label_categories:
            selected_yolo_indices = set()
            for cat in selected_label_categories:
                coco_id = cat["id"]
                if self.cat_id_to_yolo:
                    yolo_idx = self.cat_id_to_yolo.get(coco_id)
                    if yolo_idx is not None:
                        selected_yolo_indices.add(yolo_idx)
                else:
                    # Fallback: assume YOLO index = COCO ID - 1
                    selected_yolo_indices.add(coco_id - 1)
            selected_ids = selected_yolo_indices if selected_yolo_indices else None
        else:
            selected_ids = None

        target_name = (
            ", ".join(c.get("name", "?") for c in selected_label_categories)
            if selected_label_categories else "all classes"
        )

        for img_index, image_file in enumerate(images_list):
            # Progress + cancellation check (mirrors SAM2 pattern)
            if progressBar is not None:
                if not progressBar.isVisible():
                    print("[YOLOInferenceEngine] Inference stopped — progress bar closed.")
                    cancelled = True
                    break
                progressBar.setValue(img_index)

            image_path = os.path.join(self.segmentation_images_path, image_file)
            base       = os.path.splitext(image_file)[0]

            try:
                pil_image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"[YOLOInferenceEngine] Failed to open {image_path}: {e}")
                images_not_found += 1
                continue

            image_array = np.array(pil_image)

            try:
                combined_mask, conf_map, class_mask, results = self.segment_image(image_path)
            except Exception as e:
                print(f"[YOLOInferenceEngine] Inference failed for {image_file}: {e}")
                images_not_found += 1
                continue

            # Filter to selected categories only
            if selected_ids is not None:
                # Zero out pixels belonging to non-selected classes
                filter_mask = np.isin(class_mask, list(selected_ids))
                combined_mask = (combined_mask & filter_mask).astype(np.uint8)
                conf_map      = conf_map * filter_mask.astype(np.float32)

            if np.sum(combined_mask) == 0:
                # Save diagnostic panel with empty mask
                self._save_panel(image_array, combined_mask, conf_map,
                                 self.predictions_output_path, base)
                images_not_found += 1
                continue

            # Save combined outputs
            self._save_overlay(image_array, combined_mask, self.predictions_output_path, base)
            if save_masks:
                self._save_mask(combined_mask, self.predictions_output_path, base)
            self._save_heatmaps(conf_map, self.predictions_output_path, base)
            self._save_panel(image_array, combined_mask, conf_map,
                             self.predictions_output_path, base)
            self._save_components(image_array, combined_mask,
                                  self.predictions_output_path, base)

            # Per-class masks — one subfolder per selected category
            if len(selected_label_categories) > 1:
                for cat in selected_label_categories:
                    coco_id  = cat["id"]
                    cls_name = cat.get("name", str(coco_id))
                    # Resolve to YOLO index for class_mask comparison
                    if self.cat_id_to_yolo:
                        yolo_idx = self.cat_id_to_yolo.get(coco_id)
                    else:
                        yolo_idx = coco_id - 1
                    if yolo_idx is None:
                        continue
                    cls_mask = ((class_mask == yolo_idx) & (combined_mask == 1)).astype(np.uint8)
                    if np.sum(cls_mask) > 0:
                        self._save_class_mask(cls_mask, self.predictions_output_path,
                                              base, cls_name)

            if copy_original_image:
                shutil.copy(image_path,
                            os.path.join(self.predictions_output_path, image_file))

            # COCO bookkeeping
            add_coco_entries(coco_data, image_path, combined_mask, image_array,
                             image_id, annotation_id)
            image_id      += 1
            annotation_id += 1
            images_found  += 1

            # Periodic GPU cache clear (mirrors SAM2 pattern)
            if torch.cuda.is_available() and img_index % 10 == 0:
                torch.cuda.empty_cache()

        if progressBar is not None and progressBar.isVisible():
            progressBar.close()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_coco_json(coco_data, self.predictions_output_path)

        # Summary dialog (mirrors SAM2 pattern)
        total = images_found + images_not_found
        if cancelled:
            QMessageBox.warning(
                None, "Inference Cancelled",
                f"Inference was cancelled by user.\n\n"
                f"Results for '{target_name}':\n"
                f"- Images processed: {total}\n"
                f"- Detections found: {images_found}\n"
                f"- No detection: {images_not_found}"
            )
        elif images_found == 0:
            QMessageBox.warning(
                None, "No Segmentations Found",
                f"No detections were produced for any of the {total} images processed.\n\n"
                f"Check that the model was trained on the selected categories and that "
                f"the confidence threshold ({self.confidence_threshold}) is appropriate."
            )
        elif images_not_found == 0:
            QMessageBox.information(
                None, "Inference Complete",
                f"Detections found in all {images_found} images."
            )
        else:
            QMessageBox.information(
                None, "Inference Complete",
                f"Segmentation results for '{target_name}':\n\n"
                f"- Detections found: {images_found} images\n"
                f"- No detection: {images_not_found} images\n"
                f"- Total processed: {total} images"
            )

        return self

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_overlay(self, img: np.ndarray, mask: np.ndarray, out_dir: str, base: str):
        """Save a blended overlay of the mask on the original image."""
        overlay = img.copy()
        overlay[mask == 1] = (0, 150, 255)
        blended = (0.6 * img + 0.4 * overlay).astype(np.uint8)
        out_path = os.path.join(out_dir, f"{base}_overlay.png")
        cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        print(f"Saved overlay: {out_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_mask(self, mask: np.ndarray, out_dir: str, base: str):
        """Save the binary mask as a grayscale PNG."""
        mask_path = os.path.join(out_dir, f"{base}_mask.png")
        cv2.imwrite(mask_path, mask * 255)
        print(f"Saved mask: {mask_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_heatmaps(self, conf_map: np.ndarray, out_dir: str, base: str):
        """Save grayscale and jet-colormap confidence heatmaps."""
        prob_dir = os.path.normpath(os.path.join(out_dir, "heatmaps"))
        os.makedirs(prob_dir, exist_ok=True)

        # Normalize to 0-255
        min_val, max_val = conf_map.min(), conf_map.max()
        if max_val > min_val:
            norm = ((conf_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            norm = (conf_map * 255).astype(np.uint8)

        gray_path = os.path.join(prob_dir, f"{base}_heatmap_gray.png")
        cv2.imwrite(gray_path, norm)

        jet_path = os.path.join(prob_dir, f"{base}_heatmap_jet.png")
        cv2.imwrite(jet_path, cv2.applyColorMap(norm, cv2.COLORMAP_JET))

        print(f"Saved heatmaps: {gray_path}, {jet_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_panel(self, img: np.ndarray, mask: np.ndarray,
                    conf_map: np.ndarray, out_dir: str, base: str):
        """
        Save a 2x2 composite panel matching SAM2 and SegFormer panels exactly:
          [0,0] Original  [0,1] Overlay
          [1,0] Mask      [1,1] Confidence heatmap
        """
        try:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))

            axs[0, 0].imshow(img)
            axs[0, 0].set_title("Original")
            axs[0, 0].axis("off")

            overlay = img.copy()
            overlay[mask == 1] = (0, 150, 255)
            blended = (0.6 * img + 0.4 * overlay).astype(np.uint8)
            axs[0, 1].imshow(blended)
            axs[0, 1].set_title("Overlay")
            axs[0, 1].axis("off")

            axs[1, 0].imshow(mask, cmap="gray")
            axs[1, 0].set_title("Binary Mask")
            axs[1, 0].axis("off")

            axs[1, 1].imshow(conf_map, cmap="jet")
            axs[1, 1].set_title("Confidence Heatmap")
            axs[1, 1].axis("off")

            panel_dir = os.path.normpath(os.path.join(out_dir, "panels"))
            os.makedirs(panel_dir, exist_ok=True)
            panel_path = os.path.join(panel_dir, f"{base}_panel.png")

            plt.tight_layout()
            plt.savefig(panel_path)
            print(f"Saved panel: {panel_path}")
        finally:
            plt.close()

    def _save_class_mask(self, mask: np.ndarray, out_dir: str,
                         base: str, class_name: str):
        """
        Save a per-class binary mask to a dedicated subfolder.
        Only called when multiple categories are selected — produces
        one subfolder per class named 'masks_<class_name>'.
        """
        cls_dir = os.path.normpath(os.path.join(out_dir, f"masks_{class_name}"))
        os.makedirs(cls_dir, exist_ok=True)
        path = os.path.join(cls_dir, f"{base}_{class_name}_mask.png")
        cv2.imwrite(path, mask * 255)
        print(f"Saved class mask: {path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_components(self, img: np.ndarray, mask: np.ndarray,
                         out_dir: str, base: str):
        """Save a connected-components visualization with random per-component colors."""
        num_labels, labels = cv2.connectedComponents(mask)
        cc_vis = np.zeros_like(img)
        for lbl in range(1, num_labels):
            color = [random.randint(0, 255) for _ in range(3)]
            cc_vis[labels == lbl] = color

        comp_dir = os.path.normpath(os.path.join(out_dir, "mask components"))
        os.makedirs(comp_dir, exist_ok=True)
        cc_path = os.path.join(comp_dir, f"{base}_components.png")
        cv2.imwrite(cc_path, cv2.cvtColor(cc_vis, cv2.COLOR_RGB2BGR))
        print(f"Saved components: {cc_path}")
