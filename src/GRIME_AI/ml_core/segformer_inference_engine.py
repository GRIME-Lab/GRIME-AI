#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Nov 18, 2025
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# segformer_inference_engine.py

import os
import cv2
import torch
import numpy as np
import random
import shutil
from pathlib import Path

from PIL import Image

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, prevents GUI windows
import matplotlib.pyplot as plt

import torchvision.transforms as T
from transformers import SegformerForSemanticSegmentation
from peft import LoraConfig, get_peft_model
from appcore.ml_core.ml_helpers import (init_coco_structure, add_coco_entries, save_coco_json)
from appcore.ml_core.segformer_errors import SegFormerError


VALID_BACKBONES = ("b0", "b1", "b2", "b3", "b4", "b5")
VALID_IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

# Keys whose absence (or presence with no matching layer) means the trained
# weights did not load and the model would run on untrained layers.
CRITICAL_KEY_PARTS = ("classifier", "lora_A", "lora_B", "modules_to_save")

# For a 2-class softmax, "probability >= 0.5" is the same rule as argmax.
# Used only when a checkpoint predates the saved F1-optimal threshold.
ARGMAX_EQUIVALENT_THRESHOLD = 0.5


# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===     class SegFormerInferenceEngine     ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class SegFormerInferenceEngine:
    def __init__(self, device, segformer_model, input_dir, output_dir,
                 image_size: int = 512, threshold: float = None, class_index = 1,
                 use_tta: bool = False):
        # `threshold`: None means use the F1-optimal threshold saved in the
        # checkpoint (the same threshold used for the training-report metrics).
        self.device = device
        self.SEGFORMER_MODEL = segformer_model
        self.segmentation_images_path = input_dir
        self.predictions_output_path = self._output_dir_for(output_dir)

        # Parameters
        self.image_size = image_size
        self.threshold = threshold
        self.class_index = class_index
        self.use_tta = bool(use_tta)

        # Transforms
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.to_tensor = T.ToTensor()

        # Load model (raises SegFormerError with a plain statement of the problem)
        self.model = self._load_model()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _output_dir_for(output_dir):
        return output_dir + " (segformer)"

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _classifier_out_channels(state_dict):
        """Number of classes stored in the checkpoint's classifier weights, or None."""
        for k, v in state_dict.items():
            if "decode_head.classifier" in k and k.endswith("weight") and "lora_" not in k \
                    and hasattr(v, "shape") and len(v.shape) == 4:
                return int(v.shape[0])
        return None

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _load_model(self):
        path = self.SEGFORMER_MODEL
        if not path or not os.path.isfile(path):
            raise SegFormerError(f"The SegFormer model file was not found:\n{path}")

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise SegFormerError(f"The SegFormer model file could not be read:\n{path}\n\n{e}") from e

        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise SegFormerError(f"The selected file is not a SegFormer model (it contains no model weights):\n{path}")

        base_model = ckpt.get("base_model")
        if base_model not in (None, "segformer"):
            raise SegFormerError(f"The selected model was trained with {base_model}, not SegFormer:\n{path}")

        state = ckpt["model_state_dict"]

        # Number of classes: saved value, checked against the classifier weights.
        saved_classes = ckpt.get("num_classes")
        if saved_classes is None:
            saved_classes = ckpt.get("num_labels")
        weight_classes = self._classifier_out_channels(state)
        if saved_classes is not None and weight_classes is not None and int(saved_classes) != weight_classes:
            raise SegFormerError(
                f"The model file is inconsistent: it records {saved_classes} classes, "
                f"but its classifier has {weight_classes} outputs.\n{path}")
        num_classes = saved_classes if saved_classes is not None else weight_classes
        if num_classes is None:
            raise SegFormerError(f"The number of classes could not be determined from the model file:\n{path}")
        num_classes = int(num_classes)
        if num_classes <= self.class_index:
            raise SegFormerError(
                f"The model has {num_classes} class(es), so it has no target class "
                f"(class index {self.class_index}).\n{path}")

        # Backbone size the model was trained with. Checkpoints that predate the
        # saved value were always b0.
        size = str(ckpt.get("backbone_size") or "b0").lower()
        if size not in VALID_BACKBONES:
            raise SegFormerError(f"The model file records an unknown SegFormer backbone '{size}':\n{path}")

        # LoRA or full fine-tune, as trained. Older checkpoints without the flag
        # are identified by their LoRA adapter weights.
        use_lora = ckpt.get("use_lora")
        if use_lora is None:
            use_lora = any("lora_A" in k for k in state)
        use_lora = bool(use_lora)

        print(f"[SegFormer] Model: {path}")
        print(f"[SegFormer] Backbone: {size} | classes: {num_classes} | LoRA: {use_lora}")

        pretrained = f"nvidia/segformer-{size}-finetuned-cityscapes-1024-1024"
        try:
            base = SegformerForSemanticSegmentation.from_pretrained(pretrained, ignore_mismatched_sizes=True)
        except Exception as e:
            raise SegFormerError(
                f"The SegFormer {size} base model ({pretrained}) could not be loaded. It is not in the "
                f"local model cache and could not be downloaded.\n\n{e}") from e

        base.config.num_labels = num_classes
        base.decode_head.classifier = torch.nn.Conv2d(
            base.decode_head.classifier.in_channels, num_classes, kernel_size=1)

        if use_lora:
            saved_lora = ckpt.get("lora_config")
            if saved_lora:
                lora_cfg = LoraConfig(**saved_lora)
            else:
                # Checkpoints saved before the LoRA configuration was stored.
                # If these settings do not match, the key check below stops the run.
                lora_cfg = LoraConfig(
                    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                    target_modules=["query", "key", "value", "proj"],
                    modules_to_save=["decode_head.classifier"],
                )
            model = get_peft_model(base, lora_cfg)
        else:
            model = base

        try:
            load_result = model.load_state_dict(state, strict=False)
        except RuntimeError as e:
            raise SegFormerError(
                f"The model weights do not fit a SegFormer {size} network with {num_classes} classes.\n"
                f"{path}\n\n{e}") from e

        missing = list(getattr(load_result, "missing_keys", []) or [])
        unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
        crit_missing = [k for k in missing if any(p in k for p in CRITICAL_KEY_PARTS)]
        crit_unexpected = [k for k in unexpected if any(p in k for p in CRITICAL_KEY_PARTS)]
        print(f"[SegFormer] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

        if crit_missing or crit_unexpected:
            for k in (crit_missing + crit_unexpected)[:20]:
                print(f"    UNMATCHED: {k}")
            raise SegFormerError(
                "The trained layers in the model file do not match the network built to load them, "
                f"so the trained weights could not be loaded ({len(crit_missing)} missing, "
                f"{len(crit_unexpected)} unmatched).\n{path}")

        # Decision threshold: the F1-optimal value saved at training, unless the
        # caller passed one.
        if self.threshold is None:
            saved_thr = ckpt.get("val_best_threshold")
            if saved_thr is not None:
                self.threshold = float(saved_thr)
                print(f"[SegFormer] Using saved F1-optimal threshold: {self.threshold:.4f}")
            else:
                self.threshold = ARGMAX_EQUIVALENT_THRESHOLD
                print(f"[SegFormer] No saved threshold in checkpoint; using {self.threshold}.")

        return model.to(self.device).eval()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _class_probability(self, x):
        """Target-class probability map for a normalized [1,C,H,W] input, with optional flip TTA."""
        logits = self.model(pixel_values=x).logits
        logits = torch.nn.functional.interpolate(
            logits, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)[0, self.class_index]

        if self.use_tta:
            x_flip = torch.flip(x, dims=[3])
            logits_f = self.model(pixel_values=x_flip).logits
            logits_f = torch.nn.functional.interpolate(
                logits_f, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            probs_f = torch.flip(torch.softmax(logits_f, dim=1)[0, self.class_index], dims=[1])
            probs = (probs + probs_f) / 2.0

        return probs.cpu().numpy()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def segment_image(self, image_path: str, out_path: str, gt_mask_path: str = None,
                      save_masks: bool = True, save_probability_maps: bool = True,
                      save_diagnostic_panels: bool = True):
        # --- Load original image ---
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise SegFormerError(f"The image could not be read (missing or corrupt):\n{image_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # --- Resize for model input ---
        img_resized = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        x = self.to_tensor(img_resized)
        x = self.normalize(x).unsqueeze(0).to(self.device)

        # --- Forward pass ---
        class_prob_resized = self._class_probability(x)

        # Decision rule: threshold the target-class probability. Training
        # metrics, validation overlays and the training report use the same rule
        # and the same saved threshold.
        pred_resized = (class_prob_resized >= self.threshold).astype(np.uint8)

        # --- Map predictions back to original image size ---
        pred = cv2.resize(pred_resized, (w, h), interpolation=cv2.INTER_NEAREST)
        class_prob = cv2.resize(class_prob_resized, (w, h), interpolation=cv2.INTER_LINEAR)

        # --- Prepare output paths ---
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        # --- Save outputs aligned to original image ---
        self._save_overlay(img, pred, out_path)
        if save_masks:
            self._save_mask(pred, out_dir, base)
        if save_probability_maps:
            self._save_heatmaps(torch.tensor(class_prob), out_dir, base)
        if save_diagnostic_panels:
            self._save_panel(img, pred, torch.tensor(class_prob), out_dir, base)
            self._save_components(img, pred, out_dir, base)
        if gt_mask_path:
            self._save_error_map(img, pred, gt_mask_path, out_dir, base)

        # Return for downstream COCO bookkeeping
        return pred, class_prob

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def run_segformer_inference(self, copy_original_image, save_masks, selected_label_categories, progressBar,
                                save_probability_maps=True, save_diagnostic_panels=True,
                                image_filter=None, input_dir=None, output_dir=None, cancel_check=None):
        """
        Segment every image in a folder. The model is loaded once in __init__;
        input_dir/output_dir select the folder for this call so one engine
        serves a multi-folder run. Raises SegFormerError on any failure.
        """
        if input_dir is not None:
            self.segmentation_images_path = input_dir
        if output_dir is not None:
            self.predictions_output_path = self._output_dir_for(output_dir)

        coco_data = init_coco_structure(selected_label_categories)

        try:
            os.makedirs(self.predictions_output_path, exist_ok=True)
        except OSError as e:
            raise SegFormerError(f"The output folder could not be created:\n{self.predictions_output_path}\n\n{e}") from e

        images_list = sorted(f for f in os.listdir(self.segmentation_images_path)
                             if f.lower().endswith(VALID_IMAGE_EXTS))
        if image_filter is not None:
            images_list = [f for f in images_list if f in image_filter]
        if progressBar is not None:
            progressBar.setRange(0, len(images_list) + 1)

        image_id = 1
        annotation_id = 1
        images_processed = 0
        images_found = 0
        images_not_found = 0
        cancelled = False

        # Resolve the target category name from the selection for reporting.
        target_category_name = "unknown"
        if selected_label_categories:
            first = selected_label_categories[0]
            target_category_name = first.get("name", target_category_name) \
                if isinstance(first, dict) else str(first)

        for img_index, image in enumerate(images_list):
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            if progressBar is not None and progressBar.isVisible():
                progressBar.setValue(img_index)

            image_path = os.path.join(self.segmentation_images_path, image)

            out_overlay_path = os.path.join(self.predictions_output_path, f"{Path(image).stem}_overlay.png")
            pred, water_prob = self.segment_image(
                image_path, out_overlay_path,
                save_masks=save_masks,
                save_probability_maps=save_probability_maps,
                save_diagnostic_panels=save_diagnostic_panels)

            if copy_original_image:
                shutil.copy(image_path, os.path.join(self.predictions_output_path, os.path.basename(image_path)))

            # COCO entries use the same mask that was saved and overlaid.
            image_array = np.array(Image.open(image_path).convert("RGB"))
            mask = pred.astype(np.uint8)

            add_coco_entries(coco_data, image_path, mask, image_array, image_id, annotation_id)
            images_processed += 1
            if int(mask.sum()) > 0:
                images_found += 1
            else:
                images_not_found += 1
            image_id += 1
            annotation_id += 1

        # The dispatcher owns the progress bar and closes it after all folders.

        if images_processed > 0:
            save_coco_json(coco_data, self.predictions_output_path)

        return {
            "predictor": self,
            "images_processed": images_processed,
            "images_found": images_found,
            "images_not_found": images_not_found,
            "target_category_name": target_category_name,
            "cancelled": cancelled,
            "output_dir": self.predictions_output_path,
        }

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_overlay(self, orig_img, pred, out_path):
        h, w = orig_img.shape[:2]
        pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        overlay = orig_img.copy()
        overlay[pred_resized == 1] = (0, 150, 255)

        blended = (0.6 * orig_img + 0.4 * overlay).astype(np.uint8)
        out_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        cv2.imwrite(out_path, out_bgr)
        print(f"Saved overlay: {out_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_mask(self, pred, out_dir, base):
        mask_path = os.path.join(out_dir, f"{base}_mask.png")
        cv2.imwrite(mask_path, pred * 255)
        print(f"Saved mask: {mask_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_heatmaps(self, water_prob, out_dir, base):
        prob_map = (water_prob.cpu().numpy() * 255).astype(np.uint8)

        prob_path = os.path.normpath(os.path.join(out_dir, "probability maps"))
        os.makedirs(prob_path, exist_ok=True)

        gray_path = os.path.join(prob_path, f"{base}_prob_gray.png")
        cv2.imwrite(gray_path, prob_map)
        print(f"Saved probability heatmap (gray): {gray_path}")

        jet_path = os.path.join(prob_path, f"{base}_prob_jet.png")
        cv2.imwrite(jet_path, cv2.applyColorMap(prob_map, cv2.COLORMAP_JET))
        print(f"Saved probability heatmap (jet): {jet_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_panel(self, img, pred, prob_map, out_dir, base):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0,0].imshow(img); axs[0,0].set_title("Original"); axs[0,0].axis("off")
        overlay = img.copy(); overlay[pred == 1] = (0, 150, 255)

        blended = (0.6 * img + 0.4 * overlay).astype(np.uint8)

        axs[0,1].imshow(blended); axs[0,1].set_title("Overlay"); axs[0,1].axis("off")
        axs[1,0].imshow(pred, cmap="gray"); axs[1,0].set_title("Binary Mask"); axs[1,0].axis("off")
        axs[1,1].imshow(prob_map.cpu().numpy(), cmap="jet"); axs[1,1].set_title("Probability Heatmap"); axs[1,1].axis("off")

        panel_path = os.path.normpath(os.path.join(out_dir, "panels"))
        os.makedirs(panel_path, exist_ok=True)

        output_file = os.path.join(panel_path, f"{base}_panel.png")

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        print(f"Saved side-by-side panel: {panel_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_error_map(self, img, pred, gt_path, out_dir, base):
        if not os.path.exists(gt_path):
            return
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        gt = (gt > 127).astype(np.uint8)
        error_map = np.zeros_like(img)
        error_map[(gt == 1) & (pred == 1)] = (0, 255, 0)
        error_map[(gt == 1) & (pred == 0)] = (255, 0, 0)
        error_map[(gt == 0) & (pred == 1)] = (255, 0, 255)
        error_path = os.path.join(out_dir, f"{base}_error.png")
        cv2.imwrite(error_path, cv2.cvtColor(error_map, cv2.COLOR_RGB2BGR))
        print(f"Saved error map: {error_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_components(self, img, pred, out_dir, base):
        num_labels, labels = cv2.connectedComponents(pred)
        cc_vis = np.zeros_like(img)
        for lbl in range(1, num_labels):
            mask = labels == lbl
            color = [random.randint(0,255) for _ in range(3)]
            cc_vis[mask] = color

        component_path = os.path.normpath(os.path.join(out_dir, "mask components"))
        os.makedirs(component_path, exist_ok=True)

        cc_path = os.path.join(component_path, f"{base}_components.png")

        cv2.imwrite(cc_path, cv2.cvtColor(cc_vis, cv2.COLOR_RGB2BGR))
        print(f"Saved connected components visualization: {cc_path}")
