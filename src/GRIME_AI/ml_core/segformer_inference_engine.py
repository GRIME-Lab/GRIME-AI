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
from GRIME_AI.ml_core.ml_helpers import (init_coco_structure, add_coco_entries, save_coco_json)


# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===     class SegFormerInferenceEngine     ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class SegFormerInferenceEngine:
    def __init__(self, device, segformer_model, input_dir, output_dir,
                 image_size: int = 512, threshold: float = None, class_index = 1):
        # `threshold` is retained for call-site compatibility but unused: the
        # decision rule is argmax over classes, matching segformer_trainer.
        # Original attributes
        self.device = device
        self.SEGFORMER_MODEL = segformer_model
        self.segmentation_images_path = input_dir
        self.predictions_output_path = output_dir + " (segformer)"

        # Parameters
        self.image_size = image_size
        self.threshold = threshold
        self.class_index = class_index

        # Transforms
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.to_tensor = T.ToTensor()

        # Load model
        self.model = self._load_model()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _load_model(self):
        if not self.SEGFORMER_MODEL or not os.path.exists(self.SEGFORMER_MODEL):
            raise FileNotFoundError(f"SegFormer model checkpoint not found: {self.SEGFORMER_MODEL}")

        # Load checkpoint first to get num_classes
        ckpt = torch.load(self.SEGFORMER_MODEL, map_location="cpu", weights_only=False)

        # Get num_classes - explicit None checks
        num_classes = ckpt.get("num_classes")
        if num_classes is None:
            num_classes = ckpt.get("num_labels")
        if num_classes is None:
            num_classes = 2
        
        print(f"=== Inference Engine Debug ===")
        print(f"Checkpoint path: {self.SEGFORMER_MODEL}")
        print(f"num_classes from ckpt: {ckpt.get('num_classes')} (type: {type(ckpt.get('num_classes'))})")
        print(f"num_labels from ckpt: {ckpt.get('num_labels')}")
        print(f"FINAL num_classes: {num_classes} (type: {type(num_classes)})")
        print(f"=============================")

        # Build base model with correct num_classes
        base = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            ignore_mismatched_sizes=True
        )
        base.config.num_labels = int(num_classes)  # Ensure it's an integer
        base.decode_head.classifier = torch.nn.Conv2d(
            base.decode_head.classifier.in_channels, int(num_classes), kernel_size=1
        )

        print(f"Built classifier with {base.decode_head.classifier.out_channels} output channels")

        # Rebuild the LoRA config from the checkpoint so adapter key names match
        # exactly what training saved. Falls back to the legacy hardcoded config
        # only for old checkpoints that predate the saved lora_config, and warns
        # so the mismatch is visible rather than silent.
        saved_lora = ckpt.get("lora_config")
        if saved_lora:
            lora_cfg = LoraConfig(**saved_lora)
            print(f"[LoRA] Rebuilt config from checkpoint: "
                  f"target_modules={saved_lora.get('target_modules')}")
        else:
            lora_cfg = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                target_modules=["query", "key", "value", "proj"],
                modules_to_save=["decode_head.classifier"],
            )
            print("[LoRA] WARNING: checkpoint has no saved lora_config "
                  "(pre-fix checkpoint). Using legacy hardcoded target_modules; "
                  "adapter keys may not match. Retrain to embed the config.")
        model = get_peft_model(base, lora_cfg)

        print(f"Applied LoRA wrapper")

        # Load state dict
        load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)

        # strict=False silently drops weights whose key names don't match the
        # reconstructed (LoRA-wrapped) model. If the trained classifier head or
        # LoRA adapters land in missing_keys, they never load -> the model runs
        # on random/pretrained init and predicts background everywhere (empty
        # masks) with no error raised. Surface that here.
        missing = list(getattr(load_result, "missing_keys", []) or [])
        unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
        print(f"[load_state_dict] missing_keys: {len(missing)}, "
              f"unexpected_keys: {len(unexpected)}")

        def _flag(keys, needles):
            return [k for k in keys if any(n in k for n in needles)]

        crit_missing = _flag(missing, ("classifier", "lora_A", "lora_B",
                                       "modules_to_save"))
        if crit_missing:
            print("[load_state_dict] *** CRITICAL: trained weights NOT loaded "
                  "(these keys are missing from the checkpoint match):")
            for k in crit_missing[:20]:
                print(f"    MISSING: {k}")
            if len(crit_missing) > 20:
                print(f"    ... and {len(crit_missing) - 20} more")
            print("[load_state_dict] The model is running on un-trained weights "
                  "for these layers -> empty/garbage masks are expected.")
        if unexpected:
            crit_unexpected = _flag(unexpected, ("classifier", "lora_A", "lora_B",
                                                 "modules_to_save"))
            for k in crit_unexpected[:20]:
                print(f"    UNEXPECTED (in ckpt, no home in model): {k}")

        print(f"Successfully loaded checkpoint")

        # Adopt the F1-optimal probability threshold saved at training, unless the
        # caller explicitly passed one. Falls back to 0.5 for pre-fix checkpoints.
        if self.threshold is None:
            saved_thr = ckpt.get("val_best_threshold")
            if saved_thr is not None:
                self.threshold = float(saved_thr)
                print(f"[threshold] Using saved F1-optimal threshold: {self.threshold:.4f}")
            else:
                print("[threshold] No saved threshold in checkpoint; defaulting to 0.5.")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return model.to(device).eval()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def segment_image(self, image_path: str, out_path: str, gt_mask_path: str = None):
        # --- Load original image ---
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(image_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # --- Resize for model input ---
        img_resized = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        x = self.to_tensor(img_resized)
        x = self.normalize(x).unsqueeze(0).to(str(self.device))

        # --- Forward pass ---
        logits = self.model(pixel_values=x).logits
        logits = torch.nn.functional.interpolate(
            logits, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )

        probs = torch.softmax(logits, dim=1)

        # Target-class probability map.
        class_prob_resized = probs[0, self.class_index].cpu().numpy()

        # Decision rule: threshold the target-class probability, NOT argmax.
        # For a minority class (sandbar is a small fraction of pixels) argmax
        # almost never selects it -- background wins per-pixel -- so an argmax
        # mask comes out empty even when the model ranks the class well
        # (high mIoU). Thresholding the class probability is the correct binary
        # segmentation rule. self.threshold is the F1-optimal value saved at
        # training; fall back to 0.5 only if the checkpoint predates it.
        thr = self.threshold if self.threshold is not None else 0.5
        pred_resized = (class_prob_resized >= thr).astype(np.uint8)

        # --- Map predictions back to original image size ---
        pred = cv2.resize(pred_resized, (w, h), interpolation=cv2.INTER_NEAREST)
        class_prob = cv2.resize(class_prob_resized, (w, h), interpolation=cv2.INTER_LINEAR)

        # --- Prepare output paths ---
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        # --- Save outputs aligned to original image ---
        self._save_overlay(img, pred, out_path)
        self._save_mask(pred, out_dir, base)
        self._save_heatmaps(torch.tensor(class_prob), out_dir, base)
        self._save_panel(img, pred, torch.tensor(class_prob), out_dir, base)
        if gt_mask_path:
            self._save_error_map(img, pred, gt_mask_path, out_dir, base)
        self._save_components(img, pred, out_dir, base)

        # Return for downstream COCO bookkeeping
        return pred, class_prob

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def run_segformer_inference(self, copy_original_image, save_masks, selected_label_categories, progressBar):
        coco_data = init_coco_structure(selected_label_categories)

        os.makedirs(self.predictions_output_path, exist_ok=True)
        VALID_EXTS = ('.jpg', '.jpeg')
        images_list = [f for f in os.listdir(self.segmentation_images_path) if f.lower().endswith(VALID_EXTS)]
        if progressBar is not None:
            progressBar.setRange(0, len(images_list) + 1)

        image_id = 1
        annotation_id = 1
        images_processed = 0
        images_found = 0
        images_not_found = 0
        # Resolve the target category name from the selection for reporting.
        target_category_name = "sandbar"
        try:
            if selected_label_categories:
                first = selected_label_categories[0]
                target_category_name = first.get("name", target_category_name) \
                    if isinstance(first, dict) else str(first)
        except Exception:
            pass

        for img_index, image in enumerate(images_list):
            if progressBar is not None and progressBar.isVisible():
                progressBar.setValue(img_index)

            image_path = os.path.join(self.segmentation_images_path, image)

            # Engine overlays/masks
            out_overlay_path = os.path.join(self.predictions_output_path, f"{Path(image).stem}_overlay.png")
            pred, water_prob = self.segment_image(image_path, out_overlay_path)

            # Optional copy of original image
            if copy_original_image:
                shutil.copy(image_path, os.path.join(self.predictions_output_path, os.path.basename(image_path)))

            # COCO entries
            pil_image = Image.open(image_path).convert("RGB")
            image_array = np.array(pil_image)

            # COCO export uses the same argmax mask that was saved/overlaid, so
            # every artifact from this engine agrees with training/validation.
            mask = pred.astype(np.uint8)
            score = float(np.mean(water_prob))  # informational only

            add_coco_entries(coco_data, image_path, mask, image_array, image_id, annotation_id)
            images_processed += 1
            if int(mask.sum()) > 0:
                images_found += 1
            else:
                images_not_found += 1
            image_id += 1
            annotation_id += 1

        if progressBar is not None and progressBar.isVisible():
            progressBar.close()

        save_coco_json(coco_data, self.predictions_output_path)
        # Return a stats dict (not self) so ML_Segmentation_Dispatcher's
        # `isinstance(result, dict)` accumulation works — otherwise it reports
        # 0 images / "unknown" / a false cancellation.
        return {
            "predictor": self,
            "images_processed": images_processed,
            "images_found": images_found,
            "images_not_found": images_not_found,
            "target_category_name": target_category_name,
            "cancelled": False,
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
