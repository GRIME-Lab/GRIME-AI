#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# MLImageSegmentation.py

import os
from pathlib import Path
import sys
import json
import shutil
import importlib.util

import numpy as np
import torch
from PIL import Image
from PyQt5.QtWidgets import QMessageBox

from omegaconf import OmegaConf, DictConfig

# Project imports
from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils

# Engines
from GRIME_AI.ml_core.sam2_inference_engine import SAM2InferenceEngine
from GRIME_AI.ml_core.segformer_inference_engine import SegFormerInferenceEngine
from GRIME_AI.ml_core.yolo_inference_engine import YOLOInferenceEngine
from GRIME_AI.ml_core.ml_helpers import add_coco_entries


# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===       class MLImageSegmentation        ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class MLImageSegmentation:
    def __init__(self, cfg: DictConfig = None, parent_widget=None):
        self.className = "MLImageSegmentation"
        self.progress_bar_closed = False
        self.parent_widget = parent_widget

        # Load config (same as before)
        if cfg is None or "load_model" not in cfg:
            settings_folder = os.path.normpath(GRIME_AI_Save_Utils().get_settings_folder())
            CONFIG_FILENAME = "site_config.json"
            config_file = os.path.normpath(os.path.join(settings_folder, CONFIG_FILENAME))
            with open(config_file, 'r') as file:
                self.config = json.load(file).get("load_model", {})
        else:
            self.config = OmegaConf.to_container(cfg.load_model, resolve=True)

        main_dir = os.path.dirname(importlib.util.find_spec('sam2').origin)
        self.SAM2_CHECKPOINT = os.path.normpath(os.path.join(main_dir, self.config.get("SAM2_CHECKPOINT", "")))
        self.MODEL_CFG = os.path.normpath(self.config.get("MODEL_CFG", ""))

        self.segmentation_images_path = os.path.normpath(self.config.get("segmentation_images_path", ""))
        self.predictions_output_path = os.path.normpath(self.config.get("predictions_output_path", ""))

        # Multi-folder support: list of {"path": str, "recursive": bool} — each entry is already a flat folder
        self.segmentation_image_folders = self.config.get("segmentation_image_folders", [])
        if not self.segmentation_image_folders and self.segmentation_images_path:
            # Fall back to single-folder mode for backwards compatibility
            self.segmentation_image_folders = [{"path": self.segmentation_images_path, "recursive": False}]

        self.SAM2_MODEL = os.path.normpath(self.config.get("SAM2_MODEL", ""))
        self.SAM3_MODEL = os.path.normpath(self.config.get("SAM3_MODEL", ""))
        self.SEGFORMER_MODEL = os.path.normpath(self.config.get("SEGFORMER_MODEL", ""))
        self.YOLO_MODEL = os.path.normpath(self.config.get("YOLO_MODEL", ""))

        if self.SAM2_CHECKPOINT == "" or self.MODEL_CFG == "" or self.segmentation_images_path == "" or self.predictions_output_path == "" or (self.SAM2_MODEL == "" and self.SEGFORMER_MODEL == ""):
            print("ERROR: Configuration file missing items.")

        self._check_for_required_files()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _check_for_required_files(self):
        nError = 0
        # Always check the input directory
        # Check the model file that is actually configured — not always SAM2_MODEL
        model_path = ""
        for key in ("SAM2_MODEL", "SEGFORMER_MODEL", "YOLO_MODEL"):
            val = getattr(self, key, "")
            if val and val != os.path.normpath(""):
                model_path = val
                break
        paths_to_check = [
            ("Input directory",   self.segmentation_images_path),
            ("Trained model file", model_path),
        ]
        self.missing_items = [(name, path) for name, path in paths_to_check if not os.path.exists(path)]
        if self.missing_items:
            nError = -1
            self.show_missing_files_dialog(self.missing_items)
        return nError

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def show_missing_files_dialog(self, missing_items):
        lines = [f"{name}: {path}" for name, path in missing_items]
        full_msg = "The following files or directories are missing or have been moved:\n\n" + "\n".join(lines) + "\n"
        msgBox = GRIME_AI_QMessageBox('Model Configuration Error', full_msg, QMessageBox.Close, icon=QMessageBox.Critical)
        msgBox.displayMsgBox()

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    def ML_Segmentation_Dispatcher(self, copy_original_image, save_masks, selected_label_categories, mode="segformer",
                                    save_probability_maps=True, save_diagnostic_panels=False):
        if self.missing_items:
            return

        self.progress_bar_closed = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_classes = len(selected_label_categories) + 1  # +1 for background

        # Compute common root once for output mirroring
        all_paths = [os.path.normpath(e["path"]) for e in self.segmentation_image_folders]
        common_root = os.path.commonpath(all_paths) if len(all_paths) > 1 else os.path.dirname(all_paths[0])

        # Accumulated stats across all folders
        total_found = 0
        total_not_found = 0
        total_processed = 0
        target_category_name = "unknown"
        final_predictor = None
        user_cancelled = False

        # Single progress bar for the entire run
        self.progress_bar_closed = False
        progressBar = QProgressWheel(
            title="Segmenting images...", total=1,
            on_close=lambda: setattr(self, "progress_bar_closed", True),
            parent=self.parent_widget
        )

        # Load model once before the loop
        engine = None
        predictor = None
        if mode.lower() == "sam2":
            engine = SAM2InferenceEngine(
                device=torch.device("cuda"),
                model_cfg="sam2.1_hiera_l.yaml",
                trained_checkpoint_path=self.SAM2_MODEL,
                input_dir=self.segmentation_images_path,
                output_dir=self.predictions_output_path
            )
            # Validate categories before starting
            predictor = engine.load_sam2_model()
            if not self._validate_sam2_categories(engine, selected_label_categories, progressBar):
                return None

        # ── Season filter ─────────────────────────────────────────────────────
        segment_seasons      = self.config.get("segment_seasons", [])
        season_filter_active = bool(segment_seasons)
        if season_filter_active:
            try:
                from GRIME_AI.ml_core.seasonal_only import filter_seasons_only
                print(f"[MLImageSegmentation] Season filter active: "
                      f"{', '.join(segment_seasons)} (Meteorological) — only these seasons will be segmented.")
            except ImportError:
                print("[MLImageSegmentation] WARNING: seasonal_only not available; "
                      "season filter disabled.")
                season_filter_active = False

        for folder_entry in self.segmentation_image_folders:
            input_dir = os.path.normpath(folder_entry["path"])
            if not os.path.isdir(input_dir):
                print(f"[MLImageSegmentation] Skipping missing folder: {input_dir}")
                continue

            # Mirror subfolder structure under predictions_output_path
            # Use the folder's own name as the subdirectory under predictions_output_path.
            # This avoids broken paths from relpath when the output folder is unrelated
            # to the input folders (user-specified output location).
            if len(self.segmentation_image_folders) == 1:
                output_dir = self.predictions_output_path
            else:
                output_dir = os.path.normpath(
                    os.path.join(self.predictions_output_path, Path(input_dir).name)
                )

            print(f"[MLImageSegmentation] Segmenting: {input_dir}")
            print(f"[MLImageSegmentation] Output:     {output_dir}")

            # Build filtered image list if season filter is active
            image_filter = None
            if season_filter_active:
                VALID_EXTS = {'.jpg', '.jpeg', '.png'}
                all_files = [f for f in os.listdir(input_dir)
                             if os.path.splitext(f)[1].lower() in VALID_EXTS]
                kept, excluded = filter_seasons_only(all_files, segment_seasons, "Meteorological")
                image_filter = set(kept)
                print(f"[MLImageSegmentation] Season filter: {len(kept)} images kept "
                      f"({', '.join(segment_seasons)}), {len(excluded)} excluded")

            result = None

            if mode.lower() == "sam2":
                result = engine.run_inference_on_folder(
                    predictor, input_dir, output_dir,
                    copy_original_image, save_masks, selected_label_categories, progressBar,
                    save_probability_maps=save_probability_maps,
                    save_diagnostic_panels=save_diagnostic_panels,
                    image_filter=image_filter
                )

            elif mode.lower() == "segformer":
                engine = SegFormerInferenceEngine(
                    device,
                    self.SEGFORMER_MODEL,
                    input_dir,
                    output_dir,
                    class_index=1		###JES - CHANGE THIS!!!
                )
                result = engine.run_segformer_inference(
                    copy_original_image, save_masks, selected_label_categories, progressBar
                )

            elif mode.lower() == "yolo":
                if importlib.util.find_spec("ultralytics") is None:
                    QMessageBox.critical(
                        None,
                        "ultralytics Not Installed",
                        "The ultralytics package is required for YOLOv11-seg inference "
                        "but is not installed in this environment.\n\n"
                        "Install it with:\n"
                        "  pip install ultralytics\n\n"
                        "All other GRIME AI features remain available."
                    )
                    try:
                        progressBar.close()
                    except Exception:
                        pass
                    return
                engine = YOLOInferenceEngine(
                    device=device,
                    yolo_model_path=self.YOLO_MODEL,
                    input_dir=input_dir,
                    output_dir=output_dir,
                )
                result = engine.run_yolo_inference(
                    copy_original_image, save_masks, selected_label_categories, progressBar
                )

            elif mode.lower() == "maskrcnn":
                final_predictor = self.load_maskrcnn_model(device, selected_label_categories)
                continue

            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Accumulate stats from this folder
            if isinstance(result, dict):
                final_predictor = result.get("predictor")
                total_found += result.get("images_found", 0)
                total_not_found += result.get("images_not_found", 0)
                total_processed += result.get("images_processed", 0)
                target_category_name = result.get("target_category_name", target_category_name)
                if result.get("cancelled"):
                    user_cancelled = True
            else:
                final_predictor = result

            # Print GPU memory usage after each folder
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved  = torch.cuda.memory_reserved()  / 1024**2
                print(f"[GPU Memory] After '{os.path.basename(input_dir)}': "
                      f"allocated={allocated:.1f} MB, reserved={reserved:.1f} MB")
            else:
                print(f"[GPU Memory] CUDA not available — running on CPU")

            # Stop if user closed the progress bar
            if self.progress_bar_closed:
                user_cancelled = True
                break

        # Final cleanup — model loaded once, released once
        try:
            del predictor
        except Exception:
            pass
        try:
            del engine
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Close progress bar
        try:
            if progressBar is not None and progressBar.isVisible():
                progressBar.close()
        except Exception:
            pass

        # Show one combined summary dialog
        if user_cancelled:
            QMessageBox.warning(
                None, "Inference Cancelled",
                f"Segmentation was cancelled by user.\n\n"
                f"Results for '{target_category_name}':\n"
                f"- Images processed: {total_processed}\n"
                f"- Category found in: {total_found} images\n"
                f"- Category not found in: {total_not_found} images"
            )
        elif total_processed == 0:
            QMessageBox.warning(None, "No Images Processed",
                                "No images were found in the selected folders.")
        elif total_not_found == 0 and total_found == total_processed:
            QMessageBox.information(
                None, "Inference Complete",
                f"Category '{target_category_name}' was found in all {total_processed} images."
            )
        elif total_found == 0:
            QMessageBox.warning(
                None, "No Segmentations Found",
                f"Category '{target_category_name}' was not found in any of the {total_processed} images processed."
            )
        else:
            QMessageBox.information(
                None, "Inference Complete",
                f"Segmentation results for '{target_category_name}':\n\n"
                f"- Category found in: {total_found} images\n"
                f"- Category not found in: {total_not_found} images\n"
                f"- Total processed: {total_processed} images"
            )

        return final_predictor

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _validate_sam2_categories(self, engine, selected_label_categories, progressBar):
        """
        Validate that the requested category exists in the loaded model.
        Returns True if valid, False if not (and closes progress bar + shows dialog).
        Extracted from run_sam2_inference to avoid repeating per-folder.
        """
        if engine.TEST_MODE or not selected_label_categories:
            return True
        requested_category = selected_label_categories[0]
        requested_id = requested_category.get("id")
        requested_name = requested_category.get("name", "unknown")
        if int(requested_id) not in engine.category_centroids:
            try:
                if progressBar is not None and progressBar.isVisible():
                    progressBar.close()
            except Exception:
                pass
            available_msg = (f"This model was trained on: '{engine.target_category_name}'"
                             if engine.target_category_name
                             else f"Available category IDs: {list(engine.category_centroids.keys())}")
            QMessageBox.critical(None, "Category Not Found in Model",
                                 f"Category '{requested_name}' (ID: {requested_id}) not found in model.\n\n"
                                 f"{available_msg}")
            return False
        if not engine.category_centroids.get(int(requested_id)):
            try:
                if progressBar is not None and progressBar.isVisible():
                    progressBar.close()
            except Exception:
                pass
            QMessageBox.critical(None, "No Training Data for Category",
                                 f"Category '{requested_name}' has no centroid data in this model.")
            return False
        return True

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def process_image(self, image_path, predictor, mode, device, save_masks, copy_original_image, coco_data, image_id,
                      annotation_id, category_id):
        pil_image = Image.open(image_path).convert("RGB")
        image_array = np.array(pil_image)

        if mode.lower() == "sam2":
            mask, score = predictor.predict_sam2(predictor, image_array)
        elif mode.lower() == "segformer":
            from torchvision import transforms as T
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            to_tensor = T.ToTensor()
            x = to_tensor(pil_image.resize((512, 512)))
            x = normalize(x).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = predictor(pixel_values=x).logits
            probs = torch.softmax(logits, dim=1)
            water_prob = probs[0, 1]
            mask = (water_prob > 0.2).cpu().numpy().astype(np.uint8)
            score = float(water_prob.mean().item())
        elif mode.lower() == "maskrcnn":
            mask, score = self.predict_maskrcnn(predictor, pil_image, device)
        else:
            pass

        if mask is None:
            return None, None

        if mode.lower() == "segformer":
            if copy_original_image:
                shutil.copy(image_path, os.path.join(self.predictions_output_path, os.path.basename(image_path)))
            add_coco_entries(coco_data, image_path, mask, image_array, image_id, annotation_id)
        else:
            predictor.save_outputs(image_path, pil_image, mask, score, save_masks, copy_original_image, category_id)
            add_coco_entries(coco_data, image_path, mask, image_array, image_id, annotation_id)

        return image_id + 1, annotation_id + 1
