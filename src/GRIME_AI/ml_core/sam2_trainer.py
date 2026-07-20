#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
import math
import json
from datetime import datetime
import importlib.util

import numpy as np
import cv2
from PIL import Image

###JES from hydra.experimental import initialize, compose
###JES from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

import torch
from torch import nn
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast  # FALLBACK FOR OLDER PYTORCH
    
from torch.nn.attention import sdpa_kernel, SDPBackend

sys.path.append(os.path.join(os.path.dirname(__file__), '../sam2'))
from sam2.modeling import sam2_base
from sam2.sam2_image_predictor import SAM2ImagePredictor

from torch.optim.lr_scheduler import ReduceLROnPlateau

from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI.ml_core.model_training_visualization import ModelTrainingVisualization
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI.dialogs.ML_image_processing.model_config_manager import ModelConfigManager

from GRIME_AI.utils.datasetutils import DatasetUtils
from GRIME_AI.ml_core.ml_helpers import build_centroid_point_prompts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(sam2_base.__file__)


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
def _normalize_centroid(cx: float, cy: float, w: int, h: int):
    """
    Normalize centroid coordinates to [0,1] range.
    Args:
        cx, cy: pixel coordinates of centroid
        w, h: image width and height
    Returns:
        (cx_norm, cy_norm) as floats in [0,1]
    """
    if w <= 1 or h <= 1:
        raise ValueError(f"Invalid image dimensions for normalization: w={w}, h={h}")
    cx_norm = cx / float(w - 1)
    cy_norm = cy / float(h - 1)
    return (cx_norm, cy_norm)


# =======================================================================
# =======================================================================
# = = =                     HELPER FUNCTIONS                        = = =
# =======================================================================
# =======================================================================
def compute_mean_iou(y_true: list[int], y_pred: list[int]) -> float:
    arr_t = np.array(y_true, dtype=bool)
    arr_p = np.array(y_pred, dtype=bool)
    inter = np.logical_and(arr_t, arr_p).sum()
    union = np.logical_or(arr_t, arr_p).sum()
    return float(inter) / float(union) if union > 0 else 1.0

# --- Add below compute_mean_iou ---
class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # probs, targets: [B,1,H,W] in [0,1]
        probs_f = probs.view(probs.size(0), -1)
        targets_f = targets.view(targets.size(0), -1)
        inter = (probs_f * targets_f).sum(dim=1)
        union = probs_f.sum(dim=1) + targets_f.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def dice_coeff_from_probs(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item()
    return float((2.0 * inter + eps) / (union + eps))


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def iou_from_probs(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item() - inter
    return float((inter + eps) / (union + eps))

def unwrap_mask(m):
    # If SAM2 returned a list of masks, take the first
    if isinstance(m, list):
        # find the first ndarray inside the list
        for item in m:
            if hasattr(item, "ndim"):
                return item
        raise ValueError(f"List contains no ndarray: {m}")

    # If SAM2 returned (mask, score) or (mask, logits)
    if isinstance(m, tuple):
        for item in m:
            if hasattr(item, "ndim"):
                return item
        raise ValueError(f"Tuple contains no ndarray: {m}")

    return m

# ============================================================================
# ============================================================================
# = = =                       class SAM2Trainer                          = = =
# ============================================================================
# ============================================================================
class SAM2Trainer:


    def __init__(self, cfg: DictConfig = None, parent_widget=None, site_config: dict = None):
        self.parent_widget = parent_widget
        # =========================================================================
        # TEST MODE CONFIGURATION
        # Set self.TEST_MODE = True to train on hardcoded water + sky for testing
        # Set self.TEST_MODE = False to use normal configuration from site_config.json
        # =========================================================================
        self.TEST_MODE = False  # CHANGE THIS to True to enable test mode
        self.TEST_CATEGORIES = ["water", "sky"]  # Categories to use in test mode
        # =========================================================================

        self.sam2_model = None
        self._last_checkpoint_path = None
        self.selected_backend = None
        self.progress_bar_closed = False
        self.suggested_blob_radius_result = None

        # Track top N best validation checkpoints
        self.best_checkpoints = []  # List of (val_loss, path) tuples
        self.max_best_checkpoints = 3  # Keep top 3 best checkpoints

        # ALL FILES SAVED WILL BE TAGGED WITH THE DATE AND TIME THAT TRAINING STARTED.
        self.now = datetime.now()
        self.formatted_time = self.now.strftime('%Y%m%d_%H%M%S')

        self.epoch_list = []
        self.loss_values = []
        self.train_accuracy_values = []
        self.val_loss_values = []
        self.val_accuracy_values = []
        self.val_true_list = []
        self.val_pred_list = []
        self.val_score_list = []

        # track mean–IoU per epoch
        self.miou_values = []

        self.train_dice_values = []
        self.train_iou_values = []

        self.category_centroids = {}

        self.dataset_util = DatasetUtils()

        # Site config priority (race-safe):
        #   1) Use the explicit site_config dict passed in by MLModelTraining.
        #      This is the dict that MLModelTraining loaded at dispatch time,
        #      which captures the state immediately after the user's Train click.
        #      No further file I/O — immune to any later writes to site_config.json.
        #   2) Fall back to Hydra cfg.site_config if present.
        #   3) Last resort: re-read site_config.json from disk.
        if site_config is not None:
            # Defensive copy so caller's dict isn't mutated by trainer code.
            import copy as _copy
            self.site_config = _copy.deepcopy(site_config)
            print(f"[SAM2Trainer] Using site_config passed in from caller "
                  f"(holdout_seasons={self.site_config.get('holdout_seasons', [])!r}).")
        elif cfg is not None and "site_config" in cfg:
            # Convert the Hydra DictConfig to a standard dict using OmegaConf.to_container.
            self.site_config = OmegaConf.to_container(cfg.site_config, resolve=True)
        else:
            settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
            CONFIG_FILENAME = "site_config.json"
            site_configuration_file = os.path.normpath(os.path.join(settings_folder, CONFIG_FILENAME))
            print(site_configuration_file)

            # Use ModelConfigManager instead of JsonEditor
            mgr = ModelConfigManager(site_configuration_file)
            self.site_config = mgr.load_config(return_type="dict")
            print(f"[SAM2Trainer] WARNING: re-read site_config.json from disk "
                  f"(no dict was passed in). holdout_seasons={self.site_config.get('holdout_seasons', [])!r}.")

        self.site_name = self.site_config['siteName']
        self.learning_rates = self.site_config['learningRates']
        self.optimizer_type = self.site_config['optimizer']
        self.loss_function = self.site_config.get('loss_function', 'BCE + Dice + Score')
        self.weight_decay = self.site_config['weight_decay']
        self.num_epochs = self.site_config['number_of_epochs']
        self.save_model_frequency = self.site_config['save_model_frequency']
        self.early_stopping = self.site_config['early_stopping']
        self.patience = self.site_config['patience']

        # Blob filter radius — stored as fraction of image diagonal
        # Default: 50px / diagonal of a 2000x1000 image ≈ 0.02236
        import math as _math
        _DEFAULT_BLOB_FRACTION = 50.0 / _math.sqrt(2000**2 + 1000**2)
        self.blob_filter_radius = float(self.site_config.get('blob_filter_radius', _DEFAULT_BLOB_FRACTION))
        # Mahalanobis-derived fallback fraction, kept SEPARATE from
        # blob_filter_radius so the Circular filter's radius is never poisoned.
        self.blob_filter_mahal_fraction = None

        # Validation overlay settings (default: 5 samples per epoch)
        self.validation_overlay_samples = self.site_config.get('validation_overlay_samples', 5)

        self.dataset = {}

        self.val_dice_values = []
        self.val_iou_values = []

        self.folders = None
        self.annotation_files = None
        self.all_folders = []
        self.all_annotations = []
        self.categories = []
        self.annotation_index = []

        # create output folder
        try:
            self.model_output_folder = os.path.join(
                GRIME_AI_Save_Utils().get_models_folder(), 'sam2',
                f"{self.formatted_time}_{self.site_name}"
            )
            os.makedirs(self.model_output_folder, exist_ok=True)
        except OSError as e:
            self.model_output_folder = None
            print(f"Error creating folders: {e}")

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def run_training_pipeline(self):
        # Collect folders and annotations (flatten lists)
        self.all_folders = []
        self.all_annotations = []
        self.training_time_seconds = None   # set after training loop completes

        paths = self.site_config.get('Path', [])
        for path in paths:
            directory_path = path['directoryPaths']
            folders = directory_path.get('folders', [])
            annotations = directory_path.get('annotations', [])
            # Handle both list and legacy single-string values
            if isinstance(folders, str):
                folders = [folders]
            if isinstance(annotations, str):
                annotations = [annotations]
            self.all_folders.extend(folders)
            self.all_annotations.extend(annotations)

        # Build categories first (needed for both modes)
        self.categories = self.build_unique_categories(self.all_annotations)
        
        if self.TEST_MODE:
            # === TEST MODE: Train on multiple hardcoded categories ===
            print("\n" + "=" * 70)
            print("TEST MODE ENABLED: Training on", self.TEST_CATEGORIES)
            print("=" * 70 + "\n")
            
            # Don't combine datasets here - we'll reload per-category during training
            # This avoids the bug where second category's annotations get discarded
            # Just load first category to initialize the dataset structure
            first_category = self.TEST_CATEGORIES[0]
            print(f"Loading initial dataset structure using: {first_category}")
            self.dataset = self.dataset_util.load_images_and_annotations(
                self.all_folders, self.all_annotations, first_category
                )
            self.annotation_index = self.dataset_util.build_annotation_index(self.dataset)
            
        else:
            # === NORMAL MODE: Use TRAINING_CATEGORIES from train_model config ===
            training_cats = self.site_config.get("train_model", {}).get("TRAINING_CATEGORIES", [])
            configured_categories = [cat["label_name"] for cat in training_cats if "label_name" in cat]
            
            # Handle empty config
            if not configured_categories:
                print("\n" + "!" * 70)
                print("ERROR: No categories configured in TRAINING_CATEGORIES!")
                print("Please select categories in the UI")
                print("!" * 70 + "\n")
                raise ValueError("No training categories configured")
            
            # Load dataset once with first category (but keeps all annotations)
            first_category = configured_categories[0]
            print(f"Loading dataset with all annotations...")
            self.dataset = self.dataset_util.load_images_and_annotations(
                self.all_folders, self.all_annotations, first_category
            )
            self.annotation_index = self.dataset_util.build_annotation_index(self.dataset)


        # Split dataset — both ratios are user-configured and independent
        val_split   = float(self.site_config.get("val_split",   0.2))
        train_split = float(self.site_config.get("train_split", round(1.0 - val_split, 2)))

        # Seasonal hold-out: remove images from one or more hold-out seasons before splitting.
        # Config key is "holdout_seasons" (list). Legacy single-value key "knockout_season"
        # is also supported for backward compatibility.
        holdout_seasons = self.site_config.get("holdout_seasons", [])
        if not holdout_seasons:
            legacy = self.site_config.get("knockout_season", None)
            if legacy:
                holdout_seasons = [legacy]
        season_type = self.site_config.get("season_type", "Meteorological")

        if holdout_seasons:
            from GRIME_AI.ml_core.seasonal_dropout import (
                filter_seasons, extract_date_from_usgs_filename, get_season_date_range, get_season
            )

            # self.dataset is keyed by FOLDER, not by image. Each value is a dict with
            # 'images' (list of full image paths) and 'annotations' (COCO dict with its own
            # 'images' and 'annotations' arrays). Filter at the image level inside each folder.
            print(f"[Seasonal Hold-Out] Filtering — holdout_seasons={holdout_seasons!r}, "
                  f"season_type={season_type!r}, folders={len(self.dataset)}")

            total_kept     = 0
            total_excluded = 0
            excluded_image_records = []  # for manifest CSV

            holdout_set = set(holdout_seasons)

            for folder_key, folder_val in self.dataset.items():
                if not isinstance(folder_val, dict):
                    continue

                # 1) Filter the 'images' path list
                img_paths = folder_val.get("images", []) or []
                kept_paths, excluded_paths = filter_seasons(img_paths, holdout_seasons, season_type)
                folder_val["images"] = kept_paths
                total_kept     += len(kept_paths)
                total_excluded += len(excluded_paths)

                # Record excluded for manifest
                for p in excluded_paths:
                    excluded_image_records.append(p)

                # 2) Filter the COCO annotations substructure (images[] and annotations[])
                ann = folder_val.get("annotations")
                if isinstance(ann, dict):
                    excluded_basenames = {
                        str(p).replace("\\", "/").split("/")[-1] for p in excluded_paths
                    }
                    excluded_image_ids = set()

                    coco_images = ann.get("images", []) or []
                    kept_coco_images = []
                    for img_rec in coco_images:
                        fname = img_rec.get("file_name", "")
                        if fname in excluded_basenames:
                            excluded_image_ids.add(img_rec.get("id"))
                        else:
                            kept_coco_images.append(img_rec)
                    ann["images"] = kept_coco_images

                    if excluded_image_ids:
                        coco_anns = ann.get("annotations", []) or []
                        ann["annotations"] = [
                            a for a in coco_anns if a.get("image_id") not in excluded_image_ids
                        ]

            if total_excluded > 0:
                print(f"[Seasonal Hold-Out] Excluded {total_excluded} images from "
                      f"{', '.join(holdout_seasons)} ({season_type}). "
                      f"Kept {total_kept} images.")

                # Save hold-out manifest CSV
                import csv as _csv
                seasons_label     = "_".join(s.lower() for s in holdout_seasons)
                knockout_filename = os.path.join(
                    self.model_output_folder,
                    f"{season_type.lower()}_{seasons_label}_seasonal_holdouts.csv"
                )
                with open(knockout_filename, "w", newline="", encoding="utf-8") as _f:
                    writer = _csv.writer(_f)
                    writer.writerow([
                        "File Date",
                        "Image Filename",
                        "Season Type",
                        "Hold-Out Seasons",
                        "Season Date Range",
                    ])
                    for k in excluded_image_records:
                        bare       = str(k).replace("\\", "/").split("/")[-1]
                        file_date  = extract_date_from_usgs_filename(bare)
                        img_season = get_season(file_date, season_type) if file_date else ""
                        date_range = get_season_date_range(img_season, season_type) if img_season else ""
                        writer.writerow([
                            str(file_date) if file_date else "",
                            bare,
                            season_type,
                            ", ".join(holdout_seasons),
                            date_range,
                        ])
                print(f"[Seasonal Hold-Out] Manifest saved to: {knockout_filename}")
            else:
                print(f"[Seasonal Hold-Out] WARNING: filter ran but excluded 0 images. "
                      f"Either no images matched {holdout_seasons!r} in this dataset, "
                      f"or filename date regex failed to parse.")

        train_images, val_images = self.dataset_util.split_dataset(
            self.dataset, train_split=train_split, val_split=val_split
        )
        _diag_folder = os.path.join(self.model_output_folder, "diagnostics")
        os.makedirs(_diag_folder, exist_ok=True)
        split_dataset_filename = os.path.join(
            _diag_folder,
            f"{self.formatted_time}_{self.site_name}_training_and_validation_sets.json"
        )
        self.dataset_util.save_split_dataset(train_images, val_images, split_dataset_filename)

        # Preflight: verify all image files exist before training starts
        missing_files = [f for f in train_images + val_images if not os.path.isfile(f)]
        if missing_files:
            missing_list = "\n  ".join(missing_files[:20])
            if len(missing_files) > 20:
                missing_list += f"\n  ... and {len(missing_files) - 20} more."
            raise FileNotFoundError(
                f"[Preflight] {len(missing_files)} image file(s) referenced in the dataset "
                f"do not exist on disk. Training aborted.\n\n  {missing_list}\n\n"
                f"The COCO JSON and the image folder may not match."
            )

        stats_train = self.summarize_mask_imbalance(train_images)
        stats_val = self.summarize_mask_imbalance(val_images)
        print(f"Foreground ratio (train): {stats_train}")
        print(f"Foreground ratio (val):   {stats_val}")

        if len(train_images) == 0:
            print("No training images found!")
            return

        main_dir = os.path.dirname(importlib.util.find_spec('sam2').origin)
        model_cfg = os.path.normpath(os.path.join(main_dir, "configs", "sam2.1", "sam2.1_hiera_l.yaml"))
        sam2_checkpoint = os.path.normpath(os.path.join(main_dir, "checkpoints", "sam2.1_hiera_large.pt"))
        config_dir = os.path.normpath(os.path.join(main_dir, "configs", "sam2.1"))
        print("Model config path: ", model_cfg)
        print("Checkpoint path: ", sam2_checkpoint)
        print("config_dir path: ", config_dir)

        ###JES -
        '''
        # Clear Hydra if already initialized
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # Re‑initialize Hydra
        #with initialize(config_path=config_dir, version_base=None):
        with initialize(config_module="GRIME_AI.sam2.sam2.configs.sam2.1", version_base=None):
            cfg_intern = compose(config_name="sam2.1_hiera_l.yaml")
            raw_model_cfg = OmegaConf.to_container(cfg_intern.model, resolve=True)

            # Strip unsupported keys
            for key in ["no_obj_embed_spatial", "use_signed_tpos_enc_to_obj_ptrs", "device"]:
                raw_model_cfg.pop(key, None)

            new_cfg = OmegaConf.create(raw_model_cfg)
            model = instantiate(new_cfg, _recursive_=True)

            checkpoint = torch.load(
                sam2_checkpoint,
                map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            )
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        '''

        # --- REPLACEMENT FOR HYDRA THAT IS A BEAR TO USE AND MAINTAIN
        cfg_file = os.path.join(main_dir, "configs", "sam2.1", "sam2.1_hiera_l.yaml")
        checkpoint = os.path.join(main_dir, "checkpoints", "sam2.1_hiera_large.pt")

        cfg_intern = OmegaConf.load(cfg_file)
        raw_model_cfg = OmegaConf.to_container(cfg_intern.model, resolve=True)

        for k in ["no_obj_embed_spatial", "use_signed_tpos_enc_to_obj_ptrs", "device"]:
            raw_model_cfg.pop(k, None)

        new_cfg = OmegaConf.create(raw_model_cfg)
        model = instantiate(new_cfg, _recursive_=True)

        ckpt = torch.load(checkpoint,
                          map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
        # --- END HYDRA PATCH

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tuning: {total_params:,} trainable parameters")

        self.sam2_model = model.to(device).train()

        import time as _time
        _training_start = _time.perf_counter()

        if self.TEST_MODE:
            # === TEST MODE: Train on each test category separately ===
            for category_name in self.TEST_CATEGORIES:
                print("\n" + "=" * 70)
                print(f"TRAINING ON: {category_name.upper()}")
                print("=" * 70 + "\n")
                
                # RELOAD dataset for THIS specific category
                # This ensures we have the correct annotations for this category
                print(f"Loading dataset for {category_name}...")
                self.dataset = self.dataset_util.load_images_and_annotations(
                    self.all_folders, self.all_annotations, category_name
                )
                self.annotation_index = self.dataset_util.build_annotation_index(self.dataset)
                
                # DON'T re-split! Use the SAME train/val split from initial split
                # This is crucial - random.shuffle() would give different split each time
                print(f"  Using shared train/val split:")
                print(f"  Training images: {len(train_images)}")
                print(f"  Validation images: {len(val_images)}")
                
                # Get category ID
                target_id = next((c["id"] for c in self.categories if c["name"] == category_name), None)
                if target_id is None:
                    print(f"Warning: Category '{category_name}' not found in categories, skipping")
                    continue
                
                # Check if we have any annotations for this category
                if not train_images:
                    print(f"Warning: No training images for '{category_name}', skipping")
                    continue
                
                # Train on this category
                for lr in self.learning_rates:
                    print(f"\nTraining {category_name} with learning rate: {lr}")
                    self.reset_metrics()
                    self.num_train_images = len(train_images)
                    self.num_val_images = len(val_images)
                    self.train_sam(lr, self.weight_decay, train_images, val_images, 
                                 epochs=self.num_epochs, target_label=category_name)
                    self.training_time_seconds = _time.perf_counter() - _training_start
                    if self.epoch_list:
                        self._plot_training_graphs(lr)
                    else:
                        print(f"[SAM2Trainer] No epochs completed for {category_name} lr={lr} — skipping report.")
        else:
            # === NORMAL MODE: Train on ALL categories from TRAINING_CATEGORIES ===
            training_cats = self.site_config.get("train_model", {}).get("TRAINING_CATEGORIES", [])
            configured_categories = [cat["label_name"] for cat in training_cats if "label_name" in cat]
            
            print(f"\nWill train on {len(configured_categories)} category(ies): {configured_categories}")
            
            # Train on each configured category
            for category_name in configured_categories:
                print("\n" + "=" * 70)
                print(f"TRAINING ON: {category_name.upper()}")
                print("=" * 70 + "\n")
                
                # Verify category exists
                target_id = next((c["id"] for c in self.categories if c["name"] == category_name), None)
                if target_id is None:
                    print(f"Warning: Category '{category_name}' not found, skipping")
                    print(f"Available categories: {[c['name'] for c in self.categories]}")
                    continue
                
                # Train on this category (dataset already has all annotations)
                for lr in self.learning_rates:
                    print(f"\nTraining {category_name} with learning rate: {lr}")
                    self.reset_metrics()
                    self.num_train_images = len(train_images)
                    self.num_val_images = len(val_images)
                    self.train_sam(lr, self.weight_decay, train_images, val_images,
                                     epochs=self.num_epochs, target_label=category_name)
                    self.training_time_seconds = _time.perf_counter() - _training_start
                    if self.epoch_list:
                        self._plot_training_graphs(lr)
                    else:
                        print(f"[SAM2Trainer] No epochs completed for {category_name} lr={lr} — skipping report.")



        self.training_time_seconds = _time.perf_counter() - _training_start

        config_file = os.path.join(
            self.model_output_folder,
            f"{self.formatted_time}_{self.site_name}_configuration.txt"
        )
        self.save_config_to_text(config_file)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def train_sam(self, learnrate, weight_decay, train_images, val_images, epochs=20, target_label=None):

        self.reset_metrics()

        total_iterations = epochs * (len(train_images) + (len(val_images) if val_images else 0))

        progressBar = QProgressWheel(
            title="Training in-progress...", total=total_iterations,
            on_close=lambda: setattr(self, "progress_bar_closed", True),
            parent=self.parent_widget
        )

        # Set seeds for reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Select best SDPA backend once (not per-sample)
        self.selected_backend = self._select_best_sdpa_backend()

        self.sam2_model.train()

        predictor = SAM2ImagePredictor(self.sam2_model)

        optimizer = torch.optim.AdamW(self.sam2_model.parameters(), lr=learnrate, weight_decay=weight_decay)

        # ============================================================================
        # LEARNING RATE SCHEDULER: REDUCE ON PLATEAU
        # THIS SCHEDULER AUTOMATICALLY REDUCES THE LEARNING RATE WHEN VALIDATION LOSS
        # STOPS IMPROVING. IT MONITORS VALIDATION LOSS AND REDUCES LR BY FACTOR=0.5
        # AFTER PATIENCE=3 EPOCHS WITHOUT IMPROVEMENT. MIN_LR=1e-7 PREVENTS LR FROM
        # BECOMING TOO SMALL. VERBOSE=TRUE PRINTS WHEN LR CHANGES.
        # ============================================================================
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',  # minimize validation loss
            factor=0.5,  # multiply LR by 0.5 when plateau detected
            patience=3,  # wait 3 epochs before reducing
            min_lr=1e-7  # don't reduce below this value
        )

        # Note: loss_fn defined here but not passed to validation anymore
        use_amp = True
        if use_amp and device.type == "cuda":
            # MAKE PORTABLE ACROSS GPU AND CPU MACHINES
            try:
                from torch.amp import GradScaler
                scaler = GradScaler('cuda')
            except ImportError:
                from torch.cuda.amp import GradScaler  # FALLBACK FOR OLDER PYTORCH
                scaler = GradScaler()
        else:
            scaler = None

        best_val_loss = float("inf")
        patience_counter = 0
        divergence_threshold = 1e3
        last_completed_epoch = 0

        # VRAM trend tracking
        _prev_train_vram_mb = None
        self._prev_val_vram_mb = None

        # Get target_label - use parameter if provided (TEST_MODE), otherwise get from config
        if target_label is None:
            # Normal mode: get from config
            target_label = self.site_config["load_model"]["SEGMENTATION_CATEGORIES"][0]
        # else: use the provided target_label (TEST_MODE)

        # ── Pre-load validation images and masks into memory ─────────────────
        # Eliminates repeated disk I/O during validation each epoch.
        # predictor.set_image() is still called each epoch but images and masks
        # are served from memory rather than re-read from disk.
        val_cache = []
        if val_images:
            print(f"[SAM2Trainer] Pre-loading {len(val_images)} validation images into memory...")
            target_id_for_cache = next(
                (c["id"] for c in self.categories if c["name"] == target_label), None
            )
            for vf in val_images:
                try:
                    img_np = np.array(Image.open(vf).convert("RGB"))
                except Exception as e:
                    print(f"[SAM2Trainer] Could not load val image {vf}: {e}")
                    continue
                result = self.dataset_util.load_true_mask(
                    vf, self.annotation_index, mode="binary", target_id=target_id_for_cache
                )
                true_mask = result[0] if isinstance(result, tuple) else result
                if true_mask is None:
                    print(f"[SAM2Trainer] No annotation for {vf} — skipping from val cache.")
                    continue
                val_cache.append({"file": vf, "image_np": img_np, "true_mask": true_mask})
            print(f"[SAM2Trainer] Val cache ready: {len(val_cache)} images.")
        # ─────────────────────────────────────────────────────────────────────

        try:
            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")

                if use_amp:
                    avg_epoch_loss, train_accuracy, train_dice, train_iou = self._train_one_epoch(
                        epoch=epoch,
                        train_images=train_images,
                        predictor=predictor,
                        optimizer=optimizer,
                        progressBar=progressBar,
                        use_amp=use_amp,
                        scaler=scaler,
                        target_label=target_label
                    )
                else:
                    avg_epoch_loss, train_accuracy, train_dice, train_iou = self._train_one_epoch(
                        epoch=epoch,
                        train_images=train_images,
                        predictor=predictor,
                        optimizer=optimizer,
                        progressBar=progressBar,
                        use_amp=use_amp,
                        target_label = target_label
                    )

                if avg_epoch_loss is None:
                    return

                last_completed_epoch = epoch + 1

                # ALWAYS track training metrics for ALL epochs
                self.loss_values.append(avg_epoch_loss)
                self.train_accuracy_values.append(train_accuracy)
                self.train_dice_values.append(train_dice)
                self.train_iou_values.append(train_iou)
                print(f"Epoch {epoch + 1} Training "
                      f"Loss: {avg_epoch_loss:.4f} "
                      f"Acc: {train_accuracy:.4f} "
                      f"Dice: {train_dice:.4f} "
                      f"IoU: {train_iou:.4f}")

                # Clear GPU cache after training to prevent memory accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    cur_mb = torch.cuda.memory_allocated() / 1024 ** 2
                    res_mb = torch.cuda.memory_reserved() / 1024 ** 2
                    trend = ""
                    if _prev_train_vram_mb is not None:
                        delta = cur_mb - _prev_train_vram_mb
                        trend = f"  delta={delta:+.1f} MB"
                        if delta > 50:
                            trend += "  <<< VRAM INCREASING - possible leak"
                    print(f"[VRAM] Epoch {epoch + 1} post-train: allocated={cur_mb:.1f} MB  reserved={res_mb:.1f} MB{trend}")
                    _prev_train_vram_mb = cur_mb

                if math.isnan(avg_epoch_loss) or avg_epoch_loss > divergence_threshold:
                    print("Training diverged. Aborting early.")
                    break

                # Deduplicate centroids after first epoch to prevent memory leak
                if epoch == 0:
                    print("\n=== Finalizing Centroid Collection ===")
                    self._deduplicate_centroids()
                    total_centroids = sum(len(v) for v in self.category_centroids.values())
                    print(f"Total unique centroids collected: {total_centroids}")
                    print("Centroid collection frozen - will not accumulate in future epochs\n")


                if val_cache:
                    is_last_epoch = (epoch + 1 == epochs)
                    avg_val_loss, val_accuracy, miou, avg_val_dice, avg_val_iou = self._validate_one_epoch(
                        val_cache=val_cache,
                        predictor=predictor,
                        progressBar=progressBar,
                        target_label=target_label,
                        save_overlays=is_last_epoch
                    )

                    if avg_val_loss is None:
                        return

                    # Add to epoch list when validation runs
                    self.epoch_list.append(epoch + 1)

                    self.val_loss_values.append(avg_val_loss)
                    self.val_accuracy_values.append(val_accuracy)
                    self.miou_values.append(miou)
                    self.val_dice_values.append(avg_val_dice)
                    self.val_iou_values.append(avg_val_iou)

                    print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f} "
                          f"Acc: {val_accuracy:.4f} Dice: {avg_val_dice:.4f} "
                          f"IoU: {avg_val_iou:.4f} mIoU: {miou:.4f}")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        self._save_model_checkpoint(
                            predictor, learnrate, epoch + 1,
                            suffix=f"valbest_ep{epoch + 1:03d}",
                            val_loss=avg_val_loss, val_accuracy=val_accuracy, miou=miou,
                            target_category_name=target_label
                        )
                    else:
                        patience_counter += 1
                        if self.early_stopping and patience_counter >= self.patience:
                            print(f"Early stopping triggered at epoch {epoch + 1}.")
                            break

                    # ============================================================================
                    # UPDATE LEARNING RATE SCHEDULER BASED ON VALIDATION LOSS
                    # THIS STEP ALLOWS THE SCHEDULER TO MONITOR VALIDATION LOSS AND REDUCE
                    # LEARNING RATE IF NO IMPROVEMENT IS DETECTED FOR PATIENCE=3 EPOCHS
                    # ============================================================================
                    scheduler.step(avg_val_loss)
            # ----------------------------------------------------------------
            # POST-TRAINING: compute suggested blob filter radius and store
            # on self so the caller can retrieve it and prompt the user
            # ----------------------------------------------------------------
            self.suggested_blob_radius_result = self._compute_blob_filter_distribution(
                train_images, target_label
            )
            # Record the Mahalanobis-derived fallback fraction for the report ONLY.
            # Do NOT overwrite self.blob_filter_radius (the user's Manual/Computed
            # scalar radius used by the Circular filter).
            if self.suggested_blob_radius_result is not None:
                self.blob_filter_mahal_fraction = self.suggested_blob_radius_result.get(
                    "fallback_fraction", self.blob_filter_mahal_fraction
                )

        finally:
            if not self.progress_bar_closed and 'progressBar' in locals():
                progressBar.close()
            if 'progressBar' in locals():
                del progressBar

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def reset_metrics(self):
        self.epoch_list.clear()
        self.loss_values.clear()
        self.train_accuracy_values.clear()
        self.val_loss_values.clear()
        self.val_accuracy_values.clear()
        self.val_true_list.clear()
        self.val_pred_list.clear()
        self.val_score_list.clear()
        self.miou_values.clear()
        self.train_dice_values.clear()
        self.train_iou_values.clear()
        self.val_dice_values.clear()
        self.val_iou_values.clear()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    @property
    def _report_blob_radius_frac(self):
        """Report value: Mahalanobis-derived fraction when available, else the
        configured scalar radius. Separate from blob_filter_radius."""
        if self.blob_filter_mahal_fraction is not None:
            return self.blob_filter_mahal_fraction
        return self.blob_filter_radius

    def _suggest_blob_filter_mode(self, pts):
        """Advisory only — recommend a blob-filter mode from the centroid
        distribution. Never applied automatically. Returns (mode, reason)."""
        n = len(pts)
        if n < 5:
            return "circular", f"only {n} centroids — too few for a distribution"
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            best_k, best_s = 1, -1.0
            for k in (2, 3):
                if n <= k:
                    break
                labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(pts)
                s = silhouette_score(pts, labels)
                if s > best_s:
                    best_k, best_s = k, s
            if best_k >= 2 and best_s >= 0.55:
                return "knn", f"multimodal ({best_k} clusters, silhouette {best_s:.2f})"
            return "mahalanobis", f"unimodal (best silhouette {best_s:.2f})"
        except Exception:
            return "mahalanobis", "unimodal (clustering unavailable)"

    def _compute_blob_filter_distribution(self, train_images, target_label):
        """
        Post-training: fit a 2D Gaussian to per-image ground truth centroids
        and store the mean and covariance matrix in the checkpoint. At inference
        the blob filter uses Mahalanobis distance rather than a circular radius,
        so the acceptance region conforms to the actual shape and orientation of
        centroid variation across the training set.

        Falls back to a scalar radius fraction when fewer than 3 images are
        available or the covariance matrix is singular (collinear centroids).

        Returns a dict with keys:
            centroid_mean   : [cx, cy] in pixels
            centroid_cov    : 2x2 covariance matrix as nested list (or None)
            n_sigma         : float, Mahalanobis threshold (default 2.5)
            fallback_fraction: float, scalar fraction for singular/missing cov
            diagonal_px     : float
            n_images        : int
        or None on total failure.
        """
        import numpy as np
        from scipy.spatial.distance import mahalanobis

        target_id = next((c["id"] for c in self.categories if c["name"] == target_label), None)
        if target_id is None:
            print(f"[Blob Filter] Category '{target_label}' not found — skipping.")
            return None

        # Minimum blob area (fraction of frame) to include — drops degenerate
        # slivers/noise that would skew the distribution. Tunable.
        _MIN_BLOB_FRAC = 0.005
        # Covariance std floor as a fraction of the image diagonal (Fix 3).
        _COV_MIN_STD_FRAC = 0.03

        per_blob_centroids = []
        skipped = 0
        dropped_small = 0

        for image_file in train_images:
            result = self.dataset_util.load_true_mask(
                image_file, self.annotation_index, mode="binary", target_id=target_id
            )
            true_mask = result[0] if isinstance(result, tuple) else result
            if true_mask is None or true_mask.sum() == 0:
                skipped += 1
                continue
            if true_mask.ndim == 3:
                true_mask = true_mask[..., 0]
            true_mask = true_mask.astype(np.uint8)

            # Fix 1: one centroid PER connected water region, not the mean of ALL
            # water pixels — scattered regions no longer collapse to a single
            # phantom centroid between them.
            num_labels, labels = cv2.connectedComponents(true_mask)
            frame_area = float(true_mask.shape[0] * true_mask.shape[1])
            for lbl in range(1, num_labels):
                ys, xs = np.where(labels == lbl)
                if xs.size == 0:
                    continue
                # Fix 2: drop degenerate/tiny blobs (e.g. sliver annotations).
                if xs.size / frame_area < _MIN_BLOB_FRAC:
                    dropped_small += 1
                    continue
                per_blob_centroids.append([float(np.mean(xs)), float(np.mean(ys))])

        if not per_blob_centroids:
            print("[Blob Filter] No valid blobs — skipping computation.")
            return None

        try:
            img = np.array(Image.open(train_images[0]).convert("RGB"))
            h, w = img.shape[:2]
            diagonal_px = float(np.sqrt(w ** 2 + h ** 2))
        except Exception:
            diagonal_px = float(np.sqrt(1920 ** 2 + 1080 ** 2))

        pts = np.array(per_blob_centroids)  # shape (N, 2)
        mean_cx, mean_cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        centroid_mean = [mean_cx, mean_cy]

        # Distances from mean — used for fallback scalar and diagnostics
        distances = np.linalg.norm(pts - np.array([mean_cx, mean_cy]), axis=1)
        p95_px  = float(np.percentile(distances, 95))
        p50_px  = float(np.percentile(distances, 50))
        fallback_fraction = p95_px / diagonal_px

        # Attempt covariance fit — requires at least 3 non-collinear points
        centroid_cov = None
        n_sigma = 2.5
        cov_method = "none"
        if len(pts) >= 3:
            try:
                # Robust covariance (Minimum Covariance Determinant) when
                # scikit-learn is available — rejects outlier centroids at the
                # source. Falls back to the empirical np.cov otherwise.
                rob_mean = [mean_cx, mean_cy]
                try:
                    from sklearn.covariance import MinCovDet
                    _mcd = MinCovDet().fit(pts)
                    cov = np.asarray(_mcd.covariance_, dtype=np.float64)
                    rob_mean = [float(_mcd.location_[0]), float(_mcd.location_[1])]
                    cov_method = "robust (MinCovDet)"
                except Exception:
                    cov = np.cov(pts.T)
                    cov_method = "empirical (np.cov)"
                # Fix 3: regularize — floor each axis' variance so a tight cluster
                # can't create a near-singular, over-narrow ellipse.
                _min_var = (_COV_MIN_STD_FRAC * diagonal_px) ** 2
                cov = cov + _min_var * np.eye(2)
                inv_cov = np.linalg.inv(cov)
                # Pair the (robust) mean with the (robust) covariance so the
                # Mahalanobis distances are internally consistent.
                centroid_mean = rob_mean
                mah_distances = np.array([
                    mahalanobis(p, centroid_mean, inv_cov) for p in pts
                ])
                n_sigma = max(2.5, float(np.percentile(mah_distances, 95)))
                centroid_cov = cov.tolist()
                print(f"[Blob Filter] Mahalanobis fit successful ({cov_method}).")
                print(f"  n_sigma (95th pct of training Mahalanobis distances): {n_sigma:.3f}")
            except np.linalg.LinAlgError:
                print("[Blob Filter] Covariance matrix singular — using scalar fallback.")

        # kNN model: store every training centroid plus a distance threshold set
        # from the training set's own k-th-nearest-neighbour scatter. Handles
        # multimodal water (multiple channels) a single Gaussian cannot.
        knn_k = 1
        knn_threshold = None
        if len(pts) >= 2:
            try:
                from scipy.spatial import cKDTree
                _k = min(knn_k, len(pts) - 1)
                _tree = cKDTree(pts)
                _d, _ = _tree.query(pts, k=_k + 1)   # self is nearest (dist 0)
                _kth = _d[:, _k]
                knn_k = _k
                knn_threshold = float(np.percentile(_kth, 95))
            except Exception as _knn_err:
                print(f"[Blob Filter] kNN threshold computation failed ({_knn_err}).")

        # Advisory only — recommend a mode; never applied automatically.
        suggested_mode, suggested_reason = self._suggest_blob_filter_mode(pts)

        print(f"\n[Blob Filter] Category: '{target_label}'")
        print(f"  Blobs analyzed  : {len(per_blob_centroids)}  (images skipped: {skipped}, small blobs dropped: {dropped_small})")
        print(f"  Mean centroid   : ({centroid_mean[0]:.1f}, {centroid_mean[1]:.1f}) px")
        print(f"  Median distance : {p50_px:.1f} px")
        print(f"  95th pct dist   : {p95_px:.1f} px")
        print(f"  Diagonal        : {diagonal_px:.1f} px")
        print(f"  Fallback frac   : {fallback_fraction:.4f} ({fallback_fraction * 100:.2f}%)")
        print(f"  Covariance fit  : {cov_method if centroid_cov else 'no (fallback)'}")
        if knn_threshold is not None:
            print(f"  kNN threshold   : {knn_threshold:.1f} px (k={knn_k})")
        print(f"  Recommended mode: {suggested_mode}  ({suggested_reason})")

        return {
            "centroid_mean":    centroid_mean,
            "centroid_cov":     centroid_cov,
            "n_sigma":          n_sigma,
            "fallback_fraction": fallback_fraction,
            "diagonal_px":      diagonal_px,
            "n_images":         len(per_blob_centroids),
            "centroids":        pts.tolist(),
            "knn_k":            knn_k,
            "knn_threshold":    knn_threshold,
            "suggested_mode":   suggested_mode,
            "suggested_reason": suggested_reason,
        }

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _update_checkpoints_blob_radius(self, new_fraction, blob_distribution=None):
        """
        Reopen each saved checkpoint and update blob filter metadata.
        Writes blob_filter_radius (scalar fallback fraction) and, when
        blob_distribution is provided, also writes blob_centroid_mean,
        blob_centroid_cov, and blob_filter_n_sigma for Mahalanobis filtering
        at inference.
        """
        for rank, (val_loss, ckpt_path) in enumerate(self.best_checkpoints, start=1):
            if not ckpt_path or not os.path.isfile(ckpt_path):
                continue
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                # blob_filter_radius stays the scalar radius (Circular); the
                # Mahalanobis fallback fraction is stored under its own key.
                ckpt["blob_filter_radius"] = self.blob_filter_radius
                if new_fraction is not None:
                    ckpt["blob_filter_mahal_fraction"] = new_fraction
                if blob_distribution is not None:
                    ckpt["blob_centroid_mean"]  = blob_distribution.get("centroid_mean")
                    ckpt["blob_centroid_cov"]   = blob_distribution.get("centroid_cov")
                    ckpt["blob_filter_n_sigma"] = blob_distribution.get("n_sigma", 2.5)
                    ckpt["blob_centroids"]      = blob_distribution.get("centroids")
                    ckpt["blob_knn_k"]          = blob_distribution.get("knn_k")
                    ckpt["blob_knn_threshold"]  = blob_distribution.get("knn_threshold")
                    ckpt["suggested_blob_filter_mode"] = blob_distribution.get("suggested_mode")
                torch.save(ckpt, ckpt_path)
                print(f"[Blob Filter] Updated checkpoint #{rank}: {os.path.basename(ckpt_path)}")
            except Exception as e:
                print(f"[Blob Filter] Failed to update {ckpt_path}: {e}")

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _select_best_sdpa_backend(self):
        """
        Selects the best available SDPA backend once, to be reused throughout training.
        Returns the selected backend or None if all fail.
        """
        # Enable all SDP backends — allows CuDNN fused attention on RTX 40-series
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        desired = ("FLASH_ATTENTION", "XFORMERS", "EFFICIENT_ATTENTION", "MATH")
        backends = [getattr(SDPBackend, name) for name in desired if hasattr(SDPBackend, name)]

        print("\n=== Selecting SDPA Backend ===")
        for backend in backends:
            try:
                dummy_q = torch.randn(1, 8, 16, 64, device=device)
                dummy_k = torch.randn(1, 8, 16, 64, device=device)
                dummy_v = torch.randn(1, 8, 16, 64, device=device)

                with sdpa_kernel(backend):
                    _ = torch.nn.functional.scaled_dot_product_attention(dummy_q, dummy_k, dummy_v)

                print(f"✓ Selected SDPA backend: {backend.name}")
                return backend
            except Exception as e:
                print(f"✗ Backend {backend.name} failed: {e}")
                continue

        print("All SDPA backends failed; will use default attention")
        return None

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _train_one_epoch(
            self,
            epoch,
            train_images,
            predictor,
            optimizer,
            progressBar,
            use_amp=False,
            scaler=None,
            target_label=None
    ):
        """
        Runs one training epoch. Returns (avg_epoch_loss, train_accuracy, avg_dice, avg_iou).

        CORRECTIONS APPLIED:
        - Uses BCEWithLogitsLoss properly on logits
        - Only applies sigmoid when computing Dice loss and metrics
        - Maintains numerical stability
        - Direct progress_bar_closed attribute access (Issue #8)
        """
        import torch.nn.functional as F  # ✅ Moved import to top of method

        self.sam2_model.train()

        epoch_loss = 0.0
        train_correct, train_total = 0, 0
        processed_count = 0
        dice_sum, iou_sum = 0.0, 0.0

        np.random.shuffle(train_images)

        # Define loss function once (not per-sample)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        dice_loss_fn = DiceLoss()

        # --- FILTER MASK BY TARGET_LABEL ---
        if target_label is None:
            raise ValueError("target_label is required but was None.")

        for idx, image_file in enumerate(train_images):
            if self.progress_bar_closed:
                self._terminate_training(progressBar)
                return None, None, None, None

            # ----------------------------------------
            # LOAD GROUND TRUTH MASK
            # ----------------------------------------
            # RESOLVE TARGET_ID
            target_id = next((c["id"] for c in self.categories if c["name"] == target_label), None)
            if target_id is None:
                available = ", ".join(f'{c["name"]}:{c["id"]}' for c in self.categories)
                raise ValueError(f"Unknown target_label '{target_label}'. Available: {available}")

            # ============================================================
            # CENTROID DIAGNOSTIC TRACKING (epoch 0 only)
            # Tracks why each image did or did not contribute a centroid.
            # Written to a CSV report in the model output folder at end of epoch 0.
            # ============================================================
            if epoch == 0 and not hasattr(self, "_centroid_diag_rows"):
                self._centroid_diag_rows = []   # list of dicts, one per image

            # LOAD GROUND TRUTH MASK
            result = self.dataset_util.load_true_mask(image_file, self.annotation_index, mode="binary", target_id=target_id)
            true_mask = result[0] if isinstance(result, tuple) else result

            # SKIP IF NO USABLE MASK
            if true_mask is None or true_mask.sum() == 0:
                print(f"No usable mask for target_id={target_id} in {image_file}, skipping.")
                if epoch == 0:
                    self._centroid_diag_rows.append({
                        "image": os.path.basename(image_file),
                        "outcome": "mask_none_or_empty",
                        "centroid_px": "",
                        "centroid_norm": "",
                        "detail": f"load_true_mask returned None or zero mask for target_id={target_id}"
                    })
                continue

            if true_mask.ndim == 3:
                true_mask = true_mask[..., 0]
            true_mask = true_mask.astype(np.uint8)

            if not os.path.isfile(image_file):
                # File missing at training time despite passing preflight.
                # Most likely cause: antivirus, indexer, or sync service locking the file.
                # Log and skip rather than crash.
                print(f"[WARNING] Image file inaccessible at epoch {epoch}, index {idx} - skipping: {image_file}")
                if epoch == 0:
                    self._centroid_diag_rows.append({
                        "image": os.path.basename(image_file),
                        "outcome": "file_missing",
                        "centroid_px": "",
                        "centroid_norm": "",
                        "detail": "Image file inaccessible at training time"
                    })
                continue
            image = np.array(Image.open(image_file).convert("RGB"))
            predictor.set_image(image)

            # ============================================================
            # COMPUTE CENTROIDS FOR POSITIVE PROMPTS (selected category)
            # ============================================================
            h, w = true_mask.shape[:2]
            num_labels, labels = cv2.connectedComponents(true_mask.astype(np.uint8))
            positive_coords = []
            
            for lbl in range(1, num_labels):  # skip background
                ys, xs = np.nonzero(labels == lbl)
                if xs.size > 0:
                    cx = float(xs.mean())
                    cy = float(ys.mean())
                    positive_coords.append([cx, cy])

                    # Only collect centroids during first epoch to prevent memory leak.
                    # NOTE: centroid collection happens here from ground truth mask blobs,
                    # not from predicted blobs, so the blob filter does not affect what
                    # gets recorded. This ensures the Mahalanobis distribution is built
                    # from the full ground truth centroid population.
                    if epoch == 0:
                        centroid_px = (int(round(cx)), int(round(cy)))
                        centroid_norm = _normalize_centroid(cx, cy, w, h)
                        self.category_centroids.setdefault(int(target_id), []).append({
                            "centroid_px": centroid_px,
                            "centroid_norm": centroid_norm
                        })
                        self._centroid_diag_rows.append({
                            "image": os.path.basename(image_file),
                            "outcome": "centroid_recorded",
                            "centroid_px": f"({centroid_px[0]}, {centroid_px[1]})",
                            "centroid_norm": f"({centroid_norm[0]:.4f}, {centroid_norm[1]:.4f})",
                            "detail": f"blob {lbl}/{num_labels - 1}  mask_pixels={int(ys.size)}"
                        })

            if not positive_coords:
                print(f"Skipping {image_file} - no usable centroid for label {target_label}.")
                if epoch == 0:
                    self._centroid_diag_rows.append({
                        "image": os.path.basename(image_file),
                        "outcome": "no_foreground_blobs",
                        "centroid_px": "",
                        "centroid_norm": "",
                        "detail": f"Mask loaded OK but connectedComponents found 0 blobs  "
                                  f"mask_sum={int(true_mask.sum())}"
                    })
                continue

            # ============================================================
            # COMPUTE CENTROIDS FOR NEGATIVE PROMPTS (all other categories)
            # ============================================================
            negative_coords = []

            # Get all categories except the target (dynamic - not hardcoded!)
            negative_categories = [
                c for c in self.categories 
                if c["name"] != target_label
            ]
            
            for neg_category in negative_categories:
                neg_id = neg_category["id"]
                
                # Load mask for this negative category
                neg_result = self.dataset_util.load_true_mask(
                    image_file, self.annotation_index, 
                    mode="binary", target_id=neg_id
                )
                neg_mask = neg_result[0] if isinstance(neg_result, tuple) else neg_result

                # Skip if no annotation for this category in this image
                if neg_mask is None or neg_mask.sum() == 0:
                    continue
                
                if neg_mask.ndim == 3:
                    neg_mask = neg_mask[..., 0]
                
                # Compute centroids for negative regions
                num_labels_neg, labels_neg = cv2.connectedComponents(neg_mask.astype(np.uint8))
                for lbl in range(1, num_labels_neg):
                    ys, xs = np.nonzero(labels_neg == lbl)
                    if xs.size > 0:
                        cx, cy = float(xs.mean()), float(ys.mean())
                        negative_coords.append([cx, cy])
            
            # Balance negatives to avoid overwhelming positive prompts
            # (SAM2 works best with balanced prompt sets)
            if len(negative_coords) > len(positive_coords) * 3:
                negative_coords = random.sample(negative_coords, len(positive_coords) * 3)
            
            # ============================================================
            # COMBINE POSITIVE AND NEGATIVE PROMPTS
            # ============================================================
            all_coords = positive_coords + negative_coords
            all_labels = [1] * len(positive_coords) + [0] * len(negative_coords)
            
            # Log prompt counts for monitoring
            if idx % 10 == 0:  # Log every 10th image
                print(f"  Image {os.path.basename(image_file)}: "
                      f"{len(positive_coords)} positive, {len(negative_coords)} negative prompts")

            # --- Build tensors for all centroids (positive + negative) ---
            point_coords = torch.tensor(all_coords, device=device, dtype=torch.float32)
            point_labels = torch.tensor(all_labels, device=device, dtype=torch.int64)

            # Prompt embeddings using multiple points (positive and negative)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(point_coords.unsqueeze(0), point_labels.unsqueeze(0)),
                boxes=None,
                masks=None,
            )

            batched_mode = True
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

            optimizer.zero_grad()

            # Pick dtype depending on device
            if device.type == "cuda":
                autocast_dtype = torch.float16  # or torch.bfloat16 if you prefer
            else:  # CPU
                autocast_dtype = torch.bfloat16  # CPU autocast only supports bfloat16

            with autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                # Use pre-selected SDPA backend (selected once in train_sam)
                if hasattr(self, 'selected_backend') and self.selected_backend is not None:
                    with sdpa_kernel(self.selected_backend):
                        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                            repeat_image=batched_mode,
                            high_res_features=high_res_features,
                        )
                else:
                    # Fallback to default if backend selection failed
                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        repeat_image=batched_mode,
                        high_res_features=high_res_features,
                    )

                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                gt_mask = torch.tensor(true_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
                if gt_mask.sum() == 0:
                    print(f"Skipping {image_file} - ground-truth mask is empty for label {target_label}.")
                    continue

                # Keep predictions as LOGITS (don't apply sigmoid yet)
                prd_mask_logits = prd_masks[:, 0].unsqueeze(1)  # [1,1,H,W] - raw logits

                # Resize logits if mismatch
                if prd_mask_logits.shape[-2:] != gt_mask.shape[-2:]:
                    prd_mask_logits = F.interpolate(
                        prd_mask_logits,
                        size=gt_mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

                if prd_mask_logits.shape != gt_mask.shape:
                    raise ValueError(f"Shape mismatch {prd_mask_logits.shape} vs {gt_mask.shape} for {image_file}")

                # USE BCEWithLogitsLoss on logits (more stable than manual BCE on probabilities)
                seg_loss = bce_loss_fn(prd_mask_logits, gt_mask)

                # FOR DICE LOSS, WE NEED PROBABILITIES, SO APPLY SIGMOID
                prd_mask_probs = torch.sigmoid(prd_mask_logits)

                dice_loss = dice_loss_fn(prd_mask_probs, gt_mask)

                # COMPUTE IOU FOR SCORE PREDICTION LOSS
                prd_mask_binary = (prd_mask_probs > 0.5).float()

                # FILTER OUT BLOBS NOT OVERLAPPING CENTROID PROMPTS
                labels_np = prd_mask_binary.squeeze().cpu().numpy().astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(labels_np)
                valid_mask = np.zeros_like(labels, dtype=np.uint8)

                # ADAPTIVE RADIUS THRESHOLD BASED ON IMAGE SIZE.
                # FRACTION IS SET BY USER IN GUI AND STORED IN site_config + .torch METADATA.
                img_diagonal = math.sqrt(h * h + w * w)
                radius_threshold = max(10, int(self.blob_filter_radius * img_diagonal))

                for lbl in range(1, num_labels):  # skip background
                    ys, xs = np.nonzero(labels == lbl)
                    if xs.size == 0:
                        continue
                    cx_blob, cy_blob = xs.mean(), ys.mean()

                    # KEEP ONLY IF BLOB CENTROID IS CLOSE TO ONE OF THE PROMPT CENTROIDS
                    for cx, cy in positive_coords:
                        if np.linalg.norm([cx - cx_blob, cy - cy_blob]) < radius_threshold:
                            valid_mask[labels == lbl] = 1
                            break

                prd_mask_binary = torch.tensor(valid_mask, device=device).unsqueeze(0).unsqueeze(1).float()

                inter = (gt_mask * prd_mask_binary).sum((1, 2, 3))
                union = gt_mask.sum((1, 2, 3)) + prd_mask_binary.sum((1, 2, 3)) - inter
                iou = inter / (union + 1e-6)
                iou[union == 0] = 1.0

                # Align decoder score with IoU
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                # Combined loss
                loss = 0.5 * seg_loss + 0.5 * dice_loss + 0.05 * score_loss

            # Backward
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.sam2_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sam2_model.parameters(), max_norm=1.0)
                optimizer.step()

            # Metrics (use probabilities for evaluation)
            d_metric = dice_coeff_from_probs(prd_mask_probs, gt_mask)
            j_metric = iou_from_probs(prd_mask_probs, gt_mask)

            dice_sum += d_metric
            iou_sum += j_metric
            epoch_loss += loss.item()
            processed_count += 1

            # Periodically clear GPU cache to prevent VRAM fragmentation
            if processed_count % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            pred_binary = (prd_mask_probs > 0.5).cpu().numpy()
            true_binary = gt_mask.cpu().numpy()
            train_correct += np.sum(pred_binary == true_binary)
            train_total += np.prod(true_binary.shape)

            # Explicitly free tensors to prevent VRAM accumulation across iterations
            del loss, prd_mask_logits, prd_mask_probs, prd_mask_binary, gt_mask, seg_loss, dice_loss, score_loss
            del point_coords, point_labels, sparse_embeddings, dense_embeddings
            del low_res_masks, prd_scores, prd_masks, high_res_features, iou

            if not self.progress_bar_closed:  # ✅ ISSUE #8 FIX: Direct attribute access
                progressBar.setValue(progressBar.getValue() + 1)

        avg_epoch_loss = epoch_loss / processed_count if processed_count > 0 else 0.0
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        avg_dice = dice_sum / processed_count if processed_count > 0 else 0.0
        avg_iou = iou_sum / processed_count if processed_count > 0 else 0.0

        return avg_epoch_loss, train_accuracy, avg_dice, avg_iou

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _validate_one_epoch(self, val_cache, predictor, progressBar, target_label,
                            save_overlays=False):
        """
        Runs one validation epoch using pre-cached image embeddings.

        val_cache: list of dicts produced by the pre-caching block in train_sam.
                   Each dict has: file, image_np, true_mask, features, orig_hw, input_size.
        save_overlays: only write overlay PNGs when True (last epoch only).
        """
        import torch.nn.functional as F

        val_loss = 0.0
        dice_sum, iou_sum = 0.0, 0.0
        val_correct, val_total = 0, 0
        n_items = 0

        # Clear per-epoch pixel lists to prevent unbounded memory growth across epochs
        self.val_true_list.clear()
        self.val_pred_list.clear()
        self.val_score_list.clear()

        self.sam2_model.eval()
        progressBar.setWindowTitle("Validation in-progress")

        # Define loss functions (same as training)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        dice_loss_fn = DiceLoss()

        target_id = next((c["id"] for c in self.categories if c["name"] == target_label), None)

        with torch.no_grad():
            for val_idx, cached in enumerate(val_cache):
                if self.progress_bar_closed:
                    self._terminate_validation(progressBar)
                    return None, None, None, None, None

                # Initialize per-image tensors to None so del is always safe
                iou_actual = None
                logit_tensor = None
                logit_upsampled = None
                prob_upsampled = None
                pred_binary = None
                val_true_mask_tensor = None

                val_image   = cached["image_np"]
                val_true_mask = cached["true_mask"]

                # Re-encode image each epoch (images served from memory cache)
                predictor.set_image(val_image)

                # Build centroid prompts
                h_img, w_img = val_image.shape[:2]
                point_coords, point_labels = build_centroid_point_prompts(
                    category_id=target_id,
                    category_centroids=self.category_centroids,
                    image_w=w_img,
                    image_h=h_img,
                    device=device,
                    random_seed=42
                )

                if point_coords is None:
                    print(f"  Validation skipped for {cached['file']} — "
                          f"no centroids yet for category '{target_label}' (ID {target_id}).")
                    continue

                try:
                    masks, scores, low_res_logits = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=False
                    )

                    if masks.size == 0 or scores.size == 0:
                        print(f"Warning: No masks predicted for {cached['file']}")
                        continue

                except Exception as e:
                    print(f"Error during prediction for {cached['file']}: {e}")
                    continue

                best_idx = int(np.argmax(scores))
                best_logits = low_res_logits[best_idx]

                logit_tensor = torch.tensor(
                    best_logits, dtype=torch.float32, device=device
                ).unsqueeze(0).unsqueeze(0)

                if len(val_true_mask.shape) > 2:
                    val_true_mask = val_true_mask[:, :, 0]

                val_true_mask_tensor = torch.tensor(
                    val_true_mask, dtype=torch.float32, device=device
                ).unsqueeze(0).unsqueeze(0)

                H, W = val_true_mask_tensor.shape[2:]
                logit_upsampled = F.interpolate(
                    logit_tensor, size=(H, W), mode='bilinear', align_corners=False
                )

                bce_val  = bce_loss_fn(logit_upsampled, val_true_mask_tensor).item()
                prob_upsampled = torch.sigmoid(logit_upsampled)
                dice_val = dice_loss_fn(prob_upsampled, val_true_mask_tensor).item()

                pred_binary = (prob_upsampled > 0.5).float()
                inter = (val_true_mask_tensor * pred_binary).sum()
                union = val_true_mask_tensor.sum() + pred_binary.sum() - inter
                iou_actual = inter / (union + 1e-6)
                if union == 0:
                    iou_actual = torch.tensor(1.0, device=device)

                score_val = torch.abs(scores[best_idx] - iou_actual).item()
                val_loss += 0.5 * bce_val + 0.5 * dice_val + 0.05 * score_val

                d_metric = dice_coeff_from_probs(prob_upsampled, val_true_mask_tensor)
                j_metric = iou_from_probs(prob_upsampled, val_true_mask_tensor)
                dice_sum += d_metric
                iou_sum  += j_metric
                n_items  += 1

                pred_binary_np = pred_binary.cpu().numpy()
                true_binary_np = val_true_mask_tensor.cpu().numpy()
                val_correct += np.sum(pred_binary_np == true_binary_np)
                val_total   += np.prod(true_binary_np.shape)

                true_flat  = true_binary_np.flatten()
                pred_flat  = pred_binary_np.flatten()
                max_samples = self.site_config.get('max_pixel_samples', 10000)
                total_pixels = len(true_flat)
                score_flat = prob_upsampled.squeeze().cpu().numpy().flatten()

                if total_pixels > max_samples:
                    sample_indices = np.random.choice(total_pixels, max_samples, replace=False)
                    true_sampled  = true_flat[sample_indices]
                    pred_sampled  = pred_flat[sample_indices]
                    score_sampled = score_flat[sample_indices]
                else:
                    true_sampled  = true_flat
                    pred_sampled  = pred_flat
                    score_sampled = score_flat

                self.val_true_list.extend(int(x) for x in true_sampled)
                self.val_pred_list.extend(int(x) for x in pred_sampled)
                self.val_score_list.extend(score_sampled.tolist())

                # Overlay images: only on the last epoch to avoid repeated disk I/O
                if save_overlays and val_idx < self.validation_overlay_samples:
                    img_vis = val_image.copy()
                    overlay = (prob_upsampled.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    overlay_color = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
                    blended = cv2.addWeighted(img_vis, 1.0, overlay_color, 0.4, 0.0)
                    overlay_dir = os.path.join(self.model_output_folder, "validation_overlays")
                    os.makedirs(overlay_dir, exist_ok=True)
                    out_path = os.path.join(
                        overlay_dir,
                        f"{self.formatted_time}_{self.site_name}_val_overlay_e{len(self.epoch_list)}_{val_idx}.png"
                    )
                    cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

                del logit_tensor, logit_upsampled, prob_upsampled, pred_binary, val_true_mask_tensor, iou_actual

                progressBar.setValue(progressBar.getValue() + 1)

        # Compute averages
        avg_val_loss = val_loss / n_items if n_items > 0 else 0.0
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        avg_val_dice = dice_sum / n_items if n_items > 0 else 0.0
        avg_val_iou = iou_sum / n_items if n_items > 0 else 0.0
        miou = compute_mean_iou(self.val_true_list, self.val_pred_list)

        # Clear GPU cache after validation to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            cur_mb = torch.cuda.memory_allocated() / 1024 ** 2
            res_mb = torch.cuda.memory_reserved() / 1024 ** 2
            trend = ""
            if self._prev_val_vram_mb is not None:
                delta = cur_mb - self._prev_val_vram_mb
                trend = f"  delta={delta:+.1f} MB"
                if delta > 50:
                    trend += "  <<< VRAM INCREASING - possible leak"
            print(f"[VRAM] Post-validation: allocated={cur_mb:.1f} MB  reserved={res_mb:.1f} MB{trend}")
            self._prev_val_vram_mb = cur_mb

        return avg_val_loss, val_accuracy, miou, avg_val_dice, avg_val_iou

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _save_model_checkpoint(self, predictor, learnrate, epochs, suffix="final", val_loss=None, val_accuracy=None,
                               miou=None, target_category_name=None):
        """
        Save the current model state_dict and metadata to a .torch file.
        Only saves if this is a top-N best checkpoint by validation loss.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # num_classes is the number of annotation categories defined for this site
        num_classes = len(self.categories) if self.categories else None

        ckpt = {
            "model_state_dict": predictor.model.state_dict(),
            "categories": self.categories,
            "creation_UTC": timestamp,
            "site_name": self.site_name,
            "learning_rate": learnrate,
            "epochs": epochs,
            "num_classes": num_classes,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "miou": miou,
            "target_category_name": target_category_name,
            "base_model": "sam2",
            "category_centroids": self.category_centroids,
            "blob_filter_radius": self.blob_filter_radius
        }

        # Only save best validation checkpoints (not "final")
        if suffix.startswith("valbest") and val_loss is not None:
            # Check if this checkpoint would be in top N
            temp_list = self.best_checkpoints + [(val_loss, None)]
            temp_list.sort(key=lambda x: x[0])

            # Only proceed if this checkpoint makes the top N
            if len(temp_list) <= self.max_best_checkpoints or val_loss <= temp_list[self.max_best_checkpoints - 1][0]:

                # Create temporary checkpoint name (will be renamed based on rank)
                temp_filename = f"temp_{timestamp}_{self.site_name}_{target_category_name}_ep{epochs:03d}_lr{learnrate}.torch"
                temp_path = os.path.join(self.model_output_folder, temp_filename)

                # Save the checkpoint
                torch.save(ckpt, temp_path)

                # Add to tracking list with temp path
                self.best_checkpoints.append((val_loss, temp_path))

                # Sort by validation loss (ascending - best first)
                self.best_checkpoints.sort(key=lambda x: x[0])

                # Remove worst checkpoint if exceeding max
                if len(self.best_checkpoints) > self.max_best_checkpoints:
                    _, worst_path = self.best_checkpoints.pop()
                    if os.path.exists(worst_path):
                        try:
                            os.remove(worst_path)
                            print(f"Removed checkpoint: {os.path.basename(worst_path)}")
                        except Exception as e:
                            print(f"Warning: could not delete checkpoint {worst_path}: {e}")

                # Rename all checkpoints with meaningful rank names
                rank_names = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]

                for rank, (loss, old_path) in enumerate(self.best_checkpoints, start=1):
                    rank_suffix = rank_names[rank - 1] if rank <= len(rank_names) else f"{rank}th"

                    # Extract epoch number from the checkpoint metadata or filename
                    epoch_num = epochs if old_path == temp_path else self._extract_epoch_from_path(old_path)

                    # Create new meaningful name with category
                    new_filename = f"best_{rank_suffix}_{target_category_name}_epoch{epoch_num:03d}_valloss{loss:.4f}_lr{learnrate}.torch"
                    new_path = os.path.join(self.model_output_folder, new_filename)

                    # Rename if needed
                    if old_path != new_path and os.path.exists(old_path):
                        try:
                            os.rename(old_path, new_path)
                            # Update the path in the list
                            self.best_checkpoints[rank - 1] = (loss, new_path)
                        except Exception as e:
                            print(f"Warning: could not rename {old_path} to {new_path}: {e}")

                # Print current best checkpoints
                print(f"\nTop {len(self.best_checkpoints)} checkpoints:")
                for rank, (loss, path) in enumerate(self.best_checkpoints, start=1):
                    print(f"  {rank}. {os.path.basename(path)} (val_loss={loss:.4f})")

                return temp_path if temp_path in [p for _, p in self.best_checkpoints] else self.best_checkpoints[0][1]
            else:
                print(
                    f"Checkpoint at epoch {epochs} (val_loss={val_loss:.4f}) not in top {self.max_best_checkpoints}, not saved")
                return None

        return None

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _extract_epoch_from_path(self, path):
        """Extract epoch number from checkpoint filename."""
        import re
        match = re.search(r'epoch(\d+)', os.path.basename(path))
        if match:
            return int(match.group(1))
        return 0

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def get_best_checkpoint_path(self):
        """Returns the path to the best checkpoint (lowest validation loss)."""
        if self.best_checkpoints:
            return self.best_checkpoints[0][1]
        return None

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def build_unique_categories(self, annotation_files):
        merged = []
        id_to_name = {}
        name_to_id = {}

        for p in annotation_files:
            try:
                data = json.load(open(p, 'r'))
                cats = data.get('categories', [])
            except Exception as e:
                print(f"Failed loading '{p}': {e}")
                continue

            for cat in cats:
                cid = cat.get('id')
                cname = cat.get('name')
                if cid is None or cname is None:
                    print(f"Warning: bad category entry in '{p}': {cat}")
                    continue

                # Check ID conflict BEFORE assignment
                if cid in id_to_name and id_to_name[cid] != cname:
                    print(f"⚠️ ID conflict: {cid} is '{id_to_name[cid]}' and '{cname}'")
                    continue

                # Check name conflict BEFORE assignment
                if cname in name_to_id and name_to_id[cname] != cid:
                    print(f"⚠️ Name conflict: '{cname}' → {name_to_id[cname]} vs {cid}")
                    continue

                # Only assign if no conflicts
                id_to_name[cid] = cname
                name_to_id[cname] = cid
                merged.append({"id": cid, "name": cname})

        # dedupe
        unique = {(c['id'], c['name']): c for c in merged}
        return list(unique.values())

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def save_config_to_text(self, output_text_file):
        with open(output_text_file, 'w') as text_file:
            # Write the details to the text file
            text_file.write(f"Site: {self.site_name}\n")
            text_file.write(f"Learning Rates: {self.learning_rates}\n")
            text_file.write(f"Optimizer: {self.optimizer_type}\n")
            text_file.write(f"Loss Function: {self.loss_function}\n")
            text_file.write(f"Weight Decay: {self.weight_decay}\n")
            text_file.write(f"Number of Epochs: {self.num_epochs}\n")
            text_file.write(f"Save Model Frequency: {self.save_model_frequency}\n")
            text_file.write(f"Early Stopping: {self.early_stopping}\n")
            text_file.write(f"Patience: {self.patience}\n")
            text_file.write(f"Device: {device}\n")
            text_file.write(f"Folders: {self.folders}\n")
            text_file.write(f"Annotations: {self.annotation_files}\n")

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _plot_training_graphs(self, lr: float):
        """
        Generate and save all training/validation plots for a given learning rate.
        """
        progressBar = QProgressWheel(
            title="Generating graphs...", total=9,
            on_close=lambda: setattr(self, "progress_bar_closed", True),
            parent=self.parent_widget
        )

        viz = ModelTrainingVisualization(
            self.model_output_folder, self.formatted_time, self.categories
        )

        # Create epoch lists for training (all epochs) and validation (only validated epochs)
        train_epochs = list(range(1, len(self.loss_values) + 1))
        val_epochs = self.epoch_list  # Only epochs where validation ran

        # 1) Loss curves - use DIFFERENT epoch lists for train vs val
        viz.plot_loss_curves(
            train_epochs=train_epochs,
            train_loss=self.loss_values,
            val_epochs=val_epochs,
            val_loss=self.val_loss_values,
            site_name=self.site_name,
            lr=lr,
            model_name="SAM2",
            epochs=self.num_epochs,
            num_train_images=getattr(self, "num_train_images", None),
            num_val_images=getattr(self, "num_val_images", None),
            weight_decay=self.weight_decay,
            blob_radius_pct=round(self._report_blob_radius_frac * 100, 1),
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 2) Accuracy curves - same issue
        viz.plot_accuracy(
            train_epochs=train_epochs,
            train_acc=self.train_accuracy_values,
            val_epochs=val_epochs,
            val_acc=self.val_accuracy_values,
            site_name=self.site_name,
            lr=lr,
            model_name="SAM2",
            epochs=self.num_epochs,
            num_train_images=getattr(self, "num_train_images", None),
            num_val_images=getattr(self, "num_val_images", None),
            weight_decay=self.weight_decay,
            blob_radius_pct=round(self._report_blob_radius_frac * 100, 1),
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 3) Confusion matrix
        cm_png = viz.plot_confusion_matrix(
            y_true=self.val_true_list,
            y_pred=self.val_pred_list,
            site_name=self.site_name,
            lr=lr,
            normalize=True,
            file_prefix="Normalized"
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 4) ROC Curve + AUC
        roc_png = viz.plot_roc_curve(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 5) Precision–Recall
        pr_png = viz.plot_precision_recall(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 6) F1 vs. Threshold
        f1_png = viz.plot_f1_score(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 7) Mean IoU curve
        miou_png = viz.plot_miou_curve(
            epochs=self.epoch_list,
            miou_values=self.miou_values,
            site_name=self.site_name,
            lr=lr,
            model_name="SAM2",
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}",
            num_epochs=self.num_epochs,
            num_train_images=getattr(self, "num_train_images", None),
            num_val_images=getattr(self, "num_val_images", None),
            weight_decay=self.weight_decay,
            blob_radius_pct=round(self._report_blob_radius_frac * 100, 1),
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        dice_png = viz.plot_dice_curve(
            epochs=self.epoch_list,
            dice_values=self.val_dice_values,
            site_name=self.site_name,
            lr=lr,
            model_name="SAM2",
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}",
            num_epochs=self.num_epochs,
            num_train_images=getattr(self, "num_train_images", None),
            num_val_images=getattr(self, "num_val_images", None),
            weight_decay=self.weight_decay,
            blob_radius_pct=round(self._report_blob_radius_frac * 100, 1),
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        iou_png = viz.plot_iou_curve(
            epochs=self.epoch_list,
            iou_values=self.val_iou_values,
            site_name=self.site_name,
            lr=lr,
            model_name="SAM2",
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}",
            num_epochs=self.num_epochs,
            num_train_images=getattr(self, "num_train_images", None),
            num_val_images=getattr(self, "num_val_images", None),
            weight_decay=self.weight_decay,
            blob_radius_pct=round(self._report_blob_radius_frac * 100, 1),
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        progressBar.close()

        # ── PDF Diagnostic Report ──────────────────────────────────────────
        acc_png  = os.path.join(viz.graphs_folder, f"{self.formatted_time}_{self.site_name}_AccuracyCurves_lr{lr:.5f}.png")
        loss_png = os.path.join(viz.graphs_folder, f"{self.formatted_time}_{self.site_name}_LossCurves_lr{lr:.5f}.png")
        graphs = [g for g in [acc_png, loss_png, cm_png, roc_png, pr_png,
                           f1_png, miou_png, dice_png, iou_png] if g and os.path.isfile(g)]
        viz.save_training_report(
            train_acc=self.train_accuracy_values,
            val_acc=self.val_accuracy_values,
            train_loss=self.loss_values,
            val_loss=self.val_loss_values,
            site_name=self.site_name,
            lr=lr,
            miou_values=self.miou_values,
            dice_values=self.val_dice_values,
            graph_paths=graphs,
            model_type="SAM2",
            num_train_images=getattr(self, "num_train_images", None),
            num_val_images=getattr(self, "num_val_images", None),
            blob_radius_pct=round(self._report_blob_radius_frac * 100, 1),
            weight_decay=self.weight_decay,
            categories=self.categories,
            best_epoch=self._extract_epoch_from_path(self.best_checkpoints[0][1])
                if self.best_checkpoints else None,
            best_val_acc=self.val_accuracy_values[
                self.epoch_list.index(self._extract_epoch_from_path(self.best_checkpoints[0][1]))
            ] if self.best_checkpoints and self.val_accuracy_values and
                self._extract_epoch_from_path(self.best_checkpoints[0][1]) in self.epoch_list else None,
            early_stopped=getattr(self, "early_stopped", False),
            early_stop_epoch=getattr(self, "early_stop_epoch", None),
            training_time_seconds=getattr(self, "training_time_seconds", None),
        )

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def summarize_mask_imbalance(self, images: list[str]) -> dict:
        """
        Computes mean/median/std of foreground ratio (water pixels / total) across given images.
        Uses dataset_util.load_true_mask and your annotation_index.
        """
        ratios = []
        
        # Get target label - handle TEST_MODE
        if self.TEST_MODE:
            # In TEST_MODE, use first test category
            target_label = self.TEST_CATEGORIES[0]
        else:
            # In normal mode, use config
            target_label = self.site_config["train_model"]["TRAINING_CATEGORIES"][0]
        
        target_id = next((c["id"] for c in self.categories if c["name"] == target_label), None)
        for img in images:
            m_result = self.dataset_util.load_true_mask(img, self.annotation_index, mode="binary", target_id=target_id)
            m = m_result[0] if isinstance(m_result, tuple) else m_result
            if m is None:
                continue

            m = unwrap_mask(m)
            if m.ndim == 3:
                m = m[..., 0]
            total = m.size
            fg = (m > 0).sum()
            ratios.append(float(fg) / float(total) if total > 0 else 0.0)

        if not ratios:
            return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0}

        arr = np.array(ratios, dtype=np.float64)

        return {"count": len(ratios),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std())}

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _terminate_training(self, progressBar):
        msg = "You have cancelled the model training currently in-progress. A model has not been generated."
        msgBox = GRIME_AI_QMessageBox('Model Training Terminated', msg, GRIME_AI_QMessageBox.Close)
        msgBox.displayMsgBox()

        """Clean up progress bar."""
        try:
            if progressBar is not None and progressBar.isVisible():
                progressBar.stop()
                progressBar.hide()
        except RuntimeError:
            # Progress bar already deleted by Qt
            pass

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _terminate_validation(self, progressBar):
        msg = "You have cancelled the validation currently in-progress."
        msgBox = GRIME_AI_QMessageBox('Validation Terminated', msg, GRIME_AI_QMessageBox.Close)
        msgBox.displayMsgBox()

        if progressBar and progressBar.isVisible():
            progressBar.close()

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _deduplicate_centroids(self, tolerance_px=5):
        """
        Deduplicate centroids within each category to prevent memory leak.
        Keeps only unique centroids within tolerance_px distance.
        Call after first epoch to establish representative centroid set.
        """
        # ============================================================
        # WRITE CENTROID DIAGNOSTIC REPORT (epoch 0 data)
        # ============================================================
        if hasattr(self, "_centroid_diag_rows") and self._centroid_diag_rows and self.model_output_folder:
            import csv
            _diag_folder = os.path.join(self.model_output_folder, "diagnostics")
            os.makedirs(_diag_folder, exist_ok=True)
            diag_path = os.path.join(
                _diag_folder,
                f"{self.formatted_time}_{self.site_name}_centroid_diagnostic.csv"
            )
            fieldnames = ["image", "outcome", "centroid_px", "centroid_norm", "detail"]
            try:
                with open(diag_path, "w", newline="", encoding="utf-8") as f_diag:
                    writer = csv.DictWriter(f_diag, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self._centroid_diag_rows)

                # Print summary
                from collections import Counter
                outcomes = Counter(r["outcome"] for r in self._centroid_diag_rows)
                total = len(self._centroid_diag_rows)
                print(f"\n[Centroid Diagnostic] {total} images processed in epoch 0:")
                for outcome, count in sorted(outcomes.items()):
                    print(f"  {outcome:<30} : {count}")
                print(f"  Report written to: {diag_path}")
            except Exception as e:
                print(f"[Centroid Diagnostic] Failed to write report: {e}")
            finally:
                del self._centroid_diag_rows  # free memory

        print(f"\n[Centroid Deduplication] tolerance_px={tolerance_px}")
        for cat_id, centroids in self.category_centroids.items():
            if not centroids:
                print(f"  Category {cat_id}: 0 centroids (skipped)")
                continue

            before_count = len(centroids)
            unique_centroids = []
            for new_centroid in centroids:
                cx_new, cy_new = new_centroid["centroid_px"]
                is_unique = True

                for existing in unique_centroids:
                    cx_exist, cy_exist = existing["centroid_px"]
                    dist = ((cx_new - cx_exist) ** 2 + (cy_new - cy_exist) ** 2) ** 0.5
                    if dist < tolerance_px:
                        is_unique = False
                        break

                if is_unique:
                    unique_centroids.append(new_centroid)

            after_count = len(unique_centroids)
            removed = before_count - after_count
            self.category_centroids[cat_id] = unique_centroids
            print(f"  Category {cat_id}: {before_count} collected  ->  {after_count} unique  "
                  f"({removed} removed as duplicates within {tolerance_px}px)")
