#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# torch.use_deterministic_algorithms(True) (set in train_sam) requires this
# for cuBLAS matmuls on CUDA >= 10.2; without it the first GPU matmul under
# deterministic mode raises RuntimeError. cuBLAS reads it at handle creation,
# so it must be set before any CUDA matmul runs in the process. setdefault
# preserves an externally configured value. NOTE: if the host application
# performs GPU work before this module is imported, this is too late -- the
# entry point (main.py) must set it instead.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import random
import math
import time as _time
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
from GRIME_AI.dialogs.ML_image_processing.model_config_manager import ModelConfigManager

from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI.ml_core.model_training_visualization import ModelTrainingVisualization
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI.dialogs.ML_image_processing.model_config_manager import ModelConfigManager

from GRIME_AI.utils.datasetutils import DatasetUtils
from GRIME_AI.ml_core.ml_helpers import (build_centroid_point_prompts, sample_pooled_prompts,
                                         DEFAULT_MAX_POSITIVES as _DEFAULT_MAX_POSITIVES,
                                         DEFAULT_NEGATIVE_BALANCE as _DEFAULT_NEGATIVE_BALANCE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(sam2_base.__file__)

# Sentinel distinguishing "image not yet seen" from "image cached as unusable (None)"
# in the cross-epoch training cache. See _train_one_epoch.
_CACHE_MISS = object()


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
        # Populated from the site config in _load_site_config(); see schema default.
        self.max_best_checkpoints = ModelConfigManager.get_default("max_best_checkpoints")

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
        # Optimizer and loss are fixed by the trainer (AdamW; 0.5*BCE + 0.5*Dice
        # + 0.05*score). The GUI selectors were removed; do not read stale
        # 'optimizer' / 'loss_function' keys from site_config or the report
        # will describe something that never ran.
        self.optimizer_type = "AdamW"
        self.loss_function = "BCE + Dice + Score"
        self.weight_decay = self.site_config['weight_decay']
        self.num_epochs = self.site_config['number_of_epochs']
        self.max_best_checkpoints = int(self.site_config.get(
            'max_best_checkpoints', ModelConfigManager.get_default('max_best_checkpoints')))
        self.early_stopping = self.site_config['early_stopping']
        self.patience = self.site_config['patience']

        # Mini-batch size for SAM2 training (GUI "Batch Size" spinbox).
        # Accepts either key spelling; clamped to >= 1. Resolved value is
        # printed at the start of training so a misnamed config key is
        # visible immediately (it will resolve to 1).
        self.batch_size = max(1, int(self.site_config.get(
            'batch_size', self.site_config.get('batchSize', 1))))

        # Blob filter radius — stored as fraction of image diagonal
        # Default: 50px / diagonal of a 2000x1000 image ≈ 0.02236
        import math as _math
        _DEFAULT_BLOB_FRACTION = 50.0 / _math.sqrt(2000**2 + 1000**2)
        self.blob_filter_radius = float(self.site_config.get('blob_filter_radius', _DEFAULT_BLOB_FRACTION))
        # Mahalanobis-derived fallback fraction, kept SEPARATE from
        # blob_filter_radius so the Circular filter's radius is never poisoned.
        self.blob_filter_mahal_fraction = None

        # Validation overlay settings (default: 5 samples per epoch)
        _D = ModelConfigManager.get_default
        self.validation_overlay_samples = self.site_config.get(
            'validation_overlay_samples', _D('validation_overlay_samples'))

        # Overlay write policy.
        #   'last'     -> only the final epoch (legacy behavior)
        #   'every'    -> every validation epoch
        #   'interval' -> every Nth epoch, per validation_overlay_interval
        # Regardless of mode, a guaranteed final overlay pass runs if the epoch
        # loop exits early (early stopping, divergence, cancel) without having
        # written any overlays. See _ensure_overlays_written().
        self.validation_overlay_mode = str(
            self.site_config.get('validation_overlay_mode', _D('validation_overlay_mode'))
        ).lower()
        self.validation_overlay_interval = int(
            self.site_config.get('validation_overlay_interval', _D('validation_overlay_interval'))
        )
        # Set True by _validate_one_epoch the first time it writes overlay PNGs.
        self._overlays_written = False

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
        print(f"Model loaded: {total_params:,} total parameters. "
              f"Actual trained modules are reported by the gradient-flow audit at the first training step.")

        self.sam2_model = model.to(device).train()

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

        # Progress budget: per epoch, one tick per trained sample and per
        # validation image, PLUS a one-time Phase 1 budget of two ticks per
        # training image (one for preflight prep, one for encoding). Later
        # epochs re-enter Phase 1 only for transient retries, which are
        # negligible against this budget.
        total_iterations = epochs * (len(train_images) + (len(val_images) if val_images else 0)) \
            + 2 * len(train_images)

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
        # Scheduler patience must stay strictly below early-stopping patience,
        # otherwise training halts before the LR is ever reduced and the
        # scheduler is effectively dead code. Default to one-third of the
        # early-stopping patience (min 1) unless explicitly configured.
        _D = ModelConfigManager.get_default
        self.lr_scheduler_enabled = bool(
            self.site_config.get('lr_scheduler_enabled', _D('lr_scheduler_enabled')))
        self.lr_scheduler_factor = float(
            self.site_config.get('lr_scheduler_factor', _D('lr_scheduler_factor')))
        self.lr_scheduler_patience = int(
            self.site_config.get('lr_scheduler_patience', _D('lr_scheduler_patience')))
        self.lr_scheduler_min_lr = float(
            self.site_config.get('lr_scheduler_min_lr', _D('lr_scheduler_min_lr')))

        if self.early_stopping and self.lr_scheduler_patience >= int(self.patience):
            print(f"[SAM2Trainer] WARNING: LR scheduler patience "
                  f"({self.lr_scheduler_patience}) >= early-stopping patience "
                  f"({self.patience}). Training will stop before the learning "
                  f"rate is ever reduced.")

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',                        # minimize validation loss
            factor=self.lr_scheduler_factor,   # multiply LR by this on plateau
            patience=self.lr_scheduler_patience,
            min_lr=self.lr_scheduler_min_lr
        ) if self.lr_scheduler_enabled else None

        if self.lr_scheduler_enabled:
            print(f"[SAM2Trainer] ReduceLROnPlateau active: monitor=val_loss "
                  f"mode=min factor={self.lr_scheduler_factor} "
                  f"patience={self.lr_scheduler_patience} "
                  f"min_lr={self.lr_scheduler_min_lr:g} initial_lr={learnrate:g}")
        else:
            print("[SAM2Trainer] LR scheduler disabled; learning rate is fixed.")

        # Per-epoch LR history for the diagnostics report
        self.lr_values = []

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
        self._overlays_written = False

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
            self.time_train_seconds = 0.0
            self.time_val_seconds = 0.0
            self.time_encode_seconds = 0.0   # epoch-0 preflight + embedding cache (train) and val encode
            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")
                _t_epoch = _time.perf_counter()

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

                _dt = _time.perf_counter() - _t_epoch
                _enc = float(getattr(self, "_last_encode_seconds", 0.0))
                self.time_encode_seconds += _enc
                self.time_train_seconds += max(0.0, _dt - _enc)
                self._last_encode_seconds = 0.0

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
                if epoch == 0 and not getattr(self, "_centroids_finalized", False):
                    print("\n=== Finalizing Centroid Collection ===")
                    self._deduplicate_centroids()
                    self._centroids_finalized = True


                if val_cache:
                    _t_val = _time.perf_counter()
                    avg_val_loss, val_accuracy, miou, avg_val_dice, avg_val_iou = self._validate_one_epoch(
                        val_cache=val_cache,
                        predictor=predictor,
                        progressBar=progressBar,
                        target_label=target_label,
                        save_overlays=self._should_save_overlays(epoch + 1, epochs)
                    )

                    _dt_val = _time.perf_counter() - _t_val
                    _enc = float(getattr(self, "_last_encode_seconds", 0.0))
                    self.time_encode_seconds += _enc
                    self.time_val_seconds += max(0.0, _dt_val - _enc)
                    self._last_encode_seconds = 0.0

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
                    self._step_lr_scheduler(scheduler, optimizer, avg_val_loss, epoch + 1)

            # ----------------------------------------------------------------
            # POST-TRAINING: backstop so an early exit still produces overlays
            # ----------------------------------------------------------------
            self._ensure_overlays_written(val_cache, predictor, progressBar, target_label)

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
        Runs one training epoch with true mini-batching.
        Returns (avg_epoch_loss, train_accuracy, avg_dice, avg_iou).

        Structure (per epoch):
          PHASE 1 -- PREFLIGHT + ENCODE: images not yet in the cross-epoch
            cache get their mask decoded, prompts derived, and embeddings
            computed. Encoding runs through predictor.set_image_batch() in
            chunks of self.batch_size (per-image set_image() fallback if the
            installed sam2 lacks batch support). On the first epoch this is
            all images; on later epochs it is only transient retries
            (e.g. previously file-locked images).
          PHASE 2 -- BATCHED TRAINING: samples are drawn from the cache in
            chunks of self.batch_size. Per step: embeddings are stacked
            [B,...], prompt sets are padded to the max count in the batch
            (pad label -1 = SAM2 not-a-point), one prompt-encoder and one
            mask-decoder forward runs for the whole batch
            (repeat_image=False since each sample has its own embedding),
            per-sample losses are averaged, and the optimizer steps once
            per batch. Gradients are therefore averaged over batch_size
            samples per update.

        The cross-epoch cache invariant is unchanged: set_image()/
        set_image_batch() run under no_grad, so the image encoder receives
        no gradient updates and embeddings are constant across epochs. In
        this batched design the decoder consumes cached embeddings directly,
        which makes encoder training impossible by construction; the
        gradient-flow audit will flag a configuration error if encoder
        gradients ever appear.

        CORRECTIONS PRESERVED FROM PRIOR VERSIONS:
        - BCEWithLogitsLoss on logits; sigmoid only for Dice/metrics
        - Direct progress_bar_closed attribute access (Issue #8)
        - Blob filter: uint8 GPU-side cast, connectedComponentsWithStats,
          vectorized isin rebuild (output-identical to per-blob loop)
        - Pixel accuracy computed on-GPU, scalar-only transfer
        """
        import torch.nn.functional as F

        self.sam2_model.train()

        epoch_loss = 0.0
        train_correct, train_total = 0, 0
        processed_count = 0
        dice_sum, iou_sum = 0.0, 0.0

        np.random.shuffle(train_images)

        # Define loss function once (not per-sample)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        dice_loss_fn = DiceLoss()

        if target_label is None:
            raise ValueError("target_label is required but was None.")

        # RESOLVE TARGET_ID (once per epoch; deterministic per category)
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

        # ================================================================
        # CROSS-EPOCH TRAINING CACHE (embeddings + masks + prompts)
        # ----------------------------------------------------------------
        # set_image()/set_image_batch() run under no_grad, so gradients flow
        # only through the prompt encoder and mask decoder. Embeddings are
        # therefore constant across epochs and cached on first encounter,
        # along with the decoded ground-truth mask and prompt centroids.
        #
        # Cache entry per image_file:
        #   dict  -> usable sample (embeddings on CPU/pinned)
        #   None  -> structurally unusable (empty mask, no foreground
        #            blobs); skipped without I/O on later epochs
        #   absent-> transient failure (file lock); retried next epoch
        #
        # Host RAM cost is ~17 MB/image for hiera-large fp32 embeddings.
        # Negative centroids are cached UNBALANCED; the 3:1 random
        # subsampling happens per epoch at batch-assembly time.
        # ================================================================
        if not hasattr(self, "cache_embeddings"):
            self.cache_embeddings = True
        if epoch == 0 or not hasattr(self, "_train_cache"):
            self._train_cache = {}
            self._pinned_bytes = 0

        # Pinned (page-locked) host memory cannot be paged out and, on
        # Windows/WDDM, draws from a constrained pool; unbounded pinning at
        # ~17 MB/image destabilizes the OS on large datasets (1266 images
        # ~= 21 GB page-locked). Pin only up to a budget; beyond it, cache
        # in ordinary pageable memory. Batch-assembly non_blocking copies
        # silently degrade to synchronous for unpinned tensors -- slower,
        # never incorrect. Override with self.pin_budget_bytes.
        _pin_budget = getattr(self, "pin_budget_bytes", 4 * 1024 ** 3)

        def _cache_tensor(t):
            t = t.detach().cpu()
            if device.type == "cuda" and self._pinned_bytes + t.nbytes <= _pin_budget:
                try:
                    t = t.pin_memory()
                    self._pinned_bytes += t.nbytes
                except RuntimeError:
                    # Pinned allocation failed; pageable tensor is fine.
                    pass
            return t

        batch_size = max(1, int(getattr(self, "batch_size", 1)))
        if epoch == 0:
            print(f"[SAM2Trainer] Mini-batch size: {batch_size} (from site_config)")

        # ================================================================
        # PHASE 1: PREFLIGHT + BATCHED ENCODING
        # ================================================================
        _t_phase1 = _time.perf_counter()
        pending = [f for f in train_images if f not in self._train_cache]
        prepped = []   # (image_file, image_np, prep dict)

        if pending:
            print(f"[SAM2Trainer] Phase 1: preflighting {len(pending)} images "
                  f"(mask decode + prompt derivation)...", flush=True)

        for idx, image_file in enumerate(pending):
            if self.progress_bar_closed:
                self._terminate_training(progressBar)
                return None, None, None, None

            # One tick per preflighted image regardless of outcome; this both
            # advances the bar and pumps the Qt event loop so the wheel keeps
            # repainting during CPU-bound prep (training runs on the GUI thread).
            progressBar.setValue(progressBar.getValue() + 1)

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
                self._train_cache[image_file] = None
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
                # Transient rather than structural: do not cache None; the
                # image is retried on the next epoch.
                continue

            # NOTE: the image itself is deliberately NOT loaded here.
            # Preflight needs only masks and centroids; decoding every frame
            # up front and retaining it until encoding finished held all
            # full-res RGB arrays in RAM at once (~6 MB/image at 1080p,
            # ~7.5 GB for 1266 images; 4x that at 4K). Images are now
            # loaded per encode chunk below and released immediately.

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
                self._train_cache[image_file] = None
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
                nh, nw = neg_mask.shape[:2]
                for lbl in range(1, num_labels_neg):
                    ys, xs = np.nonzero(labels_neg == lbl)
                    if xs.size > 0:
                        cx, cy = float(xs.mean()), float(ys.mean())
                        negative_coords.append([cx, cy])
                        # Record non-target centroids too. The checkpoint's
                        # category_centroids feeds inference; without these,
                        # deployment has no negative prompts at all.
                        if epoch == 0:
                            self.category_centroids.setdefault(int(neg_id), []).append({
                                "centroid_px": (int(round(cx)), int(round(cy))),
                                "centroid_norm": _normalize_centroid(cx, cy, nw, nh),
                            })

            prepped.append((image_file, {
                "true_mask": true_mask,
                "positive_coords": positive_coords,
                "negative_coords": list(negative_coords),
            }))

        # ---- BATCHED ENCODING of newly prepared images ----
        # The ENCODE chunk size is deliberately decoupled from the training
        # batch size. Encoding is activation-heavy (~1.5 GB VRAM per image
        # through hiera-large at 1024x1024, fp32), and on Windows/WDDM an
        # oversubscribed batch does not raise OOM -- it silently spills to
        # shared system memory and runs orders of magnitude slower. The chunk
        # is therefore sized from actually-free VRAM, capped by batch_size.
        # Training batches (decoder-only, cheap) still use full batch_size.
        # Override with self.encode_batch_size or site_config
        # 'encode_batch_size' if the estimate misjudges a given card.
        encode_chunk = 1
        if device.type == "cuda":
            try:
                free_bytes, _total = torch.cuda.mem_get_info()
                _PER_IMAGE_ENCODE_BYTES = int(1.5 * 1024 ** 3)
                encode_chunk = max(1, int((free_bytes * 0.6) // _PER_IMAGE_ENCODE_BYTES))
            except Exception:
                encode_chunk = 1
        encode_chunk = min(encode_chunk, batch_size)
        _cfg_encode = self.site_config.get('encode_batch_size', getattr(self, 'encode_batch_size', None))
        if _cfg_encode:
            encode_chunk = max(1, int(_cfg_encode))
        if prepped:
            print(f"[SAM2Trainer] Phase 1: encoding {len(prepped)} images "
                  f"(encode chunk={encode_chunk}, training batch={batch_size})...", flush=True)

        newly_encoded = 0
        for start in range(0, len(prepped), encode_chunk):
            if self.progress_bar_closed:
                self._terminate_training(progressBar)
                return None, None, None, None

            chunk = prepped[start:start + encode_chunk]

            # Load this chunk's images only now; a file that vanished or got
            # locked between preflight and encode is skipped without caching,
            # so it is retried next epoch (same policy as the preflight
            # file-missing case).
            loaded = []
            for image_file, prep in chunk:
                try:
                    img = np.array(Image.open(image_file).convert("RGB"))
                except (OSError, ValueError) as e:
                    print(f"[WARNING] Failed to load {image_file} at encode time ({e}); "
                          f"will retry next epoch.")
                    continue
                loaded.append((image_file, img, prep))

            if not loaded:
                if not self.progress_bar_closed:
                    progressBar.setValue(progressBar.getValue() + len(chunk))
                continue

            imgs = [c[1] for c in loaded]

            encoded_batched = False
            if len(imgs) > 1 and hasattr(predictor, "set_image_batch"):
                try:
                    predictor.set_image_batch(imgs)
                    encoded_batched = True
                except Exception as e:
                    print(f"[SAM2Trainer] set_image_batch failed ({e}); "
                          f"falling back to per-image encoding for this chunk.")

            if encoded_batched:
                for i, (image_file, _img, prep) in enumerate(loaded):
                    self._train_cache[image_file] = {
                        "image_embed": _cache_tensor(predictor._features["image_embed"][i:i + 1]),
                        "high_res_feats": [_cache_tensor(t[i:i + 1]) for t in predictor._features["high_res_feats"]],
                        "orig_hw": [predictor._orig_hw[i]],
                        "true_mask": prep["true_mask"],
                        "positive_coords": prep["positive_coords"],
                        "negative_coords": prep["negative_coords"],
                    }
                    newly_encoded += 1
            else:
                for image_file, img, prep in loaded:
                    predictor.set_image(img)
                    self._train_cache[image_file] = {
                        "image_embed": _cache_tensor(predictor._features["image_embed"]),
                        "high_res_feats": [_cache_tensor(t) for t in predictor._features["high_res_feats"]],
                        "orig_hw": list(predictor._orig_hw),
                        "true_mask": prep["true_mask"],
                        "positive_coords": prep["positive_coords"],
                        "negative_coords": prep["negative_coords"],
                    }
                    newly_encoded += 1

            # One tick per encoded image, applied per chunk; also pumps the
            # Qt event loop between encoder calls so the wheel repaints.
            del loaded, imgs
            if not self.progress_bar_closed:
                progressBar.setValue(progressBar.getValue() + len(chunk))
            print(f"[SAM2Trainer]   encoded {min(start + len(chunk), len(prepped))}/{len(prepped)}", flush=True)

        del prepped
        self._last_encode_seconds = _time.perf_counter() - _t_phase1

        # Freeze and deduplicate the centroid population BEFORE the first
        # optimizer step so that pooled-prompt training, validation and the
        # checkpoint all see the same population.
        if epoch == 0 and not getattr(self, "_centroids_finalized", False):
            print("\n=== Finalizing Centroid Collection ===")
            self._deduplicate_centroids()
            self._centroids_finalized = True
            print(f"Total unique centroids: "
                  f"{sum(len(v) for v in self.category_centroids.values())} across "
                  f"{len(self.category_centroids)} categories\n")

        train_prompt_mode = getattr(self, "train_prompt_mode",
                                    self.site_config.get("train_prompt_mode", "pooled"))
        if epoch == 0:
            print(f"[SAM2Trainer] Prompt protocol: {train_prompt_mode} "
                  f"({'deployment-identical pooled centroids' if train_prompt_mode == 'pooled' else 'per-image ground-truth centroids'})")
        _prompt_rng = random.Random()  # unseeded: new draw every epoch (augmentation)

        # ================================================================
        # PHASE 2: BATCHED TRAINING from cache
        # ================================================================
        samples = [f for f in train_images if isinstance(self._train_cache.get(f), dict)]
        num_steps = 0
        _since_cache_clear = 0

        for start in range(0, len(samples), batch_size):
            if self.progress_bar_closed:
                self._terminate_training(progressBar)
                return None, None, None, None

            chunk_files = samples[start:start + batch_size]
            entries = [self._train_cache[f] for f in chunk_files]
            B = len(entries)

            # ---- Stack cached embeddings into batch tensors ----
            image_embed = torch.cat(
                [e["image_embed"] for e in entries], dim=0
            ).to(device, non_blocking=True)
            n_levels = len(entries[0]["high_res_feats"])
            high_res_features = [
                torch.cat([e["high_res_feats"][lv] for e in entries], dim=0
                          ).to(device, non_blocking=True)
                for lv in range(n_levels)
            ]

            # ---- Per-sample prompts, balanced per epoch, padded to Pmax ----
            coords_list, labels_list = [], []
            for i, e in enumerate(entries):
                if train_prompt_mode == "pooled":
                    # Deployment protocol: sample from the pooled population,
                    # exactly as sam2_inference_engine does at segmentation time.
                    hh, ww = e["orig_hw"][-1]
                    c, l = sample_pooled_prompts(
                        target_id, self.category_centroids, ww, hh,
                        max_positives=int(self.site_config.get("max_positives", _DEFAULT_MAX_POSITIVES)),
                        negative_balance=int(self.site_config.get("negative_balance", _DEFAULT_NEGATIVE_BALANCE)),
                        rng=_prompt_rng)
                    if c is None:  # cannot happen once preflight recorded >=1 target blob
                        c, l = e["positive_coords"], [1] * len(e["positive_coords"])
                    coords_list.append(c)
                    labels_list.append(l)
                    pos, neg = [p for p, lab in zip(c, l) if lab == 1], [p for p, lab in zip(c, l) if lab == 0]
                else:
                    pos = e["positive_coords"]
                    neg = list(e["negative_coords"])
                    if len(neg) > len(pos) * 3:
                        neg = random.sample(neg, len(pos) * 3)
                    coords_list.append(pos + neg)
                    labels_list.append([1] * len(pos) + [0] * len(neg))

                # Log prompt counts for monitoring (every 10th sample)
                if (processed_count + i) % 10 == 0:
                    print(f"  Image {os.path.basename(chunk_files[i])}: "
                          f"{len(pos)} positive, {len(neg)} negative prompts")

            Pmax = max(len(c) for c in coords_list)
            point_coords = torch.zeros((B, Pmax, 2), dtype=torch.float32, device=device)
            # Pad label -1 = SAM2 "not a point"; the prompt encoder embeds it
            # with not_a_point_embed, so padded slots contribute nothing.
            point_labels = torch.full((B, Pmax), -1, dtype=torch.int64, device=device)
            for i, (c, l) in enumerate(zip(coords_list, labels_list)):
                # Map pixel-space centroids into the model's 1024x1024 input
                # frame -- same transform predictor.predict() applies
                # internally. Without this the prompt encoder receives
                # out-of-frame coordinates and the prompts are noise.
                # Cached coords stay pixel-space (the blob filter needs them
                # in the original frame); the transform is applied only here.
                c_t = predictor._transforms.transform_coords(
                    torch.tensor(c, dtype=torch.float32),
                    normalize=True,
                    orig_hw=entries[i]["orig_hw"][-1],
                )
                point_coords[i, :len(c)] = c_t.to(device)
                point_labels[i, :len(l)] = torch.tensor(l, dtype=torch.int64)

            optimizer.zero_grad()

            # Pick dtype depending on device
            if device.type == "cuda":
                # bf16: same range as fp32, no loss scaling / overflow risk.
                # Falls back to fp16 + GradScaler on pre-Ampere cards.
                autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:  # CPU
                autocast_dtype = torch.bfloat16  # CPU autocast only supports bfloat16

            with autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                # Prompt embeddings for the whole batch
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None,
                )

                decoder_kwargs = dict(
                    image_embeddings=image_embed,
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,   # one embedding per prompt set
                    high_res_features=high_res_features,
                )

                # Use pre-selected SDPA backend (selected once in train_sam)
                if hasattr(self, 'selected_backend') and self.selected_backend is not None:
                    with sdpa_kernel(self.selected_backend):
                        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(**decoder_kwargs)
                else:
                    # Fallback to default if backend selection failed
                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(**decoder_kwargs)

                # ---- Per-sample losses, averaged over the batch ----
                batch_loss_sum = None
                batch_valid = 0
                for i in range(B):
                    true_mask = entries[i]["true_mask"]
                    h, w = true_mask.shape[:2]
                    gt_mask = torch.tensor(true_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
                    if gt_mask.sum() == 0:
                        # Defensive; preflight guarantees nonzero masks.
                        print(f"Skipping {chunk_files[i]} - ground-truth mask is empty for label {target_label}.")
                        continue

                    # Keep predictions as LOGITS (don't apply sigmoid yet)
                    prd_mask_logits = low_res_masks[i:i + 1, 0:1]  # [1,1,h,w] - raw logits

                    # Upsample logits to ground-truth resolution
                    if prd_mask_logits.shape[-2:] != gt_mask.shape[-2:]:
                        prd_mask_logits = F.interpolate(
                            prd_mask_logits,
                            size=gt_mask.shape[-2:],
                            mode="bilinear",
                            align_corners=False
                        )

                    if prd_mask_logits.shape != gt_mask.shape:
                        raise ValueError(f"Shape mismatch {prd_mask_logits.shape} vs {gt_mask.shape} for {chunk_files[i]}")

                    # USE BCEWithLogitsLoss on logits (more stable than manual BCE on probabilities)
                    seg_loss = bce_loss_fn(prd_mask_logits, gt_mask)

                    # FOR DICE LOSS, WE NEED PROBABILITIES, SO APPLY SIGMOID
                    prd_mask_probs = torch.sigmoid(prd_mask_logits)

                    dice_loss = dice_loss_fn(prd_mask_probs, gt_mask)

                    # COMPUTE IOU FOR SCORE PREDICTION LOSS
                    prd_mask_binary = (prd_mask_probs > 0.5).float()

                    # FILTER OUT BLOBS NOT OVERLAPPING CENTROID PROMPTS
                    # (uint8 on-GPU cast, WithStats single pass, vectorized
                    # isin rebuild -- output-identical to the per-blob loop)
                    labels_np = prd_mask_binary.squeeze().to(torch.uint8).cpu().numpy()
                    num_labels, labels, blob_stats, blob_centroids = cv2.connectedComponentsWithStats(labels_np)

                    # ADAPTIVE RADIUS THRESHOLD BASED ON IMAGE SIZE.
                    # FRACTION IS SET BY USER IN GUI AND STORED IN site_config + .torch METADATA.
                    img_diagonal = math.sqrt(h * h + w * w)
                    radius_threshold = max(10, int(self.blob_filter_radius * img_diagonal))

                    kept_labels = []
                    for lbl in range(1, num_labels):  # skip background
                        if blob_stats[lbl, cv2.CC_STAT_AREA] == 0:
                            continue
                        cx_blob, cy_blob = blob_centroids[lbl]  # (x, y) pixel-mean centroid

                        # KEEP ONLY IF BLOB CENTROID IS CLOSE TO ONE OF THE PROMPT CENTROIDS
                        for cx, cy in entries[i]["positive_coords"]:
                            if np.linalg.norm([cx - cx_blob, cy - cy_blob]) < radius_threshold:
                                kept_labels.append(lbl)
                                break

                    if kept_labels:
                        valid_mask = np.isin(labels, kept_labels).astype(np.uint8)
                    else:
                        valid_mask = np.zeros_like(labels, dtype=np.uint8)

                    prd_mask_binary = torch.tensor(valid_mask, device=device).unsqueeze(0).unsqueeze(1).float()

                    inter = (gt_mask * prd_mask_binary).sum((1, 2, 3))
                    union = gt_mask.sum((1, 2, 3)) + prd_mask_binary.sum((1, 2, 3)) - inter
                    iou = inter / (union + 1e-6)
                    iou[union == 0] = 1.0

                    # Align decoder score with IoU
                    score_loss = torch.abs(prd_scores[i:i + 1, 0] - iou).mean()

                    # Combined per-sample loss (weights unchanged)
                    sample_loss = 0.5 * seg_loss + 0.5 * dice_loss + 0.05 * score_loss
                    batch_loss_sum = sample_loss if batch_loss_sum is None else batch_loss_sum + sample_loss
                    batch_valid += 1

                    # Metrics (use probabilities for evaluation)
                    dice_sum += dice_coeff_from_probs(prd_mask_probs, gt_mask)
                    iou_sum += iou_from_probs(prd_mask_probs, gt_mask)

                    # Pixel accuracy computed on-GPU; only the scalar crosses to host.
                    with torch.no_grad():
                        train_correct += ((prd_mask_probs > 0.5).float() == gt_mask).sum().item()
                    train_total += gt_mask.numel()

                    del prd_mask_logits, prd_mask_probs, prd_mask_binary, gt_mask
                    del seg_loss, dice_loss, score_loss, iou

                if batch_valid == 0:
                    continue

                # Mean over valid samples: gradients are averaged over the batch.
                loss = batch_loss_sum / batch_valid

            # Backward -- one optimizer step per batch
            if use_amp and scaler is not None and autocast_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.sam2_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sam2_model.parameters(), max_norm=1.0)
                optimizer.step()

            # ================================================================
            # ONE-TIME GRADIENT-FLOW AUDIT (first backward of epoch 0)
            # Reports, per top-level module, how many parameters actually
            # received gradients -- measured, not assumed. In the batched
            # design the decoder consumes cached embeddings directly, so
            # encoder gradients here would indicate a configuration error.
            # ================================================================
            if epoch == 0 and not getattr(self, "_grad_audit_done", False):
                self._grad_audit_done = True
                grad_counts = {}
                total_with_grad = 0
                for name, p in self.sam2_model.named_parameters():
                    if p.grad is not None:
                        top = name.split(".")[0]
                        grad_counts[top] = grad_counts.get(top, 0) + p.numel()
                        total_with_grad += p.numel()
                print(f"[SAM2Trainer] Gradient-flow audit: {total_with_grad:,} parameters received gradients:")
                for top, n in sorted(grad_counts.items()):
                    print(f"[SAM2Trainer]     {top}: {n:,}")
                if any(k.startswith("image_encoder") for k in grad_counts):
                    print("[SAM2Trainer] ERROR: image encoder received gradients under cached-embedding "
                          "batched training. This is a configuration error; encoder updates cannot take "
                          "effect because the decoder consumes cached embeddings. Results are unreliable.")

            epoch_loss += batch_loss_sum.item()
            processed_count += batch_valid
            num_steps += 1

            # Periodically clear GPU cache to prevent VRAM fragmentation
            _since_cache_clear += batch_valid
            if _since_cache_clear >= 20 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                _since_cache_clear = 0

            # Explicitly free batch tensors to prevent VRAM accumulation
            del loss, batch_loss_sum, low_res_masks, prd_scores
            del point_coords, point_labels, sparse_embeddings, dense_embeddings
            del image_embed, high_res_features

            if not self.progress_bar_closed:  # Issue #8: direct attribute access
                progressBar.setValue(progressBar.getValue() + B)

        print(f"[SAM2Trainer] Epoch {epoch + 1}: {processed_count} samples in {num_steps} steps "
              f"(batch_size={batch_size}, {newly_encoded} newly encoded, "
              f"{processed_count - newly_encoded if processed_count >= newly_encoded else 0} pre-cached).")

        avg_epoch_loss = epoch_loss / processed_count if processed_count > 0 else 0.0
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        avg_dice = dice_sum / processed_count if processed_count > 0 else 0.0
        avg_iou = iou_sum / processed_count if processed_count > 0 else 0.0

        return avg_epoch_loss, train_accuracy, avg_dice, avg_iou

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _step_lr_scheduler(self, scheduler, optimizer, avg_val_loss, epoch_num):
        """
        Step ReduceLROnPlateau and report any LR change.

        torch's `verbose` argument is deprecated, so the change is detected and
        printed here instead. Without this the scheduler runs invisibly and
        users cannot tell it is active.
        """
        lr_before = optimizer.param_groups[0]['lr']

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        lr_after = optimizer.param_groups[0]['lr']
        self.lr_values.append(lr_after)

        if lr_after < lr_before:
            print(f"[SAM2Trainer] Epoch {epoch_num}: validation loss plateaued — "
                  f"learning rate reduced {lr_before:.3e} -> {lr_after:.3e}")
            if lr_after <= self.lr_scheduler_min_lr:
                print(f"[SAM2Trainer] Learning rate has reached min_lr "
                      f"({self.lr_scheduler_min_lr:g}); no further reductions.")

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _should_save_overlays(self, epoch_num, total_epochs):
        """
        Decide whether this validation epoch should write overlay PNGs.

        epoch_num:    1-based epoch number
        total_epochs: configured epoch count

        Note that 'last' cannot fire when the epoch loop exits early (early
        stopping, divergence, cancel). _ensure_overlays_written() is the
        backstop for those paths.
        """
        mode = getattr(self, 'validation_overlay_mode', 'last')

        if mode == 'every':
            return True

        if mode == 'interval':
            interval = max(1, int(getattr(self, 'validation_overlay_interval', 5)))
            return (epoch_num % interval == 0) or (epoch_num == total_epochs)

        # Default / 'last'
        return epoch_num == total_epochs

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _ensure_overlays_written(self, val_cache, predictor, progressBar, target_label):
        """
        Guarantee that validation_overlays/ is populated even when the epoch
        loop exited before the final epoch.

        Prior behavior gated overlay writing on (epoch + 1 == epochs), so any
        early exit — early stopping, divergence abort, user cancel — produced a
        model folder with no validation_overlays/ subfolder at all. This runs a
        single extra overlay-only validation pass in that case.
        """
        if self._overlays_written or not val_cache:
            return

        print("[SAM2Trainer] Training ended before the final epoch; "
              "running one validation pass to write validation_overlays/.")
        try:
            self._validate_one_epoch(
                val_cache=val_cache,
                predictor=predictor,
                progressBar=progressBar,
                target_label=target_label,
                save_overlays=True
            )
        except Exception as e:
            print(f"[SAM2Trainer] Final overlay pass failed: {e}")

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def _gt_prompts_for_image(self, image_file, target_label, true_mask=None,
                              negative_balance=3):
        """
        Per-image point prompts derived from ground truth -- the SAME protocol
        _train_one_epoch uses (positives = centroid of each target blob,
        negatives = centroids of every other category's blobs, balanced to
        negative_balance:1).

        Returns (coords [N,2] float32 numpy in pixel space, labels [N] int64
        numpy) or (None, None) if the image has no target blob.
        """
        target_id = next((c["id"] for c in self.categories if c["name"] == target_label), None)
        if target_id is None:
            return None, None

        if true_mask is None:
            result = self.dataset_util.load_true_mask(
                image_file, self.annotation_index, mode="binary", target_id=target_id)
            true_mask = result[0] if isinstance(result, tuple) else result
        if true_mask is None or true_mask.sum() == 0:
            return None, None
        if true_mask.ndim == 3:
            true_mask = true_mask[..., 0]

        positives = []
        n_lbl, lbls = cv2.connectedComponents(true_mask.astype(np.uint8))
        for lbl in range(1, n_lbl):
            ys, xs = np.nonzero(lbls == lbl)
            if xs.size:
                positives.append([float(xs.mean()), float(ys.mean())])
        if not positives:
            return None, None

        negatives = []
        for cat in self.categories:
            if cat["name"] == target_label:
                continue
            res = self.dataset_util.load_true_mask(
                image_file, self.annotation_index, mode="binary", target_id=cat["id"])
            neg = res[0] if isinstance(res, tuple) else res
            if neg is None or neg.sum() == 0:
                continue
            if neg.ndim == 3:
                neg = neg[..., 0]
            n_lbl, lbls = cv2.connectedComponents(neg.astype(np.uint8))
            for lbl in range(1, n_lbl):
                ys, xs = np.nonzero(lbls == lbl)
                if xs.size:
                    negatives.append([float(xs.mean()), float(ys.mean())])

        if len(negatives) > len(positives) * negative_balance:
            negatives = random.sample(negatives, len(positives) * negative_balance)

        coords = np.asarray(positives + negatives, dtype=np.float32)
        labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int64)
        return coords, labels

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

                # Encode once, reuse the embedding on later epochs (the image
                # encoder is frozen -- its output never changes).
                if "_feat" in cached:
                    predictor.reset_predictor()
                    predictor._features = {
                        "image_embed": cached["_feat"]["image_embed"].to(device, non_blocking=True),
                        "high_res_feats": [t.to(device, non_blocking=True)
                                           for t in cached["_feat"]["high_res_feats"]],
                    }
                    predictor._orig_hw = list(cached["_feat"]["orig_hw"])
                    predictor._is_image_set = True
                    predictor._is_batch = False
                else:
                    _t_enc = _time.perf_counter()
                    predictor.set_image(val_image)
                    cached["_feat"] = {
                        "image_embed": predictor._features["image_embed"].detach().cpu(),
                        "high_res_feats": [t.detach().cpu() for t in predictor._features["high_res_feats"]],
                        "orig_hw": list(predictor._orig_hw),
                    }
                    self._last_encode_seconds = getattr(self, "_last_encode_seconds", 0.0) + (_time.perf_counter() - _t_enc)

                # Build point prompts.
                #   "gt":    this image's own GT centroids + 3:1 negatives --
                #            oracle prompts; an upper bound, not a deployment number.
                #   "pooled" (default): sampled population centroids, identical to
                #            training (train_prompt_mode="pooled") and to
                #            sam2_inference_engine. This is the deployment number.
                h_img, w_img = val_image.shape[:2]
                val_prompt_mode = getattr(self, "val_prompt_mode",
                                          self.site_config.get("val_prompt_mode", "pooled"))
                if val_prompt_mode == "pooled":
                    c, l = sample_pooled_prompts(
                        target_id, self.category_centroids, w_img, h_img,
                        max_positives=int(self.site_config.get("max_positives", _DEFAULT_MAX_POSITIVES)),
                        negative_balance=int(self.site_config.get("negative_balance", _DEFAULT_NEGATIVE_BALANCE)),
                        rng=random.Random(42 + val_idx))
                    point_coords = None if c is None else np.asarray(c, dtype=np.float32)
                    point_labels = None if l is None else np.asarray(l, dtype=np.int64)
                else:
                    point_coords, point_labels = self._gt_prompts_for_image(
                        cached["file"], target_label, true_mask=val_true_mask)

                if point_coords is None:
                    print(f"  Validation skipped for {cached['file']} — "
                          f"no prompts for category '{target_label}' (ID {target_id}).")
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
                    # val_image is RGB (PIL). Work in BGR for OpenCV, and tint
                    # each pixel in proportion to its probability so background
                    # stays untouched (a uniform 0.4 JET layer darkens the frame).
                    img_bgr = cv2.cvtColor(val_image, cv2.COLOR_RGB2BGR)
                    prob_np = prob_upsampled.squeeze().cpu().numpy().astype(np.float32)
                    heat = cv2.applyColorMap((prob_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    alpha = (0.55 * prob_np)[..., None]
                    blended = (img_bgr.astype(np.float32) * (1.0 - alpha)
                               + heat.astype(np.float32) * alpha).astype(np.uint8)
                    overlay_dir = os.path.join(self.model_output_folder, "validation_overlays")
                    os.makedirs(overlay_dir, exist_ok=True)
                    out_path = os.path.join(
                        overlay_dir,
                        f"{self.formatted_time}_{self.site_name}_val_overlay_e{len(self.epoch_list)}_{val_idx}.png"
                    )
                    cv2.imwrite(out_path, blended)  # already BGR
                    self._overlays_written = True

                del logit_tensor, logit_upsampled, prob_upsampled, pred_binary, val_true_mask_tensor, iou_actual

                progressBar.setValue(progressBar.getValue() + 1)

        # Compute averages
        avg_val_loss = val_loss / n_items if n_items > 0 else 0.0
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        avg_val_dice = dice_sum / n_items if n_items > 0 else 0.0
        avg_val_iou = iou_sum / n_items if n_items > 0 else 0.0
        miou = compute_mean_iou(self.val_true_list, self.val_pred_list)

        # Calibration check: Dice/IoU at the F1-optimal threshold on the
        # sampled pixels. A large gap vs the 0.5-threshold numbers means the
        # model ranks pixels well but its logits are shifted -- a prompt /
        # protocol problem, not a capacity problem.
        self.val_best_thr, self.val_dice_best_thr, self.val_iou_best_thr = (
            self._threshold_optimal_metrics(self.val_true_list, self.val_score_list))
        print(f"[SAM2Trainer] Val @0.5: dice={avg_val_dice:.3f} iou={avg_val_iou:.3f} | "
              f"@thr={self.val_best_thr:.2f} (F1-opt): dice={self.val_dice_best_thr:.3f} "
              f"iou={self.val_iou_best_thr:.3f}")

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
    @staticmethod
    def _threshold_optimal_metrics(y_true, y_score):
        """Pixel-level Dice/IoU at the threshold maximising F1 (Dice == F1)."""
        if not y_true:
            return 0.5, 0.0, 0.0
        t = np.asarray(y_true, dtype=np.int8)
        p = np.asarray(y_score, dtype=np.float32)
        order = np.argsort(-p)
        t, p = t[order], p[order]
        tp = np.cumsum(t)
        fp = np.cumsum(1 - t)
        fn = t.sum() - tp
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        k = int(np.argmax(f1))
        dice = float(f1[k])
        iou = dice / (2 - dice) if dice < 1 else 1.0
        return float(p[k]), dice, float(iou)

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
            # Prompt protocol the decoder was trained with. Inference MUST
            # reproduce it (see sam2_inference_engine.load_sam2_model).
            "prompt_protocol": {
                "mode": getattr(self, "train_prompt_mode", self.site_config.get("train_prompt_mode", "pooled")),
                "max_positives": int(self.site_config.get("max_positives", _DEFAULT_MAX_POSITIVES)),
                "negative_balance": int(self.site_config.get("negative_balance", _DEFAULT_NEGATIVE_BALANCE)),
            },
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
            text_file.write(f"Best Checkpoints Kept: {self.max_best_checkpoints}\n")
            text_file.write(f"Early Stopping: {self.early_stopping}\n")
            text_file.write(f"Patience: {self.patience}\n")
            text_file.write("Early Stopping Monitor: validation loss (mode=min)\n")
            _sched_on = getattr(self, 'lr_scheduler_enabled', True)
            text_file.write(f"LR Scheduler: {'ReduceLROnPlateau' if _sched_on else 'None'}\n")
            if _sched_on:
                text_file.write(f"LR Scheduler Monitor: validation loss (mode=min)\n")
                text_file.write(f"LR Scheduler Factor: {getattr(self, 'lr_scheduler_factor', 0.5)}\n")
                text_file.write(f"LR Scheduler Patience: {getattr(self, 'lr_scheduler_patience', 3)}\n")
                text_file.write(f"LR Scheduler Min LR: {getattr(self, 'lr_scheduler_min_lr', 1e-7):g}\n")
                _lrs = getattr(self, 'lr_values', [])
                if _lrs:
                    text_file.write(f"Final Learning Rate: {_lrs[-1]:.3e}\n")
                    text_file.write(f"LR Reductions: {sum(1 for i in range(1, len(_lrs)) if _lrs[i] < _lrs[i-1])}\n")
            text_file.write(f"Validation Overlay Mode: {getattr(self, 'validation_overlay_mode', 'last')}\n")
            text_file.write(f"Train Prompt Mode: {self.site_config.get('train_prompt_mode', 'pooled')}\n")
            text_file.write(f"Val Prompt Mode: {self.site_config.get('val_prompt_mode', 'pooled')}\n")
            text_file.write(f"Max Positive Prompts: {self.site_config.get('max_positives', _DEFAULT_MAX_POSITIVES)}\n")
            text_file.write(f"Negative Balance: {self.site_config.get('negative_balance', _DEFAULT_NEGATIVE_BALANCE)}:1\n")
            text_file.write(f"Batch Size: {getattr(self, 'batch_size', 1)}\n")
            _amp = "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp16"
            text_file.write(f"AMP dtype: {_amp}\n")
            if getattr(self, "val_best_thr", None) is not None:
                text_file.write(f"Val F1-optimal threshold: {self.val_best_thr:.3f} "
                                f"(Dice {self.val_dice_best_thr:.4f}, IoU {self.val_iou_best_thr:.4f})\n")
            if getattr(self, "time_train_seconds", None) is not None:
                text_file.write(f"Time (encode/train/val): {self.time_encode_seconds:.0f}s / "
                                f"{self.time_train_seconds:.0f}s / {self.time_val_seconds:.0f}s\n")
            text_file.write(f"Device: {device}\n")
            # self.folders / self.annotation_files are never populated after
            # __init__; the real inputs live in all_folders / all_annotations.
            _folders = getattr(self, 'all_folders', None) or self.folders
            _annots = getattr(self, 'all_annotations', None) or self.annotation_files
            text_file.write(f"Folders: {_folders}\n")
            text_file.write(f"Annotations: {_annots}\n")

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
            lr_scheduler_enabled=getattr(self, "lr_scheduler_enabled", None),
            lr_reductions=sum(1 for i in range(1, len(getattr(self, "lr_values", [])))
                              if self.lr_values[i] < self.lr_values[i - 1]),
            config=self.site_config,
            val_best_threshold=getattr(self, "val_best_thr", None),
            val_dice_best_threshold=getattr(self, "val_dice_best_thr", None),
            time_breakdown={
                "encode": getattr(self, "time_encode_seconds", None),
                "train": getattr(self, "time_train_seconds", None),
                "validate": getattr(self, "time_val_seconds", None),
            },
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
                  
