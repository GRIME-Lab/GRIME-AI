#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

from GRIME_AI.ml_core.ML_Dependencies import *  # JES - Boy, do I have issues with this. :(

from torchvision.transforms import InterpolationMode
_ = InterpolationMode.BILINEAR  # Ensures inclusion during PyInstaller freeze

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from datetime import datetime

from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI.GRIME_AI_JSON_Editor import JsonEditor
from GRIME_AI.ml_core.segformer_trainer import SegFormerConfig, SegFormerTrainer
from GRIME_AI.ml_core.lora_wrapper import GeneralLoRAWrapper

# ----------------------------------------------------------------------------------------------------------------------
# WARNING AND ERROR LOGGING
# ----------------------------------------------------------------------------------------------------------------------
if True:
    import logging
    logging.getLogger("root").setLevel(logging.WARNING)
    logging.disable(logging.INFO)

# ----------------------------------------------------------------------------------------------------------------------
# GET DEVICE TO BE USED
# ----------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------------------------------------------------------------------------
# HYDRA (for SAM2)
# ----------------------------------------------------------------------------------------------------------------------
from omegaconf import DictConfig

DEBUG = False  # Set to True if you want print statements


# ============================================================================
# ============================================================================
# ===                             HELPER FUNCTIONS                       =====
# ============================================================================
# ============================================================================
def normalize_files(annotation_files):
    # If it's a list of lists, flatten one level
    if annotation_files and isinstance(annotation_files[0], list):
        return [item for sublist in annotation_files for item in sublist]
    # Otherwise assume it's already a flat list of strings
    return annotation_files


# ============================================================================
# ============================================================================
# ===                            class MLModelTraining                     ===
# ============================================================================
# ============================================================================
class MLModelTraining:

    def __init__(self, cfg: DictConfig = None, parent_widget=None):
        self.className = "MLModelTraining"

        self.cfg = cfg
        self.parent_widget = parent_widget

        # ALL FILES SAVED WITH BE TAGGED WITH THE DATE AND TIME THAT TRAINING STARTED.
        self.now = datetime.now()
        self.formatted_time = self.now.strftime('%Y%m%d_%H%M%S')

        self.image_shape_cache = {}

        # load site_config from saved JSON
        settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
        CONFIG_FILENAME = "site_config.json"
        site_configuration_file = os.path.normpath(os.path.join(settings_folder, CONFIG_FILENAME))
        print(site_configuration_file)

        self.site_config = JsonEditor().load_json_file(site_configuration_file)

        self.site_name = self.site_config['siteName']
        self.learning_rates = self.site_config['learningRates']
        self.optimizer_type = self.site_config['optimizer']
        self.loss_function = self.site_config['loss_function']
        self.weight_decay = self.site_config['weight_decay']
        self.num_epochs = self.site_config['number_of_epochs']
        self.save_model_frequency = self.site_config['save_model_frequency']
        self.early_stopping = self.site_config['early_stopping']
        self.patience = self.site_config['patience']
        self.device = self.site_config.get('device', str(device))

        self.folders = None
        self.annotation_files = None
        self.all_folders = []
        self.all_annotations = []
        self.categories = []
        self.model_output_folder = None

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def debug_print(self, msg):
        if DEBUG:
            print(msg)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def Model_Training_Dispatcher(self, cfg=None, mode="sam2"):
        """
        Unified training entry point.
        mode: "sam2" or "maskrcnn"
        - "sam2": run SAM2 training (Hydra + predictor + train_sam loop)
        - "maskrcnn": run Mask R-CNN training with graphs + checkpoints
        """

        now_start = datetime.now()
        print(f"Execution Started: {now_start.strftime('%y%m%d %H:%M:%S')}")

        # -------------------------------
        # CREATE OUTPUT FOLDER
        # -------------------------------
        try:
            self.model_output_folder = os.path.join(
                GRIME_AI_Save_Utils().get_models_folder(), mode,
                f"{self.formatted_time}_{self.site_name}"
            )
            os.makedirs(self.model_output_folder, exist_ok=True)
        except OSError as e:
            print(f"Error creating folders: {e}")


        ###JES TEST TEST TEST TEST TEST
        ###JES mode = "sam3"

        # --------------------------------------------------------------------
        #      SAM2   ---   SAM2   ---   SAM2   ---   SAM2   ---   SAM2
        # --------------------------------------------------------------------
        if mode.lower() == "sam2":
            from GRIME_AI.ml_core.sam2_trainer import SAM2Trainer
            # Pass self.site_config (loaded at dispatch time from the user's
            # Train-click write) directly into the trainer. This eliminates
            # any race window where another code path could rewrite
            # site_config.json between the Train click and the trainer's read.
            mySAM2_pipeline = SAM2Trainer(
                self.cfg,
                parent_widget=self.parent_widget,
                site_config=self.site_config,
            )
            mySAM2_pipeline.run_training_pipeline()

            # Get the best checkpoint path
            best_model_path = mySAM2_pipeline.get_best_checkpoint_path()
            if best_model_path:
                print(f"Best model saved at: {best_model_path}")
            else:
                print("No checkpoints were saved during training")

            # Return trainer so caller can access suggested_blob_radius_result
            return mySAM2_pipeline

        # --------------------------------------------------------------------
        #      SAM3   ---   SAM3   ---   SAM3   ---   SAM3   ---   SAM3
        # --------------------------------------------------------------------
        elif mode.lower() == "sam3":
            from GRIME_AI.ml_core.sam3_trainer import SAM3Trainer
            mySAM3_pipeline = SAM3Trainer(self.cfg)
            mySAM3_pipeline.run_training_pipeline()

        # --------------------------------------------------------------------
        #     SegFormer  ---  SegFormer  ---  SegFormer  ---  SegFormer
        # --------------------------------------------------------------------
        elif mode.lower() == "segformer":
            # Collect folders and annotations
            paths = self.site_config.get('Path', [])
            for path in paths:
                directory_path = path['directoryPaths']

                self.folders = directory_path.get('folders', [])
                self.all_folders.extend([self.folders])

                self.annotation_files = directory_path.get('annotations', [])
                self.all_annotations.extend([self.annotation_files])

            # BUILD CATEGORIES AND UPDATE CONFIG
            self.categories = self.build_unique_categories(self.all_annotations)
            cfg = SegFormerConfig(
                images_dir="",
                ann_path="",
                num_epochs=self.num_epochs,
                batch_size=self.site_config.get('batch_size', 8),
                lr=self.learning_rates[0] if self.learning_rates else 0.0003,
                weight_decay=self.weight_decay,
                early_stopping=self.early_stopping,
                patience=self.patience,
                categories=self.categories,
                target_category_name=self.site_config["train_model"]["TRAINING_CATEGORIES"][0]["label_name"],
                output_dir=self.model_output_folder,
                formatted_time=self.formatted_time,
                site_name=self.site_name,
            )

            print("Begin SegFormer Training...")
            self.run_segformer(self.all_folders, self.all_annotations, cfg,
                          use_lora=True,
                          lora_target_modules=["query", "key", "value"],
                          modules_to_save=["decode_head.classifier"])
            print("Completed SegFormer Training...")

        # -------------------------------
        # Mask R-CNN BRANCH
        # -------------------------------
        elif mode.lower() == "maskrcnn":
            from exp.trainers.maskrcnn_trainer import MaskRCNNTrainer
            from exp.datasets.coco_instance_seg import CocoInstanceSegDataset
            from torch.utils.data import DataLoader
            import torch
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

            # BUILD DATASETS/LOADERS FROM CFG
            train_dataset = CocoInstanceSegDataset(cfg.train_images, cfg.train_ann)
            val_dataset = CocoInstanceSegDataset(cfg.val_images, cfg.val_ann)

            train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                      shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
            val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                                    shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

            # BUILD MODEL
            model = maskrcnn_resnet50_fpn(weights="DEFAULT")
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.num_classes)
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, cfg.num_classes)

            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-4)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            trainer = MaskRCNNTrainer(model, train_loader, val_loader, optimizer, device)
            trainer.train(cfg.epochs)

        # --------------------------------------------------------------------
        #      YOLO   ---   YOLO   ---   YOLO   ---   YOLO   ---   YOLO
        # --------------------------------------------------------------------
        elif mode.lower() == "yolo":
            import importlib.util
            if importlib.util.find_spec("ultralytics") is None:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(
                    None,
                    "ultralytics Not Installed",
                    "The ultralytics package is required for YOLOv11-seg training "
                    "but is not installed in this environment.\n\n"
                    "Install it with:\n"
                    "  pip install ultralytics\n\n"
                    "YOLO training has been cancelled. All other GRIME AI features "
                    "remain available."
                )
                return

            from GRIME_AI.ml_core.yolo_trainer import YOLOTrainer
            myYOLO_pipeline = YOLOTrainer(self.cfg, parent_widget=self.parent_widget)
            myYOLO_pipeline.run_training_pipeline()
            return

        else:
            raise ValueError(f"Unknown training mode: {mode}")

        # FINAL TIMING
        now_end = datetime.now()
        print(f"Execution Ended: {now_end.strftime('%y%m%d %H:%M:%S')}")


    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def run_segformer(self, image_dirs, ann_paths, cfg: SegFormerConfig,
                      use_lora: bool = False,
                      lora_target_modules=None,
                      modules_to_save=None,
                      site_name_plain="segformer_plain",
                      site_name_lora="segformer_lora"):
        """
        Compose SegFormer training with optional LoRA wrapping.
        - Builds base SegFormer via SegFormerTrainer.
        - Optionally wraps with GeneralLoRAWrapper.
        - Trains using SegFormerTrainer with the correct optimizer.
        """
        trainer = SegFormerTrainer(cfg, parent_widget=self.parent_widget)
        
        # Calculate num_labels from categories
        # Background (class 0) + all annotated categories
        num_labels = len(cfg.categories) + 1  # +1 for background class
        print(f"Building SegFormer with {num_labels} classes (background + {len(cfg.categories)} categories)")

        num_labels = 2  ###JES - DIAGNOSTIC - BACK TO ORIGINAL 2 CLASS CODE
        base_model = trainer.build_model(num_labels=num_labels)

        if use_lora:
            lora = GeneralLoRAWrapper(
                r=cfg.__dict__.get("lora_r", 16),
                alpha=cfg.__dict__.get("lora_alpha", 32),
                dropout=cfg.__dict__.get("lora_dropout", 0.05),
                target_modules=lora_target_modules or ["query", "key", "value"],
                modules_to_save=modules_to_save or ["decode_head.classifier"],
                task_type="FEATURE_EXTRACTION"
            )
            model = lora.apply(base_model, device=cfg.device)
            optimizer = lora.configure_optimizer(lr=cfg.lr, weight_decay=cfg.weight_decay)
            trained = trainer.train(
                image_dirs, ann_paths,
                model=model, optimizer=optimizer,
                categories=cfg.categories, site_name=site_name_lora
            )
            # Optionally save adapters: lora.save_adapters(cfg.output_dir)
            return trained

        trained = trainer.train(
            image_dirs, ann_paths,
            model=base_model, optimizer=None,
            categories=cfg.categories, site_name=site_name_plain
        )

        return trained

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def build_unique_categories(self, annotation_files):
        merged = []
        id_to_name = {}
        name_to_id = {}

        annotation_files = normalize_files(annotation_files)

        for p in annotation_files:
            try:
                data = json.load(open(p, 'r'))
                cats = data.get('categories', [])
            except Exception as e:
                print(f"Failed loading '{p}': {e}")
                continue

            for cat in cats:
                cid = cat.get('id'); cname = cat.get('name')
                if cid is None or cname is None:
                    print(f"Warning: bad category entry in '{p}': {cat}")
                    continue

                # check ID↔name consistency
                if cid in id_to_name and id_to_name[cid] != cname:
                    print(f"⚠️ ID conflict: {cid} is '{id_to_name[cid]}' and '{cname}'")
                    continue
                id_to_name[cid] = cname

                if cname in name_to_id and name_to_id[cname] != cid:
                    print(f"⚠️ Name conflict: '{cname}' → {self.name_to_id[cname]} vs {cid}")
                    continue
                name_to_id[cname] = cid

                merged.append({"id": cid, "name": cname})

        # dedupe
        unique = {(c['id'], c['name']): c for c in merged}
        return list(unique.values())
