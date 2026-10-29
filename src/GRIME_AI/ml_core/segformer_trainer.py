#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, prevents GUI windows
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import SegformerForSemanticSegmentation

from appcore.ml_core.coco_segmentation_datasets import MultiCocoTargetDataset
from appcore.ml_core.lora_segmentation_losses import (BinaryDiceLoss, MultiClassDiceLoss,
                                                          FocalLoss, TverskyLoss)
from appcore.QProgressWheel import QProgressWheel
from appcore.ml_core.model_training_visualization import ModelTrainingVisualization
from appcore.dialogs.ML_image_processing.model_config_manager import ModelConfigManager
import torchvision.transforms.functional as TF
from appcore.ml_core.segformer_errors import SegFormerError
from appcore.ml_core.segformer_inference_engine import ARGMAX_EQUIVALENT_THRESHOLD

# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====      AUGMENTATION      =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
def apply_augmentation(image, mask, cfg):
    """
    Apply data augmentation to image and mask together.
    Applies same geometric transforms to both image and mask.
    
    Args:
        image: PIL Image or Tensor [C,H,W]
        mask: Tensor [H,W]
        cfg: SegFormerConfig with augmentation parameters
    """
    # Convert to PIL if needed
    if isinstance(image, torch.Tensor):
        image = TF.to_pil_image(image)
    
    # Random horizontal flip
    if random.random() < cfg.aug_horizontal_flip:
        image = TF.hflip(image)
        mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
    
    # Random vertical flip
    if random.random() < cfg.aug_vertical_flip:
        image = TF.vflip(image)
        mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
    
    # Random rotation. The mask MUST rotate with nearest-neighbor interpolation
    # or bilinear blending corrupts the integer class labels at boundaries.
    if cfg.aug_rotation > 0:
        angle = random.uniform(-cfg.aug_rotation, cfg.aug_rotation)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(
            mask.unsqueeze(0), angle,
            interpolation=TF.InterpolationMode.NEAREST
        ).squeeze(0)
    
    # Color jitter (image only, not mask)
    if cfg.aug_brightness > 0 or cfg.aug_contrast > 0:
        brightness_factor = 1.0 + random.uniform(-cfg.aug_brightness, cfg.aug_brightness)
        contrast_factor = 1.0 + random.uniform(-cfg.aug_contrast, cfg.aug_contrast)
        image = TF.adjust_brightness(image, brightness_factor)
        image = TF.adjust_contrast(image, contrast_factor)
    
    # Convert back to tensor
    image = TF.to_tensor(image)
    
    return image, mask

# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====      HELPER FUNCTIONS      =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
GLOBAL_SEED = 42

def _worker_init_fn(worker_id):
    np.random.seed(GLOBAL_SEED + worker_id)
    random.seed(GLOBAL_SEED + worker_id)
    torch.manual_seed(GLOBAL_SEED + worker_id)

# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====   class SegFormerConfig    =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
@dataclass
class SegFormerConfig:
    images_dir: str = ""
    ann_path: str = ""
    categories: Optional[List[str]] = field(default_factory=list)
    target_category_name: str = ""
    image_size: int = 512
    backbone_size: str = "b0"   # one of b0..b5; selects segformer-{size}-finetuned-cityscapes-1024-1024
    loss_function: str = "ce_dice"   # ce_dice | ce_focal | ce_tversky | focal | tversky | dice
    tversky_alpha: float = 0.5       # FP penalty (Tversky/Focal-Tversky)
    tversky_beta: float = 0.5        # FN penalty; beta>alpha favors recall
    focal_gamma: float = 2.0
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    val_every: int = 1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs_segformer"
    amp: bool = True
    grad_clip_norm: float = 1.0

    # Early stopping
    # Monitors validation loss (mode=min), matching the SAM2 trainer so both
    # backends stop on the same criterion.
    early_stopping: bool = False
    patience: int = 10

    # Learning rate scheduler (ReduceLROnPlateau), monitoring validation loss.
    lr_scheduler_enabled: bool = field(
        default_factory=lambda: ModelConfigManager.get_default("lr_scheduler_enabled"))
    lr_scheduler_factor: float = field(
        default_factory=lambda: ModelConfigManager.get_default("lr_scheduler_factor"))
    lr_scheduler_patience: int = field(
        default_factory=lambda: ModelConfigManager.get_default("lr_scheduler_patience"))
    lr_scheduler_min_lr: float = field(
        default_factory=lambda: ModelConfigManager.get_default("lr_scheduler_min_lr"))

    # Checkpoint management
    max_best_checkpoints: int = field(
        default_factory=lambda: ModelConfigManager.get_default("max_best_checkpoints"))

    # Validation overlays
    save_val_overlays: bool = True
    num_val_overlays: int = field(
        default_factory=lambda: ModelConfigManager.get_default("validation_overlay_samples"))
    validation_overlay_mode: str = field(
        default_factory=lambda: ModelConfigManager.get_default("validation_overlay_mode"))
    validation_overlay_interval: int = field(
        default_factory=lambda: ModelConfigManager.get_default("validation_overlay_interval"))

    # Inference threshold (should match inference engine)
    # Decision rule is argmax over classes (multi-class standard) everywhere:
    # training accuracy, validation metrics, overlays, and the inference
    # engine. No probability threshold -- any threshold here would diverge
    # from what the deployed engine does.
    
    # Multi-class support
    target_class_id: int = 1  # Which class to evaluate/visualize (for multi-class datasets)
    
    # Data augmentation (for training only)
    use_augmentation: bool = True
    aug_horizontal_flip: float = 0.5  # Probability of horizontal flip
    aug_vertical_flip: float = 0.5    # Probability of vertical flip
    aug_rotation: float = 15.0        # Max rotation degrees (+/-)
    aug_brightness: float = 0.2       # Brightness adjustment range
    aug_contrast: float = 0.2         # Contrast adjustment range

    # Timestamp — passed in from ml_model_training so graph filenames match output folder
    formatted_time: str = ""

    # Pixel sampling cap for ROC/PR/confusion matrix (matches SAM2)
    max_pixel_samples: int = 10000

    # Resolution of the validation score histograms used to find the
    # F1-optimal decision threshold (threshold step = 1 / threshold_bins).
    threshold_bins: int = 1000

    # Site name stored on cfg for convenience (also passed via train())
    site_name: str = ""


# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====   class SegFormerTrainer   =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class SegFormerTrainer:
    """
    Pure SegFormer trainer: builds model, data loaders, runs training/eval, saves checkpoints/curves.
    No LoRA logic inside. If you want LoRA, wrap the model externally before calling train().
    """
    # Background (class 0) + the selected target category (class 1).
    BINARY_NUM_LABELS = 2

    def __init__(self, cfg: SegFormerConfig, parent_widget=None):
        self.cfg = cfg
        self.parent_widget = parent_widget
        self.best_iou = 0.0
        self.progressBar = None
        self._last_checkpoint_path = None
        self._progress_total = 0

        # Early stopping tracking
        self.patience_counter = 0
        self.best_val_loss = float("inf")
        # Set True once overlay PNGs have been written for this run.
        self._overlays_written = False

        # Checkpoint management
        self.best_checkpoints = []  # List of (val_loss, epoch, filepath) tuples

        # Metric accumulators — mirrors SAM2 trainer structure
        self.loss_values: List[float] = []           # train loss per epoch
        self.val_loss_values: List[float] = []       # val loss per val epoch
        self.train_accuracy_values: List[float] = [] # train pixel accuracy per epoch
        self.val_accuracy_values: List[float] = []   # val pixel accuracy per val epoch
        self.val_iou_values: List[float] = []        # val IoU per val epoch
        self.miou_values: List[float] = []           # val mIoU per val epoch (same as IoU for single-class)
        self.val_dice_values: List[float] = []       # val Dice per val epoch
        self.val_true_list: List[int] = []           # sampled pixel true labels (for ROC/PR/CM)
        self.val_pred_list: List[int] = []           # sampled pixel predictions
        self.val_score_list: List[float] = []        # sampled pixel scores (target class prob)
        self.epoch_list: List[int] = []              # val epoch numbers

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def set_seed(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _build_loss_spec(self, name, ce_loss, dice_loss, focal, tversky):
        """Map the loss_function name to the components to combine. Returns a
        list of (kind, fn) pairs summed at compute time. 'kind' tells
        _compute_loss which input form each fn expects."""
        table = {
            "ce_dice":    [("ce", ce_loss), ("dice", dice_loss)],
            "ce_focal":   [("ce", ce_loss), ("focal", focal)],
            "ce_tversky": [("ce", ce_loss), ("tversky", tversky)],
            "focal":      [("focal", focal)],
            "tversky":    [("tversky", tversky)],
            "dice":       [("dice", dice_loss)],
        }
        # Normalize free-form UI text ("Cross Entropy + Dice", "CE+Focal", etc.)
        # to canonical keys so the selector's display strings resolve regardless
        # of exact wording.
        key = name.lower().replace(" ", "").replace("+", "_").replace("-", "_")
        key = key.replace("crossentropy", "ce").replace("cross_entropy", "ce")
        aliases = {
            "ce_dice": "ce_dice", "cedice": "ce_dice",
            "ce_focal": "ce_focal", "cefocal": "ce_focal",
            "ce_tversky": "ce_tversky", "cetversky": "ce_tversky",
            "focal": "focal", "tversky": "tversky", "dice": "dice",
        }
        key = aliases.get(key, key)
        if key not in table:
            print(f"[loss] unknown loss_function {name!r} -> {key!r}; falling back to ce_dice")
            key = "ce_dice"
        return table[key]

    # ------------------------------------------------------------------------------------------------------------------
    def _compute_loss(self, logits, masks, num_labels):
        """Combine the selected loss components. Centralizes the per-loss input
        handling (logits vs probs, binary vs multi-class) so train and eval use
        an identical loss."""
        probs = torch.softmax(logits, dim=1)
        total = 0.0
        for kind, fn in self._loss_spec:
            if kind == "ce":
                total = total + fn(logits, masks)
            elif kind == "focal":
                total = total + fn(logits, masks)          # FocalLoss takes logits
            elif kind == "tversky":
                total = total + fn(logits, masks)          # TverskyLoss takes logits
            elif kind == "dice":
                if num_labels == 2:
                    probs_dice = probs[:, 1:2]
                    masks_bin = (masks == 1).float().unsqueeze(1)
                    total = total + fn(probs_dice, masks_bin)
                else:
                    total = total + fn(probs, masks)
        return total

    def build_model(self, num_labels: int) -> nn.Module:
        """
        Build base SegFormer (no LoRA). If you want LoRA, wrap the returned model externally.
        """
        size = str(getattr(self.cfg, "backbone_size", "b0")).lower()
        if size not in {"b0", "b1", "b2", "b3", "b4", "b5"}:
            raise SegFormerError(f"The SegFormer backbone size must be b0 to b5; the setting is {size!r}.")
        pretrained = f"nvidia/segformer-{size}-finetuned-cityscapes-1024-1024"
        try:
            model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            raise SegFormerError(
                f"The SegFormer {size} base model ({pretrained}) could not be loaded. It is not in the "
                f"local model cache and could not be downloaded.\n\n{e}") from e
        model.config.num_labels = num_labels
        model.config.id2label = {i: f"class_{i}" for i in range(num_labels)}
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}
        model.decode_head.classifier = nn.Conv2d(
            model.decode_head.classifier.in_channels, num_labels, kernel_size=1
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return model.to(device).train()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        inter = ((pred == 1) & (target == 1)).sum().item()
        union = ((pred == 1) | (target == 1)).sum().item()
        return (inter / union) if union > 0 else 1.0

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        tp = ((pred == 1) & (target == 1)).sum().item()
        fp = ((pred == 1) & (target == 0)).sum().item()
        fn = ((pred == 0) & (target == 1)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, model, val_loader, ce_loss, dice_loss, num_labels):
        """
        Run one validation epoch.
        Returns (avg_val_loss, val_accuracy, mean_iou, avg_dice, avg_iou, metrics_dict),
        or None if the user cancelled.

        The decision rule is the one the inference engine uses: target-class
        probability >= threshold. The threshold is the F1-optimal value for this
        epoch's validation pixels, found from per-image score histograms, and
        every metric below is computed at that threshold. metrics_dict["threshold"]
        carries it so the checkpoint saved for this epoch stores the same value.

        Pixel samples for ROC/PR/confusion matrix are for this epoch only.
        """
        model.eval()
        bins = int(self.cfg.threshold_bins)
        tid = self.cfg.target_class_id

        val_loss = 0.0
        n_batches = 0
        per_image = []          # (positive-pixel histogram, negative-pixel histogram) per image
        true_samples = []
        score_samples = []

        for imgs, masks in val_loader:
            if self._cancel_requested():
                return None

            imgs = imgs.to(self.cfg.device, non_blocking=True)
            masks = masks.to(self.cfg.device, non_blocking=True)
            masks_long = masks.round().long()
            masks_long[(masks_long < 0) | (masks_long > num_labels - 1)] = 255

            outputs = model(pixel_values=imgs)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )

            # Val loss uses the selected loss (same as training)
            val_loss += float(self._compute_loss(logits, masks_long, num_labels).detach().cpu())
            n_batches += 1

            probs = torch.softmax(logits, dim=1)

            for b in range(probs.size(0)):
                score = probs[b, tid].float().cpu().numpy().ravel()
                truth = (masks_long[b] == tid).cpu().numpy().ravel()

                bin_idx = np.minimum((score * bins).astype(np.int64), bins - 1)
                pos_hist = np.bincount(bin_idx[truth], minlength=bins)
                neg_hist = np.bincount(bin_idx[~truth], minlength=bins)
                per_image.append((pos_hist, neg_hist))

                # Pixel sampling for ROC/PR/confusion matrix
                total_pixels = len(score)
                max_samples = self.cfg.max_pixel_samples
                if total_pixels > max_samples:
                    idx = np.random.choice(total_pixels, max_samples, replace=False)
                    true_samples.append(truth[idx].astype(int))
                    score_samples.append(score[idx])
                else:
                    true_samples.append(truth.astype(int))
                    score_samples.append(score)

            self._tick_progress()

        if not per_image:
            raise SegFormerError("The validation set produced no images to evaluate.")

        # F1-optimal threshold over all validation pixels. Predicting positive at
        # threshold k/bins selects bins k..end, so suffix sums give TP and FP.
        pos_tot = np.sum([p for p, _ in per_image], axis=0)
        neg_tot = np.sum([n for _, n in per_image], axis=0)
        tp_k = np.cumsum(pos_tot[::-1])[::-1]
        fp_k = np.cumsum(neg_tot[::-1])[::-1]
        fn_k = pos_tot.sum() - tp_k
        f1_k = 2 * tp_k / np.maximum(2 * tp_k + fp_k + fn_k, 1)
        k = int(np.argmax(f1_k))
        threshold = k / bins

        # Per-image metrics at that threshold
        ious, precisions, recalls, f1s, dices = [], [], [], [], []
        val_correct = 0
        val_total = 0
        for pos_hist, neg_hist in per_image:
            tp = float(pos_hist[k:].sum())
            fp = float(neg_hist[k:].sum())
            fn = float(pos_hist.sum()) - tp
            tn = float(neg_hist.sum()) - fp
            union = tp + fp + fn
            ious.append(tp / union if union > 0 else 1.0)
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(2 * prec * rec / (prec + rec + 1e-8))
            dices.append((2 * tp) / ((tp + fp) + (tp + fn) + 1e-8))
            val_correct += tp + tn
            val_total += tp + fp + fn + tn

        true_arr = np.concatenate(true_samples)
        score_arr = np.concatenate(score_samples)
        pred_arr = (score_arr >= threshold).astype(int)
        self.val_true_list = true_arr.tolist()
        self.val_score_list = score_arr.tolist()
        self.val_pred_list = pred_arr.tolist()

        avg_val_loss = val_loss / n_batches if n_batches > 0 else 0.0
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        mean_iou = float(np.mean(ious))
        avg_dice = float(np.mean(dices))
        avg_iou = mean_iou

        metrics_dict = {
            "mean_iou": mean_iou,
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "f1": float(np.mean(f1s)),
            "threshold": float(threshold),
        }

        return avg_val_loss, val_accuracy, mean_iou, avg_dice, avg_iou, metrics_dict

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _should_save_overlays(self, epoch_num, total_epochs):
        """Overlay write policy. Mirrors the SAM2 trainer."""
        mode = str(getattr(self.cfg, "validation_overlay_mode", "last") or "last").lower()
        if mode == "every":
            return True
        if mode == "interval":
            interval = max(1, int(getattr(self.cfg, "validation_overlay_interval", 5)))
            return (epoch_num % interval == 0) or (epoch_num == total_epochs)
        return epoch_num == total_epochs

    def save_validation_overlay(self, epoch, images, pred_masks, true_masks, output_dir):
        """Save validation overlays showing prediction vs ground truth (SAM2 style)"""
        overlay_dir = os.path.join(output_dir, "validation_overlays")
        os.makedirs(overlay_dir, exist_ok=True)

        num_samples = min(self.cfg.num_val_overlays, len(images))

        for i in range(num_samples):
            img = images[i].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            pred = pred_masks[i].cpu().numpy()
            true = true_masks[i].cpu().numpy()

            # Denormalize image (assuming ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            # Create overlay with color coding:
            # Green = True Positive (correct prediction)
            # Red = False Positive (predicted but not GT)  
            # Cyan/Blue = False Negative (GT but not predicted)
            overlay = img.copy()

            # True Positives (both pred and GT) - Green/Yellow
            true_positive = (pred == 1) & (true == 1)
            overlay[true_positive] = [0, 1, 0]  # Green

            # False Positives (pred but not GT) - Red
            false_positive = (pred == 1) & (true == 0)
            overlay[false_positive] = [1, 0, 0]  # Red
            
            # False Negatives (GT but not pred) - Cyan
            false_negative = (pred == 0) & (true == 1)
            overlay[false_negative] = [0, 1, 1]  # Cyan
            
            # Blend with original image
            blended = (0.6 * img + 0.4 * overlay).astype(np.float32)
            
            # Save as single image
            plt.figure(figsize=(10, 10))
            plt.imshow(blended)
            plt.title(f"Validation Overlay - Epoch {epoch} (Green=TP, Red=FP, Cyan=FN)", fontsize=12)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(overlay_dir, f"ep{epoch:03d}_sample{i:02d}.png"), 
                       bbox_inches='tight', pad_inches=0.1)
            plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _model_label(self) -> str:
        return "SegFormer-LoRA" if getattr(self, "_use_lora", False) else "SegFormer"

    def _plot_training_graphs(self, site_name: str, lr: float):
        """
        Generate and save all training/validation plots using ModelTrainingVisualization.
        Mirrors SAM2's _plot_training_graphs exactly — same 9 plots, same viz class.
        """
        progressBar = QProgressWheel(
            title="Generating graphs...", total=9,
            on_close=lambda: setattr(self, "progress_bar_closed", True),
            parent=self.parent_widget
        )

        viz = ModelTrainingVisualization(
            self.cfg.output_dir, self.cfg.formatted_time, self.cfg.categories
        )

        train_epochs = list(range(1, len(self.loss_values) + 1))
        val_epochs = self.epoch_list

        # 1) Loss curves
        viz.plot_loss_curves(
            train_epochs=train_epochs,
            train_loss=self.loss_values,
            val_epochs=val_epochs,
            val_loss=self.val_loss_values,
            site_name=site_name,
            lr=lr,
            model_name=self._model_label(),
            epochs=self.cfg.num_epochs,
            batch_size=self.cfg.batch_size,
            num_train_images=getattr(self, "num_train_images", None),
            num_val_images=getattr(self, "num_val_images", None),
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 2) Accuracy curves
        viz.plot_accuracy(
            train_epochs=train_epochs,
            train_acc=self.train_accuracy_values,
            val_epochs=val_epochs,
            val_acc=self.val_accuracy_values,
            site_name=site_name,
            lr=lr,
            model_name=self._model_label(),
            epochs=self.cfg.num_epochs,
            batch_size=self.cfg.batch_size,
            num_train_images=getattr(self, "num_train_images", None),
            num_val_images=getattr(self, "num_val_images", None),
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 3) Confusion matrix
        viz.plot_confusion_matrix(
            y_true=self.val_true_list,
            y_pred=self.val_pred_list,
            site_name=site_name,
            lr=lr,
            normalize=True,
            file_prefix="Normalized",
            target_label=self.cfg.target_category_name
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 4) ROC curve + AUC
        viz.plot_roc_curve(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=site_name,
            lr=lr,
            file_prefix=f"{self.cfg.formatted_time}_{site_name}_lr{lr:.5f}"
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 5) Precision-Recall
        viz.plot_precision_recall(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=site_name,
            lr=lr,
            file_prefix=f"{self.cfg.formatted_time}_{site_name}_lr{lr:.5f}"
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 6) F1 vs Threshold
        viz.plot_f1_score(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=site_name,
            lr=lr,
            file_prefix=f"{self.cfg.formatted_time}_{site_name}_lr{lr:.5f}"
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 7) Mean IoU curve
        viz.plot_miou_curve(
            epochs=self.epoch_list,
            miou_values=self.miou_values,
            site_name=site_name,
            lr=lr,
            model_name=self._model_label(),
            file_prefix=f"{self.cfg.formatted_time}_{site_name}_lr{lr:.5f}",
            num_epochs=self.cfg.num_epochs,
            batch_size=self.cfg.batch_size,
            num_train_images=getattr(self, "num_train_images", None),
            num_val_images=getattr(self, "num_val_images", None),
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 8) Dice curve
        viz.plot_dice_curve(
            epochs=self.epoch_list,
            dice_values=self.val_dice_values,
            site_name=site_name,
            lr=lr,
            model_name=self._model_label(),
            file_prefix=f"{self.cfg.formatted_time}_{site_name}_lr{lr:.5f}",
            num_epochs=self.cfg.num_epochs,
            batch_size=self.cfg.batch_size,
            num_train_images=getattr(self, "num_train_images", None),
            num_val_images=getattr(self, "num_val_images", None),
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 9) IoU curve
        viz.plot_iou_curve(
            epochs=self.epoch_list,
            iou_values=self.val_iou_values,
            site_name=site_name,
            lr=lr,
            model_name=self._model_label(),
            file_prefix=f"{self.cfg.formatted_time}_{site_name}_lr{lr:.5f}",
            num_epochs=self.cfg.num_epochs,
            batch_size=self.cfg.batch_size,
            num_train_images=getattr(self, "num_train_images", None),
            num_val_images=getattr(self, "num_val_images", None),
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        progressBar.close()

        # ── PDF Diagnostic Report ──────────────────────────────────────────
        acc_png  = os.path.join(viz.graphs_folder, f"{self.cfg.formatted_time}_{site_name}_AccuracyCurves_lr{lr:.5f}.png")
        loss_png = os.path.join(viz.graphs_folder, f"{self.cfg.formatted_time}_{site_name}_LossCurves_lr{lr:.5f}.png")
        # Best epoch = lowest validation loss, the same criterion that ranks the checkpoints.
        best_idx = (int(np.argmin(self.val_loss_values)) if self.val_loss_values else None)
        viz.save_training_report(
            train_acc=self.train_accuracy_values,
            val_acc=self.val_accuracy_values,
            train_loss=self.loss_values,
            val_loss=self.val_loss_values,
            site_name=site_name,
            lr=lr,
            miou_values=self.miou_values,
            dice_values=self.val_dice_values,
            graph_paths=[acc_png, loss_png],
            model_type=self._model_label(),
            num_train_images=getattr(self, "num_train_images", None),
            num_val_images=getattr(self, "num_val_images", None),
            categories=self.cfg.categories,
            best_epoch=self.epoch_list[best_idx] if best_idx is not None else None,
            best_val_acc=self.val_accuracy_values[best_idx] if best_idx is not None else None,
            early_stopped=getattr(self, "early_stopped", False),
            early_stop_epoch=getattr(self, "early_stop_epoch", None),
            training_time_seconds=getattr(self, "training_time_seconds", None),
            lr_scheduler_enabled=bool(self.cfg.lr_scheduler_enabled),
            lr_reductions=sum(1 for i in range(1, len(getattr(self, "lr_values", [])))
                              if self.lr_values[i] < self.lr_values[i - 1]),
            # Config snapshot in site_config key names so recommendations are
            # actionable in the GUI (same keys as SAM2).
            config={
                "learningRates": [lr],
                "number_of_epochs": int(self.cfg.num_epochs),
                "weight_decay": float(self.cfg.weight_decay),
                "early_stopping": bool(self.cfg.early_stopping),
                "patience": int(self.cfg.patience),
                "lr_scheduler_enabled": bool(self.cfg.lr_scheduler_enabled),
                "lr_scheduler_patience": int(self.cfg.lr_scheduler_patience),
                "val_split": getattr(self.cfg, "val_split", None),
            },
            val_best_threshold=self._f1_optimal_threshold(),
            # Every validation metric is computed at the F1-optimal threshold.
            val_dice_best_threshold=self.val_dice_values[best_idx] if best_idx is not None else None,
            check_threshold_calibration=False,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _f1_optimal_threshold(self):
        """F1-optimal decision threshold of the best (lowest val loss) epoch, saved in that checkpoint."""
        return getattr(self, "_best_threshold", None)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_checkpoint(self, model, optimizer, scaler, categories, site_name,
                        learnrate, epochs, output_dir,
                        suffix=None, val_loss=None, val_accuracy=None, miou=None, target_category_name=None,
                        val_iou=None, epoch_num=None, val_best_threshold=None):
        """Save checkpoint with top-N management based on validation loss (lower is better)."""
        timestamp = str(np.datetime64('now', 's')).replace('-', '').replace(':', '').replace('T', '_')

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "categories": categories,
            "creation_UTC": timestamp,
            "site_name": site_name,
            "learning_rate": learnrate,
            "epochs": epochs,
            "num_classes": getattr(model.config, "num_labels", None),
            "backbone_size": str(getattr(self.cfg, "backbone_size", "b0")).lower(),
            "lora_config": getattr(self, "_lora_config", None),
            "use_lora": bool(getattr(self, "_use_lora", False)),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "miou": miou,
            "target_category_name": target_category_name,
            "base_model": "segformer",
            "val_best_threshold": val_best_threshold,
        }

        # If this is a final save or no val_iou provided, use simple naming
        if val_iou is None or suffix == "final":
            if suffix == "final":
                torch_filename = f"{timestamp}_{site_name}_{suffix}_lr{learnrate}_epoch{epochs}.torch"
            else:
                torch_filename = f"{timestamp}_{site_name}_epoch{epochs}_lr{learnrate}.torch"
            save_path = os.path.join(output_dir, torch_filename)
            torch.save(ckpt, save_path)
            print(f"Model checkpoint saved to {save_path}")
            self._last_checkpoint_path = save_path
            return save_path

        # Top-N checkpoint management based on validation loss (lower is better),
        # matching the SAM2 trainer.
        temp_filename = f"temp_{timestamp}_{site_name}_{target_category_name}_ep{epoch_num:03d}_lr{learnrate}.torch"
        temp_path = os.path.join(output_dir, temp_filename)
        torch.save(ckpt, temp_path)

        # Add to tracking list: (val_loss, epoch the weights are from, path)
        self.best_checkpoints.append((val_loss, epoch_num, temp_path))
        self.best_checkpoints.sort(key=lambda x: x[0])  # Sort by val loss ascending

        # Remove worst checkpoints if exceeding max
        while len(self.best_checkpoints) > self.cfg.max_best_checkpoints:
            _, _, worst_path = self.best_checkpoints.pop()
            if os.path.exists(worst_path):
                try:
                    os.remove(worst_path)
                    print(f"Removed lower-ranked checkpoint: {os.path.basename(worst_path)}")
                except Exception as e:
                    print(f"Failed to remove {worst_path}: {e}")

        # Rename all checkpoints with their current rank
        # Two passes so a rename never lands on a file that is still waiting to be renamed.
        staged = []
        for rank, (loss, ep, old_path) in enumerate(self.best_checkpoints, 1):
            rank_suffix = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
            new_filename = f"best_{rank_suffix}_{target_category_name}_epoch{ep:03d}_valloss{loss:.4f}_lr{learnrate}.torch"
            new_path = os.path.join(output_dir, new_filename)
            stage_path = old_path
            if old_path != new_path and os.path.exists(old_path):
                stage_path = old_path + ".ranking"
                os.replace(old_path, stage_path)
            staged.append((loss, ep, stage_path, new_path, rank))
        self.best_checkpoints = []
        for loss, ep, stage_path, new_path, rank in staged:
            if stage_path != new_path and os.path.exists(stage_path):
                os.replace(stage_path, new_path)
                print(f"Ranked #{rank}: {os.path.basename(new_path)}")
            self.best_checkpoints.append((loss, ep, new_path))

        self._last_checkpoint_path = self.best_checkpoints[0][2] if self.best_checkpoints else temp_path
        return self._last_checkpoint_path

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _init_progress(self, train_loader, val_loader):
        self._progress_total = self.cfg.num_epochs * (len(train_loader) + len(val_loader))
        self.progressBar = QProgressWheel(
            title="SegFormer Training in-progress...",
            total=self._progress_total,
            on_close=lambda: setattr(self.cfg, "progress_bar_closed", True),
            parent=self.parent_widget
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _tick_progress(self, inc: int = 1):
        if self.progressBar and not getattr(self.cfg, "progress_bar_closed", False):
            self.progressBar.setValue(self.progressBar.getValue() + inc)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _close_progress(self):
        if self.progressBar and not getattr(self.cfg, "progress_bar_closed", False):
            self.progressBar.close()
        self.progressBar = None

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _cancel_requested(self) -> bool:
        return bool(getattr(self.cfg, "progress_bar_closed", False))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _target_predictions(self, logits, threshold):
        """Target-class mask by the deployed rule: probability >= threshold."""
        probs = torch.softmax(logits.float(), dim=1)
        return (probs[:, self.cfg.target_class_id] >= threshold).long()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _preflight(self):
        """Checks that must pass before any data is loaded."""
        if not torch.cuda.is_available():
            raise SegFormerError("SegFormer training requires an NVIDIA GPU with CUDA, "
                                 "and none is available on this computer.")
        if not self.cfg.output_dir:
            raise SegFormerError("The model output folder could not be created.")
        try:
            self.ensure_dir(self.cfg.output_dir)
        except OSError as e:
            raise SegFormerError(f"The model output folder could not be created:\n{self.cfg.output_dir}\n\n{e}") from e
        if not self.cfg.target_category_name:
            raise SegFormerError("No training category is selected.")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def train(self, image_dirs, ann_paths, model: Optional[nn.Module] = None,
              optimizer: Optional[torch.optim.Optimizer] = None,
              categories: Optional[List[str]] = None,
              site_name: str = "segformer",
              lora_config: Optional[dict] = None):
        """
        Train loop that accepts an externally provided model and optimizer.
        - If model is None, builds a plain binary SegFormer.
        - If optimizer is None, uses AdamW on all model params (no LoRA assumption).

        Raises SegFormerError when training cannot run or fails. The training
        graphs and report are produced only when at least one validation epoch
        completed (a finished run, early stop, or user cancel).
        """
        # LoRA config is applied in the caller; stored here so it is written into
        # every checkpoint and read back verbatim at inference.
        self._lora_config = lora_config
        self._use_lora = lora_config is not None
        self.cfg.progress_bar_closed = False
        self.cancelled = False
        self.early_stopped = False
        self.early_stop_epoch = None

        self._preflight()
        self.set_seed()

        import time as _time
        _training_start = _time.perf_counter()
        self.training_time_seconds = None

        try:
            train_ds = MultiCocoTargetDataset(image_dirs, ann_paths, self.cfg.target_category_name,
                                              self.cfg.image_size, split="train", cfg=self.cfg)
            val_ds = MultiCocoTargetDataset(image_dirs, ann_paths, self.cfg.target_category_name,
                                            self.cfg.image_size, split="val")
        except SegFormerError:
            raise
        except Exception as e:
            raise SegFormerError(f"The training images and annotations could not be loaded.\n\n"
                                 f"{type(e).__name__}: {e}") from e

        self.num_train_images = len(train_ds)
        self.num_val_images = len(val_ds)
        if self.num_train_images == 0 or self.num_val_images == 0:
            raise SegFormerError(
                f"The training set has {self.num_train_images} image(s) and the validation set has "
                f"{self.num_val_images} image(s) for category '{self.cfg.target_category_name}'. "
                f"Both sets need at least one image.")

        g = torch.Generator().manual_seed(self.cfg.seed)
        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True,
            generator=g, worker_init_fn=_worker_init_fn,
            persistent_workers=(self.cfg.num_workers > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
            worker_init_fn=_worker_init_fn, persistent_workers=(self.cfg.num_workers > 0),
        )

        # The model's own class count is used everywhere (labels, losses, checkpoint).
        model = model or self.build_model(num_labels=self.BINARY_NUM_LABELS)
        num_labels = int(getattr(model.config, "num_labels", self.BINARY_NUM_LABELS))
        if num_labels <= self.cfg.target_class_id:
            raise SegFormerError(f"The model has {num_labels} class(es), so it has no target class "
                                 f"(class index {self.cfg.target_class_id}).")

        optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )

        # LR scheduler: reduce on validation-loss plateau (same as SAM2Trainer).
        scheduler = None
        if self.cfg.lr_scheduler_enabled:
            sched_patience = self.cfg.lr_scheduler_patience
            if self.cfg.early_stopping and sched_patience >= self.cfg.patience:
                sched_patience = max(1, self.cfg.patience // 3)
                print(f"[SegFormer] LR scheduler patience lowered to {sched_patience} "
                      f"so it can fire before early stopping (patience={self.cfg.patience}).")
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',                       # minimize validation loss
                factor=self.cfg.lr_scheduler_factor,
                patience=sched_patience,
                min_lr=self.cfg.lr_scheduler_min_lr,
            )
            print(f"[SegFormer] ReduceLROnPlateau active: monitor=val_loss mode=min "
                  f"factor={self.cfg.lr_scheduler_factor} patience={sched_patience} "
                  f"min_lr={self.cfg.lr_scheduler_min_lr:g} initial_lr={self.cfg.lr:g}")
        else:
            print("[SegFormer] LR scheduler disabled; learning rate is fixed.")

        self.lr_values = []
        self._overlays_written = False

        ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        if num_labels == 2:
            dice_loss = BinaryDiceLoss()
        else:
            dice_loss = MultiClassDiceLoss()
        loss_name = str(getattr(self.cfg, "loss_function", "ce_dice")).lower()
        focal = FocalLoss(gamma=getattr(self.cfg, "focal_gamma", 2.0))
        tversky = TverskyLoss(alpha=getattr(self.cfg, "tversky_alpha", 0.5),
                              beta=getattr(self.cfg, "tversky_beta", 0.5))
        self._loss_spec = self._build_loss_spec(loss_name, ce_loss, dice_loss, focal, tversky)
        scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.amp)

        metrics_log_path = os.path.join(self.cfg.output_dir, "metrics.json")

        # Clear all accumulators at the start of each training run
        self.loss_values.clear()
        self.val_loss_values.clear()
        self.train_accuracy_values.clear()
        self.val_accuracy_values.clear()
        self.val_iou_values.clear()
        self.miou_values.clear()
        self.val_dice_values.clear()
        self.val_true_list = []
        self.val_pred_list = []
        self.val_score_list = []
        self.epoch_list.clear()
        self.best_checkpoints = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Decision threshold for training accuracy and overlays: the latest
        # validation epoch's F1-optimal threshold (argmax-equivalent before the first).
        self._decision_threshold = ARGMAX_EQUIVALENT_THRESHOLD
        # Validation results of the best (lowest val loss) epoch, for the report.
        self._best_epoch = None
        self._best_threshold = None
        self._best_samples = None

        self._init_progress(train_loader, val_loader)

        last_completed_epoch = 0
        epoch = 0

        try:
            for epoch in range(1, self.cfg.num_epochs + 1):
                model.train()
                total_loss = 0.0
                train_correct = 0
                train_total = 0
                n_train_batches = 0

                for imgs, masks in train_loader:
                    if self._cancel_requested():
                        self.cancelled = True
                        break

                    # Apply augmentation to training data
                    if self.cfg.use_augmentation:
                        augmented_imgs = []
                        augmented_masks = []
                        for i in range(imgs.size(0)):
                            aug_img, aug_mask = apply_augmentation(imgs[i], masks[i], self.cfg)
                            augmented_imgs.append(aug_img)
                            augmented_masks.append(aug_mask)
                        imgs = torch.stack(augmented_imgs)
                        masks = torch.stack(augmented_masks)

                    imgs = imgs.to(self.cfg.device, non_blocking=True)
                    masks = masks.to(self.cfg.device, non_blocking=True)
                    masks = masks.round().long()
                    masks[(masks < 0) | (masks > num_labels - 1)] = 255

                    optimizer.zero_grad(set_to_none=True)

                    from contextlib import contextmanager

                    @contextmanager
                    def allow_nondeterminism():
                        torch.use_deterministic_algorithms(False)
                        try:
                            yield
                        finally:
                            torch.use_deterministic_algorithms(True, warn_only=True)

                    with allow_nondeterminism():
                        with torch.amp.autocast("cuda", enabled=self.cfg.amp):
                            outputs = model(pixel_values=imgs)
                            logits = outputs.logits
                            logits = torch.nn.functional.interpolate(
                                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                            )
                            loss = self._compute_loss(logits, masks, num_labels)

                        if not torch.isfinite(loss):
                            raise SegFormerError(
                                f"The training loss became {float(loss)} at epoch {epoch}, "
                                f"so training cannot continue.")

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()

                    total_loss += float(loss.detach().cpu())
                    n_train_batches += 1

                    # Train pixel accuracy, same decision rule as validation and inference
                    with torch.no_grad():
                        preds = self._target_predictions(logits.detach(), self._decision_threshold)
                        mb = (masks == self.cfg.target_class_id).long()
                        train_correct += (preds == mb).sum().item()
                        train_total += mb.numel()

                    self._tick_progress()

                if self.cancelled:
                    print(f"[SegFormer] Training cancelled by the user during epoch {epoch}.")
                    break

                avg_loss = total_loss / max(1, n_train_batches)
                train_accuracy = train_correct / train_total if train_total > 0 else 0.0
                self.loss_values.append(avg_loss)
                self.train_accuracy_values.append(train_accuracy)
                last_completed_epoch = epoch
                print(f"Epoch {epoch}/{self.cfg.num_epochs} | Train loss: {avg_loss:.4f} | Acc: {train_accuracy:.4f}")

                # GPU memory cleanup after each epoch
                torch.cuda.empty_cache()

                if epoch % self.cfg.val_every == 0:
                    result = self.evaluate(model, val_loader, ce_loss, dice_loss, num_labels)
                    if result is None:
                        self.cancelled = True
                        print(f"[SegFormer] Training cancelled by the user during validation of epoch {epoch}.")
                        break
                    avg_val_loss, val_accuracy, mean_iou, avg_dice, avg_iou, metrics = result
                    epoch_threshold = metrics["threshold"]
                    self._decision_threshold = epoch_threshold

                    # Step LR scheduler on validation loss (plateau detection).
                    _lr_before = optimizer.param_groups[0]['lr']
                    if scheduler is not None:
                        scheduler.step(avg_val_loss)
                    _lr_after = optimizer.param_groups[0]['lr']
                    self.lr_values.append(_lr_after)
                    if _lr_after < _lr_before:
                        print(f"[SegFormer] Epoch {epoch}: validation loss plateaued - "
                              f"learning rate reduced {_lr_before:.3e} -> {_lr_after:.3e}")

                    self.epoch_list.append(epoch)
                    self.val_loss_values.append(avg_val_loss)
                    self.val_accuracy_values.append(val_accuracy)
                    self.val_iou_values.append(avg_iou)
                    self.miou_values.append(mean_iou)
                    self.val_dice_values.append(avg_dice)

                    print(
                        f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | "
                        f"Acc: {val_accuracy:.4f} | "
                        f"IoU: {metrics['mean_iou']:.4f} | "
                        f"Dice: {avg_dice:.4f} | "
                        f"Precision: {metrics['precision']:.4f} | "
                        f"Recall: {metrics['recall']:.4f} | "
                        f"F1: {metrics['f1']:.4f} | "
                        f"Threshold: {epoch_threshold:.4f}"
                    )

                    # Save validation overlays (same policy as the SAM2 trainer)
                    if self.cfg.save_val_overlays and self._should_save_overlays(
                            epoch, self.cfg.num_epochs):
                        self._write_overlays(model, val_loader, epoch, epoch_threshold)

                    # Early stopping and checkpoint management on validation loss (min).
                    current_val_iou = metrics['mean_iou']   # reported, not used for stopping

                    if avg_val_loss < self.best_val_loss:
                        self.best_val_loss = avg_val_loss
                        self.patience_counter = 0
                        self._best_epoch = epoch
                        self._best_threshold = epoch_threshold
                        self._best_samples = (list(self.val_true_list),
                                              list(self.val_pred_list),
                                              list(self.val_score_list))

                        self.save_checkpoint(
                            model, optimizer, scaler,
                            categories=self.cfg.categories,
                            site_name=site_name,
                            learnrate=self.cfg.lr,
                            epochs=epoch,
                            output_dir=self.cfg.output_dir,
                            val_accuracy=val_accuracy,
                            miou=current_val_iou,
                            target_category_name=self.cfg.target_category_name,
                            val_iou=current_val_iou,
                            val_loss=avg_val_loss,
                            epoch_num=epoch,
                            val_best_threshold=epoch_threshold
                        )
                    else:
                        self.patience_counter += 1
                        print(f"No improvement for {self.patience_counter} epochs")

                    with open(metrics_log_path, "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch,
                            "train_loss": avg_loss,
                            "train_accuracy": train_accuracy,
                            "val_loss": avg_val_loss,
                            "val_accuracy": val_accuracy,
                            "val_dice": avg_dice,
                            **metrics
                        }) + "\n")

                    if self.cfg.early_stopping and self.patience_counter >= self.cfg.patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        self.early_stopped = True
                        self.early_stop_epoch = epoch
                        break

            # Guarantee overlays if none were written (early stop, cancel, val_every).
            if self.cfg.save_val_overlays and not self._overlays_written and last_completed_epoch > 0:
                self._write_overlays(model, val_loader, last_completed_epoch, self._decision_threshold)

        except SegFormerError:
            raise
        except Exception as e:
            raise SegFormerError(f"SegFormer training failed during epoch {epoch}.\n\n"
                                 f"{type(e).__name__}: {e}") from e
        finally:
            self.training_time_seconds = _time.perf_counter() - _training_start
            self._close_progress()

        print(f"[SegFormer] Training ended after epoch {last_completed_epoch}.")

        if not self.epoch_list:
            print("[SegFormer] No validation epoch completed; training graphs and report were not created.")
            return model

        # Graphs and report describe the best (lowest val loss) epoch's validation pixels.
        if self._best_samples is not None:
            self.val_true_list, self.val_pred_list, self.val_score_list = (
                list(s) for s in self._best_samples)
        try:
            self._plot_training_graphs(site_name=site_name, lr=self.cfg.lr)
        except Exception as e:
            raise SegFormerError(
                f"Training finished and the model checkpoints were saved in\n{self.cfg.output_dir}\n\n"
                f"but the training graphs and report could not be created.\n\n{type(e).__name__}: {e}") from e

        return model

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _write_overlays(self, model, val_loader, epoch, threshold):
        sample_imgs, sample_masks = next(iter(val_loader))
        sample_imgs = sample_imgs.to(self.cfg.device)
        sample_masks = sample_masks.to(self.cfg.device)
        model.eval()
        with torch.no_grad():
            sample_logits = model(pixel_values=sample_imgs).logits
            sample_logits = torch.nn.functional.interpolate(
                sample_logits, size=sample_masks.shape[-2:], mode="bilinear", align_corners=False)
            sample_preds = self._target_predictions(sample_logits, threshold)
            sample_masks_binary = (sample_masks.round().long() == self.cfg.target_class_id).long()
        self.save_validation_overlay(epoch, sample_imgs, sample_preds, sample_masks_binary, self.cfg.output_dir)
        self._overlays_written = True
