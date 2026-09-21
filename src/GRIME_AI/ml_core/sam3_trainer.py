#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import math
import json
from datetime import datetime

import numpy as np
import cv2
from PIL import Image

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler

# ===== Local infra =====
from opsilum.QProgressWheel import QProgressWheel
from opsilum.Save_Utils import Save_Utils
from opsilum.ml_core.model_training_visualization import ModelTrainingVisualization
from opsilum.JSON_Editor import JsonEditor
from opsilum.opsilum_QMessageBox import QMessageBox
from opsilum.utils.datasetutils import DatasetUtils

# ===== Import adapter/predictor =====
from opsilum.ml_core.sam3_adapter import SAM3ImagePredictor
from opsilum.dialogs.ML_image_processing.model_config_manager import ModelConfigManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===   ===      HELPER FUNCTIONS      ===   ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
def compute_mean_iou(y_true: list[int], y_pred: list[int]) -> float:
    arr_t = np.array(y_true, dtype=bool)
    arr_p = np.array(y_pred, dtype=bool)
    inter = np.logical_and(arr_t, arr_p).sum()
    union = np.logical_or(arr_t, arr_p).sum()
    return float(inter) / float(union) if union > 0 else 1.0

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs_f = probs.view(probs.size(0), -1)
        targets_f = targets.view(targets.size(0), -1)
        inter = (probs_f * targets_f).sum(dim=1)
        union = probs_f.sum(dim=1) + targets_f.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()

def dice_coeff_from_probs(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item()
    return float((2.0 * inter + eps) / (union + eps))

def iou_from_probs(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item() - inter
    return float((inter + eps) / (union + eps))


# ======================================================================================================================
# ======================================================================================================================
# ===   ===   ===   ===   ===   ===   ===   ===      class SAM3Trainer     ===   ===   ===   ===   ===   ===   ===   ===
# ======================================================================================================================
# ======================================================================================================================
class SAM3Trainer:

    def __init__(self, cfg: DictConfig = None):
        self.sam3_model = None
        self._last_checkpoint_path = None

        self.now = datetime.now()
        self.formatted_time = self.now.strftime('%Y%m%d_%H%M%S')

        # metrics
        self.epoch_list = []
        self.loss_values = []
        self.train_accuracy_values = []
        self.val_loss_values = []
        self.val_accuracy_values = []
        self.val_true_list = []
        self.val_pred_list = []
        self.val_score_list = []
        self.miou_values = []
        self.train_dice_values = []
        self.train_iou_values = []
        self.val_dice_values = []
        self.val_iou_values = []

        self.dataset_util = DatasetUtils()

        # site_config loading
        if cfg is None or "site_config" not in cfg:
            settings_folder = Save_Utils().get_settings_folder()
            CONFIG_FILENAME = "site_config.json"
            site_configuration_file = os.path.normpath(os.path.join(settings_folder, CONFIG_FILENAME))
            print(site_configuration_file)
            self.site_config = JsonEditor().load_json_file(site_configuration_file)
        else:
            self.site_config = OmegaConf.to_container(cfg.site_config, resolve=True)

        self.site_name = self.site_config['siteName']
        self.learning_rates = self.site_config['learningRates']
        self.optimizer_type = self.site_config['optimizer']
        self.loss_function = self.site_config['loss_function']
        self.weight_decay = self.site_config['weight_decay']
        self.num_epochs = self.site_config['number_of_epochs']
        self.max_best_checkpoints = int(self.site_config.get(
            'max_best_checkpoints', ModelConfigManager.get_default('max_best_checkpoints')))
        self.early_stopping = self.site_config['early_stopping']
        self.patience = self.site_config['patience']
        self.device = self.site_config.get('device', str(device))

        self.dataset = {}
        self.folders = None
        self.annotation_files = None
        self.all_folders = []
        self.all_annotations = []
        self.categories = []

        # output folder
        try:
            self.model_output_folder = os.path.join(
                Save_Utils().get_models_folder(), 'sam3',
                f"{self.formatted_time}_{self.site_name}"
            )
            os.makedirs(self.model_output_folder, exist_ok=True)
        except OSError as e:
            self.model_output_folder = None
            print(f"Error creating folders: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def run_training_pipeline(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # collect folders/annotations
        self.all_folders = []
        self.all_annotations = []
        paths = self.site_config.get('Path', [])
        for path in paths:
            directory_path = path['directoryPaths']
            self.folders = directory_path.get('folders', [])
            self.annotation_files = directory_path.get('annotations', [])
            self.all_folders.extend([self.folders])
            self.all_annotations.extend([self.annotation_files])

        self.dataset = self.dataset_util.load_images_and_annotations(self.all_folders, self.all_annotations)
        self.annotation_index = self.dataset_util.build_annotation_index(self.dataset)
        self.categories = self.build_unique_categories(self.all_annotations)

        # split dataset
        train_images, val_images = self.dataset_util.split_dataset(self.dataset)
        split_dataset_filename = os.path.join(
            self.model_output_folder,
            f"{self.formatted_time}_{self.site_name}training_and_validation_sets.json"
        )
        self.dataset_util.save_split_dataset(train_images, val_images, split_dataset_filename)

        # stats (optional; retained for parity)
        stats_train = self.summarize_mask_imbalance(train_images)
        stats_val = self.summarize_mask_imbalance(val_images)
        print(f"Foreground ratio (train): {stats_train}")
        print(f"Foreground ratio (val):   {stats_val}")

        if len(train_images) == 0:
            print("No training images found!")
            return

        # Hydra config + checkpoint
        dirname = os.path.dirname(__file__)
        config_dir = os.path.join("../sam3", "sam3", "configs")
        model_cfg_name = "sam3_base.yaml"
        sam3_checkpoint = os.path.normpath(os.path.join(dirname, "../sam3", "checkpoints", "sam3_base.pt"))

        print("SAM3 config dir:", config_dir)
        print("SAM3 checkpoint:", sam3_checkpoint)

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize(config_path=config_dir, version_base=None):
            cfg_intern = compose(config_name=model_cfg_name)
            raw_model_cfg = OmegaConf.to_container(cfg_intern.model, resolve=True)
            for key in ["no_obj_embed_spatial", "use_signed_tpos_enc_to_obj_ptrs", "device"]:
                raw_model_cfg.pop(key, None)
            sam3_cfg = OmegaConf.create(raw_model_cfg)

        # build adapter + predictor
        ###JES - THIS NEEDS TO BE FIXED!
        # sam3_adapter = SAM3Adapter(model_cfg=sam3_cfg, checkpoint_path=sam3_checkpoint, device=device)
        # self.sam3_model = model.to(device).train()

        for lr in self.learning_rates:
            print(f"Training with learning rate: {lr}")
            self.train_sam(lr, self.weight_decay, train_images, val_images, epochs=self.num_epochs)
            self._plot_training_graphs(lr)

        config_file = os.path.join(
            self.model_output_folder,
            f"{self.formatted_time}_{self.site_name}_configuration.txt"
        )
        self.save_config_to_text(config_file)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def train_sam(self, learnrate, weight_decay, predictor, train_images, val_images, epochs=20):
        self.reset_metrics()
        total_iterations = epochs * (len(train_images) + (len(val_images) if val_images else 0))

        progressBar = QProgressWheel(
            title="Training in-progress...", total=total_iterations,
            on_close=lambda: setattr(self, "progress_bar_closed", True)
        )

        # Seeds and determinism
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

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam3_model.train()
        predictor = SAM3ImagePredictor(self.sam3_model)
        optimizer = torch.optim.AdamW(self.sam3_model.parameters(), lr=learnrate, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        scaler = GradScaler() if dev.type == "cuda" else None
        use_amp = False

        best_val_loss = float("inf")
        patience_counter = 0
        divergence_threshold = 1e3
        last_completed_epoch = 0

        try:
            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")
                avg_epoch_loss, train_accuracy, train_dice, train_iou = self._train_one_epoch(
                    epoch=epoch,
                    train_images=train_images,
                    predictor=predictor,
                    optimizer=optimizer,
                    progressBar=progressBar,
                    use_amp=use_amp,
                    scaler=scaler
                )

                if avg_epoch_loss is None:
                    return

                last_completed_epoch = epoch + 1

                self.loss_values.append(avg_epoch_loss)
                self.train_accuracy_values.append(train_accuracy)
                self.train_dice_values.append(train_dice)
                self.train_iou_values.append(train_iou)
                print(f"Epoch {epoch + 1} Training "
                      f"Loss: {avg_epoch_loss:.4f} "
                      f"Acc: {train_accuracy:.4f} "
                      f"Dice: {train_dice:.4f} "
                      f"IoU: {train_iou:.4f}")

                if math.isnan(avg_epoch_loss) or avg_epoch_loss > divergence_threshold:
                    print("Training diverged. Aborting early.")
                    break

                if val_images:
                    avg_val_loss, val_accuracy, miou, avg_val_dice, avg_val_iou = self._validate_one_epoch(
                        val_images=val_images,
                        predictor=predictor,
                        loss_fn=loss_fn,
                        progressBar=progressBar
                    )
                    if avg_val_loss is None:
                        return

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
                            val_loss=avg_val_loss, val_accuracy=val_accuracy, miou=miou
                        )
                    else:
                        patience_counter += 1
                        if self.early_stopping and patience_counter >= self.patience:
                            print(f"Early stopping triggered at epoch {epoch + 1}.")
                            break
        finally:
            if last_completed_epoch > 0:
                self._save_model_checkpoint(predictor, learnrate, last_completed_epoch, suffix="final")

            if not getattr(self, "progress_bar_closed", False) and 'progressBar' in locals():
                progressBar.close()
            if 'progressBar' in locals():
                del progressBar

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
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
        Returns (avg_epoch_loss, train_accuracy, avg_dice, avg_iou).
        """
        self.sam3_model.train()
        self.epoch_list.append(epoch + 1)

        epoch_loss = 0.0
        train_correct, train_total = 0, 0
        processed_count = 0
        dice_sum, iou_sum = 0.0, 0.0

        np.random.shuffle(train_images)

        for idx, image_file in enumerate(train_images):
            if getattr(self, "progress_bar_closed", False):
                self._terminate_training(progressBar)
                return None, None, None, None

            image = np.array(Image.open(image_file).convert("RGB"))

            true_mask = self.dataset_util.load_true_mask(image_file, self.annotation_index)
            if true_mask is None:
                print(f"No annotation found for image {image_file}, skipping.")
                continue

            if target_label is not None:
                if hasattr(self.dataset_util, "label_to_index") and target_label in self.dataset_util.label_to_index:
                    label_index = self.dataset_util.label_to_index[target_label]
                    true_mask = (true_mask == label_index).astype(np.uint8)

            predictor.set_image(image)

            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                # low-res logits and scores from adapter
                low_res_masks, prd_scores = predictor.model.decode_masks(multimask_output=False, repeat_image=True)
                prd_masks = predictor.model.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                if len(true_mask.shape) == 3:
                    true_mask = true_mask[..., 0]

                gt_mask = torch.tensor(true_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
                if gt_mask.sum() == 0:
                    print(f"Skipping {image_file} - ground-truth mask is empty for label {target_label}.")
                    continue

                prd_mask = prd_masks[:, 0]            # [1,H,W] logits
                prd_mask = torch.sigmoid(prd_mask).unsqueeze(1)  # [1,1,H,W]

                if prd_mask.shape[-2:] != gt_mask.shape[-2:]:
                    import torch.nn.functional as F
                    prd_mask = F.interpolate(prd_mask, size=gt_mask.shape[-2:], mode="bilinear", align_corners=False)

                if prd_mask.shape != gt_mask.shape:
                    raise ValueError(f"Shape mismatch {prd_mask.shape} vs {gt_mask.shape} for {image_file}")

                seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()
                dice_loss = DiceLoss()(prd_mask, gt_mask)

                inter = (gt_mask * (prd_mask > 0.5)).sum((1, 2, 3))
                union = gt_mask.sum((1, 2, 3)) + (prd_mask > 0.5).sum((1, 2, 3)) - inter
                iou = inter / (union + 1e-6)
                iou[union == 0] = 1.0
                score_loss = torch.abs(torch.sigmoid(prd_scores[:, 0]) - iou).mean()

                loss = 0.5 * seg_loss + 0.5 * dice_loss + 0.05 * score_loss

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            d_metric = dice_coeff_from_probs(prd_mask, gt_mask)
            j_metric = iou_from_probs(prd_mask, gt_mask)

            dice_sum += d_metric
            iou_sum += j_metric
            epoch_loss += loss.item()
            processed_count += 1

            pred_binary = (prd_mask > 0.5).cpu().numpy()
            true_binary = gt_mask.cpu().numpy()
            train_correct += np.sum(pred_binary == true_binary)
            train_total += np.prod(true_binary.shape)

            if not getattr(self, "progress_bar_closed", False):
                progressBar.setValue(progressBar.getValue() + 1)

        avg_epoch_loss = epoch_loss / processed_count if processed_count > 0 else 0.0
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        avg_dice = dice_sum / processed_count if processed_count > 0 else 0.0
        avg_iou = iou_sum / processed_count if processed_count > 0 else 0.0

        return avg_epoch_loss, train_accuracy, avg_dice, avg_iou

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _validate_one_epoch(self, val_images, predictor, loss_fn, progressBar):
        """
        Returns (avg_val_loss, val_accuracy, miou, avg_val_dice, avg_val_iou).
        """
        val_loss_sum, dice_sum, iou_sum = 0.0, 0.0, 0.0
        n_items = 0

        self.sam3_model.eval()
        progressBar.setWindowTitle("Validation in-progress")

        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for val_idx, val_image_file in enumerate(val_images):
                if getattr(self, "progress_bar_closed", False):
                    self._terminate_validation(progressBar)
                    return None, None, None, None, None

                val_image = np.array(Image.open(val_image_file).convert("RGB"))
                val_true_mask = self.dataset_util.load_true_mask(val_image_file, self.annotation_index)

                if val_true_mask is None:
                    print(f"No annotation found for validation image {val_image_file}, skipping.")
                    continue

                predictor.set_image(val_image)
                masks, scores, low_res_logits = predictor.predict(point_coords=None, point_labels=None, multimask_output=False)

                if masks.size > 0:
                    best_idx = int(np.argmax(scores))
                    best_mask = masks[best_idx]

                    if len(val_true_mask.shape) > 2:
                        val_true_mask = val_true_mask[:, :, 0]

                    best_mask_tensor = torch.tensor(best_mask, dtype=torch.float32).unsqueeze(0).to(device)
                    val_true_mask_tensor = torch.tensor(val_true_mask, dtype=torch.float32).unsqueeze(0).to(device)

                    bce_val = loss_fn(best_mask_tensor, val_true_mask_tensor).item()
                    dice_val = DiceLoss()(best_mask_tensor, val_true_mask_tensor).item()
                    val_loss += 0.5 * bce_val + 0.5 * dice_val

                    d_metric = dice_coeff_from_probs(best_mask_tensor, val_true_mask_tensor)
                    j_metric = iou_from_probs(best_mask_tensor, val_true_mask_tensor)
                    dice_sum += d_metric
                    iou_sum += j_metric
                    n_items += 1

                    pred_binary = (best_mask_tensor > 0.5).cpu().numpy()
                    true_binary = val_true_mask_tensor.cpu().numpy()
                    val_correct += np.sum(pred_binary == true_binary)
                    val_total += np.prod(true_binary.shape)

                    self.val_true_list.extend(int(x) for x in true_binary.flatten())
                    self.val_pred_list.extend(int(x) for x in pred_binary.flatten())

                    import torch.nn.functional as F
                    logit_lr = torch.tensor(
                        low_res_logits[best_idx],
                        dtype=torch.float32,
                        device=device
                    ).unsqueeze(0).unsqueeze(0)

                    prob_lr = torch.sigmoid(logit_lr)
                    H_full, W_full = val_true_mask_tensor.shape[1], val_true_mask_tensor.shape[2]
                    prob_full = F.interpolate(
                        prob_lr, size=(H_full, W_full), mode='bilinear', align_corners=False
                    )

                    if val_idx < 5:
                        img_vis = (val_image).copy()
                        overlay = (prob_full.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                        overlay_color = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
                        alpha = 0.4
                        blended = cv2.addWeighted(img_vis, 1.0, overlay_color, alpha, 0.0)
                        out_path = os.path.join(self.model_output_folder,
                                                f"{self.formatted_time}_{self.site_name}_val_overlay_e{len(self.epoch_list)}_{val_idx}.png")
                        cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

                    score_flat = prob_full.squeeze(0).squeeze(0).cpu().numpy().flatten()
                    self.val_score_list.extend(score_flat.tolist())

                    progressBar.setValue(progressBar.getValue() + 1)

        avg_val_loss = val_loss / len(val_images) if len(val_images) > 0 else 0.0
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        avg_val_dice = dice_sum / n_items if n_items > 0 else 0.0
        avg_val_iou = iou_sum / n_items if n_items > 0 else 0.0
        miou = compute_mean_iou(self.val_true_list, self.val_pred_list)

        return avg_val_loss, val_accuracy, miou, avg_val_dice, avg_val_iou

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _save_model_checkpoint(self, predictor, learnrate, epochs, suffix="final", val_loss=None, val_accuracy=None,
                               miou=None, target_category_name=None):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        num_classes = None
        try:
            if hasattr(predictor.model.model, "roi_heads"):
                num_classes = predictor.model.model.roi_heads.box_predictor.cls_score.out_features
        except Exception:
            pass

        ckpt = {
            "model_state_dict": predictor.model.model.state_dict(),
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
        }

        if suffix == "final":
            torch_filename = f"{timestamp}_{self.site_name}_{suffix}_lr{learnrate}_epoch{epochs}.torch"
        else:
            torch_filename = f"{self.formatted_time}_{self.site_name}_{suffix}_{learnrate}.torch"

        save_path = os.path.join(self.model_output_folder, torch_filename)
        torch.save(ckpt, save_path)
        print(f"Model checkpoint saved to {save_path}")

        if hasattr(self, "_last_checkpoint_path") and self._last_checkpoint_path and os.path.exists(self._last_checkpoint_path):
            try:
                os.remove(self._last_checkpoint_path)
                print(f"Deleted previous checkpoint: {self._last_checkpoint_path}")
            except Exception as e:
                print(f"Warning: could not delete previous checkpoint {self._last_checkpoint_path}: {e}")

        self._last_checkpoint_path = save_path
        return save_path

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
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
                cid = cat.get('id'); cname = cat.get('name')
                if cid is None or cname is None:
                    print(f"Warning: bad category entry in '{p}': {cat}")
                    continue

                if cid in id_to_name and id_to_name[cid] != cname:
                    print(f"⚠️ ID conflict: {cid} is '{id_to_name[cid]}' and '{cname}'")
                    continue
                id_to_name[cid] = cname

                if cname in name_to_id and name_to_id[cname] != cid:
                    print(f"⚠️ Name conflict: '{cname}' → {name_to_id[cname]} vs {cid}")
                    continue
                name_to_id[cname] = cid

                merged.append({"id": cid, "name": cname})

        unique = {(c['id'], c['name']): c for c in merged}
        return list(unique.values())

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def save_config_to_text(self, output_text_file):
        with open(output_text_file, 'w') as text_file:
            text_file.write(f"Site: {self.site_name}\n")
            text_file.write(f"Learning Rates: {self.learning_rates}\n")
            text_file.write(f"Optimizer: {self.optimizer_type}\n")
            text_file.write(f"Loss Function: {self.loss_function}\n")
            text_file.write(f"Weight Decay: {self.weight_decay}\n")
            text_file.write(f"Number of Epochs: {self.num_epochs}\n")
            text_file.write(f"Best Checkpoints Kept: {self.max_best_checkpoints}\n")
            text_file.write(f"Early Stopping: {self.early_stopping}\n")
            text_file.write(f"Patience: {self.patience}\n")
            text_file.write(f"Device: {device}\n")
            text_file.write(f"Folders: {self.folders}\n")
            text_file.write(f"Annotations: {self.annotation_files}\n")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _plot_training_graphs(self, lr: float):
        progressBar = QProgressWheel(
            title="Generating graphs...", total=9,
            on_close=lambda: setattr(self, "progress_bar_closed", True)
        )

        viz = ModelTrainingVisualization(
            self.model_output_folder, self.formatted_time, self.categories
        )

        viz.plot_loss_curves(
            epochs=self.epoch_list,
            train_loss=self.loss_values,
            val_loss=self.val_loss_values,
            site_name=self.site_name,
            lr=lr
        ); progressBar.setValue(progressBar.getValue() + 1); progressBar.show()

        viz.plot_accuracy(
            epochs=self.epoch_list,
            train_acc=self.train_accuracy_values,
            val_acc=self.val_accuracy_values,
            site_name=self.site_name,
            lr=lr
        ); progressBar.setValue(progressBar.getValue() + 1); progressBar.show()

        viz.plot_confusion_matrix(
            y_true=self.val_true_list,
            y_pred=self.val_pred_list,
            site_name=self.site_name,
            lr=lr,
            normalize=True,
            file_prefix="Normalized"
        ); progressBar.setValue(progressBar.getValue() + 1); progressBar.show()

        viz.plot_roc_curve(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        ); progressBar.setValue(progressBar.getValue() + 1); progressBar.show()

        viz.plot_precision_recall(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        ); progressBar.setValue(progressBar.getValue() + 1); progressBar.show()

        viz.plot_f1_score(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        ); progressBar.setValue(progressBar.getValue() + 1); progressBar.show()

        viz.plot_miou_curve(
            epochs=self.epoch_list,
            miou_values=self.miou_values,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        ); progressBar.setValue(progressBar.getValue() + 1); progressBar.show()

        viz.plot_dice_curve(
            epochs=self.epoch_list,
            dice_values=self.val_dice_values,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        ); progressBar.setValue(progressBar.getValue() + 1); progressBar.show()

        viz.plot_iou_curve(
            epochs=self.epoch_list,
            iou_values=self.val_iou_values,
            site_name=self.site_name,
            lr=lr,
            file_prefix=f"{self.formatted_time}_{self.site_name}_lr{lr:.5f}"
        ); progressBar.setValue(progressBar.getValue() + 1); progressBar.show()

        progressBar.close()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _terminate_training(self, progressBar):
        msg = "You have cancelled the model training currently in-progress. A model has not been generated."
        msgBox = QMessageBox('Model Training Terminated', msg, QMessageBox.Close)
        msgBox.displayMsgBox()
        if progressBar:
            progressBar.close()
        del progressBar

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def _terminate_validation(self, progressBar):
        msg = "You have cancelled validation. The current validation pass was not completed."
        msgBox = QMessageBox("Validation Cancelled", msg, QMessageBox.Close)
        msgBox.displayMsgBox()
        if progressBar:
            progressBar.close()
        del progressBar
