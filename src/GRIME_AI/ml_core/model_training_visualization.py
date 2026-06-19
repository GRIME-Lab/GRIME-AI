#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# !/usr/bin/env python3
# model_training_visualization.py
# OPTIMIZED VERSION - All fixes and improvements applied

import os
import numpy as np
from fpdf import FPDF

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, no GUI overhead
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)


# ======================================================================================================================
# ======================================================================================================================
#  =====     =====     =====     =====     =====     =====     =====     =====     =====     =====     =====     =====
# ======================================================================================================================
# ======================================================================================================================
class ModelTrainingVisualization:
    def __init__(self, models_folder: str, formatted_time: str, categories: list[dict]):
        """
        models_folder: path to save figures
        formatted_time: timestamp string, e.g. '2025-08-07_13-10-00'
        categories:   list of {"id": cid, "name": cname}
        """
        self.models_folder = models_folder
        self.graphs_folder = os.path.join(models_folder, "graphs")
        os.makedirs(self.graphs_folder, exist_ok=True)
        self.formatted_time = formatted_time

        # sort categories once, build label IDs + names
        sorted_cats = sorted(categories, key=lambda c: c["id"])
        self._label_ids = [c["id"] for c in sorted_cats]
        self._class_names = [c["name"] for c in sorted_cats]
        self.categories = categories

        # placeholders – populate these before plotting
        self.epoch_list = []
        self.loss_values = []
        self.train_accuracy_values = []
        self.val_accuracy_values = []
        self.val_loss_values = []

        # ✅ OPTIMIZATION: Configurable parameters
        self.max_samples = 100_000  # maximum number of points to plot
        self.plot_dpi = 150  # DPI for saved figures (consistent across all plots)
        self.grid_alpha = 0.3  # Transparency for grid lines

        # apply Seaborn global styling
        sns.set_theme(style="whitegrid", palette="deep")

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ NEW: Helper method for consistent title generation
    # ------------------------------------------------------------------------------------------------------------------
    def _make_title(self, base_title: str, site_name: str = None, lr: float = None, model_name: str = None) -> str:
        """
        Generate consistent two-line plot titles.

        Line 1: Site name
        Line 2: Model name and metric label with optional learning rate

        Args:
            base_title:  Metric label, e.g. "Accuracy" or "Loss"
            site_name:   Site identifier, e.g. "KOLA"
            lr:          Learning rate
            model_name:  Model name, e.g. "SAM2" or "SegFormer-LoRA"

        Returns:
            Formatted title string
        """
        line2 = base_title
        if model_name:
            line2 = f"{model_name} — {base_title}"
        if site_name:
            return f"{site_name}\n{line2}"
        return line2

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ NEW: Helper to draw a hyperparameter info box on any axes
    # ------------------------------------------------------------------------------------------------------------------
    def _add_info_box(
            self,
            ax,
            lr: float = None,
            epochs: int = None,
            batch_size: int = None,
            num_train_images: int = None,
            num_val_images: int = None,
            blob_radius_pct: float = None,
            weight_decay: float = None,
            lora_rank: int = None,
            lora_alpha: float = None,
            loc: str = "lower right",
    ):
        """
        Draws a small hyperparameter info box in the lower-right corner of ax.
        Only lines with non-None values are included.
        """
        lines = []
        if lr is not None:
            lines.append(f"lr: {lr:.5f}")
        if epochs is not None:
            lines.append(f"Epochs: {epochs}")
        if batch_size is not None:
            lines.append(f"Batch size: {batch_size}")
        if num_train_images is not None:
            lines.append(f"No. train images: {num_train_images}")
        if num_val_images is not None:
            lines.append(f"No. val images: {num_val_images}")
        if blob_radius_pct is not None:
            lines.append(f"Blob radius: {blob_radius_pct:.1f}%")
        if weight_decay is not None:
            lines.append(f"Weight decay: {weight_decay}")
        if lora_rank is not None:
            lines.append(f"LoRA rank: {lora_rank}")
        if lora_alpha is not None:
            lines.append(f"LoRA alpha: {lora_alpha}")

        if not lines:
            return

        text = "\n".join(lines)
        x = 0.97 if "right" in loc else 0.03
        y = 0.03 if "lower" in loc else 0.97
        va = "bottom" if "lower" in loc else "top"
        ha = "right" if "right" in loc else "left"
        ax.text(
            x, y, text,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment=va,
            horizontalalignment=ha,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7, edgecolor="gray", linewidth=0.5),
            family="monospace",
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Added grid, standardized DPI, used title helper
    # ------------------------------------------------------------------------------------------------------------------
    def plot_accuracy(
            self,
            train_epochs: list[int],
            train_acc: list[float],
            val_epochs: list[int],
            val_acc: list[float],
            site_name: str,
            lr: float,
            model_name: str = None,
            epochs: int = None,
            batch_size: int = None,
            num_train_images: int = None,
            num_val_images: int = None,
            blob_radius_pct: float = None,
            weight_decay: float = None,
            lora_rank: int = None,
            lora_alpha: float = None,
    ):
        if not train_epochs or not train_acc or not val_epochs or not val_acc:
            return

        # vectorize inputs
        tr_ep = np.asarray(train_epochs)
        tr_acc = np.asarray(train_acc)
        vl_ep = np.asarray(val_epochs)
        vl_acc = np.asarray(val_acc)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tr_ep, tr_acc, label="Train Accuracy", lw=1, color="tab:blue")
        ax.plot(vl_ep, vl_acc, label="Val Accuracy", lw=1, color="tab:orange")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(self._make_title("Accuracy", site_name, lr, model_name))  # ✅ Use helper
        _yall = list(tr_acc) + list(vl_acc)
        _ymin, _ymax = min(_yall), max(_yall)
        _ypad = max(0.005, (_ymax - _ymin) * 0.05)
        ax.set_ylim(max(0.0, _ymin - _ypad), _ymax + _ypad)
        ax.legend(loc="lower left")
        self._add_info_box(ax, lr=lr, epochs=len(vl_acc), batch_size=batch_size,
                           num_train_images=num_train_images, num_val_images=num_val_images,
                           blob_radius_pct=blob_radius_pct, weight_decay=weight_decay,
                           lora_rank=lora_rank, lora_alpha=lora_alpha, loc="lower right")
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Added grid

        out_file = os.path.join(
            self.graphs_folder,
            f"{self.formatted_time}_{site_name}_AccuracyCurves_lr{lr:.5f}.png"
        )
        fig.tight_layout()
        fig.savefig(out_file, dpi=self.plot_dpi, bbox_inches="tight")  # ✅ Consistent DPI
        plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Added grid, used title helper
    # ------------------------------------------------------------------------------------------------------------------
    def plot_loss_curves(
            self,
            train_epochs: list[int],
            train_loss: list[float],
            val_epochs: list[int],
            val_loss: list[float],
            site_name: str,
            lr: float,
            model_name: str = None,
            epochs: int = None,
            batch_size: int = None,
            num_train_images: int = None,
            num_val_images: int = None,
            blob_radius_pct: float = None,
            weight_decay: float = None,
            lora_rank: int = None,
            lora_alpha: float = None,
    ):
        if not train_epochs or not train_loss or not val_epochs or not val_loss:
            return

        # vectorize + subsample if too many points
        tr_ep = np.asarray(train_epochs)
        tr = np.asarray(train_loss)
        vl_ep = np.asarray(val_epochs)
        vl = np.asarray(val_loss)

        if tr_ep.size > self.max_samples:
            idxs = np.linspace(0, tr_ep.size - 1, self.max_samples, dtype=int)
            tr_ep = tr_ep[idxs]
            tr = tr[idxs]
            print(f"[plot_loss_curves] downsampled train to {tr_ep.size} points")

        if vl_ep.size > self.max_samples:
            idxs = np.linspace(0, vl_ep.size - 1, self.max_samples, dtype=int)
            vl_ep = vl_ep[idxs]
            vl = vl[idxs]
            print(f"[plot_loss_curves] downsampled val to {vl_ep.size} points")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tr_ep, tr, label="Train Loss", color="tab:blue", lw=1)
        ax.plot(vl_ep, vl, label="Val Loss", color="tab:orange", lw=1)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(self._make_title("Loss", site_name, lr, model_name))  # ✅ Use helper
        ax.legend(loc="upper right")
        _yall = list(tr) + list(vl)
        _ymin, _ymax = min(_yall), max(_yall)
        _ypad = max(0.005, (_ymax - _ymin) * 0.05)
        ax.set_ylim(_ymin - _ypad, _ymax + _ypad)
        self._add_info_box(ax, lr=lr, epochs=len(vl), batch_size=batch_size,
                           num_train_images=num_train_images, num_val_images=num_val_images,
                           blob_radius_pct=blob_radius_pct, weight_decay=weight_decay,
                           lora_rank=lora_rank, lora_alpha=lora_alpha, loc="lower left")
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Added grid

        out_file = os.path.join(
            self.graphs_folder,
            f"{self.formatted_time}_{site_name}_LossCurves_lr{lr:.5f}.png"
        )
        fig.tight_layout()
        fig.savefig(out_file, dpi=self.plot_dpi, bbox_inches="tight")  # ✅ Already consistent
        plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ CRITICAL FIX: Memory leak fixed - capture fig reference and close it properly
    # ✅ FIXED: Standardized DPI
    # ------------------------------------------------------------------------------------------------------------------
    def plot_confusion_matrix(
            self,
            y_true: list[int],
            y_pred: list[int],
            site_name: str = "",
            lr: float = 0.0,
            normalize: bool = False,
            file_prefix: str = "",
            target_label: str = ""
    ):
        # val_true_list and val_pred_list are binary: 0 = background, 1 = target class.
        pos_name = target_label if target_label else "target"
        labels = [0, 1]
        class_names = ["background", pos_name]

        # force inclusion of both classes even if one is absent
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if normalize:
            # compute row sums (shape: [n_classes, 1])
            row_sums = cm.sum(axis=1, keepdims=True).astype(float)

            # safe divide: wherever row_sums != 0, do cm/row_sums, else leave zeros
            cm = np.divide(
                cm.astype(float),
                row_sums,
                out=np.zeros_like(cm, dtype=float),
                where=(row_sums != 0)
            )
            fmt = ".2f"
            title = "Normalized Confusion Matrix"
        else:
            fmt = "d"
            title = "Confusion Matrix"

        fig = plt.figure(figsize=(6, 5))  # ✅ CRITICAL: Capture figure reference
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(title)

        prefix_tag = f"{file_prefix}_" if file_prefix else ""
        norm_tag = "_norm" if normalize else ""
        filename = (
            f"{self.formatted_time}_{site_name}_"
            f"{prefix_tag}ConfusionMatrix_{lr}{norm_tag}.png"
        )
        png_path = os.path.join(self.graphs_folder, filename)

        plt.tight_layout()
        plt.savefig(png_path, dpi=self.plot_dpi)  # ✅ Consistent DPI
        plt.close(fig)  # ✅ CRITICAL: Close the specific figure (was plt.close())
        return png_path

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Added grid, used title helper, simplified downsampling
    # ------------------------------------------------------------------------------------------------------------------
    def plot_precision_recall(
            self,
            y_true,
            y_scores,
            site_name=None,
            lr=None,
            model_name=None,
            file_prefix=None
    ):
        # vectorize & cast
        y_true_arr = np.asarray(y_true)
        y_scores_arr = np.asarray(y_scores)  # ✅ Let numpy decide dtype

        if y_true_arr.size == 0 or y_scores_arr.size == 0:
            print("Cannot plot precision-recall: empty inputs.")
            return

        # enforce binary
        labels = np.unique(y_true_arr)
        if not np.all(np.isin(labels, [0, 1])):
            y_true_arr = (y_true_arr == labels.max()).astype(int)

        # subsample raw inputs
        if y_scores_arr.size > self.max_samples:
            idxs = np.linspace(0, y_scores_arr.size - 1, self.max_samples, dtype=int)
            y_true_arr = y_true_arr[idxs]
            y_scores_arr = y_scores_arr[idxs]
            print(f"[plot_precision_recall] downsampled to {y_true_arr.size} points")

        precision, recall, _ = precision_recall_curve(y_true_arr, y_scores_arr)

        # subsample PR curve
        if precision.size > self.max_samples:
            idxs = np.linspace(0, precision.size - 1, self.max_samples, dtype=int)
            precision = precision[idxs]
            recall = recall[idxs]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(recall, precision, lw=1, color="tab:blue")
        ax.set_title(self._make_title("Precision-Recall Curve", site_name, lr, model_name))  # ✅ Use helper
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Added grid
        fig.tight_layout()

        if file_prefix:
            filename = f"{file_prefix}_precision_recall_curve.png"
            pr_path = os.path.join(self.graphs_folder, filename)
            fig.savefig(pr_path, dpi=self.plot_dpi)  # ✅ Consistent DPI
            plt.close(fig)
            print(f"PR plot saved to {filename}")
            return pr_path
        else:
            plt.close(fig)  # ✅ Ensure cleanup even if not saving

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Removed redundant try-catch, added grid, used title helper
    # ------------------------------------------------------------------------------------------------------------------
    def plot_roc_curve(
            self,
            y_true,
            y_scores,
            site_name=None,
            lr=None,
            model_name=None,
            file_prefix=None
    ):
        # vectorize & cast
        y_true_arr = np.asarray(y_true)
        y_scores_arr = np.asarray(y_scores)  # ✅ Let numpy decide dtype

        if y_true_arr.size == 0 or y_scores_arr.size == 0:
            print("Cannot plot ROC: empty inputs.")
            return

        labels = np.unique(y_true_arr)
        if not np.all(np.isin(labels, [0, 1])):
            y_true_arr = (y_true_arr == labels.max()).astype(int)

        # Downsample before computing ROC (prevents memory issues)
        if y_scores_arr.size > self.max_samples:
            idxs = np.linspace(0, y_scores_arr.size - 1, self.max_samples, dtype=int)
            y_true_arr = y_true_arr[idxs]
            y_scores_arr = y_scores_arr[idxs]
            print(f"[plot_roc_curve] downsampled to {y_true_arr.size} points")

        # ✅ REMOVED: Redundant try-catch (already downsampled above)
        fpr, tpr, _ = roc_curve(y_true_arr, y_scores_arr)
        roc_auc = auc(fpr, tpr)

        # subsample ROC curve points if still too large
        if fpr.size > self.max_samples:
            idxs = np.linspace(0, fpr.size - 1, self.max_samples, dtype=int)
            fpr, tpr = fpr[idxs], tpr[idxs]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color="tab:blue", lw=1, label=f"AUC={roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")

        ax.set_title(self._make_title("ROC Curve", site_name, lr, model_name))  # ✅ Use helper
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Added grid
        fig.tight_layout()

        if file_prefix:
            filename = f"{file_prefix}_roc_curve.png"
            roc_path = os.path.join(self.graphs_folder, filename)
            fig.savefig(roc_path, dpi=self.plot_dpi)  # ✅ Consistent DPI
            plt.close(fig)
            print(f"ROC plot saved to {filename}")
            return roc_path
        else:
            plt.close(fig)  # ✅ Ensure cleanup

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Added grid, used title helper
    # ------------------------------------------------------------------------------------------------------------------
    def plot_f1_score(
            self,
            y_true,
            y_scores,
            site_name: str = None,
            lr: float = None,
            model_name: str = None,
            file_prefix: str = None
    ):
        y_true_arr = np.asarray(y_true)
        y_scores_arr = np.asarray(y_scores)  # ✅ Let numpy decide dtype

        if y_true_arr.size == 0 or y_scores_arr.size == 0:
            print("Cannot plot F1: empty inputs.")
            return

        labels = np.unique(y_true_arr)
        if not np.all(np.isin(labels, [0, 1])):
            y_true_arr = (y_true_arr == labels.max()).astype(int)

        precision, recall, thresholds = precision_recall_curve(y_true_arr, y_scores_arr)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        f1 = f1_scores[:-1]
        thr = thresholds

        best_idx = np.argmax(f1)
        best_thr = thr[best_idx]
        best_f1 = f1[best_idx]

        if f1.size > self.max_samples:
            idxs = np.linspace(0, f1.size - 1, self.max_samples, dtype=int)
            f1 = f1[idxs]
            thr = thr[idxs]
            print(f"[plot_f1_score] downsampled to {f1.size} points")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thr, f1, color="tab:green", lw=1, label="F1 Score")
        ax.scatter(
            best_thr,
            best_f1,
            color="red",
            s=50,
            zorder=5,
            label=f"Max F1={best_f1:.2f} @thr={best_thr:.2f}"
        )

        ax.set_title(self._make_title("F1 Score vs Threshold", site_name, lr, model_name))  # ✅ Use helper
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1 Score")
        ax.legend(loc="best")
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Added grid
        fig.tight_layout()

        if file_prefix:
            filename = f"{file_prefix}_f1_curve.png"
            f1_path = os.path.join(self.graphs_folder, filename)
            fig.savefig(f1_path, dpi=self.plot_dpi)  # ✅ Consistent DPI
            plt.close(fig)
            print(f"F1 plot saved to {filename}")
            return f1_path
        else:
            plt.close(fig)  # ✅ Ensure cleanup

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Used title helper (grid already present)
    # ------------------------------------------------------------------------------------------------------------------
    def plot_miou_curve(
            self,
            epochs: list[int],
            miou_values: list[float],
            site_name: str = None,
            lr: float = None,
            model_name: str = None,
            file_prefix: str = "miou",
            num_epochs: int = None,
            batch_size: int = None,
            num_train_images: int = None,
            num_val_images: int = None,
            blob_radius_pct: float = None,
            weight_decay: float = None,
            lora_rank: int = None,
            lora_alpha: float = None,
    ):
        """
        Plots Mean IoU vs. epoch. Ensures x and y have the same length by
        truncating to the smaller size if they differ, then applies optional
        downsampling to max_samples.
        """
        # Early exit on empty data
        if not epochs or not miou_values:
            print("Cannot plot Mean IoU curve: empty inputs.")
            return

        # Convert to numpy arrays
        ep = np.asarray(epochs)
        mi = np.asarray(miou_values)

        # Dimension check: truncate to the smaller of two lengths
        if ep.size != mi.size:
            print(
                f"[plot_miou_curve] dimension mismatch: "
                f"epochs ({ep.size}) != miou ({mi.size}); "
                "truncating to the smaller length."
            )
            min_len = min(ep.size, mi.size)
            ep = ep[:min_len]
            mi = mi[:min_len]

        # Optional downsampling if still more points than max_samples
        if ep.size > self.max_samples:
            idxs = np.linspace(0, ep.size - 1, self.max_samples, dtype=int)
            ep = ep[idxs]
            mi = mi[idxs]
            print(f"[plot_miou_curve] downsampled to {ep.size} points")

        # Plotting
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ep, mi, color="tab:blue", lw=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean IoU")
        ax.set_title(self._make_title("Mean IoU over Epochs", site_name, lr, model_name))  # ✅ Use helper
        _yall = list(mi)
        _ymin, _ymax = min(_yall), max(_yall)
        _ypad = max(0.005, (_ymax - _ymin) * 0.05)
        ax.set_ylim(max(0.0, _ymin - _ypad), _ymax + _ypad)
        self._add_info_box(ax, lr=lr, epochs=len(mi), batch_size=batch_size,
                           num_train_images=num_train_images, num_val_images=num_val_images,
                           blob_radius_pct=blob_radius_pct, weight_decay=weight_decay,
                           lora_rank=lora_rank, lora_alpha=lora_alpha)
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Already present, now uses consistent alpha
        fig.tight_layout()

        # Save or show
        if file_prefix:
            filename = f"{file_prefix}_miou_curve.png"
            miou_path = os.path.join(self.graphs_folder, filename)
            fig.savefig(miou_path, dpi=self.plot_dpi)  # ✅ Consistent DPI
            plt.close(fig)
            print(f"Mean IoU plot saved to {filename}")
            return miou_path
        else:
            plt.close(fig)  # ✅ Ensure cleanup

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Used title helper (grid already present)
    # ------------------------------------------------------------------------------------------------------------------
    def plot_dice_curve(
            self,
            epochs: list[int],
            dice_values: list[float],
            site_name: str = None,
            lr: float = None,
            model_name: str = None,
            file_prefix: str = "dice",
            num_epochs: int = None,
            batch_size: int = None,
            num_train_images: int = None,
            num_val_images: int = None,
            blob_radius_pct: float = None,
            weight_decay: float = None,
            lora_rank: int = None,
            lora_alpha: float = None,
    ):
        if not epochs or not dice_values:
            print("Cannot plot Dice curve: empty inputs.")
            return

        ep = np.asarray(epochs)
        dv = np.asarray(dice_values)

        if ep.size != dv.size:
            print(
                f"[plot_dice_curve] dimension mismatch: "
                f"epochs ({ep.size}) != dice ({dv.size}); "
                "truncating to the smaller length."
            )
            min_len = min(ep.size, dv.size)
            ep = ep[:min_len]
            dv = dv[:min_len]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ep, dv, color="tab:purple", lw=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dice")
        ax.set_title(self._make_title("Validation Dice over Epochs", site_name, lr, model_name))  # ✅ Use helper
        _yall = list(dv)
        _ymin, _ymax = min(_yall), max(_yall)
        _ypad = max(0.005, (_ymax - _ymin) * 0.05)
        ax.set_ylim(max(0.0, _ymin - _ypad), _ymax + _ypad)
        self._add_info_box(ax, lr=lr, epochs=len(dv), batch_size=batch_size,
                           num_train_images=num_train_images, num_val_images=num_val_images,
                           blob_radius_pct=blob_radius_pct, weight_decay=weight_decay,
                           lora_rank=lora_rank, lora_alpha=lora_alpha)
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Already present, now uses consistent alpha
        fig.tight_layout()

        if file_prefix:
            filename = f"{file_prefix}_dice_curve.png"
            dice_path = os.path.join(self.graphs_folder, filename)
            fig.savefig(dice_path, dpi=self.plot_dpi)  # ✅ Consistent DPI
            plt.close(fig)
            print(f"Dice plot saved to {filename}")
            return dice_path
        else:
            plt.close(fig)  # ✅ Ensure cleanup

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ FIXED: Used title helper (grid already present)
    # ------------------------------------------------------------------------------------------------------------------
    def plot_iou_curve(
            self,
            epochs: list[int],
            iou_values: list[float],
            site_name: str = None,
            lr: float = None,
            model_name: str = None,
            file_prefix: str = "iou",
            num_epochs: int = None,
            batch_size: int = None,
            num_train_images: int = None,
            num_val_images: int = None,
            blob_radius_pct: float = None,
            weight_decay: float = None,
            lora_rank: int = None,
            lora_alpha: float = None,
    ):
        if not epochs or not iou_values:
            print("Cannot plot IoU curve: empty inputs.")
            return

        ep = np.asarray(epochs)
        iv = np.asarray(iou_values)

        if ep.size != iv.size:
            print(
                f"[plot_iou_curve] dimension mismatch: "
                f"epochs ({ep.size}) != iou ({iv.size}); "
                "truncating to the smaller length."
            )
            min_len = min(ep.size, iv.size)
            ep = ep[:min_len]
            iv = iv[:min_len]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ep, iv, color="tab:green", lw=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("IoU")
        ax.set_title(self._make_title("Validation IoU over Epochs", site_name, lr, model_name))  # ✅ Use helper
        _yall = list(iv)
        _ymin, _ymax = min(_yall), max(_yall)
        _ypad = max(0.005, (_ymax - _ymin) * 0.05)
        ax.set_ylim(max(0.0, _ymin - _ypad), _ymax + _ypad)
        self._add_info_box(ax, lr=lr, epochs=len(iv), batch_size=batch_size,
                           num_train_images=num_train_images, num_val_images=num_val_images,
                           blob_radius_pct=blob_radius_pct, weight_decay=weight_decay,
                           lora_rank=lora_rank, lora_alpha=lora_alpha)
        ax.grid(True, alpha=self.grid_alpha)  # ✅ Already present, now uses consistent alpha
        fig.tight_layout()

        if file_prefix:
            filename = f"{file_prefix}_iou_curve.png"
            iou_path = os.path.join(self.graphs_folder, filename)
            fig.savefig(iou_path, dpi=self.plot_dpi)  # ✅ Consistent DPI
            plt.close(fig)
            print(f"IoU plot saved to {filename}")
            return iou_path
        else:
            plt.close(fig)  # ✅ Ensure cleanup
    # ------------------------------------------------------------------------------------------------------------------
    # ✅ NEW: Rule-based training diagnostics — fully offline, no AI API required
    # ------------------------------------------------------------------------------------------------------------------
    def analyze_training_results(
            self,
            train_acc: list[float],
            val_acc: list[float],
            train_loss: list[float],
            val_loss: list[float],
            site_name: str = "",
            lr: float = None,
            miou_values: list[float] = None,
            dice_values: list[float] = None,
            plateau_window: int = 5,
            overfit_gap_threshold: float = 0.05,
            underfit_threshold: float = 0.75,
            instability_std_threshold: float = 0.015,
            good_val_acc_threshold: float = 0.90,
    ) -> dict:
        """
        Analyzes training curves and returns a structured diagnostic report.

        Detects:
          - Overfitting (persistent train/val accuracy gap)
          - Underfitting (both curves plateau below acceptable threshold)
          - Early plateau (val accuracy stops improving too soon)
          - Training instability (high variance in val curve)
          - Loss divergence (val loss increasing while train loss decreases)
          - Good convergence (all checks pass)

        Args:
            train_acc:                Per-epoch train accuracy values.
            val_acc:                  Per-epoch validation accuracy values.
            train_loss:               Per-epoch train loss values.
            val_loss:                 Per-epoch validation loss values.
            site_name:                Site identifier for report labeling.
            lr:                       Learning rate used during training.
            miou_values:              Optional per-epoch mean IoU values.
            dice_values:              Optional per-epoch Dice values.
            plateau_window:           Epochs at end of training to check for plateau.
            overfit_gap_threshold:    Train/val gap above which overfitting is flagged.
            underfit_threshold:       Val accuracy below which underfitting is flagged.
            instability_std_threshold: Std dev of val accuracy flagging instability.
            good_val_acc_threshold:   Val accuracy above which convergence is good.

        Returns:
            dict with keys:
              - 'severity':     'good' | 'warning' | 'critical'
              - 'findings':     list of (severity, finding_text) tuples
              - 'suggestions':  list of suggestion strings
              - 'summary':      single human-readable summary string
              - 'report_text':  full formatted report string
        """
        findings = []
        suggestions = []

        if not train_acc or not val_acc or not train_loss or not val_loss:
            return {
                "severity": "warning",
                "findings": [("warning", "Insufficient data to analyze training results.")],
                "suggestions": ["Ensure training ran for at least 2 epochs with validation enabled."],
                "summary": "Insufficient data for analysis.",
                "report_text": "No training data available for analysis."
            }

        tr_acc = np.asarray(train_acc)
        vl_acc = np.asarray(val_acc)
        tr_loss = np.asarray(train_loss)
        vl_loss = np.asarray(val_loss)

        n_epochs = len(vl_acc)
        final_train_acc = float(tr_acc[-1])
        final_val_acc = float(vl_acc[-1])
        final_train_loss = float(tr_loss[-1])
        final_val_loss = float(vl_loss[-1])

        # 1. Overfitting
        if n_epochs >= plateau_window:
            mean_gap = float(np.mean(tr_acc[-plateau_window:] - vl_acc[-plateau_window:]))
        else:
            mean_gap = float(tr_acc[-1] - vl_acc[-1])

        if mean_gap > overfit_gap_threshold * 2:
            findings.append(("critical", f"Severe overfitting: train/val accuracy gap is {mean_gap:.3f} over the final {plateau_window} epochs."))
            suggestions.append("Reduce training epochs or add regularization (increase weight decay or dropout).")
            suggestions.append("Consider using more training images or stronger data augmentation.")
        elif mean_gap > overfit_gap_threshold:
            findings.append(("warning", f"Mild overfitting: train/val accuracy gap is {mean_gap:.3f} over the final {plateau_window} epochs."))
            suggestions.append("If deploying, prefer the checkpoint with the best validation accuracy rather than the final epoch.")

        # 2. Underfitting
        max_val_acc = float(np.max(vl_acc))
        if max_val_acc < underfit_threshold:
            findings.append(("critical", f"Underfitting: peak validation accuracy is only {max_val_acc:.3f} (threshold: {underfit_threshold:.2f})."))
            suggestions.append("Increase the number of training epochs.")
            suggestions.append("Try a higher learning rate — current value may be too conservative.")
            suggestions.append("Check annotation quality: low accuracy may indicate label errors.")
            if lr is not None and lr < 1e-4:
                suggestions.append(f"Learning rate {lr:.5f} may be too low — consider trying 3e-4 or 1e-3.")

        # 3. Early plateau
        if n_epochs >= plateau_window * 2:
            midpoint = n_epochs // 2
            early_max = float(np.max(vl_acc[:midpoint]))
            late_max = float(np.max(vl_acc[midpoint:]))
            if late_max - early_max < 0.005:
                plateau_epoch = int(np.argmax(vl_acc)) + 1
                findings.append(("warning", f"Validation accuracy plateaued at epoch {plateau_epoch} ({early_max:.3f}) with no meaningful improvement in the second half of training."))
                suggestions.append(f"Consider stopping training at epoch {plateau_epoch} to save time.")
                suggestions.append("A learning rate scheduler (e.g., ReduceLROnPlateau) may help break through the plateau.")

        # 4. Instability
        if n_epochs >= 5:
            val_std = float(np.std(vl_acc))
            if val_std > instability_std_threshold:
                findings.append(("warning", f"Unstable validation accuracy detected (std dev: {val_std:.4f})."))
                suggestions.append("Reduce the learning rate — instability often indicates the learning rate is too high.")
                suggestions.append("Increase batch size if GPU memory allows.")

        # 5. Loss divergence
        if n_epochs >= plateau_window:
            train_loss_trend = float(tr_loss[-1]) - float(tr_loss[-plateau_window])
            val_loss_trend = float(vl_loss[-1]) - float(vl_loss[-plateau_window])
            if train_loss_trend < -0.01 and val_loss_trend > 0.01:
                findings.append(("critical", f"Loss divergence: train loss decreasing ({train_loss_trend:.4f}) while val loss increasing ({val_loss_trend:.4f})."))
                suggestions.append("Stop training — the model is memorizing training data.")
                suggestions.append("Use the checkpoint from the epoch with the lowest validation loss.")

        # 6. mIoU check
        if miou_values and len(miou_values) > 0:
            max_miou = float(np.max(miou_values))
            if max_miou < 0.5:
                findings.append(("critical", f"Mean IoU is very low (peak: {max_miou:.3f}). Segmentation quality is poor."))
                suggestions.append("Review mask annotations — very low mIoU may indicate annotation errors or class imbalance.")
            elif max_miou < 0.70:
                findings.append(("warning", f"Mean IoU is moderate (peak: {max_miou:.3f}). Further tuning may improve segmentation quality."))

        # 7. Dice check
        if dice_values and len(dice_values) > 0:
            max_dice = float(np.max(dice_values))
            if max_dice < 0.60:
                findings.append(("critical", f"Dice score is low (peak: {max_dice:.3f}). Segmentation boundary quality is poor."))
                suggestions.append("Consider increasing annotated images, particularly for difficult cases (shadows, glare, low water).")

        # 8. Good convergence fallback
        if not findings:
            findings.append(("good", f"Good convergence: validation accuracy reached {final_val_acc:.3f} with a train/val gap of {mean_gap:.3f}."))

        # Deduplicate suggestions
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)

        # Overall severity
        severities = [f[0] for f in findings]
        if "critical" in severities:
            overall_severity = "critical"
        elif "warning" in severities:
            overall_severity = "warning"
        else:
            overall_severity = "good"

        # Build report text
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("  GRIME AI TRAINING DIAGNOSTIC REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"  Site:             {site_name}")
        if lr is not None:
            report_lines.append(f"  Learning Rate:    {lr:.5f}")
        report_lines.append(f"  Epochs:           {n_epochs}")
        report_lines.append(f"  Final Train Acc:  {final_train_acc:.4f}")
        report_lines.append(f"  Final Val Acc:    {final_val_acc:.4f}")
        report_lines.append(f"  Final Train Loss: {final_train_loss:.4f}")
        report_lines.append(f"  Final Val Loss:   {final_val_loss:.4f}")
        if miou_values:
            report_lines.append(f"  Final mIoU:       {float(miou_values[-1]):.4f}")
        if dice_values:
            report_lines.append(f"  Final Dice:       {float(dice_values[-1]):.4f}")
        report_lines.append("-" * 70)
        report_lines.append(f"  Overall Status:   {overall_severity.upper()}")
        report_lines.append("-" * 70)
        report_lines.append("  FINDINGS:")
        tags = {"good": "[OK]      ", "warning": "[WARNING] ", "critical": "[CRITICAL]"}
        for sev, msg in findings:
            report_lines.append(f"    {tags[sev]} {msg}")
        if unique_suggestions:
            report_lines.append("-" * 70)
            report_lines.append("  SUGGESTIONS:")
            for i, s in enumerate(unique_suggestions, 1):
                report_lines.append(f"    {i}. {s}")
        report_lines.append("=" * 70)
        report_text = "\n".join(report_lines)

        summary = f"[{overall_severity.upper()}] " + " | ".join(f[1] for f in findings[:2])

        return {
            "severity": overall_severity,
            "findings": findings,
            "suggestions": unique_suggestions,
            "summary": summary,
            "report_text": report_text,
        }

    # ------------------------------------------------------------------------------------------------------------------
    # ✅ NEW: Save diagnostic report as a formatted PDF using fpdf2
    # ------------------------------------------------------------------------------------------------------------------
    def save_training_report(
            self,
            train_acc: list[float],
            val_acc: list[float],
            train_loss: list[float],
            val_loss: list[float],
            site_name: str = "",
            lr: float = None,
            miou_values: list[float] = None,
            dice_values: list[float] = None,
            graph_paths: list[str] = None,
            # Model identity
            model_type: str = None,           # e.g. "SAM2" or "SegFormer-LoRA"
            # Dataset info
            num_train_images: int = None,
            num_val_images: int = None,
            # SAM2-specific
            blob_radius_pct: float = None,    # blob filter radius as % of image diagonal
            weight_decay: float = None,        # optimizer weight decay
            batch_size: int = None,            # batch size (SegFormer)
            # Annotation info
            categories: list[dict] = None,    # list of {"id": int, "name": str}
            # Checkpoint info
            best_epoch: int = None,           # epoch with best val accuracy/mIoU
            best_val_acc: float = None,       # val accuracy at best epoch
            early_stopped: bool = False,      # whether early stopping triggered
            early_stop_epoch: int = None,     # epoch at which training stopped early
            training_time_seconds: float = None,  # total wall-clock training time
    ) -> dict:
        """
        Runs analyze_training_results() and saves a formatted PDF report
        in models_folder alongside the training graphs.

        Args:
            train_acc:          Per-epoch train accuracy values.
            val_acc:            Per-epoch validation accuracy values.
            train_loss:         Per-epoch train loss values.
            val_loss:           Per-epoch validation loss values.
            site_name:          Site identifier.
            lr:                 Learning rate used.
            miou_values:        Optional per-epoch mean IoU values.
            dice_values:        Optional per-epoch Dice values.
            graph_paths:        Optional list of PNG paths to embed in the PDF.
            model_type:         Model name string, e.g. "SAM2" or "SegFormer-LoRA".
            num_train_images:   Number of images used for training.
            num_val_images:     Number of images used for validation.
            blob_radius_pct:    SAM2 blob filter radius as % of image diagonal.
            categories:         List of {"id": int, "name": str} annotation categories.
            best_epoch:         Epoch number with best validation performance.
            best_val_acc:       Validation accuracy at best epoch.
            early_stopped:      True if early stopping triggered before max epochs.
            early_stop_epoch:   Epoch at which early stopping triggered.

        Returns:
            dict with keys: severity, findings, suggestions, summary, report_text
        """
        # If training was cancelled before any epoch completed, nothing to report.
        if not train_acc or not val_acc or not train_loss or not val_loss:
            print("[ModelTrainingVisualization] save_training_report: no epoch data — skipping report.")
            return {"severity": "critical", "findings": [], "suggestions": [],
                    "summary": "Training cancelled before first epoch completed.",
                    "report_text": ""}

        result = self.analyze_training_results(
            train_acc=train_acc,
            val_acc=val_acc,
            train_loss=train_loss,
            val_loss=val_loss,
            site_name=site_name,
            lr=lr,
            miou_values=miou_values,
            dice_values=dice_values,
        )

        severity = result["severity"]
        findings = result["findings"]
        suggestions = result["suggestions"]

        severity_colors = {
            "good":     (34,  139, 34),
            "warning":  (210, 140, 0),
            "critical": (180, 30,  30),
        }
        banner_rgb = severity_colors[severity]

        # ------------------------------------------------------------------
        # Build PDF
        # ------------------------------------------------------------------

        def _safe(text: str) -> str:
            """Replace non-latin-1 characters so Helvetica doesn't choke."""
            return text.encode("latin-1", errors="replace").decode("latin-1")

        def _section_header(title: str):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_fill_color(230, 230, 230)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(page_w, 7, _safe(title), border=0, ln=True, fill=True)
            pdf.ln(2)

        def _meta_table(rows: list[tuple[str, str]]):
            col_w = page_w / 2
            for i, (label, value) in enumerate(rows):
                fill = i % 2 == 0
                pdf.set_fill_color(245, 245, 245) if fill else pdf.set_fill_color(255, 255, 255)
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(col_w, 6, _safe(f"  {label}"), border=0, fill=fill)
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(col_w, 6, _safe(str(value)), border=0, ln=True, fill=fill)
            pdf.ln(6)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_margins(15, 15, 15)
        page_w = pdf.w - 30

        # --- Header banner ---
        pdf.set_fill_color(*banner_rgb)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(page_w, 12, _safe("GRIME AI  -  Training Diagnostic Report"), border=0, ln=True, align="C", fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

        # --- Status badge ---
        pdf.set_fill_color(*banner_rgb)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(page_w, 8, _safe(f"Overall Status: {severity.upper()}"), border=0, ln=True, align="C", fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(6)

        # --- Training Summary ---
        _section_header("Training Summary")
        meta_rows = [
            ("Site",               site_name or "N/A"),
            ("Model",              model_type or "N/A"),
            ("Learning Rate",      f"{lr:.5f}" if lr is not None else "N/A"),
            ("Epochs Completed",   str(len(val_acc))),
            ("Final Train Accuracy", f"{train_acc[-1]:.4f}" if train_acc else "N/A"),
            ("Final Val Accuracy",   f"{val_acc[-1]:.4f}"   if val_acc   else "N/A"),
            ("Final Train Loss",     f"{train_loss[-1]:.4f}" if train_loss else "N/A"),
            ("Final Val Loss",       f"{val_loss[-1]:.4f}"   if val_loss  else "N/A"),
        ]
        if miou_values:
            meta_rows.append(("Final mIoU", f"{float(miou_values[-1]):.4f}"))
        if dice_values:
            meta_rows.append(("Final Dice", f"{float(dice_values[-1]):.4f}"))
        if weight_decay is not None:
            meta_rows.append(("Weight Decay", str(weight_decay)))
        if batch_size is not None:
            meta_rows.append(("Batch Size", str(batch_size)))
        if best_epoch is not None:
            meta_rows.append(("Best Checkpoint Epoch", str(best_epoch)))
        if best_val_acc is not None:
            meta_rows.append(("Best Val Accuracy",     f"{best_val_acc:.4f}"))
        if early_stopped:
            stop_ep = str(early_stop_epoch) if early_stop_epoch is not None else "unknown"
            meta_rows.append(("Early Stopping",        f"Triggered at epoch {stop_ep}"))
        if training_time_seconds is not None:
            h = int(training_time_seconds // 3600)
            m = int((training_time_seconds % 3600) // 60)
            s = int(training_time_seconds % 60)
            if h > 0:
                time_str = f"{h}h {m}m {s}s"
            elif m > 0:
                time_str = f"{m}m {s}s"
            else:
                time_str = f"{s}s"
            meta_rows.append(("Training Time", time_str))
        _meta_table(meta_rows)

        # --- Dataset Info ---
        if num_train_images is not None or num_val_images is not None:
            _section_header("Dataset")
            ds_rows = []
            if num_train_images is not None:
                ds_rows.append(("Training Images", str(num_train_images)))
            if num_val_images is not None:
                ds_rows.append(("Validation Images", str(num_val_images)))
            _meta_table(ds_rows)

        # --- Model Parameters (SAM2-specific) ---
        if blob_radius_pct is not None:
            _section_header("Model Parameters")
            param_rows = [("Blob Filter Radius", f"{blob_radius_pct:.1f}% of image diagonal")]

            _meta_table(param_rows)

        # --- Annotation Labels ---
        cats_to_show = categories if categories is not None else self.categories
        if cats_to_show:
            _section_header("Annotation Categories")
            sorted_cats = sorted(cats_to_show, key=lambda c: c["id"])
            label_rows = [(f"ID {c['id']}", c["name"]) for c in sorted_cats]
            label_rows.insert(0, ("Total Categories", str(len(sorted_cats))))
            _meta_table(label_rows)

        # --- Findings ---
        _section_header("Findings")
        finding_colors = {
            "good":     (34,  139, 34),
            "warning":  (210, 140, 0),
            "critical": (180, 30,  30),
        }
        badge_labels = {"good": "OK", "warning": "WARNING", "critical": "CRITICAL"}

        for sev, msg in findings:
            r, g, b = finding_colors[sev]
            badge_w = 22
            pdf.set_fill_color(r, g, b)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(badge_w, 7, _safe(badge_labels[sev]), border=0, fill=True, align="C")
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(page_w - badge_w, 7, _safe(f"  {msg}"), border=0)
            pdf.ln(1)
        pdf.ln(4)

        # --- Suggestions ---
        if suggestions:
            _section_header("Suggestions")
            pdf.set_font("Helvetica", "", 10)
            for i, s in enumerate(suggestions, 1):
                pdf.set_x(15)
                pdf.multi_cell(page_w, 6, _safe(f"  {i}.  {s}"), border=0)
                pdf.ln(1)
            pdf.ln(4)

        # --- Embedded graphs ---
        if graph_paths:
            valid_graphs = [p for p in graph_paths if p and os.path.isfile(p)]
            if valid_graphs:
                _section_header("Training Graphs")
                pdf.ln(2)
                for img_path in valid_graphs:
                    img_w = page_w
                    img_h = img_w * 0.67
                    if pdf.get_y() + img_h > pdf.h - 20:
                        pdf.add_page()
                    pdf.image(img_path, x=15, w=img_w, h=img_h)
                    pdf.ln(6)

        # --- Footer ---
        pdf.set_y(-15)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(page_w, 5, _safe(
            f"Generated by GRIME AI  |  {self.formatted_time}  |  Site: {site_name}"
            + (f"  |  Model: {model_type}" if model_type else "")
        ), align="C")

        # --- Save PDF ---
        lr_str = f"_lr{lr:.5f}" if lr is not None else ""
        pdf_filename = f"{self.formatted_time}_{site_name}_Training_Report{lr_str}.pdf"
        pdf_path = os.path.join(self.models_folder, pdf_filename)

        try:
            pdf.output(pdf_path)
            print(f"Training report saved to {pdf_path}")
        except OSError as e:
            print(f"[save_training_report] Could not write PDF: {e}")

        return result
