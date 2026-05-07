#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

"""
yolo_trainer.py

YOLOv11-seg trainer for GRIME AI.  Mirrors the SAM2Trainer / SegFormerTrainer
class structure exactly:
  - Reads site_config.json for all hyperparameters (including val_split)
  - Converts COCO JSON annotations → YOLO flat-file format internally
  - Writes all outputs (weights, YOLO run artefacts, GRIME AI graphs, PDF
    report) to  <models_root>/yolo/<timestamp>_<site>/
  - Generates the same 9 ModelTrainingVisualization plots + PDF diagnostic
    report as SAM2 and SegFormer
  - Adds a QProgressWheel for graph generation, matching the other trainers
"""

import os
import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from GRIME_AI.GRIME_AI_QProgressWheel import QProgressWheel
from GRIME_AI.GRIME_AI_Save_Utils import GRIME_AI_Save_Utils
from GRIME_AI.GRIME_AI_QMessageBox import GRIME_AI_QMessageBox
from GRIME_AI.dialogs.ML_image_processing.model_config_manager import ModelConfigManager
from GRIME_AI.ml_core.model_training_visualization import ModelTrainingVisualization


# ============================================================================
# ============================================================================
# ===                     COCO → YOLO CONVERSION                          ===
# ============================================================================
# ============================================================================

def coco_segmentation_to_yolo(
    segmentation: List[List[float]],
    image_w: int,
    image_h: int,
) -> Optional[List[float]]:
    """
    Convert a COCO polygon segmentation to a YOLO normalised flat list.

    COCO stores polygons as [[x0,y0,x1,y1,...], ...].
    YOLO expects a single flat list [x0/W, y0/H, x1/W, y1/H, ...].
    Multiple polygons are merged by concatenation (YOLO supports multi-polygon
    segments in a single line — each polygon is terminated and a new one begins).

    Returns None if the segmentation is empty or unparseable.
    """
    if not segmentation or not isinstance(segmentation, list):
        return None

    yolo_flat: List[float] = []
    for poly in segmentation:
        if not poly or len(poly) < 6:
            continue
        coords = list(poly)
        if len(coords) % 2 != 0:
            coords = coords[:-1]
        norm = []
        for i in range(0, len(coords), 2):
            x = max(0.0, min(1.0, coords[i] / image_w))
            y = max(0.0, min(1.0, coords[i + 1] / image_h))
            norm.extend([x, y])
        yolo_flat.extend(norm)

    return yolo_flat if yolo_flat else None


def _coco_to_yolo_label(ann_path: str, image_src_dir: str,
                         labels_out_dir: str, cat_id_to_yolo: Dict[int, int]) -> List[str]:
    """
    Convert a single COCO JSON to YOLO .txt label files.
    Writes one .txt per image into labels_out_dir.
    Returns list of image stems successfully converted.
    """
    labels_out = Path(labels_out_dir)
    labels_out.mkdir(parents=True, exist_ok=True)

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    print(f"[COCO->YOLO] cat_id_to_yolo mapping: {cat_id_to_yolo}")
    print(f"[COCO->YOLO] JSON categories: {coco.get('categories', [])}")
    print(f"[COCO->YOLO] JSON images: {len(coco.get('images', []))}, annotations: {len(coco.get('annotations', []))}")

    id_to_image = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image: Dict[int, List[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    stems = []
    for img_id, img_info in id_to_image.items():
        fname    = img_info["file_name"]
        src_path = Path(image_src_dir) / fname
        if not src_path.exists():
            continue

        w = img_info.get("width", 0)
        h = img_info.get("height", 0)
        if w == 0 or h == 0:
            try:
                with Image.open(src_path) as im:
                    w, h = im.size
            except Exception:
                continue

        label_lines = []
        for ann in anns_by_image.get(img_id, []):
            yolo_class = cat_id_to_yolo.get(ann.get("category_id"))
            if yolo_class is None:
                continue
            seg = ann.get("segmentation", [])
            if not seg or not isinstance(seg, list):
                continue
            coords = coco_segmentation_to_yolo(seg, w, h)
            if coords:
                label_lines.append(
                    f"{yolo_class} " + " ".join(f"{v:.6f}" for v in coords)
                )

        stem = Path(fname).stem
        label_path = labels_out / f"{stem}.txt"
        with open(label_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(label_lines))
        stems.append(stem)

    non_empty = sum(1 for s in stems if (labels_out / f"{s}.txt").stat().st_size > 0)
    print(f"[COCO->YOLO] {ann_path}: {len(stems)} images, {non_empty} with annotations")
    return stems


def build_yolo_dataset(
    image_dirs: List[str],
    annotation_paths: List[str],
    output_root: str,
    categories: List[Dict[str, Any]],
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[str, str, Dict[int, int], int, int]:
    """
    Build a YOLO segmentation dataset without copying or symlinking images.

    Labels (.txt) are written to:
        dataset/labels/train/
        dataset/labels/val/

    The dataset.yaml points train/val directly at the original source image
    directories so no image duplication or symlinking occurs — safe for
    datasets of any size.

    Returns:
        (yaml_path, dataset_root, category_id_to_yolo_class, n_train, n_val)
    """
    dataset_root = Path(output_root) / "dataset"

    sorted_cats    = sorted(categories, key=lambda c: c["id"])
    cat_id_to_yolo = {cat["id"]: idx for idx, cat in enumerate(sorted_cats)}
    yolo_names     = [cat["name"] for cat in sorted_cats]

    # ── Step 1: Convert COCO annotations to YOLO labels in a staging area
    staging_labels = dataset_root / "_staging_labels"

    all_entries: List[Tuple[Path, Path]] = []  # (image_path, label_path)
    seen_stems: set = set()

    for img_dir, ann_path in zip(image_dirs, annotation_paths):
        img_dir  = Path(img_dir)
        ann_path = Path(ann_path)
        if not ann_path.exists():
            print(f"[YOLO dataset] Annotation not found, skipping: {ann_path}")
            continue

        # Delete stale YOLO cache files
        for cache in list(img_dir.glob("*.cache")) + \
                     list(img_dir.parent.glob(f"{img_dir.name}.cache")):
            try:
                cache.unlink()
                print(f"[YOLO dataset] Deleted stale cache: {cache}")
            except Exception:
                pass

        stems = _coco_to_yolo_label(
            str(ann_path), str(img_dir), str(staging_labels), cat_id_to_yolo
        )

        for stem in stems:
            if stem in seen_stems:
                continue
            label_file = staging_labels / f"{stem}.txt"
            img_file   = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                candidate = img_dir / (stem + ext)
                if candidate.exists():
                    img_file = candidate
                    break
            if img_file is None or not label_file.exists():
                continue
            seen_stems.add(stem)
            all_entries.append((img_file.resolve(), label_file.resolve()))

    if not all_entries:
        shutil.rmtree(staging_labels, ignore_errors=True)
        raise RuntimeError("[YOLO dataset] No images were successfully converted.")

    print(f"[YOLO dataset] Total: {len(all_entries)} image/label pairs.")

    # ── Step 2: Shuffle + split
    rng = random.Random(seed)
    rng.shuffle(all_entries)
    n_val         = max(1, int(len(all_entries) * val_split))
    val_entries   = all_entries[:n_val]
    train_entries = all_entries[n_val:]

    # ── Step 3: Write label files into labels/{train,val}
    # Images stay in their original locations — no copying or symlinking.
    # YOLO resolves label paths by replacing the images/ segment with labels/,
    # but since we're using absolute image paths in the YAML we must place
    # labels where YOLO will look: alongside each image in a sibling
    # 'labels' folder, or we use a flat labels dir per split and list
    # image paths explicitly.
    #
    # Simplest approach that works with any Ultralytics version:
    # Write one text file per split listing absolute image paths,
    # and place the matching label alongside each image in labels/{train,val}.

    train_img_list = dataset_root / "train_images.txt"
    val_img_list   = dataset_root / "val_images.txt"

    for split_name, entries, img_list_path in [
        ("train", train_entries, train_img_list),
        ("val",   val_entries,   val_img_list),
    ]:
        lbl_out = dataset_root / "labels" / split_name
        lbl_out.mkdir(parents=True, exist_ok=True)

        with open(img_list_path, "w", encoding="utf-8") as flist:
            for img_path, lbl_path in entries:
                # Write image path to list file
                flist.write(str(img_path).replace("\\", "/") + "\n")
                # Copy label to labels/split/ (labels are small generated files)
                dst_lbl = lbl_out / lbl_path.name
                if not dst_lbl.exists():
                    shutil.copy2(lbl_path, dst_lbl)

    shutil.rmtree(staging_labels, ignore_errors=True)

    # ── Step 4: Write dataset.yaml
    # Use the image list .txt files as train/val — Ultralytics accepts these.
    yaml_path = dataset_root / "dataset.yaml"
    train_str = str(train_img_list.resolve()).replace("\\", "/")
    val_str   = str(val_img_list.resolve()).replace("\\", "/")
    names_lines = "\n".join(f"  {i}: {n}" for i, n in enumerate(yolo_names))
    yaml_content = (
        f"train: {train_str}\n"
        f"val:   {val_str}\n"
        f"\n"
        f"nc: {len(yolo_names)}\n"
        f"names:\n"
        f"{names_lines}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    n_train = len(train_entries)
    n_val_c = len(val_entries)
    print(f"[YOLO dataset] Train: {n_train}  Val: {n_val_c}")
    print(f"[YOLO dataset] YAML: {yaml_path}")
    return str(yaml_path), str(dataset_root), cat_id_to_yolo, n_train, n_val_c



from PyQt5.QtCore import QThread, pyqtSignal


# ============================================================================
# ============================================================================
# ===                      class YOLOTrainWorker                          ===
# ============================================================================
# ============================================================================
class YOLOTrainWorker(QThread):
    """
    Runs YOLO model.train() in a background thread so the Qt event loop
    stays alive, allowing the progress wheel close button to work.

    Signals:
        epoch_end(int, float)      — epoch number, loss value
        batch_end(int, int, float) — epoch number, batch index, loss value
        finished()                 — training completed normally
        error(str)                 — training raised an exception
    """
    epoch_end = pyqtSignal(int, float)
    batch_end = pyqtSignal(int, int, float)
    finished  = pyqtSignal()
    error     = pyqtSignal(str)

    def __init__(self, model, train_kwargs: dict):
        super().__init__()
        self.model = model
        self.train_kwargs = train_kwargs
        self._stop_requested = False

    def request_stop(self):
        """Signal the worker to stop after the current epoch."""
        self._stop_requested = True
        try:
            if hasattr(self.model, "trainer") and self.model.trainer is not None:
                self.model.trainer.stop = True
        except Exception:
            pass

    def run(self):
        try:
            def _on_epoch_end(trainer):
                epoch = getattr(trainer, "epoch", 0) + 1
                loss  = getattr(trainer, "loss",  0.0)
                if hasattr(loss, "item"):
                    loss = loss.item()
                self.epoch_end.emit(epoch, float(loss))
                if self._stop_requested:
                    trainer.stop = True

            def _on_batch_end(trainer):
                epoch      = getattr(trainer, "epoch", 0) + 1
                batch_idx  = getattr(trainer, "batch", 0)
                loss       = getattr(trainer, "loss",  0.0)
                if hasattr(loss, "item"):
                    loss = loss.item()
                self.batch_end.emit(epoch, int(batch_idx), float(loss))

            self.model.add_callback("on_train_epoch_end", _on_epoch_end)
            self.model.add_callback("on_train_batch_end", _on_batch_end)
            self.model.train(**self.train_kwargs)
            self.finished.emit()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")



class YOLOTrainer:
    """
    Fine-tunes YOLOv11-seg on GRIME AI COCO-annotated data.

    Mirrors SAM2Trainer / SegFormerTrainer:
      - Reads all hyperparameters from site_config.json
      - Converts COCO → YOLO format internally
      - Writes outputs to <models_root>/yolo/<timestamp>_<site>/
      - Generates the same 9 graphs + PDF report via ModelTrainingVisualization
    """

    YOLO_MODEL_BASE = "yolo11x-seg.pt"  # extra-large seg model (recommended for 32GB+ VRAM)

    def __init__(self, cfg=None, parent_widget=None):
        self.parent_widget = parent_widget
        self.progress_bar_closed = False

        self.now = datetime.now()
        self.formatted_time = self.now.strftime("%Y%m%d_%H%M%S")

        # ── Metric accumulators (mirrors SegFormerTrainer exactly) ──────────
        self.epoch_list: List[int] = []
        self.loss_values: List[float] = []
        self.train_accuracy_values: List[float] = []
        self.val_loss_values: List[float] = []
        self.val_accuracy_values: List[float] = []
        self.val_iou_values: List[float] = []
        self.miou_values: List[float] = []
        self.val_dice_values: List[float] = []
        self.val_true_list: List[int] = []
        self.val_pred_list: List[int] = []
        self.val_score_list: List[float] = []

        # ── Load site_config ─────────────────────────────────────────────────
        settings_folder = GRIME_AI_Save_Utils().get_settings_folder()
        config_file = Path(settings_folder) / "site_config.json"
        mgr = ModelConfigManager(str(config_file))
        self.site_config: Dict[str, Any] = mgr.load_config(return_type="dict")

        self.site_name: str = self.site_config["siteName"]
        self.learning_rates: List[float] = self.site_config["learningRates"]
        self.weight_decay: float = float(self.site_config.get("weight_decay", 0.01))
        self.num_epochs: int = int(self.site_config.get("number_of_epochs", 20))
        self.batch_size: int = int(self.site_config.get("batch_size", 8))
        self.early_stopping: bool = bool(self.site_config.get("early_stopping", False))
        self.patience: int = int(self.site_config.get("patience", 10))
        self.val_split: float = float(self.site_config.get("val_split", 0.2))
        self.val_split = max(0.10, min(0.40, self.val_split))

        self.categories: List[Dict[str, Any]] = []
        self.model_output_folder: Optional[str] = None
        self.num_train_images: int = 0
        self.num_val_images: int = 0
        self.cat_id_to_yolo: Dict[int, int] = {}
        self._resolved_yolo_base: str = self.YOLO_MODEL_BASE  # updated after weights resolve

        # ── Create output folder ─────────────────────────────────────────────
        try:
            self.model_output_folder = os.path.join(
                GRIME_AI_Save_Utils().get_models_folder(), "yolo",
                f"{self.formatted_time}_{self.site_name}"
            )
            os.makedirs(self.model_output_folder, exist_ok=True)
        except OSError as e:
            print(f"[YOLOTrainer] Error creating output folder: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    def run_training_pipeline(self):
        """
        Main entry point — mirrors SAM2Trainer.run_training_pipeline().
        """
        import time as _time
        self.training_time_seconds = None
        # ── Collect folders / annotations from site_config ──────────────────
        all_folders: List[str] = []
        all_annotations: List[str] = []

        for path_entry in self.site_config.get("Path", []):
            dp = path_entry.get("directoryPaths", {})
            folders = dp.get("folders", [])
            annotations = dp.get("annotations", [])
            if isinstance(folders, str):
                folders = [folders]
            if isinstance(annotations, str):
                annotations = [annotations]
            all_folders.extend(folders)
            all_annotations.extend(annotations)

        if not all_folders:
            print("[YOLOTrainer] No training folders configured.")
            return

        # ── Build categories ─────────────────────────────────────────────────
        self.categories = self._build_unique_categories(all_annotations)
        if not self.categories:
            print("[YOLOTrainer] No categories found in annotations.")
            return

        # ── Convert COCO → YOLO dataset ──────────────────────────────────────
        print("[YOLOTrainer] Converting COCO annotations to YOLO format...")
        print(f"[YOLOTrainer] Folders: {all_folders}")
        print(f"[YOLOTrainer] Annotations: {all_annotations}")
        print(f"[YOLOTrainer] Categories: {self.categories}")
        yaml_path, dataset_root_str, cat_id_to_yolo, n_train, n_val = build_yolo_dataset(
            image_dirs=all_folders,
            annotation_paths=all_annotations,
            output_root=self.model_output_folder,
            categories=self.categories,
            val_split=self.val_split,
            seed=42,
        )
        self.cat_id_to_yolo = cat_id_to_yolo  # COCO ID → YOLO 0-based index
        self.num_train_images = n_train
        self.num_val_images   = n_val

        _training_start = _time.perf_counter()
        for lr in self.learning_rates:
            print(f"\n[YOLOTrainer] Training with lr={lr}")
            self._train_one_lr(yaml_path, dataset_root_str, lr)
        self.training_time_seconds = _time.perf_counter() - _training_start

        for lr in self.learning_rates:
            self._plot_training_graphs(lr=lr)

    # ─────────────────────────────────────────────────────────────────────────
    def _train_one_lr(self, yaml_path: str, dataset_root_str: str, lr: float):
        """
        Run YOLO training for a single learning rate in a background QThread,
        keeping the Qt event loop alive so the progress wheel close button works.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )

        # PyTorch 2.6 weights_only=True compatibility patch
        try:
            import ultralytics.nn.tasks as _ult_tasks
            import torch as _torch

            def _patched_torch_safe_load(weight):
                ckpt = _torch.load(weight, map_location="cpu", weights_only=False)
                return ckpt, weight

            _ult_tasks.torch_safe_load = _patched_torch_safe_load
        except Exception as e:
            print(f"[YOLOTrainer] Could not apply torch_safe_load patch: {e}")

        # Resolve base weights from ultralytics/assets/
        # yolo_base_weights stores just the filename (e.g. "yolo11l-seg.pt")
        weights_filename = self.site_config.get("yolo_base_weights", "") or self.YOLO_MODEL_BASE
        if os.path.sep in weights_filename or weights_filename.startswith("/"):
            # Legacy: full path stored — use as-is if it exists
            weights_path = weights_filename if os.path.isfile(weights_filename) else ""
        else:
            weights_path = ""

        if not weights_path:
            # Resolve against ultralytics/assets/ — same location as SAM2 uses site-packages
            try:
                import importlib.util
                spec = importlib.util.find_spec("ultralytics")
                if spec and spec.origin:
                    assets_dir = os.path.join(os.path.dirname(spec.origin), "assets")
                    candidate = os.path.join(assets_dir, weights_filename)
                    if os.path.isfile(candidate):
                        weights_path = candidate
                        print(f"[YOLOTrainer] Found weights in ultralytics/assets/: {candidate}")
            except Exception as e:
                print(f"[YOLOTrainer] Could not resolve ultralytics/assets/: {e}")

        if not weights_path:
            raise FileNotFoundError(
                f"[YOLOTrainer] YOLO base weights '{weights_filename}' not found.\n"
                f"Run: python -m GRIME_AI.ml_core.download_yolo_weights --default\n"
                f"Or place the file manually in the ultralytics/assets/ folder of your venv."
            )

        print(f"[YOLOTrainer] Loading base weights: {weights_path}")
        self._resolved_yolo_base = Path(weights_path).name  # e.g. "yolo11l-seg.pt"

        # Free any GPU memory held by previous models in this session
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Reset metric lists
        for attr in ("epoch_list", "loss_values", "train_accuracy_values",
                     "val_loss_values", "val_accuracy_values", "val_iou_values",
                     "miou_values", "val_dice_values", "val_true_list",
                     "val_pred_list", "val_score_list"):
            getattr(self, attr).clear()

        lr_tag   = f"lr{lr:.5f}"
        run_name = f"{self.formatted_time}_{self.site_name}_{lr_tag}"

        model = YOLO(weights_path)

        # ── Progress wheel ────────────────────────────────────────────────────
        progressBar = QProgressWheel(
            title=f"YOLOv11-seg Training — {self.site_name}  (lr={lr})",
            total=self.num_epochs,
            on_close=lambda: worker.request_stop(),
            parent=self.parent_widget,
        )
        progressBar.show()

        # ── Build train kwargs ────────────────────────────────────────────────
        patience_arg = self.patience if self.early_stopping else self.num_epochs
        train_kwargs = dict(
            data=yaml_path,
            epochs=self.num_epochs,
            batch=self.batch_size,
            lr0=lr,
            weight_decay=self.weight_decay,
            patience=patience_arg,
            project=self.model_output_folder,
            name=run_name,
            exist_ok=True,
            task="segment",
            verbose=True,
            seed=42,
            deterministic=True,
            save=True,
            save_period=max(1, self.site_config.get("save_model_frequency", 5)),
            val=True,
            plots=True,
            amp=False,
            workers=0,
            imgsz=640,            # 640 uses ~4x less VRAM than 1280
            device=0 if torch.cuda.is_available() else "cpu",
        )

        # ── Launch worker thread ──────────────────────────────────────────────
        worker = YOLOTrainWorker(model, train_kwargs)

        # Estimate batches per epoch for title display
        _n_train = getattr(self, "num_train_images", 1) or 1
        _batches_per_epoch = max(1, math.ceil(_n_train / max(1, self.batch_size)))

        def _on_epoch(epoch, loss):
            progressBar.setWindowTitle(
                f"YOLOv11-seg — {self.site_name}  "
                f"Epoch {epoch}/{self.num_epochs}  loss={loss:.4f}"
            )
            progressBar.setValue(epoch)
            progressBar.show()

        def _on_batch(epoch, batch_idx, loss):
            progressBar.setWindowTitle(
                f"YOLOv11-seg — {self.site_name}  "
                f"Epoch {epoch}/{self.num_epochs}  "
                f"Batch {batch_idx + 1}/{_batches_per_epoch}  "
                f"loss={loss:.4f}"
            )

        def _on_finished():
            progressBar.close()

        def _on_error(msg):
            progressBar.close()
            print(f"[YOLOTrainer] Training error:\n{msg}")

        worker.epoch_end.connect(_on_epoch)
        worker.batch_end.connect(_on_batch)
        worker.finished.connect(_on_finished)
        worker.error.connect(_on_error)

        worker.start()

        # Block here while keeping the Qt event loop alive so the UI responds
        from PyQt5.QtWidgets import QApplication
        while worker.isRunning():
            QApplication.processEvents()
            worker.wait(100)   # wait up to 100 ms then re-check

        # If user closed the wheel, give the worker a moment to finish cleanly
        if not worker.isFinished():
            worker.wait(5000)

        run_dir      = Path(self.model_output_folder) / run_name
        results_csv  = run_dir / "results.csv"

        if self.progress_bar_closed:
            print("[YOLOTrainer] Training cancelled by user.")
            self._cleanup_dataset(dataset_root_str, yaml_path)
            return

        run_dir = Path(self.model_output_folder) / run_name
        results_csv = run_dir / "results.csv"

        # ── Parse YOLO results.csv into GRIME AI metric lists ─────────────────
        if results_csv.exists():
            self._parse_results_csv(results_csv)
        else:
            print(f"[YOLOTrainer] results.csv not found at {results_csv}; "
                  "graphs will be skipped.")

        # ── Post-training: build pixel-level lists for ROC/PR/CM ─────────────
        best_weights = run_dir / "weights" / "best.pt"
        if best_weights.exists() and self.num_val_images > 0:
            self._extract_val_pixel_stats(
                model_path=str(best_weights),
                yaml_path=yaml_path,
                run_dir=str(run_dir),
            )

        # ── Save GRIME AI .torch checkpoint embedding metadata ────────────────
        if best_weights.exists():
            self._save_grime_ai_checkpoint(best_weights, lr)

        print(f"[YOLOTrainer] Run complete. Outputs in: {run_dir}")

        self._cleanup_dataset(dataset_root_str, yaml_path)

    # ─────────────────────────────────────────────────────────────────────────
    def _save_grime_ai_checkpoint(self, best_pt_path: Path, lr: float):
        """
        Save a GRIME AI .torch file alongside best.pt.

        The .torch file contains only GRIME AI metadata — categories, site info,
        training stats, and the path to best.pt. The inference engine loads
        best.pt directly via YOLO() since it is the complete fine-tuned model.
        """
        try:
            print(f"[YOLOTrainer] Saving GRIME AI checkpoint...")

            # Extract class names from best.pt
            class_names = {}
            try:
                raw = torch.load(str(best_pt_path), map_location="cpu", weights_only=False)
                if isinstance(raw, dict) and "model" in raw:
                    ul_model = raw["model"]
                    if hasattr(ul_model, "names"):
                        class_names = ul_model.names
            except Exception as e:
                print(f"[YOLOTrainer] Could not extract class_names from best.pt: {e}")

            val_loss     = float(min(self.val_loss_values))     if self.val_loss_values     else None
            val_accuracy = float(max(self.val_accuracy_values)) if self.val_accuracy_values else None
            miou         = float(max(self.miou_values))         if self.miou_values         else None
            epochs_run   = len(self.epoch_list)

            from datetime import datetime as _dt
            timestamp = _dt.utcnow().strftime("%Y%m%d_%H%M%S")

            ckpt = {
                # ── GRIME AI metadata ─────────────────────────────────────────
                "base_model":       "yolo",
                "best_pt_path":     str(best_pt_path),   # inference loads this directly
                "categories":       self.categories,
                "creation_UTC":     timestamp,
                "site_name":        self.site_name,
                "learning_rate":    lr,
                "epochs":           epochs_run,
                "num_classes":      len(self.categories),
                "val_loss":         val_loss,
                "val_accuracy":     val_accuracy,
                "miou":             miou,
                # ── YOLO-specific ─────────────────────────────────────────────
                "yolo_model_base":  getattr(self, "_resolved_yolo_base", self.YOLO_MODEL_BASE),
                "batch_size":       self.batch_size,
                "class_names":      class_names,
                "cat_id_to_yolo":   getattr(self, "cat_id_to_yolo", {}),
            }

            torch_filename = (
                f"{self.formatted_time}_{self.site_name}"
                f"_ep{epochs_run:03d}_lr{lr:.5f}.torch"
            )
            torch_path = best_pt_path.parent / torch_filename
            torch.save(ckpt, str(torch_path))
            print(f"[YOLOTrainer] GRIME AI checkpoint saved: {torch_path.name}")

        except Exception as e:
            print(f"[YOLOTrainer] Warning: could not save GRIME AI checkpoint: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    def _cleanup_dataset(self, dataset_root_str: str, yaml_path: str):
        """Remove the dataset/ folder after training. Source images are never touched."""
        try:
            target = Path(dataset_root_str)
            if target.exists():
                shutil.rmtree(target)
                print(f"[YOLOTrainer] Removed dataset folder: {target}")
        except Exception as e:
            print(f"[YOLOTrainer] Warning: could not remove dataset folder: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    def _parse_results_csv(self, csv_path: Path):
        """
        Read YOLO's results.csv and populate metric accumulators.

        YOLO results.csv columns (YOLOv11-seg):
          epoch, train/seg_loss, train/box_loss, train/cls_loss, train/dfl_loss,
          metrics/precision(M), metrics/recall(M), metrics/mAP50(M), metrics/mAP50-95(M),
          val/seg_loss, val/box_loss, val/cls_loss, val/dfl_loss,
          lr/pg0, lr/pg1, lr/pg2
        Column names vary slightly by version — we match flexibly.
        """
        import csv

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Strip whitespace from header keys
            rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]

        if not rows:
            print("[YOLOTrainer] results.csv is empty — no graphs will be generated.")
            return

        # Print actual column names so we can verify the mapping matches
        print(f"[YOLOTrainer] results.csv columns: {list(rows[0].keys())}")

        def _get(row: dict, *candidates, default=0.0) -> float:
            for key in candidates:
                if key in row:
                    try:
                        return float(row[key])
                    except (ValueError, TypeError):
                        pass
            return default

        for i, row in enumerate(rows):
            epoch = i + 1
            self.epoch_list.append(epoch)

            # Training loss: primary seg loss + box loss
            train_seg = _get(row, "train/seg_loss", "train/box_loss")
            train_box = _get(row, "train/box_loss", 0.0)
            self.loss_values.append(train_seg + train_box)

            # Val loss
            val_seg = _get(row, "val/seg_loss", "val/box_loss")
            val_box = _get(row, "val/box_loss", 0.0)
            self.val_loss_values.append(val_seg + val_box)

            # Precision → train accuracy proxy
            prec = _get(row, "metrics/precision(M)", "metrics/precision(B)")
            self.train_accuracy_values.append(prec)

            # Recall → val accuracy proxy
            rec = _get(row, "metrics/recall(M)", "metrics/recall(B)")
            self.val_accuracy_values.append(rec)

            # mAP50 → IoU proxy
            map50 = _get(row, "metrics/mAP50(M)", "metrics/mAP50(B)")
            self.val_iou_values.append(map50)
            self.miou_values.append(map50)

            # mAP50-95 → Dice proxy (best available continuous quality metric)
            map5095 = _get(row, "metrics/mAP50-95(M)", "metrics/mAP50-95(B)")
            self.val_dice_values.append(map5095)

        print(f"[YOLOTrainer] Parsed {len(self.epoch_list)} epochs from results.csv")
        print(f"  train_loss non-zero: {any(v != 0.0 for v in self.loss_values)}")
        print(f"  val_iou non-zero:    {any(v != 0.0 for v in self.val_iou_values)}")

    # ─────────────────────────────────────────────────────────────────────────
    def _extract_val_pixel_stats(
        self,
        model_path: str,
        yaml_path: str,
        run_dir: str,
    ):
        """
        Run inference on validation images with the best checkpoint and collect
        pixel-level true/pred/score lists for ROC, PR, and confusion-matrix plots.

        This mirrors the val_true_list / val_pred_list / val_score_list pattern
        used by SAM2 and SegFormer.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            return

        val_images_dir = Path(yaml_path).parent / "images" / "val"
        val_labels_dir = Path(yaml_path).parent / "labels" / "val"

        # With the new no-copy dataset layout, images stay in their original
        # locations and val_images.txt lists their absolute paths.
        val_txt = Path(yaml_path).parent / "val_images.txt"
        if val_txt.exists():
            with open(val_txt, encoding="utf-8") as f:
                image_files = [Path(l.strip()) for l in f if l.strip()]
            image_files = [p for p in image_files if p.exists()]
        elif val_images_dir.exists():
            # Legacy fallback: symlink/copy layout
            image_files = (
                list(val_images_dir.glob("*.jpg")) +
                list(val_images_dir.glob("*.jpeg")) +
                list(val_images_dir.glob("*.png"))
            )
        else:
            image_files = []

        if not image_files:
            print("[YOLOTrainer] No val images found — skipping pixel stats.")
            return

        model = YOLO(model_path)
        model.to(0 if torch.cuda.is_available() else "cpu")

        MAX_PIXEL_SAMPLES = 10_000
        rng = random.Random(42)

        val_overlay_dir = Path(run_dir) / "validation_overlays"
        val_overlay_dir.mkdir(exist_ok=True)
        overlay_count = 0

        for img_path in image_files:
            stem = img_path.stem
            label_path = val_labels_dir / f"{stem}.txt"

            # ── Load ground truth mask from YOLO label ──────────────────────
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            gt_mask = np.zeros((h, w), dtype=np.uint8)

            if label_path.exists():
                with open(label_path, "r") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        coords = [float(p) for p in parts[1:]]
                        pts = []
                        for j in range(0, len(coords), 2):
                            px = int(round(coords[j] * w))
                            py = int(round(coords[j + 1] * h))
                            pts.append([px, py])
                        if pts:
                            poly = np.array(pts, dtype=np.int32)
                            cv2.fillPoly(gt_mask, [poly], 1)

            # ── Run YOLO inference ───────────────────────────────────────────
            result = model(str(img_path), verbose=False)[0]

            pred_mask = np.zeros((h, w), dtype=np.uint8)
            conf_map = np.zeros((h, w), dtype=np.float32)

            if result.masks is not None:
                for seg_mask, conf in zip(
                    result.masks.data.cpu().numpy(),
                    result.boxes.conf.cpu().numpy() if result.boxes is not None else []
                ):
                    seg_resized = cv2.resize(
                        seg_mask.astype(np.float32), (w, h),
                        interpolation=cv2.INTER_LINEAR
                    )
                    binary = (seg_resized > 0.5).astype(np.uint8)
                    pred_mask = np.maximum(pred_mask, binary)
                    conf_map = np.maximum(conf_map, seg_resized * float(conf))

            # ── Save validation overlay (first 5 images) ─────────────────────
            if overlay_count < 5:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                overlay = img_rgb.copy().astype(np.float32) / 255.0
                overlay[pred_mask == 1] = [0, 1, 0]     # TP regions green
                fp_mask = (pred_mask == 1) & (gt_mask == 0)
                fn_mask = (pred_mask == 0) & (gt_mask == 1)
                overlay[fp_mask] = [1, 0, 0]             # FP red
                overlay[fn_mask] = [0, 1, 1]             # FN cyan
                blended = (0.6 * (img_rgb / 255.0) + 0.4 * overlay)
                blended = np.clip(blended, 0, 1)

                plt.figure(figsize=(10, 10))
                plt.imshow(blended)
                plt.title(
                    f"Validation Overlay — {stem}\n"
                    "(Green=TP  Red=FP  Cyan=FN)",
                    fontsize=10
                )
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(
                    val_overlay_dir / f"val_{overlay_count:02d}_{stem}.png",
                    bbox_inches="tight", pad_inches=0.05
                )
                plt.close()
                overlay_count += 1

            # ── Pixel sampling ───────────────────────────────────────────────
            true_flat = gt_mask.flatten().astype(int)
            pred_flat = pred_mask.flatten().astype(int)
            score_flat = conf_map.flatten().astype(float)

            n = len(true_flat)
            if n > MAX_PIXEL_SAMPLES:
                idx = rng.sample(range(n), MAX_PIXEL_SAMPLES)
                true_flat = true_flat[idx]
                pred_flat = pred_flat[idx]
                score_flat = score_flat[idx]

            self.val_true_list.extend(true_flat.tolist())
            self.val_pred_list.extend(pred_flat.tolist())
            self.val_score_list.extend(score_flat.tolist())

    # ─────────────────────────────────────────────────────────────────────────
    def _plot_training_graphs(self, lr: float):
        """
        Generate all 9 GRIME AI training graphs + PDF report.
        Mirrors SegFormerTrainer._plot_training_graphs exactly.
        """
        if not self.epoch_list:
            print("[YOLOTrainer] No epoch data to plot.")
            return

        progressBar = QProgressWheel(
            title="Generating graphs...", total=9,
            on_close=lambda: setattr(self, "progress_bar_closed", True),
            parent=self.parent_widget
        )

        viz = ModelTrainingVisualization(
            self.model_output_folder,
            self.formatted_time,
            self.categories,
        )

        train_epochs = self.epoch_list
        val_epochs = self.epoch_list  # YOLO validates every epoch

        lr_tag = f"{lr:.5f}"
        file_prefix = f"{self.formatted_time}_{self.site_name}_lr{lr_tag}"

        # 1) Loss curves
        viz.plot_loss_curves(
            train_epochs=train_epochs,
            train_loss=self.loss_values,
            val_epochs=val_epochs,
            val_loss=self.val_loss_values,
            site_name=self.site_name,
            lr=lr,
            model_name="YOLOv11-seg",
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            num_train_images=self.num_train_images,
            num_val_images=self.num_val_images,
            weight_decay=self.weight_decay,
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 2) Accuracy curves (precision / recall from results.csv)
        viz.plot_accuracy(
            train_epochs=train_epochs,
            train_acc=self.train_accuracy_values,
            val_epochs=val_epochs,
            val_acc=self.val_accuracy_values,
            site_name=self.site_name,
            lr=lr,
            model_name="YOLOv11-seg",
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            num_train_images=self.num_train_images,
            num_val_images=self.num_val_images,
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 3) Confusion matrix
        viz.plot_confusion_matrix(
            y_true=self.val_true_list,
            y_pred=self.val_pred_list,
            site_name=self.site_name,
            lr=lr,
            normalize=True,
            file_prefix="Normalized",
            target_label=", ".join(c["name"] for c in self.categories)
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 4) ROC curve
        viz.plot_roc_curve(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=self.site_name,
            lr=lr,
            file_prefix=file_prefix,
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 5) Precision-Recall
        viz.plot_precision_recall(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=self.site_name,
            lr=lr,
            file_prefix=file_prefix,
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 6) F1 vs Threshold
        viz.plot_f1_score(
            y_true=self.val_true_list,
            y_scores=self.val_score_list,
            site_name=self.site_name,
            lr=lr,
            file_prefix=file_prefix,
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 7) mIoU curve (mAP50 from YOLO)
        viz.plot_miou_curve(
            epochs=self.epoch_list,
            miou_values=self.miou_values,
            site_name=self.site_name,
            lr=lr,
            model_name="YOLOv11-seg",
            file_prefix=file_prefix,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            num_train_images=self.num_train_images,
            num_val_images=self.num_val_images,
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 8) Dice curve (mAP50-95 from YOLO)
        viz.plot_dice_curve(
            epochs=self.epoch_list,
            dice_values=self.val_dice_values,
            site_name=self.site_name,
            lr=lr,
            model_name="YOLOv11-seg",
            file_prefix=file_prefix,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            num_train_images=self.num_train_images,
            num_val_images=self.num_val_images,
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        # 9) IoU curve
        viz.plot_iou_curve(
            epochs=self.epoch_list,
            iou_values=self.val_iou_values,
            site_name=self.site_name,
            lr=lr,
            model_name="YOLOv11-seg",
            file_prefix=file_prefix,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            num_train_images=self.num_train_images,
            num_val_images=self.num_val_images,
        )
        progressBar.setValue(progressBar.getValue() + 1)
        progressBar.show()

        progressBar.close()

        # ── PDF Diagnostic Report ─────────────────────────────────────────────
        acc_png = os.path.join(
            self.model_output_folder,
            f"{self.formatted_time}_{self.site_name}_AccuracyCurves_lr{lr_tag}.png"
        )
        loss_png = os.path.join(
            self.model_output_folder,
            f"{self.formatted_time}_{self.site_name}_LossCurves_lr{lr_tag}.png"
        )
        viz.save_training_report(
            train_acc=self.train_accuracy_values,
            val_acc=self.val_accuracy_values,
            train_loss=self.loss_values,
            val_loss=self.val_loss_values,
            site_name=self.site_name,
            lr=lr,
            miou_values=self.miou_values,
            dice_values=self.val_dice_values,
            graph_paths=[acc_png, loss_png],
            model_type="YOLOv11-seg",
            num_train_images=self.num_train_images,
            num_val_images=self.num_val_images,
            categories=self.categories,
            best_epoch=(
                self.epoch_list[int(np.argmax(self.miou_values))]
                if self.miou_values else None
            ),
            best_val_acc=(
                max(self.val_accuracy_values) if self.val_accuracy_values else None
            ),
            early_stopped=False,
            early_stop_epoch=None,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            training_time_seconds=getattr(self, "training_time_seconds", None),
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _build_unique_categories(
        self, annotation_files: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Identical to MLModelTraining.build_unique_categories.
        Kept here so YOLOTrainer is fully self-contained.
        """
        merged = []
        id_to_name: Dict[int, str] = {}
        name_to_id: Dict[str, int] = {}

        # Flatten nested lists (matches normalize_files pattern)
        flat: List[str] = []
        for item in annotation_files:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)

        for p in flat:
            if not p or not os.path.isfile(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cats = data.get("categories", [])
            except Exception as e:
                print(f"[YOLOTrainer] Failed loading '{p}': {e}")
                continue

            for cat in cats:
                cid = cat.get("id")
                cname = cat.get("name")
                if cid is None or cname is None:
                    continue
                if cid in id_to_name and id_to_name[cid] != cname:
                    print(f"[YOLOTrainer] ⚠ ID conflict: {cid} → "
                          f"'{id_to_name[cid]}' vs '{cname}'")
                    continue
                if cname in name_to_id and name_to_id[cname] != cid:
                    print(f"[YOLOTrainer] ⚠ Name conflict: '{cname}' → "
                          f"{name_to_id[cname]} vs {cid}")
                    continue
                id_to_name[cid] = cname
                name_to_id[cname] = cid
                merged.append({"id": cid, "name": cname})

        unique = {(c["id"], c["name"]): c for c in merged}
        return list(unique.values())
