#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation: University of Nebraska-Lincoln / Blade Vision Systems, LLC
# License: Apache License, Version 2.0

"""
Holdout Validation Evaluator
=============================
Evaluates a trained SAM2 model against annotated holdout-season images
without updating any model weights. Produces per-image metrics, a summary
CSV, validation overlay images, and a PDF report.

This is distinct from training-time validation — it runs against the seasons
that were excluded from training (the holdout seasons), using ground truth
annotations from the COCO JSON files to compute pixel-level metrics.
"""

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from datetime import datetime
from PIL import Image
from typing import List, Dict, Optional, Tuple

from GRIME_AI.ml_core.seasonal_only import filter_seasons_only
from GRIME_AI.ml_core.seasonal_dropout import extract_date_from_usgs_filename, get_season
from GRIME_AI.utils.datasetutils import DatasetUtils

SEASON_TYPE = "Meteorological"


class HoldoutEvaluator:
    """
    Runs holdout validation for a trained SAM2 model against annotated
    images from the specified holdout seasons.
    """

    def __init__(
        self,
        torch_path: str,
        image_dirs: List[str],
        annotation_paths: List[str],
        eval_seasons: List[str],
        category_name: str,
        output_root: str,
        save_overlays: bool = True,
        progressBar=None,
    ):
        self.torch_path       = torch_path
        self.image_dirs       = image_dirs
        self.annotation_paths = annotation_paths
        self.eval_seasons     = eval_seasons
        self.category_name    = category_name
        self.output_root      = Path(output_root)
        self.save_overlays    = save_overlays
        self.progressBar      = progressBar
        self.progress_bar_closed = False

        self.output_dir = self.output_root / "Holdout_Validation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.dataset_util = DatasetUtils()

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> Optional[Dict]:
        """
        Run the full holdout evaluation pipeline.
        Returns a dict of aggregate metrics, or None if cancelled.
        """
        try:
            import importlib.util
            from omegaconf import OmegaConf
            from hydra.utils import instantiate
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError("sam2 / hydra / omegaconf is not installed.")

        # Load checkpoint metadata
        ckpt = torch.load(self.torch_path, map_location="cpu", weights_only=False)
        site_name   = ckpt.get("site_name", "Unknown")
        categories  = ckpt.get("categories", [])
        blob_radius = ckpt.get("blob_filter_radius", 0.0)

        # Find target category ID
        target_id = next(
            (c["id"] for c in categories if c["name"] == self.category_name), None
        )
        if target_id is None:
            raise ValueError(
                f"Category '{self.category_name}' not found in checkpoint. "
                f"Available: {[c['name'] for c in categories]}"
            )

        # Build model the same way sam2_trainer.py does — no build_sam2/Hydra
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        main_dir = os.path.dirname(importlib.util.find_spec('sam2').origin)
        cfg_file = os.path.join(main_dir, "configs", "sam2.1", "sam2.1_hiera_l.yaml")

        cfg_intern = OmegaConf.load(cfg_file)
        raw_model_cfg = OmegaConf.to_container(cfg_intern.model, resolve=True)
        for k in ["no_obj_embed_spatial", "use_signed_tpos_enc_to_obj_ptrs", "device"]:
            raw_model_cfg.pop(k, None)
        new_cfg = OmegaConf.create(raw_model_cfg)
        sam2_model = instantiate(new_cfg, _recursive_=True)

        sam2_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        sam2_model = sam2_model.to(device)
        sam2_model.eval()
        predictor = SAM2ImagePredictor(sam2_model)

        # Build annotation index via DatasetUtils (same as sam2_trainer.py)
        dataset = self.dataset_util.load_images_and_annotations(
            self.image_dirs, self.annotation_paths, self.category_name
        )
        annotation_index = self.dataset_util.build_annotation_index(dataset)

        # Build evaluation image list
        eval_pairs = self._collect_eval_pairs(dataset, annotation_index, target_id)
        if not eval_pairs:
            print(f"[HoldoutEvaluator] No annotated images found for seasons: {self.eval_seasons}")
            return None

        print(f"[HoldoutEvaluator] Evaluating {len(eval_pairs)} images "
              f"({', '.join(self.eval_seasons)}) for category '{self.category_name}'")

        if self.progressBar:
            self.progressBar.setWindowTitle(
                f"Holdout Validation — {site_name} — {', '.join(self.eval_seasons)}"
            )

        # Run evaluation
        results = []

        with torch.no_grad():
            for idx, (img_path, true_mask, season) in enumerate(eval_pairs):
                if self.progress_bar_closed:
                    print("[HoldoutEvaluator] Cancelled by user.")
                    return None

                try:
                    img_np = np.array(Image.open(img_path).convert("RGB"))
                except Exception as e:
                    print(f"[HoldoutEvaluator] Could not load {img_path}: {e}")
                    continue

                predictor.set_image(img_np)

                # Build centroid prompt from true mask
                point_coords, point_labels = self._build_centroid_prompt(true_mask)
                if point_coords is None:
                    print(f"[HoldoutEvaluator] No valid centroid for {img_path.name} — skipping.")
                    continue

                try:
                    masks, scores, low_res_logits = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=False
                    )
                except Exception as e:
                    print(f"[HoldoutEvaluator] Prediction failed for {img_path.name}: {e}")
                    continue

                if masks.size == 0:
                    continue

                best_idx    = int(np.argmax(scores))
                best_logits = low_res_logits[best_idx]

                logit_t = torch.tensor(best_logits, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                true_t  = torch.tensor(true_mask,   dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                H, W    = true_t.shape[2:]

                logit_up = F.interpolate(logit_t, size=(H, W), mode="bilinear", align_corners=False)
                prob_up  = torch.sigmoid(logit_up)
                pred_bin = (prob_up > 0.5).float()

                # Metrics
                metrics = self._compute_metrics(pred_bin, true_t)
                metrics["filename"] = img_path.name
                metrics["season"]   = season
                metrics["path"]     = str(img_path)
                results.append(metrics)

                # Overlay
                if self.save_overlays:
                    self._save_overlay(img_np, pred_bin, true_t, img_path.name, idx)

                if self.progressBar:
                    self.progressBar.setValue(self.progressBar.getValue() + 1)

        if not results:
            print("[HoldoutEvaluator] No results produced.")
            return None

        # Aggregate
        aggregate = self._aggregate(results)

        # Save CSV and report
        csv_path = self._save_csv(results, aggregate, site_name)
        self._save_report(aggregate, results, site_name, categories, csv_path)

        print(f"[HoldoutEvaluator] Done. Results in: {self.output_dir}")
        return aggregate

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _collect_eval_pairs(self, dataset, annotation_index, target_id) -> List[Tuple[Path, np.ndarray, str]]:
        """
        Collect (image_path, true_mask, season) tuples for all holdout-season
        images that have a valid ground truth annotation for the target category.
        Uses DatasetUtils.load_true_mask — same as sam2_trainer.py.
        """
        pairs = []

        for folder, data in dataset.items():
            all_paths = [Path(p) for p in data["images"]]

            # Filter to holdout seasons
            kept, _ = filter_seasons_only(
                [p.name for p in all_paths], self.eval_seasons, SEASON_TYPE
            )
            kept_set = set(kept)

            for img_path in all_paths:
                if img_path.name not in kept_set:
                    continue

                result = self.dataset_util.load_true_mask(
                    str(img_path), annotation_index, mode="binary", target_id=target_id
                )
                true_mask, found_target = result
                if not found_target or true_mask is None or true_mask.sum() == 0:
                    continue

                date = extract_date_from_usgs_filename(img_path.name)
                season = get_season(date, SEASON_TYPE) if date else "Unknown"

                pairs.append((img_path, true_mask, season))

        return pairs

    def _build_centroid_prompt(self, true_mask: np.ndarray):
        """Build a single centroid point prompt from the ground truth mask."""
        ys, xs = np.where(true_mask > 0)
        if len(ys) == 0:
            return None, None
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        point_coords = np.array([[cx, cy]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        return point_coords, point_labels

    def _compute_metrics(self, pred: torch.Tensor, true: torch.Tensor) -> Dict:
        """Compute pixel-level metrics between predicted and true binary masks."""
        p = pred.bool()
        t = true.bool()

        tp = (p & t).sum().item()
        fp = (p & ~t).sum().item()
        fn = (~p & t).sum().item()
        tn = (~p & ~t).sum().item()
        total = tp + fp + fn + tn

        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        dice      = 2 * tp / (2 * tp + fp + fn + 1e-6)
        iou       = tp / (tp + fp + fn + 1e-6)
        accuracy  = (tp + tn) / (total + 1e-6)
        f1        = dice  # F1 = Dice for binary segmentation

        return {
            "accuracy":  round(accuracy,  4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "dice":      round(dice,      4),
            "iou":       round(iou,       4),
            "f1":        round(f1,        4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    def _aggregate(self, results: List[Dict]) -> Dict:
        """Compute mean metrics across all evaluated images."""
        keys = ["accuracy", "precision", "recall", "dice", "iou", "f1"]
        agg = {}
        for k in keys:
            vals = [r[k] for r in results if k in r]
            agg[k] = round(float(np.mean(vals)), 4) if vals else 0.0
        agg["n_images"] = len(results)
        agg["seasons"]  = self.eval_seasons
        return agg

    def _save_overlay(self, img_np, pred_bin, true_t, filename, idx):
        """
        Save a side-by-side panel: original image (left) with legend in the
        upper-right corner, and the validation overlay (right).
        """
        pred_np = pred_bin.squeeze().cpu().numpy().astype(bool)
        true_np = true_t.squeeze().cpu().numpy().astype(bool)

        H, W = true_np.shape

        # Convert original image to BGR and resize to match mask dimensions
        orig_bgr = cv2.resize(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), (W, H))

        # Build overlay panel (BGR)
        color_overlay = np.zeros((H, W, 3), dtype=np.uint8)
        color_overlay[true_np & pred_np]  = [0,   255, 0]    # TP — green
        color_overlay[~true_np & pred_np] = [0,   0,   255]  # FP — red   (BGR)
        color_overlay[true_np & ~pred_np] = [255, 255, 0]    # FN — cyan  (BGR)
        blended = cv2.addWeighted(orig_bgr, 0.6, color_overlay, 0.4, 0)

        # ── Legend in upper-right of overlay (right) panel ───────────────────
        legend_items = [
            ("Green = True Positive (model predicted it, annotation agrees)", (0,   200, 0)),
            ("Red = False Positive (model predicted it, no annotation)",       (0,   0,   200)),
            ("Cyan = False Negative (annotation says yes, model missed it)",   (200, 200, 0)),
        ]
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, W / 2500)
        thickness  = 1
        pad        = 6
        swatch_sz  = max(10, int(14 * W / 1000))
        line_h     = max(18, int(22 * W / 1000))

        max_text_w = max(
            cv2.getTextSize(label, font, font_scale, thickness)[0][0]
            for label, _ in legend_items
        )
        box_w = swatch_sz + pad * 3 + max_text_w
        box_h = pad * 2 + line_h * len(legend_items)

        margin = 10
        x0 = W - box_w - margin
        y0 = margin

        # Semi-transparent dark background on the blended overlay
        legend_bg = blended.copy()
        cv2.rectangle(legend_bg, (x0, y0), (x0 + box_w, y0 + box_h), (30, 30, 30), -1)
        blended = cv2.addWeighted(blended, 0.35, legend_bg, 0.65, 0)

        for i, (label, bgr) in enumerate(legend_items):
            row_y = y0 + pad + i * line_h
            sx, sy = x0 + pad, row_y
            cv2.rectangle(blended, (sx, sy), (sx + swatch_sz, sy + swatch_sz), bgr, -1)
            cv2.putText(blended, label,
                        (sx + swatch_sz + pad, sy + swatch_sz - 2),
                        font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)

        # Side-by-side panel
        panel = np.concatenate([orig_bgr, blended], axis=1)

        overlay_dir = self.output_dir / "overlays"
        overlay_dir.mkdir(exist_ok=True)
        out_name = f"{self.timestamp}_{Path(filename).stem}_overlay.png"
        cv2.imwrite(str(overlay_dir / out_name), panel)

    def _save_csv(self, results: List[Dict], aggregate: Dict, site_name: str) -> Path:
        """Save per-image metrics to CSV."""
        csv_path = self.output_dir / f"{self.timestamp}_{site_name}_holdout_metrics.csv"
        fieldnames = ["filename", "season", "accuracy", "precision", "recall",
                      "dice", "iou", "f1", "tp", "fp", "fn", "tn"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
            writer.writerow({
                "filename": "AGGREGATE",
                "season":   ", ".join(aggregate["seasons"]),
                "accuracy":  aggregate["accuracy"],
                "precision": aggregate["precision"],
                "recall":    aggregate["recall"],
                "dice":      aggregate["dice"],
                "iou":       aggregate["iou"],
                "f1":        aggregate["f1"],
            })
        print(f"[HoldoutEvaluator] CSV saved: {csv_path.name}")
        return csv_path

    def _save_report(self, aggregate: Dict, results: List[Dict],
                     site_name: str, categories: list, csv_path: Path):
        """Save a PDF summary report."""
        try:
            from fpdf import FPDF
        except ImportError:
            print("[HoldoutEvaluator] fpdf2 not installed — skipping PDF report.")
            return

        pdf = FPDF()
        pdf.add_page()
        pdf.set_margins(20, 20, 20)

        # Title
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(31, 78, 121)
        pdf.cell(0, 10, "GRIME AI - Holdout Validation Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 6, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", ln=True, align="C")
        pdf.ln(4)

        # Summary table
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(31, 78, 121)
        pdf.cell(0, 8, "Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)

        rows = [
            ("Site",              site_name),
            ("Model",             "SAM2"),
            ("Category",          self.category_name),
            ("Holdout Seasons",   ", ".join(aggregate["seasons"])),
            ("Season Type",       SEASON_TYPE),
            ("Images Evaluated",  str(aggregate["n_images"])),
            ("",                  ""),
            ("Mean Accuracy",     f"{aggregate['accuracy']:.4f}"),
            ("Mean Precision",    f"{aggregate['precision']:.4f}"),
            ("Mean Recall",       f"{aggregate['recall']:.4f}"),
            ("Mean Dice",         f"{aggregate['dice']:.4f}"),
            ("Mean IoU",          f"{aggregate['iou']:.4f}"),
            ("Mean F1",           f"{aggregate['f1']:.4f}"),
        ]

        for label, value in rows:
            if not label:
                pdf.ln(3)
                continue
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(70, 7, f"  {label}", border=0)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 7, value, ln=True)

        pdf.ln(6)

        # Per-season breakdown
        seasons_present = sorted(set(r["season"] for r in results))
        if len(seasons_present) > 1:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(31, 78, 121)
            pdf.cell(0, 8, "Per-Season Breakdown", ln=True)
            pdf.set_text_color(0, 0, 0)

            for s in seasons_present:
                s_results = [r for r in results if r["season"] == s]
                s_agg = {k: round(float(np.mean([r[k] for r in s_results])), 4)
                         for k in ["accuracy", "precision", "recall", "dice", "iou"]}
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, f"  {s}  (n={len(s_results)})", ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 6,
                    f"    Acc={s_agg['accuracy']:.4f}  "
                    f"Prec={s_agg['precision']:.4f}  "
                    f"Rec={s_agg['recall']:.4f}  "
                    f"Dice={s_agg['dice']:.4f}  "
                    f"IoU={s_agg['iou']:.4f}", ln=True)

        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, f"Per-image metrics saved to: {csv_path.name}", ln=True)

        pdf_path = self.output_dir / f"{self.timestamp}_{site_name}_holdout_report.pdf"
        pdf.output(str(pdf_path))
        print(f"[HoldoutEvaluator] PDF saved: {pdf_path.name}")
