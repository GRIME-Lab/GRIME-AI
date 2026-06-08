#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Apr 2, 2026
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

"""\nGRIME_AI_segment.py\n-------------------\nCommand-line interface for segmentation using a trained GRIME AI model.\nAccepts a single image (--image) or a folder of images (--folder). Use one, not both.

Supports SAM2 and SegFormer-LoRA models.

Usage
-----
SAM2:
    python grime_ai_segment.py \\
        --model  path/to/model.torch \\
        --image  path/to/image.jpg \\
        --output path/to/output_folder \\
        --mode   sam2 \\
        --category-id   1 \\
        --category-name Vegetation

SegFormer:
    python grime_ai_segment.py \\
        --model  path/to/model.torch \\
        --image  path/to/image.jpg \\
        --output path/to/output_folder \\
        --mode   segformer \\
        --category-id   1 \\
        --category-name Vegetation

Optional flags
--------------
    --no-mask        Skip saving the binary mask PNG
    --no-copy        Skip copying the original image to the output folder
    --model-cfg      SAM2 model config YAML (default: sam2.1_hiera_l.yaml)
    --threshold      Probability threshold for SegFormer (default: 0.2)
    --no-gui         Suppress Qt progress wheel (useful in headless environments)
"""

import argparse
import os
import sys
import importlib.util

import numpy as np
import torch
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="GRIME AI — single-image segmentation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",         required=True,  help="Path to trained model checkpoint (.torch)")
    parser.add_argument("--image",         required=False, default=None, help="Path to a single input image")
    parser.add_argument("--folder",        required=False, default=None, help="Path to a folder of images")
    parser.add_argument("--output",        required=True,  help="Output folder for results")
    parser.add_argument("--mode",          required=True,  choices=["sam2", "segformer"],
                        help="Segmentation model type")
    parser.add_argument("--category-id",   type=int, default=1,
                        help="Label category ID to segment (default: 1)")
    parser.add_argument("--category-name", default="Vegetation",
                        help="Label category name (default: Vegetation)")
    parser.add_argument("--model-cfg",     default="sam2.1_hiera_l.yaml",
                        help="SAM2 model config YAML (default: sam2.1_hiera_l.yaml)")
    parser.add_argument("--threshold",     type=float, default=0.2,
                        help="Probability threshold for SegFormer mask (default: 0.2)")
    parser.add_argument("--no-mask",       action="store_true",
                        help="Skip saving binary mask PNG")
    parser.add_argument("--no-copy",       action="store_true",
                        help="Skip copying original image to output folder")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def collect_images(args) -> list:
    """Return sorted list of image paths from --image or --folder."""
    if args.image and args.folder:
        print("[ERROR] Specify --image or --folder, not both.", file=sys.stderr)
        sys.exit(1)
    if not args.image and not args.folder:
        print("[ERROR] Must specify either --image or --folder.", file=sys.stderr)
        sys.exit(1)
    if args.image:
        if not os.path.isfile(args.image):
            print(f"[ERROR] Image file not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        return [args.image]
    # folder mode
    if not os.path.isdir(args.folder):
        print(f"[ERROR] Folder not found: {args.folder}", file=sys.stderr)
        sys.exit(1)
    images = sorted(
        os.path.join(args.folder, f)
        for f in os.listdir(args.folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not images:
        print(f"[ERROR] No images found in folder: {args.folder}", file=sys.stderr)
        sys.exit(1)
    return images


def validate_inputs(args):
    if not os.path.isfile(args.model):
        print(f"[ERROR] Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.output, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# SAM2 segmentation
# ──────────────────────────────────────────────────────────────────────────────
def run_sam2(args, device, category, progressBar, image_list=None):
    from GRIME_AI.ml_core.sam2_inference_engine import SAM2InferenceEngine

    if image_list is None:
        image_list = collect_images(args)

    # Use first image's directory as input_dir for engine init
    input_dir = os.path.dirname(os.path.abspath(image_list[0]))

    engine = SAM2InferenceEngine(
        device=device,
        model_cfg=args.model_cfg,
        trained_checkpoint_path=args.model,
        input_dir=input_dir,
        output_dir=args.output,
    )

    # Override output path — engine appends " (sam2)" by default
    engine.predictions_output_path = args.output
    os.makedirs(args.output, exist_ok=True)

    predictor = engine.load_sam2_model()
    if predictor is None:
        print("[ERROR] Failed to load SAM2 model.", file=sys.stderr)
        sys.exit(1)

    save_masks = not args.no_mask
    copy_original = not args.no_copy
    n = len(image_list)

    for i, image_path in enumerate(image_list, 1):
        print(f"[SAM2] Processing {i}/{n}: {os.path.basename(image_path)}")
        pil_image = Image.open(image_path).convert("RGB")
        image_array = np.array(pil_image)

        mask, prob_map, score = engine.predict_with_centroids(
            predictor, image_array, category_id=args.category_id
        )

        if mask is None:
            print(f"[SAM2] WARNING: No mask returned for {os.path.basename(image_path)}, skipping.")
            continue

        score = float(np.asarray(score).flat[0])
        print(f"[SAM2] Score: {score:.4f}")

        # Ensure prob_map is a 2D array matching the image dimensions
        if prob_map is None or np.asarray(prob_map).ndim == 0:
            prob_map = np.full((image_array.shape[0], image_array.shape[1]), score, dtype=np.float32)
        prob_map = np.asarray(prob_map)
        if prob_map.ndim != 2:
            prob_map = np.squeeze(prob_map)

        engine.save_outputs(
            image_path=image_path,
            pil_image=pil_image,
            mask=mask,
            prob_map=prob_map,
            score=score,
            save_masks=save_masks,
            copy_original_image=copy_original,
            category_id=args.category_id,
            category_name=args.category_name,
        )

    print(f"[SAM2] Outputs saved to: {args.output}")


# ──────────────────────────────────────────────────────────────────────────────
# SegFormer segmentation
# ──────────────────────────────────────────────────────────────────────────────
def run_segformer(args, device, category, progressBar, image_list=None):
    from GRIME_AI.ml_core.segformer_inference_engine import SegFormerInferenceEngine
    from torchvision import transforms as T
    import shutil
    import cv2

    if image_list is None:
        image_list = collect_images(args)

    input_dir = os.path.dirname(os.path.abspath(image_list[0]))

    engine = SegFormerInferenceEngine(
        device=device,
        model_path=args.model,
        input_dir=input_dir,
        output_dir=args.output,
        class_index=args.category_id,
    )

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = T.ToTensor()

    os.makedirs(args.output, exist_ok=True)
    save_masks = not args.no_mask
    copy_original = not args.no_copy
    n = len(image_list)

    for i, image_path in enumerate(image_list, 1):
        print(f"[SegFormer] Processing {i}/{n}: {os.path.basename(image_path)}")
        pil_image = Image.open(image_path).convert("RGB")
        image_array = np.array(pil_image)

        x = to_tensor(pil_image.resize((512, 512)))
        x = normalize(x).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = engine.model(pixel_values=x).logits

        probs = torch.softmax(logits, dim=1)
        target_prob = probs[0, args.category_id]
        mask = (target_prob > args.threshold).cpu().numpy().astype(np.uint8)
        prob_map = target_prob.cpu().numpy()
        score = float(target_prob.mean().item())

        h, w = image_array.shape[:2]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)

        print(f"[SegFormer] Score: {score:.4f}")

        base = os.path.splitext(os.path.basename(image_path))[0]

        prob_vis = (prob_map * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, f"{base}_prob.png"), prob_vis)

        if save_masks:
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            cv2.imwrite(os.path.join(args.output, f"{base}_mask.png"), mask_clean * 255)

        overlay_color = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
        overlay_color_rgb = cv2.cvtColor(overlay_color, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(image_array, 0.6, overlay_color_rgb, 0.4, 0)
        cv2.imwrite(os.path.join(args.output, f"{base}_overlay.png"),
                    cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        if copy_original:
            shutil.copy(image_path, os.path.join(args.output, os.path.basename(image_path)))

    print(f"[SegFormer] Outputs saved to: {args.output}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    validate_inputs(args)

    image_list = collect_images(args)
    category = {"id": args.category_id, "name": args.category_name}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[GRIME AI] Using device: {device}")
    print(f"[GRIME AI] Mode:         {args.mode.upper()}")
    print(f"[GRIME AI] Model:        {args.model}")
    print(f"[GRIME AI] Images:       {len(image_list)} file(s)")
    print(f"[GRIME AI] Output:       {args.output}")
    print(f"[GRIME AI] Category:     {args.category_name} (ID: {args.category_id})")

    if args.mode == "sam2":
        run_sam2(args, device, category, progressBar=None, image_list=image_list)
    elif args.mode == "segformer":
        run_segformer(args, device, category, progressBar=None, image_list=image_list)

    print("[GRIME AI] Segmentation complete.")


if __name__ == "__main__":
    main()
