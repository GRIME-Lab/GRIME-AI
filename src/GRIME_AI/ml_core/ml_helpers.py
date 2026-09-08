#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import cv2
import numpy as np
from pathlib import Path


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Default cap on positive prompts drawn from the pooled centroid population.
# Training, validation and inference MUST all use the same protocol; this
# constant is the single source of truth for that protocol.
DEFAULT_MAX_POSITIVES = 8
DEFAULT_NEGATIVE_BALANCE = 3


def _centroids_to_px(entries, image_w, image_h):
    out = []
    for entry in entries:
        cx_norm, cy_norm = entry["centroid_norm"] if isinstance(entry, dict) else entry
        out.append([int(round(cx_norm * (image_w - 1))), int(round(cy_norm * (image_h - 1)))])
    return out


def sample_pooled_prompts(category_id, category_centroids, image_w, image_h,
                          max_positives=DEFAULT_MAX_POSITIVES,
                          negative_balance=DEFAULT_NEGATIVE_BALANCE,
                          rng=None):
    """
    Deployment prompt protocol, shared by training, validation and inference.

    Positives : up to max_positives centroids sampled from the target category's
                pooled population (all of them if fewer exist).
    Negatives : up to negative_balance * n_pos centroids sampled from every other
                category's pooled population.

    Returns (coords list[[x, y]], labels list[int]) in pixel space, or (None, None).
    rng: random.Random instance. Pass a seeded one for deterministic inference;
         pass an unseeded one during training so each epoch sees a different draw.
    """
    rng = rng or random.Random()
    positives = _centroids_to_px(category_centroids.get(int(category_id), []), image_w, image_h)
    if not positives:
        return None, None
    if max_positives and len(positives) > max_positives:
        positives = rng.sample(positives, max_positives)

    negatives = []
    for cat_id, centroids in category_centroids.items():
        if int(cat_id) == int(category_id):
            continue
        negatives.extend(_centroids_to_px(centroids, image_w, image_h))
    max_neg = len(positives) * negative_balance
    if len(negatives) > max_neg:
        negatives = rng.sample(negatives, max_neg)

    return positives + negatives, [1] * len(positives) + [0] * len(negatives)


def build_centroid_point_prompts(category_id, category_centroids, image_w, image_h, device,
                                 negative_balance=DEFAULT_NEGATIVE_BALANCE, random_seed=42,
                                 max_positives=DEFAULT_MAX_POSITIVES):
    """
    Build SAM2-ready point prompt tensors from stored centroid metadata.

    Positive prompts come from the target category; negative prompts come from
    all other categories, balanced to negative_balance * len(positives).
    Uses a fixed random_seed for reproducibility.

    Args:
        category_id      : int — target category ID
        category_centroids: dict — {cat_id: [{"centroid_norm": (cx, cy)}, ...], ...}
        image_w, image_h : int — pixel dimensions of the current image
        device           : torch.device
        negative_balance : int — max negatives = this * number of positives (default 3)
        random_seed      : int — seed for negative sampling (default 42)

    Returns:
        (point_coords, point_labels) as torch.Tensor, or (None, None) if no
        positive centroids exist for category_id.
    """
    import torch

    coords, labels = sample_pooled_prompts(
        category_id, category_centroids, image_w, image_h,
        max_positives=max_positives, negative_balance=negative_balance,
        rng=random.Random(random_seed),
    )
    if coords is None:
        return None, None
    point_coords = torch.tensor(coords, device=device, dtype=torch.float32)
    point_labels = torch.tensor(labels, device=device, dtype=torch.int64)
    return point_coords, point_labels

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
COLOR_MAP = {
    0: np.array([1.0, 0.0, 0.0, 0.6]),
    1: np.array([0.0, 1.0, 0.0, 0.6]),
    2: np.array([0.0, 0.0, 1.0, 0.6]),
    3: np.array([1.0, 1.0, 0.0, 0.6]),
    4: np.array([1.0, 0.0, 1.0, 0.6]),
    5: np.array([0.0, 1.0, 1.0, 0.6]),
    6: np.array([0.5, 0.0, 0.0, 0.6]),
    7: np.array([0.0, 0.5, 0.0, 0.6]),
    8: np.array([0.0, 0.0, 0.5, 0.6]),
    9: np.array([0.5, 0.5, 0.0, 0.6]),
    10: np.array([0.5, 0.0, 0.5, 0.6]),
    11: np.array([0.0, 0.5, 0.5, 0.6]),
    12: np.array([1.0, 0.5, 0.0, 0.6]),
    13: np.array([0.5, 1.0, 0.0, 0.6]),
    14: np.array([0.0, 1.0, 0.5, 0.6]),
    15: np.array([0.0, 0.5, 1.0, 0.6]),
    16: np.array([0.5, 0.0, 1.0, 0.6]),
    17: np.array([1.0, 0.0, 0.5, 0.6]),
    18: np.array([0.25, 0.25, 0.25, 0.6]),
    19: np.array([0.7, 0.7, 0.7, 0.6]),
    20: np.array([1.0, 0.85, 0.8, 0.6]),
    21: np.array([0.3, 0.3, 1.0, 0.6]),
    22: np.array([1.0, 0.9, 0.2, 0.6]),
    23: np.array([0.2, 0.8, 0.2, 0.6]),
    24: np.array([0.2, 0.4, 0.8, 0.6]),
    25: np.array([0.8, 0.2, 0.4, 0.6]),
    26: np.array([0.4, 0.8, 0.8, 0.6]),
    27: np.array([0.6, 0.3, 0.0, 0.6]),
    28: np.array([0.9, 0.6, 0.4, 0.6]),
    29: np.array([0.3, 0.6, 0.1, 0.6]),
    30: np.array([0.6, 0.1, 0.6, 0.6]),
    31: np.array([0.3, 0.7, 0.6, 0.6]),
}


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def get_color_for_category(category_id):
    """Return RGBA color for a given category id."""
    return COLOR_MAP.get(category_id, np.array([0.5, 0.5, 0.5, 0.6]))


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def init_coco_structure(selected_label_categories):
    """Initialize COCO JSON structure."""
    return {
        "images": [],
        "annotations": [],
        "categories": selected_label_categories or [{"id": 2, "name": "Vegetation"}],
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "",
                 "description": "", "url": "",
                 "version": "", "year": ""}
    }


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def mask_to_polygon(mask, min_contour_area=50):
    """Convert binary mask to polygon segmentation."""
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    if hierarchy is None:
        return segmentation
    for i, contour in enumerate(contours):
        contour = contour.flatten().tolist()
        if len(contour) < 6 or cv2.contourArea(contours[i]) < min_contour_area:
            continue
        if hierarchy[0][i][3] == -1:
            segmentation.append(contour)
        else:
            segmentation.append(contour[::-1])
    return segmentation


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def add_coco_entries(coco_data, image_path, mask, image_array, image_id, annotation_id):
    """Add image and annotation entries to COCO JSON."""
    height, width = image_array.shape[:2]
    coco_data["images"].append({
        "file_name": os.path.basename(image_path),
        "height": height,
        "width": width,
        "id": image_id,
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": 0
    })

    segmentation = mask_to_polygon(mask)
    if not segmentation:
        return

    pos = np.where(mask)
    xmin, xmax = int(np.min(pos[1])), int(np.max(pos[1]))
    ymin, ymax = int(np.min(pos[0])), int(np.max(pos[0]))
    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

    coco_data["annotations"].append({
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 2,
        "segmentation": segmentation,
        "area": int(np.sum(mask.astype(np.uint8))),
        "bbox": bbox,
        "iscrowd": 0
    })


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def save_coco_json(coco_data, output_dir):
    """Save COCO annotations to predictions.json."""
    output_file = Path(output_dir) / "predictions.json"
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=4)
    print(f"COCO annotations saved to {output_file}")
