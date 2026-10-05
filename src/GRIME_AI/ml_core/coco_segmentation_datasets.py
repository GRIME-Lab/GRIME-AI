#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Oct 31, 2025
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from typing import Dict, Any, List

# ============================================================================
# ============================================================================
# ===                         Helper Functions                             ===
# ============================================================================
# ============================================================================
def normalize_path(annotation_files):
    """
    Normalize a list of annotation file inputs into a flat list of strings.
    Handles cases like:
      ["file.json"]
      [["file.json"]]
      [[["file.json"]]]
    """
    def unwrap_path(path):
        # Recursively unwrap nested lists until a string path is found
        while isinstance(path, list):
            if not path:  # empty list guard
                raise ValueError("Got an empty list instead of a path")
            path = path[0]
        if not isinstance(path, str):
            raise ValueError(f"Expected string path, got {type(path)}")
        return path

    result = []
    for f in annotation_files:
        result.append(unwrap_path(f))
    return result

def to_string_path(x):
    """
    Unwrap arbitrarily nested lists/tuples until a single string path is returned.
    Raises if the final value is not a string.
    """
    while isinstance(x, (list, tuple)):
        if not x:
            raise ValueError("Empty list/tuple received where a path was expected.")
        x = x[0]
    if not isinstance(x, str):
        raise ValueError(f"Expected a string path, got {type(x)}")
    return x


def to_path_list(x):
    """
    Normalize input into a list[str].
    Accepts:
      - str -> [str]
      - list/tuple of str -> [str]
      - arbitrarily nested lists/tuples -> flattened [str]
    Never iterates into strings (so no character-splitting).
    """
    if isinstance(x, str):
        return [x]

    result = []

    def walk(y):
        if isinstance(y, str):
            result.append(y)
        elif isinstance(y, (list, tuple)):
            for z in y:
                walk(z)
        else:
            raise ValueError(f"Unexpected type in path list: {type(y)}")

    walk(x)
    return result

def load_image(images_dir: str, img_info: Dict[str, Any]) -> np.ndarray:

    images_dir = to_string_path(images_dir)

    path = os.path.normpath(os.path.join(images_dir, img_info['file_name']))
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def build_mask(coco, img_id: int, cat_id: int, height: int, width: int) -> np.ndarray:
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        seg = ann["segmentation"]
        if isinstance(seg, list):
            rles = mask_utils.frPyObjects(seg, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(seg, dict):
            if isinstance(seg.get("counts", None), list):
                rle = mask_utils.frPyObjects(seg, height, width)
            else:
                rle = seg
        else:
            continue
        m = mask_utils.decode(rle)
        mask = np.maximum(mask, m.astype(np.uint8))
    return mask

def maybe_augment(img_t, mask_t, split, cfg):
    """Augment only the training split, and only when cfg enables it.
    Geometric transforms apply to image AND mask together; color jitter hits
    the image only (see apply_augmentation). Call BEFORE normalize so
    brightness/contrast operate on [0,1] pixel values."""
    if split == "train" and cfg is not None and getattr(cfg, "use_augmentation", False):
        # Lazy import: segformer_trainer imports this module.
        from appcore.ml_core.segformer_trainer import apply_augmentation
        img_t, mask_t = apply_augmentation(img_t, mask_t, cfg)
    return img_t, mask_t

# ============================================================================
# ============================================================================
# = = =                      class CocoWaterDataset                      = = =
# ============================================================================
# ============================================================================
class CocoWaterDataset(Dataset):
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def __init__(self, images_dir: str, ann_path: str, target_category_name: str, image_size: int, split: str = "train", val_ratio: float = 0.1, cfg=None):
        super().__init__()
        self.images_dir = images_dir
        self.coco = COCO(ann_path)

        cat_ids = self.coco.getCatIds(catNms=[target_category_name])
        if not cat_ids:
            raise ValueError(f"Category '{target_category_name}' not found in annotations.")
        self.water_cat_id = cat_ids[0]

        img_ids = sorted(self.coco.getImgIds())
        rng = np.random.default_rng(12345)
        perm = rng.permutation(len(img_ids))
        val_count = int(len(img_ids) * val_ratio)
        val_idx = set(perm[:val_count].tolist())
        self.img_ids = [img_ids[i] for i in range(len(img_ids)) if (i not in val_idx if split == "train" else i in val_idx)]

        self.size = image_size
        self.split = split
        self.cfg = cfg
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img = load_image(self.images_dir, img_info)
        h, w = img.shape[:2]
        mask = build_mask(self.coco, img_id, self.water_cat_id, h, w)

        img_resized = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        img_t = self.to_tensor(img_resized).float()
        mask_t = torch.from_numpy(mask_resized).long()
        img_t, mask_t = maybe_augment(img_t, mask_t, self.split, self.cfg)
        img_t = self.normalize(img_t)
        return img_t, mask_t

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def __len__(self):
        return len(self.img_ids)

# ============================================================================
# ============================================================================
# = = =                  class MultiCocoTargetDataset                    = = =
# ============================================================================
# ============================================================================
class MultiCocoTargetDataset(Dataset):
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def __init__(self, images_dirs, ann_paths, target_category_name, image_size, split="train", val_ratio=0.1, cfg=None):
        super().__init__()
        assert len(images_dirs) == len(ann_paths)
        self.size = image_size
        self.split = split
        self.cfg = cfg
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.entries = []

        rng = np.random.default_rng(12345)
        for images_dir, ann_path in zip(images_dirs, ann_paths):
            ann_path = to_string_path(ann_path)
            coco = COCO(ann_path)
            cat_ids = coco.getCatIds(catNms=[target_category_name])
            if not cat_ids:
                raise ValueError(f"Category '{target_category_name}' not found in {ann_path}")
            water_cat_id = cat_ids[0]

            img_ids = sorted(coco.getImgIds())
            perm = rng.permutation(len(img_ids))
            val_count = int(len(img_ids) * val_ratio)
            val_idx = set(perm[:val_count].tolist())
            chosen = [img_ids[i] for i in range(len(img_ids)) if (i not in val_idx if split == "train" else i in val_idx)]

            for img_id in chosen:
                self.entries.append((coco, img_id, images_dir, water_cat_id))

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def __getitem__(self, idx):
        coco, img_id, images_dir, water_cat_id = self.entries[idx]
        img_info = coco.loadImgs([img_id])[0]
        img = load_image(images_dir, img_info)
        h, w = img.shape[:2]
        mask = build_mask(coco, img_id, water_cat_id, h, w)

        img_resized = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        img_t = self.to_tensor(img_resized).float()
        mask_t = torch.from_numpy(mask_resized).long()
        img_t, mask_t = maybe_augment(img_t, mask_t, self.split, self.cfg)
        img_t = self.normalize(img_t)
        return img_t, mask_t

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def __len__(self):
        return len(self.entries)
