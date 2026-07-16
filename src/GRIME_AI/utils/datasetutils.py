import os
import random
from pycocotools import mask as coco_mask
import cv2
import numpy as np
import json


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
def get_category_id(annotations: dict, target_label: str) -> int:
    target = target_label.strip().lower()
    name_to_id = {
        str(cat.get("name", "")).strip().lower(): int(cat.get("id"))
        for cat in annotations.get("categories", [])
        if "id" in cat and "name" in cat
    }
    if target not in name_to_id:
        available = [cat.get("name") for cat in annotations.get("categories", [])]
        raise ValueError(
            f"Label '{target_label}' not found in annotations categories. "
            f"Available: {available}"
        )
    return name_to_id[target]


def _seg_to_mask(segmentation, height, width):
    """Decode polygon, uncompressed RLE, or compressed RLE to an H×W mask."""
    if isinstance(segmentation, dict):                     # RLE dict
        counts = segmentation.get("counts")
        if isinstance(counts, (bytes, str)):               # compressed RLE
            rle = segmentation
        else:                                              # uncompressed RLE
            rle = coco_mask.frPyObjects(segmentation, height, width)
    else:                                                  # polygon(s)
        rle = coco_mask.frPyObjects(segmentation, height, width)
        if isinstance(rle, list):
            rle = coco_mask.merge(rle)
    mask = coco_mask.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)


class DatasetUtils:
    # ------------------------------------------------------------------------
    def __init__(self):
        self.image_shape_cache = {}

    # ------------------------------------------------------------------------
    def load_images_and_annotations(self, folders, annotation_files, target_label):
        """
        Load images and annotations from multiple folder/annotation pairs.
        Each folder is processed independently and stored under its normalized
        path to prevent basename collisions between folders with identical filenames.

        Returns dict: normalized_folder_path -> {"images": [...], "annotations": {...}}
        """
        dataset = {}

        for folder, annotation_file in zip(folders, annotation_files):
            folder = os.path.normpath(folder)
            annotation_file = os.path.normpath(annotation_file)

            try:
                disk_files = [
                    f for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".jpeg"))
                ]
            except OSError as e:
                raise OSError(f"Cannot list folder '{folder}': {e}")

            try:
                with open(annotation_file, "r", encoding="utf-8") as f:
                    annotations = json.load(f)
            except OSError as e:
                raise OSError(f"Cannot open annotation file '{annotation_file}': {e}")

            # Validate target label exists
            get_category_id(annotations, target_label)

            # Build set of JSON-referenced filenames
            json_filenames = {
                os.path.basename(img.get("file_name", ""))
                for img in annotations.get("images", [])
                if isinstance(img, dict)
            }

            disk_set = set(disk_files)

            # Only train on files present both on disk AND in JSON
            matched_files = [f for f in disk_files if f in json_filenames]
            skipped = len(disk_files) - len(matched_files)
            if skipped > 0:
                print(f"[DatasetUtils] {folder}: {skipped} on-disk file(s) not in JSON - skipped.")

            missing = [f for f in json_filenames if f not in disk_set]
            if missing:
                print(f"[DatasetUtils] {folder}: {len(missing)} JSON-referenced file(s) not on disk.")
                for m in missing[:5]:
                    print(f"  missing: {m}")
                if len(missing) > 5:
                    print(f"  ... and {len(missing) - 5} more.")

            full_paths = [
                os.path.normpath(os.path.join(folder, f))
                for f in matched_files
            ]

            dataset[folder] = {
                "images": full_paths,
                "annotations": {
                    "images": annotations.get("images", []),
                    "annotations": annotations.get("annotations", []),
                },
            }

        return dataset

    # ------------------------------------------------------------------------
    def build_annotation_index(self, dataset):
        """
        Build a mapping from FULL NORMALIZED IMAGE PATH to its annotation entry.

        Keying by full path (not basename) prevents collisions when multiple
        folders contain identically named files.
        """
        annotation_index = {}
        for folder, data in dataset.items():
            ann_data = data["annotations"]
            # basename -> image_info for this folder's JSON
            basename_to_info = {
                os.path.basename(img.get("file_name", "")): img
                for img in ann_data.get("images", [])
                if isinstance(img, dict)
            }
            for image_path in data["images"]:
                norm_path = os.path.normpath(image_path)
                base = os.path.basename(norm_path)
                image_info = basename_to_info.get(base)
                if image_info is None:
                    print(f"[DatasetUtils] Warning: no JSON entry for {norm_path} - skipping.")
                    continue
                annotation_index[norm_path] = {
                    "image_info": image_info,
                    "annotations": ann_data.get("annotations", []),
                }
        return annotation_index

    # ------------------------------------------------------------------------
    def load_true_mask(self, image_file, annotation_index, mode="binary", target_id=None):
        """
        Loads a mask for an image using COCO-style annotations.
        Looks up by full normalized path to prevent cross-folder contamination.

        Returns:
          binary mode: (mask: np.ndarray, found_target: bool)
          categorical mode: mask: np.ndarray
        """
        norm_path = os.path.normpath(image_file)

        if norm_path not in annotation_index:
            raise ValueError(
                f"Image not in annotation index: {norm_path}\n"
                f"Index has {len(annotation_index)} entries."
            )

        entry = annotation_index[norm_path]
        image_info = entry["image_info"]
        all_anns = entry["annotations"]

        image_id = image_info["id"]
        height = image_info["height"]
        width = image_info["width"]

        anns = [ann for ann in all_anns if ann["image_id"] == image_id]

        if not anns:
            if mode == "binary":
                return np.zeros((height, width), dtype=np.uint8), False
            return np.zeros((height, width), dtype=np.uint8)

        if mode == "binary":
            combined = np.zeros((height, width), dtype=np.uint8)
            found_target = False
            for ann in anns:
                if target_id is not None and ann["category_id"] != target_id:
                    continue
                mask = _seg_to_mask(ann["segmentation"], height, width)
                combined = np.logical_or(combined, mask).astype(np.uint8)
                if target_id is not None and ann["category_id"] == target_id:
                    found_target = True
            return combined, found_target

        elif mode == "categorical":
            categorical = np.zeros((height, width), dtype=np.uint8)
            for ann in anns:
                mask = _seg_to_mask(ann["segmentation"], height, width)
                categorical[mask.astype(bool)] = ann["category_id"]
            return categorical

        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'binary' or 'categorical'.")

    # ------------------------------------------------------------------------
    def split_dataset(self, dataset_dict, train_split=0.9, val_split=0.1):
        """
        Collect all image paths from all folders and split into train/val.
        Paths are already normalized full paths from load_images_and_annotations.
        """
        all_images = []
        for data in dataset_dict.values():
            for img in data["images"]:
                all_images.append(os.path.normpath(str(img)))

        random.shuffle(all_images)

        train_size = int(train_split * len(all_images))
        train_images = all_images[:train_size]
        val_images = all_images[train_size:]
        print(f"Train: {len(train_images)} images, Validation: {len(val_images)} images")

        return train_images, val_images

    # ------------------------------------------------------------------------
    def save_split_dataset(self, train_images, val_images, output_file):
        """
        Save metadata-only split to a JSON file.
        """
        from pathlib import Path

        out_dir = Path(os.path.dirname(output_file))
        out_dir.mkdir(parents=True, exist_ok=True)

        def extract_paths(images):
            return [img['path'] if isinstance(img, dict) else str(img) for img in images]

        split_data = {
            "train": extract_paths(train_images),
            "val": extract_paths(val_images)
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2)

        print(f"[split] Saved metadata-only split to {output_file}")

    # ------------------------------------------------------------------------
    def get_image_size(self, image_path: str) -> tuple[int, int]:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return img.shape[:2]

    # ------------------------------------------------------------------------
    def get_annotations_for_image(self, image_path: str, annotation_index: dict) -> list[dict]:
        """
        Returns annotation dicts for this image. Looks up by full normalized path.
        """
        norm_path = os.path.normpath(image_path)
        entry = annotation_index.get(norm_path)
        if entry is None:
            return []
        image_id = entry["image_info"]["id"]
        return [ann for ann in entry["annotations"] if ann["image_id"] == image_id]

    # ------------------------------------------------------------------------
    def rasterize_polygon(self, segmentation, image_shape: tuple[int, int]) -> np.ndarray:
        """Decode any COCO segmentation (polygon, uncompressed RLE, or
        compressed RLE) to an H×W 0/1 mask. Holes are preserved — this routes
        through coco_mask.decode, so ring masks (e.g. water around a sandbar)
        keep their interior 0s instead of being filled solid."""
        h, w = image_shape
        return _seg_to_mask(segmentation, h, w)

    # ------------------------------------------------------------------------
    def load_all_true_masks(self, image_path: str, annotation_index: dict) -> dict[int, np.ndarray]:
        """
        Returns dict mapping category_id -> H×W binary mask for all annotations in this image.
        """
        anns = self.get_annotations_for_image(image_path, annotation_index)
        h, w = self.get_image_size(image_path)

        all_masks: dict[int, np.ndarray] = {}
        for ann in anns:
            cid = ann.get("category_id")
            seg = ann.get("segmentation")
            if cid is None or seg is None:
                continue
            mask = self.rasterize_polygon(seg, (h, w))
            if mask.sum() > 0:
                all_masks[cid] = mask

        return all_masks
