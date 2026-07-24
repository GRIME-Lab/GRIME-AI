# sam2_gui/core/controller.py
import itertools
import numpy as np
from SAGE.utils.mask_ops import compute_mask_stats
from SAGE.utils.colors import get_color_for_index


class SegmentationController:
    def __init__(self, model_manager, image_np: np.ndarray):
        self.model_manager = model_manager
        self.image_np = image_np
        self.model_manager.initialize_backend(image_np)

        self.fg_points = []  # list of (x, y)
        self.bg_points = []  # list of (x, y)

        self.masks = []  # list of dicts: id, label, mask, color, visible, stats
        self._mask_id_counter = itertools.count(1)

        self.opacity = 120

    # ---- point management ----
    def add_point(self, x, y, is_fg=True):
        if is_fg:
            self.fg_points.append((x, y))
        else:
            self.bg_points.append((x, y))

    def clear_points(self):
        self.fg_points.clear()
        self.bg_points.clear()

    def remove_last_point(self):
        if self.fg_points:
            self.fg_points.pop()
        elif self.bg_points:
            self.bg_points.pop()

    # ---- point removal ----
    def remove_points_in_circle(self, cx, cy, radius):
        r2 = float(radius) * float(radius)

        def keep(p):
            dx = float(p[0]) - float(cx)
            dy = float(p[1]) - float(cy)
            return (dx * dx + dy * dy) > r2

        self.fg_points = [p for p in self.fg_points if keep(p)]
        self.bg_points = [p for p in self.bg_points if keep(p)]

    # ---- ownership ----
    def _owned_union(self, exclude_id=None):
        """Union of every mask on the canvas EXCEPT the 'Other' fill and the
        mask identified by exclude_id. Owned pixels cannot be claimed by a new
        mask. Returns a bool array, or None if nothing is owned.

        Ownership ignores visibility: a hidden mask still owns its pixels.
        To restrict ownership to visible masks, add 'or not m["visible"]' to
        the skip condition below."""
        owned = None
        for m in self.masks:
            if m.get("is_fill"):
                continue
            if exclude_id is not None and m["id"] == exclude_id:
                continue
            mm = m["mask"].astype(bool)
            owned = mm.copy() if owned is None else (owned | mm)
        return owned

    # ---- segmentation ----
    def run_segmentation(self, label=None, color=None, roi=None):
        """roi: optional bool array. When given, it is a hard boundary - the
        resulting mask is clipped to it, so a drawn shape can never annotate
        anything outside itself."""
        if not self.fg_points and not self.bg_points:
            return None

        # Owned pixels = union of every other mask (excluding the 'Other' fill).
        # Passed to the backend as a hint AND hard-subtracted below, so a new
        # mask can never claim owned pixels (first-wins).
        owned = self._owned_union()

        mask = self.model_manager.segment_from_points(
            fg_points=self.fg_points,
            bg_points=self.bg_points,
            exclude_mask=owned,
        )
        if mask is None:
            return None

        if owned is not None:
            mask = mask & ~owned

        if roi is not None:
            mask = mask & np.asarray(roi, dtype=bool)

        if not mask.any():
            self.clear_points()
            return None

        mask_id = next(self._mask_id_counter)
        if color is None:
            color = get_color_for_index(len(self.masks))
        stats = compute_mask_stats(mask)

        mask_entry = {
            "id": mask_id,
            "label": label or f"Region {mask_id}",
            "mask": mask,
            "color": color,
            "visible": True,
            "stats": stats,
        }

        self.masks.append(mask_entry)
        self.recompute_fill()
        self.clear_points()
        return mask_entry

    def run_smart_select_segmentation(self, label=None, color=None):
        """
        Smart Select. fg_points = target anchors, bg_points = exclude
        anchors. Produces a single unioned target mask.
        Returns (mask_entry_or_None, info_dict).
        """
        if not self.fg_points:
            return None, {"reason": "need at least one target anchor"}

        # Same ownership policy as run_segmentation.
        owned = self._owned_union()

        mask, info = self.model_manager.segment_smart_select(
            fg_points=self.fg_points,
            bg_points=self.bg_points,
            exclude_mask=owned,
        )
        if mask is None:
            return None, info

        if owned is not None:
            mask = mask & ~owned
            if not mask.any():
                self.clear_points()
                return None, {"reason": "region lies entirely on already-owned pixels"}

        mask_id = next(self._mask_id_counter)
        if color is None:
            color = get_color_for_index(len(self.masks))
        stats = compute_mask_stats(mask)

        mask_entry = {
            "id": mask_id,
            "label": label or f"Region {mask_id}",
            "mask": mask,
            "color": color,
            "visible": True,
            "stats": stats,
        }
        self.masks.append(mask_entry)
        self.recompute_fill()
        self.clear_points()
        return mask_entry, info

    def add_manual_mask(self, mask, label=None, color=None):
        """Add a non-SAM2 mask (e.g. manual polygon), hard-clipped against
        owned pixels. Returns the entry, or None if fully owned."""
        mask = np.asarray(mask, dtype=bool)
        owned = self._owned_union()
        if owned is not None:
            mask = mask & ~owned
        if not mask.any():
            return None
        mask_id = next(self._mask_id_counter)
        if color is None:
            color = get_color_for_index(len(self.masks))
        entry = {
            "id": mask_id,
            "label": label or f"Region {mask_id}",
            "mask": mask,
            "color": color,
            "visible": True,
            "stats": compute_mask_stats(mask),
        }
        self.masks.append(entry)
        self.recompute_fill()
        return entry

    def set_mask_label(self, mask_id, new_label):
        for m in self.masks:
            if m["id"] == mask_id:
                m["label"] = new_label
                break

    def set_mask_visibility(self, mask_id, visible):
        for m in self.masks:
            if m["id"] == mask_id:
                m["visible"] = visible
                break

    def set_opacity(self, value: int):
        self.opacity = int(value)

    def get_visible_masks(self):
        return [m for m in self.masks if m["visible"]]

    def save_coco(self, filepath):
        """
        Save all masks to a COCO 1.0 annotation file.
        Each mask becomes one annotation with polygon segmentation.
        """

        height, width = self.image_np.shape[:2]

        coco = {
            "images": [
                {
                    "id": 1,
                    "file_name": "image.png",
                    "width": width,
                    "height": height,
                }
            ],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "region"}
            ],
        }

        ann_id = 1

        for m in self.masks:
            mask = m["mask"].astype(np.uint8)

            # Extract polygon(s)
            import cv2
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            segmentation = []
            area = 0
            bbox = None

            for cnt in contours:
                if len(cnt) < 3:
                    continue

                poly = cnt.reshape(-1, 2).tolist()
                segmentation.append([coord for point in poly for coord in point])

                area += cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                if bbox is None:
                    bbox = [x, y, w, h]

            if not segmentation:
                continue

            annotation = {
                "id": ann_id,
                "image_id": 1,
                "category_id": 1,
                "segmentation": segmentation,
                "area": float(area),
                "bbox": bbox,
                "iscrowd": 0,
                "label": m["label"],
            }

            coco["annotations"].append(annotation)
            ann_id += 1

        import json
        with open(filepath, "w") as f:
            json.dump(coco, f, indent=2)

    def fill_other(self, label="Other", color=None):
        """Create one mask covering every pixel not already in a mask, labeled
        `label`. Idempotent: removes any prior fill and recomputes. Returns the
        entry, or None if the image is already fully covered."""
        self.masks = [m for m in self.masks if not m.get("is_fill")]

        h, w = self.image_np.shape[:2]
        union = np.zeros((h, w), dtype=bool)
        for m in self.masks:
            union |= m["mask"].astype(bool)

        complement = ~union
        if not complement.any():
            return None

        entry = {
            "id": next(self._mask_id_counter),
            "label": label,
            "mask": complement,
            "color": color if color is not None else (192, 38, 211),
            "visible": True,
            "stats": compute_mask_stats(complement),
            "is_fill": True,
        }
        self.masks.append(entry)
        return entry

    def recompute_fill(self):
        """Rebuild the 'Other' fill from the current complement so it stays
        exact after other masks change. No-op if no fill exists; drops it if the
        image is now fully covered."""
        fill = next((m for m in self.masks if m.get("is_fill")), None)
        if fill is None:
            return
        h, w = self.image_np.shape[:2]
        union = np.zeros((h, w), dtype=bool)
        for m in self.masks:
            if not m.get("is_fill"):
                union |= m["mask"].astype(bool)
        complement = ~union
        if not complement.any():
            self.masks = [m for m in self.masks if not m.get("is_fill")]
            return
        fill["mask"] = complement
        fill["stats"] = compute_mask_stats(complement)

    def delete_mask(self, mask_id):
        """Remove a mask by its ID."""
        self.masks = [m for m in self.masks if m["id"] != mask_id]