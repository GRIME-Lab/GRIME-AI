# SAGE/utils/coco_buffer.py
"""
Disk-backed COCO edit buffer.

Keeps a working copy of the annotation file on disk so only the currently open
image's masks live in RAM. Edits for an image are flushed to the temp file on
image change and re-read on revisit. On save, the original instances_default.json
is backed up with a datetime prefix and the temp file is promoted.

Segmentation is stored as lossless compressed COCO RLE (`{"size": [h, w],
"counts": <ascii str>}`) in the `segmentation` field. Reads accept any COCO
segmentation — polygon list, uncompressed RLE, or compressed RLE — so mixed
CVAT exports load correctly; writes are always exact RLE.
"""
import os
import json
import shutil
import tempfile
from datetime import datetime

import cv2
import numpy as np
from pycocotools import mask as _mask


# ---------------------------------------------------------------- RLE
def encode_rle(mask_bool):
    """Lossless compressed COCO RLE. `counts` is ascii str for JSON."""
    m = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = _mask.encode(m)                       # counts: bytes
    return {"size": [int(m.shape[0]), int(m.shape[1])],
            "counts": rle["counts"].decode("ascii")}


def _rle_to_coco(rle):
    """Normalize a stored/JSON RLE dict to pycocotools form (counts: bytes)."""
    counts = rle["counts"]
    return {"size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": counts.encode("ascii") if isinstance(counts, str) else counts}


def mask_to_annotation(mask_bool):
    """Return (rle, bbox, area) for a boolean mask. RLE is bit-exact."""
    m = np.asfortranarray(mask_bool.astype(np.uint8))
    coco_rle = _mask.encode(m)                  # counts: bytes
    bbox = [float(v) for v in _mask.toBbox(coco_rle).tolist()]   # [x, y, w, h]
    area = float(_mask.area(coco_rle))
    rle = {"size": [int(m.shape[0]), int(m.shape[1])],
           "counts": coco_rle["counts"].decode("ascii")}
    return rle, bbox, area


def ann_to_mask(ann, height, width):
    """
    Decode ANY COCO segmentation to a boolean mask:
      - polygon list                 -> union rasterization (no holes; legacy)
      - uncompressed RLE (counts=[]) -> exact
      - compressed RLE (counts=str)  -> exact
    """
    seg = ann.get("segmentation")
    if seg is None:
        return np.zeros((height, width), dtype=bool)

    if isinstance(seg, list):                            # polygon(s)
        if not seg:
            return np.zeros((height, width), dtype=bool)
        rle = _mask.merge(_mask.frPyObjects(seg, height, width))
    elif isinstance(seg, dict):
        if isinstance(seg.get("counts"), list):          # uncompressed RLE
            rle = _mask.frPyObjects(seg, height, width)
        else:                                            # compressed RLE
            rle = _rle_to_coco(seg)
    else:
        return np.zeros((height, width), dtype=bool)

    return _mask.decode(rle).astype(bool)


def polygons_to_mask(segmentation, height, width):
    """Legacy polygon rasterizer. Retained for callers that still pass raw
    polygon lists; the buffer read path now uses ann_to_mask()."""
    m = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


# ---------------------------------------------------------------- buffer
class CocoBuffer:
    def __init__(self, folder):
        self.folder = folder
        self.original_path = os.path.join(folder, "instances_default.json")
        self.temp_path = os.path.join(folder, ".instances_default.working.json")
        self.doc = {"images": [], "annotations": [], "categories": []}
        self._file_to_imgid = {}
        self._next_img_id = 1
        self._next_ann_id = 1
        self._dirty = False   # unsaved in-memory edits not yet written to temp

    # ---- lifecycle ----
    def load(self):
        """Seed the working copy from the original (or empty), write temp."""
        if os.path.exists(self.temp_path):
            with open(self.temp_path) as f:
                self.doc = json.load(f)
        elif os.path.exists(self.original_path):
            with open(self.original_path) as f:
                self.doc = json.load(f)
        self.doc.setdefault("images", [])
        self.doc.setdefault("annotations", [])
        self.doc.setdefault("categories", [])

        for img in self.doc["images"]:
            self._file_to_imgid[img["file_name"]] = img["id"]
            self._next_img_id = max(self._next_img_id, img["id"] + 1)
        for ann in self.doc["annotations"]:
            self._next_ann_id = max(self._next_ann_id, ann["id"] + 1)
        self._write_temp()

    def categories(self):
        return [(c["name"], int(c["id"])) for c in self.doc["categories"]
                if "name" in c and "id" in c]

    def set_categories(self, pairs):
        self.doc["categories"] = [
            {"id": int(cid), "name": name, "supercategory": ""}
            for name, cid in pairs
        ]

    # ---- per-image ----
    def annotations_for(self, filename):
        img_id = self._file_to_imgid.get(filename)
        if img_id is None:
            return []
        return [a for a in self.doc["annotations"] if a["image_id"] == img_id]

    def flush_image(self, filename, mask_entries, height, width, name_to_id):
        """Replace all annotations for `filename` with the given mask entries."""
        img_id = self._file_to_imgid.get(filename)
        if img_id is None:
            img_id = self._next_img_id
            self._next_img_id += 1
            self._file_to_imgid[filename] = img_id
            self.doc["images"].append(
                {"id": img_id, "file_name": filename, "width": width, "height": height}
            )

        # drop old annotations for this image
        self.doc["annotations"] = [
            a for a in self.doc["annotations"] if a["image_id"] != img_id
        ]

        for m in mask_entries:
            mask = m["mask"]
            if not mask.any():          # skip only genuinely empty masks
                continue
            rle, bbox, area = mask_to_annotation(mask)
            self.doc["annotations"].append({
                "id": self._next_ann_id,
                "image_id": img_id,
                "category_id": int(name_to_id.get(m["label"], 0)),
                "segmentation": rle,    # lossless RLE in the COCO-standard field
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "label": m["label"],
            })
            self._next_ann_id += 1

        # Do NOT write the whole buffer to disk here — flushing 20+ MB on every
        # image switch is what made navigation slow. In-session revisits read
        # self.doc from memory, so the temp file is only crash-recovery; persist
        # it lazily (autosave / close / save) via flush_to_disk().
        self._dirty = True

    # ---- persistence ----
    def has_unsaved_changes(self):
        """True if there are in-memory edits not yet promoted to the real
        instances_default.json via save()."""
        return self._dirty

    def flush_to_disk(self):
        """Write the working temp file only if there are unsaved edits.
        Called by autosave, on close, and by save()."""
        if self._dirty:
            self._write_temp()
            self._dirty = False

    # "Other" is a reserved catch-all; it keeps a fixed sentinel id.
    OTHER_LABEL = "Other"
    OTHER_ID = 999

    def _normalize_category_ids(self):
        """Enforce stable category IDs: the reserved "Other" label keeps id 999,
        and every real class is renumbered 1..N in its existing order. Every
        annotation's category_id is remapped through the same old->new map, so
        the written file is deterministic regardless of the IDs assigned
        upstream. Order-preserving and idempotent.
        """
        cats = self.doc.get("categories", [])
        if not cats:
            return
        remap = {}
        n = 0
        for c in cats:
            old = c.get("id")
            if c.get("name") == self.OTHER_LABEL:
                new = self.OTHER_ID
            else:
                n += 1
                new = n
            if old is not None:
                remap[old] = new
            c["id"] = new
        for a in self.doc.get("annotations", []):
            old = a.get("category_id")
            if old in remap:
                a["category_id"] = remap[old]

    def _write_temp(self):
        # Serialize once and write in a single call. Streaming json.dump() emits
        # thousands of small writes; on Windows with AV/sync watching the folder
        # that turned a 21 MB write into ~50s. One write avoids that.
        # Enforce 1-indexed, contiguous category IDs on every write.
        self._normalize_category_ids()
        payload = json.dumps(self.doc)
        fd, tmp = tempfile.mkstemp(dir=self.folder, suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            os.replace(tmp, self.temp_path)   # atomic
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def save(self):
        """Back up original with datetime prefix, promote temp → original."""
        self._write_temp()
        self._dirty = False
        if os.path.exists(self.original_path):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = os.path.join(self.folder, f"instances_default_{ts}.json")
            shutil.copy2(self.original_path, backup)
        shutil.copy2(self.temp_path, self.original_path)
        return self.original_path

    def discard_temp(self):
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)
