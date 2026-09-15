#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GRIME_AI_ROI_COCO_Export.py
#
# Export Color Segmentation ROIs to COCO 1.0, matching the format the ML /
# SAGE training path consumes. ROI name -> category; ROI shape (rectangle,
# polygon, freeform) -> polygon segmentation.
#
# The ROIs are the SAME across every image in the folder (fixed ROIs let the
# feature-file build detect change WITHIN an ROI over time), so each ROI is
# replicated as an annotation against every image in the set.
#
# Written to <image_folder>/ROI_Masks_<YYYYMMDD_HHMMSS>.json. This filename is deliberately NOT
# the SAGE/training filename (instances_default.json / _annotations.coco.json)
# so exporting never overwrites a SAGE-created COCO file. To use these ROIs for
# training, the user renames the ROI_Masks file accordingly.
#
# Author: John Edward Stranzl, Jr.
# License: Apache License, Version 2.0

import os
import json
import datetime

import cv2

from GRIME_AI.dialogs.color_segmentation.color_seg_roi_data import ROIShape


_SHAPE_NAMES = {ROIShape.RECTANGLE: "rectangle", ROIShape.POLYGON: "polygon", ROIShape.FREEFORM: "freeform"}


def _roi_polygon_points(roiObj):
    """Return a flat [x0,y0,x1,y1,...] polygon in IMAGE coordinates for any ROI
    shape. Rectangles become their four corners; polygon/freeform use their
    captured vertices."""
    shape = roiObj.getROIShape()
    if shape in (ROIShape.POLYGON, ROIShape.FREEFORM):
        pts = roiObj.getImagePolygon()
        if pts and len(pts) >= 3:
            flat = []
            for p in pts:
                flat.extend([int(p.x()), int(p.y())])
            return flat
    # Rectangle (or a polygon that never captured points): use the image rect.
    r = roiObj.getImageROI()
    x, y, w, h = r.x(), r.y(), r.width(), r.height()
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def _bbox_and_area(flat):
    xs = flat[0::2]
    ys = flat[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w, h = xmax - xmin, ymax - ymin
    # Shoelace polygon area (exact for the actual shape, not the bbox).
    area = 0.0
    n = len(xs)
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    area = abs(area) / 2.0
    return [xmin, ymin, w, h], float(area)


def _as_path(item):
    """Items may be plain path strings or imageData objects carrying
    .fullPathAndFilename. Return the path string either way."""
    if isinstance(item, str):
        return item
    for attr in ('fullPathAndFilename', 'filename', 'path'):
        v = getattr(item, attr, None)
        if v:
            return v
    return str(item)


def export_roi_masks(roi_list, images_list, image_folder,
                     filename=None):
    """Build COCO 1.0 from the ROIs and write it to image_folder/filename.
    filename defaults to ROI_Masks_<YYYYMMDD_HHMMSS>.json (same stamp format as the feature files).

    roi_list     : list of roiData objects (name + shape + image-coords geometry)
    images_list  : list of image file paths (or basenames) the ROIs apply to
    image_folder : destination folder for the JSON

    Returns the output path. Raises ValueError if there are no ROIs/images.
    """
    if not roi_list:
        raise ValueError("No ROIs to export. Draw and add at least one ROI.")
    if not images_list:
        raise ValueError("No images in the folder to associate ROIs with.")

    # Categories: one per DISTINCT ROI name (dedup, ids from 1).
    cat_id_by_name = {}
    categories = []
    for roiObj in roi_list:
        name = roiObj.getROIName() or "roi"
        if name not in cat_id_by_name:
            cid = len(cat_id_by_name) + 1
            cat_id_by_name[name] = cid
            categories.append({"id": cid, "name": name, "supercategory": ""})

    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "",
                 "description": "GRIME AI Color Segmentation ROIs",
                 "url": "", "version": "1.0", "year": ""},
    }

    # Precompute each ROI's polygon/bbox/area once (same across all images).
    roi_geo = []
    for roiObj in roi_list:
        flat = _roi_polygon_points(roiObj)
        bbox, area = _bbox_and_area(flat)
        roi_geo.append((cat_id_by_name[roiObj.getROIName() or "roi"], flat, bbox, area,
                        _SHAPE_NAMES.get(roiObj.getROIShape(), "polygon")))

    ann_id = 1
    for image_id, image_item in enumerate(images_list, start=1):
        image_path = _as_path(image_item)
        img = cv2.imread(image_path)
        if img is None:
            # still record the image with unknown size rather than skip it
            height = width = 0
        else:
            height, width = img.shape[:2]
        coco["images"].append({
            "file_name": os.path.basename(image_path),
            "height": height, "width": width, "id": image_id,
            "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0,
        })
        # Replicate every ROI against this image.
        for cat_id, flat, bbox, area, shape_name in roi_geo:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "segmentation": [flat],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                # extra key (ignored by COCO tools) so an import restores the drawing tool
                "attributes": {"shape": shape_name},
            })
            ann_id += 1

    os.makedirs(image_folder, exist_ok=True)
    if not filename:
        filename = 'ROI_Masks_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.json'
    out_path = os.path.join(image_folder, filename)
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"ROI masks (COCO 1.0) written: {out_path} "
          f"({len(categories)} categories, {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations)")
    return out_path


def load_roi_masks(json_path, image_filename=None):
    """Read ROIs back from a COCO 1.0 file (an ROI_Masks export, or any COCO polygon file).

    ROIs are the same across every image in an ROI_Masks export, so the annotations of one image
    are used: the entry whose file_name matches image_filename if present, otherwise the first image.

    Returns {'file_name', 'width', 'height', 'rois': [{'name', 'shape', 'polygon', 'bbox'}], 'skipped'}
      shape   : 'rectangle', 'polygon' or 'freeform'
      polygon : [(x, y), ...] integer image coordinates
      bbox    : (x, y, w, h) integer image coordinates
      skipped : annotations that could not be read (e.g. RLE masks)
    Raises ValueError if the file has no usable ROIs.
    """
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = coco.get("images") or []
    if not images:
        raise ValueError("The file contains no images.")
    cats = {c.get("id"): c.get("name") or "roi" for c in (coco.get("categories") or [])}

    entry = None
    if image_filename:
        entry = next((im for im in images if im.get("file_name") == image_filename), None)
    if entry is None:
        entry = images[0]

    anns = [a for a in (coco.get("annotations") or []) if a.get("image_id") == entry.get("id")]
    rois, skipped = [], 0
    for a in anns:
        seg = a.get("segmentation")
        flat = None
        if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)):
            # several polygons: keep the one with the most vertices
            flat = max(seg, key=len)
        elif isinstance(seg, list) and seg and not isinstance(seg[0], (list, tuple)):
            flat = seg
        elif not seg and a.get("bbox"):
            x, y, w, h = a["bbox"]
            flat = [x, y, x + w, y, x + w, y + h, x, y + h]
        if not flat or len(flat) < 6:
            skipped += 1          # RLE or malformed
            continue

        pts = [(int(round(flat[i])), int(round(flat[i + 1]))) for i in range(0, len(flat) - 1, 2)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

        shape = ((a.get("attributes") or {}).get("shape") or "").lower()
        if shape not in ("rectangle", "polygon", "freeform"):
            is_rect = (len(pts) == 4 and len(set(xs)) == 2 and len(set(ys)) == 2)
            shape = "rectangle" if is_rect else "polygon"

        rois.append({"name": cats.get(a.get("category_id"), "roi"), "shape": shape,
                     "polygon": pts, "bbox": bbox})

    if not rois:
        raise ValueError("No polygon or box ROIs were found for image "
                         f"'{entry.get('file_name', '?')}'.")
    return {"file_name": entry.get("file_name", ""), "width": int(entry.get("width") or 0),
            "height": int(entry.get("height") or 0), "rois": rois, "skipped": skipped}
