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
# Written to <image_folder>/ROI_Masks.json. This filename is deliberately NOT
# the SAGE/training filename (instances_default.json / _annotations.coco.json)
# so exporting never overwrites a SAGE-created COCO file. To use these ROIs for
# training, the user renames ROI_Masks.json accordingly.
#
# Author: John Edward Stranzl, Jr.
# License: Apache License, Version 2.0

import os
import json

import cv2

from GRIME_AI.GRIME_AI_roiData import ROIShape


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
                     filename="ROI_Masks.json"):
    """Build COCO 1.0 from the ROIs and write it to image_folder/filename.

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
        roi_geo.append((cat_id_by_name[roiObj.getROIName() or "roi"], flat, bbox, area))

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
        for cat_id, flat, bbox, area in roi_geo:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "segmentation": [flat],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            })
            ann_id += 1

    os.makedirs(image_folder, exist_ok=True)
    out_path = os.path.join(image_folder, filename)
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"ROI masks (COCO 1.0) written: {out_path} "
          f"({len(categories)} categories, {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations)")
    return out_path
