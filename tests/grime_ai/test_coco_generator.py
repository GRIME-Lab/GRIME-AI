"""COCO annotation generation from image/mask pairs."""
import json

import numpy as np
from PIL import Image

from GRIME_AI.coco_generator import CATEGORIES, CocoGenerator

WATER_ID = 2


def square_mask(size=100, x0=20, y0=30, side=40):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[y0:y0 + side, x0:x0 + side] = 255
    return mask


def test_extract_annotations_square():
    anns, next_id = CocoGenerator.extract_annotations(square_mask(), image_id=7, start_ann_id=1)
    water = [a for a in anns if a["category_id"] == WATER_ID]
    assert len(water) == 1
    a = water[0]
    assert a["image_id"] == 7
    assert a["bbox"] == [20, 30, 40, 40]
    assert a["area"] == 39.0 * 39.0               # contour area of a 40-pixel square
    assert a["iscrowd"] == 0
    assert len(a["segmentation"][0]) >= 6
    assert next_id == 1 + len(anns)


def test_annotation_ids_are_sequential():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    mask[60:90, 60:90] = 255
    anns, next_id = CocoGenerator.extract_annotations(mask, image_id=1, start_ann_id=5)
    ids = [a["id"] for a in anns]
    assert ids == list(range(5, 5 + len(anns)))
    assert sum(a["category_id"] == WATER_ID for a in anns) == 2


def test_generate_instances_json_from_pairs(tmp_path):
    for name in ("cam_001", "cam_002"):
        Image.fromarray(np.full((100, 100, 3), 90, dtype=np.uint8)).save(tmp_path / f"{name}.jpg")
        Image.fromarray(square_mask()).save(tmp_path / f"{name}_mask.jpg", quality=100)
    Image.fromarray(square_mask()).save(tmp_path / "orphan_mask.jpg")   # mask with no image

    CocoGenerator(tmp_path).generate_annotations()

    coco = json.loads((tmp_path / "instances_default.json").read_text())
    assert coco["categories"] == CATEGORIES
    assert sorted(i["file_name"] for i in coco["images"]) == ["cam_001.jpg", "cam_002.jpg"]
    assert all(i["width"] == 100 and i["height"] == 100 for i in coco["images"])
    image_ids = {i["id"] for i in coco["images"]}
    assert {a["image_id"] for a in coco["annotations"]} <= image_ids
    assert len({a["id"] for a in coco["annotations"]}) == len(coco["annotations"])


def test_no_pairs_writes_nothing(tmp_path):
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(tmp_path / "lonely.jpg")
    CocoGenerator(tmp_path).generate_annotations()
    assert not (tmp_path / "instances_default.json").exists()
