from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

from PIL import Image


def resize_images(example: Dict[str, Any], target_size: int) -> Dict[str, Any]:
    """
    Resize a PIL image to (target_size, target_size) and scale PASCAL-style bboxes accordingly.
    (Kept exactly as in original script; may be unused elsewhere.)
    """
    img = example["image"]
    orig_w, orig_h = img.size

    # Resize image
    img_resized = img.resize((target_size, target_size), Image.BILINEAR)

    # Scale bounding boxes
    scale_x = target_size / orig_w
    scale_y = target_size / orig_h
    boxes: List[List[float]] = []
    for box in example["objects"]["bboxes"]:
        xmin, ymin, xmax, ymax = box
        boxes.append([xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y])

    example["image"] = img_resized
    example["objects"]["bboxes"] = boxes
    example["width"] = target_size
    example["height"] = target_size
    return example


def preprocess_yolo(example: Dict[str, Any], img_size: int = 448) -> Dict[str, Any]:
    """
    Resize to (img_size, img_size) and convert bboxes to YOLO normalized format [xc, yc, w, h].
    """
    img = example["image"]

    orig_w = example.get("width", img.size[0])
    orig_h = example.get("height", img.size[1])

    # resize once
    img = img.resize((img_size, img_size), Image.BILINEAR)

    boxes: List[List[float]] = []
    labels: List[int] = []

    for box, label in zip(example["objects"]["bboxes"], example["objects"]["classes"]):
        xmin, ymin, xmax, ymax = box

        # scale to resized image
        xmin = xmin * img_size / orig_w
        xmax = xmax * img_size / orig_w
        ymin = ymin * img_size / orig_h
        ymax = ymax * img_size / orig_h

        # YOLO normalized
        xc = ((xmin + xmax) / 2) / img_size
        yc = ((ymin + ymax) / 2) / img_size
        w = (xmax - xmin) / img_size
        h = (ymax - ymin) / img_size

        boxes.append([xc, yc, w, h])
        labels.append(int(label))

    return {"image": img, "boxes": boxes, "labels": labels}


def preprocess_imagenet(example: Dict[str, Any], img_size: int = 224) -> Dict[str, Any]:
    """
    Ensure PIL RGB image, resize to (img_size, img_size), keep 'label' if present.
    """
    img = example["image"]
    if isinstance(img, dict):
        if img.get("bytes") is not None:
            img = Image.open(BytesIO(img["bytes"]))
        else:
            raise ValueError(f"Image dict has no bytes. keys={list(img.keys())}")

    img = img.convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)

    out: Dict[str, Any] = {"image": img}
    if "label" in example:
        out["label"] = int(example["label"]) if example["label"] is not None else -1
    return out
