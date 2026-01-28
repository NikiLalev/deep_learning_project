#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from datasets import (
    DatasetDict,
    concatenate_datasets,
    load_dataset,
)
from datasets import Image as HFImage


DEFAULT_PASCAL_PATH = "data/pascal_voc_yolo_448"
# DEFAULT_PASCAL_PATH = "/scratch/s4015843/data/pascal_voc_yolo_448"


# =========================
# Preprocess
# =========================
def preprocess_yolo(example: Dict[str, Any], img_size: int = 448) -> Dict[str, Any]:
    """
    Resize to (img_size, img_size) and convert bboxes to YOLO normalized format [xc, yc, w, h].
    """
    img = example["image"]

    orig_w = example.get("width", img.size[0])
    orig_h = example.get("height", img.size[1])

    img = img.resize((img_size, img_size), Image.BILINEAR)

    boxes: List[List[float]] = []
    labels: List[int] = []

    for box, label in zip(example["objects"]["bboxes"], example["objects"]["classes"]):
        xmin, ymin, xmax, ymax = box

        xmin = xmin * img_size / orig_w
        xmax = xmax * img_size / orig_w
        ymin = ymin * img_size / orig_h
        ymax = ymax * img_size / orig_h

        xc = ((xmin + xmax) / 2) / img_size
        yc = ((ymin + ymax) / 2) / img_size
        w = (xmax - xmin) / img_size
        h = (ymax - ymin) / img_size

        boxes.append([xc, yc, w, h])
        labels.append(int(label))

    return {
        "image": img,
        "boxes": boxes,
        "labels": labels,
    }


# =========================
# Builder
# =========================
def build_pascal_voc_yolo(
    save_path: str = DEFAULT_PASCAL_PATH,
    img_size: int = 448,
) -> DatasetDict:
    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print("Loading VOC 2007...")
    voc07 = load_dataset(
        "HuggingFaceM4/pascal_voc",
        "voc2007_main",
        trust_remote_code=True,
    )

    print("Loading VOC 2012...")
    voc12_train = load_dataset(
        "HuggingFaceM4/pascal_voc",
        "voc2012_main",
        split="train",
        trust_remote_code=True,
    )
    voc12_val = load_dataset(
        "HuggingFaceM4/pascal_voc",
        "voc2012_main",
        split="validation",
        trust_remote_code=True,
    )

    print("Concatenating splits...")
    train_ds = concatenate_datasets(
        [
            voc07["train"],
            voc07["validation"],
            voc12_train,
            voc12_val,
        ]
    )
    test_ds = voc07["test"]

    print("Applying preprocess_yolo...")

    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print("  num_cpus:", num_cpus)
    train_ds = train_ds.map(
        preprocess_yolo,
        fn_kwargs={"img_size": img_size},
        remove_columns=train_ds.column_names,
        num_proc=num_cpus
    )
    test_ds = test_ds.map(
        preprocess_yolo,
        fn_kwargs={"img_size": img_size},
        remove_columns=test_ds.column_names,
        num_proc=num_cpus
    )

    train_ds = train_ds.cast_column("image", HFImage())
    test_ds = test_ds.cast_column("image", HFImage())

    ds = DatasetDict(
        {
            "train": train_ds,
            "test": test_ds,
        }
    )

    print("Saving to:", save_path)
    ds.save_to_disk(save_path)

    print("Done.")
    return ds


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    save_path = os.environ.get("PASCAL_PATH", DEFAULT_PASCAL_PATH)
    img_size = int(os.environ.get("IMG_SIZE", "448"))

    print("Building Pascal VOC YOLO dataset")
    print("  save_path:", save_path)
    print("  img_size:", img_size)

    build_pascal_voc_yolo(save_path=save_path, img_size=img_size)
