from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from .loaders import load_imagenet_label_names

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def plot_example(ds, example_num: int):
    """
    Plot a PASCAL example already transformed to torch tensors (image: CxHxW).
    """

    example = next(iter(ds["train"].skip(example_num)))

    image = example["image"]   # (3, 448, 448)
    boxes = example["boxes"]   # (N, 4) in YOLO format
    labels = example["labels"]

    # Convert image for matplotlib
    img_np = image.permute(1, 2, 0).numpy()  # (448, 448, 3)

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img_np)
    ax.set_axis_off()

    H, W = img_np.shape[:2]

    # Draw bounding boxes
    for box, label in zip(boxes, labels):
        xc, yc, w, h = box.tolist()

        # Convert YOLO → pixel corner format
        x_min = (xc - w / 2) * W
        y_min = (yc - h / 2) * H
        box_w = w * W
        box_h = h * H

        rect = patches.Rectangle(
            (x_min, y_min),
            box_w,
            box_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        class_name = VOC_CLASSES[label]
        ax.text(
            x_min,
            y_min - 3,
            class_name,
            color="red",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, pad=1),
        )

    plt.show()


def plot_imagenet_example(ds, example_num: int = 0):
    """
    Plot ImageNet example from a streaming iterable. Supports torch.Tensor images.
    """
    labelnames = load_imagenet_label_names()
    ex = next(iter(ds["train"].skip(example_num)))
    img = ex["image"]

    # If you've already mapped to torch tensors
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).numpy()

    label_name = labelnames[ex["label"]] if "label" in ex else "N/A"

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"label={label_name}")
    plt.show()
