from __future__ import annotations

from typing import Any, Dict
from io import BytesIO

import torch
import torchvision.transforms.functional as F
from PIL import Image


def _ensure_pil(img: Any) -> Image.Image:
    """
    Ensure the input is a PIL Image. If it's a dict (HF Image), extract bytes and convert.
    """
    if isinstance(img, dict):
        b = img.get("bytes", None)
        if b is None:
            raise ValueError(f"Image dict has no bytes. keys={list(img.keys())}")
        img = Image.open(BytesIO(b))
    return img


def to_torch(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert images to torch tensors. Supports single example and batched example.
    """
    img = example["image"]

    # Batched case: lists of images/boxes/labels
    if isinstance(img, list):
        imgs = [_ensure_pil(im) for im in img]
        return {
            "image": torch.stack([F.to_tensor(im) for im in imgs]),
            "boxes": [torch.tensor(b, dtype=torch.float32) for b in example["boxes"]],
            "labels": [torch.tensor(l, dtype=torch.long) for l in example["labels"]],
        }

    # Single example case
    img = _ensure_pil(img)
    return {
        "image": F.to_tensor(img),
        "boxes": torch.tensor(example["boxes"], dtype=torch.float32),
        "labels": torch.tensor(example["labels"], dtype=torch.long),
    }


def to_torch_imagenet(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ImageNet images to torch tensors, keep label if present.
    """
    img = example["image"]

    # batched (if ds[:N] happens)
    if isinstance(img, list):
        imgs = [_ensure_pil(im) for im in img]
        images = torch.stack([F.to_tensor(im) for im in imgs])
        out: Dict[str, Any] = {"image": images}
        if "label" in example:
            out["label"] = torch.tensor(
                [(-1 if l is None else int(l)) for l in example["label"]],
                dtype=torch.long,
            )
        return out

    # single
    img = _ensure_pil(img)
    out = {"image": F.to_tensor(img)}
    if "label" in example:
        out["label"] = torch.tensor(
            -1 if example["label"] is None else int(example["label"]),
            dtype=torch.long,
        )
    return out
